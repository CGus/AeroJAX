#!/usr/bin/env python3
"""Persistent PhysicsNeMo inference process for the AeroJAX GUI.

The protocol is deliberately small: newline-delimited JSON requests on stdin and
length-prefixed pickle responses on stdout.  stdout is reserved for the protocol.
"""
from __future__ import annotations

import json
import pickle
import struct
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from physicsnemo.models.fno.fno import FNO


FIELDS = ("u", "v", "p")
PARAMS = ("U_inf", "nu", "Re", "aoa", "dt", "delta_t", "inlet_x", "inlet_y", "dx", "dy")


def respond(value):
    payload = pickle.dumps(value, protocol=5)
    sys.stdout.buffer.write(struct.pack("!Q", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def build_model(config):
    model = config["model"]
    return FNO(
        in_channels=3 + 2 + len(PARAMS) + 4,
        out_channels=3,
        dimension=2,
        latent_channels=model["width"],
        num_fno_layers=model["layers"],
        num_fno_modes=model["modes"],
        padding=model["padding"],
        padding_type="constant",
        coord_features=True,
    )


def case_catalog(dataset_root):
    cases = []
    for split in ("train", "validation", "test"):
        for path in sorted((dataset_root / split).glob("*.h5")):
            with h5py.File(path, "r") as handle:
                case = json.loads(handle.attrs["case"])
                resolved = json.loads(handle.attrs["resolved"])
            geometry = case["geometry"]
            flow = resolved["flow"]
            cases.append({
                "path": str(path),
                "split": split,
                "geometry_id": geometry["id"],
                "family": geometry["family"],
                "geometry": geometry,
                "U_inf": float(flow["U_inf"]),
                "nu": float(flow["nu"]),
                "Re": float(flow["Re"]),
            })
    return cases


class Runtime:
    def __init__(self, checkpoint, dataset_root):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in .venv-physicsnemo")
        self.device = torch.device("cuda:0")
        self.checkpoint_path = Path(checkpoint)
        self.dataset_root = Path(dataset_root)
        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model = build_model(checkpoint_data["config"]).to(self.device)
        self.model.load_state_dict(checkpoint_data["model"])
        self.model.eval()
        self.fm = torch.as_tensor(checkpoint_data["field_mean"], device=self.device)[None, :, None, None]
        self.fs = torch.as_tensor(checkpoint_data["field_std"], device=self.device)[None, :, None, None]
        self.dm = torch.as_tensor(checkpoint_data["delta_mean"], device=self.device)[None, :, None, None]
        self.ds = torch.as_tensor(checkpoint_data["delta_std"], device=self.device)[None, :, None, None]
        self.pm = np.asarray(checkpoint_data["param_mean"], np.float32)
        self.ps = np.asarray(checkpoint_data["param_std"], np.float32)
        self.cases = case_catalog(self.dataset_root)
        if not self.cases:
            raise RuntimeError(f"No pilot HDF5 cases found under {self.dataset_root}")
        self.reset(self.cases[0]["path"])

    def reset(self, case_path):
        with h5py.File(case_path, "r") as handle:
            self.case = json.loads(handle.attrs["case"])
            self.resolved = json.loads(handle.attrs["resolved"])
            self.state = handle["fields"][0].astype(np.float32)
            self.mask = handle["mask"][:].astype(np.float32)
            sdf = handle["sdf"][:].astype(np.float32)
            self.sdf = sdf / max(float(np.max(np.abs(sdf))), 1e-6)
            self.initial_time = float(handle["time"][0])
        self.iteration = 0
        self.time = self.initial_time
        self.delta_t = 0.002
        nx, ny = self.mask.shape
        bc = np.zeros((4, nx, ny), np.float32)
        bc[0, 0, :] = 1
        bc[1, -1, :] = 1
        bc[2, :, 0] = 1
        bc[3, :, -1] = 1
        flow = self.resolved["flow"]
        inlet = self.resolved.get("inlet_direction", [1.0, 0.0])
        aoa = float(self.case["geometry"].get("aoa", 0.0))
        raw = np.asarray([
            flow["U_inf"], flow["nu"], flow["Re"], aoa, 0.001,
            self.delta_t, inlet[0], inlet[1], self.resolved["dx"], self.resolved["dy"],
        ], np.float32)
        pn = (raw - self.pm) / self.ps
        self.condition = np.concatenate([
            self.mask[None], self.sdf[None],
            np.broadcast_to(pn[:, None, None], (len(PARAMS), nx, ny)), bc,
        ], axis=0).copy()
        self.state_gpu = torch.from_numpy(self.state)[None].to(self.device)
        self.condition_gpu = torch.from_numpy(self.condition)[None].to(self.device)
        self.mask_gpu = torch.from_numpy(self.mask)[None].to(self.device)
        return self.frame(0.0, 0.0)

    def frame(self, inference_ms, transfer_ms):
        u, v, p = self.state
        dx, dy = float(self.resolved["dx"]), float(self.resolved["dy"])
        vort = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)
        div = np.gradient(u, dx, axis=0) + np.gradient(v, dy, axis=1)
        return {
            "ok": True,
            "u": np.ascontiguousarray(u, dtype=np.float32),
            "v": np.ascontiguousarray(v, dtype=np.float32),
            "p": np.ascontiguousarray(p, dtype=np.float32),
            "vel_mag": np.ascontiguousarray(np.hypot(u, v), dtype=np.float32),
            "vort": np.ascontiguousarray(vort, dtype=np.float32),
            "div": np.ascontiguousarray(div, dtype=np.float32),
            "mask": self.mask,
            "iteration": self.iteration,
            "time": self.time,
            "dt": self.delta_t,
            "inference_ms": inference_ms,
            "gpu_to_cpu_ms": transfer_ms,
            "vram_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "vram_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "case": self.case,
            "flow": self.resolved["flow"],
        }

    @torch.inference_mode()
    def step(self):
        state_n = (self.state_gpu - self.fm) / self.fs
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dz = self.model(torch.cat([state_n, self.condition_gpu], dim=1))
        raw = state_n * self.fs + self.fm + dz * self.ds + self.dm
        outlet = self.mask_gpu[:, -1, :]
        offset = (raw[:, 2, -1, :] * outlet).sum(1) / outlet.sum(1).clamp_min(1)
        raw = torch.cat([raw[:, :2], raw[:, 2:3] - offset[:, None, None, None]], dim=1)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        self.state_gpu = raw
        self.state = raw[0].cpu().numpy().astype(np.float32, copy=False)
        transfer_ms = (time.perf_counter() - t1) * 1000.0
        self.iteration += 1
        self.time += self.delta_t
        return self.frame(inference_ms, transfer_ms)


def main():
    runtime = None
    for raw_line in sys.stdin.buffer:
        try:
            request = json.loads(raw_line)
            command = request["command"]
            if command == "hello":
                runtime = Runtime(request["checkpoint"], request["dataset_root"])
                respond({
                    "ok": True,
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": str(torch.__version__),
                    "cases": runtime.cases,
                    "checkpoint": str(runtime.checkpoint_path),
                })
            elif command == "reset":
                respond(runtime.reset(request["case_path"]))
            elif command == "step":
                respond(runtime.step())
            elif command == "close":
                respond({"ok": True})
                break
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as exc:
            respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
