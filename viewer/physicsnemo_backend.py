"""GUI-side PhysicsNeMo bridge. This module intentionally imports no torch."""
from __future__ import annotations

import json
import os
import pickle
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-physicsnemo" / "bin" / "python"
SERVER = ROOT / "physicsnemo_fno" / "inference_server.py"
CHECKPOINT = ROOT / "physicsnemo_fno" / "runs" / "residual_fno" / "best.pt"
DEFAULT_DATASET = Path("/mnt/c/users/gusta/documents/codex/2026-09-09/files-pasted-by-the-user-s/outputs/teacher-pilot")


class BridgeError(RuntimeError):
    pass


class PhysicsNeMoBridge:
    def __init__(self, dataset_root=None, autostart=True):
        self.dataset_root = Path(dataset_root or os.environ.get("AEROJAX_FNO_DATASET", DEFAULT_DATASET))
        self.process = None
        if autostart:
            self.start()

    def start(self):
        missing = [str(p) for p in (PYTHON, SERVER, CHECKPOINT, self.dataset_root) if not p.exists()]
        if missing:
            raise BridgeError("PhysicsNeMo prerequisites missing: " + ", ".join(missing))
        self.process = subprocess.Popen(
            [str(PYTHON), "-u", str(SERVER)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        hello = self.request({"command": "hello", "checkpoint": str(CHECKPOINT), "dataset_root": str(self.dataset_root)}, timeout=60.0)
        self.info = hello

    def _read_exact(self, count):
        chunks = []
        remaining = count
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                stderr = self.process.stderr.read().decode("utf-8", "replace")
                raise BridgeError(f"PhysicsNeMo process exited unexpectedly. {stderr[-2000:]}")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def request(self, message, timeout=None):
        del timeout  # synchronous worker calls are bounded by process health
        if self.process.poll() is not None:
            raise BridgeError("PhysicsNeMo process is not running")
        started = time.perf_counter()
        self.process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        self.process.stdin.flush()
        size = struct.unpack("!Q", self._read_exact(8))[0]
        response = pickle.loads(self._read_exact(size))
        response["bridge_roundtrip_ms"] = (time.perf_counter() - started) * 1000.0
        if not response.get("ok"):
            raise BridgeError(response.get("error", "Unknown PhysicsNeMo error"))
        return response

    def reset(self, case_path):
        return self.request({"command": "reset", "case_path": case_path})

    def step(self):
        return self.request({"command": "step"})

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                self.request({"command": "close"})
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self.process = None

    def abort(self):
        """Interrupt model loading or inference during Reset/application close."""
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()


class PhysicsNeMoWorker(QObject):
    failed = pyqtSignal(object, str)
    data_ready = pyqtSignal(object)
    fps_update = pyqtSignal(int)
    profiling_update = pyqtSignal(float, float, float, float)

    def __init__(self, solver, case_selector):
        super().__init__()
        self.solver = solver
        self.case_selector = case_selector
        self.bridge = None
        self.thread = None
        self.running = False
        self.paused = False
        self.pending_frame = False
        self.last_frame = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, name="PhysicsNeMo-FNO", daemon=True)
        self.thread.start()

    def _publish(self, frame):
        if self.bridge is not None:
            frame["gpu"] = self.bridge.info.get("gpu", "CUDA")
        frame["pressure"] = frame.pop("p")
        frame["scalar"] = None
        frame["rms_divergence"] = float(np.sqrt(np.mean(frame["div"] ** 2)))
        frame["source"] = self
        frame["rtf"] = frame["dt"] / max(frame["bridge_roundtrip_ms"] / 1000.0, 1e-9)
        self.last_frame = frame
        self.pending_frame = True
        self.data_ready.emit(frame)

    def _run(self):
        count = 0
        window_start = time.perf_counter()
        try:
            self.bridge = PhysicsNeMoBridge(autostart=False)
            self.bridge.start()
            matches = [case for case in self.bridge.info["cases"] if
                       case["geometry_id"] == self.case_selector["geometry_id"] and
                       abs(case["U_inf"] - self.case_selector["U_inf"]) < 1e-6 and
                       abs(case["Re"] - self.case_selector["Re"]) < 1e-3]
            if len(matches) != 1:
                raise BridgeError(f"Pilot case is unavailable or ambiguous: {self.case_selector}")
            self._publish(self.bridge.reset(matches[0]["path"]))
            while self.running:
                if self.paused or self.pending_frame:
                    time.sleep(0.001)
                    continue
                frame = self.bridge.step()
                self._publish(frame)
                count += 1
                elapsed = time.perf_counter() - window_start
                if elapsed >= 1.0:
                    fps = count / elapsed
                    self.fps_update.emit(round(fps))
                    self.profiling_update.emit(frame["inference_ms"], frame["gpu_to_cpu_ms"], frame["bridge_roundtrip_ms"], fps)
                    count = 0
                    window_start = time.perf_counter()
        except Exception as exc:
            if self.running:
                self.failed.emit(self, str(exc))
        finally:
            self.running = False
            if self.bridge:
                self.bridge.close()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop_simulation(self):
        self.running = False
        self.paused = False
        if self.bridge is not None:
            self.bridge.abort()
        if self.thread and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=5.0)
        if self.thread and self.thread.is_alive():
            raise RuntimeError("PhysicsNeMo worker is still stopping")
