"""Quantitative, synchronized validation benchmarks for AeroJAX.

This module observes the current numerical method.  It deliberately does not
change solver parameters behind the user's back or claim steady convergence
when the stopping criterion has not been reached.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from solver import (
    BaselineSolver,
    FlowConstraints,
    FlowParams,
    GeometryParams,
    GridParams,
    SimulationParams,
)
from solver.operators import divergence, divergence_nonperiodic
from validation.ldc_validator import LDCValidator


@dataclass
class ErrorNorms:
    l2: float
    linf: float
    divergence_l2: float
    divergence_linf: float


def synchronize(value):
    """Return host values only after all dependent device work completes."""
    return jax.device_get(value)


def make_solver(flow_type: str, n: int, reynolds: float, dt: float) -> BaselineSolver:
    length = 2.0 * math.pi if flow_type == "taylor_green" else 1.0
    velocity = 1.0
    viscosity = velocity * length / reynolds
    grid = GridParams(nx=n, ny=n, lx=length, ly=length)
    flow = FlowParams(
        U_inf=velocity,
        nu=viscosity,
        Re=reynolds,
        L_char=length,
        constraints=FlowConstraints(lock_U=True, lock_nu=True, lock_Re=False),
    )
    geom = GeometryParams(center_x=0.5 * length, center_y=0.5 * length, radius=0.1)
    params = SimulationParams(
        flow_type=flow_type,
        obstacle_type="cylinder",
        grid_type="collocated",
        pressure_solver="multigrid",
        adaptive_dt=False,
        fixed_dt=dt,
        use_les=False,
        multigrid_v_cycles=5,
    )
    return BaselineSolver(grid, flow, geom, params, dt=dt)


def velocity_norms(u, v, u_exact, v_exact, div, dx, dy) -> ErrorNorms:
    eu = np.asarray(u) - np.asarray(u_exact)
    ev = np.asarray(v) - np.asarray(v_exact)
    magnitude = np.sqrt(eu * eu + ev * ev)
    div_host = np.asarray(div)
    area = dx * dy
    return ErrorNorms(
        l2=float(np.sqrt(np.sum(magnitude * magnitude) * area)),
        linf=float(np.max(magnitude)),
        divergence_l2=float(np.sqrt(np.sum(div_host * div_host) * area)),
        divergence_linf=float(np.max(np.abs(div_host))),
    )


def run_taylor_green(n: int, dt: float, final_time: float, reynolds: float = 100.0):
    solver = make_solver("taylor_green", n, reynolds, dt)
    steps = round(final_time / dt)
    started = time.perf_counter()
    solver.advance_steps(steps)
    u, v, div = synchronize(
        (solver.u, solver.v, divergence(solver.u, solver.v, solver.grid.dx, solver.grid.dy))
    )
    wall = time.perf_counter() - started
    # Exact 2-D Taylor-Green velocity for wave number one.
    decay = math.exp(-2.0 * solver.flow.nu * solver.simulated_time)
    u_exact = decay * np.sin(np.asarray(solver.grid.X)) * np.cos(np.asarray(solver.grid.Y))
    v_exact = -decay * np.cos(np.asarray(solver.grid.X)) * np.sin(np.asarray(solver.grid.Y))
    result = asdict(velocity_norms(u, v, u_exact, v_exact, div, solver.grid.dx, solver.grid.dy))
    result.update(
        n=n,
        dt=dt,
        steps=steps,
        final_time=solver.simulated_time,
        wall_seconds=wall,
        rtf=solver.simulated_time / wall,
        finite=bool(np.isfinite(u).all() and np.isfinite(v).all()),
    )
    return result


def run_taylor_green_state(n: int, dt: float, final_time: float):
    solver = make_solver("taylor_green", n, 100.0, dt)
    solver.advance_steps(round(final_time / dt))
    return synchronize((solver.u, solver.v))


def temporal_self_convergence():
    """Compare against the same spatial discretization at a much smaller dt."""
    n, final_time, reference_dt = 64, 0.04, 0.00025
    reference_u, reference_v = run_taylor_green_state(n, reference_dt, final_time)
    area = (2.0 * math.pi / n) ** 2
    rows = []
    for dt in (0.004, 0.002, 0.001):
        u, v = run_taylor_green_state(n, dt, final_time)
        error = np.sqrt((u - reference_u) ** 2 + (v - reference_v) ** 2)
        rows.append(
            {
                "dt": dt,
                "reference_dt": reference_dt,
                "l2": float(np.sqrt(np.sum(error * error) * area)),
                "linf": float(np.max(error)),
            }
        )
    return observed_orders(rows, "dt")


def observed_orders(rows, independent: str):
    for previous, current in zip(rows, rows[1:]):
        ratio = previous[independent] / current[independent]
        current["observed_order_l2"] = math.log(previous["l2"] / current["l2"]) / math.log(ratio)
        current["observed_order_linf"] = math.log(previous["linf"] / current["linf"]) / math.log(ratio)
    return rows


def taylor_green_suite():
    # Short horizons isolate discretization error while keeping this suite practical.
    temporal = [run_taylor_green(64, dt, 0.04) for dt in (0.004, 0.002, 0.001)]
    spatial = [run_taylor_green(n, 0.0005, 0.02) for n in (32, 64, 128)]
    for row in spatial:
        row["h"] = 2.0 * math.pi / row["n"]
    return {
        "definition": "u=sin(x)cos(y)e^(-2nu t), v=-cos(x)sin(y)e^(-2nu t)",
        "temporal": observed_orders(temporal, "dt"),
        "temporal_self_convergence": temporal_self_convergence(),
        "spatial": observed_orders(spatial, "h"),
    }


def run_ldc(reynolds: int, n: int, max_steps: int, sample_every: int, tolerance: float):
    dt = min(0.1 / n, 0.2 / reynolds)
    solver = make_solver("lid_driven_cavity", n, float(reynolds), dt)
    previous_u = np.asarray(solver.u)
    previous_v = np.asarray(solver.v)
    converged = False
    relative_change = float("inf")
    started = time.perf_counter()
    for completed in range(sample_every, max_steps + 1, sample_every):
        solver.advance_steps(sample_every)
        u, v = synchronize((solver.u, solver.v))
        difference = np.sqrt(np.mean((u - previous_u) ** 2 + (v - previous_v) ** 2))
        scale = max(float(np.sqrt(np.mean(u * u + v * v))), 1e-12)
        relative_change = float(difference / scale)
        previous_u, previous_v = u, v
        if not np.isfinite(relative_change):
            break
        if relative_change < tolerance:
            converged = True
            break
    wall = time.perf_counter() - started
    div = synchronize(divergence_nonperiodic(solver.u, solver.v, solver.grid.dx, solver.grid.dy))
    snapshot = SimpleNamespace(
        u=u,
        v=v,
        dx=solver.grid.dx,
        dy=solver.grid.dy,
        nx=n,
        ny=n,
        iteration=solver.iteration,
        timestamp=solver.simulated_time,
    )
    validator = LDCValidator.load(reynolds)
    validator.compute(snapshot)
    center = validator.get_vortex_center()
    error = validator.get_error()
    return {
        "reynolds": reynolds,
        "n": n,
        "dt": dt,
        "steps": solver.iteration,
        "simulated_time": solver.simulated_time,
        "wall_seconds": wall,
        "rtf": solver.simulated_time / wall,
        "converged": converged,
        "stopping_tolerance": tolerance,
        "sample_relative_change": relative_change,
        "center": [center.x, center.y],
        "reference_center": list(error.reference_center),
        "center_l2_error": error.l2_distance,
        "divergence_l2": float(np.sqrt(np.sum(np.asarray(div) ** 2) * solver.grid.dx * solver.grid.dy)),
        "divergence_linf": float(np.max(np.abs(np.asarray(div)))),
        "finite": bool(np.isfinite(u).all() and np.isfinite(v).all()),
    }


def ldc_suite(n: int, max_steps: int):
    return [run_ldc(reynolds, n, max_steps, 100, 1e-5) for reynolds in (100, 400, 1000, 3200, 5000)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("tgv", "ldc", "all"), default="all")
    parser.add_argument("--ldc-grid", type=int, default=64)
    parser.add_argument("--ldc-steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {
        "device": str(jax.devices()),
        "jax_version": jax.__version__,
        "grid_convention": "GridParams currently reports dx=L/n while coordinates use linspace endpoints L/(n-1)",
    }
    if args.suite in ("tgv", "all"):
        data["taylor_green"] = taylor_green_suite()
    if args.suite in ("ldc", "all"):
        data["lid_driven_cavity"] = ldc_suite(args.ldc_grid, args.ldc_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
