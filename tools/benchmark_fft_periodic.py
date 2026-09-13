"""FFT versus multigrid benchmark for a periodic Taylor-Green pressure RHS."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from pressure_solvers.FFT import poisson_fft_dirichlet_neumann
from pressure_solvers.multigrid_solver import poisson_multigrid
from tools.validate_cfd import make_solver
from tools.benchmark_multigrid import residual_metrics


def time_solver(function, rhs, repeats):
    output = function(rhs)
    jax.block_until_ready(output)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        output = function(rhs)
        jax.block_until_ready(output)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return output, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    solver = make_solver("taylor_green", 512, 100.0, 0.001)
    x, y = solver.grid.X, solver.grid.Y
    # Smooth, zero-mean periodic RHS with several resolved modes.
    rhs = (
        jnp.sin(2.0 * x) * jnp.cos(3.0 * y)
        + 0.25 * jnp.cos(5.0 * x) * jnp.sin(y)
    )
    rhs = rhs - jnp.mean(rhs)
    fft = jax.jit(
        lambda value: poisson_fft_dirichlet_neumann(
            value, solver.grid.dx, solver.grid.dy, flow_type="taylor_green"
        )
    )
    multigrid = jax.jit(
        lambda value: poisson_multigrid(
            value, solver.mask, solver.grid.dx, solver.grid.dy,
            v_cycles=5, flow_type="taylor_green"
        )
    )
    fft_pressure, fft_ms = time_solver(fft, rhs, args.repeats)
    mg_pressure, mg_ms = time_solver(multigrid, rhs, args.repeats)
    fft_residual = residual_metrics(
        rhs, fft_pressure, solver.grid.dx, solver.grid.dy, "taylor_green"
    )
    mg_residual = residual_metrics(
        rhs, mg_pressure, solver.grid.dx, solver.grid.dy, "taylor_green"
    )
    # Pressure is compared after removing its arbitrary constant.
    fft_centered = np.asarray(fft_pressure) - float(jnp.mean(fft_pressure))
    mg_centered = np.asarray(mg_pressure) - float(jnp.mean(mg_pressure))
    difference = fft_centered - mg_centered
    result = {
        "device": str(jax.devices()),
        "grid": [solver.grid.nx, solver.grid.ny],
        "fft": {"median_ms": fft_ms, **fft_residual},
        "multigrid_5_cycles": {"median_ms": mg_ms, **mg_residual},
        "fft_speedup": mg_ms / fft_ms,
        "pressure_difference_l2": float(
            np.sqrt(np.sum(difference * difference) * solver.grid.dx * solver.grid.dy)
        ),
        "pressure_difference_linf": float(np.max(np.abs(difference))),
        "scope": "periodic domains only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
