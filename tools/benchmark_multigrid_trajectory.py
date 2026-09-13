"""End-to-end trajectory comparison for the corrected multigrid hierarchy."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import solver.simulation.step_handlers as step_handlers
from pressure_solvers.multigrid_solver import poisson_multigrid as original_multigrid
from solver.operators import divergence_nonperiodic
from tools.benchmark_multigrid import make_solver


CORRECTED_CYCLES = 1


def corrected_multigrid(rhs, mask, dx, dy, v_cycles=5, tolerance=1e-6,
                        flow_type="von_karman"):
    del v_cycles
    return original_multigrid(
        rhs, mask, dx, dy, v_cycles=CORRECTED_CYCLES, tolerance=tolerance,
        flow_type=flow_type, level_scaled_spacing=True,
    )


def run_steps(solver, steps):
    samples = []
    # Compile before timing.
    result = solver._step_jit(solver.u, solver.v, solver.mask, solver.dt, solver.iteration)
    jax.block_until_ready(result)
    for _ in range(steps):
        started = time.perf_counter_ns()
        solver.u, solver.v, solver.current_pressure = solver._step_jit(
            solver.u, solver.v, solver.mask, solver.dt, solver.iteration
        )
        solver.iteration += 1
        jax.block_until_ready((solver.u, solver.v, solver.current_pressure))
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return samples


def field_error(candidate, baseline, dx, dy):
    delta = np.asarray(candidate) - np.asarray(baseline)
    return {
        "l2": float(np.sqrt(np.sum(delta * delta) * dx * dy)),
        "linf": float(np.max(np.abs(delta))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--candidate-cycles", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    global CORRECTED_CYCLES
    CORRECTED_CYCLES = args.candidate_cycles
    baseline = make_solver()
    baseline._jit_cache.clear()
    baseline._step_jit = baseline.get_step_jit()

    step_handlers.poisson_multigrid = corrected_multigrid
    candidate = make_solver()
    # Force a distinct JIT cache key so JAX cannot reuse the baseline trace
    # after the module-level pressure implementation is replaced.
    candidate.sim_params.multigrid_v_cycles = args.candidate_cycles
    candidate._jit_cache.clear()
    candidate._step_jit = candidate.get_step_jit()
    step_handlers.poisson_multigrid = original_multigrid

    baseline_samples = run_steps(baseline, args.steps)
    candidate_samples = run_steps(candidate, args.steps)
    baseline_ms = statistics.median(baseline_samples)
    candidate_ms = statistics.median(candidate_samples)
    dx, dy = baseline.grid.dx, baseline.grid.dy
    baseline_div = divergence_nonperiodic(baseline.u, baseline.v, dx, dy)
    candidate_div = divergence_nonperiodic(candidate.u, candidate.v, dx, dy)
    baseline_div, candidate_div = jax.device_get((baseline_div, candidate_div))

    result = {
        "device": str(jax.devices()),
        "grid": [baseline.grid.nx, baseline.grid.ny],
        "steps": args.steps,
        "dt": float(baseline.dt),
        "baseline": {
            "variant": "legacy spacing, 5 V-cycles",
            "full_step_median_ms": baseline_ms,
            "step_per_s": 1000.0 / baseline_ms,
            "rtf": float(baseline.dt) * 1000.0 / baseline_ms,
            "divergence_l2": float(np.sqrt(np.sum(baseline_div ** 2) * dx * dy)),
            "divergence_linf": float(np.max(np.abs(baseline_div))),
        },
        "candidate": {
            "variant": f"level-scaled spacing, {args.candidate_cycles} V-cycle(s)",
            "full_step_median_ms": candidate_ms,
            "step_per_s": 1000.0 / candidate_ms,
            "rtf": float(candidate.dt) * 1000.0 / candidate_ms,
            "divergence_l2": float(np.sqrt(np.sum(candidate_div ** 2) * dx * dy)),
            "divergence_linf": float(np.max(np.abs(candidate_div))),
        },
        "speedup": baseline_ms / candidate_ms,
        "error_vs_baseline": {
            "u": field_error(candidate.u, baseline.u, dx, dy),
            "v": field_error(candidate.v, baseline.v, dx, dy),
            "pressure": field_error(
                candidate.current_pressure, baseline.current_pressure, dx, dy
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
