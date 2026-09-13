"""Controlled residual and timing campaign for AeroJAX pressure solvers."""

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

from advection_schemes.rk3_simple_new import rk_step_unified
from pressure_solvers.multigrid_solver import poisson_multigrid
from solver import BaselineSolver, FlowConstraints, FlowParams, GeometryParams, GridParams, SimulationParams
from solver.operators import divergence_nonperiodic, grad_x_nonperiodic, grad_y_nonperiodic


def make_solver():
    grid = GridParams(512, 192, 20.0, 7.5)
    flow = FlowParams(
        U_inf=1.0,
        nu=0.001,
        Re=3000.0,
        L_char=3.0,
        constraints=FlowConstraints(lock_U=True, lock_nu=True, lock_Re=False),
    )
    params = SimulationParams(
        flow_type="von_karman",
        obstacle_type="naca_airfoil",
        pressure_solver="multigrid",
        multigrid_v_cycles=5,
        adaptive_dt=False,
        use_les=False,
    )
    return BaselineSolver(grid, flow, GeometryParams(5.0, 3.75, 0.5), params, dt=0.001)


def representative_rhs(solver):
    u_star, v_star = rk_step_unified(
        solver.u,
        solver.v,
        solver.dt,
        solver.flow.nu,
        solver.grid.dx,
        solver.grid.dy,
        solver.mask,
        sdf=solver.sdf,
        U_inf=solver.flow.U_inf,
        nu_hyper_ratio=solver.nu_hyper_ratio,
        slip_walls=solver.slip_walls,
        fast_mode=False,
        brinkman_eta=solver.sim_params.brinkman_eta,
        flow_type=solver.sim_params.flow_type,
    )
    rhs = divergence_nonperiodic(
        u_star, v_star, solver.grid.dx, solver.grid.dy
    ) / solver.dt
    return u_star, v_star, rhs.at[-1, :].set(0.0)


def laplacian_pressure(p, dx, dy, flow_type):
    mode = "edge" if flow_type in ("von_karman", "lid_driven_cavity") else "wrap"
    padded = jnp.pad(p, ((1, 1), (1, 1)), mode=mode)
    return (
        (padded[2:, 1:-1] + padded[:-2, 1:-1] - 2.0 * p) / (dx * dx)
        + (padded[1:-1, 2:] + padded[1:-1, :-2] - 2.0 * p) / (dy * dy)
    )


def residual_metrics(rhs, pressure, dx, dy, flow_type):
    residual = rhs - laplacian_pressure(pressure, dx, dy, flow_type)
    residual, rhs_host = jax.device_get((residual, rhs))
    area = dx * dy
    l2 = float(np.sqrt(np.sum(np.asarray(residual) ** 2) * area))
    rhs_l2 = float(np.sqrt(np.sum(np.asarray(rhs_host) ** 2) * area))
    return {
        "l2": l2,
        "relative_l2": l2 / max(rhs_l2, 1e-30),
        "linf": float(np.max(np.abs(np.asarray(residual)))),
        "rhs_l2": rhs_l2,
    }


def timed_solve(function, rhs, repeats):
    pressure = function(rhs)
    jax.block_until_ready(pressure)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        pressure = function(rhs)
        jax.block_until_ready(pressure)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return pressure, {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def baseline_sweep(repeats):
    solver = make_solver()
    _, _, rhs = representative_rhs(solver)
    jax.block_until_ready(rhs)
    initial = residual_metrics(
        rhs, jnp.zeros_like(rhs), solver.grid.dx, solver.grid.dy, solver.sim_params.flow_type
    )
    rows = []
    previous_l2 = initial["l2"]
    for cycles in (1, 2, 3, 4, 5, 6, 8):
        solve = jax.jit(
            lambda value, count=cycles: poisson_multigrid(
                value,
                solver.mask,
                solver.grid.dx,
                solver.grid.dy,
                v_cycles=count,
                tolerance=solver.sim_params.pressure_tolerance,
                flow_type=solver.sim_params.flow_type,
            )
        )
        pressure, timing = timed_solve(solve, rhs, repeats)
        residual = residual_metrics(
            rhs, pressure, solver.grid.dx, solver.grid.dy, solver.sim_params.flow_type
        )
        rows.append(
            {
                "variant": "baseline_zero_guess",
                "v_cycles": cycles,
                **timing,
                **residual,
                "total_reduction": initial["l2"] / max(residual["l2"], 1e-30),
                "incremental_reduction": previous_l2 / max(residual["l2"], 1e-30),
            }
        )
        previous_l2 = residual["l2"]
    return solver, rhs, initial, rows


def warm_start_sequence(solver, steps=100, max_corrections=5):
    """Compare zero-guess baseline with defect-correction warm starting."""
    dx, dy = solver.grid.dx, solver.grid.dy
    mask, dt, flow_type = solver.mask, solver.dt, solver.sim_params.flow_type

    solve_five = jax.jit(
        lambda value: poisson_multigrid(
            value, mask, dx, dy, v_cycles=5,
            tolerance=solver.sim_params.pressure_tolerance, flow_type=flow_type
        )
    )
    solve_one = lambda value: poisson_multigrid(
        value, mask, dx, dy, v_cycles=1,
        tolerance=solver.sim_params.pressure_tolerance, flow_type=flow_type
    )

    @jax.jit
    def warm_solve(rhs, initial_pressure, target_l2_sq):
        def body(_, state):
            pressure, used, done = state
            def correct():
                residual = rhs - laplacian_pressure(pressure, dx, dy, flow_type)
                candidate = pressure + solve_one(residual)
                candidate_residual = rhs - laplacian_pressure(candidate, dx, dy, flow_type)
                candidate_l2_sq = jnp.sum(candidate_residual * candidate_residual) * dx * dy
                return candidate, used + 1, candidate_l2_sq <= target_l2_sq

            return jax.lax.cond(done, lambda: state, correct)

        return jax.lax.fori_loop(
            0,
            max_corrections,
            body,
            (initial_pressure, jnp.asarray(0, jnp.int32), jnp.asarray(False)),
        )

    previous_pressure = jnp.zeros_like(solver.current_pressure)
    # Warm both compiled paths without retaining their outputs.
    u_star, v_star, rhs = representative_rhs(solver)
    baseline_pressure = solve_five(rhs)
    baseline_residual = rhs - laplacian_pressure(baseline_pressure, dx, dy, flow_type)
    target = jnp.sum(baseline_residual * baseline_residual) * dx * dy
    jax.block_until_ready(warm_solve(rhs, previous_pressure, target))

    rows = []
    for index in range(steps):
        u_star, v_star, rhs = representative_rhs(solver)
        started = time.perf_counter_ns()
        baseline_pressure = solve_five(rhs)
        jax.block_until_ready(baseline_pressure)
        baseline_ms = (time.perf_counter_ns() - started) / 1e6
        baseline_residual = rhs - laplacian_pressure(baseline_pressure, dx, dy, flow_type)
        target = jnp.sum(baseline_residual * baseline_residual) * dx * dy

        started = time.perf_counter_ns()
        warm_pressure, corrections, _ = warm_solve(rhs, previous_pressure, target)
        warm_pressure, corrections = jax.device_get((warm_pressure, corrections))
        warm_ms = (time.perf_counter_ns() - started) / 1e6
        previous_pressure = jnp.asarray(warm_pressure)

        baseline_metrics = residual_metrics(rhs, baseline_pressure, dx, dy, flow_type)
        warm_metrics = residual_metrics(rhs, warm_pressure, dx, dy, flow_type)
        pressure_delta = np.asarray(warm_pressure) - np.asarray(baseline_pressure)

        baseline_u = u_star - dt * grad_x_nonperiodic(baseline_pressure, dx)
        baseline_v = v_star - dt * grad_y_nonperiodic(baseline_pressure, dy)
        warm_u = u_star - dt * grad_x_nonperiodic(previous_pressure, dx)
        warm_v = v_star - dt * grad_y_nonperiodic(previous_pressure, dy)
        velocity_delta = jax.device_get(
            jnp.sqrt((warm_u - baseline_u) ** 2 + (warm_v - baseline_v) ** 2)
        )
        rows.append(
            {
                "step": index,
                "baseline_ms": baseline_ms,
                "warm_ms": warm_ms,
                "corrections": int(corrections),
                "baseline_relative_l2": baseline_metrics["relative_l2"],
                "warm_relative_l2": warm_metrics["relative_l2"],
                "target_met": warm_metrics["l2"] <= baseline_metrics["l2"] * (1.0 + 1e-6),
                "pressure_error_l2": float(np.sqrt(np.sum(pressure_delta ** 2) * dx * dy)),
                "pressure_error_linf": float(np.max(np.abs(pressure_delta))),
                "velocity_error_l2": float(np.sqrt(np.sum(velocity_delta ** 2) * dx * dy)),
                "velocity_error_linf": float(np.max(velocity_delta)),
            }
        )
        solver.u, solver.v, solver.current_pressure = solver._step_jit(
            solver.u, solver.v, solver.mask, solver.dt, solver.iteration
        )
        solver.iteration += 1
        jax.block_until_ready((solver.u, solver.v))

    timed = rows[1:]  # Exclude first physical step from aggregate warm-start statistics.
    return {
        "steps": steps,
        "max_corrections": max_corrections,
        "baseline_median_ms": statistics.median(row["baseline_ms"] for row in timed),
        "warm_median_ms": statistics.median(row["warm_ms"] for row in timed),
        "speedup": statistics.median(row["baseline_ms"] for row in timed)
        / statistics.median(row["warm_ms"] for row in timed),
        "median_corrections": statistics.median(row["corrections"] for row in timed),
        "target_met_fraction": sum(row["target_met"] for row in timed) / len(timed),
        "rows": rows,
    }


def controlled_sweeps(solver, rhs, repeats):
    """Change one smoother control at a time around the 5-cycle baseline."""
    dx, dy = solver.grid.dx, solver.grid.dy
    baseline = poisson_multigrid(
        rhs, solver.mask, dx, dy, v_cycles=5,
        tolerance=solver.sim_params.pressure_tolerance,
        flow_type=solver.sim_params.flow_type,
    )
    jax.block_until_ready(baseline)
    baseline_residual = residual_metrics(rhs, baseline, dx, dy, solver.sim_params.flow_type)
    configurations = []
    for omega in (0.5, 2.0 / 3.0, 0.8, 1.0):
        configurations.append((f"omega_{omega:.3f}", 2, 2, 10, omega))
    for count in (1, 2, 3):
        configurations.append((f"prepost_{count}", count, count, 10, 1.0))
    for count in (4, 10, 20):
        configurations.append((f"coarse_{count}", 2, 2, count, 1.0))
    rows = []
    for name, pre, post, coarse, omega in configurations:
        solve = jax.jit(
            lambda value, pre_count=pre, post_count=post, coarse_count=coarse, weight=omega:
            poisson_multigrid(
                value, solver.mask, dx, dy, v_cycles=5,
                tolerance=solver.sim_params.pressure_tolerance,
                flow_type=solver.sim_params.flow_type,
                pre_smooth_steps=pre_count, post_smooth_steps=post_count,
                coarse_smooth_steps=coarse_count, jacobi_omega=weight,
            )
        )
        pressure, timing = timed_solve(solve, rhs, repeats)
        metrics = residual_metrics(rhs, pressure, dx, dy, solver.sim_params.flow_type)
        delta = np.asarray(pressure) - np.asarray(baseline)
        rows.append(
            {
                "variant": name,
                "v_cycles": 5,
                "pre_smooth": pre,
                "post_smooth": post,
                "coarse_smooth": coarse,
                "omega": omega,
                **timing,
                **metrics,
                "meets_baseline_residual": metrics["l2"] <= baseline_residual["l2"] * (1 + 1e-6),
                "pressure_error_l2": float(np.sqrt(np.sum(delta * delta) * dx * dy)),
                "pressure_error_linf": float(np.max(np.abs(delta))),
            }
        )
    return {"baseline_residual": baseline_residual, "rows": rows}


def hierarchy_spacing_sweep(solver, rhs, repeats):
    """Test the coarse-grid spacing correction independently of other controls."""
    dx, dy = solver.grid.dx, solver.grid.dy
    rows = []
    for scaled in (False, True):
        for cycles in (1, 2, 3, 4, 5):
            solve = jax.jit(
                lambda value, use_scaled=scaled, count=cycles: poisson_multigrid(
                    value, solver.mask, dx, dy, v_cycles=count,
                    tolerance=solver.sim_params.pressure_tolerance,
                    flow_type=solver.sim_params.flow_type,
                    level_scaled_spacing=use_scaled,
                )
            )
            pressure, timing = timed_solve(solve, rhs, repeats)
            rows.append({
                "variant": "scaled_coarse_spacing" if scaled else "legacy_coarse_spacing",
                "v_cycles": cycles,
                **timing,
                **residual_metrics(rhs, pressure, dx, dy, solver.sim_params.flow_type),
            })
    baseline_l2 = next(
        row["l2"] for row in rows
        if row["variant"] == "legacy_coarse_spacing" and row["v_cycles"] == 5
    )
    for row in rows:
        row["meets_baseline_residual"] = row["l2"] <= baseline_l2 * (1.0 + 1e-6)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    solver, rhs, initial, rows = baseline_sweep(args.repeats)
    warm_start = warm_start_sequence(make_solver())
    smoothing = controlled_sweeps(solver, rhs, args.repeats)
    hierarchy_spacing = hierarchy_spacing_sweep(solver, rhs, args.repeats)
    data = {
        "device": str(jax.devices()),
        "grid": [solver.grid.nx, solver.grid.ny],
        "flow_type": solver.sim_params.flow_type,
        "levels_requested": solver.sim_params.multigrid_levels,
        "pre_smoothing": 2,
        "post_smoothing": 2,
        "coarse_smoothing": 10,
        "smoother": "unweighted simultaneous Jacobi",
        "initial_guess": "zeros",
        "null_space": "not applicable; outlet pressure is pinned to zero",
        "initial_residual": initial,
        "rows": rows,
        "warm_start": warm_start,
        "controlled_smoothing_sweeps": smoothing,
        "hierarchy_spacing_sweep": hierarchy_spacing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
