"""Synchronized GPU microprofile of AeroJAX's collocated CFD blocks.

Each phase is compiled and timed in isolation.  The normalized shares describe
the sum of isolated phase costs; fusion makes them an attribution model rather
than an additive decomposition of the fused production kernel.
"""

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

from advection_schemes.rk3_simple_new import rk_step_unified
from pressure_solvers import poisson_multigrid
from solver import (
    BaselineSolver,
    FlowConstraints,
    FlowParams,
    GeometryParams,
    GridParams,
    SimulationParams,
)
from solver.operators import divergence_nonperiodic, grad_x_nonperiodic, grad_y_nonperiodic


def make_solver(n: int) -> BaselineSolver:
    grid = GridParams(n, round(n * 192 / 512), 20.0, 7.5)
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
        naca_airfoil="NACA 0012",
        naca_angle=0.0,
        pressure_solver="multigrid",
        grid_type="collocated",
        adaptive_dt=False,
        use_les=False,
        multigrid_v_cycles=5,
    )
    return BaselineSolver(grid, flow, GeometryParams(5.0, 3.75, 0.5), params, dt=0.001)


def complete(value):
    jax.block_until_ready(value)
    return value


def measure(function, arguments, repeats: int):
    compiled = jax.jit(function)
    complete(compiled(*arguments))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        complete(compiled(*arguments))
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-x", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    solver = make_solver(args.grid_x)
    u, v, mask, dt = solver.u, solver.v, solver.mask, solver.dt
    dx, dy, nu = solver.grid.dx, solver.grid.dy, solver.flow.nu

    def advection(u_value, v_value):
        return rk_step_unified(
            u_value,
            v_value,
            dt,
            nu,
            dx,
            dy,
            mask,
            sdf=solver.sdf,
            U_inf=solver.flow.U_inf,
            nu_hyper_ratio=solver.nu_hyper_ratio,
            slip_walls=solver.slip_walls,
            fast_mode=False,
            brinkman_eta=solver.sim_params.brinkman_eta,
            flow_type=solver.sim_params.flow_type,
        )

    u_star, v_star = complete(jax.jit(advection)(u, v))

    def differential_operators(u_value, v_value):
        return divergence_nonperiodic(u_value, v_value, dx, dy)

    rhs = complete(differential_operators(u_star, v_star)) / dt

    def pressure(right_hand_side):
        return poisson_multigrid(
            right_hand_side,
            mask,
            dx,
            dy,
            v_cycles=solver.sim_params.multigrid_v_cycles,
            flow_type=solver.sim_params.flow_type,
        )

    pressure_value = complete(jax.jit(pressure)(rhs))

    def pressure_gradient(p):
        return grad_x_nonperiodic(p, dx), grad_y_nonperiodic(p, dy)

    dp_dx, dp_dy = complete(jax.jit(pressure_gradient)(pressure_value))

    def correction_and_boundaries(u_value, v_value, gx, gy):
        corrected_u = u_value - dt * gx
        corrected_v = v_value - dt * gy
        corrected_u = corrected_u.at[0, 1:-1].set(solver.flow.U_inf)
        corrected_v = corrected_v.at[0, :].set(0.0)
        corrected_u = corrected_u.at[-1, :].set(corrected_u[-2, :])
        corrected_v = corrected_v.at[-1, :].set(corrected_v[-2, :])
        corrected_u = corrected_u.at[:, 0].set(0.0)
        corrected_u = corrected_u.at[:, -1].set(0.0)
        corrected_v = corrected_v.at[:, 0].set(0.0)
        corrected_v = corrected_v.at[:, -1].set(0.0)
        return corrected_u, corrected_v

    def obstacle_mask(u_value, v_value):
        return jnp.where(mask > 0.5, u_value, 0.0), jnp.where(mask > 0.5, v_value, 0.0)

    def standalone_penalization(u_value, v_value):
        chi = jax.nn.sigmoid(-solver.sdf / (0.1 * dx))
        denominator = 1.0 + dt * chi / solver.sim_params.brinkman_eta
        return (
            jnp.where(mask > 0.5, u_value / denominator, 0.0),
            jnp.where(mask > 0.5, v_value / denominator, 0.0),
        )

    phases = {
        "rk3_advection_diffusion_penalization": measure(advection, (u, v), args.repeats),
        "divergence_operator": measure(differential_operators, (u_star, v_star), args.repeats),
        "pressure_multigrid": measure(pressure, (rhs,), args.repeats),
        "pressure_gradient": measure(pressure_gradient, (pressure_value,), args.repeats),
        "pressure_correction_and_boundaries": measure(
            correction_and_boundaries, (u_star, v_star, dp_dx, dp_dy), args.repeats
        ),
        "final_obstacle_mask": measure(obstacle_mask, (u_star, v_star), args.repeats),
        "standalone_penalization_diagnostic": measure(
            standalone_penalization, (u, v), args.repeats
        ),
        "full_fused_step": measure(solver._step_jit, (u, v, mask, dt, 0), args.repeats),
    }
    isolated_total = sum(
        value["median_ms"]
        for key, value in phases.items()
        if key not in ("full_fused_step", "standalone_penalization_diagnostic")
    )
    for key, value in phases.items():
        if key not in ("full_fused_step", "standalone_penalization_diagnostic"):
            value["isolated_share_percent"] = 100.0 * value["median_ms"] / isolated_total
    data = {
        "device": str(jax.devices()),
        "grid": [solver.grid.nx, solver.grid.ny],
        "repeats": args.repeats,
        "method": "median synchronized execution after warm-up; blocks compiled in isolation",
        "interpretation_limit": (
            "XLA fusion and memory reuse mean isolated block medians do not add to the fused step. "
            "Shares are normalized isolated costs, not a hardware-trace attribution."
        ),
        "penalization_limit": (
            "Penalization is fused into all RK stages. Its standalone diagnostic is intentionally "
            "excluded from the percentage sum because adding it would double count RK3 work."
        ),
        "phases": phases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
