"""Long baseline/corrected-hierarchy validation without changing production defaults."""

from __future__ import annotations

import argparse, json, math, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import solver.simulation.step_handlers as step_handlers
from advection_schemes.rk3_simple_new import rk_step_unified
from pressure_solvers.multigrid_solver import poisson_multigrid as legacy_multigrid
from solver import BaselineSolver, FlowConstraints, FlowParams, GeometryParams, GridParams, SimulationParams
from solver.metrics import compute_forces_ibm
from solver.operators import divergence_nonperiodic, vorticity_nonperiodic
from tools.benchmark_multigrid import residual_metrics


def corrected_multigrid(rhs, mask, dx, dy, v_cycles=5, tolerance=1e-6, flow_type="von_karman"):
    del v_cycles
    return legacy_multigrid(rhs, mask, dx, dy, v_cycles=2, tolerance=tolerance,
                            flow_type=flow_type, level_scaled_spacing=True)


CASES = {
    "naca_standard": dict(nx=512, ny=192, lx=20.0, ly=7.5, re=3000.0, dt=0.001,
                          flow_type="von_karman", obstacle="naca_airfoil", radius=0.5),
    "cylinder_vk": dict(nx=256, ny=128, lx=20.0, ly=7.5, re=200.0, dt=0.001,
                        flow_type="von_karman", obstacle="cylinder", radius=0.5),
    "lid_driven_cavity": dict(nx=128, ny=128, lx=1.0, ly=1.0, re=1000.0, dt=0.0002,
                              flow_type="lid_driven_cavity", obstacle="cylinder", radius=0.1),
    "high_re_naca": dict(nx=512, ny=192, lx=20.0, ly=7.5, re=5000.0, dt=0.0005,
                         flow_type="von_karman", obstacle="naca_airfoil", radius=0.5),
}

TOLERANCES = {
    "kinetic_energy_relative": 0.01,
    "divergence_l2_relative": 0.05,
    "mass_imbalance_absolute": 0.02,
    "velocity_l2_relative": 0.01,
    "velocity_linf_absolute": 0.05,
    "cl_mean_relative_or_absolute": 0.05,
    "cd_mean_relative_or_absolute": 0.05,
    "strouhal_relative": 0.05,
}


def make_case(spec, candidate=False):
    grid = GridParams(spec["nx"], spec["ny"], spec["lx"], spec["ly"])
    length = 2.0 * spec["radius"] if spec["obstacle"] == "cylinder" else 3.0
    flow = FlowParams(U_inf=1.0, nu=length / spec["re"], Re=spec["re"], L_char=length,
                      constraints=FlowConstraints(lock_U=True, lock_nu=True, lock_Re=False))
    geom = GeometryParams(0.25 * spec["lx"], 0.5 * spec["ly"], spec["radius"])
    params = SimulationParams(flow_type=spec["flow_type"], obstacle_type=spec["obstacle"],
                              pressure_solver="multigrid", multigrid_v_cycles=2 if candidate else 5,
                              adaptive_dt=False, fixed_dt=spec["dt"], use_les=False)
    return BaselineSolver(grid, flow, geom, params, dt=spec["dt"])


def build_pair(spec):
    step_handlers.poisson_multigrid = legacy_multigrid
    baseline = make_case(spec, False)
    baseline._jit_cache.clear(); baseline._step_jit = baseline.get_step_jit()
    step_handlers.poisson_multigrid = corrected_multigrid
    candidate = make_case(spec, True)
    candidate._jit_cache.clear(); candidate._step_jit = candidate.get_step_jit()
    step_handlers.poisson_multigrid = legacy_multigrid
    return baseline, candidate


def advance(solver, count):
    samples=[]
    for _ in range(count):
        t=time.perf_counter_ns()
        solver.u, solver.v, solver.current_pressure = solver._step_jit(
            solver.u, solver.v, solver.mask, solver.dt, solver.iteration)
        solver.iteration += 1
        jax.block_until_ready((solver.u, solver.v, solver.current_pressure))
        samples.append((time.perf_counter_ns()-t)/1e6)
    return samples


def norms(delta, area):
    a=np.asarray(delta)
    return {"l2": float(np.sqrt(np.sum(a*a)*area)), "linf": float(np.max(np.abs(a)))}


def observables(solver, with_forces):
    u,v,p,mask=jax.device_get((solver.u,solver.v,solver.current_pressure,solver.mask))
    dx,dy=solver.grid.dx,solver.grid.dy; area=dx*dy
    div=np.asarray(jax.device_get(divergence_nonperiodic(solver.u,solver.v,dx,dy)))
    fluid=np.asarray(mask)>0.5
    out={
        "kinetic_energy": float(0.5*np.sum((u*u+v*v)*fluid)*area),
        "divergence_l2": float(np.sqrt(np.sum(div*div)*area)),
        "divergence_linf": float(np.max(np.abs(div))),
        "mass_imbalance": float(np.sum(div)*area),
        "finite": bool(np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(p).all()),
    }
    if with_forces:
        w=vorticity_nonperiodic(solver.u,solver.v,dx,dy)
        chord=2.0*float(solver.geom.radius) if solver.sim_params.obstacle_type=="cylinder" else float(solver.sim_params.naca_chord)
        try:
            cl,cd=compute_forces_ibm(solver.u,solver.v,w,solver.grid.X,solver.grid.Y,solver.mask,
                dx,dy,solver.flow.U_inf,chord,float(solver.geom.center_x),float(solver.geom.center_y),solver.grid.lx)
            out.update(cl=cl,cd=cd)
        except Exception:
            out.update(cl=None,cd=None)
    return out


def pressure_check(solver, corrected):
    u_star,v_star=rk_step_unified(solver.u,solver.v,solver.dt,solver.flow.nu,solver.grid.dx,solver.grid.dy,
        solver.mask,sdf=solver.sdf,U_inf=solver.flow.U_inf,nu_hyper_ratio=solver.nu_hyper_ratio,
        slip_walls=solver.slip_walls,fast_mode=False,brinkman_eta=solver.sim_params.brinkman_eta,
        flow_type=solver.sim_params.flow_type)
    rhs=divergence_nonperiodic(u_star,v_star,solver.grid.dx,solver.grid.dy)/solver.dt
    solve=corrected_multigrid if corrected else legacy_multigrid
    p=solve(rhs,solver.mask,solver.grid.dx,solver.grid.dy,v_cycles=2 if corrected else 5,
            tolerance=solver.sim_params.pressure_tolerance,flow_type=solver.sim_params.flow_type)
    jax.block_until_ready(p)
    metric_rhs = rhs.at[-1,:].set(0.0) if solver.sim_params.flow_type=="von_karman" else rhs
    if solver.sim_params.flow_type == "lid_driven_cavity": metric_rhs = metric_rhs - jnp.mean(metric_rhs)
    return residual_metrics(metric_rhs,
                            p,solver.grid.dx,solver.grid.dy,solver.sim_params.flow_type)


def dominant_strouhal(history, dt_sample, length):
    if len(history)<64: return None
    signal=np.asarray(history)-np.mean(history)
    spectrum=np.abs(np.fft.rfft(signal))**2; spectrum[0]=0
    freq=np.fft.rfftfreq(len(signal),d=dt_sample)[int(np.argmax(spectrum))]
    return float(freq*length)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-steps",type=int,default=10000); ap.add_argument("--sample-every",type=int,default=100)
    ap.add_argument("--cases",nargs="*",default=list(CASES)); args=ap.parse_args()
    result={"device":str(jax.devices()),"tolerances":TOLERANCES,"cases":{}}
    for name in args.cases:
        spec=CASES[name]; baseline,candidate=build_pair(spec)
        # Compile both before measured batches.
        for s in (baseline,candidate):
            tmp=s._step_jit(s.u,s.v,s.mask,s.dt,s.iteration); jax.block_until_ready(tmp)
        checkpoints=[]; b_times=[]; c_times=[]; b_cl=[]; c_cl=[]
        target=0; stable=True
        for checkpoint in (1000,5000,10000):
            if checkpoint>args.max_steps or not stable: break
            remaining=checkpoint-target
            while remaining:
                batch=min(args.sample_every,remaining)
                b_times.extend(advance(baseline,batch)); c_times.extend(advance(candidate,batch))
                bo=observables(baseline,name!="lid_driven_cavity")
                co=observables(candidate,name!="lid_driven_cavity")
                if bo.get("cl") is not None: b_cl.append(bo["cl"]); c_cl.append(co["cl"])
                stable=bo["finite"] and co["finite"]
                remaining-=batch
                if not stable: break
            area=baseline.grid.dx*baseline.grid.dy
            bo=observables(baseline,name!="lid_driven_cavity"); co=observables(candidate,name!="lid_driven_cavity")
            uerr=norms(candidate.u-baseline.u,area); verr=norms(candidate.v-baseline.v,area)
            perr=norms(candidate.current_pressure-baseline.current_pressure,area)
            checkpoints.append({"step":checkpoint,"baseline":bo,"candidate":co,
                "errors":{"u":uerr,"v":verr,"p":perr},
                "pressure_residual":{"baseline":pressure_check(baseline,False),"candidate":pressure_check(candidate,True)}})
            target=checkpoint
        length=2*spec["radius"] if spec["obstacle"]=="cylinder" else 3.0
        sample_dt=spec["dt"]*args.sample_every
        result["cases"][name]={"spec":spec,"completed_steps":target,"stable":stable,"checkpoints":checkpoints,
            "performance":{"baseline_ms":statistics.median(b_times),"candidate_ms":statistics.median(c_times),
                "speedup":statistics.median(b_times)/statistics.median(c_times),
                "baseline_step_s":1000/statistics.median(b_times),"candidate_step_s":1000/statistics.median(c_times),
                "baseline_rtf":spec["dt"]*1000/statistics.median(b_times),
                "candidate_rtf":spec["dt"]*1000/statistics.median(c_times)},
            "strouhal":{"baseline":dominant_strouhal(b_cl,sample_dt,length),
                         "candidate":dominant_strouhal(c_cl,sample_dt,length)}}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print(f"completed {name}: {target} steps, stable={stable}",flush=True)
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
