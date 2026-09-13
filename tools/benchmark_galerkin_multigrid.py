"""Experimental Galerkin multigrid screening; production defaults are untouched."""

from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from pressure_solvers.multigrid_solver import poisson_multigrid
from tools.benchmark_multigrid import make_solver, representative_rhs, residual_metrics, laplacian_pressure


def restrict_average(x):
    return 0.25*(x[::2,::2]+x[1::2,::2]+x[::2,1::2]+x[1::2,1::2])


def prolong_constant(x):
    return jnp.repeat(jnp.repeat(x,2,axis=0),2,axis=1)


def fine_operator(p,dx,dy,flow_type):
    return laplacian_pressure(p,dx,dy,flow_type)


def galerkin_operator(p,level,dx,dy,flow_type):
    value=p
    for _ in range(level): value=prolong_constant(value)
    value=fine_operator(value,dx,dy,flow_type)
    for _ in range(level): value=restrict_average(value)
    return value


def galerkin_solve(rhs,dx,dy,flow_type="von_karman",cycles=2,levels=4,omega=0.8):
    """Matrix-free A_H=R A_h P with adjoint cell-centered transfers."""
    b=rhs.at[-1,:].set(0.0) if flow_type=="von_karman" else rhs
    if flow_type=="lid_driven_cavity": b=b-jnp.mean(b)
    diag0=-2.0/(dx*dx)-2.0/(dy*dy)

    def smooth(p,rhs_level,level,count):
        diag=diag0/(2**level)
        def body(_,state):
            q=state+omega*(rhs_level-galerkin_operator(state,level,dx,dy,flow_type))/diag
            if flow_type=="von_karman": q=q.at[-1,:].set(0.0)
            return q
        return jax.lax.fori_loop(0,count,body,p)

    def cycle(p,rhs_level,level):
        if level==levels-1: return smooth(p,rhs_level,level,20)
        p=smooth(p,rhs_level,level,2)
        residual=rhs_level-galerkin_operator(p,level,dx,dy,flow_type)
        correction=cycle(jnp.zeros_like(restrict_average(residual)),restrict_average(residual),level+1)
        p=p+prolong_constant(correction)
        return smooth(p,rhs_level,level,2)

    def body(_,p): return cycle(p,b,0)
    p=jax.lax.fori_loop(0,cycles,body,jnp.zeros_like(b))
    if flow_type=="lid_driven_cavity": p=p-p[p.shape[0]//2,p.shape[1]//2]
    return p


def spectral_bands(residual,rhs):
    power=np.abs(np.fft.rfft2(np.asarray(residual)))**2
    rhs_power=max(float((np.abs(np.fft.rfft2(np.asarray(rhs)))**2).sum()),1e-30)
    kx=np.fft.fftfreq(residual.shape[0])[:,None]
    ky=np.fft.rfftfreq(residual.shape[1])[None,:]
    radius=np.sqrt(kx*kx+ky*ky)/np.sqrt(0.5)
    return {name:float(np.sqrt(power[(radius>=lo)&(radius<hi)].sum()/rhs_power))
            for name,lo,hi in (("low",0,.15),("mid",.15,.45),("high",.45,1.01))}


def time_solve(fn,rhs,repeats):
    out=fn(rhs); jax.block_until_ready(out); samples=[]
    for _ in range(repeats):
        t=time.perf_counter_ns(); out=fn(rhs); jax.block_until_ready(out)
        samples.append((time.perf_counter_ns()-t)/1e6)
    return out,statistics.median(samples)


def rhs_suite(shape,dx,dy):
    nx,ny=shape; x=jnp.arange(nx)[:,None]/nx; y=jnp.arange(ny)[None,:]/ny
    return {
        "smooth":jnp.sin(jnp.pi*x)*jnp.cos(2*jnp.pi*y),
        "single_fourier":jnp.sin(4*jnp.pi*x)*jnp.cos(6*jnp.pi*y),
        "multiscale":jnp.sin(jnp.pi*x)*jnp.cos(2*jnp.pi*y)+.3*jnp.sin(16*jnp.pi*x)*jnp.cos(12*jnp.pi*y)+.1*jnp.sin(64*jnp.pi*x),
        "high_frequency":jnp.sin(96*jnp.pi*x)*jnp.cos(64*jnp.pi*y),
    }


def evaluate(name,rhs,solver,repeats):
    dx,dy=solver.grid.dx,solver.grid.dy; ft=solver.sim_params.flow_type
    configs={
        "legacy_5":lambda b:poisson_multigrid(b,solver.mask,dx,dy,v_cycles=5,flow_type=ft),
        "rediscretized_2":lambda b:poisson_multigrid(b,solver.mask,dx,dy,v_cycles=2,flow_type=ft,level_scaled_spacing=True),
        "galerkin_pc_2_w06":lambda b:galerkin_solve(b,dx,dy,ft,2,4,.6),
        "galerkin_pc_2_w08":lambda b:galerkin_solve(b,dx,dy,ft,2,4,.8),
        "galerkin_pc_3_w08":lambda b:galerkin_solve(b,dx,dy,ft,3,4,.8),
    }
    rows={}
    metric_rhs=rhs.at[-1,:].set(0.0) if ft=="von_karman" else rhs
    for key,raw in configs.items():
        fn=jax.jit(raw); p,ms=time_solve(fn,rhs,repeats)
        residual=metric_rhs-laplacian_pressure(p,dx,dy,ft)
        rows[key]={"ms":ms,**residual_metrics(metric_rhs,p,dx,dy,ft),
                   "spectral":spectral_bands(jax.device_get(residual),jax.device_get(metric_rhs))}
    legacy=rows["legacy_5"]
    for key,row in rows.items():
        row["passes"]=(row["relative_l2"]<=legacy["relative_l2"]*1.001 and
                       all(row["spectral"][b]<=legacy["spectral"][b]*1.05+1e-12 for b in ("low","mid","high")))
    return {"rhs":name,"rows":rows}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--repeats",type=int,default=15)
    args=ap.parse_args(); solver=make_solver(); _,_,evolved=representative_rhs(solver)
    suite=rhs_suite(evolved.shape,solver.grid.dx,solver.grid.dy); suite["naca_initial"]=evolved
    result={"device":str(jax.devices()),"grid":[solver.grid.nx,solver.grid.ny],
            "transfer":{"R":"2x2 cell average","P":"piecewise constant","operator":"matrix-free R A_h P"},
            "screening":[evaluate(name,rhs,solver,args.repeats) for name,rhs in suite.items()]}
    passed=set.intersection(*[{k for k,v in case["rows"].items() if v["passes"]} for case in result["screening"]])
    result["passed_all_synthetic_and_initial"] = sorted(passed-{"legacy_5"})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
