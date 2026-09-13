"""Benchmark a boundary-consistent matrix-free CG pressure alternative."""

from __future__ import annotations
import argparse,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from pressure_solvers.multigrid_solver import poisson_multigrid
from tools.benchmark_multigrid import make_solver,representative_rhs,residual_metrics
from tools.benchmark_galerkin_multigrid import rhs_suite


def cg_pressure(rhs,mask,dx,dy,flow_type="von_karman",maxiter=100,tol=1e-6):
    del mask
    b=rhs
    if flow_type=="von_karman": b=b.at[-1,:].set(0.0)
    elif flow_type=="lid_driven_cavity": b=b-jnp.mean(b)
    ax=1/(dx*dx); ay=1/(dy*dy); diagonal=2*ax+2*ay
    def operator(p):
        original=p
        if flow_type=="von_karman": p=p.at[-1,:].set(0.0)
        elif flow_type=="lid_driven_cavity": p=p-jnp.mean(p)
        mode="edge" if flow_type in ("von_karman","lid_driven_cavity") else "wrap"
        q=jnp.pad(p,((1,1),(1,1)),mode=mode)
        value=-(ax*(q[2:,1:-1]+q[:-2,1:-1]-2*p)+ay*(q[1:-1,2:]+q[1:-1,:-2]-2*p))
        if flow_type=="von_karman": value=value.at[-1,:].set(original[-1,:])
        return value
    def precondition(r):
        z=r/diagonal
        if flow_type=="von_karman": z=z.at[-1,:].set(r[-1,:])
        return z
    p,_=cg(operator,-b,x0=jnp.zeros_like(b),tol=tol,maxiter=maxiter,M=precondition)
    if flow_type=="von_karman": p=p.at[-1,:].set(0.0)
    elif flow_type=="lid_driven_cavity": p=p-jnp.mean(p)
    return p


def timed(fn,rhs,repeats):
    p=fn(rhs);jax.block_until_ready(p);s=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();p=fn(rhs);jax.block_until_ready(p);s.append((time.perf_counter_ns()-t)/1e6)
    return p,statistics.median(s)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--repeats",type=int,default=10)
    a=ap.parse_args();solver=make_solver();_,_,initial=representative_rhs(solver)
    suite=rhs_suite(initial.shape,solver.grid.dx,solver.grid.dy);suite["naca_initial"]=initial
    rows=[]
    for name,rhs in suite.items():
        cases={"legacy_5":lambda x:poisson_multigrid(x,solver.mask,solver.grid.dx,solver.grid.dy,v_cycles=5,flow_type="von_karman")}
        for count in (50,100,200):cases[f"cg_spd_{count}"]=lambda x,n=count:cg_pressure(x,solver.mask,solver.grid.dx,solver.grid.dy,"von_karman",n,1e-6)
        result={}
        metric_rhs=rhs.at[-1,:].set(0.0)
        for key,raw in cases.items():
            p,ms=timed(jax.jit(raw),rhs,a.repeats);result[key]={"ms":ms,**residual_metrics(metric_rhs,p,solver.grid.dx,solver.grid.dy,"von_karman")}
        rows.append({"rhs":name,"rows":result})
    out={"device":str(jax.devices()),"grid":[solver.grid.nx,solver.grid.ny],"operator":"SPD outlet-eliminated -L with Jacobi preconditioner","screening":rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
