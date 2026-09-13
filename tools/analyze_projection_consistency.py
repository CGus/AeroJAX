"""Compare the Poisson stencil with the actual collocated divergence-gradient operator."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from pressure_solvers.multigrid_solver import poisson_multigrid
from solver.operators import divergence_nonperiodic,grad_x_nonperiodic,grad_y_nonperiodic
from tools.benchmark_multigrid import make_solver,representative_rhs,laplacian_pressure
from tools.benchmark_krylov_pressure import cg_pressure

def norm(x,dx,dy):
    a=np.asarray(jax.device_get(x));return {"l2":float(np.sqrt(np.sum(a*a)*dx*dy)),"linf":float(np.max(np.abs(a)))}
def main():
    a=argparse.ArgumentParser();a.add_argument("--output",type=Path,required=True);args=a.parse_args()
    s=make_solver();_,_,rhs=representative_rhs(s);dx,dy=s.grid.dx,s.grid.dy;rhs=rhs.at[-1,:].set(0)
    variants={
      "legacy_5":poisson_multigrid(rhs,s.mask,dx,dy,v_cycles=5,flow_type="von_karman"),
      "rediscretized_2":poisson_multigrid(rhs,s.mask,dx,dy,v_cycles=2,flow_type="von_karman",level_scaled_spacing=True),
      "cg_spd_100":cg_pressure(rhs,s.mask,dx,dy,"von_karman",100,1e-6)}
    out={"device":str(jax.devices()),"grid":[s.grid.nx,s.grid.ny],"variants":{}}
    for name,p in variants.items():
        nominal=rhs-laplacian_pressure(p,dx,dy,"von_karman")
        dg=divergence_nonperiodic(grad_x_nonperiodic(p,dx),grad_y_nonperiodic(p,dy),dx,dy)
        actual=rhs-dg
        mismatch=dg-laplacian_pressure(p,dx,dy,"von_karman")
        out["variants"][name]={"poisson_r_b_minus_Lp":norm(nominal,dx,dy),
            "projection_r_b_minus_DGp":norm(actual,dx,dy),"operator_mismatch_DG_minus_L":norm(mismatch,dx,dy)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
