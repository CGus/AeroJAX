"""Synchronized fixed-dt benchmark; optional repository path permits before/after comparison."""
import os,sys,time,json,cProfile,pstats,argparse
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ap=argparse.ArgumentParser();ap.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));ap.add_argument('--backend',default='gpu');ap.add_argument('--repeats',type=int,default=3);args=ap.parse_args()
sys.path.insert(0,args.repo)
import jax,numpy as np
from solver import GridParams,FlowParams,FlowConstraints,GeometryParams,SimulationParams,BaselineSolver
jax.config.update('jax_platform_name',args.backend)
s=BaselineSolver(GridParams(512,192,20.,7.5),FlowParams(U_inf=1.,nu=.001,Re=3000.,constraints=FlowConstraints(True,True,False)),GeometryParams(5.,3.75,.5),SimulationParams(adaptive_dt=False,pressure_solver='multigrid',grid_type='collocated'),dt=.001)
u0,v0,p0=s.u,s.v,s.current_pressure

def cycle():
 if hasattr(s,'advance_steps'):
  s.advance_steps(8);u,v,w,_=s.visualization_fields()
 else:
  for _ in range(8):u,v,w,_=s.step_for_visualization()
 # Same displayed payload on both versions. Synchronization included.
 return jax.device_get((u,v,w,(u*u+v*v)**.5))
for _ in range(3):cycle()
results=[]
profile=cProfile.Profile()
for repeat in range(args.repeats):
 s.u,s.v,s.current_pressure=u0,v0,p0;s.iteration=0;s.simulated_time=0.
 if repeat==0:profile.enable()
 t=time.perf_counter()
 for _ in range(100):cycle()
 elapsed=time.perf_counter()-t
 if repeat==0:profile.disable()
 results.append({'seconds':elapsed,'steps_per_second':800/elapsed,'rtf':.8/elapsed,'finite':bool(np.isfinite(s.u).all())})
print('RESULT',json.dumps({'backend':str(jax.devices()),'grid':[512,192],'dt':.001,'steps':800,'batch':8,'runs':results}),flush=True)
if args.repeats:
 pstats.Stats(profile).sort_stats('cumtime').print_stats(12)
# Separate completed numerical work from derived-field extraction.
for _ in range(2):jax.block_until_ready(s._step_jit(s.u,s.v,s.mask,s.dt))
t=time.perf_counter()
for _ in range(20):
 out=s._step_jit(s.u,s.v,s.mask,s.dt);jax.block_until_ready(out)
print('isolated_kernel_ms',(time.perf_counter()-t)*1000/20)
