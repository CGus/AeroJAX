"""Offline frozen AeroJAX teacher -> appendable HDF5 sequences (no GUI)."""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, os, signal, sys, time, traceback
from pathlib import Path
import numpy as np
import h5py

SCHEMA = 'aerojax.teacher.v1'

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)

def source_manifest(repo):
    files = {}
    for folder in ('solver', 'pressure_solvers', 'advection_schemes', 'obstacles', 'timestepping'):
        for p in sorted((repo / folder).rglob('*.py')):
            files[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {'sha256': digest(files), 'files': files}

def pilot():
    geometries = []
    for i, code in enumerate(('NACA 0012', 'NACA 2412', 'NACA 4415')):
        geometries.append(dict(id=f'naca-{i}', family='naca', split='train', code=code, aoa=3*i))
    for i, radius in enumerate((.4, .6, .8)):
        geometries.append(dict(id=f'cylinder-{i}', family='cylinder', split='train', radius=radius))
    for i in range(3):
        geometries.append(dict(id=f'ellipse-{i}', family='ellipse', split='validation', a=.7+.25*i, b=.3+.12*i))
        geometries.append(dict(id=f'rectangle-{i}', family='rectangle', split='test', a=.45+.2*i, b=.45+.15*i))
    return dict(schema=SCHEMA, seed=20260909, grid=[128,64,20.,7.5], frames=200,
                stride=2, warmup=20, dt=.001, geometries=geometries,
                conditions=[dict(U_inf=.75, Re=200.), dict(U_inf=1.25, Re=600.)])

def build_solver(case, repo):
    import jax
    from solver import BaselineSolver, GridParams, FlowParams, FlowConstraints, GeometryParams, SimulationParams
    nx,ny,lx,ly = case['grid']; g = case['geometry']; family=g['family']
    radius=g.get('radius', g.get('a', .5))
    length=.15*lx if family=='naca' else 2*radius
    c=case['condition']; nu=c.get('nu', c['U_inf']*length/c['Re'])
    params=SimulationParams(grid_type='collocated', advection_scheme='rk3', pressure_solver='multigrid',
                            multigrid_v_cycles=5, adaptive_dt=False, use_les=False,
                            obstacle_type='naca_airfoil' if family=='naca' else 'cylinder')
    if c.get('inlet_angle_deg', 0) != 0:
        raise ValueError('Nonzero inlet direction is not supported by the frozen horizontal-inlet teacher; use geometry AoA')
    if family=='naca': params.naca_airfoil=g['code']; params.naca_angle=g.get('aoa',0.)
    if family in ('ellipse','rectangle','solid_wall','png','mask','tesla_valve'):
        from PIL import Image
        if family in ('png','tesla_valve'):
            path=Path(g.get('path', str(repo/'assets/tesla_valve_mask.png')))
            image=Image.open(path).convert('L').resize((nx,ny),Image.Resampling.NEAREST)
            mask=(np.asarray(image).T>=128).astype(np.float32)
        elif family=='mask':
            mask=np.load(g['path'], allow_pickle=False).astype(np.float32)
            if mask.shape!=(nx,ny) or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
                raise ValueError('Mask must be finite, shape [nx,ny], and in [0,1]')
        else:
            x,y=np.meshgrid(np.linspace(0,lx,nx),np.linspace(0,ly,ny),indexing='ij')
            x=x-g.get('center_x',.25*lx); y=y-g.get('center_y',.5*ly)
            theta=np.deg2rad(g.get('aoa',0.)); xx=np.cos(theta)*x+np.sin(theta)*y; yy=-np.sin(theta)*x+np.cos(theta)*y
            a,b=g.get('a',.2),g.get('b',.3*ly)
            solid=(xx/a)**2+(yy/b)**2<=1 if family=='ellipse' else (abs(xx)<=a)&(abs(yy)<=b)
            mask=(~solid).astype(np.float32)
        if mask.min()==mask.max(): raise ValueError('Geometry raster is empty or blocks the entire domain')
        params.obstacle_type='custom'; params.custom_mask=mask; params.grayscale_penalization=1-mask
    elif family not in ('naca','cylinder'):
        raise ValueError(f'Unknown geometry family: {family}')
    with contextlib.redirect_stdout(io.StringIO()):
        solver=BaselineSolver(GridParams(nx,ny,lx,ly),
            FlowParams(U_inf=c['U_inf'],nu=nu,Re=c.get('Re',c['U_inf']*length/nu),
                       constraints=FlowConstraints(True,True,False)),
            GeometryParams(.25*lx,.5*ly,radius),params,dt=case['dt'],seed=case['seed'])
    assert solver.sim_params.multigrid_v_cycles==5 and solver.sim_params.grid_type=='collocated'
    assert solver.sim_params.pressure_solver=='multigrid'
    return solver

def simple_config(obj):
    return {k:v for k,v in vars(obj).items() if isinstance(v,(str,int,float,bool,type(None)))}

def init_file(path, case, s, source):
    from scipy.ndimage import distance_transform_edt
    mask=np.asarray(s.mask,dtype='float32'); fluid=mask>=.5
    sdf=np.asarray(s.sdf,dtype='float32') if s.sdf is not None else (
        distance_transform_edt(fluid,sampling=(s.grid.dx,s.grid.dy))-
        distance_transform_edt(~fluid,sampling=(s.grid.dx,s.grid.dy))).astype('float32')
    with h5py.File(path,'w') as f:
        f.attrs.update(schema=SCHEMA, case_hash=digest(case), case=json.dumps(case), source_sha256=source,
                       committed=0, status='partial', seconds_total=0., seconds_step=0.)
        meta=dict(flow=simple_config(s.flow), simulation=simple_config(s.sim_params),
                  geometry=simple_config(s.geom), dx=s.grid.dx,dy=s.grid.dy,
                  nu_hyper_ratio=s.nu_hyper_ratio,slip_walls=s.slip_walls,seed=s.seed,
                  sdf_source='native' if s.sdf is not None else 'raster_edt_metadata_only',
                  pressure_gauge='outlet_zero',inlet_direction=[1.,0.],
                  axis_order='frame,channel,x,y',channels=['u','v','p'],
                  geometry_hash=hashlib.sha256(mask.tobytes()).hexdigest())
        f.attrs['resolved']=json.dumps(meta)
        for name,value in [('mask',mask),('sdf',sdf),('x',np.asarray(s.grid.X[:,0])),('y',np.asarray(s.grid.Y[0,:]))]:
            f.create_dataset(name,data=value,compression='lzf')
        nx,ny=mask.shape
        f.create_dataset('fields',shape=(0,3,nx,ny),maxshape=(None,3,nx,ny),dtype='float32',
                         chunks=(1,3,nx,ny),compression='lzf',shuffle=True,fletcher32=True)
        for name,dtype in [('time','float64'),('dt','float64'),('iteration','int64'),('frame_index','int64')]:
            f.create_dataset(name,shape=(0,),maxshape=(None,),dtype=dtype,chunks=(64,))

def generate_one(case, repo, root, source, frame_budget=None):
    import jax
    import jax.numpy as jnp
    t0=time.perf_counter()
    path=root/case['geometry']['split']/f"{case['sequence_id']}.h5"; path.parent.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()): s=build_solver(case,repo)
    if not path.exists(): init_file(path,case,s,source)
    with h5py.File(path,'r+') as f:
        if f.attrs['case_hash']!=digest(case) or f.attrs['source_sha256']!=source:
            raise ValueError('Resume refused: case or teacher source changed')
        if json.loads(f.attrs['resolved'])['geometry_hash']!=hashlib.sha256(np.asarray(s.mask,dtype='float32').tobytes()).hexdigest():
            raise ValueError('Resume refused: rasterized geometry changed')
        n=int(f.attrs['committed'])
        for key in ('fields','time','dt','iteration','frame_index'): f[key].resize(n,axis=0)
        if n==case['frames']: return 'complete'
        if n:
            s.u,s.v,s.current_pressure=map(jnp.asarray,f['fields'][n-1])
            s.iteration=int(f['iteration'][n-1]); s.simulated_time=float(f['time'][n-1])
        else:
            for _ in range(case['warmup']):
                s.u,s.v,s.current_pressure=s._step_jit(s.u,s.v,s.mask,s.dt,s.iteration); s.iteration+=1
            s.simulated_time=case['warmup']*float(s.dt)
        jax.block_until_ready(s._step_jit(s.u,s.v,s.mask,s.dt,s.iteration))
        end=min(case['frames'],n+frame_budget) if frame_budget else case['frames']
        step_seconds=0.
        for i in range(n,end):
            started=time.perf_counter()
            for _ in range(case['stride']):
                s.u,s.v,s.current_pressure=s._step_jit(s.u,s.v,s.mask,s.dt,s.iteration); s.iteration+=1
            fields=np.asarray(jax.device_get(jnp.stack((s.u,s.v,s.current_pressure))),dtype='float32')
            step_seconds+=time.perf_counter()-started
            if not np.isfinite(fields).all(): raise FloatingPointError(f'Nonfinite frame {i}')
            # Payload is flushed before commit marker. Graceful stop resumes exactly.
            for key in ('fields','time','dt','iteration','frame_index'): f[key].resize(i+1,axis=0)
            f['fields'][i]=fields; f['time'][i]=s.iteration*float(s.dt)
            f['dt'][i]=float(s.dt); f['iteration'][i]=s.iteration; f['frame_index'][i]=i
            f.flush(); f.attrs['committed']=i+1; f.flush()
        f.attrs['seconds_total']=float(f.attrs['seconds_total'])+time.perf_counter()-t0
        f.attrs['seconds_step']=float(f.attrs['seconds_step'])+step_seconds
        f.attrs['status']='complete' if end==case['frames'] else 'partial'
        return str(f.attrs['status'])

def cases_from(config):
    groups={}
    for g in config['geometries']:
        group=g.get('leakage_group',g['family'])
        if group in groups and groups[group]!=g['split']: raise ValueError('Leakage group spans splits')
        if g['split'] not in ('train','validation','test'): raise ValueError('Invalid split')
        groups[group]=g['split']
        for cond in config['conditions']:
            case={k:config[k] for k in ('seed','grid','frames','stride','warmup','dt')}
            case.update(geometry=g,condition=cond)
            # Hash referenced assets, not only paths.
            if 'path' in g: case['asset_sha256']=hashlib.sha256(Path(g['path']).read_bytes()).hexdigest()
            case['sequence_id']=g['id']+'-'+digest(case)[:12]
            yield case

def summarize(root):
    summary=dict(schema=SCHEMA,sequences=[],splits={},fields={},finite=True,geometry_split_integrity=True)
    sums=np.zeros(3); squares=np.zeros(3); mins=np.full(3,np.inf); maxs=-mins; count=0; seen={}
    train_sums=np.zeros(3); train_squares=np.zeros(3); train_count=0
    for path in sorted(root.glob('*/*.h5')):
        with h5py.File(path,'r') as f:
            n=int(f.attrs['committed']); case=json.loads(f.attrs['case']); meta=json.loads(f.attrs['resolved'])
            split=case['geometry']['split']; geom_hash=meta['geometry_hash']
            if geom_hash in seen and seen[geom_hash]!=split: summary['geometry_split_integrity']=False
            seen[geom_hash]=split
            for i in range(n):
                a=f['fields'][i].astype('float64'); summary['finite'] &= bool(np.isfinite(a).all())
                sums+=a.sum(axis=(1,2)); squares+=(a*a).sum(axis=(1,2)); count+=a.shape[1]*a.shape[2]
                mins=np.minimum(mins,a.min(axis=(1,2))); maxs=np.maximum(maxs,a.max(axis=(1,2)))
                if split=='train': train_sums+=a.sum(axis=(1,2));train_squares+=(a*a).sum(axis=(1,2));train_count+=a.shape[1]*a.shape[2]
            summary['sequences'].append(dict(sequence_id=case['sequence_id'],geometry_id=case['geometry']['id'],
                family=case['geometry']['family'],split=split,frames=n,status=str(f.attrs['status']),
                U_inf=meta['flow']['U_inf'],Re=meta['flow']['Re'],nu=meta['flow']['nu'],
                seconds=float(f.attrs['seconds_total']),step_seconds=float(f.attrs['seconds_step']),
                bytes=path.stat().st_size,physical_time=float(f['time'][n-1]) if n else 0.,
                time_monotonic=bool(np.all(np.diff(f['time'][:n])>0)),path=str(path.relative_to(root))))
            summary['splits'][split]=summary['splits'].get(split,0)+n
    if count:
        mean=sums/count; std=np.sqrt(np.maximum(0,squares/count-mean*mean))
        summary['fields']={name:dict(min=mins[i],max=maxs[i],mean=mean[i],std=std[i]) for i,name in enumerate(('u','v','p'))}
    if train_count:
        mean=train_sums/train_count; std=np.sqrt(np.maximum(0,train_squares/train_count-mean*mean))
        atomic_json(root/'normalization-train.json',dict(mean=mean.tolist(),std=std.tolist(),source_split='train',count=train_count))
    summary['frames']=sum(s['frames'] for s in summary['sequences'])
    summary['bytes']=sum(s['bytes'] for s in summary['sequences'])
    summary['generation_seconds']=sum(s['seconds'] for s in summary['sequences'])
    summary['frame_s_including_io']=summary['frames']/max(summary['generation_seconds'],1e-9)
    atomic_json(root/'quality.json',summary); return summary

def main():
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}: stopping after committed HDF5 frame')
    signal.signal(signal.SIGTERM, stop)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pilot-config','generate','inspect'])
    p.add_argument('--repo',type=Path,default=Path.home()/'AeroJAX');p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--frame-budget',type=int)
    args=p.parse_args()
    if args.action=='pilot-config': atomic_json(args.output,pilot());return
    if args.action=='inspect': print(json.dumps(summarize(args.output),indent=2));return
    config=json.loads(args.config.read_text()); cases=list(cases_from(config))
    sys.path.insert(0,str(args.repo)); os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    if jax.default_backend()!='gpu': raise RuntimeError('GPU required: refusing silent CPU fallback')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with open(args.output/'.writer.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=source_manifest(args.repo)
        provenance=args.output/'teacher-source.json'
        if provenance.exists() and json.loads(provenance.read_text())['sha256']!=source['sha256']:
            raise ValueError('Campaign teacher changed')
        if (args.output/'campaign.json').exists() and json.loads((args.output/'campaign.json').read_text())!=config:
            raise ValueError('Campaign configuration changed: use a new output directory')
        atomic_json(provenance,source);atomic_json(args.output/'campaign.json',config)
        import jaxlib
        runtime=dict(jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=np.__version__,h5py=h5py.__version__,
                     devices=[str(d) for d in jax.devices()],device_kind=jax.devices()[0].device_kind)
        runtime_file=args.output/'runtime.json'
        if runtime_file.exists() and json.loads(runtime_file.read_text())!=runtime:
            raise ValueError('Campaign runtime changed: start a new output directory')
        atomic_json(runtime_file,runtime)
        failures=[]
        for case in cases[:args.limit]:
            try:
                status=generate_one(case,args.repo,args.output,source['sha256'],args.frame_budget)
                print(case['sequence_id'],status,flush=True)
            except (KeyboardInterrupt,SystemExit): raise
            except Exception as exc:
                failures.append(dict(sequence_id=case['sequence_id'],error=str(exc),traceback=traceback.format_exc()))
                with open(args.output/'failure-history.jsonl','a') as history:
                    history.write(json.dumps(failures[-1])+'\n')
                atomic_json(args.output/'failures.json',failures)
                print(case['sequence_id'],'FAILED',str(exc),flush=True)
        atomic_json(args.output/'failures.json',failures)
        quality=summarize(args.output)
        print(json.dumps({k:v for k,v in quality.items() if k!='sequences'},indent=2))
        if failures or not quality['finite'] or not quality['geometry_split_integrity']: raise SystemExit(1)

if __name__=='__main__':main()
