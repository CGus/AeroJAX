"""Integration tests: restart equivalence, geometry adapters and leakage rejection."""
import json, tempfile, sys
from pathlib import Path
import h5py
import numpy as np
import teacher_dataset as td

def main():
    repo=Path('/home/gus/AeroJAX');sys.path.insert(0,str(repo))
    import jax
    assert jax.default_backend()=='gpu'
    config=td.pilot(); config.update(frames=6,warmup=2,grid=[64,32,20.,7.5])
    case=next(td.cases_from(config)); source=td.source_manifest(repo)['sha256']
    with tempfile.TemporaryDirectory(prefix='teacher-test-') as tmp:
        root=Path(tmp)
        td.generate_one(case,repo,root/'restart',source,3)
        td.generate_one(case,repo,root/'restart',source)
        td.generate_one(case,repo,root/'continuous',source)
        name=f"train/{case['sequence_id']}.h5"
        with h5py.File(root/'restart'/name) as a,h5py.File(root/'continuous'/name) as b:
            for key in ('fields','time','iteration'): np.testing.assert_array_equal(a[key][:],b[key][:])
        bad=dict(case,dt=.002)
        try: td.generate_one(bad,repo,root/'restart',source)
        except ValueError: pass
        else: raise AssertionError('Changed resume config accepted')
        from PIL import Image
        arr=np.full((32,64),255,np.uint8);arr[10:20,10:20]=0
        png=root/'mask.png';Image.fromarray(arr).save(png)
        for family in ('ellipse','rectangle','solid_wall','png','mask','tesla_valve'):
            geom=dict(id=family,family=family,split='train',a=.8,b=.6)
            if family=='png':geom['path']=str(png)
            if family=='mask':
                np.save(root/'mask.npy',(arr.T/255).astype('float32'));geom['path']=str(root/'mask.npy')
            s=td.build_solver(dict(case,geometry=geom),repo)
            assert np.asarray(s.mask).shape==(64,32)
            output=s._step_jit(s.u,s.v,s.mask,s.dt,s.iteration)
            assert all(np.isfinite(x).all() for x in jax.device_get(output))
        config['geometries']=[dict(id='a',family='naca',split='train'),dict(id='b',family='naca',split='test')]
        try: list(td.cases_from(config))
        except ValueError: pass
        else: raise AssertionError('Family leakage accepted')
    print(json.dumps(dict(resume_bitwise_equal=True,changed_resume_rejected=True,
                         adapter_gpu_smokes=6,family_leakage_rejected=True,device=str(jax.devices()))))
if __name__=='__main__':main()
