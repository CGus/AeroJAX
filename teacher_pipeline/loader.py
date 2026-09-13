"""Map-style reader accepted by torch DataLoader; torch is not required to import it."""
from pathlib import Path
import json
import h5py
import numpy as np

class TeacherPairs:
    def __init__(self, root, split='train', horizon=1):
        if split not in ('train','validation','test') or horizon<1: raise ValueError('Invalid split/horizon')
        self.root=Path(root); self.entries=[];self.horizon=horizon
        norm=json.loads((self.root/'normalization-train.json').read_text())
        self.mean=np.asarray(norm['mean'],dtype='float32')[:,None,None]
        self.std=np.maximum(np.asarray(norm['std'],dtype='float32')[:,None,None],1e-6)
        for path in sorted((self.root/split).glob('*.h5')):
            with h5py.File(path) as f:
                if f.attrs['status']=='complete':
                    self.entries.extend((path,i) for i in range(int(f.attrs['committed'])-horizon))
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        path,i=self.entries[index]
        # Open per access: no HDF5 handle crosses worker/fork boundaries.
        with h5py.File(path) as f:
            resolved=json.loads(f.attrs['resolved']);case=json.loads(f.attrs['case'])
            params=np.array([resolved['flow']['U_inf'],resolved['flow']['nu'],resolved['flow']['Re'],
                             case['geometry'].get('aoa',0.),f['dt'][i],f['time'][i+self.horizon]-f['time'][i]],dtype='float32')
            return dict(state=(f['fields'][i]-self.mean)/self.std,
                        target=(f['fields'][i+self.horizon]-self.mean)/self.std,
                        mask=f['mask'][:],sdf=f['sdf'][:],params=params,
                        sequence_id=case['sequence_id'],geometry_id=case['geometry']['id'],frame_index=i)
