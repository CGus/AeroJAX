#!/usr/bin/env python3
"""Train and evaluate an autoregressive PhysicsNeMo FNO on the frozen AeroJAX pilot."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from physicsnemo.models.fno.fno import FNO

FIELDS = ("u", "v", "p")
PARAMS = ("U_inf", "nu", "Re", "aoa", "dt", "delta_t", "inlet_x", "inlet_y", "dx", "dy")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True)); tmp.replace(path)

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_meta(path):
    with h5py.File(path,"r") as f:
        case=json.loads(f.attrs["case"]); r=json.loads(f.attrs["resolved"])
        return case,r,int(f.attrs["committed"]),f["fields"].shape

def raw_params(f, i):
    r=json.loads(f.attrs["resolved"]); c=json.loads(f.attrs["case"])
    flow=r["flow"]; inlet=r.get("inlet_direction",[1.,0.])
    return np.asarray([flow["U_inf"],flow["nu"],flow["Re"],c["geometry"].get("aoa",0.),
        f["dt"][i],f["time"][i+1]-f["time"][i],inlet[0],inlet[1],r["dx"],r["dy"]],np.float32)

def collect_manifest(root):
    out={}
    for split in ("train","validation","test"):
        out[split]=[]
        for p in sorted((root/split).glob("*.h5")):
            c,r,n,shape=read_meta(p)
            out[split].append({"path":str(p),"sha256":file_sha256(p),"sequence_id":c["sequence_id"],
                "geometry_id":c["geometry"]["id"],"family":c["geometry"]["family"],"frames":n,"shape":shape})
    ids=[{x["geometry_id"] for x in out[s]} for s in out]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]), "geometry leakage"
    return out

def param_stats(files):
    vals=[]
    for path in files:
        with h5py.File(path,"r") as f:
            for i in range(int(f.attrs["committed"])-1): vals.append(raw_params(f,i))
    a=np.stack(vals); mean=a.mean(0); std=a.std(0); std[std<1e-12]=1.
    return mean.astype(np.float32),std.astype(np.float32)

class Pairs(Dataset):
    def __init__(self, files, field_mean, field_std, param_mean, param_std):
        self.entries=[]; self.records=[]; self.fm=field_mean[:,None,None]; self.fs=field_std[:,None,None]
        self.pm=param_mean; self.ps=param_std
        for p in files:
            with h5py.File(p,"r") as f:
                n=int(f.attrs["committed"]); fields=f["fields"][:n].astype(np.float32)
                mask=f["mask"][:].astype(np.float32); sdf=f["sdf"][:].astype(np.float32)
                sdf=sdf/max(float(np.max(np.abs(sdf))),1e-6)
                params=np.stack([raw_params(f,i) for i in range(n-1)])
            rid=len(self.records); self.records.append((fields,mask,sdf,params))
            self.entries += [(rid,i) for i in range(n-1)]
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        rid,i=self.entries[idx]; fields,mask,sdf,params=self.records[rid]
        s=fields[i]; y=fields[i+1]; pn=(params[i]-self.pm)/self.ps
        nx,ny=mask.shape; bc=np.zeros((4,nx,ny),np.float32)
        bc[0,0,:]=1; bc[1,-1,:]=1; bc[2,:,0]=1; bc[3,:,-1]=1
        inp=np.concatenate([(s-self.fm)/self.fs,mask[None],sdf[None],
            np.broadcast_to(pn[:,None,None],(len(PARAMS),nx,ny)),bc],axis=0)
        return torch.from_numpy(inp.copy()),torch.from_numpy(((y-self.fm)/self.fs).copy()),torch.from_numpy(mask.copy())

def build_model(cfg):
    return FNO(in_channels=3+2+len(PARAMS)+4,out_channels=3,dimension=2,
        latent_channels=cfg["model"]["width"],num_fno_layers=cfg["model"]["layers"],
        num_fno_modes=cfg["model"]["modes"],padding=cfg["model"]["padding"],padding_type="constant",coord_features=True)

def gauged(pred_n, fm, fs, mask):
    raw=pred_n*fs+fm
    outlet=mask[:,-1,:]
    offset=(raw[:,2,-1,:]*outlet).sum(1)/outlet.sum(1).clamp_min(1)
    raw=torch.cat((raw[:,:2],raw[:,2:3]-offset[:,None,None,None]),1)
    return (raw-fm)/fs

def loss_fn(pred,target,mask):
    w=mask[:,None]; return (((pred-target)**2*w).sum()/w.sum().clamp_min(1)/3)

@torch.no_grad()
def eval_loader(model,loader,device,fm,fs):
    sums={f:{"sq":0.,"n":0,"linf":0.,"den":0.} for f in FIELDS}
    for x,y,m in loader:
        x=x.to(device); y=y.to(device); m=m.to(device)
        p=gauged(model(x),fm,fs,m); pr=p*fs+fm; tr=y*fs+fm; w=m[:,None]
        for j,f in enumerate(FIELDS):
            e=(pr[:,j]-tr[:,j])*m; sums[f]["sq"]+=float((e*e).sum()); sums[f]["n"]+=float(m.sum())
            sums[f]["linf"]=max(sums[f]["linf"],float(e.abs().max())); sums[f]["den"]+=float(((tr[:,j]*m)**2).sum())
    out={}
    for f,v in sums.items(): out[f]={"l2_rmse":math.sqrt(v["sq"]/v["n"]),"linf":v["linf"],"relative_l2":math.sqrt(v["sq"]/max(v["den"],1e-30))}
    sq=sum(v["sq"] for v in sums.values()); n=sum(v["n"] for v in sums.values()); den=sum(v["den"] for v in sums.values())
    out["total"]={"l2_rmse":math.sqrt(sq/n),"linf":max(v["linf"] for v in sums.values()),"relative_l2":math.sqrt(sq/den)}
    return out

def make_input(state, f, i, fm_np, fs_np, pm, ps):
    mask=f["mask"][:].astype(np.float32); sdf=f["sdf"][:].astype(np.float32); sdf/=max(np.abs(sdf).max(),1e-6)
    pn=(raw_params(f,min(i,len(f["dt"])-2))-pm)/ps; nx,ny=mask.shape; bc=np.zeros((4,nx,ny),np.float32)
    bc[0,0,:]=1;bc[1,-1,:]=1;bc[2,:,0]=1;bc[3,:,-1]=1
    return np.concatenate([(state-fm_np[:,None,None])/fs_np[:,None,None],mask[None],sdf[None],np.broadcast_to(pn[:,None,None],(len(PARAMS),nx,ny)),bc],0),mask

def spatial_metrics(pred,teacher,mask,dx,dy):
    out={}
    for j,name in enumerate(FIELDS):
        e=(pred[j]-teacher[j])*mask; den=np.sum((teacher[j]*mask)**2)
        out[name]={"l2_rmse":float(np.sqrt(np.sum(e*e)/np.sum(mask))),"linf":float(np.max(np.abs(e))),"relative_l2":float(np.sqrt(np.sum(e*e)/max(den,1e-30)))}
    ep=0.5*(pred[0]**2+pred[1]**2); et=0.5*(teacher[0]**2+teacher[1]**2)
    ke_p=float(np.sum(ep*mask)/np.sum(mask));ke_t=float(np.sum(et*mask)/np.sum(mask))
    dp=np.gradient(pred[0],dx,axis=0)+np.gradient(pred[1],dy,axis=1)
    dt=np.gradient(teacher[0],dx,axis=0)+np.gradient(teacher[1],dy,axis=1)
    out["kinetic_energy"]={"pred":ke_p,"teacher":ke_t,"relative_error":abs(ke_p-ke_t)/max(abs(ke_t),1e-30)}
    out["divergence"]={"pred_l2":float(np.sqrt(np.sum((dp*mask)**2)/np.sum(mask))),"teacher_l2":float(np.sqrt(np.sum((dt*mask)**2)/np.sum(mask))),"difference_l2":float(np.sqrt(np.sum(((dp-dt)*mask)**2)/np.sum(mask)))}
    return out

@torch.no_grad()
def rollouts(model, files, horizons, device, fm_np,fs_np,pm,ps):
    fm=torch.tensor(fm_np,device=device)[None,:,None,None];fs=torch.tensor(fs_np,device=device)[None,:,None,None]
    result={str(h):[] for h in horizons}
    for path in files:
        with h5py.File(path,"r") as f:
            n=int(f.attrs["committed"]); r=json.loads(f.attrs["resolved"])
            maxh=min(max(horizons),n-1); state=f["fields"][0].astype(np.float32); timeline=[]
            finite=True
            for step in range(1,maxh+1):
                inp,mask=make_input(state,f,step-1,fm_np,fs_np,pm,ps)
                x=torch.from_numpy(inp[None].copy()).to(device); mt=torch.from_numpy(mask[None].copy()).to(device)
                yn=gauged(model(x),fm,fs,mt); state=(yn*fs+fm)[0].cpu().numpy(); teacher=f["fields"][step].astype(np.float32)
                finite=finite and bool(np.isfinite(state).all())
                met=spatial_metrics(state,teacher,mask,r["dx"],r["dy"]); met["step"]=step; timeline.append(met)
                if step in horizons:
                    rel0=timeline[0]["u"]["relative_l2"]+timeline[0]["v"]["relative_l2"]+timeline[0]["p"]["relative_l2"]
                    reln=met["u"]["relative_l2"]+met["v"]["relative_l2"]+met["p"]["relative_l2"]
                    result[str(step)].append({"sequence":Path(path).stem,"finite":finite,"growth_ratio":reln/max(rel0,1e-30),"timeline":timeline.copy()})
    return result

def summarize_rollouts(raw):
    out={}
    for h,runs in raw.items():
        finals=[x["timeline"][-1] for x in runs]; d={"sequences":len(runs),"all_finite":all(x["finite"] for x in runs),"growth_ratio_mean":float(np.mean([x["growth_ratio"] for x in runs]))}
        for f in FIELDS: d[f]={k:float(np.mean([x[f][k] for x in finals])) for k in ("l2_rmse","linf","relative_l2")}
        d["kinetic_energy_relative_error"]=float(np.mean([x["kinetic_energy"]["relative_error"] for x in finals]))
        d["divergence_difference_l2"]=float(np.mean([x["divergence"]["difference_l2"] for x in finals]))
        out[h]=d
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--epochs",type=int);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());
    if args.epochs: cfg["training"]["epochs"]=args.epochs
    seed_all(cfg["seed"]); root=Path(cfg["dataset_root"]); out=Path(cfg["output_root"]);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda"); assert torch.cuda.is_available(); torch.cuda.reset_peak_memory_stats()
    manifest=collect_manifest(root); atomic_json(out/"dataset_manifest.json",manifest); atomic_json(out/"config.resolved.json",cfg)
    norm=json.loads((root/"normalization-train.json").read_text());fm_np=np.asarray(norm["mean"],np.float32);fs_np=np.asarray(norm["std"],np.float32)
    train_files=[x["path"] for x in manifest["train"]];val_files=[x["path"] for x in manifest["validation"]];test_files=[x["path"] for x in manifest["test"]]
    pm,ps=param_stats(train_files); atomic_json(out/"normalization.json",{"field_source":str(root/"normalization-train.json"),"field_mean":fm_np.tolist(),"field_std":fs_np.tolist(),"parameter_names":PARAMS,"parameter_mean":pm.tolist(),"parameter_std":ps.tolist(),"training_only":True})
    tr=Pairs(train_files,fm_np,fs_np,pm,ps);va=Pairs(val_files,fm_np,fs_np,pm,ps);te=Pairs(test_files,fm_np,fs_np,pm,ps)
    fm=torch.tensor(fm_np,device=device)[None,:,None,None];fs=torch.tensor(fs_np,device=device)[None,:,None,None]
    # Smoke test.
    smoke=build_model(cfg).to(device); opt=torch.optim.Adam(smoke.parameters(),lr=1e-3)
    smoke_loader=DataLoader(Subset(tr,range(min(cfg["smoke"]["pairs"],len(tr)))),batch_size=4,shuffle=False)
    smoke_losses=[]
    for k,(x,y,m) in zip(range(cfg["smoke"]["steps"]),smoke_loader):
        x=x.to(device);y=y.to(device);m=m.to(device);opt.zero_grad();p=gauged(smoke(x),fm,fs,m);l=loss_fn(p,y,m);l.backward();opt.step();smoke_losses.append(float(l))
    del smoke,opt;torch.cuda.empty_cache();atomic_json(out/"smoke.json",{"losses":smoke_losses,"passed":all(np.isfinite(smoke_losses))})
    # Explicit tiny-subset overfit.
    over=build_model(cfg).to(device);opt=torch.optim.Adam(over.parameters(),lr=2e-3); subset=Subset(tr,range(min(cfg["overfit"]["pairs"],len(tr))))
    ov=[]
    for k in range(cfg["overfit"]["steps"]):
        for x,y,m in DataLoader(subset,batch_size=len(subset),shuffle=False):
            x=x.to(device);y=y.to(device);m=m.to(device);opt.zero_grad();p=gauged(over(x),fm,fs,m);l=loss_fn(p,y,m);l.backward();opt.step();ov.append(float(l))
        if ov[-1] <= cfg["overfit"]["target_loss"]: break
    atomic_json(out/"overfit.json",{"initial_loss":ov[0],"final_loss":ov[-1],"steps":len(ov),"target":cfg["overfit"]["target_loss"],"strong_reduction_passed":ov[-1]<ov[0]*0.1,"target_reached":ov[-1]<=cfg["overfit"]["target_loss"]})
    del over,opt;torch.cuda.empty_cache()
    model=build_model(cfg).to(device); params=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["training"]["epochs"])
    gen=torch.Generator().manual_seed(cfg["seed"])
    tl=DataLoader(tr,batch_size=cfg["training"]["batch_size"],shuffle=True,num_workers=0,generator=gen,pin_memory=True)
    vl=DataLoader(va,batch_size=cfg["training"]["batch_size"],shuffle=False,num_workers=0,pin_memory=True)
    history=[];best=float("inf");bad=0;start=time.perf_counter();steps=0
    for epoch in range(1,cfg["training"]["epochs"]+1):
        model.train();acc=0.;seen=0;t0=time.perf_counter()
        for x,y,m in tl:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);m=m.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True);p=gauged(model(x),fm,fs,m);l=loss_fn(p,y,m);l.backward();opt.step();acc+=float(l)*x.shape[0];seen+=x.shape[0];steps+=1
        model.eval();vacc=0.;vn=0
        with torch.no_grad():
            for x,y,m in vl:
                x=x.to(device);y=y.to(device);m=m.to(device);l=loss_fn(gauged(model(x),fm,fs,m),y,m);vacc+=float(l)*x.shape[0];vn+=x.shape[0]
        row={"epoch":epoch,"train_loss":acc/seen,"validation_loss":vacc/vn,"seconds":time.perf_counter()-t0,"lr":opt.param_groups[0]["lr"]};history.append(row)
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,"config":cfg,"field_mean":fm_np,"field_std":fs_np,"param_mean":pm,"param_std":ps},out/"latest.pt")
        if row["validation_loss"]<best:
            best=row["validation_loss"];bad=0;torch.save({"model":model.state_dict(),"epoch":epoch,"config":cfg,"field_mean":fm_np,"field_std":fs_np,"param_mean":pm,"param_std":ps},out/"best.pt")
        else: bad+=1
        atomic_json(out/"history.json",history);sched.step()
        print(json.dumps(row),flush=True)
        if bad>=cfg["training"]["patience"]: break
    train_seconds=time.perf_counter()-start
    ck=torch.load(out/"best.pt",map_location=device,weights_only=False);model.load_state_dict(ck["model"]);model.eval()
    one={"validation":eval_loader(model,vl,device,fm,fs),"test":eval_loader(model,DataLoader(te,batch_size=cfg["training"]["batch_size"]),device,fm,fs)};atomic_json(out/"one_step.json",one)
    raw={}
    for split,files in (("validation",val_files),("test",test_files)):
        rr=rollouts(model,files,cfg["rollout_horizons"],device,fm_np,fs_np,pm,ps);raw[split]=rr;atomic_json(out/f"rollouts_{split}_raw.json",rr)
    roll={s:summarize_rollouts(raw[s]) for s in raw};atomic_json(out/"rollouts_summary.json",roll)
    sample=tr[0][0][None].to(device)
    with torch.no_grad():
        for _ in range(cfg["inference"]["warmup"]): model(sample)
        torch.cuda.synchronize();t0=time.perf_counter()
        for _ in range(cfg["inference"]["iterations"]): model(sample)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t0
    fps=cfg["inference"]["iterations"]/elapsed; teacher_frame_fps=cfg["teacher_measured_step_fps"]/cfg["teacher_export_stride"]
    perf={"training_seconds":train_seconds,"optimizer_steps":steps,"samples_per_second":len(tr)*len(history)/train_seconds,"peak_vram_bytes":torch.cuda.max_memory_allocated(),"inference_batch1_ms":1000/fps,"surrogate_frames_per_second":fps,"teacher_comparable_stored_frame_fps":teacher_frame_fps,"speedup_vs_teacher":fps/teacher_frame_fps,"checkpoint_bytes":(out/"best.pt").stat().st_size}
    env={"python":os.sys.version,"torch":torch.__version__,"torch_cuda":torch.version.cuda,"physicsnemo":__import__("physicsnemo").__version__,"gpu":torch.cuda.get_device_name(0),"parameters":params}
    atomic_json(out/"performance.json",perf);atomic_json(out/"environment.json",env)
    print(json.dumps({"complete":True,"best_epoch":ck["epoch"],"best_validation_loss":best,"performance":perf},indent=2))
if __name__=="__main__": main()
