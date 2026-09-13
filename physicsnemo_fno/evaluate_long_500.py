#!/usr/bin/env python3
"""Evaluate frozen absolute and residual checkpoints on the separate 520-frame campaign."""
import json
from pathlib import Path
import h5py, numpy as np, torch
from train_residual import (FIELDS, atomic_json, build_model, delta_stats, param_stats,
    residual_rollouts, summarize_residual, rollouts, summarize_rollouts)

ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/"config_residual.json").read_text())
pilot=Path(cfg["dataset_root"]); long_root=ROOT/"datasets/long_eval_500"; out=Path(cfg["output_root"])
norm=json.loads((pilot/"normalization-train.json").read_text());fm=np.asarray(norm["mean"],np.float32);fs=np.asarray(norm["std"],np.float32)
train_files=[str(p) for p in sorted((pilot/"train").glob("*.h5"))];pm,ps=param_stats(train_files);dm,ds=delta_stats(train_files)
files={s:[str(p) for p in sorted((long_root/s).glob("*.h5"))] for s in ("validation","test")}
device=torch.device("cuda")
res=build_model(cfg).to(device);res.load_state_dict(torch.load(out/"curriculum_final.pt",map_location=device,weights_only=False)["model"]);res.eval()
base=build_model(cfg).to(device);base.load_state_dict(torch.load(Path(cfg["baseline_run"])/"best.pt",map_location=device,weights_only=False)["model"]);base.eval()
h=[10,50,100,500]; raw={"residual":{},"absolute":{}}
for split,paths in files.items():
    raw["residual"][split]=residual_rollouts(res,paths,h,device,fm,fs,pm,ps,dm,ds)
    raw["absolute"][split]=rollouts(base,paths,h,device,fm,fs,pm,ps)
summary={kind:{s:(summarize_residual(raw[kind][s]) if kind=="residual" else summarize_rollouts(raw[kind][s])) for s in files} for kind in raw}
atomic_json(out/"long_500_raw.json",raw);atomic_json(out/"long_500_comparison.json",summary)
print(json.dumps(summary,indent=2))
