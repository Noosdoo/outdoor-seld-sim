# -*- coding: utf-8 -*-
"""長尺（60 秒など任意長）版 × 装着高さ入力（v17 系）: _causal_infer_long.py と同じ因果推論（10 秒の右詰め窓を 0.1 秒ずつ）に、
_causal_infer_v17.py と同じ height_patch（HEIGHT_COND=1・HEIGHT_TABLE/HEIGHT_DEFAULT/HEIGHT_OFFSET）を足したもの。2026-09-07。
長尺セット v1 はマイク 1.5 m 固定なので HEIGHT_TABLE="" ・ HEIGHT_DEFAULT=1.5 で使う。
出力: <logdir>/runs/<experiment_name>/submissions/<clip>.csv と val_all_causal.csv（frame 0..L/HOP-1）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
PS = Path(os.environ.get("PS", HERE))
sys.path.insert(0, str(PS / "src"))
sys.path.insert(0, str(PS))
sys.path.insert(0, str(HERE))

DATASET = os.environ.get("CAUSAL_DATASET", "outdoor_siren_v12")
CKPT = os.environ["CAUSAL_CKPT"]
OUTNAME = os.environ.get("CAUSAL_OUT", "infer_long_v17_causal")
LOGDIR = Path(os.environ.get("CAUSAL_LOGDIR", Path.home() / "PSELDNets_logs"))
DSDIR = Path(os.environ.get("CAUSAL_DSDIR", PS / "datasets_v12"))
PREFIX = os.environ.get("CAUSAL_PREFIX", "long_")
LIMIT = int(os.environ.get("CAUSAL_LIMIT", "0"))
BATCH = int(os.environ.get("CAUSAL_BATCH", "50"))

SR, N, HOP = 24000, 240000, 2400

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()

HEIGHT = os.environ.get("HEIGHT_COND", "0") == "1"
if HEIGHT:
    import height_patch  # noqa: E402
    height_patch.apply()

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

GlobalHydra.instance().clear()
initialize_config_dir(config_dir=str(PS / "configs"), version_base="1.3")
cfg = compose(config_name="infer.yaml",
              overrides=[f"experiment={DATASET}", "mode=test",
                         "model.kwargs.pretrained_path=null",
                         f"paths.dataset_dir={DSDIR}"])

from models.model_module import SELDModelModule  # noqa: E402
from utils.config import get_dataset  # noqa: E402
import utils.data_utilities as du  # noqa: E402

ds = get_dataset(dataset_name=DATASET, cfg=cfg)
model = SELDModelModule(cfg, ds, test_meta={})
model.setup("predict")
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
missing, _ = model.load_state_dict(sd, strict=False)
netmiss = [k for k in missing if k.startswith("net.") and not (HEIGHT and ".h_mlp." in k)]
if HEIGHT and any(".h_mlp." in k for k in missing):
    print("[height] ckpt に h_mlp が無い → ゼロ初期化のまま推論（無条件モデルと同一の出力）", flush=True)
assert not netmiss, f"netの重みが欠けている: {netmiss[:5]}"
model.eval().cuda()
NCLS = ds.num_classes
print(f"[causal] ckpt={CKPT}\n[causal] classes={NCLS} dataset={DSDIR}", flush=True)
if HEIGHT:
    print(f"[causal] height input ON: table={height_patch.TABLE} default={height_patch.DEFAULT} offset={height_patch.OFFSET}", flush=True)

foa = (Path(os.environ["CAUSAL_FOA"]) if os.environ.get("CAUSAL_FOA") else DSDIR / DATASET / "foa")
clips = sorted(p.name for p in foa.glob(f"{PREFIX}*.flac"))
if LIMIT:
    clips = clips[:LIMIT]
subs = LOGDIR / DATASET / "runs" / OUTNAME / "submissions"
subs.mkdir(parents=True, exist_ok=True)
todo = [c for c in clips if not (subs / f"{c[:-5]}.csv").exists()]
print(f"[causal] 対象 {len(clips):,} / 未処理 {len(todo):,} → {subs}", flush=True)

t0 = time.time()
for ci, fn in enumerate(todo):
    wav, sr = sf.read(str(foa / fn), dtype="float32")
    assert sr == SR and wav.shape[0] % HOP == 0, f"{fn}: sr={sr} n={wav.shape[0]}"
    L = wav.shape[0]
    x = torch.zeros(4, N + L)
    x[:, N:] = torch.from_numpy(wav.T.copy())
    nfr = L // HOP
    outs = []
    if HEIGHT:
        z_in = height_patch.norm(height_patch.lookup(fn[:-5]) + height_patch.OFFSET)   # 表に無ければ DEFAULT（長尺 v1 は 1.5 m）
    with torch.no_grad():
        for s in range(0, nfr, BATCH):
            ks = list(range(s + 1, min(s + BATCH, nfr) + 1))
            buf = torch.stack([x[:, k * HOP:k * HOP + N] for k in ks])
            xb = buf.cuda()
            if HEIGHT:
                getattr(model.net, "_orig_mod", model.net)._mic_z = torch.full((xb.shape[0],), z_in, device="cuda")
            feat = model.standardize(xb)
            y = model.net(feat)["multi_accdoa"]
            outs.append(y[:, -1, :].float().cpu())
    causal = torch.cat(outs, dim=0)[None]
    sed, doa = du.get_multi_accdoa_labels(causal, NCLS, 0.5)
    dcase = du.multi_accdoa_to_dcase_format(sed[:, 0].numpy(), doa[:, 0].numpy(), nb_classes=NCLS)
    polar = du.convert_output_format_cartesian_to_polar(in_dict=dcase)
    tmp = subs / f"{fn[:-5]}.csv.part"
    du.write_output_format_file(str(tmp), polar)
    tmp.replace(subs / f"{fn[:-5]}.csv")
    if (ci + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  {ci+1}/{len(todo)}  {el:.0f}s  残り約{el/(ci+1)*(len(todo)-ci-1)/60:.0f}分", flush=True)

out = subs.parent / "val_all_causal.csv"
n = 0
with open(out, "w") as w:
    for f in sorted(subs.glob("*.csv")):
        clip = f.stem
        for line in open(f):
            line = line.strip()
            if line:
                w.write(f"{clip},{line}\n")
                n += 1
print(f"[causal] wrote {out} ({n:,} lines, {time.time()-t0:.0f}s)", flush=True)
print("CAUSAL_INFER_DONE")
