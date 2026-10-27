# -*- coding: utf-8 -*-
"""v12(w3) の**因果推論**（未来を一切見ない）— 通知v4.1のリアルタイム検証用。

## 何をするか

通常の推論は10秒クリップを丸ごとモデルに入れ、全フレームの出力を得る。これは
**判定時刻より後の音を見ている**ので、実機では成立しない。

ここでは各判定時刻 t = 0.1, 0.2, ..., 10.0 秒について
「その時点までの音声」を右詰め・先頭ゼロ埋めした10秒窓を作り、
**モデル出力の最終フレームだけ**を採用する。未来参照はゼロになる。

## なぜ v4.1 で測り直すのか

2026-07-19 に v9.2 + 通知v1 で同じ実験をしており、初通知の遅れは中央値 +0.0〜0.25s だった。
だが v1 は**距離の微分を使っていない**。v4.1 は距離の傾き（接近速度）と方位の傾きを使うので、
因果推論で1フレームあたりの推定が粗くなる影響を v1 より強く受ける可能性がある。
その大小は測らないと分からない。

## 出力

<logdir>/runs/<experiment_name>/submissions/<clip>.csv （既存推論と同一の6列形式
[frame,class,track=0,az,el,dist]）。1クリップ書けるたびに保存するので、
QoS lowで中断されても再投入すれば続きから走る。

## 安全性

読むのは FOA音声 と ckpt だけ。書くのは新しい experiment_name のディレクトリのみ。
既存の推論結果・_hdf5・データセットには一切触れない。
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
sys.path.insert(0, str(PS))        # sde_patch.py はPSELDNets直下にある
sys.path.insert(0, str(HERE))

DATASET = os.environ.get("CAUSAL_DATASET", "outdoor_siren_v12")
CKPT = os.environ["CAUSAL_CKPT"]
OUTNAME = os.environ.get("CAUSAL_OUT", "infer_v12_w3_causal_val")
LOGDIR = Path(os.environ.get("CAUSAL_LOGDIR",
                             Path.home() / "PSELDNets_logs"))
DSDIR = Path(os.environ.get("CAUSAL_DSDIR", PS / "datasets_v12"))
PREFIX = os.environ.get("CAUSAL_PREFIX", "fold2_")   # val = fold2
LIMIT = int(os.environ.get("CAUSAL_LIMIT", "0"))     # 0=全部（試走用）
BATCH = int(os.environ.get("CAUSAL_BATCH", "25"))

SR, N, HOP = 24000, 240000, 2400                     # 24kHz / 10秒 / 0.1秒

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()                      # 推論デコードもここで入る

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
netmiss = [k for k in missing if k.startswith("net.")]
assert not netmiss, f"netの重みが欠けている: {netmiss[:5]}"
model.eval().cuda()
NCLS = ds.num_classes
print(f"[causal] ckpt={CKPT}\n[causal] classes={NCLS} dataset={DSDIR}", flush=True)

# 確定評価セットなど、別ディレクトリの音を使うときは CAUSAL_FOA で上書きする
foa = (Path(os.environ["CAUSAL_FOA"]) if os.environ.get("CAUSAL_FOA")
       else DSDIR / DATASET / "foa")
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
    assert sr == SR and wav.shape[0] == N, f"{fn}: sr={sr} n={wav.shape[0]}"
    x = torch.from_numpy(wav.T.copy())
    nfr = N // HOP                                   # 100
    buf = torch.zeros(nfr, 4, N)
    for k in range(1, nfr + 1):                      # 右詰め・先頭ゼロ埋め
        buf[k - 1, :, N - k * HOP:] = x[:, :k * HOP]
    outs = []
    with torch.no_grad():
        for s in range(0, nfr, BATCH):
            feat = model.standardize(buf[s:s + BATCH].cuda())
            y = model.net(feat)["multi_accdoa"]
            outs.append(y[:, -1, :].float().cpu())   # **最終フレームだけ**
    causal = torch.cat(outs, dim=0)[None]
    sed, doa = du.get_multi_accdoa_labels(causal, NCLS, 0.5)
    dcase = du.multi_accdoa_to_dcase_format(sed[:, 0].numpy(), doa[:, 0].numpy(),
                                            nb_classes=NCLS)
    polar = du.convert_output_format_cartesian_to_polar(in_dict=dcase)
    tmp = subs / f"{fn[:-5]}.csv.part"
    du.write_output_format_file(str(tmp), polar)
    tmp.replace(subs / f"{fn[:-5]}.csv")             # 途中で落ちても壊れない
    if (ci + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  {ci+1}/{len(todo)}  {el:.0f}s  "
              f"残り約{el/(ci+1)*(len(todo)-ci-1)/60:.0f}分", flush=True)

# --- 連結（既存の val_all.csv と同じ7列形式）---
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
