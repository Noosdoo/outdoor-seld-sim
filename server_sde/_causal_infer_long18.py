# -*- coding: utf-8 -*-
"""長尺（60 秒 = 600 フレーム）の因果推論 v18 = _causal_infer_long17.py（10 秒の右詰め窓を 0.1 秒ずつ・装着高さ入力）に、
_causal_infer_v17.py と**同じ**検出の設定（入り CAUSAL_SED_TH／保持 CAUSAL_SED_STAY／対応幅 CAUSAL_SED_ASSOC のヒステリシス）と
**同じ**来歴（_prediction_integrity: run_meta.json・val_all_causal.done.json・排他ロック・完了の照合）を足したもの。2026-10-20・Astra RN20-01／RN20-04。

long17 との違い:
  - 検出の閾値は環境変数から読む（long17 は 0.5 固定で CAUSAL_SED_TH を読まなかった）。ヒステリシスは hysteresis_patch.apply（v17 と同じ関数）。
  - run_meta.json に frames（1 本のフレーム数・長尺 v1 は 600）を記録し、完了の照合（canonical_rows）はその上限で行う。
  - CAUSAL_EXPECTED_CLIPS（本数の契約）・CAUSAL_FRAMES（1 本のフレーム数の契約・既定 600）。本数・長さが違えば止まる。
  - 既存の CSV は run_meta が同一のときだけ再開に使う（begin_run の規則）。val_all_causal.csv は publish（submissions の全本・余分なファイルがあれば拒否）で作る。
使い方（環境変数）: CAUSAL_CKPT・CAUSAL_OUT・CAUSAL_FOA（長尺 v1 の foa）・CAUSAL_PREFIX=long_・HEIGHT_COND=1・HEIGHT_TABLE=""（1.5 m 固定）・
  CAUSAL_SED_TH=0.7・CAUSAL_SED_STAY=0.3・CAUSAL_EXPECTED_CLIPS=400・CAUSAL_FRAMES=600
"""
from __future__ import annotations

import json
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
OUTNAME = os.environ.get("CAUSAL_OUT", "infer_long_v18_causal")
LOGDIR = Path(os.environ.get("CAUSAL_LOGDIR", Path.home() / "PSELDNets_logs"))
DSDIR = Path(os.environ.get("CAUSAL_DSDIR", PS / "datasets_v12"))
PREFIX = os.environ.get("CAUSAL_PREFIX", "long_")
LIMIT = int(os.environ.get("CAUSAL_LIMIT", "0"))
BATCH = int(os.environ.get("CAUSAL_BATCH", "50"))
FRAMES = int(os.environ.get("CAUSAL_FRAMES", "600"))                      # 1 本のフレーム数（長尺 v1 = 60 s = 600）
SED_TH = float(os.environ.get("CAUSAL_SED_TH", "0.5"))                  # 既定は v17 と同じ（本番の台本は 0.7／0.3 を渡す）
SED_STAY = float(os.environ.get("CAUSAL_SED_STAY", str(SED_TH)))
SED_ASSOC = float(os.environ.get("CAUSAL_SED_ASSOC", "20"))

SR, N, HOP = 24000, 240000, 2400

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()
try:  # 本体 §3-2: 入力の倍率（SDE_INPUT_GAIN_DB・既定 0 = 何もしない）。standardize の直前に 1 回
    import gain_patch  # noqa: E402
except ImportError:  # ジョブの写しの置き場に無い（旧の台本）ときは E2E の server_sde から読む（既定 0 なら何もしないので旧の動きのまま）
    import os as _gos, sys as _gsys
    from pathlib import Path as _GPath
    _gsys.path.insert(0, str(_GPath(_gos.environ.get("E2E", _GPath.home() / "research/outdoor_seld_e2e")) / "server_sde"))
    import gain_patch  # noqa: E402
gain_patch.apply()

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
from omegaconf import OmegaConf  # noqa: E402

ds = get_dataset(dataset_name=DATASET, cfg=cfg)
model = SELDModelModule(cfg, ds, test_meta={})
model.setup("predict")
if not (HERE / "_prediction_integrity.py").exists():
    sys.path.insert(0, str(Path(os.environ.get("E2E", Path.home() / "research/outdoor_seld_e2e")) / "server_sde"))
import _prediction_integrity as integrity  # noqa: E402
import hysteresis_patch  # noqa: E402

CK_SHA = integrity.sha256(CKPT)
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
gain_patch.check_ckpt(_ck)                            # 本体 §3-2
if integrity.sha256(CKPT) != CK_SHA:
    raise RuntimeError("checkpoint changed during loading")
CK_EPOCH = _ck.get("epoch")
sd = _ck["state_dict"]
del _ck
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
missing, _ = model.load_state_dict(sd, strict=False)
netmiss = [k for k in missing if k.startswith("net.") and not (HEIGHT and ".h_mlp." in k)]
if HEIGHT and any(".h_mlp." in k for k in missing):
    print("[height] ckpt に h_mlp が無い → ゼロ初期化のまま推論（無条件モデルと同一の出力）", flush=True)
assert not netmiss, f"netの重みが欠けている: {netmiss[:5]}"
model.eval().cuda()
NCLS = ds.num_classes
print(f"[causal-long] ckpt={CKPT}\n[causal-long] classes={NCLS} dataset={DSDIR} frames={FRAMES} sed={SED_TH}/{SED_STAY}/{SED_ASSOC}", flush=True)
if HEIGHT:
    print(f"[causal-long] height input ON: table={height_patch.TABLE} default={height_patch.DEFAULT} offset={height_patch.OFFSET}", flush=True)

foa = (Path(os.environ["CAUSAL_FOA"]) if os.environ.get("CAUSAL_FOA") else DSDIR / DATASET / "foa")
GLOB = os.environ.get("CAUSAL_GLOB", f"{PREFIX}*.flac")
clips = sorted(p.name for p in foa.glob(GLOB))
if LIMIT:
    clips = clips[:LIMIT]
expected_clips = int(os.environ.get("CAUSAL_EXPECTED_CLIPS", "0"))
if expected_clips and len(clips) != expected_clips:
    raise ValueError(f"input clip count {len(clips)} != expected {expected_clips}")
runs_root = Path(os.environ.get("CAUSAL_RUNS_DIR", LOGDIR / DATASET / "runs"))
subs = runs_root / OUTNAME / "submissions"
subs.mkdir(parents=True, exist_ok=True)
_run_lock = integrity.acquire_lock(subs.parent)
code_paths = {Path(__file__).resolve(), Path(integrity.__file__).resolve(), Path(sde_patch.__file__).resolve(), Path(hysteresis_patch.__file__).resolve()}
for module in list(sys.modules.values()):
    fn = getattr(module, "__file__", None)
    if fn and str(fn).endswith(".py"):
        path = Path(fn).resolve()
        if path.is_relative_to((PS / "src").resolve()):
            code_paths.add(path)
if HEIGHT:
    code_paths.add(Path(height_patch.__file__).resolve())
height = {"on": HEIGHT, "table": height_patch.TABLE if HEIGHT else "",
          "offset": height_patch.OFFSET if HEIGHT else None,
          "default": height_patch.DEFAULT if HEIGHT else None}
height["table_sha256"] = integrity.sha256(height["table"]) if height["table"] else None
height["values"] = height_patch.load_table() if HEIGHT else {}
CODE_REAL = {integrity.logical_code_name(p, PS): p for p in code_paths}
assert len(CODE_REAL) == len(code_paths), f"論理名が衝突: {sorted(str(p) for p in code_paths)}"
META = {"schema": integrity.SCHEMA, "ckpt": str(CKPT), "ckpt_sha256": CK_SHA, "ckpt_epoch": CK_EPOCH,
        "dataset": DATASET, "dsdir": str(DSDIR), "foa": str(foa), "glob": GLOB, "prefix": PREFIX, "frames": FRAMES,
        "clips": clips, "audio_sha256": integrity.audio_manifest(foa, clips), "height": height,
        "code_sha256": {integrity.logical_code_name(p, PS): integrity.sha256(p) for p in sorted(code_paths)},
        "settings": {"cfg": OmegaConf.to_container(cfg, resolve=False), "batch": BATCH, "kind": "long_v18",
                     "sde": gain_patch.meta_settings({"distance_mode": sde_patch.DIST_MODE, "distance_weight": sde_patch.W_DIST,
                                                      "distance_scale": sde_patch.DIST_SCALE, "log_k": sde_patch.LOG_K}),
                     "sed_threshold": SED_TH, "sed_stay": SED_STAY, "sed_assoc": SED_ASSOC,
                     "hysteresis": SED_STAY < SED_TH, "hysteresis_pool_min": hysteresis_patch.POOL_MIN,
                     "sr": SR, "samples": N, "hop": HOP, "frames": FRAMES, "torch": str(torch.__version__),
                     "numpy": np.__version__, "soundfile": sf.__version__}}
META = json.loads(json.dumps(META, default=str))
integrity.begin_run(subs.parent, META)
todo = [c for c in clips if not (subs / f"{c[:-5]}.csv").exists()]


def _exec_record(event: str, **extra) -> None:
    import datetime
    import platform
    import subprocess
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        drv = "unknown"
    rec = {"event": event, "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "job": os.environ.get("SLURM_JOB_ID"), "node": platform.node(),
           "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"),
           "driver": drv, "cuda": torch.version.cuda, "cudnn": (torch.backends.cudnn.version() if torch.cuda.is_available() else None),
           "torch": str(torch.__version__), "code_copies": {k: str(v) for k, v in CODE_REAL.items()}, **extra}
    with open(subs.parent / "run_exec.jsonl", "a", encoding="utf-8") as h:
        h.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


_exec_record("begin", n_clips=len(clips), n_todo=len(todo), n_done_before=len(clips) - len(todo))
print(f"[causal-long] ckpt epoch={CK_EPOCH} sha256={CK_SHA[:12]} 対象 {len(clips):,} / 未処理 {len(todo):,} → {subs}", flush=True)

t0 = time.time()
for ci, fn in enumerate(todo):
    wav, sr = sf.read(str(foa / fn), dtype="float32")
    assert sr == SR and wav.shape[0] % HOP == 0, f"{fn}: sr={sr} n={wav.shape[0]}"
    L = wav.shape[0]
    nfr = L // HOP
    if nfr != FRAMES:
        raise ValueError(f"{fn}: frames {nfr} != CAUSAL_FRAMES {FRAMES}")
    x = torch.zeros(4, N + L)
    x[:, N:] = torch.from_numpy(wav.T.copy())
    outs = []
    if HEIGHT:
        z_in = height_patch.norm(height_patch.lookup(fn[:-5]) + height_patch.OFFSET)   # 表に無ければ DEFAULT（長尺 v1 は 1.5 m）
    with torch.no_grad():
        for s in range(0, nfr, BATCH):
            ks = list(range(s + 1, min(s + BATCH, nfr) + 1))
            buf = torch.stack([x[:, k * HOP:k * HOP + N] for k in ks])      # 右詰め 10 秒窓（先頭はゼロ）
            xb = buf.cuda()
            if HEIGHT:
                getattr(model.net, "_orig_mod", model.net)._mic_z = torch.full((xb.shape[0],), z_in, device="cuda")
            feat = model.standardize(xb)
            y = model.net(feat)["multi_accdoa"]
            outs.append(y[:, -1, :].float().cpu())                            # 最終フレームだけ
    causal = torch.cat(outs, dim=0)[None]
    sed, doa = du.get_multi_accdoa_labels(causal, NCLS, SED_TH)              # 入りの閾値（v17 と同じ）
    sed_np, doa_np = sed[:, 0].numpy(), doa[:, 0].numpy()
    if SED_STAY < SED_TH:
        sed_np = hysteresis_patch.apply(doa_np, NCLS, SED_TH, SED_STAY, SED_ASSOC)   # v17 と同じ関数・同じ規則（T = 600 でも同じ）
    dcase = du.multi_accdoa_to_dcase_format(sed_np, doa_np, nb_classes=NCLS)
    polar = du.convert_output_format_cartesian_to_polar(in_dict=dcase)
    tmp = subs / f"{fn[:-5]}.csv.part"
    du.write_output_format_file(str(tmp), polar)
    tmp.replace(subs / f"{fn[:-5]}.csv")
    if (ci + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  {ci+1}/{len(todo)}  {el:.0f}s  残り約{el/(ci+1)*(len(todo)-ci-1)/60:.0f}分", flush=True)

if integrity.audio_manifest(foa, clips) != META["audio_sha256"]:
    raise RuntimeError("audio changed during inference")
if height["table"] and integrity.sha256(height["table"]) != height["table_sha256"]:
    raise RuntimeError("height table changed during inference")
if any(integrity.sha256(CODE_REAL[k]) != h for k, h in META["code_sha256"].items()):
    raise RuntimeError("inference code changed during execution")
done = integrity.publish(subs.parent, META)
_exec_record("publish", n_processed_here=len(todo), n_rows=done["n_rows"], n_files=done["n_files"], sha256=done["sha256"])
_run_lock.close()
print(f"[causal-long] wrote {subs.parent / 'val_all_causal.csv'} ({done['n_rows']:,} lines / {done['n_files']:,} clips, sha256 {done['sha256'][:12]})", flush=True)
print("CAUSAL_INFER_DONE")
