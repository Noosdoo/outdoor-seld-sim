# -*- coding: utf-8 -*-
"""相手 A（変更票 #73）: 既存の部品の組み合わせ = PANNs（Cnn14_DecisionLevelMax・AudioSet 527 クラス・モノラル 32 kHz）で「音の種類」、
FOA の擬似強度ベクトル（Jarrett ら 2010: I ∝ W·[X, Y, Z] の短時間平均）で「向き」。距離は出さない（nan）。2026-10-20・Fable。

この研究と同じ条件で測る（#73・5）: 右詰め 10 s の窓を 0.1 s ごと（未来を見ない）・窓の終わりの時刻の出力・同じヒステリシス（入り／保持／対応幅）・
同じ 7 列の予測 csv（clip,frame,cls,track,az,el,dist）・来歴つき（_prediction_integrity）。
- 種類: 窓の W チャンネル × 固定ゲイン（BASELINE_GAIN_DB・Q4）→ 32 kHz にリサンプル → PANNs の framewise_output（10 ms 刻み・実際は 0.32 s ごとに変わる）の最終フレーム
  → 対応表（md/design/baselines_class_map_v1.json の A）で 9 クラスへ（複数は最大値）。対象外のクラスは出さない。
- 向き: 窓の最後の 0.1 s の FOA から I = mean(W·X, W·Y, W·Z)（帯域 100〜4000 Hz・4 次のバターワース）→ az = atan2(Iy, Ix)・el = atan2(Iz, √(Ix²+Iy²))。
  音源を分けられないので、同じ時刻に検出した全クラスに同じ向きを与える（古典的な方法の限界・注記）。
- ヒステリシス: 「確率 × 向きの単位ベクトル」を doa の形（3 トラック × (x,y,z,dist) × 9）に入れ、hysteresis_patch.apply（v17 と同じ関数）を掛ける。
  norm = 確率なので、入り 0.7／保持 0.3 がそのまま確率の閾値になる。トラックは 0 だけ（1 クラス 1 音源）。
環境変数: CAUSAL_FOA・CAUSAL_PREFIX／CAUSAL_GLOB・CAUSAL_LIMIT・CAUSAL_EXPECTED_CLIPS・CAUSAL_RUNS_DIR・CAUSAL_OUT・CAUSAL_SED_TH／STAY／ASSOC・
  BASELINE_GAIN_DB（既定 0）・BASELINE_CLASSMAP・PANNS_CKPT（Cnn14_DecisionLevelMax_mAP=0.385.pth の場所）・PANNS_VENDOR（panns_inference と torchlibrosa を置いた dir）・E2E
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
E2E = Path(os.environ.get("E2E", Path.home() / "research/outdoor_seld_e2e"))
sys.path.insert(0, str(HERE))
VENDOR = os.environ.get("PANNS_VENDOR")
if VENDOR:
    sys.path.insert(0, VENDOR)

import _prediction_integrity as integrity  # noqa: E402
import hysteresis_patch  # noqa: E402
from panns_inference.models import Cnn14_DecisionLevelMax  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
import torchaudio  # noqa: E402

OUTNAME = os.environ.get("CAUSAL_OUT", "infer_baselineA_panns_iv")
LOGDIR = Path(os.environ.get("CAUSAL_LOGDIR", Path.home() / "PSELDNets_logs"))
PREFIX = os.environ.get("CAUSAL_PREFIX", "val_syn_")
LIMIT = int(os.environ.get("CAUSAL_LIMIT", "0"))
BATCH = int(os.environ.get("CAUSAL_BATCH", "20"))
SED_TH = float(os.environ.get("CAUSAL_SED_TH", "0.5"))
SED_STAY = float(os.environ.get("CAUSAL_SED_STAY", str(SED_TH)))
SED_ASSOC = float(os.environ.get("CAUSAL_SED_ASSOC", "20"))
GAIN_DB = float(os.environ.get("BASELINE_GAIN_DB", "0"))
CLASSMAP = Path(os.environ.get("BASELINE_CLASSMAP", E2E / "md/design/baselines_class_map_v1.json"))
PANNS_CKPT = Path(os.environ["PANNS_CKPT"])
N_OURS, N_TRACK = 9, 3
SR, N, HOP = 24000, 240000, 2400
SR_PANNS = 32000
IV_BAND = (100.0, 4000.0)

CMJ = json.loads(CLASSMAP.read_text(encoding="utf-8"))
CM = CMJ["A_panns_audioset"]
SRC_OF = {int(v["ours_index"]): [int(i) for i in v["source_index"]] for v in CM["map"].values()}
N_SRC = int(CM["n_source_classes"])

# PANNs（panns_inference の Cnn14_DecisionLevelMax・重みは Zenodo 3987831・CC BY 4.0）
model = Cnn14_DecisionLevelMax(sample_rate=SR_PANNS, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=N_SRC)
ck = torch.load(str(PANNS_CKPT), map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval().cuda()
CK_SHA = integrity.sha256(PANNS_CKPT)
print(f"[baselineA] panns={PANNS_CKPT.name} sha={CK_SHA[:12]} gain={GAIN_DB:+.1f} dB sed={SED_TH}/{SED_STAY}/{SED_ASSOC}", flush=True)
resampler = torchaudio.transforms.Resample(SR, SR_PANNS).cuda()
sos = butter(4, [IV_BAND[0] / (SR / 2), IV_BAND[1] / (SR / 2)], btype="band", output="sos")

foa = Path(os.environ["CAUSAL_FOA"])
GLOB = os.environ.get("CAUSAL_GLOB", f"{PREFIX}*.flac")
clips = sorted(p.name for p in foa.glob(GLOB))
if LIMIT:
    clips = clips[:LIMIT]
expected_clips = int(os.environ.get("CAUSAL_EXPECTED_CLIPS", "0"))
if expected_clips and len(clips) != expected_clips:
    raise ValueError(f"input clip count {len(clips)} != expected {expected_clips}")
runs_root = Path(os.environ.get("CAUSAL_RUNS_DIR", LOGDIR / "outdoor_siren_v12" / "runs"))
subs = runs_root / OUTNAME / "submissions"
subs.mkdir(parents=True, exist_ok=True)
_run_lock = integrity.acquire_lock(subs.parent)
import panns_inference  # noqa: E402
code_paths = {Path(__file__).resolve(), Path(integrity.__file__).resolve(), Path(hysteresis_patch.__file__).resolve(), CLASSMAP.resolve(), Path(panns_inference.models.__file__).resolve()}
CODE_REAL = {integrity.logical_code_name(p, E2E): p for p in code_paths}
assert len(CODE_REAL) == len(code_paths)
META = {"schema": integrity.SCHEMA, "ckpt": str(PANNS_CKPT), "ckpt_sha256": CK_SHA, "ckpt_epoch": 0,
        "dataset": "panns_audioset", "dsdir": "", "foa": str(foa), "glob": GLOB, "prefix": PREFIX,
        "clips": clips, "audio_sha256": integrity.audio_manifest(foa, clips),
        "height": {"on": False, "table": "", "table_sha256": None, "offset": None, "default": None, "values": {}},
        "code_sha256": {integrity.logical_code_name(p, E2E): integrity.sha256(p) for p in sorted(code_paths)},
        "settings": {"kind": "baseline_A_panns_iv", "batch": BATCH, "gain_db": GAIN_DB, "panns": "Cnn14_DecisionLevelMax", "sr_panns": SR_PANNS,
                     "classmap": str(CLASSMAP), "classmap_sha256": integrity.sha256(CLASSMAP), "classmap_A": {str(k): v for k, v in SRC_OF.items()}, "absent": CM["absent"],
                     "iv_band_hz": list(IV_BAND), "iv_window_s": HOP / SR, "sed_threshold": SED_TH, "sed_stay": SED_STAY, "sed_assoc": SED_ASSOC,
                     "hysteresis": SED_STAY < SED_TH, "hysteresis_pool_min": hysteresis_patch.POOL_MIN, "distance": "none (nan)",
                     "sr": SR, "samples": N, "hop": HOP, "torch": str(torch.__version__), "numpy": np.__version__, "soundfile": sf.__version__}}
META = json.loads(json.dumps(META, default=str))
integrity.begin_run(subs.parent, META)
todo = [c for c in clips if not (subs / f"{c[:-5]}.csv").exists()]
print(f"[baselineA] 対象 {len(clips):,} / 未処理 {len(todo):,} → {subs}", flush=True)
GAIN = 10.0 ** (GAIN_DB / 20.0)
DEBUG = os.environ.get("BASELINE_DEBUG") == "1"


def intensity_dirs(wav4: np.ndarray) -> np.ndarray:
    """(N, 4) FOA → (nfr, 3) 各フレーム（最後の 0.1 s）の強度ベクトルの単位ベクトル（x, y, z）。帯域 100〜4000 Hz。"""
    f = sosfiltfilt(sos, wav4, axis=0).astype(np.float64)
    w, x, y, z = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    nfr = wav4.shape[0] // HOP
    out = np.zeros((nfr, 3))
    for k in range(nfr):
        s = slice(k * HOP, (k + 1) * HOP)
        v = np.array([(w[s] * x[s]).mean(), (w[s] * y[s]).mean(), (w[s] * z[s]).mean()])
        n = np.linalg.norm(v)
        out[k] = v / n if n > 0 else np.array([1.0, 0.0, 0.0])
    return out


t0 = time.time()
for ci, fn in enumerate(todo):
    wav, sr = sf.read(str(foa / fn), dtype="float32")
    assert sr == SR and wav.shape[0] == N, f"{fn}: sr={sr} n={wav.shape[0]}"
    nfr = N // HOP
    mono = torch.from_numpy(wav[:, 0].copy()) * GAIN                       # W チャンネル × 固定ゲイン
    buf = torch.zeros(nfr, N)
    for k in range(1, nfr + 1):
        buf[k - 1, N - k * HOP:] = mono[:k * HOP]                         # 右詰め・先頭ゼロ埋め（この研究と同じ）
    probs = []
    with torch.no_grad():
        for s in range(0, nfr, BATCH):
            xb = resampler(buf[s:s + BATCH].cuda())
            fw = model(xb)["framewise_output"]                             # (B, frames, 527)
            probs.append(fw[:, -1, :].float().cpu())                       # 窓の終わり = 判定時刻
    P = torch.cat(probs, dim=0).numpy()                                    # (nfr, 527)
    dirs = intensity_dirs(wav)                                              # (nfr, 3)
    doa9 = np.zeros((N_TRACK, nfr, 4 * N_OURS), dtype=np.float32)
    for ours, idxs in SRC_OF.items():
        p = P[:, idxs].max(axis=1)                                          # 複数は最大値
        for b in range(3):
            doa9[0, :, b * N_OURS + ours] = (p * dirs[:, b]).astype(np.float32)   # norm = 確率
    if DEBUG:
        mx = {CMJ["ours"][str(k)]: float(P[:, v].max()) for k, v in SRC_OF.items()}
        print(f"[debug] {fn}: 対象クラスの確率の最大 " + "・".join(f"{k}={v:.2f}" for k, v in mx.items()) + f"・全 527 の最大 {float(P.max()):.2f}", flush=True)
    sed9 = hysteresis_patch.apply(doa9, N_OURS, SED_TH, SED_STAY, SED_ASSOC)
    rows = []
    for k in range(nfr):
        for c in range(N_OURS):
            if sed9[0, k, c]:
                xx, yy, zz = (float(doa9[0, k, b * N_OURS + c]) for b in range(3))
                az = math.degrees(math.atan2(yy, xx)); el = math.degrees(math.atan2(zz, math.hypot(xx, yy)))
                rows.append(f"{k},{c},0,{az:.2f},{el:.2f},nan\n")
    tmp = subs / f"{fn[:-5]}.csv.part"
    tmp.write_text("".join(rows), encoding="utf-8")
    tmp.replace(subs / f"{fn[:-5]}.csv")
    if (ci + 1) % 25 == 0:
        el_ = time.time() - t0
        print(f"  {ci+1}/{len(todo)}  {el_:.0f}s  残り約{el_/(ci+1)*(len(todo)-ci-1)/60:.0f}分", flush=True)

if integrity.audio_manifest(foa, clips) != META["audio_sha256"]:
    raise RuntimeError("audio changed during inference")
if any(integrity.sha256(CODE_REAL[k]) != h for k, h in META["code_sha256"].items()):
    raise RuntimeError("inference code changed during execution")
done = integrity.publish(subs.parent, META)
_run_lock.close()
print(f"[baselineA] wrote {subs.parent / 'val_all_causal.csv'} ({done['n_rows']:,} lines / {done['n_files']:,} clips, sha256 {done['sha256'][:12]})", flush=True)
print("CAUSAL_INFER_DONE")
