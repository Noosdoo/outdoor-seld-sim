# -*- coding: utf-8 -*-
# 【派生・2026-09-15】_causal_infer_v17.py に「検出の度合い（ACCDOA ベクトルの長さ）の全保存」を足した版。CAUSAL_SCORES=1 のとき scores/<clip>.csv に
#   frame,track,cls,長さ,x,y,z,距離[m] を書く（長さ > CAUSAL_SCORE_MIN・既定 0.05）。それ以外は本番と同じ。確定評価には使わない（検出層の閾値掃引の材料）。
"""v17 版の因果推論 = v12 版 ＋ 装着高さの入力（HEIGHT_COND=1 で height_patch を適用し、クリップごとの mic_z を net に渡す。
HEIGHT_OFFSET で感度評価のずらし）。HEIGHT_COND 未設定なら v12 版と同一挙動。2026-09-05。

（以下 v12 版の説明）v12(w3) の**因果推論**（未来を一切見ない）— 通知v4.1のリアルタイム検証用。

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
PREFIX = os.environ.get("CAUSAL_PREFIX", "val_syn_")   # val（2026-09-22 改名。旧 fold2_）
LIMIT = int(os.environ.get("CAUSAL_LIMIT", "0"))     # 0=全部（試走用）
BATCH = int(os.environ.get("CAUSAL_BATCH", "25"))

SR, N, HOP = 24000, 240000, 2400                     # 24kHz / 10秒 / 0.1秒

import sde_patch  # noqa: E402

sde_patch.apply_train_patches()                      # 推論デコードもここで入る
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

ds = get_dataset(dataset_name=DATASET, cfg=cfg)
model = SELDModelModule(cfg, ds, test_meta={})
model.setup("predict")
import json  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
# The entry point is also copied into PSELDNets; find the helper in the source repo.
if not (HERE / "_prediction_integrity.py").exists():
    sys.path.insert(0, str(Path(os.environ.get("E2E", Path.home() / "research/outdoor_seld_e2e")) / "server_sde"))
import _prediction_integrity as integrity  # noqa: E402

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
netmiss = [k for k in missing if k.startswith("net.") and not (HEIGHT and ".h_mlp." in k)]   # v17: h_mlp 無し ckpt はゼロ初期化のまま（無条件モデルと同一）
if HEIGHT and any(".h_mlp." in k for k in missing):
    print("[height] ckpt に h_mlp が無い → ゼロ初期化のまま推論（無条件モデルと同一の出力）", flush=True)
assert not netmiss, f"netの重みが欠けている: {netmiss[:5]}"
model.eval().cuda()
NCLS = ds.num_classes
print(f"[causal] ckpt={CKPT}\n[causal] classes={NCLS} dataset={DSDIR}", flush=True)
if HEIGHT:
    print(f"[causal] height input ON: table={height_patch.TABLE} default={height_patch.DEFAULT} offset={height_patch.OFFSET}", flush=True)

# 確定評価セットなど、別ディレクトリの音を使うときは CAUSAL_FOA で上書きする
foa = (Path(os.environ["CAUSAL_FOA"]) if os.environ.get("CAUSAL_FOA")
       else DSDIR / DATASET / "foa")
GLOB = os.environ.get("CAUSAL_GLOB", f"{PREFIX}*.flac")          # v17: 部分集合の指定（感度評価は v15 val 1,800 本だけ）
clips = sorted(p.name for p in foa.glob(GLOB))
if LIMIT:
    clips = clips[:LIMIT]
expected_clips = int(os.environ.get("CAUSAL_EXPECTED_CLIPS", "0"))
if expected_clips and len(clips) != expected_clips:
    raise ValueError(f"input clip count {len(clips)} != expected {expected_clips}")
# A fresh runs root preserves unverifiable legacy outputs without deleting them.
runs_root = Path(os.environ.get("CAUSAL_RUNS_DIR", LOGDIR / DATASET / "runs"))
subs = runs_root / OUTNAME / "submissions"
subs.mkdir(parents=True, exist_ok=True)
SCORES = os.environ.get("CAUSAL_SCORES", "0") == "1"
SCORE_MIN = float(os.environ.get("CAUSAL_SCORE_MIN", "0.05"))
SCORES_DIR = subs.parent / "scores"
if SCORES:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[scores] 検出の度合いを保存: {SCORES_DIR} (長さ > {SCORE_MIN})", flush=True)
_run_lock = integrity.acquire_lock(subs.parent)  # held through inference and publication
code_paths = {Path(__file__).resolve(), Path(integrity.__file__).resolve(), Path(sde_patch.__file__).resolve()}
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
if height["table"]:
    import csv
    with open(height["table"], newline="", encoding="utf-8") as handle:
        table_values = {r["clip_id"]: float(r["mic_z"]) for r in csv.DictReader(handle) if r.get("mic_z")}
    if table_values != height["values"] or integrity.sha256(height["table"]) != height["table_sha256"]:
        raise RuntimeError("height table changed after loading")
CODE_REAL = {integrity.logical_code_name(p, PS): p for p in code_paths}
assert len(CODE_REAL) == len(code_paths), f"論理名が衝突: {sorted(str(p) for p in code_paths)}"
META = {"schema": integrity.SCHEMA, "ckpt": str(CKPT), "ckpt_sha256": CK_SHA, "ckpt_epoch": CK_EPOCH,
        "dataset": DATASET, "dsdir": str(DSDIR), "foa": str(foa), "glob": GLOB, "prefix": PREFIX,
        "clips": clips, "audio_sha256": integrity.audio_manifest(foa, clips), "height": height,
        # 監査 R1/A3（2026-09-09）: 鍵はジョブ固有の写しや置き場所に依らない論理名。実パスは CODE_REAL に持ち、終了時の検査に使う
        "code_sha256": {integrity.logical_code_name(p, PS): integrity.sha256(p) for p in sorted(code_paths)},
        # resolve=False: 合成した設定には ${hydra:...} の補間が残っており、HydraConfig 無しで resolve=True にすると
        # InterpolationResolutionError で落ちる（2026-09-08 14:16・ジョブ 4037 の e139 で実際に発生）。未解決のまま記録する
        "settings": {"cfg": OmegaConf.to_container(cfg, resolve=False), "batch": BATCH,
                     "sde": gain_patch.meta_settings({"distance_mode": sde_patch.DIST_MODE, "distance_weight": sde_patch.W_DIST,
                                                      "distance_scale": sde_patch.DIST_SCALE, "log_k": sde_patch.LOG_K}),
                     "sr": SR, "samples": N, "hop": HOP, "torch": str(torch.__version__),
                     "numpy": np.__version__, "soundfile": sf.__version__}}
META = json.loads(json.dumps(META, default=str))
integrity.begin_run(subs.parent, META)
todo = [c for c in clips if not (subs / f"{c[:-5]}.csv").exists()]


def _exec_record(event: str, **extra) -> None:
    """実行記録（依頼 2-3・Astra R2・2026-09-14）: GPU 名・ノード・ドライバ・cuDNN・ジョブ・処理した clip 数を run_meta とは**別の**
    `run_exec.jsonl` に追記する。identity（run_meta の digest）には入れないので、別の GPU で仕上げた run や再開を拒まない。
    再開区間ごとに 1 行ずつ足し、最後の GPU で上書きしない（複数 producer の履歴が残る）。"""
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
print(f"[causal] ckpt epoch={CK_EPOCH} sha256={CK_SHA[:12]} 対象 {len(clips):,} / 未処理 {len(todo):,} → {subs}", flush=True)

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
    if HEIGHT:
        z_in = height_patch.norm(height_patch.lookup(fn[:-5]) + height_patch.OFFSET)
    with torch.no_grad():
        for s in range(0, nfr, BATCH):
            xb = buf[s:s + BATCH].cuda()
            if HEIGHT:
                getattr(model.net, "_orig_mod", model.net)._mic_z = torch.full((xb.shape[0],), z_in, device="cuda")
            feat = model.standardize(xb)
            y = model.net(feat)["multi_accdoa"]
            outs.append(y[:, -1, :].float().cpu())   # **最終フレームだけ**
    causal = torch.cat(outs, dim=0)[None]
    if SCORES:
        # 検出層の閾値掃引用（2026-09-15・Astra A4/A5 で可逆化 2026-09-16）: 3 track × 全クラスの (長さ, x, y, z, 距離[m]) を、長さ > SCORE_MIN の行だけ保存。
        # 長さは本番と同じ torch float32 で計算し、x/y/z は float32 の値を桁落ちなく書く（repr）。sed05 = 本番の「長さ > 0.5」判定そのもの（torch）。
        # 本番の 0.5 での CSV はこの下でこれまでどおり書く（この保存は submissions/ とは別の scores/ に置き、publish に混ぜない）
        srows = []
        with torch.no_grad():
            for t3 in range(3):
                b = 4 * t3 * NCLS
                xs_t = causal[0, :, b:b + NCLS]; ys_t = causal[0, :, b + NCLS:b + 2 * NCLS]; zs_t = causal[0, :, b + 2 * NCLS:b + 3 * NCLS]
                nrm_t = torch.sqrt(xs_t ** 2 + ys_t ** 2 + zs_t ** 2)              # 本番 get_multi_accdoa_labels_sde と同じ式・同じ精度
                sed05_t = nrm_t > 0.5
                xs = xs_t.numpy(); ys = ys_t.numpy(); zs = zs_t.numpy(); nrm = nrm_t.numpy(); sed05 = sed05_t.numpy()
                dd = causal[0, :, b + 3 * NCLS:b + 4 * NCLS].numpy()
                for f_, c_ in zip(*np.where(nrm > SCORE_MIN)):
                    srows.append(f"{f_},{t3},{c_},{float(nrm[f_, c_])!r},{float(xs[f_, c_])!r},{float(ys[f_, c_])!r},{float(zs[f_, c_])!r},"
                                 f"{float(dd[f_, c_])!r},{int(sed05[f_, c_])}")
        stmp = SCORES_DIR / f"{fn[:-5]}.csv.part"
        stmp.write_text("\n".join(srows) + ("\n" if srows else ""), encoding="utf-8")
        stmp.replace(SCORES_DIR / f"{fn[:-5]}.csv")
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

# Refuse to publish if inputs, table, or code changed during this run.
if integrity.audio_manifest(foa, clips) != META["audio_sha256"]:
    raise RuntimeError("audio changed during inference")
if height["table"] and integrity.sha256(height["table"]) != height["table_sha256"]:
    raise RuntimeError("height table changed during inference")
if any(integrity.sha256(CODE_REAL[k]) != h for k, h in META["code_sha256"].items()):
    raise RuntimeError("inference code changed during execution")
done = integrity.publish(subs.parent, META)
_exec_record("publish", n_processed_here=len(todo), n_rows=done["n_rows"], n_files=done["n_files"], sha256=done["sha256"])
_run_lock.close()
print(f"[causal] wrote {subs.parent / 'val_all_causal.csv'} ({done['n_rows']:,} lines / {done['n_files']:,} clips, sha256 {done['sha256'][:12]})", flush=True)
if SCORES:
    # Astra A5: scores の一覧・行数・sha256・列の形式を manifest に残す（掃引側はこれを必須にして欠損・改変・不完全行を拒む）
    import hashlib
    man = {"schema": "frame,track,cls,norm,x,y,z,dist_raw,sed05", "score_min": SCORE_MIN, "ckpt_sha256": CK_SHA, "n_clips": len(clips), "files": {}}
    for c in clips:
        f = SCORES_DIR / f"{c[:-5]}.csv"
        if not f.is_file():
            raise RuntimeError(f"scores missing: {f}")
        data = f.read_bytes()
        n_rows = data.count(b"\n")
        bad = [ln for ln in data.decode("utf-8").splitlines() if ln and ln.count(",") != 8]
        if bad:
            raise RuntimeError(f"scores malformed: {f} ({len(bad)} rows)")
        man["files"][c[:-5]] = {"rows": n_rows, "sha256": hashlib.sha256(data).hexdigest()}
    (subs.parent / "scores_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"[scores] manifest: {len(man['files']):,} clips → {subs.parent / 'scores_manifest.json'}", flush=True)
print("CAUSAL_INFER_DONE")
