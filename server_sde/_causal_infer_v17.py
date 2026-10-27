# -*- coding: utf-8 -*-
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
SED_TH = float(os.environ.get("CAUSAL_SED_TH", "0.5"))   # 検出の閾値（矢印の長さ・入り）。既定 0.5 = 従来。本番の 2 設定は 0.7（scripts/_notify_modes.py と同じ値にする）
SED_STAY = float(os.environ.get("CAUSAL_SED_STAY", str(SED_TH)))   # 保持の閾値（変更票 #34・2026-09-16 本人「①採用」= 入り 0.7／保持 0.3）。既定 = 入りと同じ = 1 閾値（従来と同一出力）
SED_ASSOC = float(os.environ.get("CAUSAL_SED_ASSOC", "20"))        # 前フレームとの対応幅（度）
# 本体 §3-5（§9-1 の 5・2026-10-26）: クラス別の検出のしきい値の表（md/design/thresholds_v30.json・threshold_table.py が確かめる）。未設定 = 旧（上の定数）。
#   CAUSAL_THRESHOLDS_MODE = hyst（表の入り／保持）か nohyst（入りは表・保持 = 入り＝アブレーション nohyst）。表と CAUSAL_SED_TH／STAY は同時に置かない。
THRESH_JSON = os.environ.get("CAUSAL_THRESHOLDS_JSON", "").strip()
THRESH_MODE = os.environ.get("CAUSAL_THRESHOLDS_MODE", "hyst").strip() or "hyst"
if THRESH_JSON and (os.environ.get("CAUSAL_SED_TH", "").strip() or os.environ.get("CAUSAL_SED_STAY", "").strip()):
    raise SystemExit("CAUSAL_THRESHOLDS_JSON と CAUSAL_SED_TH／CAUSAL_SED_STAY を同時に置かない（表か定数のどちらか・本体 §3-5）")
if os.environ.get("CAUSAL_THRESHOLDS_MODE", "").strip() and not THRESH_JSON:
    raise SystemExit("CAUSAL_THRESHOLDS_MODE は CAUSAL_THRESHOLDS_JSON（表）と一緒に置く")
# 本体 §3-5・§9-4 の 5: DUMP_NORMS=1 でヒステリシス前の norm（3 トラック・T・9・float32）を runs/<出力>/norms/<clip>.npy に間引かず全部保存し、
#   最後に norms_manifest.json（ckpt の sha・本ごとの sha）。しきい値の選択 scripts/v30_select_thresholds.py の入力（B・D の norms と同じ型）
DUMP_NORMS = os.environ.get("DUMP_NORMS", "0") == "1"

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
# 本体 §9-4 の 1（変更票 #75 決め直し 3）: ヒステリシスを frame HYST_START_FRAME から始める（それより前の枠は出力しない・状態もそこから）。既定 0 = 旧
HYST_START = int(os.environ.get("HYST_START_FRAME", "0") or 0)
HYST_START_EXPLICIT = os.environ.get("HYST_START_FRAME", "").strip() != ""   # 明示（0 でも）なら run_meta に記録・未設定なら旧のまま（Fable 10/26 16:2x）
assert 0 <= HYST_START < 100, f"HYST_START_FRAME は 0〜99: {HYST_START}"

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
import hysteresis_patch  # noqa: E402  検出のヒステリシス（入り／保持）。_prediction_integrity と同じ置き場所から読む

CK_SHA = integrity.sha256(CKPT)
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
gain_patch.check_ckpt(_ck)                            # 本体 §3-2: ckpt の倍率と SDE_INPUT_GAIN_DB が違えば止める（無い ckpt は 0）
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
TABLE = None
if THRESH_JSON:                                       # 本体 §3-5: 表を読んで確かめる（版・クラス・候補・保持の比・表を作った ckpt = 今の ckpt）
    try:
        import threshold_table  # noqa: E402
    except ImportError:
        sys.path.insert(0, str(Path(os.environ.get("E2E", Path.home() / "research/outdoor_seld_e2e")) / "server_sde"))
        import threshold_table  # noqa: E402
    TABLE = threshold_table.load(THRESH_JSON, NCLS, THRESH_MODE, ckpt_sha=CK_SHA)
    print(f"[causal] しきい値の表 {THRESH_JSON}（{TABLE['label']}・{THRESH_MODE}・sha {TABLE['sha256'][:12]}）: "
          f"{hysteresis_patch.describe(TABLE['enter'], TABLE['stay'], SED_ASSOC)}", flush=True)
ENTER, STAY = (TABLE["enter"], TABLE["stay"]) if TABLE is not None else (SED_TH, SED_STAY)
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
_run_lock = integrity.acquire_lock(subs.parent)  # held through inference and publication
NORMS_DIR = subs.parent / "norms"                 # 本体 §3-5: DUMP_NORMS の置き場（B・D と同じ runs/<出力>/norms）
if DUMP_NORMS:
    NORMS_DIR.mkdir(exist_ok=True)
code_paths = {Path(__file__).resolve(), Path(integrity.__file__).resolve(), Path(sde_patch.__file__).resolve(), Path(hysteresis_patch.__file__).resolve()}
if gain_patch.EXPLICIT:                              # 本体 §3-2: 倍率を明示した run だけ gain_patch の sha も残す（0 でも・未設定の run の identity は旧のまま）
    code_paths.add(Path(gain_patch.__file__).resolve())
if TABLE is not None:                                # 本体 §3-5: 表を使う run だけ threshold_table の sha も残す
    code_paths.add(Path(threshold_table.__file__).resolve())
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
                     "sed_threshold": SED_TH, "sed_stay": SED_STAY, "sed_assoc": SED_ASSOC,
                     "hysteresis": SED_STAY < SED_TH, "hysteresis_pool_min": hysteresis_patch.POOL_MIN,
                     "sr": SR, "samples": N, "hop": HOP, "torch": str(torch.__version__),
                     "numpy": np.__version__, "soundfile": sf.__version__}}
if HYST_START_EXPLICIT:                               # §9-4 の 1: 明示なら 0 でも鍵を足す（未設定の run の identity は変えない）
    META["settings"]["hyst_start_frame"] = HYST_START
if TABLE is not None:                                 # §3-5: 表を使う run だけ（定数の欄は None・表の sha・入り／保持・mode を記録）
    META["settings"].update(sed_threshold=None, sed_stay=None, hysteresis=hysteresis_patch.has_hysteresis(ENTER, STAY),
                            thresholds=threshold_table.record(TABLE))
if DUMP_NORMS:                                        # §3-5: norm を保存する run だけ鍵を足す
    META["settings"]["dump_norms"] = True
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
    sed, doa = du.get_multi_accdoa_labels(causal, NCLS, SED_TH)   # 入りの閾値（CAUSAL_SED_TH）。従来の 1 閾値の判定
    sed_np, doa_np = sed[:, 0].numpy(), doa[:, 0].numpy()
    if DUMP_NORMS:                                   # §3-5: ヒステリシス前の norm を全部（書きかけを残さない）
        _np_tmp = NORMS_DIR / f"{fn[:-5]}.npy.part"
        with open(_np_tmp, "wb") as _h:
            np.save(_h, hysteresis_patch.norms(doa_np, NCLS))
        os.replace(_np_tmp, NORMS_DIR / f"{fn[:-5]}.npy")
    if TABLE is not None:
        # §3-5: クラス別の入り／保持（表）。入りだけの判定もクラス別に作り直す（定数 SED_TH の sed は使わない）。HYST_START も同じ道
        sed_in = hysteresis_patch.norms(doa_np, NCLS).astype(np.float64) > ENTER[None, None, :]
        sed_np = hysteresis_patch.apply_with_start(doa_np, sed_in, NCLS, ENTER, STAY, SED_ASSOC, start=HYST_START)
    elif HYST_START > 0:
        # 本体 §9-4 の 1: frame HYST_START より前は出力しない・ヒステリシスの状態も HYST_START から（入りだけのときも同じ枠を落とす・hysteresis_patch.apply_with_start）
        sed_np = hysteresis_patch.apply_with_start(doa_np, sed_np, NCLS, SED_TH, SED_STAY, SED_ASSOC, start=HYST_START)
    elif SED_STAY < SED_TH:
        # ヒステリシス（変更票 #34・2026-09-16）: 長さ > 保持 で、前フレームで「いる」だった同じ種類の相手が ±assoc° 以内なら「いる」を続ける。
        # 規則は scripts/_hysteresis_grid2.py の decode_general（型 amp）と同一（scripts/test_hysteresis_patch.py で一致を検査済み）
        sed_np = hysteresis_patch.apply(doa_np, NCLS, SED_TH, SED_STAY, SED_ASSOC)
    dcase = du.multi_accdoa_to_dcase_format(sed_np, doa_np, nb_classes=NCLS)
    polar = du.convert_output_format_cartesian_to_polar(in_dict=dcase)
    tmp = subs / f"{fn[:-5]}.csv.part"
    du.write_output_format_file(str(tmp), polar)
    tmp.replace(subs / f"{fn[:-5]}.csv")             # 途中で落ちても壊れない
    if (ci + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  {ci+1}/{len(todo)}  {el:.0f}s  "
              f"残り約{el/(ci+1)*(len(todo)-ci-1)/60:.0f}分", flush=True)

if TABLE is not None and integrity.sha256(THRESH_JSON) != TABLE["sha256"]:
    raise RuntimeError("thresholds table changed during inference")
if DUMP_NORMS:                                        # §3-5: 全部の本の norm がそろってから一覧を書く（欠けがあれば止める）
    _nmiss = [c for c in clips if not (NORMS_DIR / f"{c[:-5]}.npy").is_file()]
    if _nmiss:
        raise RuntimeError(f"norms が欠けている: {len(_nmiss)} 本（例 {_nmiss[:2]}）")
    _nm = {"schema": "norms/<clip>.npy = (3, T, C) float32 の矢印の長さ（ヒステリシスの前・間引かない・hysteresis_patch.norms）",
           "ckpt_sha256": CK_SHA, "ckpt_epoch": CK_EPOCH, "n_clips": len(clips), "nb_classes": NCLS,
           "files": {c[:-5]: integrity.sha256(NORMS_DIR / f"{c[:-5]}.npy") for c in clips}}
    (subs.parent / "norms_manifest.json.part").write_text(json.dumps(_nm, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(subs.parent / "norms_manifest.json.part", subs.parent / "norms_manifest.json")
    print(f"[causal] norms {len(clips):,} 本 → {NORMS_DIR}（一覧 norms_manifest.json）", flush=True)
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
print("CAUSAL_INFER_DONE")
