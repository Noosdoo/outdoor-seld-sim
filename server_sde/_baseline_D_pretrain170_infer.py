# -*- coding: utf-8 -*-
"""相手 D（変更票 #73）: 屋外で追加学習していない PSELDNets（事前学習の重み ckpts/mACCDOA-HTSAT-0.567.ckpt・170 クラス・mACCDOA・3 トラック）の
因果推論（_causal_infer_v17.py と同じ右詰め 10 s 窓・0.1 s ごと・最終フレームだけ）→ 170 クラスの出力を対応表（md/design/baselines_class_map_v1.json）で
この研究の 9 クラスへ写し、同じヒステリシス（入り／保持／対応幅）を掛けて、同じ 7 列の予測 csv（clip,frame,cls,track,az,el,dist〔dist は nan〕）を来歴つきで書く。2026-10-20・Fable。

v17 との違い: sde_patch（距離の頭）は当てない・高さの入力なし（HEIGHT_COND=0 固定）・experiment=synth_maccdoa・dataset 名 synth（paths.dataset_dir の直下に 170 行の
cls_indices_train.tsv を置く＝Synthesis クラスの読み方・出力の index は行順）。DCASE 形式の変換は du を使わず自前（az = atan2(y,x)・el = atan2(z, √(x²+y²))・度）。
環境変数: CAUSAL_CKPT（事前学習の重み）・CAUSAL_OUT・CAUSAL_FOA・CAUSAL_PREFIX／CAUSAL_GLOB・CAUSAL_LIMIT・CAUSAL_EXPECTED_CLIPS・CAUSAL_RUNS_DIR・
  CAUSAL_DSDIR（= 170 行の tsv を置いた dataset_dir）・CAUSAL_SED_TH／CAUSAL_SED_STAY／CAUSAL_SED_ASSOC・BASELINE_CLASSMAP（対応表 json）・E2E
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
PS = Path(os.environ.get("PS", HERE))
E2E = Path(os.environ.get("E2E", Path.home() / "research/outdoor_seld_e2e"))
sys.path.insert(0, str(PS / "src"))
sys.path.insert(0, str(PS))
sys.path.insert(0, str(HERE))

EXPERIMENT = os.environ.get("BASELINE_EXPERIMENT", "synth_maccdoa")
DATASET = os.environ.get("CAUSAL_DATASET", "synth")
CKPT = os.environ["CAUSAL_CKPT"]
OUTNAME = os.environ.get("CAUSAL_OUT", "infer_baselineD_pretrain170")
LOGDIR = Path(os.environ.get("CAUSAL_LOGDIR", Path.home() / "PSELDNets_logs"))
DSDIR = Path(os.environ.get("CAUSAL_DSDIR", PS / "datasets_pretrain170"))
PREFIX = os.environ.get("CAUSAL_PREFIX", "val_syn_")
LIMIT = int(os.environ.get("CAUSAL_LIMIT", "0"))
BATCH = int(os.environ.get("CAUSAL_BATCH", "25"))
SED_TH = float(os.environ.get("CAUSAL_SED_TH", "0.5"))
SED_STAY = float(os.environ.get("CAUSAL_SED_STAY", str(SED_TH)))
SED_ASSOC = float(os.environ.get("CAUSAL_SED_ASSOC", "20"))
# 本体 §3-5（Fable 16:5x: クラス別のしきい値は主・B・D の全員）: CAUSAL_THRESHOLDS_JSON = クラス別の表（threshold_table.load・D は対応表の共通クラスだけ・倍率 40 は据え置き）。未設定 = 旧（上の定数）。
#   表と CAUSAL_SED_TH／STAY は同時に置かない。HYST_START_FRAME（§9-4 の 1・v17 と同じ・既定 0 = 旧・明示なら 0 でも記録）。
#   BASELINE_DUMP_NORMS=1 は norms/<clip>.npy を .part → 置き換えで書き、最後に norms_manifest.json（ckpt の欄 = 重みの sha・しきい値の選択 v30_select_thresholds の入力）。
THRESH_JSON = os.environ.get("CAUSAL_THRESHOLDS_JSON", "").strip()
if THRESH_JSON and (os.environ.get("CAUSAL_SED_TH", "").strip() or os.environ.get("CAUSAL_SED_STAY", "").strip()):
    raise SystemExit("CAUSAL_THRESHOLDS_JSON と CAUSAL_SED_TH／CAUSAL_SED_STAY を同時に置かない（表か定数のどちらか・本体 §3-5）")
HYST_START = int(os.environ.get("HYST_START_FRAME", "0") or 0)
HYST_START_EXPLICIT = os.environ.get("HYST_START_FRAME", "").strip() != ""
assert 0 <= HYST_START < 100, f"HYST_START_FRAME は 0〜99: {HYST_START}"
DUMP_NORMS = os.environ.get("BASELINE_DUMP_NORMS") == "1"
CLASSMAP = Path(os.environ.get("BASELINE_CLASSMAP", E2E / "md/design/baselines_class_map_v2.json"))
GAIN_DB = float(os.environ.get("BASELINE_GAIN_DB", "0"))          # Q4: 入力の固定ゲイン（FOA の 4 チャンネル全部に同じ倍率・原音は変えない）
if float(os.environ.get("SDE_INPUT_GAIN_DB", "0") or 0) != 0.0 and GAIN_DB != 0.0:   # 本体 §3-2: D は自分の 40 dB のまま・主の倍率を重ねない
    raise RuntimeError("相手 D に SDE_INPUT_GAIN_DB（主の倍率）が置かれている。D は BASELINE_GAIN_DB だけ（二重に掛けない）")
GAIN = 10.0 ** (GAIN_DB / 20.0)
MANIFEST = os.environ.get("CAUSAL_MANIFEST", "")                   # 任意: clip_id の名簿 csv（ゲイン選択の部分集合など）。無ければ CAUSAL_GLOB
N_OURS = 9
N_TRACK = 3
SR, N, HOP = 24000, 240000, 2400

if not (DSDIR / "cls_indices_train.tsv").is_file():
    raise SystemExit(f"170 行の cls_indices_train.tsv が無い: {DSDIR}")
CM = json.loads(CLASSMAP.read_text(encoding="utf-8"))["D_pseldnets_pretrained"]
N_SRC = int(CM["n_source_classes"])
SRC_OF = {int(v["ours_index"]): [int(i) for i in v["source_index"]] for v in CM["map"].values()}   # 対象クラス → 元クラスの index の並び
tsv_lines = [l for l in (DSDIR / "cls_indices_train.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
if len(tsv_lines) != N_SRC:
    raise SystemExit(f"tsv の行数 {len(tsv_lines)} ≠ 対応表の n_source_classes {N_SRC}")
for ours, idxs in SRC_OF.items():                       # 対応表のラベル名と tsv の行が一致するか（取り違えを止める）
    names = CM["map"][{int(k): v for k, v in json.loads(CLASSMAP.read_text(encoding="utf-8"))["ours"].items()}[ours]]["source_labels"]
    for i, nm in zip(idxs, names):
        if tsv_lines[i].split("\t")[2] != nm:
            raise SystemExit(f"対応表と tsv の行が合わない: index {i} は {tsv_lines[i].split(chr(9))[2]!r}・対応表は {nm!r}")

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

GlobalHydra.instance().clear()
initialize_config_dir(config_dir=str(PS / "configs"), version_base="1.3")
cfg = compose(config_name="infer.yaml",
              overrides=[f"experiment={EXPERIMENT}", "mode=test", "model.kwargs.pretrained_path=null", f"paths.dataset_dir={DSDIR}"])

from models.model_module import SELDModelModule  # noqa: E402
from utils.config import get_dataset  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# Astra RR73-01（2026-10-21）: 読込先と版の照合。凍結 JSON（_baselines_env.py）が PS・BASELINE_TORCH_VERSION・BASELINE_NUMPY_VERSION を渡す。
#   モデルのソースが凍結した PS の下から読まれていること・外部ライブラリの版が凍結と同じことを、推論の前に確かめる（違えば止める）
import models.model_module as _mm  # noqa: E402
_mm_file = Path(_mm.__file__).resolve()
if not _mm_file.is_relative_to(PS.resolve()):
    raise SystemExit(f"モデルのソースが PS（{PS}）の下から読まれていない: {_mm_file}")
for _k, _have in (("BASELINE_TORCH_VERSION", torch.__version__), ("BASELINE_NUMPY_VERSION", np.__version__)):
    _want = os.environ.get(_k, "")
    if _want and _want != _have:
        raise SystemExit(f"{_k} が凍結と違う: 凍結 {_want}・実際 {_have}")

if not (HERE / "_prediction_integrity.py").exists():
    sys.path.insert(0, str(E2E / "server_sde"))
import _prediction_integrity as integrity  # noqa: E402
import hysteresis_patch  # noqa: E402
import _baseline_merge as merge  # noqa: E402  Astra R73-01: 同じ音の複製（元クラス違い・トラック違い・向き 20° 以内）をまとめ、向きの違う音を長さ順に 3 本まで残す（selftest 7/7）

ds = get_dataset(dataset_name=DATASET, cfg=cfg)
if ds.num_classes != N_SRC:
    raise SystemExit(f"dataset のクラス数 {ds.num_classes} ≠ {N_SRC}（tsv の置き場 {DSDIR} を確かめる）")
model = SELDModelModule(cfg, ds, test_meta={})
model.setup("predict")
CK_SHA = integrity.sha256(CKPT)
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
if integrity.sha256(CKPT) != CK_SHA:
    raise RuntimeError("checkpoint changed during loading")
sd = _ck["state_dict"]; del _ck
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
netmiss = [k for k in missing if k.startswith("net.")]
assert not netmiss, f"net の重みが欠けている: {netmiss[:5]}"
model.eval().cuda()
head = getattr(model.net, "_orig_mod", model.net)
print(f"[baselineD] ckpt={CKPT} sha={CK_SHA[:12]} classes={N_SRC} experiment={EXPERIMENT} sed={SED_TH}/{SED_STAY}/{SED_ASSOC} gain={GAIN_DB:+.1f} dB unexpected={len(unexpected)}", flush=True)
TABLE = None
if THRESH_JSON or DUMP_NORMS:
    try:
        import threshold_table  # noqa: E402
    except ImportError:
        sys.path.insert(0, str(Path(os.environ.get("E2E", Path.home() / "research/outdoor_seld_e2e")) / "server_sde"))
        import threshold_table  # noqa: E402
if THRESH_JSON:                                                   # 本体 §3-5: 表を作った重み = 今の重み・D は対応表の共通クラス（対象外は +inf）
    TABLE = threshold_table.load(THRESH_JSON, N_OURS, "hyst", ckpt_sha=CK_SHA, classes=sorted(SRC_OF))
    if os.environ.get("BASELINE_THRESHOLDS_SHA256", "").strip() and os.environ["BASELINE_THRESHOLDS_SHA256"].strip() != TABLE["sha256"]:
        raise SystemExit("しきい値の表の sha が凍結（BASELINE_THRESHOLDS_SHA256）と違う（本体 §3-6）")
    print(f"[baselineD] しきい値の表 {THRESH_JSON}（sha {TABLE['sha256'][:12]}）: {hysteresis_patch.describe(TABLE['enter'], TABLE['stay'], SED_ASSOC)}", flush=True)
ENTER, STAY = (TABLE["enter"], TABLE["stay"]) if TABLE is not None else (SED_TH, SED_STAY)

foa = Path(os.environ["CAUSAL_FOA"]) if os.environ.get("CAUSAL_FOA") else DSDIR / DATASET / "foa"
GLOB = os.environ.get("CAUSAL_GLOB", f"{PREFIX}*.flac")
if MANIFEST:
    import csv as _csv
    with open(MANIFEST, encoding="utf-8", newline="") as _f:
        clips = sorted(f"{r['clip_id']}.flac" for r in _csv.DictReader(_f))
    missing = [c for c in clips if not (foa / c).is_file()]
    if missing:
        raise SystemExit(f"名簿の本が foa に無い: {missing[:3]}（{len(missing)} 本）")
else:
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
code_paths = {Path(__file__).resolve(), Path(integrity.__file__).resolve(), Path(hysteresis_patch.__file__).resolve(), Path(merge.__file__).resolve(), CLASSMAP.resolve(), (DSDIR / "cls_indices_train.tsv").resolve()}
if THRESH_JSON or DUMP_NORMS:                                     # §3-5: 表か norms を使う run だけ
    code_paths.add(Path(threshold_table.__file__).resolve())
for module in list(sys.modules.values()):
    fn = getattr(module, "__file__", None)
    if fn and str(fn).endswith(".py"):
        path = Path(fn).resolve()
        if path.is_relative_to((PS / "src").resolve()):
            code_paths.add(path)
CODE_REAL = {integrity.logical_code_name(p, PS): p for p in code_paths}
assert len(CODE_REAL) == len(code_paths), f"論理名が衝突: {sorted(str(p) for p in code_paths)}"
# Astra RF73-01（2026-10-21）: 凍結 JSON（BASELINE_FROZEN_JSON）があれば、実際に読んだファイル（PS/src のモデルのソース・推論の補助コード・tsv・対応表）の
#   sha が、凍結一覧の**同じパス**の sha と一致することを推論の前に確かめる（凍結に無い PS/src のモジュールを読んでいたら止める）
_frozen = os.environ.get("BASELINE_FROZEN_JSON", "")
if _frozen:
    _FJ = json.loads(Path(_frozen).read_text(encoding="utf-8"))
    _entry = next((b for b in _FJ.get("baselines", {}).values() if str(b.get("kind", "")).startswith("pretrained_pseldnets")), None)
    if _entry is None:
        raise SystemExit("凍結 JSON に相手 D（pretrained_pseldnets_*）が無い")
    _fcode = {str(Path(k).expanduser().resolve()): v for k, v in (_entry.get("code") or {}).items()}
    _bad = []
    for _p in sorted(code_paths):
        _rp = str(Path(_p).resolve())
        if _rp in _fcode:
            if integrity.sha256(_p) != _fcode[_rp]:
                _bad.append(f"sha が違う: {_p}")
        elif _rp.startswith(str((PS / "src").resolve())):
            _bad.append(f"凍結に無いモデルのソースを読んでいる: {_p}")
    if _bad:
        raise SystemExit("凍結一覧と実際に読んだファイルが合わない（RF73-01）: " + "／".join(_bad[:5]))
    print(f"[baselineD] 凍結との照合 OK: 読んだ {len(code_paths)} 本のうち凍結一覧にある {sum(1 for p in code_paths if str(Path(p).resolve()) in _fcode)} 本の sha が一致", flush=True)
META = {"schema": integrity.SCHEMA, "ckpt": str(CKPT), "ckpt_sha256": CK_SHA, "ckpt_epoch": 0,
        "dataset": DATASET, "dsdir": str(DSDIR), "foa": str(foa), "glob": (f"manifest:{MANIFEST}" if MANIFEST else GLOB), "prefix": PREFIX,
        "clips": clips, "audio_sha256": integrity.audio_manifest(foa, clips),
        "height": {"on": False, "table": "", "table_sha256": None, "offset": None, "default": None, "values": {}},
        "code_sha256": {integrity.logical_code_name(p, PS): integrity.sha256(p) for p in sorted(code_paths)},
        "settings": {"cfg": OmegaConf.to_container(cfg, resolve=False), "batch": BATCH, "kind": "baseline_D_pretrain170",
                     "experiment": EXPERIMENT, "n_source_classes": N_SRC, "classmap": str(CLASSMAP), "classmap_sha256": integrity.sha256(CLASSMAP),
                     "gain_db": GAIN_DB, "manifest": MANIFEST,
                     "classmap_D": {str(k): v for k, v in SRC_OF.items()}, "absent": CM["absent"],
                     "sed_threshold": SED_TH, "sed_stay": SED_STAY, "sed_assoc": SED_ASSOC, "hysteresis": SED_STAY < SED_TH,
                     "hysteresis_pool_min": hysteresis_patch.POOL_MIN, "distance": "none (nan)", "sr": SR, "samples": N, "hop": HOP,
                     "torch": str(torch.__version__), "numpy": np.__version__, "soundfile": sf.__version__}}
META = json.loads(json.dumps(META, default=str))
if HYST_START_EXPLICIT:                                           # §9-4 の 1: 明示なら 0 でも（未設定の run の identity は変えない）
    META["settings"]["hyst_start_frame"] = HYST_START
if TABLE is not None:                                             # §3-5: 表を使う run だけ（定数の欄は None）
    META["settings"].update(sed_threshold=None, sed_stay=None, hysteresis=hysteresis_patch.has_hysteresis(ENTER, STAY), thresholds=threshold_table.record(TABLE))
if DUMP_NORMS:
    META["settings"]["dump_norms"] = True
integrity.begin_run(subs.parent, META)
todo = [c for c in clips if not (subs / f"{c[:-5]}.csv").exists()]
print(f"[baselineD] 対象 {len(clips):,} / 未処理 {len(todo):,} → {subs}", flush=True)


def split_tracks(y: np.ndarray) -> np.ndarray:
    """(T, 3*3*C) の mACCDOA 出力 → (3 トラック, T, 3, C) の x,y,z。PSELDNets の mACCDOA の並びは [track][xyz][class]（1530 = 3 × 3 × 170）。"""
    T = y.shape[0]
    return y.reshape(T, N_TRACK, 3, N_SRC).transpose(1, 0, 2, 3)


def map_to_ours(v: np.ndarray) -> np.ndarray:
    """(3, T, 3, C_src) → (3, T, 4*9) の doa（x,y,z の 3 ブロック＋距離ブロック 0）。対象クラスごとに重複をまとめて 3 本まで（_baseline_merge.map_tracks）。"""
    return merge.map_tracks(v, SRC_OF, n_ours=N_OURS, merge_deg=SED_ASSOC, n_track=N_TRACK)


t0 = time.time()
for ci, fn in enumerate(todo):
    wav, sr = sf.read(str(foa / fn), dtype="float32")
    assert sr == SR and wav.shape[0] == N, f"{fn}: sr={sr} n={wav.shape[0]}"
    x = torch.from_numpy(wav.T.copy()) * GAIN                         # 固定ゲイン（4 チャンネル同じ・float のまま・飽和させない）
    nfr = N // HOP
    buf = torch.zeros(nfr, 4, N)
    for k in range(1, nfr + 1):
        buf[k - 1, :, N - k * HOP:] = x[:, :k * HOP]
    outs = []
    with torch.no_grad():
        for s in range(0, nfr, BATCH):
            xb = buf[s:s + BATCH].cuda()
            feat = model.standardize(xb)
            o = model.net(feat)
            y = o["multi_accdoa"] if isinstance(o, dict) else o
            outs.append(y[:, -1, :].float().cpu())
    causal = torch.cat(outs, dim=0).numpy()                      # (T, 1530)
    assert causal.shape[1] == N_TRACK * 3 * N_SRC, f"出力の幅が違う: {causal.shape}"
    if os.environ.get("BASELINE_DEBUG") == "1":                   # 試走用: 生の出力の大きさ（norm）の分布と上位クラス
        v_all = split_tracks(causal)                              # (3, T, 3, C)
        nrm_all = np.sqrt((v_all ** 2).sum(axis=2))               # (3, T, C)
        mx = nrm_all.max(axis=(0, 1))                             # (C,)
        top = np.argsort(-mx)[:8]
        mapped = sorted({i for v in SRC_OF.values() for i in v})
        print(f"[debug] {fn}: 全クラスの norm の最大 {float(mx.max()):.3f}・対象クラスの最大 {float(mx[mapped].max()):.3f}・"
              f"norm>0.3 の割合 {float((nrm_all > 0.3).mean()):.4f}・>0.5 {float((nrm_all > 0.5).mean()):.4f}・出力の絶対値の平均 {float(np.abs(causal).mean()):.4f}", flush=True)
        print("[debug]   上位: " + "・".join(f"{tsv_lines[i].split(chr(9))[2]}({i})={mx[i]:.2f}" for i in top), flush=True)
    doa9 = map_to_ours(split_tracks(causal))                      # (3, T, 36)
    if os.environ.get("BASELINE_DUMP_NORMS") == "1":              # ゲイン・閾値の選択用（R73-05）: ヒステリシス前の 9 クラスの norm（3 トラック・T・9）を残す
        nd = subs.parent / "norms"; nd.mkdir(exist_ok=True)
        nrm9 = np.sqrt(sum(doa9[:, :, b * N_OURS:(b + 1) * N_OURS] ** 2 for b in range(3)))
        threshold_table.save_norms(nd, fn[:-5], nrm9.astype(np.float32))   # §3-5: .part → 置き換え（式は前と同じ）
    if TABLE is not None or HYST_START > 0:                        # §3-5・§9-4 の 1: クラス別の表と frame HYST_START からの判定（v17 と同じ道）
        sed_in = hysteresis_patch.norms(doa9, N_OURS).astype(np.float64) > ENTER
        sed9 = hysteresis_patch.apply_with_start(doa9, sed_in, N_OURS, ENTER, STAY, SED_ASSOC, start=HYST_START)
    else:
        sed9 = hysteresis_patch.apply(doa9, N_OURS, SED_TH, SED_STAY, SED_ASSOC)   # (3, T, 9) bool・v17 と同じ規則
    rows = []
    for t in range(N_TRACK):
        for k in range(nfr):
            for c in range(N_OURS):
                if sed9[t, k, c]:
                    xx, yy, zz = (float(doa9[t, k, b * N_OURS + c]) for b in range(3))
                    az = math.degrees(math.atan2(yy, xx)); el = math.degrees(math.atan2(zz, math.hypot(xx, yy)))
                    rows.append(f"{k},{c},{t},{az:.2f},{el:.2f},nan\n")
    rows.sort(key=lambda r: (int(r.split(",")[0]), int(r.split(",")[1]), int(r.split(",")[2])))
    tmp = subs / f"{fn[:-5]}.csv.part"
    tmp.write_text("".join(rows), encoding="utf-8")
    tmp.replace(subs / f"{fn[:-5]}.csv")
    if (ci + 1) % 25 == 0:
        el_ = time.time() - t0
        print(f"  {ci+1}/{len(todo)}  {el_:.0f}s  残り約{el_/(ci+1)*(len(todo)-ci-1)/60:.0f}分", flush=True)

if TABLE is not None and integrity.sha256(THRESH_JSON) != TABLE["sha256"]:
    raise RuntimeError("thresholds table changed during inference")
if DUMP_NORMS:                                                    # §3-5: 全部の本の norms がそろってから一覧（欠けは止める）
    threshold_table.write_norms_manifest(subs.parent, [c[:-5] for c in clips], CK_SHA, N_OURS,
                                         "norms/<clip>.npy = (3, T, 9) float32 の矢印の長さ（ヒステリシスの前・間引かない）・相手 D（対応表で 9 クラスに写したあと）", extra={"kind": "baseline_D_pretrain170"})
if integrity.audio_manifest(foa, clips) != META["audio_sha256"]:
    raise RuntimeError("audio changed during inference")
if any(integrity.sha256(CODE_REAL[k]) != h for k, h in META["code_sha256"].items()):
    raise RuntimeError("inference code changed during execution")
done = integrity.publish(subs.parent, META)
_run_lock.close()
print(f"[baselineD] wrote {subs.parent / 'val_all_causal.csv'} ({done['n_rows']:,} lines / {done['n_files']:,} clips, sha256 {done['sha256'][:12]})", flush=True)
print("CAUSAL_INFER_DONE")
