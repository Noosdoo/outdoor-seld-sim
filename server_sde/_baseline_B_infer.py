# -*- coding: utf-8 -*-
"""相手 B（変更票 #73）: 学習した公式ベースライン（SeldModel・multi-ACCDOA＋距離・9 クラス）を、この研究と同じ条件で因果推論する。
  - 窓: 公式の学習と同じ 5 s（特徴 250 フレーム × 20 ms・ラベル 50 フレーム × 0.1 s）を右詰め・先頭は無音の特徴で埋める（未来を見ない）・判定時刻 = 窓の最後のラベルフレーム
  - 特徴: 公式の FeatureClass（log-mel 64 × 4ch ＋ 強度ベクトル 3ch）を 10 s の音全体で 1 回計算し、公式の StandardScaler（train だけで fit）で標準化。先頭の埋めは
    「無音（ゼロ）の音の特徴」を同じ標準化に通した 1 フレームを繰り返す
  - 出力の復号: 公式の get_multi_accdoa_labels（3 トラック × (x,y,z) ＋ 距離〔m〕）→ 重複の整理（_baseline_merge・向き 20° 以内は 1 本・3 本まで）→ 同じヒステリシス
    （入り／保持・hysteresis_patch）→ 7 列 csv（clip,frame,cls,track,az,el,dist[m]）・来歴つき（_prediction_integrity）
環境変数: B_MODEL（学習した .h5）・B_FEAT_DIR（標準化の重み wts のため）・CAUSAL_FOA・CAUSAL_PREFIX／CAUSAL_GLOB／CAUSAL_MANIFEST・CAUSAL_LIMIT・CAUSAL_EXPECTED_CLIPS・
  CAUSAL_RUNS_DIR・CAUSAL_OUT・CAUSAL_SED_TH／STAY／ASSOC・E2E
  本体 §3-6（2026-10-26）: B_INPUT_GAIN_DB（特徴の前の波形に掛ける・学習の置き場の印と照合）・B_DIST_ENCODING（log1p なら復号で m に戻す）。
  run_meta の settings に input_gain_db（B_INPUT_GAIN_DB を明示したときだけ・0 でも）・dist_encoding（log1p のときだけ）。
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
sys.path.insert(0, str(HERE))
import _baseline_B_common as C  # noqa: E402

C.setup_path()
import _prediction_integrity as integrity  # noqa: E402
import hysteresis_patch  # noqa: E402
import _baseline_merge as merge  # noqa: E402
import joblib  # noqa: E402
import seldnet_model  # noqa: E402  公式
import train_seldnet  # noqa: E402  公式（get_multi_accdoa_labels）

MODEL = Path(os.environ["B_MODEL"])
# Astra RR73-01（2026-10-21）: 読込先と版の照合。公式コードが凍結した置き場（B_OFFICIAL = settings.official_dir）から import されていること・版が凍結と同じこと
_official_file = Path(seldnet_model.__file__).resolve()
if _official_file.parent != C.OFFICIAL.resolve():
    raise SystemExit(f"公式コードが凍結の置き場（{C.OFFICIAL}）から読まれていない: {_official_file}")
for _k, _have in (("BASELINE_TORCH_VERSION", torch.__version__), ("BASELINE_NUMPY_VERSION", np.__version__)):
    _want = os.environ.get(_k, "")
    if _want and _want != _have:
        raise SystemExit(f"{_k} が凍結と違う: 凍結 {_want}・実際 {_have}")
OUTNAME = os.environ.get("CAUSAL_OUT", "infer_baselineB")
LOGDIR = Path(os.environ.get("CAUSAL_LOGDIR", Path.home() / "PSELDNets_logs"))
PREFIX = os.environ.get("CAUSAL_PREFIX", "val_syn_")
LIMIT = int(os.environ.get("CAUSAL_LIMIT", "0"))
SED_TH = float(os.environ.get("CAUSAL_SED_TH", "0.5"))
SED_STAY = float(os.environ.get("CAUSAL_SED_STAY", str(SED_TH)))
SED_ASSOC = float(os.environ.get("CAUSAL_SED_ASSOC", "20"))
# 本体 §3-5（Fable 16:5x: クラス別のしきい値は主・B・D の全員）: CAUSAL_THRESHOLDS_JSON = クラス別の表（threshold_table.load・B は 0〜8 の全部）。未設定 = 旧（上の定数）。
#   表と CAUSAL_SED_TH／STAY は同時に置かない。HYST_START_FRAME（§9-4 の 1・v17 と同じ・既定 0 = 旧・明示なら 0 でも記録）。
#   BASELINE_DUMP_NORMS=1 は norms/<clip>.npy を .part → 置き換えで書き、最後に norms_manifest.json（ckpt の欄 = 重みの sha・しきい値の選択 v30_select_thresholds の入力）。
THRESH_JSON = os.environ.get("CAUSAL_THRESHOLDS_JSON", "").strip()
if THRESH_JSON and (os.environ.get("CAUSAL_SED_TH", "").strip() or os.environ.get("CAUSAL_SED_STAY", "").strip()):
    raise SystemExit("CAUSAL_THRESHOLDS_JSON と CAUSAL_SED_TH／CAUSAL_SED_STAY を同時に置かない（表か定数のどちらか・本体 §3-5）")
HYST_START = int(os.environ.get("HYST_START_FRAME", "0") or 0)
HYST_START_EXPLICIT = os.environ.get("HYST_START_FRAME", "").strip() != ""
assert 0 <= HYST_START < 100, f"HYST_START_FRAME は 0〜99: {HYST_START}"
DUMP_NORMS = os.environ.get("BASELINE_DUMP_NORMS") == "1"
MANIFEST = os.environ.get("CAUSAL_MANIFEST", "")
N_OURS, N_TRACK = C.N_CLASSES, 3
SR, N, HOP = 24000, 240000, 2400
NFR = N // HOP                                                  # 100 ラベルフレーム

P = C.get_params()
fc = C.feature_class(P)
C.check_feat_marker(C.FEAT)                                       # 本体 §3-6: 学習の置き場の倍率・距離の符号化と環境が同じか（違えば止める）
C.patch_dist_decode()                                             # 本体 §3-6: log1p なら公式の復号を m に戻す（get_multi_accdoa_labels を包む）
FEAT_PER_LABEL = int(P["label_hop_len_s"] // P["hop_len_s"])     # 5
SEQ_FEAT = int(P["feature_sequence_length"])                     # 250
SEQ_LAB = int(P["label_sequence_length"])                        # 50
scaler = joblib.load(fc.get_normalized_wts_file())
NB_CH = 7 if P["dataset"] == "foa" else None


def _feat_rows(spect: np.ndarray) -> np.ndarray:
    """公式の線形スペクトル (T, nfft/2, 4) → 公式の特徴（T, 7*64）を標準化したもの。log-mel は top_db なし（_baseline_B_common の 11）・強度ベクトルはフレームごと。"""
    mel = fc._get_mel_spectrogram(spect)
    iv = fc._get_foa_intensity_vectors(spect)
    return scaler.transform(np.concatenate((mel, iv), axis=-1)).astype(np.float32)


def feat_of_audio(wav: np.ndarray) -> np.ndarray:
    """(N, 4) float → 公式の特徴（T_feat, 7*64）を標準化したもの（本全体で計算・学習の特徴と同じ作り方）。"""
    audio = (wav.astype(np.float64) * C.GAIN if C.INPUT_GAIN_DB != 0.0 else wav.astype(np.float64)) + fc._eps   # 本体 §3-6: 倍率は eps の前（学習の _load_audio と同じ）
    nb_feat_frames = int(len(audio) / float(fc._hop_len))
    return _feat_rows(fc._spectrogram(audio, nb_feat_frames))


TAIL_HOPS = 4                                                    # 判定時刻の最後の特徴フレームを「その時刻までの音」だけで作り直すときに使う尻の長さ（hop の 4 倍）


def feat_causal(wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Astra RI73-02（2026-10-21）: 判定時刻 e_k = (k+1)×0.1 s より後の音を入力に入れない特徴。
    公式の特徴はフレームごとの変換（top_db なしの log-mel・フレームごとに正規化した強度ベクトル）なので、未来が混ざるのは STFT の窓だけ:
    中心 j×hop の窓は j×hop + nfft/2 まで使う。判定時刻 e_k の窓の最後の特徴フレーム j = 5(k+1)−1 だけが e_k を 32 サンプル（1.3 ms）越える。
    → 本全体の STFT から作った特徴（feat_full）に加え、各ラベルフレーム k の最後の特徴フレームを「e_k までの音の尻（hop×4）」だけから作り直した
      行（last_rows[k]）を返す。窓 k の他のフレームは中心 + nfft/2 ≤ e_k − 448 なので本全体の STFT と同じ（未来を含まない）。
    返り値: feat_full (T_feat, 448)・last_rows (NFR, 448)。"""
    audio = (wav.astype(np.float64) * C.GAIN if C.INPUT_GAIN_DB != 0.0 else wav.astype(np.float64)) + fc._eps   # 本体 §3-6: 倍率は eps の前
    nb_feat_frames = int(len(audio) / float(fc._hop_len))
    feat_full = _feat_rows(fc._spectrogram(audio, nb_feat_frames))
    hop = int(fc._hop_len); L = TAIL_HOPS * hop
    last_rows = np.empty((NFR, feat_full.shape[1]), dtype=np.float32)
    for k in range(NFR):
        e = FEAT_PER_LABEL * (k + 1) * hop                      # 判定時刻までのサンプル数
        tail = audio[max(0, e - L):e]
        if tail.shape[0] < L:                                    # 先頭は無音（0 ＋ eps）で左に埋める（本の前は無音と見なす）
            tail = np.concatenate([np.full((L - tail.shape[0], tail.shape[1]), fc._eps), tail], axis=0)
        sp = fc._spectrogram(tail, TAIL_HOPS)                   # 尻の STFT・最後のフレーム（中心 L − hop）の窓は L を 32 サンプル越え → 公式と同じ零埋め
        last_rows[k] = _feat_rows(sp)[-1]
    return feat_full, last_rows


SIL = feat_of_audio(np.zeros((N, 4), dtype=np.float32))[0]       # 無音の特徴 1 フレーム（標準化済み）


def to_model_input(feat: np.ndarray) -> np.ndarray:
    """(T, 7*64) → (7, T, 64)（公式 split_multi_channels と同じ並び）。"""
    T = feat.shape[0]
    return feat.reshape(T, NB_CH, -1).transpose(1, 0, 2)


def causal_windows(feat: np.ndarray, last_rows: np.ndarray | None = None) -> np.ndarray:
    """(T_feat, 448) → (NFR, 7, SEQ_FEAT, 64): ラベルフレーム k の窓 = 特徴フレーム [5(k+1)-250, 5(k+1)) を右詰め・先頭は無音で埋める。
    last_rows (NFR, 448) があれば、窓 k の最後の特徴フレーム（5(k+1)−1）をその行に置き換える（判定時刻までの音だけで作った行・RI73-02）。"""
    T = feat.shape[0]
    out = np.empty((NFR, NB_CH, SEQ_FEAT, feat.shape[1] // NB_CH), dtype=np.float32)
    for k in range(NFR):
        end = min(T, FEAT_PER_LABEL * (k + 1)); start = end - SEQ_FEAT
        if start >= 0:
            w = feat[start:end].copy()
        else:
            w = np.concatenate([np.repeat(SIL[None], -start, axis=0), feat[:end]], axis=0)
        if last_rows is not None:
            w[-1] = last_rows[k]
        out[k] = to_model_input(w)
    return out


foa = Path(os.environ["CAUSAL_FOA"])
GLOB = os.environ.get("CAUSAL_GLOB", f"{PREFIX}*.flac")
if MANIFEST:
    import csv as _csv
    with open(MANIFEST, encoding="utf-8", newline="") as _f:
        clips = sorted(f"{r['clip_id']}.flac" for r in _csv.DictReader(_f))
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

# モデル（公式）
data_in = (P["batch_size"], NB_CH, SEQ_FEAT, P["nb_mel_bins"])
data_out = (P["batch_size"], SEQ_LAB, N_OURS * 3 * 4)
model = seldnet_model.SeldModel(data_in, data_out, P)
MODEL_SHA = integrity.sha256(MODEL)
sd = torch.load(str(MODEL), map_location="cpu")
model.load_state_dict(sd)
model.eval().cuda()
TABLE = None
if THRESH_JSON or DUMP_NORMS:
    try:
        import threshold_table  # noqa: E402
    except ImportError:
        sys.path.insert(0, str(Path(os.environ.get("E2E", Path.home() / "research/outdoor_seld_e2e")) / "server_sde"))
        import threshold_table  # noqa: E402
if THRESH_JSON:                                                   # 本体 §3-5: 表を作った重み = 今の重み・B は 0〜8 の全部
    TABLE = threshold_table.load(THRESH_JSON, N_OURS, "hyst", ckpt_sha=MODEL_SHA, classes=None)
    if os.environ.get("BASELINE_THRESHOLDS_SHA256", "").strip() and os.environ["BASELINE_THRESHOLDS_SHA256"].strip() != TABLE["sha256"]:
        raise SystemExit("しきい値の表の sha が凍結（BASELINE_THRESHOLDS_SHA256）と違う（本体 §3-6）")
    print(f"[baselineB] しきい値の表 {THRESH_JSON}（sha {TABLE['sha256'][:12]}）: {hysteresis_patch.describe(TABLE['enter'], TABLE['stay'], SED_ASSOC)}", flush=True)
ENTER, STAY = (TABLE["enter"], TABLE["stay"]) if TABLE is not None else (SED_TH, SED_STAY)
code_paths = {Path(__file__).resolve(), Path(C.__file__).resolve(), Path(integrity.__file__).resolve(), Path(hysteresis_patch.__file__).resolve(), Path(merge.__file__).resolve(),
              Path(seldnet_model.__file__).resolve(), Path(train_seldnet.__file__).resolve(), Path(fc.get_normalized_wts_file()).resolve()}
if THRESH_JSON or DUMP_NORMS:                                     # §3-5: 表か norms を使う run だけ
    code_paths.add(Path(threshold_table.__file__).resolve())
CODE_REAL = {integrity.logical_code_name(p, C.OFFICIAL): p for p in code_paths}
assert len(CODE_REAL) == len(code_paths)
# Astra RF73-01（2026-10-21）: 凍結 JSON（BASELINE_FROZEN_JSON）があれば、実際に読んだファイル（公式コード・標準化の重み・推論の補助コード）の sha が
#   凍結一覧の**同じパス**の sha と一致することを推論の前に確かめる（公式の import 元が凍結に無ければ止める）
_frozen = os.environ.get("BASELINE_FROZEN_JSON", "")
if _frozen:
    _FJ = json.loads(Path(_frozen).read_text(encoding="utf-8"))
    _entry = next((b for b in _FJ.get("baselines", {}).values() if str(b.get("kind", "")).startswith("dcase2024")), None)
    if _entry is None:
        raise SystemExit("凍結 JSON に相手 B（dcase2024_*）が無い")
    _fcode = {str(Path(k).expanduser().resolve()): v for k, v in (_entry.get("code") or {}).items()}
    _bad = []
    for _p in sorted(code_paths):
        _rp = str(Path(_p).resolve())
        if _rp in _fcode:
            if integrity.sha256(_p) != _fcode[_rp]:
                _bad.append(f"sha が違う: {_p}")
        elif _rp.startswith(str(C.OFFICIAL.resolve())) or _rp == str(Path(fc.get_normalized_wts_file()).resolve()):
            _bad.append(f"凍結に無いファイルを読んでいる: {_p}")
    if _bad:
        raise SystemExit("凍結一覧と実際に読んだファイルが合わない（RF73-01）: " + "／".join(_bad[:5]))
    print(f"[baselineB] 凍結との照合 OK: 読んだ {len(code_paths)} 本のうち凍結一覧にある {sum(1 for p in code_paths if str(Path(p).resolve()) in _fcode)} 本の sha が一致", flush=True)
META = {"schema": integrity.SCHEMA, "ckpt": str(MODEL), "ckpt_sha256": MODEL_SHA, "ckpt_epoch": 0,
        "dataset": "dcase2024_baseline_v29", "dsdir": str(C.DS), "foa": str(foa), "glob": (f"manifest:{MANIFEST}" if MANIFEST else GLOB), "prefix": PREFIX,
        "clips": clips, "audio_sha256": integrity.audio_manifest(foa, clips),
        "height": {"on": False, "table": "", "table_sha256": None, "offset": None, "default": None, "values": {}},
        "code_sha256": {integrity.logical_code_name(p, C.OFFICIAL): integrity.sha256(p) for p in sorted(code_paths)},
        "settings": {"kind": "baseline_B_dcase2024", "official_commit": os.environ.get("B_OFFICIAL_COMMIT", ""),
                     "causal_features": "log-mel top_db=None (per-frame); last feature frame of each window recomputed from audio up to the decision time (tail 4 hops); RI73-02",
                     "feat_dir": str(C.FEAT), "wts_sha256": integrity.sha256(Path(fc.get_normalized_wts_file())), "window_s": SEQ_FEAT * P["hop_len_s"], "feat_per_label": FEAT_PER_LABEL,
                     "params": {k: P[k] for k in ("unique_classes", "nb_mel_bins", "hop_len_s", "label_hop_len_s", "feature_sequence_length", "label_sequence_length", "multi_accdoa", "nb_epochs", "batch_size", "lr")},
                     "sed_threshold": SED_TH, "sed_stay": SED_STAY, "sed_assoc": SED_ASSOC, "hysteresis": SED_STAY < SED_TH, "hysteresis_pool_min": hysteresis_patch.POOL_MIN,
                     "merge_deg": SED_ASSOC, "distance": "m (official labels cm/100)", "sr": SR, "samples": N, "hop": HOP, "torch": str(torch.__version__), "numpy": np.__version__}}
if C.INPUT_GAIN_EXPLICIT:                                         # 本体 §3-6: 倍率を明示した run だけ（0 でも）
    META["settings"]["input_gain_db"] = C.INPUT_GAIN_DB
if C.DIST_ENCODING != "linear":                                   # 本体 §3-6: 距離の符号化を替えた run だけ
    META["settings"]["dist_encoding"] = C.DIST_ENCODING
    META["settings"]["distance"] = "m (labels u = log1p(d)/log1p(100); decoded d = expm1(max(u,0)*log1p(100)))"
META = json.loads(json.dumps(META, default=str))
if HYST_START_EXPLICIT:                                           # §9-4 の 1: 明示なら 0 でも（未設定の run の identity は変えない）
    META["settings"]["hyst_start_frame"] = HYST_START
if TABLE is not None:                                             # §3-5: 表を使う run だけ（定数の欄は None）
    META["settings"].update(sed_threshold=None, sed_stay=None, hysteresis=hysteresis_patch.has_hysteresis(ENTER, STAY), thresholds=threshold_table.record(TABLE))
if DUMP_NORMS:
    META["settings"]["dump_norms"] = True
integrity.begin_run(subs.parent, META)
todo = [c for c in clips if not (subs / f"{c[:-5]}.csv").exists()]
print(f"[baselineB] model={MODEL.name} sha={MODEL_SHA[:12]} sed={SED_TH}/{SED_STAY}/{SED_ASSOC} 対象 {len(clips):,} / 未処理 {len(todo):,} → {subs}", flush=True)

t0 = time.time()
for ci, fn in enumerate(todo):
    wav, sr = sf.read(str(foa / fn), dtype="float32")
    assert sr == SR and wav.shape[0] == N, f"{fn}: sr={sr} n={wav.shape[0]}"
    feat_full, last_rows = feat_causal(wav)                                    # (500, 448)・(100, 448)（RI73-02: 判定時刻より後の音を入れない）
    win = torch.from_numpy(causal_windows(feat_full, last_rows))               # (100, 7, 250, 64)
    outs = []
    with torch.no_grad():
        for s in range(0, NFR, 25):
            y = model(win[s:s + 25].cuda())                                    # (b, 50, 108)
            outs.append(y[:, -1, :].float().cpu())                             # 窓の最後のラベルフレームだけ
    out = torch.cat(outs, dim=0).numpy()[None]                                 # (1, 100, 108)
    sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2 = train_seldnet.get_multi_accdoa_labels(out, N_OURS)   # 各 (1, 100, 9) / (1, 100, 27) / (1, 100, 9)
    v = np.zeros((N_TRACK, NFR, 3, N_OURS), dtype=np.float32)                  # 重複の整理に渡す形（x,y,z・長さ = 確からしさ）
    dist = np.zeros((N_TRACK, NFR, N_OURS), dtype=np.float32)
    for t, (doa, dd) in enumerate(((doa0, dist0), (doa1, dist1), (doa2, dist2))):
        d3 = doa[0].reshape(NFR, 3, N_OURS)                                    # 公式の並び [x(9), y(9), z(9)]
        v[t] = d3; dist[t] = dd[0]
    doa9, src = merge.map_tracks(v, {c: [c] for c in range(N_OURS)}, n_ours=N_OURS, merge_deg=SED_ASSOC, n_track=N_TRACK, return_src=True)   # (3, 100, 36)・向き 20° 以内の複製は 1 本
    # 距離ブロック: 残した候補と**同じ元トラック**の距離を付ける（Astra RI73-03: 向きの内積で選ぶと、長さの大きい別の音の距離が付いていた・検査は _baseline_merge の 8）
    doa9 = merge.attach_distance(doa9, src, dist, N_OURS)
    if os.environ.get("BASELINE_DUMP_NORMS") == "1":              # 閾値の選択用（#73・D と同じ手順 _baseline_select_gain.py --target B --gains 0）: ヒステリシス前の 9 クラスの norm（3 トラック・T・9）
        nd = subs.parent / "norms"; nd.mkdir(exist_ok=True)
        nrm9 = np.sqrt(sum(doa9[:, :, b * N_OURS:(b + 1) * N_OURS] ** 2 for b in range(3)))
        threshold_table.save_norms(nd, fn[:-5], nrm9.astype(np.float32))   # §3-5: .part → 置き換え（式は前と同じ）
    if TABLE is not None or HYST_START > 0:                        # §3-5・§9-4 の 1: クラス別の表と frame HYST_START からの判定（v17 と同じ道）
        sed_in = hysteresis_patch.norms(doa9, N_OURS).astype(np.float64) > ENTER
        sed9 = hysteresis_patch.apply_with_start(doa9, sed_in, N_OURS, ENTER, STAY, SED_ASSOC, start=HYST_START)
    else:
        sed9 = hysteresis_patch.apply(doa9, N_OURS, SED_TH, SED_STAY, SED_ASSOC)
    rows = []
    for t in range(N_TRACK):
        for k in range(NFR):
            for c in range(N_OURS):
                if sed9[t, k, c]:
                    xx, yy, zz = (float(doa9[t, k, b * N_OURS + c]) for b in range(3))
                    az = math.degrees(math.atan2(yy, xx)); el = math.degrees(math.atan2(zz, math.hypot(xx, yy)))
                    rows.append(f"{k},{c},{t},{az:.2f},{el:.2f},{float(doa9[t, k, 3 * N_OURS + c]):.3f}\n")
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
    threshold_table.write_norms_manifest(subs.parent, [c[:-5] for c in clips], MODEL_SHA, N_OURS,
                                         "norms/<clip>.npy = (3, T, 9) float32 の矢印の長さ（ヒステリシスの前・間引かない）・相手 B（重複の整理と距離の付け直しのあと）", extra={"kind": "baseline_B_dcase2024"})
if integrity.audio_manifest(foa, clips) != META["audio_sha256"]:
    raise RuntimeError("audio changed during inference")
if any(integrity.sha256(CODE_REAL[k]) != h for k, h in META["code_sha256"].items()):
    raise RuntimeError("inference code changed during execution")
done = integrity.publish(subs.parent, META)
_run_lock.close()
print(f"[baselineB] wrote {subs.parent / 'val_all_causal.csv'} ({done['n_rows']:,} lines / {done['n_files']:,} clips, sha256 {done['sha256'][:12]})", flush=True)
print("CAUSAL_INFER_DONE")
