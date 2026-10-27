# -*- coding: utf-8 -*-
"""相手 B（変更票 #73）: DCASE 2024 Task 3 の公式ベースライン（partha2409/DCASE2024_seld_baseline・MIT）を、この研究の学習データ v29 に当てはめるための共通部品。
2026-10-20〜21・Fable。公式のモデル（SeldModel）・損失・学習の流れ（train_seldnet.main）をそのまま使う。公式の**ファイルは編集しない**が、このデータの読み込み・
標準化・因果推論・評価には下の適応を実行時に加えている（子クラスと monkeypatch）= 計算手順まで公式と同一ではない（Astra RI73-07・言い切らない）。

公式との違い（全部ここに書く・変更票 #73 と引き継ぎに写す）:
  1. 音声の読み方: 公式は int16 の wav を scipy で読んで /32768。この研究の音は flac（float32・0 dBFS ≡ 143 dB SPL の較正で小さい）なので、
     `fold3_<clip>.wav` という名前のシンボリックリンクを flac に張り、soundfile で float のまま読む（FeatureClassFlac._load_audio）。
     学習と推論で同じ尺度・同じ標準化（train だけで fit した StandardScaler）を使う（尺度の影響を「無い」とは言わない・log の下限や eps の効き方は残る）。
  2. フレーム数の数え方: 公式は wave モジュール（PCM wav 専用）→ soundfile.info に差し替え（FeatureClassFlac.get_frame_stats）。
  3. 標準化の fit: 公式は feat_dir の全ファイルで fit（val も混ざる）→ **fold3（train）だけで fit**（Astra R73-03）。transform は全部。
  4. クラス数 9・学習データの置き場・特徴の置き場・出力の置き場を params で上書き。finetune_mode=False（白紙から・公式の合成データ学習済みの重みは使わない）。
  5. 公式が import する cv2・matplotlib・IPython・torchvision は映像用で、音だけの学習では使わない。venv に無いので _vendor に空の代用を置く（公式コードは触らない）。
  6. 距離の単位: この研究の metadata_dist は m。公式の ADPIT ラベルは csv の 6 列目を cm として /100 で m にする → csv には round(m × 100) の cm を書く（往復検査 --check）。
  7. 正解の無い本（負例・この研究の train の 20.9%・val の 20.8%）: 公式の評価（ComputeSELDResults.__init__・最良 epoch 選びの val の指標）は正解の csv が空だと
     max() で止まる（試走 8598・2026-10-21）。公式ファイルは変えず、__init__ を包んで「中身のある本だけで公式の初期化 → 空の本は枠 100・正解なしとして足す」
     （patch_compute_results）。学習データは全部そのまま（負例も学習に入る）。公式の val の指標が負例の誤検出も数えるようになるだけ。この研究の採点には関係ない。
  8. 置き場の名前: 公式 train_seldnet.main は dataset_dir の名前で分割を決める（'2024' → train fold3・val／test fold4。年が無いと止まる）→ 置き場は B_dataset2024_v29・
     FOLD_OF train=3（get_params が名前を確かめる）。
  9. NumPy 2（venv は 2.4）: 公式の SELD_evaluation_metrics.py は np.NaN（NumPy 2.0 で削除・試走 8604 で止まった）、cls_feature_class.py の salsalite の枝は np.int／np.float
     を使う。公式ファイルは変えず、setup_path で np.NaN = np.nan・np.int = int・np.float = float の別名を numpy に足す（値は同じ・計算は変わらない）。
  10. 学習の最後の test の印字（jackknife の信頼区間）: 公式 get_SELD_Results(is_jackknife=True) はクラス別の区間を 13 クラス固定で reshape(7, 13, 2) する
     （試走 8610・9 クラスだと落ちる）。公式ファイルは変えず patch_compute_results で包み、jackknife のときは区間を nan にして同じ形で返す（点の値は公式のまま・
     モデルの保存は印字より前・学習と val の最良 epoch 選びには関係ない。この研究は公式の指標を報告に使わない）。
  11. log-mel の top_db（Astra RI73-02・2026-10-21）: 公式は librosa.power_to_db の既定 top_db=80 = 本全体の最大から 80 dB 下を下限にする → 未来の大きな音で
     過去のフレームの値が変わる（学習の特徴 300 本で 22% の本・最大で要素の 76% が下限に当たっていた）。top_db=None（フレームごとの純粋な log）にし、
     学習の特徴もこれで作り直した（B_feat_v29c・旧 B_feat_v29 は使わない）。推論は _baseline_B_infer.feat_causal: 窓の最後の特徴フレームだけが判定時刻を
     32 サンプル（1.3 ms）越えるので、その 1 フレームを「判定時刻までの音の尻」から作り直す（他のフレームは本全体の STFT と同じ = 未来を含まない）。
  12. 公式の評価の本の長さ（Astra RI73-06）: 公式は正解の最後のフレーム番号を本の長さにするので、それ以後の予測（誤検出）が落ちる → 全部の本を 10 s = 枠 100 で
     読み直す（patch_compute_results・正解はそのまま）。= 公式の val の指標（最良 epoch 選び）が本の後半の誤検出も数える。公式からの追加の適応。
  13. 距離の付け方（Astra RI73-03・_baseline_B_infer.py）: 重複の整理（_baseline_merge）で残した候補と**同じ元トラック**の距離を付ける（map_tracks の return_src）。
     向きの内積で選ぶと長さの大きい別の音の距離が付いていた。
  14. 公式の既定の学習は「合成データで学習した重みからの微調整（finetune_mode）」。この研究の B は白紙から（finetune_mode=False・4）。章に書く。
本体 §3-6（2026-10-26・§9-1 の 6・§9-4 の 3・M11）:
  15. 入力の倍率: B_INPUT_GAIN_DB（主と同じ決めた倍率・既定 0 = 何もしない）を、公式の特徴を作る前の波形に掛ける（_load_audio・推論の feat_causal も同じ）。
      特徴の置き場（B_FEAT_DIR）に印 B_input_settings.json（input_gain_db・dist_encoding）を書き、学習・推論は環境の値と照合して違えば止める
      （印の無い旧の置き場 = 倍率 0・距離 linear と読む）。「B にゲインは無い」は v29 までの話（M11 で直した）。
  16. 距離のラベルの圧縮: B_DIST_ENCODING = linear（公式のまま m・既定）か log1p（主と同じ u = log1p(d)/log1p(100)）。log1p のとき、公式の ADPIT ラベルの
      距離の欄（探り当てた軸と位置・probe_dist_axis）を u に替え、出力の復号（公式 get_multi_accdoa_labels を包む）で d = expm1(max(u, 0)·log1p(100)) に戻す
      （公式の val の指標〔最良 epoch 選び〕もこの研究の推論も m で比べる）。sde_patch.dist_encode／dist_decode と同じ式。
  17. 1 epoch の損失の内訳（patch_loss_breakdown・B_LOSS_BREAKDOWN=1）: 公式 MSELoss_ADPIT の _each_calc の値はそのまま返し（学習は変えない）、
      同じ二乗誤差から距離の欄（軸 2 の 4 つおきの最後 = 3, 7, 11）の和 ÷ 要素数を別に取り、公式が選ぶ順列（13 通りの最小）と同じ所を足す。
      再構成した損失が公式の損失と相対 1e-4 を超えて違えば止める（分け方の前提が崩れた）。比率 = Σ(W_DIST × 距離の項) ÷ Σ(損失の全体)（1 epoch の全バッチ・
      W_DIST = 1〔公式に距離の重みは無い〕）。> 0.5 で支配（ちょうど 0.5 は支配でない・非有限は止める）。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HOME = Path.home()
E2E = Path(os.environ.get("E2E", HOME / "research/outdoor_seld_e2e"))
OFFICIAL = Path(os.environ.get("B_OFFICIAL", HOME / "research/_external/DCASE2024_seld_baseline"))
VENDOR = Path(os.environ.get("B_VENDOR", E2E / "server_sde/_vendor"))
SRC = Path(os.environ.get("B_SRC_BUNDLE", E2E / "out/dataset_outdoor_siren_v2final_v29"))      # v29 の束（foa/ metadata_dist/ plan/）
DS = Path(os.environ.get("B_DATASET_DIR", HOME / "research/_external/B_dataset2024_v29"))   # 名前に 2024 が要る（公式が分割を名前で決める）            # 公式の層: foa_dev/<sub>/fold?_*.wav・metadata_dev/<sub>/fold?_*.csv
FEAT = Path(os.environ.get("B_FEAT_DIR", HOME / "research/_external/B_feat_v29c"))               # 公式の feat_label_dir（c = top_db なしの特徴・RI73-02 で作り直し。旧 B_feat_v29 は使わない）
MODEL_DIR = Path(os.environ.get("B_MODEL_DIR", HOME / "research/_external/B_models_v29"))
OUT_DIR = Path(os.environ.get("B_OUT_DIR", HOME / "research/_external/B_out_v29"))
N_CLASSES = 9
FOLD_OF = {"train": 3, "val": 4}            # 公式 2024 の dev モードの分割 = train [3]・val [4]・test [4]（train_seldnet.main:360〜363・dataset_dir に '2024' を含むとき）

# ---- 本体 §3-6（docstring の 15〜17）
LOG_K = math.log1p(100.0)
DIST_ENCODING = os.environ.get("B_DIST_ENCODING", "linear").strip() or "linear"
if DIST_ENCODING not in ("linear", "log1p"):
    raise SystemExit(f"B_DIST_ENCODING は linear か log1p（{DIST_ENCODING}）")
INPUT_GAIN_DB = float(os.environ.get("B_INPUT_GAIN_DB", "0") or 0.0)
if not math.isfinite(INPUT_GAIN_DB):
    raise SystemExit(f"B_INPUT_GAIN_DB が有限でない: {INPUT_GAIN_DB}")
INPUT_GAIN_EXPLICIT = os.environ.get("B_INPUT_GAIN_DB", "").strip() != ""
GAIN = 10.0 ** (INPUT_GAIN_DB / 20.0)
FEAT_MARKER = "B_input_settings.json"
W_DIST = 1.0                                # 公式の損失に距離の重みは無い（比率の式の W_DIST = 1）


def dist_encode_log(d):
    """m → u = log1p(d)/log1p(100)（負は 0 に・配列可）。"""
    import numpy as np
    return np.log1p(np.maximum(np.asarray(d, dtype=np.float64), 0.0)) / LOG_K


def dist_decode_log(u):
    """u → d = expm1(max(u, 0)·log1p(100))（numpy か torch）。"""
    if hasattr(u, "detach"):
        import torch
        return torch.expm1(torch.clamp(u, min=0.0) * LOG_K)
    import numpy as np
    return np.expm1(np.maximum(np.asarray(u, dtype=np.float64), 0.0) * LOG_K)


def loss_ratio_verdict(sum_dist_weighted: float, sum_total: float) -> tuple[float, bool]:
    """比率 = Σ(W_DIST × 距離の項) ÷ Σ(損失の全体)。> 0.5 で支配（ちょうど 0.5 は支配でない）。非有限・全体 ≤ 0 は止める（ValueError）。"""
    if not (math.isfinite(sum_dist_weighted) and math.isfinite(sum_total)) or sum_total <= 0 or sum_dist_weighted < 0:
        raise ValueError(f"損失の内訳が非有限か全体が 0 以下（距離 {sum_dist_weighted}・全体 {sum_total}）→ 止める")
    r = sum_dist_weighted / sum_total
    return r, bool(r > 0.5)


def feat_marker_want() -> dict:
    return {"input_gain_db": INPUT_GAIN_DB, "dist_encoding": DIST_ENCODING}


def check_feat_marker(feat_dir: Path, write: bool = False) -> dict:
    """特徴の置き場の印（倍率・距離の符号化）を環境の値と照合する。無い置き場は旧（0 dB・linear）と読む。write=True（準備）なら無ければ書く。違えば止める。"""
    p = Path(feat_dir) / FEAT_MARKER
    want = feat_marker_want()
    if p.is_file():
        have = json.loads(p.read_text(encoding="utf-8"))
    elif write:
        Path(feat_dir).mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(want, ensure_ascii=False) + "\n", encoding="utf-8")
        have = dict(want)
    else:
        have = {"input_gain_db": 0.0, "dist_encoding": "linear"}
    if float(have.get("input_gain_db", 0.0)) != INPUT_GAIN_DB or have.get("dist_encoding", "linear") != DIST_ENCODING:
        raise SystemExit(f"特徴の置き場 {feat_dir} の印 {have} と環境（B_INPUT_GAIN_DB={INPUT_GAIN_DB:g}・B_DIST_ENCODING={DIST_ENCODING}）が違う（作り直すか置き場を替える）")
    return have


def ensure_stubs() -> None:
    """映像用の import（cv2・matplotlib・IPython・torchvision）の空の代用を _vendor に置く（音だけの学習では呼ばれない）。"""
    V = VENDOR
    (V / "matplotlib").mkdir(parents=True, exist_ok=True)
    if not (V / "matplotlib/__init__.py").exists():
        (V / "matplotlib/__init__.py").write_text("# stub（映像の描画用・音だけの学習では使わない）\n", encoding="utf-8")
    # 公式の cls_feature_class.py:14 と train_seldnet.py:17 は読み込み時に plot.switch_backend('agg') を呼ぶ（描画はしない）。
    # 空の代用だと AttributeError（試走 job 8554・2026-10-21 02:21）→ 何もしない switch_backend を置く（A の試走が作った空の代用も上書きして直す）
    pp = V / "matplotlib/pyplot.py"
    if not pp.exists() or "def switch_backend" not in pp.read_text(encoding="utf-8"):
        pp.write_text("# stub（映像の描画用・音だけの学習では使わない）\n\n\ndef switch_backend(*a, **k):\n    return None\n", encoding="utf-8")
    if not (V / "cv2.py").exists():
        (V / "cv2.py").write_text("# stub（映像の読み込み用・音だけの学習では使わない）\n", encoding="utf-8")
    (V / "IPython").mkdir(exist_ok=True)
    if not (V / "IPython/__init__.py").exists():
        (V / "IPython/__init__.py").write_text("def embed(*a, **k):\n    raise RuntimeError('IPython stub')\n", encoding="utf-8")
    (V / "torchvision/models").mkdir(parents=True, exist_ok=True)
    for p in (V / "torchvision/__init__.py",):
        if not p.exists():
            p.write_text("# stub（映像の特徴用・音だけの学習では使わない）\n", encoding="utf-8")
    if not (V / "torchvision/models/__init__.py").exists():
        (V / "torchvision/models/__init__.py").write_text("def resnet50(*a, **k):\n    raise RuntimeError('torchvision stub')\n\nclass ResNet50_Weights:\n    DEFAULT = None\n", encoding="utf-8")


def numpy2_aliases() -> None:
    """docstring の 9: NumPy 2 で消えた別名（np.NaN・np.int・np.float）を同じ値で足す（公式コードが使う・公式ファイルは変えない）。"""
    import numpy as np
    for name, val in (("NaN", np.nan), ("int", int), ("float", float)):
        try:
            getattr(np, name)
        except AttributeError:
            setattr(np, name, val)


def setup_path() -> None:
    ensure_stubs()
    numpy2_aliases()
    for p in (str(VENDOR), str(OFFICIAL)):
        if p not in sys.path:
            sys.path.insert(0, p)


def get_params(nb_epochs: int | None = None, quick: bool = False, batch_size: int | None = None) -> dict:
    """公式の task 3（FOA・multi-ACCDOA）の params を取り、この研究用に上書きする。"""
    setup_path()
    import parameters  # noqa: E402  公式
    P = parameters.get_params("3")
    P["dataset"] = "foa"
    P["multi_accdoa"] = True
    P["modality"] = "audio"
    P["use_salsalite"] = False
    P["finetune_mode"] = False
    P["dataset_dir"] = str(DS) + "/"
    if "2024" not in str(DS) or "2023" in str(DS) or "2022" in str(DS):   # 公式は dataset_dir の名前で分割を決める（2024 → train fold3・val/test fold4）。試走 8586 の教訓
        raise SystemExit(f"B の置き場の名前に 2024 を入れる（2023／2022 は入れない）: {DS}")
    P["feat_label_dir"] = str(FEAT) + "/"
    P["model_dir"] = str(MODEL_DIR)
    P["dcase_output_dir"] = str(OUT_DIR)
    P["unique_classes"] = N_CLASSES
    if nb_epochs is not None:
        P["nb_epochs"] = int(nb_epochs)
        P["patience"] = int(nb_epochs)
    if batch_size is not None:
        P["batch_size"] = int(batch_size)
    P["quick_test"] = bool(quick)
    return P


def feature_class(params: dict, is_eval: bool = False):
    import cls_feature_class  # noqa: E402  公式
    import numpy as np
    import soundfile as sf

    class FeatureClassFlac(cls_feature_class.FeatureClass):
        """公式 FeatureClass の音声の読み方だけ差し替え（flac のシンボリックリンク・float のまま）。特徴の計算は公式のまま。"""

        def _load_audio(self, audio_path):
            audio, fs = sf.read(audio_path, dtype="float32")
            if audio.ndim == 1:
                audio = audio[:, None]
            if INPUT_GAIN_DB != 0.0:                                  # docstring の 15: 倍率は eps の前（推論の feat_causal と同じ）
                return audio[:, : self._nb_channels].astype(np.float64) * GAIN + self._eps, fs
            return audio[:, : self._nb_channels].astype(np.float64) + self._eps, fs

        def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
            """docstring の 16: B_DIST_ENCODING=log1p のとき、公式の ADPIT ラベルの距離の欄を u = log1p(d)/log1p(100) に替える（他の欄はそのまま）。"""
            lab = super().get_adpit_labels_for_file(_desc_file, _nb_label_frames)
            if DIST_ENCODING != "log1p":
                return lab
            ax, idx = probe_dist_axis(self)
            sl = [slice(None)] * lab.ndim
            sl[ax] = idx
            lab[tuple(sl)] = dist_encode_log(lab[tuple(sl)])
            return lab

        def get_frame_stats(self):
            if len(self._filewise_frames) != 0:
                return
            for sub_folder in sorted(os.listdir(self._aud_dir)):
                loc = os.path.join(self._aud_dir, sub_folder)
                for file_name in sorted(os.listdir(loc)):
                    n = sf.info(os.path.join(loc, file_name)).frames
                    self._filewise_frames[file_name.split(".")[0]] = [int(n / float(self._hop_len)), int(n / float(self._label_hop_len))]

        def _get_mel_spectrogram(self, linear_spectra):
            """docstring の 11（Astra RI73-02）: 公式は librosa.power_to_db の既定 top_db=80 で「本全体の最大 − 80 dB」を下限にする = 未来の大きな音で
            過去のフレームの値が変わる（学習の特徴 300 本で 22% の本に起きていた・最大で要素の 76%）。top_db=None にして純粋にフレームごとの log にする。
            学習と推論で同じこの変換を使う（学習の特徴もこれで作り直した）。式はそれ以外は公式と同じ。"""
            import librosa
            mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
            for ch_cnt in range(linear_spectra.shape[-1]):
                mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
                mel_spectra = np.dot(mag_spectra, self._mel_wts)
                mel_feat[:, :, ch_cnt] = librosa.power_to_db(mel_spectra, top_db=None)
            return mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    return FeatureClassFlac(params=params, is_eval=is_eval)


_PROBE = {}


def probe_dist_axis(fc) -> tuple[int, int]:
    """公式の ADPIT ラベルで距離が入る軸と位置を、架空の本で探り当てる（公式の実装に依らない）: 同じ frame に クラス 4 の 2 音（2.00 m・3.00 m）と
    クラス 1（2.50 m）を置き、3 つの値がどれも同じ位置に出る軸を探す（クラスの軸は 1 と 4 で、重複の軸は 2 音で分かれる）。1 つに決まらなければ止める。"""
    if "axis" in _PROBE:
        return _PROBE["axis"]
    import tempfile
    import numpy as np
    tmp = Path(tempfile.mkdtemp(prefix="B_probe_")) / "p.csv"
    tmp.write_text("".join(f"{k},4,0,30,0,200\n{k},4,1,-60,0,300\n{k},1,0,120,0,250\n" for k in range(10, 20)), encoding="utf-8")
    desc = fc.load_output_format_file(str(tmp))
    cart = fc.convert_output_format_polar_to_cartesian(desc)
    lab = np.asarray(type(fc).__mro__[1].get_adpit_labels_for_file(fc, cart, 100))      # 公式の関数（包む前）
    hits = {v: np.argwhere(np.isclose(lab, v, atol=1e-6)) for v in (2.0, 3.0, 2.5)}
    if any(len(h) == 0 for h in hits.values()):
        raise SystemExit(f"距離の値（2.0・3.0・2.5 m）が ADPIT ラベルに出ない（形 {lab.shape}）→ 止める")
    cand = []
    for ax in range(1, lab.ndim):
        idxs = {int(i) for h in hits.values() for i in h[:, ax]}
        if len(idxs) == 1:
            cand.append((ax, idxs.pop()))
    if len(cand) != 1:
        raise SystemExit(f"ADPIT ラベルの距離の軸が 1 つに決まらない（候補 {cand}・形 {lab.shape}）→ 止める")
    _PROBE["axis"] = cand[0]
    _PROBE["shape"] = list(lab.shape)
    print(f"[B-common] ADPIT ラベルの距離の欄 = 軸 {cand[0][0]} の位置 {cand[0][1]}（形 {lab.shape}）", flush=True)
    return cand[0]


def patch_dist_decode() -> None:
    """docstring の 16: B_DIST_ENCODING=log1p のとき、公式 get_multi_accdoa_labels の距離の出力（dist0・dist1・dist2）を m に戻す（val の指標と推論の両方）。"""
    if DIST_ENCODING != "log1p":
        return
    import train_seldnet  # noqa: E402  公式
    if getattr(train_seldnet.get_multi_accdoa_labels, "_dist_decoded", False):
        return
    orig = train_seldnet.get_multi_accdoa_labels

    def get_multi_accdoa_labels(*a, **k):
        r = list(orig(*a, **k))
        if len(r) != 9:
            raise RuntimeError(f"公式 get_multi_accdoa_labels の返り値が 9 個でない（{len(r)}）→ 距離の位置が分からないので止める")
        for i in (2, 5, 8):
            r[i] = dist_decode_log(r[i])
        return tuple(r)
    get_multi_accdoa_labels._dist_decoded = True
    train_seldnet.get_multi_accdoa_labels = get_multi_accdoa_labels


def patch_loss_breakdown(out_json: Path, params: dict | None = None) -> dict:
    """docstring の 17: 学習の損失（公式 MSELoss_ADPIT）を距離の項と全体に分けて、train_epoch ごとに out_json へ書く（学習の値は変えない）。返り値 = 状態。"""
    import seldnet_model  # noqa: E402  公式
    import train_seldnet  # noqa: E402  公式
    import torch
    L = seldnet_model.MSELoss_ADPIT
    st = {"on": False, "calls": [], "sum_total": 0.0, "sum_dist": 0.0, "n_batches": 0, "max_rel_err": 0.0, "epochs": 0, "shape": None}
    orig_each, orig_call = L._each_calc, L.__call__

    def _each_calc(self, output, target):
        tot = orig_each(self, output, target)
        if st["on"]:
            sq = self._each_loss(output, target)
            if sq.dim() != 4 or sq.shape[2] % 4 != 0:
                raise RuntimeError(f"二乗誤差の形が (B, T, 3×4, C) でない: {tuple(sq.shape)} → 止める")
            if not torch.allclose(sq.mean(dim=2), tot, rtol=1e-5, atol=1e-8):
                raise RuntimeError("公式の _each_calc が「二乗誤差の軸 2 の平均」でない → 分け方の前提が崩れたので止める")
            st["shape"] = list(sq.shape)
            st["calls"].append((tot.detach(), (sq[:, :, 3::4, :].sum(dim=2) / sq.shape[2]).detach()))
        return tot

    def __call__(self, output, target):
        st["calls"] = []
        loss = orig_call(self, output, target)
        if st["on"]:
            tots = torch.stack([c[0] for c in st["calls"]]); dists = torch.stack([c[1] for c in st["calls"]])
            idx = tots.argmin(dim=0, keepdim=True)
            sel_t = tots.gather(0, idx)[0]; sel_d = dists.gather(0, idx)[0]
            lt, rt, rd = float(loss.detach()), float(sel_t.mean()), float(sel_d.mean())
            if not (math.isfinite(lt) and math.isfinite(rd)):
                raise RuntimeError(f"損失が非有限（全体 {lt}・距離 {rd}）→ 止める")
            rel = abs(rt - lt) / max(abs(lt), 1e-12)
            if rel > 1e-4:
                raise RuntimeError(f"再構成した損失 {rt} が公式の損失 {lt} と違う（相対 {rel:.2e}）→ 分け方の前提が崩れたので止める")
            st["max_rel_err"] = max(st["max_rel_err"], rel)
            st["sum_total"] += lt; st["sum_dist"] += W_DIST * rd; st["n_batches"] += 1
            st["calls"] = []
        return loss

    L._each_calc, L.__call__ = _each_calc, __call__
    orig_train = train_seldnet.train_epoch

    def train_epoch(*a, **k):
        st["on"] = True
        try:
            r = orig_train(*a, **k)
        except BaseException as e:                                  # 途中で止まったとき: 元のエラーを記録に書いて、元のエラーのまま止める
            st["on"] = False
            st["epochs"] += 1
            write_breakdown(out_json, st, params, error=f"{type(e).__name__}: {e}")
            raise
        st["on"] = False
        st["epochs"] += 1
        write_breakdown(out_json, st, params)
        return r
    train_seldnet.train_epoch = train_epoch
    return st


def write_breakdown(out_json: Path, st: dict, params: dict | None = None, error: str | None = None) -> dict:
    """内訳を書く。比率が出せない（非有限・全体 0）ときは error を書いてから止める。error（途中で止まった元のエラー）を渡したときは書くだけ（止めるのは呼ぶ側）。"""
    rec = {"rule": "比率 = Σ(W_DIST × 距離の項) ÷ Σ(損失の全体)（1 epoch の全バッチ・公式 MSELoss_ADPIT の選んだ順列）。> 0.5 で支配（ちょうど 0.5 は支配でない）・非有限は止める（本体 §9-4 の 3）",
           "w_dist": W_DIST, "n_batches": st["n_batches"], "epochs": st["epochs"], "sum_total": st["sum_total"], "sum_dist_weighted": st["sum_dist"],
           "max_rel_err_reconstruction": st["max_rel_err"], "sq_shape": st["shape"], "dist_encoding": DIST_ENCODING, "input_gain_db": INPUT_GAIN_DB,
           "official_dist_params": {k: v for k, v in (params or {}).items() if "dist" in str(k).lower()}}
    if error is not None:
        rec["error"] = error
    else:
        try:
            rec["ratio"], rec["dominant"] = loss_ratio_verdict(st["sum_dist"], st["sum_total"])
        except ValueError as e:
            rec["error"] = str(e)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out_json).with_suffix(".part")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out_json)
    print(f"[B-loss] epoch {st['epochs']}・バッチ {st['n_batches']}・比率 {rec.get('ratio')}・支配 {rec.get('dominant')} → {out_json}", flush=True)
    if "error" in rec and error is None:
        raise RuntimeError(rec["error"])
    return rec


NB_LABEL_FRAMES = 100       # 10 s クリップのラベルの枠数（label_hop 0.1 s）


def patch_compute_results() -> None:
    """公式 ComputeSELDResults.__init__ を包む（docstring の 7）。正解の csv が空の本で max() が止まらないように、
    中身のある本だけ（シンボリックリンクの一時の置き場）で公式の初期化を通し、空の本は『枠 100・正解なし』として足す。公式ファイルは変えない。"""
    import cls_compute_seld_results as CR  # noqa: E402  公式
    if getattr(CR.ComputeSELDResults, "_empty_ok", False):
        return
    orig = CR.ComputeSELDResults.__init__

    def init(self, params, ref_files_folder=None):
        desc = Path(ref_files_folder) if ref_files_folder else Path(params["dataset_dir"]) / "metadata_dev"
        tmp = Path(params["feat_label_dir"]) / "_ref_nonempty"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        empties = []
        for split in sorted(os.listdir(desc)):
            (tmp / split).mkdir(parents=True)
            for f in sorted(os.listdir(desc / split)):
                src = desc / split / f
                if src.stat().st_size == 0:
                    empties.append(f)
                else:
                    (tmp / split / f).symlink_to(src.resolve())
        orig(self, params, ref_files_folder=str(tmp))
        self._desc_dir = str(desc)
        # docstring の 12（Astra RI73-06）: 公式は正解の最後のフレーム番号を本の長さにするので、それ以後の予測（誤検出）が採点から落ちる。
        #   中身のある本も全部 10 s = 枠 100 で読み直す（正解はそのまま・採点の範囲だけ本の長さに）。
        n_full = 0
        for split in sorted(os.listdir(desc)):
            for f in sorted(os.listdir(desc / split)):
                if f in empties:
                    continue
                gt = self._feat_cls.load_output_format_file(str(desc / split / f), cm2m=True)
                gt = self._feat_cls.convert_output_format_polar_to_cartesian(gt)
                lab = self._feat_cls.segment_labels(gt, NB_LABEL_FRAMES) if self.segment_level else self._feat_cls.organize_labels(gt, NB_LABEL_FRAMES)
                self._ref_labels[f] = [lab, NB_LABEL_FRAMES]
                n_full += 1
        for f in empties:
            lab = self._feat_cls.segment_labels({}, NB_LABEL_FRAMES) if self.segment_level else self._feat_cls.organize_labels({}, NB_LABEL_FRAMES)
            self._ref_labels[f] = [lab, NB_LABEL_FRAMES]
        self._nb_ref_files = len(self._ref_labels)
        self._n_empty_ref = len(empties)
        print(f"[B-common] 公式の評価の正解: 中身あり {n_full:,}・正解なし {len(empties):,}・どちらも枠 {NB_LABEL_FRAMES}（10 s）で採点", flush=True)

    CR.ComputeSELDResults.__init__ = init
    orig_get = CR.ComputeSELDResults.get_SELD_Results

    def get_results(self, pred_files_path, is_jackknife=False):
        """docstring の 10: jackknife の区間（13 クラス固定で落ちる）は nan にし、点の値は公式のまま同じ形で返す。"""
        if not is_jackknife:
            return orig_get(self, pred_files_path, is_jackknife=False)
        import numpy as np
        res = orig_get(self, pred_files_path, is_jackknife=False)
        *scalars, classwise = res
        cw = np.asarray(classwise)
        cw_ci = np.full((cw.shape[0], cw.shape[1], 2), np.nan) if cw.size else []
        return tuple([v, [float("nan"), float("nan")]] for v in scalars) + ([classwise, cw_ci],)

    CR.ComputeSELDResults.get_SELD_Results = get_results
    CR.ComputeSELDResults._empty_ok = True
