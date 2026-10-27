# -*- coding: utf-8 -*-
"""v30 描画 = v29 の鎖 ＋ 学習し直し（変更票 #75）の描画の直し（Fable 仕様（先行分）・仕様（本体）2026-10-26）。

**v30 の直しは v30 の入口（generate_clip・sample_scene・active()）の中だけで有効**（Astra R30-08「v30 のファイルの入口だけで有効化・共有関数の既定は旧」）。
import しただけでは何も変わらない: 同じプロセスで v29 の入口（v19r.generate_clip）を呼べば v29 と 1 バイトも違わない
（v29 単独・v30 単独・v29 → v30 の順に import の 3 通りで試験 = scripts/test_render_mic_heading.py の B11）。
足したものから順に書く。FEATURES は試験で 1 つずつ切って比べるためのもので、本番は全部入。

  1. 先行分 §1（決定 ⑩）: マイクの正面を進む向き ψ(t) に回す（入口の中で m9.MIC_HEADING = True）。
     FOA（m9._render_stem と v13._render_stem_preroll の direct・mirror）とラベルの方位（frame_label_rows(heading_deg=…)）を同じ R_z(−ψ) で回す。
     距離・仰角・時刻・masks（W は向きで変わらない）・乱数の消費は変えない。暗騒音・雨（拡散音）は回さない（等方）。
     規約（src/outdoor_seld/geometry.py の mic_heading_deg・rotate_to_mic_frame・az_to_mic_frame）:
       世界座標 +x 前・+y 左・+z 上、az = atan2(y, x)（左が正）、FOA は W,Y,Z,X（ACN／SN3D）。
       ψ(t) = atan2(vy, vx)（マイク経路の水平速度・上から見て反時計回りが正＝実録の yaw と同じ向き）。止まっている区間は直前の向き
       （最初から止まっていれば最初に動いた向き）・全部静止なら 0。ψ は**受音時刻**のもの（放射時刻ではない）。az_mic = wrap(az_world − ψ)（(−180, 180]）。
       歩行 dirx = +1 → ψ = 0（v29 と同じ出力）・dirx = −1 → ψ = 180・walk_stop_x → 0・walk_cross_y → 90・静止 → 0。
     scene.json: mic.mic_heading_deg（一定なら数・変わるなら [[t0, ψ], …]）・音源ごとの side_mic（最接近の見かけの方位の符号・マイク座標の L/R）。
     plan の列（car_side・w1_side など）は世界座標のまま。距離ラベル（step19_dist_labels_v16）は mic_heading_deg を読んで照合の門に同じ向きを通す。
  2. 先行分 §2（決定 ⑪）: クラクションの地面反射の二重がけを直す。法規の前方 7 m の値 lv を 20 log10(7) で 1 m に戻した l1m から
     HORN_REFL_DELTA_DB を引く（l1m = lv + 20 log10(7) − Δ・場面のあと・ドライ音の前＝サイレンの #61 と同じ置き場所）。
     Δ = 代表型（単音・長押し・音源はクラクションの高さの真ん中・マイク 1.0 m・前方 7 m・本番の鏡像）の反射込みの 7 m の受音（LAFmax）− lv
     （Fable 10/26 14:39 の (b)・#61 と同じ定義・基準点の代表値＝Astra R30-11）。出し方は scripts/level_match_horn_v30.py・md/design/level_match_horn_v30.md
     （二音・短押し・高さの両端の残差もそこ）。scene.json のクラクションに horn_refl_delta_db と l1m_before_horn_delta_db（引く前の値）。
     v24 の最大値合わせは引いたあとの値を宣言値 l1m_decl_db にする。ほかのクラスは触らない。
  3. 本体 §2-1（決定 ①・R30-01）: 正解の線 AUDIBLE_SNR_DB_V30 = −5.0 dB（学習用ラベルの操作的な線・人の可聴性を保証しない）。背景 = 暗騒音＋雨＋
     同じ場面の自分以外の全 stem（ラベル無しの電車を含む）: 音源 i の SN 比 = frame_spl_a(stem_i) − frame_spl_a(noise + Σ_{j≠i} stem_j)（= mix − stem_i）。
     masks は frame,class,track,src,snr_a_db,snr_a_bg_db（src = scene の音源の番号〔R30-10 の安定した source id〕・snr_a_db = 新しい定義・snr_a_bg_db = 旧の量〔暗騒音＋雨〕）。scene.json に gate_rule "v30"・
     gate_line_db −5.0・gate_bg "noise+others"。audible_frac・n_label_frames は新しい定義。隙間埋め GAP_FILL 2 はそのまま・D14 の電車は正解なしのまま。
  5. 本体 §2-3（R1 (b)）: 背景の形を本ごとに変える（折れ点 63〜1,000 Hz の対数一様・その上の傾き U(−5, −3) dB/oct・63 Hz より下は v12 のまま・
     src/outdoor_seld/noise_v30.py）。乱数は別の流れ rng([seed, v29.SALT29, V30_KEYS["bg_shape"]])（既存の抽選の列を 1 つも動かさない・高さの双子は同じ seed なので
     同じ形）。暗騒音の白色雑音は同じ乱数をそのまま使い、形だけ変える。scene.json の noise に shape {knee_hz, slope_db_oct}。
     **2026-10-27（決め直し 5 の結果・§9-7・Fable 07:3x）: 既定は入・傾きの幅は (ii) = U(−6.5, −2.5) dB/oct**（val のクラス平均 F で (ii) が最大・
     md/design/v30_bgcmp_decision_2026-10-27.json）。折れ点の幅は 63〜1,000 Hz のまま。V30_BG_SHAPE=0 で切（試験と比べの再現用）。部品 noise_v30 の既定の幅（−5〜−3）は変えず、
     v30 の描画が渡す幅だけ (ii) にする。scene.json の noise.shape に幅（knee_range_hz・slope_range_db_oct）を書く。
  4. 本体 §2-2: 踏切（crossing）もゲートに入れる（GATED_V30 = v13.VEHICLE_GATED_V13 ∪ {crossing}・背景は 3 の定義＝電車が入る）。
     t_first_audible_s（鳴り始め）= 最初に正解の付いた枠（v30 のゲートを掛けるクラス全部）。
  5. 帯域ゲート（**決め直し 4・本人 10/26 19:32・仕様 §9-5 = v30 の既定は入・線 0 dB**・3 の広帯域 −5 の線を置き換える・+3 は分析の併記だけ）: 判定量 = 一番有利な 1/3 オクターブ帯の
     SN 比（src/outdoor_seld/band_gate.py・125 Hz〜10 kHz の 20 帯・0.1 s・窓なし rfft・背景は 3 と同じ）≥ GATE_BAND_DB（環境変数 V30_GATE_BAND_DB・候補 0／+3）で正解・
     隙間埋め 2 はそのまま。masks は 9 列（＋ snr_band_max_db・band_hz・band_valid〔§9-6: stem と背景が両方とも床以下の帯は無効・全帯が無効な枠は 0 = 不通過〕・広帯域の値は併記のまま）・非有限は band_gate が止める・scene に来歴 gate_band（内容 hash・線・旗・枠長・床）・scene の gate_rule "v30_band"・gate_line_db = 帯の線・
     gate_broadband_line_db −5（参考）・音源ごとに gate_band_max_db／gate_band_hz_at_max。切（FEATURES["band_gate"] = False）のときは 3 の広帯域 −5 の道で、
     出力は 8ce0a65d3 の前（74c7718c6）と 1 バイトも同じ（試験の比べ・感度用に残す）。

使い方: v30.generate_clip(row)（描き直しの生成ドライバは発効のあとにこれを呼ぶ）。場面だけ: v30.sample_scene(row)。
試験: scripts/test_render_mic_heading.py（§1）・scripts/test_v30_horn_delta.py（§2）・scripts/test_v30_gate.py（3・4）・scripts/test_v30_band_gate.py（5）
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import step11_v29_render as v29  # noqa: E402  鎖の前（v9 → … → v29）を全部読む

m9, v13, v19r = v29.m9, v29.v19r.v13, v29.v19r

FEATURES = {"mic_heading": True, "horn_delta": True, "gate_v30": True,
            "bg_shape": True,    # 試験で 1 つずつ切るときだけ False（本番は全部入）。2026-10-27 決め直し 5 の結果で既定は入・幅は (ii)（V30_BG_SHAPE=0 で切）
            "band_gate": True}   # 帯域ゲート（決め直し 4・本人 10/26 19:32・既定は入）= 一番有利な 1/3 オクターブ帯の SN 比 ≥ GATE_BAND_DB で正解（広帯域 −5 は masks に併記だけ）
# 帯域ゲートの線（既定 0 dB・+3 は分析の併記だけ・環境変数 V30_GATE_BAND_DB で替える・FEATURES["band_gate"] が切なら使わない）
GATE_BAND_DB = float(os.environ.get("V30_GATE_BAND_DB", "0") or 0.0)


def _range_env(name: str, default: tuple) -> tuple:
    v = os.environ.get(name, "").strip()
    if not v:
        return (float(default[0]), float(default[1]))
    a, b = (float(x) for x in v.split(","))
    assert a < b, f"{name} は 小さい方,大きい方: {v}"
    return (a, b)


# 決め直し 5（本人 10/26 20:03・仕様 §9-7）: 背景の形の旗。2026-10-27 の結果で既定は入・傾きの幅は (ii)。V30_BG_SHAPE=0 で切・=1 で入（未設定 = 既定の入）。
#   幅 = V30_BG_KNEE_RANGE（既定 "63,1000"）・V30_BG_SLOPE_RANGE（既定 "-6.5,-2.5" = (ii)・比べの (i) は "-5,-3"）。乱数の鍵 bg_shape=21 は同じ（幅だけ変える・同じ U）。
from outdoor_seld import noise_v30 as _nv30  # noqa: E402
BG_SLOPE_RANGE_V30 = (-6.5, -2.5)          # 決め直し 5 の (ii)（部品 noise_v30 の既定 −5〜−3 とは別に、v30 の描画の既定として持つ）
BG_KNEE_RANGE = _range_env("V30_BG_KNEE_RANGE", _nv30.KNEE_RANGE_HZ)
BG_SLOPE_RANGE = _range_env("V30_BG_SLOPE_RANGE", BG_SLOPE_RANGE_V30)
_bgs = os.environ.get("V30_BG_SHAPE", "").strip()
if _bgs in ("0", "1"):
    FEATURES["bg_shape"] = (_bgs == "1")
elif _bgs:
    raise ValueError(f"V30_BG_SHAPE は 0 か 1（{_bgs!r}）")
_ACTIVE = {"on": False}

# 3・4. 正解の線と背景の定義・踏切のゲート（本体 §2-1・§2-2）
AUDIBLE_SNR_DB_V30 = -5.0
GATED_V30 = frozenset(set(v13.VEHICLE_GATED_V13) | {"crossing"})

# 5. 背景の形の乱数の鍵（v29 の KEYS に足す形・v29 のファイルは触らない・値は一意）
V30_KEYS = dict(v29.KEYS, bg_shape=21)
assert len(set(V30_KEYS.values())) == len(V30_KEYS), V30_KEYS


def bg_shape_of(s: dict, row: dict) -> dict:
    """本ごとの背景の形（別の乱数の流れ・2 回だけ引く）。幅は BG_KNEE_RANGE・BG_SLOPE_RANGE（§9-7 の旗・既定は今の幅）。scene に幅も書く。"""
    import numpy as _np
    from outdoor_seld.noise_v30 import draw_shape
    d = draw_shape(_np.random.default_rng([int(row["seed"]), v29.SALT29, V30_KEYS["bg_shape"]]), BG_KNEE_RANGE, BG_SLOPE_RANGE)
    return dict(d, knee_range_hz=list(BG_KNEE_RANGE), slope_range_db_oct=list(BG_SLOPE_RANGE))


# 2. クラクションの地面反射の Δ（scripts/level_match_horn_v30.py --n 12 --n-resid 4・2026-10-26 の出力・md/design/level_match_horn_v30.md）
HORN_REFL_DELTA_DB = 2.14


@contextlib.contextmanager
def active():
    """v30 の直しを有効にする範囲（入れ子可・抜けると元の値に戻す）。"""
    prev = (m9.MIC_HEADING, _ACTIVE["on"], v13.GATE_V30, v13.NOISE_SHAPE_V30)
    m9.MIC_HEADING = bool(FEATURES["mic_heading"])
    _ACTIVE["on"] = True
    v13.GATE_V30 = ({"line_db": AUDIBLE_SNR_DB_V30, "gated": GATED_V30, "band_db": (GATE_BAND_DB if FEATURES["band_gate"] else None)}
                    if FEATURES["gate_v30"] else None)
    v13.NOISE_SHAPE_V30 = bg_shape_of if FEATURES["bg_shape"] else None
    try:
        yield
    finally:
        m9.MIC_HEADING, _ACTIVE["on"], v13.GATE_V30, v13.NOISE_SHAPE_V30 = prev


def apply_horn_delta(s: dict) -> dict:
    """場面の音源のうちクラクションだけ、1 m の値から Δ を引く（乱数は使わない）。v30 の入口の外では何もしない。"""
    if not (_ACTIVE["on"] and FEATURES["horn_delta"]):
        return s
    for src in s["sources"]:
        if src.get("class") == "horn":
            src["l1m_before_horn_delta_db"] = float(src["l1m_db"])
            src["l1m_db"] = float(src["l1m_db"]) - HORN_REFL_DELTA_DB
            src["horn_refl_delta_db"] = HORN_REFL_DELTA_DB
    return s


def _scene_v30(prev):
    def fn(row: dict) -> dict:
        return apply_horn_delta(prev(row))
    fn._v30 = True
    return fn


# 場面の関数を包む（v24 が包んだ版の外側・generate_clip_v13 はこの名前を呼ぶ）。包みは入口の外では素通し
if not getattr(v13.sample_scene_v13, "_v30", False):
    _wrapped = _scene_v30(v13.sample_scene_v13)
    v13.sample_scene_v13 = _wrapped
    v19r.sample_scene_v19 = _wrapped


def generate_clip(row: dict) -> None:
    """v30 の 1 本（v29 の生成の梯子をそのまま使い、v30 の直しを有効にして呼ぶ）。"""
    with active():
        return v19r.generate_clip(row)


def sample_scene(row: dict) -> dict:
    """v30 の場面だけ（描かない・検査用）。"""
    with active():
        return v13.sample_scene_v13(row)
