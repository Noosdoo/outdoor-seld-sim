# -*- coding: utf-8 -*-
"""通知規則 v4.2 — 方位主体の最接近予測（2026-08-30 新規。v4/v4.1は変更しない）。

## なぜ作るか（着手前診断 out/notify_v42_diag*/diag.md の結果）

確定評価の至近「強」到達73.0%のボトルネックを、見逃し1件ずつゲート帰属した結果:

  - **confirm不足が58〜74%**: 強条件(dc≤1.0 ∧ tc≤2.5)は成立するのに、推定ノイズで
    フレームごとに条件が明滅し、4フレーム「連続」が成立しない
  - d_cpa の膨張は 11〜15% にすぎない（メモの想定と違い、主因ではなかった）
  - t_cpa 単独の失敗は ≈1%
  - 残りは追跡・帰属（案4の領分）

つまり主敵は**バイアスではなく不安定さ**。対策は部品ごとにフラグで切れるようにする:

  robust    … d²のスケールを瞬時値→窓内中央値、ḋ を最小二乗→Theil–Sen（中央値傾き）
  brg_win   … 方位の窓だけ長くする（方位は誤差中央1〜5度と正確なので長窓の害が小さい）
  m_of_n    … 確認を「4連続」→「直近N中M」（明滅を1フレームの断絶で没収しない）
  route_c   … 方位主導の定方位接近判定: |dθ/dt|≤TH ∧ 接近中 ∧ 近距離。
              船の衝突回避の古典原則（方位が変わらないまま近づく物体は衝突コース）。
              診断で「遠くをまっすぐ通る安全車」も拾うと判明したため距離ゲートDN必須
  link_pred … (案4最小版) 同一物体の連結を「前フレーム方位±60°」→
              「dθ/dt で外挿した予測方位±60°」に

⚠️ しきい値の既定値は v4.1 のまま。**選定はチューニング専用の新val（新しい乱数）で行う。**
既存valは v4.1 の選定で使用済みのため、ここでは回帰確認と煙試験にしか使わない。

全部品OFFの既定設定は fires_cpa (v4.1) と**完全に同一の出力**になる（回帰試験で保証）。

使い方:
    python scripts/step12_notify_v42_bearing.py <pred_val_all.csv> <出力dir> [--set k=v ...]
評価・比較は scripts/_notify_v42_eval.py（importlibの罠があるため専用evalを使う）。
"""
from __future__ import annotations

import importlib.util
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location(
    "nv4", ROOT / "scripts" / "step12_notify_v4_ttc.py")
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)

FPS = v4.FPS


@dataclass(frozen=True)
class Cfg:
    """v4.2 の部品スイッチとしきい値。既定＝v4.1と同一挙動。

    robust は当初1フラグだったが、煙試験で「中央値スケールは接近中に遅れて
    d_cpa を膨らませる（強到達 86.7→74.0%）」と判明したため、傾きとスケールに分割した。
    """
    robust_slope: bool = False  # ḋ を最小二乗 → Theil–Sen（中央値傾き）
    robust_scale: bool = False  # d²のスケールを瞬時値 → 窓内中央値（⚠️遅れの害あり）
    vel_win: int = v4.VEL_WIN   # 距離の窓（5=0.5s）
    brg_win: int = v4.VEL_WIN   # 方位の窓（v4.1は距離と同じ5）
    confirm_m: int = v4.CONFIRM_CPA   # 直近N中M（M=N=4 → v4.1の「4連続」と同一）
    confirm_n: int = v4.CONFIRM_CPA
    route_c: bool = False       # 定方位接近の判定経路
    adot_th: float = 0.10       # [rad/s] route_c: これ以下なら「方位が動かない」
    dn: float = 15.0            # [m]     route_c: 距離ゲート（遠方の直進通過を除外）
    v_close: float = 0.3        # [m/s]   route_c: 接近判定のしきい値
    link_pred: bool = False     # (案4) 予測方位で連結
    cpa_strong: float = v4.CPA_STRONG_M
    cpa_mid: float = v4.CPA_MID_M


def robust_stats(d_at, j, win):
    """窓内の (中央値[m], Theil–Sen傾き[m/s])。窓が埋まらなければ (None, None)。"""
    ds = [d_at.get(k) for k in range(j - win + 1, j + 1)]
    if any(x is None for x in ds):
        return None, None
    med = float(np.median(ds))
    sl = [(ds[b] - ds[a]) * FPS / (b - a)
          for a in range(len(ds)) for b in range(a + 1, len(ds))]
    return med, float(np.median(sl))


def _stream_mn(d_at, az_at, nframes, cond, m, n):
    """cond が「直近n個中m個」成立し、かつ当該フレームでも成立なら発火列に載せる。

    m=n なら v4 の _trigger_stream（m連続）と同一。予測が無いフレームで履歴を
    切るのも v4 と同じ（run=0 相当）。
    """
    hits, hist = [], deque(maxlen=n)
    for j in range(nframes):
        d = d_at.get(j)
        if d is None:
            hist.clear()
            continue
        ok = bool(cond(j, d))
        hist.append(ok)
        if ok and sum(hist) >= m:
            hits.append((j, az_at[j], d))
    return hits


def fires_cpa2(d_at, az_at, nframes, C: Cfg):
    """v4.2: 頑健化した最接近予測 ＋（任意で）定方位接近の経路。"""
    pre = {}
    for j in range(nframes):
        d = d_at.get(j)
        if d is None:
            continue
        if C.robust_slope or C.robust_scale:
            med, ts = robust_stats(d_at, j, C.vel_win)
        else:
            med = ts = None
        if C.robust_slope:
            ddot = ts
        else:
            v = v4.closing_speed(d_at, j, win=C.vel_win)
            ddot = None if v is None else -v
        dbar = med if (C.robust_scale and med is not None) else d
        adot = v4.azimuth_rate(az_at, j, win=C.brg_win)
        dc, tc = ((None, None) if dbar is None
                  else v4.cpa_of(dbar, ddot, adot))
        rc = (C.route_c and adot is not None and abs(adot) <= C.adot_th
              and ddot is not None and ddot < -C.v_close
              and (dbar if dbar is not None else d) <= C.dn)
        pre[j] = (dc, tc, rc)

    def _cond(j, dc_th, tc_th):
        dc, tc, rc = pre.get(j, (None, None, False))
        return (dc is not None and dc <= dc_th and tc <= tc_th) or rc

    # v4.1 と同じく「予測」「距離保険」は別々の列（保険をM/Nで遅らせない）。
    # route_c は強・中の両方の予測列に入れる（強⊂中を保ち、エピソード昇格を壊さない）
    def _stream(dc_th, tc_th, d_th):
        a = _stream_mn(d_at, az_at, nframes,
                       lambda j, d: _cond(j, dc_th, tc_th),
                       C.confirm_m, C.confirm_n)
        b = _stream_mn(d_at, az_at, nframes, lambda j, d: d <= d_th,
                       v4.CONFIRM, v4.CONFIRM)
        return sorted(set(a) | set(b))

    return v4._episodes_with_upgrade(
        _stream(C.cpa_mid, v4.TTC_CAUTION, v4.SUPP),
        _stream(C.cpa_strong, v4.TTC_WARN, v4.T3))


def track_series2(pred_clip, cls, nframes, C: Cfg):
    """(案4最小版) 連結の基準を「前フレームの方位」→「外挿した予測方位」へ。

    link_pred=False なら v4.track_series と同一。予測は直近≤5点の方位の
    最小二乗傾き（±180°折り返しを展開）で1フレーム先へ外挿する。
    """
    if not C.link_pred:
        return v4.track_series(pred_clip, cls, nframes)
    d_at, az_at, hist = {}, {}, []      # hist=[(frame, az)] 連続区間のみ
    for j in range(nframes):
        cand = [(a, d) for (c, a, d) in pred_clip.get(j, [])
                if c == cls and d is not None]
        if not cand:
            hist = []                   # v4 と同じく、途切れたら連結情報を捨てる
            continue
        if hist:
            if len(hist) >= 3:
                pts = hist[-5:]
                t = np.array([p[0] for p in pts]) / FPS
                unw = np.unwrap(np.radians([p[1] for p in pts]))
                slope = np.polyfit(t, unw, 1)[0]
                az_pred = np.degrees(unw[-1] + slope * (j - pts[-1][0]) / FPS)
            else:
                az_pred = hist[-1][1]
            linked = [x for x in cand if v4.cdiff(x[0], az_pred) <= v4.LINK_DEG]
            if linked:
                cand = linked
        a, d = min(cand, key=lambda x: x[1])
        d_at[j], az_at[j] = d, a
        hist.append((j, a))
    return d_at, az_at


def run_rule2(pred, C: Cfg, nframes: int = 100):
    """clip -> cls -> [episode]。v4.run_rule(pred, "cpa") の v4.2 版。"""
    res = {}
    for clip, frames in pred.items():
        per_cls = {}
        for cls in v4.DIST_CLASSES:
            d_at, az_at = track_series2(frames, cls, nframes, C)
            if not d_at:
                continue
            eps = fires_cpa2(d_at, az_at, nframes, C)
            if eps:
                per_cls[cls] = eps
        res[clip] = per_cls
    return res


def cfg_from_args(argv) -> Cfg:
    """--set key=value ... で Cfg を上書きする（煙試験用の簡易CLI）。"""
    C = Cfg()
    if "--set" in argv:
        for kv in argv[argv.index("--set") + 1:]:
            if "=" not in kv:
                break
            k, v = kv.split("=", 1)
            t = type(getattr(C, k))
            C = replace(C, **{k: (v == "True") if t is bool else t(v)})
    return C


def main() -> int:
    pred = v4.load_pred(Path(sys.argv[1]))
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    C = cfg_from_args(sys.argv)
    res = run_rule2(pred, C)
    n_ep = sum(len(e) for c in res.values() for e in c.values())
    n_strong = sum(1 for c in res.values() for e in c.values()
                   for ep in e if ep[2] == "強")
    txt = f"# 通知 v4.2 {C}\n\n- エピソード {n_ep:,}（うち強 {n_strong:,}）\n"
    (outdir / "notify_v42.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
