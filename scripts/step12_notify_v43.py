# -*- coding: utf-8 -*-
"""通知 v4.3 — v4.2 の上に「強=衝突コースだけ」の部品を足す（2026-09-03）。

宣言= md/design/通知v4.3_検討の事前宣言_2026-09-03.md 追記1。v4.2 本体（step12_notify_v42_bearing.py）は
不変更。Cfg43 の追加フィールドが既定値なら v4.2 採用構成と**発火が完全一致**する（--verify で確認）。

追加部品:
  strong_adot_max : cpa経路の強は |adot|（方位変化率・窓=brg_win）がこれ以下のときだけ。None=無効
  rc_brg_win      : routeC の |adot| 判定に使う方位窓（v4.2 は brg_win と同じ5）
（dn / adot_th は v4.2 Cfg の既存フィールド）

使い方:
  python scripts/step12_notify_v43.py --verify <pred.csv>   # 既定 Cfg43 = v4.2 採用構成 と一致確認
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v4 = _load("nv4", "step12_notify_v4_ttc.py")
V42 = _load("nv42", "step12_notify_v42_bearing.py")

ADOPTED42 = dict(route_c=True, adot_th=0.10, dn=15.0, link_pred=True, cpa_strong=1.3, cpa_mid=1.6)


@dataclass(frozen=True)
class Cfg43(V42.Cfg):
    strong_adot_max: float | None = None   # [rad/s] None=無効（v4.2と同一）
    rc_brg_win: int = 5                    # routeC 専用の方位窓（5=v4.2と同一）


def cfg43(**kw) -> Cfg43:
    return Cfg43(**{**ADOPTED42, **kw})


def fires_cpa3(d_at, az_at, nframes, C: Cfg43):
    """v4.2 fires_cpa2 ＋ 強の方位ゲート ＋ routeC 専用窓。既定値では fires_cpa2 と同一。"""
    pre = {}
    for j in range(nframes):
        d = d_at.get(j)
        if d is None:
            continue
        v = v4.closing_speed(d_at, j, win=C.vel_win)
        ddot = None if v is None else -v
        adot = v4.azimuth_rate(az_at, j, win=C.brg_win)
        adot_rc = (adot if C.rc_brg_win == C.brg_win
                   else v4.azimuth_rate(az_at, j, win=C.rc_brg_win))
        dc, tc = v4.cpa_of(d, ddot, adot)
        rc = (C.route_c and adot_rc is not None and abs(adot_rc) <= C.adot_th
              and ddot is not None and ddot < -C.v_close and d <= C.dn)
        bearing_ok = (C.strong_adot_max is None or
                      (adot is not None and abs(adot) <= C.strong_adot_max))
        pre[j] = (dc, tc, rc, bearing_ok)

    def _cond(j, dc_th, tc_th, strong):
        dc, tc, rc, bok = pre.get(j, (None, None, False, True))
        cpa = dc is not None and dc <= dc_th and tc <= tc_th
        if strong:
            cpa = cpa and bok
        return cpa or rc

    def _stream(dc_th, tc_th, d_th, strong):
        a = V42._stream_mn(d_at, az_at, nframes,
                           lambda j, d: _cond(j, dc_th, tc_th, strong),
                           C.confirm_m, C.confirm_n)
        b = V42._stream_mn(d_at, az_at, nframes, lambda j, d: d <= d_th,
                           v4.CONFIRM, v4.CONFIRM)
        return sorted(set(a) | set(b))

    return v4._episodes_with_upgrade(
        _stream(C.cpa_mid, v4.TTC_CAUTION, v4.SUPP, False),
        _stream(C.cpa_strong, v4.TTC_WARN, v4.T3, True))


def run_rule3(pred, C: Cfg43, nframes: int = 100):
    res = {}
    for clip, frames in pred.items():
        per_cls = {}
        for cls in v4.DIST_CLASSES:
            d_at, az_at = V42.track_series2(frames, cls, nframes, C)
            if not d_at:
                continue
            eps = fires_cpa3(d_at, az_at, nframes, C)
            if eps:
                per_cls[cls] = eps
        res[clip] = per_cls
    return res


def label43(C: Cfg43) -> str:
    sa = "∞" if C.strong_adot_max is None else f"{C.strong_adot_max:.2f}"
    return f"sa{sa}+dn{C.dn:.0f}+ath{C.adot_th:.2f}+rcw{C.rc_brg_win}"


def apply_mode(mode: str) -> Cfg43:
    """システムの 2 設定（本人決定 2026-09-16・変更票 #32）。定義は scripts/_notify_modes.py の 1 か所に置き、規則そのものは変えない。
    控えめ = 現行の値（既定と同一）、厳格 = 距離の閾値 4 つを ×1.2。呼ぶと v4.T3 / v4.SUPP を書き換え、cpa の閾値を入れた Cfg43 を返す。
    実機・デモ・採点器はすべてこれを通して閾値を決める（採点器は --mode）。"""
    import json as _json
    _spec = importlib.util.spec_from_file_location("notify_modes", ROOT / "scripts" / "_notify_modes.py")
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
    m = _m.MODES[mode]
    v4.T3, v4.SUPP = m["T3"], m["SUPP"]
    base = _json.loads((ROOT / "out/notify_v43_sweep/winner.json").read_text(encoding="utf-8"))
    return Cfg43(**{**base, "cpa_strong": m["cpa_strong"], "cpa_mid": m["cpa_mid"]})


if __name__ == "__main__":
    if "--verify" in sys.argv:
        pred = v4.load_pred(Path(sys.argv[sys.argv.index("--verify") + 1]))
        a = V42.run_rule2(pred, V42.Cfg(**ADOPTED42))
        b = run_rule3(pred, cfg43())
        print("v4.2採用構成 と Cfg43既定 の発火:", "完全一致 ✅" if a == b else "不一致 ❌")
        sys.exit(0 if a == b else 1)
