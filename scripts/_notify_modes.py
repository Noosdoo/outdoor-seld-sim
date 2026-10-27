# -*- coding: utf-8 -*-
"""通知層・検出層の **2 設定**（本人決定 2026-09-16 19:25・教員「厳しい閾値と楽観的な閾値の 2 つを設けて差を見る」を**装置の設定**として採用）。

  控えめ（ひかえめ）= 無駄鳴りを抑える側。通知 ×1.0（強 1.5 m・中 3.2 m・予測最接近 1.3/1.6 m）
  【2026-09-23 撤回・変更票 #37】2 設定（控えめ／厳格）は本人決定で撤回。本番は 1 設定（控えめ = v4.3 の既定・×1.0）だけ。
  この定義は掃引の道具（--mode）のために残す。「厳格」を本番の採点・報告に使わない。
  厳格（げんかく）  = 鳴らしやすい側。通知 ×1.2（強 1.8 m・中 3.84 m・予測最接近 1.56/1.92 m）。倍率は仮（×1.1 案あり・本人未決 2026-09-16 23:51）

検出層は**両設定とも同じ**: 入り 0.70／保持 0.30 のヒステリシス（変更票 #34・2026-09-16 本人「①採用」。0.01 刻みの探索
`md/results/検出層_2閾値の細密探索_2026-09-16.md` で、検出の閾値は精度を動かさず、保持はちらつきだけ減らすと分かったため。
2 設定の違いは通知層の倍率だけに置く = `md/results/2設定の閾値の探索_2026-09-16.md` §3）。
推論側の環境変数 = CAUSAL_SED_TH（入り）・CAUSAL_SED_STAY（保持）・CAUSAL_SED_ASSOC（対応幅 20°）。既定は 0.5／0.5 = 従来と同一出力。

装置は利用者がどちらかを選んで使う。評価（開発 val・確定評価）は**両方の設定を主表として並べ、差も出す**。
ckpt の選択（設計凍結 §4）は「控えめ」で行う（選択規則は不変）。
使い方: from _notify_modes import MODES, apply_mode / 掃引の道具には --scales 1.0,1.2 を渡せば同じ値になる。
"""
from __future__ import annotations

from dataclasses import replace

DET = {"det_th": 0.7, "det_stay": 0.3, "det_assoc": 20.0}   # 検出層（両設定で共通・変更票 #34）

MODES = {
    "控えめ": {**DET, "notify_scale": 1.0, "T3": 1.5, "SUPP": 3.2, "cpa_strong": 1.3, "cpa_mid": 1.6,
              "説明": "通知は現行の閾値。無駄鳴りを抑える側（適合率寄り）"},
    "厳格":   {**DET, "notify_scale": 1.2, "T3": 1.8, "SUPP": 3.84, "cpa_strong": 1.56, "cpa_mid": 1.92,
              "説明": "鳴らしやすい側。倍率 ×1.2 は仮（×1.1 案 = 強 1.65 m・中 3.52 m も残す・本人未決）"},
}
DEFAULT_MODE = "控えめ"


def apply_mode(S, mode: str):
    """統一採点器モジュール S（_score_unified）に設定を当てる。戻り値は元に戻すための関数。
    通知の距離閾値 4 つを一斉に変える（S.C43 と S.V43.v4 の T3/SUPP）。検出の閾値は推論側（CAUSAL_SED_TH／CAUSAL_SED_STAY）で掛ける。"""
    m = MODES[mode]
    v4r = S.V43.v4
    keep = (v4r.T3, v4r.SUPP, S.C43)
    v4r.T3, v4r.SUPP = m["T3"], m["SUPP"]
    S.C43 = replace(S.C43, cpa_strong=m["cpa_strong"], cpa_mid=m["cpa_mid"])

    def restore():
        v4r.T3, v4r.SUPP, S.C43 = keep
    return restore


def describe(mode: str) -> str:
    m = MODES[mode]
    return (f"{mode}: 検出 入り {m['det_th']}／保持 {m['det_stay']} / 通知 強 {m['T3']:.2f} m・中 {m['SUPP']:.2f} m・予測最接近 {m['cpa_strong']:.2f}/{m['cpa_mid']:.2f} m"
            f"（×{m['notify_scale']:.1f}）— {m['説明']}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for k in MODES:
        print(describe(k))
