# -*- coding: utf-8 -*-
"""検出のヒステリシス（入りの閾値と保持の閾値の 2 つ）— 推論の後処理（2026-09-16・変更票 #34・本人「①採用」）。

規則（scripts/_hysteresis_grid2.py の decode_general・型 amp と同じ。開発 val で 900 組以上を比べて 入り 0.70・保持 0.30 を選んだ）:
  - 矢印の長さ（ACCDOA ベクトルの norm）が **入り（enter）** を超えたら「いる」
  - そうでなくても、長さが **保持（stay）** を超え、かつ **前のフレームで「いる」だった同じ種類の相手**が方位 ±assoc° 以内にいれば「いる」を続ける
    （前フレームの相手 = 長さ > pool_min の候補のうち方位が最も近いもの。それが「いる」でなければ続けない）
  - 入り == 保持 なら 1 閾値（本番の従来の判定）と同じ
入力は numpy（sed/doa を .numpy() したもの）。torch には依存しない。長さは本番と同じ float32 の式で計算し、比較は float64 で行う
（掃引の道具と同じ。float32 の閾値 0.3 との等号の差を避ける）。

使い方（_causal_infer_v17.py）:
  sed_np = hysteresis_patch.apply(doa_np, nb_classes, enter=SED_TH, stay=SED_STAY, assoc=SED_ASSOC)
クラス別の値（本体 §3-5・2026-10-26）: enter／stay に長さ C の配列を渡すと、クラスごとの入り／保持で同じ規則を掛ける（規則はクラスごとに独立なので、
列 c の出力 = 定数 enter[c]／stay[c] で掛けた列 c と同じ）。定数を渡したときは前と同じ道（1 行も変えていない）。norm は `norms(doa, C)`。
"""
from __future__ import annotations

import numpy as np

POOL_MIN = 0.05        # 前フレームの候補に入れる最小の長さ（掃引の材料 scores の保存下限 CAUSAL_SCORE_MIN と同じ）
ASSOC_DEG = 20.0       # 前フレームとの対応幅（度）


def dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def norms(doa: np.ndarray, nb_classes: int) -> np.ndarray:
    """(3, T, 4C) → (3, T, C) float32 の矢印の長さ（本番 get_multi_accdoa_labels_sde と同じ float32 の式・ヒステリシスの前）。"""
    doa = np.asarray(doa)
    C = nb_classes
    assert doa.ndim == 3 and doa.shape[2] == 4 * C, f"doa の形が違う: {doa.shape}"
    x = doa[:, :, 0:C].astype(np.float32); y = doa[:, :, C:2 * C].astype(np.float32); z = doa[:, :, 2 * C:3 * C].astype(np.float32)
    return np.sqrt(x * x + y * y + z * z).astype(np.float32)


def has_hysteresis(enter, stay) -> bool:
    """保持 < 入り のクラスが 1 つでもあれば True（定数なら stay < enter と同じ）。"""
    return bool(np.any(np.asarray(stay, dtype=np.float64) < np.asarray(enter, dtype=np.float64)))


def _per_class(v, C: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        return np.full(C, float(a))
    assert a.shape == (C,) and not np.any(np.isnan(a)) and not np.any(a == -np.inf), f"{name} はクラス数 {C} の配列（NaN・−inf なし・対象外のクラスは +inf）: {a.shape}"
    return a


def _apply_per_class(doa: np.ndarray, C: int, enter, stay, assoc: float, pool_min: float) -> np.ndarray:
    """apply のクラス別の値の道（規則は apply と同じ・値だけクラスごと）。"""
    en, st = _per_class(enter, C, "enter"), _per_class(stay, C, "stay")
    x = doa[:, :, 0:C].astype(np.float32); y = doa[:, :, C:2 * C].astype(np.float32); z = doa[:, :, 2 * C:3 * C].astype(np.float32)
    nrm = np.sqrt(x * x + y * y + z * z).astype(np.float64)
    az = np.degrees(np.arctan2(y.astype(np.float64), x.astype(np.float64)))
    ntrk, T, _ = nrm.shape
    sed = nrm > en[None, None, :]
    if not np.any(st < en):
        return sed
    pm = np.minimum(pool_min, st)            # RI73-05 と同じ下限（クラスごと）
    out = np.zeros_like(sed)
    prev = [[] for _ in range(C)]
    for t in range(T):
        cur = [[] for _ in range(C)]
        for c in range(C):
            pool = prev[c]
            e_c, s_c, p_c = en[c], st[c], pm[c]
            for k in range(ntrk):
                n = nrm[k, t, c]
                if n <= p_c:
                    continue
                a = az[k, t, c]
                on = bool(n > e_c)
                if not on and n > s_c and pool:
                    best = None; bd = assoc + 1.0
                    for pa, pon in pool:
                        d = dang(a, pa)
                        if d <= assoc and d < bd:
                            best, bd = pon, d
                    if best:
                        on = True
                out[k, t, c] = on
                cur[c].append((a, on))
        prev = cur
    return out


def apply(doa: np.ndarray, nb_classes: int, enter: float, stay: float, assoc: float = ASSOC_DEG, pool_min: float = POOL_MIN) -> np.ndarray:
    """doa: (3, T, 4*C) float32（x, y, z, dist の 4 ブロック）。戻り値 sed: (3, T, C) bool。
    stay >= enter なら 1 閾値（長さ > enter）。enter／stay が長さ C の配列ならクラス別の値（本体 §3-5）。"""
    doa = np.asarray(doa)
    if np.ndim(enter) or np.ndim(stay):
        assert doa.ndim == 3 and doa.shape[2] == 4 * nb_classes, f"doa の形が違う: {doa.shape}"
        return _apply_per_class(doa, nb_classes, enter, stay, assoc, pool_min)
    assert doa.ndim == 3 and doa.shape[2] == 4 * nb_classes, f"doa の形が違う: {doa.shape}"
    C = nb_classes
    x = doa[:, :, 0:C].astype(np.float32); y = doa[:, :, C:2 * C].astype(np.float32); z = doa[:, :, 2 * C:3 * C].astype(np.float32)
    nrm = np.sqrt(x * x + y * y + z * z).astype(np.float64)          # (3, T, C)・本番 get_multi_accdoa_labels_sde と同じ式（float32）→ 比較は float64
    az = np.degrees(np.arctan2(y.astype(np.float64), x.astype(np.float64)))
    ntrk, T, _ = nrm.shape
    sed = nrm > enter
    if stay >= enter:
        return sed
    # Astra RI73-05（2026-10-21）: 候補の下限 pool_min が保持 stay より大きいと、保持以下で先に候補から落ちて「継続」できない（D の 0.1／0.0429 で起きる）。
    #   下限は min(pool_min, stay) にする。提案法の 0.7／0.3 では pool_min 0.05 < 0.3 なので出力は変わらない。
    pool_min = min(pool_min, stay)
    out = np.zeros_like(sed)
    prev = [[] for _ in range(C)]            # クラスごとの前フレームの候補 [(az, active)]
    for t in range(T):
        cur = [[] for _ in range(C)]
        for c in range(C):
            pool = prev[c]
            for k in range(ntrk):
                n = nrm[k, t, c]
                if n <= pool_min:
                    continue                  # 候補にならない（掃引の材料に無い行と同じ扱い）
                a = az[k, t, c]
                on = bool(n > enter)
                if not on and n > stay and pool:
                    best = None; bd = assoc + 1.0
                    for pa, pon in pool:
                        d = dang(a, pa)
                        if d <= assoc and d < bd:
                            best, bd = pon, d
                    if best:
                        on = True
                out[k, t, c] = on
                cur[c].append((a, on))
        prev = cur
    return out


def apply_with_start(doa: np.ndarray, sed: np.ndarray, nb_classes: int, enter: float, stay: float, assoc: float = ASSOC_DEG,
                     start: int = 0, pool_min: float = POOL_MIN) -> np.ndarray:
    """因果推論の後処理（2026-10-26 本体 §9-4 の 1・変更票 #75 決め直し 3）: start（HYST_START_FRAME）より前の枠は出力しない（全部 False）・
    ヒステリシスの状態も start から始める。start = 0 は旧と同じ（stay < enter ならヒステリシス・そうでなければ入りだけの sed をそのまま返す）。
    doa: (3, T, 4C)・sed: (3, T, C) bool（入りの閾値の判定）。"""
    if start <= 0:
        return apply(doa, nb_classes, enter, stay, assoc, pool_min) if has_hysteresis(enter, stay) else sed
    out = np.zeros_like(sed)
    out[:, start:] = (apply(doa[:, start:], nb_classes, enter, stay, assoc, pool_min) if has_hysteresis(enter, stay) else sed[:, start:])
    return out


def describe(enter: float, stay: float, assoc: float = ASSOC_DEG) -> str:
    if np.ndim(enter) or np.ndim(stay):
        e, s = np.atleast_1d(enter), np.atleast_1d(stay)
        return f"検出 クラス別 入り {'/'.join(f'{v:.2f}' for v in e)}／保持 {'/'.join(f'{v:.4g}' for v in s)}（対応幅 ±{assoc:.0f}°）" + ("・ヒステリシス" if has_hysteresis(enter, stay) else "・1 閾値")
    return f"検出 入り {enter:.2f}／保持 {stay:.2f}（対応幅 ±{assoc:.0f}°）" + ("・1 閾値" if stay >= enter else "・ヒステリシス")
