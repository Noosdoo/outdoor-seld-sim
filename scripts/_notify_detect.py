# -*- coding: utf-8 -*-
"""知らせの規則「対象音を検出したら知らせる（距離を見ない）」（変更票 #73・Q2・Astra Q2・本人 2026-10-20 23:34「2 おけ」）。

この研究の規則（v4.3: 最接近の予測と距離で 強／中 を分ける）と比べるための、既存アプリの型の規則。**知覚の予測はこの研究のものを固定**し、規則だけ替える。
同じにするもの（v4.3 と共通の部品をそのまま使う・Astra Q2「知らせの強さ・繰り返しを抑える時間・通知の対応付けは事前にそろえる」）:
  - 追跡（track_series2・同じ Cfg43）・確認（_stream_mn・v4.CONFIRM = 2 フレーム連続）・エピソード化（group_episodes・連続フレーム＆方位 ±AZ_MATCH で 1 回だけ鳴る）
  - 帰属（_notify_v43_attrib.attribute）・警告音の規則（v9c）は採点器側で同じ
違うもの: 距離・最接近・到達時間を一切見ず、確認が取れた検出を全部「強」で鳴らす（段を分けない）。
使い方: _score_unified.py --notify-rule detect（既定は v43）
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, fname):
    import sys
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)   # dataclass の解決に sys.modules の登録が要る
    return m


v4 = _load("nv4_det", "step12_notify_v4_ttc.py")
V42 = _load("nv42_det", "step12_notify_v42_bearing.py")
RULE_NAME = "detect"


def fires_detect(d_at, az_at, nframes: int):
    """確認（CONFIRM フレーム連続で検出）が取れたフレームを全部「強」の候補にし、エピソードの先頭で 1 回鳴らす。"""
    hits = V42._stream_mn(d_at, az_at, nframes, lambda j, d: True, v4.CONFIRM, v4.CONFIRM)
    return v4._episodes_with_upgrade(hits, hits)          # 中の列 = 強の列 → 全部 強


def run_rule_detect(pred: dict, C, nframes: int = 100) -> dict:
    """V43.run_rule3 と同じ入出力: {clip: {cls: [(frame, az, tier, d), ...]}}。tier は全部 強。"""
    res = {}
    for clip, frames in pred.items():
        per_cls = {}
        for cls in v4.DIST_CLASSES:
            d_at, az_at = V42.track_series2(frames, cls, nframes, C)
            if not d_at:
                continue
            eps = fires_detect(d_at, az_at, nframes)
            if eps:
                per_cls[cls] = eps
        res[clip] = per_cls
    return res
