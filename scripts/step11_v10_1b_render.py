# -*- coding: utf-8 -*-
"""step11_v10_1b_render.py — v10.1b（backup_beepのdBレンジ修正）。

2026-07-21のFable精査で、v9.3/v10のbackup_beepレンジ「60-92dB@1m」が
別々の2規定（UN R165通常60-75dB@後方7m評価／音声式併設時の合計上限92dB@1m）の
混同だったと判明したことへの対応。本人決定に従い、①実勢ブザー系（85-95dB@1m）
②新基準系（UN R165通常60-75dB@7m）の50:50混合抽選に修正する
（根拠と出典は out/japan_stage_audit_2026-07-21.md 8節）。

v10.1（step11_v10_1_render.py、消防車サイレン）をそのまま継承した上で
V10_1B=True を立てる1点のみが差分——**backup_beep入りクリップにはsiren入りも
含まれるため、fire型の割当を保存したまま再生成するにはV10_1の継承が必須**。
出力先はv10と同一フォルダへの上書きパッチ。対象特定・再生成は
scripts/_run_v10_1b_patch.py が行う（このファイル自体はモジュール設定のみ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v10_1_render as m10_1  # noqa: E402 (V10_1=Trueとv10設定を継承)

m9 = m10_1.m9
m9.V10_1B = True
