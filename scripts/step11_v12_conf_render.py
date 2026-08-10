# -*- coding: utf-8 -*-
"""v12確定評価セット(conf)のレンダ環境: DSをconf側に切替えてm12生成器をそのまま使う。

import時に m9.DS / m12関連の出力先を out/dataset_outdoor_siren_v12_conf に向ける。
生成器・サンプラ・物理は本番(step11_v12_render)と完全同一（事前登録の凍結対象）。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step11_v12_render as m12  # noqa: E402

m9 = m12.m9
DS = ROOT / "out" / "dataset_outdoor_siren_v12_conf"
m9.DS = DS
m9.WORK = DS / "work"   # scene.jsonはm9.WORK定数経由（job421の失敗原因=これの欠落）
for sub in ("foa", "metadata", "masks", "work"):
    (DS / sub).mkdir(parents=True, exist_ok=True)


def load_plan_v12conf():
    p = DS / "plan" / "assignment_v12conf.csv"
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["seed"] = int(r["seed"])
        if r.get("n_car"):
            r["n_car"] = int(r["n_car"])
        if r.get("n_warnings"):
            r["n_warnings"] = int(r["n_warnings"])
    return rows
