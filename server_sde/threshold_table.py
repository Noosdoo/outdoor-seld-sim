# -*- coding: utf-8 -*-
"""検出のしきい値の表（本体 §3-5・§9-1 の 5・R30-05・2026-10-26）を読む・確かめる。表は `scripts/v30_select_thresholds.py` が作る `md/design/thresholds_v30.json`。

読む側: 因果推論 `_causal_infer_v17.py`（`CAUSAL_THRESHOLDS_JSON`・`CAUSAL_THRESHOLDS_MODE`）と契約 `_conf_exec_contract.py --thresholds-json`。
確かめること（違えば ThresholdTableError で止める）:
  - 版の頭が thresholds_v30/1・クラスが 0〜C−1 の全部・入り（enter）と保持（stay）がクラス数ぶんの有限の数・入りは候補（0.1〜0.7）のどれか
  - 保持 = round(入り × 3/7, 4)（B・D と同じ比・`_baseline_select_gain.py` の stay と同じ丸め）
  - ckpt_sha を渡したら、表を作った推論の ckpt（inputs.ckpt_sha256）と同じ（別の ckpt の表を掛けない）
mode: "hyst" = 表の入り／保持・"nohyst" = 入りは表・保持 = 入り（アブレーション nohyst = 主と同じ入りを共有・§9-1 の 7）。
対象クラス（Fable 16:5x・クラス別は主・B・D の全員）: 主と B は 0〜8 の全部。D は対応表の共通クラスだけ（表の classes = その一覧・enter／stay はその順）。
  load(classes=…) で読む側の対象クラスを渡すと、表の classes がそれと同じかを確かめ、対象でないクラスの入り／保持は +inf（決して「いる」にならない＝D は対象外の
  クラスを出さない）にした長さ C の配列を返す。classes を渡さなければ 0〜C−1 の全部を求める（主・B）。
B・D の norms の保存と一覧（save_norms・write_norms_manifest）もここ（v17 と同じ型・同じ一覧の形）。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION_PREFIX = "thresholds_v30/1"
CANDIDATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MODES = ("hyst", "nohyst")


class ThresholdTableError(Exception):
    pass


def stay_of(th: float) -> float:
    """保持 = 入り × 3/7（丸め 4 桁・0.7 → 0.3）。"""
    return round(float(th) * 3.0 / 7.0, 4)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path, nb_classes: int = 9, mode: str = "hyst", ckpt_sha: str | None = None, classes: list[int] | None = None) -> dict:
    """表を読んで確かめる。返り値 = {path, sha256, version, label, mode, classes, enter (C,) float64, stay (C,) float64, ckpt_sha256}（対象外のクラスは +inf）。"""
    p = Path(path)
    if mode not in MODES:
        raise ThresholdTableError(f"mode は {MODES} のどれか: {mode}")
    if not p.is_file():
        raise ThresholdTableError(f"しきい値の表が無い: {p}")
    sha = _sha256(p)
    try:
        J = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ThresholdTableError(f"しきい値の表が JSON でない: {e}")
    if not str(J.get("version", "")).startswith(VERSION_PREFIX):
        raise ThresholdTableError(f"しきい値の表の版が違う: {J.get('version')}（{VERSION_PREFIX}）")
    want = list(range(nb_classes)) if classes is None else sorted(int(c) for c in classes)
    if J.get("classes") != want:
        raise ThresholdTableError(f"しきい値の表のクラス {J.get('classes')} が読む側の対象クラス {want} と違う")
    if not want or any(not (0 <= c < nb_classes) for c in want):
        raise ThresholdTableError(f"対象クラスが 0〜{nb_classes - 1} の中の空でない一覧でない: {want}")
    en, st = J.get("enter"), J.get("stay")
    if not (isinstance(en, list) and isinstance(st, list) and len(en) == len(want) and len(st) == len(want)):
        raise ThresholdTableError("enter／stay が対象クラスぶんの配列でない")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in en + st):
        raise ThresholdTableError("enter／stay に数でない値か非有限がある")
    if any(min(abs(v - c) for c in CANDIDATES) > 1e-9 for v in en):
        raise ThresholdTableError(f"入りが候補 {CANDIDATES} にない: {en}")
    if any(abs(s - stay_of(e)) > 1e-9 for e, s in zip(en, st)):
        raise ThresholdTableError("保持が round(入り × 3/7, 4) でない")
    ck = (J.get("inputs") or {}).get("ckpt_sha256")
    if ckpt_sha is not None and ck != ckpt_sha:
        raise ThresholdTableError(f"しきい値の表を作った ckpt（{str(ck)[:12]}）と今の ckpt（{ckpt_sha[:12]}）が違う（別の ckpt の表を掛けない）")
    enter = np.full(nb_classes, np.inf)
    stay = np.full(nb_classes, np.inf)
    enter[want] = np.asarray(en, dtype=np.float64)
    stay[want] = enter[want] if mode == "nohyst" else np.asarray(st, dtype=np.float64)
    return {"path": str(p), "sha256": sha, "version": J["version"], "label": J.get("label"), "mode": mode, "classes": want,
            "enter": enter, "stay": stay, "ckpt_sha256": ck}


def record(t: dict) -> dict:
    """run_meta・契約の記録に入れる形（配列は list・対象外のクラスは None）。"""
    def _l(a):
        return [float(v) if math.isfinite(v) else None for v in a]
    return {"path": t["path"], "sha256": t["sha256"], "version": t["version"], "label": t["label"], "mode": t["mode"], "classes": t["classes"],
            "enter": _l(t["enter"]), "stay": _l(t["stay"]), "ckpt_sha256": t["ckpt_sha256"]}


def save_norms(norms_dir: Path, clip_id: str, nrm: np.ndarray) -> None:
    """norms/<clip>.npy を書く（.part → 置き換え・書きかけを残さない）。"""
    import os
    norms_dir = Path(norms_dir)
    tmp = norms_dir / f"{clip_id}.npy.part"
    with open(tmp, "wb") as h:
        np.save(h, np.asarray(nrm, dtype=np.float32))
    os.replace(tmp, norms_dir / f"{clip_id}.npy")


def write_norms_manifest(run_dir: Path, clip_ids: list[str], ckpt_sha: str, nb_classes: int, schema: str, extra: dict | None = None) -> dict:
    """全部の本の norms がそろっていることを確かめて norms_manifest.json を書く（欠けがあれば RuntimeError）。v17 と同じ形。"""
    import json
    import os
    run_dir = Path(run_dir)
    nd = run_dir / "norms"
    miss = [c for c in clip_ids if not (nd / f"{c}.npy").is_file()]
    if miss:
        raise RuntimeError(f"norms が欠けている: {len(miss)} 本（例 {miss[:2]}）")
    m = {"schema": schema, "ckpt_sha256": ckpt_sha, "n_clips": len(clip_ids), "nb_classes": nb_classes, **(extra or {}),
         "files": {c: _sha256(nd / f"{c}.npy") for c in clip_ids}}
    tmp = run_dir / "norms_manifest.json.part"
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, run_dir / "norms_manifest.json")
    return m
