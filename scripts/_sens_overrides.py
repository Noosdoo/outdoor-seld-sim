# -*- coding: utf-8 -*-
"""感度の変種（〔仮〕の値を変えた評価用の小さな束）のための「定数の上書き」（2026-10-21 Fable・本人 13:57／23:39「それでお願いします」）。

目的: 学習し直さず、評価側の小さな変種（200 本ほど・封印の eval v3 とは別）を、合成の設計の定数を 1 つだけ変えて描き、固定したモデルで推論して差を見る。
やり方: 描画の連なり（scripts/step11_*・src/outdoor_seld/*）の module の定数を、描く前に setattr で置き換える（コードは変えない・上書きの記録を束に残す）。
対象の例（研究 md「仮定の値の根拠探し」の無根拠の項目）:
  - scripts.step11_v22_render.BIC_SLOPE_HIGH     自転車の走行音の 12 km/h 超の傾き（+30 log・無根拠）→ 例 20・40
  - outdoor_seld.level_v24.L2M_POINTS_DB         EV の下地 = 車の曲線（EV は低速で静かかもしれない）→ 例 低速 2 点を −5 dB
  - outdoor_seld.avas_v26.AVAS_L2M_DBA           通報音の大きさの幅（46〜70）→ 例 (50, 70)
  - outdoor_seld.scene.*                         地面の反射（1.0 → 0.98 は実測の根拠あり・感度は小さいはず）
  - 踏切の警報音の大きさ・クラクションの押し方 = 定数の名前は scripts/step11_v9_render.py・step11_v23_render.py で確かめてから
使い方（描画の直前に・_sens_render.py が呼ぶ）: SENS_OVERRIDES_JSON=<json> python -c "import _sens_overrides as S; S.apply_from_env(); ..."
JSON の形: {"overrides": {"<module>.<ATTR>": <JSON の値>, ...}, "tag": "...", "note": "..."}。値の型（数・tuple・list・dict）は元と同じ種類に直して入れる。
  python scripts/_sens_overrides.py --selftest
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "scripts"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class OverrideError(Exception):
    pass


def _coerce(old, new):
    """元の値の種類に合わせる（tuple は tuple に・int は int に・float は float に）。dict は辞書のまま。"""
    if isinstance(old, tuple):
        if not isinstance(new, (list, tuple)):
            raise OverrideError(f"tuple の定数には list を渡す（{new!r}）")
        return tuple(float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in new) if all(isinstance(o, float) for o in old) else tuple(new)
    if isinstance(old, bool):
        if not isinstance(new, bool):
            raise OverrideError("bool の定数には true/false を渡す")
        return new
    if isinstance(old, int) and not isinstance(old, bool):
        if isinstance(new, bool) or not isinstance(new, (int, float)):
            raise OverrideError(f"int の定数には数を渡す（{new!r}）")
        return int(new)
    if isinstance(old, float):
        if isinstance(new, bool) or not isinstance(new, (int, float)):
            raise OverrideError(f"float の定数には数を渡す（{new!r}）")
        return float(new)
    if isinstance(old, dict):
        if not isinstance(new, dict):
            raise OverrideError("dict の定数には dict を渡す")
        return dict(new)
    if isinstance(old, list):
        if not isinstance(new, list):
            raise OverrideError("list の定数には list を渡す")
        return list(new)
    raise OverrideError(f"上書きできない種類の定数: {type(old).__name__}")


def apply(overrides: dict) -> dict:
    """{"module.ATTR": value} を当てる。module は import 可能な名前（scripts/ と src/ を path に足してある）。
    戻り値 = 記録 {"module.ATTR": {"old": ..., "new": ...}}（束に残す）。無い module／属性・種類違いは止める。"""
    rec = {}
    for key, new in overrides.items():
        if "." not in key:
            raise OverrideError(f"上書きの鍵は module.ATTR の形: {key}")
        modname, attr = key.rsplit(".", 1)
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            raise OverrideError(f"module が読めない: {modname}（{e}）") from e
        if not hasattr(mod, attr):
            raise OverrideError(f"{modname} に {attr} が無い")
        old = getattr(mod, attr)
        if not attr.isupper():
            raise OverrideError(f"上書きできるのは大文字の定数だけ: {key}")
        val = _coerce(old, new)
        setattr(mod, attr, val)
        rec[key] = {"old": old, "new": val}
    return rec


def apply_from_env(var: str = "SENS_OVERRIDES_JSON") -> dict:
    p = os.environ.get(var, "")
    if not p:
        return {}
    J = json.loads(Path(p).read_text(encoding="utf-8"))
    rec = apply(J.get("overrides") or {})
    print(f"[sens] 上書き {len(rec)} 件（{J.get('tag', '')}）: " + "・".join(f"{k}: {v['old']!r} → {v['new']!r}" for k, v in rec.items()), flush=True)
    return rec


def _selftest() -> int:
    ok = []
    from outdoor_seld import avas_v26 as A  # noqa: E402
    from outdoor_seld import level_v24 as L  # noqa: E402
    old_a = A.AVAS_L2M_DBA; old_l = L.L2M_POINTS_DB
    try:
        rec = apply({"outdoor_seld.avas_v26.AVAS_L2M_DBA": [50, 70], "outdoor_seld.level_v24.L2M_POINTS_DB": [45.0, 49.0, 54.0, 58.25, 62.25, 69.0]})
        ok.append(("tuple の定数を list で上書き → tuple のまま・値が変わる・記録に old/new", A.AVAS_L2M_DBA == (50.0, 70.0) and isinstance(A.AVAS_L2M_DBA, tuple)
                   and L.L2M_POINTS_DB[0] == 45.0 and rec["outdoor_seld.avas_v26.AVAS_L2M_DBA"]["old"] == old_a))
        try:
            apply({"outdoor_seld.avas_v26.NO_SUCH": 1}); ok.append(("無い属性 → 止まる", False))
        except OverrideError:
            ok.append(("無い属性 → 止まる", True))
        try:
            apply({"outdoor_seld.avas_v26.AVAS_L2M_DBA": "x"}); ok.append(("種類違い（tuple に文字列） → 止まる", False))
        except OverrideError:
            ok.append(("種類違い（tuple に文字列） → 止まる", True))
        try:
            apply({"outdoor_seld.avas_v26.avas_params": 1}); ok.append(("小文字（関数）は上書きできない → 止まる", False))
        except OverrideError:
            ok.append(("小文字（関数）は上書きできない → 止まる", True))
        import step11_v22_render as V22  # noqa: E402
        old_s = V22.BIC_SLOPE_HIGH
        rec2 = apply({"step11_v22_render.BIC_SLOPE_HIGH": 20})
        ok.append(("scripts 側の定数（自転車の傾き）も上書きできる・float のまま", V22.BIC_SLOPE_HIGH == 20.0 and isinstance(V22.BIC_SLOPE_HIGH, float) and rec2["step11_v22_render.BIC_SLOPE_HIGH"]["old"] == old_s))
        V22.BIC_SLOPE_HIGH = old_s
    finally:
        A.AVAS_L2M_DBA = old_a; L.L2M_POINTS_DB = old_l
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 2)
