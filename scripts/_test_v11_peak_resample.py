# -*- coding: utf-8 -*-
"""ピーク棄却再抽選の検証: ①salt=0クリップのビット不変（md5） ②mix3642がsaltで解決。

注意: 本テストは対象2クリップ（mix3601/mix3642）を実際に再生成する。生成は決定論なので
既存成果物と同一ビットに収束する＝上書きは無害（①はまさにそれをmd5で証明する）。
実行: dynamic-sound/.venv の python。パスは本ファイル位置から解決（絶対パスなし）。
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step11_v11_render as m11  # noqa: E402

m9 = m11.m9
rows = m9.load_plan("core")

# ① 既存クリップ（p2が正常生成済みのmix3601）を再生成してmd5一致を確認
p = m9.DS / "foa" / "fold1_room1_mix3601.flac"
before = hashlib.md5(p.read_bytes()).hexdigest()
m9.generate_clip(rows[3600])
after = hashlib.md5(p.read_bytes()).hexdigest()
print("salt0不変性(mix3601):", "OK" if before == after else f"NG {before}->{after}")
assert before == after

# ② クラッシュしたmix3642: saltラダーで解決するか
m9.generate_clip(rows[3641])
s = json.loads((m9.WORK / "fold1_room1_mix3642" / "scene.json").read_text())
print("mix3642: salt =", s.get("peak_resample_salt", 0),
      "/ peak =", s["stats"]["peak"])
print("構成:", [(x["class"], x.get("params", {}).get("siren_type", ""),
                x.get("min_dist_m")) for x in s["sources"]])
