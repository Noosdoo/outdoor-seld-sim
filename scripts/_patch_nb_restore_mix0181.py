# -*- coding: utf-8 -*-
"""v11ノートブックのmix0181除外を撤回するパッチ（ゲートv2合格diff-0.44dBによる復帰、
2026-07-31決定。test=1,200本に）。ipynb JSONを直接書き換える。"""
import json
from pathlib import Path

NB = Path(r"C:\Users\satos\research\outdoor_seld_e2e\colab\PSELDNets_outdoor_siren_v11_Colab.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

OLD_HDR = ("- 学習 fold1_room1 4,800 / val fold2_room1 1,200 / test fold3_room1 1,199"
           "（検品FAIL1本除外。**testは最終1回まで触らない**）")
NEW_HDR = ("- 学習 fold1_room1 4,800 / val fold2_room1 1,200 / test fold3_room1 1,200"
           "（**testは最終1回まで触らない**。検品は全数合格=旧FAIL1本はゲートv2再判定で"
           "diff-0.44dB合格、2026-07-31復帰）")

OLD_BLK = """# 検品FAIL（inspection.csv 2026-07-28確定）: 受音ゲート±3.5dBの良性の裾（ベルの
# 打撃×距離の偶然相関、-4.33dBのうち-3.29dBを打撃タイミングで説明済み）。
# 物理異常ではないが v10 mix119 と同方針で除外（fold3=1,199本）
INSPECT_FAIL = ['fold3_room1_mix0181']

if not os.path.exists(f'datasets/{DATASET}/foa'):
    with zipfile.ZipFile(V11_ZIP) as z:
        z.extractall('.')
    print('v11 unzipped')
else:
    print('v11は展開済み')

_, n11 = _dir_digest(DATASET, 'foa')
if n11 == 7200:
    for sub in ('foa', 'metadata', 'masks'):
        dg, cnt = _dir_digest(DATASET, sub)
        assert (cnt, dg) == (7200, V11_DIGEST_FULL[sub]), \\
            f'⚠ {sub}: 展開結果がマニフェスト(7200)と不一致 {cnt} {dg}'
    for stem in INSPECT_FAIL:
        for sub, ext in (('foa', 'flac'), ('metadata', 'csv'), ('masks', 'csv')):
            os.remove(f'datasets/{DATASET}/{sub}/{stem}.{ext}')
    print(f'manifest(7200)照合OK -> INSPECT_FAIL除外: {INSPECT_FAIL}')
for sub in ('foa', 'metadata', 'masks'):
    dg, cnt = _dir_digest(DATASET, sub)
    assert (cnt, dg) == (7199, V11_DIGEST_FINAL[sub]), \\
        f'⚠ {sub}: 最終状態がマニフェスト(7199)と不一致 {cnt} {dg}'
print('v11 manifest digest OK (7,199 x3 subdirs)')"""

NEW_BLK = """# 検品: 7,200本全数合格。旧FAIL1本(fold3_room1_mix0181)は受音検査式の構造的予測誤り
# （ベル打撃×距離の偶然相関）で、改良検査式ゲートv2ではdiff-0.44dBで合格→除外しない
# （評価拡張31本と同一基準に統一、2026-07-31決定）

if not os.path.exists(f'datasets/{DATASET}/foa'):
    with zipfile.ZipFile(V11_ZIP) as z:
        z.extractall('.')
    print('v11 unzipped')
else:
    print('v11は展開済み')

for sub in ('foa', 'metadata', 'masks'):
    dg, cnt = _dir_digest(DATASET, sub)
    assert (cnt, dg) == (7200, V11_DIGEST_FULL[sub]), \\
        f'⚠ {sub}: 展開結果がマニフェスト(7200)と不一致 {cnt} {dg}'
print('v11 manifest digest OK (7,200 x3 subdirs)')"""

n_hit = 0
for c in nb["cells"]:
    s = c["source"]
    for old, new in ((OLD_HDR, NEW_HDR), (OLD_BLK, NEW_BLK),
                     ("'v11 FOA (7199=7200-検品FAIL1)'", "'v11 FOA (7200)'")):
        if old in s:
            c["source"] = s = s.replace(old, new)
            n_hit += 1
assert n_hit == 3, f"置換ヒット数が想定外: {n_hit}"
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("patched OK (3 replacements) ->", NB.name)
