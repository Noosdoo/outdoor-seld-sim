# === v10: 空ラベルクリップ＋検品FAILクリップの自動検出・除外 ===
# （v9.1 mix035手動除外の一般化。必ず前処理(preproc)セルの直前に実行）
# v10はtrain2400本のためmix035のような「車が最後まで暗騒音に埋もれ可聴フレームゼロ」の
# 物理的に正しい負例が複数本発生する（ローカル実測27本）。ラベルCSVが0バイトだと
# PSELDNetsの前処理がpd.read_csvでクラッシュするため、該当クリップを学習/検証/テストの
# 各splitから機械的に除外する（ダミー行を挿入する対策は使わない——それは
# 損失計算に使われる学習ラベルに「frame0にsirenがある」という偽のイベントを
# 注入してしまうため、評価専用データ(scn2等)でのみ許容される手法）。
# あわせて、ローカル検品(inspect_all)でFAILした1本（fold2_room1_mix119、受音レベルの
# ±3.5dB照合ゲート不合格）も除外する。

import os
import glob
import shutil

# 検品FAIL（ローカルの out/dataset_outdoor_siren_v10/inspection.csv で確認済み）
INSPECT_FAIL = ['fold2_room1_mix119']

removed_by_split = {}
for room in ("fold1_room1", "fold2_room1", "fold3_room1"):
    removed = []
    for f in sorted(glob.glob(f'datasets/{DATASET}/metadata/{room}_*.csv')):
        stem = os.path.basename(f)[:-4]
        if os.path.getsize(f) > 0 and stem not in INSPECT_FAIL:
            continue
        flac = f'datasets/{DATASET}/foa/{stem}.flac'
        mask = f'datasets/{DATASET}/masks/{stem}.csv'
        os.remove(f)
        if os.path.exists(flac):
            os.remove(flac)
        if os.path.exists(mask):
            os.remove(mask)
        removed.append(stem)
    removed_by_split[room] = removed
    print(f'{room}: removed {len(removed)} clips', removed if removed else '')

total = sum(len(v) for v in removed_by_split.values())
print('---')
print(f'total removed: {total}')
shutil.rmtree('_hdf5', ignore_errors=True)
print('cleaned _hdf5 -> 次に前処理セル(preproc)を実行してください')
