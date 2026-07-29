#!/usr/bin/env python
"""v11+v10評価データを ~/PSELDNets_data の zip から展開・検証する。

v11 Colab ノートブック セル6（マニフェスト照合・検品FAIL除外・v10剪定）の移植。
実行場所: PSELDNets リポジトリ直下（cwd）で
    .venv/bin/python ~/research/outdoor_seld_e2e/server/prepare_data.py
冪等（再実行しても壊れない）。
"""
import glob
import hashlib
import os
import shutil
import zipfile

DATA = os.path.expanduser('~/PSELDNets_data')
DATASET = 'outdoor_siren_v11'
EVAL_DS = 'outdoor_siren_v10'
V11_ZIP = f'{DATA}/dataset_outdoor_siren_v11.zip'
V10_ZIP = f'{DATA}/dataset_outdoor_siren_v10.zip'
V10_ADD = f'{DATA}/dataset_outdoor_siren_v10_2_add.zip'

# マニフェストダイジェスト（正= out/dataset_outdoor_siren_v11/manifest_v11.csv、
# full=7200本 / final=検品FAIL除外後7199本。Colabノートブックと同一値）
V11_DIGEST_FULL = {'foa': 'a9030a52e4eba2c318b6e05389872cc3',
                   'metadata': '7610c36b3c8bee697ff4485c9c9cc997',
                   'masks': '90042e05428a5383cba3fded51d469ef'}
V11_DIGEST_FINAL = {'foa': '19d017feb2810468f3aee4be4cdee7aa',
                    'metadata': '347a54db4b0bc49485e496f5cb699d52',
                    'masks': '6c3cb53722a56d5abc7815710dd41185'}
# 検品FAIL（inspection.csv 2026-07-28確定、v10 mix119 と同方針で除外）
INSPECT_FAIL = ['fold3_room1_mix0181']

CKPT_SHA256 = '813083ac938c5974a6f36ceca29ea66c0382091db5df1d6d47ece9572d5ac71b'


def _dir_digest(ds, sub):
    entries = sorted((os.path.basename(p), os.path.getsize(p))
                     for p in glob.glob(f'datasets/{ds}/{sub}/*'))
    dg = hashlib.md5('\n'.join(f'{n},{s}' for n, s in entries).encode()).hexdigest()
    return dg, len(entries)


assert os.path.isdir('src'), '⚠ PSELDNets リポジトリ直下で実行してください'
for p in (V11_ZIP, V10_ZIP, V10_ADD):
    assert os.path.exists(p), f'⚠ zip がありません（転送は完了していますか）: {p}'

# --- v11 展開＋照合＋検品FAIL除外 ---
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
        assert (cnt, dg) == (7200, V11_DIGEST_FULL[sub]), \
            f'⚠ {sub}: 展開結果がマニフェスト(7200)と不一致 {cnt} {dg}'
    for stem in INSPECT_FAIL:
        for sub, ext in (('foa', 'flac'), ('metadata', 'csv'), ('masks', 'csv')):
            os.remove(f'datasets/{DATASET}/{sub}/{stem}.{ext}')
    print(f'manifest(7200)照合OK -> INSPECT_FAIL除外: {INSPECT_FAIL}')
for sub in ('foa', 'metadata', 'masks'):
    dg, cnt = _dir_digest(DATASET, sub)
    assert (cnt, dg) == (7199, V11_DIGEST_FINAL[sub]), \
        f'⚠ {sub}: 最終状態がマニフェスト(7199)と不一致 {cnt} {dg}'
print('v11 manifest digest OK (7,199 x3 subdirs)')

# --- v10 展開＋評価専用に剪定（fold1系・fold3を削除。冪等） ---
if not os.path.exists(f'datasets/{EVAL_DS}/foa'):
    with zipfile.ZipFile(V10_ZIP) as z:
        z.extractall('.')
    with zipfile.ZipFile(V10_ADD) as z:
        z.extractall('.')
    print('v10+add unzipped')
else:
    print('v10は展開済み')

n10 = len(os.listdir(f'datasets/{EVAL_DS}/foa'))
if n10 > 858:
    removed = 0
    for sub in ('foa', 'metadata', 'masks'):
        for p in glob.glob(f'datasets/{EVAL_DS}/{sub}/fold1_*') + \
                 glob.glob(f'datasets/{EVAL_DS}/{sub}/fold3_*'):
            os.remove(p)
            removed += 1
    n10 = len(os.listdir(f'datasets/{EVAL_DS}/foa'))
    print(f'v10剪定: {removed}ファイル削除')
assert n10 == 858, f'⚠ v10評価セットの本数が想定外: {n10} (期待858)'
n_cls = len(open('datasets/cls_indices_train.tsv').readlines())
assert n_cls == 6, f'⚠ クラス数が想定外: {n_cls}'
print(f'v10 eval foa: {n10} / classes: {n_cls}')

# --- 事前学習 ckpt 配置＋SHA256照合 ---
os.makedirs('ckpts', exist_ok=True)
CKPT = 'ckpts/mACCDOA-HTSAT-0.567.ckpt'
if not os.path.exists(CKPT):
    shutil.copy(f'{DATA}/mACCDOA-HTSAT-0.567.ckpt', CKPT)
h = hashlib.sha256(open(CKPT, 'rb').read()).hexdigest()
assert h == CKPT_SHA256, f'⚠ ckpt SHA256不一致: {h}'
print(f'ckpt OK: {CKPT} ({os.path.getsize(CKPT)/1e6:.0f} MB, sha256一致)')
print('=== prepare_data.py 完了 ===')
