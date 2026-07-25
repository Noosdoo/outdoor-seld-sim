# === fold3（最終テスト600本）評価: v10.2モデル ===
# ⚠️ このセルは**1回だけ**実行する。fold3は一度も使っていないheld-outで、
#    実行した瞬間に「開発に使っていないテスト性能」の主張が確定する。
#    **実行タイミングは先生と相談してから**（全結論確定後・発表/卒論の直前推奨）。
#
# 前提: v10_2ノートブックのセル1〜7実行済みランタイム（学習不要・ckpt読込のみ）。T4で15分前後。
# 出力: DRIVE_LOGS直下 infer_outdoor_siren_v10_2_run1_fold3_all.csv
# ローカル採点: step12ランナー（DS=v10, --split fold3, --pred このCSV）＋step13系解剖

import os, glob, time, zipfile, sys
import numpy as np
import torch
import soundfile as sf

DATASET = 'outdoor_siren_v10'
EXP = 'outdoor_siren_v10_2_run1'
DRIVE_DATA = '/content/drive/MyDrive/PSELDNets_data'

cands = sorted(glob.glob('/content/drive/MyDrive/PSELDNets_logs*'))
cands += sorted(glob.glob('/content/drive/.shortcut-targets-by-id/*/PSELDNets_logs'))
logdir, ck = None, None
for c in cands:
    hits = sorted(glob.glob(f'{c}/{DATASET}/runs/{EXP}/checkpoints/epoch_*.ckpt'))
    if hits:
        logdir, ck = c, hits[-1]
        break
assert ck, 'v10_2_run1 の ckpt が見えません'
print('ckpt =', ck)
assert os.path.exists(f'configs/experiment/{DATASET}_scn2.yaml'), 'セル7を先に実行'

# --- fold3 の600本を用意（無ければ本体zipから個別展開） ---
foa_dir = f'datasets/{DATASET}/foa'
os.makedirs(foa_dir, exist_ok=True)
have = set(os.listdir(foa_dir))
slash = chr(47)
zp = f'{DRIVE_DATA}/dataset_{DATASET}.zip'
if os.path.exists(zp):
    with zipfile.ZipFile(zp) as z:
        for nm in z.namelist():
            base = os.path.basename(nm)
            if (nm.endswith('.flac') and (slash + 'foa' + slash) in nm
                    and base.startswith('fold3_') and base not in have):
                z.extract(nm, '.')
                have.add(base)
clips = sorted(p for p in os.listdir(foa_dir)
               if p.endswith('.flac') and p.startswith('fold3_'))
print('fold3 clips:', len(clips), '(期待596〜600、空ラベル除外の有無で変動)')
assert 590 <= len(clips) <= 600, 'fold3の本数が想定外（本体zipを確認）'

# --- モデル構築 ---
sys.path.insert(0, 'src')
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
initialize_config_dir(config_dir=os.path.abspath('configs'), version_base='1.3')
cfg = compose(config_name='infer.yaml',
              overrides=[f'experiment={DATASET}_scn2', 'mode=test',
                         'model.kwargs.pretrained_path=null'])
from utils.config import get_dataset
from models.model_module import SELDModelModule
from utils.data_utilities import (get_multi_accdoa_labels,
                                  multi_accdoa_to_dcase_format,
                                  convert_output_format_cartesian_to_polar)
ds = get_dataset(dataset_name=DATASET, cfg=cfg)
model = SELDModelModule(cfg, ds, test_meta={})
model.setup('predict')
try:
    sd = torch.load(ck, map_location='cpu')['state_dict']
except Exception:
    sd = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
netmiss = [k for k in missing if k.startswith('net.')]
assert not netmiss, f'netの重みが欠けています: {netmiss[:5]}'
model.eval(); model.cuda()

# --- 通常（全クリップ一括＝非因果）推論。テスト性能はこの標準推論で報告 ---
SR, N = 24000, 240000
NCLS = ds.num_classes
out_lines = []
t0 = time.time()
for ci, fn in enumerate(clips):
    wav, sr = sf.read(f'{foa_dir}/{fn}', dtype='float32')
    assert sr == SR and wav.shape[0] == N, fn
    x = torch.from_numpy(wav.T.copy())[None].cuda()
    with torch.no_grad():
        feat = model.standardize(x)
        y = model.net(feat)['multi_accdoa'].float().cpu()
    sed, doa = get_multi_accdoa_labels(y, NCLS, 0.5)
    dcase = multi_accdoa_to_dcase_format(sed[:, 0].numpy(), doa[:, 0].numpy(),
                                         nb_classes=NCLS)
    polar = convert_output_format_cartesian_to_polar(in_dict=dcase)
    stem = fn[:-5]
    for fr in sorted(polar.keys()):
        for v in polar[fr]:
            out_lines.append(f'{stem},{int(fr)},{int(v[0])},{int(v[1])},{int(v[2])}')
    if (ci + 1) % 100 == 0:
        print(f'{ci + 1}/{len(clips)}  {time.time() - t0:.0f}s  lines={len(out_lines)}')

out = f'{logdir}/infer_{EXP}_fold3_all.csv'
with open(out, 'w') as f:
    for line in out_lines:
        print(line, file=f)
print('wrote', out, len(out_lines), 'lines,', f'{time.time() - t0:.0f}s total')
