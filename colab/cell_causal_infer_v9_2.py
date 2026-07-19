# === 因果推論（過去のみ）実験: v9.2モデル / 6シナリオ120本 ===
# Colabノートブック末尾に新セルとして貼り付けて実行。
# 前提: セル1〜9（環境+v9.1 zip展開）とセル38（v9.2 configs）実行済みのランタイム。
#       学習不要・ckpt読み込みのみ。T4で15〜20分。
# 出力: DRIVE_LOGS直下 infer_outdoor_siren_v9_2_run1_causal_all.csv
#       （列 = clip,frame,class,azimuth,elevation。既存の *_all.csv と同形式）
# 方式: 各判定時刻 t=0.1..10.0s で「その時点までの音声」を右詰め・先頭ゼロ埋めした
#       10秒窓を作り、モデル出力の最終フレームだけ採用（未来参照ゼロの因果予測）。
#       デコードは本番と同一（sed_threshold=0.5, unify=15°）。

import os, glob, time, zipfile, sys
import numpy as np
import torch
import soundfile as sf

DATASET = 'outdoor_siren_v9_1'
EXP = 'outdoor_siren_v9_2_run1'
DRIVE_DATA = '/content/drive/MyDrive/PSELDNets_data'

cands = ['/content/drive/.shortcut-targets-by-id/1R9wsQpgsphuly312IUJZe5PgVpUjzAPa/PSELDNets_logs']
cands += sorted(glob.glob('/content/drive/MyDrive/PSELDNets_logs*'))
logdir, ck = None, None
for c in cands:
    hits = sorted(glob.glob(f'{c}/{DATASET}/runs/{EXP}/checkpoints/epoch_*.ckpt'))
    if hits:
        logdir, ck = c, hits[-1]
        break
assert ck, 'v9_2_run1 の ckpt が見えません（Drive共有/ショートカットを確認）'
print('ckpt =', ck)
assert os.path.exists(f'configs/experiment/{DATASET}_scn2.yaml'), 'セル38（v9.2 configs）を先に実行してください'

# --- 対象クリップ（fold2_room4〜9 = 120本）。ランタイムに無ければzipから個別展開 ---
rooms = ['fold2_room4', 'fold2_room5', 'fold2_room6',
         'fold2_room7', 'fold2_room8', 'fold2_room9']
foa_dir = f'datasets/{DATASET}/foa'
os.makedirs(foa_dir, exist_ok=True)
have = set(os.listdir(foa_dir))
slash = chr(47)
for zp in [f'{DRIVE_DATA}/dataset_{DATASET}.zip', f'{DRIVE_DATA}/dataset_{DATASET}_scn2.zip']:
    if not os.path.exists(zp):
        continue
    with zipfile.ZipFile(zp) as z:
        for nm in z.namelist():
            base = os.path.basename(nm)
            ok = nm.endswith('.flac') and (slash + 'foa' + slash) in nm
            if ok and base not in have and any(r in base for r in rooms):
                z.extract(nm, '.')
                have.add(base)
clips = sorted(p for p in os.listdir(foa_dir)
               if p.endswith('.flac') and any(r in p for r in rooms))
print('clips:', len(clips), '(期待120)')
assert len(clips) == 120, '120本ありません（v9.1本体zipとscn2 zipがDriveのPSELDNets_data/にあるか確認）'

# --- モデル構築（inferと同一のcfg合成・同一クラス） ---
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
print('state_dict: missing', len(missing), '/ unexpected', len(unexpected))
netmiss = [k for k in missing if k.startswith('net.')]
if netmiss:
    print('missing例:', netmiss[:5])
    print('unexpected例:', list(unexpected)[:5])
assert not netmiss, 'netの重みが欠けています'
model.eval()
model.cuda()

# --- 因果推論本体 ---
SR, N, HOP = 24000, 240000, 2400
NCLS = ds.num_classes
out_lines = []
t0 = time.time()
for ci, fn in enumerate(clips):
    wav, sr = sf.read(f'{foa_dir}/{fn}', dtype='float32')
    assert sr == SR and wav.shape[0] == N, fn
    x = torch.from_numpy(wav.T.copy())
    buf = torch.zeros(100, 4, N)
    for k in range(1, 101):
        buf[k - 1, :, N - k * HOP:] = x[:, :k * HOP]
    outs = []
    with torch.no_grad():
        for s in range(0, 100, 25):
            feat = model.standardize(buf[s:s + 25].cuda())
            y = model.net(feat)['multi_accdoa']
            outs.append(y[:, 99, :].float().cpu())
    causal = torch.cat(outs, dim=0)[None]
    sed, doa = get_multi_accdoa_labels(causal, NCLS, 0.5)
    dcase = multi_accdoa_to_dcase_format(sed[:, 0].numpy(), doa[:, 0].numpy(),
                                         nb_classes=NCLS)
    polar = convert_output_format_cartesian_to_polar(in_dict=dcase)
    stem = fn[:-5]
    for fr in sorted(polar.keys()):
        for v in polar[fr]:
            out_lines.append(f'{stem},{int(fr)},{int(v[0])},{int(v[1])},{int(v[2])}')
    if (ci + 1) % 10 == 0:
        print(f'{ci + 1}/120  {time.time() - t0:.0f}s  lines={len(out_lines)}')

out = f'{logdir}/infer_{EXP}_causal_all.csv'
with open(out, 'w') as f:
    for line in out_lines:
        print(line, file=f)
print('wrote', out, len(out_lines), 'lines,', f'{time.time() - t0:.0f}s total')
