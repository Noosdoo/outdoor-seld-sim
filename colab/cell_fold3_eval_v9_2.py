# === fold3（最終テスト240本）評価: v9.2モデル ===
# 監査 o3-r6 指摘1 への対応。fold2(val)/scenario/probe/v10a は開発に反復使用したため
# 開発セット扱い。fold3 は一度も使っていない held-out。**この推論は1回だけ**走らせ、
# 出力を「テスト性能」として報告する（以後 fold3 は触らない）。
#
# 前提: 因果セル(cell_causal_infer_v9_2.py)と同じランタイム（環境+v9.1 zip展開+v9.2 configs、
#       ckpt読込のみ、学習不要）。T4で5〜8分。
# 出力: DRIVE_LOGS直下 infer_outdoor_siren_v9_2_run1_fold3_all.csv
#       （列 = clip,frame,class,azimuth,elevation。既存 *_all.csv と同形式）
# ローカル採点: scripts/step12_notify_v9.py --v91 --split fold3 --pred <このCSV>
#       （step12 に fold3 名簿を追加済み。SELD系メトリクスは PSELDNets の val ログ準拠）

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
assert os.path.exists(f'configs/experiment/{DATASET}_scn2.yaml'), 'セル38（v9.2 configs）を先に実行'

# --- fold3 の240本を用意（ランタイムに無ければ本体zipから個別展開） ---
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
print('fold3 clips:', len(clips), '(期待240)')
assert len(clips) == 240, 'fold3が240本ありません（本体zipがDriveにあるか確認）'

# --- モデル構築（因果セルと同一） ---
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
    x = torch.from_numpy(wav.T.copy())[None].cuda()   # (1,4,N)
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
    if (ci + 1) % 40 == 0:
        print(f'{ci + 1}/240  {time.time() - t0:.0f}s  lines={len(out_lines)}')

out = f'{logdir}/infer_{EXP}_fold3_all.csv'
with open(out, 'w') as f:
    for line in out_lines:
        print(line, file=f)
print('wrote', out, len(out_lines), 'lines,', f'{time.time() - t0:.0f}s total')
# → Driveから out/predictions_v9_2/fold3_all.csv に置き、ローカルで採点する
