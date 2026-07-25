# === 装着傾き耐性テスト: v10.2モデル / FOA回転（実録スモーク準備） ===
# 目的: チェスト装着による傾き（ピッチ=前傾/ロール=左右）でsim性能がどれだけ落ちるかを
#       事前に定量化し、実録の装着精度要件を数値で決める（実録スモーク計画書 4節3項）。
# 方式: プローブ48本(fold9_room1)＋S2背後ベル20本(fold2_room6)のFOAを
#       ±10°/±20°回転させて推論（1次アンビソニックスの回転は厳密）。
# 前提: v10_2ノートブックのセル1〜7実行済み。T4で約10分。
# 出力: 各回転タグごとに infer_..._tilt_<tag>_all.csv（ローカルでstep18等で採点）

import os, glob, time, sys, math
import numpy as np
import torch
import soundfile as sf

DATASET = 'outdoor_siren_v10'
EXP = 'outdoor_siren_v10_2_run1'

cands = sorted(glob.glob('/content/drive/MyDrive/PSELDNets_logs*'))
cands += sorted(glob.glob('/content/drive/.shortcut-targets-by-id/*/PSELDNets_logs'))
logdir, ck = None, None
for c in cands:
    hits = sorted(glob.glob(f'{c}/{DATASET}/runs/{EXP}/checkpoints/epoch_*.ckpt'))
    if hits:
        logdir, ck = c, hits[-1]
        break
assert ck, 'ckptが見えません'
print('ckpt =', ck)

foa_dir = f'datasets/{DATASET}/foa'
clips = sorted(p for p in os.listdir(foa_dir)
               if p.endswith('.flac')
               and (p.startswith('fold9_room1') or p.startswith('fold2_room6')))
print('clips:', len(clips), '(期待68 = プローブ48+S2 20)')
assert len(clips) == 68

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
sd = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval(); model.cuda()

# --- FOA回転（チャンネル順 = W,Y,Z,X / ACN。X=前, Y=左, Z=上）---
def rot_matrix(pitch_deg, roll_deg):
    """マイクを前傾pitch・右ロールrollさせた状態の音場を模擬する回転行列（XYZ順）。"""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    Rp = np.array([[math.cos(p), 0, math.sin(p)],
                   [0, 1, 0],
                   [-math.sin(p), 0, math.cos(p)]])   # pitch: X-Z面
    Rr = np.array([[1, 0, 0],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r), math.cos(r)]])    # roll: Y-Z面
    return Rr @ Rp


def rotate_foa(wav, R):
    """wav (N,4)=W,Y,Z,X → 回転後 (N,4)。Wは不変、(X,Y,Z)ベクトルにRを適用。"""
    out = wav.copy()
    xyz = wav[:, [3, 1, 2]]          # X,Y,Z
    xyz = xyz @ R.T
    out[:, 3], out[:, 1], out[:, 2] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return out


TAGS = {'p10': (10, 0), 'p20': (20, 0), 'r10': (0, 10), 'r20': (0, 20)}
SR, N = 24000, 240000
NCLS = ds.num_classes
for tag, (pitch, roll) in TAGS.items():
    R = rot_matrix(pitch, roll)
    out_lines = []
    t0 = time.time()
    for fn in clips:
        wav, sr = sf.read(f'{foa_dir}/{fn}', dtype='float32')
        assert sr == SR and wav.shape[0] == N
        wav = rotate_foa(wav, R).astype('float32')
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
    out = f'{logdir}/infer_{EXP}_tilt_{tag}_all.csv'
    with open(out, 'w') as f:
        for line in out_lines:
            print(line, file=f)
    print(f'{tag}: wrote {out} {len(out_lines)} lines ({time.time() - t0:.0f}s)')
print('done — 4ファイルをローカルへ（採点はstep18のPRED差し替え＋S2方位比較）')
