# === 単一ストリーム実時間ベンチ（batch=1）: v9.2モデル ===
# 監査 o3-r6 指摘2 への対応。causal実験の「11.3ms」は25窓バッチのスループットで、
# 単一ストリームのオンライン応答時間ではない。ここでは実運用に即した
# **batch=1・warm-up後・CUDA同期・前処理(standardize)込み**の1判定レイテンシを
# p50/p95/p99 で測る。判定周期100ms（0.1s）に対する余裕を見る。
#
# 前提: 因果セルと同じランタイム（ckpt読込済みのmodelがあれば再利用可）。
# 注意: T4基準。装着デバイス級ハードの実測・量子化/蒸留は別途（本セルの対象外）。

import os, glob, sys, time
import numpy as np
import torch
import soundfile as sf

DATASET = 'outdoor_siren_v9_1'
EXP = 'outdoor_siren_v9_2_run1'
DRIVE_DATA = '/content/drive/MyDrive/PSELDNets_data'
N_WARMUP = 30
N_TRIAL = 300

# --- model が未定義なら因果セルと同じ手順で構築 ---
if 'model' not in dir():
    cands = ['/content/drive/.shortcut-targets-by-id/1R9wsQpgsphuly312IUJZe5PgVpUjzAPa/PSELDNets_logs']
    cands += sorted(glob.glob('/content/drive/MyDrive/PSELDNets_logs*'))
    ck = None
    for c in cands:
        hits = sorted(glob.glob(f'{c}/{DATASET}/runs/{EXP}/checkpoints/epoch_*.ckpt'))
        if hits:
            ck = hits[-1]; break
    assert ck, 'ckptが見えません'
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
    ds = get_dataset(dataset_name=DATASET, cfg=cfg)
    model = SELDModelModule(cfg, ds, test_meta={})
    model.setup('predict')
    sd = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval(); model.cuda()

from utils.data_utilities import get_multi_accdoa_labels
NCLS = model.dataset.num_classes if hasattr(model, 'dataset') else 6
SR, N = 24000, 240000
dev = torch.device('cuda')

# 代表窓（10秒・batch=1）。中身は何でもよいが実データで測る
foa_dir = f'datasets/{DATASET}/foa'
fn = sorted(p for p in os.listdir(foa_dir) if p.endswith('.flac'))[0]
wav, _ = sf.read(f'{foa_dir}/{fn}', dtype='float32')
x1 = torch.from_numpy(wav[:N].T.copy())[None].to(dev)   # (1,4,N)


def one_decision(x):
    """1判定 = standardize（前処理）→ net → 最終フレームのデコード。"""
    feat = model.standardize(x)
    y = model.net(feat)['multi_accdoa']
    last = y[:, -1:, :].float().cpu()          # 最終フレームだけ（因果運用）
    get_multi_accdoa_labels(last, NCLS, 0.5)   # デコードも計上
    return None


# --- warm-up（CUDAカーネルのJIT・cudnnオートチューン込みで暖機） ---
with torch.no_grad():
    for _ in range(N_WARMUP):
        one_decision(x1)
    torch.cuda.synchronize()

# --- 計測（各試行でsynchronizeして純GPU+CPU往復を含む壁時計時間） ---
lat = np.empty(N_TRIAL)
with torch.no_grad():
    for i in range(N_TRIAL):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_decision(x1)
        torch.cuda.synchronize()
        lat[i] = (time.perf_counter() - t0) * 1e3   # ms

p = np.percentile(lat, [50, 90, 95, 99])
print(f'batch=1 単一判定レイテンシ（前処理+net+最終フレームdecode、warm-up{N_WARMUP}/計測{N_TRIAL}）')
print(f'  mean {lat.mean():.2f} ms / p50 {p[0]:.2f} / p90 {p[1]:.2f} / p95 {p[2]:.2f} / p99 {p[3]:.2f} ms')
print(f'  判定周期100ms に対する余裕: p99で {100 / p[3]:.1f}x')
print('  ※T4基準。装着デバイス級ハード・量子化/蒸留での実測は今後。')
