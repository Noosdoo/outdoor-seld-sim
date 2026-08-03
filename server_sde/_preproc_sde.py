# -*- coding: utf-8 -*-
"""距離付きADPITラベル(adpit_dist)の抽出。PSELDNetsルートに置いて実行する。

- 入力: datasets/<ds>/metadata_dist/*.csv （6列 [frame,class,track,az,el,dist]）
- 出力: <hdf5_dir>/label/adpit_dist/<dataset_type>/<ds>.h5
        各クリップ {fn}/adpit/{se,azi,ele,dist} = [frames, 6, classes]
- se/azi/ele は既存 extract_adpit_label と同一のスロット規則
  （フレーム内をクラスでソート→同一クラス連続グループ: 1個→slot0 / 2個→slot1,2 /
    3個以上→slot3,4,5）。distだけが追加列。
- 音声特徴・indexは既存を流用（本スクリプトはラベルh5のみ生成）

使い方(サーバ):
    python _preproc_sde.py mode=extract_data dataset=outdoor_siren_v11 \
        dataset_type=train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import sde_patch  # noqa: E402

sde_patch.apply_loader_patch()

from preproc.preprocess import Preprocess  # noqa: E402
from utils.config import get_dataset  # noqa: E402
import utils.data_utilities as du  # noqa: E402


def build_adpit_dist(desc_file, num_classes):
    """[frames,6,C]のse/azi/ele/dist行列を作る（既存スロット規則の等価実装）。"""
    nb = list(desc_file.keys())[-1] + 1
    se = np.zeros((nb, 6, num_classes), dtype=bool)
    azi = np.zeros((nb, 6, num_classes), dtype=np.int16)
    ele = np.zeros((nb, 6, num_classes), dtype=np.int8)
    dist = np.zeros((nb, 6, num_classes), dtype=np.float32)
    for f, evs in desc_file.items():
        if f >= nb:
            continue
        evs = sorted(evs, key=lambda x: x[0])
        # 同一クラスの連続グループへ分割
        groups, cur = [], []
        for i, ev in enumerate(evs):
            cur.append(ev)
            if i == len(evs) - 1 or ev[0] != evs[i + 1][0]:
                groups.append(cur)
                cur = []
        for g in groups:
            slots = [0] if len(g) == 1 else ([1, 2] if len(g) == 2
                                             else [3, 4, 5])
            for slot, ev in zip(slots, g):
                c = int(ev[0])
                se[f, slot, c] = True
                azi[f, slot, c] = int(round(ev[1]))
                ele[f, slot, c] = int(round(ev[2]))
                dist[f, slot, c] = float(ev[3]) if len(ev) > 3 else 0.0
    return se, azi, ele, dist


@hydra.main(version_base="1.3", config_path="configs", config_name="preproc.yaml")
def main(cfg: DictConfig):
    dataset = get_dataset(dataset_name=cfg.dataset, cfg=cfg)
    pre = Preprocess(cfg, dataset)
    meta_dir = Path(str(pre.meta_dir)).parent / "metadata_dist"
    assert meta_dir.is_dir(), f"metadata_dist が無い: {meta_dir}"
    # label/adpit/<type>/<ds>.h5 -> label/adpit_dist/<type>/<ds>.h5
    p = Path(str(pre.meta_adpit_path))
    out_path = p.parent.parent.parent / "adpit_dist" / p.parent.name / p.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        out_path.unlink()

    meta_list = [p for p in sorted(meta_dir.glob("*.csv"))
                 if not p.name.startswith(".")]
    print(f"[SDE] adpit_dist抽出: {len(meta_list)}本 -> {out_path}")
    hf = h5py.File(out_path, "w")
    for idx, meta_file in enumerate(tqdm(meta_list, unit="it")):
        fn = meta_file.stem
        desc = du.load_output_format_file(meta_file)   # パッチ済(6列dist保持)
        se, azi, ele, dist = build_adpit_dist(desc, pre.num_classes)
        hf.create_dataset(f"{fn}/adpit/se", data=se)
        hf.create_dataset(f"{fn}/adpit/azi", data=azi)
        hf.create_dataset(f"{fn}/adpit/ele", data=ele)
        hf.create_dataset(f"{fn}/adpit/dist", data=dist)
    hf.close()
    print("[SDE] done")


if __name__ == "__main__":
    main()
