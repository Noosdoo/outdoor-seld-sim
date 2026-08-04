# -*- coding: utf-8 -*-
"""E0用の最小前処理: extract_index のみ実行（ラベル抽出はしない）。

HPF派生データセットは音だけが変わり正解ラベルは v11 と同一なので、
ラベルh5は本体のものを symlink して使う（sbatch側で張る）。
使い方: .venv/bin/python _e0_preproc_dataonly.py dataset=outdoor_siren_v11_hpf50 \
            dataset_type=dev wav_format=.flac mode=extract_data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import hydra
from omegaconf import DictConfig
from utils.config import get_dataset
from preproc.preprocess import Preprocess


@hydra.main(version_base="1.3", config_path="configs", config_name="preproc.yaml")
def main(cfg: DictConfig):
    dataset = get_dataset(dataset_name=cfg.dataset, cfg=cfg)
    preprocessor = Preprocess(cfg, dataset)
    preprocessor.extract_index()   # データindexのみ。ラベル抽出は呼ばない


if __name__ == "__main__":
    main()
