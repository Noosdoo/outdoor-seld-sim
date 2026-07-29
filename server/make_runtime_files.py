#!/usr/bin/env python
"""設定yamlと空ラベル許容ラッパを PSELDNets リポジトリ内に生成する。

v11 Colab ノートブック セル7（configs）＋セル8（wrappers）の移植。中身は同一。
実行場所: PSELDNets リポジトリ直下（cwd）で
    python ~/research/outdoor_seld_e2e/server/make_runtime_files.py
"""
import os
from pathlib import Path

DATASET = 'outdoor_siren_v11'
EVAL_DS = 'outdoor_siren_v10'

assert os.path.isdir('configs'), '⚠ PSELDNets リポジトリ直下で実行してください'


def data_yaml(ds, train_rooms, valid_rooms, test_rooms):
    return f"""audio_type: foa
audio_feature: logmelIV
sample_rate: 24000
nfft: 1024
n_mels: 64
hoplen: 240
window: hann

train_chunklen_sec: 10
train_hoplen_sec: 10
test_chunklen_sec: 10
test_hoplen_sec: 10

train_dataset:
  {ds}: {train_rooms}
valid_dataset:
  {ds}: {valid_rooms}
test_dataset:
  {ds}: {test_rooms}
"""


def exp_yaml(ds, data_name):
    return f"""# @package _global_
defaults:
 - override /data: {data_name}.yaml
 - override /loss: multi_accdoa.yaml
 - _self_

task_name: {ds}

model:
  batch_size: 8
  kwargs:
    pretrained_path: ckpts/mACCDOA-HTSAT-0.567.ckpt
    audioset_pretrain: false
  optimizer:
    kwargs: {{lr: 0.0003}}
  lr_scheduler:
    kwargs: {{step_size: 60}}

trainer:
  max_epochs: 100
  check_val_every_n_epoch: 5
"""


# v11: 学習本体 + val推論variant
open(f'configs/data/{DATASET}.yaml', 'w').write(
    data_yaml(DATASET, '[fold1_room1]', '[fold2_room1]', '[fold3_room1]'))
open(f'configs/experiment/{DATASET}.yaml', 'w').write(exp_yaml(DATASET, DATASET))
open(f'configs/data/{DATASET}_valinfer.yaml', 'w').write(
    data_yaml(DATASET, '[fold1_room1]', '[fold2_room1]', '[fold2_room1]'))
open(f'configs/experiment/{DATASET}_valinfer.yaml', 'w').write(
    exp_yaml(DATASET, f'{DATASET}_valinfer'))

# v10: 評価5種（学習には使わない。train/validは存在するroomを指すだけ）
EVAL_TAGS = [('scenario', '[fold2_room9]'),
             ('probe', '[fold9_room1]'),
             ('scn2', '[fold2_room4, fold2_room5, fold2_room6, fold2_room7, fold2_room8]'),
             ('v10a', '[fold8_room1]'),
             ('halluc', '[fold2_room3]')]
for tag, rooms in EVAL_TAGS:
    open(f'configs/data/{EVAL_DS}_{tag}.yaml', 'w').write(
        data_yaml(EVAL_DS, '[fold2_room1]', '[fold2_room1]', rooms))
    open(f'configs/experiment/{EVAL_DS}_{tag}.yaml', 'w').write(
        exp_yaml(EVAL_DS, f'{EVAL_DS}_{tag}'))
print('wrote configs (v11 train/valinfer + v10 eval x5)')

WRAPPER = r'''# _preproc_emptyok.py (auto-generated)
import os
import runpy
import sys

sys.path.insert(0, "src")
import pandas as pd
import utils.data_utilities as du
import preproc.preprocess as pp

_load, _read = du.load_output_format_file, pd.read_csv


def _is_empty_file(path):
    try:
        return os.path.getsize(path) == 0
    except (TypeError, OSError, ValueError):
        return False


def load_ok(path, *a, **kw):
    if _is_empty_file(path):
        return {99: []}          # 最終フレームのみ・イベント0件（ゼロ行列になる）
    return _load(path, *a, **kw)


def read_ok(path, *a, **kw):
    if _is_empty_file(path):
        return pd.DataFrame([[99, 0, 0, 0, 0]])   # num_frames=100 算出専用の番兵
    return _read(path, *a, **kw)


du.load_output_format_file = load_ok
pp.load_output_format_file = load_ok   # from-import名ごと差し替え
pd.read_csv = read_ok                  # 別プロセス内のみのグローバル差し替え

sys.argv = ["src/preproc.py"] + sys.argv[1:]
runpy.run_path("src/preproc.py", run_name="__main__")
'''
Path('_preproc_emptyok.py').write_text(WRAPPER, encoding='utf-8')

WRAPPER_TRAIN = r'''# _train_emptyok.py (auto-generated)
import os
import runpy
import sys

sys.path.insert(0, "src")
import utils.data_utilities as du
import data.components.data as dcd

_load = du.load_output_format_file


def _is_empty_file(path):
    try:
        return os.path.getsize(path) == 0
    except (TypeError, OSError, ValueError):
        return False


def load_ok(path, *a, **kw):
    if _is_empty_file(path):
        return {99: []}   # イベント0件のGT
    return _load(path, *a, **kw)


du.load_output_format_file = load_ok
dcd.load_output_format_file = load_ok   # data.pyはfrom-importなので名前ごと差し替え

sys.argv = ["src/train.py"] + sys.argv[1:]
runpy.run_path("src/train.py", run_name="__main__")
'''
Path('_train_emptyok.py').write_text(WRAPPER_TRAIN, encoding='utf-8')
print('wrote _preproc_emptyok.py / _train_emptyok.py')
