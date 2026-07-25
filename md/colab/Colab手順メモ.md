# Colab で outdoor_siren_v1（train30/val10）を学習する手順

ローカルで `scripts/step6_batch_scenes.py pack` が作る
`out/dataset_outdoor_siren_v1.zip` を使う。**以下の設定ファイル2つは
Colabノートブック側で自分で作成する**（差分テキスト方式）。

## 1. zip を Drive にアップロード

`dataset_outdoor_siren_v1.zip` を Drive の PSELDNets 用フォルダ
（例: `MyDrive/PSELDNets_data/`）へ。中身は `datasets/...` 構成なので
**PSELDNets リポジトリ直下で unzip するだけで配置が完了**する。

## 2. ノートブックに追加するセル（テキスト）

```bash
# データ展開（リポジトリ直下で）
%cd {PSELDNetsリポジトリのパス}
!unzip -o /content/drive/MyDrive/PSELDNets_data/dataset_outdoor_siren_v1.zip -d .
!ls datasets/outdoor_siren_v1/foa | head -3 && wc -l datasets/cls_indices_train.tsv
```

```python
# 設定ファイル2つを作成（新規追加のみ・既存ファイル無編集）
data_yaml = """\
audio_type: foa
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
  outdoor_siren_v1: [fold1_room1]
valid_dataset:
  outdoor_siren_v1: [fold2_room1]
test_dataset:
  outdoor_siren_v1: [fold2_room1]
"""
exp_yaml = """\
# @package _global_
defaults:
 - override /data: outdoor_siren_v1.yaml
 - override /loss: multi_accdoa.yaml
 - _self_

task_name: outdoor_siren_v1

model:
  batch_size: 16
  kwargs:
    pretrained_path: ckpts/mACCDOA-HTSAT-0.567.ckpt
    audioset_pretrain: false
  optimizer:
    kwargs: {lr: 0.0003}
  lr_scheduler:
    kwargs: {step_size: 60}

trainer:
  max_epochs: 70
  check_val_every_n_epoch: 5
"""
open('configs/data/outdoor_siren_v1.yaml', 'w').write(data_yaml)
open('configs/experiment/outdoor_siren_v1.yaml', 'w').write(exp_yaml)
```

```bash
# 前処理（Linux なので Windows で必要だった index パス修正は不要）
!python src/preproc.py dataset=outdoor_siren_v1

# 学習（T4 で 15〜30 分見込み。ログの val/macro 行が成績）
!python src/train.py experiment=outdoor_siren_v1
```

## 3. 注意点（ローカル版との違い・既知の対処）

- `trainer.accelerator` は既定の gpu のまま（ローカル版だけ cpu 上書きしていた）。
  `compile` も既定 true のままで可（Linux では動く。6月のfinetune実績と同条件）
- `ckpts/mACCDOA-HTSAT-0.567.ckpt` が Colab 側に必要。Drive の PSELDNets_ckpts に
  ある想定。なければ:
  `!wget -O ckpts/mACCDOA-HTSAT-0.567.ckpt "https://huggingface.co/datasets/Jinbo-HU/PSELDNets/resolve/main/model/mACCDOA-HTSAT-0.567.ckpt"`
- requirements の numpy/torch 矛盾は6月に確立した手順どおり（numpy 1.26.4）
- セッション切断対策はいつもの Drive 永続化＋固定 experiment_name＋resume 方式
- batch_size=16 の理由: train が30クリップしかないため（既定128は過大）
- augment はまず既定（なし）で1回通す。数値を見てから augmix1 を検討（慎重に）
- 期待の見方: val/macro の ER/F/LE/LR/SELD_scr。今回から SELD_scr を必ず併記
- **数値の位置づけ**: val は学習に見せていない10本なので、今回初めて
  「見たことないシーンへの汎化」の数字になる。1クリップ貫通時の数値
  （ER 0.000 等）とは意味が違うことに注意
