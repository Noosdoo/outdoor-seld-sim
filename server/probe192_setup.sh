set -e
cd ~/research/PSELDNet/PSELDNets
DS=outdoor_siren_probe192
echo "=== unzip ==="
unzip -oq ~/PSELDNets_data/dataset_${DS}.zip -d datasets/
echo "foa: $(ls datasets/${DS}/foa/*.flac | wc -l) / meta: $(ls datasets/${DS}/metadata/*.csv | wc -l)"

echo "=== write data config ==="
cat > configs/data/${DS}.yaml <<'EOF'
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
  outdoor_siren_probe192: [fold9_room3]
valid_dataset:
  outdoor_siren_probe192: [fold9_room3]
test_dataset:
  outdoor_siren_probe192: [fold9_room3]
EOF

echo "=== write experiment config ==="
cat > configs/experiment/${DS}.yaml <<'EOF'
# @package _global_
defaults:
 - override /data: outdoor_siren_probe192.yaml
 - override /loss: multi_accdoa.yaml
 - _self_

task_name: outdoor_siren_probe192

model:
  batch_size: 8
  kwargs:
    pretrained_path: ckpts/mACCDOA-HTSAT-0.567.ckpt
    audioset_pretrain: false
  optimizer:
    kwargs: {lr: 0.0003}
  lr_scheduler:
    kwargs: {step_size: 60}

trainer:
  max_epochs: 100
  check_val_every_n_epoch: 5
EOF
echo "configs written"

echo "=== preproc (login node, CPU) ==="
timeout 300 .venv/bin/python _preproc_emptyok.py dataset=${DS} 2>&1 | tail -20
echo "=== index files ==="
ls -l _hdf5/data/24000fs/wav/dev/*probe192* 2>/dev/null
echo "=== SETUP DONE ==="
