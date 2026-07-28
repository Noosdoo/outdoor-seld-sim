# === v11評価拡張の推論（学習不要、既存run1のckptで7セット一括） =====================
# 前提: v11ノートブックのセル1〜5,7,8を実行済みのランタイム（clone済み・pin済み・
#       v11のconfigsあり）。学習セルは不要。新規に必要なのは
#       Drive PSELDNets_data/dataset_outdoor_siren_v11_eval.zip（約4GB）のみ。
# 評価拡張は独立データセット名 outdoor_siren_v11e として展開する（本体v11の前処理に
# 触れないため）。前処理は空ラベル許容版（_preproc_emptyok.py、セル8が作成済み）を使う。
# 出力: Drive PSELDNets_data/infer_{EXP_NAME}_ev{tag}_all.csv ×7本
import glob
import os
import zipfile

EVAL_DS = "outdoor_siren_v11e"
EVAL_ZIP = f"{DRIVE_DATA}/dataset_outdoor_siren_v11_eval.zip"
assert os.path.exists(EVAL_ZIP), "⚠ eval zipがDriveにありません"

# --- 展開＋マニフェスト照合（16室3,246本、name+sizeダイジェスト。除外なし） ---
import hashlib
if not os.path.exists(f"datasets/{EVAL_DS}/foa"):
    with zipfile.ZipFile(EVAL_ZIP) as z:
        z.extractall(".")
    print("eval unzipped")
WANT = {"fold4_room1": 600, "fold4_room2": 600, "fold5_room1": 200,
        "fold5_room2": 100, "fold5_room3": 100, "fold5_room5": 200,
        "fold5_room9": 100, "fold8_room2": 200, "fold9_room2": 96,
        "fold6_room1": 150, "fold6_room2": 150, "fold6_room3": 150,
        "fold6_room4": 150, "fold6_room5": 150, "fold6_room6": 150,
        "fold6_room7": 150}
DIGEST = {"foa": "5bcd0f66999e5a316e08db084da8fdf4",       # manifest_v11eval.csv由来
          "metadata": "59058f3870a369f616b606fe1a42f188",
          "masks": "49301d62096712c6c18fa7fe13931875"}
for sub in ("foa", "metadata", "masks"):
    entries = sorted((os.path.basename(p), os.path.getsize(p))
                     for p in glob.glob(f"datasets/{EVAL_DS}/{sub}/*"))
    assert len(entries) == 3246, (sub, len(entries))
    for room, n in WANT.items():
        got = sum(1 for x, _ in entries if x.startswith(room + "_"))
        assert got == n, (sub, room, got, n)
    dg = hashlib.md5("\n".join(f"{n},{s}" for n, s in entries).encode()).hexdigest()
    assert dg == DIGEST[sub], f"⚠ {sub}: マニフェスト不一致 {dg}"
print("eval dataset: 3,246 x3 subdirs / room内訳・ダイジェストOK")

# --- 設定（データyaml＋推論7 variant。train/validは存在するroomを指すだけ） ---
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

EV_TAGS = [("evhalluc", "[fold4_room1]"),
           ("evsafe", "[fold4_room2]"),
           ("evscn2", "[fold5_room1, fold5_room2, fold5_room3, fold5_room5]"),
           ("evcross", "[fold5_room9]"),
           ("evmulti", "[fold8_room2]"),
           ("evprobe", "[fold9_room2]"),
           ("evn", "[fold6_room1, fold6_room2, fold6_room3, fold6_room4, fold6_room5, fold6_room6, fold6_room7]")]
open(f"configs/data/{EVAL_DS}.yaml", "w").write(
    data_yaml(EVAL_DS, "[fold4_room1]", "[fold4_room1]", "[fold4_room1]"))
for tag, rooms in EV_TAGS:
    open(f"configs/data/{EVAL_DS}_{tag}.yaml", "w").write(
        data_yaml(EVAL_DS, "[fold4_room1]", "[fold4_room1]", rooms))
    open(f"configs/experiment/{EVAL_DS}_{tag}.yaml", "w").write(
        exp_yaml(EVAL_DS, f"{EVAL_DS}_{tag}"))
print("wrote eval configs (x7)")

# --- 前処理（空ラベル許容版。3,246本のみ、15分前後） ---
idx = f"_hdf5/data/24000fs/wav/dev/{EVAL_DS}_10sChunklen_10sHoplen_train.csv"
if os.path.exists(idx):
    print("eval前処理済み")
else:
    !python _preproc_emptyok.py dataset={EVAL_DS}

# --- 推論（v11 run1のベストckptで7セット） ---
cands = sorted(glob.glob("/content/drive/MyDrive/PSELDNets_logs*"))
cands += sorted(glob.glob("/content/drive/.shortcut-targets-by-id/*/PSELDNets_logs"))
best_ckpt = None
for c in cands:
    hits = sorted(glob.glob(f"{c}/{DATASET}/runs/{EXP_NAME}/checkpoints/epoch_*.ckpt"))
    if hits:
        DRIVE_LOGS = c
        best_ckpt = hits[-1]
        break
print("best_ckpt =", best_ckpt)
assert best_ckpt, "⚠ v11のckptが見えません"

for tag, _ in EV_TAGS:
    exp = f"infer_{EXP_NAME}_{tag}"
    !python src/infer.py experiment={EVAL_DS}_{tag} \
        mode=test \
        ckpt_path="{best_ckpt}" \
        model.kwargs.pretrained_path=null \
        experiment_name={exp} \
        paths.log_dir={DRIVE_LOGS}
    sub = f"{DRIVE_LOGS}/{EVAL_DS}/runs/{exp}/submissions"
    out_lines = []
    for p in sorted(glob.glob(f"{sub}/*.csv")):
        stem = os.path.basename(p)[:-4]
        for line in open(p):
            if line.strip():
                out_lines.append(f"{stem},{line.strip()}")
    out = f"{DRIVE_DATA}/{exp}_all.csv"
    open(out, "w").write("\n".join(out_lines))
    print("wrote", out, len(out_lines), "lines")
print("EVAL INFER ALL DONE")
