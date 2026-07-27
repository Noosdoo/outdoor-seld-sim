# -*- coding: utf-8 -*-
"""v11 Colabノートブック生成（v10.2版の構成を踏襲、v11差分を反映）。"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\satos\research\outdoor_seld_e2e\colab\PSELDNets_outdoor_siren_v11_Colab.ipynb")

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s}

cells = []

cells.append(md(
"# PSELDNets × v11（core 7,200本=20h、自然頻度＋重要セルフロア、純静穏・警告のみ・複数車をcoreに統合）\n"
"\n"
"事前準備（Drive `MyDrive/PSELDNets_data/`）:\n"
"- `dataset_outdoor_siren_v11.zip`（約9GB、core 7,200本）← **今回新規にアップロード**\n"
"- `dataset_outdoor_siren_v10.zip`（4.7GB）と `dataset_outdoor_siren_v10_2_add.zip`（884MB）← v10.2学習で配置済みのはず（評価専用セットに使う）\n"
"\n"
"- 学習 fold1_room1 4,800 / val fold2_room1 1,200 / test fold3_room1 1,199（検品FAIL1本除外。**testは最終1回まで触らない**）\n"
"- 評価枠はv10側を共用（物理同一でビット一致のため再生成なし）: 交差点20・プローブ48・6シナリオ100・交通量60・幻覚30\n"
"- **v10.2までの「空ラベル除外セル」は廃止**。純静穏582本は設計上の教師なので、空ラベル許容preproc（セル8）で学習に載せる\n"
"- 設計の正: `md/design/v11データセット拡張_設計書_2026-07-27.md`。**データを変えたら EXP_NAME を必ず変える**\n"
"- 学習は約7時間（T4/100ep）。**切れても再実行すれば last.ckpt から自動再開**（Drive永続化）"))

cells.append(md("---\n## 1. GPU 確認"))
cells.append(code(
"import torch\n"
"assert torch.cuda.is_available(), '⚠ GPUがありません。ランタイム→T4 GPU を選択してください'\n"
"print(f'PyTorch : {torch.__version__}')\n"
"print(f'GPU     : {torch.cuda.get_device_name(0)}')\n"
"print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"))

cells.append(md("## 2. Drive マウントと設定"))
cells.append(code(
"from google.colab import drive\n"
"drive.mount('/content/drive')\n"
"\n"
"# ==== 設定 ====\n"
"DRIVE_DATA = '/content/drive/MyDrive/PSELDNets_data'\n"
"DRIVE_LOGS = '/content/drive/MyDrive/PSELDNets_logs'\n"
"DRIVE_CKPT = '/content/drive/MyDrive/PSELDNets_ckpts'\n"
"DATASET    = 'outdoor_siren_v11'    # 学習用\n"
"EVAL_DS    = 'outdoor_siren_v10'    # 評価専用セット（v11と物理同一・ビット一致）\n"
"EXP_NAME   = 'outdoor_siren_v11_run1'\n"
"\n"
"import os\n"
"for d in [DRIVE_DATA, DRIVE_LOGS, DRIVE_CKPT]:\n"
"    os.makedirs(d, exist_ok=True)\n"
"V11_ZIP = f'{DRIVE_DATA}/dataset_outdoor_siren_v11.zip'\n"
"V10_ZIP = f'{DRIVE_DATA}/dataset_outdoor_siren_v10.zip'\n"
"V10_ADD = f'{DRIVE_DATA}/dataset_outdoor_siren_v10_2_add.zip'\n"
"assert os.path.exists(V11_ZIP), '⚠ v11 zipがDriveにありません'\n"
"assert os.path.exists(V10_ZIP), '⚠ v10 zip（評価用）がDriveにありません'\n"
"assert os.path.exists(V10_ADD), '⚠ v10.2追補zip（幻覚評価room）がDriveにありません'\n"
"print(f'OK: v11 {os.path.getsize(V11_ZIP)/1e9:.2f}GB / v10 {os.path.getsize(V10_ZIP)/1e9:.2f}GB / add {os.path.getsize(V10_ADD)/1e6:.0f}MB')"))

cells.append(md("## 3. リポジトリ clone"))
cells.append(code(
"import os\n"
"\n"
"REPO = '/content/PSELDNets'\n"
"\n"
"if not os.path.exists(f'{REPO}/src'):\n"
"    !git clone https://github.com/Jinbo-Hu/PSELDNets {REPO}\n"
"else:\n"
"    print(f'既に存在: {REPO}')\n"
"\n"
"os.chdir(REPO)\n"
"print(f'CWD: {os.getcwd()}')"))

cells.append(md("## 4. 依存インストール"))
cells.append(code(
"!pip install -q \\\n"
"    librosa \\\n"
"    soundfile \\\n"
"    lightning==2.2.1 \\\n"
"    hydra-core==1.3.2 \\\n"
"    hydra-colorlog==1.2.0 \\\n"
"    hydra-joblib-launcher==1.2.0 \\\n"
"    torchmetrics==1.3.1\n"
"\n"
"import numpy, lightning, torchmetrics, librosa\n"
"print(f'numpy {numpy.__version__} / lightning {lightning.__version__} / '\n"
"      f'torchmetrics {torchmetrics.__version__} / librosa {librosa.__version__}')"))

cells.append(md("## 5. 事前学習チェックポイント（Drive キャッシュ → なければ HF から）"))
cells.append(code(
"import shutil\n"
"\n"
"os.makedirs('ckpts', exist_ok=True)\n"
"CKPT = 'ckpts/mACCDOA-HTSAT-0.567.ckpt'\n"
"CACHE = f'{DRIVE_CKPT}/mACCDOA-HTSAT-0.567.ckpt'\n"
"\n"
"if not os.path.exists(CKPT):\n"
"    if os.path.exists(CACHE):\n"
"        print('Drive キャッシュからコピー...')\n"
"        shutil.copy(CACHE, CKPT)\n"
"    else:\n"
"        print('HuggingFace からダウンロード...')\n"
"        from huggingface_hub import hf_hub_download\n"
"        src = hf_hub_download(repo_id='Jinbo-HU/PSELDNets',\n"
"                              filename='model/mACCDOA-HTSAT-0.567.ckpt',\n"
"                              repo_type='dataset')\n"
"        shutil.copy(src, CKPT)\n"
"        shutil.copy(CKPT, CACHE)   # 次回用に Drive にキャッシュ\n"
"print(f'OK: {CKPT} ({os.path.getsize(CKPT)/1e6:.0f} MB)')"))

cells.append(md(
"## 6. データセット展開（v11約9GB＋v10評価用、20分前後）\n"
"\n"
"v10側は評価専用room（fold2_room3〜9 / fold8 / fold9）と fold2_room1 だけ残して間引く\n"
"（前処理の時間とディスクの節約。学習はv11のみで行うため fold1系・fold3 は不要）。"))
cells.append(code(
"import zipfile, os, glob\n"
"\n"
"if not os.path.exists(f'datasets/{DATASET}/foa'):\n"
"    with zipfile.ZipFile(V11_ZIP) as z:\n"
"        z.extractall('.')\n"
"    print('v11 unzipped')\n"
"else:\n"
"    print('v11は展開済み')\n"
"\n"
"if not os.path.exists(f'datasets/{EVAL_DS}/foa'):\n"
"    with zipfile.ZipFile(V10_ZIP) as z:\n"
"        z.extractall('.')\n"
"    with zipfile.ZipFile(V10_ADD) as z:\n"
"        z.extractall('.')\n"
"    print('v10+add unzipped')\n"
"else:\n"
"    print('v10は展開済み')\n"
"\n"
"# 検品FAIL（ローカルinspection.csvで確定・2026-07-28）: 受音ゲート±3.5dBの良性の裾\n"
"# （ベルの打撃×距離の偶然相関、実測-4.33dBのうち-3.29dBを打撃タイミングで説明済み）。\n"
"# 物理異常ではないが v10 の mix119 と同じ方針で除外する（fold3は1,199本になる）\n"
"INSPECT_FAIL = ['fold3_room1_mix0181']\n"
"n11 = len(os.listdir(f'datasets/{DATASET}/foa'))\n"
"if n11 == 7200:\n"
"    for stem in INSPECT_FAIL:\n"
"        for sub, ext in (('foa', 'flac'), ('metadata', 'csv'), ('masks', 'csv')):\n"
"            p = f'datasets/{DATASET}/{sub}/{stem}.{ext}'\n"
"            if os.path.exists(p):\n"
"                os.remove(p)\n"
"    n11 = len(os.listdir(f'datasets/{DATASET}/foa'))\n"
"    print(f'INSPECT_FAIL removed: {INSPECT_FAIL}')\n"
"assert n11 == 7200 - len(INSPECT_FAIL), f'⚠ v11の本数が想定外: {n11}'\n"
"\n"
"# --- v10を評価専用に間引く（fold1系・fold3を削除。冪等） ---\n"
"n10 = len(os.listdir(f'datasets/{EVAL_DS}/foa'))\n"
"if n10 > 858:\n"
"    removed = 0\n"
"    for sub in ('foa', 'metadata', 'masks'):\n"
"        for p in glob.glob(f'datasets/{EVAL_DS}/{sub}/fold1_*') + \\\n"
"                 glob.glob(f'datasets/{EVAL_DS}/{sub}/fold3_*'):\n"
"            os.remove(p)\n"
"            removed += 1\n"
"    n10 = len(os.listdir(f'datasets/{EVAL_DS}/foa'))\n"
"    print(f'v10剪定: {removed}ファイル削除')\n"
"assert n10 == 858, f'⚠ v10評価セットの本数が想定外: {n10} (期待858)'\n"
"n_cls = len(open('datasets/cls_indices_train.tsv').readlines())\n"
"assert n_cls == 6\n"
"print(f'v11 foa: {n11} / v10 eval foa: {n10} / classes: {n_cls}')"))

cells.append(md(
"## 7. 設定ファイル（v11学習＋val推論、v10評価5種）\n"
"\n"
"roomsフィルタは部分文字列マッチ。学習は v11 の [fold1_room1] のみ。"))
cells.append(code(
"def data_yaml(ds, train_rooms, valid_rooms, test_rooms):\n"
"    return f\"\"\"audio_type: foa\n"
"audio_feature: logmelIV\n"
"sample_rate: 24000\n"
"nfft: 1024\n"
"n_mels: 64\n"
"hoplen: 240\n"
"window: hann\n"
"\n"
"train_chunklen_sec: 10\n"
"train_hoplen_sec: 10\n"
"test_chunklen_sec: 10\n"
"test_hoplen_sec: 10\n"
"\n"
"train_dataset:\n"
"  {ds}: {train_rooms}\n"
"valid_dataset:\n"
"  {ds}: {valid_rooms}\n"
"test_dataset:\n"
"  {ds}: {test_rooms}\n"
"\"\"\"\n"
"\n"
"def exp_yaml(ds, data_name):\n"
"    return f\"\"\"# @package _global_\n"
"defaults:\n"
" - override /data: {data_name}.yaml\n"
" - override /loss: multi_accdoa.yaml\n"
" - _self_\n"
"\n"
"task_name: {ds}\n"
"\n"
"model:\n"
"  batch_size: 8\n"
"  kwargs:\n"
"    pretrained_path: ckpts/mACCDOA-HTSAT-0.567.ckpt\n"
"    audioset_pretrain: false\n"
"  optimizer:\n"
"    kwargs: {{lr: 0.0003}}\n"
"  lr_scheduler:\n"
"    kwargs: {{step_size: 60}}\n"
"\n"
"trainer:\n"
"  max_epochs: 100\n"
"  check_val_every_n_epoch: 5\n"
"\"\"\"\n"
"\n"
"# v11: 学習本体 + val推論variant\n"
"open(f'configs/data/{DATASET}.yaml', 'w').write(\n"
"    data_yaml(DATASET, '[fold1_room1]', '[fold2_room1]', '[fold3_room1]'))\n"
"open(f'configs/experiment/{DATASET}.yaml', 'w').write(exp_yaml(DATASET, DATASET))\n"
"open(f'configs/data/{DATASET}_valinfer.yaml', 'w').write(\n"
"    data_yaml(DATASET, '[fold1_room1]', '[fold2_room1]', '[fold2_room1]'))\n"
"open(f'configs/experiment/{DATASET}_valinfer.yaml', 'w').write(\n"
"    exp_yaml(DATASET, f'{DATASET}_valinfer'))\n"
"\n"
"# v10: 評価5種（学習には使わない。train/validは存在するroomを指すだけ）\n"
"EVAL_TAGS = [('scenario', '[fold2_room9]'),\n"
"             ('probe', '[fold9_room1]'),\n"
"             ('scn2', '[fold2_room4, fold2_room5, fold2_room6, fold2_room7, fold2_room8]'),\n"
"             ('v10a', '[fold8_room1]'),\n"
"             ('halluc', '[fold2_room3]')]\n"
"for tag, rooms in EVAL_TAGS:\n"
"    open(f'configs/data/{EVAL_DS}_{tag}.yaml', 'w').write(\n"
"        data_yaml(EVAL_DS, '[fold2_room1]', '[fold2_room1]', rooms))\n"
"    open(f'configs/experiment/{EVAL_DS}_{tag}.yaml', 'w').write(\n"
"        exp_yaml(EVAL_DS, f'{EVAL_DS}_{tag}'))\n"
"print('wrote configs (v11 train/valinfer + v10 eval x5)')"))

cells.append(md(
"## 8. 前処理（**空ラベル許容版**、両データセット。初回のみ40〜50分規模）\n"
"\n"
"v11の純静穏582本はラベルCSVが0バイト（設計上の教師）。素のpreprocは\n"
"EmptyDataErrorで停止するため、読み込み2関数を0バイト判定でラップした別プロセスで\n"
"実行する（pin済みPSELDNets本体は非接触。設計= v11設計書§1.5、ローカル実証済み）。\n"
"**v10.2までの除外セルはv11では使わない。**"))
cells.append(code(
"from pathlib import Path\n"
"\n"
"WRAPPER = r'''# _preproc_emptyok.py (auto-generated)\n"
"import os\n"
"import runpy\n"
"import sys\n"
"\n"
"sys.path.insert(0, \"src\")\n"
"import pandas as pd\n"
"import utils.data_utilities as du\n"
"import preproc.preprocess as pp\n"
"\n"
"_load, _read = du.load_output_format_file, pd.read_csv\n"
"\n"
"\n"
"def _is_empty_file(path):\n"
"    try:\n"
"        return os.path.getsize(path) == 0\n"
"    except (TypeError, OSError, ValueError):\n"
"        return False\n"
"\n"
"\n"
"def load_ok(path, *a, **kw):\n"
"    if _is_empty_file(path):\n"
"        return {99: []}          # 最終フレームのみ・イベント0件（ゼロ行列になる）\n"
"    return _load(path, *a, **kw)\n"
"\n"
"\n"
"def read_ok(path, *a, **kw):\n"
"    if _is_empty_file(path):\n"
"        return pd.DataFrame([[99, 0, 0, 0, 0]])   # num_frames=100 算出専用の番兵\n"
"    return _read(path, *a, **kw)\n"
"\n"
"\n"
"du.load_output_format_file = load_ok\n"
"pp.load_output_format_file = load_ok   # from-import名ごと差し替え\n"
"pd.read_csv = read_ok                  # 別プロセス内のみのグローバル差し替え\n"
"\n"
"sys.argv = [\"src/preproc.py\"] + sys.argv[1:]\n"
"runpy.run_path(\"src/preproc.py\", run_name=\"__main__\")\n"
"'''\n"
"Path('_preproc_emptyok.py').write_text(WRAPPER, encoding='utf-8')\n"
"\n"
"# 学習プロセス用ラッパ: data.py(BaseDataset)がvalの正解ラベルを生CSVから読む経路も\n"
"# 空CSVで落ちる（data/components/data.py:98、2026-07-28にColab実地で発見）。\n"
"# 同じ0バイト判定で {99: []}=イベント0件のGT を返す（val指標への偽イベント混入なし）\n"
"WRAPPER_TRAIN = r'''# _train_emptyok.py (auto-generated)\n"
"import os\n"
"import runpy\n"
"import sys\n"
"\n"
"sys.path.insert(0, \"src\")\n"
"import utils.data_utilities as du\n"
"import data.components.data as dcd\n"
"\n"
"_load = du.load_output_format_file\n"
"\n"
"\n"
"def _is_empty_file(path):\n"
"    try:\n"
"        return os.path.getsize(path) == 0\n"
"    except (TypeError, OSError, ValueError):\n"
"        return False\n"
"\n"
"\n"
"def load_ok(path, *a, **kw):\n"
"    if _is_empty_file(path):\n"
"        return {99: []}   # イベント0件のGT\n"
"    return _load(path, *a, **kw)\n"
"\n"
"\n"
"du.load_output_format_file = load_ok\n"
"dcd.load_output_format_file = load_ok   # data.pyはfrom-importなので名前ごと差し替え\n"
"\n"
"sys.argv = [\"src/train.py\"] + sys.argv[1:]\n"
"runpy.run_path(\"src/train.py\", run_name=\"__main__\")\n"
"'''\n"
"Path('_train_emptyok.py').write_text(WRAPPER_TRAIN, encoding='utf-8')\n"
"print('wrote _preproc_emptyok.py / _train_emptyok.py')\n"
"\n"
"for ds in [DATASET, EVAL_DS]:\n"
"    idx = f'_hdf5/data/24000fs/wav/dev/{ds}_10sChunklen_10sHoplen_train.csv'\n"
"    if os.path.exists(idx):\n"
"        print(f'{ds}: 前処理済み')\n"
"    else:\n"
"        !python _preproc_emptyok.py dataset={ds}\n"
"    import pandas as pd\n"
"    print(ds, 'index rows:', len(pd.read_csv(idx, header=None)))"))

cells.append(md("## 9. 最終チェック"))
cells.append(code(
"IDX = f'_hdf5/data/24000fs/wav/dev/{DATASET}_10sChunklen_10sHoplen_train.csv'\n"
"checks = [\n"
"    ('ckpts/mACCDOA-HTSAT-0.567.ckpt',     '事前学習チェックポイント'),\n"
"    ('datasets/cls_indices_train.tsv',      'クラス辞書 TSV (6クラス)'),\n"
"    (f'datasets/{DATASET}/foa',             'v11 FOA (7199=7200-検品FAIL1)'),\n"
"    (f'datasets/{EVAL_DS}/foa',             'v10評価 FOA (858)'),\n"
"    (f'configs/experiment/{DATASET}.yaml',  '実験設定'),\n"
"    (IDX,                                   'v11前処理インデックス'),\n"
"]\n"
"for path, name in checks:\n"
"    ok = os.path.exists(path) and (not os.path.isdir(path) or len(os.listdir(path)) > 0)\n"
"    print(f'  [{\"OK\" if ok else \"NG\"}] {name}')"))

cells.append(md(
"## 10. 学習（T4 で約7時間 / 100epoch）\n"
"\n"
"- **切れても再実行すれば last.ckpt から自動再開**（Drive永続化）→ 寝る前にこのセルまで実行\n"
"- val は fold2_room1 のみ。test(fold3) はここでは一切使わない\n"
"- **データを変えて学習し直すときは必ず EXP_NAME を変えること**"))
cells.append(code(
"LAST = f'{DRIVE_LOGS}/{DATASET}/runs/{EXP_NAME}/checkpoints/last.ckpt'\n"
"resume = f'ckpt_path={LAST}' if os.path.exists(LAST) else ''\n"
"print('resume:', resume or '(new run)')\n"
"\n"
"!python _train_emptyok.py experiment={DATASET} \\\n"
"    experiment_name={EXP_NAME} \\\n"
"    paths.log_dir={DRIVE_LOGS} \\\n"
"    {resume}"))

cells.append(md("## 11. 学習曲線（val 抜粋）"))
cells.append(code(
"import re\n"
"\n"
"log_path = f'{DRIVE_LOGS}/{DATASET}/runs/{EXP_NAME}/train.log'\n"
"lines = [l for l in open(log_path, errors='ignore')\n"
"         if 'val/macro' in l or 'train: loss_all' in l]\n"
"print(f'--- {log_path} ---')\n"
"for l in lines:\n"
"    print(re.sub(r'\\x1b\\[[0-9;]*m', '', l).rstrip())\n"
"\n"
"vals = [l for l in lines if 'val/macro' in l]\n"
"if vals:\n"
"    print('\\n=== 最終 val/macro ===')\n"
"    print(re.sub(r'\\x1b\\[[0-9;]*m', '', vals[-1]).strip())"))

cells.append(md(
"## 12. 推論（6セット一括: val=v11 / 交差点・プローブ・6シナリオ・交通量・幻覚=v10をv11ckptで）\n"
"\n"
"予測CSVをセットごとに1本へ連結してDriveに保存 → ローカルの解剖・採点が読む\n"
"（ファイル名はv10.2と同じ規約 `infer_{EXP_NAME}_{tag}_all.csv`）。"))
cells.append(code(
"import glob, os\n"
"cands = sorted(glob.glob('/content/drive/MyDrive/PSELDNets_logs*'))\n"
"cands += sorted(glob.glob('/content/drive/.shortcut-targets-by-id/*/PSELDNets_logs'))\n"
"best_ckpt = None\n"
"for c in cands:\n"
"    hits = sorted(glob.glob(f'{c}/{DATASET}/runs/{EXP_NAME}/checkpoints/epoch_*.ckpt'))\n"
"    if hits:\n"
"        DRIVE_LOGS = c\n"
"        best_ckpt = hits[-1]\n"
"        break\n"
"print('best_ckpt =', best_ckpt)\n"
"assert best_ckpt, '⚠ ckptが見えません（学習が終わっていますか）'\n"
"\n"
"JOBS = [(f'{DATASET}_valinfer', 'val'),\n"
"        (f'{EVAL_DS}_scenario', 'scenario'),\n"
"        (f'{EVAL_DS}_probe', 'probe'),\n"
"        (f'{EVAL_DS}_scn2', 'scn2'),\n"
"        (f'{EVAL_DS}_v10a', 'v10a'),\n"
"        (f'{EVAL_DS}_halluc', 'halluc')]\n"
"for experiment, short in JOBS:\n"
"    exp = f'infer_{EXP_NAME}_{short}'\n"
"    !python src/infer.py experiment={experiment} \\\n"
"        mode=test \\\n"
"        ckpt_path=\"{best_ckpt}\" \\\n"
"        model.kwargs.pretrained_path=null \\\n"
"        experiment_name={exp} \\\n"
"        paths.log_dir={DRIVE_LOGS}\n"
"    task = experiment.rsplit('_', 1)[0]\n"
"    sub = f'{DRIVE_LOGS}/{task}/runs/{exp}/submissions'\n"
"    out_lines = []\n"
"    for p in sorted(glob.glob(f'{sub}/*.csv')):\n"
"        stem = os.path.basename(p)[:-4]\n"
"        for line in open(p):\n"
"            if line.strip():\n"
"                out_lines.append(f'{stem},{line.strip()}')\n"
"    out = f'{DRIVE_DATA}/{exp}_all.csv'\n"
"    open(out, 'w').write('\\n'.join(out_lines))\n"
"    print('wrote', out, len(out_lines), 'lines')"))

cells.append(md(
"## 13. このあと（ローカル側）\n"
"\n"
"1. 6つの `infer_..._all.csv` をローカルへ → 解剖・通知層採点・シナリオ採点・v10a同時検出・幻覚検定\n"
"2. fold3(test)は全分析が固まった後に**最終1回だけ**（先生と相談してから）\n"
"3. デコーダ閾値掃引（threshold_unify 5/10/15/20°）は**統合前トラック別出力の保存セル**を別途用意して実施（v11設計書§4.5）\n"
"4. 因果推論・batch=1ベンチ・傾き耐性のセルはv10.2用をパス替えで再利用可"))

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU"},
      "cells": cells}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)

# 検証: JSON妥当性＋全コードセルの構文チェック（{}をf-str扱いしない素の構文）
import ast
nb2 = json.loads(OUT.read_text(encoding="utf-8"))
n_code = 0
for c in nb2["cells"]:
    if c["cell_type"] != "code":
        continue
    n_code += 1
    src = c["source"]
    # Colabマジック(!/%%)行はコメント化して構文チェック
    lines = []
    for ln in src.split("\n"):
        if ln.lstrip().startswith("!") or ln.lstrip().startswith("%"):
            lines.append("pass  #" + ln)
        elif ln.rstrip().endswith("\\") and (ln.lstrip().startswith("!") or
                                             (lines and lines[-1].lstrip().startswith("pass  #"))):
            lines.append("pass  #" + ln)
        else:
            lines.append(ln)
    try:
        ast.parse("\n".join(lines))
    except SyntaxError as e:
        print("SYNTAX?", e, "\n---\n", src[:300])
print(f"cells: {len(nb2['cells'])} (code {n_code}) — JSON/構文チェック完了")
