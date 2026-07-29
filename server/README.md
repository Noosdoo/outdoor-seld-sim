# IS計算サーバーでの v11 実行（Colab からの移植）

正= `colab/PSELDNets_outdoor_siren_v11_Colab.ipynb`（run1）。本フォルダはその**サーバー移植版**。
run1（Colab T4）とハイパラ同一の **run2** として学習し、環境間再現性の確認を兼ねる。

## 前提（済んでいるもの）

- サーバーに `~/research` を clone 済み（submodule PSELDNets = 8092a14、ノートブックの pin と同一）
- `~/PSELDNets_data/` に zip 4本 + 事前学習 ckpt を転送済み
- Drive の代わり: データ= `~/PSELDNets_data`、ログ= `~/PSELDNets_logs`（チェックポイントもここ）

## 実行手順（ログインノードで）

```bash
# 1. venv 構築（初回のみ、~10分）
bash ~/research/outdoor_seld_e2e/server/setup_env.sh

# 2. configs・空ラベルラッパ生成 ＋ データ展開検証（初回のみ）
cd ~/research/PSELDNet/PSELDNets
.venv/bin/python ~/research/outdoor_seld_e2e/server/make_runtime_files.py
.venv/bin/python ~/research/outdoor_seld_e2e/server/prepare_data.py

# 3+4. 前処理→学習を依存連結で一括投入（2026-07-30 run2 の実績構成）
cd ~/research/PSELDNet/PSELDNets
PREP=$(sbatch --parsable -J v11prep -p a100_3g --gres=gpu:3g.40gb:1 --qos=low \
    ~/research/outdoor_seld_e2e/server/preproc_v11.sbatch)
sbatch -J v11run2 -p pro_6000 --gres=gpu:pro_6000:1 --qos=low \
    --dependency=afterok:$PREP \
    ~/research/outdoor_seld_e2e/server/train_v11_run2.sbatch
# 実績: prep 1分20秒（preprocはラベルh5+index生成のみ）/ 学習 100ep=純計算23.5分
# （中断1回→自動requeueを含め壁時計約1h）。a100 で流す場合は -p a100 --gres=gpu:a100:1

# 監視
squeue -u "$USER"
tail -f ~/v11run2-<ジョブID>.out
```

## Colab との対応表

| Colab | サーバー |
| --- | --- |
| Drive `PSELDNets_data`（zipキャッシュ） | `~/PSELDNets_data/` |
| Drive `PSELDNets_logs`（ckpt永続化） | `~/PSELDNets_logs/` |
| セル4 pip install | `setup_env.sh`（uv venv, torch 2.8.0+cu128） |
| セル5 ckpt DL+SHA照合 | `prepare_data.py`（scp済みファイルを配置+照合） |
| セル6 zip展開+マニフェスト照合 | `prepare_data.py` |
| セル7-8 configs+ラッパ生成 | `make_runtime_files.py` |
| セル8後半 preproc | `preproc_v11.sbatch` |
| セル10 学習 | `train_v11_run2.sbatch` |
| セッション切れ→再実行で resume | low QoS 中断→自動再実行で resume（同じ仕組み） |

## 決定事項・注意

- **EXP_NAME= `outdoor_siren_v11_run2`**（run1 は Colab T4。データを変えたら必ず改名）
- torch は run1 と完全一致ではない（Colab のプリインストール版 vs 2.8.0+cu128。
  当初 2.13.0+cu132 を試したが cu132 索引に torchaudio の x86 wheel が無く断念）。
  lightning==2.2.1 等の学習系ピンはノートブックと同一。run2 の位置づけは
  「環境・シード違いの頑健性チェック」なので、val 指標が run1 と大差ないことを確認する
- **fold3（test）で指標を算出せず、モデル選択にも使用しない**（データとしての展開・前処理は
  行う。最終1回の評価は先生と相談後。ノートブック セル13 と同じ）
- 推論6セット一括（セル12 相当）は学習完了後に別途 sbatch 化する
- Windows罠（index csv の `\` 正規化）はサーバー（Linux）では不要
- ⚠️ **preproc のスキップ判定は index csv の存在のみ**（Colab版と同じ制約、Sol監査#3）:
  データ・config・PSELDNets版を変えたら `_hdf5/` を手で消してから再実行すること
- 中断時の自動再実行はクラスタ既定の requeue（利用ガイド3.3）+ `#SBATCH --requeue` 明示。
  run2 で中断1回→自動再実行を実測済み。last.ckpt 生成後の中断なら自動 resume が効く
- **実績と証跡**= md/results/v11run2_server_2026-07-30.md、
  **監査対応**= md/audit/Sol自動監査01_server移行run2_2026-07-30.md
