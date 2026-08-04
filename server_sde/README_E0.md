# E0 実行手順（VPN復帰後に上から順に）

事前登録ゲートは md/design/E可聴外帯域_計画_2026-08-04.md の「E0: 事前登録」節。
**実行前にゲート節が宣言済みであることを確認**（2026-08-05 00:05 宣言済み）。

## 0. 投入前チェック（5分、ここだけ目視確認が要る）

```bash
ssh is-server
cd ~/research/PSELDNet/PSELDNets
ls datasets/outdoor_siren_v11/foa | head -3          # fold2_*.flac の存在
ls configs/experiment/ | grep valinfer               # 実験config名の確認
grep -n "dataset" configs/experiment/outdoor_siren_v11_valinfer.yaml
ls $HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/checkpoints/
df -h ~                                              # 空き（派生3セット ≈ val flac×3 ≈ 数GB）
```

- valinfer yaml内のdataset参照が `outdoor_siren_v11` の文字列でない場合、
  e0_infer.sbatch の sed 置換パターンを実際のキーに合わせて直す
- preprocess.py の呼び出し形（src/preprocess.py か scripts/…か）を `ls src/ scripts/` で確認し
  e0_prep.sbatch を合わせる

## 1. スクリプト転送と投入

```bash
scp server_sde/_e0_hpf_render.py server_sde/e0_prep.sbatch server_sde/e0_infer.sbatch \
    is-server:~/research/PSELDNet/PSELDNets/
ssh is-server 'cd ~/research/PSELDNet/PSELDNets && jid=$(sbatch --parsable e0_prep.sbatch) && \
    sbatch --dependency=afterok:$jid e0_infer.sbatch && squeue -u $USER'
```

投入後は必ず1分後に `squeue` で生存確認（job194の教訓）。バックグラウンド監視:
`ssh is-server 'while squeue -u $USER | grep -q e0; do sleep 60; done; echo DONE'`

## 2. 回収

```bash
# 各条件の予測CSVを連結（run3のときと同じ concat 手順）して取得
# → out/predictions_v11sde_run3_hpf{50,100,200}/val_all.csv （新フォルダ）
```

## 3. 採点（ローカル）

- SELD系: run3 val採点と同じ経路
- 距離: scripts/_score_sde_dist.py <pred> <outdir>
- 通知: scripts/step12_notify_v3.py <pred> <outdir>
- 比較レポート: out/e0_lowfreq_probe/ に「素 vs HPF50/100/200」の
  クラス別検出率・距離MAE・SELDの表＋ゲート判定を出力

## 検品基準（_e0_hpf_render.py が自動表示）

- fc点で約-6dB以上、fc/2で-20dB級、1kHzで±0.1dB以内（ゼロ位相8次相当）
- fold2以外を触っていないこと（レンダ対象はfold2のみ）
