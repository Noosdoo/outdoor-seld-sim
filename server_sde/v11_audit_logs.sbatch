#!/bin/bash
#SBATCH -J auditlog
#SBATCH -p a100_3g
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --qos=low
#SBATCH -o auditlog-%j.out
# Sol第8回再監査・条件3対応: T1-T6を実行しコミットSHA付き生ログをリポジトリに保存。
# あわせてW0/run3/w3のtrain.logとjob317標準出力を回収し、
# v12 core通知をv3.3で再採点する（manifest基準・Sol条件1/2/4込み）。
set -ex
PS=~/research/PSELDNet/PSELDNets
E2E=~/research/outdoor_seld_e2e
LOGS=$E2E/out/audit_logs
mkdir -p $LOGS

# 実行時リポジトリ状態の記録（Sol C1: 実行時SHAをログへ）
SHA=$(git -C $E2E rev-parse HEAD)
git -C $E2E status --porcelain > $LOGS/repo_status_job${SLURM_JOB_ID}.txt || true

cd $PS
cp $E2E/server_sde/test_sde_units.py $E2E/server_sde/test_sde_units2.py \
   $E2E/server_sde/test_sde_units3.py .
{
  echo "e2e repo SHA: $SHA"
  echo "date: $(date -Iseconds)  job: ${SLURM_JOB_ID}"
  echo "== T1-T3 (test_sde_units.py) =="
  .venv/bin/python test_sde_units.py
  echo "== T4-T5+旧T6 (test_sde_units2.py) =="
  .venv/bin/python test_sde_units2.py
  echo "== T6実データ版 (test_sde_units3.py) =="
  .venv/bin/python test_sde_units3.py
} 2>&1 | tee $LOGS/tests_T1-T6_job${SLURM_JOB_ID}.log

# 生ログの回収（Sol C2/C3: W0の0.072等の一次証拠をリポジトリへ）
cp $E2E/server_sde/v11sdew0-317.out $LOGS/ || true
cp $HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_w0/train.log \
   $LOGS/w0_train.log
cp $HOME/PSELDNets_logs/outdoor_siren_v11/runs/outdoor_siren_v11_sde_run3/train.log \
   $LOGS/run3_train.log
cp $HOME/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v12_sde_w3/train.log \
   $LOGS/v12_w3_train.log

# v12 core通知 v3.3（w3予測）
cd $E2E
PYTHONPATH=scripts:src $PS/.venv/bin/python scripts/_v12_notify_server33.py \
  $HOME/PSELDNets_logs/outdoor_siren_v12/runs/infer_v12_sde_w3_val/val_all.csv \
  $E2E/out/v12_notify_w3_v33
echo AUDIT_LOGS_DONE
