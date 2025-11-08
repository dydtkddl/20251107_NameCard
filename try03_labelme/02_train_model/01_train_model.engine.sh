#!/usr/bin/bash

#SBATCH -J train-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/dydtkddhkdwk/20251107_NameCard/try02/01_makedataset/logs/slurm-%A.out
#SBATCH -e /data/dydtkddhkdwk/20251107_NameCard/try02/01_makedataset/logs/slurm-%A.err

# =======================
# 1. 환경 설정
# =======================
set -e  # 에러 시 종료
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1  # 실시간 출력
export WORKDIR=$(pwd)
# 현재 작업 디렉토리를 기준으로 설정
WORKDIR=$(pwd)
cd "$WORKDIR"
mkdir -p logs

date
echo "Working Directory: $WORKDIR"
nvidia-smi | head -n 20

# =======================
# 2. 실행
# =======================
python "$WORKDIR/01_train_model.engine.py" \
    2>&1 | tee "$WORKDIR/logs/train_$SLURM_JOB_ID.log"
echo "=== Job Completed ==="
date

