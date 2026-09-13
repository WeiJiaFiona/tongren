#!/usr/bin/bash
#SBATCH -J train_qlora
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 120:00:00
#SBATCH -o train_qlora.out
#SBATCH -e train_qlora.err

set -euo pipefail

echo "Running on node: $SLURM_JOB_NODELIST"
nvidia-smi
echo "Starting at: $(date)"

cd /public_bme/home/jiawei2022/tongren_bme_transition
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tongren_m1
mkdir -p logs

PYTHONPATH=/public_bme/home/jiawei2022/tongren_bme_transition/code \
python code/src/train/train_qlora.py

echo "Finished at: $(date)"
