#!/usr/bin/bash
#SBATCH -J curate_and_split
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 120:00:00
#SBATCH -o curate_and_split.out
#SBATCH -e curate_and_split.err

set -euo pipefail

echo "Running on node: $SLURM_JOB_NODELIST"
nvidia-smi
echo "Starting at: $(date)"

cd /public_bme/home/jiawei2022/tongren_bme_transition
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tongren_m1
mkdir -p logs

PYTHONPATH=/public_bme/home/jiawei2022/tongren_bme_transition/code \
python code/src/data/build_patient_state.py
PYTHONPATH=/public_bme/home/jiawei2022/tongren_bme_transition/code \
python code/src/data/split_dataset.py

echo "Finished at: $(date)"
