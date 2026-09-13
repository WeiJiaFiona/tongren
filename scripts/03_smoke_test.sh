#!/usr/bin/bash
#SBATCH -J smoke_test
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 120:00:00
#SBATCH -o smoke_test.out
#SBATCH -e smoke_test.err

set -euo pipefail

echo "Running on node: $SLURM_JOB_NODELIST"
nvidia-smi
echo "Starting at: $(date)"

cd /public_bme/home/jiawei2022/tongren_bme_transition
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tongren_m1
mkdir -p logs

PYTHONPATH=/public_bme/home/jiawei2022/tongren_bme_transition/code \
python code/src/train/check_environment.py
PYTHONPATH=/public_bme/home/jiawei2022/tongren_bme_transition/code \
python code/src/train/token_length_stats.py
PYTHONPATH=/public_bme/home/jiawei2022/tongren_bme_transition/code \
pytest code/src/tests/test_model_smoke.py
PYTHONPATH=/public_bme/home/jiawei2022/tongren_bme_transition/code \
python code/src/train/baichuan_smoke_test.py

echo "Finished at: $(date)"
