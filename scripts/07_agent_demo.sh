#!/usr/bin/bash
#SBATCH -J agent_demo
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 120:00:00
#SBATCH -o agent_demo.out
#SBATCH -e agent_demo.err

set -euo pipefail

echo "Running on node: $SLURM_JOB_NODELIST"
nvidia-smi
echo "Starting at: $(date)"

cd /public_bme/home/jiawei2022/tongren_bme_transition
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tongren_m1
mkdir -p logs

echo "Agent demo requires a trained, calibrated checkpoint and frozen_agent_config.json."

echo "Finished at: $(date)"
