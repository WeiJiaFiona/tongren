#!/usr/bin/bash
#SBATCH -J final_ablation
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 120:00:00
#SBATCH -o final_ablation.out
#SBATCH -e final_ablation.err

set -euo pipefail

echo "Running on node: $SLURM_JOB_NODELIST"
nvidia-smi
echo "Starting at: $(date)"

cd /public_bme/home/jiawei2022/tongren_bme_transition
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tongren_m1
mkdir -p logs

echo "Final ablation must run only after train/validation model selection is frozen."

echo "Finished at: $(date)"
