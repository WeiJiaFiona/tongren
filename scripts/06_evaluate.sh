#!/usr/bin/bash
#SBATCH -J evaluate
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 120:00:00
#SBATCH -o evaluate.out
#SBATCH -e evaluate.err

set -euo pipefail

echo "Running on node: ${SLURM_JOB_NODELIST:-local}"
nvidia-smi || true
echo "Starting at: $(date)"

PROJECT_ROOT="${TONGREN_PROJECT_ROOT:-/public_bme/home/jiawei2022/tongren_bme_transition}"
if [[ ! -d "$PROJECT_ROOT/code" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tongren_m1
set -u

mkdir -p logs

export TONGREN_PROJECT_ROOT="$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/code" \
python -u "$PROJECT_ROOT/code/src/train/evaluate.py"

echo "Finished at: $(date)"
