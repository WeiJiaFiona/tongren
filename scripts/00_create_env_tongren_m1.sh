#!/usr/bin/bash
#SBATCH -J create_env_tongren_m1
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 120:00:00
#SBATCH -o create_env_tongren_m1.out
#SBATCH -e create_env_tongren_m1.err

set -euo pipefail

echo "Running on node: $SLURM_JOB_NODELIST"
nvidia-smi
echo "Starting at: $(date)"

cd /public_bme/home/jiawei2022/tongren_bme_transition
mkdir -p logs

conda create -y -n tongren_m1 python=3.11 pip
conda run -n tongren_m1 python -m pip install --upgrade pip
conda run -n tongren_m1 python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda run -n tongren_m1 python -m pip install \
  transformers==4.48.1 peft bitsandbytes accelerate safetensors \
  scikit-learn numpy pandas pyyaml tqdm pyarrow sentencepiece ninja packaging psutil

# Pin flash-attn to a torch-2.5-compatible release and build locally.
# Prebuilt wheels may require a newer system GLIBC than the cluster provides.
conda run -n tongren_m1 env \
  FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
  python -m pip install flash-attn==2.7.4.post1 --no-build-isolation --no-binary flash-attn

echo "Finished at: $(date)"
