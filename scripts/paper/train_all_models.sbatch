#!/bin/bash
#SBATCH -J nn-paper-train-models
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 02:00:00
#SBATCH -o results/paper/training/slurm_%j.out
#SBATCH -e results/paper/training/slurm_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/paper/training

echo "=========================================="
echo "  Training All NN Turbulence Models"
echo "=========================================="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Use pytorch container for Python+torch+sklearn+pandas
module load pytorch/2.1.0

# Install deps inside the container's Python
python3 -m pip install --user pandas scikit-learn 2>&1 | tail -3

# Verify
python3 -c "
import torch, pandas, sklearn, numpy
print(f'torch={torch.__version__}, pandas={pandas.__version__}, sklearn={sklearn.__version__}, numpy={numpy.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Training on McConkey dataset ==="
echo ""

# Run unified training script
python3 scripts/paper/train_all_models.py \
    --data_dir mcconkey_data \
    --output_dir data/models \
    --device cuda

echo ""
echo "=========================================="
echo "  Training Complete"
echo "=========================================="
