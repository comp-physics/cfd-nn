#!/bin/bash
#SBATCH -J nn-paper-eval-apriori
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-l40s
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH --exclude=atl1-1-01-004-31-0
#SBATCH -t 00:30:00
#SBATCH -o results/paper/apriori/slurm_%j.out
#SBATCH -e results/paper/apriori/slurm_%j.err

set -euo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
mkdir -p results/paper/apriori

echo "=========================================="
echo "  A Priori Evaluation — NN Turbulence Models"
echo "=========================================="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

module load pytorch/2.1.0

python3 -m pip install --user scikit-learn 2>&1 | tail -3

python3 -c "
import torch, sklearn, numpy
print(f'torch={torch.__version__}, sklearn={sklearn.__version__}, numpy={numpy.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Evaluating all models ==="
echo ""

python3 -u scripts/paper/evaluate_apriori.py \
    --data_dir mcconkey_data \
    --model_dir data/models \
    --output_dir results/paper/apriori \
    --device cuda

echo ""
echo "=========================================="
echo "  Evaluation Complete"
echo "=========================================="
