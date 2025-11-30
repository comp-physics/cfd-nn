#!/bin/bash
# Convenience script to activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Ready to train! Try:"
echo "  python scripts/train_tbnn_mcconkey.py --epochs 10"
