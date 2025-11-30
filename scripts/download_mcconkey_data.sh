#!/bin/bash
#
# Download McConkey et al. (2021) turbulence modeling dataset
#
# Reference:
#   McConkey, R. et al. "A curated dataset for data-driven turbulence modelling."
#   Scientific Data 8, 255 (2021).
#   DOI: 10.1038/s41597-021-01034-2
#
# Dataset URL: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset
#

set -e

DATASET_NAME="ryleymcconkey/ml-turbulence-dataset"
OUTPUT_DIR="mcconkey_data"

echo "=========================================="
echo "McConkey Dataset Downloader"
echo "=========================================="
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: Kaggle CLI not found."
    echo ""
    echo "Please install it with:"
    echo "  pip install kaggle"
    echo ""
    echo "Then setup your API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Save kaggle.json to ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Check if API credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API credentials not found."
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Save kaggle.json to ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

echo "Downloading dataset: $DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Download dataset
kaggle datasets download -d "$DATASET_NAME" -p "$OUTPUT_DIR" --unzip

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Dataset location: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Train TBNN model:"
echo "     python scripts/train_tbnn_mcconkey.py --data_dir $OUTPUT_DIR --case periodic_hills"
echo ""
echo "  2. Train MLP model:"
echo "     python scripts/train_mlp_mcconkey.py --data_dir $OUTPUT_DIR --case periodic_hills"
echo ""
echo "  3. See docs/TRAINING_GUIDE.md for full instructions"
echo ""


