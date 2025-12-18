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

# Check if API credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API credentials not found at ~/.kaggle/kaggle.json"
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

# Download dataset using Kaggle CLI
# Note: some Kaggle versions do NOT support `python -m kaggle` (no __main__).
export KAGGLE_CONFIG_DIR="${HOME}/.kaggle"

ZIP_DIR="${OUTPUT_DIR}/_kaggle_zip"
mkdir -p "${ZIP_DIR}"

kaggle datasets download -d "$DATASET_NAME" -p "$ZIP_DIR"

ZIP_FILE="$(ls -1t "${ZIP_DIR}"/*.zip 2>/dev/null | head -n 1 || true)"
if [[ -z "${ZIP_FILE}" ]]; then
    echo "ERROR: Kaggle download completed but no .zip was found in ${ZIP_DIR}" >&2
    exit 1
fi

echo "Unzipping ${ZIP_FILE} -> ${OUTPUT_DIR}"
unzip -q "${ZIP_FILE}" -d "${OUTPUT_DIR}"

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


