#!/usr/bin/env bash
#
# Local training helper (real McConkey dataset) for CFD-NN.
# - Creates a Python venv in scratch (so $HOME quota stays small)
# - Downloads McConkey dataset to scratch using existing ~/.kaggle/kaggle.json
# - Trains either TBNN or MLP for a selected case
#
# Examples:
#   bash scripts/train_realdata_local.sh --case periodic_hills --model tbnn --epochs 50 --device cpu
#   bash scripts/train_realdata_local.sh --case channel --model mlp --epochs 100 --device cpu
#
# Notes:
# - Dataset goes to:   /storage/home/hcoda1/6/sbryngelson3/scratch/mcconkey_data
# - Venv goes to:      /storage/home/hcoda1/6/sbryngelson3/scratch/cfd-nn-py/venv
# - Model output goes to data/models/<model>_<case> (override with --output)
#

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_realdata_local.sh [options]

Options:
  --case CASE         Flow case: periodic_hills | channel | square_duct (default: periodic_hills)
  --model MODEL       Model: tbnn | mlp (default: tbnn)
  --epochs N          Training epochs (default: 50)
  --device DEV        cpu | cuda | mps | auto (default: cpu)
  --batch_size N      Batch size (default: 512)
  --lr LR             Learning rate (default: 1e-3)
  --output DIR        Output directory for exported C++ weights (default: data/models/<model>_<case>)
  --data_dir DIR      Dataset root directory (default: /storage/home/hcoda1/6/sbryngelson3/scratch/mcconkey_data)
  --scratch_base DIR  Scratch base for venv/caches (default: /storage/home/hcoda1/6/sbryngelson3/scratch/cfd-nn-py)
  --skip_download     Do not download dataset (assume it already exists)
  --force_download    Re-download + unzip dataset into --data_dir
  --help              Show this help

EOF
}

CASE="periodic_hills"
MODEL="tbnn"
EPOCHS="50"
DEVICE="cpu"
BATCH_SIZE="512"
LR="1e-3"

SCRATCH_BASE_DEFAULT="/storage/home/hcoda1/6/sbryngelson3/scratch/cfd-nn-py"
DATA_DIR_DEFAULT="/storage/home/hcoda1/6/sbryngelson3/scratch/data-repo/mcconkey_data_processed"

SCRATCH_BASE="$SCRATCH_BASE_DEFAULT"
DATA_DIR="$DATA_DIR_DEFAULT"
OUTPUT_DIR=""
SKIP_DOWNLOAD="0"
FORCE_DOWNLOAD="0"

module load cuda

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --output) OUTPUT_DIR="$2"; shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    --scratch_base) SCRATCH_BASE="$2"; shift 2;;
    --skip_download) SKIP_DOWNLOAD="1"; shift 1;;
    --force_download) FORCE_DOWNLOAD="1"; shift 1;;
    --help|-h) usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="data/models/${MODEL}_${CASE}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================="
echo "CFD-NN local training (real data)"
echo "========================================="
echo "Repo root:     ${REPO_ROOT}"
echo "Case:          ${CASE}"
echo "Model:         ${MODEL}"
echo "Epochs:        ${EPOCHS}"
echo "Device:        ${DEVICE}"
echo "Batch size:    ${BATCH_SIZE}"
echo "LR:            ${LR}"
echo "Dataset dir:   ${DATA_DIR}"
echo "Scratch base:  ${SCRATCH_BASE}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "========================================="

VENV_DIR="${SCRATCH_BASE}/venv"
PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
TMPDIR_DIR="${SCRATCH_BASE}/tmp"

mkdir -p "${SCRATCH_BASE}" "${PIP_CACHE_DIR}" "${TMPDIR_DIR}" "${DATA_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export TMPDIR="${TMPDIR_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo ""
  echo "Creating venv in: ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi



source "${VENV_DIR}/bin/activate"

echo ""
echo "Python: $(command -v python)"
python --version

echo ""
echo "Installing Python dependencies (from requirements.txt)..."
pip install -U pip
pip install -r "${REPO_ROOT}/requirements.txt"

echo ""
echo "Dependency sanity checks:"
python - <<'PY'
import importlib.metadata as m
import torch
print("  torch:", torch.__version__)
try:
    print("  kaggle:", m.version("kaggle"))
except Exception as e:
    print("  kaggle: (version unavailable)", e)
print("  cuda available:", torch.cuda.is_available())
PY

if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
  if [[ "${FORCE_DOWNLOAD}" == "1" ]]; then
    echo ""
    echo "Forcing dataset re-download into: ${DATA_DIR}"
    rm -rf "${DATA_DIR}"
    mkdir -p "${DATA_DIR}"
  fi

  # Determine whether expected case structure exists already
  if [[ -f "${DATA_DIR}/${CASE}/train/data.npz" && -f "${DATA_DIR}/${CASE}/val/data.npz" && -f "${DATA_DIR}/${CASE}/test/data.npz" ]]; then
    echo ""
    echo "Dataset appears present for case ${CASE}; skipping download."
  else
    echo ""
    echo "Downloading McConkey dataset to: ${DATA_DIR}"
    echo "Using Kaggle credentials from: ~/.kaggle/kaggle.json"

    if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
      echo "ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json" >&2
      echo "Please set up Kaggle credentials, then re-run." >&2
      exit 1
    fi

    # Kaggle CLI (preferred). `python -m kaggle` is not supported in some versions.
    if ! command -v kaggle >/dev/null 2>&1; then
      echo "ERROR: kaggle CLI not found on PATH after installing requirements.txt" >&2
      echo "Try: pip install kaggle" >&2
      exit 1
    fi
    export KAGGLE_CONFIG_DIR="${HOME}/.kaggle"
    ZIP_DIR="${DATA_DIR}/_kaggle_zip"
    mkdir -p "${ZIP_DIR}"

    echo "Downloading zip to: ${ZIP_DIR}"
    kaggle datasets download \
      -d ryleymcconkey/ml-turbulence-dataset \
      -p "${ZIP_DIR}"

    ZIP_FILE="$(ls -1t "${ZIP_DIR}"/*.zip 2>/dev/null | head -n 1 || true)"
    if [[ -z "${ZIP_FILE}" ]]; then
      echo "ERROR: Kaggle download completed but no .zip was found in ${ZIP_DIR}" >&2
      exit 1
    fi

    echo "Unzipping ${ZIP_FILE} -> ${DATA_DIR}"
    if command -v unzip >/dev/null 2>&1; then
      unzip -q "${ZIP_FILE}" -d "${DATA_DIR}"
    else
      # Fallback: pure-python unzip
      python - <<PY
import zipfile
zip_path = r"${ZIP_FILE}"
out_dir = r"${DATA_DIR}"
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(out_dir)
print("Extracted", zip_path, "to", out_dir)
PY
    fi
  fi
else
  echo ""
  echo "Skipping dataset download (per --skip_download)."
fi

echo ""
echo "Checking dataset structure for case '${CASE}'..."
if [[ -f "${DATA_DIR}/${CASE}/train/data.npz" && -f "${DATA_DIR}/${CASE}/val/data.npz" && -f "${DATA_DIR}/${CASE}/test/data.npz" ]]; then
  echo "  OK: Found expected NPZ files under ${DATA_DIR}/${CASE}/{train,val,test}/data.npz"
else
  echo "ERROR: Expected files not found:" >&2
  echo "  ${DATA_DIR}/${CASE}/train/data.npz" >&2
  echo "  ${DATA_DIR}/${CASE}/val/data.npz" >&2
  echo "  ${DATA_DIR}/${CASE}/test/data.npz" >&2
  echo "" >&2
  echo "This usually means the dataset unzipped into a nested folder." >&2
  echo "Try locating where ${CASE}/train/data.npz landed and pass that parent as --data_dir." >&2
  exit 1
fi

mkdir -p "$(dirname "${REPO_ROOT}/${OUTPUT_DIR}")"

echo ""
echo "Starting training..."
cd "${REPO_ROOT}"

if [[ "${MODEL}" == "tbnn" ]]; then
  python scripts/train_tbnn_mcconkey.py \
    --data_dir "${DATA_DIR}" \
    --case "${CASE}" \
    --output "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden 64 64 64 \
    --device "${DEVICE}"
elif [[ "${MODEL}" == "mlp" ]]; then
  python scripts/train_mlp_mcconkey.py \
    --data_dir "${DATA_DIR}" \
    --case "${CASE}" \
    --output "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden 32 32 \
    --device "${DEVICE}"
else
  echo "ERROR: Unknown --model '${MODEL}'. Use 'tbnn' or 'mlp'." >&2
  exit 2
fi

echo ""
echo "========================================="
echo "DONE"
echo "Model exported to: ${REPO_ROOT}/${OUTPUT_DIR}"
echo "========================================="


