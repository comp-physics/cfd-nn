# Quick Start: Train Your Own Turbulence Model

Train a neural network turbulence model in **30 minutes** (with dataset) or **5 minutes** (with dummy data for testing).

## Prerequisites

```bash
# Python 3.7+ with pip
pip install -r requirements.txt
```

## Option A: Train on Real Data (30 min)

### Step 1: Download Dataset (10 min)

```bash
# Setup Kaggle API (one-time)
pip install kaggle
# Get API key from https://www.kaggle.com/settings
# Save to ~/.kaggle/kaggle.json

# Download McConkey dataset (~500 MB)
bash scripts/download_mcconkey_data.sh
```

### Step 2: Train TBNN (15 min on CPU)

```bash
python scripts/train_tbnn_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/my_tbnn \
    --epochs 50
```

### Step 3: Run in CFD Solver (5 min)

```bash
cd build
./channel --model nn_tbnn --nn_preset my_tbnn --max_iter 10000
```

## Option B: Quick Test with Dummy Data (5 min)

If you don't have the dataset, test the pipeline with synthetic data:

```bash
# Train on dummy data (fast - just testing)
python scripts/train_tbnn_mcconkey.py \
    --data_dir /nonexistent/path \
    --output data/models/test_tbnn \
    --epochs 10

# Run in solver
cd build
./channel --model nn_tbnn --nn_preset test_tbnn --max_iter 1000
```

**Note**: Dummy data trains quickly but results are not physically meaningful. Use real data for actual research.

## Train Different Models

### TBNN (Tensor Basis Neural Network)

Best for: Accurate Reynolds stress prediction

```bash
python scripts/train_tbnn_mcconkey.py \
    --case channel \
    --output data/models/tbnn_channel \
    --hidden 64 64 64 \
    --epochs 100
```

### MLP (Simpler, Faster)

Best for: Quick prototyping, faster inference

```bash
python scripts/train_mlp_mcconkey.py \
    --case channel \
    --output data/models/mlp_channel \
    --hidden 32 32 \
    --epochs 100
```

## Compare All Models

```bash
# Run all available models and compare
python scripts/run_all_models.py --case channel --plot
```

This will:
- Run solver with baseline, GEP, and all trained NN models
- Generate comparison plots
- Create performance report

## Training Options

```bash
# Train on GPU (if available)
python scripts/train_tbnn_mcconkey.py \
    --device cuda \
    --batch_size 512 \
    --epochs 200

# Train on different flow case
python scripts/train_tbnn_mcconkey.py \
    --case channel \
    --output data/models/tbnn_channel

# Bigger network for complex flows
python scripts/train_tbnn_mcconkey.py \
    --hidden 128 128 128 \
    --epochs 200
```

## Troubleshooting

### "Dataset not found"
- Script will generate dummy data automatically for testing
- For real training, download with `bash scripts/download_mcconkey_data.sh`

### "Training loss is NaN"
- Reduce learning rate: `--lr 1e-4`
- Reduce batch size: `--batch_size 128`

### "Model runs but results are bad"
- Trained on dummy data? --> Download real dataset
- Need more epochs: `--epochs 200`
- Try different architecture: `--hidden 64 64 64`

## What You Get

After training, you'll have:

```
data/models/my_tbnn/
├── layer0_W.txt        # Network weights
├── layer0_b.txt
├── layer1_W.txt
├── ...
├── input_means.txt     # Feature normalization
├── input_stds.txt
└── metadata.json       # Model documentation
```

Use in solver: `./channel --model nn_tbnn --nn_preset my_tbnn`

## Full Documentation

- **Complete training guide**: `docs/TRAINING_GUIDE.md`
- **Dataset information**: `docs/DATASET_INFO.md`
- **Main project docs**: `README.md`

## Expected Results

On channel flow with real McConkey data:

| Model | Training Time | Inference Speed | Accuracy vs DNS |
|-------|---------------|-----------------|-----------------|
| TBNN  | 15-30 min    | ~2 ms/cell      | ~15% error     |
| MLP   | 5-10 min     | ~0.4 ms/cell    | ~25% error     |

(CPU times on typical laptop)

## Next Steps

1. **Validate**: Compare against DNS data
2. **Generalize**: Train on one case, test on another
3. **Optimize**: Try different architectures
4. **Publish**: Document your results!

---

**Need help?** See full documentation in `docs/TRAINING_GUIDE.md`

