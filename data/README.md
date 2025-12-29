# Data Directory

This directory contains data assets for the CFD-NN project.

## Neural Network Models

**All neural network turbulence model weights are stored in `data/models/`.**

See `data/models/README.md` for:
- Available trained models
- How to use models with `--nn_preset`
- Training your own models
- Model architecture details

### Quick Start

```bash
# Use trained TBNN model for channel flow
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout

# Use example/demo models (random weights - for testing only)
./channel --model nn_tbnn --nn_preset example_tbnn
./channel --model nn_mlp --nn_preset example_scalar_nut
```

**Note:** NN models now require explicit selection via `--nn_preset` or `--weights/--scaling`. There are no default/fallback weights.