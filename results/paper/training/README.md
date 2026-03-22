# Training Logs

Final training run logs for the paper's NN turbulence models. See `docs/paper/training_methodology.md` for full details.

## full_5266424.out

SLURM job 5266424 (2026-03-21). Final training of all 5 models on McConkey dataset with 1000 max epochs, cosine annealing LR, patience 150. GPU: NVIDIA L40S. Results:

| Model | Val RMSE(b) | Epochs |
|-------|------------|--------|
| TBRF | 0.0637 | n/a |
| TBNN | 0.0845 | 694 (early stop) |
| MLP-Large | 0.1045 | 344 (early stop) |
| MLP | 0.1096 | 1000 (max) |
| PI-TBNN | 0.1209 | 152 (early stop, L2 bug) |

Note: The PI-TBNN result in this run had an L2 regularization bug (alpha=0.01, ~825x larger than MSE). The corrected results are in the sweep below.

## pi_sweep_5300440.out

SLURM job 5300440 (2026-03-21). PI-TBNN beta sweep with L2 bug fixed (alpha=1e-6). Swept beta = {0.001, 0.01}. Hit 2-hour wall time before beta=0.1 completed.

| Beta | Val RMSE(b) | Epochs |
|------|------------|--------|
| 0.001 | 0.0852 | 729 (early stop) |
| 0.01 | 0.0909 | 676 (early stop) |

Key finding: beta=0.001 matches unconstrained TBNN (0.0845); beta=0.01 is 7.6% worse. Realizability penalty is either negligible or harmful.
