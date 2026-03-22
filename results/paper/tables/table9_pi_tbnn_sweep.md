## Table 9: PI-TBNN Beta Sweep

| Beta | L2 Reg (alpha) | Val RMSE(b) | Epochs | vs TBNN (0.0845) |
|---|---|---|---|---|
| 0 (TBNN) | 0 | 0.0845 | 694 | baseline |
| 0.001 | 1e-6 | 0.0852 | 729 | +0.8% |
| 0.01 | 1e-6 | 0.0909 | 676 | +7.6% |
| 0.1 (L2 bug) | 0.01 | 0.1215 | 152 | +43.8% |
| 1.0 (L2 bug) | 0.01 | 0.1203 | 151 | +42.4% |

*The L2 bug (alpha=0.01) caused ~825x larger regularization than MSE, dominating the loss.*

