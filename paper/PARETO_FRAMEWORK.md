# Pareto Plot Framework

## Definition

**x-axis**: Computational cost = ms/step on H200 GPU (from timing data)
**y-axis**: Accuracy = error metric vs DNS reference

## Error Metrics by Case

### Duct (primary Pareto case)
- **Accuracy metric**: L2 error of U(y) profile at z=0 (midplane) vs Vinuesa DNS
- Alternative: |U_b - U_b_DNS| / U_b_DNS (bulk velocity error)
- Alternative: |v|_max / U_b (secondary flow strength, higher = closer to DNS)

### Hills
- **Accuracy metric**: |x_reattach - x_reattach_DNS| / L (reattachment point error)
- Alternative: L2 error of Cf(x) vs Krank DNS

### Cylinder
- **Accuracy metric**: |Cd - Cd_DNS| / Cd_DNS (drag coefficient error)
- Secondary: |St - St_DNS| / St_DNS

### Sphere
- **Accuracy metric**: |Cd - Cd_DNS| / Cd_DNS

## Cost Data (from H200 timing, duct 884K cells)

| Model | ms/step | Relative to SST |
|-------|---------|-----------------|
| None | 15.0 | 0.98× |
| Baseline | 15.3 | 1.00× |
| k-omega | 15.1 | 0.99× |
| SST | 15.3 | 1.00× |
| GEP | 15.2 | 1.00× |
| RSM-SSG | 15.8 | 1.03× |
| MLP | 19.7 | 1.29× |
| MLP-med | 31.4 | 2.06× |
| MLP-large | ~90* | ~5.9× |
| TBNN-small | 39.1 | 2.56× |
| TBNN | 64.3 | 4.20× |
| TBNN-large | ~140* | ~9.2× |
| TBRF-1t | ~16* | ~1.05× |
| TBRF-5t | ~17* | ~1.11× |
| TBRF-10t | ~18* | ~1.18× |

*Estimated from prior runs

## Pareto Plot Observations (expected)

Three clusters will be visible:
1. **Lower-left**: Classical RANS (SST, k-omega, RSM, EARSM) — low cost, moderate accuracy
2. **Upper-right**: MLP — moderate cost, WORSE accuracy (strictly dominated)
3. **Right-middle**: TBNN — higher cost, better accuracy on duct (Pareto frontier)

SST is the "knee" of the classical cluster.
TBNN-small is the Pareto-optimal NN closure (best accuracy/cost ratio).
MLP is below and to the right of SST — dominated on both axes.

## Implementation Plan

Generate plot using Python matplotlib:
- One main plot: duct case (THE differentiating case)
- Supplementary: one plot per case
- Color code: blue=classical, red=scalar NN, green=tensor NN
- Mark SST and DNS reference lines
- Annotate dominated region (MLP)
