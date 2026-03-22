## Table 1: Model Summary

| Model | Architecture | Parameters | FLOPs/cell | Weight Size | Deployable |
|---|---|---|---|---|---|
| Baseline (mixing length) | Algebraic | 0 | ~2 | 0 | Yes |
| k-omega | 2-eq transport | 0 | ~50 | 0 | Yes |
| SST k-omega | 2-eq transport | 0 | ~80 | 0 | Yes |
| GEP | Algebraic | 0 | ~40 | 0 | Yes |
| EARSM-WJ | SST + tensor algebra | 0 | ~120 | 0 | Yes |
| EARSM-GS | SST + tensor algebra | 0 | ~120 | 0 | Yes |
| EARSM-Pope | SST + tensor algebra | 0 | ~120 | 0 | Yes |
| MLP | 5->32->32->1 (Tanh) | 1,249 | 1,249 | 56 KB | Yes |
| MLP-Large | 5->128^4->1 (Tanh) | 50,049 | 50,049 | 896 KB | Yes |
| TBNN | 5->64^3->10 (Tanh) | 9,354 | 10,254 | 196 KB | Yes |
| PI-TBNN | 5->64^3->10 (Tanh) | 9,354 | 10,254 | 196 KB | Yes |
| TBRF (1 tree) | 10 forests x 1 tree | 282,902 nodes | ~2,200 | 5.7 MB | Experimental |
| TBRF (5 trees) | 10 forests x 5 trees | 1,432,612 nodes | ~10,200 | 29 MB | Experimental |
| TBRF (10 trees) | 10 forests x 10 trees | 2,797,130 nodes | ~20,200 | 56 MB | Experimental |
| TBRF (200 trees) | 10 forests x 200 trees | 55,051,026 nodes | ~400,200 | 3.3 GB | No |

*FLOPs/cell for NN includes feature computation (~900 FLOPs for invariants + tensor basis for TBNN/TBRF).*
*TBRF FLOPs estimated as: 900 (features) + n_coefficients x n_trees x avg_depth x 2 (comparison + branch).*
*For MLP, FLOPs = weights count (each weight = 1 multiply + 1 add) + activations.*

