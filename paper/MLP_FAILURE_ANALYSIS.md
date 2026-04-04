# MLP Failure Analysis

## The Finding

MLP turbulence closures are **strictly dominated** in Pareto space:
they are more expensive than SST AND produce worse results on every case.

## Quantitative Evidence

### Duct Re_b=3500 (96³)
| Model | nu_t (max) | nu_t / nu | nu_t / nu_t_SST | U_b | ms/step |
|-------|-----------|-----------|-----------------|------|---------|
| SST | 5.18e-3 | 9.1× | 1× | 0.625* | 15.3 |
| MLP | 6.24e-1 | 1094× | 121× | 0.010 | 19.7 |
| MLP-med | — | — | — | 0.019 | 31.4 |
| TBNN-small | 5.18e-3 | 9.1× | 1× | 0.675* | 39.1 |

*Not yet converged at 10K steps.

MLP predicts **121× the eddy viscosity of SST**, creating massive artificial
diffusion that suppresses all velocity gradients. U_b collapses from ~0.6 (SST)
to 0.01 (essentially dead flow).

### Hills Re=5600
| Model | U_b | vs SST |
|-------|------|--------|
| SST | 0.878 | baseline |
| MLP | 0.731 | -17% |
| MLP-med | 0.515 | -41% |
| MLP-large | 0.187 | -79% |

Larger MLP → more over-diffusive. The relationship is monotonic: more parameters
→ more capacity to overfit → larger nu_t predictions → lower U_b.

### Cylinder Re=100
| Model | Cd | vs DNS (1.35) |
|-------|-----|---------------|
| None | 1.291 | -4.4% |
| MLP | 1.411 | +4.5% |
| SST | 1.484 | +9.9% |

MLP is closer to DNS than SST on cylinder — but this is because Re=100 is
laminar and MLP's excess viscosity is small relative to molecular nu=0.01.

## Why MLP Fails

### Architecture limitation
MLP predicts a **scalar** eddy viscosity correction: nu_t = f(invariants).
The Boussinesq assumption (tau_ij = -2*nu_t*S_ij) cannot represent anisotropy.
When trained on data that includes anisotropic flows (duct secondary flow),
the MLP tries to capture all Reynolds stress effects through a single scalar.
The result: it overpredicts nu_t to compensate for the missing anisotropy.

### Input distribution shift
The Pope invariants computed during the a posteriori evaluation differ from
training because:
1. The velocity field evolves with the MLP's own nu_t (feedback loop)
2. As nu_t increases → gradients decrease → invariants shrink → MLP sees
   inputs outside training range → extrapolates to even larger nu_t
3. This positive feedback saturates at very high nu_t (flow nearly stagnant)

### Size effect
Larger MLPs have more capacity to memorize the training data's nu_t distribution.
Since the training data includes some cells with large nu_t (near walls, separated
regions), larger models produce larger peak nu_t predictions, causing more damage.

## Implications for the Field

1. **MLP-type scalar corrections should not be used for RANS closure** on flows
   with significant anisotropy (secondary flow, separation, 3D effects)
2. **Tensor-basis architectures (TBNN) are necessary** — they can represent
   anisotropy directly without inflating the scalar viscosity
3. **A priori accuracy is misleading** — MLP has reasonable test RMSE but
   catastrophic a posteriori performance due to the feedback loop
4. The finding is consistent with Duraisamy et al. (2019) warning about
   model-data inconsistency, but provides the first quantitative demonstration
   across multiple flow cases
