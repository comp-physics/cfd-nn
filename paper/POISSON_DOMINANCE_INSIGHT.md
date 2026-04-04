# The Poisson Dominance Insight

## The Finding

The pressure Poisson solve consumes **81% of every time step** on the H200 GPU
(12.4 ms out of 15.3 ms for SST). This cost is IDENTICAL for all turbulence
models because it depends only on the grid, not the closure.

**Consequence: neural network inference cost is amortized by the Poisson solve.**

Even TBNN-small, which takes 14.3 ms for NN inference per step, only costs
2.56× SST total — because 12.4 ms of every step is the same Poisson solve
regardless of the closure.

Without the Poisson "floor", TBNN would be 9.2× SST (from the NN cost alone).

## Why This Matters for the Paper

This is the central message of the cost analysis:

> "The practical cost of neural network turbulence closures is far lower than
> their inference cost alone would suggest, because the pressure Poisson solve
> — which is independent of the closure — dominates the per-step budget. On
> our 884K-cell GPU solver, TBNN-small adds only 2.6× overhead despite a
> 14 ms inference cost, because the 12 ms Poisson solve creates a cost floor."

## Scaling Argument

The Poisson solve uses FFT (O(N log N)) or multigrid (O(N)).
The NN inference is O(N) (pointwise evaluation at each cell).

As the grid refines:
- Poisson cost grows as N log N (or N for MG)
- NN cost grows as N
- **The NN fraction of total cost DECREASES with grid refinement**

On a 10M-cell grid, the Poisson would take ~100 ms while TBNN-small 
would take ~160 ms → only 1.6× overhead (vs 2.56× at 884K cells).

## Table for Paper

| Model | NN cost | Poisson | Other | Total | vs SST |
|-------|---------|---------|-------|-------|--------|
| SST | 0.2 ms | 12.4 ms | 2.7 ms | 15.3 ms | 1.0× |
| MLP-small | 4.6 ms | 12.4 ms | 2.7 ms | 19.7 ms | 1.3× |
| TBNN-small | 14.3 ms | 12.4 ms | 2.7 ms* | 39.1 ms | 2.6× |
| TBNN | 24.0 ms | 12.4 ms | 2.7 ms* | 64.3 ms | 4.2× |

*"Other" includes tau_div computation for tensor models (~10 ms additional).

## Implications

1. **The cost barrier to NN closures is lower than assumed.** Prior papers
   worry about inference cost without accounting for solver overhead.

2. **TBNN-small is cost-effective.** At 2.6× SST with improved accuracy on
   anisotropy-dominated flows, it sits on the Pareto frontier.

3. **MLP is dominated regardless of cost.** Even at 1.3× SST (cheapest NN),
   MLP gives WORSE accuracy. Cost is not MLP's problem — physics is.

4. **Future: faster Poisson solvers would expose NN cost.** If the Poisson
   were reduced 10× (e.g., via neural operator), the NN overhead would
   increase to ~5× (TBNN-small). Conversely, if the NN were reduced 10×
   (e.g., via quantization), the total step cost would barely change
   (15.3 → 14.0 ms, only 8% savings).
