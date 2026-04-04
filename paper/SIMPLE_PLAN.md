# GPU SIMPLE Implementation Plan

## Architecture

```
SIMPLE outer loop (500-2000 iterations):
│
├─ 1. Momentum solve: A * u* = b(u_old, p)
│     └─ BiCGSTAB + MG-preconditioner (5-10 Krylov iters)
│        └─ MG V-cycle: Chebyshev-Jacobi smoother (1 cycle per BiCGSTAB iter)
│
├─ 2. Compute a_P = diag(A)
│
├─ 3. Pressure solve: div(1/a_P · grad(p')) = div(u*)
│     └─ Variable-coefficient MG (ALREADY IMPLEMENTED)
│
├─ 4. Correction: u = u* - 1/a_P · grad(p'), p += alpha_p · p'
│     └─ (ALREADY IMPLEMENTED)
│
├─ 5. Under-relax: u = alpha_u * u + (1-alpha_u) * u_old
│
├─ 6. Update turbulence (k, omega, nu_t)
│
└─ 7. Check convergence
```

## What exists vs what's needed

| Component | Status | Effort |
|-----------|--------|--------|
| SIMPLE outer loop | EXISTS (solver_time_simple.cpp) | 0 |
| RB-GS momentum kernel | EXISTS (solver_time_kernels_simple.cpp) | 0 |
| BiCGSTAB scratch buffers | EXISTS (solver.hpp, 7 arrays) | 0 |
| BiCGSTAB algorithm | PARTIAL (was in earlier code, removed) | ~50 LOC |
| MG Poisson solver | EXISTS (poisson_solver_multigrid.cpp) | 0 |
| Variable-coeff MG | EXISTS at level 0 (for pressure) | 0 |
| MG for momentum | NEEDS MODIFICATION | ~100 LOC |
| Pressure correction | EXISTS (correct_velocity_simple) | 0 |
| a_P computation | EXISTS (simple_compute_aP_2d/3d) | 0 |
| Patankar under-relaxation | EXISTS (in RB-GS kernel) | 0 |

## Key implementation: MG for momentum

The momentum matrix A has the same sparsity pattern as the Poisson Laplacian
(5-point in 2D, 7-point in 3D) but with:
- Variable coefficients on each face (nu_eff + upwind convection)
- An extra diagonal term (outflow convection)
- A non-symmetric contribution from upwind convection

Our MG already supports variable coefficients at level 0 via
`set_variable_coefficients()`. For momentum, we need:

1. **General stencil setter**: Instead of just D_u/D_v (diffusion-like),
   accept 5 arrays: a_W, a_E, a_S, a_N, a_P at each cell.
   
2. **Smoother**: Chebyshev-Jacobi works for any diagonally-dominant system.
   The eigenvalue bounds need updating for the momentum operator.
   
3. **Coarsening**: Galerkin (A_c = R * A * P) works for any operator.
   Already implemented for constant-coefficient coarse levels.

4. **BiCGSTAB wrapper**: Use MG as preconditioner. Outer loop:
   ```
   r = b - A*u
   for k = 1, 2, ..., max_iter:
       z = MG_Vcycle(r)  // preconditioner
       ... standard BiCGSTAB update ...
       if ||r|| < tol: break
   ```

## Estimated timeline

- Day 1: Generalize MG stencil for momentum (modify smooth_jacobi, compute_residual)
- Day 2: Implement BiCGSTAB with MG preconditioner
- Day 3: Test on laminar Poiseuille, then SST channel
- Day 4: Integrate into SIMPLE loop, test on duct
- Day 5: GPU optimization, production runs

## Alternative: Simpler approach first

Before full MG-BiCGSTAB, try:
1. **Jacobi-preconditioned BiCGSTAB** (no MG, just diagonal scaling)
   - ~50 LOC, uses existing BiCGSTAB buffers
   - May need 20-50 iterations (vs 5-10 with MG preconditioner)
   - Still much better than bare RB-GS

2. **Polynomial (Chebyshev) preconditioner**
   - ~30 LOC, use eigenvalue estimates from MG
   - 3rd-order Chebyshev polynomial as preconditioner
   - Embarrassingly parallel on GPU

## Performance target

OpenFOAM simpleFoam on duct 884K cells:
- 5000 SIMPLE iterations, 3.34s/iter (CPU) = 16702s total
- Momentum: 2 GS sweeps per iteration
- Pressure: GAMG, ~10 V-cycles per iteration

Our GPU SIMPLE target:
- 1000-2000 SIMPLE iterations (SIMPLEC converges faster)
- Momentum: BiCGSTAB + MG, ~5-10 Krylov iters, ~2ms per solve
- Pressure: varcoeff MG, ~5 V-cycles, ~5ms per solve
- Total per iteration: ~10ms
- Total: 1500 iters × 10ms = 15s (vs OpenFOAM 16702s → 1000× speedup)
