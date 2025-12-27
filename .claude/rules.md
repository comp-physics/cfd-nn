# Claude Code Rules for CFD-NN Project

## Core Principles

### Minimal Scope & Token Usage
- Goal: Minimal token usage and minimal code churn
- Default scope: <= 3 files and <= 200 LOC touched
- If more needed: STOP and ask with proposed file list + why (<= 4 bullets)
- Output: Default to unified diff only. If text needed: <= 4 bullets, 1 sentence each
- Do NOT: Scan entire repository, create new markdown/doc files unless explicitly asked
- SLURM jobs: Always submit using the embers QOS

### Behavior Guidelines
- When unsure, ask 1 targeted question instead of exploring widely
- Keep existing style/conventions; no large renames/reformats unless asked
- Do NOT scan the entire repository by default

## Debugging Protocol

### Lean Debugging: Packets Only
- Never request full logs. Use packets only.
- Do not proceed with diagnosis until the relevant packet output is provided.

### Result Packet Format (ask for only this)
1. Exact command
2. Where it ran (OS/arch + compiler/runtime versions if relevant)
3. Exit status (code/signal)
4. First error block + <= 40 lines context
5. Tail: stderr <= 120 lines, stdout <= 80 lines
6. File:line references

### Preferred Commands
- GitHub CI: Run ./tools/ci_packet.sh <runid> via terminal tool (if <runid> missing, run ./tools/ci_packet.sh)
- SLURM: Run ./tools/spkt <JOBID> via terminal tool (if <JOBID> missing, run ./tools/spkt)
- Local: Run ./tools/local_packet.sh <command...> via terminal tool

If terminal execution is unavailable or blocked by approvals, ask for approval or ask user to run the single command.

### Output Format
- Patch/diff first
- If more info needed: ask for ONE additional item only

## Testing Requirements

### Comprehensive Physics Validation Suite

The project has a gold-standard validation suite that rigorously verifies the Navier-Stokes solver:

#### Core Tests (tests/test_physics_validation.cpp) - ~2 min on GPU:

1. Poiseuille Flow (Analytical)
   - Tests: Viscous terms, pressure gradient, parabolic profile
   - Pass criterion: <5% L2 error (expected for 64x128 grid)
   - Validates: Solver produces correct velocity profiles

2. Divergence-Free Constraint
   - Tests: ∇·u = 0 (incompressibility)
   - Pass criterion: Machine precision (0% divergence!)
   - Validates: Projection method works perfectly

3. Momentum Balance (Integral)
   - Tests: ∫ f_body dV = ∫ τ_wall dA
   - Pass criterion: <10% imbalance
   - Validates: Global momentum conservation

4. Channel Symmetry
   - Tests: u(y) = u(-y) about centerline
   - Pass criterion: Machine precision
   - Validates: BCs and discretization correctness

5. Cross-Model Consistency
   - Tests: All turbulence models agree in laminar limit
   - Pass criterion: <5% difference
   - Validates: Model implementations correct

6. Sanity Checks
   - Tests: No NaN/Inf, realizability (ν_t >= 0)
   - Validates: Numerical stability

#### Advanced Validation (tests/test_tg_validation.cpp) - ~30 sec:

Taylor-Green Vortex Test
- Initial: u=sin(x)cos(y), v=-cos(x)sin(y) (divergence-free)
- Theory: Energy decays as KE(t) = KE(0)·exp(-4νt)
- Pass criterion: <5% error in energy decay over 100 timesteps
- Validates: Viscous terms, time integration, periodic BCs
- Result: Currently achieving 0.5% error (excellent!)

What These Tests Prove:
- Solver correctly solves incompressible Navier-Stokes
- All conservation laws satisfied
- 2nd-order spatial accuracy
- Stable time integration
- GPU produces identical results to CPU

### CRITICAL: Pre-Push Testing Protocol

BEFORE EVERY PUSH TO REPOSITORY:

1. Run the pre-CI test script:
   ./test_before_ci.sh
   This script MUST pass before pushing. It tests:
   - Debug build (like CI does)
   - Release build (like CI does)
   - All unit tests in both configurations
   - Checks for compiler warnings

2. For GPU-related changes, ALSO run:
   ./test_before_ci_gpu.sh
   This runs the COMPLETE GPU CI test suite locally:
   - All unit tests on GPU (including physics validation tests)
   - Fast turbulence model validation (Baseline, GEP, SST, k-omega, EARSM)
   - Periodic Hills complex geometry tests
   - CPU/GPU consistency validation
   - Takes 10-15 minutes and ensures GPU CI will pass

3. Why this matters:
   - CPU CI runs 4 configurations: (Ubuntu + macOS) × (Debug + Release)
   - GPU CI runs comprehensive physics validation on actual GPU hardware
   - Local development often only tests one configuration
   - Debug builds have different behavior than Release
   - Tests that pass locally in Release may fail in CI's Debug build
   - GPU tests that work on small problems may fail on large-scale runs

### Test Development Guidelines

1. No Platform-Specific Tolerances
   - DO NOT add #ifdef __APPLE__ or similar platform checks
   - DO NOT relax tolerances for different compilers
   - If a test is numerically sensitive, fix the root cause, don't hide it

2. Build-Type Independence
   - Tests should pass in BOTH Debug and Release builds
   - If different behavior is needed, it's a code smell - investigate why

3. Numerical Robustness
   - Avoid overly strict floating-point comparisons (e.g., == for doubles)
   - Use appropriate tolerances based on algorithm, not platform
   - For iterative solvers: check residual convergence, not exact iteration count

4. Test Isolation
   - Each test should be independent
   - Don't rely on execution order
   - Clean up any state/files created during tests

### Common CI Failure Patterns to Avoid

Bad: Tests that only pass in Release
```cpp
// This might fail in Debug due to different optimization
assert(iterations < 100);  // Iteration count is optimization-dependent!
```

Good: Tests that work in both Debug and Release
```cpp
// Check convergence, not iteration count
assert(residual < tolerance);
assert(std::isfinite(result));
```

Bad: Missing body force setup
```cpp
RANSSolver solver(mesh, config);
// Forgot to set body force - flow won't develop!
solver.step();
```

Good: Complete initialization
```cpp
RANSSolver solver(mesh, config);
solver.set_body_force(-config.dp_dx, 0.0);
solver.initialize_uniform(0.1, 0.0);
solver.step();
```

Bad: Exact floating-point comparisons
```cpp
assert(max_u == 1.0);  // Will fail due to numerical precision
```

Good: Tolerance-based comparisons
```cpp
assert(std::abs(max_u - 1.0) < 1e-10);
```

## Code Quality Standards

1. No Compiler Warnings
   - Fix ALL warnings before pushing
   - Use [[maybe_unused]] for intentionally unused variables in assertions
   - Don't suppress warnings with flags

2. Const Correctness
   - Use const for read-only references
   - Mark methods const if they don't modify state

3. RAII for Resource Management
   - Use smart pointers or RAII wrappers
   - No manual new/delete or malloc/free
   - GPU buffers use OmpDeviceBuffer wrapper

4. Error Handling
   - Check return values
   - Use exceptions for error conditions
   - Provide informative error messages

## Performance Guidelines

1. Data Movement (CPU-GPU)
   - Minimize CPU↔GPU transfers
   - Keep frequently accessed data on GPU
   - Batch operations when possible

2. Algorithmic Choices
   - Profile before optimizing
   - Document complexity assumptions
   - Use appropriate data structures

## Documentation Requirements

1. Public API Documentation
   - Every public function needs a comment
   - Explain parameters, return values, side effects
   - Include usage examples for complex functions

2. Code Comments
   - Explain WHY, not WHAT
   - Document numerical algorithms
   - Note any non-obvious optimizations

3. README Updates
   - Keep command-line options up to date
   - Document new features
   - Update examples when behavior changes

## Workflow Best Practices

### Before Starting Work
1. Pull latest changes from main
2. Create feature branch
3. Run ./test_before_ci.sh to verify starting state

### During Development
1. Build and test frequently
2. Test in Debug mode during development
3. Fix warnings immediately (don't accumulate)

### Before Committing
1. Run all tests: cd build && ctest
2. Check for warnings: Review build output
3. Verify changes don't break existing functionality

### Before Pushing
1. MANDATORY: Run ./test_before_ci.sh
2. Only push if script passes completely
3. If it fails, fix issues before pushing

### After Pushing
1. Monitor CI status on GitHub
2. If CI fails, fix immediately (don't push more changes on top)
3. Learn from CI failures to prevent similar issues

## Git Commit Guidelines

1. Descriptive Messages
   - First line: Brief summary (<50 chars)
   - Blank line
   - Detailed explanation if needed

2. Atomic Commits
   - One logical change per commit
   - Don't mix refactoring with feature additions
   - Each commit should leave code in working state

3. Test Coverage
   - Add tests for new features
   - Update tests when changing behavior
   - Don't commit broken tests

## File Organization

- /src/ - Implementation files
- /include/ - Header files (public API)
- /app/ - Application entry points
- /tests/ - Unit and integration tests
- /data/models/ - Neural network model files
- /output/ - Generated output (VTK, data files)
- /scripts/ - Utility scripts
- /docs/ - Documentation

## Common Gotchas

1. Solver Initialization
   - Always call set_body_force() for driven flows
   - Initialize velocity field before solving
   - Set turbulence model before first step

2. Boundary Conditions
   - Ensure BCs are consistent across solver components
   - Periodic BCs require special handling in Poisson solver

3. GPU Offload
   - Data must be explicitly uploaded to GPU
   - Check USE_GPU_OFFLOAD is defined
   - Verify omp_get_num_devices() > 0 at runtime

4. VTK Output
   - Use solve_steady_with_snapshots() for automatic output
   - Specify num_snapshots in config
   - Files are numbered sequentially + final

## Quick Reference

Run tests locally (like CI):
./test_before_ci.sh

Build for debugging:
mkdir -p build_debug && cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j4
ctest --output-on-failure

Build for performance:
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

Check for warnings:
cd build
make clean
make 2>&1 | grep -i "warning:"

Run single test:
cd build
./test_solver

Run with verbose output:
cd build
ctest --output-on-failure --verbose

## Emergency: CI is Failing

1. Check which configuration failed:
   - Ubuntu Debug? Ubuntu Release?
   - macOS Debug? macOS Release?

2. Reproduce locally:
   For Ubuntu Debug failure:
   ./test_before_ci.sh
   Check Debug output specifically

3. Common fixes:
   - Missing initialization (body force, etc.)
   - Unused variables in assertions (use [[maybe_unused]])
   - Platform paths (use relative paths)
   - Missing #include that worked locally

4. Don't:
   - Push "fixes" without running test_before_ci.sh
   - Add platform-specific workarounds
   - Disable failing tests
   - Relax tolerances to hide issues

Remember: CI failures are usually legitimate bugs or test issues, not platform quirks.
Fix the root cause, don't work around it!


