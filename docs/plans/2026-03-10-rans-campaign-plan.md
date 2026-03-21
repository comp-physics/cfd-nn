# RANS Validation Campaign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 38 RANS simulations (9 turbulence models x 4 flow cases + 2 extra NN variants for periodic hills), profile each, and sanity-check against benchmark data.

**Architecture:** Add two new IBM geometries (forward-facing step, periodic hills), two new app drivers, generate 38 config files, and dispatch as a SLURM array job. Post-processing script parses timing/profiling data and compares key quantities (Cd, reattachment length, etc.) against published benchmarks.

**Tech Stack:** C++17, OpenMP target offload (GPU), SLURM, Python 3 (post-processing), existing TimingStats profiling infrastructure.

---

## File Structure

### New files
| File | Purpose |
|------|---------|
| `include/ibm_geometry.hpp` | Add `StepBody`, `PeriodicHillBody` class declarations (modify existing) |
| `src/ibm_geometry.cpp` | Add `StepBody`, `PeriodicHillBody` implementations + factory entries (modify existing) |
| `app/main_step.cpp` | Forward-facing step driver (Inflow/Outflow BCs) |
| `app/main_hills.cpp` | Periodic hills driver (Periodic BCs, body force) |
| `tests/test_ibm_step_sdf.cpp` | Unit tests for StepBody signed distance |
| `tests/test_ibm_hills_sdf.cpp` | Unit tests for PeriodicHillBody signed distance |
| `scripts/rans_campaign/generate_configs.py` | Generate all 38 config files |
| `scripts/rans_campaign/submit_campaign.sbatch` | SLURM array job script |
| `scripts/rans_campaign/job_list.txt` | Maps array index to (executable, config) |
| `scripts/rans_campaign/analyze_campaign.py` | Parse results, timing, sanity checks |
| `examples/13_rans_campaign/*.cfg` | 38 config files (generated) |

### Modified files
| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `step`, `hills` executables + 2 tests |
| `app/main_cylinder.cpp` | Line 156: use `config.dp_dx` instead of hardcoded 0 |
| `app/main_airfoil.cpp` | Line 151: use `config.dp_dx` instead of hardcoded 0 |

---

## Flow Cases

| Case | Executable | Re | Domain (in ref lengths) | Grid | BCs (x/y/z) | Reference |
|------|-----------|-----|------------------------|------|-------------|-----------|
| Cylinder | `cylinder` | 100 (D) | [0,30]x[-10,10]x[0,pi] | 192x128x4 | Per/Per/Per + dp_dx | Zdravkovich: Cd~1.33, St~0.164 |
| NACA 0012 | `airfoil` | 1000 (c) | [-5,15]x[-8,8]x[0,1] | 256x128x4 | Per/Per/Per + dp_dx | Exp: Cd, Cl at alpha=0 |
| Fwd step | `step` | 5000 (s) | [-10,20]x[0,6]x[0,1] | 256x128x4 | In-Out/NoSlip/Per | Shih&Ho: reattach ~1-2s upstream |
| Periodic hills | `hills` | 10595 (h) | [0,9]x[0,3.035]x[0,1] | 192x96x4 | Per/NoSlip/Per + dp_dx | Breuer: reattach x/h~4.7 |

## Turbulence Models (9 per case, +2 for hills)

| ID | Model string | Config key `turb_model` | Extra config |
|----|-------------|------------------------|--------------|
| 0 | Baseline | `baseline` | — |
| 1 | KOmega | `komega` | — |
| 2 | SSTKOmega | `sst` | — |
| 3 | GEP | `gep` | — |
| 4 | EARSM_WJ | `earsm_wj` | — |
| 5 | EARSM_GS | `earsm_gs` | — |
| 6 | EARSM_Pope | `earsm_pope` | — |
| 7 | NNMLP | `nn_mlp` | `nn_preset = mlp_channel_caseholdout` |
| 8 | NNTBNN | `nn_tbnn` | `nn_preset = tbnn_channel_caseholdout` |
| 9 | NNMLP (phll) | `nn_mlp` | `nn_preset = mlp_phll_caseholdout` (hills only) |
| 10 | NNTBNN (phll) | `nn_tbnn` | `nn_preset = tbnn_phll_caseholdout` (hills only) |

**Total: 9x4 + 2 = 38 simulations**

## Benchmark Sanity Checks

| Case | Metric | Expected range | Source |
|------|--------|---------------|--------|
| Cylinder Re=100 | Cd | 1.2 - 1.5 | Zdravkovich (1997) |
| Cylinder Re=100 | St | 0.15 - 0.18 | Williamson (1996) |
| NACA 0012 Re=1000 | Cd | 0.05 - 0.15 | Exp/DNS |
| NACA 0012 Re=1000 | Cl (alpha=0) | -0.02 - 0.02 | Symmetry |
| Fwd step Re=5000 | Reattach (upstream) | 0.5s - 3.0s | Shih & Ho (1994) |
| Fwd step Re=5000 | Diverged? | false | Basic stability |
| Hills Re=10595 | Reattach x/h | 3.5 - 6.0 | Breuer et al. (2009) |
| Hills Re=10595 | Max Cf < 0 region | x/h in [0.5, 5] | Breuer et al. (2009) |

---

## Chunk 1: New IBM Geometries

### Task 1: StepBody IBM geometry

**Files:**
- Modify: `include/ibm_geometry.hpp`
- Modify: `src/ibm_geometry.cpp`
- Create: `tests/test_ibm_step_sdf.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add StepBody declaration to header**

In `include/ibm_geometry.hpp`, add after `NACABody` class:

```cpp
/// Forward-facing step: solid region for x >= x_step, y <= y_step
class StepBody : public IBMBody {
public:
    StepBody(double x_step, double y_step);
    double phi(double x, double y, double z) const override;
    std::tuple<double, double, double> normal(double x, double y, double z) const override;
    std::string name() const override;

private:
    double x_step_;
    double y_step_;
};
```

- [ ] **Step 2: Add StepBody implementation**

In `src/ibm_geometry.cpp`, add after `NACABody` implementation:

```cpp
// ============================================================
// StepBody
// ============================================================

StepBody::StepBody(double x_step, double y_step)
    : x_step_(x_step), y_step_(y_step) {}

double StepBody::phi(double x, double y, double z) const {
    double dx = x - x_step_;  // positive = past step face
    double dy = y - y_step_;  // positive = above step top

    if (dx >= 0.0 && dy <= 0.0) {
        // Inside step body
        return -std::min(dx, -dy);
    } else if (dx < 0.0 && dy <= 0.0) {
        // In front of step face
        return -dx;  // positive (outside)
    } else if (dx >= 0.0 && dy > 0.0) {
        // Above step top
        return dy;  // positive (outside)
    } else {
        // Corner region (front and above)
        return std::sqrt(dx * dx + dy * dy);
    }
}

std::tuple<double, double, double> StepBody::normal(double x, double y, double z) const {
    double dx = x - x_step_;
    double dy = y - y_step_;

    if (dx >= 0.0 && dy <= 0.0) {
        // Inside: normal toward nearest surface
        if (dx < -dy) return {-1.0, 0.0, 0.0};  // vertical face
        else return {0.0, 1.0, 0.0};              // horizontal top
    } else if (dx < 0.0 && dy <= 0.0) {
        return {-1.0, 0.0, 0.0};  // vertical face
    } else if (dx >= 0.0 && dy > 0.0) {
        return {0.0, 1.0, 0.0};  // horizontal top
    } else {
        double r = std::sqrt(dx * dx + dy * dy);
        if (r < 1e-12) return {-0.707107, 0.707107, 0.0};
        return {dx / r, dy / r, 0.0};
    }
}

std::string StepBody::name() const {
    return "ForwardFacingStep";
}
```

- [ ] **Step 3: Add StepBody to factory function**

In `create_ibm_body()` in `src/ibm_geometry.cpp`, add case:

```cpp
} else if (type == "step") {
    return std::make_unique<StepBody>(param1, param2);
```

- [ ] **Step 4: Write StepBody SDF test**

Create `tests/test_ibm_step_sdf.cpp`:

```cpp
#include "ibm_geometry.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

using namespace nncfd;

int main() {
    StepBody step(5.0, 1.0);  // step at x=5, height=1

    int failures = 0;
    auto check = [&](const char* desc, double got, double expected, double tol = 1e-10) {
        if (std::abs(got - expected) > tol) {
            std::cerr << "FAIL: " << desc << ": got " << got
                      << ", expected " << expected << "\n";
            failures++;
        }
    };

    // Inside step
    check("deep inside", step.phi(10.0, 0.5, 0.0), -0.5);
    check("inside near face", step.phi(5.1, 0.5, 0.0), -0.1);
    check("inside near top", step.phi(10.0, 0.9, 0.0), -0.1);

    // On surface
    check("on vertical face", step.phi(5.0, 0.5, 0.0), 0.0);
    check("on horizontal top", step.phi(7.0, 1.0, 0.0), 0.0);
    check("corner", step.phi(5.0, 1.0, 0.0), 0.0);

    // Outside
    check("in front of face", step.phi(4.0, 0.5, 0.0), 1.0);
    check("above top", step.phi(7.0, 2.0, 0.0), 1.0);
    check("corner region", step.phi(4.0, 2.0, 0.0), std::sqrt(2.0));

    // Name
    if (step.name() != "ForwardFacingStep") {
        std::cerr << "FAIL: name() returned " << step.name() << "\n";
        failures++;
    }

    if (failures > 0) {
        throw std::runtime_error("StepBody SDF test failed with "
                                 + std::to_string(failures) + " failures");
    }
    std::cout << "StepBody SDF: all tests passed\n";
    return 0;
}
```

- [ ] **Step 5: Register test in CMakeLists.txt**

Add to `CMakeLists.txt` test section:

```cmake
add_nncfd_test(test_ibm_step_sdf TEST_NAME_SUFFIX IBMStepSDFTest LABELS fast)
```

- [ ] **Step 6: Build and run test**

Run: `cd build && cmake .. -DBUILD_TESTS=ON && make test_ibm_step_sdf && ./test_ibm_step_sdf`
Expected: `StepBody SDF: all tests passed`

- [ ] **Step 7: Commit**

```bash
git add include/ibm_geometry.hpp src/ibm_geometry.cpp tests/test_ibm_step_sdf.cpp CMakeLists.txt
git commit -m "Add StepBody IBM geometry for forward-facing step"
```

---

### Task 2: PeriodicHillBody IBM geometry

**Files:**
- Modify: `include/ibm_geometry.hpp`
- Modify: `src/ibm_geometry.cpp`
- Create: `tests/test_ibm_hills_sdf.cpp`
- Modify: `CMakeLists.txt`

The hill profile uses the Breuer et al. (2009) / ERCOFTAC UFR 3-30 piecewise polynomial.
Original coefficients are for x in mm (h=28mm), y/h as output.
We normalize: input x/h, output y/h.

Conversion: if original coeff uses x_mm, then A_i = a_i * 28^i.

Normalized coefficients (x/h in, y/h out):

| Seg | x/h range | a0 | a1 | a2 | a3 |
|-----|-----------|------|---------|----------|----------|
| 1 | [0, 0.3214] | 1.0 | 0.0 | 0.18973 | -1.66518 |
| 2 | [0.3214, 0.5] | 0.8955 | 0.97552 | -2.84514 | 1.48159 |
| 3 | [0.5, 0.7143] | 0.9213 | 0.82068 | -2.53546 | 1.27499 |
| 4 | [0.7143, 1.071] | 1.445 | -1.37956 | 0.54488 | -0.16231 |
| 5 | [1.071, 1.429] | 0.6401 | 0.87444 | -1.55859 | 0.49216 |
| 6 | [1.429, 1.929] | 2.0139 | -2.01040 | 0.46060 | 0.02097 |

Rules:
- Segment 1: y/h = min(1.0, poly)
- Segment 6: y/h = max(0.0, poly)
- x/h in [1.929, 7.071]: y/h = 0 (flat)
- x/h in [7.071, 9.0]: y/h = hill_profile(9.0 - x/h) (mirror about centerline)
- Periodic: x wraps modulo 9h

- [ ] **Step 1: Add PeriodicHillBody declaration to header**

In `include/ibm_geometry.hpp`:

```cpp
/// Periodic hills (Breuer et al. 2009, ERCOFTAC UFR 3-30)
/// Hill profile defined by 6 piecewise cubic polynomials, periodic in x
/// with period 9h. Domain height = 3.035h.
class PeriodicHillBody : public IBMBody {
public:
    /// h = hill height (reference length)
    explicit PeriodicHillBody(double h);
    double phi(double x, double y, double z) const override;
    std::string name() const override;

    /// Hill profile height y_hill(x) for arbitrary x (periodic, in physical coords)
    double hill_height(double x) const;

private:
    double h_;
    /// Evaluate normalized profile y/h for x/h in [0, 1.929]
    double hill_profile_normalized(double xn) const;
};
```

- [ ] **Step 2: Add PeriodicHillBody implementation**

In `src/ibm_geometry.cpp`:

```cpp
// ============================================================
// PeriodicHillBody — Breuer et al. (2009) / ERCOFTAC UFR 3-30
// ============================================================

PeriodicHillBody::PeriodicHillBody(double h) : h_(h) {}

double PeriodicHillBody::hill_profile_normalized(double xn) const {
    // xn = x/h in [0, 1.929], returns y/h
    // Coefficients from ERCOFTAC, converted to normalized form (x/h input)
    if (xn <= 0.3214) {
        double y = 1.0 + 0.0 * xn + 0.18973 * xn * xn - 1.66518 * xn * xn * xn;
        return std::min(1.0, y);
    } else if (xn <= 0.5) {
        return 0.8955 + 0.97552 * xn - 2.84514 * xn * xn + 1.48159 * xn * xn * xn;
    } else if (xn <= 0.7143) {
        return 0.9213 + 0.82068 * xn - 2.53546 * xn * xn + 1.27499 * xn * xn * xn;
    } else if (xn <= 1.071) {
        return 1.445 - 1.37956 * xn + 0.54488 * xn * xn - 0.16231 * xn * xn * xn;
    } else if (xn <= 1.429) {
        return 0.6401 + 0.87444 * xn - 1.55859 * xn * xn + 0.49216 * xn * xn * xn;
    } else {
        double y = 2.0139 - 2.01040 * xn + 0.46060 * xn * xn + 0.02097 * xn * xn * xn;
        return std::max(0.0, y);
    }
}

double PeriodicHillBody::hill_height(double x) const {
    // Map x to [0, 9h) (periodic)
    double period = 9.0 * h_;
    double xp = std::fmod(x, period);
    if (xp < 0.0) xp += period;

    double xn = xp / h_;  // normalized x/h in [0, 9)

    if (xn <= 1.929) {
        return hill_profile_normalized(xn) * h_;
    } else if (xn <= 7.071) {
        return 0.0;  // flat region
    } else {
        // Mirror about x/h = 4.5: use profile at (9 - xn)
        return hill_profile_normalized(9.0 - xn) * h_;
    }
}

double PeriodicHillBody::phi(double x, double y, double z) const {
    // Signed distance approximation: vertical distance to hill surface
    // phi > 0 above hill (fluid), phi < 0 below (solid)
    return y - hill_height(x);
}

std::string PeriodicHillBody::name() const {
    return "PeriodicHills";
}
```

- [ ] **Step 3: Add PeriodicHillBody to factory function**

In `create_ibm_body()`:

```cpp
} else if (type == "periodic_hill" || type == "hills") {
    return std::make_unique<PeriodicHillBody>(param1);
```

- [ ] **Step 4: Write PeriodicHillBody SDF test**

Create `tests/test_ibm_hills_sdf.cpp`:

```cpp
#include "ibm_geometry.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

using namespace nncfd;

int main() {
    double h = 1.0;
    PeriodicHillBody hill(h);

    int failures = 0;
    auto check = [&](const char* desc, double got, double expected, double tol) {
        if (std::abs(got - expected) > tol) {
            std::cerr << "FAIL: " << desc << ": got " << got
                      << ", expected " << expected << " (tol=" << tol << ")\n";
            failures++;
        }
    };

    // Hill crest: x=0, y_hill = h
    check("crest height", hill.hill_height(0.0), h, 1e-10);

    // Flat region: x/h in [1.929, 7.071], y_hill = 0
    check("flat at x=3h", hill.hill_height(3.0), 0.0, 1e-10);
    check("flat at x=5h", hill.hill_height(5.0), 0.0, 1e-10);

    // Second hill (mirrored): crest at x=9h (=0 due to periodicity)
    check("second crest", hill.hill_height(9.0), h, 1e-10);

    // Symmetry: hill(x) == hill(9h - x) for x in [0, 4.5h]
    for (double xn = 0.0; xn <= 1.929; xn += 0.1) {
        double y1 = hill.hill_height(xn * h);
        double y2 = hill.hill_height((9.0 - xn) * h);
        check(("symmetry at x/h=" + std::to_string(xn)).c_str(), y1, y2, 1e-10);
    }

    // Continuity at segment boundaries
    double eps = 1e-6;
    double bounds[] = {0.3214, 0.5, 0.7143, 1.071, 1.429, 1.929};
    for (double b : bounds) {
        double y_left = hill.hill_height((b - eps) * h);
        double y_right = hill.hill_height((b + eps) * h);
        check(("continuity at x/h=" + std::to_string(b)).c_str(),
              y_left, y_right, 0.01);  // loose tol for segment joins
    }

    // SDF sign: above hill = positive, below = negative
    check("phi above crest", hill.phi(0.0, h + 0.5, 0.0) > 0.0, true, 0.0);
    check("phi below crest", hill.phi(0.0, h - 0.5, 0.0) < 0.0, true, 0.0);
    check("phi in flat region", hill.phi(5.0, 0.5, 0.0) > 0.0, true, 0.0);

    // Periodicity
    check("periodic phi", hill.phi(1.0, 0.5, 0.0), hill.phi(10.0, 0.5, 0.0), 1e-10);

    if (failures > 0) {
        throw std::runtime_error("PeriodicHillBody SDF test failed with "
                                 + std::to_string(failures) + " failures");
    }
    std::cout << "PeriodicHillBody SDF: all tests passed\n";
    return 0;
}
```

- [ ] **Step 5: Register test in CMakeLists.txt**

```cmake
add_nncfd_test(test_ibm_hills_sdf TEST_NAME_SUFFIX IBMHillsSDFTest LABELS fast)
```

- [ ] **Step 6: Build and run test**

Run: `cd build && cmake .. -DBUILD_TESTS=ON && make test_ibm_step_sdf test_ibm_hills_sdf && ./test_ibm_step_sdf && ./test_ibm_hills_sdf`
Expected: Both pass.

- [ ] **Step 7: Commit**

```bash
git add include/ibm_geometry.hpp src/ibm_geometry.cpp tests/test_ibm_hills_sdf.cpp CMakeLists.txt
git commit -m "Add PeriodicHillBody IBM geometry (Breuer et al. 2009)"
```

---

## Chunk 2: App Executables

### Task 3: Forward-facing step app driver

**Files:**
- Create: `app/main_step.cpp`
- Modify: `CMakeLists.txt`

The step app uses Inflow/Outflow in x, NoSlip in y, Periodic in z. The step is placed at x=0 with configurable height. Inflow is uniform velocity.

- [ ] **Step 1: Create main_step.cpp**

Create `app/main_step.cpp` following the `main_cylinder.cpp` pattern but with Inflow/Outflow BCs:

```cpp
/// Forward-facing step solver with immersed boundary method
/// Domain: [x_min, x_max] x [0, H] x [0, Lz]
/// Step at x=0 from y=0 to y=step_height
/// Inflow: uniform u = U_inf
/// Outflow: zero-gradient

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "ibm_geometry.hpp"
#include "ibm_forcing.hpp"
#include "decomposition.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace nncfd;

int main(int argc, char** argv) {
#ifdef USE_MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "[MPI] MPI_Init failed\n";
        return 1;
    }
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif

    if (mpi_rank == 0) {
        std::cout << "=== Forward-Facing Step Solver (IBM) ===\n\n";
    }

    Config config;
    config.Nx = 256;
    config.Ny = 128;
    config.Nz = 1;
    config.x_min = -10.0;
    config.x_max = 20.0;
    config.y_min = 0.0;
    config.y_max = 6.0;
    config.z_min = 0.0;
    config.z_max = 1.0;
    config.nu = 0.0002;
    config.dp_dx = 0.0;
    config.dt = 0.001;
    config.max_steps = 50000;
    config.tol = 1e-8;
    config.output_freq = 100;
    config.verbose = true;
    config.simulation_mode = SimulationMode::Unsteady;
    config.adaptive_dt = true;
    config.poisson_tol = 1e-6;
    config.poisson_max_vcycles = 20;
    config.turb_model = TurbulenceModelType::None;

    config.parse_args(argc, argv);

    double step_x = 0.0;
    double step_h = 1.0;      // Step height (reference length)
    double U_inf = 1.0;

    double Re = U_inf * step_h / config.nu;

    if (mpi_rank == 0) {
        config.print();
        std::cout << "\nStep: x=" << step_x << ", height=" << step_h << "\n";
        std::cout << "Re (step height) = " << Re << "\n";
        std::cout << "U_inf = " << U_inf << "\n\n";
    }

    try {
        std::filesystem::create_directories(config.output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create output directory: " << e.what() << "\n";
    }

    bool is3D = config.Nz > 1;

    Mesh mesh;
    if (is3D) {
        mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max,
                          config.z_min, config.z_max);
    } else {
        mesh.init_uniform(config.Nx, config.Ny,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max);
    }

    if (mpi_rank == 0) {
        std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny;
        if (is3D) std::cout << " x " << mesh.Nz;
        std::cout << " cells\n";
        std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy;
        if (is3D) std::cout << ", dz = " << mesh.dz;
        std::cout << "\n\n";
    }

#ifdef USE_MPI
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    Decomposition decomp(MPI_COMM_WORLD, config.Nz);
#else
    Decomposition decomp(config.Nz);
#endif

    auto body = std::make_shared<StepBody>(step_x, step_h);
    IBMForcing ibm(mesh, body);

    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);
    solver.set_ibm_forcing(&ibm);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Inflow;
    bc.x_hi = VelocityBC::Outflow;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    if (is3D) {
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
    }
    solver.set_velocity_bc(bc);

    // Uniform inflow profile
    double H = config.y_max - config.y_min;
    solver.set_inflow_profile([U_inf, H](double y) { return U_inf; },
                              [](double y) { return 0.0; });

    solver.set_body_force(0.0, 0.0);
    solver.print_solver_info();
    solver.initialize_uniform(U_inf, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    std::ofstream force_file;
    if (mpi_rank == 0) {
        force_file.open(config.output_dir + "forces.dat");
        if (force_file.is_open()) {
            force_file << "# step  time  Fx  Fy  residual\n";
        }
    }

    ScopedTimer total_timer("Total simulation", false);

    for (int step = 1; step <= config.max_steps; ++step) {
        if (config.adaptive_dt) {
            solver.set_dt(solver.compute_adaptive_dt());
        }
        double residual = solver.step();

#ifdef USE_GPU_OFFLOAD
        solver.sync_solution_from_gpu();
#endif
        auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), solver.current_dt());
        double time = solver.current_time();

        if (mpi_rank == 0) {
            if (force_file.is_open()) {
                force_file << step << " " << time << " "
                           << Fx << " " << Fy << " " << residual << "\n";
                if (step % config.output_freq == 0) force_file.flush();
            }
            if (step % config.output_freq == 0 || step == 1) {
                std::cout << "Step " << std::setw(6) << step
                          << "  t=" << std::fixed << std::setprecision(4) << time
                          << "  res=" << std::scientific << std::setprecision(3) << residual
                          << "  Fx=" << std::fixed << std::setprecision(4) << Fx
                          << "  Fy=" << std::setprecision(4) << Fy
                          << "\n" << std::flush;
            }
        }

        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "ERROR: Solver diverged at step " << step << "\n";
            break;
        }

        if (residual < config.tol && step > 100) {
            if (mpi_rank == 0) {
                std::cout << "Converged at step " << step
                          << " (residual=" << residual << ")\n";
            }
            break;
        }
    }

    total_timer.stop();

    if (mpi_rank == 0) {
        std::cout << "\n=== Simulation complete ===\n";
        std::cout << "Re = " << Re << "\n";
        TimingStats::instance().print_summary();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
```

- [ ] **Step 2: Register executable in CMakeLists.txt**

Add alongside other executables:

```cmake
add_executable(step app/main_step.cpp)
target_link_libraries(step nn_cfd_core)
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd build && cmake .. && make step`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add app/main_step.cpp CMakeLists.txt
git commit -m "Add forward-facing step app driver"
```

---

### Task 4: Periodic hills app driver

**Files:**
- Create: `app/main_hills.cpp`
- Modify: `CMakeLists.txt`

Periodic BCs in x/z, NoSlip in y, body force dp/dx drives the flow. The hill geometry is created via IBM.

- [ ] **Step 1: Create main_hills.cpp**

Create `app/main_hills.cpp`:

```cpp
/// Periodic hills solver with immersed boundary method
/// Breuer et al. (2009) / ERCOFTAC UFR 3-30 geometry
/// Domain: [0, 9h] x [0, 3.035h] x [0, Lz]
/// Periodic in x and z, NoSlip in y, body force drives flow

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "ibm_geometry.hpp"
#include "ibm_forcing.hpp"
#include "decomposition.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace nncfd;

int main(int argc, char** argv) {
#ifdef USE_MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "[MPI] MPI_Init failed\n";
        return 1;
    }
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif

    if (mpi_rank == 0) {
        std::cout << "=== Periodic Hills Solver (IBM) ===\n\n";
    }

    Config config;

    double h = 1.0;  // Hill height (reference length)

    config.Nx = 192;
    config.Ny = 96;
    config.Nz = 1;
    config.x_min = 0.0;
    config.x_max = 9.0 * h;
    config.y_min = 0.0;
    config.y_max = 3.035 * h;
    config.z_min = 0.0;
    config.z_max = 1.0;

    // Re_h = U_b * h / nu = 10595 => nu = U_b * h / 10595
    // With U_b = 1.0, h = 1.0: nu = 9.438e-5
    config.nu = 9.438e-5;
    config.dp_dx = -1.0;  // Will be tuned to maintain bulk velocity

    config.dt = 0.001;
    config.max_steps = 50000;
    config.tol = 1e-8;
    config.output_freq = 100;
    config.verbose = true;
    config.simulation_mode = SimulationMode::Unsteady;
    config.adaptive_dt = true;
    config.poisson_tol = 1e-6;
    config.poisson_max_vcycles = 20;
    config.turb_model = TurbulenceModelType::None;

    config.parse_args(argc, argv);

    double Re = 1.0 * h / config.nu;

    if (mpi_rank == 0) {
        config.print();
        std::cout << "\nHill height h = " << h << "\n";
        std::cout << "Re_h = " << Re << "\n";
        std::cout << "dp/dx = " << config.dp_dx << "\n\n";
    }

    try {
        std::filesystem::create_directories(config.output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create output directory: " << e.what() << "\n";
    }

    bool is3D = config.Nz > 1;

    Mesh mesh;
    if (is3D) {
        mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max,
                          config.z_min, config.z_max);
    } else {
        mesh.init_uniform(config.Nx, config.Ny,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max);
    }

    if (mpi_rank == 0) {
        std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny;
        if (is3D) std::cout << " x " << mesh.Nz;
        std::cout << " cells\n";
        std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy;
        if (is3D) std::cout << ", dz = " << mesh.dz;
        std::cout << "\n\n";
    }

#ifdef USE_MPI
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    Decomposition decomp(MPI_COMM_WORLD, config.Nz);
#else
    Decomposition decomp(config.Nz);
#endif

    auto body = std::make_shared<PeriodicHillBody>(h);
    IBMForcing ibm(mesh, body);

    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);
    solver.set_ibm_forcing(&ibm);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    if (is3D) {
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
    }
    solver.set_velocity_bc(bc);

    solver.set_body_force(-config.dp_dx, 0.0);
    solver.print_solver_info();
    solver.initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    std::ofstream force_file;
    if (mpi_rank == 0) {
        force_file.open(config.output_dir + "forces.dat");
        if (force_file.is_open()) {
            force_file << "# step  time  Fx  Fy  residual  bulk_u\n";
        }
    }

    ScopedTimer total_timer("Total simulation", false);

    for (int step = 1; step <= config.max_steps; ++step) {
        if (config.adaptive_dt) {
            solver.set_dt(solver.compute_adaptive_dt());
        }
        double residual = solver.step();

#ifdef USE_GPU_OFFLOAD
        solver.sync_solution_from_gpu();
#endif
        auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), solver.current_dt());
        double bulk_u = solver.bulk_velocity_x();
        double time = solver.current_time();

        if (mpi_rank == 0) {
            if (force_file.is_open()) {
                force_file << step << " " << time << " "
                           << Fx << " " << Fy << " "
                           << residual << " " << bulk_u << "\n";
                if (step % config.output_freq == 0) force_file.flush();
            }
            if (step % config.output_freq == 0 || step == 1) {
                std::cout << "Step " << std::setw(6) << step
                          << "  t=" << std::fixed << std::setprecision(4) << time
                          << "  res=" << std::scientific << std::setprecision(3) << residual
                          << "  Fx=" << std::fixed << std::setprecision(4) << Fx
                          << "  U_b=" << std::setprecision(4) << bulk_u
                          << "\n" << std::flush;
            }
        }

        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "ERROR: Solver diverged at step " << step << "\n";
            break;
        }

        if (residual < config.tol && step > 100) {
            if (mpi_rank == 0) {
                std::cout << "Converged at step " << step
                          << " (residual=" << residual << ")\n";
            }
            break;
        }
    }

    total_timer.stop();

    if (mpi_rank == 0) {
        std::cout << "\n=== Simulation complete ===\n";
        std::cout << "Re_h = " << Re << "\n";
        TimingStats::instance().print_summary();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
```

- [ ] **Step 2: Register in CMakeLists.txt**

```cmake
add_executable(hills app/main_hills.cpp)
target_link_libraries(hills nn_cfd_core)
```

- [ ] **Step 3: Build and verify**

Run: `cd build && cmake .. && make hills`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add app/main_hills.cpp CMakeLists.txt
git commit -m "Add periodic hills app driver (Breuer 2009)"
```

---

### Task 5: Enable configurable body force in cylinder/airfoil apps

**Files:**
- Modify: `app/main_cylinder.cpp` (line 156)
- Modify: `app/main_airfoil.cpp` (line 151)

Both apps hardcode `set_body_force(0.0, 0.0)`. Change to use `config.dp_dx` so we can drive periodic-domain flows with a pressure gradient.

- [ ] **Step 1: Fix main_cylinder.cpp**

Change line 156 from:
```cpp
    solver.set_body_force(0.0, 0.0);
```
to:
```cpp
    solver.set_body_force(-config.dp_dx, 0.0);
```

- [ ] **Step 2: Fix main_airfoil.cpp**

Change line 151 from:
```cpp
    solver.set_body_force(0.0, 0.0);
```
to:
```cpp
    solver.set_body_force(-config.dp_dx, 0.0);
```

- [ ] **Step 3: Build and verify existing tests still pass**

Run: `cd build && cmake .. && make cylinder airfoil && ctest -R IBM -L fast --output-on-failure`
Expected: All existing IBM tests pass (dp_dx defaults to 0.0, so behavior unchanged).

- [ ] **Step 4: Commit**

```bash
git add app/main_cylinder.cpp app/main_airfoil.cpp
git commit -m "Use config dp_dx for cylinder/airfoil body force"
```

---

## Chunk 3: Campaign Infrastructure

### Task 6: Config file generator

**Files:**
- Create: `scripts/rans_campaign/generate_configs.py`
- Output: `examples/13_rans_campaign/*.cfg` (38 files)

- [ ] **Step 1: Create config generator script**

Create `scripts/rans_campaign/generate_configs.py`:

```python
#!/usr/bin/env python3
"""Generate 38 config files for the RANS validation campaign."""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', '13_rans_campaign')

# Base configs per case
CASES = {
    'cylinder_re100': {
        'exe': 'cylinder',
        'params': {
            'Nx': 192, 'Ny': 128, 'Nz': 4,
            'x_min': 0.0, 'x_max': 30.0,
            'y_min': -10.0, 'y_max': 10.0,
            'z_min': 0.0, 'z_max': 3.14159,
            'nu': 0.01, 'dp_dx': -0.001,
            'max_steps': 50000, 'CFL_max': 0.5,
            'adaptive_dt': 'true', 'mode': 'unsteady',
            'scheme': 'upwind',
            'poisson_solver': 'auto', 'poisson_tol': 1e-6,
            'poisson_max_vcycles': 20,
            'output_freq': 500,
            'write_fields': 'true', 'num_snapshots': 10,
        },
    },
    'airfoil_re1000': {
        'exe': 'airfoil',
        'params': {
            'Nx': 256, 'Ny': 128, 'Nz': 4,
            'x_min': -5.0, 'x_max': 15.0,
            'y_min': -8.0, 'y_max': 8.0,
            'z_min': 0.0, 'z_max': 1.0,
            'nu': 0.001, 'dp_dx': -0.0005,
            'max_steps': 50000, 'CFL_max': 0.5,
            'adaptive_dt': 'true', 'mode': 'unsteady',
            'scheme': 'upwind',
            'poisson_solver': 'auto', 'poisson_tol': 1e-6,
            'poisson_max_vcycles': 20,
            'output_freq': 500,
            'write_fields': 'true', 'num_snapshots': 10,
        },
    },
    'step_re5000': {
        'exe': 'step',
        'params': {
            'Nx': 256, 'Ny': 128, 'Nz': 4,
            'x_min': -10.0, 'x_max': 20.0,
            'y_min': 0.0, 'y_max': 6.0,
            'z_min': 0.0, 'z_max': 1.0,
            'nu': 0.0002, 'dp_dx': 0.0,
            'max_steps': 50000, 'CFL_max': 0.5,
            'adaptive_dt': 'true', 'mode': 'unsteady',
            'scheme': 'upwind',
            'poisson_solver': 'auto', 'poisson_tol': 1e-6,
            'poisson_max_vcycles': 20,
            'output_freq': 500,
            'write_fields': 'true', 'num_snapshots': 10,
        },
    },
    'hills_re10595': {
        'exe': 'hills',
        'params': {
            'Nx': 192, 'Ny': 96, 'Nz': 4,
            'x_min': 0.0, 'x_max': 9.0,
            'y_min': 0.0, 'y_max': 3.035,
            'z_min': 0.0, 'z_max': 1.0,
            'nu': 9.438e-5, 'dp_dx': -1.0,
            'max_steps': 50000, 'CFL_max': 0.5,
            'adaptive_dt': 'true', 'mode': 'unsteady',
            'scheme': 'upwind',
            'poisson_solver': 'auto', 'poisson_tol': 1e-6,
            'poisson_max_vcycles': 20,
            'output_freq': 500,
            'write_fields': 'true', 'num_snapshots': 10,
        },
    },
}

# Standard RANS models (all cases)
MODELS = [
    ('baseline', {}),
    ('komega', {}),
    ('sst', {}),
    ('gep', {}),
    ('earsm_wj', {}),
    ('earsm_gs', {}),
    ('earsm_pope', {}),
    ('nn_mlp', {'nn_preset': 'mlp_channel_caseholdout'}),
    ('nn_tbnn', {'nn_preset': 'tbnn_channel_caseholdout'}),
]

# Extra NN models for periodic hills only
HILLS_EXTRA_MODELS = [
    ('nn_mlp', {'nn_preset': 'mlp_phll_caseholdout', 'suffix': '_phll'}),
    ('nn_tbnn', {'nn_preset': 'tbnn_phll_caseholdout', 'suffix': '_phll'}),
]


def write_config(filepath, case_params, model_name, model_extra):
    with open(filepath, 'w') as f:
        f.write(f"# Auto-generated RANS campaign config\n")
        f.write(f"# Case: {os.path.basename(filepath)}\n\n")
        for key, val in case_params.items():
            f.write(f"{key} = {val}\n")
        f.write(f"\nturb_model = {model_name}\n")
        for key, val in model_extra.items():
            if key == 'suffix':
                continue
            f.write(f"{key} = {val}\n")
        out_name = os.path.splitext(os.path.basename(filepath))[0]
        f.write(f"\noutput_dir = output/rans_campaign/{out_name}/\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    job_list = []

    for case_name, case_info in CASES.items():
        exe = case_info['exe']
        params = case_info['params']

        models = list(MODELS)
        if 'hills' in case_name:
            models += HILLS_EXTRA_MODELS

        for model_name, model_extra in models:
            suffix = model_extra.get('suffix', '')
            safe_model = model_name.replace('nn_', 'nn')
            cfg_name = f"{case_name}_{safe_model}{suffix}.cfg"
            cfg_path = os.path.join(OUT_DIR, cfg_name)
            write_config(cfg_path, params, model_name, model_extra)
            job_list.append((exe, f"examples/13_rans_campaign/{cfg_name}"))

    # Write job list
    job_list_path = os.path.join(os.path.dirname(__file__), 'job_list.txt')
    with open(job_list_path, 'w') as f:
        f.write("# index  executable  config_path\n")
        for i, (exe, cfg) in enumerate(job_list):
            f.write(f"{i} {exe} {cfg}\n")

    print(f"Generated {len(job_list)} configs in {OUT_DIR}")
    print(f"Job list written to {job_list_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the generator**

Run: `cd /storage/scratch1/6/sbryngelson3/cfd-nn && python3 scripts/rans_campaign/generate_configs.py`
Expected: `Generated 38 configs in .../examples/13_rans_campaign`

- [ ] **Step 3: Verify config count and spot-check**

Run: `ls examples/13_rans_campaign/*.cfg | wc -l` → 38
Run: `cat examples/13_rans_campaign/hills_re10595_nntbnn_phll.cfg` → should have `nn_preset = tbnn_phll_caseholdout`

- [ ] **Step 4: Commit**

```bash
git add scripts/rans_campaign/generate_configs.py scripts/rans_campaign/job_list.txt examples/13_rans_campaign/
git commit -m "Add RANS campaign config generator (38 cases)"
```

---

### Task 7: SLURM submission script

**Files:**
- Create: `scripts/rans_campaign/submit_campaign.sbatch`
- Create: `scripts/rans_campaign/submit.sh`

- [ ] **Step 1: Create SLURM array job script**

Create `scripts/rans_campaign/submit_campaign.sbatch`:

```bash
#!/bin/bash
#SBATCH -J rans_campaign
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h100,gpu-h200,gpu-a100,gpu-v100
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 02:00:00
#SBATCH -o output/rans_campaign/slurm-%A_%a.out
#SBATCH -e output/rans_campaign/slurm-%A_%a.err
#SBATCH --array=0-37

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build_gpu_campaign"
JOB_LIST="$SCRIPT_DIR/job_list.txt"

# Read job parameters for this array task
LINE=$(grep "^${SLURM_ARRAY_TASK_ID} " "$JOB_LIST")
EXE=$(echo "$LINE" | awk '{print $2}')
CONFIG=$(echo "$LINE" | awk '{print $3}')

echo "=== RANS Campaign Job ==="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Executable: $EXE"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "========================="

# Ensure output directory exists
CONFIG_BASE=$(basename "$CONFIG" .cfg)
mkdir -p "$PROJECT_DIR/output/rans_campaign/$CONFIG_BASE"

# Set GPU environment
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_NUM_THREADS=1

cd "$PROJECT_DIR"

# Run with timing
START_TIME=$(date +%s%N)
"$BUILD_DIR/$EXE" --config "$CONFIG"
EXIT_CODE=$?
END_TIME=$(date +%s%N)

ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
echo ""
echo "=== Job Summary ==="
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED_MS}ms ($(echo "scale=1; $ELAPSED_MS/1000" | bc)s)"
echo "End: $(date)"
echo "==================="

exit $EXIT_CODE
```

- [ ] **Step 2: Create convenience submit script**

Create `scripts/rans_campaign/submit.sh`:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build_gpu_campaign"

echo "=== RANS Campaign Submission ==="

# Check build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found: $BUILD_DIR"
    echo "Build first with:"
    echo "  mkdir -p $BUILD_DIR && cd $BUILD_DIR"
    echo "  cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  make -j\$(nproc) cylinder airfoil step hills"
    exit 1
fi

# Check executables
for exe in cylinder airfoil step hills; do
    if [ ! -f "$BUILD_DIR/$exe" ]; then
        echo "ERROR: Missing executable: $BUILD_DIR/$exe"
        exit 1
    fi
done

# Check configs
N_CONFIGS=$(ls "$PROJECT_DIR/examples/13_rans_campaign/"*.cfg 2>/dev/null | wc -l)
if [ "$N_CONFIGS" -ne 38 ]; then
    echo "ERROR: Expected 38 configs, found $N_CONFIGS"
    echo "Run: python3 scripts/rans_campaign/generate_configs.py"
    exit 1
fi

# Create output directory
mkdir -p "$PROJECT_DIR/output/rans_campaign"

# Submit
JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/submit_campaign.sbatch")
echo "Submitted array job: $JOB_ID (38 tasks)"
echo "Monitor: squeue -j $JOB_ID"
echo "Logs: output/rans_campaign/slurm-${JOB_ID}_*.out"
echo ""
echo "After completion, analyze with:"
echo "  python3 scripts/rans_campaign/analyze_campaign.py"
```

- [ ] **Step 3: Make scripts executable**

Run: `chmod +x scripts/rans_campaign/submit_campaign.sbatch scripts/rans_campaign/submit.sh`

- [ ] **Step 4: Commit**

```bash
git add scripts/rans_campaign/submit_campaign.sbatch scripts/rans_campaign/submit.sh
git commit -m "Add SLURM array job for RANS campaign (38 jobs)"
```

---

### Task 8: Post-processing and profiling analysis

**Files:**
- Create: `scripts/rans_campaign/analyze_campaign.py`

This script parses all 38 SLURM output logs, extracts timing/profiling data, force coefficients, convergence status, and compares against benchmarks.

- [ ] **Step 1: Create analysis script**

Create `scripts/rans_campaign/analyze_campaign.py`:

```python
#!/usr/bin/env python3
"""Analyze RANS campaign results: timing, convergence, and sanity checks."""

import os
import re
import glob
import sys

# Benchmark ranges for sanity checks
BENCHMARKS = {
    'cylinder_re100': {
        'Cd': (1.0, 1.8),
        'St': (0.14, 0.19),
    },
    'airfoil_re1000': {
        'Cd': (0.02, 0.20),
        'Cl_abs': (0.0, 0.05),  # symmetric airfoil at alpha=0
    },
    'step_re5000': {
        'converged': True,
    },
    'hills_re10595': {
        'converged': True,
    },
}


def parse_log(filepath):
    """Parse a SLURM .out log file for timing and physics data."""
    result = {
        'file': filepath,
        'converged': False,
        'diverged': False,
        'final_residual': None,
        'total_steps': 0,
        'wall_time_ms': None,
        'timing': {},
        'Cd': None,
        'Cl': None,
        'Fx': None,
        'Fy': None,
        'bulk_u': None,
        'gpu_util': None,
    }

    try:
        with open(filepath, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        result['error'] = 'File not found'
        return result

    # Convergence
    if 'Converged at step' in text:
        result['converged'] = True
    if 'Solver diverged' in text or 'NaN' in text.lower():
        result['diverged'] = True

    # Final step data
    step_lines = re.findall(r'Step\s+(\d+)\s+.*?res=\s*([\d.eE+-]+)', text)
    if step_lines:
        result['total_steps'] = int(step_lines[-1][0])
        result['final_residual'] = float(step_lines[-1][1])

    # Force coefficients (last line with Cd)
    cd_matches = re.findall(r'Cd=\s*([\d.eE+-]+)', text)
    cl_matches = re.findall(r'Cl=\s*([\d.eE+-]+)', text)
    if cd_matches:
        result['Cd'] = float(cd_matches[-1])
    if cl_matches:
        result['Cl'] = float(cl_matches[-1])

    # Forces (last values)
    fx_matches = re.findall(r'Fx=\s*([\d.eE+-]+)', text)
    fy_matches = re.findall(r'Fy=\s*([\d.eE+-]+)', text)
    if fx_matches:
        result['Fx'] = float(fx_matches[-1])
    if fy_matches:
        result['Fy'] = float(fy_matches[-1])

    # Bulk velocity
    ub_matches = re.findall(r'U_b=\s*([\d.eE+-]+)', text)
    if ub_matches:
        result['bulk_u'] = float(ub_matches[-1])

    # Wall time
    wt_match = re.search(r'Wall time:\s*(\d+)ms', text)
    if wt_match:
        result['wall_time_ms'] = int(wt_match.group(1))

    # Timing summary
    timing_section = re.search(
        r'=== Timing Summary ===(.*?)(?:===|$)', text, re.DOTALL)
    if timing_section:
        for line in timing_section.group(1).split('\n'):
            match = re.match(r'\s*(\S+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)', line)
            if match:
                result['timing'][match.group(1)] = {
                    'total_s': float(match.group(2)),
                    'calls': int(match.group(3)),
                    'avg_ms': float(match.group(4)),
                }

    # GPU utilization
    gpu_match = re.search(r'GPU utilization:\s*([\d.]+)\s*%', text)
    if gpu_match:
        result['gpu_util'] = float(gpu_match.group(1))

    return result


def identify_case(cfg_name):
    """Extract case name from config filename."""
    for case in ['cylinder_re100', 'airfoil_re1000', 'step_re5000', 'hills_re10595']:
        if cfg_name.startswith(case):
            return case
    return 'unknown'


def sanity_check(case_name, result):
    """Check results against benchmarks. Returns list of (status, message)."""
    checks = []
    bench = BENCHMARKS.get(case_name, {})

    if result['diverged']:
        checks.append(('FAIL', 'Simulation diverged'))
        return checks

    if 'Cd' in bench and result['Cd'] is not None:
        lo, hi = bench['Cd']
        if lo <= result['Cd'] <= hi:
            checks.append(('PASS', f"Cd={result['Cd']:.4f} in [{lo}, {hi}]"))
        else:
            checks.append(('WARN', f"Cd={result['Cd']:.4f} outside [{lo}, {hi}]"))

    if 'Cl_abs' in bench and result['Cl'] is not None:
        lo, hi = bench['Cl_abs']
        if lo <= abs(result['Cl']) <= hi:
            checks.append(('PASS', f"|Cl|={abs(result['Cl']):.4f} in [{lo}, {hi}]"))
        else:
            checks.append(('WARN', f"|Cl|={abs(result['Cl']):.4f} outside [{lo}, {hi}]"))

    if bench.get('converged') and not result['converged']:
        checks.append(('WARN', 'Did not converge'))

    if not checks:
        checks.append(('OK', 'No benchmark data to compare'))

    return checks


def main():
    project_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    log_dir = os.path.join(project_dir, 'output', 'rans_campaign')

    logs = sorted(glob.glob(os.path.join(log_dir, 'slurm-*.out')))
    if not logs:
        print(f"No logs found in {log_dir}")
        print("Have you run the campaign yet?")
        sys.exit(1)

    print(f"Found {len(logs)} log files\n")

    # Parse all logs
    results = []
    for log_path in logs:
        result = parse_log(log_path)
        # Extract config name from log content
        with open(log_path, 'r') as f:
            text = f.read()
        cfg_match = re.search(r'Config:\s*(\S+)', text)
        if cfg_match:
            result['config'] = os.path.basename(cfg_match.group(1))
        else:
            result['config'] = os.path.basename(log_path)
        results.append(result)

    # === Summary Table ===
    print("=" * 120)
    print(f"{'Config':<45} {'Status':<10} {'Steps':>7} {'Residual':>12} "
          f"{'Wall(s)':>8} {'Cd':>8} {'GPU%':>6} {'Check'}")
    print("-" * 120)

    for r in results:
        cfg = r.get('config', '?')[:44]
        status = 'DIVERG' if r['diverged'] else ('CONV' if r['converged'] else 'RUNNING')
        steps = r['total_steps']
        res = f"{r['final_residual']:.2e}" if r['final_residual'] else '—'
        wall = f"{r['wall_time_ms']/1000:.1f}" if r['wall_time_ms'] else '—'
        cd = f"{r['Cd']:.4f}" if r['Cd'] else '—'
        gpu = f"{r['gpu_util']:.0f}" if r['gpu_util'] else '—'

        case_name = identify_case(r.get('config', ''))
        checks = sanity_check(case_name, r)
        check_str = '; '.join(f"{s}: {m}" for s, m in checks)

        print(f"{cfg:<45} {status:<10} {steps:>7} {res:>12} "
              f"{wall:>8} {cd:>8} {gpu:>6} {check_str}")

    print("=" * 120)

    # === Profiling Summary ===
    print("\n=== Profiling Summary (avg ms/step) ===\n")
    print(f"{'Config':<45} {'step':>8} {'poisson':>8} {'turb':>8} "
          f"{'conv':>8} {'diff':>8}")
    print("-" * 85)

    for r in results:
        cfg = r.get('config', '?')[:44]
        t = r.get('timing', {})

        def avg(key):
            for k, v in t.items():
                if key in k.lower():
                    return f"{v['avg_ms']:.2f}"
            return '—'

        print(f"{cfg:<45} {avg('solver_step'):>8} {avg('poisson'):>8} "
              f"{avg('turbulence'):>8} {avg('convect'):>8} {avg('diffus'):>8}")

    print()

    # === Failure Summary ===
    failures = [r for r in results if r['diverged']]
    warnings = [r for r in results
                if any(s == 'WARN' for s, _ in sanity_check(
                    identify_case(r.get('config', '')), r))]

    if failures:
        print(f"\n!!! {len(failures)} DIVERGED simulations:")
        for r in failures:
            print(f"  - {r.get('config', '?')}")

    if warnings:
        print(f"\n!!! {len(warnings)} simulations with benchmark WARNINGS:")
        for r in warnings:
            case_name = identify_case(r.get('config', ''))
            checks = sanity_check(case_name, r)
            for s, m in checks:
                if s == 'WARN':
                    print(f"  - {r.get('config', '?')}: {m}")

    if not failures and not warnings:
        print("All simulations passed sanity checks.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/rans_campaign/analyze_campaign.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/rans_campaign/analyze_campaign.py
git commit -m "Add RANS campaign analysis script (timing + sanity checks)"
```

---

## Chunk 4: Build, Submit, and Validate

### Task 9: GPU build

- [ ] **Step 1: Create GPU build directory and build**

```bash
mkdir -p build_gpu_campaign && cd build_gpu_campaign
cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
make -j$(nproc) cylinder airfoil step hills test_ibm_step_sdf test_ibm_hills_sdf
```

- [ ] **Step 2: Run SDF tests on login node (CPU mode)**

```bash
cd build_gpu_campaign
./test_ibm_step_sdf
./test_ibm_hills_sdf
```

Expected: Both pass.

- [ ] **Step 3: Smoke-test each executable with a tiny config**

Quick sanity: run each executable for 10 steps to confirm they start and produce output.

```bash
cd /storage/scratch1/6/sbryngelson3/cfd-nn
./build_gpu_campaign/step --config examples/13_rans_campaign/step_re5000_baseline.cfg --max_steps 10
./build_gpu_campaign/hills --config examples/13_rans_campaign/hills_re10595_baseline.cfg --max_steps 10
```

Expected: Runs 10 steps without crash, prints step output.

- [ ] **Step 4: Commit any build fixes if needed**

### Task 10: Submit campaign

- [ ] **Step 1: Generate configs if not already done**

```bash
python3 scripts/rans_campaign/generate_configs.py
```

- [ ] **Step 2: Submit SLURM array job**

```bash
bash scripts/rans_campaign/submit.sh
```

Expected: `Submitted array job: <JOBID> (38 tasks)`

- [ ] **Step 3: Monitor jobs**

```bash
squeue -u $(whoami) --qos=embers
```

Wait for all 38 tasks to complete.

### Task 11: Analyze results

- [ ] **Step 1: Run analysis script**

```bash
python3 scripts/rans_campaign/analyze_campaign.py
```

Expected: Summary table with convergence status, Cd values, timing profiling, and sanity check results for all 38 simulations.

- [ ] **Step 2: Review and document findings**

Check for:
- Any diverged simulations (may need CFL/scheme adjustments)
- Cd/Cl within benchmark ranges
- Timing: Poisson fraction, turbulence overhead per model
- GPU utilization (expect >50% for these grid sizes)

- [ ] **Step 3: Commit analysis output**

```bash
# Save analysis output
python3 scripts/rans_campaign/analyze_campaign.py > output/rans_campaign/analysis_summary.txt
git add output/rans_campaign/analysis_summary.txt
git commit -m "RANS campaign results: analysis summary"
```
