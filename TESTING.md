# Unit Testing

## Overview

The project includes comprehensive unit tests to ensure correctness and prevent regressions. All tests are built with CMake and can be run individually or via the test runner script.

## Running Tests

### Run All Tests

```bash
./run_tests.sh
```

### Run Individual Tests

```bash
cd build
./test_mesh
./test_poisson
./test_solver
./test_features
./test_nn_core
./test_turbulence
```

### Using CTest

```bash
cd build
ctest --output-on-failure
```

## Test Suites

### test_mesh
Tests mesh initialization, indexing, and field operations.

**Coverage:**
- Uniform mesh creation
- Stretched mesh with tanh stretching
- Wall distance computation
- Scalar and vector field operations

### test_poisson
Tests the Poisson solver with various boundary conditions.

**Coverage:**
- Laplacian computation accuracy
- Dirichlet boundary conditions
- Periodic boundary conditions
- Mixed boundary conditions (periodic x, Neumann y)
- Convergence to analytical solutions

### test_solver
Tests the RANS solver against analytical solutions.

**Coverage:**
- Laminar Poiseuille flow validation (< 5% error)
- Convergence to steady state
- Residual computation

### test_features
Tests feature computation for neural network inputs.

**Coverage:**
- Velocity gradient computation
- Strain and rotation tensor computation
- Tensor basis functions for TBNN
- Invariant computation
- Feature vector assembly

### test_nn_core
Tests the neural network forward pass implementation.

**Coverage:**
- Dense layer forward pass
- Multi-layer perceptron forward pass
- Weight loading from files
- Activation functions

### test_turbulence
Tests all turbulence model implementations.

**Coverage:**
- Baseline mixing length model
- GEP symbolic model
- NN-MLP scalar eddy viscosity model
- NN-TBNN tensor basis model
- Positivity and boundedness checks
- NaN/Inf detection

## Test Results

All tests currently pass:

```
==========================================
Test Summary
==========================================
Passed: 6
Failed: 0
==========================================
```

## Adding New Tests

1. Create a new test file in `tests/`:

```cpp
#include "mesh.hpp"
// ... other includes
#include <cassert>

using namespace nncfd;

void test_my_feature() {
    std::cout << "Testing my feature... ";
    
    // Test code
    assert(condition);
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== My Test Suite ===\n\n";
    test_my_feature();
    std::cout << "\nAll tests passed!\n";
    return 0;
}
```

2. Add to CMakeLists.txt:

```cmake
add_executable(test_my_feature tests/test_my_feature.cpp)
target_link_libraries(test_my_feature nn_cfd_core)
add_test(NAME MyFeatureTest COMMAND test_my_feature)
```

3. Add to run_tests.sh:

```bash
run_test "My Feature Tests" "test_my_feature"
```

## Continuous Integration

### GitHub Actions

The project includes automated CI that runs on every push and pull request:

**`.github/workflows/ci.yml`** - Build and test on Ubuntu and macOS:
- Builds in both Release and Debug modes
- Runs all 6 test suites
- Checks for compiler warnings
- Tests on multiple platforms

**`.github/workflows/documentation.yml`** - Documentation checks:
- Verifies all required docs exist
- Checks for large files accidentally committed

### CI Status

Check the [Actions tab](https://github.com/YOUR_USERNAME/nn-cfd/actions) to see CI results.

### Running Tests Locally

Tests should be run:
- Before committing changes
- After pulling updates
- Before releases
- When modifying core solver components

To run exactly what CI runs:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

## Test Philosophy

- Tests should be fast (< 1 second per suite)
- Tests should be deterministic
- Tests should validate against known analytical solutions where possible
- Tests should check edge cases and error conditions
- Tests should not depend on external data (except model weights which can be skipped)

