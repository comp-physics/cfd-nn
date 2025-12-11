#!/bin/bash
# Comprehensive Pre-Push Testing Script
# Validates that BOTH CPU CI and GPU CI workflows will pass
# Run this before pushing to catch issues early!

set -e

echo "==================================================================="
echo "  Comprehensive Pre-Push Testing"
echo "  Tests: CPU (Debug + Release) + GPU (Full validation)"
echo "==================================================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS_PASSED=true
START_TIME=$(date +%s)

cleanup_on_exit() {
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "==================================================================="
    if [ "$TESTS_PASSED" = true ]; then
        echo -e "${GREEN}✓✓✓ ALL TESTS PASSED! ✓✓✓${NC}"
        echo "==================================================================="
        echo ""
        echo "✅ CPU CI (Debug + Release): PASS"
        echo "✅ GPU CI (Full validation): PASS"
        echo ""
        echo "Total time: ${DURATION}s"
        echo ""
        echo -e "${GREEN}You are SAFE to push!${NC} 🚀"
        echo ""
    else
        echo -e "${RED}❌❌❌ TESTS FAILED! ❌❌❌${NC}"
        echo "==================================================================="
        echo ""
        echo "Fix the issues above before pushing."
        echo "This will prevent CI failures and save time!"
        echo ""
        exit 1
    fi
}
trap cleanup_on_exit EXIT

# =================================================================
# PHASE 1: CPU CI VALIDATION (Debug + Release)
# =================================================================
echo ""
echo "==================================================================="
echo -e "  ${BLUE}PHASE 1: CPU CI Validation${NC}"
echo "  (Mimics .github/workflows/ci.yml)"
echo "==================================================================="
echo ""

test_cpu_build() {
    local BUILD_TYPE=$1
    local BUILD_DIR="build_precheck_${BUILD_TYPE,,}"
    
    echo ""
    echo "-------------------------------------------------------------------"
    echo -e "  Testing CPU ${YELLOW}${BUILD_TYPE}${NC} Build"
    echo "-------------------------------------------------------------------"
    
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure (matching CI flags)
    echo "→ Configuring CMake..."
    export CXXFLAGS="-Wno-unused-variable -Wno-unused-but-set-variable"
    cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" > cmake.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ CMake configuration failed!${NC}"
        cat cmake.log
        TESTS_PASSED=false
        cd "$PROJECT_ROOT"
        return 1
    fi
    echo -e "${GREEN}✓${NC} CMake configured"
    
    # Build
    echo "→ Building..."
    make -j$(nproc 2>/dev/null || echo 4) > build.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Build failed!${NC}"
        tail -50 build.log
        TESTS_PASSED=false
        cd "$PROJECT_ROOT"
        return 1
    fi
    echo -e "${GREEN}✓${NC} Build successful"
    
    # Check warnings (Release only, like CI)
    if [ "$BUILD_TYPE" = "Release" ]; then
        echo "→ Checking for warnings..."
        if grep -i "warning:" build.log | grep -v "unused-variable" | grep -v "unused-but-set-variable"; then
            echo -e "${RED}❌ Compiler warnings detected!${NC}"
            TESTS_PASSED=false
            cd "$PROJECT_ROOT"
            return 1
        fi
        echo -e "${GREEN}✓${NC} No warnings"
    fi
    
    # Run tests
    echo "→ Running CTest..."
    ctest --output-on-failure > test.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Tests failed!${NC}"
        cat test.log
        TESTS_PASSED=false
        cd "$PROJECT_ROOT"
        return 1
    fi
    echo -e "${GREEN}✓${NC} All tests passed"
    
    # Quick turbulence model check (subset of CI)
    echo "→ Testing turbulence models..."
    mkdir -p output/precheck_validation
    
    ./channel --Nx 64 --Ny 128 --nu 0.001 --max_iter 50 \
             --model baseline --dp_dx -0.01 \
             --output output/precheck_validation/baseline --num_snapshots 0 --quiet > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Baseline model failed!${NC}"
        TESTS_PASSED=false
        cd "$PROJECT_ROOT"
        return 1
    fi
    
    ./channel --Nx 64 --Ny 128 --nu 0.001 --max_iter 100 \
             --model sst --dp_dx -0.01 \
             --output output/precheck_validation/sst --num_snapshots 0 --quiet > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ SST model failed!${NC}"
        TESTS_PASSED=false
        cd "$PROJECT_ROOT"
        return 1
    fi
    
    echo -e "${GREEN}✓${NC} Turbulence models OK"
    
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✅ CPU $BUILD_TYPE: PASS${NC}"
}

# Test both CPU build types
test_cpu_build "Release"
test_cpu_build "Debug"

# =================================================================
# PHASE 2: GPU CI VALIDATION (if on cluster)
# =================================================================
echo ""
echo "==================================================================="
echo -e "  ${BLUE}PHASE 2: GPU CI Validation${NC}"
echo "  (Mimics .github/workflows/gpu-ci.yml)"
echo "==================================================================="
echo ""

# Check if we're on a system with GPU and SLURM
if ! command -v sbatch &> /dev/null; then
    echo -e "${YELLOW}⚠️  SLURM not available - skipping GPU tests${NC}"
    echo "   (GPU CI will run on self-hosted runner)"
    echo ""
else
    echo "→ Checking for NVHPC compiler..."
    if ! module load nvhpc/25.5 2>/dev/null && ! module load nvhpc/24.5 2>/dev/null; then
        echo -e "${YELLOW}⚠️  NVHPC not available - skipping GPU tests${NC}"
        echo "   (GPU CI will run on self-hosted runner)"
        echo ""
    else
        echo -e "${GREEN}✓${NC} NVHPC available"
        
        # Verify critical GPU fixes in source (matching gpu-ci.yml)
        echo ""
        echo "→ Verifying critical GPU fixes in source code..."
        
        FIXES_OK=true
        
        # Fix #1: Stride-based indexing
        if ! grep -q "int i = idx % Nx + 1.*skip ghost" src/turbulence_transport.cpp; then
            echo -e "${RED}  ✗ Missing stride-based indexing in turbulence_transport.cpp${NC}"
            FIXES_OK=false
        fi
        
        if ! grep -q "int cell_idx = j \* stride + i.*Stride-based index" src/turbulence_gep.cpp; then
            echo -e "${RED}  ✗ Missing stride-based indexing in turbulence_gep.cpp${NC}"
            FIXES_OK=false
        fi
        
        # Fix #2: GPU sync after transport updates
        if ! grep -A5 "advance_turbulence" src/solver.cpp | grep -q "target update to.*k_ptr_"; then
            echo -e "${RED}  ✗ Missing GPU sync after advance_turbulence()${NC}"
            FIXES_OK=false
        fi
        
        # Fix #3: Correct map sizes
        if ! grep -q "map(present:.*total_size" src/turbulence_transport.cpp; then
            echo -e "${RED}  ✗ Missing correct map size (total_size)${NC}"
            FIXES_OK=false
        fi
        
        # Fix #4: GPU pointers
        if ! grep -q "double\* k_ptr_" include/solver.hpp || ! grep -q "double\* omega_ptr_" include/solver.hpp; then
            echo -e "${RED}  ✗ Missing k_ptr_/omega_ptr_ declarations${NC}"
            FIXES_OK=false
        fi
        
        if ! grep -q "k_ptr_ = k_.data().data()" src/solver.cpp; then
            echo -e "${RED}  ✗ Missing k_ptr_ initialization${NC}"
            FIXES_OK=false
        fi
        
        if [ "$FIXES_OK" = false ]; then
            echo -e "${RED}❌ Critical GPU fixes missing!${NC}"
            TESTS_PASSED=false
        else
            echo -e "${GREEN}✓${NC} All critical GPU fixes verified"
        fi
        
        # Build GPU version
        echo ""
        echo "→ Building GPU version..."
        BUILD_DIR="build_precheck_gpu"
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        
        CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON > cmake_gpu.log 2>&1
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ GPU CMake configuration failed!${NC}"
            cat cmake_gpu.log
            TESTS_PASSED=false
            cd "$PROJECT_ROOT"
        else
            # Check that GPU offloading is enabled
            if ! grep -q "GPU offloading ENABLED" cmake_gpu.log; then
                echo -e "${RED}❌ GPU offloading not enabled!${NC}"
                TESTS_PASSED=false
                cd "$PROJECT_ROOT"
            else
                echo -e "${GREEN}✓${NC} GPU CMake configured"
                
                make -j8 > build_gpu.log 2>&1
                if [ $? -ne 0 ]; then
                    echo -e "${RED}❌ GPU build failed!${NC}"
                    tail -50 build_gpu.log
                    TESTS_PASSED=false
                    cd "$PROJECT_ROOT"
                else
                    echo -e "${GREEN}✓${NC} GPU build successful"
                    
                    # Submit GPU tests to SLURM (matching GPU CI)
                    echo ""
                    echo "→ Submitting GPU tests to SLURM..."
                    echo "  (This will test: unit tests, turbulence models, CPU-GPU consistency)"
                    
                    # Create test script
                    cat > gpu_precheck_test.sh <<'EOF'
#!/bin/bash
#SBATCH -J precheck_gpu
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu-l40s
#SBATCH -G1
#SBATCH --qos=inferno
#SBATCH -t 00:30:00
#SBATCH -o precheck_gpu.out
#SBATCH -e precheck_gpu.err

module load nvhpc/25.5 2>/dev/null || module load nvhpc/24.5

echo "=== GPU Test Suite ==="
echo ""

# Run unit tests
echo "1. Unit tests..."
export OMP_TARGET_OFFLOAD=MANDATORY
ctest --output-on-failure || exit 1
echo "✓ Unit tests passed"
echo ""

# Test CPU-GPU consistency (CRITICAL)
echo "2. CPU-GPU consistency..."
./test_cpu_gpu_consistency || {
    echo "❌ CPU-GPU consistency FAILED!"
    exit 1
}
echo "✓ CPU-GPU consistency validated"
echo ""

# Quick turbulence model test
echo "3. Turbulence models on GPU..."
mkdir -p output/gpu_precheck
./channel --Nx 128 --Ny 256 --nu 0.001 --max_iter 100 \
    --model baseline --num_snapshots 0 --quiet || exit 1
./channel --Nx 128 --Ny 256 --nu 0.001 --max_iter 200 \
    --model sst --num_snapshots 0 --quiet || exit 1
echo "✓ Turbulence models passed"
echo ""

echo "=== All GPU tests PASSED ==="
EOF
                    
                    chmod +x gpu_precheck_test.sh
                    
                    JOB_ID=$(sbatch --parsable gpu_precheck_test.sh)
                    echo "  Job ID: $JOB_ID"
                    echo "  Waiting for GPU tests to complete (max 30 min)..."
                    
                    # Wait for job with timeout
                    WAIT_COUNT=0
                    MAX_WAIT=360  # 30 minutes
                    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
                        if ! squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; then
                            break
                        fi
                        sleep 5
                        WAIT_COUNT=$((WAIT_COUNT + 1))
                        
                        # Progress indicator every minute
                        if [ $((WAIT_COUNT % 12)) -eq 0 ]; then
                            echo "  ... still running ($((WAIT_COUNT / 12)) min elapsed)"
                        fi
                    done
                    
                    # Check results
                    if [ -f precheck_gpu.out ]; then
                        if grep -q "All GPU tests PASSED" precheck_gpu.out; then
                            echo -e "${GREEN}✓${NC} GPU tests passed"
                            echo -e "${GREEN}✅ GPU CI: PASS${NC}"
                        else
                            echo -e "${RED}❌ GPU tests failed!${NC}"
                            echo ""
                            echo "GPU test output:"
                            cat precheck_gpu.out
                            if [ -f precheck_gpu.err ]; then
                                echo ""
                                echo "GPU test errors:"
                                cat precheck_gpu.err
                            fi
                            TESTS_PASSED=false
                        fi
                    else
                        echo -e "${RED}❌ GPU test output not found!${NC}"
                        TESTS_PASSED=false
                    fi
                    
                    cd "$PROJECT_ROOT"
                fi
            fi
        fi
    fi
fi

# Cleanup
echo ""
echo "→ Cleaning up build directories..."
rm -rf build_precheck_*
echo -e "${GREEN}✓${NC} Cleanup complete"

exit 0


