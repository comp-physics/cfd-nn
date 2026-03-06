#!/bin/bash
#
# Build and run CFD-NN simulations
#
# Usage:
#   ./run.sh gpu --config file.cfg            # Build GPU + run
#   ./run.sh cpu --config file.cfg            # Build CPU + run
#   ./run.sh gpu --build-only                 # Build only (replaces make.sh)
#   ./run.sh clean                            # Remove build artifacts
#   ./run.sh --run-only --config file.cfg     # Run only (skip build)
#   ./run.sh gpu --dry-run --config file.cfg  # Show what would run
#   ./run.sh gpu --slurm --config file.cfg    # Submit SLURM job
#
# Extra solver args after '--':
#   ./run.sh gpu --config file.cfg -- --max_steps 5000 --nu 0.001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────
BUILD_MODE=""
BUILD_TYPE="Release"
BUILD_DIR=""
REBUILD=false
JOBS=$(nproc 2>/dev/null || echo 4)
USE_HDF5=false
USE_MPI=false
USE_HYPRE=false
GPU_CC=""
ALL_FEATURES=false

CONFIG_FILE=""
BUILD_ONLY=false
RUN_ONLY=false
DRY_RUN=false
USE_SLURM=false
SLURM_TIME="06:00:00"
SLURM_PARTITION=""
EXECUTABLE=""
SOLVER_ARGS=()
PARSING_SOLVER_ARGS=false

# ── Parse arguments ───────────────────────────────────────
while [[ $# -gt 0 ]]; do
    if [[ "$PARSING_SOLVER_ARGS" == true ]]; then
        SOLVER_ARGS+=("$1")
        shift
        continue
    fi

    case "$1" in
        --)
            PARSING_SOLVER_ARGS=true
            shift
            ;;
        cpu|CPU)
            BUILD_MODE="cpu"
            shift
            ;;
        gpu|GPU)
            BUILD_MODE="gpu"
            shift
            ;;
        clean)
            echo "Cleaning build directories..."
            rm -rf "$PROJECT_ROOT"/build/CMake* "$PROJECT_ROOT"/build/Makefile "$PROJECT_ROOT"/build/*.cmake
            rm -rf "$PROJECT_ROOT"/build/*.a "$PROJECT_ROOT"/build/compile_commands.json
            rm -rf "$PROJECT_ROOT"/build/channel "$PROJECT_ROOT"/build/duct "$PROJECT_ROOT"/build/taylor_green_3d
            rm -rf "$PROJECT_ROOT"/build/cylinder "$PROJECT_ROOT"/build/airfoil
            rm -rf "$PROJECT_ROOT"/build/compare_channel_cpu_gpu
            rm -rf "$PROJECT_ROOT"/build/profile_* "$PROJECT_ROOT"/build/test_* "$PROJECT_ROOT"/build/bench_*
            echo "Clean complete."
            exit 0
            ;;
        --config|-c)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --config requires a file argument"
                exit 1
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --jobs)
            if [[ $# -lt 2 ]]; then echo "ERROR: --jobs requires an argument"; exit 1; fi
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        --hdf5)
            USE_HDF5=true
            shift
            ;;
        --mpi)
            USE_MPI=true
            shift
            ;;
        --hypre)
            USE_HYPRE=true
            shift
            ;;
        --gpu-cc)
            if [[ $# -lt 2 ]]; then echo "ERROR: --gpu-cc requires an argument"; exit 1; fi
            GPU_CC="$2"
            shift 2
            ;;
        --build-dir)
            if [[ $# -lt 2 ]]; then echo "ERROR: --build-dir requires an argument"; exit 1; fi
            BUILD_DIR="$2"
            shift 2
            ;;
        --all-features)
            ALL_FEATURES=true
            shift
            ;;
        --slurm)
            USE_SLURM=true
            shift
            ;;
        --slurm-time)
            if [[ $# -lt 2 ]]; then echo "ERROR: --slurm-time requires an argument"; exit 1; fi
            SLURM_TIME="$2"
            shift 2
            ;;
        --slurm-partition)
            if [[ $# -lt 2 ]]; then echo "ERROR: --slurm-partition requires an argument"; exit 1; fi
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --exe)
            if [[ $# -lt 2 ]]; then echo "ERROR: --exe requires an argument"; exit 1; fi
            EXECUTABLE="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'HELP'
Usage: ./run.sh [cpu|gpu|clean] [options] [--config file.cfg] [-- solver_args...]

Build + Run:
  ./run.sh gpu --config examples/01_laminar_channel/poiseuille.cfg
  ./run.sh cpu --debug --config my_case.cfg
  ./run.sh gpu --config case.cfg -- --max_steps 5000

Build only:
  ./run.sh gpu --build-only
  ./run.sh gpu --debug --build-only
  ./run.sh clean                           # Remove build artifacts

Run only (skip build):
  ./run.sh --run-only --config examples/01_laminar_channel/poiseuille.cfg
  ./run.sh --run-only --exe duct --config duct_case.cfg

SLURM submission:
  ./run.sh gpu --slurm --config dns_case.cfg
  ./run.sh gpu --slurm --slurm-time 12:00:00 --config big_case.cfg

Options:
  --config FILE       Config file for the solver (required unless --build-only)
  --build-only        Build without running
  --run-only          Run without building (uses existing build/)
  --dry-run           Show commands without executing
  --exe NAME          Executable name (default: auto-detect from config)
  --slurm             Submit as SLURM job instead of running directly
  --slurm-time HH:MM  SLURM wall time (default: 06:00:00)
  --slurm-partition P  SLURM partition (default: auto)

Build options:
  --debug             Debug build (default: Release)
  --rebuild           Force clean rebuild
  --jobs N / -jN      Parallel compile jobs (default: auto)
  --hdf5              Enable HDF5 checkpoint/restart
  --mpi               Enable MPI domain decomposition
  --hypre             Enable HYPRE Poisson solver (GPU only)
  --gpu-cc N          GPU compute capability (e.g., 80=A100, 90=H200)
  --build-dir DIR     Custom build directory (default: build/)
  --all-features      Enable all optional features

Extra solver arguments after '--':
  ./run.sh gpu --config case.cfg -- --nu 0.005 --Nx 128

Test commands (run from build directory):
  ctest --output-on-failure        All tests
  ctest -L fast                    Fast tests (<30s)
  ctest -L gpu                     GPU-only tests
  ctest -LE slow                   Skip slow tests
HELP
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

# ── Validate arguments ────────────────────────────────────
if [[ "$BUILD_ONLY" == false && "$RUN_ONLY" == false && -z "$CONFIG_FILE" && -z "$BUILD_MODE" ]]; then
    echo "ERROR: Specify a build mode (cpu/gpu) and --config, or use --help"
    exit 1
fi

if [[ "$BUILD_ONLY" == false && -z "$CONFIG_FILE" ]]; then
    echo "ERROR: --config is required (unless using --build-only)"
    echo "Usage: ./run.sh [cpu|gpu] --config file.cfg"
    exit 1
fi

if [[ -n "$CONFIG_FILE" && ! -f "$CONFIG_FILE" ]]; then
    # Try relative to script dir
    if [[ -f "$SCRIPT_DIR/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
    else
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi
fi

# Make config path absolute
if [[ -n "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
fi

# ── Apply --all-features ─────────────────────────────────
if [[ "$ALL_FEATURES" == true ]]; then
    USE_HDF5=true
    USE_MPI=true
    if [[ "$BUILD_MODE" == "gpu" ]]; then
        USE_HYPRE=true
    fi
fi

# ── Set build directory ──────────────────────────────────
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="$PROJECT_ROOT/build"
fi
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR"
fi

# ── Auto-detect executable ────────────────────────────────
detect_executable() {
    if [[ -n "$EXECUTABLE" ]]; then
        echo "$EXECUTABLE"
        return
    fi

    local cfg="$1"
    local dir_name
    dir_name="$(basename "$(dirname "$cfg")")"

    case "$dir_name" in
        *duct*)         echo "duct" ;;
        *taylor_green*) echo "taylor_green_3d" ;;
        *cylinder*)     echo "cylinder" ;;
        *airfoil*)      echo "airfoil" ;;
        *)              echo "channel" ;;
    esac
}

if [[ -n "$CONFIG_FILE" ]]; then
    EXE_NAME=$(detect_executable "$CONFIG_FILE")
else
    EXE_NAME="channel"
fi

EXE_PATH="$BUILD_DIR/$EXE_NAME"

# ══════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════
if [[ "$RUN_ONLY" == false ]]; then
    if [[ -z "$BUILD_MODE" ]]; then
        BUILD_MODE="cpu"
    fi

    # Load NVHPC module if needed
    if ! command -v nvc++ &>/dev/null; then
        if [[ -f /etc/profile.d/lmod.sh ]]; then
            source /etc/profile.d/lmod.sh
        fi
        if command -v module &>/dev/null; then
            if [[ "$DRY_RUN" == false ]]; then
                echo "Loading nvhpc module..."
            fi
            module load nvhpc 2>/dev/null || true
        fi
    fi

    # Verify compiler
    if ! command -v nvc++ &>/dev/null; then
        echo "ERROR: nvc++ not found."
        echo ""
        echo "Please load the NVIDIA HPC SDK:"
        echo "  module load nvhpc"
        exit 1
    fi

    CXX_COMPILER=$(command -v nvc++)

    # Clean if rebuild requested
    if [[ "$REBUILD" == true && "$DRY_RUN" == false ]]; then
        echo "Cleaning for rebuild..."
        rm -rf "$BUILD_DIR"/CMake* "$BUILD_DIR"/Makefile "$BUILD_DIR"/*.cmake
        rm -rf "$BUILD_DIR"/*.a "$BUILD_DIR"/compile_commands.json
    fi

    mkdir -p "$BUILD_DIR"

    # Build cmake options
    CMAKE_OPTS=(
        -DCMAKE_CXX_COMPILER="$CXX_COMPILER"
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    )

    if [[ "$BUILD_MODE" == "gpu" ]]; then
        CMAKE_OPTS+=(-DUSE_GPU_OFFLOAD=ON)
        if [[ -n "$GPU_CC" ]]; then
            CMAKE_OPTS+=(-DGPU_CC="$GPU_CC")
        fi
    else
        CMAKE_OPTS+=(-DUSE_GPU_OFFLOAD=OFF)
    fi

    [[ "$USE_HDF5" == true ]] && CMAKE_OPTS+=(-DUSE_HDF5=ON)
    [[ "$USE_MPI" == true ]] && CMAKE_OPTS+=(-DUSE_MPI=ON)
    [[ "$USE_HYPRE" == true ]] && CMAKE_OPTS+=(-DUSE_HYPRE=ON)

    # Print configuration
    echo "=============================================="
    if [[ "$BUILD_MODE" == "gpu" ]]; then
        echo "  GPU Build (OpenMP Target Offloading)"
    else
        echo "  CPU Build"
    fi
    echo "=============================================="
    echo "Compiler:    $CXX_COMPILER"
    echo "Build type:  $BUILD_TYPE"
    echo "Build dir:   $BUILD_DIR"
    echo "GPU offload: $(if [[ $BUILD_MODE == gpu ]]; then echo ON; else echo OFF; fi)"
    [[ -n "$GPU_CC" ]] && echo "GPU CC:      $GPU_CC"
    echo "HDF5:        $(if $USE_HDF5; then echo ON; else echo OFF; fi)"
    echo "MPI:         $(if $USE_MPI; then echo ON; else echo OFF; fi)"
    echo "HYPRE:       $(if $USE_HYPRE; then echo ON; else echo OFF; fi)"
    echo "Jobs:        $JOBS"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] cmake $PROJECT_ROOT ${CMAKE_OPTS[*]}"
        echo "[dry-run] make -j$JOBS"
    else
        # Run cmake if needed
        if [[ ! -f "$BUILD_DIR/Makefile" ]]; then
            echo "Running cmake..."
            cd "$BUILD_DIR"
            cmake "$PROJECT_ROOT" "${CMAKE_OPTS[@]}"
            echo ""
        fi

        # Build
        echo "Building with $JOBS parallel jobs..."
        cd "$BUILD_DIR"
        make -j"$JOBS"
    fi

    if [[ "$DRY_RUN" == false ]]; then
        echo ""
        echo "=============================================="
        echo "  Build complete!"
        echo "=============================================="
        echo ""
    fi
fi

if [[ "$BUILD_ONLY" == true ]]; then
    echo "Executables:"
    for exe in channel duct taylor_green_3d cylinder airfoil; do
        if [[ -x "$BUILD_DIR/$exe" ]]; then
            echo "  ./run.sh --run-only --config <file.cfg>"
            break
        fi
    done
    echo ""
    echo "Run tests:   cd $BUILD_DIR && ctest --output-on-failure"
    echo "Fast tests:  cd $BUILD_DIR && ctest -L fast --output-on-failure"
    echo ""
    exit 0
fi

# ══════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════

# Verify executable exists
if [[ "$DRY_RUN" == false && ! -x "$EXE_PATH" ]]; then
    echo "ERROR: Executable not found: $EXE_PATH"
    echo ""
    echo "Available executables in $BUILD_DIR:"
    for exe in channel duct taylor_green_3d cylinder airfoil; do
        [[ -x "$BUILD_DIR/$exe" ]] && echo "  $exe"
    done
    echo ""
    echo "Try: ./run.sh gpu --config $CONFIG_FILE --exe <name>"
    exit 1
fi

# Build run command
RUN_CMD=("$EXE_PATH" "--config" "$CONFIG_FILE")
if [[ ${#SOLVER_ARGS[@]} -gt 0 ]]; then
    RUN_CMD+=("${SOLVER_ARGS[@]}")
fi

# GPU environment
GPU_ENV=""
if [[ "$BUILD_MODE" == "gpu" ]]; then
    GPU_ENV="OMP_TARGET_OFFLOAD=MANDATORY"
elif [[ "$RUN_ONLY" == true && -z "$BUILD_MODE" ]]; then
    echo "NOTE: No build mode specified with --run-only. Use 'gpu --run-only' to set OMP_TARGET_OFFLOAD=MANDATORY."
fi

# ── SLURM submission ──────────────────────────────────────
if [[ "$USE_SLURM" == true ]]; then
    JOB_NAME="cfdnn_$(basename "$CONFIG_FILE" .cfg)"
    SLURM_SCRIPT="$SCRIPT_DIR/.slurm_${JOB_NAME}_$$.sh"

    # Detect partition
    if [[ -z "$SLURM_PARTITION" ]]; then
        if [[ "$BUILD_MODE" == "gpu" ]]; then
            SLURM_PARTITION="gpu-h200"
        else
            SLURM_PARTITION="cpu"
        fi
    fi

    # Detect account
    SLURM_ACCOUNT=""
    if command -v sacctmgr &>/dev/null; then
        SLURM_ACCOUNT=$(sacctmgr -n -P show assoc user="$USER" format=account 2>/dev/null | awk 'NR==1' || true)
    fi

    # Build properly quoted command string for SLURM script
    RUN_CMD_STR=$(printf '%q ' "${RUN_CMD[@]}")

    cat > "$SLURM_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=${JOB_NAME}_%j.out
#SBATCH --error=${JOB_NAME}_%j.err
#SBATCH --time=$SLURM_TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --qos=embers
$(if [[ -n "$SLURM_ACCOUNT" ]]; then echo "#SBATCH --account=$SLURM_ACCOUNT"; fi)
$(if [[ "$BUILD_MODE" == "gpu" ]]; then echo "#SBATCH --gpus=1"; fi)

# Load modules
if [[ -f /etc/profile.d/lmod.sh ]]; then
    source /etc/profile.d/lmod.sh
fi
module load nvhpc 2>/dev/null || true

# Set GPU environment
$(if [[ -n "$GPU_ENV" ]]; then echo "export $GPU_ENV"; fi)

echo "=== CFD-NN SLURM Job ==="
echo "Config: $CONFIG_FILE"
echo "Binary: $EXE_PATH"
echo "Node:   \$(hostname)"
echo "GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

$RUN_CMD_STR
SLURM

    if [[ "$DRY_RUN" == true ]]; then
        echo "=== SLURM Script ==="
        cat "$SLURM_SCRIPT"
        echo ""
        echo "[dry-run] sbatch $SLURM_SCRIPT"
        rm -f "$SLURM_SCRIPT"
    else
        echo "=== Submitting SLURM Job ==="
        echo "  Config:    $CONFIG_FILE"
        echo "  Binary:    $EXE_PATH"
        echo "  Partition: $SLURM_PARTITION"
        echo "  Wall time: $SLURM_TIME"
        echo ""
        sbatch "$SLURM_SCRIPT"
        echo ""
        echo "Monitor with: squeue -u $USER"
        echo "SLURM script: $SLURM_SCRIPT"
    fi
    exit 0
fi

# ── Direct execution ──────────────────────────────────────
echo "=== Running ==="
echo "  Binary: $EXE_PATH"
echo "  Config: $CONFIG_FILE"
if [[ ${#SOLVER_ARGS[@]} -gt 0 ]]; then
    echo "  Extra:  ${SOLVER_ARGS[*]}"
fi
echo ""

if [[ "$DRY_RUN" == true ]]; then
    if [[ -n "$GPU_ENV" ]]; then
        echo "[dry-run] $GPU_ENV ${RUN_CMD[*]}"
    else
        echo "[dry-run] ${RUN_CMD[*]}"
    fi
    exit 0
fi

if [[ -n "$GPU_ENV" ]]; then
    export OMP_TARGET_OFFLOAD=MANDATORY
fi

exec "${RUN_CMD[@]}"
