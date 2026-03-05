#!/bin/bash
#
# Build and run CFD-NN simulations
#
# Usage:
#   ./run.sh gpu --config file.cfg          # Build GPU + run
#   ./run.sh cpu --config file.cfg          # Build CPU + run
#   ./run.sh gpu --build-only               # Build only
#   ./run.sh --run-only --config file.cfg   # Run only (skip build)
#   ./run.sh gpu --dry-run --config file.cfg  # Show what would run
#   ./run.sh gpu --slurm --config file.cfg  # Submit SLURM job
#
# Extra solver args after '--':
#   ./run.sh gpu --config file.cfg -- --max_steps 5000 --nu 0.001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────
BUILD_MODE=""
CONFIG_FILE=""
BUILD_ONLY=false
RUN_ONLY=false
DRY_RUN=false
USE_SLURM=false
SLURM_TIME="06:00:00"
SLURM_PARTITION=""
EXECUTABLE=""
MAKE_ARGS=()
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
            MAKE_ARGS+=("cpu")
            shift
            ;;
        gpu|GPU)
            BUILD_MODE="gpu"
            MAKE_ARGS+=("gpu")
            shift
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
        --slurm)
            USE_SLURM=true
            shift
            ;;
        --slurm-time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --slurm-partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --exe)
            EXECUTABLE="$2"
            shift 2
            ;;
        --debug|--rebuild|--hdf5|--mpi|--hypre|--all-features)
            MAKE_ARGS+=("$1")
            shift
            ;;
        --gpu-cc|--build-dir|--jobs)
            MAKE_ARGS+=("$1" "$2")
            shift 2
            ;;
        -j*)
            MAKE_ARGS+=("$1")
            shift
            ;;
        --help|-h)
            cat <<'HELP'
Usage: ./run.sh [cpu|gpu] [options] --config file.cfg [-- solver_args...]

Build + Run:
  ./run.sh gpu --config examples/01_laminar_channel/poiseuille.cfg
  ./run.sh cpu --debug --config my_case.cfg
  ./run.sh gpu --config case.cfg -- --max_steps 5000

Build only:
  ./run.sh gpu --build-only
  ./run.sh gpu --debug --build-only

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

Build options (passed to make.sh):
  --debug             Debug build
  --rebuild           Force clean rebuild
  --hdf5              Enable HDF5
  --mpi               Enable MPI
  --hypre             Enable HYPRE
  --gpu-cc N          GPU compute capability
  --build-dir DIR     Custom build directory
  --all-features      Enable all optional features
  --jobs N / -jN      Parallel compile jobs

Extra solver arguments after '--':
  ./run.sh gpu --config case.cfg -- --nu 0.005 --Nx 128
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

# ── Determine build directory ─────────────────────────────
BUILD_DIR="$SCRIPT_DIR/build"
for i in "${!MAKE_ARGS[@]}"; do
    if [[ "${MAKE_ARGS[$i]}" == "--build-dir" ]]; then
        next=$((i + 1))
        BUILD_DIR="${MAKE_ARGS[$next]}"
        # Make absolute
        if [[ "$BUILD_DIR" != /* ]]; then
            BUILD_DIR="$SCRIPT_DIR/$BUILD_DIR"
        fi
    fi
done

# ── Auto-detect executable ────────────────────────────────
detect_executable() {
    if [[ -n "$EXECUTABLE" ]]; then
        echo "$EXECUTABLE"
        return
    fi

    local cfg="$1"
    local basename
    basename="$(basename "$(dirname "$cfg")")"

    # Match by example directory naming or config content
    case "$basename" in
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

# ── Build ─────────────────────────────────────────────────
if [[ "$RUN_ONLY" == false ]]; then
    if [[ ${#MAKE_ARGS[@]} -eq 0 ]]; then
        # Default to cpu if no mode specified
        MAKE_ARGS=("cpu")
        BUILD_MODE="cpu"
    fi

    echo "=== Building ($BUILD_MODE) ==="
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] $SCRIPT_DIR/make.sh ${MAKE_ARGS[*]}"
    else
        "$SCRIPT_DIR/make.sh" "${MAKE_ARGS[@]}"
    fi
    echo ""
fi

if [[ "$BUILD_ONLY" == true ]]; then
    echo "Build complete. Run with:"
    echo "  $EXE_PATH --config <file.cfg>"
    exit 0
fi

# ── Verify executable exists ──────────────────────────────
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

# ── Build run command ─────────────────────────────────────
RUN_CMD=("$EXE_PATH" "--config" "$CONFIG_FILE")
if [[ ${#SOLVER_ARGS[@]} -gt 0 ]]; then
    RUN_CMD+=("${SOLVER_ARGS[@]}")
fi

# ── GPU environment ───────────────────────────────────────
GPU_ENV=""
if [[ "$BUILD_MODE" == "gpu" ]]; then
    GPU_ENV="OMP_TARGET_OFFLOAD=MANDATORY"
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
        SLURM_ACCOUNT=$(sacctmgr -n -P show assoc user="$USER" format=account 2>/dev/null | head -1 || true)
    fi

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

${RUN_CMD[*]}
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
        # Clean up script after submission
        rm -f "$SLURM_SCRIPT"
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
