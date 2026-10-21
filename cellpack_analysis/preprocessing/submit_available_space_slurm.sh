#!/usr/bin/env bash
# ============================================================================
# submit_available_space_slurm.sh
#
# Lightweight launcher that submits the calculate_available_space workflow to
# SLURM.  No Python runs on the login node — all work happens on compute nodes.
#
# It submits a single orchestrator job that:
#   1. Discovers valid meshes for the given structure.
#   2. Partitions cells into batches and writes batch manifests.
#   3. Submits a SLURM job array (one task per batch, up to --max-jobs concurrent).
#
# Usage:
#   bash submit_available_space_slurm.sh -s <structure_id> [OPTIONS]
#
# Required:
#   -s, --structure-id    Structure ID to process
#
# Options:
#   -b, --batch-size      Cells per SLURM array task (default: 4)
#       --spacing         Grid spacing for distance calculations (default: 2.0)
#       --use-struct-mesh Exclude structure volume from available space
#       --use-mean-shape  Process mean-shape cell only
#       --use-all-cells   Use all cells, including those outside the 8D sphere
#       --recalculate     Recalculate even if output files already exist. Default: false (skips existing)
#       --chunk-size      Override adaptive chunk size
#   -v, --venv            Path to venv activate script (auto-detected from
#                         VIRTUAL_ENV if omitted)
#   -p, --partition       SLURM partition (default: cluster default)
#   -t, --time            Wall-clock limit per array task (default: 02:00:00)
#   -m, --mem             Memory per array task (default: 64G)
#       --cpus            CPUs per array task (default: batch-size)
#       --max-jobs        Max concurrent array tasks (default: 16)
#       --job-name        SLURM job name prefix (default: cellpack_avspace)
#       --orch-time       Wall-clock limit for orchestrator job (default: 1:00:00)
#       --orch-mem        Memory for orchestrator job (default: 16G)
#       --orch-partition  Partition for orchestrator job (defaults to --partition)
#       --dry-run         Write scripts but do not submit
#   -h, --help            Show this help message
#
# Examples:
#   # Basic — auto-detect venv, default settings
#   bash submit_available_space_slurm.sh -s SLC25A17
#
#   # With structure mesh exclusion and explicit partition
#   bash submit_available_space_slurm.sh -s SLC25A17 --use-struct-mesh -p my_partition
#
#   # Force recalculation with larger memory
#   bash submit_available_space_slurm.sh -s SLC25A17 --recalculate -m 128G
#
# Monitoring:
#   squeue -u $USER --name=cellpack_avspace
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
# ============================================================================
set -euo pipefail

# ----------------------------  defaults  ------------------------------------
STRUCTURE_ID=""
BATCH_SIZE=4
SPACING="2.0"
USE_STRUCT_MESH=""
USE_MEAN_SHAPE=""
USE_ALL_CELLS=""
RECALCULATE=""
CHUNK_SIZE=""
PARTITION=""
TIME="02:00:00"
MEM="64G"
CPUS="$BATCH_SIZE"
MAX_JOBS=16
JOB_NAME="cellpack_avspace"
ORCH_TIME="1-00:00:00"
ORCH_MEM="16G"
ORCH_PARTITION=""
VENV=""
DRY_RUN=""
NO_WAIT=""

# ----------------------------  parse args  ----------------------------------
usage() {
    sed -n '2,/^# ===/s/^# //p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--structure-id)  STRUCTURE_ID="$2";   shift 2 ;;
        -b|--batch-size)    BATCH_SIZE="$2";      shift 2 ;;
        --spacing)          SPACING="$2";          shift 2 ;;
        --use-struct-mesh)  USE_STRUCT_MESH="--use-struct-mesh"; shift ;;
        --use-mean-shape)   USE_MEAN_SHAPE="--use-mean-shape";   shift ;;
        --use-all-cells)    USE_ALL_CELLS="--use-all-cells";     shift ;;
        --recalculate)      RECALCULATE="--recalculate";          shift ;;
        --chunk-size)       CHUNK_SIZE="--chunk-size $2";         shift 2 ;;
        -v|--venv)          VENV="$2";             shift 2 ;;
        -p|--partition)     PARTITION="$2";        shift 2 ;;
        -t|--time)          TIME="$2";             shift 2 ;;
        -m|--mem)           MEM="$2";              shift 2 ;;
        --cpus)             CPUS="$2";             shift 2 ;;
        --max-jobs)         MAX_JOBS="$2";          shift 2 ;;
        --job-name)         JOB_NAME="$2";         shift 2 ;;
        --orch-time)        ORCH_TIME="$2";        shift 2 ;;
        --orch-mem)         ORCH_MEM="$2";         shift 2 ;;
        --orch-partition)   ORCH_PARTITION="$2";   shift 2 ;;
        --dry-run)          DRY_RUN="--dry-run";   shift ;;
        --no-wait)          NO_WAIT="--no-wait";   shift ;;
        -h|--help)          usage 0 ;;
        *)                  echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$STRUCTURE_ID" ]]; then
    echo "Error: --structure-id (-s) is required." >&2
    usage 1
fi

# ----------------------------  auto-detect venv  ----------------------------
if [[ -z "$VENV" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        VENV="${VIRTUAL_ENV}/bin/activate"
    else
        # exit if venv is not set
        echo "Warning: No --venv given and VIRTUAL_ENV not set." >&2
        echo "Stopping execution." >&2
        exit 1
    fi
fi

if [[ -n "$VENV" && ! -f "$VENV" ]]; then
    echo "Error: virtualenv activate script not found: $VENV" >&2
    exit 1
fi

ORCH_PARTITION="${ORCH_PARTITION:-$PARTITION}"

# ----------------------------  detect python  -------------------------------
if [[ -n "$VENV" ]]; then
    PYTHON_EXEC="$(dirname "$VENV")/python"
else
    PYTHON_EXEC="$(command -v python3 || command -v python)"
fi

if [[ ! -x "$PYTHON_EXEC" ]]; then
    echo "Error: Cannot find Python executable at $PYTHON_EXEC" >&2
    exit 1
fi

# ----------------------------  determine log dir  ---------------------------
# Resolve project root relative to this script's location.
TS=$(date +"%Y%m%d")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/structure_data/${STRUCTURE_ID}/slurm_logs/${TS}"
ORCH_LOG_DIR="${LOG_DIR}/orchestrator"
mkdir -p "$ORCH_LOG_DIR"

# ----------------------------  build orchestrator cmd  ----------------------
ACTIVATE_CMD=""
if [[ -n "$VENV" ]]; then
    ACTIVATE_CMD="source ${VENV}"
fi

VENV_ARG=""
if [[ -n "$VENV" ]]; then
    VENV_ARG="--venv-path ${VENV}"
fi

PARTITION_DIRECTIVE=""
if [[ -n "$PARTITION" ]]; then
    PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
fi

ORCH_PARTITION_DIRECTIVE=""
if [[ -n "$ORCH_PARTITION" ]]; then
    ORCH_PARTITION_DIRECTIVE="#SBATCH --partition=${ORCH_PARTITION}"
fi

ORCHESTRATOR_CMD="${PYTHON_EXEC} -m cellpack_analysis.preprocessing.run_available_space_slurm \
    --orchestrate \
    --structure-id ${STRUCTURE_ID} \
    --spacing ${SPACING} \
    --batch-size ${BATCH_SIZE} \
    --max-jobs ${MAX_JOBS} \
    --slurm-time ${TIME} \
    --slurm-mem ${MEM} \
    --slurm-cpus-per-task ${CPUS} \
    --slurm-job-name ${JOB_NAME} \
    ${USE_STRUCT_MESH} \
    ${USE_MEAN_SHAPE} \
    ${USE_ALL_CELLS} \
    ${RECALCULATE} \
    ${CHUNK_SIZE} \
    ${DRY_RUN} \
    ${VENV_ARG} \
    ${NO_WAIT}"

if [[ -n "$PARTITION" ]]; then
    ORCHESTRATOR_CMD="${ORCHESTRATOR_CMD} --slurm-partition ${PARTITION}"
fi

# ----------------------------  submit orchestrator  -------------------------
ORCH_SCRIPT=$(mktemp /tmp/avspace_orch_XXXXXX.sh)

cat > "$ORCH_SCRIPT" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}_orch
${ORCH_PARTITION_DIRECTIVE}
#SBATCH --time=${ORCH_TIME}
#SBATCH --mem=${ORCH_MEM}
#SBATCH --cpus-per-task=1
#SBATCH --output=${ORCH_LOG_DIR}/${JOB_NAME}_orch_%j.out
#SBATCH --error=${ORCH_LOG_DIR}/${JOB_NAME}_orch_%j.err

set -euo pipefail

${ACTIVATE_CMD}

echo "=== Orchestrator job \$SLURM_JOB_ID on \$(hostname) ==="
echo "Structure: ${STRUCTURE_ID}"
echo "Batch size: ${BATCH_SIZE}"
echo "Start: \$(date)"

${ORCHESTRATOR_CMD}

echo "Orchestrator finished: \$(date)"
SBATCH_EOF
chmod 755 "$ORCH_SCRIPT"

echo "=== Available Space SLURM Launcher ==="
echo "Structure:       $STRUCTURE_ID"
echo "Batch size:      $BATCH_SIZE"
echo "Spacing:         $SPACING"
echo "Use struct mesh: ${USE_STRUCT_MESH:-no}"
echo "Use mean shape:  ${USE_MEAN_SHAPE:-no}"
echo "Use all cells:   ${USE_ALL_CELLS:-no}"
echo "Recalculate:     ${RECALCULATE:-no}"
echo "Partition:       ${PARTITION:-<cluster default>}"
echo "Time/task:       $TIME"
echo "Mem/task:        $MEM"
echo "CPUs/task:       $CPUS"
echo "Max concurrent:  $MAX_JOBS"
echo "Job name:        $JOB_NAME"
echo "Venv:            ${VENV:-<none>}"
echo "Wait:            ${NO_WAIT:+no-wait}${NO_WAIT:-wait for completion}"
echo ""

ORCH_SUBMIT=$(sbatch "$ORCH_SCRIPT" 2>&1)
ORCH_JOB_ID=$(echo "$ORCH_SUBMIT" | awk '{print $NF}')

if [[ -z "$ORCH_JOB_ID" || "$ORCH_JOB_ID" == "$ORCH_SUBMIT" ]]; then
    echo "Error: Failed to submit orchestrator job:" >&2
    echo "  $ORCH_SUBMIT" >&2
    rm -f "$ORCH_SCRIPT"
    exit 1
fi

TRACKING_FILE="${PROJECT_ROOT}/data/structure_data/${STRUCTURE_ID}/slurm_staging/${TS}/job_tracking.json"
ORCH_LOG="${ORCH_LOG_DIR}/${JOB_NAME}_orch_${ORCH_JOB_ID}.out"

echo "Submitted orchestrator job: $ORCH_JOB_ID"
echo "  Script: $ORCH_SCRIPT"
echo "  Stdout: ${ORCH_LOG}"
echo "  Stderr: ${ORCH_LOG_DIR}/${JOB_NAME}_orch_${ORCH_JOB_ID}.err"
echo ""
echo "The orchestrator will discover meshes on the compute node and submit"
echo "a worker array job automatically."
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --name=${JOB_NAME}"
echo "  scontrol show job $ORCH_JOB_ID"
echo "  # Once the array is submitted:"
echo "  grep 'Submitted job array' ${ORCH_LOG}"
echo "  # or:"
echo "  cat ${TRACKING_FILE}"
echo ""
echo "After all array tasks finish, aggregate results with:"
echo "  ${PYTHON_EXEC} -m cellpack_analysis.preprocessing.run_available_space_slurm \\"
echo "      --aggregate ${TRACKING_FILE}"
