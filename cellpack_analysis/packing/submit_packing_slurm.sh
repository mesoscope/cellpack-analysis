#!/usr/bin/env bash
# ============================================================================
# submit_packing_slurm.sh
#
# Lightweight launcher that submits the entire cellPACK packing workflow to
# SLURM.  **No Python runs on the login node.**
#
# It submits three chained SLURM jobs:
#   1. orchestrator — generates configs/recipes, writes batch manifests &
#      sbatch scripts, then submits the packing worker jobs.
#   2. (worker jobs) — created by the orchestrator; each packs a batch of
#      recipes.
#   3. aggregator — runs after all workers finish; collects per-batch
#      result JSONs into a single summary.
#
# Usage:
#   bash submit_packing_slurm.sh -c path/to/workflow_config.json [OPTIONS]
#
# Options:
#   -c, --config       Path to the workflow configuration JSON (required)
#   -b, --batch-size   Recipes per worker SLURM job (default: 8)
#   -v, --venv         Path to virtualenv activate script (auto-detected
#                      from VIRTUAL_ENV if omitted)
#   -p, --partition    SLURM partition (default: cluster default)
#   -t, --time         Wall-clock limit for worker jobs (default: 00:30:00)
#   -m, --mem          Memory per worker job (default: 16G)
#   --cpus             CPUs per worker task (default: 4)
#   --job-name         SLURM job name prefix (default: cellpack)
#   --orch-time        Wall-clock limit for orchestrator job (default: 1-00:00:00)
#   --orch-mem         Memory for orchestrator job (default: 16G)
#   --orch-cpus        CPUs for orchestrator job (default: same as --cpus)
#   --orch-partition   Partition for orchestrator job (defaults to --partition)
#   --max-jobs         Max concurrent worker jobs (uses SLURM job arrays)
#   --dry-run          Pass --dry-run to orchestrator (no workers submitted)
#   -h, --help         Show this help message
#
# Examples:
#   # Basic — auto-detect venv, default batch size
#   bash submit_packing_slurm.sh -c configs/peroxisome.json
#
#   # Explicit venv and batch size
#   bash submit_packing_slurm.sh \
#       -c configs/peroxisome.json \
#       -b 16 \
#       -v /path/to/.venv/bin/activate
#
#   # Custom resources
#   bash submit_packing_slurm.sh \
#       -c configs/peroxisome.json \
#       -p my_partition -t 24:00:00 -m 32G
#
#   # Limit to max 20 concurrent worker jobs
#   bash submit_packing_slurm.sh \
#       -c configs/peroxisome.json \
#       --max-jobs 20
#
# Monitoring:
#   squeue -u $USER                     # list your jobs
#   squeue -u $USER --name=cellpack     # filter by job name
#   scontrol show job <JOBID>           # detailed job info
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
#   scancel <JOBID>                     # cancel a job
#   sinfo                               # partition/node overview
#   sinfo -p <partition> -N -l          # per-node detail
# ============================================================================
set -euo pipefail

# ----------------------------  defaults  ------------------------------------
BATCH_SIZE=8
PARTITION=""
TIME="00:30:00"
MEM="16G"
CPUS="4"
JOB_NAME="cellpack"
ORCH_TIME="1-00:00:00"
ORCH_CPUS="4"
ORCH_MEM="16G"
ORCH_PARTITION=""
CONFIG=""
VENV=""
DRY_RUN=""
MAX_JOBS=0

# ----------------------------  parse args  ----------------------------------
usage() {
    sed -n '2,/^# ===/s/^# //p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)       CONFIG="$2";        shift 2 ;;
        -b|--batch-size)   BATCH_SIZE="$2";    shift 2 ;;
        -v|--venv)         VENV="$2";          shift 2 ;;
        -p|--partition)    PARTITION="$2";      shift 2 ;;
        -t|--time)         TIME="$2";          shift 2 ;;
        -m|--mem)          MEM="$2";           shift 2 ;;
        --cpus)            CPUS="$2";          shift 2 ;;
        --job-name)        JOB_NAME="$2";      shift 2 ;;
        --orch-time)       ORCH_TIME="$2";     shift 2 ;;
        --orch-mem)        ORCH_MEM="$2";      shift 2 ;;
        --orch-partition)  ORCH_PARTITION="$2"; shift 2 ;;
        --orch-cpus)       ORCH_CPUS="$2";     shift 2 ;;
        --max-jobs)        MAX_JOBS="$2";       shift 2 ;;
        --dry-run)         DRY_RUN="--dry-run"; shift ;;
        -h|--help)         usage 0 ;;
        *)                 echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config (-c) is required." >&2
    usage 1
fi

# Resolve config to absolute path
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG" >&2
    exit 1
fi

# ----------------------------  auto-detect venv  ----------------------------
if [[ -z "$VENV" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        VENV="${VIRTUAL_ENV}/bin/activate"
    else
        echo "Warning: No --venv given and VIRTUAL_ENV not set." >&2
        echo "         Workers will inherit the submitter's bare environment." >&2
    fi
fi

if [[ -n "$VENV" && ! -f "$VENV" ]]; then
    echo "Error: virtualenv activate script not found: $VENV" >&2
    exit 1
fi

# Default orchestrator partition to the worker partition
ORCH_PARTITION="${ORCH_PARTITION:-$PARTITION}"
ORCH_CPUS="${ORCH_CPUS:-$CPUS}"

# ----------------------------  detect python  -------------------------------
# We need the python path inside the venv for the sbatch scripts.
if [[ -n "$VENV" ]]; then
    PYTHON_EXEC="$(dirname "$VENV")/python"
else
    PYTHON_EXEC="$(command -v python3 || command -v python)"
fi

if [[ ! -x "$PYTHON_EXEC" ]]; then
    echo "Error: Cannot find Python executable at $PYTHON_EXEC" >&2
    exit 1
fi

# ----------------------------  build commands  ------------------------------
ACTIVATE_CMD=""
if [[ -n "$VENV" ]]; then
    ACTIVATE_CMD="source ${VENV}"
fi

SLURM_ARGS=(
    "--slurm-time"      "$TIME"
    "--slurm-mem"       "$MEM"
    "--slurm-cpus-per-task" "$CPUS"
    "--slurm-job-name"  "$JOB_NAME"
)
if [[ -n "$PARTITION" ]]; then
    SLURM_ARGS+=("--slurm-partition" "$PARTITION")
fi

ORCHESTRATOR_CMD="${PYTHON_EXEC} -m cellpack_analysis.packing.run_packing_workflow_slurm \
    --orchestrate \
    -c ${CONFIG} \
    -b ${BATCH_SIZE} \
    --max-jobs ${MAX_JOBS} \
    ${DRY_RUN} \
    $(printf '%s ' "${SLURM_ARGS[@]}")"

# Trim trailing whitespace the venv arg if needed
if [[ -n "$VENV" ]]; then
    ORCHESTRATOR_CMD="${ORCHESTRATOR_CMD} --venv-path ${VENV}"
fi

# ----------------------------  submit orchestrator  -------------------------
ORCH_SCRIPT=$(mktemp /tmp/cellpack_orch_XXXXXX.sh)
# Build the partition directive only if a partition was specified
ORCH_PARTITION_LINE=""
if [[ -n "$ORCH_PARTITION" ]]; then
    ORCH_PARTITION_LINE="#SBATCH --partition=${ORCH_PARTITION}"
fi

# Save orchestrator SLURM logs next to the config file so they are easy to find.
# The Python-level log ends up in <output_path>/logs/ once the workflow config
# is parsed, but that path is not known at bash submission time.
ORCH_LOG_DIR="$(dirname "$CONFIG")/slurm_logs"
mkdir -p "$ORCH_LOG_DIR"

cat > "$ORCH_SCRIPT" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}_orch
${ORCH_PARTITION_LINE}
#SBATCH --time=${ORCH_TIME}
#SBATCH --mem=${ORCH_MEM}
#SBATCH --cpus-per-task=${ORCH_CPUS}
#SBATCH --output=${ORCH_LOG_DIR}/${JOB_NAME}_orch_%j.out
#SBATCH --error=${ORCH_LOG_DIR}/${JOB_NAME}_orch_%j.err

set -euo pipefail

${ACTIVATE_CMD}

echo "=== Orchestrator job \$SLURM_JOB_ID on \$(hostname) ==="
echo "Config: ${CONFIG}"
echo "Batch size: ${BATCH_SIZE}"
echo "Start: \$(date)"

# Run the orchestrator on the compute node.
# --orchestrate: prep + submit workers + write tracking file
# --no-wait:     don't block (the aggregator job handles collection)
${ORCHESTRATOR_CMD}

echo "Orchestrator finished: \$(date)"
SBATCH_EOF
chmod 755 "$ORCH_SCRIPT"

echo "=== cellPACK SLURM Launcher ==="
echo "Config:          $CONFIG"
echo "Batch size:      $BATCH_SIZE"
echo "Partition:       ${PARTITION:-<cluster default>}"
echo "Worker time:     $TIME"
echo "Worker mem:      $MEM"
echo "Orch partition:  ${ORCH_PARTITION:-<cluster default>}"
echo "Orch time:       $ORCH_TIME"
echo "Orch mem:        $ORCH_MEM"
echo "Orch cpus:       $ORCH_CPUS"
echo "Max jobs:        ${MAX_JOBS:-0 (unlimited)}"
echo "Job name:        $JOB_NAME"
echo "Venv:            ${VENV:-<none>}"
echo ""

ORCH_SUBMIT=$(sbatch "$ORCH_SCRIPT" 2>&1)
ORCH_JOB_ID=$(echo "$ORCH_SUBMIT" | awk '{print $NF}')

if [[ -z "$ORCH_JOB_ID" || "$ORCH_JOB_ID" == "$ORCH_SUBMIT" ]]; then
    echo "Error: Failed to submit orchestrator job:" >&2
    echo "  $ORCH_SUBMIT" >&2
    rm -f "$ORCH_SCRIPT"
    exit 1
fi

echo "Submitted orchestrator job: $ORCH_JOB_ID"
echo "  Script: $ORCH_SCRIPT"
echo "  Stdout: ${ORCH_LOG_DIR}/${JOB_NAME}_orch_${ORCH_JOB_ID}.out"
echo "  Stderr: ${ORCH_LOG_DIR}/${JOB_NAME}_orch_${ORCH_JOB_ID}.err"
echo ""
echo "The orchestrator will generate recipes/configs on the compute node"
echo "and then submit worker batch jobs automatically."
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --name=${JOB_NAME}"
echo "  scontrol show job $ORCH_JOB_ID"
echo ""
echo "After all jobs finish, aggregate results with:"
echo "  ${PYTHON_EXEC} -m cellpack_analysis.packing.run_packing_workflow_slurm \\"
echo "      --aggregate <output_path>/slurm_staging/job_tracking.json"
echo ""
echo "Done. No further compute on this node."
