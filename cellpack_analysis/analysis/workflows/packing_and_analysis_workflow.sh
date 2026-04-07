#! /bin/bash
# ============================================================================
# packing_and_analysis_workflow.sh
#
# Unified launcher that runs the full cellPACK packing → analysis pipeline.
#
# By default, packing runs locally.  Pass --slurm to submit packing via SLURM
# instead (using submit_packing_slurm.sh).  Because SLURM packing is
# asynchronous (fire-and-forget), analysis is NOT chained automatically in
# SLURM mode; the script prints the analysis command to run manually after all
# SLURM jobs complete.
#
# Usage:
#   bash packing_and_analysis_workflow.sh -p <packing_config> -a <analysis_config> [OPTIONS]
#
# Required (at least one):
#   -p, --packing-config   Path to the packing workflow JSON config
#   -a, --analysis-config  Path to the analysis workflow JSON config
#
# Workflow control:
#   --slurm                Use SLURM for packing (analysis remains local; see note above)
#   --skip-packing         Skip packing; run analysis only
#   --skip-analysis        Skip analysis; run packing only
#
# Analysis options:
#   --log-level            Log level for run_analysis_workflow.py (default: INFO)
#
# SLURM pass-through options (only used with --slurm):
#   -b, --batch-size       Recipes per SLURM worker job (default: 8)
#   -v, --venv             Path to virtualenv activate script (auto-detected)
#   --partition            SLURM partition
#   -t, --time             Wall-clock limit for worker jobs (default: 00:30:00)
#   -m, --mem              Memory per worker job (default: 16G)
#   --cpus                 CPUs per worker task (default: 4)
#   --max-jobs             Max concurrent worker jobs (default: 16)
#   --job-name             SLURM job name prefix (default: cellpack)
#   --dry-run              Pass --dry-run to the packing orchestrator
#
#   -h, --help             Show this help message
#
# Examples:
#   # Local packing + analysis
#   bash packing_and_analysis_workflow.sh \
#       -p cellpack_analysis/packing/configs/peroxisome.json \
#       -a cellpack_analysis/analysis/workflows/configs/analysis_config_peroxisome.json
#
#   # SLURM packing + print analysis command for later
#   bash packing_and_analysis_workflow.sh \
#       -p cellpack_analysis/packing/configs/peroxisome.json \
#       -a cellpack_analysis/analysis/workflows/configs/analysis_config_peroxisome.json \
#       --slurm -b 16 --partition my_partition
#
#   # Analysis only (packing already done)
#   bash packing_and_analysis_workflow.sh \
#       -a cellpack_analysis/analysis/workflows/configs/analysis_config_peroxisome.json \
#       --skip-packing
#
#   # Packing only (no analysis)
#   bash packing_and_analysis_workflow.sh \
#       -p cellpack_analysis/packing/configs/peroxisome.json \
#       --skip-analysis
# ============================================================================
set -euo pipefail

# -------------------------  defaults  ---------------------------------------
PACKING_CONFIG=""
ANALYSIS_CONFIG=""
USE_SLURM=""
SKIP_PACKING=""
SKIP_ANALYSIS=""
LOG_LEVEL="INFO"

# SLURM pass-through args (accumulated as an array)
SLURM_ARGS=()

# -------------------------  parse args  ------------------------------------
usage() {
    sed -n '2,/^# ===/s/^# //p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--packing-config)    PACKING_CONFIG="$2";    shift 2 ;;
        -a|--analysis-config)   ANALYSIS_CONFIG="$2";   shift 2 ;;
        --slurm)                USE_SLURM=1;            shift ;;
        --skip-packing)         SKIP_PACKING=1;         shift ;;
        --skip-analysis)        SKIP_ANALYSIS=1;        shift ;;
        --log-level)            LOG_LEVEL="$2";         shift 2 ;;
        # SLURM pass-throughs
        -b|--batch-size)        SLURM_ARGS+=("-b" "$2");        shift 2 ;;
        -v|--venv)              SLURM_ARGS+=("-v" "$2");        shift 2 ;;
        --partition)            SLURM_ARGS+=("-p" "$2");        shift 2 ;;
        -t|--time)              SLURM_ARGS+=("-t" "$2");        shift 2 ;;
        -m|--mem)               SLURM_ARGS+=("-m" "$2");        shift 2 ;;
        --cpus)                 SLURM_ARGS+=("--cpus" "$2");    shift 2 ;;
        --max-jobs)             SLURM_ARGS+=("--max-jobs" "$2"); shift 2 ;;
        --job-name)             SLURM_ARGS+=("--job-name" "$2"); shift 2 ;;
        --dry-run)              SLURM_ARGS+=("--dry-run");      shift ;;
        -h|--help)              usage 0 ;;
        *)                      echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# -------------------------  validate args  ----------------------------------
if [[ -z "$PACKING_CONFIG" && -z "$SKIP_PACKING" ]]; then
    echo "Error: --packing-config (-p) is required unless --skip-packing is set." >&2
    usage 1
fi

if [[ -z "$ANALYSIS_CONFIG" && -z "$SKIP_ANALYSIS" ]]; then
    echo "Error: --analysis-config (-a) is required unless --skip-analysis is set." >&2
    usage 1
fi

if [[ -n "$PACKING_CONFIG" && ! -f "$PACKING_CONFIG" ]]; then
    echo "Error: Packing config not found: $PACKING_CONFIG" >&2
    exit 1
fi

if [[ -n "$ANALYSIS_CONFIG" && ! -f "$ANALYSIS_CONFIG" ]]; then
    echo "Error: Analysis config not found: $ANALYSIS_CONFIG" >&2
    exit 1
fi

# -------------------------  locate scripts  ---------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$(cd "$SCRIPT_DIR/../../packing" && pwd)/submit_packing_slurm.sh"

echo "=== cellPACK Packing + Analysis Workflow ==="
[[ -n "$PACKING_CONFIG" ]]  && echo "Packing config:  $PACKING_CONFIG"
[[ -n "$ANALYSIS_CONFIG" ]] && echo "Analysis config: $ANALYSIS_CONFIG"
echo "SLURM mode:      ${USE_SLURM:+yes}${USE_SLURM:-no}"
echo "Skip packing:    ${SKIP_PACKING:+yes}${SKIP_PACKING:-no}"
echo "Skip analysis:   ${SKIP_ANALYSIS:+yes}${SKIP_ANALYSIS:-no}"
echo ""

# -------------------------  packing  ----------------------------------------
if [[ -z "$SKIP_PACKING" ]]; then
    if [[ -n "$USE_SLURM" ]]; then
        # --- SLURM packing (fire-and-forget) ---
        echo ">>> Submitting packing via SLURM ..."
        bash "$SLURM_SCRIPT" -c "$PACKING_CONFIG" "${SLURM_ARGS[@]}"
        echo ""
        echo "SLURM packing jobs submitted.  Packing workers run asynchronously."
        echo "Analysis cannot be chained automatically in SLURM mode."
        if [[ -n "$ANALYSIS_CONFIG" && -z "$SKIP_ANALYSIS" ]]; then
            echo ""
            echo "After all SLURM jobs complete, run analysis with:"
            echo "  python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \\"
            echo "      --config_file $ANALYSIS_CONFIG \\"
            echo "      --log_level $LOG_LEVEL"
        fi
        exit 0
    else
        # --- Local packing ---
        echo ">>> Running local packing ..."
        echo "python cellpack_analysis/packing/run_packing_workflow.py -c $PACKING_CONFIG"
        python cellpack_analysis/packing/run_packing_workflow.py -c "$PACKING_CONFIG"
        echo ""
    fi
fi

# -------------------------  analysis  ---------------------------------------
if [[ -z "$SKIP_ANALYSIS" ]]; then
    echo ">>> Running analysis ..."
    echo "python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \\"
    echo "    --config_file $ANALYSIS_CONFIG --log_level $LOG_LEVEL"
    python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \
        --config_file "$ANALYSIS_CONFIG" \
        --log_level "$LOG_LEVEL"
    echo ""
fi

echo "=== Workflow complete ==="
