#!/usr/bin/env bash
# ============================================================================
# rule_interpolation_workflow.sh
#
# End-to-end rule interpolation workflow launcher.
#
# Phases:
#   fit       — Load component packings, compute occupancy, run CV, generate
#               a single aggregated mixed-rule packing config.
#   pack      — Submit the mixed-rule packing config via SLURM (or locally).
#   validate  — Run orthogonal distance + occupancy comparisons including the
#               mixed rule.
#   all       — Run all three phases sequentially (requires --local for pack).
#
# Usage:
#   bash rule_interpolation_workflow.sh -c <config> --phase <fit|pack|validate|all> [OPTIONS]
#
# Required:
#   -c, --config       Path to the unified e2e JSON config file
#
# Options:
#   --phase PHASE      Phase to run: fit, pack, validate, all (default: fit)
#   --local            Run packing locally instead of via SLURM
#   --dry-run          Print actions without executing
#   --log-level LEVEL  Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
#   -h, --help         Show this help message
#
# Examples:
#   # Fit (generate packing config from CV)
#   bash rule_interpolation_workflow.sh \
#       -c cellpack_analysis/analysis/workflows/configs/rule_interpolation/rule_interpolation_e2e_peroxisome.json \
#       --phase fit
#
#   # Submit mixed-rule packing to SLURM
#   bash rule_interpolation_workflow.sh \
#       -c cellpack_analysis/analysis/workflows/configs/rule_interpolation/rule_interpolation_e2e_peroxisome.json \
#       --phase pack
#
#   # Validate after packing completes
#   bash rule_interpolation_workflow.sh \
#       -c cellpack_analysis/analysis/workflows/configs/rule_interpolation/rule_interpolation_e2e_peroxisome.json \
#       --phase validate
#
#   # Run everything locally (small test)
#   bash rule_interpolation_workflow.sh \
#       -c cellpack_analysis/analysis/workflows/configs/rule_interpolation/rule_interpolation_e2e_peroxisome.json \
#       --phase all --local
# ============================================================================
set -euo pipefail

# -------------------------  defaults  ----------------------------------------
CONFIG=""
PHASE="fit"
LOCAL=""
DRY_RUN=""
LOG_LEVEL="INFO"

# -------------------------  parse args  --------------------------------------
usage() {
    sed -n '2,/^# ===/s/^# //p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)      CONFIG="$2";      shift 2 ;;
        --phase)          PHASE="$2";       shift 2 ;;
        --local)          LOCAL="--local";  shift ;;
        --dry-run)        DRY_RUN="--dry_run"; shift ;;
        --log-level)      LOG_LEVEL="$2";   shift 2 ;;
        -h|--help)        usage 0 ;;
        *)                echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# -------------------------  validate  ----------------------------------------
if [[ -z "$CONFIG" ]]; then
    echo "Error: --config (-c) is required." >&2
    usage 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG" >&2
    exit 1
fi

case "$PHASE" in
    fit|pack|validate|all) ;;
    *) echo "Error: Unknown phase '$PHASE'. Must be: fit, pack, validate, or all." >&2; exit 1 ;;
esac

if [[ "$PHASE" == "all" && -z "$LOCAL" ]]; then
    echo "Error: --phase all requires --local (SLURM packing is async)." >&2
    echo "       Run phases individually for SLURM: --phase fit, then --phase pack, then --phase validate." >&2
    exit 1
fi

# -------------------------  locate script  -----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_SCRIPT="$SCRIPT_DIR/run_rule_interpolation_workflow.py"

if [[ ! -f "$WORKFLOW_SCRIPT" ]]; then
    echo "Error: Python orchestrator not found: $WORKFLOW_SCRIPT" >&2
    exit 1
fi

# -------------------------  run  ---------------------------------------------
echo "=== Rule Interpolation E2E Workflow ==="
echo "Config:    $CONFIG"
echo "Phase:     $PHASE"
echo "Local:     ${LOCAL:+yes}${LOCAL:-no}"
echo "Dry run:   ${DRY_RUN:+yes}${DRY_RUN:-no}"
echo "Log level: $LOG_LEVEL"
echo ""

python "$WORKFLOW_SCRIPT" \
    -c "$CONFIG" \
    --phase "$PHASE" \
    --log_level "$LOG_LEVEL" \
    $LOCAL $DRY_RUN

echo ""
echo "=== Done ==="
