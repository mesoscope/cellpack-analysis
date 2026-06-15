#!/usr/bin/env bash
# ============================================================================
# submit_replicate_99_slurm.sh
#
# Submits the 99-replicate ("replicate_99_v2") cellPACK packing run for BOTH
# structures (peroxisome + endosome) over the 8d-sphere cell set, using the
# four standard rules (random, nucleus_gradient, membrane_gradient,
# apical_gradient).  Calls the generic launcher `submit_packing_slurm.sh` once
# per workflow config.  All Python work runs on compute nodes.
#
# ---------------------------------------------------------------------------
# HOW THE WORK PARALLELIZES (read before tuning resources)
# ---------------------------------------------------------------------------
# Three nested layers:
#   1. SLURM job array        -- MAX_JOBS array tasks run concurrently.
#   2. Worker process pool    -- within a task, a ProcessPoolExecutor with
#                                CPUS workers packs CPUS recipes at once
#                                (num_processes == --slurm-cpus-per-task).
#   3. cellPACK per recipe    -- each recipe = ONE `pack.py` subprocess that
#                                runs all 99 seeds SEQUENTIALLY on ONE core
#                                (grid loaded once, reused).  The 99 replicates
#                                add NO parallel width -- they make each recipe
#                                ~99x longer.
#
#   IMPORTANT: cellPACK has its own seed-level multiprocessing, gated by the
#   config key "parallel".  Our config templates set "parallel": false on
#   purpose so cellPACK does NOT spawn its own pool (it ignores the SLURM
#   allocation and would use floor(0.8*node_cpus) procs PER recipe, badly
#   oversubscribing layer 2).  Do not set "parallel": true for SLURM runs.
#
# Wall-time per task ~= ceil(BATCH_SIZE / CPUS) waves x (per-recipe time).
# With BATCH_SIZE == CPUS that is exactly one wave (task time ~= one recipe).
#
# FOOTPRINT: this wrapper launches BOTH structures, and their worker arrays
# run CONCURRENTLY (the launcher submits its orchestrator and returns).  So
# total concurrent cores ~= 2 * MAX_JOBS * CPUS.  Defaults below
# (CPUS=16, MAX_JOBS=2) => 2*2*16 = 64 logical CPUs ~= 2 aics nodes.
# Use `-s peroxisome|endosome` to run one structure, or raise --max-jobs to
# go faster (each +1 max-jobs ~= +16 logical CPUs per structure).
# ---------------------------------------------------------------------------
#
# Usage:
#   bash submit_replicate_99_slurm.sh [OPTIONS]
#
# Options (forwarded to submit_packing_slurm.sh):
#   -s, --structure   peroxisome | endosome | both        (default: both)
#   -b, --batch-size  Recipes per worker SLURM job         (default: 16)
#   --cpus            CPUs per worker task (recipes/task)   (default: 16)
#   -m, --mem         Memory per worker job                (default: 32G)
#   -t, --time        Wall-clock limit per worker job       (default: 02:00:00)
#   --max-jobs        Max concurrent tasks PER STRUCTURE    (default: 2)
#   -p, --partition   SLURM partition                      (default: aics)
#   -v, --venv        Path to virtualenv activate script   (auto-detected)
#   --dry-run         Generate scripts/manifests but do NOT submit workers
#   --no-wait         Submit and return immediately (aggregate later)
#   -h, --help        Show this help message
#
# NOTE: keep BATCH_SIZE a multiple of CPUS so cores are not left idle.
#       MEM/TIME defaults are calibrated for ~99 sequential packings per
#       recipe; re-tune if you change CPUS/BATCH_SIZE or the cell set.
#
# Examples:
#   # Dry-run both structures (no jobs submitted) -- recommended first step
#   bash cellpack_analysis/packing/submit_replicate_99_slurm.sh --dry-run
#
#   # Submit both structures with default ~2-node footprint
#   bash cellpack_analysis/packing/submit_replicate_99_slurm.sh
#
#   # Only peroxisome, go faster (max 4 tasks)
#   bash cellpack_analysis/packing/submit_replicate_99_slurm.sh -s peroxisome --max-jobs 4
#
# Monitoring:
#   squeue -u $USER --name=cellpack
#   scancel -u $USER --name=cellpack
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/submit_packing_slurm.sh"
CONFIG_DIR="${SCRIPT_DIR}/configs/replicate_99_v2"

# ---- tuned defaults for the 99-replicate workload on the aics partition ----
STRUCTURE="both"
BATCH_SIZE=16          # recipes per task (== CPUS => 1 wave per task)
CPUS=16                # recipes packed concurrently per task (1 core each)
MEM="32G"              # per task; calibrated: ~0.8GB/proc peak x16 + headroom
TIME="02:00:00"        # per task; calibrated: worst recipe ~14-34min/99-seeds + margin
MAX_JOBS=2             # PER STRUCTURE; both run => ~2*2*16 = 64 CPUs ~= 2 nodes
PARTITION="aics"
VENV=""
DRY_RUN=""
NO_WAIT=""

usage() { sed -n '2,/^# ===/s/^# //p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--structure)   STRUCTURE="$2"; shift 2 ;;
        -b|--batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --cpus)           CPUS="$2"; shift 2 ;;
        -m|--mem)         MEM="$2"; shift 2 ;;
        -t|--time)        TIME="$2"; shift 2 ;;
        --max-jobs)       MAX_JOBS="$2"; shift 2 ;;
        -p|--partition)   PARTITION="$2"; shift 2 ;;
        -v|--venv)        VENV="$2"; shift 2 ;;
        --dry-run)        DRY_RUN="--dry-run"; shift ;;
        --no-wait)        NO_WAIT="--no-wait"; shift ;;
        -h|--help)        usage 0 ;;
        *)                echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

case "$STRUCTURE" in
    peroxisome) CONFIGS=("${CONFIG_DIR}/peroxisome_replicate_99_v2.json") ;;
    endosome)   CONFIGS=("${CONFIG_DIR}/endosome_replicate_99_v2.json") ;;
    both)       CONFIGS=("${CONFIG_DIR}/peroxisome_replicate_99_v2.json" \
                         "${CONFIG_DIR}/endosome_replicate_99_v2.json") ;;
    *)          echo "Error: --structure must be peroxisome | endosome | both" >&2; exit 1 ;;
esac

# Build the flag list forwarded to the generic launcher.
LAUNCHER_ARGS=(-b "$BATCH_SIZE" --cpus "$CPUS" -m "$MEM" -t "$TIME" \
               --max-jobs "$MAX_JOBS" -p "$PARTITION")
[[ -n "$VENV"    ]] && LAUNCHER_ARGS+=(-v "$VENV")
[[ -n "$DRY_RUN" ]] && LAUNCHER_ARGS+=("$DRY_RUN")
[[ -n "$NO_WAIT" ]] && LAUNCHER_ARGS+=("$NO_WAIT")

echo "Resource plan (per worker task): cpus=$CPUS mem=$MEM time=$TIME batch=$BATCH_SIZE"
echo "Concurrency: max-jobs=$MAX_JOBS per structure x ${#CONFIGS[@]} structure(s)"
echo "  => up to $(( MAX_JOBS * CPUS * ${#CONFIGS[@]} )) logical CPUs concurrently on '$PARTITION'"
echo

for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "$cfg" ]]; then
        echo "Error: config not found: $cfg" >&2
        exit 1
    fi
    echo "=========================================================================="
    echo "Submitting replicate_99_v2 packing run for: $(basename "$cfg")"
    echo "=========================================================================="
    bash "$LAUNCHER" -c "$cfg" "${LAUNCHER_ARGS[@]}"
done
