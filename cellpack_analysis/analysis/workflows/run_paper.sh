#!/usr/bin/env bash
# ============================================================================
# run_paper.sh
#
# Regenerates the analyses of the Protein Science paper from the released
# input data.  One stage per call, in the order below; `analysis` runs every
# analysis stage in order.  Packing stages are optional (the analysis uses
# the published packings) and submit SLURM jobs.
#
# ---------------------------------------------------------------------------
# STAGES (figure / table numbers as published)
# ---------------------------------------------------------------------------
#   fetch          download + verify the input archives into data/
#   calibration    envelope calibration (Methods rates); writes the distance
#                  cache the next stage reuses, so it runs first
#   distance       Fig 3a-d, S5, S6b-c, S7b-c; single-rule rows of Tables S2-S3
#   distance-supp  Fig S4
#   occupancy      Fig 4d (per-rule curves); single-rule arms of Fig 5c, Table S4
#   fit            Fig 4d (targets), S8a/c, S9a/c, mixing weights; writes the
#                  mixed-rule packing and validation configs
#   validate       Fig 5b-c, S6a, S7a, S8b/d, S9b/d; mixed-rule rows of
#                  Tables S2-S4
#   ladder         Fig S10
#   biovar         Fig 2b-e, S3a/b/d, Table S1, and the caches behind 2f, S3c/e
#   illustrations  Fig 4a-c, 5a, Tables 1-2, Fig S1-S2 (notebooks)
#   manuscript     drop-in figure panels (incl. Fig 2f, S3c/e), numbers sheets,
#                  calibration summary
#
#   analysis       calibration .. manuscript, in order
#
# Optional packing (SLURM; reproduces the published packings):
#   pack           stage-1 packings, 4 rules x 2 structures x 99 seeds
#   stage2-config  copy the fit's mixed-rule packing config into place
#   pack2          mixed-rule packings (after fit + stage2-config)
#
# Usage:
#   bash cellpack_analysis/analysis/workflows/run_paper.sh <stage>
# Env overrides: STRUCTS (default "peroxisome endosome").
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONDITION="replicate_99_v4"
STRUCTS="${STRUCTS:-peroxisome endosome}"
PY="${REPO}/.venv/bin/python"
CFG="${SCRIPT_DIR}/configs"
NB="${REPO}/cellpack_analysis/analysis/notebooks"
PACK_CFG="${REPO}/cellpack_analysis/packing/configs/${CONDITION}"

usage() { sed -n '2,/^# ===/s/^# \?//p' "$0"; }
case "${1:-}" in ""|-h|--help) usage; exit 0 ;; esac

[[ -x "$PY" ]] || { echo "Error: run 'uv sync --frozen' first ($PY not found)" >&2; exit 1; }
cd "$REPO"
unset CELLPACK PYTHONPATH  # the engine is the pinned venv copy

banner() { printf '\n========================================\n%s\n========================================\n' "$1"; }
run()    { echo "+ $*"; "$@"; }
analysis() { run "$PY" -m cellpack_analysis.analysis.workflows.run_analysis_workflow -c "$1"; }
interp()   { run "$PY" -m cellpack_analysis.analysis.workflows.run_rule_interpolation_workflow \
                 -c "$1" --phase "$2"; }

stage_fetch() {
    banner "Fetch input data"
    run "$PY" -m cellpack_analysis.data_release.paper_inputs fetch --groups all
}

stage_calibration() {
    banner "Envelope calibration (Methods)"
    for s in $STRUCTS; do
        analysis "${CFG}/distance_analysis/distance_analysis_${s}_${CONDITION}_calibration.json"
    done
}

stage_distance() {
    banner "Distance analysis: Fig 3, S5, S6b-c, S7b-c, Table S2"
    for s in $STRUCTS; do
        analysis "${CFG}/distance_analysis/distance_analysis_${s}_${CONDITION}.json"
    done
}

stage_distance_supp() {
    banner "Distance analysis: Fig S4"
    for s in $STRUCTS; do
        analysis "${CFG}/distance_analysis/distance_analysis_supp_${s}_${CONDITION}.json"
    done
}

stage_occupancy() {
    banner "Occupancy analysis: Fig 4d per-rule curves"
    for s in $STRUCTS; do
        analysis "${CFG}/occupancy_analysis/occupancy_analysis_${s}_${CONDITION}.json"
    done
}

stage_fit() {
    banner "NNLS fit + CV: Fig 4d targets, S8a/c, S9a/c"
    for s in $STRUCTS; do
        interp "${CFG}/rule_interpolation_${CONDITION}/rule_interpolation_${CONDITION}_${s}.json" fit
    done
}

stage_validate() {
    banner "Mixed-rule validation: Fig 5b-c, S6a, S7a, S8b/d, S9b/d, Tables S3-S4"
    for s in $STRUCTS; do
        interp "${CFG}/rule_interpolation_${CONDITION}/rule_interpolation_${CONDITION}_${s}.json" validate
    done
}

stage_ladder() {
    banner "Decay-length sensitivity: Fig S10"
    for s in $STRUCTS; do
        interp "${CFG}/rule_interpolation/rule_interpolation_tune_gradients_v2_${s}.json" fit
    done
}

stage_biovar() {
    banner "Sources of variation: Fig 2b-f, S3, Table S1"
    analysis "${CFG}/biological_variation/biological_variation_config.json"
}

stage_illustrations() {
    banner "Illustrations and descriptive tables: Fig 4a-c, 5a, Tables 1-2, S1-S2"
    for nb in cell_metrics_viz structure_size_and_count_viz grid_points_illustration \
              available_distance_distribution_viz occupancy_analysis_workflow; do
        MPLBACKEND=Agg run "$PY" "${NB}/${nb}.py"
    done
}

stage_manuscript() {
    banner "Manuscript panels and numbers"
    run "$PY" -m cellpack_analysis.analysis.workflows.run_manuscript_workflow \
        -c "${CFG}/manuscript/manuscript_${CONDITION}.json"
}

stage_analysis() {
    stage_calibration
    stage_distance
    stage_distance_supp
    stage_occupancy
    stage_fit
    stage_validate
    stage_ladder
    stage_biovar
    stage_illustrations
    stage_manuscript
}

stage_pack() {
    banner "Stage-1 packings (SLURM)"
    run bash "${REPO}/cellpack_analysis/packing/submit_replicate_99_slurm.sh" \
        --condition "$CONDITION"
}

stage_stage2_config() {
    banner "Stage-2 packing config (from the fit)"
    for s in $STRUCTS; do
        src="${REPO}/results/rule_interpolation_${CONDITION}/${s}/mixed_rule_packing_configs/mixed_rule_joint_aggregated.json"
        [[ -f "$src" ]] || { echo "Error: run 'fit' first: $src" >&2; exit 1; }
        run cp "$src" "${PACK_CFG}/${s}_${CONDITION}_interpolated.json"
    done
}

stage_pack2() {
    banner "Stage-2 mixed-rule packings (SLURM)"
    run bash "${REPO}/cellpack_analysis/packing/submit_replicate_99_slurm.sh" \
        --condition "$CONDITION" --suffix _interpolated -b 56
}

case "${1:-}" in
    fetch)          stage_fetch ;;
    calibration)    stage_calibration ;;
    distance)       stage_distance ;;
    distance-supp)  stage_distance_supp ;;
    occupancy)      stage_occupancy ;;
    fit)            stage_fit ;;
    validate)       stage_validate ;;
    ladder)         stage_ladder ;;
    biovar)         stage_biovar ;;
    illustrations)  stage_illustrations ;;
    manuscript)     stage_manuscript ;;
    analysis)       stage_analysis ;;
    pack)           stage_pack ;;
    stage2-config)  stage_stage2_config ;;
    pack2)          stage_pack2 ;;
    *) usage; exit 1 ;;
esac
