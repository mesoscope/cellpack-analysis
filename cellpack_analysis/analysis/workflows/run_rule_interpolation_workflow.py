#!/usr/bin/env python3
"""End-to-end rule interpolation workflow.

Orchestrates the full cycle: **fit** component-rule occupancy curves with
cross-validated NNLS → **pack** the derived mixed rule via SLURM → **validate**
the mixed-rule packings with orthogonal distance-based and occupancy-based
comparisons.

All three phases are driven by a single JSON config file.  The fit phase
generates intermediate artifacts (a packing config and a validation analysis
config) that the subsequent phases consume.

Phases
------
``fit``
    Load component packing outputs, compute occupancy ratios, run k-fold CV,
    generate an aggregated mixed-rule packing config and a validation analysis
    config.
``pack``
    Submit the mixed-rule packing config to SLURM (or run locally).
``validate``
    Re-run the analysis workflow with ``"interpolated"`` added to the channel
    map so that *all* standard distance-based and occupancy-based comparisons
    include the mixed rule — ensuring orthogonal (non-occupancy) metrics.

Usage::

    # Run a single phase
    python run_rule_interpolation_workflow.py -c config.json --phase fit
    python run_rule_interpolation_workflow.py -c config.json --phase pack
    python run_rule_interpolation_workflow.py -c config.json --phase validate

    # Run all phases sequentially (local packing, no SLURM)
    python run_rule_interpolation_workflow.py -c config.json --phase all --local
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from cellpack_analysis import setup_logging
from cellpack_analysis.analysis.workflows.run_analysis_workflow import (
    AnalysisConfig,
    AnalysisRunner,
)
from cellpack_analysis.lib import rule_interpolation
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.io import format_time

logger = logging.getLogger(__name__)

# Analysis steps used internally by the fit phase.
_FIT_ANALYSIS_STEPS = [
    "load_common_data",
    "calculate_distances",
    "run_occupancy_analysis",
    "run_rule_interpolation_cv",
]

# Default validation analysis steps when not specified in the config.
_DEFAULT_VALIDATION_STEPS = [
    "load_common_data",
    "calculate_distances",
    "run_emd_analysis",
    "run_ks_analysis",
    "run_pairwise_envelope_test",
    "run_occupancy_analysis",
    "run_occupancy_emd_analysis",
    "run_occupancy_ks_analysis",
    "run_occupancy_pairwise_envelope_test",
]

_DEFAULT_VALIDATION_DISTANCE_MEASURES = ["nearest", "pairwise", "nucleus", "z"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_e2e_config(config_path: str | Path) -> dict[str, Any]:
    """Load and return the raw JSON config dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as fh:
        return json.load(fh)


def _resolve_results_dir(raw_config: dict[str, Any]) -> Path:
    """Derive the results directory from the raw config (mirrors AnalysisConfig)."""
    project_root = get_project_root()
    base_results_dir = project_root / "results"
    output_path = raw_config.get("output_path")
    if output_path:
        return base_results_dir / output_path
    return base_results_dir / raw_config["name"] / raw_config.get("packing_id", "default")


def _packing_config_dir(results_dir: Path) -> Path:
    return results_dir / "mixed_rule_packing_configs"


def _aggregated_config_path(results_dir: Path, scope: str = "joint") -> Path:
    return _packing_config_dir(results_dir) / f"mixed_rule_{scope}_aggregated.json"


def _validation_config_path(results_dir: Path) -> Path:
    return results_dir / "mixed_rule_validation_config.json"


# ---------------------------------------------------------------------------
# Fit phase
# ---------------------------------------------------------------------------

def _build_fit_config_dict(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Build a transient config dict for the fit phase AnalysisRunner.

    Starts from the user's e2e config, overrides ``analysis_steps`` and
    ensures ``generate_packing_configs`` is **disabled** (we generate the
    aggregated config ourselves after the runner finishes).
    """
    fit_config = dict(raw_config)
    fit_config["analysis_steps"] = list(_FIT_ANALYSIS_STEPS)

    # Ensure CV runs but does NOT generate per-fold configs (we do aggregated-only).
    cv_params = dict(fit_config.get("rule_interpolation_cv_params", {}))
    cv_params["generate_packing_configs"] = False
    fit_config["rule_interpolation_cv_params"] = cv_params

    return fit_config


def _write_validation_analysis_config(
    raw_config: dict[str, Any],
    results_dir: Path,
) -> Path:
    """Auto-generate a validation analysis config JSON for the validate phase.

    Adds ``"interpolated"`` to the channel map and sets the analysis steps to
    the validation steps specified in the e2e config.
    """
    structure_id = raw_config.get("structure_id", "")
    packing_id = raw_config.get("packing_id", "default")

    # Build channel_map with "interpolated" added.
    channel_map: dict[str, str] = dict(raw_config.get("channel_map", {}))
    if structure_id:
        channel_map.setdefault("interpolated", structure_id)

    validation_steps = raw_config.get("validation_analysis_steps", _DEFAULT_VALIDATION_STEPS)
    validation_distance_measures = raw_config.get(
        "validation_distance_measures", _DEFAULT_VALIDATION_DISTANCE_MEASURES
    )

    validation_config: dict[str, Any] = {
        "name": f"{raw_config.get('name', 'rule_interpolation')}_validation",
        "structure_id": structure_id,
        "packing_id": packing_id,
        "structure_name": raw_config.get("structure_name", packing_id),
        "packing_output_folder": raw_config.get("packing_output_folder", ""),
        "channel_map": channel_map,
        "distribution_method": raw_config.get("distribution_method", "kde"),
        "distance_measures": validation_distance_measures,
        "occupancy_distance_measures": raw_config.get(
            "occupancy_distance_measures", ["nucleus", "z"]
        ),
        "occupancy_params": raw_config.get("occupancy_params", {}),
        "analysis_steps": validation_steps,
        "recalculate": raw_config.get("recalculate", False),
    }

    # Use the same output_path so results go to the same directory tree.
    output_path = raw_config.get("output_path")
    if output_path:
        validation_config["output_path"] = output_path

    out_path = _validation_config_path(results_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(validation_config, fh, indent=4)

    logger.info("Wrote validation analysis config: %s", out_path)
    return out_path


def fit(config_path: str, dry_run: bool = False) -> None:
    """Phase 1: Fit component-rule occupancy and generate packing config.

    1. Run ``AnalysisRunner`` with the fit-phase steps (load data, distances,
       occupancy, rule-interpolation CV).
    2. Generate a single aggregated mixed-rule packing config.
    3. Write a validation analysis config for Phase 3.
    """
    raw_config = _load_e2e_config(config_path)
    results_dir = _resolve_results_dir(raw_config)
    results_dir.mkdir(parents=True, exist_ok=True)
    project_root = get_project_root()
    scope = raw_config.get("packing_config_scope", "joint")

    # --- Step 1: run AnalysisRunner for fit steps ---
    fit_config_dict = _build_fit_config_dict(raw_config)

    # Write a temporary config JSON for AnalysisRunner.
    tmp_fit_config_path = results_dir / "_fit_phase_config.json"
    with open(tmp_fit_config_path, "w") as fh:
        json.dump(fit_config_dict, fh, indent=4)

    logger.info("=== Phase 1: Fit ===")
    runner = AnalysisRunner()
    runner.run_analysis(str(tmp_fit_config_path))

    # --- Step 2: generate aggregated packing config ---
    cv_result = runner.shared_data.get("rule_interpolation_cv_result")
    if cv_result is None:
        raise RuntimeError(
            "CV result not found in runner shared_data. "
            "Ensure 'run_rule_interpolation_cv' completed successfully."
        )

    base_packing_config_path = raw_config.get("base_packing_config_path")
    if base_packing_config_path is None:
        raise ValueError("'base_packing_config_path' must be set in the config.")

    base_packing_config_path = project_root / base_packing_config_path
    mode_to_gradient_name = raw_config.get("mode_to_gradient_name", {})

    output_config_dir = _packing_config_dir(results_dir)
    written = rule_interpolation.generate_mixed_rule_packing_configs(
        cv_result=cv_result,
        base_config_path=base_packing_config_path,
        output_config_dir=output_config_dir,
        mode_to_gradient_name=mode_to_gradient_name,
        scope=scope,
        aggregated_only=True,
        dry_run=dry_run,
    )

    if written:
        logger.info("Generated aggregated packing config: %s", written[0])
    elif dry_run:
        logger.info("[dry-run] Would generate aggregated packing config in %s", output_config_dir)

    # --- Step 3: write validation analysis config ---
    val_config_path = _write_validation_analysis_config(raw_config, results_dir)

    # --- Summary ---
    agg_path = _aggregated_config_path(results_dir, scope)
    logger.info("")
    logger.info("Fit phase complete.")
    logger.info("  Packing config:    %s", agg_path)
    logger.info("  Validation config: %s", val_config_path)
    logger.info("")
    logger.info("Next: run --phase pack to submit packing jobs.")


# ---------------------------------------------------------------------------
# Pack phase
# ---------------------------------------------------------------------------

def pack(
    config_path: str,
    local: bool = False,
    dry_run: bool = False,
) -> None:
    """Phase 2: Submit the mixed-rule packing to SLURM (or run locally).

    Locates the aggregated packing config generated by the fit phase and
    submits it via ``trigger_packing_workflow``.
    """
    raw_config = _load_e2e_config(config_path)
    results_dir = _resolve_results_dir(raw_config)
    scope = raw_config.get("packing_config_scope", "joint")
    slurm_kwargs = raw_config.get("slurm_kwargs", {})

    agg_path = _aggregated_config_path(results_dir, scope)
    if not agg_path.exists():
        raise FileNotFoundError(
            f"Aggregated packing config not found: {agg_path}\n"
            "Run --phase fit first to generate it."
        )

    logger.info("=== Phase 2: Pack ===")
    logger.info("Packing config: %s", agg_path)

    if dry_run:
        logger.info("[dry-run] Would submit packing for: %s", agg_path)
        logger.info("[dry-run] use_slurm=%s, slurm_kwargs=%s", not local, slurm_kwargs)
    else:
        rule_interpolation.trigger_packing_workflow(
            config_paths=[agg_path],
            use_slurm=not local,
            slurm_kwargs=slurm_kwargs if slurm_kwargs else None,
        )

    if not local:
        logger.info("")
        logger.info("SLURM packing jobs submitted (async).")
        logger.info("After all jobs complete, run --phase validate.")
    else:
        logger.info("")
        logger.info("Local packing complete.")


# ---------------------------------------------------------------------------
# Validate phase
# ---------------------------------------------------------------------------

def validate(config_path: str) -> None:
    """Phase 3: Validate mixed-rule packings with orthogonal comparisons.

    Runs the analysis workflow with ``"interpolated"`` in the channel map,
    producing distance-based (EMD, KS, envelope) and occupancy-based
    comparisons across all packing modes including the mixed rule.
    """
    raw_config = _load_e2e_config(config_path)
    results_dir = _resolve_results_dir(raw_config)

    val_config_path = _validation_config_path(results_dir)
    if not val_config_path.exists():
        raise FileNotFoundError(
            f"Validation config not found: {val_config_path}\n"
            "Run --phase fit first to generate it."
        )

    logger.info("=== Phase 3: Validate ===")
    logger.info("Validation config: %s", val_config_path)

    runner = AnalysisRunner()
    runner.run_analysis(str(val_config_path))

    # If raw distance data is available, run the unified validation that
    # includes both occupancy-based AND distance-based orthogonal tests.
    all_distance_dict = runner.shared_data.get("all_distance_dict")
    occupancy_dict = runner.shared_data.get("occupancy_dict")

    if occupancy_dict is not None:
        val_analysis_config = AnalysisConfig(str(val_config_path))
        packing_modes = val_analysis_config.packing_modes
        occupancy_dm = val_analysis_config.occupancy_distance_measures
        baseline_mode = val_analysis_config.baseline_mode

        logger.info("Running unified mixed-rule validation (occupancy + distance)")
        validation_result = rule_interpolation.run_mixed_rule_validation(
            combined_occupancy_dict=occupancy_dict,
            channel_map=val_analysis_config.channel_map,
            baseline_mode=baseline_mode,
            packing_modes=packing_modes,
            distance_measures=occupancy_dm,
            mixed_rule_distance_dict=all_distance_dict,
            results_dir=results_dir,
            recalculate=val_analysis_config.recalculate.get(
                "run_rule_interpolation_cv", False
            ),
        )

        # Log summary
        logger.info("Occupancy EMD summary:\n%s", validation_result.emd_df.to_string())
        logger.info("Occupancy KS test summary:\n%s", validation_result.ks_df.to_string())
        if validation_result.distance_emd_df is not None:
            logger.info(
                "Distance EMD summary:\n%s",
                validation_result.distance_emd_df.to_string(),
            )

    logger.info("")
    logger.info("Validation complete. Results saved to: %s", results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PHASES = ("fit", "pack", "validate", "all")


def main() -> None:
    """CLI entry point for the end-to-end rule interpolation workflow."""
    parser = argparse.ArgumentParser(
        description="End-to-end rule interpolation workflow: fit → pack → validate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Fit phase only
    python run_rule_interpolation_workflow.py -c config.json --phase fit

    # Submit packing to SLURM
    python run_rule_interpolation_workflow.py -c config.json --phase pack

    # Validate after packing completes
    python run_rule_interpolation_workflow.py -c config.json --phase validate

    # Run all phases locally (no SLURM)
    python run_rule_interpolation_workflow.py -c config.json --phase all --local
        """,
    )
    parser.add_argument(
        "--config_file", "-c",
        type=str,
        required=True,
        help="Path to the unified e2e JSON config file.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="fit",
        choices=_PHASES,
        help="Which phase to run (default: fit).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run packing locally instead of via SLURM.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print actions without executing packing or writing configs.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO).",
    )

    args = parser.parse_args()

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    setup_logging(level=level_map[args.log_level])

    start_time = time.time()
    phase = args.phase

    if phase in ("fit", "all"):
        fit(args.config_file, dry_run=args.dry_run)

    if phase in ("pack", "all"):
        pack(args.config_file, local=args.local, dry_run=args.dry_run)

    if phase in ("validate", "all"):
        validate(args.config_file)

    logger.info("Total time: %s", format_time(time.time() - start_time))


if __name__ == "__main__":
    main()
