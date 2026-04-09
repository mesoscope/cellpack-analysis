"""
Rule interpolation module for cellpack analysis.

Fits occupancy data from a baseline experimental mode as a non-negative linear
combination of simulated packing-rule occupancy curves (NNLS), with k-fold
cross-validation on baseline cell IDs.  Provides utilities for generating mixed-rule
packing configurations and running orthogonal validation tests.

Design principles
-----------------
- **Dict-agnostic**: only reads ``individual[cell_id]["xvals"]``,
  ``individual[cell_id]["occupancy"]``, ``combined["xvals"]``, and
  ``combined["occupancy"]`` — the same keys produced by both the KDE and
  discrete histogram pipelines.
- **Single fitting strategy**: the target vector ``b`` is the mean of a
  (possibly held-out) set of baseline cells' occupancy curves; the design matrix
  columns are the *full* combined occupancy of each simulated mode (never split).
  This separates the cell pool used to define the target from the simulated mode
  data, avoiding the main sources of overfitting.
- **CV split on baseline cells only**: folds are drawn from the baseline-mode
  ``individual`` dict; simulated modes always contribute their aggregate combined
  curve.
"""

import json
import logging
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from cellpack_analysis.lib import distance, occupancy
from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    get_project_root,
    remove_file_handler_from_logger,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Results from a single NNLS fitting run.

    Parameters
    ----------
    coefficients
        ``{mode: {dm: coeff, "joint": coeff}}``.  Per-distance-measure
        coefficients are keyed by distance-measure name; the joint coefficient
        is under the ``"joint"`` key.
    relative_contributions
        Normalised coefficients: ``{mode: {dm: fraction, "joint": fraction}}``.
    train_mse
        Per-distance-measure MSE: ``{"individual": {dm: mse}, "joint": {dm: mse}}``.
    train_cell_ids
        Baseline cell IDs used for this fit.
    packing_modes
        Ordered list of simulated packing modes (columns of ``A``).
    distance_measures
        Ordered list of distance measures used.
    reconstructed_occupancy
        Fitted occupancy curves: ``{dm: {"individual": array, "joint": array}}``.
    """

    coefficients: dict[str, dict[str, float]]
    relative_contributions: dict[str, dict[str, float]]
    train_mse: dict[str, dict[str, float]]
    train_cell_ids: list[str]
    packing_modes: list[str]
    distance_measures: list[str]
    reconstructed_occupancy: dict[str, dict[str, np.ndarray]]

    def __repr__(self) -> str:
        fmt_relative_contributions: dict[str, dict[str, str]] = {
            mode: {
                **{dm: f"{frac:.3f}" for dm, frac in data.items() if dm != "joint"},
                "joint": f"{data['joint']:.3f}",
            }
            for mode, data in self.relative_contributions.items()
        }
        fmt_train_mse = {
            scope: {dm: f"{mse:.4f}" for dm, mse in dm_dict.items()}
            for scope, dm_dict in self.train_mse.items()
        }
        return (
            f"FitResult(\n"
            f"    train_cells={len(self.train_cell_ids)},\n"
            f"    packing_modes={self.packing_modes},\n"
            f"    distance_measures={self.distance_measures},\n"
            f"    relative_contributions={fmt_relative_contributions},\n"
            f"    train_mse={fmt_train_mse},\n"
            f")"
        )

    @property
    def coefficients_individual(self) -> dict[str, dict[str, float]]:
        """Per-distance-measure coefficients: ``{dm: {mode: coeff}}``."""
        return {
            dm: {mode: self.coefficients[mode][dm] for mode in self.packing_modes}
            for dm in self.distance_measures
        }

    @property
    def coefficients_joint(self) -> dict[str, float]:
        """Joint-fit coefficients: ``{mode: coeff}``."""
        return {mode: self.coefficients[mode]["joint"] for mode in self.packing_modes}

    @property
    def relative_contributions_individual(self) -> dict[str, dict[str, float]]:
        """Per-distance-measure relative contributions: ``{dm: {mode: fraction}}``."""
        return {
            dm: {mode: self.relative_contributions[mode][dm] for mode in self.packing_modes}
            for dm in self.distance_measures
        }

    @property
    def relative_contributions_joint(self) -> dict[str, float]:
        """Joint-fit relative contributions: ``{mode: fraction}``."""
        return {mode: self.relative_contributions[mode]["joint"] for mode in self.packing_modes}

    @property
    def train_mse_individual(self) -> dict[str, float]:
        """Per-distance-measure training MSE: ``{dm: mse}``."""
        return self.train_mse["individual"]

    @property
    def train_mse_joint(self) -> dict[str, float]:
        """Joint-fit training MSE per distance measure: ``{dm: mse}``."""
        return self.train_mse["joint"]


@dataclass
class FoldResult:
    """Results for a single cross-validation fold.

    Parameters
    ----------
    fold_idx
        Zero-based fold index.
    train_cell_ids
        Baseline cell IDs in the training split.
    test_cell_ids
        Baseline cell IDs in the held-out test split.
    fit_result
        NNLS fit trained on ``train_cell_ids``.
    test_mse
        Per-cell MSE on the held-out baseline cells:
        ``{"individual": {dm: array}, "joint": : array}``.
    """

    fold_idx: int
    train_cell_ids: list[str]
    test_cell_ids: list[str]
    fit_result: FitResult
    test_mse: dict[str, dict[str, np.ndarray]]
    repeat_idx: int = 0


@dataclass
class CVResult:
    """Aggregated results from k-fold cross-validation.

    Parameters
    ----------
    folds
        Per-fold results.
    aggregated_coefficients
        Mean and std of coefficients across folds:
        ``{mode: {dm: (mean, std), "joint": (mean, std)}}``.  Per-distance-measure
        entries are keyed by distance-measure name; the joint entry is under ``"joint"``.
    mean_train_mse
        Mean training MSE across folds: ``{"individual": {dm: float}, "joint": {dm: float}}``.
    std_train_mse
        Std of training MSE across folds.
    mean_test_mse
        Mean test MSE across folds.
    std_test_mse
        Std of test MSE across folds.
    n_folds
        Number of folds used.
    baseline_mode
        The baseline packing mode used for fitting.
    """

    folds: list[FoldResult]
    aggregated_coefficients: dict[str, dict[str, tuple[float, float]]]
    mean_train_mse: dict[str, dict[str, float]]
    std_train_mse: dict[str, dict[str, float]]
    mean_test_mse: dict[str, dict[str, float]]
    std_test_mse: dict[str, dict[str, float]]
    n_folds: int
    baseline_mode: str
    n_repeats: int = 10

    @property
    def aggregated_coefficients_individual(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Per-distance-measure aggregated coefficients: ``{dm: {mode: (mean, std)}}``."""
        packing_modes = list(self.aggregated_coefficients.keys())
        if not packing_modes:
            return {}
        distance_measures = [
            k for k in self.aggregated_coefficients[packing_modes[0]] if k != "joint"
        ]
        return {
            dm: {mode: self.aggregated_coefficients[mode][dm] for mode in packing_modes}
            for dm in distance_measures
        }

    @property
    def aggregated_coefficients_joint(self) -> dict[str, tuple[float, float]]:
        """Joint-fit aggregated coefficients: ``{mode: (mean, std)}``."""
        return {mode: data["joint"] for mode, data in self.aggregated_coefficients.items()}


@dataclass
class ValidationResult:
    """Results from orthogonal validation of the mixed rule.

    Parameters
    ----------
    emd_df
        Pairwise Earth Mover's Distance on occupancy curves.
    ks_df
        Kolmogorov-Smirnov test results vs the baseline mode.
    envelope_test
        Pairwise Monte Carlo rank-envelope test on occupancy curves.
    distance_emd_df
        Pairwise EMD on distance distributions (only if packings were run).
    distance_envelope_test
        Pairwise envelope test on distance distributions (only if packings were run).
    distance_ks_df
        Per-cell KS test on distance distributions vs baseline (only if packings were run).
    distance_ks_bootstrap_df
        Bootstrapped KS similarity fractions on distance distributions
        (only if packings were run).
    """

    emd_df: pd.DataFrame
    ks_df: pd.DataFrame
    envelope_test: dict[str, Any]
    distance_emd_df: pd.DataFrame | None = None
    distance_envelope_test: dict[str, Any] | None = None
    distance_ks_df: pd.DataFrame | None = None
    distance_ks_bootstrap_df: pd.DataFrame | None = None
    aic_result: "AICComparisonResult | None" = None


@dataclass
class AICModelResult:
    """AIC/BIC statistics for a single candidate model.

    Parameters
    ----------
    model_name
        Human-readable label (e.g. ``"mixed_rule"``, ``"single:random"``, ``"null"``).
    k
        Number of free model parameters (excluding error-variance).
    n
        Number of data points (occupancy grid length, or stacked length for joint).
    rss
        Residual sum of squares.
    aic
        Akaike Information Criterion.
    aicc
        Small-sample corrected AIC.
    bic
        Bayesian Information Criterion.
    """

    model_name: str
    k: int
    n: int
    rss: float
    aic: float
    aicc: float
    bic: float


@dataclass
class AICComparisonResult:
    """Model comparison results for AIC/BIC across scopes and distance measures.

    Parameters
    ----------
    comparisons
        ``{scope: {dm: list[AICModelResult]}}``.
        *scope* is ``"individual"`` or ``"joint"``; *dm* is a distance-measure
        name (or ``"joint"`` for the stacked joint scope).
    delta_aic
        ``{scope: {dm: {model_name: float}}}``.  ΔAIC relative to the best model.
    delta_bic
        ``{scope: {dm: {model_name: float}}}``.  ΔBIC relative to the best model.
    akaike_weights
        ``{scope: {dm: {model_name: float}}}``.  Akaike weights (sum to 1).
    bic_weights
        ``{scope: {dm: {model_name: float}}}``.  BIC-based weights (sum to 1).
    best_model_aic
        ``{scope: {dm: str}}``.  Name of the best model per AIC.
    best_model_bic
        ``{scope: {dm: str}}``.  Name of the best model per BIC.
    """

    comparisons: dict[str, dict[str, list[AICModelResult]]]
    delta_aic: dict[str, dict[str, dict[str, float]]]
    delta_bic: dict[str, dict[str, dict[str, float]]]
    akaike_weights: dict[str, dict[str, dict[str, float]]]
    bic_weights: dict[str, dict[str, dict[str, float]]]
    best_model_aic: dict[str, dict[str, str]]
    best_model_bic: dict[str, dict[str, str]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_baseline_cell_ids(
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    baseline_mode: str,
) -> list[str]:
    """Return sorted list of baseline cell IDs consistent across distance measures.

    Parameters
    ----------
    occupancy_dict
        ``{distance_measure: {mode: {"individual": {cell_id: ...}, "combined": ...}}}``.
    baseline_mode
        Packing-mode key used as the experimental baseline.

    Returns
    -------
    :
        Sorted cell ID list drawn from the first distance measure;
        a warning is logged if the cell set differs across distance measures.
    """
    distance_measures = list(occupancy_dict.keys())
    if not distance_measures:
        raise ValueError("occupancy_dict is empty — no distance measures found.")

    first_dm = distance_measures[0]
    baseline_individual = occupancy_dict[first_dm].get(baseline_mode, {}).get("individual", {})
    if not baseline_individual:
        raise ValueError(
            f"Baseline mode '{baseline_mode}' has no 'individual' data in distance measure "
            f"'{first_dm}'."
        )

    cell_ids = sorted(baseline_individual.keys())
    logger.info(f"Found {len(cell_ids)} baseline cell IDs in distance measure '{first_dm}'.")

    for dm in distance_measures[1:]:
        dm_individual = occupancy_dict[dm].get(baseline_mode, {}).get("individual", {})
        dm_cell_ids = sorted(dm_individual.keys())
        if dm_cell_ids != cell_ids:
            logger.warning(
                f"Baseline cell IDs differ between distance measure '{first_dm}' "
                f"({len(cell_ids)} cells) and '{dm}' ({len(dm_cell_ids)} cells). "
                "Using the cell IDs from the first distance measure for CV splits."
            )

    return cell_ids


def _compute_mean_occupancy_from_cells(
    mode_individual_dict: dict[str, dict[str, np.ndarray]],
    cell_ids: list[str],
    common_xvals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean occupancy from a subset of cells interpolated onto a common grid.

    Parameters
    ----------
    mode_individual_dict
        ``{cell_id: {"xvals": array, "occupancy": array}}``.
    cell_ids
        Subset of cell IDs to use.
    common_xvals
        Target x-axis grid for interpolation.

    Returns
    -------
    mean_occupancy
        Shape ``(len(common_xvals),)``.
    per_cell_matrix
        Shape ``(n_cells, len(common_xvals))`` — each row is one cell's
        interpolated occupancy curve.
    """
    rows: list[np.ndarray] = []
    for cell_id in cell_ids:
        cell_data = mode_individual_dict.get(cell_id)
        if cell_data is None:
            logger.warning(f"Cell '{cell_id}' not found in individual dict; skipping.")
            continue
        interp = np.interp(
            common_xvals, cell_data["xvals"], cell_data["occupancy"], right=0.0, left=0.0
        )
        rows.append(interp)

    if not rows:
        raise ValueError(f"No valid cells found for the given cell_id subset ({cell_ids[:3]} ...)")

    per_cell_matrix = np.vstack(rows)
    mean_occupancy = np.nan_to_num(np.nanmean(per_cell_matrix, axis=0), nan=0.0)
    return mean_occupancy, per_cell_matrix


def _normalize_coefficients(
    coefficients: np.ndarray, packing_modes: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert a coefficient array to labelled dicts with relative contributions.

    Parameters
    ----------
    coefficients
        Raw NNLS coefficients, shape ``(n_modes,)``.
    packing_modes
        Mode labels corresponding to each coefficient.

    Returns
    -------
    coeff_dict
        ``{mode: raw_coeff}``.
    relative_dict
        ``{mode: normalised_fraction}`` (sum = 1, or all-zero if sum is 0).
    """
    coeff_dict = dict(zip(packing_modes, coefficients.tolist(), strict=True))
    total = float(np.sum(coefficients))
    if total > 0:
        relative_dict = {mode: float(c / total) for mode, c in coeff_dict.items()}
    else:
        relative_dict = dict.fromkeys(packing_modes, 0.0)
    return coeff_dict, relative_dict


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------


def fit_rule_interpolation(
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    channel_map: dict[str, str],
    baseline_mode: str,
    train_cell_ids: list[str] | None = None,
    distance_measures: list[str] | None = None,
) -> FitResult:
    """Fit a mixed packing rule as a non-negative linear combination of simulated modes.

    The target vector ``baseline_occupancy`` for each distance measure is the mean occupancy
    curve of the ``train_cell_ids`` baseline cells (interpolated onto the
    baseline combined ``xvals`` grid).  The design-matrix columns are each
    simulated mode's combined occupancy curve.

    Both per-distance-measure (individual) and joint (stacked across all
    distance measures) NNLS fits are performed.

    Parameters
    ----------
    occupancy_dict
        ``{distance_measure: {mode: {"individual": {cell_id: {"xvals", "occupancy", ...}},
        "combined": {"xvals", "occupancy", ...}}}}``.
        Compatible with both KDE and discrete histogram pipelines.
    channel_map
        ``{mode: structure_id}`` — used to identify which modes are simulated
        (i.e. all modes except ``baseline_mode`` and ``"interpolated"``).
    baseline_mode
        Key for the experimental baseline packing mode.
    train_cell_ids
        Baseline cell IDs to average into the target vector.  If ``None``,
        all available baseline cells are used.
    distance_measures
        Subset of distance measures to use.  Defaults to all keys in
        ``occupancy_dict``.

    Returns
    -------
    :
        :class:`FitResult` with individual and joint coefficients, MSEs, and
        reconstructed occupancy curves.
    """
    if distance_measures is None:
        distance_measures = list(occupancy_dict.keys())

    packing_modes = [
        mode for mode in channel_map.keys() if mode not in (baseline_mode, "interpolated")
    ]
    if not packing_modes:
        raise ValueError(
            f"No simulated packing modes found in channel_map after excluding "
            f"'{baseline_mode}' and 'interpolated'. channel_map keys: {list(channel_map)}"
        )

    if train_cell_ids is None:
        train_cell_ids = _get_baseline_cell_ids(occupancy_dict, baseline_mode)

    # Accumulators for joint fit
    stacked_baseline_occupancy: list[np.ndarray] = []
    stacked_simulated_occupancy: list[np.ndarray] = []

    _coefficients_individual: dict[str, dict[str, float]] = {}
    _relative_contributions_individual: dict[str, dict[str, float]] = {}
    train_mse_individual: dict[str, float] = {}
    reconstructed_occupancy: dict[str, dict[str, np.ndarray]] = {}

    for dm in distance_measures:
        dm_data = occupancy_dict.get(dm)
        if dm_data is None:
            raise KeyError(f"Distance measure '{dm}' not found in occupancy_dict.")

        baseline_dm = dm_data.get(baseline_mode)
        if baseline_dm is None:
            raise KeyError(f"Baseline mode '{baseline_mode}' not found in occupancy_dict['{dm}'].")

        common_xvals: np.ndarray = baseline_dm["combined"]["xvals"]

        # Build target: mean of training baseline cells on common_xvals
        baseline_occupancy, _ = _compute_mean_occupancy_from_cells(
            mode_individual_dict=baseline_dm["individual"],
            cell_ids=train_cell_ids,
            common_xvals=common_xvals,
        )

        # Build design matrix A: each column is a simulated mode's combined occupancy
        simulated_occupancy_cols: list[np.ndarray] = []
        for mode in packing_modes:
            mode_dm = dm_data.get(mode)
            if mode_dm is None:
                raise KeyError(f"Simulated mode '{mode}' not found in occupancy_dict['{dm}'].")
            mode_combined_xvals: np.ndarray = mode_dm["combined"]["xvals"]
            mode_combined_occ: np.ndarray = mode_dm["combined"]["occupancy"]
            # Interpolate simulated combined onto baseline common_xvals if grids differ
            if len(mode_combined_xvals) != len(common_xvals) or not np.allclose(
                mode_combined_xvals, common_xvals
            ):
                col = np.interp(
                    common_xvals, mode_combined_xvals, mode_combined_occ, right=0.0, left=0.0
                )
            else:
                col = mode_combined_occ
            col = np.nan_to_num(col, nan=0.0)
            simulated_occupancy_cols.append(col)

        simulated_occupancy_matrix = np.column_stack(
            simulated_occupancy_cols
        )  # shape (n_bins, n_modes)

        # Per-distance-measure NNLS
        coeffs_ind, _ = nnls(simulated_occupancy_matrix, baseline_occupancy)
        recon_ind = simulated_occupancy_matrix @ coeffs_ind
        mse_ind = float(np.mean((baseline_occupancy - recon_ind) ** 2))

        coeff_dict_ind, rel_dict_ind = _normalize_coefficients(coeffs_ind, packing_modes)
        _coefficients_individual[dm] = coeff_dict_ind
        _relative_contributions_individual[dm] = rel_dict_ind
        train_mse_individual[dm] = mse_ind
        reconstructed_occupancy[dm] = {"individual": recon_ind}

        stacked_baseline_occupancy.append(baseline_occupancy)
        stacked_simulated_occupancy.append(simulated_occupancy_matrix)

    # Joint NNLS across all distance measures
    baseline_occupancy_joint = np.concatenate(stacked_baseline_occupancy)
    simulated_occupancy_joint = np.vstack(stacked_simulated_occupancy)
    coeffs_joint, _ = nnls(simulated_occupancy_joint, baseline_occupancy_joint)
    coeff_dict_joint, rel_dict_joint = _normalize_coefficients(coeffs_joint, packing_modes)

    # Evaluate joint fit per distance measure
    train_mse_joint: dict[str, float] = {}
    idx = 0
    for i, dm in enumerate(distance_measures):
        n = len(stacked_baseline_occupancy[i])
        baseline_occupancy_dm = stacked_baseline_occupancy[i]
        simulated_occupancy_dm = stacked_simulated_occupancy[i]
        recon_joint = simulated_occupancy_dm @ coeffs_joint
        train_mse_joint[dm] = float(np.mean((baseline_occupancy_dm - recon_joint) ** 2))
        reconstructed_occupancy[dm]["joint"] = recon_joint
        idx += n

    # Build consolidated coefficients and relative_contributions
    coefficients: dict[str, dict[str, float]] = {
        mode: {
            **{dm: _coefficients_individual[dm][mode] for dm in distance_measures},
            "joint": coeff_dict_joint[mode],
        }
        for mode in packing_modes
    }
    relative_contributions: dict[str, dict[str, float]] = {
        mode: {
            **{dm: _relative_contributions_individual[dm][mode] for dm in distance_measures},
            "joint": rel_dict_joint[mode],
        }
        for mode in packing_modes
    }

    return FitResult(
        coefficients=coefficients,
        relative_contributions=relative_contributions,
        train_mse={"individual": train_mse_individual, "joint": train_mse_joint},
        train_cell_ids=list(train_cell_ids),
        packing_modes=packing_modes,
        distance_measures=list(distance_measures),
        reconstructed_occupancy=reconstructed_occupancy,
    )


def fit_result_from_cv(
    cv_result: CVResult,
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    channel_map: dict[str, str],
    baseline_mode: str,
    distance_measures: list[str] | None = None,
) -> FitResult:
    """Construct a :class:`FitResult` from the mean CV coefficients.

    Extracts the mean coefficients from ``cv_result.aggregated_coefficients``
    and reconstructs occupancy curves by applying them to the same design
    matrix used in :func:`fit_rule_interpolation`.  The returned
    :class:`FitResult` is compatible with
    :func:`~cellpack_analysis.lib.visualization.plot_rule_interpolation_fit`.

    MSE values are evaluated against all baseline cells' combined occupancy
    curve (i.e. ``occupancy_dict[dm][baseline_mode]["combined"]["occupancy"]``),
    so they are directly comparable to the full-data fit.

    Parameters
    ----------
    cv_result
        Output from :func:`run_rule_interpolation_cv`.
    occupancy_dict
        Same ``{distance_measure: {mode: {"individual": ..., "combined": ...}}}``
        dict passed to :func:`run_rule_interpolation_cv`.
    channel_map
        ``{mode: structure_id}`` — used to identify simulated modes.
    baseline_mode
        Key for the experimental baseline packing mode.
    distance_measures
        Subset of distance measures to use.  Defaults to all keys in
        ``occupancy_dict``.

    Returns
    -------
    :
        :class:`FitResult` whose ``reconstructed_occupancy``, ``coefficients``,
        ``relative_contributions``, and ``train_mse`` reflect the mean CV
        coefficients.
    """
    if distance_measures is None:
        distance_measures = list(occupancy_dict.keys())

    packing_modes = [
        mode for mode in channel_map.keys() if mode not in (baseline_mode, "interpolated")
    ]

    # Extract mean coefficients from aggregated CV results
    mean_coeffs_individual: dict[str, dict[str, float]] = {
        dm: {mode: cv_result.aggregated_coefficients[mode][dm][0] for mode in packing_modes}
        for dm in distance_measures
    }
    mean_coeffs_joint: dict[str, float] = {
        mode: cv_result.aggregated_coefficients[mode]["joint"][0] for mode in packing_modes
    }

    reconstructed_occupancy: dict[str, dict[str, np.ndarray]] = {}
    train_mse_individual: dict[str, float] = {}
    train_mse_joint: dict[str, float] = {}
    stacked_simulated_occupancy: list[np.ndarray] = []
    stacked_baseline_occupancy: list[np.ndarray] = []

    for dm in distance_measures:
        dm_data = occupancy_dict[dm]
        baseline_dm = dm_data[baseline_mode]
        common_xvals: np.ndarray = baseline_dm["combined"]["xvals"]

        # Reference target: all baseline cells' combined occupancy curve
        baseline_occ = np.nan_to_num(
            np.interp(
                common_xvals,
                baseline_dm["combined"]["xvals"],
                baseline_dm["combined"]["occupancy"],
            )
        )

        # Build design matrix (same construction as fit_rule_interpolation)
        simulated_occupancy_cols: list[np.ndarray] = []
        for mode in packing_modes:
            mode_combined_xvals: np.ndarray = dm_data[mode]["combined"]["xvals"]
            mode_combined_occ: np.ndarray = dm_data[mode]["combined"]["occupancy"]
            if len(mode_combined_xvals) != len(common_xvals) or not np.allclose(
                mode_combined_xvals, common_xvals
            ):
                col = np.interp(
                    common_xvals, mode_combined_xvals, mode_combined_occ, right=0.0, left=0.0
                )
            else:
                col = mode_combined_occ
            simulated_occupancy_cols.append(np.nan_to_num(col, nan=0.0))

        design_matrix = np.column_stack(simulated_occupancy_cols)  # (n_bins, n_modes)

        # Reconstruct using mean individual coefficients
        coeffs_ind = np.array([mean_coeffs_individual[dm][mode] for mode in packing_modes])
        recon_ind = design_matrix @ coeffs_ind
        train_mse_individual[dm] = float(np.mean((baseline_occ - recon_ind) ** 2))
        reconstructed_occupancy[dm] = {"individual": recon_ind}

        stacked_simulated_occupancy.append(design_matrix)
        stacked_baseline_occupancy.append(baseline_occ)

    # Reconstruct using mean joint coefficients (applied per-dm)
    coeffs_joint = np.array([mean_coeffs_joint[mode] for mode in packing_modes])
    for i, dm in enumerate(distance_measures):
        recon_joint = stacked_simulated_occupancy[i] @ coeffs_joint
        train_mse_joint[dm] = float(np.mean((stacked_baseline_occupancy[i] - recon_joint) ** 2))
        reconstructed_occupancy[dm]["joint"] = recon_joint

    # Build consolidated coefficients and relative_contributions
    coefficients: dict[str, dict[str, float]] = {
        mode: {
            **{dm: mean_coeffs_individual[dm][mode] for dm in distance_measures},
            "joint": mean_coeffs_joint[mode],
        }
        for mode in packing_modes
    }
    relative_contributions_individual: dict[str, dict[str, float]] = {}
    for dm in distance_measures:
        _, rel_dm = _normalize_coefficients(
            np.array([mean_coeffs_individual[dm][mode] for mode in packing_modes]),
            packing_modes,
        )
        relative_contributions_individual[dm] = rel_dm
    _, rel_dict_joint = _normalize_coefficients(coeffs_joint, packing_modes)

    relative_contributions: dict[str, dict[str, float]] = {
        mode: {
            **{dm: relative_contributions_individual[dm][mode] for dm in distance_measures},
            "joint": rel_dict_joint[mode],
        }
        for mode in packing_modes
    }

    # Collect all cell IDs seen across folds
    all_cell_ids: list[str] = sorted(
        {
            cell_id
            for fold in cv_result.folds
            for cell_id in fold.train_cell_ids + fold.test_cell_ids
        }
    )

    return FitResult(
        coefficients=coefficients,
        relative_contributions=relative_contributions,
        train_mse={"individual": train_mse_individual, "joint": train_mse_joint},
        train_cell_ids=all_cell_ids,
        packing_modes=packing_modes,
        distance_measures=list(distance_measures),
        reconstructed_occupancy=reconstructed_occupancy,
    )


def _evaluate_on_cells(
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    baseline_mode: str,
    test_cell_ids: list[str],
    fit_result: FitResult,
    distance_measures: list[str] | None = None,
    grouping: Literal["per-cell", "combined"] = "per-cell",
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate fitted occupancy against held-out baseline cells.

    For each test cell and each distance measure, the cell's occupancy curve
    is interpolated onto the common grid and compared against the fitted
    reconstruction (using both individual and joint coefficients).

    Parameters
    ----------
    occupancy_dict
        Same structure as passed to :func:`fit_rule_interpolation`.
    baseline_mode
        Baseline packing mode key.
    test_cell_ids
        Held-out baseline cell IDs.
    fit_result
        Trained :class:`FitResult` from :func:`fit_rule_interpolation`.
    distance_measures
        Subset of distance measures to evaluate.  Defaults to
        ``fit_result.distance_measures``.
    grouping
        Whether to compute test MSE on per cell occupancy or on the combined mean curve
        across test cells.
            If ``"per-cell"``, returns an array of MSE values for each test cell;
            if ``"combined"``, returns a single MSE value comparing the fit to the
            mean occupancy curve of the test cells.


    Returns
    -------
    :
        ``{"individual": {dm: per_cell_mse_array}, "joint": {dm: per_cell_mse_array}}``
        where each per_cell_mse_array has shape ``(n_test_cells,)``.
    """
    if distance_measures is None:
        distance_measures = fit_result.distance_measures

    test_mse: dict[str, dict[str, np.ndarray]] = {"individual": {}, "joint": {}}

    for dm in distance_measures:
        baseline_individual = occupancy_dict[dm][baseline_mode]["individual"]
        common_xvals: np.ndarray = occupancy_dict[dm][baseline_mode]["combined"]["xvals"]
        recon_ind: np.ndarray = fit_result.reconstructed_occupancy[dm]["individual"]
        recon_joint: np.ndarray = fit_result.reconstructed_occupancy[dm]["joint"]

        per_cell_mse_ind: list[float] = []
        per_cell_mse_joint: list[float] = []

        test_cell_occs: list[np.ndarray] = []
        for cell_id in test_cell_ids:
            cell_data = baseline_individual.get(cell_id)
            if cell_data is None:
                logger.warning(
                    f"Test cell '{cell_id}' not found in baseline individual dict "
                    f"for distance measure '{dm}'; skipping."
                )
                continue
            cell_occ = np.interp(
                common_xvals, cell_data["xvals"], cell_data["occupancy"], right=0.0, left=0.0
            )
            cell_occ = np.nan_to_num(cell_occ, nan=0.0)
            per_cell_mse_ind.append(float(np.mean((cell_occ - recon_ind) ** 2)))
            per_cell_mse_joint.append(float(np.mean((cell_occ - recon_joint) ** 2)))
            test_cell_occs.append(cell_occ)
        mean_cell_occ = (
            np.mean(test_cell_occs, axis=0) if test_cell_occs else np.zeros_like(common_xvals)
        )
        if grouping == "combined":
            mse_ind = float(np.mean((mean_cell_occ - recon_ind) ** 2))
            mse_joint = float(np.mean((mean_cell_occ - recon_joint) ** 2))
            test_mse["individual"][dm] = np.array([mse_ind])
            test_mse["joint"][dm] = np.array([mse_joint])
        else:
            test_mse["individual"][dm] = np.array(per_cell_mse_ind)
            test_mse["joint"][dm] = np.array(per_cell_mse_joint)

    return test_mse


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_rule_interpolation_coeffs(
    fit_result: FitResult,
    baseline_mode: str,
    file_path: Path | None = None,
) -> None:
    """Log interpolation coefficients and MSE from a :class:`FitResult`.

    Parameters
    ----------
    fit_result
        Output from :func:`fit_rule_interpolation`.
    baseline_mode
        Baseline packing mode (for display).
    file_path
        Optional path to write the log output to a file.
    """
    if file_path is not None:
        dist_logger = add_file_handler_to_logger(logger, file_path)
    else:
        dist_logger = logger

    dist_logger.info(f"Baseline mode: {baseline_mode}")
    dist_logger.info(f"Training cells: {len(fit_result.train_cell_ids)}")
    dist_logger.info("=" * 80)

    dist_logger.info("\nJoint Optimization (across all distance measures):")
    dist_logger.info("-" * 80)
    for mode in fit_result.packing_modes:
        coeff = fit_result.coefficients[mode]["joint"]
        rel = fit_result.relative_contributions[mode]["joint"]
        dist_logger.info(f"  Mode: {mode:<30s}  coeff={coeff:.4f}  relative={rel:.4f}")

    dist_logger.info("\n" + "=" * 80)
    dist_logger.info("\nIndividual Optimization (per distance measure):")
    dist_logger.info("-" * 80)
    for dm in fit_result.distance_measures:
        dist_logger.info(f"\n  Distance measure: {dm}")
        dist_logger.info(
            f"    Train MSE (individual): {fit_result.train_mse['individual'][dm]:.6f}"
        )
        dist_logger.info(f"    Train MSE (joint):      {fit_result.train_mse['joint'][dm]:.6f}")
        for mode in fit_result.packing_modes:
            coeff = fit_result.coefficients[mode][dm]
            rel = fit_result.relative_contributions[mode][dm]
            dist_logger.info(f"    Mode: {mode:<30s}  coeff={coeff:.4f}  relative={rel:.4f}")

    remove_file_handler_from_logger(dist_logger, file_path)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def run_rule_interpolation_cv(
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    channel_map: dict[str, str],
    baseline_mode: str,
    n_folds: int = 5,
    n_repeats: int = 10,
    random_state: int | None = None,
    distance_measures: list[str] | None = None,
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    grouping: Literal["per-cell", "combined"] = "per-cell",
) -> CVResult:
    """Run k-fold cross-validation for occupancy rule interpolation.

    The baseline cell IDs are shuffled and split into ``n_folds`` folds.
    For each fold, coefficients are fitted on the training baseline cells and
    evaluated on the held-out test cells.  Simulated modes always contribute
    their full combined occupancy to the design matrix.

    The full k-fold procedure is repeated ``n_repeats`` times with independent
    random shuffles.  Aggregated statistics (mean/std MSE and coefficients) are
    computed across all ``n_folds x n_repeats`` fold results.

    Parameters
    ----------
    occupancy_dict
        ``{distance_measure: {mode: {"individual": {...}, "combined": {...}}}}``.
    channel_map
        ``{mode: structure_id}``.
    baseline_mode
        Key for the experimental baseline packing mode.
    n_folds
        Number of folds (default 5).
    n_repeats
        Number of independent repeats of the full k-fold procedure (default 1).
        Per-repeat seeds are derived deterministically from ``random_state``.
    random_state
        Seed for reproducible shuffling.
    distance_measures
        Subset of distance measures to use.  Defaults to all keys in
        ``occupancy_dict``.
    results_dir
        If provided, cache the result as a pickle file.
    recalculate
        If ``True``, ignore any cached result and recompute.
    suffix
        Appended to the cache filename.
    grouping
        Whether to compute test MSE on per cell occupancy or on the combined mean curve
        across test cells.
        If ``"per-cell"``, returns an array of MSE values for each test cell;
        if ``"combined"``, returns a single MSE value comparing the fit to the
        mean occupancy curve of the test cells.

    Returns
    -------
    :
        :class:`CVResult` with per-fold results and aggregated statistics.
    """
    if distance_measures is None:
        distance_measures = list(occupancy_dict.keys())

    save_path: Path | None = None
    if results_dir is not None:
        save_path = results_dir / f"rule_interpolation_cv{suffix}.pkl"

    if not recalculate and save_path is not None and save_path.exists():
        try:
            with open(save_path, "rb") as fh:
                cv_result: CVResult = pickle.load(fh)
            logger.info(f"Loaded cached CV result from {save_path}")
            return cv_result
        except Exception as exc:
            logger.warning(f"Could not load cached CV result from {save_path}: {exc}. Recomputing.")

    all_cell_ids = _get_baseline_cell_ids(occupancy_dict, baseline_mode)
    if len(all_cell_ids) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} folds from only {len(all_cell_ids)} baseline cells."
        )

    # Derive independent per-repeat seeds from the master random state.
    rng = np.random.default_rng(random_state)
    repeat_seeds = rng.integers(0, 2**31, size=n_repeats)

    fold_results: list[FoldResult] = []
    for repeat_idx in range(n_repeats):
        rng_r = np.random.default_rng(int(repeat_seeds[repeat_idx]))
        shuffled = list(rng_r.permutation(all_cell_ids))
        folds_array = np.array_split(shuffled, n_folds)
        logger.info(f"Repeat {repeat_idx + 1}/{n_repeats}")

        for fold_idx, test_arr in enumerate(folds_array):
            test_cells = list(test_arr)
            train_cells = [c for c in shuffled if c not in set(test_cells)]
            logger.info(
                f"  Fold {fold_idx + 1}/{n_folds}: "
                f"{len(train_cells)} train cells, {len(test_cells)} test cells"
            )

            fit_result = fit_rule_interpolation(
                occupancy_dict=occupancy_dict,
                channel_map=channel_map,
                baseline_mode=baseline_mode,
                train_cell_ids=train_cells,
                distance_measures=distance_measures,
            )
            test_mse = _evaluate_on_cells(
                occupancy_dict=occupancy_dict,
                baseline_mode=baseline_mode,
                test_cell_ids=test_cells,
                fit_result=fit_result,
                distance_measures=distance_measures,
                grouping=grouping,
            )
            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    train_cell_ids=train_cells,
                    test_cell_ids=test_cells,
                    fit_result=fit_result,
                    test_mse=test_mse,
                    repeat_idx=repeat_idx,
                )
            )

    cv_result = _aggregate_cv_results(
        fold_results=fold_results,
        distance_measures=distance_measures,
        packing_modes=fold_results[0].fit_result.packing_modes,
        baseline_mode=baseline_mode,
        n_folds=n_folds,
        n_repeats=n_repeats,
    )

    if save_path is not None:
        with open(save_path, "wb") as fh:
            pickle.dump(cv_result, fh)
        logger.info(f"Saved CV result to {save_path}")

    return cv_result


def _aggregate_cv_results(
    fold_results: list[FoldResult],
    distance_measures: list[str],
    packing_modes: list[str],
    baseline_mode: str,
    n_folds: int,
    n_repeats: int = 10,
) -> CVResult:
    """Aggregate per-fold FitResults into a CVResult."""
    # --- aggregated coefficients ---
    agg_coeffs: dict[str, dict[str, tuple[float, float]]] = {}
    for mode in packing_modes:
        ind_by_dm: dict[str, tuple[float, float]] = {}
        for dm in distance_measures:
            vals = [fr.fit_result.coefficients[mode][dm] for fr in fold_results]
            ind_by_dm[dm] = (float(np.mean(vals)), float(np.std(vals)))
        joint_vals = [fr.fit_result.coefficients[mode]["joint"] for fr in fold_results]
        agg_coeffs[mode] = {
            **ind_by_dm,
            "joint": (float(np.mean(joint_vals)), float(np.std(joint_vals))),
        }

    # --- MSE aggregation ---
    def _agg_mse(
        mse_getter,  # callable(FoldResult, dm) -> float | np.ndarray
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for scope in ("individual", "joint"):
            result[scope] = {}
            for dm in distance_measures:
                vals = [mse_getter(fr, scope, dm) for fr in fold_results]
                result[scope][dm] = float(np.mean(np.concatenate([np.atleast_1d(v) for v in vals])))
        return result

    def train_getter(fr: FoldResult, scope: str, dm: str) -> float:
        return fr.fit_result.train_mse[scope][dm]

    def test_getter(fr: FoldResult, scope: str, dm: str) -> np.ndarray:
        return fr.test_mse[scope][dm]

    def _agg_std(
        mse_getter,
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for scope in ("individual", "joint"):
            result[scope] = {}
            for dm in distance_measures:
                per_fold = [
                    np.mean(np.atleast_1d(mse_getter(fr, scope, dm))) for fr in fold_results
                ]
                result[scope][dm] = float(np.std(per_fold))
        return result

    return CVResult(
        folds=fold_results,
        aggregated_coefficients=agg_coeffs,
        mean_train_mse=_agg_mse(train_getter),
        std_train_mse=_agg_std(train_getter),
        mean_test_mse=_agg_mse(test_getter),
        std_test_mse=_agg_std(test_getter),
        n_folds=n_folds,
        baseline_mode=baseline_mode,
        n_repeats=n_repeats,
    )


def summarize_cv_results(cv_result: CVResult) -> pd.DataFrame:
    """Tabulate CV results for easy inspection and plotting.

    Parameters
    ----------
    cv_result
        Output from :func:`run_rule_interpolation_cv`.

    Returns
    -------
    :
        DataFrame with columns ``fold_idx``, ``scope``, ``distance_measure``,
        ``split``, ``mse``.  Each row corresponds to one scalar MSE observation
        (a single cell for the test split, or the fold-level scalar for train).
    """
    records: list[dict[str, Any]] = []

    for fold in cv_result.folds:
        for scope in ("individual", "joint"):
            for dm in fold.fit_result.distance_measures:
                # Train: one scalar per fold
                train_val = fold.fit_result.train_mse[scope][dm]
                records.append(
                    {
                        "repeat_idx": fold.repeat_idx,
                        "fold_idx": fold.fold_idx,
                        "scope": scope,
                        "distance_measure": dm,
                        "split": "train",
                        "mse": train_val,
                        "cell_id": None,
                    }
                )
                # Test: one row per held-out cell
                test_arr = fold.test_mse[scope][dm]
                for cell_id, mse_val in zip(fold.test_cell_ids, test_arr, strict=False):
                    records.append(
                        {
                            "repeat_idx": fold.repeat_idx,
                            "fold_idx": fold.fold_idx,
                            "scope": scope,
                            "distance_measure": dm,
                            "split": "test",
                            "mse": float(mse_val),
                            "cell_id": cell_id,
                        }
                    )

    return pd.DataFrame.from_records(records)


def log_cv_summary(
    cv_result: CVResult,
    file_path: Path | None = None,
) -> None:
    """Log a human-readable summary of cross-validation results.

    Parameters
    ----------
    cv_result
        Output from :func:`run_rule_interpolation_cv`.
    file_path
        Optional path to write the log to a file.
    """
    if file_path is not None:
        cv_logger = add_file_handler_to_logger(logger, file_path)
    else:
        cv_logger = logger

    cv_logger.info(
        f"Cross-validation summary ({cv_result.n_folds} folds x {cv_result.n_repeats} repeats)"
    )
    cv_logger.info(f"Baseline mode: {cv_result.baseline_mode}")
    cv_logger.info("=" * 80)

    cv_logger.info("\nAggregated joint coefficients (mean ± std across folds):")
    cv_logger.info("-" * 80)
    for mode, data in cv_result.aggregated_coefficients.items():
        mean, std = data["joint"]
        cv_logger.info(f"  {mode:<30s}  {mean:.4f} ± {std:.4f}")

    cv_logger.info("\nMSE summary (mean across folds):")
    cv_logger.info("-" * 80)
    cv_logger.info(
        f"  {'Distance measure':<20s}  {'Scope':<12s}  {'Train MSE':>12s}  {'Test MSE':>12s}"
    )
    for scope in ("individual", "joint"):
        for dm in cv_result.mean_train_mse.get(scope, {}):
            train_mse = cv_result.mean_train_mse[scope][dm]
            test_mse = cv_result.mean_test_mse[scope][dm]
            cv_logger.info(f"  {dm:<20s}  {scope:<12s}  {train_mse:>12.6f}  {test_mse:>12.6f}")

    remove_file_handler_from_logger(cv_logger, file_path)


# ---------------------------------------------------------------------------
# Packing config generation
# ---------------------------------------------------------------------------


def _get_gradient_weights_from_coefficients(
    coefficients: dict[str, float],
    packing_modes: list[str],
    mode_to_gradient_name: dict[str, str],
) -> dict[str, float]:
    """Convert mode coefficients to normalised gradient_weights for a cellPACK config.

    Parameters
    ----------
    coefficients
        ``{mode: raw_coeff}`` — raw NNLS output (need not sum to 1).
    packing_modes
        Ordered list of simulated packing modes.
    mode_to_gradient_name
        Mapping from packing mode names to cellPACK gradient names.

    Returns
    -------
    :
        ``{gradient_name: normalised_weight}`` summing to 1.0.
    """
    total = sum(coefficients.values())
    if total <= 0:
        raise ValueError("All coefficients are zero; cannot produce a valid gradient_weights dict.")
    return {
        mode_to_gradient_name[mode]: coefficients[mode] / total
        for mode in packing_modes
        if mode in mode_to_gradient_name
    }


def generate_mixed_rule_packing_configs(
    cv_result: CVResult,
    base_config_path: Path,
    output_config_dir: Path,
    mode_to_gradient_name: dict[str, str],
    scope: Literal["individual", "joint"] = "joint",
    distance_measure: str | None = None,
    fold_idx: int | None = None,
    dry_run: bool = False,
    aggregated_only: bool = False,
) -> list[Path]:
    """Generate cellPACK packing config files with the fitted mixed rule.

    For each fold (or for the aggregated coefficients), writes a JSON config
    based on ``base_config_path`` with:

    - ``recipe_data["interpolated"]`` set to the fitted ``gradient_weights``.
    - ``packings_to_run.rules`` restricted to ``["interpolated"]``.
    - ``num_cells`` / ``cell_ids`` set to the held-out test cells for the fold
      (or omitted when using aggregated coefficients).

    Parameters
    ----------
    cv_result
        Output from :func:`run_rule_interpolation_cv`.
    base_config_path
        Path to an existing cellPACK workflow config JSON to use as a template.
    output_config_dir
        Directory where generated configs are written.
    mode_to_gradient_name
        Maps each simulated packing mode to the cellPACK gradient name used in
        ``recipe_data`` (e.g. ``{"nucleus_gradient": "nucleus_gradient"}``).
        Must cover all modes in ``cv_result``.
    scope
        Which fit to use: ``"individual"`` uses per-distance-measure coefficients
        (requires ``distance_measure`` to be set); ``"joint"`` uses the joint fit.
    distance_measure
        Required when ``scope="individual"``.
    fold_idx
        Which fold's coefficients to use.  ``None`` uses the aggregated mean
        coefficients (averaged across folds).
    dry_run
        If ``True``, print the configs but do not write files.
    aggregated_only
        If ``True`` and ``fold_idx`` is ``None``, generate only the aggregated
        config (mean coefficients across folds) and skip per-fold configs.

    Returns
    -------
    :
        List of paths to the written config files.
    """
    if scope == "individual" and distance_measure is None:
        raise ValueError("'distance_measure' must be set when scope='individual'.")

    output_config_dir = Path(output_config_dir)
    if not dry_run:
        output_config_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path) as fh:
        base_config: dict[str, Any] = json.load(fh)

    packing_modes = list(cv_result.aggregated_coefficients.keys())

    # Determine which folds to produce configs for
    fold_indices: list[int | None]
    if fold_idx is not None:
        fold_indices = [fold_idx]
    elif aggregated_only:
        fold_indices = [None]
    else:
        # One config per fold (using that fold's fitted coefficients) plus
        # one aggregated config
        fold_indices = [*list(range(cv_result.n_folds)), None]

    written_paths: list[Path] = []
    for fi in fold_indices:
        # Get coefficients
        if fi is None:
            # Aggregated: take mean coefficient across folds
            if scope == "joint":
                raw_coeffs = {
                    mode: data["joint"][0]
                    for mode, data in cv_result.aggregated_coefficients.items()
                }
            else:
                assert distance_measure is not None
                raw_coeffs = {
                    mode: data[distance_measure][0]
                    for mode, data in cv_result.aggregated_coefficients.items()
                }
            test_cells: list[str] | None = None
            label = "aggregated"
        else:
            fold = cv_result.folds[fi]
            if scope == "joint":
                raw_coeffs = {
                    mode: fold.fit_result.coefficients[mode]["joint"] for mode in packing_modes
                }
            else:
                assert distance_measure is not None
                raw_coeffs = {
                    mode: fold.fit_result.coefficients[mode][distance_measure]
                    for mode in packing_modes
                }
            test_cells = fold.test_cell_ids
            label = f"fold{fi}"

        gradient_weights = _get_gradient_weights_from_coefficients(
            coefficients=raw_coeffs,
            packing_modes=packing_modes,
            mode_to_gradient_name=mode_to_gradient_name,
        )
        gradient_names = list(gradient_weights.keys())

        # Build the modified config
        config = dict(base_config)  # shallow copy to avoid mutating the original
        recipe_data: dict[str, Any] = config.get("recipe_data", {}).copy()
        recipe_data["interpolated"] = {
            "gradients": gradient_names,
            "gradient_weights": gradient_weights,
        }
        config["recipe_data"] = recipe_data
        config["packings_to_run"] = {"rules": ["interpolated"]}

        if test_cells is not None:
            config["cell_ids"] = test_cells

        dm_label = f"_{distance_measure}" if scope == "individual" and distance_measure else ""
        fname = f"mixed_rule_{scope}{dm_label}_{label}.json"

        if dry_run:
            logger.info(f"[dry_run] Would write: {output_config_dir / fname}")
            logger.info(json.dumps(config, indent=2))
        else:
            out_path = output_config_dir / fname
            with open(out_path, "w") as fh:
                json.dump(config, fh, indent=4)
            os.chmod(out_path, 0o644)
            logger.info(f"Wrote packing config: {out_path}")
            written_paths.append(out_path)

    return written_paths


def trigger_packing_workflow(
    config_paths: list[Path],
    use_slurm: bool = True,
    slurm_script: Path | None = None,
    slurm_kwargs: dict[str, Any] | None = None,
) -> None:
    """Submit packing jobs for a list of config files.

    .. warning::
        Packing is highly compute-intensive.  Each config may require many
        cpu-hours.  Prefer submitting to SLURM on a compute cluster.

    Parameters
    ----------
    config_paths
        Paths to cellPACK workflow config JSON files to run.
    use_slurm
        If ``True`` (default), submit each config via the SLURM launcher script.
        If ``False``, run the Python packing workflow locally (not recommended
        for large jobs).
    slurm_script
        Path to the SLURM launcher shell script.  If ``None``, attempts to
        locate ``submit_packing_slurm.sh`` relative to this module.
    slurm_kwargs
        Extra keyword arguments passed to the SLURM launcher as flags
        (e.g. ``{"b": "8", "p": "celltypes"}``).
    """
    logger.warning(
        "trigger_packing_workflow: packing is very time-consuming. "
        "Each job may take many cpu-hours. Review generated configs before submitting."
    )

    if use_slurm:
        if slurm_script is None:
            slurm_script = (
                get_project_root() / "cellpack_analysis" / "packing" / "submit_packing_slurm.sh"
            )
        if not slurm_script.exists():
            raise FileNotFoundError(f"SLURM launcher not found: {slurm_script}")

        for config_path in config_paths:
            cmd = ["bash", str(slurm_script), "-c", str(config_path)]
            if slurm_kwargs:
                for key, val in slurm_kwargs.items():
                    cmd += [f"-{key}", str(val)]

            logger.info(f"Submitting SLURM job: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    else:
        from cellpack_analysis.packing.run_packing_workflow import _run_packing_workflow

        for config_path in config_paths:
            logger.info(f"Running packing workflow locally for: {config_path}")
            _run_packing_workflow(workflow_config_path=config_path)


# ---------------------------------------------------------------------------
# AIC / BIC model comparison
# ---------------------------------------------------------------------------


def _compute_aic(n: int, k: int, rss: float) -> float:
    r"""Compute the Akaike Information Criterion for Gaussian least-squares.

    Uses :math:`k_{\\text{eff}} = k + 1` (model parameters plus error variance).

    .. math::
        \\text{AIC} = n \\ln(\\text{RSS} / n) + 2 k_{\\text{eff}}
    """
    k_eff = k + 1
    return n * np.log(rss / n) + 2 * k_eff


def _compute_aicc(n: int, k: int, rss: float) -> float:
    r"""Compute the small-sample corrected AIC (AICc).

    .. math::
        \\text{AIC}_c = \\text{AIC} + \\frac{2 k_{\\text{eff}} (k_{\\text{eff}} + 1)}
        {n - k_{\\text{eff}} - 1}
    """
    k_eff = k + 1
    aic = _compute_aic(n, k, rss)
    denom = n - k_eff - 1
    if denom <= 0:
        return float("inf")
    return aic + 2 * k_eff * (k_eff + 1) / denom


def _compute_bic(n: int, k: int, rss: float) -> float:
    r"""Compute the Bayesian Information Criterion for Gaussian least-squares.

    Uses :math:`k_{\\text{eff}} = k + 1`.

    .. math::
        \\text{BIC} = n \\ln(\\text{RSS} / n) + k_{\\text{eff}} \\ln(n)
    """
    k_eff = k + 1
    return n * np.log(rss / n) + k_eff * np.log(n)


def _fit_single_rule(
    baseline_occupancy: np.ndarray,
    mode_occupancy: np.ndarray,
) -> tuple[float, int]:
    """Fit a single-mode NNLS model and return (RSS, n).

    Parameters
    ----------
    baseline_occupancy
        Target vector (mean baseline occupancy), shape ``(n,)``.
    mode_occupancy
        Single simulated mode's combined occupancy on the same grid, shape ``(n,)``.

    Returns
    -------
    rss
        Residual sum of squares.
    n
        Number of data points.
    """
    mode_occ_matrix = mode_occupancy.reshape(-1, 1)
    coeff, _ = nnls(mode_occ_matrix, baseline_occupancy)
    recon = mode_occ_matrix @ coeff
    rss = float(np.sum((baseline_occupancy - recon) ** 2))
    return rss, len(baseline_occupancy)


def _compute_model_weights(
    delta_values: dict[str, float],
) -> dict[str, float]:
    r"""Convert ΔAIC or ΔBIC values to model weights via the Akaike formula.

    .. math::
        w_i = \\frac{\\exp(-\\Delta_i / 2)}{\\sum_j \\exp(-\\Delta_j / 2)}
    """
    raw = {name: np.exp(-0.5 * d) for name, d in delta_values.items()}
    total = sum(raw.values())
    if total == 0:
        n = len(raw)
        return dict.fromkeys(raw, 1.0 / n)
    return {name: val / total for name, val in raw.items()}


def compute_aic_comparison(
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    channel_map: dict[str, str],
    baseline_mode: str,
    fit_result: FitResult | None = None,
    distance_measures: list[str] | None = None,
) -> AICComparisonResult:
    """Compare mixed-rule, single-rule, and null models using AIC and BIC.

    For each scope (individual per-DM and joint across all DMs) computes RSS
    for the mixed rule (from ``fit_result``, or fits from scratch), each
    single-mode NNLS model, and a constant-mean null model.  Returns AIC,
    AICc, BIC, ΔAIC, ΔBIC, and model weights.

    Parameters
    ----------
    occupancy_dict
        ``{dm: {mode: {"individual": {...}, "combined": {...}}}}``.
    channel_map
        ``{mode: structure_id}``.
    baseline_mode
        Experimental baseline packing mode key.
    fit_result
        Pre-computed :class:`FitResult` (e.g. from :func:`fit_rule_interpolation`).
        If ``None``, a full-data fit is performed internally.
    distance_measures
        Subset of distance measures to use.  Defaults to all keys in
        ``occupancy_dict``.

    Returns
    -------
    :
        :class:`AICComparisonResult`.
    """
    if distance_measures is None:
        distance_measures = list(occupancy_dict.keys())

    packing_modes = [mode for mode in channel_map if mode not in (baseline_mode, "interpolated")]

    # Fit if not provided
    if fit_result is None:
        fit_result = fit_rule_interpolation(
            occupancy_dict, channel_map, baseline_mode, distance_measures=distance_measures
        )

    n_modes = len(packing_modes)

    # Pre-compute per-DM baseline targets and design matrices
    dm_baseline: dict[str, np.ndarray] = {}
    dm_sim_cols: dict[str, dict[str, np.ndarray]] = {}

    for dm in distance_measures:
        dm_data = occupancy_dict[dm]
        baseline_dm = dm_data[baseline_mode]
        common_xvals: np.ndarray = baseline_dm["combined"]["xvals"]

        baseline_occ, _ = _compute_mean_occupancy_from_cells(
            mode_individual_dict=baseline_dm["individual"],
            cell_ids=fit_result.train_cell_ids,
            common_xvals=common_xvals,
        )
        dm_baseline[dm] = baseline_occ

        cols: dict[str, np.ndarray] = {}
        for mode in packing_modes:
            mode_xvals = dm_data[mode]["combined"]["xvals"]
            mode_occ = dm_data[mode]["combined"]["occupancy"]
            if len(mode_xvals) != len(common_xvals) or not np.allclose(mode_xvals, common_xvals):
                col = np.interp(common_xvals, mode_xvals, mode_occ, right=0.0, left=0.0)
            else:
                col = mode_occ
            cols[mode] = np.nan_to_num(col, nan=0.0)
        dm_sim_cols[dm] = cols

    # Collect results per scope/dm
    comparisons: dict[str, dict[str, list[AICModelResult]]] = {
        "individual": {},
        "joint": {},
    }

    for dm in distance_measures:
        b = dm_baseline[dm]
        n = len(b)

        models: list[AICModelResult] = []

        # Mixed rule (individual scope)
        recon_ind = fit_result.reconstructed_occupancy[dm]["individual"]
        rss_mixed = float(np.sum((b - recon_ind) ** 2))
        models.append(
            AICModelResult(
                model_name="mixed_rule",
                k=n_modes,
                n=n,
                rss=rss_mixed,
                aic=_compute_aic(n, n_modes, rss_mixed),
                aicc=_compute_aicc(n, n_modes, rss_mixed),
                bic=_compute_bic(n, n_modes, rss_mixed),
            )
        )

        # Single-rule models
        for mode in packing_modes:
            rss_s, n_s = _fit_single_rule(b, dm_sim_cols[dm][mode])
            models.append(
                AICModelResult(
                    model_name=f"single:{mode}",
                    k=1,
                    n=n_s,
                    rss=rss_s,
                    aic=_compute_aic(n_s, 1, rss_s),
                    aicc=_compute_aicc(n_s, 1, rss_s),
                    bic=_compute_bic(n_s, 1, rss_s),
                )
            )

        # Null model: predict constant mean of baseline occupancy
        mean_val = float(np.mean(b))
        rss_null = float(np.sum((b - mean_val) ** 2))
        models.append(
            AICModelResult(
                model_name="null",
                k=1,
                n=n,
                rss=rss_null,
                aic=_compute_aic(n, 1, rss_null),
                aicc=_compute_aicc(n, 1, rss_null),
                bic=_compute_bic(n, 1, rss_null),
            )
        )

        comparisons["individual"][dm] = models

    # Joint scope: stack all DMs
    b_joint = np.concatenate([dm_baseline[dm] for dm in distance_measures])
    n_joint = len(b_joint)

    # Mixed rule (joint scope) — use joint reconstructed occupancy
    recon_joint = np.concatenate(
        [fit_result.reconstructed_occupancy[dm]["joint"] for dm in distance_measures]
    )
    rss_mixed_joint = float(np.sum((b_joint - recon_joint) ** 2))
    joint_models: list[AICModelResult] = [
        AICModelResult(
            model_name="mixed_rule",
            k=n_modes,
            n=n_joint,
            rss=rss_mixed_joint,
            aic=_compute_aic(n_joint, n_modes, rss_mixed_joint),
            aicc=_compute_aicc(n_joint, n_modes, rss_mixed_joint),
            bic=_compute_bic(n_joint, n_modes, rss_mixed_joint),
        )
    ]

    # Single-rule (joint scope): stack all DMs per mode
    for mode in packing_modes:
        stacked_mode = np.concatenate([dm_sim_cols[dm][mode] for dm in distance_measures])
        rss_s, n_s = _fit_single_rule(b_joint, stacked_mode)
        joint_models.append(
            AICModelResult(
                model_name=f"single:{mode}",
                k=1,
                n=n_s,
                rss=rss_s,
                aic=_compute_aic(n_s, 1, rss_s),
                aicc=_compute_aicc(n_s, 1, rss_s),
                bic=_compute_bic(n_s, 1, rss_s),
            )
        )

    # Null (joint scope)
    mean_joint = float(np.mean(b_joint))
    rss_null_joint = float(np.sum((b_joint - mean_joint) ** 2))
    joint_models.append(
        AICModelResult(
            model_name="null",
            k=1,
            n=n_joint,
            rss=rss_null_joint,
            aic=_compute_aic(n_joint, 1, rss_null_joint),
            aicc=_compute_aicc(n_joint, 1, rss_null_joint),
            bic=_compute_bic(n_joint, 1, rss_null_joint),
        )
    )
    comparisons["joint"]["joint"] = joint_models

    # Compute deltas and weights
    delta_aic: dict[str, dict[str, dict[str, float]]] = {}
    delta_bic: dict[str, dict[str, dict[str, float]]] = {}
    akaike_weights: dict[str, dict[str, dict[str, float]]] = {}
    bic_weights: dict[str, dict[str, dict[str, float]]] = {}
    best_model_aic: dict[str, dict[str, str]] = {}
    best_model_bic: dict[str, dict[str, str]] = {}

    for scope, dm_dict in comparisons.items():
        delta_aic[scope] = {}
        delta_bic[scope] = {}
        akaike_weights[scope] = {}
        bic_weights[scope] = {}
        best_model_aic[scope] = {}
        best_model_bic[scope] = {}

        for dm, models in dm_dict.items():
            min_aic = min(m.aic for m in models)
            min_bic = min(m.bic for m in models)

            d_aic = {m.model_name: m.aic - min_aic for m in models}
            d_bic = {m.model_name: m.bic - min_bic for m in models}

            delta_aic[scope][dm] = d_aic
            delta_bic[scope][dm] = d_bic
            akaike_weights[scope][dm] = _compute_model_weights(d_aic)
            bic_weights[scope][dm] = _compute_model_weights(d_bic)
            best_model_aic[scope][dm] = min(d_aic, key=d_aic.get)  # type: ignore[arg-type]
            best_model_bic[scope][dm] = min(d_bic, key=d_bic.get)  # type: ignore[arg-type]

    return AICComparisonResult(
        comparisons=comparisons,
        delta_aic=delta_aic,
        delta_bic=delta_bic,
        akaike_weights=akaike_weights,
        bic_weights=bic_weights,
        best_model_aic=best_model_aic,
        best_model_bic=best_model_bic,
    )


def summarize_aic_comparison(result: AICComparisonResult) -> pd.DataFrame:
    """Return a tidy DataFrame summarising the AIC/BIC model comparison.

    Columns: ``scope``, ``distance_measure``, ``model``, ``k``, ``n``, ``rss``,
    ``aic``, ``aicc``, ``bic``, ``delta_aic``, ``delta_bic``,
    ``akaike_weight``, ``bic_weight``.
    """
    rows: list[dict[str, Any]] = []
    for scope, dm_dict in result.comparisons.items():
        for dm, models in dm_dict.items():
            for m in models:
                rows.append(
                    {
                        "scope": scope,
                        "distance_measure": dm,
                        "model": m.model_name,
                        "k": m.k,
                        "n": m.n,
                        "rss": m.rss,
                        "aic": m.aic,
                        "aicc": m.aicc,
                        "bic": m.bic,
                        "delta_aic": result.delta_aic[scope][dm][m.model_name],
                        "delta_bic": result.delta_bic[scope][dm][m.model_name],
                        "akaike_weight": result.akaike_weights[scope][dm][m.model_name],
                        "bic_weight": result.bic_weights[scope][dm][m.model_name],
                    }
                )
    return pd.DataFrame(rows)


def log_evidence_ratios(
    result: AICComparisonResult,
    file_path: Path,
) -> None:
    """Write AIC/BIC evidence ratios to a dedicated log file.

    For each scope and distance measure, logs:

    * ΔAIC / ΔBIC interpretation thresholds (Burnham & Anderson 2002).
    * Per-model AIC, BIC, Akaike weight, BIC weight, and evidence ratios
      relative to the best model.

    Parameters
    ----------
    result
        Output from :func:`compute_aic_comparison`.
    file_path
        Absolute path for the log file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    er_logger = logging.getLogger(__name__ + ".evidence_ratios")
    add_file_handler_to_logger(er_logger, file_path)

    er_logger.info("AIC / BIC Evidence Ratio Report")
    er_logger.info("=" * 90)
    er_logger.info(
        "Interpretation guide (Burnham & Anderson 2002):\n"
        "  ΔAIC  0-2  : Substantial support (model nearly as good as best)\n"
        "  ΔAIC  4-7  : Considerably less support\n"
        "  ΔAIC  > 10 : Essentially no support\n"
        "  Evidence ratio = w_best / w_i  (how many times more likely the best model is)\n"
    )

    for scope, dm_dict in result.comparisons.items():
        for dm, models in dm_dict.items():
            er_logger.info("-" * 90)
            er_logger.info(f"Scope: {scope}  |  Distance measure: {dm}")
            er_logger.info(
                f"  Best model (AIC): {result.best_model_aic[scope][dm]}  |  "
                f"Best model (BIC): {result.best_model_bic[scope][dm]}"
            )
            er_logger.info("-" * 90)

            header = (
                f"  {'Model':<25s} {'k':>3s} {'AIC':>12s} {'ΔAIC':>10s} "
                f"{'w_AIC':>10s} {'BIC':>12s} {'ΔBIC':>10s} {'w_BIC':>10s} "
                f"{'ER_AIC':>10s} {'ER_BIC':>10s}"
            )
            er_logger.info(header)

            best_w_aic = result.akaike_weights[scope][dm][result.best_model_aic[scope][dm]]
            best_w_bic = result.bic_weights[scope][dm][result.best_model_bic[scope][dm]]

            for m in models:
                name = m.model_name
                w_aic = result.akaike_weights[scope][dm][name]
                w_bic = result.bic_weights[scope][dm][name]
                er_aic = best_w_aic / w_aic if w_aic > 0 else float("inf")
                er_bic = best_w_bic / w_bic if w_bic > 0 else float("inf")
                d_aic = result.delta_aic[scope][dm][name]
                d_bic = result.delta_bic[scope][dm][name]

                er_logger.info(
                    f"  {name:<25s} {m.k:>3d} {m.aic:>12.2f} {d_aic:>10.2f} "
                    f"{w_aic:>10.4f} {m.bic:>12.2f} {d_bic:>10.2f} {w_bic:>10.4f} "
                    f"{er_aic:>10.2f} {er_bic:>10.2f}"
                )

            er_logger.info("")

    remove_file_handler_from_logger(er_logger, file_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def run_mixed_rule_validation(
    combined_occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    channel_map: dict[str, str],
    baseline_mode: str,
    packing_modes: list[str],
    distance_measures: list[str],
    mixed_rule_distance_dict: dict[str, Any] | None = None,
    results_dir: Path | None = None,
    recalculate: bool = False,
    fit_result: FitResult | None = None,
) -> ValidationResult:
    """Run orthogonal validation checks for the mixed packing rule.

    Always computes occupancy-based checks reusing the existing functions in
    :mod:`cellpack_analysis.lib.occupancy`.  If actual packing simulation
    outputs are available (``mixed_rule_distance_dict`` is not ``None``),
    also runs distance-distribution-based tests.  If ``fit_result`` is
    provided, additionally runs AIC/BIC model comparison.

    Parameters
    ----------
    combined_occupancy_dict
        ``{distance_measure: {mode: {"individual": {...}, "combined": {...}}}}``.
        Must include the mixed-rule mode (typically keyed ``"interpolated"``).
    channel_map
        ``{mode: structure_id}``.
    baseline_mode
        Experimental baseline packing mode key.
    packing_modes
        All packing modes to include in comparisons (including the mixed rule).
    distance_measures
        Distance measures to compare across.
    mixed_rule_distance_dict
        Raw distance data from mixed-rule packings in the same format as
        ``all_distance_dict`` used by the distance analysis workflow.
        If ``None``, only occupancy-based tests are run.
    results_dir
        Directory for saving intermediate results.
    recalculate
        Whether to recompute even if cached results exist.
    fit_result
        Pre-computed :class:`FitResult` from :func:`fit_rule_interpolation`.
        When provided, AIC/BIC model comparison is computed and attached to
        the returned :class:`ValidationResult`.

    Returns
    -------
    :
        :class:`ValidationResult` bundling all test outputs.
    """
    logger.info("Running mixed-rule occupancy EMD analysis")
    emd_df = occupancy.get_occupancy_emd_df(
        combined_occupancy_dict=combined_occupancy_dict,
        packing_modes=packing_modes,
        distance_measures=distance_measures,
        results_dir=results_dir,
        recalculate=recalculate,
        suffix="_mixed_rule_validation",
    )

    logger.info("Running mixed-rule occupancy KS test")
    ks_df = occupancy.get_occupancy_ks_test_df(
        distance_measures=distance_measures,
        packing_modes=packing_modes,
        combined_occupancy_dict=combined_occupancy_dict,
        baseline_mode=baseline_mode,
        results_dir=results_dir,
        recalculate=recalculate,
    )

    logger.info("Running mixed-rule occupancy pairwise envelope test")
    envelope_test = occupancy.pairwise_envelope_test_occupancy(
        combined_occupancy_dict=combined_occupancy_dict,
        packing_modes=packing_modes,
    )

    distance_emd_df: pd.DataFrame | None = None
    distance_envelope_test: dict[str, Any] | None = None
    distance_ks_df: pd.DataFrame | None = None
    distance_ks_bootstrap_df: pd.DataFrame | None = None

    if mixed_rule_distance_dict is not None:
        logger.info("Running mixed-rule distance EMD analysis")
        distance_emd_df = distance.get_distance_distribution_emd_df(
            all_distance_dict=mixed_rule_distance_dict,
            packing_modes=packing_modes,
            distance_measures=distance_measures,
            results_dir=results_dir,
            recalculate=recalculate,
            suffix="_mixed_rule_validation",
        )

        from cellpack_analysis.lib.stats import pairwise_envelope_test

        logger.info("Running mixed-rule distance pairwise envelope test")
        distance_envelope_test = pairwise_envelope_test(
            all_distance_dict=mixed_rule_distance_dict,
            packing_modes=packing_modes,
            distance_measures=distance_measures,
        )

        logger.info("Running mixed-rule distance KS test")
        distance_ks_df = distance.get_ks_test_df(
            distance_measures=distance_measures,
            packing_modes=packing_modes,
            all_distance_dict=mixed_rule_distance_dict,
            baseline_mode=baseline_mode,
        )

        logger.info("Running mixed-rule distance KS bootstrap")
        non_baseline_modes = [m for m in packing_modes if m != baseline_mode]
        distance_ks_bootstrap_df = distance.bootstrap_ks_tests(
            ks_test_df=distance_ks_df,
            distance_measures=distance_measures,
            packing_modes=non_baseline_modes,
        )

    # AIC/BIC model comparison
    aic_result: AICComparisonResult | None = None
    if fit_result is not None:
        logger.info("Running AIC/BIC model comparison for mixed-rule validation")
        aic_result = compute_aic_comparison(
            occupancy_dict=combined_occupancy_dict,
            channel_map=channel_map,
            baseline_mode=baseline_mode,
            fit_result=fit_result,
            distance_measures=distance_measures,
        )

    return ValidationResult(
        emd_df=emd_df,
        ks_df=ks_df,
        envelope_test=envelope_test,
        distance_emd_df=distance_emd_df,
        distance_envelope_test=distance_envelope_test,
        distance_ks_df=distance_ks_df,
        distance_ks_bootstrap_df=distance_ks_bootstrap_df,
        aic_result=aic_result,
    )
