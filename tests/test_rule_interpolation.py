"""Unit tests for cellpack_analysis.analysis.rule_interpolation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import nnls

from cellpack_analysis.lib.rule_interpolation import (
    AICComparisonResult,
    AICModelResult,
    CVResult,
    FitResult,
    _compute_aic,
    _compute_aicc,
    _compute_bic,
    _compute_mean_occupancy_from_cells,
    _fit_single_rule,
    _count_nonzero_coeffs,
    _get_baseline_cell_ids,
    _get_gradient_weights_from_coefficients,
    _masked_mse,
    _masked_reconstruction,
    _normalize_coefficients,
    _solve_nnls_masked,
    _valid_row_mask,
    compute_aic_comparison,
    fit_rule_interpolation,
    generate_mixed_rule_packing_configs,
    log_cv_summary,
    log_evidence_ratios,
    log_rule_interpolation_coeffs,
    run_rule_interpolation_cv,
    summarize_aic_comparison,
    summarize_cv_results,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

NUM_BINS = 30
XVALS = np.linspace(0, 6, NUM_BINS)


def _make_occupancy_curve(seed: int, scale: float = 1.0) -> np.ndarray:
    """Return a simple synthetic occupancy curve (positive-valued)."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale, size=NUM_BINS) + 0.1
    return raw


def _make_occupancy_dict(
    baseline_mode: str,
    simulated_modes: list[str],
    n_baseline_cells: int,
    distance_measures: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Build a minimal occupancy_dict compatible with rule_interpolation functions.

    Both KDE-style (non-uniform per-cell xvals) and discrete-style (shared
    common_xvals) use identical dict keys, so this helper produces the simplest
    compatible structure.
    """
    if distance_measures is None:
        distance_measures = ["nucleus"]

    rng = np.random.default_rng(seed)
    occ_dict: dict = {}

    for dm_idx, dm in enumerate(distance_measures):
        occ_dict[dm] = {}

        # Baseline mode: individual cells + combined
        individual: dict = {}
        for c_idx in range(n_baseline_cells):
            # Use slightly jittered xvals to exercise the np.interp path
            jitter = rng.uniform(-0.01, 0.01, size=NUM_BINS)
            cell_xvals = np.clip(XVALS + jitter, 0, None)
            cell_xvals = np.sort(cell_xvals)
            individual[f"cell_{c_idx:03d}"] = {
                "xvals": cell_xvals,
                "occupancy": _make_occupancy_curve(seed + dm_idx * 100 + c_idx),
            }
        mean_occ = np.mean(np.vstack([v["occupancy"] for v in individual.values()]), axis=0)
        occ_dict[dm][baseline_mode] = {
            "individual": individual,
            "combined": {"xvals": XVALS.copy(), "occupancy": mean_occ},
        }

        # Simulated modes: only combined (no individual cells — different structure)
        for s_idx, mode in enumerate(simulated_modes):
            sim_occ = _make_occupancy_curve(seed + dm_idx * 200 + s_idx * 10 + 1)
            occ_dict[dm][mode] = {
                "individual": {},
                "combined": {"xvals": XVALS.copy(), "occupancy": sim_occ},
            }

    return occ_dict


def _make_channel_map(baseline_mode: str, simulated_modes: list[str]) -> dict[str, str]:
    """Build a channel_map where all modes map to the same structure_id."""
    cm = {baseline_mode: "STRUCT_A"}
    for mode in simulated_modes:
        cm[mode] = "STRUCT_A"
    return cm


# ---------------------------------------------------------------------------
# Tests: _get_baseline_cell_ids
# ---------------------------------------------------------------------------


class TestGetBaselineCellIds:
    def test_returns_sorted_cell_ids(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=8)
        cell_ids = _get_baseline_cell_ids(occ, "real")
        assert cell_ids == sorted(cell_ids)
        assert len(cell_ids) == 8

    def test_raises_if_baseline_mode_missing(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=5)
        with pytest.raises(ValueError, match="no 'individual' data"):
            _get_baseline_cell_ids(occ, "nonexistent_mode")

    def test_warns_when_cell_ids_differ_across_dms(self, caplog):
        occ = _make_occupancy_dict(
            "real", ["random"], n_baseline_cells=4, distance_measures=["nucleus", "z"]
        )
        # Artificially remove a cell from the second distance measure
        occ["z"]["real"]["individual"].pop("cell_003")
        import logging

        with caplog.at_level(logging.WARNING):
            _get_baseline_cell_ids(occ, "real")
        assert any("differ" in record.message for record in caplog.records)

    def test_raises_if_empty_occupancy_dict(self):
        with pytest.raises(ValueError, match="empty"):
            _get_baseline_cell_ids({}, "real")


# ---------------------------------------------------------------------------
# Tests: _compute_mean_occupancy_from_cells
# ---------------------------------------------------------------------------


class TestComputeMeanOccupancyFromCells:
    def test_output_shape(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=6)
        individual = occ["nucleus"]["real"]["individual"]
        cell_ids = list(individual.keys())
        mean_occ, matrix = _compute_mean_occupancy_from_cells(individual, cell_ids, XVALS)
        assert mean_occ.shape == (NUM_BINS,)
        assert matrix.shape == (6, NUM_BINS)

    def test_mean_matches_manual_computation(self):
        """The mean is the NaN-aware column mean of the per-cell matrix.

        Jittered per-cell grids can end just inside the common grid, so edge
        points fall outside some cells' support and are NaN in those rows --
        `nanmean`, not `mean`, is the estimator under test.
        """
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=5, seed=42)
        individual = occ["nucleus"]["real"]["individual"]
        cell_ids = list(individual.keys())
        mean_occ, matrix = _compute_mean_occupancy_from_cells(individual, cell_ids, XVALS)
        np.testing.assert_allclose(mean_occ, np.nanmean(matrix, axis=0))

    def test_interpolation_handles_nonuniform_xvals(self):
        """Cells with jittered grids are correctly interpolated onto common_xvals."""
        rng = np.random.default_rng(77)
        individual = {}
        for i in range(4):
            jitter = rng.uniform(-0.05, 0.05, size=NUM_BINS)
            xv = np.sort(np.clip(XVALS + jitter, 0, None))
            individual[f"c{i}"] = {
                "xvals": xv,
                "occupancy": _make_occupancy_curve(i),
            }
        mean_occ, matrix = _compute_mean_occupancy_from_cells(
            individual, list(individual.keys()), XVALS
        )
        assert not np.any(np.isnan(mean_occ))
        assert matrix.shape == (4, NUM_BINS)

    def test_subset_of_cells(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        individual = occ["nucleus"]["real"]["individual"]
        subset = list(individual.keys())[:3]
        _, matrix = _compute_mean_occupancy_from_cells(individual, subset, XVALS)
        assert matrix.shape == (3, NUM_BINS)

    def test_raises_when_no_valid_cells(self):
        with pytest.raises(ValueError, match="No valid cells"):
            _compute_mean_occupancy_from_cells({}, ["nonexistent"], XVALS)


# ---------------------------------------------------------------------------
# Tests: D1.1 — NaN-aware fit target with short-support cells
# ---------------------------------------------------------------------------


def _make_short_support_individual(
    n_full: int,
    n_short: int,
    level: float = 2.0,
    short_x_max: float = 3.0,
    x_max: float = 6.0,
    common_xvals: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Individual dict where ``n_short`` cells' support stops at ``short_x_max``.

    Every cell carries constant occupancy ``level``, so the NaN-aware mean is
    exactly ``level`` at every grid point, whereas zero-filling the unsupported
    tail drags the mean down to ``level * n_full / (n_full + n_short)`` beyond
    ``short_x_max``.  Passing ``common_xvals`` also attaches a per-cell
    ``occupancy_common`` row (NaN outside support), as the production occupancy
    pipelines do.
    """
    individual: dict[str, dict[str, np.ndarray]] = {}

    def _entry(hi: float, n: int) -> dict[str, np.ndarray]:
        entry = {"xvals": np.linspace(0.0, hi, n), "occupancy": np.full(n, level)}
        if common_xvals is not None:
            occ_common = np.full(common_xvals.shape, level)
            occ_common[common_xvals > hi] = np.nan
            entry["occupancy_common"] = occ_common
        return entry

    for i in range(n_full):
        individual[f"full_{i:03d}"] = _entry(x_max, 40)
    for i in range(n_short):
        individual[f"short_{i:03d}"] = _entry(short_x_max, 20)
    return individual


class TestFitTargetShortSupport:
    """D1.1: the fit target is the NaN-aware mean, not a zero-filled mean."""

    def test_target_does_not_decay_in_tail(self):
        common = np.linspace(0, 6, 60)
        individual = _make_short_support_individual(n_full=2, n_short=2)
        mean_occ, matrix = _compute_mean_occupancy_from_cells(
            individual, list(individual.keys()), common
        )
        # Constant-level cells => the NaN-aware mean is `level` everywhere.
        np.testing.assert_allclose(mean_occ, 2.0)
        # Short cells contribute NaN beyond their support, never 0.
        tail = common > 3.0
        assert np.isnan(matrix[:, tail]).sum() == 2 * tail.sum()

    def test_target_matches_nanmean_combined_curve(self):
        """Target == the plotted `nanmean` combined curve wherever it is finite."""
        common = np.linspace(0, 6, 60)
        individual = _make_short_support_individual(n_full=3, n_short=2, common_xvals=common)
        # Mirror occupancy.py: combined = nanmean over per-cell `occupancy_common`.
        combined = np.nanmean(
            np.vstack([v["occupancy_common"] for v in individual.values()]), axis=0
        )
        mean_occ, _ = _compute_mean_occupancy_from_cells(
            individual, list(individual.keys()), common
        )
        finite = np.isfinite(combined)
        assert finite.any()
        np.testing.assert_allclose(mean_occ[finite], combined[finite])

    def test_prefers_precomputed_occupancy_common(self):
        """When present, `occupancy_common` is used verbatim (no re-interpolation)."""
        common = np.linspace(0, 6, 60)
        sentinel = np.linspace(1.0, 2.0, 60)
        individual = {
            "c0": {
                "xvals": np.linspace(0, 6, 40),
                "occupancy": np.full(40, 99.0),
                "occupancy_common": sentinel.copy(),
            }
        }
        mean_occ, matrix = _compute_mean_occupancy_from_cells(individual, ["c0"], common)
        np.testing.assert_allclose(mean_occ, sentinel)
        np.testing.assert_allclose(matrix[0], sentinel)

    def test_ignores_mismatched_occupancy_common_length(self):
        """A stale `occupancy_common` of the wrong length falls back to interp."""
        common = np.linspace(0, 6, 60)
        individual = {
            "c0": {
                "xvals": np.linspace(0, 6, 40),
                "occupancy": np.full(40, 2.0),
                "occupancy_common": np.full(11, 99.0),
            }
        }
        mean_occ, _ = _compute_mean_occupancy_from_cells(individual, ["c0"], common)
        np.testing.assert_allclose(mean_occ, 2.0)


# ---------------------------------------------------------------------------
# Tests: D1.2 — masked rows are dropped from the least-squares system
# ---------------------------------------------------------------------------


class TestMaskedLeastSquares:
    """D1.2: masked support is excluded from the fit, not driven to zero."""

    def test_valid_row_mask_requires_target_and_all_columns(self):
        target = np.array([1.0, np.nan, 3.0, 4.0])
        design = np.array([[1.0, 1.0], [1.0, 1.0], [np.nan, 1.0], [1.0, 1.0]])
        np.testing.assert_array_equal(_valid_row_mask(target, design), [True, False, False, True])

    def test_solved_row_count_excludes_masked_rows(self):
        rng = np.random.default_rng(3)
        design = np.abs(rng.normal(size=(20, 2))) + 0.5
        target = design @ np.array([1.5, 0.5])
        design[15:, 0] = np.nan  # 5 rows lose a column
        target[2] = np.nan  # 1 more row loses the target
        coeffs, valid = _solve_nnls_masked(target, design, "test")
        assert int(valid.sum()) == 14
        expected, _ = nnls(design[valid], target[valid])
        np.testing.assert_allclose(coeffs, expected)

    def test_dropping_differs_from_zero_filling(self):
        """The drop-vs-zero distinction changes the coefficient, so it is not cosmetic."""
        design = np.full((20, 1), 2.0)
        target = np.concatenate([np.full(10, 3.0), np.full(10, np.nan)])
        coeffs, valid = _solve_nnls_masked(target, design, "test")
        assert int(valid.sum()) == 10
        # Only the 10 supported rows are fitted: 3.0 / 2.0.
        np.testing.assert_allclose(coeffs, [1.5])
        # Zero-filling the unsupported half instead makes those rows vote
        # "occupancy 0", halving the fitted coefficient.
        zero_filled, _ = nnls(design, np.nan_to_num(target, nan=0.0))
        np.testing.assert_allclose(zero_filled, [0.75])

    def test_zero_rows_deflate_mse_and_inflate_n(self):
        """A fully zero-filled row is a free perfect fit: deflates MSE, inflates n."""
        design = np.vstack([np.full((10, 1), 2.0), np.full((10, 1), np.nan)])
        target = np.concatenate([np.full(10, 5.0), np.full(10, np.nan)])
        coeffs, valid = _solve_nnls_masked(target, design, "test")
        recon = _masked_reconstruction(design, coeffs)
        masked_mse = _masked_mse(target, recon)
        # Zero-filling both sides turns the masked half into 10 zero-residual
        # rows: same total squared error spread over twice as many points.
        zf_mse = float(
            np.mean((np.nan_to_num(target, nan=0.0) - np.nan_to_num(recon, nan=0.0)) ** 2)
        )
        assert zf_mse == pytest.approx(masked_mse / 2.0)
        assert int(valid.sum()) == 10 and target.size == 20

    def test_reconstruction_is_nan_on_masked_rows(self):
        design = np.array([[1.0, 2.0], [1.0, 2.0], [np.nan, 2.0]])
        recon = _masked_reconstruction(design, np.array([1.0, 0.5]))
        np.testing.assert_allclose(recon[:2], [2.0, 2.0])
        assert np.isnan(recon[2])

    def test_masked_mse_ignores_masked_points(self):
        target = np.array([1.0, 2.0, np.nan, 4.0])
        recon = np.array([1.0, 3.0, 100.0, 4.0])
        assert _masked_mse(target, recon) == pytest.approx(1.0 / 3.0)

    def test_masked_mse_all_masked_is_nan(self):
        assert np.isnan(_masked_mse(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))

    def test_raises_when_too_few_valid_rows(self):
        target = np.array([1.0, np.nan, np.nan])
        design = np.ones((3, 2))
        with pytest.raises(ValueError, match="fewer than the 2 coefficient"):
            _solve_nnls_masked(target, design, "test")

    def test_fit_recovers_coefficient_despite_masked_tail(self):
        """End-to-end: a mode masked in the tail is fitted on its support only."""
        common = np.linspace(0, 6, 50)
        supported = common <= 4.0
        mode_curve = np.where(supported, 2.0, np.nan)
        # Baseline cells are exactly 1.5x the mode over the full grid.
        individual = {
            f"c{i}": {"xvals": common.copy(), "occupancy": np.full(common.size, 3.0)}
            for i in range(4)
        }
        occ = {
            "nucleus": {
                "real": {
                    "individual": individual,
                    "combined": {"xvals": common, "occupancy": np.full(common.size, 3.0)},
                },
                "random": {
                    "individual": {},
                    "combined": {"xvals": common, "occupancy": mode_curve},
                },
            }
        }
        fit = fit_rule_interpolation(occ, _make_channel_map("real", ["random"]), "real")
        assert fit.coefficients["random"]["nucleus"] == pytest.approx(1.5)
        recon = fit.reconstructed_occupancy["nucleus"]["individual"]
        np.testing.assert_allclose(recon[supported], 3.0)
        assert np.all(np.isnan(recon[~supported]))
        # MSE is computed over supported rows only -> an exact fit is exactly 0.
        assert fit.train_mse["individual"]["nucleus"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: D1.5 — the "expected" curve and the recipe weights share a normalization
# ---------------------------------------------------------------------------


class TestExpectedOccupancyNormalization:
    """D1.5: the drawn expected curve must be realizable by the recipe."""

    @staticmethod
    def _fit_two_mode(scale_a: float, scale_b: float):
        """Fit a baseline that is `scale_a`*mode_a + `scale_b`*mode_b exactly."""
        common = np.linspace(0, 6, 40)
        mode_a = np.linspace(1.0, 2.0, 40)
        mode_b = np.linspace(2.0, 0.5, 40)
        target = scale_a * mode_a + scale_b * mode_b
        individual = {
            f"c{i}": {"xvals": common.copy(), "occupancy": target.copy()} for i in range(4)
        }
        occ = {
            "nucleus": {
                "real": {
                    "individual": individual,
                    "combined": {"xvals": common, "occupancy": target},
                },
                "mode_a": {
                    "individual": {},
                    "combined": {"xvals": common, "occupancy": mode_a},
                },
                "mode_b": {
                    "individual": {},
                    "combined": {"xvals": common, "occupancy": mode_b},
                },
            }
        }
        cm = _make_channel_map("real", ["mode_a", "mode_b"])
        return fit_rule_interpolation(occ, cm, "real"), occ, {"mode_a": mode_a, "mode_b": mode_b}

    def test_expected_equals_recipe_weighted_mixture(self):
        """expected_occupancy == sum(gradient_weights_i * M_i), exactly."""
        fit, _, curves = self._fit_two_mode(0.8, 0.4)  # sum(c) = 1.2 != 1
        for plot_type in ("individual", "joint"):
            coeffs = (
                fit.coefficients_individual["nucleus"]
                if plot_type == "individual"
                else fit.coefficients_joint
            )
            weights = _get_gradient_weights_from_coefficients(
                coeffs, fit.packing_modes, {m: m for m in fit.packing_modes}
            )
            assert sum(weights.values()) == pytest.approx(1.0)
            realizable = sum(weights[m] * curves[m] for m in fit.packing_modes)
            np.testing.assert_allclose(fit.expected_occupancy("nucleus", plot_type), realizable)

    def test_expected_differs_from_raw_reconstruction_when_sum_not_one(self):
        """The raw reconstruction is the un-realizable target this fixes."""
        fit, _, _ = self._fit_two_mode(0.8, 0.4)
        raw = fit.reconstructed_occupancy["nucleus"]["joint"]
        expected = fit.expected_occupancy("nucleus", "joint")
        total = sum(fit.coefficients_joint.values())
        assert total == pytest.approx(1.2)
        np.testing.assert_allclose(expected, raw / total)
        # Raw sits high by exactly the coefficient sum's excess.
        assert np.nanmax(raw) > np.nanmax(expected)

    def test_expected_matches_raw_when_coefficients_sum_to_one(self):
        fit, _, _ = self._fit_two_mode(0.6, 0.4)  # sum(c) = 1
        np.testing.assert_allclose(
            fit.expected_occupancy("nucleus", "joint"),
            fit.reconstructed_occupancy["nucleus"]["joint"],
        )

    def test_expected_preserves_nan_mask(self):
        fit, _, _ = self._fit_two_mode(0.8, 0.4)
        fit.reconstructed_occupancy["nucleus"]["joint"][:3] = np.nan
        assert np.all(np.isnan(fit.expected_occupancy("nucleus", "joint")[:3]))

    def test_expected_returns_reconstruction_when_all_coefficients_zero(self):
        fit, _, _ = self._fit_two_mode(0.8, 0.4)
        for mode in fit.packing_modes:
            fit.coefficients[mode]["joint"] = 0.0
        np.testing.assert_allclose(
            fit.expected_occupancy("nucleus", "joint"),
            fit.reconstructed_occupancy["nucleus"]["joint"],
        )


# ---------------------------------------------------------------------------
# Tests: uncovered grid points render as a gap, not a plunge to zero
# ---------------------------------------------------------------------------


class TestOccupancyPlotsPreserveGaps:
    """A3: zero-filling a coverage-masked tail draws a false "occupancy = 0"."""

    @staticmethod
    def _occ_with_masked_tail():
        common = np.linspace(0, 6, 30)
        masked = common > 4.5
        curve = np.where(masked, np.nan, 1.2)
        individual = {
            f"c{i}": {"xvals": common.copy(), "occupancy": np.full(common.size, 1.2)}
            for i in range(3)
        }
        occ = {
            "nucleus": {
                "real": {
                    "individual": individual,
                    "combined": {"xvals": common, "occupancy": curve.copy()},
                },
                "random": {
                    "individual": {},
                    "combined": {"xvals": common, "occupancy": curve.copy()},
                },
            }
        }
        return occ, _make_channel_map("real", ["random"]), masked

    def test_plot_rule_interpolation_fit_keeps_nan_gap(self):
        from cellpack_analysis.lib.visualization import plot_rule_interpolation_fit

        occ, cm, masked = self._occ_with_masked_tail()
        fit = fit_rule_interpolation(occ, cm, "real")
        _fig, ax = plot_rule_interpolation_fit(
            fit_result=fit,
            occupancy_dict=occ,
            channel_map=cm,
            baseline_mode="real",
            distance_measure="nucleus",
            figures_dir=None,
        )
        # Every drawn occupancy curve must retain the gap, not drop to zero.
        drawn = [
            np.asarray(line.get_ydata(), dtype=float)
            for line in ax.get_lines()
            if np.asarray(line.get_ydata(), dtype=float).size == masked.size
        ]
        assert drawn, "expected at least one full-length curve on the axes"
        for ydata in drawn:
            assert np.all(np.isnan(ydata[masked])), ydata[masked]

    def test_plot_occupancy_ratio_keeps_nan_gap(self):
        from cellpack_analysis.lib.visualization import plot_occupancy_ratio

        occ, cm, masked = self._occ_with_masked_tail()
        _fig, ax = plot_occupancy_ratio(
            occupancy_dict=occ["nucleus"],
            channel_map=cm,
            baseline_mode="real",
            distance_measure="nucleus",
            figures_dir=None,
        )
        drawn = [
            np.asarray(line.get_ydata(), dtype=float)
            for line in ax.get_lines()
            if np.asarray(line.get_ydata(), dtype=float).size == masked.size
        ]
        assert drawn
        for ydata in drawn:
            assert np.all(np.isnan(ydata[masked]))


# ---------------------------------------------------------------------------
# Tests: _normalize_coefficients
# ---------------------------------------------------------------------------


class TestNormalizeCoefficients:
    def test_relative_contributions_sum_to_one(self):
        coeffs = np.array([0.3, 0.5, 0.2])
        modes = ["a", "b", "c"]
        _, rel_dict = _normalize_coefficients(coeffs, modes)
        assert abs(sum(rel_dict.values()) - 1.0) < 1e-10

    def test_zero_total_returns_zeros(self):
        coeffs = np.array([0.0, 0.0])
        modes = ["a", "b"]
        _, rel_dict = _normalize_coefficients(coeffs, modes)
        assert all(v == 0.0 for v in rel_dict.values())

    def test_coeff_dict_values_match_input(self):
        coeffs = np.array([1.0, 2.0])
        modes = ["x", "y"]
        coeff_dict, _ = _normalize_coefficients(coeffs, modes)
        assert coeff_dict["x"] == pytest.approx(1.0)
        assert coeff_dict["y"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests: fit_rule_interpolation
# ---------------------------------------------------------------------------


class TestFitRuleInterpolation:
    def test_returns_fitresult(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=8)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        assert isinstance(result, FitResult)

    def test_coefficients_nonneg(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=8)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        for dm, mode_coeffs in result.coefficients_individual.items():
            for mode, coeff in mode_coeffs.items():
                assert coeff >= 0.0, f"Negative coeff for {mode} in {dm}"
        for coeff in result.coefficients_joint.values():
            assert coeff >= 0.0

    def test_relative_contributions_sum_to_one(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=8)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        for dm in result.distance_measures:
            total = sum(result.relative_contributions_individual[dm].values())
            assert abs(total - 1.0) < 1e-9, f"Relative contributions don't sum to 1 for {dm}"
        total_joint = sum(result.relative_contributions_joint.values())
        assert abs(total_joint - 1.0) < 1e-9

    def test_reconstructed_occupancy_shape(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=6)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        for dm in result.distance_measures:
            assert result.reconstructed_occupancy[dm]["individual"].shape == (NUM_BINS,)
            assert result.reconstructed_occupancy[dm]["joint"].shape == (NUM_BINS,)

    def test_train_cell_ids_recorded(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=5)
        channel_map = _make_channel_map("real", ["random"])
        train_cells = ["cell_000", "cell_001", "cell_002"]
        result = fit_rule_interpolation(occ, channel_map, "real", train_cell_ids=train_cells)
        assert result.train_cell_ids == train_cells

    def test_mse_is_nonneg(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        for mse in result.train_mse_individual.values():
            assert mse >= 0.0
        for mse in result.train_mse_joint.values():
            assert mse >= 0.0

    def test_multiple_distance_measures(self):
        occ = _make_occupancy_dict(
            "real", ["random", "nucleus"], n_baseline_cells=6, distance_measures=["nucleus", "z"]
        )
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        assert set(result.distance_measures) == {"nucleus", "z"}
        assert "nucleus" in result.train_mse_individual
        assert "z" in result.train_mse_individual

    def test_raises_for_missing_baseline(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=4)
        channel_map_with_missing = {"missing_baseline": "STRUCT_A", "random": "STRUCT_A"}
        with pytest.raises((KeyError, ValueError)):
            fit_rule_interpolation(occ, channel_map_with_missing, "missing_baseline")

    def test_raises_for_no_simulated_modes(self):
        occ = _make_occupancy_dict("real", [], n_baseline_cells=4)
        channel_map = {"real": "STRUCT_A"}
        with pytest.raises(ValueError, match="No simulated packing modes"):
            fit_rule_interpolation(occ, channel_map, "real")

    def test_recovery_when_one_mode_is_true_generator(self):
        """When one simulated mode matches the baseline exactly, its coefficient
        should dominate."""
        xvals = np.linspace(0, 5, 50)
        true_occ = np.exp(-0.5 * xvals) + 0.1
        # Baseline individual cells: all identical to true_occ
        individual = {
            f"cell_{i}": {"xvals": xvals, "occupancy": true_occ.copy()} for i in range(10)
        }
        occ = {
            "nucleus": {
                "real": {
                    "individual": individual,
                    "combined": {"xvals": xvals, "occupancy": true_occ},
                },
                "true_mode": {
                    "individual": {},
                    "combined": {"xvals": xvals, "occupancy": true_occ},
                },
                "noise_mode": {
                    "individual": {},
                    "combined": {
                        "xvals": xvals,
                        "occupancy": np.random.default_rng(0).uniform(0, 2, 50),
                    },
                },
            }
        }
        channel_map = {"real": "S", "true_mode": "S", "noise_mode": "S"}
        result = fit_rule_interpolation(occ, channel_map, "real")
        # true_mode should get a substantially larger relative contribution
        rel = result.relative_contributions_individual["nucleus"]
        assert rel["true_mode"] > rel["noise_mode"]


# ---------------------------------------------------------------------------
# Tests: run_rule_interpolation_cv
# ---------------------------------------------------------------------------


class TestRunRuleInterpolationCV:
    def test_returns_cv_result(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=12)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=42
        )
        assert isinstance(cv, CVResult)
        assert cv.n_folds == 3
        assert cv.n_repeats == 1

    def test_correct_number_of_folds(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=4, n_repeats=1, random_state=0
        )
        assert len(cv.folds) == 4

    def test_train_test_splits_partition_all_cells(self):
        n_cells = 15
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=n_cells)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=5, n_repeats=1, random_state=1
        )
        all_cell_ids = _get_baseline_cell_ids(occ, "real")
        for fold in cv.folds:
            # Train + test = all cells
            combined = set(fold.train_cell_ids) | set(fold.test_cell_ids)
            assert combined == set(all_cell_ids)
            # No overlap
            assert len(set(fold.train_cell_ids) & set(fold.test_cell_ids)) == 0

    def test_aggregated_coefficients_have_correct_structure(self):
        occ = _make_occupancy_dict(
            "real", ["random", "nucleus"], n_baseline_cells=10, distance_measures=["nucleus", "z"]
        )
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=5
        )
        for dm in ["nucleus", "z"]:
            assert dm in cv.aggregated_coefficients_individual
            for mode in ["random", "nucleus"]:
                mean, std = cv.aggregated_coefficients_individual[dm][mode]
                assert isinstance(mean, float)
                assert std >= 0.0
        for mode in ["random", "nucleus"]:
            mean, std = cv.aggregated_coefficients_joint[mode]
            assert isinstance(mean, float)

    def test_raises_when_too_few_cells(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=3)
        channel_map = _make_channel_map("real", ["random"])
        with pytest.raises(ValueError, match="folds"):
            run_rule_interpolation_cv(occ, channel_map, "real", n_folds=5, n_repeats=1)

    def test_cache_round_trip(self, tmp_path):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random"])
        cv1 = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=7, results_dir=tmp_path
        )
        # Second call should load from cache
        cv2 = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=7, results_dir=tmp_path
        )
        assert cv1.n_folds == cv2.n_folds
        assert cv1.baseline_mode == cv2.baseline_mode

    def test_recalculate_ignores_cache(self, tmp_path):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random"])
        _ = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=7, results_dir=tmp_path
        )
        # Corrupt the cache
        cache_path = tmp_path / "rule_interpolation_cv.pkl"
        cache_path.write_bytes(b"garbage")
        # recalculate=True must bypass cache
        cv2 = run_rule_interpolation_cv(
            occ,
            channel_map,
            "real",
            n_folds=3,
            n_repeats=1,
            random_state=7,
            results_dir=tmp_path,
            recalculate=True,
        )
        assert cv2.n_folds == 3

    def test_test_mse_shape_matches_test_cell_count(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=12)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=4, n_repeats=1, random_state=2
        )
        for fold in cv.folds:
            n_test = len(fold.test_cell_ids)
            for scope in ("individual", "joint"):
                for dm, arr in fold.test_mse[scope].items():
                    assert len(arr) == n_test, (
                        f"fold {fold.fold_idx}, scope={scope}, dm={dm}: "
                        f"expected {n_test} test cells, got {len(arr)}"
                    )

    def test_reproducibility_with_same_seed(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=12)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        cv1 = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=99
        )
        cv2 = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=99
        )
        for f1, f2 in zip(cv1.folds, cv2.folds, strict=True):
            assert f1.train_cell_ids == f2.train_cell_ids
            assert f1.test_cell_ids == f2.test_cell_ids

    def test_n_repeats_produces_correct_fold_count(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=12)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=2, random_state=0
        )
        assert cv.n_repeats == 2
        assert len(cv.folds) == 3 * 2

    def test_repeat_idx_values(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=12)
        channel_map = _make_channel_map("real", ["random"])
        n_repeats = 3
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=2, n_repeats=n_repeats, random_state=1
        )
        for fold in cv.folds:
            assert 0 <= fold.repeat_idx < n_repeats

    def test_reproducibility_with_repeats(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=12)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        cv1 = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=2, random_state=7
        )
        cv2 = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=2, random_state=7
        )
        for f1, f2 in zip(cv1.folds, cv2.folds, strict=True):
            assert f1.train_cell_ids == f2.train_cell_ids
            assert f1.test_cell_ids == f2.test_cell_ids
            assert f1.repeat_idx == f2.repeat_idx


# ---------------------------------------------------------------------------
# Tests: summarize_cv_results
# ---------------------------------------------------------------------------


class TestSummarizeCVResults:
    def test_returns_dataframe(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=0
        )
        df = summarize_cv_results(cv)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=0
        )
        df = summarize_cv_results(cv)
        for col in ("repeat_idx", "fold_idx", "scope", "distance_measure", "split", "mse"):
            assert col in df.columns

    def test_contains_both_train_and_test_rows(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=0
        )
        df = summarize_cv_results(cv)
        assert "train" in df["split"].values
        assert "test" in df["split"].values

    def test_all_mse_nonneg(self):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=12)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        cv = run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=3
        )
        df = summarize_cv_results(cv)
        assert (df["mse"] >= 0).all()


# ---------------------------------------------------------------------------
# Tests: generate_mixed_rule_packing_configs
# ---------------------------------------------------------------------------


class TestGenerateMixedRulePackingConfigs:
    def _base_config(self, tmp_path: Path) -> Path:
        """Write a minimal cellPACK workflow config to tmp_path."""
        config = {
            "structure_name": "test_struct",
            "structure_id": "TEST",
            "condition": "test",
            "recipe_data": {
                "random": {},
                "nucleus_gradient": {"gradients": ["nucleus_gradient"]},
            },
            "packings_to_run": {"rules": ["random", "nucleus_gradient"]},
        }
        p = tmp_path / "base_config.json"
        with open(p, "w") as fh:
            json.dump(config, fh)
        return p

    def _make_cv_result(self) -> CVResult:
        occ = _make_occupancy_dict("real", ["nucleus_gradient", "random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["nucleus_gradient", "random"])
        return run_rule_interpolation_cv(
            occ, channel_map, "real", n_folds=3, n_repeats=1, random_state=11
        )

    def test_writes_one_config_per_fold_plus_aggregated(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        mode_to_grad = {"nucleus_gradient": "nucleus_gradient", "random": "uniform"}
        paths = generate_mixed_rule_packing_configs(
            cv_result=cv,
            base_config_path=base,
            output_config_dir=tmp_path / "configs",
            mode_to_gradient_name=mode_to_grad,
            scope="joint",
        )
        # n_folds configs + 1 aggregated
        assert len(paths) == cv.n_folds + 1

    def test_gradient_weights_sum_to_one(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        mode_to_grad = {"nucleus_gradient": "nucleus_gradient", "random": "uniform"}
        paths = generate_mixed_rule_packing_configs(
            cv_result=cv,
            base_config_path=base,
            output_config_dir=tmp_path / "configs",
            mode_to_gradient_name=mode_to_grad,
            scope="joint",
        )
        for p in paths:
            with open(p) as fh:
                cfg = json.load(fh)
            weights = cfg["recipe_data"]["interpolated"]["gradient_weights"]
            assert abs(sum(weights.values()) - 1.0) < 1e-9, (
                f"Weights don't sum to 1 in {p.name}: {weights}"
            )

    def test_packings_to_run_is_interpolated_only(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        mode_to_grad = {"nucleus_gradient": "nucleus_gradient", "random": "uniform"}
        paths = generate_mixed_rule_packing_configs(
            cv_result=cv,
            base_config_path=base,
            output_config_dir=tmp_path / "configs",
            mode_to_gradient_name=mode_to_grad,
        )
        for p in paths:
            with open(p) as fh:
                cfg = json.load(fh)
            assert cfg["packings_to_run"]["rules"] == ["interpolated"]

    def test_fold_configs_include_test_cell_ids(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        mode_to_grad = {"nucleus_gradient": "nucleus_gradient", "random": "uniform"}
        paths = generate_mixed_rule_packing_configs(
            cv_result=cv,
            base_config_path=base,
            output_config_dir=tmp_path / "configs",
            mode_to_gradient_name=mode_to_grad,
        )
        # All fold configs (not the aggregated one) should have cell_ids
        fold_paths = [p for p in paths if "aggregated" not in p.name]
        for p in fold_paths:
            with open(p) as fh:
                cfg = json.load(fh)
            assert "cell_ids" in cfg
            assert len(cfg["cell_ids"]) > 0

    def test_aggregated_config_has_no_cell_ids(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        mode_to_grad = {"nucleus_gradient": "nucleus_gradient", "random": "uniform"}
        paths = generate_mixed_rule_packing_configs(
            cv_result=cv,
            base_config_path=base,
            output_config_dir=tmp_path / "configs",
            mode_to_gradient_name=mode_to_grad,
        )
        agg_paths = [p for p in paths if "aggregated" in p.name]
        assert len(agg_paths) == 1
        with open(agg_paths[0]) as fh:
            cfg = json.load(fh)
        assert "cell_ids" not in cfg

    def test_single_fold_config(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        mode_to_grad = {"nucleus_gradient": "nucleus_gradient", "random": "uniform"}
        paths = generate_mixed_rule_packing_configs(
            cv_result=cv,
            base_config_path=base,
            output_config_dir=tmp_path / "configs",
            mode_to_gradient_name=mode_to_grad,
            fold_idx=0,
        )
        # Only fold0 + aggregated (fold_idx=None is skipped when fold_idx is set)
        assert len(paths) == 1

    def test_raises_if_individual_scope_without_dm(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        with pytest.raises(ValueError, match="distance_measure"):
            generate_mixed_rule_packing_configs(
                cv_result=cv,
                base_config_path=base,
                output_config_dir=tmp_path / "configs",
                mode_to_gradient_name={"nucleus_gradient": "nucleus_gradient"},
                scope="individual",
            )

    def test_dry_run_writes_no_files(self, tmp_path):
        cv = self._make_cv_result()
        base = self._base_config(tmp_path)
        mode_to_grad = {"nucleus_gradient": "nucleus_gradient", "random": "uniform"}
        out_dir = tmp_path / "dry_run_configs"
        paths = generate_mixed_rule_packing_configs(
            cv_result=cv,
            base_config_path=base,
            output_config_dir=out_dir,
            mode_to_gradient_name=mode_to_grad,
            dry_run=True,
        )
        assert paths == []
        # Directory should not be created during dry_run
        assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Tests: logging helpers (smoke tests)
# ---------------------------------------------------------------------------


class TestLoggingHelpers:
    def test_log_rule_interpolation_coeffs_runs(self, tmp_path):
        occ = _make_occupancy_dict("real", ["random", "nucleus"], n_baseline_cells=6)
        channel_map = _make_channel_map("real", ["random", "nucleus"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        log_path = tmp_path / "interp_coeffs.log"
        log_rule_interpolation_coeffs(result, "real", file_path=log_path)
        assert log_path.exists()
        assert log_path.stat().st_size > 0

    def test_log_cv_summary_runs(self, tmp_path):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=10)
        channel_map = _make_channel_map("real", ["random"])
        cv = run_rule_interpolation_cv(occ, channel_map, "real", n_folds=3, random_state=0)
        log_path = tmp_path / "cv_summary.log"
        log_cv_summary(cv, file_path=log_path)
        assert log_path.exists()
        assert log_path.stat().st_size > 0

    def test_log_rule_interpolation_coeffs_no_file(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=4)
        channel_map = _make_channel_map("real", ["random"])
        result = fit_rule_interpolation(occ, channel_map, "real")
        # Should not raise when file_path=None
        log_rule_interpolation_coeffs(result, "real", file_path=None)


# ---------------------------------------------------------------------------
# Tests: _compute_aic / _compute_aicc / _compute_bic
# ---------------------------------------------------------------------------


class TestComputeAIC:
    """Tests for the pure AIC/AICc/BIC functions."""

    def test_aic_known_values(self):
        # n=100, k=2, RSS=50 → k_eff=3, AIC = 100*ln(0.5) + 6
        n, k, rss = 100, 2, 50.0
        expected = 100 * np.log(0.5) + 2 * 3
        assert _compute_aic(n, k, rss) == pytest.approx(expected)

    def test_bic_known_values(self):
        # n=100, k=2, RSS=50 → k_eff=3, BIC = 100*ln(0.5) + 3*ln(100)
        n, k, rss = 100, 2, 50.0
        expected = 100 * np.log(0.5) + 3 * np.log(100)
        assert _compute_bic(n, k, rss) == pytest.approx(expected)

    def test_aicc_agrees_with_aic_for_large_n(self):
        # For very large n, AICc → AIC
        n, k, rss = 100_000, 3, 500.0
        aic = _compute_aic(n, k, rss)
        aicc = _compute_aicc(n, k, rss)
        assert aicc == pytest.approx(aic, rel=1e-3)

    def test_aicc_correction_positive(self):
        # AICc should be >= AIC
        n, k, rss = 20, 3, 10.0
        assert _compute_aicc(n, k, rss) >= _compute_aic(n, k, rss)

    def test_aicc_inf_when_insufficient_data(self):
        # n - k_eff - 1 <= 0 should give inf
        assert _compute_aicc(3, 2, 1.0) == float("inf")

    def test_lower_rss_gives_lower_aic(self):
        aic_low = _compute_aic(50, 2, 10.0)
        aic_high = _compute_aic(50, 2, 100.0)
        assert aic_low < aic_high

    def test_more_params_increases_aic(self):
        # Same RSS, more params → higher AIC penalty
        aic_simple = _compute_aic(50, 1, 10.0)
        aic_complex = _compute_aic(50, 5, 10.0)
        assert aic_complex > aic_simple

    def test_bic_penalizes_more_than_aic_for_large_n(self):
        # For n > e^2 ≈ 7.4, BIC penalty > AIC penalty per parameter
        n, k, rss = 100, 3, 50.0
        bic = _compute_bic(n, k, rss)
        aic = _compute_aic(n, k, rss)
        # BIC has k_eff*ln(n) vs AIC has 2*k_eff; ln(100) > 2
        assert bic > aic


# ---------------------------------------------------------------------------
# Tests: _fit_single_rule
# ---------------------------------------------------------------------------


class TestFitSingleRule:
    """Tests for the single-mode NNLS helper."""

    def test_returns_rss_n_and_coeff(self):
        rng = np.random.default_rng(42)
        b = rng.random(30)
        mode_occ = rng.random(30)
        rss, n, coeff = _fit_single_rule(b, mode_occ)
        assert n == 30
        assert rss >= 0.0
        assert coeff >= 0.0

    def test_perfect_fit_gives_zero_rss(self):
        mode_occ = np.array([1.0, 2.0, 3.0])
        b = 2.0 * mode_occ
        rss, _, coeff = _fit_single_rule(b, mode_occ)
        assert rss == pytest.approx(0.0, abs=1e-10)
        assert coeff == pytest.approx(2.0)

    def test_rss_nonnegative(self):
        rng = np.random.default_rng(99)
        b = rng.random(50)
        mode_occ = rng.random(50)
        rss, _, _ = _fit_single_rule(b, mode_occ)
        assert rss >= 0.0

    def test_n_counts_only_unmasked_rows(self):
        """Masked grid points are excluded from n, not scored as zero residuals."""
        mode_occ = np.concatenate([np.full(20, 2.0), np.full(10, np.nan)])
        b = np.concatenate([np.full(20, 5.0), np.full(10, 5.0)])
        rss, n, coeff = _fit_single_rule(b, mode_occ)
        assert n == 20
        assert coeff == pytest.approx(2.5)
        assert rss == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests: compute_aic_comparison
# ---------------------------------------------------------------------------


class TestComputeAICComparison:
    """Tests for the full AIC/BIC model comparison."""

    def test_returns_aic_comparison_result(self):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        result = compute_aic_comparison(occ, channel_map, "real")
        assert isinstance(result, AICComparisonResult)

    def test_akaike_weights_sum_to_one(self):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        result = compute_aic_comparison(occ, channel_map, "real")
        for scope_dict in result.akaike_weights.values():
            for dm, weights in scope_dict.items():
                total = sum(weights.values())
                assert total == pytest.approx(1.0, abs=1e-10)

    def test_bic_weights_sum_to_one(self):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        result = compute_aic_comparison(occ, channel_map, "real")
        for scope_dict in result.bic_weights.values():
            for dm, weights in scope_dict.items():
                total = sum(weights.values())
                assert total == pytest.approx(1.0, abs=1e-10)

    def test_delta_aic_best_is_zero(self):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        result = compute_aic_comparison(occ, channel_map, "real")
        for scope_dict in result.delta_aic.values():
            for dm, deltas in scope_dict.items():
                best = result.best_model_aic[
                    "individual" if dm != "joint" else "joint"
                ][dm]
                assert deltas[best] == pytest.approx(0.0)

    def test_delta_bic_best_is_zero(self):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        result = compute_aic_comparison(occ, channel_map, "real")
        for scope_dict in result.delta_bic.values():
            for dm, deltas in scope_dict.items():
                best = result.best_model_bic[
                    "individual" if dm != "joint" else "joint"
                ][dm]
                assert deltas[best] == pytest.approx(0.0)

    def test_correct_k_values(self):
        modes = ["random", "gradient"]
        occ = _make_occupancy_dict("real", modes, n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", modes)
        result = compute_aic_comparison(occ, channel_map, "real")
        n_modes = len(modes)
        for scope_dict in result.comparisons.values():
            for dm, models in scope_dict.items():
                for m in models:
                    if m.model_name == "mixed_rule":
                        assert m.k == n_modes
                    else:
                        assert m.k == 1

    def test_all_deltas_nonnegative(self):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        result = compute_aic_comparison(occ, channel_map, "real")
        for scope_dict in result.delta_aic.values():
            for deltas in scope_dict.values():
                for v in deltas.values():
                    assert v >= -1e-10
        for scope_dict in result.delta_bic.values():
            for deltas in scope_dict.values():
                for v in deltas.values():
                    assert v >= -1e-10

    def test_includes_individual_and_joint_scopes(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random"])
        result = compute_aic_comparison(occ, channel_map, "real")
        assert "individual" in result.comparisons
        assert "joint" in result.comparisons
        assert "joint" in result.comparisons["joint"]

    def test_multiple_distance_measures(self):
        occ = _make_occupancy_dict(
            "real", ["random"], n_baseline_cells=8,
            distance_measures=["nucleus", "z"], seed=42,
        )
        channel_map = _make_channel_map("real", ["random"])
        result = compute_aic_comparison(occ, channel_map, "real")
        assert "nucleus" in result.comparisons["individual"]
        assert "z" in result.comparisons["individual"]

    def test_single_mode_prefers_single_when_baseline_matches(self):
        """When baseline IS the single mode occupancy, AIC should prefer single."""
        # Build occupancy dict where baseline combined = mode's combined
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=10, seed=0)
        channel_map = _make_channel_map("real", ["random", "gradient"])

        # Overwrite all baseline cells to exactly match "random" combined
        for dm in occ:
            random_occ = occ[dm]["random"]["combined"]["occupancy"].copy()
            for cell_id in occ[dm]["real"]["individual"]:
                occ[dm]["real"]["individual"][cell_id]["occupancy"] = random_occ.copy()
                occ[dm]["real"]["individual"][cell_id]["xvals"] = XVALS.copy()
            occ[dm]["real"]["combined"]["occupancy"] = random_occ.copy()

        result = compute_aic_comparison(occ, channel_map, "real")
        # single:random should be best or at least have lower AIC than mixed
        for dm in occ:
            d_aic = result.delta_aic["individual"][dm]
            assert d_aic["single:random"] <= d_aic["mixed_rule"]

    def test_with_precomputed_fit_result(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random"])
        fit = fit_rule_interpolation(occ, channel_map, "real")
        result = compute_aic_comparison(occ, channel_map, "real", fit_result=fit)
        assert isinstance(result, AICComparisonResult)
        # Should give same result as without fit_result
        result2 = compute_aic_comparison(occ, channel_map, "real")
        # Compare mixed_rule AIC in individual scope
        dm = list(occ.keys())[0]
        m1 = [m for m in result.comparisons["individual"][dm] if m.model_name == "mixed_rule"][0]
        m2 = [m for m in result2.comparisons["individual"][dm] if m.model_name == "mixed_rule"][0]
        assert m1.aic == pytest.approx(m2.aic)


# ---------------------------------------------------------------------------
# Tests: D1.4 — consistent AIC/BIC scoring (k = nonzero coeffs, n = fitted rows)
# ---------------------------------------------------------------------------


class TestAICScoringConsistency:
    """D1.4: every model is scored on the same rows, and k counts real parameters."""

    @staticmethod
    def _occ_with_masked_tail(n_points: int = 40, supported_to: float = 4.0):
        """Baseline exactly reproducible from mode_a; mode_b masked in the tail."""
        common = np.linspace(0, 6, n_points)
        supported = common <= supported_to
        mode_a = np.linspace(1.0, 2.0, n_points)
        mode_b = np.where(supported, np.linspace(2.0, 0.5, n_points), np.nan)
        # Rises faster than mode_a, so the unconstrained fit wants a negative
        # mode_b weight and NNLS clamps it to zero -- with a non-zero residual,
        # avoiding the degenerate log(RSS=0) case.
        target = np.linspace(0.5, 3.0, n_points)
        individual = {
            f"c{i}": {"xvals": common.copy(), "occupancy": target.copy()} for i in range(4)
        }
        occ = {
            "nucleus": {
                "real": {
                    "individual": individual,
                    "combined": {"xvals": common, "occupancy": target},
                },
                "mode_a": {
                    "individual": {},
                    "combined": {"xvals": common, "occupancy": mode_a},
                },
                "mode_b": {
                    "individual": {},
                    "combined": {"xvals": common, "occupancy": mode_b},
                },
            }
        }
        return occ, _make_channel_map("real", ["mode_a", "mode_b"]), int(supported.sum())

    def test_count_nonzero_coeffs(self):
        coeffs = {"a": 0.5, "b": 0.0, "c": 1.2}
        assert _count_nonzero_coeffs(coeffs, ["a", "b", "c"]) == 2
        assert _count_nonzero_coeffs(coeffs, ["b"]) == 0

    def test_n_reflects_fitted_rows_not_grid_length(self):
        occ, cm, n_supported = self._occ_with_masked_tail()
        result = compute_aic_comparison(occ, cm, "real")
        for model in result.comparisons["individual"]["nucleus"]:
            assert model.n == n_supported, model.model_name

    def test_all_models_share_one_row_count(self):
        """Comparable information criteria require an identical n across models."""
        occ, cm, _ = self._occ_with_masked_tail()
        result = compute_aic_comparison(occ, cm, "real")
        for scope_dict in result.comparisons.values():
            for models in scope_dict.values():
                assert len({m.n for m in models}) == 1

    def test_k_counts_only_nonzero_coefficients(self):
        """A mode NNLS drives to zero is not charged as a free parameter."""
        occ, cm, _ = self._occ_with_masked_tail()
        fit = fit_rule_interpolation(occ, cm, "real")
        n_nonzero = _count_nonzero_coeffs(fit.coefficients_joint, fit.packing_modes)
        assert n_nonzero == 1  # NNLS zeroes mode_b
        assert fit.coefficients_joint["mode_b"] < 1e-8
        result = compute_aic_comparison(occ, cm, "real", fit_result=fit)
        mixed = next(
            m for m in result.comparisons["joint"]["joint"] if m.model_name == "mixed_rule"
        )
        assert mixed.k == n_nonzero

    def test_joint_n_is_sum_of_per_dm_fitted_rows(self):
        """Joint n sums covered points across DMs, not len(grid) * n_dms."""
        occ, cm, n_supported = self._occ_with_masked_tail()
        occ["z"] = occ["nucleus"]
        result = compute_aic_comparison(occ, cm, "real", distance_measures=["nucleus", "z"])
        joint_mixed = next(
            m for m in result.comparisons["joint"]["joint"] if m.model_name == "mixed_rule"
        )
        assert joint_mixed.n == 2 * n_supported


# ---------------------------------------------------------------------------
# Tests: summarize_aic_comparison
# ---------------------------------------------------------------------------


class TestSummarizeAICComparison:
    """Tests for summarize_aic_comparison."""

    def test_returns_dataframe(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random"])
        aic_result = compute_aic_comparison(occ, channel_map, "real")
        df = summarize_aic_comparison(aic_result)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random"])
        aic_result = compute_aic_comparison(occ, channel_map, "real")
        df = summarize_aic_comparison(aic_result)
        expected_cols = {
            "scope", "distance_measure", "model", "k", "n", "rss",
            "aic", "aicc", "bic", "delta_aic", "delta_bic",
            "akaike_weight", "bic_weight",
        }
        assert expected_cols == set(df.columns)

    def test_row_count_matches_models(self):
        modes = ["random", "gradient"]
        occ = _make_occupancy_dict("real", modes, n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", modes)
        aic_result = compute_aic_comparison(occ, channel_map, "real")
        df = summarize_aic_comparison(aic_result)
        # 1 DM individual: 2 single + 1 mixed + 1 null = 4
        # 1 DM joint: same = 4
        assert len(df) == 8


# ---------------------------------------------------------------------------
# Tests: log_evidence_ratios
# ---------------------------------------------------------------------------


class TestLogEvidenceRatios:
    """Tests for log_evidence_ratios."""

    def test_creates_log_file(self, tmp_path):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        aic_result = compute_aic_comparison(occ, channel_map, "real")
        log_path = tmp_path / "logs" / "test_evidence_ratios.log"
        log_evidence_ratios(aic_result, log_path)
        assert log_path.exists()
        assert log_path.stat().st_size > 0

    def test_log_contains_expected_content(self, tmp_path):
        occ = _make_occupancy_dict("real", ["random", "gradient"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random", "gradient"])
        aic_result = compute_aic_comparison(occ, channel_map, "real")
        log_path = tmp_path / "evidence_ratios.log"
        log_evidence_ratios(aic_result, log_path)
        content = log_path.read_text()
        assert "Evidence Ratio Report" in content
        assert "mixed_rule" in content
        assert "null" in content
        assert "Interpretation guide" in content

    def test_creates_parent_dirs(self, tmp_path):
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random"])
        aic_result = compute_aic_comparison(occ, channel_map, "real")
        log_path = tmp_path / "deep" / "nested" / "evidence.log"
        log_evidence_ratios(aic_result, log_path)
        assert log_path.exists()


# ---------------------------------------------------------------------------
# Tests: ValidationResult with AIC
# ---------------------------------------------------------------------------


class TestValidationResultAIC:
    """Tests for AIC integration in run_mixed_rule_validation."""

    def test_aic_result_none_without_fit_result(self):
        from cellpack_analysis.lib.rule_interpolation import ValidationResult

        vr = ValidationResult(
            emd_df=pd.DataFrame(),
            ks_df=pd.DataFrame(),
            envelope_test={},
        )
        assert vr.aic_result is None

    def test_aic_result_populated_with_fit_result(self):
        from cellpack_analysis.lib.rule_interpolation import ValidationResult

        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=8, seed=42)
        channel_map = _make_channel_map("real", ["random"])
        fit = fit_rule_interpolation(occ, channel_map, "real")
        aic = compute_aic_comparison(occ, channel_map, "real", fit_result=fit)

        vr = ValidationResult(
            emd_df=pd.DataFrame(),
            ks_df=pd.DataFrame(),
            envelope_test={},
            aic_result=aic,
        )
        assert vr.aic_result is not None
        assert isinstance(vr.aic_result, AICComparisonResult)
