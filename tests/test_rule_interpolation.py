"""Unit tests for cellpack_analysis.analysis.rule_interpolation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    _get_baseline_cell_ids,
    _normalize_coefficients,
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
        occ = _make_occupancy_dict("real", ["random"], n_baseline_cells=5, seed=42)
        individual = occ["nucleus"]["real"]["individual"]
        cell_ids = list(individual.keys())
        mean_occ, matrix = _compute_mean_occupancy_from_cells(individual, cell_ids, XVALS)
        np.testing.assert_allclose(mean_occ, np.mean(matrix, axis=0))

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

    def test_returns_rss_and_n(self):
        rng = np.random.default_rng(42)
        b = rng.random(30)
        mode_occ = rng.random(30)
        rss, n = _fit_single_rule(b, mode_occ)
        assert n == 30
        assert rss >= 0.0

    def test_perfect_fit_gives_zero_rss(self):
        mode_occ = np.array([1.0, 2.0, 3.0])
        b = 2.0 * mode_occ
        rss, _ = _fit_single_rule(b, mode_occ)
        assert rss == pytest.approx(0.0, abs=1e-10)

    def test_rss_nonnegative(self):
        rng = np.random.default_rng(99)
        b = rng.random(50)
        mode_occ = rng.random(50)
        rss, _ = _fit_single_rule(b, mode_occ)
        assert rss >= 0.0


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
