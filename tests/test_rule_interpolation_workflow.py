"""Tests for the rule-interpolation end-to-end workflow helpers.

Focus: the fit and validate phases must agree on one set of occupancy curves.
The coefficients are fitted in the fit phase; scoring them against the validate
phase's independently rebuilt curves silently evaluates a fit against data it was
never fitted to.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cellpack_analysis.analysis.workflows.run_rule_interpolation_workflow import (
    _REQUIRED_FIT_ANALYSIS_STEPS,
    _build_fit_config_dict,
    _fit_occupancy_cache_path,
    _merge_fit_and_validation_occupancy,
)


class TestBuildFitConfigDict:
    """The fit phase must rebuild both the occupancy cache and the CV pickle."""

    def test_forces_recalculate_for_all_required_steps(self) -> None:
        cfg = _build_fit_config_dict({"name": "x"})
        for step in _REQUIRED_FIT_ANALYSIS_STEPS:
            assert cfg["recalculate"][step] is True

    def test_forces_recalculate_over_an_explicit_false(self) -> None:
        cfg = _build_fit_config_dict(
            {"name": "x", "recalculate": {"run_occupancy_analysis": False}}
        )
        assert cfg["recalculate"]["run_occupancy_analysis"] is True
        assert cfg["recalculate"]["run_rule_interpolation_cv"] is True

    def test_forces_recalculate_over_a_bool_false(self) -> None:
        cfg = _build_fit_config_dict({"name": "x", "recalculate": False})
        for step in _REQUIRED_FIT_ANALYSIS_STEPS:
            assert cfg["recalculate"][step] is True

    def test_preserves_unrelated_recalculate_keys(self) -> None:
        cfg = _build_fit_config_dict({"name": "x", "recalculate": {"calculate_distances": True}})
        assert cfg["recalculate"]["calculate_distances"] is True

    def test_does_not_mutate_the_input_config(self) -> None:
        raw: dict[str, Any] = {"name": "x", "recalculate": {"run_occupancy_analysis": False}}
        _build_fit_config_dict(raw)
        assert raw["recalculate"]["run_occupancy_analysis"] is False

    def test_required_steps_are_added(self) -> None:
        cfg = _build_fit_config_dict({"name": "x", "analysis_steps": ["load_common_data"]})
        for step in _REQUIRED_FIT_ANALYSIS_STEPS:
            assert step in cfg["analysis_steps"]


class TestMergeFitAndValidationOccupancy:
    """Component modes come from the fit cache; only the mixed rule is new."""

    XVALS = np.linspace(0, 6, 25)

    @classmethod
    def _mode_entry(cls, level: float, xvals: np.ndarray | None = None) -> dict[str, Any]:
        xv = cls.XVALS if xvals is None else xvals
        return {
            "individual": {"c0": {"xvals": xv, "occupancy": np.full(xv.size, level)}},
            "combined": {"xvals": xv, "occupancy": np.full(xv.size, level)},
        }

    @classmethod
    def _write_fit_cache(
        cls,
        results_dir: Path,
        dm: str = "nucleus",
        suffix: str = "",
        xvals: np.ndarray | None = None,
    ) -> Path:
        payload = {
            "real": cls._mode_entry(1.0, xvals),
            "membrane_gradient": cls._mode_entry(1.1644, xvals),
        }
        path = _fit_occupancy_cache_path(results_dir, dm, suffix)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        return path

    @classmethod
    def _validation_dict(
        cls, dm: str = "nucleus", xvals: np.ndarray | None = None
    ) -> dict[str, Any]:
        return {
            dm: {
                "real": cls._mode_entry(1.0, xvals),
                # Drifted: a second cache built later disagrees with the fit's.
                "membrane_gradient": cls._mode_entry(1.2244, xvals),
                "interpolated": cls._mode_entry(0.9, xvals),
            }
        }

    def test_component_modes_come_from_the_fit_cache(self, tmp_path: Path) -> None:
        self._write_fit_cache(tmp_path)
        merged = _merge_fit_and_validation_occupancy(
            results_dir=tmp_path,
            fit_suffix="",
            distance_measures=["nucleus"],
            validation_occupancy_dict=self._validation_dict(),
            baseline_mode="real",
        )
        curve = merged["nucleus"]["membrane_gradient"]["combined"]["occupancy"]
        np.testing.assert_allclose(curve, 1.1644)  # the fit's value, not 1.2244

    def test_interpolated_comes_from_the_validation_run(self, tmp_path: Path) -> None:
        self._write_fit_cache(tmp_path)
        merged = _merge_fit_and_validation_occupancy(
            results_dir=tmp_path,
            fit_suffix="",
            distance_measures=["nucleus"],
            validation_occupancy_dict=self._validation_dict(),
            baseline_mode="real",
        )
        assert "interpolated" in merged["nucleus"]
        np.testing.assert_allclose(merged["nucleus"]["interpolated"]["combined"]["occupancy"], 0.9)

    def test_missing_fit_cache_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Fit-phase occupancy cache not found"):
            _merge_fit_and_validation_occupancy(
                results_dir=tmp_path,
                fit_suffix="",
                distance_measures=["nucleus"],
                validation_occupancy_dict=self._validation_dict(),
                baseline_mode="real",
            )

    def test_grid_mismatch_raises(self, tmp_path: Path) -> None:
        self._write_fit_cache(tmp_path, xvals=np.linspace(0, 6, 25))
        with pytest.raises(ValueError, match="occupancy grids differ"):
            _merge_fit_and_validation_occupancy(
                results_dir=tmp_path,
                fit_suffix="",
                distance_measures=["nucleus"],
                validation_occupancy_dict=self._validation_dict(xvals=np.linspace(0, 5.17, 25)),
                baseline_mode="real",
            )

    def test_missing_validation_grid_raises(self, tmp_path: Path) -> None:
        """A malformed side must fail the grid check, not slip past it (F7).

        The old check was ``if fit_xvals is not None and val_xvals is not None``,
        which skipped exactly the case it exists to catch.
        """
        self._write_fit_cache(tmp_path)
        validation: dict[str, Any] = {"nucleus": {"interpolated": self._mode_entry(0.9)}}
        with pytest.raises(ValueError, match="Cannot verify the occupancy grid"):
            _merge_fit_and_validation_occupancy(
                results_dir=tmp_path,
                fit_suffix="",
                distance_measures=["nucleus"],
                validation_occupancy_dict=validation,
                baseline_mode="real",
            )

    def test_missing_fit_grid_raises(self, tmp_path: Path) -> None:
        """Same for a fit cache that has no baseline grid."""
        path = _fit_occupancy_cache_path(tmp_path, "nucleus", "")
        with open(path, "wb") as fh:
            pickle.dump({"membrane_gradient": self._mode_entry(1.1644)}, fh)
        with pytest.raises(ValueError, match="Cannot verify the occupancy grid"):
            _merge_fit_and_validation_occupancy(
                results_dir=tmp_path,
                fit_suffix="",
                distance_measures=["nucleus"],
                validation_occupancy_dict=self._validation_dict(),
                baseline_mode="real",
            )

    def test_missing_grid_error_names_the_offending_side(self, tmp_path: Path) -> None:
        self._write_fit_cache(tmp_path)
        validation: dict[str, Any] = {"nucleus": {"interpolated": self._mode_entry(0.9)}}
        with pytest.raises(ValueError, match="validation occupancy data"):
            _merge_fit_and_validation_occupancy(
                results_dir=tmp_path,
                fit_suffix="",
                distance_measures=["nucleus"],
                validation_occupancy_dict=validation,
                baseline_mode="real",
            )

    def test_suffix_is_used_for_the_cache_filename(self, tmp_path: Path) -> None:
        """The suffix must be the composed one, matching AnalysisConfig.suffix."""
        self._write_fit_cache(tmp_path, suffix="_norm_max")
        merged = _merge_fit_and_validation_occupancy(
            results_dir=tmp_path,
            fit_suffix="_norm_max",
            distance_measures=["nucleus"],
            validation_occupancy_dict=self._validation_dict(),
            baseline_mode="real",
        )
        assert "membrane_gradient" in merged["nucleus"]

    def test_all_requested_distance_measures_are_merged(self, tmp_path: Path) -> None:
        for dm in ("nucleus", "z"):
            self._write_fit_cache(tmp_path, dm=dm)
        validation = {**self._validation_dict("nucleus"), **self._validation_dict("z")}
        merged = _merge_fit_and_validation_occupancy(
            results_dir=tmp_path,
            fit_suffix="",
            distance_measures=["nucleus", "z"],
            validation_occupancy_dict=validation,
            baseline_mode="real",
        )
        assert set(merged) == {"nucleus", "z"}
