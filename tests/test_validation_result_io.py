"""Tests for ``load_validation_result`` and the ``ks_bootstrap_seed`` hook of the validation."""

import numpy as np
import pandas as pd
import pytest

from cellpack_analysis.lib import distance as distance_mod
from cellpack_analysis.lib import rule_interpolation as ri

OCC_EMD_FILE = "occupancy_emd_mixed_rule_validation.parquet"
DIST_EMD_FILE = "distance_pairwise_emd_mixed_rule_validation.parquet"


def _emd_frame(offset: float = 0.0) -> pd.DataFrame:
    """Cross-mode EMD rows, two of which pair different cells."""
    return pd.DataFrame(
        {
            "distance_measure": ["nucleus"] * 4,
            "packing_mode_1": ["real", "real", "random", "real"],
            "packing_mode_2": ["random", "random", "random", "interpolated"],
            "cell_id_1": ["c0", "c0", "c0", "c1"],
            "cell_id_2": ["c0", "c1", "c1", "c1"],
            "emd": np.array([0.1, 0.2, 0.3, 0.4]) + offset,
        }
    )


def _run_validation(results_dir, **kwargs):
    """Run the validation on precomputed EMD frames, so no EMD cache is written."""
    return ri.run_mixed_rule_validation(
        combined_occupancy_dict={},
        channel_map={"real": "S", "random": "S", "interpolated": "S"},
        baseline_mode="real",
        packing_modes=["random", "interpolated"],
        distance_measures=["nucleus"],
        results_dir=results_dir,
        occupancy_emd_df=_emd_frame(),
        distance_emd_df=_emd_frame(offset=1.0),
        **kwargs,
    )


@pytest.fixture()
def saved_dir(tmp_path):
    """A results dir as ``validate`` leaves it: slim pickle plus both unfiltered parquets."""
    _run_validation(tmp_path)
    _emd_frame().to_parquet(tmp_path / OCC_EMD_FILE, index=False)
    _emd_frame(offset=1.0).to_parquet(tmp_path / DIST_EMD_FILE, index=False)
    return tmp_path


class TestLoadValidationResult:
    def test_restores_both_emd_frames(self, saved_dir):
        result = ri.load_validation_result(saved_dir, same_cell_emd=False)
        pd.testing.assert_frame_equal(result.emd_df, _emd_frame())
        assert result.distance_emd_df is not None
        pd.testing.assert_frame_equal(result.distance_emd_df, _emd_frame(offset=1.0))

    def test_same_cell_flag_matches_the_in_memory_result(self, saved_dir, tmp_path_factory):
        in_memory = _run_validation(tmp_path_factory.mktemp("mem"), same_cell_emd=True)
        loaded = ri.load_validation_result(saved_dir, same_cell_emd=True)
        assert len(loaded.emd_df) == 3  # the c0-vs-c1 cross-mode row is dropped
        pd.testing.assert_frame_equal(loaded.emd_df, in_memory.emd_df)
        assert in_memory.distance_emd_df is not None and loaded.distance_emd_df is not None
        pd.testing.assert_frame_equal(loaded.distance_emd_df, in_memory.distance_emd_df)

    def test_other_fields_survive_the_pickle(self, tmp_path):
        ks_df = pd.DataFrame({"cell_id": ["c0", "c1"], "ks_stat": [0.1, 0.2]})
        slim_source = ri.ValidationResult(
            emd_df=_emd_frame(),
            distance_emd_df=_emd_frame(offset=1.0),
            distance_ks_df=ks_df,
            distance_envelope_test={"alpha": 0.05},
        )
        ri._save_validation_result(slim_source, tmp_path)
        _emd_frame().to_parquet(tmp_path / OCC_EMD_FILE, index=False)
        _emd_frame(offset=1.0).to_parquet(tmp_path / DIST_EMD_FILE, index=False)
        loaded = ri.load_validation_result(tmp_path, same_cell_emd=False)
        pd.testing.assert_frame_equal(loaded.distance_ks_df, ks_df)
        assert loaded.distance_envelope_test == {"alpha": 0.05}

    def test_occupancy_only_result_loads_with_no_distance_frame(self, tmp_path):
        ri._save_validation_result(ri.ValidationResult(emd_df=_emd_frame()), tmp_path)
        _emd_frame().to_parquet(tmp_path / OCC_EMD_FILE, index=False)
        result = ri.load_validation_result(tmp_path, same_cell_emd=True)
        assert result.distance_emd_df is None
        assert len(result.emd_df) == 3

    def test_missing_pickle_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="validation_result"):
            ri.load_validation_result(tmp_path, same_cell_emd=False)

    def test_missing_occupancy_parquet_raises(self, saved_dir):
        (saved_dir / OCC_EMD_FILE).unlink()
        with pytest.raises(FileNotFoundError, match="occupancy_emd"):
            ri.load_validation_result(saved_dir, same_cell_emd=False)

    def test_missing_distance_parquet_raises_when_distance_tests_exist(self, tmp_path):
        source = ri.ValidationResult(
            emd_df=_emd_frame(),
            distance_ks_df=pd.DataFrame({"cell_id": ["c0"]}),
        )
        ri._save_validation_result(source, tmp_path)
        _emd_frame().to_parquet(tmp_path / OCC_EMD_FILE, index=False)
        with pytest.raises(FileNotFoundError, match="distance_pairwise_emd"):
            ri.load_validation_result(tmp_path, same_cell_emd=False)


def _distance_dict(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "nucleus": {
            m: {f"c{i}": {str(s): rng.uniform(0, 2, 40) for s in range(5)} for i in range(6)}
            for m in ("real", "random", "interpolated")
        }
    }


class TestKsBootstrapSeed:
    def test_seed_reaches_bootstrap_ks_tests(self, monkeypatch):
        seen: dict = {}

        def _spy(**kwargs):
            seen.update(kwargs)
            return pd.DataFrame()

        monkeypatch.setattr(distance_mod, "bootstrap_ks_tests", _spy)
        _run_validation(None, mixed_rule_distance_dict=_distance_dict(), ks_bootstrap_seed=11)
        assert seen["seed"] == 11

    def test_default_is_unseeded(self, monkeypatch):
        seen: dict = {}

        def _spy(**kwargs):
            seen.update(kwargs)
            return pd.DataFrame()

        monkeypatch.setattr(distance_mod, "bootstrap_ks_tests", _spy)
        _run_validation(None, mixed_rule_distance_dict=_distance_dict())
        assert seen["seed"] is None
