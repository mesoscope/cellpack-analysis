"""Tests for central-tendency logging functions in cellpack_analysis.lib.distance."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellpack_analysis.lib.distance import (
    log_central_tendencies_for_distance_distributions,
    log_central_tendencies_for_ks,
    log_pairwise_emd_central_tendencies,
)

_LOGGER_NAME = "cellpack_analysis.lib.distance"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_emd_df() -> pd.DataFrame:
    """Minimal df_emd with two modes and one distance measure.

    mode_a vs mode_b: emd values [1.0, 2.0, 3.0] → mean=2, median=2
    mode_a vs mode_a: emd values [0.5, 0.5, 0.5] → mean=0.5, median=0.5
    """
    records = [
        # intra mode_a
        {"distance_measure": "nucleus", "packing_mode_1": "mode_a", "packing_mode_2": "mode_a", "emd": 0.5},
        {"distance_measure": "nucleus", "packing_mode_1": "mode_a", "packing_mode_2": "mode_a", "emd": 0.5},
        {"distance_measure": "nucleus", "packing_mode_1": "mode_a", "packing_mode_2": "mode_a", "emd": 0.5},
        # mode_a vs mode_b
        {"distance_measure": "nucleus", "packing_mode_1": "mode_a", "packing_mode_2": "mode_b", "emd": 1.0},
        {"distance_measure": "nucleus", "packing_mode_1": "mode_a", "packing_mode_2": "mode_b", "emd": 2.0},
        {"distance_measure": "nucleus", "packing_mode_1": "mode_a", "packing_mode_2": "mode_b", "emd": 3.0},
        # intra mode_b
        {"distance_measure": "nucleus", "packing_mode_1": "mode_b", "packing_mode_2": "mode_b", "emd": 4.0},
        {"distance_measure": "nucleus", "packing_mode_1": "mode_b", "packing_mode_2": "mode_b", "emd": 4.0},
        {"distance_measure": "nucleus", "packing_mode_1": "mode_b", "packing_mode_2": "mode_b", "emd": 4.0},
    ]
    return pd.DataFrame(records)


@pytest.fixture()
def simple_ks_df() -> pd.DataFrame:
    """Minimal df_ks_bootstrap with one distance measure and two modes."""
    records = []
    for i in range(10):
        records.append({
            "distance_measure": "nucleus",
            "packing_mode": "mode_a",
            "experiment_number": i,
            "similar_fraction": 0.8,
        })
        records.append({
            "distance_measure": "nucleus",
            "packing_mode": "mode_b",
            "experiment_number": i,
            "similar_fraction": 0.4,
        })
    return pd.DataFrame(records)


@pytest.fixture()
def simple_distance_dict() -> dict:
    """Minimal all_distance_dict with one measure and two modes."""
    rng = np.random.default_rng(0)
    return {
        "nucleus": {
            "mode_a": {
                "cell_1": {"seed_0": rng.uniform(0.0, 1.0, size=50)},
            },
            "mode_b": {
                "cell_1": {"seed_0": rng.uniform(2.0, 3.0, size=50)},
            },
        }
    }


# ---------------------------------------------------------------------------
# log_pairwise_emd_central_tendencies
# ---------------------------------------------------------------------------


class TestLogPairwiseEmdCentralTendencies:
    def test_logs_header(self, simple_emd_df, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_pairwise_emd_central_tendencies(
                df_emd=simple_emd_df,
                distance_measures=["nucleus"],
                packing_modes=["mode_a", "mode_b"],
            )
        assert "Pairwise EMD central tendencies" in caplog.text

    def test_logs_correct_stats_for_pair(self, simple_emd_df, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_pairwise_emd_central_tendencies(
                df_emd=simple_emd_df,
                distance_measures=["nucleus"],
                packing_modes=["mode_a", "mode_b"],
            )
        # mode_a vs mode_b [nucleus]: mean=2.00, median=2.00
        assert "mode_a vs mode_b [nucleus]: 2.00" in caplog.text
        assert "median: 2.00" in caplog.text

    def test_logs_pooled_line(self, simple_emd_df, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_pairwise_emd_central_tendencies(
                df_emd=simple_emd_df,
                distance_measures=["nucleus"],
                packing_modes=["mode_a", "mode_b"],
            )
        assert "[pooled]" in caplog.text

    def test_skips_empty_pairs_without_crash(self, simple_emd_df, caplog):
        """A mode with no matching rows should not appear in the log and must not raise."""
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_pairwise_emd_central_tendencies(
                df_emd=simple_emd_df,
                distance_measures=["nucleus"],
                packing_modes=["mode_a", "mode_b", "mode_c"],
            )
        # mode_c has no data — the function should silently skip it
        assert "mode_c vs mode_c [nucleus]" not in caplog.text

    def test_writes_to_file(self, simple_emd_df, tmp_path):
        log_path = tmp_path / "emd_pairwise.log"
        log_pairwise_emd_central_tendencies(
            df_emd=simple_emd_df,
            distance_measures=["nucleus"],
            packing_modes=["mode_a", "mode_b"],
            log_file_path=log_path,
        )
        assert log_path.exists()
        content = log_path.read_text()
        assert "mode_a vs mode_b [nucleus]" in content
        assert "[pooled]" in content

    def test_intra_mode_stats_correct(self, simple_emd_df, caplog):
        """Intra-mode (mode_a vs mode_a) should log mean=0.50."""
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_pairwise_emd_central_tendencies(
                df_emd=simple_emd_df,
                distance_measures=["nucleus"],
                packing_modes=["mode_a"],
            )
        assert "mode_a vs mode_a [nucleus]: 0.50" in caplog.text

    def test_multiple_distance_measures(self, caplog):
        """All requested distance measures appear in the output."""
        records = [
            {"distance_measure": dm, "packing_mode_1": "mode_a", "packing_mode_2": "mode_b", "emd": 1.0}
            for dm in ["nucleus", "z", "membrane"]
        ]
        df = pd.DataFrame(records)
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_pairwise_emd_central_tendencies(
                df_emd=df,
                distance_measures=["nucleus", "z", "membrane"],
                packing_modes=["mode_a", "mode_b"],
            )
        for dm in ["nucleus", "z", "membrane"]:
            assert f"[{dm}]" in caplog.text


# ---------------------------------------------------------------------------
# log_central_tendencies_for_ks
# ---------------------------------------------------------------------------


class TestLogCentralTendenciesForKs:
    def test_logs_correct_stats(self, simple_ks_df, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_central_tendencies_for_ks(
                df_ks_bootstrap=simple_ks_df,
                distance_measures=["nucleus"],
            )
        # mode_a: mean=0.80, mode_b: mean=0.40
        assert "mode_a: 0.80" in caplog.text
        assert "mode_b: 0.40" in caplog.text

    def test_logs_each_distance_measure_label(self, caplog):
        records = [
            {"distance_measure": dm, "packing_mode": "mode_a", "experiment_number": i, "similar_fraction": 0.5}
            for dm in ["nucleus", "z"]
            for i in range(5)
        ]
        df = pd.DataFrame(records)
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_central_tendencies_for_ks(
                df_ks_bootstrap=df,
                distance_measures=["nucleus", "z"],
            )
        assert "Distance measure: nucleus" in caplog.text
        assert "Distance measure: z" in caplog.text

    def test_stats_written_to_file(self, simple_ks_df, tmp_path):
        """Verify bug fix: per-mode stats must appear in the file, not just on console."""
        log_path = tmp_path / "ks.log"
        log_central_tendencies_for_ks(
            df_ks_bootstrap=simple_ks_df,
            distance_measures=["nucleus"],
            file_path=log_path,
        )
        assert log_path.exists()
        content = log_path.read_text()
        assert "mode_a: 0.80" in content
        assert "mode_b: 0.40" in content

    def test_median_correct(self, simple_ks_df, caplog):
        """All similar_fraction values are constant → median equals mean."""
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_central_tendencies_for_ks(
                df_ks_bootstrap=simple_ks_df,
                distance_measures=["nucleus"],
            )
        assert "median: 0.80" in caplog.text
        assert "median: 0.40" in caplog.text


# ---------------------------------------------------------------------------
# log_central_tendencies_for_distance_distributions
# ---------------------------------------------------------------------------


class TestLogCentralTendenciesForDistanceDistributions:
    def test_logs_correct_stats(self, simple_distance_dict, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_central_tendencies_for_distance_distributions(
                all_distance_dict=simple_distance_dict,
                distance_measures=["nucleus"],
                packing_modes=["mode_a", "mode_b"],
                minimum_distance=None,
            )
        # mode_a distances are in [0, 1] → mean ~0.5; mode_b in [2, 3] → mean ~2.5
        assert "mode_a:" in caplog.text
        assert "mode_b:" in caplog.text

    def test_logs_correct_distance_measure_label(self, simple_distance_dict, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_central_tendencies_for_distance_distributions(
                all_distance_dict=simple_distance_dict,
                distance_measures=["nucleus"],
                packing_modes=["mode_a"],
                minimum_distance=None,
            )
        assert "Distance measure: nucleus" in caplog.text

    def test_logs_warning_for_empty_distances(self, caplog):
        """All-NaN distances should trigger a warning and not raise."""
        nan_dict = {
            "nucleus": {
                "mode_a": {
                    "cell_1": {"seed_0": np.array([np.nan, np.nan])},
                }
            }
        }
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            log_central_tendencies_for_distance_distributions(
                all_distance_dict=nan_dict,
                distance_measures=["nucleus"],
                packing_modes=["mode_a"],
                minimum_distance=None,
            )
        assert "No valid distances" in caplog.text

    def test_writes_to_file(self, simple_distance_dict, tmp_path):
        log_path = tmp_path / "dist.log"
        log_central_tendencies_for_distance_distributions(
            all_distance_dict=simple_distance_dict,
            distance_measures=["nucleus"],
            packing_modes=["mode_a", "mode_b"],
            file_path=log_path,
            minimum_distance=None,
        )
        assert log_path.exists()
        content = log_path.read_text()
        assert "mode_a:" in content
        assert "mode_b:" in content

    def test_mean_value_is_accurate(self, caplog):
        """Use a deterministic array so we can check the exact logged mean."""
        exact_dict = {
            "nucleus": {
                "mode_a": {
                    "cell_1": {"seed_0": np.array([1.0, 2.0, 3.0])},
                }
            }
        }
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            log_central_tendencies_for_distance_distributions(
                all_distance_dict=exact_dict,
                distance_measures=["nucleus"],
                packing_modes=["mode_a"],
                minimum_distance=None,
            )
        # mean=2.00, median=2.00
        assert "2.00" in caplog.text
