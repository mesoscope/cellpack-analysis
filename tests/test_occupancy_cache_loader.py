"""Tests for the strict occupancy cache loader ``load_occupancy_cache``."""

import pickle

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from cellpack_analysis.lib.occupancy import (
    _occupancy_cache_meta_path,
    get_kde_occupancy_dict,
    load_occupancy_cache,
)

NUM_POINTS = 40
FLOORS: dict[str, float | None] = {
    "minimum_distance_um": -0.108,
    "available_minimum_distance_um": -1.0,
}


@pytest.fixture()
def cache_dir(tmp_path):
    """Build a real KDE occupancy cache (pickle + sidecar) in a temp dir."""
    rng = np.random.default_rng(5)
    kde_dict = {
        f"cell_{i}": {
            "real": gaussian_kde(rng.exponential(2.0, size=200)),
            "random": gaussian_kde(rng.exponential(2.5, size=200)),
            "available": gaussian_kde(rng.exponential(3.0, size=600)),
        }
        for i in range(3)
    }
    built = get_kde_occupancy_dict(
        distance_kde_dict=kde_dict,
        channel_map={"real": "S1", "random": "S1"},
        results_dir=tmp_path,
        suffix="_x",
        distance_measure="nucleus",
        bandwidth=0.2,
        num_points=NUM_POINTS,
        x_min=0,
        x_max=8.0,
        floors_um=FLOORS,
    )
    return tmp_path, built


def _expected() -> dict:
    return {
        "distance_measure": "nucleus",
        "num_points": NUM_POINTS,
        "x_min": 0,
        "x_max": 8.0,
        "bandwidth": 0.2,
        **FLOORS,
    }


class TestLoadOccupancyCache:
    def test_round_trips_what_the_builder_wrote(self, cache_dir):
        results_dir, built = cache_dir
        loaded = load_occupancy_cache(
            results_dir,
            "nucleus",
            suffix="_x",
            expected_meta=_expected(),
            channel_map={"real": "S1", "random": "S1"},
        )
        assert set(loaded) == set(built)
        np.testing.assert_array_equal(
            loaded["real"]["combined"]["all_occupancy"],
            built["real"]["combined"]["all_occupancy"],
        )

    def test_no_expected_meta_only_requires_the_sidecar(self, cache_dir):
        results_dir, _ = cache_dir
        assert set(load_occupancy_cache(results_dir, "nucleus", suffix="_x")) == {"real", "random"}

    def test_missing_pickle_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"nucleus_occupancy\.dat"):
            load_occupancy_cache(tmp_path, "nucleus")

    def test_wrong_suffix_raises(self, cache_dir):
        results_dir, _ = cache_dir
        with pytest.raises(FileNotFoundError):
            load_occupancy_cache(results_dir, "nucleus", suffix="")

    def test_missing_sidecar_raises_even_without_expected_meta(self, cache_dir):
        results_dir, _ = cache_dir
        _occupancy_cache_meta_path(results_dir / "nucleus_occupancy_x.dat").unlink()
        with pytest.raises(FileNotFoundError, match="sidecar"):
            load_occupancy_cache(results_dir, "nucleus", suffix="_x")

    def test_corrupt_sidecar_raises(self, cache_dir):
        results_dir, _ = cache_dir
        _occupancy_cache_meta_path(results_dir / "nucleus_occupancy_x.dat").write_text("{nope")
        with pytest.raises(ValueError, match="Could not read"):
            load_occupancy_cache(results_dir, "nucleus", suffix="_x")

    @pytest.mark.parametrize(
        "change",
        [
            {"num_points": 250},
            {"x_max": 6},
            {"bandwidth": 0.5},
            {"minimum_distance_um": -0.5},
            {"available_minimum_distance_um": -0.5},
        ],
    )
    def test_mismatched_meta_raises_and_names_the_key(self, cache_dir, change):
        results_dir, _ = cache_dir
        with pytest.raises(ValueError, match=next(iter(change))):
            load_occupancy_cache(
                results_dir, "nucleus", suffix="_x", expected_meta={**_expected(), **change}
            )

    def test_expected_key_absent_from_sidecar_raises(self, cache_dir):
        results_dir, _ = cache_dir
        with pytest.raises(ValueError, match="not_stamped"):
            load_occupancy_cache(
                results_dir, "nucleus", suffix="_x", expected_meta={"not_stamped": 1}
            )

    def test_mode_keys_checked_only_when_channel_map_given(self, cache_dir):
        results_dir, _ = cache_dir
        wrong = {"real": "S1", "apical": "S1"}
        with pytest.raises(ValueError, match="modes"):
            load_occupancy_cache(results_dir, "nucleus", suffix="_x", channel_map=wrong)
        assert load_occupancy_cache(results_dir, "nucleus", suffix="_x") is not None

    def test_cache_without_unified_keys_raises(self, tmp_path):
        save_path = tmp_path / "nucleus_occupancy.dat"
        with open(save_path, "wb") as f:
            pickle.dump({"real": {"individual": {}, "combined": {}}}, f)
        _occupancy_cache_meta_path(save_path).write_text("{}")
        with pytest.raises(ValueError, match="unified format"):
            load_occupancy_cache(tmp_path, "nucleus")
