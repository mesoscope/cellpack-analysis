"""Tests for the distance floors, ``minimum_distance_um`` and ``available_minimum_distance_um``.

Occupancy is ``pdf(packed) / pdf(available)``. As in round 1, puncta and available-space
distances each have their own floor, both in microns.
"""

import json

import numpy as np
import pytest

from cellpack_analysis.analysis.workflows import run_analysis_workflow as raw
from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import (
    filter_invalids_from_distance_distribution_dict,
    get_distance_distribution_kde,
    normalize_distance_dictionary,
)

CELL_ID = "12345"
MODE = "random"
STRUCTURE_ID = "SLC25A17"
CHANNEL_MAP = {MODE: STRUCTURE_ID}

# Both packed and grid distances are stored in pixels.
AVAILABLE_PIXELS = np.arange(-5.0, 6.0)
PACKED_PIXELS = np.array([-4.0, -3.0, -2.0, 0.0, 3.0, 6.0, np.nan])

PUNCTA_FLOOR_UM = -0.28
AVAILABLE_FLOOR_UM = -0.5

FLOOR_KEYS = [
    ("minimum_distance_um", "minimum_distance"),
    ("available_minimum_distance_um", "available_minimum_distance"),
]


def _kde_after_workflow_prep(puncta_floor, available_floor):
    """Floor, normalize and KDE the fixture exactly as ``AnalysisRunner`` does."""
    all_distance_dict = {"nucleus": {MODE: {CELL_ID: {"0": PACKED_PIXELS.copy()}}}}
    mesh_information_dict = {
        STRUCTURE_ID: {CELL_ID: {"nuc_grid_distances": AVAILABLE_PIXELS.copy()}}
    }
    all_distance_dict = filter_invalids_from_distance_distribution_dict(
        all_distance_dict, minimum_distance=puncta_floor
    )
    all_distance_dict = normalize_distance_dictionary(
        all_distance_dict, mesh_information_dict, CHANNEL_MAP
    )
    kde_dict = get_distance_distribution_kde(
        all_distance_dict=all_distance_dict,
        mesh_information_dict=mesh_information_dict,
        channel_map=CHANNEL_MAP,
        save_dir=None,
        distance_measure="nucleus",
        minimum_distance=available_floor,
    )
    return kde_dict[CELL_ID]


def test_each_floor_trims_only_its_own_side():
    """The puncta floor trims packed distances; the available floor trims the grid."""
    cell = _kde_after_workflow_prep(
        PUNCTA_FLOOR_UM / PIXEL_SIZE_IN_UM, AVAILABLE_FLOOR_UM / PIXEL_SIZE_IN_UM
    )

    packed_um = PACKED_PIXELS[~np.isnan(PACKED_PIXELS)] * PIXEL_SIZE_IN_UM
    available_um = AVAILABLE_PIXELS * PIXEL_SIZE_IN_UM
    expected_packed = packed_um[packed_um >= PUNCTA_FLOOR_UM]
    expected_available = available_um[available_um >= AVAILABLE_FLOOR_UM]
    assert len(expected_packed) == 4 and len(expected_available) == 10, "fixture drift"

    np.testing.assert_allclose(np.sort(cell[MODE].dataset.ravel()), expected_packed, atol=1e-6)
    np.testing.assert_allclose(
        np.sort(cell["available"].dataset.ravel()), expected_available, atol=1e-6
    )


def test_none_keeps_every_finite_distance():
    """``None`` drops only NaN/inf on both sides."""
    cell = _kde_after_workflow_prep(None, None)

    assert cell[MODE].dataset.size == np.isfinite(PACKED_PIXELS).sum()
    assert cell["available"].dataset.size == AVAILABLE_PIXELS.size


def _write_config(tmp_path, **extra):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"name": "floor_test", "analysis_steps": [], **extra}))
    return path


def test_defaults_are_the_round_1_floors():
    assert raw.defaults.MINIMUM_DISTANCE_UM == -0.108
    assert raw.defaults.AVAILABLE_MINIMUM_DISTANCE_UM == -1.0


@pytest.mark.parametrize("legacy_key", ["filter_minimum_distance", "available_minimum_distance"])
def test_legacy_keys_are_rejected(tmp_path, legacy_key):
    """A retired key must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="minimum_distance_um"):
        raw.AnalysisConfig(str(_write_config(tmp_path, **{legacy_key: -1})))


@pytest.mark.parametrize(("key", "px_attr"), FLOOR_KEYS)
def test_config_converts_microns_to_pixels(tmp_path, monkeypatch, key, px_attr):
    """Each floor is read in microns and applied to raw distances in pixels."""
    monkeypatch.setattr(raw, "get_project_root", lambda: tmp_path)
    default = raw.AnalysisConfig(str(_write_config(tmp_path)))
    config = raw.AnalysisConfig(str(_write_config(tmp_path, **{key: -0.28})))

    assert getattr(config, key) == -0.28
    assert getattr(config, px_attr) == pytest.approx(-0.28 / PIXEL_SIZE_IN_UM)
    assert getattr(default, px_attr) == pytest.approx(getattr(default, key) / PIXEL_SIZE_IN_UM)
    other = next(k for k, _ in FLOOR_KEYS if k != key)
    assert getattr(config, other) == getattr(default, other)


@pytest.mark.parametrize(("key", "px_attr"), FLOOR_KEYS)
def test_none_disables_the_floor(tmp_path, monkeypatch, key, px_attr):
    """An explicit ``null`` turns a floor off rather than falling back to the default."""
    monkeypatch.setattr(raw, "get_project_root", lambda: tmp_path)
    config = raw.AnalysisConfig(str(_write_config(tmp_path, **{key: None})))

    assert getattr(config, key) is None
    assert getattr(config, px_attr) is None
