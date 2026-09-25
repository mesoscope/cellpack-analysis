"""Tests for the single distance floor, ``minimum_distance_um``.

Occupancy is ``pdf(packed) / pdf(available)``. Both sides must be floored at the same
depth in the same units, or the ratio is biased near the nucleus surface.
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

FLOOR_UM = -0.28


def _kde_after_workflow_prep(minimum_distance):
    """Floor, normalize and KDE the fixture exactly as ``AnalysisRunner`` does."""
    all_distance_dict = {"nucleus": {MODE: {CELL_ID: {"0": PACKED_PIXELS.copy()}}}}
    mesh_information_dict = {
        STRUCTURE_ID: {CELL_ID: {"nuc_grid_distances": AVAILABLE_PIXELS.copy()}}
    }
    all_distance_dict = filter_invalids_from_distance_distribution_dict(
        all_distance_dict, minimum_distance=minimum_distance
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
        minimum_distance=minimum_distance,
    )
    return kde_dict[CELL_ID]


def test_numerator_and_denominator_share_the_floor():
    """The same micron floor must trim packed and available distances alike."""
    cell = _kde_after_workflow_prep(FLOOR_UM / PIXEL_SIZE_IN_UM)

    packed_um = PACKED_PIXELS[~np.isnan(PACKED_PIXELS)] * PIXEL_SIZE_IN_UM
    available_um = AVAILABLE_PIXELS * PIXEL_SIZE_IN_UM
    expected_packed = packed_um[packed_um >= FLOOR_UM]
    expected_available = available_um[available_um >= FLOOR_UM]
    assert len(expected_packed) == 4 and len(expected_available) == 8, "fixture drift"

    np.testing.assert_allclose(np.sort(cell[MODE].dataset.ravel()), expected_packed, atol=1e-6)
    np.testing.assert_allclose(
        np.sort(cell["available"].dataset.ravel()), expected_available, atol=1e-6
    )


def test_none_keeps_every_finite_distance():
    """``None`` drops only NaN/inf on both sides."""
    cell = _kde_after_workflow_prep(None)

    assert cell[MODE].dataset.size == np.isfinite(PACKED_PIXELS).sum()
    assert cell["available"].dataset.size == AVAILABLE_PIXELS.size


def _write_config(tmp_path, **extra):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"name": "floor_test", "analysis_steps": [], **extra}))
    return path


@pytest.mark.parametrize("legacy_key", ["filter_minimum_distance", "available_minimum_distance"])
def test_legacy_keys_are_rejected(tmp_path, legacy_key):
    """A retired key must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="minimum_distance_um"):
        raw.AnalysisConfig(str(_write_config(tmp_path, **{legacy_key: -1})))


@pytest.mark.parametrize(
    ("extra", "expected_um"),
    [({}, raw.defaults.MINIMUM_DISTANCE_UM), ({"minimum_distance_um": -0.28}, -0.28)],
)
def test_config_converts_microns_to_pixels(tmp_path, monkeypatch, extra, expected_um):
    """The config value is in microns; the raw-distance floor is that value in pixels."""
    monkeypatch.setattr(raw, "get_project_root", lambda: tmp_path)
    config = raw.AnalysisConfig(str(_write_config(tmp_path, **extra)))

    assert config.minimum_distance_um == expected_um
    assert config.minimum_distance == pytest.approx(expected_um / PIXEL_SIZE_IN_UM)


def test_none_disables_the_floor(tmp_path, monkeypatch):
    """An explicit ``null`` turns the floor off rather than falling back to the default."""
    monkeypatch.setattr(raw, "get_project_root", lambda: tmp_path)
    config = raw.AnalysisConfig(str(_write_config(tmp_path, minimum_distance_um=None)))

    assert config.minimum_distance_um is None
    assert config.minimum_distance is None
