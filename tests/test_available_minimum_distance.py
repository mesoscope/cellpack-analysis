"""Tests for the denominator-only distance filter in ``get_distance_distribution_kde``.

Occupancy is ``pdf(packed) / pdf(available)``.  A single ``minimum_distance`` filtered
both sides; ``available_minimum_distance`` decouples them so a standoff margin can be
applied to the denominator alone.
"""

import numpy as np
import pytest

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import get_distance_distribution_kde

CELL_ID = "12345"
MODE = "random"
STRUCTURE_ID = "SLC25A17"
CHANNEL_MAP = {MODE: STRUCTURE_ID}

# Grid distances are stored in pixels; the KDE normalizes to microns, so these
# span -0.540 to +0.540 um.
AVAILABLE_PIXELS = np.arange(-5.0, 6.0)
AVAILABLE_MICRONS = AVAILABLE_PIXELS * PIXEL_SIZE_IN_UM

# Packed distances reach this function already normalized, i.e. in microns.
PACKED_MICRONS = np.array([-0.4, -0.2, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5])

# Peroxisome-scale standoff, more negative than the numerator's -1 px = -0.108 um.
MARGIN_UM = -0.28


@pytest.fixture
def kde_inputs():
    """A one-cell, one-mode distance dict and the matching mesh information dict."""
    all_distance_dict = {"nucleus": {MODE: {CELL_ID: {"0": PACKED_MICRONS.copy()}}}}
    mesh_information_dict = {
        STRUCTURE_ID: {CELL_ID: {"nuc_grid_distances": AVAILABLE_PIXELS.copy()}}
    }
    return all_distance_dict, mesh_information_dict


def _run(all_distance_dict, mesh_information_dict, **kwargs):
    """Call the KDE builder with caching disabled, returning the one cell's entry."""
    kde_dict = get_distance_distribution_kde(
        all_distance_dict=all_distance_dict,
        mesh_information_dict=mesh_information_dict,
        channel_map=CHANNEL_MAP,
        save_dir=None,
        distance_measure="nucleus",
        **kwargs,
    )
    return kde_dict[CELL_ID]


def test_default_is_bit_identical_to_the_coupled_filter(kde_inputs):
    """Omitting the key must reproduce the coupled behaviour exactly.

    This is the fidelity gate for the refactor: every shipped result was produced
    without the key, so ``None`` has to mean "inherit ``minimum_distance``".
    """
    inherited = _run(*kde_inputs, minimum_distance=-1)
    explicit = _run(*kde_inputs, minimum_distance=-1, available_minimum_distance=-1)

    np.testing.assert_array_equal(inherited["available"].dataset, explicit["available"].dataset)
    np.testing.assert_array_equal(inherited[MODE].dataset, explicit[MODE].dataset)


def test_margin_trims_the_denominator(kde_inputs):
    """The margin must drop available points closer to the surface than the standoff."""
    baseline = _run(*kde_inputs, minimum_distance=-1)
    with_margin = _run(*kde_inputs, minimum_distance=-1, available_minimum_distance=MARGIN_UM)

    # At -1 um nothing is trimmed: the whole grid is above the floor.
    np.testing.assert_allclose(
        np.sort(baseline["available"].dataset.ravel()), AVAILABLE_MICRONS, atol=1e-6
    )

    expected = AVAILABLE_MICRONS[AVAILABLE_MICRONS >= MARGIN_UM]
    assert len(expected) == 8, "fixture drift: the margin should remove exactly 3 of 11 points"
    np.testing.assert_allclose(
        np.sort(with_margin["available"].dataset.ravel()), expected, atol=1e-6
    )


def test_margin_leaves_the_numerator_untouched(kde_inputs):
    """Only the availability denominator may move; the packed distances must not.

    The approved change is to the denominator. If the key reached the numerator's
    filter as well it would alter observed puncta counts, which is a different --
    and unapproved -- analysis.
    """
    baseline = _run(*kde_inputs, minimum_distance=-1)
    with_margin = _run(*kde_inputs, minimum_distance=-1, available_minimum_distance=MARGIN_UM)

    np.testing.assert_array_equal(baseline[MODE].dataset, with_margin[MODE].dataset)
    np.testing.assert_allclose(
        np.sort(with_margin[MODE].dataset.ravel()), PACKED_MICRONS, atol=1e-6
    )
