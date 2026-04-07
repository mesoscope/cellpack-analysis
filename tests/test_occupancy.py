"""Tests for the refactored histogram-based occupancy functions."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from cellpack_analysis.lib.occupancy import (
    _compute_single_cell_occupancy,
    get_binned_occupancy_dict_from_distance_dict,
    get_kde_occupancy_dict,
)


class TestComputeSingleCellOccupancy:
    """Unit tests for the per-cell occupancy helper."""

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        occupied = rng.exponential(2.0, size=500)
        available = rng.exponential(3.0, size=2000)
        result = _compute_single_cell_occupancy(occupied, available, bin_width=0.5)
        expected_keys = {
            "xvals",
            "occupancy",
            "pdf_occupied",
            "pdf_available",
            "raw_occupied",
            "raw_available",
        }
        assert set(result.keys()) == expected_keys

    def test_xvals_uniformly_spaced(self):
        rng = np.random.default_rng(0)
        available = rng.uniform(0, 10, size=1000)
        occupied = rng.uniform(0, 5, size=200)
        result = _compute_single_cell_occupancy(occupied, available, bin_width=0.5)
        diffs = np.diff(result["xvals"])
        # Grid is uniform but spacing may differ slightly from bin_width
        # due to make_r_grid_from_pooled's extend / percentile logic
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-12)

    def test_pdfs_integrate_to_one(self):
        rng = np.random.default_rng(7)
        occupied = rng.normal(3, 1, size=500)
        available = rng.normal(4, 2, size=2000)
        occupied = occupied[occupied > 0]
        available = available[available > 0]
        result = _compute_single_cell_occupancy(occupied, available, bin_width=0.3)
        integral_occ = np.trapezoid(result["pdf_occupied"], result["xvals"])
        integral_avail = np.trapezoid(result["pdf_available"], result["xvals"])
        np.testing.assert_allclose(integral_occ, 1.0, atol=0.05)
        np.testing.assert_allclose(integral_avail, 1.0, atol=0.05)

    def test_min_count_masks_sparse_bins(self):
        rng = np.random.default_rng(1)
        available = rng.uniform(0, 2, size=50)
        occupied = rng.uniform(0, 2, size=50)
        result = _compute_single_cell_occupancy(occupied, available, bin_width=0.2, min_count=10)
        # Some bins should be NaN where available count < 10
        assert np.any(np.isnan(result["occupancy"]))

    def test_no_masking_when_min_count_zero(self):
        rng = np.random.default_rng(2)
        available = rng.uniform(0, 2, size=100)
        occupied = rng.uniform(0, 2, size=100)
        result = _compute_single_cell_occupancy(occupied, available, bin_width=0.5, min_count=0)
        assert not np.any(np.isnan(result["occupancy"]))

    def test_occupancy_near_one_for_identical_distributions(self):
        rng = np.random.default_rng(99)
        data = rng.exponential(2.0, size=5000)
        result = _compute_single_cell_occupancy(
            data[:2500], data[2500:], bin_width=0.3, min_count=0
        )
        finite = result["occupancy"][np.isfinite(result["occupancy"])]
        # Ratio should hover around 1.0 when both are from the same distribution
        np.testing.assert_allclose(np.nanmean(finite), 1.0, atol=0.3)


class TestGetBinnedOccupancyDictFromDistanceDict:
    """Integration tests for the refactored main function."""

    @pytest.fixture()
    def mock_inputs(self):
        """Build minimal synthetic inputs matching the expected dict structure."""
        rng = np.random.default_rng(42)
        n_cells = 5
        cell_ids = [f"cell_{i}" for i in range(n_cells)]

        # Build all_distance_dict
        # {distance_measure: {mode: {cell_id: {seed: distances}}}}
        all_distance_dict: dict = {
            "nucleus": {
                "real": {},
            }
        }
        for cid in cell_ids:
            all_distance_dict["nucleus"]["real"][cid] = {
                "seed_0": rng.exponential(2.0, size=200),
            }

        # Build combined_mesh_information_dict
        # {structure_id: {cell_id: {"nuc_grid_distances": ...}}}
        combined_mesh = {"ST001": {}}
        for cid in cell_ids:
            # Grid distances in pixel units (will be multiplied by PIXEL_SIZE_IN_UM)
            combined_mesh["ST001"][cid] = {
                "nuc_grid_distances": rng.exponential(30.0, size=5000),
            }

        channel_map = {"real": "ST001"}
        return all_distance_dict, combined_mesh, channel_map, cell_ids

    def test_output_structure(self, mock_inputs):
        all_dd, mesh, cmap, cell_ids = mock_inputs
        result = get_binned_occupancy_dict_from_distance_dict(
            all_distance_dict=all_dd,
            combined_mesh_information_dict=mesh,
            channel_map=cmap,
            distance_measure="nucleus",
            bin_width=0.3,
            x_max=8.0,
            recalculate=True,
        )
        assert "real" in result
        assert "individual" in result["real"]
        assert "combined" in result["real"]
        combined = result["real"]["combined"]
        for key in [
            "xvals",
            "occupancy",
            "occupancy_pooled",
            "std_occupancy",
            "envelope_lo",
            "envelope_hi",
            "pdf_occupied",
            "pdf_available",
            "all_occupancy",
        ]:
            assert key in combined, f"Missing combined key: {key}"

    def test_individual_per_cell_grids(self, mock_inputs):
        all_dd, mesh, cmap, cell_ids = mock_inputs
        result = get_binned_occupancy_dict_from_distance_dict(
            all_distance_dict=all_dd,
            combined_mesh_information_dict=mesh,
            channel_map=cmap,
            distance_measure="nucleus",
            bin_width=0.3,
            recalculate=True,
        )
        individual = result["real"]["individual"]
        # Each cell should have its own xvals
        grids = [individual[cid]["xvals"] for cid in individual]
        # Grids may differ in length (per-cell basis)
        assert all(isinstance(g, np.ndarray) for g in grids)
        for cid, data in individual.items():
            assert len(data["xvals"]) == len(data["occupancy"])
            assert len(data["xvals"]) == len(data["pdf_occupied"])
            assert len(data["xvals"]) == len(data["pdf_available"])

    def test_combined_all_occupancy_shape(self, mock_inputs):
        all_dd, mesh, cmap, cell_ids = mock_inputs
        result = get_binned_occupancy_dict_from_distance_dict(
            all_distance_dict=all_dd,
            combined_mesh_information_dict=mesh,
            channel_map=cmap,
            distance_measure="nucleus",
            bin_width=0.3,
            x_max=8.0,
            recalculate=True,
        )
        combined = result["real"]["combined"]
        n_cells_processed = len(result["real"]["individual"])
        n_bins = len(combined["xvals"])
        assert combined["all_occupancy"].shape == (n_cells_processed, n_bins)

    def test_pooled_vs_averaged_occupancy_differ(self, mock_inputs):
        all_dd, mesh, cmap, cell_ids = mock_inputs
        result = get_binned_occupancy_dict_from_distance_dict(
            all_distance_dict=all_dd,
            combined_mesh_information_dict=mesh,
            channel_map=cmap,
            distance_measure="nucleus",
            bin_width=0.3,
            x_max=8.0,
            recalculate=True,
        )
        combined = result["real"]["combined"]
        # Pooled and averaged occupancy should generally differ
        assert not np.allclose(
            np.nan_to_num(combined["occupancy"]),
            np.nan_to_num(combined["occupancy_pooled"]),
        )

    def test_num_cells_limits_output(self, mock_inputs):
        all_dd, mesh, cmap, cell_ids = mock_inputs
        result = get_binned_occupancy_dict_from_distance_dict(
            all_distance_dict=all_dd,
            combined_mesh_information_dict=mesh,
            channel_map=cmap,
            distance_measure="nucleus",
            bin_width=0.3,
            num_cells=2,
            recalculate=True,
        )
        assert len(result["real"]["individual"]) == 2


class TestGetKdeOccupancyDictUnifiedFormat:
    """Verify get_kde_occupancy_dict produces the same unified combined-key format as discrete."""

    @pytest.fixture()
    def mock_kde_inputs(self):
        """Build minimal synthetic kde_dict and channel_map."""
        rng = np.random.default_rng(77)
        n_cells = 4
        cell_ids = [f"cell_{i}" for i in range(n_cells)]
        n_points = 50  # small for speed

        kde_dict: dict = {}
        for cid in cell_ids:
            occupied_data = rng.exponential(2.0, size=300)
            available_data = rng.exponential(3.0, size=1000)
            kde_dict[cid] = {
                "real": gaussian_kde(occupied_data),
                "available": gaussian_kde(available_data),
            }

        channel_map = {"real": "ST001"}
        return kde_dict, channel_map, n_cells, n_points

    def test_combined_has_all_unified_keys(self, mock_kde_inputs):
        kde_dict, channel_map, n_cells, n_points = mock_kde_inputs
        result = get_kde_occupancy_dict(
            distance_kde_dict=kde_dict,
            channel_map=channel_map,
            num_points=n_points,
            x_min=0,
            x_max=8.0,
            recalculate=True,
        )
        combined = result["real"]["combined"]
        # KDE path produces mean-of-per-cell occupancy; pooled KDE and pooled
        # PDFs are not computed (those live on the binned/discrete path).
        expected_keys = {
            "xvals",
            "occupancy",
            "std_occupancy",
            "envelope_lo",
            "envelope_hi",
            "all_occupancy",
        }
        assert expected_keys == set(combined.keys())

    def test_all_occupancy_shape(self, mock_kde_inputs):
        kde_dict, channel_map, n_cells, n_points = mock_kde_inputs
        result = get_kde_occupancy_dict(
            distance_kde_dict=kde_dict,
            channel_map=channel_map,
            num_points=n_points,
            x_min=0,
            x_max=8.0,
            recalculate=True,
        )
        combined = result["real"]["combined"]
        assert combined["all_occupancy"].shape == (n_cells, n_points)
        assert len(combined["xvals"]) == n_points

    def test_mean_occupancy_matches_all_occupancy_nanmean(self, mock_kde_inputs):
        kde_dict, channel_map, n_cells, n_points = mock_kde_inputs
        result = get_kde_occupancy_dict(
            distance_kde_dict=kde_dict,
            channel_map=channel_map,
            num_points=n_points,
            x_min=0,
            x_max=8.0,
            recalculate=True,
        )
        combined = result["real"]["combined"]
        # combined["occupancy"] is the nanmean of the per-cell curves stacked
        # in combined["all_occupancy"] (shape: n_cells × n_points).
        expected_mean = np.nanmean(combined["all_occupancy"], axis=0)
        np.testing.assert_allclose(combined["occupancy"], expected_mean)

    def test_envelope_bounds_combined_mean(self, mock_kde_inputs):
        kde_dict, channel_map, n_cells, n_points = mock_kde_inputs
        result = get_kde_occupancy_dict(
            distance_kde_dict=kde_dict,
            channel_map=channel_map,
            num_points=n_points,
            x_min=0,
            x_max=8.0,
            recalculate=True,
        )
        combined = result["real"]["combined"]
        lo = combined["envelope_lo"]
        hi = combined["envelope_hi"]
        mu = combined["occupancy"]
        # At most grid points the mean should lie within [lo, hi]
        within = (mu >= lo - 1e-9) & (mu <= hi + 1e-9)
        assert np.mean(within) >= 0.9

    def test_individual_per_cell_keys(self, mock_kde_inputs):
        kde_dict, channel_map, n_cells, n_points = mock_kde_inputs
        result = get_kde_occupancy_dict(
            distance_kde_dict=kde_dict,
            channel_map=channel_map,
            num_points=n_points,
            x_min=0,
            x_max=8.0,
            recalculate=True,
        )
        individual = result["real"]["individual"]
        assert len(individual) == n_cells
        for cell_data in individual.values():
            assert {
                "xvals",
                "occupancy",
                "pdf_occupied",
                "pdf_available",
                "occupancy_common",
                "pdf_occupied_common",
                "pdf_available_common",
            } == set(cell_data.keys())
