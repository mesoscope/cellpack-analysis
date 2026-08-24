"""Tests for the refactored histogram-based occupancy functions."""

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from cellpack_analysis.analysis.workflows.configs import defaults
from cellpack_analysis.lib.occupancy import (
    _apply_coverage_mask,
    _compute_single_cell_occupancy,
    _get_common_xvals_from_kde,
    get_binned_occupancy_dict_from_distance_dict,
    get_kde_occupancy_dict,
    get_kde_occupancy_for_single_cell,
)
from cellpack_analysis.lib.stats import normalize_pdf, pdf_ratio


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
        all_dd, mesh, cmap, _ = mock_inputs
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
        all_dd, mesh, cmap, _ = mock_inputs
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
        for _, data in individual.items():
            assert len(data["xvals"]) == len(data["occupancy"])
            assert len(data["xvals"]) == len(data["pdf_occupied"])
            assert len(data["xvals"]) == len(data["pdf_available"])

    def test_combined_all_occupancy_shape(self, mock_inputs):
        all_dd, mesh, cmap, _ = mock_inputs
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
        kde_dict, channel_map, _, n_points = mock_kde_inputs
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
            min_coverage_fraction=0.0,
        )
        combined = result["real"]["combined"]
        # With the coverage mask disabled, combined["occupancy"] is exactly the
        # nanmean of the per-cell curves stacked in combined["all_occupancy"].
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
        # At most grid points where mu is finite, the mean should lie within [lo, hi]
        finite_mask = np.isfinite(mu)
        assert finite_mask.any(), "Expected at least some finite occupancy values"
        within = (mu[finite_mask] >= lo[finite_mask] - 1e-9) & (
            mu[finite_mask] <= hi[finite_mask] + 1e-9
        )
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


class TestNormalizePdfEdgeCases:
    """Edge-case tests for normalize_pdf."""

    def test_zero_integral_returns_nan(self):
        xvals = np.linspace(0, 5, 50)
        density = np.zeros(50)
        result = normalize_pdf(xvals, density)
        assert np.all(np.isnan(result)), "All-zero density must return all-NaN"

    def test_positive_density_integrates_to_one(self):
        xvals = np.linspace(0, 5, 100)
        density = np.exp(-xvals)
        result = normalize_pdf(xvals, density)
        np.testing.assert_allclose(np.trapezoid(result, xvals), 1.0, atol=1e-6)


class TestPdfRatioEdgeCases:
    """Edge-case tests for pdf_ratio."""

    def test_all_zero_denominator_returns_nan(self):
        xvals = np.linspace(0, 5, 50)
        numerator = np.exp(-xvals)
        denominator = np.zeros(50)
        ratio, _, _ = pdf_ratio(xvals, numerator, denominator)
        # normalize_pdf returns all-NaN for all-zero denominator; that NaN
        # must propagate through the ratio rather than being zeroed out.
        assert np.all(np.isnan(ratio)), "NaN denominator must propagate NaN ratio, not zero"

    def test_identical_densities_ratio_near_one(self):
        rng = np.random.default_rng(0)
        xvals = np.linspace(0, 10, 100)
        density = np.exp(-xvals / 3.0)
        ratio, _, _ = pdf_ratio(xvals, density, density)
        finite = ratio[np.isfinite(ratio)]
        np.testing.assert_allclose(finite, 1.0, atol=1e-6)


class TestGetKdeOccupancyForSingleCell:
    """Unit tests for get_kde_occupancy_for_single_cell fixes."""

    @pytest.fixture()
    def kde_inputs(self):
        rng = np.random.default_rng(42)
        available_data = rng.exponential(3.0, size=1000)
        occupied_data = rng.exponential(2.0, size=300)
        kde_dict = {
            "cell_0": {
                "real": gaussian_kde(occupied_data),
                "available": gaussian_kde(available_data),
            }
        }
        return kde_dict, available_data

    def test_bandwidth_does_not_mutate_original_kde(self, kde_inputs):
        kde_dict, _ = kde_inputs
        original_factor = float(kde_dict["cell_0"]["real"].factor)
        x_vals = np.linspace(0, 8, 50)
        get_kde_occupancy_for_single_cell(
            cell_id="cell_0",
            mode="real",
            distance_kde_dict=kde_dict,
            common_xvals=x_vals,
            bandwidth=0.5,
            num_points=50,
        )
        assert float(kde_dict["cell_0"]["real"].factor) == original_factor, (
            "Original KDE bandwidth must not be mutated by get_kde_occupancy_for_single_cell"
        )

    def test_occupancy_common_nan_outside_support(self, kde_inputs):
        kde_dict, available_data = kde_inputs
        avail_min = float(np.min(available_data))
        avail_max = float(np.max(available_data))
        # Extend x_vals well beyond the cell's data range
        x_vals = np.linspace(avail_min - 5.0, avail_max + 5.0, 200)
        result = get_kde_occupancy_for_single_cell(
            cell_id="cell_0",
            mode="real",
            distance_kde_dict=kde_dict,
            common_xvals=x_vals,
            bandwidth=None,
            num_points=50,
        )
        assert result is not None, "Expected a result dict from get_kde_occupancy_for_single_cell"
        occ = result[1]["occupancy_common"]
        out_of_support = (x_vals < avail_min) | (x_vals > avail_max)
        assert np.all(np.isnan(occ[out_of_support])), (
            "occupancy_common must be NaN for x_vals outside the available-data support"
        )
        in_support = ~out_of_support
        assert np.any(np.isfinite(occ[in_support])), (
            "occupancy_common must contain finite values within the available-data support"
        )

    def test_no_infinities_when_objects_exceed_support(self, kde_inputs):
        kde_dict, available_data = kde_inputs
        avail_max = float(np.max(available_data))
        # "Skip filtering" scenario: inject occupied objects well beyond the
        # available-space support and confirm the ratio never blows up.
        rng = np.random.default_rng(3)
        occupied_beyond = np.concatenate([
            rng.exponential(2.0, size=300),
            avail_max + rng.uniform(1.0, 5.0, size=50),
        ])
        kde_dict["cell_0"]["real"] = gaussian_kde(occupied_beyond)
        x_vals = np.linspace(0.0, avail_max + 10.0, 200)
        result = get_kde_occupancy_for_single_cell(
            cell_id="cell_0",
            mode="real",
            distance_kde_dict=kde_dict,
            common_xvals=x_vals,
            bandwidth=None,
            num_points=50,
        )
        assert result is not None
        occ = result[1]["occupancy_common"]
        assert not np.any(np.isinf(occ)), "occupancy_common must never contain infinities"
        assert np.any(np.isfinite(occ)), (
            "occupancy_common must have finite values within the available-data support"
        )


class TestApplyCoverageMask:
    """Unit tests for the trailing coverage mask applied to the combined aggregate."""

    def test_masks_high_x_tail_keeps_low_x_start(self):
        # 4 cells, 6 grid points. Coverage: idx0 = 1 cell (leading, low),
        # idx1-3 = 4 cells (full), idx4-5 = 1 cell (trailing, low).
        all_curves = np.full((4, 6), np.nan)
        all_curves[0, 0] = 1.0
        all_curves[:, 1:4] = 1.0
        all_curves[0, 4:6] = 1.0
        mean = np.nanmean(all_curves, axis=0)
        env_lo = mean.copy()
        env_hi = mean.copy()
        _apply_coverage_mask(all_curves, 0.95, mean, env_lo, env_hi)
        # Trailing low-coverage points are blanked...
        assert np.isnan(mean[4]) and np.isnan(mean[5])
        assert np.isnan(env_lo[4]) and np.isnan(env_hi[5])
        # ...the full-coverage core is kept...
        assert np.all(np.isfinite(mean[1:4]))
        # ...and the leading low-x point is preserved (keeps the range start).
        assert np.isfinite(mean[0])
        # The per-cell data is untouched.
        assert np.isfinite(all_curves[0, 0]) and np.isfinite(all_curves[0, 5])

    def test_disabled_when_fraction_zero(self):
        all_curves = np.full((3, 5), np.nan)
        all_curves[0, :] = 1.0  # only 1/3 coverage everywhere
        mean = np.nanmean(all_curves, axis=0)
        _apply_coverage_mask(all_curves, 0.0, mean)
        assert np.all(np.isfinite(mean))  # nothing masked when disabled


# ---------------------------------------------------------------------------
# Tests: D2 — grid density and full-support range
# ---------------------------------------------------------------------------


class TestOccupancyGridResolution:
    """D2: num_points is 1000 and the grid spans the full [0, xlim] support."""

    def test_default_num_points_is_1000(self):
        assert defaults.OCCUPANCY_NUM_POINTS == 1000
        for dm in ("nucleus", "z"):
            assert defaults.OCCUPANCY_PARAMS[dm]["num_points"] == 1000

    def test_default_xlims_span_full_support(self):
        assert defaults.OCCUPANCY_PARAMS["nucleus"]["xlim"] == 6
        assert defaults.OCCUPANCY_PARAMS["z"]["xlim"] == 8
        for dm in ("nucleus", "z"):
            assert defaults.OCCUPANCY_PARAMS[dm]["x_min"] == 0

    def test_shallow_merge_drops_num_points_so_fallback_is_needed(self):
        """The real configs omit num_points; the shallow merge must not KeyError.

        `AnalysisConfig._setup_occupancy_params` does defaults.copy() then
        .update(), so a config overriding any key of occupancy_params[dm]
        replaces the whole inner dict. Reading num_points must therefore use
        .get() with the module default, which is what the workflow does.
        """
        config_params = {
            "nucleus": {"xlim": 6, "ylim": 3, "bandwidth": 0.2},
            "z": {"xlim": 8, "ylim": 2, "bandwidth": 0.2},
        }
        merged = defaults.OCCUPANCY_PARAMS.copy()
        merged.update(config_params)
        for dm in ("nucleus", "z"):
            assert "num_points" not in merged[dm]
            assert merged[dm].get("num_points", defaults.OCCUPANCY_NUM_POINTS) == 1000

    def test_explicit_x_max_gives_full_support_grid(self):
        """An explicit x_max produces linspace(x_min, x_max, num_points) exactly."""
        rng = np.random.default_rng(11)
        # Cell data stops well short of x_max; the grid must still reach it.
        kde_dict = {"cell_0": {"available": gaussian_kde(rng.exponential(1.0, size=200))}}
        grid = _get_common_xvals_from_kde(
            distance_kde_dict=kde_dict,
            cell_id_map={"real": ["cell_0"]},
            num_points=defaults.OCCUPANCY_NUM_POINTS,
            pct_min_cells=0,
            x_min=0,
            x_max=6,
        )
        np.testing.assert_allclose(grid, np.linspace(0, 6, 1000))

    def test_z_measure_grid_reaches_eight(self):
        rng = np.random.default_rng(12)
        kde_dict = {"cell_0": {"available": gaussian_kde(rng.exponential(1.0, size=200))}}
        grid = _get_common_xvals_from_kde(
            distance_kde_dict=kde_dict,
            cell_id_map={"real": ["cell_0"]},
            num_points=defaults.OCCUPANCY_NUM_POINTS,
            pct_min_cells=0,
            x_min=0,
            x_max=8,
        )
        assert grid[0] == 0.0
        assert grid[-1] == pytest.approx(8.0)
        assert grid.size == 1000
