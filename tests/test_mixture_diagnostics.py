"""Tests for the virtual-mixture diagnostics (D2 of the rule-mixing handoff).

The virtual mixture draws, for every cell and seed, ``round(c_i * N)`` puncta from
rule *i*'s existing packing of that same cell/seed and unions them. By the mixture
identity its expected occupancy equals ``sum(c_i * occ_i)``, so it previews the
achievable packed-vs-expected match without any engine change or new packing.
"""

import numpy as np
import pytest

from cellpack_analysis.lib import mixture_diagnostics


class TestAllocateMixtureCounts:
    def test_counts_sum_to_total(self):
        counts = mixture_diagnostics.allocate_mixture_counts(
            59, {"a": 0.325, "b": 0.654, "c": 0.021}
        )
        assert sum(counts.values()) == 59

    def test_unnormalized_coefficients_are_normalized(self):
        counts_raw = mixture_diagnostics.allocate_mixture_counts(100, {"a": 2.0, "b": 6.0})
        counts_norm = mixture_diagnostics.allocate_mixture_counts(100, {"a": 0.25, "b": 0.75})
        assert counts_raw == counts_norm == {"a": 25, "b": 75}

    def test_largest_remainder_is_proportional_within_one(self):
        coeffs = {"a": 0.325, "b": 0.654, "c": 0.021}
        total = sum(coeffs.values())
        n = 137
        counts = mixture_diagnostics.allocate_mixture_counts(n, coeffs)
        for rule, c in coeffs.items():
            assert abs(counts[rule] - n * c / total) < 1.0

    def test_zero_coefficient_gets_zero_count(self):
        counts = mixture_diagnostics.allocate_mixture_counts(50, {"a": 0.0, "b": 1.0})
        assert counts == {"a": 0, "b": 50}

    def test_deterministic(self):
        coeffs = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        results = {
            tuple(sorted(mixture_diagnostics.allocate_mixture_counts(10, coeffs).items()))
            for _ in range(20)
        }
        assert len(results) == 1

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            mixture_diagnostics.allocate_mixture_counts(10, {"a": 0.0, "b": 0.0})

    def test_zero_points(self):
        counts = mixture_diagnostics.allocate_mixture_counts(0, {"a": 0.5, "b": 0.5})
        assert counts == {"a": 0, "b": 0}


def _synthetic_distance_dict(n_cells=3, n_seeds=5, n_points=40, offset=100.0):
    """Two rules with disjoint constant values; measure ``m2`` = ``m1`` + offset.

    The offset relationship lets tests verify that the same point indices are
    drawn from every distance measure (joint consistency).
    """
    rng = np.random.default_rng(0)
    all_distance_dict: dict = {"m1": {}, "m2": {}}
    for rule, loc in [("rule_a", 1.0), ("rule_b", 3.0)]:
        all_distance_dict["m1"][rule] = {}
        all_distance_dict["m2"][rule] = {}
        for cell in [f"70{i}" for i in range(n_cells)]:
            all_distance_dict["m1"][rule][cell] = {}
            all_distance_dict["m2"][rule][cell] = {}
            for seed in [str(s) for s in range(n_seeds)]:
                base = rng.normal(loc, 0.05, size=n_points)
                all_distance_dict["m1"][rule][cell][seed] = base
                all_distance_dict["m2"][rule][cell][seed] = base + offset
    return all_distance_dict


class TestBuildVirtualMixtureDistances:
    def test_structure_and_counts(self):
        add = _synthetic_distance_dict()
        mixed = mixture_diagnostics.build_virtual_mixture_distances(
            all_distance_dict=add,
            coefficients={"rule_a": 0.3, "rule_b": 0.7},
            distance_measures=["m1", "m2"],
        )
        assert set(mixed.keys()) == {"m1", "m2"}
        assert set(mixed["m1"].keys()) == set(add["m1"]["rule_a"].keys())
        for cell, seed_dict in mixed["m1"].items():
            assert set(seed_dict.keys()) == set(add["m1"]["rule_a"][cell].keys())
            for arr in seed_dict.values():
                assert arr.shape == (40,)

    def test_mixture_fractions_realized_exactly(self):
        add = _synthetic_distance_dict(n_points=40)
        mixed = mixture_diagnostics.build_virtual_mixture_distances(
            all_distance_dict=add,
            coefficients={"rule_a": 0.3, "rule_b": 0.7},
            distance_measures=["m1"],
        )
        # rule_a values sit near 1.0, rule_b near 3.0 — count by threshold.
        for seed_dict in mixed["m1"].values():
            for arr in seed_dict.values():
                n_a = int(np.sum(arr < 2.0))
                assert (
                    n_a
                    == mixture_diagnostics.allocate_mixture_counts(
                        40, {"rule_a": 0.3, "rule_b": 0.7}
                    )["rule_a"]
                )

    def test_same_indices_across_measures(self):
        add = _synthetic_distance_dict(offset=100.0)
        mixed = mixture_diagnostics.build_virtual_mixture_distances(
            all_distance_dict=add,
            coefficients={"rule_a": 0.5, "rule_b": 0.5},
            distance_measures=["m1", "m2"],
        )
        for cell in mixed["m1"]:
            for seed in mixed["m1"][cell]:
                np.testing.assert_allclose(mixed["m2"][cell][seed] - mixed["m1"][cell][seed], 100.0)

    def test_deterministic_given_base_seed(self):
        add = _synthetic_distance_dict()
        kwargs = {
            "all_distance_dict": add,
            "coefficients": {"rule_a": 0.3, "rule_b": 0.7},
            "distance_measures": ["m1"],
        }
        a = mixture_diagnostics.build_virtual_mixture_distances(base_seed=7, **kwargs)
        b = mixture_diagnostics.build_virtual_mixture_distances(base_seed=7, **kwargs)
        c = mixture_diagnostics.build_virtual_mixture_distances(base_seed=8, **kwargs)
        cell, seed = "700", "0"
        np.testing.assert_array_equal(a["m1"][cell][seed], b["m1"][cell][seed])
        assert not np.array_equal(a["m1"][cell][seed], c["m1"][cell][seed])

    def test_mixture_mean_matches_identity(self):
        add = _synthetic_distance_dict(n_cells=2, n_seeds=30, n_points=200)
        mixed = mixture_diagnostics.build_virtual_mixture_distances(
            all_distance_dict=add,
            coefficients={"rule_a": 0.3, "rule_b": 0.7},
            distance_measures=["m1"],
        )
        pooled = np.concatenate(
            [arr for seed_dict in mixed["m1"].values() for arr in seed_dict.values()]
        )
        assert abs(pooled.mean() - (0.3 * 1.0 + 0.7 * 3.0)) < 0.05

    def test_unequal_point_counts_raise(self):
        add = _synthetic_distance_dict()
        add["m1"]["rule_b"]["700"]["0"] = add["m1"]["rule_b"]["700"]["0"][:-3]
        with pytest.raises(ValueError, match="point counts"):
            mixture_diagnostics.build_virtual_mixture_distances(
                all_distance_dict=add,
                coefficients={"rule_a": 0.5, "rule_b": 0.5},
                distance_measures=["m1"],
            )

    def test_missing_rule_raises(self):
        add = _synthetic_distance_dict()
        with pytest.raises(ValueError, match="not present"):
            mixture_diagnostics.build_virtual_mixture_distances(
                all_distance_dict=add,
                coefficients={"rule_a": 0.5, "rule_z": 0.5},
                distance_measures=["m1"],
            )


class TestExpectedMixtureOccupancy:
    def test_weighted_sum_per_cell(self):
        xvals = np.linspace(0, 5, 11)
        occupancy_dict = {}
        for rule, level in [("rule_a", 1.0), ("rule_b", 2.0)]:
            occupancy_dict[rule] = {
                "individual": {
                    cell: {"xvals": xvals, "occupancy_common": np.full_like(xvals, level + i)}
                    for i, cell in enumerate(["700", "701"])
                },
                "combined": {"xvals": xvals},
            }
        expected = mixture_diagnostics.expected_mixture_occupancy(
            occupancy_dict, {"rule_a": 0.25, "rule_b": 0.75}
        )
        assert set(expected.keys()) == {"700", "701"}
        np.testing.assert_allclose(expected["700"], 0.25 * 1.0 + 0.75 * 2.0)
        np.testing.assert_allclose(expected["701"], 0.25 * 2.0 + 0.75 * 3.0)

    def test_nan_propagates_not_zero(self):
        xvals = np.linspace(0, 5, 5)
        curve_a = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        curve_b = np.array([2.0, 2.0, 2.0, np.nan, 2.0])
        occupancy_dict = {
            "rule_a": {"individual": {"700": {"xvals": xvals, "occupancy_common": curve_a}}},
            "rule_b": {"individual": {"700": {"xvals": xvals, "occupancy_common": curve_b}}},
        }
        expected = mixture_diagnostics.expected_mixture_occupancy(
            occupancy_dict, {"rule_a": 0.5, "rule_b": 0.5}
        )
        assert np.isnan(expected["700"][1]) and np.isnan(expected["700"][3])
        np.testing.assert_allclose(expected["700"][[0, 2, 4]], 1.5)
