"""Tests comparing the custom _wasserstein_1d_presorted against scipy's wasserstein_distance."""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from cellpack_analysis.lib.distance import (
    POOLED_SEED_LABEL,
    _wasserstein_1d_presorted,
    get_distance_distribution_emd_df,
)


def _custom_emd(u: np.ndarray, v: np.ndarray) -> float:
    """Sort inputs and delegate to the presorted implementation, mirroring real call sites."""
    return _wasserstein_1d_presorted(np.sort(u), np.sort(v))


# ---------------------------------------------------------------------------
# Parametrised agreement tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "u, v, description",
    [
        # identical distributions → EMD = 0
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            "identical arrays",
        ),
        # constant shift → EMD = shift amount
        (
            np.zeros(50),
            np.ones(50),
            "constant shift by 1",
        ),
        # different-length arrays
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.5, 1.5]),
            "different-length arrays",
        ),
        # single-element arrays
        (
            np.array([3.0]),
            np.array([7.0]),
            "single-element arrays",
        ),
        # float32 input (dtype used in the pipeline)
        (
            np.array([0.1, 0.4, 0.9], dtype=np.float32),
            np.array([0.2, 0.5, 0.8], dtype=np.float32),
            "float32 dtype",
        ),
        # large random arrays (seeded for reproducibility)
        (
            np.random.default_rng(0).normal(0, 1, 1000),
            np.random.default_rng(1).normal(0.5, 1, 1000),
            "large random normal arrays",
        ),
        # exponential vs uniform
        (
            np.random.default_rng(42).exponential(1.0, 500),
            np.random.default_rng(42).uniform(0, 3, 500),
            "exponential vs uniform",
        ),
        # arrays with duplicate values
        (
            np.array([1.0, 1.0, 1.0, 2.0, 2.0]),
            np.array([1.0, 2.0, 2.0, 2.0, 3.0]),
            "arrays with duplicate values",
        ),
        # already-sorted input (common case in the pipeline after np.sort)
        (
            np.linspace(0, 1, 200),
            np.linspace(0.1, 1.1, 200),
            "linspace shift (pre-sorted)",
        ),
    ],
)
def test_custom_emd_matches_scipy(u: np.ndarray, v: np.ndarray, description: str) -> None:
    """Custom EMD should match scipy's wasserstein_distance to within float32 tolerance."""
    expected = wasserstein_distance(u, v)
    actual = _custom_emd(u, v)
    assert actual == pytest.approx(
        expected, rel=1e-5, abs=1e-7
    ), f"Mismatch for '{description}': custom={actual}, scipy={expected}"


# ---------------------------------------------------------------------------
# Analytical value tests
# ---------------------------------------------------------------------------


def test_emd_identical_distributions_is_zero() -> None:
    arr = np.linspace(0, 10, 100)
    assert _custom_emd(arr, arr) == pytest.approx(0.0, abs=1e-10)


def test_emd_constant_shift() -> None:
    """For two same-size uniform distributions offset by `shift`, EMD == shift."""
    shift = 5.0
    u = np.zeros(100)
    v = np.full(100, shift)
    assert _custom_emd(u, v) == pytest.approx(shift, rel=1e-6)


def test_emd_symmetry() -> None:
    """EMD(u, v) == EMD(v, u)."""
    rng = np.random.default_rng(7)
    u = rng.normal(0, 1, 300)
    v = rng.normal(1, 2, 300)
    assert _custom_emd(u, v) == pytest.approx(_custom_emd(v, u), rel=1e-6)


def test_emd_non_negative() -> None:
    """EMD is always non-negative."""
    rng = np.random.default_rng(99)
    for _ in range(10):
        u = rng.uniform(-5, 5, 50)
        v = rng.uniform(-5, 5, 50)
        assert _custom_emd(u, v) >= 0.0


# ---------------------------------------------------------------------------
# Seed pooling in get_distance_distribution_emd_df
# ---------------------------------------------------------------------------


def _toy_distance_dict() -> dict:
    """Two modes x two cells x three seeds, one distance measure."""
    rng = np.random.default_rng(0)
    return {
        "nucleus": {
            mode: {
                cell: {f"seed_{s}": rng.normal(size=5) for s in range(3)}
                for cell in ("cell_a", "cell_b")
            }
            for mode in ("observed", "random")
        }
    }


def test_pool_seeds_false_keeps_one_curve_per_seed() -> None:
    """Default behaviour is unchanged: every (mode, cell, seed) is its own curve."""
    df = get_distance_distribution_emd_df(
        all_distance_dict=_toy_distance_dict(),
        packing_modes=["observed", "random"],
        distance_measures=["nucleus"],
        results_dir=None,
    )
    n_curves = 2 * 2 * 3
    assert len(df) == n_curves * (n_curves - 1) // 2
    assert set(df["seed_1"]) == {"seed_0", "seed_1", "seed_2"}


def test_pool_seeds_collapses_each_cell_to_one_curve() -> None:
    """With pooling, each (mode, cell) contributes exactly one labelled curve."""
    df = get_distance_distribution_emd_df(
        all_distance_dict=_toy_distance_dict(),
        packing_modes=["observed", "random"],
        distance_measures=["nucleus"],
        results_dir=None,
        pool_seeds=True,
    )
    n_curves = 2 * 2
    assert len(df) == n_curves * (n_curves - 1) // 2
    assert set(df["seed_1"]) == {POOLED_SEED_LABEL}
    assert set(df["seed_2"]) == {POOLED_SEED_LABEL}


def test_pool_seeds_concatenates_that_cells_seeds() -> None:
    """The pooled curve is the concatenation of the cell's seeds, not a mean of EMDs."""
    dist_dict = _toy_distance_dict()
    df = get_distance_distribution_emd_df(
        all_distance_dict=dist_dict,
        packing_modes=["observed", "random"],
        distance_measures=["nucleus"],
        results_dir=None,
        pool_seeds=True,
    )
    row = df[
        (df["packing_mode_1"] == "observed")
        & (df["cell_id_1"] == "cell_a")
        & (df["packing_mode_2"] == "random")
        & (df["cell_id_2"] == "cell_a")
    ]
    assert len(row) == 1
    expected = _custom_emd(
        np.concatenate(list(dist_dict["nucleus"]["observed"]["cell_a"].values())),
        np.concatenate(list(dist_dict["nucleus"]["random"]["cell_a"].values())),
    )
    assert row["emd"].iloc[0] == pytest.approx(expected)


def test_emd_pair_guard_raises_before_materialising_pairs(monkeypatch) -> None:
    """The quadratic blow-up is caught with a message naming the escape hatch."""
    monkeypatch.setattr("cellpack_analysis.lib.distance.MAX_EMD_PAIRS", 10)
    with pytest.raises(ValueError, match="pool_seeds=True"):
        get_distance_distribution_emd_df(
            all_distance_dict=_toy_distance_dict(),
            packing_modes=["observed", "random"],
            distance_measures=["nucleus"],
            results_dir=None,
        )


def test_emd_pair_guard_not_triggered_when_pooling(monkeypatch) -> None:
    """Pooling brings the same dataset back under the guard."""
    monkeypatch.setattr("cellpack_analysis.lib.distance.MAX_EMD_PAIRS", 10)
    df = get_distance_distribution_emd_df(
        all_distance_dict=_toy_distance_dict(),
        packing_modes=["observed", "random"],
        distance_measures=["nucleus"],
        results_dir=None,
        pool_seeds=True,
    )
    assert len(df) == 6
