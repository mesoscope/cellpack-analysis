"""Tests comparing the custom _wasserstein_1d_presorted against scipy's wasserstein_distance."""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from cellpack_analysis.lib.distance import _wasserstein_1d_presorted


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
