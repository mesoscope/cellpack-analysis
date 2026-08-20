"""Regression tests for the batched pooled-envelope statistic.

``_batch_test_obs_vs_sim`` precomputes the fixed simulation statistic once and
reuses it across observed curves (O(M log M) instead of O(M^2)).  These tests
lock it to the per-row reference :func:`get_test_statistic_and_pvalue` so the
optimization is behaviour-preserving.
"""

import numpy as np
import pytest

from cellpack_analysis.lib.stats import _batch_test_obs_vs_sim, get_test_statistic_and_pvalue


def _reference(obs_curves: np.ndarray, sim: np.ndarray, statistic: str):
    pvals, signs = [], []
    for obs in obs_curves:
        p, _, _, s = get_test_statistic_and_pvalue(obs, sim, statistic=statistic)
        pvals.append(p)
        signs.append(s)
    return np.array(pvals), np.array(signs)


@pytest.mark.parametrize("statistic", ["intdev", "supremum"])
def test_batch_matches_per_row_reference(statistic):
    rng = np.random.default_rng(0)
    length = 40
    # Monotone-ish ECDF-like curves.
    sim = np.sort(rng.random((60, length)), axis=1)
    obs = np.sort(rng.random((17, length)), axis=1)

    p_batch, s_batch = _batch_test_obs_vs_sim(
        obs, sim, statistic=statistic, envelope_type="pointwise"
    )
    p_ref, s_ref = _reference(obs, sim, statistic)

    np.testing.assert_allclose(p_batch, p_ref, atol=1e-12)
    np.testing.assert_array_equal(s_batch, s_ref)


def test_batch_empty_obs_returns_empty():
    rng = np.random.default_rng(1)
    sim = rng.random((10, 20))
    pvals, signs = _batch_test_obs_vs_sim(
        np.empty((0, 20)), sim, statistic="intdev", envelope_type="pointwise"
    )
    assert pvals.size == 0
    assert signs.size == 0


def test_batch_rank_matches_reference():
    rng = np.random.default_rng(2)
    length = 30
    sim = np.sort(rng.random((50, length)), axis=1)
    obs = np.sort(rng.random((8, length)), axis=1)

    p_batch, s_batch = _batch_test_obs_vs_sim(obs, sim, envelope_type="rank")
    # Reference: compute_rank_envelope per row via get_test_statistic_and_pvalue("rank").
    p_ref, s_ref = _reference(obs, sim, "rank")

    np.testing.assert_allclose(p_batch, p_ref, atol=1e-12)
    np.testing.assert_array_equal(s_batch, s_ref)
