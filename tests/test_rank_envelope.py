"""Tests for compute_rank_envelope and the 'rank' statistic in get_test_statistic_and_pvalue."""

import numpy as np
import pytest

from cellpack_analysis.lib.stats import compute_rank_envelope, get_test_statistic_and_pvalue


@pytest.fixture()
def rank_envelope_data() -> tuple[np.ndarray, np.ndarray]:
    """Return (sims, obs_interior) with M=99 monotone curves of shape (M, L=50)."""
    rng = np.random.default_rng(42)
    M, L = 99, 50
    sims = rng.uniform(0, 1, (M, L))
    sims = np.sort(sims, axis=1)
    obs = sims.mean(axis=0)
    return sims, obs


class TestComputeRankEnvelope:
    def test_interior_obs_not_rejected(self, rank_envelope_data: tuple) -> None:
        """Obs equal to the pointwise mean of sims should not be rejected."""
        sims, obs = rank_envelope_data
        lo, hi, pval, sign, r_obs = compute_rank_envelope(obs, sims)
        assert pval > 0.05, f"Expected p > 0.05 for interior obs, got {pval}"
        assert np.all(lo <= obs + 1e-10) and np.all(obs <= hi + 1e-10), (
            "Interior obs should lie within the envelope"
        )

    def test_extreme_above_gives_minimum_pvalue(self, rank_envelope_data: tuple) -> None:
        """Obs strictly above all sims at every grid point yields the minimum p-value 1/(M+1)."""
        sims, _ = rank_envelope_data
        M = sims.shape[0]
        obs_hi = sims.max(axis=0) + 1.0
        _, _, pval, sign, r_obs = compute_rank_envelope(obs_hi, sims)
        expected_p = 1.0 / (M + 1)
        assert abs(pval - expected_p) < 1e-10, f"Expected p={expected_p}, got {pval}"
        assert sign == 1

    def test_extreme_below_gives_minimum_pvalue(self, rank_envelope_data: tuple) -> None:
        """Obs strictly below all sims at every grid point yields the minimum p-value 1/(M+1)."""
        sims, _ = rank_envelope_data
        M = sims.shape[0]
        obs_lo = sims.min(axis=0) - 1.0
        _, _, pval, sign, r_obs = compute_rank_envelope(obs_lo, sims)
        expected_p = 1.0 / (M + 1)
        assert abs(pval - expected_p) < 1e-10, f"Expected p={expected_p}, got {pval}"
        assert sign == -1

    def test_envelope_bounds_shape(self, rank_envelope_data: tuple) -> None:
        """lo and hi have the same shape as the input curves."""
        sims, obs = rank_envelope_data
        lo, hi, _, _, _ = compute_rank_envelope(obs, sims)
        assert lo.shape == obs.shape
        assert hi.shape == obs.shape

    def test_envelope_bounds_ordered(self, rank_envelope_data: tuple) -> None:
        """lo <= hi at every grid point."""
        sims, obs = rank_envelope_data
        lo, hi, _, _, _ = compute_rank_envelope(obs, sims)
        assert np.all(lo <= hi + 1e-10)


class TestGetTestStatisticRank:
    def test_rank_statistic_extreme_obs(self, rank_envelope_data: tuple) -> None:
        """'rank' statistic yields minimum p-value for an obs above all sims."""
        sims, _ = rank_envelope_data
        M = sims.shape[0]
        obs_hi = sims.max(axis=0) + 1.0
        pval, t_obs, t_sim, sign = get_test_statistic_and_pvalue(obs_hi, sims, statistic="rank")
        expected_p = 1.0 / (M + 1)
        assert abs(pval - expected_p) < 1e-10, f"Expected p={expected_p}, got {pval}"
        assert sign == 1

    def test_rank_statistic_t_sim_shape(self, rank_envelope_data: tuple) -> None:
        """t_sim has shape (M,) when statistic='rank'."""
        sims, obs = rank_envelope_data
        _, _, t_sim, _ = get_test_statistic_and_pvalue(obs, sims, statistic="rank")
        assert t_sim.shape == (sims.shape[0],)

    @pytest.mark.parametrize("statistic", ["supremum", "intdev"])
    def test_backward_compat(
        self, rank_envelope_data: tuple, statistic: str
    ) -> None:
        """Existing 'supremum' and 'intdev' statistics are unaffected."""
        sims, obs = rank_envelope_data
        pval, t_obs, t_sim, sign = get_test_statistic_and_pvalue(obs, sims, statistic=statistic)  # type: ignore[arg-type]
        assert 0.0 < pval <= 1.0
        assert t_sim.shape == (sims.shape[0],)
