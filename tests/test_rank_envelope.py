"""Tests for compute_rank_envelope and the 'rank' statistic in get_test_statistic_and_pvalue."""

import numpy as np
import pytest

from scipy import integrate

from cellpack_analysis.lib.stats import (
    _batch_test_obs_vs_sim,
    compute_rank_envelope,
    get_test_statistic_and_pvalue,
)


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


class TestDegenerateGridPoints:
    """Grid points where every simulation agrees carry no information.

    ``sd`` is zero there, so a variance-standardized deviation is undefined. Flooring
    ``sd`` to 1e-9 instead of excluding the point turns any observed difference into a
    deviation of ~1e7, which swamps the whole statistic and forces the minimum possible
    p-value regardless of the data everywhere else. Both statistic paths must exclude
    such points so every analysis treats them identically.
    """

    @staticmethod
    def _sims_with_degenerate_tail(rng, n_sims=99, n_pts=50, n_degenerate=5):
        """Simulations that are identical over a trailing block of grid points."""
        sims = np.sort(rng.uniform(0, 1, (n_sims, n_pts)), axis=1)
        sims[:, -n_degenerate:] = 1.0  # every ECDF saturates: sd == 0 here
        return sims

    def test_degenerate_point_does_not_dominate_the_statistic(self):
        rng = np.random.default_rng(0)
        sims = self._sims_with_degenerate_tail(rng)
        # Observed sits at the simulated mean everywhere it is informative, but
        # differs slightly in the degenerate tail.
        obs = sims.mean(axis=0)
        obs[-1] = 0.99

        pval, t_obs, _, _ = get_test_statistic_and_pvalue(obs, sims, statistic="intdev")

        # An interior observation must not be rejected because of one dead point.
        assert np.isfinite(t_obs)
        assert t_obs < 1e3, f"degenerate point still dominates: T={t_obs}"
        assert pval > 0.05

    def test_batch_and_scalar_paths_agree_on_degenerate_input(self):
        """The per-cell and pairwise tests must treat dead points the same way."""
        rng = np.random.default_rng(1)
        sims = self._sims_with_degenerate_tail(rng)
        obs = sims.mean(axis=0)
        obs[-1] = 0.99

        scalar_p, _, _, _ = get_test_statistic_and_pvalue(obs, sims, statistic="intdev")
        batch_p, _ = _batch_test_obs_vs_sim(
            obs[np.newaxis, :], sims, statistic="intdev", envelope_type="pointwise"
        )
        assert batch_p[0] == pytest.approx(scalar_p)

    def test_no_degenerate_points_is_a_no_op(self):
        """Excluding dead points must not perturb a well-behaved comparison."""
        rng = np.random.default_rng(2)
        sims = np.sort(rng.uniform(0, 1, (99, 50)), axis=1)
        obs = np.sort(rng.uniform(0, 1, 50))

        pval, t_obs, _, sign = get_test_statistic_and_pvalue(obs, sims, statistic="intdev")
        mu, sd = sims.mean(axis=0), sims.std(axis=0, ddof=1)
        expected = integrate.trapezoid(np.abs((obs - mu) / sd))
        assert t_obs == pytest.approx(expected)
        assert np.isfinite(pval)

    def test_all_points_degenerate_yields_no_evidence(self):
        """If nothing varies, there is nothing to test -> the largest possible p-value."""
        sims = np.ones((99, 20))
        obs = np.full(20, 0.5)
        pval, t_obs, _, _ = get_test_statistic_and_pvalue(obs, sims, statistic="intdev")
        assert t_obs == 0.0
        assert pval == pytest.approx(1.0)
