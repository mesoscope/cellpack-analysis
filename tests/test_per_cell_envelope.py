"""Tests for the per-cell Monte Carlo rank-envelope (CSR) test.

``per_cell_envelope_test`` pairs each cell's baseline (experimental) distance
distribution with that same cell's simulated replicate envelope, then aggregates
per-cell p-values across cells with BH correction.  These tests verify the two
ends of the behaviour: a baseline drawn from the null should rarely be rejected,
and a baseline shifted far outside the null should be rejected everywhere, with
the rank-envelope p-value bounded below by ``1 / (R + 1)``.
"""

import numpy as np
import pytest

from cellpack_analysis.lib.stats import per_cell_envelope_test

N_REPS = 99
N_PTS = 200


def _make_distance_dict(
    rng: np.random.Generator,
    n_cells: int,
    baseline_low: float,
    baseline_high: float,
    distance_measures: tuple[str, ...] = ("nucleus",),
) -> tuple[dict, str, list[str]]:
    """Build an ``all_distance_dict`` with one baseline mode and one null model.

    Null replicates are drawn from ``U(0, 2)``; the baseline for each cell is drawn
    from ``U(baseline_low, baseline_high)``.
    """
    baseline_mode = "exp"
    null_mode = "random"
    all_distance_dict: dict = {dm: {baseline_mode: {}, null_mode: {}} for dm in distance_measures}
    for c in range(n_cells):
        cid = f"cell{c}"
        for dm in distance_measures:
            all_distance_dict[dm][baseline_mode][cid] = {
                "0": rng.uniform(baseline_low, baseline_high, N_PTS)
            }
            all_distance_dict[dm][null_mode][cid] = {
                str(s): rng.uniform(0.0, 2.0, N_PTS) for s in range(N_REPS)
            }
    return all_distance_dict, baseline_mode, [baseline_mode, null_mode]


def test_output_structure():
    rng = np.random.default_rng(0)
    d, baseline, modes = _make_distance_dict(rng, n_cells=8, baseline_low=0.0, baseline_high=2.0)
    res = per_cell_envelope_test(
        d, baseline, modes, ["nucleus"], alpha=0.05, bin_width=0.2, envelope_type="rank"
    )
    assert res["null_modes"] == ["random"]
    assert res["baseline_mode"] == "exp"
    assert len(res["cell_ids"]) == 8
    assert len(res["per_cell_results"]) == 8
    # Summary carries BH q-values and rejection fractions.
    assert ("random", "nucleus") in res["summary"]["rejection_per_distance_measure"].index


def test_null_baseline_low_rejection():
    """Baseline drawn from the null -> few cells rejected (near alpha)."""
    rng = np.random.default_rng(1)
    d, baseline, modes = _make_distance_dict(rng, n_cells=25, baseline_low=0.0, baseline_high=2.0)
    res = per_cell_envelope_test(
        d, baseline, modes, ["nucleus"], alpha=0.05, bin_width=0.2, envelope_type="rank"
    )
    rej = res["summary"]["rejection_per_distance_measure"]
    assert float(rej[("random", "nucleus")]) < 0.35


def test_shifted_baseline_high_rejection():
    """Baseline shifted far outside the null -> nearly all cells rejected."""
    rng = np.random.default_rng(2)
    d, baseline, modes = _make_distance_dict(rng, n_cells=25, baseline_low=5.0, baseline_high=7.0)
    res = per_cell_envelope_test(
        d, baseline, modes, ["nucleus"], alpha=0.05, bin_width=0.2, envelope_type="rank"
    )
    rej = res["summary"]["rejection_per_distance_measure"]
    assert float(rej[("random", "nucleus")]) > 0.8

    # Rank-envelope p-value is bounded below by 1 / (R + 1).
    pvals = res["summary"]["per_distance_measure_pvals"][("random", "nucleus")].to_numpy()
    assert pvals.min() >= 1.0 / (N_REPS + 1) - 1e-9


def test_joint_two_measures_shifted():
    """Joint test over two measures also rejects a clearly-shifted baseline."""
    rng = np.random.default_rng(3)
    d, baseline, modes = _make_distance_dict(
        rng, n_cells=15, baseline_low=5.0, baseline_high=7.0, distance_measures=("nucleus", "z")
    )
    res = per_cell_envelope_test(
        d,
        baseline,
        modes,
        ["nucleus", "z"],
        alpha=0.05,
        bin_width=0.2,
        envelope_type="rank",
    )
    assert float(res["summary"]["rejection_joint"]["random"]) > 0.8


def test_requires_a_null_model():
    rng = np.random.default_rng(4)
    d, baseline, _ = _make_distance_dict(rng, n_cells=3, baseline_low=0.0, baseline_high=2.0)
    with pytest.raises(ValueError, match="null model"):
        per_cell_envelope_test(d, baseline, [baseline], ["nucleus"], envelope_type="rank")
