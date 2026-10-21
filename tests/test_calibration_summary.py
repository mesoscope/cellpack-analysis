"""Tests for ``summarize_per_cell_calibration``."""

import numpy as np
import pandas as pd
import pytest

from cellpack_analysis.lib.stats import summarize_across_cells, summarize_per_cell_calibration

COLUMNS = [
    "null_mode",
    "scope",
    "raw_rejection",
    "bh_rejection",
    "at_floor",
    "min_p",
    "expected_raw",
]
R = 98


def _per_cell_results(pvals: dict[tuple[str, str], list[int]], r: int = R) -> dict:
    """Wrap ``{(mode, scope): [k, ...]}`` (p = k / (r + 1)) like ``per_cell_envelope_test``."""
    modes = list(dict.fromkeys(mode for mode, _ in pvals))
    measures = [s for s in dict.fromkeys(scope for _, scope in pvals) if s != "joint"]
    n_cells = len(next(iter(pvals.values())))
    all_results = []
    for i in range(n_cells):
        cell = {}
        for mode in modes:
            cell[mode] = {
                "per_distance_measure": {
                    dm: {"pval": pvals[(mode, dm)][i] / (r + 1), "supremum_sign": 1}
                    for dm in measures
                },
                "joint": {"pval": pvals[(mode, "joint")][i] / (r + 1), "supremum_sign": 1},
            }
        all_results.append(cell)
    return {
        "summary": summarize_across_cells(all_results, alpha=0.05),
        "per_cell_results": all_results,
        "cell_ids": [f"c{i}" for i in range(n_cells)],
        "null_modes": modes,
        "distance_measures": measures,
        "alpha": 0.05,
        "statistic": "intdev",
        "envelope_type": "pointwise",
        "calibration_holdout_index": 0,
    }


@pytest.fixture()
def results() -> dict:
    # k is the p-value in units of 1/(R+1); k <= 4 is p < 0.05 for R = 98
    return _per_cell_results(
        {
            ("random", "nucleus"): [1, 1, 1, 1, 40],
            ("random", "z"): [4, 5, 50, 60, 70],
            ("random", "joint"): [30, 31, 32, 33, 34],
            ("apical", "nucleus"): [2, 3, 4, 5, 6],
            ("apical", "z"): [99, 99, 99, 99, 99],
            ("apical", "joint"): [1, 50, 60, 70, 80],
        }
    )


@pytest.fixture()
def table(results) -> pd.DataFrame:
    return summarize_per_cell_calibration(results, n_envelope_replicates=R, alpha=0.05)


def _row(table: pd.DataFrame, mode: str, scope: str) -> pd.Series:
    return table[(table.null_mode == mode) & (table.scope == scope)].iloc[0]


class TestSummarizePerCellCalibration:
    def test_columns_and_row_order(self, table):
        assert list(table.columns) == COLUMNS
        # modes in null_modes order; each mode lists its measures, then the joint scope
        assert list(zip(table.null_mode, table.scope, strict=True)) == [
            ("random", "nucleus"),
            ("random", "z"),
            ("random", "joint"),
            ("apical", "nucleus"),
            ("apical", "z"),
            ("apical", "joint"),
        ]

    def test_raw_rejection_counts_p_below_alpha(self, table):
        assert _row(table, "random", "nucleus").raw_rejection == pytest.approx(4 / 5)
        assert _row(table, "random", "z").raw_rejection == pytest.approx(1 / 5)
        assert _row(table, "apical", "nucleus").raw_rejection == pytest.approx(3 / 5)
        assert _row(table, "apical", "z").raw_rejection == 0.0
        # p = 5/99 = 0.0505 is above alpha, so only k <= 4 is counted

    def test_at_floor_and_min_p(self, table):
        row = _row(table, "random", "nucleus")
        assert row.at_floor == pytest.approx(4 / 5)
        assert row.min_p == pytest.approx(1 / 99)
        assert _row(table, "apical", "nucleus").at_floor == 0.0
        assert _row(table, "apical", "z").min_p == pytest.approx(1.0)

    def test_bh_rejection_follows_the_summary_qvals(self, table, results):
        # four cells at the floor of five: BH q = 0.0101 * 5 / 4 < 0.05
        assert _row(table, "random", "nucleus").bh_rejection == pytest.approx(4 / 5)
        # a single cell at the floor cannot survive BH over five cells
        assert _row(table, "apical", "joint").raw_rejection == pytest.approx(1 / 5)
        assert _row(table, "apical", "joint").bh_rejection == 0.0
        for r in table.itertuples():
            key = r.null_mode, r.scope
            q = (
                results["summary"]["joint_qvals"][r.null_mode]
                if r.scope == "joint"
                else results["summary"]["per_distance_measure_qvals"][key]
            )
            assert r.bh_rejection == np.mean(q.to_numpy(dtype=float) < 0.05)

    def test_expected_raw_for_98_replicates(self, table):
        assert (table.expected_raw == 4 / 99).all()
        assert round(float(table.expected_raw.iloc[0]), 4) == 0.0404

    @pytest.mark.parametrize(
        ("n_reps", "alpha", "expected"),
        [(99, 0.05, 0.04), (98, 0.05, 4 / 99), (199, 0.05, 9 / 200), (99, 0.01, 0.0)],
    )
    def test_expected_raw_formula(self, results, n_reps, alpha, expected):
        out = summarize_per_cell_calibration(results, n_envelope_replicates=n_reps, alpha=alpha)
        assert out.expected_raw.iloc[0] == pytest.approx(expected)

    def test_floor_follows_the_replicate_count(self):
        # with R = 99 the smallest p is 1/100, so k = 1 sits at the floor
        res = _per_cell_results(
            {("random", "nucleus"): [1, 2, 3], ("random", "joint"): [1, 1, 50]}, r=99
        )
        out = summarize_per_cell_calibration(res, n_envelope_replicates=99)
        assert _row(out, "random", "nucleus").at_floor == pytest.approx(1 / 3)
        assert _row(out, "random", "joint").at_floor == pytest.approx(2 / 3)
        assert _row(out, "random", "joint").min_p == pytest.approx(0.01)
        assert (out.expected_raw == 0.04).all()

    def test_alpha_sets_the_rejection_threshold(self, results):
        out = summarize_per_cell_calibration(results, n_envelope_replicates=R, alpha=0.02)
        # p < 0.02 needs k <= 1
        assert _row(out, "apical", "nucleus").raw_rejection == 0.0
        assert _row(out, "random", "nucleus").raw_rejection == pytest.approx(4 / 5)
