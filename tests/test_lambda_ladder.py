"""Decay-length ladder grouping and figures.

The ladder figures key off mode names (``nucleus_gradient_0pt03``), so the parser
has to reject every non-rung mode in a real ``channel_map`` -- the structure id,
``random``, and the canonical rules -- or a family panel picks up a stray curve.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cellpack_analysis.lib.rule_interpolation import CVResult
from cellpack_analysis.lib.visualization import (
    group_lambda_rungs,
    parse_lambda_rung,
    plot_lambda_ladder_profile,
)

BASELINE = "SLC25A17"
RUNGS = ["nucleus_gradient_0pt01", "nucleus_gradient_0pt1", "apical_gradient_0pt7"]
CHANNEL_MAP = dict.fromkeys([BASELINE, "random", *RUNGS], BASELINE)
XVALS = np.linspace(0, 6, 40)


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("nucleus_gradient_0pt03", ("nucleus_gradient", 0.03)),
        ("apical_gradient_10", ("apical_gradient", 10.0)),
        ("membrane_gradient_1", ("membrane_gradient", 1.0)),
        ("nucleus_gradient", None),
        ("random", None),
        ("SLC25A17", None),
        ("1_random", None),
        ("interpolated", None),
    ],
)
def test_parse_lambda_rung(mode: str, expected: tuple[str, float] | None) -> None:
    """Only numeric-suffixed gradient modes parse as ladder rungs."""
    assert parse_lambda_rung(mode) == expected


def test_group_lambda_rungs_sorts_by_decay_length() -> None:
    """Families collect their own rungs, ascending, and drop everything else."""
    families = group_lambda_rungs(CHANNEL_MAP)
    assert set(families) == {"nucleus_gradient", "apical_gradient"}
    assert families["nucleus_gradient"] == [
        (0.01, "nucleus_gradient_0pt01"),
        (0.1, "nucleus_gradient_0pt1"),
    ]


def _occupancy_dict() -> dict:
    """Minimal occupancy dict with a NaN far tail, as the coverage mask leaves."""
    occupancy = np.linspace(2.0, 0.5, XVALS.size)
    occupancy[-5:] = np.nan
    return {
        mode: {
            "individual": {},
            "combined": {"xvals": XVALS.copy(), "occupancy": occupancy.copy()},
        }
        for mode in CHANNEL_MAP
    }


def test_plot_lambda_ladder_profile_draws_one_panel_per_distance_measure() -> None:
    """Fitted-contribution panel plus one RMSE panel per distance measure."""
    mse: dict = {"individual": {"nucleus": 0.0, "z": 0.0}, "joint": {"joint": 0.0}}
    cv_result = CVResult(
        folds=[],
        aggregated_coefficients={
            mode: {"nucleus": (0.25, 0.05), "z": (0.25, 0.05), "joint": (0.25, 0.05)}
            for mode in RUNGS
        },
        mean_train_mse=mse,
        std_train_mse=mse,
        mean_test_mse=mse,
        std_test_mse=mse,
        n_folds=5,
        baseline_mode=BASELINE,
    )
    _, axes = plot_lambda_ladder_profile(
        cv_result=cv_result,
        occupancy_dicts={"nucleus": _occupancy_dict(), "z": _occupancy_dict()},
        channel_map=CHANNEL_MAP,
        baseline_mode=BASELINE,
    )
    assert len(axes) == 3
    assert all(ax.get_xscale() == "log" for ax in axes)
