"""Packing-mode order must follow the run's ``channel_map``, end to end.

``rule_interpolation`` used to re-derive its order with ``sorted()``, which left every
dict downstream of the fit alphabetical and leaked into the coefficient and AIC/BIC
panels.  Order now comes from ``channel_map`` at construction.
"""

import numpy as np
import pytest

from cellpack_analysis.lib.rule_interpolation import (
    compute_aic_comparison,
    fit_rule_interpolation,
)

BASELINE = "STRUCT_A"
CONFIG_ORDER = ["random", "nucleus_gradient", "membrane_gradient", "apical_gradient"]
NUM_BINS = 30
XVALS = np.linspace(0, 6, NUM_BINS)


def _occupancy_dict(modes: list[str], n_cells: int = 6) -> dict:
    """Minimal occupancy_dict: per-cell baseline curves plus one combined curve per mode."""
    rng = np.random.default_rng(0)
    occ: dict = {}
    for dm in ("nucleus", "z"):
        individual = {
            f"cell_{i:03d}": {
                "xvals": XVALS.copy(),
                "occupancy": rng.exponential(1.0, NUM_BINS) + 0.1,
            }
            for i in range(n_cells)
        }
        mean_occ = np.mean([v["occupancy"] for v in individual.values()], axis=0)
        occ[dm] = {
            BASELINE: {
                "individual": individual,
                "combined": {"xvals": XVALS.copy(), "occupancy": mean_occ},
            }
        }
        for mode in modes:
            occ[dm][mode] = {
                "individual": {},
                "combined": {
                    "xvals": XVALS.copy(),
                    "occupancy": rng.exponential(1.0, NUM_BINS) + 0.1,
                },
            }
    return occ


def _channel_map(modes: list[str]) -> dict[str, str]:
    return {BASELINE: "STRUCT_A", **dict.fromkeys(modes, "STRUCT_A")}


class TestFitOrder:
    def test_coefficient_keys_follow_channel_map(self) -> None:
        result = fit_rule_interpolation(
            _occupancy_dict(CONFIG_ORDER), _channel_map(CONFIG_ORDER), BASELINE
        )
        assert list(result.coefficients) == CONFIG_ORDER

    def test_reordering_channel_map_reorders_keys(self) -> None:
        reversed_modes = list(reversed(CONFIG_ORDER))
        result = fit_rule_interpolation(
            _occupancy_dict(CONFIG_ORDER), _channel_map(reversed_modes), BASELINE
        )
        assert list(result.coefficients) == reversed_modes

    def test_coefficients_invariant_to_column_order(self) -> None:
        """Permuting the design matrix must not move the NNLS solution."""
        occ = _occupancy_dict(CONFIG_ORDER)
        a = fit_rule_interpolation(occ, _channel_map(CONFIG_ORDER), BASELINE)
        b = fit_rule_interpolation(occ, _channel_map(list(reversed(CONFIG_ORDER))), BASELINE)
        for mode in CONFIG_ORDER:
            for scope, value in a.coefficients[mode].items():
                assert value == pytest.approx(b.coefficients[mode][scope], abs=1e-10)


class TestAICOrder:
    def test_model_keys_are_mixed_then_config_order_then_null(self) -> None:
        result = compute_aic_comparison(
            _occupancy_dict(CONFIG_ORDER), _channel_map(CONFIG_ORDER), BASELINE
        )
        assert list(result.akaike_weights["joint"]["joint"]) == [
            "mixed_rule",
            *[f"single:{mode}" for mode in CONFIG_ORDER],
            "null",
        ]
