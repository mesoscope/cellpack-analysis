"""Tests for the fit -> validate config passthrough.

Any key that changes what occupancy is computed on has to survive the fit -> validate
hand-off, or the fitted coefficients are scored against different terms than they were
fitted on.  Only the x-grids are checked downstream, so a mismatch would go unnoticed.
"""

import json
from pathlib import Path

import pytest

from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw

BEHAVIOUR_KEYS = {
    "suffix": "_margin028",
    "minimum_distance_um": -0.5,
    "emd_pool_seeds": True,
    "skip_mixed_rule_validation": False,
}
PERFORMANCE_KEYS = {
    "num_workers": 16,
    "distance_backend": "process",
    "occupancy_backend": "process",
}


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    """Write a validation config into ``tmp_path`` and return it parsed."""
    out = tmp_path / "mixed_rule_validation_config_peroxisome.json"
    monkeypatch.setattr(riw, "_resolve_validation_config_path", lambda raw_config: out)

    def _write(**extra):
        raw_config = {
            "name": "rule_interpolation_test",
            "structure_id": "SLC25A17",
            "packing_id": "peroxisome",
            "channel_map": {"SLC25A17": "SLC25A17", "random": "SLC25A17"},
            **extra,
        }
        path = riw._write_validation_analysis_config(raw_config, tmp_path)
        return json.loads(Path(path).read_text())

    return _write


def test_behaviour_keys_reach_the_validation_config(write_config):
    """A key that changes what occupancy is computed on must be propagated."""
    written = write_config(**BEHAVIOUR_KEYS)
    for key, value in BEHAVIOUR_KEYS.items():
        assert written[key] == value, f"{key} was dropped by the fit -> validate hand-off"


def test_performance_keys_reach_the_validation_config(write_config):
    """The shipped validation configs carry these; a regeneration must not lose them."""
    written = write_config(**PERFORMANCE_KEYS)
    for key, value in PERFORMANCE_KEYS.items():
        assert written[key] == value, f"{key} was dropped by the fit -> validate hand-off"


def test_absent_keys_are_not_written(write_config):
    """Passthrough must be a no-op for a config that sets none of these keys.

    Writing a key the fit config never set would hand the validation phase a value
    it did not choose -- for ``emd_pool_seeds`` that is the difference between the
    module default and the shipped behaviour.
    """
    written = write_config()
    for key in {**BEHAVIOUR_KEYS, **PERFORMANCE_KEYS}:
        assert key not in written, f"{key} was invented by the generator"
