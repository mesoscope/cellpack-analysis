"""Tests for the command line of the SLURM packing workflow."""

import sys
from pathlib import Path

import pytest

from cellpack_analysis.packing import run_packing_workflow_slurm as slurm


def _main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_packing_workflow_slurm", *args])
    slurm.main()


def test_orchestrating_needs_a_config(monkeypatch, capsys):
    for args in ([], ["--orchestrate"], ["--dry-run"]):
        with pytest.raises(SystemExit) as exc:
            _main(monkeypatch, *args)
        assert exc.value.code == 2
        assert "--workflow-config-path/-c is required" in capsys.readouterr().err


def test_orchestrating_passes_the_config(monkeypatch):
    seen = {}

    def fake(workflow_config_path: Path, **kwargs: object) -> int:
        seen["path"] = workflow_config_path
        return 0

    monkeypatch.setattr(slurm, "_run_orchestrator", fake)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "-c", "x.json")
    assert exc.value.code == 0 and seen["path"] == Path("x.json")


def test_worker_and_aggregate_need_no_config(monkeypatch):
    seen = {}
    monkeypatch.setattr(slurm, "_run_worker", lambda manifest_path, result_path: seen.update(w=1))
    _main(monkeypatch, "--worker", "--batch-manifest", "m.json", "--result-path", "r.json")
    assert seen == {"w": 1}
    monkeypatch.setattr(slurm, "aggregate_from_tracking", lambda tracking_file: 0)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "--aggregate", "tracking.json")
    assert exc.value.code == 0
