"""Tests for the run-provenance helpers in ``lib.file_io``."""

import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cellpack_analysis.lib import file_io

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
NOT_A_CHECKOUT = "unknown (not a git checkout)"
KEYS = [
    "utc",
    "argv",
    "code_version",
    "engine_commit",
    "python",
    "uv_lock_sha256",
    "config_file",
    "config_sha256",
]


def _git(path: Path, *args: str) -> None:
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.org"]
    subprocess.run(["git", *identity, *args], cwd=path, check=True, capture_output=True)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake project root (not a git checkout)."""
    path = tmp_path / "proj"
    path.mkdir()
    monkeypatch.setattr(file_io, "get_project_root", lambda: path)
    return path


def test_engine_commit(monkeypatch, caplog):
    class Dist:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return self.text

    def missing(name: str) -> None:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.delenv("CELLPACK", raising=False)
    text = json.dumps({"vcs_info": {"commit_id": "abc123"}})
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Dist(text))
    assert file_io.engine_commit() == "abc123"
    caplog.set_level(logging.WARNING, logger=file_io.__name__)
    for fake in (lambda name: Dist(None), lambda name: Dist("{}"), missing):
        monkeypatch.setattr(importlib.metadata, "distribution", fake)
        caplog.clear()
        assert file_io.engine_commit() == "unknown"
        assert caplog.records
    monkeypatch.setenv("CELLPACK", "/somewhere")
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Dist(text))
    caplog.clear()
    assert file_io.engine_commit() == "abc123"
    assert "CELLPACK is set" in caplog.text


class TestCodeVersion:
    def test_not_a_work_tree(self, root):
        assert file_io.code_version() == NOT_A_CHECKOUT

    def test_missing_root_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_io, "get_project_root", lambda: tmp_path / "nope")
        assert file_io.code_version() == NOT_A_CHECKOUT

    def test_git_absent(self, root, monkeypatch):
        def no_git(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", no_git)
        assert file_io.code_version() == NOT_A_CHECKOUT

    @needs_git
    def test_describe_of_a_checkout(self, root):
        _git(root, "init", "-q")
        (root / "a.txt").write_text("a")
        _git(root, "add", "a.txt")
        _git(root, "commit", "-q", "-m", "first")
        _git(root, "tag", "v1.0")
        assert file_io.code_version() == "v1.0"
        (root / "a.txt").write_text("b")
        assert file_io.code_version() == "v1.0-dirty"
        _git(root, "commit", "-q", "-am", "second")
        version = file_io.code_version()
        assert version.startswith("v1.0-1-g") and not version.endswith("-dirty")

    @needs_git
    def test_untagged_checkout_gives_the_hash(self, root):
        _git(root, "init", "-q")
        (root / "a.txt").write_text("a")
        _git(root, "add", "a.txt")
        _git(root, "commit", "-q", "-m", "first")
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        assert file_io.code_version() == head

    @needs_git
    def test_a_root_inside_another_repository_is_not_described(self, tmp_path, monkeypatch):
        _git(tmp_path, "init", "-q")
        (tmp_path / "a.txt").write_text("a")
        _git(tmp_path, "add", "a.txt")
        _git(tmp_path, "commit", "-q", "-m", "first")
        inner = tmp_path / "inner"
        inner.mkdir()
        monkeypatch.setattr(file_io, "get_project_root", lambda: inner)
        assert file_io.code_version() == NOT_A_CHECKOUT


class TestWriteRunProvenance:
    @staticmethod
    def _lines(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_two_calls_append_two_lines(self, root, monkeypatch):
        monkeypatch.setattr(file_io, "engine_commit", lambda: "abc123")
        config = root / "configs" / "c.json"
        config.parent.mkdir()
        config.write_text('{"a": 1}')
        logs = root / "results" / "x" / "logs"
        path = file_io.write_run_provenance(logs, config)
        again = file_io.write_run_provenance(logs, config, {"phase": "fit"})
        assert path == again == logs / "run_provenance.jsonl"
        first, second = self._lines(path)
        assert list(first) == KEYS
        assert list(second) == [*KEYS, "phase"] and second["phase"] == "fit"
        assert "phase" not in first

    def test_values(self, root, monkeypatch):
        monkeypatch.setattr(file_io, "engine_commit", lambda: "abc123")
        monkeypatch.setattr(sys, "argv", ["prog", "-c", "x.json"])
        config = root / "configs" / "c.json"
        config.parent.mkdir()
        config.write_text('{"a": 1}')
        (root / "uv.lock").write_text("lock")
        (line,) = self._lines(file_io.write_run_provenance(root / "logs", config))
        assert line["utc"].endswith("+00:00") and line["utc"][:4].isdigit()
        assert line["argv"] == ["prog", "-c", "x.json"]
        assert line["code_version"] == NOT_A_CHECKOUT
        assert line["engine_commit"] == "abc123"
        assert line["python"] == platform.python_version()
        assert line["uv_lock_sha256"] == hashlib.sha256(b"lock").hexdigest()
        assert line["config_file"] == "configs/c.json"
        assert line["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()

    def test_config_outside_the_root_is_absolute(self, root, tmp_path):
        config = tmp_path / "elsewhere.json"
        config.write_text("{}")
        (line,) = self._lines(file_io.write_run_provenance(root / "logs", config))
        assert line["config_file"] == config.as_posix()

    def test_symlinked_config_stays_relative(self, root, tmp_path):
        target = tmp_path / "target.json"
        target.write_text("{}")
        link = root / "link.json"
        link.symlink_to(target)
        (line,) = self._lines(file_io.write_run_provenance(root / "logs", link))
        assert line["config_file"] == "link.json"

    def test_no_config_and_no_lock(self, root):
        (line,) = self._lines(file_io.write_run_provenance(root / "logs", None))
        assert line["config_file"] is None and line["config_sha256"] is None
        assert line["uv_lock_sha256"] is None

    def test_extra_never_overrides_a_core_key(self, root):
        (line,) = self._lines(
            file_io.write_run_provenance(root / "logs", None, {"python": "0", "n": 3})
        )
        assert line["python"] == platform.python_version() and line["n"] == 3
