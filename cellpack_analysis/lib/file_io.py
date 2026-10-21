import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROVENANCE_FILE = "run_provenance.jsonl"
NOT_A_CHECKOUT = "unknown (not a git checkout)"


def get_datadir_path() -> Path:
    """
    Get the data directory path and create it if it doesn't exist.

    Returns
    -------
    :
        Path to the data directory
    """
    datadir = Path(__file__).parents[2] / "data"
    datadir.mkdir(exist_ok=True, parents=True)
    return datadir


def get_results_path() -> Path:
    """
    Get the results directory path and create it if it doesn't exist.

    Returns
    -------
    :
        Path to the results directory
    """
    results_path = Path(__file__).parents[2] / "results"
    results_path.mkdir(exist_ok=True, parents=True)
    return results_path


def get_project_root() -> Path:
    """
    Get the project root directory path.

    Returns
    -------
    :
        Path to the project root directory
    """
    project_root = Path(__file__).parents[2]
    return project_root


def read_json(file_path: str | Path) -> dict[str, Any]:
    """
    Read a JSON file and return its contents.

    Parameters
    ----------
    file_path
        Path to the JSON file to read

    Returns
    -------
    :
        Dictionary containing the JSON file contents
    """
    with open(file_path) as f:
        return json.load(f)


def write_json(file_path: str | Path, data: dict[str, Any]) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    file_path
        Path where the JSON file will be written
    data
        Dictionary data to write to the JSON file
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    os.chmod(file_path, 0o644)


def add_file_handler_to_logger(
    logger: logging.Logger, file_path: str | Path, level: int = logging.INFO
) -> logging.Logger:
    """
    Add a file handler to a logger for writing logs to a file.

    Parameters
    ----------
    logger
        Logger instance to add the file handler to
    file_path
        Path where log messages will be written
    level
        Logging level for the file handler
    """
    file_handler = logging.FileHandler(file_path, mode="w")
    file_handler.setLevel(level)

    # Try to use the formatter from an existing handler (e.g., from logging.conf)
    formatter = None
    for h in logging.getLogger().handlers:
        if hasattr(h, "formatter") and h.formatter is not None:
            formatter = h.formatter
            break
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def remove_file_handler_from_logger(
    logger: logging.Logger, file_path: str | Path | None = None
) -> logging.Logger:
    """
    Remove a file handler from a logger.

    Parameters
    ----------
    logger
        Logger instance to remove the file handler from
    file_path
        Path of the log file associated with the handler to remove
    """
    if file_path is None:
        return logger

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(file_path):
            logger.removeHandler(handler)
            handler.close()
            break

    return logger


def setup_workflow_logging(log_file_path: str | Path) -> Path:
    """
    Set up integrated debug logging for workflow scripts.

    Adds a DEBUG-level file handler to the root logger to capture all debug messages
    from all modules into a single log file.

    Parameters
    ----------
    log_file_path
        Path where the debug log file will be written

    Returns
    -------
    :
        Path object of the log file
    """
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    # Ensure root logger is at DEBUG level to allow DEBUG messages through
    root_logger.setLevel(logging.DEBUG)

    add_file_handler_to_logger(root_logger, log_path, level=logging.DEBUG)

    return log_path


def engine_commit() -> str:
    """
    Return the engine commit recorded in the installed cellpack ``direct_url.json``.

    Returns
    -------
    :
        The commit id, or ``"unknown"`` (with a warning) if it is not recorded.
    """
    if os.environ.get("CELLPACK"):
        logger.warning("CELLPACK is set; packing would not use the installed engine")
    try:
        text = importlib.metadata.distribution("cellpack").read_text("direct_url.json")
        commit = json.loads(text or "{}").get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, ValueError) as e:
        logger.warning("Could not read the cellpack engine commit: %s", e)
        return "unknown"
    if not commit:
        logger.warning("cellpack direct_url.json records no commit_id")
        return "unknown"
    return str(commit)


def code_version() -> str:
    """
    Return ``git describe --always --dirty --tags`` for the project root.

    Git must find the repository at the project root itself, so a checkout that merely
    contains the project does not describe the outer repository.

    Returns
    -------
    :
        The description, or ``"unknown (not a git checkout)"`` if git is absent or the
        project root is not a work tree. Never raises.
    """
    root = get_project_root().resolve()
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(root.parent)}
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return NOT_A_CHECKOUT
    out = result.stdout.strip()
    if result.returncode != 0 or not out:
        return NOT_A_CHECKOUT
    return out


def _sha256(path: Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or ``None`` if it is not a file."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_to_root(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when inside it, else its absolute path (posix)."""
    path = Path(os.path.abspath(path))
    for base, target in ((Path(os.path.abspath(root)), path), (root.resolve(), path.resolve())):
        try:
            return target.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def write_run_provenance(
    logs_dir: Path, config_file: Path | None, extra: dict[str, Any] | None = None
) -> Path:
    """
    Append one JSON line recording which code and engine produced a run.

    Parameters
    ----------
    logs_dir
        Directory for ``run_provenance.jsonl``; created if missing
    config_file
        Config that drives the run, or ``None``
    extra
        Further keys, merged at the top level; they never override the keys below

    Returns
    -------
    :
        Path of ``run_provenance.jsonl``.

    Notes
    -----
    The line has ``utc``, ``argv``, ``code_version``, ``engine_commit``, ``python``,
    ``uv_lock_sha256`` (``null`` if the project root has no ``uv.lock``), ``config_file``
    (relative to the project root when inside it) and ``config_sha256`` (``null`` without a
    readable config file).
    """
    root = get_project_root()
    record: dict[str, Any] = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "code_version": code_version(),
        "engine_commit": engine_commit(),
        "python": platform.python_version(),
        "uv_lock_sha256": _sha256(root / "uv.lock"),
        "config_file": None if config_file is None else _relative_to_root(Path(config_file), root),
        "config_sha256": None if config_file is None else _sha256(Path(config_file)),
    }
    for key, value in (extra or {}).items():
        record.setdefault(key, value)
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / PROVENANCE_FILE
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return path
