import json
import logging
import os
from pathlib import Path
from typing import Any


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


def add_file_handler_to_logger(logger: logging.Logger, file_path: str | Path) -> logging.Logger:
    """
    Add a file handler to a logger for writing logs to a file.

    Parameters
    ----------
    logger
        Logger instance to add the file handler to
    file_path
        Path where log messages will be written
    """
    file_handler = logging.FileHandler(file_path, mode="w")
    file_handler.setLevel(logging.INFO)
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
