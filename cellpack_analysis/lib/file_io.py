import json
from pathlib import Path


def get_datadir_path() -> Path:
    datadir = Path(__file__).parents[2] / "data"
    datadir.mkdir(exist_ok=True, parents=True)
    return datadir


def get_results_path() -> Path:
    results_path = Path(__file__).parents[2] / "results"
    results_path.mkdir(exist_ok=True, parents=True)
    return results_path


def get_project_root() -> Path:
    project_root = Path(__file__).parents[2]
    return project_root


def read_json(file_path):
    """
    Read a JSON file.

    Args:
    ----
        file_path (str): The path to the JSON file.

    Returns:
    -------
        dict: The JSON file as a dictionary.
    """
    with open(file_path) as f:
        return json.load(f)


def write_json(file_path, data):
    """
    Write a dictionary to a JSON file.

    Args:
    ----
        file_path (str): The path to the JSON file.
        data (dict): The dictionary to write.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
