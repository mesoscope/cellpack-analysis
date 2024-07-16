import json


def read_json(file_path):
    """
    Read a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The JSON file as a dictionary.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path, data):
    """
    Write a dictionary to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        data (dict): The dictionary to write.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
