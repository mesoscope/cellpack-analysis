import json
from pathlib import Path
import numpy as np
import pickle

import pandas as pd


from cellpack_analysis.analyses.stochastic_variation_analysis.label_tables import (
    STRUCTURE_NAME_DICT,
    VARIABLE_SHAPE_MODES,
)


def combine_multiple_seeds_to_dictionary(
    data_folder,
    ingredient_key="membrane_interior_peroxisome",
    search_prefix="positions_",
    rule_name="random",
    save_name="positions_peroxisome_analyze_random_mean",
):
    """
    Combine data from multiple seeds into a dictionary.

    Args:
        data_folder (str): Path to the folder containing the data files.
        ingredient_key (str, optional): Key of the ingredient to extract from the data. Defaults to "membrane_interior_peroxisome".
        search_prefix (str, optional): Prefix to search for in the data file names. Defaults to "positions_".
        rule_name (str, optional): Name of the rule to filter the data files. Defaults to "random".
        save_name (str, optional): Name of the output file. Defaults to "positions_peroxisome_analyze_random_mean".

    Returns:
        dict: A dictionary containing the combined data from multiple seeds.
    """

    data_folder = Path(data_folder)
    output_dict = {}
    for file in data_folder.glob(f"{search_prefix}*.json"):
        if rule_name not in file.name:
            continue
        if save_name in file.name:
            continue
        if ("invert" not in rule_name) and ("invert" in file.name):
            continue

        seed = file.stem.split("_")[-1].split("_")[0]
        with open(file) as j:
            raw_data = json.load(j)
        for seed_key, ingr_dict in raw_data.items():
            output_dict[f"{seed}_{seed_key}"] = {}
            for ingr_key, positions in ingr_dict.items():
                if ingr_key == ingredient_key:
                    output_dict[f"{seed}_{seed_key}"][ingr_key] = positions

    with open(data_folder / f"{save_name}.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return output_dict


def get_positions_dictionary_from_file(
    filename, ingredient_key="membrane_interior_peroxisome", drop_random_seed=False
):
    """
    Retrieve positions dictionary from a file.

    Args:
        filename (str): The path to the file containing the positions data.
        ingredient_key (str, optional): The key for the ingredient in the raw data. Defaults to "membrane_interior_peroxisome".

    Returns:
        dict: A dictionary containing the positions data, where the keys are integers and the values are NumPy arrays.
    """
    with open(filename) as j:
        raw_data = json.load(j)
    if drop_random_seed:
        positions = {
            k.split("_")[0]: np.array(v[ingredient_key]) for k, v in raw_data.items()
        }
    else:
        positions = {k: np.array(v[ingredient_key]) for k, v in raw_data.items()}

    return positions


def get_position_data_from_outputs(
    structure_id,
    packing_modes,
    base_datadir,
    results_dir,
    recalculate=False,
    baseline_analysis=False,
):
    """
    Retrieves position data from outputs.

    Args:
        structure_id (str): The ID of the structure.
        packing_modes (list): List of packing modes.
        base_datadir (str): The base directory for data.
        results_dir (str): The directory to save results.
        recalculate (bool, optional): Whether to recalculate the position data. Defaults to False.

    Returns:
        dict: A dictionary containing the position data for each packing mode.
    """
    if not recalculate:
        file_path = results_dir / "packing_modes_positions.dat"
        if file_path.exists():
            with open(file_path, "rb") as f:
                all_positions = pickle.load(f)
            return all_positions

    print("Reading position data from outputs")
    all_positions = {}

    for mode in packing_modes:
        if mode == "observed_data":
            file_path = (
                base_datadir
                / f"structure_data/{structure_id}/sample_8d/positions_{structure_id}.json"
            )
        else:
            if baseline_analysis:
                folder_str = f"packing_outputs/stochastic_variation_analysis/{mode}/{STRUCTURE_NAME_DICT[structure_id]}/spheresSST/"
            else:
                folder_str = f"packing_outputs/8d_sphere_data/RS/{STRUCTURE_NAME_DICT[structure_id]}/spheresSST/"
            data_folder = base_datadir / folder_str
            file_path = (
                data_folder
                / f"all_positions_{STRUCTURE_NAME_DICT[structure_id]}_analyze_{mode}.json"
            )

            if file_path.exists() and not recalculate:
                positions = get_positions_dictionary_from_file(file_path)
                all_positions[mode] = positions
                continue

            rule_name = mode if not baseline_analysis else "random"

            combine_multiple_seeds_to_dictionary(
                data_folder,
                ingredient_key=f"membrane_interior_{STRUCTURE_NAME_DICT[structure_id]}",
                search_prefix="positions_",
                rule_name=rule_name,
                save_name=f"all_positions_{STRUCTURE_NAME_DICT[structure_id]}_analyze_{mode}",
            )

        positions = get_positions_dictionary_from_file(
            file_path,
            drop_random_seed=mode in VARIABLE_SHAPE_MODES,
        )

        all_positions[mode] = positions
        print(f"Read {len(positions)} packings for {mode}")
    # save all positions dictionary
    file_path = results_dir / "packing_modes_positions.dat"
    with open(file_path, "wb") as f:
        pickle.dump(all_positions, f)

    return all_positions


def get_cellid_list(structure_id, filter_8d=False):
    """
    Get a list of cell IDs for a given structure ID.

    Parameters:
        structure_id (int): The ID of the structure.
        filter_8d (bool, optional): Whether to filter the cell IDs based on 8D criteria. Default is False.

    Returns:
        list: A list of cell IDs.
    """
    df = pd.read_csv("s3://cellpack-analysis-data/all_cellids.csv")
    df = df[df["structure_name"] == structure_id]
    if filter_8d:
        df = df[df["8dsphere"]]
    return df["CellId"].tolist()
