import json
import logging
import pickle
from pathlib import Path

import numpy as np

from cellpack_analysis.lib.file_io import get_project_root, read_json, write_json
from cellpack_analysis.lib.label_tables import STATIC_SHAPE_MODES, STRUCTURE_NAME_DICT

log = logging.getLogger(__name__)
PROJECT_ROOT = get_project_root()


def combine_multiple_seeds_to_dictionary(
    data_folder,
    ingredient_key="membrane_interior_peroxisome",
    search_prefix="positions_",
    rule_name="random",
    save_name="positions_peroxisome_analyze_random_mean.json",
):
    """
    Combine data from multiple seeds into a dictionary.

    Args:
    ----
        data_folder (str): Path to the folder containing the data files.
        ingredient_key (str, optional): Key of the ingredient to extract from the data.
            Defaults to "membrane_interior_peroxisome".
        search_prefix (str, optional): Prefix to search for in the data file names.
            Defaults to "positions_".
        rule_name (str, optional): Name of the rule to filter the data files.
            Defaults to "random".
        save_name (str, optional): Name of the output file.
            Defaults to "positions_peroxisome_analyze_random_mean".

    Returns:
    -------
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

        raw_data = read_json(file)

        seed = file.stem.split("_")[-1].split("_")[0]
        for seed_key, ingr_dict in raw_data.items():
            output_dict[f"{seed}_{seed_key}"] = {}
            for ingr_key, positions in ingr_dict.items():
                if ingr_key == ingredient_key:
                    output_dict[f"{seed}_{seed_key}"][ingr_key] = positions

    write_json(data_folder / save_name, output_dict)

    return output_dict


def get_positions_dictionary_from_file(
    filename, ingredient_key="membrane_interior_peroxisome", drop_random_seed=False
):
    """
    Retrieve positions dictionary from a file.

    Args:
    ----
        filename (str):
            The path to the file containing the positions data.
        ingredient_key (str, optional):
            The key for the ingredient in the raw data.
            Defaults to "membrane_interior_peroxisome".

    Returns:
    -------
        dict: A dictionary containing the positions data,
            where the keys are integers and the values are NumPy arrays.
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
    structure_name,
    packing_modes,
    base_datadir,
    results_dir,
    packing_output_folder,
    recalculate=False,
    ingredient_key=None,
):
    """
    Retrieves position data from outputs.

    Parameters
    ----------
    structure_id : str
        The ID of the structure.
    structure_name : str
        The name of the structure.
    packing_modes : list
        List of packing modes.
    base_datadir : str
        The base directory for data.
    results_dir : str
        The directory to save results.
    packing_output_folder : str
        The folder containing packing outputs.
    recalculate : bool, optional
        Whether to recalculate the position data. Default is False.
    ingredient_key : str, optional
        The key for the ingredient in the raw data. Default is None.

    Returns
    -------
    :
        A dictionary containing the position data for each packing mode.
    """
    save_file_name = f"{structure_name}_positions.dat"
    save_file_path = results_dir / save_file_name
    if not recalculate and save_file_path.exists():
        log.info(f"Loading positions from {save_file_path.relative_to(PROJECT_ROOT)}")
        with open(save_file_path, "rb") as f:
            all_positions = pickle.load(f)
        return all_positions

    if ingredient_key is None:
        ingredient_key = f"membrane_interior_{structure_name}"

    log.info("Reading position data from outputs")
    all_positions = {}

    for mode in packing_modes:
        if mode in STRUCTURE_NAME_DICT:  # if the mode is from observed microscopy data
            mode_file_path = (
                base_datadir
                / f"structure_data/{structure_id}/sample_8d/positions_{structure_id}.json"
            )
        else:  # if the mode is from cellPACK outputs
            subfolder = f"{mode}/{structure_name}/spheresSST/"

            data_folder = base_datadir / f"{packing_output_folder}/{subfolder}"

            mode_position_filename = (
                f"all_positions_{structure_name}_analyze_{mode}.json"
            )

            mode_file_path = data_folder / mode_position_filename

            if mode_file_path.exists() and not recalculate:
                positions = get_positions_dictionary_from_file(
                    mode_file_path,
                    ingredient_key=ingredient_key,
                    drop_random_seed=mode not in STATIC_SHAPE_MODES,
                )
                all_positions[mode] = positions
                continue

            rule_name = mode if mode not in STATIC_SHAPE_MODES else "random"

            combine_multiple_seeds_to_dictionary(
                data_folder,
                ingredient_key=ingredient_key,
                search_prefix="positions_",
                rule_name=rule_name,
                save_name=mode_position_filename,
            )

        log.info(
            f"Reading positions for {mode} from {mode_file_path.relative_to(PROJECT_ROOT)}"
        )
        positions = get_positions_dictionary_from_file(
            mode_file_path,
            ingredient_key=ingredient_key,
            drop_random_seed=mode not in STATIC_SHAPE_MODES,
        )

        all_positions[mode] = positions
        log.info(f"Read {len(positions)} cellids for {mode}")

    # save all positions dictionary
    with open(save_file_path, "wb") as f:
        pickle.dump(all_positions, f)

    return all_positions
