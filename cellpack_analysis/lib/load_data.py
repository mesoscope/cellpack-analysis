import json
import logging
import pickle
from pathlib import Path

import numpy as np

from cellpack_analysis.lib.file_io import get_project_root, read_json, write_json
from cellpack_analysis.lib.label_tables import STATIC_SHAPE_MODES, STRUCTURE_NAME_DICT

logger = logging.getLogger(__name__)
PROJECT_ROOT = get_project_root()


def combine_multiple_seeds_to_dictionary(
    data_folder: str | Path,
    ingredient_key: str = "membrane_interior_peroxisome",
    search_prefix: str = "positions_",
    rule_name: str = "random",
    save_name: str = "positions_peroxisome_analyze_random_mean.json",
) -> dict[str, dict[str, list[list[float]]]]:
    """
    Combine data from multiple seeds into a dictionary.

    Parameters
    ----------
    data_folder
        Path to the folder containing the data files
    ingredient_key
        Key of the ingredient to extract from the data
    search_prefix
        Prefix to search for in the data file names
    rule_name
        Name of the rule to filter the data files
    save_name
        Name of the output file

    Returns
    -------
    :
        Dictionary containing the combined data from multiple seeds with structure
        {seed_key: {ingredient_key: positions}}
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
    filename: str | Path,
    ingredient_key: str = "membrane_interior_peroxisome",
    drop_random_seed: bool = False,
) -> dict[str, np.ndarray]:
    """
    Retrieve positions dictionary from a file.

    Parameters
    ----------
    filename
        Path to the file containing the positions data
    ingredient_key
        Key for the ingredient in the raw data
    drop_random_seed
        Whether to remove random seed suffix from keys

    Returns
    -------
    :
        Dictionary containing positions data where keys are cell IDs and values are NumPy arrays
    """
    with open(filename) as j:
        raw_data = json.load(j)
    if drop_random_seed:
        positions = {k.split("_")[0]: np.array(v[ingredient_key]) for k, v in raw_data.items()}
    else:
        positions = {k: np.array(v[ingredient_key]) for k, v in raw_data.items()}

    return positions


def _get_mode_file_path(
    mode: str,
    structure_id: str,
    packing_id: str,
    base_datadir: Path,
    packing_output_folder: str,
) -> Path:
    """Get the file path for a specific packing mode."""
    if mode in STRUCTURE_NAME_DICT:  # observed microscopy data
        return (
            base_datadir / f"structure_data/{structure_id}/sample_8d/positions_{structure_id}.json"
        )
    else:  # cellPACK outputs
        subfolder = f"{mode}/{packing_id}/spheresSST/"
        data_folder = base_datadir / f"{packing_output_folder}/{subfolder}"
        mode_position_filename = f"all_positions_{packing_id}_analyze_{mode}.json"
        return data_folder / mode_position_filename


def _ensure_positions_file_exists(
    mode_file_path: Path, mode: str, ingredient_key: str, recalculate: bool
) -> None:
    """Ensure the positions file exists, creating it if necessary."""
    if mode_file_path.exists() and not recalculate:
        return

    if mode in STRUCTURE_NAME_DICT:
        return  # microscopy data files should already exist

    # Create combined file for cellPACK outputs
    data_folder = mode_file_path.parent
    rule_name = mode if mode not in STATIC_SHAPE_MODES else "random"
    combine_multiple_seeds_to_dictionary(
        data_folder,
        ingredient_key=ingredient_key,
        search_prefix="positions_",
        rule_name=rule_name,
        save_name=mode_file_path.name,
    )


def _load_positions_for_mode(
    mode: str,
    structure_id: str,
    packing_id: str,
    base_datadir: Path,
    packing_output_folder: str,
    ingredient_key: str,
    recalculate: bool,
) -> dict[str, np.ndarray]:
    """Load positions for a single packing mode."""
    mode_file_path = _get_mode_file_path(
        mode, structure_id, packing_id, base_datadir, packing_output_folder
    )

    _ensure_positions_file_exists(mode_file_path, mode, ingredient_key, recalculate)

    logger.info(f"Reading positions for {mode} from {mode_file_path.relative_to(PROJECT_ROOT)}")
    positions = get_positions_dictionary_from_file(
        mode_file_path,
        ingredient_key=ingredient_key,
        drop_random_seed=mode not in STATIC_SHAPE_MODES,
    )

    logger.info(f"Read {len(positions)} cell_ids for {mode}")
    return positions


def get_position_data_from_outputs(
    structure_id: str,
    packing_id: str,
    packing_modes: list[str],
    base_datadir: Path,
    results_dir: Path,
    packing_output_folder: str,
    recalculate: bool = False,
    ingredient_key: str | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Retrieve position data from packing outputs.

    Parameters
    ----------
    structure_id
        ID of the structure
    packing_id
        ID of the packing
    packing_modes
        List of packing modes to retrieve data for
    base_datadir
        Base directory for data
    results_dir
        Directory to save results
    packing_output_folder
        Folder containing packing outputs
    recalculate
        Whether to recalculate the position data
    ingredient_key
        Key for the ingredient in the raw data, defaults to membrane_interior_{packing_id}

    Returns
    -------
    :
        Dictionary containing position data for each packing mode with structure
        {mode: {cell_id: positions_array}}
    """
    save_file_path = results_dir / f"{packing_id}_positions.dat"

    # Load cached results if available
    if not recalculate and save_file_path.exists():
        logger.info(f"Loading positions from {save_file_path.relative_to(PROJECT_ROOT)}")
        with open(save_file_path, "rb") as f:
            all_positions = pickle.load(f)
            if set(all_positions.keys()) == set(packing_modes):
                return all_positions
            else:
                logger.warning(
                    f"Cached data in {save_file_path} does not match requested packing modes.\n"
                    f"Recalculating positions."
                )

    ingredient_key = ingredient_key or f"membrane_interior_{packing_id}"
    logger.info("Reading position data from outputs")

    all_positions = {}
    for mode in packing_modes:
        all_positions[mode] = _load_positions_for_mode(
            mode,
            structure_id,
            packing_id,
            base_datadir,
            packing_output_folder,
            ingredient_key,
            recalculate,
        )

    # Cache results
    with open(save_file_path, "wb") as f:
        pickle.dump(all_positions, f)

    return all_positions
