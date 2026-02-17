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
    Combine position data from multiple seed files into a single dictionary.

    Searches for JSON files matching the prefix and rule name, extracts positions
    for the specified ingredient, and writes the combined results to a new file.

    Parameters
    ----------
    data_folder
        Directory containing the position data files
    ingredient_key
        Name of the ingredient to extract from each file
    search_prefix
        Filename prefix to filter files
    rule_name
        Packing rule name to filter files (e.g., "random", "invert")
    save_name
        Output filename for the combined dictionary

    Returns
    -------
    :
        Combined data with structure {seed_key: {ingredient_key: positions}} where
        positions is a list of [x, y, z] coordinates
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
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load positions dictionary from a JSON file and convert to NumPy arrays.

    Parameters
    ----------
    filename
        Path to JSON file containing position data
    ingredient_key
        Key for the ingredient in the raw data

    Returns
    -------
    :
        Nested dictionary with structure {cell_id: {seed: positions_array}} where
        positions_array is a NumPy array of shape (n_points, 3)

    Raises
    ------
    KeyError
        If ingredient_key is not found in the data for any cell ID
    """
    with open(filename) as j:
        raw_data = json.load(j)
    positions = {}
    for cellid_seed, seed_positions in raw_data.items():
        split_cellid = cellid_seed.split("_")
        if len(split_cellid) <= 0 or len(split_cellid) > 2:
            raise ValueError(f"Invalid cell ID format: '{cellid_seed}'")
        elif len(split_cellid) == 1:
            cellid = cellid_seed
            seed = "0"
        else:
            cellid = split_cellid[0]
            seed = split_cellid[1]

        if ingredient_key not in seed_positions:
            raise KeyError(
                f"Ingredient key '{ingredient_key}' not found in data for cell ID '{cellid}'."
            )
        if cellid not in positions:
            positions[cellid] = {}
        positions[cellid][seed] = np.array(seed_positions[ingredient_key])

    return positions


def _get_mode_file_path(
    mode: str,
    structure_id: str,
    packing_id: str,
    base_datadir: Path,
    packing_output_folder: Path | str,
) -> Path:
    """
    Determine the file path for position data based on packing mode.

    Handles both experimental microscopy data (stored in structure_data/) and
    cellPACK simulation outputs (stored in packing_output_folder/).

    Parameters
    ----------
    mode
        Packing mode or structure name from STRUCTURE_NAME_DICT
    structure_id
        Structure identifier for microscopy data
    packing_id
        Packing run identifier for simulation outputs
    base_datadir
        Base data directory path
    packing_output_folder
        Subdirectory name for packing outputs

    Returns
    -------
    :
        Full path to the positions JSON file
    """
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
    """
    Verify positions file exists or create it by combining seed files.

    For experimental data, assumes the file already exists. For cellPACK outputs,
    combines multiple seed files if the target file is missing or recalculation
    is requested.

    Parameters
    ----------
    mode_file_path
        Path where the positions file should exist
    mode
        Packing mode name
    ingredient_key
        Ingredient name to extract from raw data
    recalculate
        If True, recreate the file even if it exists
    """
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
    packing_output_folder: Path | str,
    ingredient_key: str,
    recalculate: bool,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load position data for a single packing mode.

    Ensures the positions file exists, then reads and returns the position data
    for all cells in that mode.

    Parameters
    ----------
    mode
        Packing mode or structure name
    structure_id
        Structure identifier for microscopy data
    packing_id
        Packing run identifier
    base_datadir
        Base data directory path
    packing_output_folder
        Subdirectory name for packing outputs
    ingredient_key
        Ingredient name to extract
    recalculate
        If True, regenerate combined position files

    Returns
    -------
    :
        Dictionary with structure {cell_id: {seed: positions_array}}
    """
    mode_file_path = _get_mode_file_path(
        mode, structure_id, packing_id, base_datadir, packing_output_folder
    )

    _ensure_positions_file_exists(mode_file_path, mode, ingredient_key, recalculate)

    logger.info(f"Reading positions for {mode} from {mode_file_path.relative_to(PROJECT_ROOT)}")
    positions = get_positions_dictionary_from_file(
        mode_file_path,
        ingredient_key=ingredient_key,
    )
    num_seeds = 0
    for cellid in positions:
        num_seeds += len(positions[cellid])

    logger.info(f"Read {num_seeds} seeds from {len(positions)} cell_ids for {mode}")
    return positions


def get_position_data_from_outputs(
    structure_id: str,
    packing_id: str,
    packing_modes: list[str],
    base_datadir: Path,
    results_dir: Path,
    packing_output_folder: Path | str,
    recalculate: bool = False,
    ingredient_key: str | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Retrieve and cache position data for multiple packing modes.

    Loads position data from packing outputs or cached pickle files. Caches results
    to a .dat file for faster subsequent access. Automatically combines multiple seed
    files if needed.

    Parameters
    ----------
    structure_id
        Structure identifier for microscopy data
    packing_id
        Packing run identifier, also used to construct default ingredient_key
    packing_modes
        List of mode names to load (e.g., ["random", "gradient", "observed"])
    base_datadir
        Base directory containing data and structure_data folders
    results_dir
        Directory for saving cached position data
    packing_output_folder
        Name of subdirectory containing packing outputs
    recalculate
        If True, ignore cached data and reload from source files. Default is False
    ingredient_key
        Ingredient name to extract. If None, defaults to "membrane_interior_{packing_id}"

    Returns
    -------
    :
        Nested dictionary with structure {mode: {cell_id: {seed: positions_array}}}
        where positions_array has shape (n_points, 3)
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
            mode=mode,
            structure_id=structure_id,
            packing_id=packing_id,
            base_datadir=base_datadir,
            packing_output_folder=packing_output_folder,
            ingredient_key=ingredient_key,
            recalculate=recalculate,
        )

    # Cache results
    with open(save_file_path, "wb") as f:
        pickle.dump(all_positions, f)

    return all_positions
