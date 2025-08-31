import gc
import importlib.util
import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from cellpack_analysis.lib.file_io import read_json
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.io import format_time

log = logging.getLogger(__name__)

# set cellPACK path
load_dotenv()
CELLPACK_PATH = os.getenv("CELLPACK")
if CELLPACK_PATH is None:
    spec = importlib.util.find_spec("cellpack")
    if spec is None:
        raise Exception("cellPACK not found")
    CELLPACK_PATH = spec.submodule_search_locations[0]  # type: ignore

PACK_PATH: str = CELLPACK_PATH + "/bin/pack.py"  # type: ignore
assert os.path.exists(PACK_PATH), f"PACK path {PACK_PATH} does not exist"
log.debug(f"Using cellPACK at {PACK_PATH}")


def check_recipe_completed(recipe_path, config_data, workflow_config):
    """
    Check if a recipe has completed packing.
    Also checks for single recipes that pack multiple outputs.
    """
    recipe_data = read_json(recipe_path)
    number_of_packings = config_data.get("number_of_packings", 1)
    seed_vals = recipe_data.get("randomness_seed", [0])
    if isinstance(seed_vals, int):
        seed_vals = [seed_vals]

    base_folder = Path(config_data["out"]) / f"{workflow_config.structure_name}/spheresSST"

    if workflow_config.result_type == "image":
        folder_to_check = base_folder / "figures"
        prefix = "voxelized_image"
        suffix = "ome.tiff"
    elif workflow_config.result_type == "simularium":
        folder_to_check = base_folder
        prefix = "results"
        suffix = "simularium"
    else:
        raise ValueError("check_type must be 'image' or 'simularium'")

    if len(seed_vals) == number_of_packings:
        # usually this means only packing one seed
        # check whether all files exist explicitly
        result_file_list = [
            folder_to_check
            / f"{prefix}_{recipe_data['name']}_{config_data['name']}_{recipe_data['version']}_seed_{seed_val}.{suffix}"
            for seed_val in seed_vals
        ]
    else:
        result_file_list = list(
            folder_to_check.glob(
                f"{prefix}_{recipe_data['name']}_{config_data['name']}_"
                f"{recipe_data['version']}_seed_*.{suffix}"
            )
        )
    num_existing_files = sum([result_file.exists() for result_file in result_file_list])

    return num_existing_files == number_of_packings


def log_update(count, skipped_count, failed_count, start, num_files, rule_start):
    """
    Logs the update of the packing process.

    Args:
    ----
        count (int): The number of completed runs.
        skipped_count (int): The number of skipped runs.
        failed_count (int): The number of failed runs.
        start (float): The start time of the packing process.
        num_files (int): The total number of files to process.
        rule_start (float): The start time of the current rule.
    """
    done = count + skipped_count
    remaining = num_files - done - failed_count
    log.info(
        f"Completed: {count}, Failed: {failed_count}, Skipped: {skipped_count}, "
        f"Total: {num_files}, Done: {done}, Remaining: {remaining}"
    )
    rule_time = time.time() - rule_start
    total_time = time.time() - start
    per_count = np.inf
    time_left = np.inf
    if count > 0:
        per_count = rule_time / count
        time_left = per_count * remaining

    log.info(
        f"Total time: {format_time(total_time)}, Rule time: {format_time(rule_time)}, "
        f"Time per run: {format_time(per_count)}, "
        f"Estimated time left for rule: {format_time(time_left)}",
    )


def get_cell_ids_to_pack(workflow_config):
    """
    Get list of cell IDs to pack for a structure.

    Parameters
    ----------
    workflow_config: type
        workflow configuration
    """
    if workflow_config.use_mean_cell:
        return ["mean"]
    else:
        # get list of all cell_ids
        all_cell_ids = get_cell_id_list_for_structure(
            structure_id=workflow_config.structure_id,
            df_cell_id=None,
            dsphere=workflow_config.use_cells_in_8d_sphere,
            load_local=True,
        )

        # get packing information from config
        packing_info = workflow_config.data.get("packings_to_run", {})

        # check if cell_ids to run are specified
        cell_ids = packing_info.get("cell_ids", all_cell_ids)

        # check if number of packings is specified
        number_of_packings = packing_info.get("number_of_packings")
        if number_of_packings:
            cell_ids = cell_ids[:number_of_packings]

        return cell_ids


def get_input_file_dictionary(workflow_config):
    """
    Get the input file dictionary containing the configuration path and recipe paths for each rule.

    Args:
    ----
        workflow_config: The workflow configuration object.

    Returns:
    -------
        input_file_dict:
            The dictionary containing the configuration path and recipe paths for each rule.
    """
    packing_info = workflow_config.data.get("packings_to_run", {})
    rule_list = packing_info.get("rules", [])

    cell_ids_to_pack = get_cell_ids_to_pack(workflow_config)

    input_file_dict = {}
    for rule in rule_list:
        input_file_dict[rule] = {}
        rule_config_path = (
            f"{workflow_config.generated_config_path}"
            f"/{rule}/{workflow_config.structure_name}_{rule}_config.json"
        )
        input_file_dict[rule]["config_path"] = rule_config_path

        rule_recipe_folder = Path(f"{workflow_config.generated_recipe_path}/{rule}")

        rule_recipe_list = []
        for cell_id in cell_ids_to_pack:
            cell_id_recipe_path = (
                rule_recipe_folder / f"{workflow_config.structure_name}_{rule}_{cell_id}.json"
            )
            if cell_id_recipe_path.exists():
                rule_recipe_list.append(cell_id_recipe_path)
        log.info(f"Found {len(rule_recipe_list)} recipes for rule {rule}")
        input_file_dict[rule]["recipe_paths"] = rule_recipe_list

    return input_file_dict


def run_single_packing(
    recipe_path,
    config_path,
):
    try:
        log.debug(f"Running {recipe_path}")
        result = subprocess.run(
            [
                "python",
                PACK_PATH,
                "-r",
                recipe_path,
                "-c",
                config_path,
            ],
            check=True,
        )
        return result.returncode == 0
    except Exception as e:
        log.error(f"Error: {e}")
        return False


def pack_recipes(workflow_config):
    """
    Pack recipes using cellPACK.

    Parameters
    ----------
    workflow_config: type
        workflow configuration
    """
    input_file_dict = get_input_file_dictionary(workflow_config)
    start = time.time()

    log_folder = workflow_config.output_path / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    total_count = total_failed_count = 0
    for rule, input_files in input_file_dict.items():
        rule_start = time.time()
        skipped_count = count = failed_count = 0
        log.info(f"Packing rule: {rule}")

        config_path = input_files["config_path"]
        config_data = read_json(config_path)

        num_files = len(input_files["recipe_paths"])

        futures = []
        with ProcessPoolExecutor(max_workers=workflow_config.num_processes) as executor:
            for recipe_path in input_files["recipe_paths"]:

                # check if recipe can be skipped
                if workflow_config.skip_completed and check_recipe_completed(
                    recipe_path, config_data, workflow_config
                ):
                    skipped_count += 1
                    log.debug(
                        f"Skipping packing for completed recipe {recipe_path}. "
                        f"{skipped_count} skipped"
                    )
                    continue

                futures.append(
                    executor.submit(
                        run_single_packing,
                        recipe_path,
                        config_path,
                    )
                )
                log.debug(f"Submitted packing for {recipe_path}")

            log.info(f"Submitted {len(futures)} packings for rule {rule}. {skipped_count} skipped")
            for future in as_completed(futures):
                if future.result():
                    count += 1
                else:
                    failed_count += 1
                    log.info(f"Failed packing for {future}")
                log_update(
                    count,
                    skipped_count,
                    failed_count,
                    start,
                    num_files,
                    rule_start,
                )
                gc.collect()
        total_count += count
        total_failed_count += failed_count
    log.info(f"Packing complete. Total time: {format_time(time.time() - start)}")
    log.info(f"Total count: {total_count}, Total failed: {total_failed_count}")
