import concurrent.futures
import logging
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cellpack_analysis.lib.file_io import read_json, write_json
from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure
from cellpack_analysis.lib.get_structure_stats_dataframe import (
    get_structure_stats_dataframe,
)
from cellpack_analysis.lib.mesh_tools import get_bounding_box

log = logging.getLogger(__name__)


def update_and_save_recipe(
    cellid,
    structure_name,
    recipe_template,
    rule_name,
    rule_dict,
    grid_path,
    mesh_path,
    generated_recipe_path,
    multiple_replicates,
    count=None,
    radius=None,
    get_bounding_box_from_mesh=False,
):
    updated_recipe = recipe_template.copy()

    # update recipe version
    updated_recipe["version"] = f"{rule_name}_{cellid}"

    # update seed if needed
    if not multiple_replicates and isinstance(cellid, int):
        updated_recipe["randomness_seed"] = cellid

    # update grid path
    updated_recipe["grid_file_path"] = f"{grid_path}/{cellid}_grid.dat"

    # update mesh paths
    for obj, short_name in zip(["nucleus_mesh", "membrane_mesh"], ["nuc", "mem"]):
        updated_recipe["objects"][obj]["representations"]["mesh"][
            "path"
        ] = f"{mesh_path}"
        updated_recipe["objects"][obj]["representations"]["mesh"][
            "name"
        ] = f"{short_name}_mesh_{cellid}.obj"

    # update recipe rule data
    for rule_key, rule_data in rule_dict.items():
        updated_recipe[rule_key] = rule_data
        if "gradients" in rule_key:
            updated_recipe["objects"][structure_name]["packing_mode"] = "gradient"
            gradient_keys = list(rule_data.keys())
            if len(gradient_keys) > 1:
                raise ValueError("Only one gradient is supported")
            gradient_key = gradient_keys[0]

            updated_recipe["objects"][structure_name]["gradient"] = gradient_key

    # update counts
    if count is not None:
        updated_recipe["composition"]["membrane"]["regions"]["interior"] = [
            "nucleus",
            {
                "object": f"{structure_name}",
                "count": int(count),
            },
        ]

    # update size
    if radius is not None:
        updated_recipe["objects"][structure_name]["radius"] = radius

    # update bounding box
    if get_bounding_box_from_mesh:
        bounding_box = get_bounding_box(
            mesh_path / f"mem_mesh_{cellid}.obj", expand=1.05
        ).tolist()
        # print(f"Bounding box for {structure_name}_{cellid}: {bounding_box}")
        updated_recipe["bounding_box"] = bounding_box

    # save recipe
    rule_path = f"{generated_recipe_path}/{rule_name}"
    Path(rule_path).mkdir(parents=True, exist_ok=True)
    recipe_path = f"{rule_path}/{structure_name}_{rule_name}_{cellid}.json"
    # print(f"Saving recipe to {recipe_path}")
    write_json(recipe_path, updated_recipe)

    return updated_recipe


def get_cellids(workflow_config):
    """
    Get list of cell IDs to pack for a given structure

    Parameters
    ----------
    workflow_config: type
        workflow configuration
    """
    if workflow_config.use_mean_cell:
        return ["mean"]
    else:
        return get_cellid_list_for_structure(
            structure_id=workflow_config.structure_id,
            df_cellID=None,
            dsphere=workflow_config.use_cells_in_8d_sphere,
            load_local=True,
        )


def generate_recipes(
    workflow_config,
):
    """
    Generates cellPACK recipes.
    Operations that need to get data from a dataframe are
    performed in this function.
    Other operations are performed in parallel inside the function
    update_and_save_recipe.

    Parameters
    ----------
    workflow_config: type
        workflow configuration
    """
    cellid_list = get_cellids(workflow_config)

    recipe_data = workflow_config.data.get("recipe_data", [])

    num_processes = np.min(
        [int(np.floor(0.8 * multiprocessing.cpu_count())), len(cellid_list)]
    )

    recipe_template = read_json(workflow_config.recipe_template_path)

    stats_df = get_structure_stats_dataframe()

    for data_dict in recipe_data:
        for rule_name, rule_dict in data_dict.items():
            print("Generating recipes for rule:", rule_name)
            with tqdm(total=len(cellid_list)) as pbar:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_processes
                ) as executor:
                    futures = []
                    for cellid in cellid_list:
                        # get count from cell stats
                        count = None
                        if workflow_config.get_counts_from_data:
                            count = int(stats_df.loc[cellid, "count"])

                        # get size from cell stats
                        radius = None
                        if workflow_config.get_size_from_data:
                            radius = stats_df.loc[cellid, "radius"]

                        future = executor.submit(
                            update_and_save_recipe,
                            cellid,
                            workflow_config.structure_name,
                            recipe_template,
                            rule_name,
                            rule_dict,
                            workflow_config.grid_path,
                            workflow_config.mesh_path,
                            workflow_config.generated_recipe_path,
                            workflow_config.multiple_replicates,
                            count=count,
                            radius=radius,
                            get_bounding_box_from_mesh=workflow_config.get_bounding_box_from_mesh,
                        )
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        pbar.update(1)


def generate_configs(workflow_config):
    """
    Generate cellpack config file

    Parameters
    ----------
    workflow_config: type
        workflow configuration
    """
    config_template = read_json(workflow_config.config_template_path)

    packing_info = workflow_config.data.get("packings_to_run", {})
    rule_list = packing_info.get("rules", [])

    for rule in rule_list:
        rule_config_path = (
            f"{workflow_config.generated_config_path}"
            f"/{rule}/{workflow_config.structure_name}_{rule}_config.json"
        )
        Path(rule_config_path).parent.mkdir(parents=True, exist_ok=True)

        rule_output_path = workflow_config.output_path / rule
        rule_output_path.mkdir(parents=True, exist_ok=True)

        config_template["out"] = str(rule_output_path)

        write_json(rule_config_path, config_template)
