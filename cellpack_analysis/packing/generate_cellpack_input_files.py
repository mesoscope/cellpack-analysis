import concurrent.futures
import json
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure
from cellpack_analysis.lib.get_structure_stats_dataframe import (
    get_structure_stats_dataframe,
)
from cellpack_analysis.lib.mesh_tools import get_bounding_box


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
            gradient_list = []
            for gradient_key in rule_data:
                gradient_list.append(gradient_key)

            updated_recipe["objects"][structure_name]["gradient"] = gradient_list

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
    with open(recipe_path, "w") as f:
        json.dump(updated_recipe, f, indent=4)

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


def load_recipe_template(workflow_config):
    """
    Load recipe template

    Parameters
    ----------
    workflow_config: type
        workflow configuration
    """
    recipe_template_path = workflow_config.recipe_template_path

    with open(recipe_template_path, "r") as f:
        recipe_template = json.load(f)

    return recipe_template


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

    recipe_template = load_recipe_template(workflow_config)

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
                            count = stats_df.loc[cellid, ("count", "mean")]

                        # get size from cell stats
                        radius = None
                        if workflow_config.get_size_from_data:
                            radius = stats_df.loc[cellid, ("radius", "mean")]

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
