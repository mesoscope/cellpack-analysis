import concurrent.futures
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from cellpack_analysis.lib.file_io import read_json, write_json
from cellpack_analysis.lib.get_cell_id_list import sample_cell_ids_for_structure
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.mesh_tools import get_bounding_box
from cellpack_analysis.packing import rule_repository

logger = logging.getLogger(__name__)


def set_gradient_mode_center(
    mode_settings: dict[str, Any], bounding_box: list[list[float]]
) -> dict[str, Any]:
    """
    Set gradient mode center based on mode settings and bounding box.

    Parameters
    ----------
    mode_settings
        Mode settings dictionary
    bounding_box
        Bounding box for the packing

    Returns
    -------
    :
        Updated mode settings dictionary with center position
    """
    center = mode_settings.get("center")
    bounding_box_np = np.array(bounding_box)

    if center is not None:
        if center == "center":
            center_position = bounding_box_np.mean(axis=0).tolist()
        elif center == "random":
            center_position = np.random.uniform(
                low=bounding_box_np[0], high=bounding_box_np[1]
            ).tolist()
        elif center == "max":
            direction = mode_settings.get("direction", [0, 0, 0])
            center_position = bounding_box_np.mean(axis=0)
            for i in range(3):
                if direction[i]:
                    center_position[i] = np.max(bounding_box_np[:, i])
                else:
                    center_position[i] = np.mean(bounding_box_np[:, i])
            center_position = center_position.tolist()
        elif center == "min":
            direction = mode_settings.get("direction", [0, 0, 0])
            center_position = bounding_box_np.mean(axis=0)
            for i in range(3):
                if direction[i]:
                    center_position[i] = np.max(bounding_box_np[:, i])
                else:
                    center_position[i] = np.mean(bounding_box_np[:, i])
            center_position = center_position.tolist()
        else:
            center_position = center
    else:
        center_position = bounding_box_np.mean(axis=0).tolist()

    mode_settings["center"] = center_position

    return mode_settings


def resolve_gradient_names(gradient_list: list[str]) -> dict[str, Any]:
    """
    Resolve gradient names from gradient list.

    Parameters
    ----------
    gradient_list
        Gradient list

    Returns
    -------
    :
        Dictionary mapping gradient names to their definitions
    """
    resolved_gradient_dict = {}
    for gradient in gradient_list:
        if gradient in rule_repository.GRADIENTS:
            resolved_gradient_dict[gradient] = rule_repository.GRADIENTS[gradient]
        else:
            logger.error(f"Gradient {gradient} not found in GRADIENTS")

    return resolved_gradient_dict


def process_gradient_data(
    recipe_entry: list[str] | dict[str, Any],
    recipe: dict[str, Any],
    gradient_structure_name: str,
) -> dict[str, Any]:
    """
    Process gradient data for recipe.

    Parameters
    ----------
    recipe_entry
        Recipe entry containing gradient information
    recipe
        Recipe dictionary to update
    gradient_structure_name
        Name of the structure to apply gradient to

    Returns
    -------
    :
        Updated recipe with gradient data processed
    """
    bounding_box = recipe["bounding_box"]

    # resolve gradient names
    if isinstance(recipe_entry, list):
        recipe_entry = resolve_gradient_names(recipe_entry)

    for _, gradient_dict in recipe_entry.items():
        if gradient_dict["mode"] == "vector":
            gradient_dict["mode_settings"] = set_gradient_mode_center(
                gradient_dict["mode_settings"], bounding_box
            )

    recipe["gradients"] = recipe_entry

    # set packing mode of structure to gradient
    recipe["objects"][gradient_structure_name]["packing_mode"] = "gradient"

    # set the gradients required for the structure
    gradient_keys = list(recipe_entry.keys())
    if len(gradient_keys) == 1:
        gradient_keys = gradient_keys[0]

    recipe["objects"][gradient_structure_name]["gradient"] = gradient_keys

    return recipe


def process_rule_dict(
    updated_recipe: dict[str, Any], rule_dict: dict[str, Any], gradient_structure_name: str
) -> dict[str, Any]:
    """
    Process rule dictionary and update recipe.

    Parameters
    ----------
    updated_recipe
        Updated recipe dictionary
    rule_dict
        Rule dictionary to process
    gradient_structure_name
        Name of the structure to apply gradient to

    Returns
    -------
    :
        Updated recipe with rule data processed
    """
    for recipe_key, recipe_entry in rule_dict.items():

        if recipe_key == "gradients":
            updated_recipe = process_gradient_data(
                recipe_entry, updated_recipe, gradient_structure_name
            )
        if recipe_key == "gradient_weights":
            updated_recipe["objects"][gradient_structure_name]["gradient_weights"] = recipe_entry

    return updated_recipe


def update_and_save_recipe(
    cell_id: int | str,
    structure_name: str,
    recipe_template: dict[str, Any],
    rule_name: str,
    rule_dict: dict[str, Any],
    grid_path: str | Path,
    mesh_path: str | Path,
    generated_recipe_path: str | Path,
    multiple_replicates: bool,
    count: int | None = None,
    radius: float | None = None,
    get_bounding_box_from_mesh: bool = False,
    use_additional_struct: bool = False,
    gradient_structure_name: str | None = None,
) -> dict[str, Any]:
    """
    Update the recipe template with provided parameters and save the updated recipe to a JSON file.

    Parameters
    ----------
    cell_id
        The ID of the cell
    structure_name
        The name of the structure
    recipe_template
        The template recipe to be updated
    rule_name
        The name of the rule
    rule_dict
        The rule dictionary
    grid_path
        The path to the grid file
    mesh_path
        The path to the mesh file
    generated_recipe_path
        The path to save the generated recipe
    multiple_replicates
        Indicates whether multiple replicates are used
    count
        The count of the structure
    radius
        The radius of the structure
    get_bounding_box_from_mesh
        Indicates whether to get the bounding box from the mesh
    use_additional_struct
        Indicates whether to use an additional structure
    gradient_structure_name
        The name of the structure to apply the gradient

    Returns
    -------
    :
        The updated recipe
    """
    updated_recipe = recipe_template.copy()

    # update recipe version
    updated_recipe["version"] = f"{rule_name}_{cell_id}"

    # update bounding box
    if get_bounding_box_from_mesh:
        bounding_box = get_bounding_box(
            Path(mesh_path) / f"mem_mesh_{cell_id}.obj", expand=1.05
        ).tolist()
        # print(f"Bounding box for {structure_name}_{cell_id}: {bounding_box}")
        updated_recipe["bounding_box"] = bounding_box

    # update seed if needed
    if not multiple_replicates and isinstance(cell_id, int):
        updated_recipe["randomness_seed"] = cell_id

    # update grid path
    updated_recipe["grid_file_path"] = f"{grid_path}/{cell_id}_grid.dat"

    # update mesh paths
    for obj, short_name in zip(["nucleus_mesh", "membrane_mesh"], ["nuc", "mem"], strict=False):
        updated_recipe["objects"][obj]["representations"]["mesh"]["path"] = f"{mesh_path}"
        updated_recipe["objects"][obj]["representations"]["mesh"][
            "name"
        ] = f"{short_name}_mesh_{cell_id}.obj"

    # update mesh path for additional structure if needed
    if use_additional_struct:
        updated_recipe["objects"]["struct_mesh"]["representations"]["mesh"]["path"] = f"{mesh_path}"
        updated_recipe["objects"]["struct_mesh"]["representations"]["mesh"][
            "name"
        ] = f"struct_mesh_{cell_id}.obj"

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

    # update recipe rule data
    if gradient_structure_name is None:
        gradient_structure_name = structure_name
    updated_recipe = process_rule_dict(updated_recipe, rule_dict, gradient_structure_name)

    # save recipe
    rule_path = f"{generated_recipe_path}/{rule_name}"
    Path(rule_path).mkdir(parents=True, exist_ok=True)
    recipe_path = f"{rule_path}/{structure_name}_{rule_name}_{cell_id}.json"
    logger.debug(f"Saving recipe to {recipe_path}")
    write_json(recipe_path, updated_recipe)

    return updated_recipe


def get_cell_ids(workflow_config: Any) -> list[str]:
    """
    Get list of cell IDs to pack for a given structure.

    Parameters
    ----------
    workflow_config
        Workflow configuration

    Returns
    -------
    :
        List of cell IDs
    """
    if workflow_config.use_mean_cell:
        return ["mean"]
    else:
        return sample_cell_ids_for_structure(
            structure_id=workflow_config.structure_id,
            num_cells=workflow_config.num_cells,
            dsphere=workflow_config.use_cells_in_8d_sphere,
        )


def generate_recipes(workflow_config: Any) -> None:
    """
    Generate cellPACK recipes.

    Operations that need to get data from a dataframe are
    performed in this function.
    Other operations are performed in parallel inside the function
    update_and_save_recipe.

    Parameters
    ----------
    workflow_config
        Workflow configuration
    """
    cell_id_list = get_cell_ids(workflow_config)

    recipe_data = workflow_config.data.get("recipe_data", {})

    if hasattr(workflow_config, "num_processes"):
        num_processes = workflow_config.num_processes
    else:
        num_processes = np.min(
            [int(np.floor(0.8 * multiprocessing.cpu_count())), len(cell_id_list)]
        )

    recipe_template = read_json(workflow_config.recipe_template_path)

    stats_df = get_structure_stats_dataframe().set_index("CellId")

    for rule_name, rule_dict in recipe_data.items():
        if rule_name not in workflow_config.data.get("packings_to_run", {}).get("rules", []):
            continue
        logger.info(f"Generating recipes for rule: {rule_name}")
        with tqdm(total=len(cell_id_list)) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for cell_id in cell_id_list:
                    # get count from cell stats
                    count = None
                    if workflow_config.get_counts_from_data:
                        count = stats_df.loc[cell_id, "count"].astype(int)  # type: ignore

                    # get size from cell stats
                    radius = None
                    if workflow_config.get_size_from_data:
                        radius = stats_df.loc[cell_id, "radius"].astype(float)  # type: ignore

                    future = executor.submit(
                        update_and_save_recipe,
                        cell_id=cell_id,
                        structure_name=workflow_config.structure_name,
                        recipe_template=recipe_template,
                        rule_name=rule_name,
                        rule_dict=rule_dict,
                        grid_path=workflow_config.grid_path,
                        mesh_path=workflow_config.mesh_path,
                        generated_recipe_path=workflow_config.generated_recipe_path,
                        multiple_replicates=workflow_config.multiple_replicates,
                        count=count,
                        radius=radius,
                        get_bounding_box_from_mesh=workflow_config.get_bounding_box_from_mesh,
                        use_additional_struct=workflow_config.use_additional_struct,
                        gradient_structure_name=workflow_config.gradient_structure_name,
                    )
                    futures.append(future)

                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)


def generate_configs(workflow_config: Any) -> None:
    """
    Generate cellpack config file.

    Parameters
    ----------
    workflow_config
        Workflow configuration
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
