import concurrent.futures
import copy
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import read_json, write_json
from cellpack_analysis.lib.get_cell_id_list import sample_cell_ids_for_structure
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.mesh_tools import expand_bounding_box, get_bounding_box
from cellpack_analysis.packing import rule_repository

logger = logging.getLogger(__name__)

MESH_BOUNDS_FILENAME = "mem_mesh_bounds.parquet"
_BOUND_COLUMNS = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]


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
            # Deep-copy so callers (e.g. process_gradient_data -> set_gradient_mode_center)
            # can mutate the returned dict in place without aliasing/corrupting the shared
            # module-level rule_repository.GRADIENTS definitions.
            resolved_gradient_dict[gradient] = copy.deepcopy(rule_repository.GRADIENTS[gradient])
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
    gradient_keys: str | list[str] = list(recipe_entry.keys())
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


def _validate_recipe(recipe: dict[str, Any]) -> None:
    """
    Validate the updated recipe.

    Parameters
    ----------
    recipe
        The recipe to validate

    Raises
    ------
    ValueError
        If the recipe is invalid
    """
    cell_id = recipe["version"].split("_")[-1]
    for mesh_type in ["membrane_mesh", "nucleus_mesh", "struct_mesh"]:
        mesh_name = (
            recipe["objects"]
            .get(mesh_type, {})
            .get("representations", {})
            .get("mesh", {})
            .get("name")
        )
        if mesh_name is None:
            continue
        mesh_id = mesh_name.split("_")[-1].split(".")[0]
        if cell_id != mesh_id:
            raise ValueError(f"Cell ID {cell_id} does not match mesh ID {mesh_id}")


def _load_mesh_bounds(mesh_file: str) -> np.ndarray:
    return trimesh.load_mesh(mesh_file).bounds


def get_membrane_mesh_bounds(
    mesh_path: str | Path, cell_ids: list[str], num_processes: int = 1
) -> dict[str, np.ndarray]:
    """
    Get tight membrane mesh bounds per cell, cached beside the mesh directory.

    A cached row is reused only while its mesh file's size and mtime are unchanged.

    Parameters
    ----------
    mesh_path
        Directory holding ``mem_mesh_{cell_id}.obj``
    cell_ids
        Cells to return bounds for; cells without a mesh file are skipped
    num_processes
        Maximum worker processes for meshes not yet cached

    Returns
    -------
    :
        ``{cell_id: (2, 3) array of [min, max] bounds}``
    """
    mesh_path = Path(mesh_path)
    cache_path = mesh_path.parent / MESH_BOUNDS_FILENAME

    stats = {}
    for cell_id in map(str, cell_ids):
        mesh_file = mesh_path / f"mem_mesh_{cell_id}.obj"
        if mesh_file.exists():
            st = mesh_file.stat()
            stats[cell_id] = (st.st_size, st.st_mtime_ns)

    rows = {}
    if cache_path.exists():
        rows = {r["cell_id"]: r for r in pd.read_parquet(cache_path).to_dict("records")}

    bounds = {}
    missing = []
    for cell_id, (size, mtime_ns) in stats.items():
        row = rows.get(cell_id)
        if row is not None and (row["size"], row["mtime_ns"]) == (size, mtime_ns):
            bounds[cell_id] = np.array([row[c] for c in _BOUND_COLUMNS]).reshape(2, 3)
        else:
            missing.append(cell_id)
    if not missing:
        return bounds

    mesh_files = [str(mesh_path / f"mem_mesh_{cell_id}.obj") for cell_id in missing]
    max_workers = min(num_processes, len(os.sched_getaffinity(0)), len(missing))
    logger.info(f"Computing {len(missing)} mesh bounds with {max_workers} processes")
    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            computed = list(executor.map(_load_mesh_bounds, mesh_files))
    else:
        computed = [_load_mesh_bounds(mesh_file) for mesh_file in mesh_files]

    for cell_id, cell_bounds in zip(missing, computed, strict=True):
        bounds[cell_id] = cell_bounds
        size, mtime_ns = stats[cell_id]
        rows[cell_id] = {
            "cell_id": cell_id,
            "size": size,
            "mtime_ns": mtime_ns,
            **dict(zip(_BOUND_COLUMNS, cell_bounds.ravel(), strict=True)),
        }

    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    pd.DataFrame(list(rows.values())).to_parquet(tmp_path, index=False)
    os.replace(tmp_path, cache_path)
    return bounds


def update_and_save_recipe(
    cell_id: int | str,
    packing_id: str,
    recipe_template: dict[str, Any],
    rule_name: str,
    rule_dict: dict[str, Any],
    grid_path: str | Path,
    mesh_path: str | Path,
    generated_recipe_path: str | Path,
    number_of_replicates: int,
    count: int | None = None,
    radius: float | None = None,
    get_bounding_box_from_mesh: bool = False,
    use_additional_struct: bool = False,
    gradient_structure_name: str | None = None,
    mesh_bounds: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Update the recipe template with provided parameters and save the updated recipe to a JSON file.

    Parameters
    ----------
    cell_id
        The ID of the cell
    packing_id
        The ID of the packing
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
    number_of_replicates
        The number of replicates to pack
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
    mesh_bounds
        Precomputed tight membrane mesh bounds; skips loading the mesh

    Returns
    -------
    :
        The updated recipe
    """
    updated_recipe = copy.deepcopy(recipe_template)

    # update recipe version
    updated_recipe["version"] = f"{rule_name}_{cell_id}"

    # update bounding box
    if get_bounding_box_from_mesh:
        if mesh_bounds is not None:
            bounding_box = expand_bounding_box(mesh_bounds, expand=1.05).tolist()
        else:
            bounding_box = get_bounding_box(
                Path(mesh_path) / f"mem_mesh_{cell_id}.obj", expand=1.05
            ).tolist()
        updated_recipe["bounding_box"] = bounding_box

    # update seed if needed
    # this ensures that the same cell ID will have the same seed across different rules,
    # which is important for comparing results across rules
    # but also each cell ID will have a different seed, which is important for ensuring variability
    # across cells
    if number_of_replicates == 1 and cell_id != "mean":
        try:
            updated_recipe["randomness_seed"] = int(cell_id)
        except ValueError:
            pass

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

    # resolve gradient_structure_name before using it for counts/radius/rules
    if gradient_structure_name is None:
        gradient_structure_name = packing_id

    # update counts
    if count is not None:
        updated_recipe["composition"]["membrane"]["regions"]["interior"] = [
            "nucleus",
            {
                "object": gradient_structure_name,
                "count": int(count),
            },
        ]

    # update size
    if radius is not None:
        updated_recipe["objects"][gradient_structure_name]["radius"] = radius

    # update recipe rule data
    updated_recipe = process_rule_dict(updated_recipe, rule_dict, gradient_structure_name)

    # validate updated recipe
    _validate_recipe(updated_recipe)

    # save recipe
    rule_path = f"{generated_recipe_path}/{rule_name}"
    Path(rule_path).mkdir(parents=True, exist_ok=True)
    recipe_path = f"{rule_path}/{packing_id}_{rule_name}_{cell_id}.json"
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

    num_processes = max(1, getattr(workflow_config, "num_processes", 1))

    recipe_template = read_json(workflow_config.recipe_template_path)

    stats_df = get_structure_stats_dataframe().set_index("CellId")

    mesh_bounds: dict[str, np.ndarray] = {}
    if workflow_config.get_bounding_box_from_mesh:
        mesh_bounds = get_membrane_mesh_bounds(
            workflow_config.mesh_path, cell_id_list, num_processes
        )

    for rule_name, rule_dict in recipe_data.items():
        if rule_name not in workflow_config.data.get("packings_to_run", {}).get("rules", []):
            continue
        logger.info(f"Generating recipes for rule: {rule_name}")
        logger.debug(f"Processing {len(cell_id_list)} cell IDs for rule {rule_name}")
        with tqdm(total=len(cell_id_list)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for cell_id in cell_id_list:
                    # get count from cell stats
                    count = None
                    if workflow_config.get_counts_from_data:
                        count = stats_df.loc[cell_id, "count"].astype(int)  # type: ignore

                    # get size from cell stats
                    radius = None
                    if workflow_config.get_size_from_data:
                        # structure_stats radii are microns; recipes are authored in
                        # pixels.  Unconverted, the packed sphere is 9.26x too small.
                        radius = (
                            stats_df.loc[cell_id, "radius"].astype(float)  # type: ignore
                            / PIXEL_SIZE_IN_UM
                        )
                    future = executor.submit(
                        update_and_save_recipe,
                        cell_id=cell_id,
                        packing_id=workflow_config.packing_id,
                        recipe_template=recipe_template,
                        rule_name=rule_name,
                        rule_dict=rule_dict,
                        grid_path=workflow_config.grid_path,
                        mesh_path=workflow_config.mesh_path,
                        generated_recipe_path=workflow_config.generated_recipe_path,
                        number_of_replicates=workflow_config.number_of_replicates,
                        count=count,
                        radius=radius,
                        get_bounding_box_from_mesh=workflow_config.get_bounding_box_from_mesh,
                        use_additional_struct=workflow_config.use_additional_struct,
                        gradient_structure_name=workflow_config.gradient_structure_name,
                        mesh_bounds=mesh_bounds.get(str(cell_id)),
                    )
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        logger.debug(
                            f"Successfully generated recipe {result['name']}_{result['version']}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate recipe: {e}", exc_info=True)
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
    config_template["name"] = workflow_config.condition

    if (
        "number_of_packings" in config_template
        and config_template["number_of_packings"] != workflow_config.number_of_replicates
    ):
        logger.warning(
            f"Number of packings in config template ({config_template['number_of_packings']}) "
            f"does not match number of replicates in workflow config"
            f" ({workflow_config.number_of_replicates}). "
            f"Using number of replicates from workflow config."
        )
    config_template["number_of_packings"] = workflow_config.number_of_replicates

    # Only set seed list in config when number_of_replicates > 1 or using mean cell
    # For single replicates with real cell IDs, seed is set per-recipe instead
    if workflow_config.number_of_replicates > 1 or workflow_config.use_mean_cell:
        config_template["randomness_seed"] = list(range(workflow_config.number_of_replicates))

    for rule in rule_list:
        rule_config_path = (
            f"{workflow_config.generated_config_path}"
            f"/{rule}/{workflow_config.packing_id}_{rule}_config.json"
        )
        Path(rule_config_path).parent.mkdir(parents=True, exist_ok=True)

        rule_output_path = workflow_config.output_path / rule
        rule_output_path.mkdir(parents=True, exist_ok=True)

        config_template["out"] = str(rule_output_path)

        write_json(rule_config_path, config_template)
