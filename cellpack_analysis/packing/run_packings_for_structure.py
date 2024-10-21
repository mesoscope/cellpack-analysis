import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import subprocess
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from cellpack_analysis.lib.mesh_tools import get_mesh_boundaries

np.random.seed(42)

AXIS_TO_VEC = {
    "X": [1, 0, 0],
    "Y": [0, 1, 0],
    "Z": [0, 0, 1],
}

GENERATE_RECIPES = False
RUN_PACKINGS = False
SKIP_COMPLETED = False
DRY_RUN = False  # if True, will not run packings, just print commands
OUT_FOLDER = Path("/allen/aics/animated-cell/Saurabh/cellpack/out/")
NUM_PROCESSES = 32
NUM_CELLS = 0  # if 0, will use all cells

RULE_LIST = [
    # "random",
    # "surface_gradient_nucleus_weak",
    # "surface_gradient_nucleus_moderate",
    # "surface_gradient_nucleus_strong",
    # "surface_gradient_membrane_weak",
    # "surface_gradient_membrane_moderate",
    "surface_gradient_membrane_strong",
    # "surface_gradient_nucleus_weak_invert",
    # "surface_gradient_nucleus_moderate_invert",
    # "surface_gradient_nucleus_strong_invert",
    # "surface_gradient_membrane_weak_invert",
    # "surface_gradient_membrane_moderate_invert",
    "surface_gradient_membrane_strong_invert",
    # "planar_gradient_Z_weak",
    # "planar_gradient_Z_moderate",
    # "planar_gradient_Z_strong",
    # "planar_gradient_Y_moderate",
]


def set_paths(
    structure_name,
    structure_id,
    datadir=None,
    recipe_template_path=None,
    config_path=None,
    generated_recipe_path=None,
    use_mean_cell=False,
):
    if datadir is None:
        datadir = Path("/allen/aics/animated-cell/Saurabh/cellpack/")
    else:
        datadir = Path(datadir)

    if recipe_template_path is None:
        recipe_template_path = datadir / f"templates/{structure_name}_template.json"
    else:
        recipe_template_path = Path(recipe_template_path)

    if config_path is None:
        config_path = datadir / f"configs/{structure_name}_packing_config.json"
    else:
        config_path = Path(config_path)

    if generated_recipe_path is None:
        generated_recipe_path = datadir / f"generated_recipes/{structure_name}/"
    else:
        generated_recipe_path = Path(generated_recipe_path)

    if use_mean_cell:
        mesh_path = datadir / "average_shape_meshes"
        grid_path = datadir / "average_shape_grids"
    else:
        mesh_path = datadir / f"structure_data/{structure_id}/meshes/"
        grid_path = datadir / f"structure_data/{structure_id}/grids/"

    cellid_df_path = datadir / "all_cellids.csv"

    return (
        recipe_template_path,
        config_path,
        generated_recipe_path,
        mesh_path,
        grid_path,
        cellid_df_path,
    )


def transform_and_save_dict_for_rule(
    input_dict,
    rule,
    cellID,
    base_output_path,
    mesh_base_path,
    grid_path,
    structure_name,
    use_cellid_as_seed=False,
):
    output_dict = input_dict.copy()
    base_mesh_name = f"mesh_{cellID}.obj"
    output_dict["version"] = f"{rule}_{cellID}"
    grid_file_path = grid_path / f"{cellID}_grid.dat"
    output_dict["grid_file_path"] = f"{grid_file_path}"
    base_output_path.mkdir(parents=True, exist_ok=True)
    if use_cellid_as_seed:
        output_dict["randomness_seed"] = [cellID]
    for obj, short_name in zip(["nucleus_mesh", "membrane_mesh"], ["nuc", "mem"]):
        output_dict["objects"][obj]["representations"]["mesh"][
            "path"
        ] = f"{mesh_base_path}"
        output_dict["objects"][obj]["representations"]["mesh"][
            "name"
        ] = f"{short_name}_{base_mesh_name}"

    if rule == "random":
        output_dict.pop("gradients")
        output_dict["objects"][structure_name].pop("gradient")
        output_dict["objects"][structure_name]["packing_mode"] = "random"
    elif "surface_gradient" in rule:
        output_dict["gradients"]["surface_gradient"] = {
            "description": "gradient based on distance from a surface",
            "pick_mode": "rnd",
            "mode": "surface",
            "mode_settings": {
                "object": "nucleus",
                "scale_to_next_surface": False,
            },
            "weight_mode": "exponential",
            "weight_mode_settings": {"decay_length": 0.1},
        }
        if "nucleus" in rule:
            output_dict["gradients"]["surface_gradient"]["mode_settings"][
                "object"
            ] = "nucleus"
        if "membrane" in rule:
            output_dict["gradients"]["surface_gradient"]["mode_settings"][
                "object"
            ] = "membrane"
        if "weak" in rule:
            output_dict["gradients"]["surface_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.3
            }
        if "moderate" in rule:
            output_dict["gradients"]["surface_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.1
            }
        if "strong" in rule:
            output_dict["gradients"]["surface_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.03
            }
        if "invert" in rule:
            output_dict["gradients"]["surface_gradient"]["invert"] = "distance"
        output_dict["objects"][structure_name]["packing_mode"] = "gradient"
        output_dict["objects"][structure_name]["gradient"] = "surface_gradient"
    elif "planar_gradient" in rule:
        output_dict["gradients"].pop("surface_gradient")
        output_dict["gradients"]["planar_gradient"] = {
            "description": "gradient based on distance from a plane",
            "pick_mode": "rnd",
            "mode": "vector",
            "mode_settings": {
                "direction": [0, 0, 1],
                "center": [0, 0, -150],
            },
            "weight_mode": "exponential",
            "weight_mode_settings": {"decay_length": 0.1},
        }

        output_dict["objects"][structure_name]["packing_mode"] = "gradient"
        output_dict["objects"][structure_name]["gradient"] = "planar_gradient"

        # set vector and center for planar gradient
        axis = rule.split("_")[2]
        vec = AXIS_TO_VEC[axis]
        output_dict["gradients"]["planar_gradient"]["mode_settings"]["direction"] = vec

        mem_mesh_path = mesh_base_path / f"mem_{base_mesh_name}"
        max_coordinates, _ = get_mesh_boundaries(mem_mesh_path)
        output_dict["gradients"]["planar_gradient"]["mode_settings"]["center"] = [
            round(coord, 3) for coord in max_coordinates
        ]

        if "weak" in rule:
            output_dict["gradients"]["planar_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.3
            }
        if "moderate" in rule:
            output_dict["gradients"]["planar_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.1
            }
        if "strong" in rule:
            output_dict["gradients"]["planar_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.03
            }
        if "invert" in rule:
            output_dict["gradients"]["planar_gradient"]["invert"] = "weight"

    # save transformed dict
    with open(base_output_path / f"{structure_name}_{rule}_{cellID}.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return output_dict


def update_config_file(config_path, output_path):
    # read json
    with open(config_path) as j:
        config = json.load(j)

    # update paths
    config["out"] = str(output_path)

    # save transformed dict
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config


def generate_recipes(
    cellID_list,
    template_path,
    output_path,
    mesh_base_path,
    grid_path,
    structure_name,
    rule_list=RULE_LIST,
    use_cellid_as_seed=False,
):
    # if cellID_list is None, will use mean cell
    # read json template
    with open(template_path) as j:
        template = json.load(j)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for rule in rule_list:
        print(f"Creating files for rule {rule}")
        if cellID_list is None:
            transform_and_save_dict_for_rule(
                input_dict=template,
                rule=rule,
                cellID="mean",
                base_output_path=output_path,
                mesh_base_path=mesh_base_path,
                grid_path=grid_path,
                structure_name=structure_name,
                use_cellid_as_seed=False,
            )
            continue

        # transform dicts in parallel
        num_processes = np.min(
            [
                int(np.floor(0.8 * multiprocessing.cpu_count())),
                len(cellID_list),
            ]
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            executor.map(
                transform_and_save_dict_for_rule,
                [template] * len(cellID_list),
                [rule] * len(cellID_list),
                cellID_list,
                [output_path] * len(cellID_list),
                [mesh_base_path] * len(cellID_list),
                [grid_path] * len(cellID_list),
                [structure_name] * len(cellID_list),
                [use_cellid_as_seed] * len(cellID_list),
            )


def get_cell_ids_to_use(
    cellid_df_path,
    structure_id,
    num_cells=0,
    use_mean_cell=False,
    use_cells_in_8d_sphere=True,
):
    # get cell id list for given structure
    # uses all cells by default
    if use_mean_cell:
        return None

    df_cellID = pd.read_csv(cellid_df_path)
    if use_cells_in_8d_sphere:
        df_cellID = df_cellID[df_cellID["8dsphere"]]

    cellid_list = df_cellID.loc[
        df_cellID["structure_name"] == structure_id, "CellId"
    ].values.tolist()

    cellid_to_use = cellid_list
    if num_cells > 0:
        cellid_to_use = np.random.choice(cellid_list, num_cells, replace=False).tolist()

    print(
        f"Using {len(cellid_to_use)} cell ids out of {len(cellid_list)} for {structure_id}"
    )

    return cellid_to_use


def run_single_packing(recipe_path, config_path):
    try:
        print(f"Running {recipe_path}")
        result = subprocess.run(
            [
                "pack",
                "-r",
                recipe_path,
                "-c",
                config_path,
            ],
            check=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(e)
        return False


def get_recipes_to_use(
    generated_recipe_path,
    cellid_df_path,
    structure_id,
    num_cells=0,
    rule_list=RULE_LIST,
    cell_ids_to_use=None,
    use_mean_cell=False,
    use_cells_in_8d_sphere=True,
):
    if use_mean_cell:
        cell_ids_to_use = ["mean"]
    elif cell_ids_to_use is None:
        cell_ids_to_use = get_cell_ids_to_use(
            cellid_df_path=cellid_df_path,
            structure_id=structure_id,
            num_cells=num_cells,
            use_mean_cell=use_mean_cell,
            use_cells_in_8d_sphere=use_cells_in_8d_sphere,
        )
    input_file_list = list(generated_recipe_path.glob("*.json"))
    input_files_to_use = []
    for file in input_file_list:
        for cellid in cell_ids_to_use:
            fstem = file.stem
            if str(cellid) in fstem and any([rule in fstem for rule in rule_list]):
                input_files_to_use.append(file)
    print("Found", len(input_files_to_use), "files")
    return input_files_to_use, cell_ids_to_use


def check_run_this_recipe(recipe_path, config_data, structure_name, check_type="image"):
    with open(recipe_path) as j:
        recipe_data = json.load(j)

    number_of_packings = config_data.get("number_of_packings", 1)
    seed_vals = recipe_data.get("randomness_seed", [0])

    base_folder = Path(config_data["out"]) / f"{structure_name}/spheresSST/"

    if check_type == "image":
        folder_to_check = base_folder / "figures"
        prefix = "voxelized_image"
    elif check_type == "simularium":
        folder_to_check = base_folder
        prefix = "results"
    else:
        raise ValueError("check_type must be 'image' or 'simularium'")

    if len(seed_vals) == number_of_packings:
        # usually this means only packing one seed
        # check whether all files exist explicitly
        result_file_list = [
            folder_to_check
            / f"{prefix}_{recipe_data['name']}_{config_data['name']}_{recipe_data['version']}_seed_{seed_val}.ome.tiff"
            for seed_val in seed_vals
        ]
    else:
        result_file_list = list(
            folder_to_check.glob(
                f"{prefix}_{recipe_data['name']}_{config_data['name']}_{recipe_data['version']}_seed_*.ome.tiff"
            )
        )
    num_existing_files = sum([result_file.exists() for result_file in result_file_list])

    return num_existing_files == number_of_packings


def run_packing_workflow(
    generated_recipe_path,
    num_processes,
    num_packings,
    config_path,
    structure_name,
    structure_id,
    cellid_df_path,
    out_path=OUT_FOLDER,
    skip_completed=SKIP_COMPLETED,
    dry_run=DRY_RUN,
    cell_ids_to_use=None,
    use_mean_cell=False,
    use_cells_in_8d_sphere=True,
):
    input_recipes_to_use, _ = get_recipes_to_use(
        generated_recipe_path=generated_recipe_path,
        cellid_df_path=cellid_df_path,
        structure_id=structure_id,
        num_cells=num_packings,
        cell_ids_to_use=cell_ids_to_use,
        use_mean_cell=use_mean_cell,
        use_cells_in_8d_sphere=use_cells_in_8d_sphere,
    )
    num_files = len(input_recipes_to_use)

    # save list of input_recipes
    with open(out_path / f"{structure_name}_input_recipes.txt", "w") as f:
        f.write("\n".join([str(x) for x in input_recipes_to_use]))

    # update config file
    config_data = update_config_file(config_path=config_path, output_path=out_path)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = out_path.parent / f"{structure_name}_log_{current_datetime}.txt"

    skipped_count = 0
    count = 0
    failed_count = 0
    futures = []
    if num_processes == 1:
        for recipe_path in input_recipes_to_use:
            if check_run_this_recipe(
                recipe_path, config_data, structure_name, check_type="image"
            ):
                if skip_completed:
                    skipped_count += 1
                    print(
                        f"Skipping {recipe_path} because result file exists, {skipped_count} skipped"
                    )
                    continue

            if dry_run:
                continue

            result = run_single_packing(recipe_path, config_path)
            if result:
                count += 1
            else:
                failed_count += 1
                with open(log_file_path, "a") as f:
                    f.write(f"Failed: {recipe_path}\n")
            done = count + skipped_count
            remaining = num_files - done - failed_count
            print(
                f"Completed: {count}, Failed: {failed_count}, Skipped: {skipped_count},",
                f"Total: {num_files}, Done: {done}, Remaining: {remaining}",
            )
            t = time() - start
            per_count = np.inf
            time_left = np.inf
            if count > 0:
                per_count = t / count
                time_left = per_count * remaining
            print(
                f"Total time: {t:.2f}s, Time per run: {per_count:.2f}s,",
                f"Estimated time left: {time_left:.2f}s",
            )
            gc.collect()
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            for recipe_path in input_recipes_to_use:
                if check_run_this_recipe(
                    recipe_path, config_data, structure_name, check_type="image"
                ):
                    if skip_completed:
                        skipped_count += 1
                        print(
                            f"Skipping {recipe_path} because result file exists, {skipped_count} skipped"
                        )
                        continue

                if dry_run:
                    continue

                futures.append(
                    executor.submit(run_single_packing, recipe_path, config_path)
                )
                print(f"Submitted {recipe_path}")

            print(f"Submitted {len(futures)} jobs, {skipped_count} skipped")
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    count += 1
                else:
                    failed_count += 1
                    with open(log_file_path, "a") as f:
                        f.write(f"Failed: {recipe_path}\n")
                done = count + skipped_count
                remaining = num_files - done - failed_count
                print(
                    f"Completed: {count}, Failed: {failed_count}, Skipped: {skipped_count},",
                    f"Total: {num_files}, Done: {done}, Remaining: {remaining}",
                )
                t = time() - start
                per_count = np.inf
                time_left = np.inf
                if count > 0:
                    per_count = t / count
                    time_left = per_count * remaining
                print(
                    f"Total time: {t:.2f}s, Time per run: {per_count:.2f}s,",
                    f"Estimated time left: {time_left:.2f}s",
                )
                gc.collect()

    return count


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_recipes",
        action="store_true",
        default=GENERATE_RECIPES,
        help="If true, will create files",
    )
    parser.add_argument(
        "--run_packings",
        action="store_true",
        default=RUN_PACKINGS,
        help="If true, will run packings",
    )
    parser.add_argument(
        "--num_packings",
        type=int,
        default=0,
        help="Number of packings to run, if 0, will run all packings",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default=OUT_FOLDER,
        help="cellpack output folder",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        default=SKIP_COMPLETED,
        help="If true, will skip completed files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=DRY_RUN,
        help="If true, will not run packings, just print commands",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=NUM_PROCESSES,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--structure_name",
        type=str,
        default="peroxisome",
        help="Name of structure",
    )
    parser.add_argument(
        "--structure_id",
        type=str,
        default="SLC25A17",
        help="ID of structure",
    )
    parser.add_argument(
        "--recipe_template_path",
        type=str,
        default=None,
        help="Path to recipe template",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--generated_recipe_path",
        type=str,
        default=None,
        help="Path to generated recipes",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=None,
        help="Path to data directory",
    )
    parser.add_argument(
        "--use_cellid_as_seed",
        action="store_true",
        default=False,
        help="If true, will use cellid as seed",
    )
    parser.add_argument(
        "--use_mean_cell",
        action="store_true",
        default=False,
        help="If true, will use mean cell",
    )
    parser.add_argument(
        "--use_cells_in_8d_sphere",
        action="store_true",
        default=False,
        help="If true, will use cells in 8d sphere",
    )

    args = parser.parse_args()

    # set paths
    (
        recipe_template_path,
        config_path,
        generated_recipe_path,
        mesh_path,
        grid_path,
        cellid_df_path,
    ) = set_paths(
        structure_name=args.structure_name,
        structure_id=args.structure_id,
        datadir=args.datadir,
        recipe_template_path=args.recipe_template_path,
        config_path=args.config_path,
        generated_recipe_path=args.generated_recipe_path,
        use_mean_cell=args.use_mean_cell,
    )

    # create files if needed
    cellid_list = None
    if args.generate_recipes:
        cellid_list = get_cell_ids_to_use(
            cellid_df_path=cellid_df_path,
            structure_id=args.structure_id,
            num_cells=args.num_packings,
            use_mean_cell=args.use_mean_cell,
            use_cells_in_8d_sphere=args.use_cells_in_8d_sphere,
        )
        generate_recipes(
            cellID_list=cellid_list,
            template_path=recipe_template_path,
            output_path=generated_recipe_path,
            mesh_base_path=mesh_path,
            grid_path=grid_path,
            structure_name=args.structure_name,
            rule_list=RULE_LIST,
            use_cellid_as_seed=args.use_cellid_as_seed,
        )

    # run packings
    if args.run_packings:
        out_folder = Path(args.out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        count = run_packing_workflow(
            generated_recipe_path=generated_recipe_path,
            num_processes=args.num_processes,
            num_packings=args.num_packings,
            config_path=config_path,
            structure_name=args.structure_name,
            structure_id=args.structure_id,
            cellid_df_path=cellid_df_path,
            out_path=out_folder,
            skip_completed=args.skip_completed,
            dry_run=args.dry_run,
            cell_ids_to_use=cellid_list,
            use_mean_cell=args.use_mean_cell,
            use_cells_in_8d_sphere=args.use_cells_in_8d_sphere,
        )

        print(f"Finished running {count} files in {time() - start:.2f} seconds")

    print(f"The workflow took {time() - start:.2f} seconds")
