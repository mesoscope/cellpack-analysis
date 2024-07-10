import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import subprocess
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

AXIS_TO_VEC = {
    "X": [1, 0, 0],
    "Y": [0, 1, 0],
    "Z": [0, 0, 1],
}

GENERATE_RECIPES = True
RUN_PACKINGS = False
SKIP_COMPLETED = False
DRY_RUN = False  # if True, will not run packings, just print commands
OUT_FOLDER = Path("/allen/aics/animated-cell/Saurabh/cellpack/out/")
NUM_PROCESSES = 32
NUM_CELLS = 0  # if 0, will use all cells

RULE_LIST = [
    "random",
    "surface_gradient_nucleus_weak",
    "surface_gradient_nucleus_moderate",
    "surface_gradient_nucleus_strong",
    "surface_gradient_membrane_weak",
    "surface_gradient_membrane_moderate",
    "surface_gradient_membrane_strong",
    "surface_gradient_nucleus_weak_invert",
    "surface_gradient_nucleus_moderate_invert",
    "surface_gradient_nucleus_strong_invert",
    "surface_gradient_membrane_weak_invert",
    "surface_gradient_membrane_moderate_invert",
    "surface_gradient_membrane_strong_invert",
    "planar_gradient_Z_weak",
    "planar_gradient_Z_moderate",
    "planar_gradient_Z_strong",
]


def set_paths(
    structure_name,
    structure_id,
    datadir=Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data"),
):
    recipe_template_path = datadir / f"templates/{structure_name}_template.json"

    config_path = datadir / f"configs/{structure_name}_packing_config.json"

    generated_recipe_path = datadir / f"generated_recipes/{structure_name}/"

    mesh_path = datadir / f"structure_data/{structure_id}/meshes/"

    grid_path = datadir / f"structure_data/{structure_id}/grids/"

    cellid_df_path = datadir / "8dsphere_ids.csv"

    return (
        recipe_template_path,
        config_path,
        generated_recipe_path,
        mesh_path,
        grid_path,
        cellid_df_path,
    )


def get_mesh_vertices(mesh_file_path):
    """
    Given a mesh file path, returns the coordinates of the mesh vertices.
    """
    coordinates = []
    with open(mesh_file_path, "r") as mesh_file:
        for line in mesh_file:
            if line.startswith("v"):
                coordinates.append([float(x) for x in line.split()[1:]])
    coordinates = np.array(coordinates)
    return coordinates


def get_mesh_center(mesh_file_path):
    """
    Given a mesh file path, returns the center of the mesh.
    """
    coordinates = get_mesh_vertices(mesh_file_path)
    center = np.mean(coordinates, axis=0)
    return center


def get_mesh_boundaries(mesh_file_path):
    """
    Given a mesh file path, returns the coordinates:
    [max_x, max_y, max_z] , [min_x, min_y, min_z]
    """
    coordinates = get_mesh_vertices(mesh_file_path)
    max_coordinates = np.max(coordinates, axis=0)
    min_coordinates = np.min(coordinates, axis=0)
    return max_coordinates, min_coordinates


def transform_and_save_dict_for_rule(
    input_dict,
    rule,
    cellID,
    base_output_path,
    mesh_base_path,
    grid_path,
    structure_name,
):
    output_dict = input_dict.copy()
    base_mesh_name = f"mesh_{cellID}.obj"
    output_dict["version"] = f"{rule}_{cellID}"
    grid_file_path = grid_path / f"{cellID}_grid.dat"
    output_dict["grid_file_path"] = f"{grid_file_path}"
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
        _, min_coordinates = get_mesh_boundaries(mem_mesh_path)
        output_dict["gradients"]["planar_gradient"]["mode_settings"]["center"] = [
            0,
            0,
            round(min_coordinates[2], 3),
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
    with open(config_path, "r") as j:
        config = json.load(j)

    # update paths
    config["out"] = str(output_path)

    # save transformed dict
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def generate_recipes(
    cellID_list,
    template_path,
    output_path,
    mesh_base_path,
    grid_path,
    structure_name,
    rule_list=RULE_LIST,
):
    # read json
    with open(template_path, "r") as j:
        template = json.load(j)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for rule in rule_list:
        print(f"Creating files for rule {rule}")
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
            )


def get_cell_ids_to_use(df_path, structure_id, num_cells=0):
    # get cell id list for given structure
    # uses all cells by default
    df_cellID = pd.read_csv(df_path)
    df_cellID.set_index("structure", inplace=True)
    all_cellid_as_strings = df_cellID.loc[structure_id, "CellIds"].split(",")

    cellid_list = []
    for cellid in all_cellid_as_strings:
        cellid_list.append(int(cellid.replace("[", "").replace("]", "")))

    cellid_to_use = cellid_list
    if num_cells > 0:
        cellid_to_use = np.random.choice(cellid_list, num_cells)

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


def chunk_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : min(i + chunk_size, len(input_list))]


def get_recipes_to_use(generated_recipe_path, num_cells=0, rule_list=RULE_LIST):
    cell_ids_to_use = get_cell_ids_to_use(num_cells=num_cells)
    input_file_list = list(generated_recipe_path.glob("*.json"))
    input_files_to_use = []
    for file in input_file_list:
        for cellid in cell_ids_to_use:
            fstem = file.stem
            if str(cellid) in fstem and any([rule in fstem for rule in rule_list]):
                input_files_to_use.append(file)
    print("Found", len(input_files_to_use), "files")
    return input_files_to_use


def run_packing_workflow(
    generated_recipe_path,
    num_processes,
    num_packings,
    config_path,
    structure_name,
    out_path=OUT_FOLDER,
    skip_completed=SKIP_COMPLETED,
    dry_run=DRY_RUN,
    config_name="analyze",
):
    input_recipes_to_use = get_recipes_to_use(
        num_cells=num_packings, generated_recipe_path=generated_recipe_path
    )
    num_files = len(input_recipes_to_use)

    # update config file
    update_config_file(config_path=config_path, output_path=out_path)

    result_path = out_path / f"{structure_name}/spheresSST/"

    skipped_count = 0
    count = 0
    failed_count = 0
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        for recipe_path in input_recipes_to_use:
            fname = recipe_path.stem
            fname = fname.split(f"{structure_name}_")[-1].split(".")[0]
            result_file = (
                result_path
                / f"figures/voxelized_image_{structure_name}_{config_name}_{fname}_seed_0.ome.tiff"
            )
            # result_file = out_path / f"results_{structure_name}_{config_name}_{fname}_seed_0.simularium"
            if result_file.exists():
                if skip_completed:
                    skipped_count += 1
                    print(
                        f"Skipping {recipe_path} because result file exists, {skipped_count} skipped"
                    )
                    continue
            # sleep_time = np.random.random_sample() * 0.01
            # sleep(sleep_time)
            print(f"Submitted {recipe_path}")
            if dry_run:
                continue
            futures.append(
                executor.submit(run_single_packing, recipe_path, config_path)
            )
        # print number of futures completed
        print(f"Submitted {len(futures)} jobs, {skipped_count} skipped")
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                count += 1
            else:
                failed_count += 1
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
                f"Total time: {t:.2f} seconds, Time per run: {per_count:.2f} seconds,",
                f"Estimated time left: {time_left:.2f} seconds",
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

    args = parser.parse_args()

    # set paths
    (
        recipe_template_path,
        config_path,
        generated_recipe_path,
        mesh_path,
        grid_path,
        cellid_df_path,
    ) = set_paths()

    # create files if needed
    if args.generate_recipes:
        cellID_list = get_cell_ids_to_use(
            df_path=cellid_df_path,
            structure_id=args.structure_id,
            num_cells=args.num_packings,
        )
        generate_recipes(
            cellID_list=cellID_list,
            template_path=recipe_template_path,
            output_path=generated_recipe_path,
            mesh_base_path=mesh_path,
            grid_path=grid_path,
            structure_name=args.structure_name,
            rule_list=RULE_LIST,
        )

    # run packings
    if args.run_packings:
        count = run_packing_workflow(
            generated_recipe_path=Path(args.generated_recipe_path),
            num_processes=args.num_processes,
            num_packings=args.num_packings,
            config_path=Path(args.config_path),
            structure_name=args.structure_name,
            out_path=Path(args.out_folder),
            skip_completed=args.skip_completed,
            dry_run=args.dry_run,
        )

        print(f"Finished running {count} files in {time() - start:.2f} seconds")

    print(f"The workflow took {time() - start:.2f} seconds")
