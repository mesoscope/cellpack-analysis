import gc
import json
import subprocess
import numpy as np
import concurrent.futures
import multiprocessing
from time import time
from pathlib import Path
import pandas as pd
import argparse

GENERATE_RECIPES = False
RUN_PACKINGS = False
SKIP_COMPLETED = False
DRY_RUN = False  # if True, will not run packings, just print commands
OUT_FOLDER = Path("/allen/aics/animated-cell/Saurabh/cellpack/out/")
NUM_PROCESSES = 32
NUM_CELLS = 0  # if 0, will use all cells

RULE_LIST = [
    "random",
    "nucleus_weak_gradient",
    "nucleus_moderate_gradient",
    "nucleus_strong_gradient",
    "membrane_weak_gradient",
    "membrane_moderate_gradient",
    "membrane_strong_gradient",
    "nucleus_weak_gradient_invert",
    "nucleus_moderate_gradient_invert",
    "nucleus_strong_gradient_invert",
    "membrane_weak_gradient_invert",
    "membrane_moderate_gradient_invert",
    "membrane_strong_gradient_invert",
]

datadir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data")

RECIPE_TEMPLATE_PATH = datadir / "templates/peroxisome_template.json"

CONFIG_PATH = datadir / "configs/peroxisome_packing_config_noload.json"

GENERATED_RECIPE_PATH = datadir / "generated_recipes/peroxisomes/"

MESH_PATH = datadir / "meshes/SLC25A17/"

CELLID_DF_PATH = datadir / "8dsphere_ids.csv"


def transform_and_save_dict_for_rule(
    input_dict,
    rule,
    cellID,
    base_output_path=GENERATED_RECIPE_PATH,
    mesh_base_path=MESH_PATH,
):
    output_dict = input_dict.copy()

    base_mesh_name = f"mesh_{cellID}.obj"
    output_dict["version"] = f"{rule}_{cellID}"
    for obj, short_name in zip(["nucleus_mesh", "membrane_mesh"], ["nuc", "mem"]):
        output_dict["objects"][obj]["representations"]["mesh"][
            "path"
        ] = f"{mesh_base_path}"
        output_dict["objects"][obj]["representations"]["mesh"][
            "name"
        ] = f"{short_name}_{base_mesh_name}"

    if rule == "random":
        output_dict.pop("gradients")
        output_dict["objects"]["peroxisome"].pop("gradient")
        output_dict["objects"]["peroxisome"]["packing_mode"] = "random"
    elif "gradient" in rule:
        output_dict["gradients"]["surface_gradient"]["weight_mode"] = "exponential"
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
            output_dict["gradients"]["surface_gradient"]["invert"] = True

    # save transformed dict
    with open(base_output_path / f"peroxisomes_{rule}_{cellID}.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return output_dict


def update_config_file(config_path=CONFIG_PATH, output_path=OUT_FOLDER):
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
    template_path=RECIPE_TEMPLATE_PATH,
    output_path=GENERATED_RECIPE_PATH,
    rule_list=RULE_LIST,
):
    # read json
    with open(template_path, "r") as j:
        template = json.load(j)

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
            )


def get_cell_ids_to_use(df_path=CELLID_DF_PATH, structure_name="SLC25A17", num_cells=0):
    # get cell id list for given structure
    # uses all cells by default
    df_cellID = pd.read_csv(df_path)
    df_cellID.set_index("structure", inplace=True)
    all_cellid_as_strings = df_cellID.loc[structure_name, "CellIds"].split(",")

    cellid_list = []
    for cellid in all_cellid_as_strings:
        cellid_list.append(int(cellid.replace("[", "").replace("]", "")))

    cellid_to_use = cellid_list
    if num_cells > 0:
        cellid_to_use = np.random.choice(cellid_list, num_cells)

    print(
        f"Using {len(cellid_to_use)} cell ids out of {len(cellid_list)} for {structure_name}"
    )

    return cellid_to_use


def run_single_packing(recipe_path, config_path=CONFIG_PATH):
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


def get_recipes_to_use(
    num_cells=0, generated_recipe_path=GENERATED_RECIPE_PATH, rule_list=RULE_LIST
):
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
    generated_recipe_path=GENERATED_RECIPE_PATH,
    num_processes=NUM_PROCESSES,
    num_packings=NUM_CELLS,
    config_path=CONFIG_PATH,
    out_path=OUT_FOLDER,
    skip_completed=SKIP_COMPLETED,
    dry_run=DRY_RUN,
    recipe_name="peroxisomes",
    config_name="analyze",
):
    input_recipes_to_use = get_recipes_to_use(
        num_cells=num_packings, generated_recipe_path=generated_recipe_path
    )
    num_files = len(input_recipes_to_use)

    # update config file
    update_config_file(config_path=config_path, output_path=out_path)

    result_path = out_path / f"{recipe_name}/spheresSST/"

    skipped_count = 0
    count = 0
    failed_count = 0
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        for recipe_path in input_recipes_to_use:
            fname = recipe_path.stem
            fname = fname.split(f"{recipe_name}_")[-1].split(".")[0]
            result_file = (
                result_path
                / f"figures/voxelized_image_{recipe_name}_{config_name}_{fname}_seed_0.ome.tiff"
            )
            # result_file = out_path / f"results_{recipe_name}_{config_name}_{fname}_seed_0.simularium"
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
        "--generated_recipe_path",
        type=str,
        default=GENERATED_RECIPE_PATH,
        help="Path to generated recipes",
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
        "--config_path",
        type=str,
        default=CONFIG_PATH,
        help="Path to config file",
    )

    args = parser.parse_args()

    # create files if needed
    if args.generate_recipes:
        cellID_list = get_cell_ids_to_use()
        generate_recipes(
            cellID_list=cellID_list,
        )

    # run packings
    if args.run_packings:
        count = run_packing_workflow(
            generated_recipe_path=Path(args.generated_recipe_path),
            num_processes=args.num_processes,
            num_packings=args.num_packings,
            config_path=Path(args.config_path),
            out_path=Path(args.out_folder),
            skip_completed=args.skip_completed,
            dry_run=args.dry_run,
        )

        print(f"Finished running {count} files in {time() - start:.2f} seconds")

    print(f"The workflow took {time() - start:.2f} seconds")
