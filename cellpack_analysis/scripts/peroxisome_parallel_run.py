import os
import json
import numpy as np
import concurrent.futures
import multiprocessing
from pathlib import Path
from time import sleep
import pandas as pd

CREATE_FILES = False
RUN_PACKINGS = True

RULE_LIST = [
    "random",
    "nucleus_weak_gradient",
    "nucleus_moderate_gradient",
    "nucleus_strong_gradient",
    "membrane_weak_gradient",
    "membrane_moderate_gradient",
    "membrane_strong_gradient",
]


recipe_template_path = Path(
    "/allen/aics/animated-cell/Saurabh/cellpack_PILR_analysis/data/templates/peroxisome_template.json"
)
config_path = Path(
    "/allen/aics/animated-cell/Saurabh/cellpack_PILR_analysis/data/configs/peroxisome_packing_config.json"
)
generated_recipe_path = Path(
    "/allen/aics/animated-cell/Saurabh/cellpack_PILR_analysis/data/generated_recipes"
)
mesh_path = Path(
    "/allen/aics/animated-cell/Saurabh/cellpack_PILR_analysis/data/meshes/SLC25A17/"
)
cellid_df_path = Path(
    "/allen/aics/animated-cell/Saurabh/cellpack_PILR_analysis/data/8dsphere_ids.csv"
)


def transform_and_save_dict_for_rule(
    input_dict,
    rule,
    cellID,
    base_output_path=generated_recipe_path,
    mesh_base_path=mesh_path,
):

    output_dict = input_dict.copy()

    base_mesh_name = f"mesh_{cellID}.obj"
    output_dict["version"] = f"{rule}_{cellID}"
    for obj, short_name in zip(["nucleus", "membrane"], ["nuc", "mem"]):
        output_dict["objects"][obj]["representations"]["mesh"]["path"] = f"{mesh_base_path}"
        output_dict["objects"][obj]["representations"]["mesh"][
            "name"
        ] = f"{short_name}_{base_mesh_name}"

    if rule == "random":
        output_dict.pop("gradients")
        output_dict["objects"]["peroxisome"].pop("gradient")
        output_dict["objects"]["peroxisome"]["packing_mode"] = "random"
    elif "gradient" in rule:
        output_dict["gradients"]["surface_gradient"]["mode_settings"][
            "weight_mode"
        ] = "exponential"
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
                "decay_length": 0.9
            }
        if "moderate" in rule:
            output_dict["gradients"]["surface_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.3
            }
        if "strong" in rule:
            output_dict["gradients"]["surface_gradient"]["weight_mode_settings"] = {
                "decay_length": 0.1
            }

    # save transformed dict
    with open(base_output_path / f"peroxisomes_{rule}_{cellID}.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return output_dict


def create_files(
    cellID_list,
    template_path=recipe_template_path,
    output_path=generated_recipe_path,
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


def get_cellID_list(df_path=cellid_df_path, structure_name="SLC25A17"):
    # get cell id list for peroxisomes
    df_cellID = pd.read_csv(df_path)
    df_cellID.set_index("structure", inplace=True)
    str_cellid = df_cellID.loc[structure_name, "CellIds"].split(",")
    cellid_list = []
    for cellid in str_cellid:
        cellid_list.append(int(cellid.replace("[", "").replace("]", "")))

    return cellid_list


if CREATE_FILES:
    cellID_list = get_cellID_list()
    create_files(
        cellID_list=cellID_list,
    )


def run_packing(recipe_path, config_path=config_path):
    print(f"Running {recipe_path}")
    os.system(f"pack -r {recipe_path} -c {config_path}")


def chunk_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:min(i + chunk_size, len(input_list))]


# run in parallel
if RUN_PACKINGS:    
    cellID_list = get_cellID_list()
    cellid_to_use = cellID_list

    input_file_list = list(generated_recipe_path.glob("*.json"))
    input_files_to_use = []
    for cellid in cellid_to_use:
        for file in input_file_list:
            if str(cellid) in file.stem:
                input_files_to_use.append(file)

    print(f"Running {len(input_files_to_use)} files")

    num_processes = 16
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for chunked_files in chunk_list(input_files_to_use, num_processes):  
            print(f"Starting new chunk with {len(chunked_files)} files")
            for file in chunked_files:
                sleep_time = np.random.random_sample() * 5
                sleep(sleep_time)
                futures.append(executor.submit(run_packing, file))
            done, not_done = concurrent.futures.wait(futures)

            # print number of futures completed            
            print(f"Completed {len(done)} out of {len(input_files_to_use)}")
