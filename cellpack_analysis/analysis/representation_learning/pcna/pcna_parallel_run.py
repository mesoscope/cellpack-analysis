import os
import json
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
from time import sleep, time
import subprocess
from pathlib import Path
import gc

rules = [
#     "random",
#     "radial_gradient",
    "surface_gradient",
#     "planar_gradient_0deg",
#     "planar_gradient_15deg",
#     "planar_gradient_30deg",
#     "planar_gradient_45deg",
#     "planar_gradient_60deg",
#     "planar_gradient_75deg",
#     "planar_gradient_90deg",
]

CREATE_FILES = True
RUN_PACKINGS = True

recipe_template_path = (
    "/allen/aics/animated-cell/Saurabh/forRitvik/pcna_cellPACK/templates/"
)
config_path = "/allen/aics/animated-cell/Saurabh/forRitvik/pcna_cellPACK/config/pcna_parallel_packing_config.json"
cellpack_rules = os.listdir(recipe_template_path)
cellpack_rules = [
    recipe_template_path + i for i in cellpack_rules if i.split(".")[-1] == "json"
]
generated_recipe_path = (
    "/allen/aics/animated-cell/Saurabh/forRitvik/pcna_cellPACK/generated_recipes/"
)
mesh_path = "/allen/aics/modeling/ritvik/forSaurabh/"

shape_df = pd.read_csv("/allen/aics/modeling/ritvik/forSaurabh/manifest.csv")

IDS = shape_df["CellId"].unique()
ANGLES = shape_df["angle"].unique()


def create_rule_files(
    cellpack_rules,
    shape_df,
    output_path="./tmp/",
    shape_ids=IDS,
    shape_angles=ANGLES,
    config_path=config_path,
):
    # read json
    for rule in cellpack_rules:
        print(f"Creating files for {rule}")
        with open(rule, "r") as j:
            contents = json.load(j)
            contents_shape = contents.copy()
            base_version = contents_shape["version"]
            for this_id in shape_ids:
                for ang in shape_angles:
                    this_row = shape_df.loc[shape_df["CellId"] == this_id]
                    this_row = this_row.loc[this_row["angle"] == ang]

                    contents_shape["version"] = f"{base_version}_{this_id}_{ang}"
                    contents_shape["objects"]["mean_nucleus"]["representations"][
                        "mesh"
                    ]["name"] = f"{this_id}_{ang}.obj"
                    contents_shape["objects"]["mean_nucleus"]["representations"][
                        "mesh"
                    ]["path"] = mesh_path
                    # save json
                    with open(
                        output_path + f"/{base_version}_{this_id}_rotation_{ang}.json",
                        "w",
                    ) as f:
                        json.dump(contents_shape, f, indent=4)


if CREATE_FILES:
    create_rule_files(cellpack_rules, shape_df, generated_recipe_path, IDS, ANGLES)

rules_to_use = [
    # "planar_gradient_0deg",
    # "planar_gradient_45deg",
    # "planar_gradient_90deg",
    "surface_gradient",
    # "random",
    # "radial_gradient",
]

shape_rotations = [
    "rotation_0",
    # "rotation_1",
    # "rotation_2",
]

files = os.listdir(generated_recipe_path)
# max_num_cellIDs = 24
max_num_cellIDs = np.inf
# max_num_files = max_num_cellIDs * len(shape_rotations) * len(rules_to_use)  # 9 rules x number of files per rule
max_num_files = np.inf
input_files_to_use = []
num_files = 0

for rule in rules_to_use:
    for rot in shape_rotations:
        for file in files:
            if (rule in file) and (rot in file):
                input_files_to_use.append(generated_recipe_path + file)
                num_files += 1
        if "random" in rule or "radial" in rule:
            break
    if num_files >= max_num_files:
        break

# output list of files to txt
with open("input_files_to_use.txt", "w") as f:
    for file in input_files_to_use:
        f.write(file + "\n")


def run_packing(recipe_path, config_path=config_path):
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


out_path = Path("/allen/aics/animated-cell/Saurabh/cellpack/out/pcna/spheresSST")

# run in parallel
skip_completed = False
num_files = len(input_files_to_use)
print(f"Found {num_files} files")
start = time()
futures = []
if RUN_PACKINGS:
    # input_files_to_use = [input_files_to_use[0]]
    num_processes = np.min(
        [
            int(np.floor(0.8 * multiprocessing.cpu_count())),
            num_files,
        ]
    )
    num_processes = 16
    skipped_count = 0
    count = 0
    failed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        for file in input_files_to_use:
            fname = Path(file).stem
            fname = "".join(fname.split("_rotation"))
            simularium_file = (
                out_path / f"results_pcna_analyze_{fname}_seed_0.simularium"
            )
            if simularium_file.exists():
                if skip_completed:
                    skipped_count += 1
                    print(
                        f"Skipping {file} because simularium file exists, {skipped_count} skipped"
                    )
                    continue
                # else:
                # remove simularium file
                # os.remove(simularium_file)
            # sleep_time = np.random.random_sample() * 0.01
            # sleep(sleep_time)
            print(f"Running {file}")
            futures.append(executor.submit(run_packing, file))
        # print number of futures completed
        print(f"Submitted {len(futures)} jobs, {skipped_count} skipped")
        for future in concurrent.futures.as_completed(futures):
            # check for exceptions
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

print(f"Finished running {len(futures)} files in {time() - start:.2f} seconds")
