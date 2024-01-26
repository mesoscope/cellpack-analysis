import os
import json
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
from time import time
import subprocess
from pathlib import Path
import gc

MAX_NUM_CLUSTERS = 3

CREATE_FILES = False
RUN_PACKINGS = True

recipe_template_path = "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/templates/npm1_template.json"
config_path = "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/config/npm1_parallel_packing_config.json"

generated_recipe_path = (
    "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/generated_recipes/"
)
MESH_PATH = "/allen/aics/modeling/ritvik/forSaurabh/"

shape_df = pd.read_csv("/allen/aics/modeling/ritvik/forSaurabh/manifest.csv")

IDS = shape_df["CellId"].unique()

COPY_GRID_FILE = True


def create_recipe_files(
    recipe_template_path,
    shape_df,
    output_path="./tmp/",
    num_clusters=MAX_NUM_CLUSTERS,
    shape_ids=IDS,
):
    # read json
    with open(recipe_template_path, "r") as j:
        contents = json.load(j)
    base_version = contents["version"]
    contents["objects"]["mean_nucleus"]["representations"]["mesh"]["path"] = MESH_PATH

    for clust in np.arange(1, num_clusters + 1):
        print(f"Creating files for {clust} clusters")
        contents_clust = contents.copy()
        contents_clust["composition"]["nucleus"]["regions"]["interior"][1][
            "count"
        ] = int(clust)
        for this_id in shape_ids:
            this_row = shape_df.loc[shape_df["CellId"] == this_id]
            this_row = this_row.loc[this_row["angle"] == 0]

            contents_clust["version"] = f"{clust}_clust_{base_version}_{this_id}"
            contents_clust["objects"]["mean_nucleus"]["representations"]["mesh"][
                "name"
            ] = f"{this_id}_0.obj"

            # save json
            with open(
                output_path + f"/{clust}_clust_{base_version}_{this_id}.json",
                "w",
            ) as f:
                json.dump(contents_clust, f, indent=4)


if CREATE_FILES:
    create_recipe_files(
        recipe_template_path=recipe_template_path,
        shape_df=shape_df,
        output_path=generated_recipe_path,
        num_clusters=MAX_NUM_CLUSTERS,
        shape_ids=IDS,
    )


files = os.listdir(generated_recipe_path)
# max_num_cellIDs = 24
max_num_cellIDs = np.inf
max_num_files = np.inf
input_files_to_use = []
num_files = 0

for file in files:
    num_clust = int(file.split("_")[0])
    input_files_to_use.append(generated_recipe_path + file)
    num_files += 1
    if num_files >= max_num_files:
        break

# output list of files to txt
with open("input_files_to_use.txt", "w") as f:
    for file in input_files_to_use:
        f.write(file + "\n")


def copy_grid_file(recipe_path, config_path, max_num_clust=MAX_NUM_CLUSTERS):
    # get recipe version
    with open(recipe_path, "r") as j:
        contents = json.load(j)
        recipe_name = contents["name"]
        recipe_version = contents["version"]
    with open(config_path, "r") as j:
        contents = json.load(j)
        config_name = contents["name"]
        out_path = Path(contents["out"])

    recipe_tail = "_".join(recipe_version.split("_")[1:])
    grid_file_name = f"{recipe_name}_{config_name}_{recipe_version}_grid.dat"

    # check if grid file exists in out_path
    grid_file_path = out_path / recipe_name / "spheresSST" / grid_file_name

    if grid_file_path.exists():
        # create files with all num_clust
        for clust in np.arange(1, max_num_clust + 1):
            new_recipe_version = f"{clust}_{recipe_tail}"
            new_grid_file_name = (
                f"{recipe_name}_{config_name}_{new_recipe_version}_grid.dat"
            )
            new_grid_file_path = (
                out_path / recipe_name / "spheresSST" / new_grid_file_name
            )

            if not new_grid_file_path.exists():
                print(f"Copying {grid_file_path} to {new_grid_file_path}")
                subprocess.run(["cp", grid_file_path, new_grid_file_path])
            else:
                print(f"{new_grid_file_path} already exists")


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

        if COPY_GRID_FILE:
            copy_grid_file(recipe_path, config_path)

        return result.returncode == 0
    except Exception as e:
        print(e)
        return False


output_folder = Path(
    "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/out/npm1/spheresSST/"
)

# run in parallel
skip_completed = True
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
    num_processes = 32
    skipped_count = 0
    count = 0
    failed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        for file in input_files_to_use:
            fname = Path(file).stem
            # output_file = (
            #     output_folder / f"results_npm1_analyze_{fname}_seed_0.simularium"
            # )
            output_file = (
                output_folder
                / f"figures/voxelized_image_npm1_analyze_{fname}_seed_0.ome.tiff"
            )
            if output_file.exists():
                if skip_completed:
                    skipped_count += 1
                    print(
                        f"Skipping {file} because output file exists, {skipped_count} skipped"
                    )
                    continue
                # else:
                # remove simularium file
                # os.remove(simularium_file)
            # sleep_time = np.random.random_sample() * 0.01
            # sleep(sleep_time)
            print(f"Submitted {file}")
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
