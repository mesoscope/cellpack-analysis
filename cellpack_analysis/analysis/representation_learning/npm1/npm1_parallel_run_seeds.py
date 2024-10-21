import concurrent.futures
import gc
import json
import multiprocessing
import os
import subprocess
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

MAX_NUM_CLUSTERS = 5

CREATE_FILES = True
RUN_PACKINGS = True

recipe_template_path = "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/templates/npm1_template.json"
config_path = "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/config/npm1_parallel_packing_config.json"

generated_recipe_path = (
    "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/generated_recipes/"
)
MESH_PATH = "/allen/aics/modeling/ritvik/forSaurabh/"
BASE_GRID_PATH = "/allen/aics/animated-cell/Saurabh/forRitvik/npm1_cellPACK/grids/"

shape_df = pd.read_csv("/allen/aics/modeling/ritvik/forSaurabh/manifest.csv")

IDS = shape_df["CellId"].unique()

PLANAR_GRADIENT_DICT = {
    "planar_gradient": {
        "description": "gradient based on distance from a plane",
        "weight_mode": "exponential",
        "pick_mode": "rnd",
        "mode": "vector",
        "invert": "distance",
        "mode_settings": {"direction": [0.0, 1.0, 0.0], "center": [0, 0, 0]},
        "weight_mode_settings": {"decay_length": 0.01},
    }
}

RADIAL_GRADIENT_DICT = {
    "radial_gradient": {
        "description": "radial gradient from the center",
        "mode": "radial",
        "pick_mode": "rnd",
        "weight_mode": "exponential",
        "weight_mode_settings": {"decay_length": 0.01},
    }
}

SURFACE_GRADIENT_DICT = {
    "surface_gradient": {
        "description": "gradient based on distance from a surface",
        "pick_mode": "rnd",
        "mode": "surface",
        "mode_settings": {"object": "nucleus", "scale_to_next_surface": False},
        "weight_mode": "exponential",
        "weight_mode_settings": {"decay_length": 0.01},
    }
}


def update_shape_path_and_save_recipe(
    shape_ids, shape_df, clust, contents_clust, base_version, output_path
):
    # update nucleus meshes
    for this_id in shape_ids:
        this_row = shape_df.loc[shape_df["CellId"] == this_id]
        this_row = this_row.loc[this_row["angle"] == 0]

        grid_file_path = BASE_GRID_PATH + f"{this_id}_grid.dat"
        contents_clust["grid_file_path"] = f"{grid_file_path}"

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


def create_recipe_files(
    recipe_template_path,
    shape_df,
    output_path="./tmp/",
    num_clusters=MAX_NUM_CLUSTERS,
    shape_ids=IDS,
):
    # read json
    with open(recipe_template_path) as j:
        contents = json.load(j)
    contents["objects"]["mean_nucleus"]["representations"]["mesh"]["path"] = MESH_PATH

    for clust in np.arange(1, num_clusters + 1):
        print(f"Creating files for {clust} clusters")
        contents_clust = contents.copy()

        if clust == 1:
            contents_clust["composition"]["nucleus"]["regions"]["interior"] = [
                {"object": "seed", "count": int(clust)},
            ]

            # put cluster randomly
            version = "random"
            update_shape_path_and_save_recipe(
                shape_ids, shape_df, clust, contents_clust, version, output_path
            )

            # put cluster at center
            contents_clust["gradients"] = RADIAL_GRADIENT_DICT
            contents_clust["objects"]["seed"]["packing_mode"] = "gradient"
            contents_clust["objects"]["seed"]["gradient"] = "radial_gradient"

            version = "center"
            update_shape_path_and_save_recipe(
                shape_ids, shape_df, clust, contents_clust, version, output_path
            )

        if clust == 2:
            contents_clust["composition"]["nucleus"]["regions"]["interior"] = [
                {"object": "seed", "count": int(clust)}
            ]

            # put clusters at end points
            contents_clust["gradients"] = PLANAR_GRADIENT_DICT
            contents_clust["objects"]["seed"]["packing_mode"] = "gradient"
            contents_clust["objects"]["seed"]["gradient"] = "planar_gradient"

            version = "polar"
            update_shape_path_and_save_recipe(
                shape_ids, shape_df, clust, contents_clust, version, output_path
            )

        if clust == 3:
            # put 1 cluster at center
            contents_clust["gradients"] = {
                **RADIAL_GRADIENT_DICT,
                **SURFACE_GRADIENT_DICT,
            }
            contents_clust["objects"]["seed_center"] = contents_clust["objects"][
                "seed"
            ].copy()
            contents_clust["objects"]["seed_center"]["packing_mode"] = "gradient"
            contents_clust["objects"]["seed_center"]["gradient"] = "radial_gradient"

            # put 2 clusters along the nuclear boundary
            contents_clust["objects"]["seed_surface"] = contents_clust["objects"][
                "seed"
            ].copy()
            contents_clust["objects"]["seed_surface"]["packing_mode"] = "gradient"
            contents_clust["objects"]["seed_surface"]["gradient"] = "surface_gradient"

            # remove old seed
            contents_clust["objects"].pop("seed")

            # update composition
            contents_clust["composition"]["nucleus"]["regions"]["interior"] = [
                {"object": "seed_center", "count": 1},
                {"object": "seed_surface", "count": 2},
            ]

            version = "center_surface"
            update_shape_path_and_save_recipe(
                shape_ids, shape_df, clust, contents_clust, version, output_path
            )

        if clust > 3:
            contents_clust["composition"]["nucleus"]["regions"]["interior"] = [
                {"object": "seed", "count": int(clust)},
            ]

            # put cluster randomly
            version = "random"
            update_shape_path_and_save_recipe(
                shape_ids, shape_df, clust, contents_clust, version, output_path
            )


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
        with open("failed_recipes.log", "a") as f:
            f.write(f"{recipe_path}\n")
            f.write(f"{e}\n")
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
