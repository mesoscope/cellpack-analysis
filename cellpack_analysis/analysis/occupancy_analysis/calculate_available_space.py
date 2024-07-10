# %% [markdown]
"""
# Calculate available space for a structure

This script calculates the available space for intracellular structures by discretizing
the space into a grid and calculating the distance of each grid point to the
nearest nucleus and membrane.
The distance is calculated using the signed distance function from the trimesh library and the
distances are saved in a grid directory for each cellid.
Distances are normalized by the cell diameter and saved in the grid directory.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cellpack_analysis.lib.mesh_tools import calculate_grid_distances

# %% set structure id
STRUCTURE_ID = "RAB5A"  # SLC25A17: peroxisomes, RAB5A: early endosomes
# %% set up parameters for grid
SPACING = 1.5
# %% set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
print(f"Data directory: {base_datadir}")

# %% select cellids to use
use_mean_shape = True
if use_mean_shape:
    mesh_folder = base_datadir / "average_shape_meshes"
    cellids_to_use = ["mean"]
else:
    mesh_folder = base_datadir / f"structure_data/{STRUCTURE_ID}/meshes/"
    df_cellid = pd.read_csv("s3://cellpack-analysis-data/all_cellids.csv")
    df_struct = df_cellid.loc[df_cellid["structure_name"] == STRUCTURE_ID]
    cellids_to_use = df_struct.loc[df_struct["8dsphere"], "CellId"]
print(f"Using {len(cellids_to_use)} cellids")
# %% get meshes for cellids used
cellid_list = []
nuc_meshes_to_use = []
mem_meshes_to_use = []
for cellid in cellids_to_use:
    nuc_mesh = mesh_folder / f"nuc_mesh_{cellid}.obj"
    mem_mesh = mesh_folder / f"mem_mesh_{cellid}.obj"
    if nuc_mesh.exists() and mem_mesh.exists():
        cellid_list.append(cellid)
        nuc_meshes_to_use.append(nuc_mesh)
        mem_meshes_to_use.append(mem_mesh)
print(f"Found {len(nuc_meshes_to_use)} meshes")
# %% set up grid results directory
grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances"
grid_dir.mkdir(exist_ok=True, parents=True)
# %% run in parallel
PARALLEL = False
recalculate = False
calc_nuc_distances = True
calc_mem_distances = True
calc_z_distances = True
calc_inside_mem = True
chunk_size = 50000
if PARALLEL:
    num_cores = 4
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i in range(len(nuc_meshes_to_use)):
            futures.append(
                executor.submit(
                    calculate_grid_distances,
                    nuc_meshes_to_use[i],
                    mem_meshes_to_use[i],
                    cellid_list[i],
                    SPACING,
                    grid_dir,
                    recalculate,
                    calc_nuc_distances,
                    calc_mem_distances,
                    calc_z_distances,
                    chunk_size,
                )
            )

        with tqdm(total=len(futures), desc="CellIDs done") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
else:
    results = []
    for i in tqdm(range(len(nuc_meshes_to_use)), desc="CellIDs done"):
        results.append(
            calculate_grid_distances(
                nuc_mesh_path=nuc_meshes_to_use[i],
                mem_mesh_path=mem_meshes_to_use[i],
                cellid=cellid_list[i],
                spacing=SPACING,
                save_dir=grid_dir,
                recalculate=recalculate,
                calc_nuc_distances=calc_nuc_distances,
                calc_mem_distances=calc_mem_distances,
                calc_z_distances=calc_z_distances,
                chunk_size=chunk_size,
            )
        )
