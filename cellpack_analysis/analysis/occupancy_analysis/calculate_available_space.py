# %% [markdown]
"""
# Calculate available space for a structure.

This script calculates the available space for intracellular structures by discretizing
the space into a grid and calculating the distance of each grid point to the
nearest nucleus and membrane.
The distance is calculated using the signed distance function from the trimesh library and the
distances are saved in a grid directory for each cell_id.
Distances are normalized by the cell diameter and saved in the grid directory.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.mesh_tools import calculate_grid_distances

log = logging.getLogger(__name__)
# %% set structure id
STRUCTURE_ID = "mean"  # SLC25A17: peroxisomes, RAB5A: early endosomes
# %% set up parameters for grid
SPACING = 2
# %% set file paths and setup parameters
base_datadir = get_project_root() / "data"
log.info(f"Data directory: {base_datadir}")

# %% select cell_ids to use
use_mean_shape = True
if use_mean_shape:
    cell_ids_to_use = ["mean"]
else:
    df_cell_id = pd.read_csv("s3://cellpack-analysis-data/all_cell_ids.csv")
    df_struct = df_cell_id.loc[df_cell_id["structure_name"] == STRUCTURE_ID]
    cell_ids_to_use = df_struct.loc[df_struct["8dsphere"], "CellId"].tolist()
mesh_folder = base_datadir / f"structure_data/{STRUCTURE_ID}/meshes/"
log.info(f"Using {len(cell_ids_to_use)} cell_ids")
# %% get meshes for cell_ids used
# cell_ids_to_use = [cell_ids_to_use[0]]
cell_id_list = []
nuc_meshes_to_use = []
mem_meshes_to_use = []
for cell_id in cell_ids_to_use:
    nuc_mesh_path = mesh_folder / f"nuc_mesh_{cell_id}.obj"
    mem_mesh_path = mesh_folder / f"mem_mesh_{cell_id}.obj"
    if nuc_mesh_path.exists() and mem_mesh_path.exists():
        cell_id_list.append(cell_id)
        nuc_meshes_to_use.append(nuc_mesh_path)
        mem_meshes_to_use.append(mem_mesh_path)
log.info(f"Found {len(nuc_meshes_to_use)} meshes")
# %% set up grid results directory
grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances/"
grid_dir.mkdir(exist_ok=True, parents=True)
# %% run the workflow
save_dir = None
PARALLEL = False
recalculate = False
calc_nuc_distances = True
calc_mem_distances = True
calc_z_distances = True
calc_scaled_nuc_distances = True
chunk_size = 20000
if PARALLEL:
    num_cores = 8
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for i in range(len(nuc_meshes_to_use)):
            results.append(
                executor.submit(
                    calculate_grid_distances,
                    nuc_meshes_to_use[i],
                    mem_meshes_to_use[i],
                    cell_id_list[i],
                    SPACING,
                    grid_dir,
                    recalculate,
                    calc_nuc_distances,
                    calc_mem_distances,
                    calc_z_distances,
                    calc_scaled_nuc_distances,
                    chunk_size,
                )
            )

        with tqdm(total=len(results), desc="CellIDs done") as pbar:
            for result in as_completed(results):
                if result.result:
                    pbar.update(1)
else:
    results = []
    for i in tqdm(range(len(nuc_meshes_to_use)), desc="CellIDs done"):
        results.append(
            calculate_grid_distances(
                nuc_mesh_path=nuc_meshes_to_use[i],
                mem_mesh_path=mem_meshes_to_use[i],
                cell_id=cell_id_list[i],
                spacing=SPACING,
                save_dir=grid_dir,
                recalculate=recalculate,
                calc_nuc_distances=calc_nuc_distances,
                calc_mem_distances=calc_mem_distances,
                calc_z_distances=calc_z_distances,
                calc_scaled_nuc_distances=calc_scaled_nuc_distances,
                chunk_size=chunk_size,
            )
        )

# %%
