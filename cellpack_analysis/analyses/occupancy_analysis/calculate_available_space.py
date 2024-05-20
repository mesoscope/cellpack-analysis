# %%
import pickle
import numpy as np
import trimesh
import trimesh.proximity
import matplotlib.pyplot as plt

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import seaborn as sns

import gc
plt.rcParams.update({"font.size": 14})

# %% set structure id
STRUCTURE_ID = "SLC25A17"  # peroxisomes
# %% set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

results_dir = base_results_dir / "stochastic_variation_analysis/rules/"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)
print(f"Results directory: {results_dir}")
print(f"Figures directory: {figures_dir}")

# %% set up grid
SPACING = 2
bounding_box = np.array([[-190, -224, -66], [172, 216, 74]])
grid = np.mgrid[
    bounding_box[0, 0] : bounding_box[1, 0] : SPACING,
    bounding_box[0, 1] : bounding_box[1, 1] : SPACING,
    bounding_box[0, 2] : bounding_box[1, 2] : SPACING,
]
all_points = grid.reshape(3, -1).T

# %%
CHUNK_SIZE = 50000


def get_nuc_distances(
    nuc_mesh_path, mem_mesh_path, cellid, points, save_dir=None, skip_completed=True
):
    # check if distances already calculated
    if skip_completed and save_dir is not None:
        file_name = save_dir / f"nuc_distances_{cellid}.npy"
        if file_name.exists():
            return np.load(file_name), np.load(save_dir / f"mem_distances_{cellid}.npy")

    # load meshes
    nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
    mem_mesh = trimesh.load_mesh(mem_mesh_path)

    # get points inside mem using chunking
    inside_mem = np.empty(len(points), dtype=bool)
    for i in range(0, len(points), CHUNK_SIZE):
        inside_mem[i : i + CHUNK_SIZE] = mem_mesh.contains(points[i : i + CHUNK_SIZE])
    inside_mem_points = points[inside_mem]
    del points

    gc.collect()

    # get points inside nuc
    # inside_nuc = np.empty(len(inside_mem_points), dtype=bool)
    # for i in range(0, len(inside_mem_points), CHUNK_SIZE):
    #     inside_nuc[i : i + CHUNK_SIZE] = nuc_mesh.contains(
    #         inside_mem_points[i : i + CHUNK_SIZE]
    #     )

    # # get points inside mem but outside nuc
    # inter_points = inside_mem_points[~inside_nuc]

    # get distances to nuc_mesh
    nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, inside_mem_points)
    mem_distances = trimesh.proximity.signed_distance(mem_mesh, inside_mem_points)
    if save_dir is not None:
        np.save(save_dir / f"nuc_distances_{cellid}.npy", nuc_distances)
        np.save(save_dir / f"mem_distances_{cellid}.npy", mem_distances)
    return nuc_distances, mem_distances


# %% select cellids to use
mesh_folder = base_datadir / "structure_data/SLC25A17/meshes/"
df_cellid = pd.read_csv("s3://cellpack-analysis-data/all_cellids.csv")
df_pex = df_cellid.loc[df_cellid["structure_name"] == "SLC25A17"]
cellids_to_use = df_pex.loc[df_pex["8dsphere"], "CellId"]
print(f"Using {len(cellids_to_use)} cellids")
# %% use mean cell and nuclear shape
mesh_folder = base_datadir / "average_shape_meshes"
cellids_to_use = ["mean"]
# %%
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
print(f"Found {len(cellid_list)} meshes")
# %% set up grid results directory
grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances"
grid_dir.mkdir(exist_ok=True, parents=True)
# %% run in parallel
PARALLEL = False
skip_completed = False
if PARALLEL:
    num_cores = 4
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i in range(len(nuc_meshes_to_use)):
            futures.append(
                executor.submit(
                    get_nuc_distances,
                    nuc_meshes_to_use[i],
                    mem_meshes_to_use[i],
                    cellid_list[i],
                    all_points,
                    grid_dir,
                    skip_completed,
                )
            )

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
else:
    results = []
    for i in tqdm(range(len(nuc_meshes_to_use))):
        results.append(
            get_nuc_distances(
                nuc_meshes_to_use[i],
                mem_meshes_to_use[i],
                cellid_list[i],
                all_points,
                grid_dir,
            )
        )
# %% load meshes
nuc_mesh = trimesh.load_mesh(nuc_meshes_to_use[0])
mem_mesh = trimesh.load_mesh(mem_meshes_to_use[0])
# %% use average mesh
nuc_mesh = trimesh.load_mesh(
    base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
)
mem_mesh = trimesh.load_mesh(
    base_datadir / "average_shape_meshes/mem_mesh_mean.obj"
)
# %% try explicit inside-outside check
print("Calculating nuc inside check")
inside_nuc = nuc_mesh.contains(all_points)
print("Calculating mem inside check")
inside_mem = mem_mesh.contains(all_points)

# %% find points inside mem but outside nuc
inside_mem_outside_nuc = inside_mem & ~inside_nuc
# %% plot grid point scatter plot
fig, ax = plt.subplots(dpi=300)
ax.scatter(
    all_points[inside_mem_outside_nuc, 0],
    all_points[inside_mem_outside_nuc, 1],
    c="magenta",
    label="inside mem outside nuc",
    s=0.1,
    alpha=0.7,
)
ax.scatter(
    all_points[inside_nuc, 0],
    all_points[inside_nuc, 1],
    c="cyan",
    label="inside nuc",
    s=0.1,
    alpha=0.7,
)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
ax.set_aspect("equal")
plt.show()

# %% load mesh information
file_path = grid_dir.parent / "mesh_information.dat"
with open(file_path, "rb") as f:
    mesh_information_dict = pickle.load(f)
# %% load saved distances
# normalization = "cell_diameter"
normalization = None
nuc_distances = []
mem_distances = []
for cellid in cellids_to_use:
    normalization_factor = mesh_information_dict[str(cellid)].get(normalization, 1)
    nuc_distances.append(
        np.load(grid_dir / f"nuc_distances_{cellid}.npy") / normalization_factor
    )
    mem_distances.append(
        np.load(grid_dir / f"mem_distances_{cellid}.npy") / normalization_factor
    )

# %% plot distance distribution kdeplot
fig, ax = plt.subplots(dpi=300)
cmap = plt.get_cmap("jet", len(nuc_distances))
pix_size = 0.108
all_nuc_distances = []
for i in tqdm(range(len(nuc_distances))):
    distances_to_plot = nuc_distances[i] * pix_size
    distances_to_plot = distances_to_plot[distances_to_plot > 0]
    sns.kdeplot(distances_to_plot, ax=ax, color=cmap(i + 1), alpha=0.3)
    all_nuc_distances.append(distances_to_plot)
all_nuc_distances = np.concatenate(all_nuc_distances)
sns.kdeplot(all_nuc_distances, ax=ax, color=cmap(0), linewidth=2)
mean_distance = np.mean(all_nuc_distances)
ax.axvline(mean_distance, color="black", linestyle="--")
ax.set_title(f"Distance to nucleus\nMean: {mean_distance:.2f}\u03BCm")
ax.set_xlabel("Distance (\u03BCm)")
ax.set_ylabel("Probability density")
plt.show()
# %% plot distance distribution histogram for mean shape
fig, ax = plt.subplots(dpi=300)
pix_size = 0.108
nuc_distances = np.array(nuc_distances)
ax.hist(nuc_distances[0] * 0.108)
ax.set_xlabel("Distance (\u03BCm)")
ax.set_ylabel("Number of points")
ax.set_title("Distance to nucleus")
plt.show()



# %%
