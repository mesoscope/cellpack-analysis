# %% [markdown]
"""
## Analyze and visualize available space distribution
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
import trimesh
from tqdm import tqdm

from cellpack_analysis.lib.mesh_tools import (
    get_list_of_grid_points,
    round_away_from_zero,
)

# %% set pixel size
PIX_SIZE = 0.108  # um per pixel

# %% set structure id
STRUCTURE_ID = "SLC25A17"  # SLC25A17: peroxisomes, RAB5A: early endosomes
# %% set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

results_dir = base_results_dir / f"stochastic_variation_analysis/{STRUCTURE_ID}/rules/"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances"
grid_dir.mkdir(exist_ok=True, parents=True)

print(f"Results directory: {results_dir}")
print(f"Figures directory: {figures_dir}")
print(f"Grid directory: {grid_dir}")

# %%
use_mean_shape = False

# %% select cellids to use
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

# %% load meshes
if use_mean_shape:
    nuc_mesh = trimesh.load_mesh(
        base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
    )
    mem_mesh = trimesh.load_mesh(
        base_datadir / "average_shape_meshes/mem_mesh_mean.obj"
    )
else:
    nuc_mesh = trimesh.load_mesh(nuc_meshes_to_use[0])
    mem_mesh = trimesh.load_mesh(mem_meshes_to_use[0])

# %% set up grid
SPACING = 1
bounds = mem_mesh.bounds
bounding_box = round_away_from_zero(bounds)
all_points = get_list_of_grid_points(bounding_box, SPACING)
# %% explicit inside-outside check
print("Calculating nuc inside check")
inside_nuc = nuc_mesh.contains(all_points)
print("Calculating mem inside check")
inside_mem = mem_mesh.contains(all_points)

# %% find points inside mem but outside nuc
inside_mem_outside_nuc = inside_mem & ~inside_nuc
# %% plot grid point scatter plot
fig, ax = plt.subplots(dpi=300)
all_points_scaled = all_points * PIX_SIZE
ax.scatter(
    all_points_scaled[inside_mem_outside_nuc, 0],
    all_points_scaled[inside_mem_outside_nuc, 1],
    c="magenta",
    label="inside mem outside nuc",
    s=0.1,
    alpha=0.7,
)
ax.scatter(
    all_points_scaled[inside_nuc, 0],
    all_points_scaled[inside_nuc, 1],
    c="cyan",
    label="inside nuc",
    s=0.1,
    alpha=0.7,
)
ax.set_xlabel("x (\u03BCm)")
ax.set_ylabel("y (\u03BCm)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
ax.set_aspect("equal")
plt.show()
fig.savefig(figures_dir / "grid_points.png", bbox_inches="tight")

# %% load mesh information
file_path = grid_dir.parent / "mesh_information.dat"
with open(file_path, "rb") as f:
    mesh_information_dict = pickle.load(f)
# %% load saved distances
# normalization = "cell_diameter"
normalization = None
nuc_distances = []
mem_distances = []
for cellid in tqdm(cellids_to_use):
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
all_nuc_distances = []
for i in tqdm(range(len(nuc_distances))):
    distances_to_plot = nuc_distances[i] * PIX_SIZE
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
# %% plot distance distribution histogram
fig, ax = plt.subplots(dpi=300)
nuc_distances = np.array(nuc_distances)
distances_to_plot = nuc_distances[0][nuc_distances[0] > 0] * PIX_SIZE
ax.hist(distances_to_plot, bins=100)
ax.set_xlabel("Distance (\u03BCm)")
ax.set_ylabel("Number of points")
ax.set_title("Distance to nucleus")
plt.show()
