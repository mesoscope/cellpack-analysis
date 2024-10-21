# %% [markdown]
# # Analyze and visualize available space distribution
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import trimesh
from tqdm import tqdm

from cellpack_analysis.lib.mesh_tools import (
    get_list_of_grid_points,
    round_away_from_zero,
)

log = logging.getLogger(__name__)

# %% [markdown]
# ## Set parameters and file paths
MEAN_SHAPE = False

PIX_SIZE = 0.108  # um per pixel

STRUCTURE_ID = "SEC61B"  # SLC25A17: peroxisomes, RAB5A: early endosomes
STRUCTURE_NAME = "ER_peroxisome"

base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

results_dir = (
    base_results_dir / f"stochastic_variation_analysis/{STRUCTURE_NAME}/rules/"
)
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances"
grid_dir.mkdir(exist_ok=True, parents=True)

log.info(f"Results directory: {results_dir}")
log.info(f"Figures directory: {figures_dir}")
log.info(f"Grid directory: {grid_dir}")

# %% [markdown]
# ## Select cellids to use
if MEAN_SHAPE:
    mesh_folder = base_datadir / "average_shape_meshes"
    cellids_to_use = ["mean"]
else:
    mesh_folder = base_datadir / f"structure_data/{STRUCTURE_ID}/meshes/"
    df_cellid = pd.read_csv("s3://cellpack-analysis-data/all_cellids.csv")
    df_struct = df_cellid.loc[df_cellid["structure_name"] == STRUCTURE_ID]
    cellids_to_use = df_struct.loc[df_struct["8dsphere"], "CellId"].tolist()
log.info(f"Using {len(cellids_to_use)} cellids")
# %% [markdown]
# ## Get meshes for cellids used
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
log.info(f"Found {len(nuc_meshes_to_use)} meshes")

# %% [markdown]
# # Grid spacing illustration
# ## Load meshes
if MEAN_SHAPE:
    nuc_mesh = trimesh.load_mesh(
        base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
    )
    mem_mesh = trimesh.load_mesh(
        base_datadir / "average_shape_meshes/mem_mesh_mean.obj"
    )
else:
    nuc_mesh = trimesh.load_mesh(nuc_meshes_to_use[0])
    mem_mesh = trimesh.load_mesh(mem_meshes_to_use[0])

# %% [markdown]
# ## Get grid points
SPACING = 5
bounds = mem_mesh.bounds
bounding_box = round_away_from_zero(bounds)
all_points = get_list_of_grid_points(bounding_box, SPACING)

# %% [markdown]
# ## Run inside-outside check
log.info("Calculating nuc inside check")
inside_nuc = nuc_mesh.contains(all_points)
log.info("Calculating mem inside check")
inside_mem = mem_mesh.contains(all_points)
# %% [markdown]
# ## Plot grid points
inside_mem_outside_nuc = inside_mem & ~inside_nuc

fig, ax = plt.subplots(dpi=300)
all_points_scaled = all_points * PIX_SIZE
centroid = np.mean(all_points_scaled, axis=0)
ax.scatter(
    all_points_scaled[inside_mem_outside_nuc, 0] - centroid[0],
    all_points_scaled[inside_mem_outside_nuc, 1] - centroid[1],
    c="magenta",
    label="Available points",
    s=0.5,
    alpha=1,
)
ax.scatter(
    all_points_scaled[inside_nuc, 0] - centroid[0],
    all_points_scaled[inside_nuc, 1] - centroid[1],
    c="cyan",
    label="inside nuc",
    s=0.5,
    alpha=1,
)
ax.set_xlabel("x (\u03bcm)")
ax.set_ylabel("y (\u03bcm)")
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
ax.set_aspect("equal")
# ax.set_aspect(1.3)
plt.show()
file_name = "grid_points"
if MEAN_SHAPE:
    file_name += "_mean"
fig.savefig(figures_dir / f"{file_name}.png", bbox_inches="tight")

# %% [markdown]
# # Plot distance from nucleus for all shapes
# ## Load mesh information
file_path = grid_dir.parent / "mesh_information.dat"
with open(file_path, "rb") as f:
    mesh_information_dict = pickle.load(f)
# %% [markdown]
# ## Load distances
# normalization = "cell_diameter"
normalization = None
nuc_distances = []
mem_distances = []
for cellid in tqdm(cellids_to_use):
    if normalization is not None:
        normalization_factor = mesh_information_dict[str(cellid)].get(normalization, 1)
    else:
        normalization_factor = 1
    nuc_distances.append(
        np.load(grid_dir / f"nuc_distances_{cellid}.npy") / normalization_factor
    )
    mem_distances.append(
        np.load(grid_dir / f"mem_distances_{cellid}.npy") / normalization_factor
    )

# %% [markdown]
# ## Plot distance distribution as kde
fig, ax = plt.subplots(dpi=300)
cmap = plt.get_cmap("viridis", len(nuc_distances))
color_inds = np.random.permutation(len(nuc_distances))
all_nuc_distances = []
for i in tqdm(range(len(nuc_distances))):
    distances_to_plot = nuc_distances[i] * PIX_SIZE
    distances_to_plot = distances_to_plot[distances_to_plot > 0]
    sns.kdeplot(
        distances_to_plot, ax=ax, color=cmap(color_inds[i]), alpha=0.4, linewidth=1
    )
    all_nuc_distances.append(distances_to_plot)
    # break
all_nuc_distances = np.concatenate(all_nuc_distances)
sns.kdeplot(all_nuc_distances, ax=ax, color="k", linewidth=3)
mean_distance = np.mean(all_nuc_distances).item()
ax.axvline(mean_distance, color="black", linestyle="--")
# %% [markdown]
# Adjust plot
# xlim = ax.get_xlim()
plt.rcdefaults()
ax.set_xlim((-1, 10))
ax.set_title(f"Distance to nucleus\nMean: {mean_distance:.2f}\u03bcm")
ax.set_xlabel("Distance (\u03bcm)")
ax.set_ylabel("Probability density")
file_name = "nuc_distance_kde"
fig.savefig(figures_dir / f"{file_name}.png", bbox_inches="tight")
fig.savefig(figures_dir / f"{file_name}.svg")
# plt.show()
fig
# %% [markdown]
# ## Plot distance distribution histogram
fig, ax = plt.subplots(dpi=300)
nuc_distances = np.array(nuc_distances)
distances_to_plot = nuc_distances[0][nuc_distances[0] > 0] * PIX_SIZE
ax.hist(distances_to_plot, bins=100)
ax.set_xlabel("Distance (\u03bcm)")
ax.set_ylabel("Number of points")
ax.set_title("Distance to nucleus")
file_name = "nuc_distance_hist"
fig.savefig(figures_dir / f"{file_name}.png", bbox_inches="tight")
plt.show()

# %%
