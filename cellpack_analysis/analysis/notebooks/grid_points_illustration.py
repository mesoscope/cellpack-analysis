# %% [markdown]
# # Visualize grid points
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import trimesh
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import sample_cell_ids_for_structure
from cellpack_analysis.lib.mesh_tools import get_list_of_grid_points, round_away_from_zero

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Set parameters and file paths
STRUCTURE_ID = "SLC25A17"  # SLC25A17: peroxisomes, RAB5A: early endosomes
STRUCTURE_NAME = "peroxisome"

project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

figures_dir = base_results_dir / "occupancy_analysis/available_space"
figures_dir.mkdir(exist_ok=True, parents=True)

grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances"
grid_dir.mkdir(exist_ok=True, parents=True)

logger.info(f"Figures directory: {figures_dir}")
logger.info(f"Grid directory: {grid_dir}")
# %% [markdown]
# ## Select cell_ids to use
mesh_folder = base_datadir / f"structure_data/{STRUCTURE_ID}/meshes/"
cell_ids_to_use = sample_cell_ids_for_structure(
    structure_id=STRUCTURE_ID, num_cells=10, dsphere=True
)
cell_id = "834350"
logger.info(f"Using cell id: {cell_id} for grid spacing illustration")

# %% [markdown]
# ## Get meshes for cell_ids used
nuc_mesh_path = mesh_folder / f"nuc_mesh_{cell_id}.obj"
mem_mesh_path = mesh_folder / f"mem_mesh_{cell_id}.obj"
nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
mem_mesh = trimesh.load_mesh(mem_mesh_path)

# %% [markdown]
# ## Get grid points
SPACING = 1.7
bounds = mem_mesh.bounds
bounding_box = round_away_from_zero(bounds)
all_grid_points = get_list_of_grid_points(bounding_box, SPACING)
logger.info(f"Total grid points to check: {all_grid_points.shape[0]}")

# %% [markdown]
# ## Select z slice to plot
z_values = all_grid_points[:, 2]
median_z = np.median(z_values)
point_indexes = np.isclose(z_values, median_z, atol=SPACING / 2)
logger.info(
    f"Selected z slice at z={median_z * PIXEL_SIZE_IN_UM:.2f} (\u03bcm) "
    f"with {np.sum(point_indexes)} points"
)
grid_points_center_slice = all_grid_points[point_indexes]

# %% [markdown]
# ## Run inside-outside check
logger.info("Calculating nuc inside check")
inside_nuc = nuc_mesh.contains(grid_points_center_slice)
logger.info("Calculating mem inside check")
inside_mem = mem_mesh.contains(grid_points_center_slice)

# %% [markdown]
# ## Get points inside membrane but outside nucleus
inside_mem_outside_nuc = inside_mem & ~inside_nuc

# %% [markdown]
# ## Calculate distance from nucleus for points inside membrane but outside nucleus
nuc_distances = nuc_mesh.nearest.signed_distance(grid_points_center_slice[inside_mem_outside_nuc])
nuc_distances_um = -nuc_distances * PIXEL_SIZE_IN_UM
scaled_nuc_distances = nuc_distances_um / np.max(nuc_distances_um)
decay_length = 0.1
nuc_weights = np.exp(-scaled_nuc_distances / decay_length)
scaled_nuc_weights = nuc_weights / np.max(nuc_weights)

# %% [markdown]
# ## Calculate distance from membrane for points inside membrane but outside nucleus
mem_distances = mem_mesh.nearest.signed_distance(grid_points_center_slice[inside_mem_outside_nuc])
mem_distances_um = mem_distances * PIXEL_SIZE_IN_UM
scaled_mem_distances = mem_distances_um / np.max(mem_distances_um)
decay_length = 0.1
mem_weights = np.exp(-scaled_mem_distances / decay_length)
scaled_mem_weights = mem_weights / np.max(mem_weights)

# %% [markdown]
# ## Uniform weights for points
uniform_weights = np.ones(np.count_nonzero(inside_mem_outside_nuc))
# %% [markdown]
# ## Plot center slice
file_name = f"nucleus_distance_{cell_id}"
plt.rcParams.update({"font.size": 8})

grid_points_um = grid_points_center_slice * PIXEL_SIZE_IN_UM
centroid = np.mean(grid_points_um, axis=0)
cyto_points = grid_points_um[inside_mem_outside_nuc]
nuc_points = grid_points_um[inside_nuc]
dot_size = 2
custom_cmap = LinearSegmentedColormap.from_list("cyan_to_magenta", ["cyan", "magenta"])

color_var = nuc_distances_um
cmap = custom_cmap
cbar_label = "Distance from nucleus (\u03bcm)"
# cbar_label = "Uniform weight"

fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
# Plot cytoplasm points with weight coloring
sns.scatterplot(
    x=grid_points_um[inside_mem_outside_nuc, 0] - centroid[0],
    y=grid_points_um[inside_mem_outside_nuc, 1] - centroid[1],
    c=color_var,
    cmap=cmap,
    s=dot_size,
)
# Plot nucleus points in gray
sns.scatterplot(
    x=grid_points_um[inside_nuc, 0] - centroid[0],
    y=grid_points_um[inside_nuc, 1] - centroid[1],
    c="gray",
    s=dot_size,
)
ax.set_xlabel("x (\u03bcm)")
ax.set_ylabel("y (\u03bcm)")
ax.set_aspect("equal")
ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
sns.despine(ax=ax)
cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
cbar.set_label(cbar_label, rotation=270, labelpad=15)
plt.show()

fig.savefig(figures_dir / f"{file_name}.pdf", bbox_inches="tight")

# %%
