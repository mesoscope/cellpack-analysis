# %% [markdown]
# # Visualize grid points
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import trimesh
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.label_tables import COLOR_PALETTE
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
cell_ids_to_use = get_cell_id_list_for_structure(
    structure_id=STRUCTURE_ID, dsphere=True, load_local=True
)
cell_id = cell_ids_to_use[0]
logger.info(f"Using cell id: {cell_id} for grid spacing illustration")
# %% [markdown]
# ## Get meshes for cell_ids used
nuc_mesh_path = mesh_folder / f"nuc_mesh_{cell_id}.obj"
mem_mesh_path = mesh_folder / f"mem_mesh_{cell_id}.obj"
nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
mem_mesh = trimesh.load_mesh(mem_mesh_path)

# %% [markdown]
# ## Get grid points
SPACING = 5
bounds = mem_mesh.bounds
bounding_box = round_away_from_zero(bounds)
all_points = get_list_of_grid_points(bounding_box, SPACING)

# %% [markdown]
# ## Run inside-outside check
logger.info("Calculating nuc inside check")
inside_nuc = nuc_mesh.contains(all_points)
logger.info("Calculating mem inside check")
inside_mem = mem_mesh.contains(all_points)
# %% [markdown]
# ## Plot grid points
inside_mem_outside_nuc = inside_mem & ~inside_nuc
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
all_points_scaled = all_points * PIXEL_SIZE_IN_UM
centroid = np.mean(all_points_scaled, axis=0)
dot_size = 10
sns.scatterplot(
    x=all_points_scaled[inside_mem_outside_nuc, 0] - centroid[0],
    y=all_points_scaled[inside_mem_outside_nuc, 1] - centroid[1],
    c=COLOR_PALETTE["membrane"],
    label="Points inside membrane",
    s=dot_size,
)
sns.scatterplot(
    x=all_points_scaled[inside_nuc, 0] - centroid[0],
    y=all_points_scaled[inside_nuc, 1] - centroid[1],
    c=COLOR_PALETTE["nucleus"],
    label="Points inside nucleus",
    s=dot_size,
)
ax.set_xlabel("x (\u03bcm)")
ax.set_ylabel("y (\u03bcm)")
ax.set_aspect("equal")
ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
sns.despine(ax=ax)
plt.show()
file_name = "grid_points"
fig.savefig(figures_dir / f"{file_name}.svg", bbox_inches="tight")
