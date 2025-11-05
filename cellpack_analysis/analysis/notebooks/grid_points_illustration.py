# %% [markdown]
# # Visualize grid points
import logging

import matplotlib.pyplot as plt
import trimesh

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import sample_cell_ids_for_structure
from cellpack_analysis.lib.mesh_tools import (
    get_distances_from_mesh,
    get_grid_points_slice,
    get_inside_outside_check,
    get_list_of_grid_points,
    get_weights_from_distances,
    round_away_from_zero,
)
from cellpack_analysis.lib.visualization import plot_grid_points_slice

logger = logging.getLogger(__name__)
plt.rcParams["font.size"] = 6

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
SPACING = 1.7063
bounds = mem_mesh.bounds
bounding_box = round_away_from_zero(bounds)
all_grid_points = get_list_of_grid_points(bounding_box, SPACING)
logger.info(f"Total grid points to check: {all_grid_points.shape[0]}")

# %% [markdown]
# ## Get grid points for slice
grid_points_slice = {}
inside_nuc = {}
inside_mem = {}
inside_mem_outside_nuc = {}
nuc_distances_um = {}
mem_distances_um = {}
z_distances_um = {}
nuc_weights = {}
mem_weights = {}
z_weights = {}

for projection_axis in ["x", "y", "z"]:
    grid_points_slice[projection_axis] = get_grid_points_slice(
        all_grid_points=all_grid_points,
        projection_axis=projection_axis,
        spacing=SPACING,
    )
    (
        inside_nuc[projection_axis],
        inside_mem[projection_axis],
        inside_mem_outside_nuc[projection_axis],
    ) = get_inside_outside_check(
        nuc_mesh=nuc_mesh,
        mem_mesh=mem_mesh,
        grid_points_slice=grid_points_slice[projection_axis],
    )

    nuc_distances_um[projection_axis] = get_distances_from_mesh(
        points=grid_points_slice[projection_axis][inside_mem_outside_nuc[projection_axis]],
        mesh=nuc_mesh,
        invert=True,
    )
    nuc_weights[projection_axis] = get_weights_from_distances(
        nuc_distances_um[projection_axis], decay_length=0.1
    )

    mem_distances_um[projection_axis] = get_distances_from_mesh(
        points=grid_points_slice[projection_axis][inside_mem_outside_nuc[projection_axis]],
        mesh=mem_mesh,
    )
    mem_weights[projection_axis] = get_weights_from_distances(
        mem_distances_um[projection_axis], decay_length=0.1
    )

    z_distances_um[projection_axis] = (
        grid_points_slice[projection_axis][inside_mem_outside_nuc[projection_axis], 2]
        * PIXEL_SIZE_IN_UM
    )
    z_weights[projection_axis] = get_weights_from_distances(
        z_distances_um[projection_axis], decay_length=0.1
    )
# %% [markdown]
# ## Plot nucleus distance slice
projection_axis = "z"
file_name = f"nucleus_distance_{cell_id}"
cbar_label = "Distance from nucleus (\u03bcm)"

fig_nuc, ax_nuc = plot_grid_points_slice(
    grid_points_slice=grid_points_slice[projection_axis],
    inside_mem_outside_nuc=inside_mem_outside_nuc[projection_axis],
    inside_nuc=inside_nuc[projection_axis],
    color_var=nuc_distances_um[projection_axis],
    cbar_label=cbar_label,
    dot_size=2,
    projection_axis=projection_axis,
)

plt.show()
fig_nuc.savefig(figures_dir / f"{file_name}.pdf", bbox_inches="tight")
# %% [markdown]
# ## Plot membrane distance slice
projection_axis = "z"
file_name = f"membrane_distance_{cell_id}"
cbar_label = "Distance from membrane (\u03bcm)"
fig_mem, ax_mem = plot_grid_points_slice(
    grid_points_slice=grid_points_slice[projection_axis],
    inside_mem_outside_nuc=inside_mem_outside_nuc[projection_axis],
    inside_nuc=inside_nuc[projection_axis],
    color_var=mem_distances_um[projection_axis],
    cbar_label=cbar_label,
    dot_size=2,
    projection_axis=projection_axis,
    reverse_cmap=True,
)
plt.show()
fig_mem.savefig(figures_dir / f"{file_name}.pdf", bbox_inches="tight")
# %% [markdown]
# ## Plot z distance slice
projection_axis = "y"
file_name = f"z_distance_{cell_id}"
cbar_label = "Z Distance (\u03bcm)"
fig_z, ax_z = plot_grid_points_slice(
    grid_points_slice=grid_points_slice[projection_axis],
    inside_mem_outside_nuc=inside_mem_outside_nuc[projection_axis],
    inside_nuc=inside_nuc[projection_axis],
    color_var=z_distances_um[projection_axis],
    cbar_label=cbar_label,
    dot_size=2,
    projection_axis=projection_axis,
)
plt.show()
fig_z.savefig(figures_dir / f"{file_name}.pdf", bbox_inches="tight")

# %%
