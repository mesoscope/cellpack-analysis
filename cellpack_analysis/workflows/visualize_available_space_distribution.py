# %% [markdown]
# # Analyze and visualize available space distribution
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import trimesh
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from cellpack_analysis.analysis.punctate_analysis.lib.distance import filter_invalid_distances
from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.label_tables import COLOR_PALETTE, DISTANCE_LIMITS
from cellpack_analysis.lib.mesh_tools import get_list_of_grid_points, round_away_from_zero

log = logging.getLogger(__name__)

# %% [markdown]
# ## Set parameters and file paths
MEAN_SHAPE = False

STRUCTURE_ID = "RAB5A"  # SLC25A17: peroxisomes, RAB5A: early endosomes
STRUCTURE_NAME = "endosome"
PACKING_ID = "endosome"

project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

figures_dir = base_results_dir / f"punctate_analysis/{PACKING_ID}/figures/available_space"
figures_dir.mkdir(exist_ok=True, parents=True)

grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances"
grid_dir.mkdir(exist_ok=True, parents=True)

log.info(f"Figures directory: {figures_dir}")
log.info(f"Grid directory: {grid_dir}")
# %% [markdown]
# ## Select cell_ids to use
if MEAN_SHAPE:
    mesh_folder = base_datadir / "average_shape_meshes"
    cell_ids_to_use = ["mean"]
else:
    mesh_folder = base_datadir / f"structure_data/{STRUCTURE_ID}/meshes/"
    cell_ids_to_use = get_cell_id_list_for_structure(
        structure_id=STRUCTURE_ID, dsphere=True, load_local=True
    )
log.info(f"Using {len(cell_ids_to_use)} cell_ids")
# %% [markdown]
# ## Get meshes for cell_ids used
cell_id_list = []
nuc_meshes_to_use = []
mem_meshes_to_use = []
for cell_id in cell_ids_to_use:
    nuc_mesh = mesh_folder / f"nuc_mesh_{cell_id}.obj"
    mem_mesh = mesh_folder / f"mem_mesh_{cell_id}.obj"
    if nuc_mesh.exists() and mem_mesh.exists():
        cell_id_list.append(cell_id)
        nuc_meshes_to_use.append(nuc_mesh)
        mem_meshes_to_use.append(mem_mesh)
log.info(f"Found {len(nuc_meshes_to_use)} meshes")

# %% [markdown]
# # Grid spacing illustration
# ## Load meshes
if MEAN_SHAPE:
    nuc_mesh = trimesh.load_mesh(base_datadir / "average_shape_meshes/nuc_mesh_mean.obj")
    mem_mesh = trimesh.load_mesh(base_datadir / "average_shape_meshes/mem_mesh_mean.obj")
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

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
all_points_scaled = all_points * PIXEL_SIZE_IN_UM
centroid = np.mean(all_points_scaled, axis=0)
dot_size = 50
ax.scatter(
    all_points_scaled[inside_mem_outside_nuc, 0] - centroid[0],
    all_points_scaled[inside_mem_outside_nuc, 1] - centroid[1],
    c=COLOR_PALETTE["membrane"],
    label="Available points",
    s=dot_size,
)
ax.scatter(
    all_points_scaled[inside_nuc, 0] - centroid[0],
    all_points_scaled[inside_nuc, 1] - centroid[1],
    c=COLOR_PALETTE["nucleus"],
    label="Points inside nucleus",
    s=dot_size,
)
ax.set_xlabel("x (\u03bcm)")
ax.set_ylabel("y (\u03bcm)")
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
ax.set_aspect("equal")
ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
# ax.set_aspect(1.3)
plt.show()
file_name = "grid_points"
if MEAN_SHAPE:
    file_name += "_mean"
fig.savefig(figures_dir / f"{file_name}.svg", bbox_inches="tight")

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
z_distances = []
for cell_id in tqdm(cell_ids_to_use):
    if normalization is not None:
        normalization_factor = mesh_information_dict[str(cell_id)].get(normalization, 1)
    else:
        normalization_factor = 1
    nuc_distances.append(np.load(grid_dir / f"nuc_distances_{cell_id}.npy") / normalization_factor)
    mem_distances.append(np.load(grid_dir / f"mem_distances_{cell_id}.npy") / normalization_factor)
    z_distances.append(np.load(grid_dir / f"z_distances_{cell_id}.npy") / normalization_factor)
log.info(f"Loaded distances for {len(nuc_distances)} cells")
# %% [markdown]
# ## Plot distribution of cell diameter
cell_diameters = [
    mesh_information_dict[str(cell_id)]["cell_diameter"] * PIXEL_SIZE_IN_UM
    for cell_id in cell_id_list
]
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
sns.kdeplot(
    cell_diameters,
    ax=ax,
    color=COLOR_PALETTE[STRUCTURE_ID],
    linewidth=3,
    cut=0,
    label=STRUCTURE_NAME,
)
mean_cell_diameter = np.mean(cell_diameters).item()
std_cell_diameter = np.std(cell_diameters).item()
ax.axvline(mean_cell_diameter, color="black", linestyle="--")
ax.set_xlabel("Cell diameter (\u03bcm)")
ax.set_ylabel("Probability density")
# ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
# ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title(
    f"Mean cell diameter for {STRUCTURE_NAME}: "
    f"{mean_cell_diameter:.2f} +/- {std_cell_diameter:.2f}\u03bcm"
)
fig.savefig(figures_dir / f"cell_diameter_distribution_{STRUCTURE_ID}.svg", bbox_inches="tight")
plt.show()
# %% [markdown]
# ## Plot distance distribution as kde
fig_z, ax_z = plt.subplots(figsize=(6, 3), dpi=300)
fig_nuc, ax_nuc = plt.subplots(figsize=(6, 3), dpi=300)
all_nuc_distances = []
all_z_distances = []
for i in tqdm(range(len(nuc_distances))):
    nuc_distances_to_plot = filter_invalid_distances(nuc_distances[i] * PIXEL_SIZE_IN_UM)
    z_distances_to_plot = filter_invalid_distances(z_distances[i] * PIXEL_SIZE_IN_UM)

    sns.kdeplot(
        nuc_distances_to_plot,
        ax=ax_nuc,
        color=COLOR_PALETTE[STRUCTURE_ID],
        alpha=0.1,
        linewidth=0.5,
        cut=0,
        label="_nolegend_",
    )
    sns.kdeplot(
        z_distances_to_plot,
        ax=ax_z,
        color=COLOR_PALETTE[STRUCTURE_ID],
        alpha=0.1,
        linewidth=0.5,
        cut=0,
        label="_nolegend_",
    )

    all_z_distances.append(z_distances_to_plot)
    all_nuc_distances.append(nuc_distances_to_plot)

all_nuc_distances = np.concatenate(all_nuc_distances)
sns.kdeplot(
    all_nuc_distances, ax=ax_nuc, color=COLOR_PALETTE[STRUCTURE_ID], linewidth=3, bw_method=0.2
)

all_z_distances = np.concatenate(all_z_distances)
sns.kdeplot(all_z_distances, ax=ax_z, color=COLOR_PALETTE[STRUCTURE_ID], linewidth=3, bw_method=0.1)

mean_nuc_distance = np.mean(all_nuc_distances).item()
std_nuc_distance = np.std(all_nuc_distances).item()
ax_nuc.axvline(mean_nuc_distance, color="black", linestyle="--")

mean_z_distance = np.mean(all_z_distances).item()
std_z_distance = np.std(all_z_distances).item()
ax_z.axvline(mean_z_distance, color="black", linestyle="--")

ax_nuc.set_xlim(DISTANCE_LIMITS["nucleus"])
ax_z.set_xlim(DISTANCE_LIMITS["z"])


for ax in [ax_nuc, ax_z]:
    ax.set_xlabel("Distance (\u03bcm)")
    ax.set_ylabel("Probability density")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig_nuc.savefig(
    figures_dir / f"nucleus_distance_distribution_{STRUCTURE_ID}.svg", bbox_inches="tight"
)
fig_z.savefig(figures_dir / f"z_distance_distribution_{STRUCTURE_ID}.svg", bbox_inches="tight")
plt.show()

log.info(
    f"Mean nucleus distance for {STRUCTURE_NAME}: "
    f"{mean_nuc_distance:.2f} +/- {std_nuc_distance}\u03bcm"
)
log.info(f"Mean z distance for {STRUCTURE_NAME}: {mean_z_distance:.2f} +/- {std_z_distance}\u03bcm")

# %%
