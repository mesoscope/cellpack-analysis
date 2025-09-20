# %% [markdown]
# # Plot cell diameter and distance measures for grid points
# %% [markdown]
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import filter_invalid_distances
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import COLOR_PALETTE, DISTANCE_LIMITS
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

log = logging.getLogger(__name__)

# %% [markdown]
# ## Set parameters and file paths
all_structures = [
    "SLC25A17",  # SLC25A17: peroxisomes, RAB5A: early endosomes
    "RAB5A",
]

distance_measures = ["nucleus", "z"]

project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

figures_dir = base_results_dir / "occupancy_analysis/available_space"
figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ## Plot cell diameter distributions
fontsize = 6
plt.rcParams.update({"font.size": fontsize})
for structure_id in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )
    cell_diameters = [
        seed_dict["cell_diameter"] * PIXEL_SIZE_IN_UM
        for seed_dict in mesh_information_dict.values()
    ]
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    sns.kdeplot(
        cell_diameters,
        ax=ax,
        color=COLOR_PALETTE[structure_id],
        linewidth=1.5,
        cut=0,
        label=structure_id,
    )
    mean_cell_diameter = np.mean(cell_diameters).item()
    std_cell_diameter = np.std(cell_diameters).item()
    ax.axvspan(
        mean_cell_diameter - std_cell_diameter,
        mean_cell_diameter + std_cell_diameter,
        facecolor="gray",
        alpha=0.5,
        edgecolor="none",
    )
    ax.axvline(mean_cell_diameter, color="gray", linestyle="--")
    ax.set_xlabel("Cell diameter (\u03bcm)")
    ax.set_ylabel("Probability density")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    sns.despine(ax=ax)
    ax.text(
        0.95,
        0.95,
        f"{structure_id}\n{mean_cell_diameter:.2f} +/- {std_cell_diameter:.2f}\u03bcm",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / f"cell_diameter_distribution_{structure_id}.svg", bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## Load distances
normalization = None
nuc_distances = []
mem_distances = []
z_distances = []
for cell_id in tqdm(cell_ids_to_use):
    nuc_distances.append(np.load(grid_dir / f"nuc_distances_{cell_id}.npy"))
    mem_distances.append(np.load(grid_dir / f"mem_distances_{cell_id}.npy"))
    z_distances.append(np.load(grid_dir / f"z_distances_{cell_id}.npy"))
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
