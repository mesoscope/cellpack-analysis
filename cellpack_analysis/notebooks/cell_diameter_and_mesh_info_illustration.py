# %% [markdown]
# # Plot cell diameter and distance measures for grid points
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import filter_invalid_distances
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import (
    COLOR_PALETTE,
    DISTANCE_LIMITS,
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
    MODE_LABELS,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

logger = logging.getLogger(__name__)

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
# ## Plot grid distance distribution
fig, axs = plt.subplots(2, 2, figsize=(4, 2.5), dpi=300, sharex="col", sharey="col")

for row, structure_id in enumerate(all_structures):
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )
    for col, distance_measure in enumerate(distance_measures):
        ax = axs[row, col]

        combined_distances = []
        for cell_id in tqdm(mesh_information_dict.keys()):
            distances = mesh_information_dict[cell_id][GRID_DISTANCE_LABELS[distance_measure]]
            distances_to_plot = filter_invalid_distances(
                distances * PIXEL_SIZE_IN_UM, minimum_distance=0
            )

            sns.kdeplot(
                distances_to_plot,
                ax=ax,
                color=COLOR_PALETTE[structure_id],
                alpha=0.1,
                linewidth=0.1,
                cut=0,
                label="_nolegend_",
                bw_method=0.4,
            )
            combined_distances.append(distances_to_plot)

        sns.kdeplot(
            np.concatenate(combined_distances),
            ax=ax,
            color=COLOR_PALETTE[structure_id],
            linewidth=1.5,
            cut=0,
            label=structure_id,
            bw_method=0.4,
        )
        if row == 1:
            ax.set_xlabel(f"{DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)")
        if col == 0:
            ax.set_ylabel(f"{MODE_LABELS[structure_id]}\n PDF")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        sns.despine(ax=ax)
        ax.set_xlim(DISTANCE_LIMITS[distance_measure])
fig.tight_layout()
fig.savefig(figures_dir / "grid_distance_distribution.pdf", bbox_inches="tight")

# %%
