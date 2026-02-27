# %% [markdown]
# # Plot distribution of available distances
#
# See `cell_metrics_viz.py` for cell-level metric logging and distribution plots.
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
    DISTANCE_LIMITS,
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

fontsize = 6
plt.rcParams["font.size"] = fontsize

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Set parameters and file paths
all_structures = [
    "SLC25A17",  # peroxisomes
    "RAB5A",  # early endosomes
]

distance_measures = ["nucleus", "z"]

DISTANCE_YLIMS = {
    "nucleus": (0, 0.35),
    "z": (0, 0.25),
}

minimum_distance = -1
bandwidth = 0.2

project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

figures_dir = base_results_dir / "occupancy_analysis/available_space"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load mesh information dicts
combined_mesh_information_dict: dict = {}
for structure_id in all_structures:
    combined_mesh_information_dict[structure_id] = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )

# %% [markdown]
# ## Plot grid distance distribution
fig, axs = plt.subplots(len(distance_measures), 1, figsize=(2.5, 2.5), dpi=300)
distances = {
    structure_id: {distance_measure: [] for distance_measure in distance_measures}
    for structure_id in all_structures
}
combined_distances = {distance_measure: [] for distance_measure in distance_measures}

for row, distance_measure in enumerate(distance_measures):
    ax = axs[row]

    for structure_id in all_structures:
        mesh_information_dict = combined_mesh_information_dict[structure_id]
        for _cell_id, cellid_dict in tqdm(
            mesh_information_dict.items(), desc=f"{structure_id} - {distance_measure}"
        ):
            cell_grid_distances = cellid_dict[GRID_DISTANCE_LABELS[distance_measure]]
            cell_distances_um = filter_invalid_distances(
                cell_grid_distances * PIXEL_SIZE_IN_UM, minimum_distance=minimum_distance
            )
            distances[structure_id][distance_measure].append(cell_distances_um)
            combined_distances[distance_measure].extend(cell_distances_um)
            sns.kdeplot(
                cell_distances_um,
                ax=ax,
                color="gray",
                alpha=0.1,
                linewidth=0.1,
                cut=0,
                label="_nolegend_",
                bw_method=bandwidth,
            )

    sns.kdeplot(
        combined_distances[distance_measure],
        ax=ax,
        color="black",
        linewidth=1.5,
        cut=0,
        bw_method=bandwidth,
    )
    ax.set_xlabel(f"{DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)")
    sns.despine(ax=ax)
    ax.set_ylim(DISTANCE_YLIMS[distance_measure])
    ax.set_xlim(DISTANCE_LIMITS[distance_measure])
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))

fig.tight_layout()
fig.savefig(figures_dir / "grid_distance_distribution.pdf", bbox_inches="tight")

# %% [markdown]
# ## Plot combined CDF for distance measures
fig, axs = plt.subplots(len(distance_measures), 1, figsize=(2.5, 2.5), dpi=300, sharey=True)
for row, distance_measure in enumerate(distance_measures):
    ax = axs[row]
    all_combined_distances = np.concatenate(
        [distances[structure_id][distance_measure] for structure_id in all_structures]
    )
    sorted_distances = np.sort(all_combined_distances)
    cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    sns.lineplot(
        x=sorted_distances,
        y=cdf,
        ax=ax,
        color="k",
        label="all",
        linewidth=1.5,
    )
    sns.despine(ax=ax)
    ax.set_xlabel(f"{DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)")
    ax.set_ylabel("CDF")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.set_xlim(DISTANCE_LIMITS[distance_measure])
    ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(figures_dir / "grid_distance_cdf.pdf", bbox_inches="tight")

# %% [markdown]
# ## Calculate and log radius of gyration
for distance_measure in distance_measures:
    radius_of_gyration = np.sqrt(
        np.mean(np.array(combined_distances[distance_measure]) ** 2)
    ).item()
    logger.info(f"Radius of gyration for {distance_measure}: {radius_of_gyration:.2f} \u03bcm")
