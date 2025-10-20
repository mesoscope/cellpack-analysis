# %% [markdown]
# # Plot distribution of available distances
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
    intracellular_radii = [
        seed_dict["intracellular_radius"] * PIXEL_SIZE_IN_UM
        for seed_dict in mesh_information_dict.values()
    ]
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    sns.kdeplot(
        intracellular_radii,
        ax=ax,
        color=COLOR_PALETTE[structure_id],
        linewidth=1.5,
        cut=0,
        label=structure_id,
    )
    mean_intracellular_radius = np.mean(intracellular_radii).item()
    std_intracellular_radius = np.std(intracellular_radii).item()
    ax.axvspan(
        mean_intracellular_radius - std_intracellular_radius,
        mean_intracellular_radius + std_intracellular_radius,
        facecolor="gray",
        alpha=0.5,
        edgecolor="none",
    )
    ax.axvline(mean_intracellular_radius, color="gray", linestyle="--")
    ax.set_xlabel("Intracellular radius (\u03bcm)")
    ax.set_ylabel("Probability density")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    sns.despine(ax=ax)
    ax.text(
        0.95,
        0.95,
        f"{structure_id}\n{mean_intracellular_radius:.2f} +/- "
        f"{std_intracellular_radius:.2f}\u03bcm",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"intracellular_radius_distribution_{structure_id}.pdf", bbox_inches="tight"
    )
    plt.show()


# %% [markdown]
# ## Plot grid distance distribution
fig, axs = plt.subplots(2, 1, figsize=(2.5, 2.5), dpi=300)
minimum_distance = -1
bandwidth = 0.2
combined_distances = {}

for row, distance_measure in enumerate(distance_measures):
    ax = axs[row]

    for structure_id in all_structures:
        combined_distances[distance_measure] = {structure_id: []}

        mesh_information_dict = get_mesh_information_dict_for_structure(
            structure_id=structure_id,
            base_datadir=base_datadir,
            recalculate=False,
        )
        for cell_id in tqdm(mesh_information_dict.keys()):
            distances = mesh_information_dict[cell_id][GRID_DISTANCE_LABELS[distance_measure]]
            distances_to_plot = filter_invalid_distances(
                distances * PIXEL_SIZE_IN_UM, minimum_distance=minimum_distance
            )

            sns.kdeplot(
                distances_to_plot,
                ax=ax,
                color=COLOR_PALETTE[structure_id],
                alpha=0.1,
                linewidth=0.1,
                cut=0,
                label="_nolegend_",
                bw_method=bandwidth,
            )
            combined_distances[distance_measure][structure_id].append(distances_to_plot)
    all_combined_distances = np.concatenate(
        [np.concatenate(v) for v in combined_distances[distance_measure].values()]
    )
    combined_distances[distance_measure]["all"] = all_combined_distances
    sns.kdeplot(
        all_combined_distances,
        ax=ax,
        color=COLOR_PALETTE["membrane"],
        linewidth=1.5,
        cut=0,
        bw_method=bandwidth,
    )
    ax.set_xlabel(f"{DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    sns.despine(ax=ax)
    ax.set_xlim(DISTANCE_LIMITS[distance_measure])
fig.tight_layout()
fig.savefig(figures_dir / "grid_distance_distribution.pdf", bbox_inches="tight")

# %% [markdown]
# ## Plot combined cdf for distance measures
fig, axs = plt.subplots(2, 1, figsize=(2.5, 2.5), dpi=300, sharey=True)
for row, distance_measure in enumerate(distance_measures):
    ax = axs[row]
    all_distances = combined_distances[distance_measure]["all"]
    sorted_distances = np.sort(all_distances)
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


# %% [markdown]
# ### Calculate and log radius of gyration
for distance_measure in distance_measures:
    radius_of_gyration = np.sqrt(np.mean(combined_distances[distance_measure]["all"] ** 2)).item()
    logger.info(f"Radius of gyration for {distance_measure}: {radius_of_gyration:.2f} \u03bcm")
