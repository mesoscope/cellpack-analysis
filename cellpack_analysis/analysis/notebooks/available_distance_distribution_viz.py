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
from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    get_project_root,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.label_tables import (
    COLOR_PALETTE,
    DISTANCE_LIMITS,
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
    MODE_LABELS,
    STATS_LABELS,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
fontsize = 6
plt.rcParams["font.size"] = fontsize

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
# ## Get mesh information dicts
combined_mesh_information_dict = {}
for structure_id in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )
    combined_mesh_information_dict[structure_id] = mesh_information_dict
# %% [markdown]
# ## Log averages for cell metrics
log_file_path = base_results_dir / "cell_metrics/punctate_cell_metrics.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)
logger = add_file_handler_to_logger(logger, log_file_path)
cell_stats = {
    structure_id: {
        "cell_volumes": [],
        "nuc_volumes": [],
        "cell_heights": [],
        "nuc_heights": [],
        "cell_diameters": [],
        "nuc_diameters": [],
        "intracellular_radii": [],
    }
    for structure_id in all_structures
}
for structure_id, mesh_information_dict in combined_mesh_information_dict.items():
    logger.info(f"{structure_id} cell metrics:")
    logger.info("===================================")
    for cellid_dict in mesh_information_dict.values():
        cell_stats[structure_id]["cell_volumes"].append(
            cellid_dict["cell_volume"] * (PIXEL_SIZE_IN_UM**3)
        )
        cell_stats[structure_id]["nuc_volumes"].append(
            cellid_dict["nuc_volume"] * (PIXEL_SIZE_IN_UM**3)
        )

        cell_bounds = cellid_dict["cell_bounds"]
        nuc_bounds = cellid_dict["nuc_bounds"]
        cell_bottom = cell_bounds[:, 2].min() * PIXEL_SIZE_IN_UM
        cell_top = cell_bounds[:, 2].max() * PIXEL_SIZE_IN_UM
        nuc_top = nuc_bounds[:, 2].max() * PIXEL_SIZE_IN_UM
        cell_stats[structure_id]["cell_heights"].append(cell_top - cell_bottom)
        cell_stats[structure_id]["nuc_heights"].append(nuc_top - cell_bottom)

        cell_stats[structure_id]["cell_diameters"].append(
            cellid_dict["cell_diameter"] * PIXEL_SIZE_IN_UM
        )
        cell_stats[structure_id]["nuc_diameters"].append(
            cellid_dict["nuc_diameter"] * PIXEL_SIZE_IN_UM
        )
        cell_stats[structure_id]["intracellular_radii"].append(
            cellid_dict["intracellular_radius"] * PIXEL_SIZE_IN_UM
        )

    mean_nuc_volume = np.mean(cell_stats[structure_id]["nuc_volumes"]).item()
    std_nuc_volume = np.std(cell_stats[structure_id]["nuc_volumes"]).item()
    mean_cell_volume = np.mean(cell_stats[structure_id]["cell_volumes"]).item()
    std_cell_volume = np.std(cell_stats[structure_id]["cell_volumes"]).item()

    mean_cell_height = np.mean(cell_stats[structure_id]["cell_heights"]).item()
    std_cell_height = np.std(cell_stats[structure_id]["cell_heights"]).item()
    mean_nuc_height = np.mean(cell_stats[structure_id]["nuc_heights"]).item()
    std_nuc_height = np.std(cell_stats[structure_id]["nuc_heights"]).item()

    mean_cell_diameter = np.mean(cell_stats[structure_id]["cell_diameters"]).item()
    std_cell_diameter = np.std(cell_stats[structure_id]["cell_diameters"]).item()
    mean_nuc_diameter = np.mean(cell_stats[structure_id]["nuc_diameters"]).item()
    std_nuc_diameter = np.std(cell_stats[structure_id]["nuc_diameters"]).item()

    mean_intracellular_radius = np.mean(cell_stats[structure_id]["intracellular_radii"]).item()
    std_intracellular_radius = np.std(cell_stats[structure_id]["intracellular_radii"]).item()

    logger.info(
        f"{structure_id} cell volume: {mean_cell_volume:.2f} +/- {std_cell_volume:.2f} \u03bcm^3"
    )
    logger.info(
        f"{structure_id} nucleus volume: {mean_nuc_volume:.2f} +/- {std_nuc_volume:.2f} \u03bcm^3"
    )
    logger.info(
        f"{structure_id} cell height: {mean_cell_height:.2f} +/- {std_cell_height:.2f} \u03bcm"
    )
    logger.info(
        f"{structure_id} nucleus height: {mean_nuc_height:.2f} +/- {std_nuc_height:.2f} \u03bcm"
    )
    logger.info(
        f"{structure_id} cell diameter: {mean_cell_diameter:.2f} +/- "
        f"{std_cell_diameter:.2f} \u03bcm"
    )
    logger.info(
        f"{structure_id} nucleus diameter: {mean_nuc_diameter:.2f} +/- "
        f"{std_nuc_diameter:.2f} \u03bcm"
    )
    logger.info(
        f"{structure_id} intracellular radius: {mean_intracellular_radius:.2f} +/- "
        f"{std_intracellular_radius:.2f} \u03bcm"
    )
    logger.info("===================================")
    logger.info("===================================")
logger = remove_file_handler_from_logger(logger, log_file_path)

# %% [markdown]
# ## Plot distributions of cell stats
measurement_names = list(cell_stats[all_structures[0]].keys())
fig, axs = plt.subplots(
    figsize=(6.5, 5),
    dpi=300,
    nrows=len(measurement_names),
    ncols=len(all_structures),
    sharex="row",
    sharey="row",
    squeeze=False,
)
for ct, structure_id in enumerate(all_structures):
    structure_stats = cell_stats[structure_id]
    for rt, measurement_name in enumerate(measurement_names):
        measurement_values = structure_stats[measurement_name]
        measurement_label = STATS_LABELS.get(measurement_name, measurement_name)
        ax = axs[rt, ct]
        sns.histplot(
            measurement_values,
            ax=ax,
            color=COLOR_PALETTE[structure_id],
            # linewidth=0,
            bins=12,
            # alpha=0.7,
        )
        ax.yaxis.label.set_visible(True)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel(measurement_label)
        ax.set_ylabel("Number of cells")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        sns.despine(ax=ax)

fig.tight_layout()
fig.savefig(
    figures_dir / "cell_measurements_distribution.pdf",
    bbox_inches="tight",
)
plt.show()


# %% [markdown]
# ## Plot grid distance distribution
fig, axs = plt.subplots(2, 1, figsize=(2.5, 2.5), dpi=300)
minimum_distance = -1
bandwidth = 0.2
combined_distances = {}

DISTANCE_YLIMS = {
    "nucleus": (0, 0.35),
    "z": (0, 0.25),
}

for row, distance_measure in enumerate(distance_measures):
    ax = axs[row]

    for structure_id in all_structures:
        combined_distances[distance_measure] = {structure_id: []}

        mesh_information_dict = get_mesh_information_dict_for_structure(
            structure_id=structure_id,
            base_datadir=base_datadir,
            recalculate=False,
        )
        logger.info(f"Plotting {structure_id} - {distance_measure}")
        for cell_id in tqdm(mesh_information_dict.keys()):
            distances = mesh_information_dict[cell_id][GRID_DISTANCE_LABELS[distance_measure]]
            distances_to_plot = filter_invalid_distances(
                distances * PIXEL_SIZE_IN_UM, minimum_distance=minimum_distance
            )

            sns.histplot(
                distances_to_plot,
                ax=ax,
                color="gray",
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
