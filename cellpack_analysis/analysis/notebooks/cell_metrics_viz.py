# %% [markdown]
# # Cell metrics: logging and distribution plots
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib import label_tables
from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    get_project_root,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

fontsize = 6
plt.rcParams["font.size"] = fontsize

logger = logging.getLogger(__name__)

# Label and unit for each tracked metric (used in log output)
METRIC_LOG_META: dict[str, tuple[str, str]] = {
    "mem_volumes": ("mem volume", "μm^3"),
    "nuc_volumes": ("nucleus volume", "μm^3"),
    "mem_heights": ("mem height", "μm"),
    "nuc_heights": ("nucleus height", "μm"),
    "mem_diameters": ("mem diameter", "μm"),
    "nuc_diameters": ("nucleus diameter", "μm"),
    "mem_sphericities": ("membrane sphericity", ""),
    "nuc_sphericities": ("nucleus sphericity", ""),
}

# %% [markdown]
# ## Set parameters and file paths
all_structures = [
    "SLC25A17",  # peroxisomes
    "RAB5A",  # early endosomes
]

project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

figures_dir = base_results_dir / "cell_metrics/figures"
figures_dir.mkdir(exist_ok=True, parents=True)

log_file_path = base_results_dir / "cell_metrics/punctate_cell_metrics.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load mesh information and collect per-cell metrics
combined_mesh_information_dict: dict = {}
cell_stats: dict = {
    structure_id: {metric: [] for metric in METRIC_LOG_META} for structure_id in all_structures
}

for structure_id in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=True,
    )
    combined_mesh_information_dict[structure_id] = mesh_information_dict

    for cellid_dict in mesh_information_dict.values():
        mem_bounds = cellid_dict["mem_bounds"]
        nuc_bounds = cellid_dict["nuc_bounds"]
        mem_bottom = mem_bounds[:, 2].min() * PIXEL_SIZE_IN_UM
        mem_top = mem_bounds[:, 2].max() * PIXEL_SIZE_IN_UM
        nuc_top = nuc_bounds[:, 2].max() * PIXEL_SIZE_IN_UM

        cell_stats[structure_id]["mem_volumes"].append(
            cellid_dict["mem_volume"] * (PIXEL_SIZE_IN_UM**3)
        )
        cell_stats[structure_id]["nuc_volumes"].append(
            cellid_dict["nuc_volume"] * (PIXEL_SIZE_IN_UM**3)
        )
        cell_stats[structure_id]["mem_heights"].append(mem_top - mem_bottom)
        cell_stats[structure_id]["nuc_heights"].append(nuc_top - mem_bottom)
        cell_stats[structure_id]["mem_diameters"].append(
            cellid_dict["mem_diameter"] * PIXEL_SIZE_IN_UM
        )
        cell_stats[structure_id]["nuc_diameters"].append(
            cellid_dict["nuc_diameter"] * PIXEL_SIZE_IN_UM
        )
        cell_stats[structure_id]["mem_sphericities"].append(cellid_dict["mem_sphericity"])
        cell_stats[structure_id]["nuc_sphericities"].append(cellid_dict["nuc_sphericity"])

# %% [markdown]
# ## Log summary statistics for each structure
logger = add_file_handler_to_logger(logger, log_file_path)
for structure_id in all_structures:
    logger.info(f"{structure_id} cell metrics:")
    logger.info("===================================")
    for metric, (label, unit) in METRIC_LOG_META.items():
        values = cell_stats[structure_id][metric]
        mean_val = np.mean(values).item()
        std_val = np.std(values).item()
        unit_str = f" {unit}" if unit else ""
        logger.info(f"{structure_id} {label}: {mean_val:.2f} +/- {std_val:.2f}{unit_str}")
    logger.info("===================================")
    logger.info("===================================")
logger = remove_file_handler_from_logger(logger, log_file_path)

# %% [markdown]
# ## Plot distributions of cell metrics
# Each tuple: (base_measure, bin_width) — bin_width is in axis/data units (μm, μm^3, or unitless)
prefixes = ["mem", "nuc"]
base_measure_tuples: list[tuple[str, float]] = [
    ("volumes", 100.0),  # μm^3
    ("heights", 0.5),  # μm
    ("diameters", 2.0),  # μm
    ("sphericities", 0.025),  # unitless
]

# Layout: rows = base measures, cols = struct_1_mem, struct_1_nuc, struct_2_mem, struct_2_nuc, …
nrows = len(base_measure_tuples)
ncols = len(all_structures) * len(prefixes)

fig, axs = plt.subplots(
    figsize=(6.5, 5),
    dpi=300,
    nrows=nrows,
    ncols=ncols,
    sharex="row",
    sharey="row",
    squeeze=False,
)
for rt, (base_measure, bin_width) in enumerate(base_measure_tuples):
    for ct, structure_id in enumerate(all_structures):
        for pt, prefix in enumerate(prefixes):
            col = ct * len(prefixes) + pt
            metric = f"{prefix}_{base_measure}"
            measurement_values = cell_stats[structure_id][metric]
            measurement_label = label_tables.STATS_LABELS.get(metric, metric)
            ax = axs[rt, col]
            sns.histplot(
                measurement_values,
                ax=ax,
                color=label_tables.COLOR_PALETTE[structure_id],
                binwidth=bin_width,
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

# %%
