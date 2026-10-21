# %% [markdown]
"""
# Cell and nucleus metrics (Table 2, Fig S2)

Logs mean +/- SD (``np.std``, ddof 0) of cell and nucleus volume, height, diameter and
sphericity for the peroxisome (n = 305) and endosome (n = 442) cells, and plots their
distributions.

Inputs: ``data/structure_data/{SLC25A17,RAB5A}/mesh_information.dat`` (read only).

Outputs:
- ``results/illustrations/Table2/punctate_cell_metrics.log`` (Table 2)
- ``results/illustrations/FigS2/FigS2_cell_measurements_distribution.pdf`` (Fig S2)
"""

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

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Frozen parameters
STRUCTURE_IDS = ["SLC25A17", "RAB5A"]  # peroxisome, endosome cohorts
RECALCULATE = False  # never True: it rebuilds and overwrites mesh_information.dat
FONT_SIZE = 6  # published font size
FIG_SIZE = (6.5, 5)  # published figure size (in)
# (measure, bin width in axis units): rows of Fig S2
BASE_MEASURE_BINS: list[tuple[str, float]] = [
    ("volumes", 100.0),  # µm^3
    ("heights", 0.5),  # µm
    ("diameters", 2.0),  # µm
    ("sphericities", 0.025),  # unitless
]
PREFIXES = ["mem", "nuc"]  # columns per structure: cell, nucleus

# Log label and unit per metric
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

plt.rcParams["font.size"] = FONT_SIZE

project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

log_file_path = base_results_dir / "illustrations/Table2/punctate_cell_metrics.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)
figures_dir = base_results_dir / "illustrations/FigS2"
figures_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Collect per-cell metrics
cell_stats: dict = {
    structure_id: {metric: [] for metric in METRIC_LOG_META} for structure_id in STRUCTURE_IDS
}
for structure_id in STRUCTURE_IDS:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=RECALCULATE,
    )
    for cellid_dict in mesh_information_dict.values():
        mem_bounds = cellid_dict["mem_bounds"]
        nuc_bounds = cellid_dict["nuc_bounds"]
        mem_bottom = mem_bounds[:, 2].min() * PIXEL_SIZE_IN_UM
        mem_top = mem_bounds[:, 2].max() * PIXEL_SIZE_IN_UM
        nuc_top = nuc_bounds[:, 2].max() * PIXEL_SIZE_IN_UM

        stats = cell_stats[structure_id]
        stats["mem_volumes"].append(cellid_dict["mem_volume"] * (PIXEL_SIZE_IN_UM**3))
        stats["nuc_volumes"].append(cellid_dict["nuc_volume"] * (PIXEL_SIZE_IN_UM**3))
        stats["mem_heights"].append(mem_top - mem_bottom)
        stats["nuc_heights"].append(nuc_top - mem_bottom)
        stats["mem_diameters"].append(cellid_dict["mem_diameter"] * PIXEL_SIZE_IN_UM)
        stats["nuc_diameters"].append(cellid_dict["nuc_diameter"] * PIXEL_SIZE_IN_UM)
        stats["mem_sphericities"].append(cellid_dict["mem_sphericity"])
        stats["nuc_sphericities"].append(cellid_dict["nuc_sphericity"])
    del mesh_information_dict

# %% [markdown]
# ## Log Table 2 (mean +/- SD, ddof 0)
logger = add_file_handler_to_logger(logger, log_file_path)
for structure_id in STRUCTURE_IDS:
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
# ## Plot Fig S2
# Rows: measures; columns: struct_1 cell, struct_1 nucleus, struct_2 cell, struct_2 nucleus
fig, axs = plt.subplots(
    figsize=FIG_SIZE,
    dpi=300,
    nrows=len(BASE_MEASURE_BINS),
    ncols=len(STRUCTURE_IDS) * len(PREFIXES),
    sharex="row",
    sharey="row",
    squeeze=False,
)
for rt, (base_measure, bin_width) in enumerate(BASE_MEASURE_BINS):
    for ct, structure_id in enumerate(STRUCTURE_IDS):
        for pt, prefix in enumerate(PREFIXES):
            metric = f"{prefix}_{base_measure}"
            ax = axs[rt, ct * len(PREFIXES) + pt]
            sns.histplot(
                cell_stats[structure_id][metric],
                ax=ax,
                color=label_tables.COLOR_PALETTE[structure_id],
                binwidth=bin_width,
            )
            ax.yaxis.label.set_visible(True)
            ax.tick_params(axis="y", labelleft=True)
            ax.set_xlabel(label_tables.STATS_LABELS.get(metric, metric))
            ax.set_ylabel("Number of cells")
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            sns.despine(ax=ax)

fig.tight_layout()
fig.savefig(figures_dir / "FigS2_cell_measurements_distribution.pdf", bbox_inches="tight")
plt.show()

# %%
