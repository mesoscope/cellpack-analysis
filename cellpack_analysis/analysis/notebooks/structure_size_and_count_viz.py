# %% [markdown]
"""
# Puncta count and size (Table 1, Fig S1)

Logs per-structure mean +/- SD (pandas ``std``, ddof 1) of puncta count per cell, equivalent
spherical radius (nm) and volume per punctum, and plots their distributions, for the peroxisome
(n = 305) and endosome (n = 442) cells.

Input: ``data/structure_stats.parquet`` (made by
``preprocessing/get_structure_size_and_count.py``).

Outputs:
- ``results/illustrations/Table1/puncta_size_and_count.log`` (Table 1)
- ``results/illustrations/FigS1/FigS1_puncta_distribution_histograms.pdf`` (Fig S1)
"""

import logging

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    get_results_path,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.label_tables import COLOR_PALETTE

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Frozen parameters
STRUCTURE_IDS = ["SLC25A17", "RAB5A"]  # peroxisome, endosome cohorts
DDOF = 1  # SD convention for Table 1 (pandas default)
FONT_SIZE = 6  # published font size
FIG_SIZE = (6.5, 5)  # published figure size (in)
# (column, x label, bin width): rows of Fig S1
MEASUREMENTS = [
    ("count", "Number of puncta", 10),
    ("volume", "Puncta volume (µm³)", 10 * 0.1**3),
    ("radius", "Equivalent spherical puncta radius (µm)", 0.01),
]

plt.rcParams["font.size"] = FONT_SIZE

results_dir = get_results_path() / "illustrations"
log_file_path = results_dir / "Table1/puncta_size_and_count.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)
figures_dir = results_dir / "FigS1"
figures_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load structure stats
df_stats = get_structure_stats_dataframe()

# %% [markdown]
# ## Log Table 1 (mean +/- SD, ddof 1)
logger = add_file_handler_to_logger(logger, log_file_path)
for structure_id in STRUCTURE_IDS:
    df_structure = df_stats.loc[df_stats["structure_name"] == structure_id]
    count = df_structure["count"].dropna()
    radius_nm = df_structure["radius"].dropna() * 1000
    volume = df_structure["volume"].dropna()
    logger.info(f"{structure_id}: n = {len(df_structure)} cells (SD with ddof={DDOF})")
    logger.info(
        f"{structure_id} count per cell: {count.mean():.2f} +/- {count.std(ddof=DDOF):.2f} "
        f"(n = {len(count)})"
    )
    logger.info(
        f"{structure_id} radius: {radius_nm.mean():.2f} +/- {radius_nm.std(ddof=DDOF):.2f} nm "
        f"(n = {len(radius_nm)})"
    )
    logger.info(
        f"{structure_id} volume: {volume.mean():.4f} +/- {volume.std(ddof=DDOF):.4f} µm^3 "
        f"(n = {len(volume)})"
    )
logger = remove_file_handler_from_logger(logger, log_file_path)

# %% [markdown]
# ## Plot Fig S1
fig, axs = plt.subplots(
    nrows=len(MEASUREMENTS),
    ncols=len(STRUCTURE_IDS),
    figsize=FIG_SIZE,
    dpi=300,
    sharex="row",
    sharey="row",
    squeeze=False,
)
for ct, structure_id in enumerate(STRUCTURE_IDS):
    for rt, (measurement_name, measurement_label, bin_width) in enumerate(MEASUREMENTS):
        measurement_values = (
            df_stats.loc[df_stats["structure_name"] == structure_id, measurement_name]
            .dropna()
            .to_numpy()
        )
        ax = axs[rt, ct]
        sns.histplot(
            measurement_values,
            ax=ax,
            color=COLOR_PALETTE[structure_id],
            bins=12,
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
fig.savefig(figures_dir / "FigS1_puncta_distribution_histograms.pdf", bbox_inches="tight")
plt.show()

# %%
