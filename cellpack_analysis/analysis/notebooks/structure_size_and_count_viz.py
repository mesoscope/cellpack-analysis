# %% [markdown]
# # Structure size and count distribution histograms
# Visualize the distributions of puncta count, volume, and equivalent spherical radius
# for structures of interest, loaded from the pre-computed structure_stats.parquet file.

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib.file_io import get_results_path
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.label_tables import COLOR_PALETTE

plt.rcParams["font.size"] = 6

# %% [markdown]
# ## Load pre-computed structure stats
df_stats = get_structure_stats_dataframe()

structures_of_interest = ["SLC25A17", "RAB5A"]

measurement_tuples = [
    ("count", "Number of puncta", 10),
    ("volume", "Puncta volume (µm³)", 10),
    ("radius", "Equivalent spherical puncta radius (µm)", 0.1),
]

# %% [markdown]
# ## Plot histograms
figures_dir = get_results_path() / "cell_metrics/figures"
figures_dir.mkdir(parents=True, exist_ok=True)

fig, axs = plt.subplots(
    nrows=len(measurement_tuples),
    ncols=len(structures_of_interest),
    figsize=(6.5, 5),
    dpi=300,
    sharex="row",
    sharey="row",
    squeeze=False,
)

for ct, structure_id in enumerate(structures_of_interest):
    for rt, (measurement_name, measurement_label, bin_width) in enumerate(measurement_tuples):
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
fig.savefig(figures_dir / "puncta_distribution_histograms.pdf", bbox_inches="tight")
plt.show()

# %%
