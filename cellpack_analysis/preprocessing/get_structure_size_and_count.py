# %% [markdown]
# ## Get distribution of counts and sizes from manifest

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib.file_io import get_datadir_path, get_results_path
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.get_variance_dataset import get_variance_dataframe
from cellpack_analysis.lib.label_tables import COLOR_PALETTE

logger = logging.getLogger(__name__)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
fontsize = 6
plt.rcParams["font.size"] = fontsize

# %% [markdown]
# ## Load the variance dataframe
meta_df = get_variance_dataframe()
structures_of_interest = ["SLC25A17", "RAB5A"]
dsphere = True  # Get cell IDs for one of the structures (they should be the same for both)
# %% [markdown]
# Process each structure and calculate statistics
stats_data = []

for gene, df_gene in meta_df.groupby("structure_name"):
    cell_id_list = get_cell_id_list_for_structure(structure_id=gene, dsphere=dsphere)
    df_gene = df_gene[df_gene.index.astype(str).isin(cell_id_list)]
    counts = df_gene["STR_connectivity_cc"].to_numpy().astype(float)

    # Handle division by zero for counts
    valid_counts = counts > 0
    volume_per_unit = np.full_like(counts, np.nan)
    volume_per_unit[valid_counts] = (
        df_gene["STR_shape_volume"].to_numpy()[valid_counts] / counts[valid_counts]
    )

    # Calculate unit radius (assuming spherical structures)
    unit_radius = np.full_like(volume_per_unit, np.nan)
    valid_volumes = ~np.isnan(volume_per_unit) & (volume_per_unit > 0)
    unit_radius[valid_volumes] = (volume_per_unit[valid_volumes] / (4 / 3 * np.pi)) ** (1 / 3)

    # Calculate sphericity more efficiently
    mem_surface_area = df_gene["MEM_roundness_surface_area"].to_numpy()
    mem_volume = df_gene["MEM_shape_volume"].to_numpy()
    sphericity = np.full_like(mem_volume, np.nan)
    valid_sphericity = mem_volume > 0
    sphericity[valid_sphericity] = (
        np.pi ** (1 / 3)
        * (6 * mem_volume[valid_sphericity]) ** (2 / 3)
        / mem_surface_area[valid_sphericity]
    )  # 1 is perfect sphere

    # Create records for this structure
    for i, cell_id in enumerate(df_gene.index.astype(str)):
        stats_data.append(
            {
                "CellId": cell_id,
                "structure_name": gene,
                "count": counts[i],
                "volume": volume_per_unit[i],
                "radius": unit_radius[i],
                "cell_stage": df_gene["cell_stage"].iloc[i],
                "cell_volume": df_gene["MEM_shape_volume"].iloc[i],
                "nuc_volume": df_gene["NUC_shape_volume"].iloc[i],
                "cell_height": df_gene["MEM_position_depth"].iloc[i],
                "nuc_height": df_gene["NUC_position_depth"].iloc[i],
                "sphericity": sphericity[i],
            }
        )

    # Print statistics for structures of interest
    if gene in structures_of_interest:
        valid_data = ~np.isnan(volume_per_unit) & ~np.isnan(unit_radius)
        if np.any(valid_data):
            print(f"\n{gene}:")
            print(f"  Count: {np.mean(counts):0.2f} ± {np.std(counts):.2f}")
            print(
                f"  Volume: {np.nanmean(volume_per_unit):0.2f} ± {np.nanstd(volume_per_unit):.2f}"
            )
            print(f"  Radius: {np.nanmean(unit_radius):0.2f} ± {np.nanstd(unit_radius):.2f}")
        else:
            print(f"\n{gene}: No valid data (all counts are zero)")

df_stats = pd.DataFrame(stats_data)

# %% [markdown]
# ## Save updated dataframe
df_stats_path = get_datadir_path() / "structure_stats.parquet"
df_stats.to_parquet(df_stats_path)
# %% [markdown]
# Create histograms of counts and sizes for structures of interest
dsphere = True
figures_dir = get_results_path() / "cell_metrics/figures"
figures_dir.mkdir(parents=True, exist_ok=True)
measurement_tuples = [
    ("count", "Number of puncta", 10),
    ("volume", "Puncta volume (µm\u00b3)", 10),
    ("radius", "Puncta radius (µm)", 0.1),
    ("sphericity", "Sphericity", 0.01),
]
fig, axs = plt.subplots(
    figsize=(6.5, 5),
    dpi=300,
    # nrows=len(measurement_tuples) // 2,
    # ncols=len(structures_of_interest) * 2,
    nrows=len(measurement_tuples),
    ncols=len(structures_of_interest),
    sharex="row",
    sharey="row",
    squeeze=False,
)
for ct, structure_id in enumerate(structures_of_interest):
    cell_id_list = get_cell_id_list_for_structure(structure_id=structure_id, dsphere=dsphere)
    for rt, (measurement_name, measurement_label, bin_width) in enumerate(measurement_tuples):
        measurement_values = (
            df_stats.loc[df_stats["structure_name"] == structure_id, measurement_name]
            .dropna()
            .to_numpy()
        )
        # row_index = rt // 2
        # col_index = ct * 2 + (rt % 2)

        ax = axs[rt, ct]
        sns.histplot(
            measurement_values,
            ax=ax,
            color=COLOR_PALETTE[structure_id],
            # linewidth=0,
            bins=12,
            binwidth=bin_width,
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
    figures_dir / "puncta_distribution_histograms.pdf",
    bbox_inches="tight",
)
plt.show()

# %%
