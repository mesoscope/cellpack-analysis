# %% [markdown]
# ## Get distribution of counts and sizes from manifest

import numpy as np
import pandas as pd

from cellpack_analysis.lib.get_structure_stats_dataframe import (
    get_local_stats_dataframe_path,
)
from cellpack_analysis.lib.get_variance_dataset import get_variance_dataframe

# %%
meta_df = get_variance_dataframe()

# %%
structures_of_interest = ["SLC25A17", "RAB5A", "LAMP1", "RAB7A"]

df_stats = pd.DataFrame(
    columns=[
        "count",
        "volume",
        "radius",
        "cell_stage",
        "cell_volume",
        "nuc_volume",
        "cell_height",
        "nuc_height",
        "sphericity",
    ],
    index=meta_df.index,
)

# %%
for gene, df_gene in meta_df.groupby("structure_name"):
    counts = df_gene["STR_connectivity_cc"].values.astype(float)
    volume_per_unit = df_gene["STR_shape_volume"].values / counts
    unit_radius = (volume_per_unit / (4 / 3 * np.pi)) ** (1 / 3)

    df_stats.loc[df_gene.index, "structure_name"] = gene

    df_stats.loc[df_gene.index, "count"] = counts
    df_stats.loc[df_gene.index, "volume"] = volume_per_unit
    df_stats.loc[df_gene.index, "radius"] = unit_radius
    df_stats.loc[df_gene.index, "cell_stage"] = df_gene["cell_stage"].values
    df_stats.loc[df_gene.index, "cell_volume"] = df_gene["MEM_shape_volume"].values
    df_stats.loc[df_gene.index, "nuc_volume"] = df_gene["NUC_shape_volume"].values
    df_stats.loc[df_gene.index, "cell_height"] = df_gene["MEM_position_depth"].values
    df_stats.loc[df_gene.index, "nuc_height"] = df_gene["NUC_position_depth"].values
    df_stats.loc[df_gene.index, "sphericity"] = df_gene[
        "MEM_roundness_surface_area"
    ].values ** 1.5 / (df_gene["MEM_shape_volume"].values)

    if gene in structures_of_interest:
        print(gene)
        print(f"Count: {np.mean(counts):0.2f} +/- {np.std(counts):.2f}")
        print(
            f"Volume: {np.mean(volume_per_unit):0.2f} +/- {np.std(volume_per_unit):.2f}"
        )
        print(f"Radius: {np.mean(unit_radius):0.2f} +/- {np.std(unit_radius):.2f}")

# %% save updated dataframe
df_stats_path = get_local_stats_dataframe_path()
df_stats.to_parquet(df_stats_path)
