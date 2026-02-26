# %% [markdown]
# # Get distribution of counts and sizes for punctate structures

import logging

import numpy as np
import pandas as pd

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.get_variance_dataset import get_variance_dataframe

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Load the variance dataframe
meta_df = get_variance_dataframe()
structures_of_interest = ["SLC25A17", "RAB5A"]
dsphere = True

# %% [markdown]
# ## Filter cells per structure using dsphere cell ID lists
filtered_frames: list[pd.DataFrame] = []
for gene, df_gene in meta_df.groupby("structure_name"):
    cell_id_list = get_cell_id_list_for_structure(structure_id=str(gene), dsphere=dsphere)
    filtered_frames.append(df_gene[df_gene.index.astype(str).isin(cell_id_list)])

df_filtered = pd.concat(filtered_frames)

# %% [markdown]
# ## Compute per-punctum measurements (vectorized)
counts = df_filtered["STR_connectivity_cc"].astype(float)
str_volume = df_filtered["STR_shape_volume"]
mem_surface_area = df_filtered["MEM_roundness_surface_area"]
mem_volume = df_filtered["MEM_shape_volume"]
nuc_surface_area = df_filtered["NUC_roundness_surface_area"]
nuc_volume = df_filtered["NUC_shape_volume"]

# Volume and radius per unit (NaN where count == 0 or volume == 0)
volume_per_unit = str_volume.where(counts > 0) / counts.where(counts > 0)
unit_radius = (volume_per_unit.where(volume_per_unit > 0) / (4 / 3 * np.pi)) ** (1 / 3)

# Sphericity: 1 = perfect sphere; NaN where volume == 0
mem_sphericity = (
    np.pi ** (1 / 3) * (6 * mem_volume.where(mem_volume > 0)) ** (2 / 3) / mem_surface_area
)
nuc_sphericity = (
    np.pi ** (1 / 3) * (6 * nuc_volume.where(nuc_volume > 0)) ** (2 / 3) / nuc_surface_area
)

df_stats = pd.DataFrame(
    {
        "CellId": df_filtered.index.astype(str),
        "structure_name": df_filtered["structure_name"],
        "count": counts.values,
        "volume": volume_per_unit.values,
        "radius": unit_radius.values,
        "cell_stage": df_filtered["cell_stage"].values,
        "cell_volume": mem_volume.values,
        "nuc_volume": nuc_volume.values,
        "cell_height": df_filtered["MEM_position_depth"].values,
        "nuc_height": df_filtered["NUC_position_depth"].values,
        "mem_sphericity": mem_sphericity.values,
        "nuc_sphericity": nuc_sphericity.values,
    }
)

# %% [markdown]
# ## Print summary statistics for structures of interest
for gene in structures_of_interest:
    gene_data = df_stats[df_stats["structure_name"] == gene]
    if gene_data.empty or (gene_data["count"] == 0).all():
        print(f"\n{gene}: No valid data (all counts are zero)")
        continue
    print(f"\n{gene}:")
    print(f"  Count: {gene_data['count'].mean():.2f} ± {gene_data['count'].std():.2f}")
    print(f"  Volume: {gene_data['volume'].mean():.2f} ± {gene_data['volume'].std():.2f}")
    print(f"  Radius: {gene_data['radius'].mean():.2f} ± {gene_data['radius'].std():.2f}")

# %% [markdown]
# ## Save updated dataframe
df_stats_path = get_datadir_path() / "structure_stats.parquet"
df_stats.to_parquet(df_stats_path)
logger.info("Saved structure stats to %s", df_stats_path)

# %%
