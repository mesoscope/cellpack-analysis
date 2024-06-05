# %% [markdown]
# ## Get distribution of counts and sizes from manifest

# %%
import pandas as pd
import numpy as np

# %%
s3_df_path = "s3://cellpack-analysis-data/variance_dataset.csv"
meta_df = pd.read_csv(s3_df_path, index_col="CellId")

# %%
meta_df["mean_count"] = np.nan
meta_df["std_count"] = np.nan
meta_df["mean_volume"] = np.nan
meta_df["std_volume"] = np.nan
meta_df["mean_radius"] = np.nan
meta_df["std_radius"] = np.nan

structures_of_interest = ["SLC25A17", "RAB5A"]

for gene, df_gene in meta_df.groupby("structure_name"):
    counts = df_gene["STR_connectivity_cc"].values.astype(float)
    volume_per_unit = df_gene["STR_shape_volume"].values / counts
    unit_radius = (volume_per_unit / (4 / 3 * np.pi)) ** (1 / 3)
    df_gene["mean_count"] = np.mean(counts)
    df_gene["std_count"] = np.std(counts)
    df_gene["mean_volume"] = np.mean(volume_per_unit)
    df_gene["std_volume"] = np.std(volume_per_unit)
    df_gene["mean_radius"] = np.mean(unit_radius)
    df_gene["std_radius"] = np.std(unit_radius)

    meta_df.loc[df_gene.index, "mean_count"] = df_gene["mean_count"]
    meta_df.loc[df_gene.index, "std_count"] = df_gene["std_count"]
    meta_df.loc[df_gene.index, "mean_volume"] = df_gene["mean_volume"]
    meta_df.loc[df_gene.index, "std_volume"] = df_gene["std_volume"]
    meta_df.loc[df_gene.index, "mean_radius"] = df_gene["mean_radius"]
    meta_df.loc[df_gene.index, "std_radius"] = df_gene["std_radius"]

    if gene in structures_of_interest:
        print(gene)
        print(f"Count: {np.mean(counts):0.2f} +/- {np.std(counts):.2f}")
        print(
            f"Volume: {np.mean(volume_per_unit):0.2f} +/- {np.std(volume_per_unit):.2f}"
        )
        print(f"Radius: {np.mean(unit_radius):0.2f} +/- {np.std(unit_radius):.2f}")

# %% save updated csv to s3
meta_df.to_csv(s3_df_path)
