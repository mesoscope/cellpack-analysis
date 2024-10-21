# %% [markdown]
# # Workflow to create individual PILR correlation plots for colocolization analysis
import itertools
import logging
import logging.config

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_project_root

plt.rcParams.update({"font.size": 16})

project_root = get_project_root()
log_config_path = project_root / "logging.conf"
logging.config.fileConfig(log_config_path)
log = logging.getLogger(__name__)

base_folder = (
    project_root
    / "results/PILR_correlation_analysis/colocalization/individual_PILR_correlations"
)

# %% [markdown]
# ## load previously calculated correlations
#
# Run `raw_data_individual_PILR_correlation_workflow.py`
# to generate the individual PILR correlations

df = pd.read_parquet(base_folder / "individual_PILR_correlations.parquet")

# %% [markdown]
# ## Define channel names and colors
STRUCTURE_IDS = {
    "SLC25A17": "Peroxisome",
    "RAB5A": "Endosome",
    "SEC61B": "ER",
    "ST6GAL1": "Golgi",
    "TOMM20": "Mitochondria",
}

color_dict = {
    "SLC25A17": "#509750",
    "RAB5A": "#978250",
    "SEC61B": "blue",
    "ST6GAL1": "orange",
    "TOMM20": "purple",
}

# %% [markdown]
# ## Transform dataframe to be used for plotting
recalculate = False
df_clean = df.loc[list(STRUCTURE_IDS.keys()), list(STRUCTURE_IDS.keys())]

df_plot_path = base_folder / "individual_PILR_correlations_plot.parquet"
if not recalculate and df_plot_path.exists():
    df_plot = pd.read_parquet(df_plot_path)
else:
    df_plot = pd.DataFrame(
        index=np.arange(df_clean.shape[0] * df_clean.shape[1]),
        columns=["cellid1", "cellid2", "structure1", "structure2", "correlation"],
    )
    ct = 0
    for struct1, struct2 in itertools.combinations_with_replacement(
        STRUCTURE_IDS.keys(), 2
    ):
        log.info(f"Processing {struct1} vs {struct2}")
        df_struct = df_clean.loc[struct1, struct2]
        cellids1 = df_struct.index
        cellids2 = df_struct.columns
        for cellid1, cellid2 in tqdm(
            itertools.product(cellids1, cellids2),
            total=len(cellids1) * len(cellids2),
            desc=f"Processing {struct1} vs {struct2}",
        ):
            row = {
                "cellid1": cellid1,
                "cellid2": cellid2,
                "structure1": STRUCTURE_IDS[struct1],
                "structure2": STRUCTURE_IDS[struct2],
                "correlation": df_struct.loc[cellid1, cellid2],
            }
            df_plot.loc[ct] = row
            ct += 1

        df_plot.to_parquet(df_plot_path)
        log.info(f"df_plot contains {df_plot.count()} entries")
    df_plot.dropna(inplace=True)
    df_plot.to_parquet(df_plot_path)
    log.info(f"Saved df_plot to {df_plot_path}")
# %% [markdown]
# ## Create df for stacked barplot
df_struct = df_plot.loc[
    ((df_plot["structure1"] == "Peroxisome") | (df_plot["structure1"] == "Endosome"))
    & (
        (df_plot["structure2"] == "ER")
        | (df_plot["structure2"] == "Golgi")
        | (df_plot["structure2"] == "Mitochondria")
    )
]
# %% [markdown]
# ## draw plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
ax = sns.barplot(
    data=df_struct,
    orient="h",
    x="correlation",
    y="structure2",
    hue="structure1",
    palette=[color_dict["SLC25A17"], color_dict["RAB5A"]],
)
# %%
ax.set_xlabel("Individual PILR Correlation")
ax.set_ylabel("")
ax.set_title("")
lgd = ax.legend()
fig.savefig(
    base_folder / "figures/peroxisome_endosome_correlation.svg", bbox_inches="tight"
)
fig
# %% [markdown]
# ## Draw violin plot
fig_v, ax_v = plt.subplots(figsize=(5, 5), dpi=300)
ax_v = sns.violinplot(
    data=df_struct,
    orient="h",
    x="correlation",
    y="structure2",
    hue="structure1",
    palette=[color_dict["SLC25A17"], color_dict["RAB5A"]],
)
# %%
ax_v.set_xlim(-0.015, 0.035)
ax_v.set_xlabel("Individual PILR Correlation")
ax_v.set_ylabel("")
ax_v.set_title("")
lgd = ax_v.legend()
fig_v.savefig(
    base_folder / "figures/peroxisome_endosome_correlation_violin.png",
    bbox_inches="tight",
)
fig_v

# %%
