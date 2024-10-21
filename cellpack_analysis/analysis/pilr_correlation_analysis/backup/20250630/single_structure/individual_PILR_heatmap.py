# %% [markdown]
# # Workflow to create individual PILR correlation plots
import itertools
import logging
import logging.config
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

log_config_path = Path(__file__).parents[4] / "logging.conf"
logging.config.fileConfig(log_config_path)
log = logging.getLogger(__name__)

# %% [markdown]
# ## Define channel names
STRUCTURE_ID = "SLC25A17"
STRUCTURE_NAME = "peroxisome"
EXPERIMENT = "rules_shape"

CHANNEL_IDS = {
    "SLC25A17": "Peroxisome",
    # "RAB5A": "Endosome",
    "random": "Random",
    "nucleus_gradient_strong": "Nucleus",
    "membrane_gradient_strong": "Membrane",
    # "struct_gradient": "Structure",
    # "apical_gradient": "Apical",
}

# %% [markdown]
# ## Set up folders
base_folder = (
    Path(__file__).parents[4]
    / f"results/PILR_correlation_analysis/{STRUCTURE_NAME}/{EXPERIMENT}/"
)

fig_folder = base_folder / "figures"
fig_folder.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## load previously calculated correlations
#
# Run `analysis/pilr_correlation_analysis/individual_PILR_correlation_workflow.py`
# to generate the individual PILR correlations

df = pd.read_parquet(base_folder / "individual_PILR_correlations.parquet")

# %% [markdown]
# ## Define channel names and colors
color_dict = {
    "Peroxisome": "green",
    "Endosome": "gold",
    "Random": "gray",
    "Nucleus": "cyan",
    "Membrane": "magenta",
    "Structure": "blue",
    "Apical": "brown",
    "SLC25A17": "green",
    "SEC61B": "blue",
}

# %% [markdown]
# ## Transform dataframe to be used for plotting
recalculate = False
df_clean = df.loc[list(CHANNEL_IDS.keys()), list(CHANNEL_IDS.keys())]

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
        CHANNEL_IDS.keys(), 2
    ):
        log.info(f"Processing {struct1} vs {struct2}")
        df_struct = df_clean.loc[[struct1], [struct2]]
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
                "structure1": CHANNEL_IDS[struct1],
                "structure2": CHANNEL_IDS[struct2],
                "correlation": df_struct.loc[cellid1, cellid2],
            }
            df_plot.loc[ct] = pd.Series(row)
            ct += 1

        df_plot.to_parquet(df_plot_path)
        log.info(f"df_plot contains {df_plot.count()} entries")
    df_plot.dropna(inplace=True)
    df_plot.to_parquet(df_plot_path)
    log.info(f"Saved df_plot to {df_plot_path}")
# %% [markdown]
# ## Create df for stacked barplot
# x_structs = ["Peroxisome", "Endosome"]
# y_structs = ["ER", "Golgi", "Mitochondria"]
# x_structs = ["Peroxisome"]
# y_structs = ["Peroxisome", "Random", "Nucleus", "Membrane"]
x_structs = ["Peroxisome"]
y_structs = ["Random", "Nucleus", "Membrane", "Structure"]
# y_structs = ["Peroxisome", "Random", "Nucleus", "Membrane"]
df_struct = df_plot.loc[
    (df_plot["structure1"].isin(x_structs))
    & (df_plot["structure2"].isin(y_structs))
    & (df_plot["correlation"] != 1)  # remove self-correlations
]
# %% [markdown]
# ## Get average PILR correlation values
df_avg = df_plot.groupby(["structure1", "structure2"])["correlation"].mean()
df_std = df_plot.groupby(["structure1", "structure2"])["correlation"].std()
df_avg.to_csv(base_folder / "average_correlation.csv")
df_std.to_csv(base_folder / "std_correlation.csv")
print(df_avg)
# %% [markdown]
# ## draw plot

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax = sns.barplot(
    data=df_struct,
    orient="h",
    x="correlation",
    y="structure2",
    hue="structure2",
    legend=False,
    palette=color_dict,
)
# %%
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.set_xlabel("Individual PILR Correlation")
ax.set_ylabel("")
ax.set_xlim(-0.0021, 0.0015)
ax.set_title("Correlation with observed peroxisomes")
# ax.set_xticks(np.linspace(-0.008, 0.008, 5))
# lgd = ax.legend()
fig.savefig(fig_folder / f"{STRUCTURE_ID}_correlation_barplot.png", bbox_inches="tight")
fig
# %% [markdown]
# ## Draw violin plot
fig_v, ax_v = plt.subplots(figsize=(5, 5), dpi=300)
plt.rcParams.update({"font.size": 14})
ax_v = sns.violinplot(
    data=df_struct,
    orient="h",
    x="correlation",
    y="structure2",
    hue="structure2",
    legend=False,
    palette=color_dict,
    inner=None,
    cut=0,
    linewidth=1,
    linecolor="k",
)
# %%
ax_v.set_xlim(-0.012, 0.012)
# ax_v.set_xticks(np.linspace(-0.008, 0.008, 3))
# ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
# ax_v.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax_v.set_xlabel("Individual PILR Correlation")
# ax_v.set_ylabel("")
# ax_v.set_title("")
# lgd = ax_v.legend()
fig_v.savefig(
    fig_folder / f"{STRUCTURE_ID}_correlation_violin.png",
    bbox_inches="tight",
)
fig_v

# %%
df_compare = pd.DataFrame(columns=["Rule", "Label", "Correlation"])
# %%
df_compare.loc[len(df_compare)] = ["Random", "SLC25A17", 0.000334]
df_compare.loc[len(df_compare)] = ["Nucleus", "SLC25A17", 0.001677]
df_compare.loc[len(df_compare)] = ["Membrane", "SLC25A17", -0.000988]
df_compare.loc[len(df_compare)] = ["Random", "SEC61B", -0.000270]
df_compare.loc[len(df_compare)] = ["Nucleus", "SEC61B", 0.000599]
df_compare.loc[len(df_compare)] = ["Membrane", "SEC61B", -0.001964]
df_compare.loc[len(df_compare)] = ["Structure", "SEC61B", 0.001373]

# %%
fig_c, ax_c = plt.subplots(figsize=(5, 5), dpi=300)
plt.rcParams.update({"font.size": 14})
ax_c = sns.barplot(
    data=df_compare,
    orient="h",
    x="Correlation",
    y="Rule",
    hue="Label",
    legend=True,
    palette=color_dict,
)
sns.move_legend(ax_c, "upper left", bbox_to_anchor=(1, 1))
# %%
