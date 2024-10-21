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
from IPython.display import display
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

log_config_path = Path(__file__).parents[4] / "logging.conf"
logging.config.fileConfig(log_config_path)
log = logging.getLogger(__name__)

# %% [markdown]
# ## Define channel names, colors, and labels
FOLDER_ID = "multi_structure_colocalization"

STRUCTURE_ID = "SLC25A17"
STRUCTURE_NAME = "ER_peroxisome"

LABELS = {
    "peroxisome": "Peroxisome",
    "ER_peroxisome": "ER Peroxisome",
    "ER_peroxisome_no_struct": "ER Peroxisome no struct",
    "observed": "Observed",
    "random": "Random",
    "nucleus_gradient_strong": "Nucleus",
    "membrane_gradient_strong": "Membrane",
}

RULE_MAP = {
    "peroxisome": [
        "observed",
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
    ],
    # "ER_peroxisome": [
    #     "random",
    #     "nucleus_gradient_strong",
    #     "membrane_gradient_strong",
    #     "struct_gradient",
    # ],
    # "ER_peroxisome_no_struct": [
    #     "random",
    #     "nucleus_gradient_strong",
    #     "membrane_gradient_strong",
    # ],
    # "golgi_endosome": [
    #     "random",
    #     "nucleus_gradient_strong",
    #     "membrane_gradient_strong",
    #     "struct_gradient",
    # ],
    # "golgi_endosome_no_struct": [
    #     "random",
    #     "nucleus_gradient_strong",
    #     "membrane_gradient_strong",
    # ],
}

COLORS = {
    "Peroxisome": "green",
    "Endosome": "gold",
    "Random": "gray",
    "Nucleus": "cyan",
    "Membrane": "magenta",
    "Structure": "blue",
    "Apical": "brown",
    "ER_peroxisome": "C0",
    "ER_peroxisome_no_struct": "C1",
    "Observed": "green",
}
# %% [markdown]
# ## Set up folders
base_folder = (
    Path(__file__).parents[4]
    / f"results/PILR_correlation_analysis/{FOLDER_ID}/{STRUCTURE_NAME}"
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
# ## Transform dataframe to be used for plotting
recalculate = True
df_clean = df.loc[list(RULE_MAP.keys()), list(RULE_MAP.keys())]

df_plot_path = base_folder / "individual_PILR_correlations_plot.parquet"
if not recalculate and df_plot_path.exists():
    df_plot = pd.read_parquet(df_plot_path)
else:
    df_plot = pd.DataFrame(
        index=np.arange(df_clean.shape[0] * df_clean.shape[1]),
        columns=[
            "cellid1",
            "cellid2",
            "rule1",
            "rule2",
            "structure1",
            "structure2",
            "correlation",
        ],
    )
    ct = 0
    for struct1, struct2 in itertools.combinations_with_replacement(RULE_MAP.keys(), 2):
        print(f"structure1: {struct1}, structure2: {struct2}")
        log.info(f"Processing structure {struct1} vs {struct2}")
        df_struct = df_clean.loc[struct1, struct2]
        for rule1, rule2 in itertools.product(RULE_MAP[struct1], RULE_MAP[struct2]):
            log.info(f"Processing rule {rule1} vs {rule2}")
            df_rule = df_struct.loc[rule1, rule2]
            cellids1 = df_rule.index
            cellids2 = df_rule.columns
            for cellid1, cellid2 in tqdm(
                itertools.product(cellids1, cellids2),
                total=len(cellids1) * len(cellids2),
                desc=f"Processing {rule1} vs {rule2}",
            ):
                row = {
                    "cellid1": cellid1,
                    "cellid2": cellid2,
                    "rule1": rule1,
                    "rule2": rule2,
                    "structure1": struct1,
                    "structure2": struct2,
                    "correlation": df_rule.loc[cellid1, cellid2],
                }
                df_plot.loc[ct] = pd.Series(row)
                ct += 1

        df_plot.to_parquet(df_plot_path)
        log.info(f"df_plot contains {df_plot.count()} entries")
    df_plot.dropna(inplace=True)
    df_plot.to_parquet(df_plot_path)
    log.info(f"Saved df_plot to {df_plot_path}")
# %% [markdown]
# ## Relabel entries
cols = ["rule1", "rule2", "structure1", "structure2"]
for col in cols:
    df_plot[col] = df_plot[col].map(LABELS).fillna(df_plot[col])
# %% [markdown]
# ## Create df for stacked barplot
# x_structs = ["Peroxisome", "Endosome"]
# y_structs = ["ER", "Golgi", "Mitochondria"]
x_structs = ["Peroxisome"]
y_structs = ["Peroxisome", "Random", "Nucleus", "Membrane"]
# x_structs = ["peroxisome"]
# y_structs = ["peroxisome", "ER_peroxisome", "ER_peroxisome_no_struct"]
x_rules = ["Observed"]
df_struct = df_plot.loc[
    (df_plot["structure1"].isin(x_structs))
    & (df_plot["rule1"].isin(x_rules))
    # & (df_plot["structure2"].isin(y_structs))
    & (df_plot["correlation"] != 1)  # remove self-correlations
]
log.info(f"df_struct contains {df_struct.count()} entries")
# %% [markdown]
# ## Barplot of correlations across rules
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax = sns.barplot(
    data=df_struct,
    orient="h",
    x="correlation",
    y="rule2",
    # hue="structure2",
    legend=True,
    palette=COLORS,
)
# %%
# ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
ax.set_xlabel("Individual PILR Correlation")
ax.set_ylabel("")
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
# ax.set_xlim(-0.008, 0.008)
# ax.set_title("Correlation with observed peroxisomes")
# ax.set_xticks(np.linspace(-0.008, 0.008, 5))
# lgd = ax.legend()
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig.savefig(fig_folder / f"{STRUCTURE_ID}_correlation.png", bbox_inches="tight")
display(fig)
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
    palette=COLORS,
    inner=None,
    cut=0,
    linewidth=1,
    linecolor="k",
)
# %%
ax_v.set_xlim(-0.012, 0.012)
ax_v.set_xticks(np.linspace(-0.008, 0.008, 3))
ax_v.set_xlabel("Individual PILR Correlation")
ax_v.set_ylabel("")
ax_v.set_title("")
# lgd = ax_v.legend()
fig_v.savefig(
    base_folder / "figures/pilr_correlation_violin.png",
    bbox_inches="tight",
)
fig_v

# %% [markdown]
# ## Get average PILR correlation values
df_observed = df_plot.loc[
    (df_plot["structure1"] == "peroxisome")
    & (df_plot["rule1"] == "observed")
    & (df_plot["correlation"] != 1)  # remove self-correlations
]
# %%
df_avg = df_observed.groupby(["structure2", "rule2"])["correlation"].mean()
df_std = df_observed.groupby(["structure2", "rule2"])["correlation"].std()
df_avg.to_csv(base_folder / "average_correlation.csv")
df_std.to_csv(base_folder / "std_correlation.csv")
print(df_avg)
# %%
