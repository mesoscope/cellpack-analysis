# %% [markdown]
# # Plot figures for single structure radial bias comparisons
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib.file_io import get_project_root, read_json

log = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 14})
# %% [markdown]
# ## Set up folders
project_root = get_project_root()
config_folder = project_root / "cellpack_analysis/analysis/pilr_correlation_analysis/configs"
config_file = "peroxisome_radial_bias.json"
base_structure = "peroxisome"
config = read_json(config_folder / config_file)
workflow = config["workflow"]
base_folder = project_root / f"results/PILR_correlation_analysis/{workflow}"
fig_folder = base_folder / "figures"
fig_folder.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# ## Load PILR correlations
try:
    df = pd.read_parquet(base_folder / "pilr_correlation.parquet")
except FileNotFoundError:
    raise FileNotFoundError(
        f"File {base_folder / 'pilr_correlation.parquet'} not found. "
        "Please run the correlation workflow first."
    )

# %% [markdown]
# ## Get label and color maps for rules
rule_label_map = {}
color_map = {}
for _strucure_name, structure_info in config["data"].items():
    for rule, rule_dict in structure_info["rules"].items():
        if rule not in rule_label_map:
            rule_label_map[rule] = rule_dict["label"]
            color_map[rule_dict["label"]] = rule_dict["color"]
# %% [markdown]
# ## Map rule labels
cols = ["rule_1", "rule_2"]
for col in cols:
    df[col] = df[col].map(rule_label_map)
# %% [markdown]
# ## Prepare data for plotting
base_rule = "Observed"
df_plot = df.loc[
    (df["structure_1"] == base_structure)
    & (df["structure_2"] == base_structure)
    & (df["rule_1"] == base_rule)
    & (df["correlation"] < 0.99),
    ["rule_2", "correlation"],
]
# %% [markdown]
# ## Get average and std deviation for each rule
df_avg = (
    df_plot.groupby("rule_2")
    .agg(
        correlation_mean=("correlation", "mean"),
        correlation_std=("correlation", "std"),
    )
    .reset_index()
)
print(df_avg)
# %% [markdown]
# ## Generate barplot
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax = sns.barplot(
    data=df_plot,
    orient="h",
    x="correlation",
    y="rule_2",
    hue="rule_2",
    legend=False,
    palette=color_map,
)
ax.set_xlabel("Individual PILR Correlation")
ax.set_ylabel("")
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
for ext in ["png", "svg"]:
    fig.savefig(fig_folder / f"correlation_barplot.{ext}", bbox_inches="tight")
display(fig)
# %% [markdown]
# ## Generate violin plot
fig_v, ax_v = plt.subplots(figsize=(5, 5), dpi=300)
ax_v = sns.violinplot(
    data=df_plot,
    orient="h",
    x="correlation",
    y="rule_2",
    hue="rule_2",
    legend=False,
    palette=color_map,
    inner=None,
    cut=0,
    linewidth=1,
    linecolor="k",
)
ax_v.set_xlabel("Individual PILR Correlation")
ax_v.set_ylabel("")
ax_v.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax_v.set_xlim(-0.012, 0.012)
for ext in ["png", "svg"]:
    fig_v.savefig(fig_folder / f"correlation_violinplot.{ext}", bbox_inches="tight")
display(fig_v)

# %%
