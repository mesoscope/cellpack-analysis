# %% [markdown]
# # Plot figures for dual structure exclusion comparisons
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.lib.file_io import get_project_root, read_json

log = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 14})
# %% [markdown]
# ## Set up folders
project_root = get_project_root()
config_folder = (
    project_root / "cellpack_analysis/analysis/pilr_correlation_analysis/configs"
)
config_file = "golgi_endosome_exclusion.json"
base_structure = "Endosome"
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
# ## Get label and color maps for rules and structures
rule_label_map = {}
structure_label_map = {}
color_map = {}
for structure_name, structure_info in config["data"].items():
    if structure_name not in structure_label_map:
        structure_label_map[structure_name] = structure_info["structure_label"]
    for rule, rule_dict in structure_info["rules"].items():
        if rule not in rule_label_map:
            rule_label_map[rule] = rule_dict["label"]
            color_map[rule_dict["label"]] = rule_dict["color"]
# %% [markdown]
# ## Map rule labels
rule_cols = ["rule_1", "rule_2"]
structure_cols = ["structure_1", "structure_2"]
for cols, label_map in [
    (rule_cols, rule_label_map),
    (structure_cols, structure_label_map),
]:
    for col in cols:
        df[col] = df[col].map(label_map)
# %% [markdown]
# ## Prepare data for plotting
base_rule = "Observed"
df_plot = df.loc[
    (df["structure_1"] == base_structure)
    & (df["rule_1"] == base_rule)
    & (df["correlation"] < 0.99),
    ["structure_2", "rule_2", "correlation"],
]
log.info(
    f"Plotting {len(df_plot)} correlations for {base_structure} vs other structures."
)
# %% [markdown]
# ## Get raw correlation values
log.info(
    df_plot.groupby(["rule_2", "structure_2"])["correlation"]
    .agg(["mean", "std"])
    .sort_values(by="mean", ascending=False)
)
# %% [markdown]
# ## Generate barplot
fig, ax = plt.subplots(figsize=(5, 9), dpi=300)
ax = sns.barplot(
    data=df_plot,
    orient="h",
    x="correlation",
    y="rule_2",
    hue="structure_2",
    # legend=False,
    # palette=color_map,
)
lgd = ax.legend()
sns.move_legend(ax, "center right")
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_xlabel("Individual PILR Correlation")
ax.set_ylabel("")
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
for ext in ["png", "svg"]:
    fig.savefig(fig_folder / f"correlation_barplot.{ext}", bbox_inches="tight")
# %% [markdown]
# ## Generate violin plot
fig_v, ax_v = plt.subplots(figsize=(5, 9), dpi=300)
ax_v = sns.violinplot(
    data=df_plot,
    orient="h",
    x="correlation",
    y="rule_2",
    hue="structure_2",
    # legend=False,
    # palette=color_map,
    inner=None,
    cut=0,
    linewidth=1,
    linecolor="k",
)
lgd = ax_v.legend()
sns.move_legend(ax_v, "upper left", bbox_to_anchor=(1, 1))
ax_v.set_xlabel("Individual PILR Correlation")
ax_v.set_ylabel("")
ax_v.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax_v.set_xlim(-0.012, 0.012)
for ext in ["png", "svg"]:
    fig_v.savefig(fig_folder / f"correlation_violinplot.{ext}", bbox_inches="tight")

# %%
