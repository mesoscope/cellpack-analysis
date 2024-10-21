# %% [markdown]
# # Plot figures for concordance
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_ind

from cellpack_analysis.lib.file_io import get_project_root, read_json
from cellpack_analysis.lib.stats import cohens_d

log = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 14})
# %% [markdown]
# ## Set up folders
project_root = get_project_root()
config_folder = (
    project_root / "cellpack_analysis/analysis/pilr_correlation_analysis/configs"
)
config_file = "concordance.json"
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
for strucure_name, structure_info in config["data"].items():
    if strucure_name not in structure_label_map:
        structure_label_map[strucure_name] = structure_info["structure_label"]
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
base_structures = ["Endosome", "Peroxisome"]
base_rules = ["Peroxisome"]
df_plot = df.loc[
    (df["structure_1"].isin(base_structures))
    & ~(df["structure_2"].isin(base_structures))
    & (df["correlation"] < 0.99),
    ["structure_1", "structure_2", "correlation"],
]
# %% [markdown]
# ## Get average and std deviation for each rule
df_avg = (
    df_plot.groupby(["structure_1", "structure_2"])
    .agg(
        correlation_mean=("correlation", "mean"),
        correlation_std=("correlation", "std"),
    )
    .reset_index()
)
print(df_avg)
# %% [markdown]
# ## Run a statistical test to get p-value
samples = {}
for struct, df_struct in df_plot.groupby("structure_2"):
    if struct not in samples:
        samples[struct] = {}
    for puncta, df_puncta in df_struct.groupby("structure_1"):
        samples[struct][puncta] = df_puncta["correlation"].values
    t_stat, p_value = ttest_ind(
        samples[struct][base_structures[0]],
        samples[struct][base_structures[1]],
        equal_var=False,
    )
    cohens_d_value = cohens_d(
        samples[struct][base_structures[0]],
        samples[struct][base_structures[1]],
    )
    effect_size = (
        "small"
        if cohens_d_value < 0.2
        else "medium" if cohens_d_value < 0.5 else "large"
    )
    log.info(
        (
            f"Statistical test for {struct}: t-statistic={t_stat:.3f}, p-value={p_value:.3e}\n"
            f"Cohen's d={cohens_d_value:.3f} (effect size: {effect_size})"
        )
    )

# %% [markdown]
# ## Generate barplot
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax = sns.barplot(
    data=df_plot,
    orient="h",
    x="correlation",
    y="structure_2",
    hue="structure_1",
    # legend=False,
    # palette=color_map,
)
lgd = ax.legend()
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
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
    y="structure_2",
    hue="structure_1",
    # legend=False,
    # palette=color_map,
    inner=None,
    cut=1,
    linewidth=1,
    linecolor="k",
)
lgd = ax_v.legend()
sns.move_legend(ax_v, "upper left", bbox_to_anchor=(1, 1))
ax_v.set_xlabel("Individual PILR Correlation")
ax_v.set_ylabel("")
ax_v.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
# ax_v.set_xlim(-0.012, 0.012)
for ext in ["png", "svg"]:
    fig_v.savefig(fig_folder / f"correlation_violinplot.{ext}", bbox_inches="tight")
display(fig_v)

# %%
