# %% [markdown]
# # Workflow to analyze multi-structure colocalization
#
# Compare colocalization of endosomes and peroxisomes with the ER and Golgi
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator

from cellpack_analysis.lib.file_io import get_project_root

plt.rcParams["font.size"] = 6

logger = logging.getLogger(__name__)
# %% [markdown]
# ## Set up parameters
save_format = "pdf"
distance_measures = [
    "nucleus",
    "z",
]

packing_id_info = {
    "ER_peroxisome": "SLC25A17",
    "golgi_peroxisome": "SLC25A17",
    "ER_endosome": "RAB5A",
    "golgi_endosome": "RAB5A",
    "ER_peroxisome_no_struct": "SLC25A17",
    "golgi_peroxisome_no_struct": "SLC25A17",
    "ER_endosome_no_struct": "RAB5A",
    "golgi_endosome_no_struct": "RAB5A",
}

# %% [markdown]
# ### Set file paths and setup parameters
packing_output_folder = "packing_outputs/8d_sphere_data/norm_weights/"
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

occupancy_emd_folder = base_results_dir / "norm_weights_mixed_rule"

results_dir = base_results_dir / "multi_structure_colocalization_akaike"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures/"
figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Load occupancy EMD data
df_list: list[pd.DataFrame] = []
for packing_id, comparison_mode in packing_id_info.items():  # noqa: B007
    logger.info(f"Loading data for packing id: {packing_id}")

    occupancy_emd_df_path = occupancy_emd_folder / packing_id / "occupancy_emd.parquet"
    packing_id_occupancy_emd_df = pd.read_parquet(occupancy_emd_df_path)
    tmp_df = packing_id_occupancy_emd_df.query(
        "packing_mode_1==@comparison_mode and packing_mode_2=='interpolated'"
    ).copy()
    tmp_df["packing_id"] = packing_id
    tmp_df["condition"] = packing_id.split("_no_struct")[0]
    tmp_df["structure_on"] = "no_struct" not in packing_id
    df_list.append(tmp_df)
occupancy_emd_df = pd.concat(df_list, axis=0, ignore_index=True)

# %% [markdown]
# ## Compare average EMD
pairs_list = [
    [
        (("ER_peroxisome", True), ("ER_peroxisome", False)),
        (("golgi_peroxisome", True), ("golgi_peroxisome", False)),
    ],
    [
        (("ER_endosome", True), ("ER_endosome", False)),
        (("golgi_endosome", True), ("golgi_endosome", False)),
    ],
]

for pairs in pairs_list:
    # Extract unique condition names and get both with and without structure versions
    condition_names = list(set([p[0][0] for p in pairs] + [p[1][0] for p in pairs]))
    packing_id_names = []
    for condition in condition_names:
        packing_id_names.append(condition)
        packing_id_names.append(f"{condition}_no_struct")
    data_df = occupancy_emd_df.query("packing_id in @packing_id_names").copy()
    fig, ax = plt.subplots(figsize=(7, 2.5), dpi=300)
    plot_params = {
        "data": data_df,
        "x": "condition",
        "y": "emd",
        "hue": "structure_on",
        "errorbar": "sd",
    }
    sns.barplot(**plot_params, ax=ax, linewidth=0.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("EMD")
    annotator = Annotator(
        ax,
        pairs,
        plot="barplot",
        **plot_params,
    )
    annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
    annotator.apply_and_annotate()
    sns.despine(ax=ax)
    plt.tight_layout()
    fig_path = (
        figures_dir
        / f"average_occupancy_emd_comparison_{pairs[0][0][0].split('_')[-1]}.{save_format}"
    )
    plt.savefig(fig_path, bbox_inches="tight")
    logger.info(f"Saved figure to {fig_path}")
# %% [markdown]
# ## Bayesian Model Comparison and Effect Size Analysis
#
# For each packing pair, we'll compare:
# - Model with structure (e.g., ER_peroxisome)
# - Model without structure (e.g., ER_peroxisome_no_struct)
#
# Using:
# 1. Bayes Factor: ratio of marginal likelihoods
# 2. Effect Size: Cohen's d and Mann-Whitney U test


def calculate_bayes_factor(emd_with_struct, emd_without_struct):
    """
    Calculate Bayes Factor comparing two models using EMD distributions.

    Assumes EMD values are normally distributed under each model.
    BF > 1 indicates evidence for model with structure.
    BF < 1 indicates evidence for model without structure.

    Interpretation (Kass & Raftery, 1995):
    1-3: Weak evidence
    3-20: Positive evidence
    20-150: Strong evidence
    >150: Very strong evidence
    """
    from scipy.stats import norm

    # Fit normal distributions to each model's EMD values
    mu_with = np.mean(emd_with_struct)
    sigma_with = np.std(emd_with_struct, ddof=1)
    mu_without = np.mean(emd_without_struct)
    sigma_without = np.std(emd_without_struct, ddof=1)

    # Avoid division by zero
    if sigma_with == 0 or sigma_without == 0:
        logger.warning("Zero variance detected, returning BF = 1")
        return 1.0, 0.0, 0.0

    # Calculate log marginal likelihoods
    # For model with structure, evaluate likelihood of all data points
    log_l_with = np.sum(norm.logpdf(emd_with_struct, mu_with, sigma_with))
    # For model without structure
    log_l_without = np.sum(norm.logpdf(emd_without_struct, mu_without, sigma_without))

    # Bayes Factor = L(with) / L(without)
    # Using log: log(BF) = log(L_with) - log(L_without)
    log_bf = log_l_with - log_l_without
    bayes_factor = np.exp(log_bf)

    return bayes_factor, log_bf, mu_with - mu_without


def calculate_effect_size(emd_with_struct, emd_without_struct):
    """
    Calculate effect size (Cohen's d) for the difference in EMD distributions.

    Negative d: EMD lower with structure (better fit with structure)
    Positive d: EMD higher with structure (better fit without structure)

    Interpretation:
    |d| < 0.2: negligible
    0.2 <= |d| < 0.5: small
    0.5 <= |d| < 0.8: medium
    |d| >= 0.8: large
    """
    from scipy.stats import mannwhitneyu

    mean_with = np.mean(emd_with_struct)
    mean_without = np.mean(emd_without_struct)

    # Pooled standard deviation
    n_with = len(emd_with_struct)
    n_without = len(emd_without_struct)
    var_with = np.var(emd_with_struct, ddof=1)
    var_without = np.var(emd_without_struct, ddof=1)

    pooled_std = np.sqrt(
        ((n_with - 1) * var_with + (n_without - 1) * var_without) / (n_with + n_without - 2)
    )

    if pooled_std == 0:
        cohens_d = 0.0
    else:
        cohens_d = (mean_with - mean_without) / pooled_std

    # Mann-Whitney U test (non-parametric alternative to t-test)
    u_stat, mw_p_value = mannwhitneyu(emd_with_struct, emd_without_struct, alternative="two-sided")

    # Effect size r for Mann-Whitney: r = Z / sqrt(N)
    n_total = n_with + n_without
    z_score = stats.norm.ppf(1 - mw_p_value / 2)  # Convert p-value to Z
    r_effect = z_score / np.sqrt(n_total)

    return cohens_d, mw_p_value, u_stat, r_effect


def calculate_aic(emd_values, n_params):
    """
    Calculate AIC for EMD values.

    AIC = 2k - 2ln(L)
    where k is number of parameters and L is likelihood.

    For EMD, we treat it as residuals and assume normal distribution.
    """
    n = len(emd_values)
    # Calculate log-likelihood assuming normal distribution
    # L = -n/2 * ln(2π) - n/2 * ln(sigma²) - 1/(2sigma²) * Σ(residuals²)
    # For simplicity, we use mean squared error as proxy for -2ln(L)
    mse = np.mean(emd_values**2)
    # sigma2 = np.var(emd_values, ddof=1)
    # AIC = 2k + n*ln(MSE)
    # aic = 2 * n_params - n * (np.log(2 * np.pi * sigma2) + 1)
    aic = 2 * n_params + n * np.log(mse)
    return aic


# Create pairs for comparison
structure_pairs = [
    ("ER_peroxisome", "ER_peroxisome_no_struct"),
    ("golgi_peroxisome", "golgi_peroxisome_no_struct"),
    ("ER_endosome", "ER_endosome_no_struct"),
    ("golgi_endosome", "golgi_endosome_no_struct"),
]

# %% [markdown]
# ### Run Bayesian and Effect Size Analysis
analysis_results = []

for with_struct, without_struct in structure_pairs:
    logger.info(f"Comparing {with_struct} vs {without_struct}")

    # Get EMD values for both models
    emd_with_struct = np.array(
        occupancy_emd_df[occupancy_emd_df["packing_id"] == with_struct]["emd"].values
    )
    emd_without_struct = np.array(
        occupancy_emd_df[occupancy_emd_df["packing_id"] == without_struct]["emd"].values
    )

    if len(emd_with_struct) == 0 or len(emd_without_struct) == 0:
        logger.warning(f"Missing data for {with_struct} or {without_struct}")
        continue

    # Calculate Bayes Factor
    bayes_factor, log_bf, mean_diff = calculate_bayes_factor(emd_with_struct, emd_without_struct)

    # Calculate Effect Size
    cohens_d, mw_p_value, u_stat, r_effect = calculate_effect_size(
        emd_with_struct, emd_without_struct
    )

    # Calculate AIC for comparison (kept for reference)
    aic_with_struct = calculate_aic(emd_with_struct, 5)  # 5 parameters with structure
    aic_without_struct = calculate_aic(emd_without_struct, 4)  # 4 parameters without structure
    delta_aic = aic_with_struct - aic_without_struct

    # Interpret Bayes Factor
    if bayes_factor > 150:
        bf_interpretation = "Very strong evidence for structure"
    elif bayes_factor > 20:
        bf_interpretation = "Strong evidence for structure"
    elif bayes_factor > 3:
        bf_interpretation = "Positive evidence for structure"
    elif bayes_factor > 1:
        bf_interpretation = "Weak evidence for structure"
    elif bayes_factor > 1 / 3:
        bf_interpretation = "Weak evidence against structure"
    elif bayes_factor > 1 / 20:
        bf_interpretation = "Positive evidence against structure"
    elif bayes_factor > 1 / 150:
        bf_interpretation = "Strong evidence against structure"
    else:
        bf_interpretation = "Very strong evidence against structure"

    # Interpret Effect Size
    abs_d = abs(cohens_d)
    if abs_d >= 0.8:
        effect_interpretation = "Large"
    elif abs_d >= 0.5:
        effect_interpretation = "Medium"
    elif abs_d >= 0.2:
        effect_interpretation = "Small"
    else:
        effect_interpretation = "Negligible"

    # Determine preferred model based on Bayes Factor
    if bayes_factor > 1:
        preferred_model = "with structure"
    else:
        preferred_model = "without structure"

    result = {
        "structure_pair": f"{with_struct} vs {without_struct}",
        "with_structure": with_struct,
        "without_structure": without_struct,
        # Bayes Factor metrics
        "bayes_factor": bayes_factor,
        "log_bayes_factor": log_bf,
        "bf_interpretation": bf_interpretation,
        # Effect size metrics
        "cohens_d": cohens_d,
        "effect_size_interpretation": effect_interpretation,
        "mann_whitney_p": mw_p_value,
        "mann_whitney_u": u_stat,
        "r_effect_size": r_effect,
        # Descriptive statistics
        "mean_emd_with_struct": np.mean(emd_with_struct),
        "mean_emd_without_struct": np.mean(emd_without_struct),
        "std_emd_with_struct": np.std(emd_with_struct),
        "std_emd_without_struct": np.std(emd_without_struct),
        "median_emd_with_struct": np.median(emd_with_struct),
        "median_emd_without_struct": np.median(emd_without_struct),
        "emd_mean_difference": np.mean(emd_with_struct) - np.mean(emd_without_struct),
        "emd_median_difference": np.median(emd_with_struct) - np.median(emd_without_struct),
        # Sample sizes
        "n_with_struct": len(emd_with_struct),
        "n_without_struct": len(emd_without_struct),
        # AIC for reference
        "aic_with_struct": aic_with_struct,
        "aic_without_struct": aic_without_struct,
        "delta_aic": delta_aic,
        # Preferred model
        "preferred_model": preferred_model,
    }

    analysis_results.append(result)

    # Log results
    logger.info(f"  Bayes Factor: {bayes_factor:.3f} ({bf_interpretation})")
    logger.info(f"  Cohen's d: {cohens_d:.3f} ({effect_interpretation} effect)")
    logger.info(f"  Mann-Whitney p-value: {mw_p_value:.4f}")
    logger.info(f"  Mean EMD difference: {result['emd_mean_difference']:.4f}")
    logger.info(f"  Preferred model: {preferred_model}")

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(analysis_results)

# %% [markdown]
# ## Results Summary

print("=== Bayesian Model Comparison & Effect Size Analysis ===")
print()
for _, row in results_df.iterrows():
    print(f"{row['structure_pair']}:")
    print(f"  Preferred model: {row['preferred_model']}")
    print(f"  Bayes Factor: {row['bayes_factor']:.3f} ({row['bf_interpretation']})")
    print(f"  Cohen's d: {row['cohens_d']:.3f} ({row['effect_size_interpretation']} effect)")
    print(f"  Mean EMD difference: {row['emd_mean_difference']:.4f}")
    print(f"  Mann-Whitney p-value: {row['mann_whitney_p']:.4f}")
    print(f"  Delta AIC (reference): {row['delta_aic']:.3f}")
    print()

# Save results
results_df.to_csv(results_dir / "bayesian_effect_size_comparison.csv", index=False)
print(f"Results saved to {results_dir / 'bayesian_effect_size_comparison.csv'}")

# %% [markdown]
# ## Visualization

# Create comprehensive visualization
fig = plt.figure(figsize=(14, 10), dpi=300)
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

structure_names = [
    pair.split(" vs ")[0].replace("_", " + ") for pair in results_df["structure_pair"]
]

# Plot 1: Bayes Factor (log scale)
ax1 = fig.add_subplot(gs[0, 0])
log_bf = results_df["log_bayes_factor"].values
colors_bf = ["blue" if x > 0 else "red" for x in log_bf]

bars = ax1.bar(range(len(results_df)), log_bf.tolist(), color=colors_bf, alpha=0.7)
ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax1.axhline(y=np.log(3), color="gray", linestyle="--", linewidth=0.5, alpha=0.7, label="Positive")
ax1.axhline(y=-np.log(3), color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
ax1.axhline(y=np.log(20), color="gray", linestyle=":", linewidth=0.5, alpha=0.7, label="Strong")
ax1.axhline(y=-np.log(20), color="gray", linestyle=":", linewidth=0.5, alpha=0.7)

ax1.set_xlabel("Structure Pairs")
ax1.set_ylabel("log(Bayes Factor)")
ax1.set_title("Bayes Factor: Evidence for Structure")
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels(structure_names, rotation=45, ha="right")
ax1.text(
    0.02,
    0.98,
    "Blue: Evidence FOR structure\nRed: Evidence AGAINST structure",
    transform=ax1.transAxes,
    verticalalignment="top",
    fontsize=8,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
)

# Plot 2: Effect Size (Cohen's d)
ax2 = fig.add_subplot(gs[0, 1])
cohens_d = results_df["cohens_d"].values
colors_d = ["blue" if x < 0 else "red" for x in cohens_d]

bars = ax2.bar(range(len(results_df)), cohens_d.tolist(), color=colors_d, alpha=0.7)
ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax2.axhline(y=0.2, color="gray", linestyle="--", linewidth=0.5, alpha=0.7, label="Small")
ax2.axhline(y=-0.2, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
ax2.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.7, label="Medium")
ax2.axhline(y=-0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
ax2.axhline(y=0.8, color="orange", linestyle="-.", linewidth=0.5, alpha=0.7, label="Large")
ax2.axhline(y=-0.8, color="orange", linestyle="-.", linewidth=0.5, alpha=0.7)

ax2.set_xlabel("Structure Pairs")
ax2.set_ylabel("Cohen's d")
ax2.set_title("Effect Size: EMD Difference")
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels(structure_names, rotation=45, ha="right")
ax2.text(
    0.02,
    0.98,
    "Blue: Lower EMD with structure\nRed: Higher EMD with structure",
    transform=ax2.transAxes,
    verticalalignment="top",
    fontsize=8,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
)

# Plot 3: Mean EMD comparison
ax3 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(results_df))
width = 0.35

bars1 = ax3.bar(
    x_pos - width / 2,
    results_df["mean_emd_with_struct"],
    width,
    label="With structure",
    color="blue",
    alpha=0.7,
)
bars2 = ax3.bar(
    x_pos + width / 2,
    results_df["mean_emd_without_struct"],
    width,
    label="Without structure",
    color="red",
    alpha=0.7,
)

# Add error bars
ax3.errorbar(
    x_pos - width / 2,
    results_df["mean_emd_with_struct"],
    yerr=results_df["std_emd_with_struct"],
    fmt="none",
    ecolor="black",
    capsize=3,
    linewidth=0.5,
)
ax3.errorbar(
    x_pos + width / 2,
    results_df["mean_emd_without_struct"],
    yerr=results_df["std_emd_without_struct"],
    fmt="none",
    ecolor="black",
    capsize=3,
    linewidth=0.5,
)

ax3.set_xlabel("Structure Pairs")
ax3.set_ylabel("Mean EMD")
ax3.set_title("EMD Comparison (mean ± SD)")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(structure_names, rotation=45, ha="right")
ax3.legend()

# Plot 4: Statistical significance
ax4 = fig.add_subplot(gs[1, 1])
p_values = results_df["mann_whitney_p"].values
log_p = -np.log10(p_values)
colors_p = ["green" if p < 0.05 else "gray" for p in p_values]

bars = ax4.bar(range(len(results_df)), log_p, color=colors_p, alpha=0.7)
ax4.axhline(y=-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05")
ax4.axhline(y=-np.log10(0.01), color="orange", linestyle="--", linewidth=1, label="p=0.01")
ax4.axhline(y=-np.log10(0.001), color="darkred", linestyle="--", linewidth=1, label="p=0.001")

ax4.set_xlabel("Structure Pairs")
ax4.set_ylabel("-log10(p-value)")
ax4.set_title("Statistical Significance (Mann-Whitney U)")
ax4.set_xticks(range(len(results_df)))
ax4.set_xticklabels(structure_names, rotation=45, ha="right")
ax4.legend(fontsize=7)

# Plot 5: Delta AIC (for comparison with new methods)
ax5 = fig.add_subplot(gs[2, 0])
delta_aic = results_df["delta_aic"].values
colors_aic = ["red" if x > 0 else "blue" for x in delta_aic]

bars = ax5.bar(range(len(results_df)), delta_aic.tolist(), color=colors_aic, alpha=0.7)
ax5.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax5.axhline(y=2, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
ax5.axhline(y=-2, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

ax5.set_xlabel("Structure Pairs")
ax5.set_ylabel("Δ AIC (with - without)")
ax5.set_title("AIC Comparison (for reference)")
ax5.set_xticks(range(len(results_df)))
ax5.set_xticklabels(structure_names, rotation=45, ha="right")
ax5.text(
    0.02,
    0.98,
    "Note: AIC may not be appropriate\nfor this analysis. Use BF instead.",
    transform=ax5.transAxes,
    verticalalignment="top",
    fontsize=7,
    bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.8},
)

# Plot 6: Combined summary (BF vs Effect Size)
ax6 = fig.add_subplot(gs[2, 1])
scatter = ax6.scatter(
    results_df["cohens_d"],
    results_df["log_bayes_factor"],
    s=100,
    c=results_df["mann_whitney_p"],
    cmap="RdYlGn_r",
    alpha=0.7,
    edgecolors="black",
    linewidth=0.5,
)

# Add labels for each point
for i in range(len(results_df)):
    ax6.annotate(
        structure_names[i],
        (results_df.iloc[i]["cohens_d"], results_df.iloc[i]["log_bayes_factor"]),
        fontsize=7,
        ha="left",
        va="bottom",
    )

ax6.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
ax6.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
ax6.set_xlabel("Cohen's d (Effect Size)")
ax6.set_ylabel("log(Bayes Factor)")
ax6.set_title("Effect Size vs Bayes Factor")
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label("p-value", fontsize=8)

plt.tight_layout()
fig_path = figures_dir / f"bayesian_effect_size_analysis.{save_format}"
plt.savefig(fig_path, bbox_inches="tight")
logger.info(f"Saved comprehensive analysis figure to {fig_path}")

# %% [markdown]
# ## Summary Table

fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
ax.axis("tight")
ax.axis("off")

# Prepare table data
table_data = []
for _, row in results_df.iterrows():
    structure_name = row["structure_pair"].split(" vs ")[0].replace("_", " + ")
    bf_str = f"{row['bayes_factor']:.2f}"
    effect_str = f"{row['cohens_d']:.3f}"
    p_val = f"{row['mann_whitney_p']:.3f}" if row["mann_whitney_p"] >= 0.001 else "<0.001"
    preferred = "With" if row["preferred_model"] == "with structure" else "Without"

    table_data.append(
        [
            structure_name,
            bf_str,
            row["bf_interpretation"],
            effect_str,
            row["effect_size_interpretation"],
            p_val,
            preferred,
        ]
    )

headers = [
    "Structure Pair",
    "Bayes Factor",
    "BF Interpretation",
    "Cohen's d",
    "Effect Size",
    "p-value",
    "Preferred",
]
table = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.2, 2.0)

# Color code the preferred model column
for i in range(len(table_data)):
    if "With" in table_data[i][6]:
        table[(i + 1, 6)].set_facecolor("#add8e6")  # Light blue
    else:
        table[(i + 1, 6)].set_facecolor("#ffcccb")  # Light red

    # Color code p-values
    try:
        p_val_float = float(table_data[i][5].replace("<", ""))
        if p_val_float < 0.001:
            table[(i + 1, 5)].set_facecolor("#90EE90")  # Light green
        elif p_val_float < 0.05:
            table[(i + 1, 5)].set_facecolor("#FFFFE0")  # Light yellow
    except (ValueError, IndexError):
        pass

plt.title("Bayesian Model Comparison & Effect Size Summary", pad=20, fontsize=12)
fig_path = figures_dir / f"bayesian_effect_size_summary_table.{save_format}"
plt.savefig(fig_path, bbox_inches="tight")
logger.info(f"Saved summary table to {fig_path}")

plt.show()

# %%
