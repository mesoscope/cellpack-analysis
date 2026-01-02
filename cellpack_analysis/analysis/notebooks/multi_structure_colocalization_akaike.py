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
df_list = []
for packing_id, comparison_mode in packing_id_info.items():
    logger.info(f"Loading data for packing id: {packing_id}")

    occupancy_emd_df_path = occupancy_emd_folder / packing_id / "occupancy_emd.parquet"
    occupancy_emd_df = pd.read_parquet(occupancy_emd_df_path)
    tmp_df = occupancy_emd_df.query(
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
    packing_id_names = [p[0][0] for p in pairs] + [p[1][0] for p in pairs]
    data_df = occupancy_emd_df.query(
        "packing_id in @ [p[0][0] for p in pairs] + [p[1][0] for p in pairs]"
    )
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
# ## AIC Analysis: Compare 5-parameter vs 4-parameter models
#
# For each packing pair, we'll compare:
# - 5-parameter model: includes the structure (e.g., ER_peroxisome)
# - 4-parameter model: no structure (e.g., ER_peroxisome_no_struct)


def calculate_aic(emd_values, n_params):
    """
    Calculate AIC for EMD values.
    AIC = 2k - 2ln(L)
    where k is number of parameters and L is likelihood

    For EMD, we treat it as residuals and assume normal distribution
    """
    n = len(emd_values)
    # Calculate log-likelihood assuming normal distribution
    # L = -n/2 * ln(2π) - n/2 * ln(σ²) - 1/(2σ²) * Σ(residuals²)
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

aic_results = []

for with_struct, without_struct in structure_pairs:
    logger.info(f"Comparing {with_struct} vs {without_struct}")

    # Get EMD values for both models
    emd_5param = np.array(
        occupancy_emd_df[occupancy_emd_df["packing_id"] == with_struct]["emd"].values
    )
    emd_4param = np.array(
        occupancy_emd_df[occupancy_emd_df["packing_id"] == without_struct]["emd"].values
    )

    if len(emd_5param) == 0 or len(emd_4param) == 0:
        logger.warning(f"Missing data for {with_struct} or {without_struct}")
        continue

    # Calculate AIC for both models
    aic_5param = calculate_aic(emd_5param, 5)  # 5 parameters with structure
    aic_4param = calculate_aic(emd_4param, 4)  # 4 parameters without structure

    # Calculate AIC difference (positive means 4-param is better, negative means 5-param is better)
    delta_aic = aic_5param - aic_4param

    # Calculate Akaike weights
    weight_5param = np.exp(-0.5 * max(0, -delta_aic))
    weight_4param = np.exp(-0.5 * max(0, delta_aic))
    total_weight = weight_5param + weight_4param
    weight_5param /= total_weight
    weight_4param /= total_weight

    # Statistical test: paired t-test on EMD values if same length
    if len(emd_5param) == len(emd_4param):
        t_stat, p_value = stats.ttest_rel(emd_5param, emd_4param)
    else:
        t_stat, p_value = stats.ttest_ind(emd_5param, emd_4param)

    result = {
        "structure_pair": f"{with_struct} vs {without_struct}",
        "with_structure": with_struct,
        "without_structure": without_struct,
        "aic_5param": aic_5param,
        "aic_4param": aic_4param,
        "delta_aic": delta_aic,
        "preferred_model": (
            "4-param (no structure)" if delta_aic > 0 else "5-param (with structure)"
        ),
        "weight_5param": weight_5param,
        "weight_4param": weight_4param,
        "mean_emd_5param": np.mean(emd_5param),
        "mean_emd_4param": np.mean(emd_4param),
        "std_emd_5param": np.std(emd_5param),
        "std_emd_4param": np.std(emd_4param),
        "emd_difference": np.mean(emd_5param) - np.mean(emd_4param),
        "t_statistic": t_stat,
        "p_value": p_value,
        "n_5param": len(emd_5param),
        "n_4param": len(emd_4param),
    }

    aic_results.append(result)

    logger.info(f"  AIC 5-param: {aic_5param:.3f}")
    logger.info(f"  AIC 4-param: {aic_4param:.3f}")
    logger.info(f"  Delta AIC: {delta_aic:.3f}")
    logger.info(f"  Preferred: {result['preferred_model']}")
    logger.info(f"  p-value: {p_value:.3f}")

# Convert to DataFrame for easier analysis
aic_df = pd.DataFrame(aic_results)

# %% [markdown]
# ## Results Summary

print("=== AIC Model Comparison Results ===")
print()
for _, row in aic_df.iterrows():
    print(f"{row['structure_pair']}:")
    print(f"  Preferred model: {row['preferred_model']}")
    print(f"  Delta AIC: {row['delta_aic']:.3f}")
    print(f"  Evidence ratio: {max(row['weight_5param'], row['weight_4param']):.3f}")
    print(f"  Mean EMD difference: {row['emd_difference']:.4f}")
    print(f"  Statistical significance: p={row['p_value']:.3f}")

    # Interpretation
    if abs(row["delta_aic"]) < 2:
        strength = "weak"
    elif abs(row["delta_aic"]) < 4:
        strength = "moderate"
    else:
        strength = "strong"

    print(f"  Evidence strength: {strength}")
    print()

# Save results
aic_df.to_csv(results_dir / "aic_model_comparison.csv", index=False)
print(f"Results saved to {results_dir / 'aic_model_comparison.csv'}")

# %% [markdown]
# ## Visualization

# Create visualization comparing AIC values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# Plot 1: Delta AIC comparison
structure_names = [pair.split(" vs ")[0].replace("_", " + ") for pair in aic_df["structure_pair"]]
colors = ["red" if x > 0 else "blue" for x in aic_df["delta_aic"]]

bars = ax1.bar(range(len(aic_df)), aic_df["delta_aic"], color=colors, alpha=0.7)
ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax1.axhline(y=2, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
ax1.axhline(y=-2, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
ax1.set_xlabel("Structure Pairs")
ax1.set_ylabel("Δ AIC (5-param - 4-param)")
ax1.set_title("AIC Comparison: Model Preference")
ax1.set_xticks(range(len(aic_df)))
ax1.set_xticklabels(structure_names, rotation=45, ha="right")

# Add text annotations for interpretation
ax1.text(
    0.02,
    0.98,
    "Blue: 5-param preferred\nRed: 4-param preferred",
    transform=ax1.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Add strength indicators
for i, (_, row) in enumerate(aic_df.iterrows()):
    if abs(row["delta_aic"]) >= 2:
        marker = "**" if abs(row["delta_aic"]) >= 4 else "*"
        ax1.text(
            i,
            row["delta_aic"] + 0.1 * np.sign(row["delta_aic"]),
            marker,
            ha="center",
            va="bottom" if row["delta_aic"] > 0 else "top",
        )

# Plot 2: Model weights
x_pos = np.arange(len(aic_df))
width = 0.35

bars1 = ax2.bar(
    x_pos - width / 2,
    aic_df["weight_5param"],
    width,
    label="5-param (with structure)",
    color="blue",
    alpha=0.7,
)
bars2 = ax2.bar(
    x_pos + width / 2,
    aic_df["weight_4param"],
    width,
    label="4-param (no structure)",
    color="red",
    alpha=0.7,
)

ax2.set_xlabel("Structure Pairs")
ax2.set_ylabel("Akaike Weight")
ax2.set_title("Model Evidence Weights")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(structure_names, rotation=45, ha="right")
ax2.legend()
ax2.set_ylim(0, 1)

plt.tight_layout()
fig_path = figures_dir / f"aic_model_comparison.{save_format}"
plt.savefig(fig_path, bbox_inches="tight")
logger.info(f"Saved AIC comparison figure to {fig_path}")

# Create a summary table visualization
fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
ax.axis("tight")
ax.axis("off")

# Prepare table data
table_data = []
for _, row in aic_df.iterrows():
    structure_name = row["structure_pair"].split(" vs ")[0].replace("_", " + ")
    preferred = "With Structure" if row["delta_aic"] < 0 else "No Structure"
    strength = (
        "Strong"
        if abs(row["delta_aic"]) >= 4
        else "Moderate" if abs(row["delta_aic"]) >= 2 else "Weak"
    )
    p_val = f"{row['p_value']:.3f}" if row["p_value"] >= 0.001 else "<0.001"

    table_data.append([structure_name, f"{row['delta_aic']:.2f}", preferred, strength, p_val])

headers = ["Structure Pair", "Δ AIC", "Preferred Model", "Evidence", "p-value"]
table = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)

# Color code the preferred model column
for i in range(len(table_data)):
    if "With Structure" in table_data[i][2]:
        table[(i + 1, 2)].set_facecolor("#add8e6")  # Light blue
    else:
        table[(i + 1, 2)].set_facecolor("#ffcccb")  # Light red

plt.title("AIC Model Comparison Summary", pad=20)
fig_path = figures_dir / f"aic_summary_table.{save_format}"
plt.savefig(fig_path, bbox_inches="tight")
logger.info(f"Saved AIC summary table to {fig_path}")

plt.show()

# %%
