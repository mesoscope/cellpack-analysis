# %% [markdown]
# # Workflow to analyze multi-structure colocalization
#
# Compare colocalization of endosomes and peroxisomes with the ER and Golgi
# using the following packing modes:
#   1. interpolated with structure (K=5)
#   2. interpolated without structure (K=4)

import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from statannotations.Annotator import Annotator

from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    get_project_root,
    make_dir,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.label_tables import COLOR_PALETTE
from cellpack_analysis.lib.stats import calculate_emd_effect_size, calculate_model_comparison

plt.rcParams["font.size"] = 8

logger = logging.getLogger(__name__)


def _interpret_effect(d: float) -> str:
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "Large"
    elif abs_d >= 0.5:
        return "Medium"
    elif abs_d >= 0.2:
        return "Small"
    return "Negligible"


def _fmt_p(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


# %% [markdown]
# ## Set up parameters
save_format = "pdf"

mode_config = {
    "ER_peroxisome": {"baseline": "SLC25A17", "interpolated": "SEC61B"},
    "golgi_peroxisome": {"baseline": "SLC25A17", "interpolated": "ST6GAL1"},
    "ER_endosome": {"baseline": "RAB5A", "interpolated": "SEC61B"},
    "golgi_endosome": {"baseline": "RAB5A", "interpolated": "ST6GAL1"},
    "ER_peroxisome_no_struct": {"baseline": "SLC25A17", "interpolated": "SEC61B"},
    "golgi_peroxisome_no_struct": {"baseline": "SLC25A17", "interpolated": "ST6GAL1"},
    "ER_endosome_no_struct": {"baseline": "RAB5A", "interpolated": "SEC61B"},
    "golgi_endosome_no_struct": {"baseline": "RAB5A", "interpolated": "ST6GAL1"},
}
base_conditions = ["ER_peroxisome", "golgi_peroxisome", "ER_endosome", "golgi_endosome"]
occupancy_distance_measures = ["nucleus", "z"]

# %%
# Model complexity parameters (number of free parameters excluding error variance)
K_INTERPOLATED_WITH = 5
K_INTERPOLATED_NO = 4

# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_results_dir = project_root / "results"

output_dir = base_results_dir / "rule_interpolation_e2e_validation"

results_dir = make_dir(base_results_dir / "multi_structure_colocalization")

figures_dir = make_dir(results_dir / "figures")

log_path = results_dir / "multi_structure_colocalization.log"

# %% [markdown]
# ### Load data

occupancy_emd_df_list: list[pd.DataFrame] = []
distance_emd_df_list: list[pd.DataFrame] = []
occupancy_dict = {packing_id: {} for packing_id in mode_config.keys()}

for packing_id, config_dict in mode_config.items():
    logger.info(f"Loading data for packing id: {packing_id}")
    baseline_mode = config_dict["baseline"]

    for dm in occupancy_distance_measures:
        occupancy_path = output_dir / packing_id / f"{dm}_occupancy.dat"
        with open(occupancy_path, "rb") as f:
            dm_occupancy_dict = pickle.load(f)
        occupancy_dict[packing_id][dm] = {
            mode: dm_occupancy_dict[mode] for mode in [baseline_mode, "interpolated"]
        }

    for emd_type, emd_list in zip(
        ["occupancy_emd", "distance_pairwise_emd"],
        [occupancy_emd_df_list, distance_emd_df_list],
        strict=True,
    ):
        emd_df_path = output_dir / packing_id / f"{emd_type}.parquet"
        emd_df = pd.read_parquet(emd_df_path)
        condition = packing_id.split("_no_struct")[0]

        tmp = emd_df.query(
            "packing_mode_1 == @baseline_mode and packing_mode_2 == 'interpolated'"
        ).copy()
        tmp = tmp.groupby("cell_id_1")["emd"].mean().reset_index()
        tmp["mode_category"] = (
            "interpolated_no" if "no_struct" in packing_id else "interpolated_with"
        )
        tmp["packing_id"] = packing_id
        tmp["condition"] = condition
        emd_list.append(tmp)

occupancy_emd_df = pd.concat(occupancy_emd_df_list, ignore_index=True)
distance_emd_df = pd.concat(distance_emd_df_list, ignore_index=True)
# %% [markdown]
# ## Plot occupancy ratio
color_palette_adjusted = {
    "SLC25A17": COLOR_PALETTE["SLC25A17"],
    "RAB5A": COLOR_PALETTE["RAB5A"],
    "interpolated_no_struct": "gray",
    "interpolated_with_struct": "black",
}
ylims = {"nucleus": (0, 2), "z": (0, 1.5)}
plt.rcParams["font.size"] = 6
for condition in base_conditions:
    baseline_mode = mode_config[condition]["baseline"]
    for dm in occupancy_distance_measures:
        fig, ax = plt.subplots(figsize=(1.5, 1), dpi=300)

        for mode in [baseline_mode, "interpolated", "no_struct"]:
            condition_key = condition if mode != "no_struct" else f"{condition}_no_struct"
            color_key = (
                mode
                if mode == baseline_mode
                else f"interpolated_{'with_struct' if 'no_struct' not in condition_key else 'no_struct'}"
            )
            mode_key = mode if mode != "no_struct" else "interpolated"
            zorder = 3 if mode == baseline_mode else 2
            alpha = 1.0 if mode == baseline_mode else 0.8
            xvals = occupancy_dict[condition_key][dm][mode_key]["combined"]["xvals"]
            yvals = occupancy_dict[condition_key][dm][mode_key]["combined"]["occupancy"]
            envelope_hi = occupancy_dict[condition_key][dm][mode_key]["combined"]["envelope_hi"]
            envelope_lo = occupancy_dict[condition_key][dm][mode_key]["combined"]["envelope_lo"]
            ax.plot(
                xvals,
                yvals,
                color=color_palette_adjusted[color_key],
                linewidth=1.5,
                label=color_key.replace("_", " ").title(),
                zorder=zorder,
                alpha=alpha,
            )
            ax.fill_between(
                xvals,
                envelope_lo,
                envelope_hi,
                color=color_palette_adjusted[color_key],
                linewidth=0,
                alpha=0.1,
                zorder=1,
            )
            xlim = (0, max(xvals))
            ax.set_xlim(xlim)
            ax.set_ylim(ylims[dm])
            # ax.set_xlabel(DISTANCE_MEASURE_LABELS[dm])
            ax.set_ylabel("Occupancy Ratio")
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5, alpha=0.7, zorder=1)
            sns.despine(ax=ax)
            plt.tight_layout()
            fig_path = figures_dir / f"combined_occupancy_{condition}_{dm}.{save_format}"
            plt.savefig(fig_path, bbox_inches="tight")
            logger.info(f"Saved figure to {fig_path}")


# %% [markdown]
# ## Bar plot comparisons
#
# For each organelle group (peroxisome, endosome), plot EMD comparisons.
# Pairwise Wilcoxon annotations with and without struct
organelle_groups: dict[str, list[str]] = {
    "peroxisome": ["ER_peroxisome", "golgi_peroxisome"],
    "endosome": ["ER_endosome", "golgi_endosome"],
}

hue_order = ["interpolated_with", "interpolated_no"]
hue_color_mapping = {
    "interpolated_with": color_palette_adjusted["interpolated_with_struct"],
    "interpolated_no": color_palette_adjusted["interpolated_no_struct"],
}

for organelle, conditions in organelle_groups.items():
    for emd_type, emd_df in zip(
        ["Occupancy EMD", "Distance EMD"], [occupancy_emd_df, distance_emd_df], strict=True
    ):
        logger.info(f"Plotting {emd_type} for {organelle}")
        data_df = emd_df[emd_df["condition"].isin(conditions)].copy()

        pairs = []
        for cond in conditions:
            pairs.append(((cond, "interpolated_with"), (cond, "interpolated_no")))

        fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
        plot_params = {
            "data": data_df,
            "x": "condition",
            "y": "emd",
            "hue": "mode_category",
            "hue_order": hue_order,
            "whis": [2.5, 97.5],
            "showfliers": False,
            "palette": hue_color_mapping,
        }
        sns.boxplot(**plot_params, ax=ax, linewidth=0.1)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("EMD")
        ax.set_title(f"{organelle.capitalize()} colocalization, {emd_type}")

        annotator = Annotator(ax, pairs, plot="boxplot", **plot_params)
        annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
        # annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
        annotator.apply_and_annotate()

        sns.despine(ax=ax)
        plt.tight_layout()
        fig_path = (
            figures_dir
            / f"average_{'_'.join(emd_type.split()).lower()}_comparison_{organelle}.{save_format}"
        )
        fig.savefig(fig_path, bbox_inches="tight")
        logger.info(f"Saved figure to {fig_path}")

# %% [markdown]
# ## Model Comparison and Effect Size Analysis
#
# For each base condition and each EMD type, compare interpolated_with (K=5) vs interpolated_no (K=4)
analysis_results = []

for emd_type, emd_df in zip(
    ["occupancy", "distance"], [occupancy_emd_df, distance_emd_df], strict=True
):
    for condition in base_conditions:
        logger.info(f"Analyzing condition: {condition}, EMD type: {emd_type}")

        emd_iw = np.array(
            emd_df.loc[
                (emd_df["condition"] == condition)
                & (emd_df["mode_category"] == "interpolated_with"),
                "emd",
            ].values
        )
        emd_no = np.array(
            emd_df.loc[
                (emd_df["condition"] == condition) & (emd_df["mode_category"] == "interpolated_no"),
                "emd",
            ].values
        )

        if any(len(x) == 0 for x in [emd_iw, emd_no]):
            logger.warning(f"Missing data for condition {condition} ({emd_type}), skipping")
            continue

        mc = calculate_model_comparison(emd_iw, emd_no, K_INTERPOLATED_WITH, K_INTERPOLATED_NO)
        cohens_d, p_value, test_statistic, r = calculate_emd_effect_size(
            emd_iw, emd_no, test="wilcoxon"
        )

        result = {
            "emd_type": emd_type,
            "condition": condition,
            "mean_emd_iw": np.mean(emd_iw),
            "mean_emd_no": np.mean(emd_no),
            "n_iw": len(emd_iw),
            "n_no": len(emd_no),
            "aic_with": mc["aic_with"],
            "aic_without": mc["aic_without"],
            "w_aic_with": mc["w_aic_with"],
            "w_aic_without": mc["w_aic_without"],
            "er_aic": mc["er_aic"],
            "best_model_aic": mc["best_model_aic"],
            "delta_aic_interpretation": mc["delta_aic_interpretation"],
            "cohens_d": cohens_d,
            "effect_interpretation": _interpret_effect(cohens_d),
            "p_value": p_value,
            "test_statistic": test_statistic,
            "r_effect": r,
            "emd_mean_diff": np.mean(emd_iw) - np.mean(emd_no),
        }
        analysis_results.append(result)

        logger.info(
            f"  interpolated_with vs interpolated_no ({emd_type}): "
            f"w_AIC={mc['w_aic_with']:.3f}/{mc['w_aic_without']:.3f}, "
            f"d={cohens_d:.3f}, p={p_value:.4f}"
        )

results_df = pd.DataFrame(analysis_results)

# %% [markdown]
# ## Results Summary
add_file_handler_to_logger(logger, log_path)
logger.info(
    "=== Model Comparison & Effect Size Analysis (interpolated_with vs interpolated_no) ==="
)
for _, row in results_df.iterrows():
    logger.info(f"{row['condition']} ({row['emd_type']} EMD):")
    logger.info(f"      Best model (AIC): {row['best_model_aic']}")
    logger.info(
        f"      Akaike weights: with={row['w_aic_with']:.3f}, without={row['w_aic_without']:.3f}"
    )
    logger.info(
        f"      Evidence ratio: {row['er_aic']:.1g}x  |  ΔAIC: {row['delta_aic_interpretation']}"
    )
    logger.info(
        f"      Cohen's d={row['cohens_d']:.3f} ({row['effect_interpretation']}), "
        f"p={row['p_value']:.4f}"
    )
remove_file_handler_from_logger(logger, log_path)
results_df.to_csv(results_dir / "model_comparison_effect_size.csv", index=False)
logger.info(f"Results saved to {results_dir / 'model_comparison_effect_size.csv'}")

# %% [markdown]
# ## AIC Weights and Cohen's d

emd_types = ["occupancy", "distance"]
bar_w = 0.35

for emd_type in emd_types:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300)

    # Akaike weights
    ax = axes[0]
    df_sub = results_df[results_df["emd_type"] == emd_type].reset_index(drop=True)
    condition_labels = [c.replace("_", "\n") for c in df_sub["condition"]]
    x_pos = np.arange(len(df_sub))

    ax.bar(
        x_pos - bar_w / 2,
        df_sub["w_aic_with"],
        bar_w,
        label="With structure",
        color="black",
        alpha=0.7,
    )
    ax.bar(
        x_pos + bar_w / 2,
        df_sub["w_aic_without"],
        bar_w,
        label="Without structure",
        color="grey",
        alpha=0.7,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Akaike Weight", fontsize=9)
    ax.set_title(f"{emd_type.capitalize()} EMD", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(condition_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # Cohen's d
    ax = axes[1]

    vals = df_sub["cohens_d"].to_numpy()
    colors_d = ["grey" if v < 0 else "black" for v in vals]
    ax.bar(x_pos, vals.tolist(), color=colors_d, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    for level, style, lbl in [
        (0.2, "--", "Small (0.2)"),
        (0.5, ":", "Medium (0.5)"),
        (0.8, "-.", "Large (0.8)"),
    ]:
        color = "orange" if level == 0.8 else "gray"
        ax.axhline(y=level, color=color, linestyle=style, linewidth=0.5, alpha=0.7, label=lbl)
        ax.axhline(y=-level, color=color, linestyle=style, linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Cohen's d", fontsize=9)
    ax.set_title(f"{emd_type.capitalize()} EMD", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(condition_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    ax.text(
        0.02,
        0.98,
        "Black: lower EMD with structure\nGrey: higher EMD with structure",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=7,
    )
    sns.despine(ax=ax)

    fig.suptitle(
        "Model Comparison & Effect Size (interpolated_with vs interpolated_no)", fontsize=10
    )
    plt.tight_layout()
    fig_path = figures_dir / f"{emd_type}_model_comparison_effect_size_analysis.{save_format}"
    plt.savefig(fig_path, bbox_inches="tight")


# %%
