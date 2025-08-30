import logging
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import squareform
from tqdm import tqdm

from cellpack_analysis.analysis.punctate_analysis.lib.distance import (
    filter_invalid_distances,
    get_normalization_factor,
    get_scaled_structure_radius,
)
from cellpack_analysis.analysis.punctate_analysis.lib.stats_functions import (
    create_padded_numpy_array,
    get_pdf_ratio,
    sample_cellids_from_distance_dict,
)
from cellpack_analysis.lib.default_values import PIX_SIZE
from cellpack_analysis.lib.label_tables import (
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
    MODE_LABELS,
    NORMALIZATION_LABELS,
)

log = logging.getLogger(__name__)


def plot_cell_diameter_distribution(
    mesh_information_dict: dict[Any, dict[str, float]],
) -> None:
    """
    Plots the distribution of cell and nucleus diameters.

    This function calculates the cell and nucleus diameters from the mesh information dictionary,
    and then plots the histograms of the diameters. It also displays the mean cell and nucleus
    diameters in the plot title.

    Parameters
    ----------
    mesh_information_dict
        A dictionary containing mesh information with cell_diameter and nuc_diameter keys
    """
    cell_diameters = [
        cellid_dict["cell_diameter"] * PIX_SIZE
        for _, cellid_dict in mesh_information_dict.items()
    ]
    nuc_diameters = [
        cellid_dict["nuc_diameter"] * PIX_SIZE
        for _, cellid_dict in mesh_information_dict.items()
    ]
    fig, ax = plt.subplots(dpi=300)
    ax.hist(cell_diameters, bins=20, alpha=0.5, label="cell")
    ax.hist(nuc_diameters, bins=20, alpha=0.5, label="nucleus")
    ax.set_title(
        f"Mean cell diameter: {np.mean(cell_diameters):.2f}\u03bcm\n"
        f"Mean nucleus diameter: {np.mean(nuc_diameters):.2f}\u03bcm"
    )
    ax.set_xlabel("Diameter (\u03bcm)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_distance_distributions_kde(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    overlay: bool = False,
    distance_limits: dict[str, tuple[float, float]] | None = None,
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> None:
    """
    Plot distance distributions using kernel density estimation (KDE).

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    packing_modes
        List of packing modes to plot
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    suffix
        Suffix to append to the figure filenames
    normalization
        Normalization method to apply to the distance measures
    overlay
        If True, overlay the pooled KDE
    distance_limits
        Dictionary containing limits for each distance measure
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value
    save_format
        Format to save the figures in
    """
    log.info("Starting distance distribution kde plot")
    num_cols = len(packing_modes)
    cmap = plt.get_cmap("tab10")

    for distance_measure in distance_measures:
        distance_dict = all_distance_dict[distance_measure]
        fig, axs = plt.subplots(
            1,
            num_cols,
            figsize=(num_cols * 3, 4),
            dpi=300,
            sharex=True,
            sharey=True,
        )
        for mode_index, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]
            log.info(
                f"Plotting distance histogram for: {distance_measure}, Mode: {mode}"
            )

            ax = axs[mode_index]
            # plot individual kde plots of distance distributions
            for k, (_, distances) in tqdm(
                enumerate(mode_dict.items()), total=len(mode_dict)
            ):
                sns.kdeplot(
                    distances,
                    ax=ax,
                    color=cmap(mode_index),
                    linewidth=1,
                    alpha=0.05,
                    bw_method=bandwidth,
                    cut=0,
                )
                # break

            # plot combined kde plot of distance distributions
            combined_mode_distances = np.concatenate(list(mode_dict.values()))
            combined_mode_distances = filter_invalid_distances(combined_mode_distances)
            if overlay:
                sns.kdeplot(
                    combined_mode_distances,
                    ax=ax,
                    color=cmap(mode_index),
                    linewidth=3,
                    label=MODE_LABELS.get(mode, mode),
                    bw_method=bandwidth,
                    cut=0,
                )

            # plot mean distance and add title
            mean_distance = np.nanmean(combined_mode_distances)
            if normalization is None and "scaled" not in distance_measure:
                unit = "\u03bcm"
            else:
                unit = ""
            ax.set_title(
                f"{MODE_LABELS.get(mode, mode)}\nMean: {mean_distance:.2f}{unit}"
            )
            if distance_limits is not None:
                ax.set_xlim(distance_limits.get(distance_measure, (0, 1)))
            else:
                min_xlim = np.nanmin(combined_mode_distances)
                max_xlim = np.nanmax(combined_mode_distances)
                min_xlim = min_xlim - 0.1 * (max_xlim - min_xlim)
                max_xlim = max_xlim + 0.1 * (max_xlim - min_xlim)
                ax.set_xlim(min_xlim, max_xlim)

            ax.axvline(mean_distance, color="k", linestyle="--")

        if normalization is not None:
            distance_label = (
                f"{DISTANCE_MEASURE_LABELS[distance_measure]}"
                f" / {NORMALIZATION_LABELS[normalization]}"
            )
        elif "scaled" in distance_measure:
            distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
        else:
            distance_label = f"{DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)"

        fig.supxlabel(distance_label)
        # fig.suptitle(f"{DISTANCE_MEASURE_LABELS[distance_measure]}")
        fig.tight_layout()

        if figures_dir is not None:
            distance_kde_dir = figures_dir / "distance_kde"
            distance_kde_dir.mkdir(exist_ok=True)
            fig.savefig(
                distance_kde_dir
                / f"distance_distributions_{distance_measure}{suffix}.{save_format}",
                dpi=300,
            )

        plt.show()


def plot_distance_distributions_kde_vertical(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    overlay: bool = False,
    distance_limits: dict[str, tuple[float, float]] | None = None,
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> tuple[list[Any], list[Any]]:
    """
    Plot distance distributions using kernel density estimation (KDE) in vertical layout.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    packing_modes
        List of packing modes to plot
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    suffix
        Suffix to append to the figure filenames
    normalization
        Normalization method to apply to the distance measures
    overlay
        If True, overlay the pooled KDE
    distance_limits
        Dictionary containing limits for each distance measure
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing lists of figure and axis objects
    """
    log.info("Starting distance distribution kde plot")
    num_rows = len(packing_modes)

    fig_list, ax_list = [], []

    for col, distance_measure in enumerate(distance_measures):
        fig, axs = plt.subplots(
            num_rows,
            1,
            figsize=(6, num_rows * 1.5),
            dpi=300,
            sharex="col",
            sharey="col",
        )
        fig_list.append(fig)
        ax_list.append(axs)
        distance_dict = all_distance_dict[distance_measure]
        for row, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]
            log.info(f"Plotting distance kde for: {distance_measure}, Mode: {mode}")
            cmap = plt.get_cmap("jet", len(mode_dict))

            ax = axs[row]
            # plot individual kde plots of distance distributions
            color_inds = np.random.permutation(len(mode_dict))
            for k, (_, distances) in tqdm(
                enumerate(mode_dict.items()), total=len(mode_dict)
            ):
                color = cmap(color_inds[k])

                sns.kdeplot(
                    distances,
                    ax=ax,
                    color=color,
                    linewidth=1,
                    alpha=0.2,
                    bw_method=bandwidth,
                    cut=0,
                )

            # plot combined kde plot of distance distributions
            combined_mode_distances = np.concatenate(list(mode_dict.values()))
            combined_mode_distances = filter_invalid_distances(combined_mode_distances)
            if overlay:
                sns.kdeplot(
                    combined_mode_distances,
                    ax=ax,
                    color="k",
                    linewidth=1.5,
                    label=MODE_LABELS.get(mode, mode),
                    bw_method=bandwidth,
                    cut=0,
                )

            # plot mean distance and add title
            mean_distance = np.nanmean(combined_mode_distances)
            # if normalization is None and "scaled" not in distance_measure:
            #     unit = "\u03BCm"
            # else:
            #     unit = ""
            # ax.set_title(
            #     f"{MODE_LABELS.get(mode, mode)}\nMean: {mean_distance:.2f}{unit}"
            # )
            if distance_limits is not None:
                ax.set_xlim(distance_limits.get(distance_measure, (0, 1)))
            else:
                min_xlim = np.nanmin(combined_mode_distances)
                max_xlim = np.nanmax(combined_mode_distances)
                min_xlim = min_xlim - 0.1 * (max_xlim - min_xlim)
                max_xlim = max_xlim + 0.1 * (max_xlim - min_xlim)
                ax.set_xlim(min_xlim, max_xlim)

            ax.axvline(mean_distance, color="k", linestyle="--")

            if col == 0:
                ax.set_ylabel(f"{MODE_LABELS.get(mode, mode)}\nDensity")

            if row == num_rows - 1:
                if normalization is not None:
                    distance_label = (
                        f"{DISTANCE_MEASURE_LABELS[distance_measure]}"
                        f" / {NORMALIZATION_LABELS[normalization]}"
                    )
                elif "scaled" in distance_measure:
                    distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
                else:
                    distance_label = (
                        f"{DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)"
                    )
                ax.set_xlabel(distance_label)

        # fig.suptitle(f"{DISTANCE_MEASURE_LABELS[distance_measure]}")
        # fig.supylabel(MODE_LABELS.get(mode, mode))
        fig.tight_layout()

        if figures_dir is not None:
            distance_kde_dir = figures_dir / "distance_kde"
            distance_kde_dir.mkdir(exist_ok=True)
            fig.savefig(
                distance_kde_dir
                / f"distance_distributions_vertical_{distance_measure}{suffix}.{save_format}",
                dpi=300,
            )

        plt.show()
    return fig_list, ax_list


def plot_ks_test_distance_distributions_kde(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    ks_observed_dict: dict[str, dict[str, dict[str, float]]],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    baseline_mode: str | None = None,
    significance_level: float = 0.05,
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> None:
    """
    Plot distance distributions for packings with significant and non-significant KS test results.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    packing_modes
        List of packing modes to plot
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure
    ks_observed_dict
        Dictionary containing KS test results for each packing mode and distance measure
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    suffix
        Suffix to append to the figure filenames
    normalization
        Normalization method to apply to the distance measures
    baseline_mode
        Packing mode to use as baseline for comparison
    significance_level
        Significance level for the KS test
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value
    save_format
    """
    log.info("Plotting distance distributions")

    if baseline_mode is not None:
        packing_modes = [mode for mode in packing_modes if mode != baseline_mode]

    num_cols = len(packing_modes)

    for i, distance_measure in enumerate(distance_measures):
        distance_dict = all_distance_dict[distance_measure]
        ks_measure_dict = ks_observed_dict[distance_measure]
        fig, axs = plt.subplots(
            2,
            num_cols,
            figsize=(num_cols * 4, 8),
            dpi=300,
            sharex="row",
            sharey="row",
        )
        for j, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]
            ks_mode_dict = ks_measure_dict[mode]
            log.info(f"Distance measure: {distance_measure}, Mode: {mode}")

            sig_ax = axs[0, j]
            ns_ax = axs[1, j]

            # get significant and non-significant seeds
            all_seeds = list(mode_dict.keys())
            significant_seeds = [
                seed
                for seed, p_value in ks_mode_dict.items()
                if p_value < significance_level
            ]
            ns_seeds = [seed for seed in all_seeds if seed not in significant_seeds]

            # kde plots for significant seeds
            for seed in tqdm(significant_seeds):
                distances = mode_dict[seed]
                sns.kdeplot(
                    distances,
                    ax=sig_ax,
                    color="r",
                    linewidth=1,
                    alpha=0.2,
                    bw_method=bandwidth,
                    cut=0,
                )
            # plot combined kde plot of distance distributions
            if len(significant_seeds) > 0:
                combined_sig_distances = np.concatenate(
                    [
                        distance
                        for seed, distance in mode_dict.items()
                        if seed in significant_seeds
                    ]
                )
                sns.kdeplot(
                    combined_sig_distances,
                    ax=sig_ax,
                    color="k",
                    linewidth=2,
                    bw_method=bandwidth,
                    cut=0,
                )
                # plot mean distance and add title
                mean_distance = combined_sig_distances.mean()
                title_str = (
                    f"Mean: {mean_distance:.2f}\nKS p-value < {significance_level}"
                )
                sig_ax.axvline(mean_distance, color="k", linestyle="--")
            else:
                title_str = "No significant obs."
            if i == 0:
                sig_ax.set_title(f"{MODE_LABELS.get(mode, mode)}\n{title_str}")
            else:
                sig_ax.set_title(title_str)

            # kde plots for non-significant seeds
            for seed in tqdm(ns_seeds):
                distances = mode_dict[seed]
                sns.kdeplot(
                    distances,
                    ax=ns_ax,
                    color="g",
                    linewidth=1,
                    alpha=0.2,
                    bw_method=bandwidth,
                    cut=0,
                )
            # plot combined kde plot of distance distributions
            if len(ns_seeds) > 0:
                combined_ns_distances = np.concatenate(
                    [
                        distance
                        for seed, distance in mode_dict.items()
                        if seed in ns_seeds
                    ]
                )
                sns.kdeplot(
                    combined_ns_distances,
                    ax=ns_ax,
                    color="k",
                    linewidth=2,
                    bw_method=bandwidth,
                    cut=0,
                )
                # plot mean distance and add title
                mean_distance = combined_ns_distances.mean()
                title_str = (
                    f"Mean: {mean_distance:.2f}\nKS p-value >= {significance_level}"
                )
                ns_ax.axvline(mean_distance, color="k", linestyle="--")
            else:
                title_str = "No non-significant obs."
            ns_ax.set_title(title_str)

        distance_label = "distance"
        if normalization is not None:
            distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
        else:
            distance_label = f"{distance_label} (\u03bcm)"
        fig.supylabel(f"{distance_measure} PDF")
        fig.supxlabel(distance_label)
        fig.tight_layout()

        if figures_dir is not None:
            fig.savefig(
                figures_dir
                / f"distance_distributions_ks_test_{distance_measure}{suffix}.{save_format}",
                dpi=300,
            )

    plt.show()


def plot_distance_distributions_overlay(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> None:
    """
    Plot overlaid distance distributions using kernel density estimation (KDE).

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    packing_modes
        List of packing modes to plot
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    suffix
        Suffix to append to the figure filenames
    normalization
        Normalization method to apply to the distance measures
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value
    save_format
        Format to save the figures in
    """
    log.info("Plotting overlaid distance distributions")

    for distance_measure in distance_measures:
        distance_dict = all_distance_dict[distance_measure]
        fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
        cmap = plt.get_cmap("tab10")
        for j, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]
            log.info(f"Distance measure: {distance_measure}, Mode: {mode}")

            # plot combined kde plot of distance distributions
            combined_mode_distances = np.concatenate(list(mode_dict.values()))
            sns.kdeplot(
                combined_mode_distances,
                ax=ax,
                color=cmap(j),
                linewidth=2,
                label=MODE_LABELS.get(mode, mode),
                bw_method=bandwidth,
                cut=0,
            )

            # plot mean distance
            mean_distance = combined_mode_distances.mean()
            ax.axvline(mean_distance, color=cmap(j), linestyle="--", label="_nolegend_")

        distance_label = "distance"
        if normalization is not None:
            distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
        else:
            distance_label = f"{distance_label} (\u03bcm)"

        ax.set_xlabel(distance_label)
        ax.set_title(f"{distance_measure} distance")
        ax.legend()
        fig.tight_layout()

        if figures_dir is not None:
            fig.savefig(
                figures_dir
                / f"distance_distributions_{distance_measure}{suffix}.{save_format}",
                dpi=300,
            )

        plt.show()


def plot_distance_distributions_histogram(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    save_format: str = "png",
) -> None:
    """
    Plot distance distributions using histograms.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    packing_modes
        List of packing modes to plot
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    suffix
        Suffix to append to the figure filenames
    normalization
        Normalization method to apply to the distance measures
    save_format
        Format to save the figures in
    """
    num_cols = len(packing_modes)

    for i, distance_measure in enumerate(distance_measures):
        fig, axs = plt.subplots(
            1,
            num_cols,
            figsize=(num_cols * 3, 4),
            dpi=300,
            sharex=True,
            sharey=True,
        )
        distance_dict = all_distance_dict[distance_measure]
        for j, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]

            combined_mode_distances = np.concatenate(list(mode_dict.values()))

            # plot histogram
            axs[j].hist(
                combined_mode_distances,
                bins=50,
            )

            # plot mean distance and add title
            unit = ""
            if normalization is None and "scaled" not in distance_measure:
                unit = "(\u03bcm)"
            mean_distance = combined_mode_distances.mean()
            title_str = f"Mean: {mean_distance:.2f}"
            axs[j].set_title(f"{MODE_LABELS.get(mode, mode)}\n{title_str}{unit}")

            axs[j].axvline(mean_distance, color="k", linestyle="--")

            # add x and y labels

        if normalization is not None:
            distance_label = (
                f"{DISTANCE_MEASURE_LABELS[distance_measure]}"
                f" / {NORMALIZATION_LABELS[normalization]}"
            )
        elif "scaled" in distance_measure:
            distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
        else:
            distance_label = f"{DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)"

        fig.supylabel("Count")
        fig.supxlabel(distance_label)
        fig.tight_layout()

        if figures_dir is not None:
            hist_dir = figures_dir / "distance_histogram"
            hist_dir.mkdir(exist_ok=True)
            fig.savefig(
                hist_dir
                / f"distance_distributions_histogram_{distance_measure}{suffix}.{save_format}",
                dpi=300,
            )

        plt.show()


def plot_ks_observed_barplots(
    df_melt: pd.DataFrame,
    figures_dir: Path | None = None,
    suffix: str = "",
    significance_level: float = 0.05,
    save_format: str = "png",
) -> tuple[Any, Any]:
    """
    Plot KS observed results as bar plots.

    Parameters
    ----------
    df_melt
        Melted DataFrame with KS test results
    figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    significance_level
        Significance level for the KS test
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    sns.barplot(
        data=df_melt,
        x="packing_mode",
        y="ks_observed",
        hue="distance_measure",
        ax=ax,
    )
    # xticklabels = [
    #     MODE_LABELS.get(mode.get_text(), mode.get_text()) for mode in ax.get_xticklabels()
    # ]
    # xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    # ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_xlabel("Packing Mode")
    ax.set_ylabel(
        f"Statistically indistinguishable fraction\n(p-value >= {significance_level})"
    )
    ax.legend(
        title="Distance Measure",
    )
    plt.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"observed_ks_test_barplot{suffix}.{save_format}",
            dpi=300,
        )
    plt.show()
    return fig, ax


def plot_emd_heatmaps(
    distance_measures: list[str],
    all_pairwise_emd: dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]],
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: str = "png",
) -> None:
    """
    Plot EMD values as heatmaps.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    all_pairwise_emd
        Dictionary containing pairwise EMD values
    figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    save_format
        Format to save the figures in
    """
    log.info("Plotting pairwise EMD heatmaps")
    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure]
        fig, axs = plt.subplots(len(emd_dict), len(emd_dict), figsize=(6, 6), dpi=300)
        for rt, (mode_1, mode_1_dict) in enumerate(emd_dict.items()):
            mode_1_label = MODE_LABELS[mode_1]
            for ct, (mode_2, emd) in enumerate(mode_1_dict.items()):
                log.info(f"{distance_measure}, {mode_1}, {mode_2}")
                mode_2_label = MODE_LABELS[mode_2]
                if mode_1 == mode_2:
                    values = squareform(list(emd.values()))
                else:
                    values = list(emd.values())
                    values = np.array(values)
                    dim_len = int(np.sqrt(len(values)))
                    values = values.reshape((dim_len, -1))
                ax = axs[rt, ct]
                if ct == len(emd_dict) - 1:
                    cbar = True
                    cbar_kws = {"label": "EMD"}
                else:
                    cbar = False
                    cbar_kws = None
                sns.heatmap(
                    values,
                    ax=ax,
                    cmap="PuOr",
                    # vmin=0,
                    # vmax=0.05,
                    square=True,
                    cbar=cbar,
                    cbar_kws=cbar_kws,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if rt == 0:
                    ax.set_title(mode_2_label)
                if ct == 0:
                    ax.set_ylabel(mode_1_label)
        fig.suptitle(f"{distance_measure} EMD")
        fig.tight_layout()

        if figures_dir is not None:
            fig.savefig(
                figures_dir / f"{distance_measure}_emd{suffix}.{save_format}", dpi=300
            )

        plt.show()


def plot_average_emd_correlation_heatmap(
    distance_measures: list[str],
    corr_df_dict: dict[str, dict[str, pd.DataFrame]],
    emd_figures_dir: Path | None = None,
    suffix: str = "",
) -> None:
    """
    Plot the correlation heatmap for Earth Mover's Distance (EMD) values.

    Parameters
    ----------
    distance_measures
        List of distance measures
    corr_df_dict
        Dictionary containing correlation DataFrames
    emd_figures_dir
        Directory to save the heatmap images
    suffix
        Suffix to add to the heatmap image filenames
    """
    for distance_measure in distance_measures:
        fig, ax = plt.subplots(dpi=300)
        df_corr = corr_df_dict[distance_measure]["mean"]
        sns.heatmap(df_corr, cmap="PuOr", annot=True, ax=ax)
        plt.tight_layout()
        ax.set_title(f"{distance_measure} EMD")
        if emd_figures_dir is not None:
            fig.savefig(
                emd_figures_dir / f"{distance_measure}_emd_heatmap{suffix}.png",
                dpi=300,
            )


def plot_emd_correlation_circles(
    distance_measures: list[str],
    corr_df_dict: dict[str, dict[str, pd.DataFrame]],
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: str = "png",
) -> None:
    """
    Plot EMD correlation as circles with size indicating variation.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    corr_df_dict
        Dictionary containing correlation and standard deviation DataFrames
    figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    save_format
        Format to save the figures in
    """
    log.info("Plotting EMD variation circles")

    for distance_measure in distance_measures:
        corr_dict = corr_df_dict[distance_measure]
        df_corr = corr_dict["mean"]
        df_std = corr_dict["std"]

        N = M = len(df_corr)
        xlabels = ylabels = [MODE_LABELS.get(col, col) for col in df_corr.columns]

        xvals = np.arange(M)
        yvals = np.arange(N)

        x, y = np.meshgrid(xvals, yvals)
        stdev = df_std.to_numpy()
        corr = df_corr.to_numpy()

        fig, ax = plt.subplots(dpi=300)

        R = stdev / stdev.max() / 2.5
        circles = [
            Circle((j, i), radius=r)
            for r, j, i in zip(R.flat, x.flat, y.flat, strict=False)
        ]
        col = PatchCollection(
            circles,
            array=corr.flatten(),
            cmap="PuOr",
        )
        ax.add_collection(col)

        ax.set(
            xticks=np.arange(M),
            yticks=np.arange(N),
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlim(-0.5, M - 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.invert_yaxis()

        ax.set(
            xticks=np.arange(M),
            yticks=np.arange(N),
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
        ax.grid(which="minor")
        # ax.grid(which="minor")
        # for item in ax.spines.__dict__["_dict"].values():
        #     item.set_visible(False)
        # ax.tick_params(which="both", length=0)

        # for i in range(N):
        #     for j in range(M):
        #         ax.text(j, i, f"{corr[i, j]:.3g}", ha="center", va="center", color="black")

        cbar = fig.colorbar(col)
        cbar.set_label(f"EMD for {DISTANCE_MEASURE_LABELS[distance_measure]}")
        # ax.set_frame_on(False)
        # ax.set_title(f"{distance_measure} EMD")

        if figures_dir is not None:
            fig.savefig(
                figures_dir
                / f"{distance_measure}_emd_heatmap_circ{suffix}.{save_format}",
                dpi=300,
            )
        plt.show()


def plot_emd_boxplots(
    distance_measures: list[str],
    all_pairwise_emd: dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]],
    baseline_mode: str = "SLC25A17",
    suffix: str = "",
    figures_dir: Path | None = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> None:
    """
    Plot boxplots of Earth Mover's Distance (EMD) values for different distance measures.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    all_pairwise_emd
        Dictionary containing pairwise EMD values for each distance measure and mode
    baseline_mode
        The baseline packing mode to use for comparison
    suffix
        Suffix to append to the figure filenames
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    save_format
        Format to save the figures in
    """
    log.info("Plotting EMD variation boxplots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_dict = {
            k: v for k, v in emd_dict.items() if k != baseline_mode
        }  # exclude baseline mode for self comparison
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
        sns.boxplot(data=emd_df, ax=ax)  # type:ignore
        ax.set_ylabel("Earth mover's distance (EMD)")
        ax.set_title(DISTANCE_MEASURE_LABELS[distance_measure])
        ax.set_xticklabels([MODE_LABELS[m] for m in emd_df.columns], rotation=45)
        if figures_dir is not None:
            fig.savefig(
                figures_dir / f"{distance_measure}_emd_boxplot{suffix}.{save_format}",
                dpi=300,
            )


def plot_emd_barplots(
    distance_measures: list[str],
    all_pairwise_emd: dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]],
    baseline_mode: str = "SLC25A17",
    suffix: str = "",
    figures_dir: Path | None = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> None:
    """
    Plot barplots of Earth Mover's Distance (EMD) values for different distance measures.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    all_pairwise_emd
        Dictionary containing pairwise EMD values for each distance measure and mode
    baseline_mode
        The baseline packing mode to use for comparison
    suffix
        Suffix to append to the figure filenames
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    save_format
        Format to save the figures in
    """
    log.info("Plotting EMD variation barplots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_dict = {
            k: v for k, v in emd_dict.items() if k != baseline_mode
        }  # exclude baseline mode for self comparison
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
        sns.barplot(data=emd_df, ax=ax)  # type:ignore
        ax.set_ylabel("EMD")
        ax.set_title(DISTANCE_MEASURE_LABELS[distance_measure])
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([MODE_LABELS[m] for m in emd_df.columns], rotation=45)
        if figures_dir is not None:
            fig.savefig(
                figures_dir / f"{distance_measure}_emd_barplot{suffix}.{save_format}",
                dpi=300,
            )


def plot_emd_violinplots(
    distance_measures: list[str],
    all_pairwise_emd: dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]],
    baseline_mode: str = "SLC25A17",
    suffix: str = "",
    figures_dir: Path | None = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> None:
    """
    Plot violinplots of Earth Mover's Distance (EMD) values for different distance measures.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    all_pairwise_emd
        Dictionary containing pairwise EMD values for each distance measure and mode
    baseline_mode
        The baseline packing mode to use for comparison
    suffix
        Suffix to append to the figure filenames
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    save_format
        Format to save the figures in
    """
    log.info("Plotting EMD variation violinplots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_dict = {
            k: v for k, v in emd_dict.items() if k != baseline_mode
        }  # exclude baseline mode for self comparison
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
        sns.violinplot(
            data=emd_df,
            orient="h",
            legend=False,
            inner=None,
            cut=0,
            linewidth=1,
            linecolor="k",
            ax=ax,
        )
        ax.set_xlabel("EMD")
        ax.set_title(DISTANCE_MEASURE_LABELS[distance_measure])
        ax.set_yticklabels([MODE_LABELS[m] for m in emd_df.columns])
        if figures_dir is not None:
            fig.savefig(
                figures_dir
                / f"{distance_measure}_emd_violinplot{suffix}.{save_format}",
                dpi=300,
            )


def plot_emd_histograms(
    distance_measures: list[str],
    all_pairwise_emd: dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]],
    baseline_mode: str = "SLC25A17",
    emd_figures_dir: Path | None = None,
    suffix: str = "",
    save_format: str = "png",
) -> None:
    """
    Plot histograms of EMD variation.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    all_pairwise_emd
        Dictionary containing pairwise EMD values
    baseline_mode
        Baseline packing mode for comparison
    emd_figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    save_format
        Format to save the figures in
    """
    log.info("Plotting EMD variation histograms")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_df = pd.DataFrame.from_dict(emd_dict)
        ncol = len(emd_df.columns)
        fig, axs = plt.subplots(
            1, ncol, dpi=300, sharey=True, sharex=True, figsize=(ncol * 6, 6)
        )
        for ct, col in enumerate(emd_df.columns):
            axs[ct].hist(emd_df[col], bins=50, density=True)
            axs[ct].set_title(
                f"{MODE_LABELS[col]}\nn={np.count_nonzero(~np.isnan(emd_df[col]))}"
            )
        fig.supxlabel(f"{distance_measure} EMD")
        fig.supylabel("Density")
        plt.tight_layout()

        if emd_figures_dir is not None:
            fig.savefig(
                emd_figures_dir
                / f"{distance_measure}_emd_boxplot{suffix}.{save_format}",
                dpi=300,
            )


def plot_emd_kdeplots(
    distance_measures: list[str],
    all_pairwise_emd: dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]],
    baseline_mode: str = "SLC25A17",
    emd_figures_dir: Path | None = None,
    suffix: str = "",
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    save_format: str = "png",
) -> None:
    """
    Plot KDE plots of EMD variation.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot
    all_pairwise_emd
        Dictionary containing pairwise EMD values
    baseline_mode
        Baseline packing mode for comparison
    emd_figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    bandwidth
        Bandwidth method for KDE
    save_format
        Format to save the figures in
    """
    log.info("Plotting EMD variation kde plots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
        for col in emd_df.columns:
            sns.kdeplot(
                x=emd_df[col],
                ax=ax,
                label=MODE_LABELS[col],
                bw_method=bandwidth,  # type:ignore
                cut=0,
            )
        ax.set_xlabel(f"{distance_measure} EMD")
        ax.legend()
        fig.tight_layout()

        if emd_figures_dir is not None:
            fig.savefig(
                emd_figures_dir
                / f"{distance_measure}_emd_kdeplot{suffix}.{save_format}",
                dpi=300,
            )


def plot_ripley_k(
    mean_ripleyK: dict[str, np.ndarray],
    ci_ripleyK: dict[str, np.ndarray],
    r_values: np.ndarray,
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: str = "png",
) -> None:
    """
    Plot Ripley's K function results.

    Parameters
    ----------
    mean_ripleyK
        Dictionary containing mean Ripley K values for each mode
    ci_ripleyK
        Dictionary containing confidence intervals for each mode
    r_values
        Array of r values used in calculation
    figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    save_format
        Format to save the figures in
    """
    log.info("Plotting Ripley K")
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    for mode, mode_values in mean_ripleyK.items():
        mode_values = mode_values - np.pi * r_values**2
        err_values = ci_ripleyK[mode] - np.pi * r_values**2

        ax.plot(r_values, mode_values, label=MODE_LABELS.get(mode, mode))
        ax.fill_between(r_values, err_values[0], err_values[1], alpha=0.2)
    ax.set_xlabel("r")
    ax.set_ylabel("$K(r) - \\pi r^2$")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)

    if figures_dir is not None:
        fig.savefig(figures_dir / f"ripleyK{suffix}.{save_format}", dpi=300)

    plt.show()


def plot_occupancy_illustration(
    distance_dict: dict[str, dict[str, np.ndarray]],
    kde_dict: dict[str, dict[str, dict[str, Any]]],
    baseline_mode: str = "random",
    figures_dir: Path | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    normalization: str | None = None,
    method: str = "pdf",
    xlim: float | None = None,
    seed_index: int | None = None,
    save_format: str = "png",
) -> tuple[Any, Any, np.ndarray, np.ndarray, Any, list[Any]]:
    """
    Plot an illustration of occupancy analysis.

    Parameters
    ----------
    distance_dict
        Dictionary containing distance information
    kde_dict
        Dictionary containing KDE information
    baseline_mode
        Baseline packing mode to illustrate
    figures_dir
        Directory to save the figures
    suffix
        Suffix to add to the figure filename
    distance_measure
        The distance measure to plot
    normalization
        Normalization method applied
    method
        Method for ratio calculation ("pdf" or "cumulative")
    xlim
        X-axis limit for plots
    seed_index
        Index of seed to use for illustration
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing KDE objects, x-values, y-values, figure, and axes
    """
    log.info("Plotting occupancy illustration")

    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=(6, 6))
    mode_dict = distance_dict[baseline_mode]
    all_cellids = list(mode_dict.keys())
    if seed_index is not None:
        seed = all_cellids[seed_index]
    else:
        seed = all_cellids[0]
    distances = mode_dict[seed]
    distances = filter_invalid_distances(distances)

    kde_distance = kde_dict[seed][baseline_mode]["kde"]
    kde_available_space = kde_dict[seed]["available_distance"]["kde"]

    xvals = np.linspace(np.nanmin(distances), np.nanmax(distances), 100)

    kde_distance_values = kde_distance(xvals)
    kde_available_space_values = kde_available_space(xvals)

    yvals, kde_distance_values_normalized, kde_available_space_values_normalized = (
        get_pdf_ratio(xvals, kde_distance_values, kde_available_space_values, method)
    )

    # plot occupied distance values
    ax = axs[0]
    ax.plot(xvals, kde_distance_values_normalized, c="r")
    if xlim is None:
        xlim = ax.get_xlim()[1]
    ax.set_xlim([0, xlim])
    ax.set_ylabel("Probability Density")
    ax.set_title("Occupied Space")

    # plot available space values
    ax = axs[1]
    ax.plot(
        xvals, kde_available_space_values_normalized, label="available space", c="b"
    )
    ax.set_xlim([0, xlim])
    ax.set_ylabel("Probability Density")
    ax.set_title("Available Space")

    ylims = [axs[i].get_ylim() for i in range(2)]
    ylim = [min([y[0] for y in ylims]), max([y[1] for y in ylims])]

    for i in range(2):
        axs[i].set_ylim(ylim)

    # plot ratio
    ax = axs[2]
    ax.plot(xvals, yvals, label="Occupancy Ratio", c="g")
    ax.set_ylim([0, 2])
    ax.set_xlim([0, xlim])
    ax.axhline(1, color="k", linestyle="--")
    ax.set_ylabel("Ratio")
    title_str = "Occupancy Ratio"
    if method == "cumulative":
        title_str = "Cumulative Ratio"
    ax.set_title(title_str)

    for ax in axs:
        distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
        if normalization is not None:
            distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
        elif "scaled" not in distance_measure:
            distance_label = f"{distance_label} (\u03bcm)"
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(distance_label)

    plt.tight_layout()
    plt.show()

    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_{method}_occupancy_ratio_illustration{suffix}.{save_format}",
            dpi=300,
        )
    plt.show()

    return kde_distance, kde_available_space, xvals, yvals, fig, axs


def add_struct_radius_to_plot(
    ax: Any,
    structure_id: str,
    mesh_information_dict: dict[str, dict[str, Any]],
    normalization: str | None = None,
) -> Any:
    """
    Add the mean and std structure radius to the plot.

    Parameters
    ----------
    ax
        The axes object to add the structural diameter information to
    structure_id
        ID of the structure to analyze
    mesh_information_dict
        Dictionary containing mesh information
    normalization
        Normalization method to apply

    Returns
    -------
    :
        The modified axes object
    """
    avg_radius, std_radius = get_scaled_structure_radius(
        structure_id, mesh_information_dict, normalization=normalization
    )

    ax.axvspan(
        avg_radius - std_radius,
        avg_radius + std_radius,
        color="r",
        alpha=0.2,
        linewidth=0,
    )
    ax.axvline(avg_radius, color="r", linestyle="--")

    return ax


def plot_individual_occupancy_ratio(
    distance_dict: dict[str, dict[str, np.ndarray]],
    kde_dict: dict[str, dict[str, dict[str, Any]]],
    packing_modes: list[str],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    distance_measure: str = "nucleus",
    method: str = "pdf",
    xlim: float | None = None,
    ylim: float | None = None,
    sample_size: int | None = None,
    save_format: str = "png",
) -> tuple[list[Any], list[Any]]:
    """
    Plot the individual occupancy ratio based on the given parameters.

    Parameters
    ----------
    distance_dict
        A dictionary containing distance information
    kde_dict
        A dictionary containing KDE information
    packing_modes
        A list of packing modes
    figures_dir
        The directory to save the figures
    suffix
        A suffix to add to the figure filename
    normalization
        Normalization method applied
    distance_measure
        The distance measure to plot
    method
        The ratio to plot ("pdf" or "cumulative")
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    sample_size
        Number of samples to plot
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing lists of figure and axis objects
    """
    log.info("Plotting individual occupancy values")
    cmap = plt.get_cmap("tab10")
    figs = []
    axs = []

    for ct, mode in enumerate(packing_modes):
        fig, ax = plt.subplots(
            dpi=300,
            figsize=(9, 3),
        )
        log.info(f"Calculating occupancy for {mode}")
        mode_dict = distance_dict[mode]
        cellids_to_use = sample_cellids_from_distance_dict(mode_dict, sample_size)

        for cellid in tqdm(cellids_to_use):
            distances = mode_dict[cellid]
            distances = filter_invalid_distances(distances)
            if len(distances) == 0:
                log.warning(
                    f"No valid distances found for cellid {cellid} and mode {mode}"
                )
                continue
            kde_distance = kde_dict[cellid][mode]["kde"]
            kde_available_space = kde_dict[cellid]["available_distance"]["kde"]
            min_distance = np.nanmin(distances)
            max_distance = np.nanmax(distances)
            xvals = np.linspace(min_distance, max_distance, 100)

            kde_distance_values = kde_distance(xvals)
            kde_available_space_values = kde_available_space(xvals)

            yvals, _, _ = get_pdf_ratio(
                xvals, kde_distance_values, kde_available_space_values, method
            )

            ax.plot(xvals, yvals, c=cmap(ct), alpha=0.25)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        if xlim is not None:
            ax.set_xlim((0, xlim))
        if ylim is not None:
            ax.set_ylim((0, ylim))
        ax.axhline(1, color="k", linestyle="--")
        distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
        if normalization is not None:
            distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
        else:
            distance_label = f"{distance_label} (\u03bcm)"
        ax.set_xlabel(distance_label)
        ylabel = "Occupancy Ratio"
        if method == "cumulative":
            ylabel = "Cumulative Ratio"
        ax.set_ylabel(ylabel)
        ax.set_title(f"{MODE_LABELS.get(mode, mode)}")

        fig.tight_layout()
        if figures_dir is not None:
            fig.savefig(
                figures_dir
                / (
                    f"{distance_measure}_{mode}_individual_{method}_occupancy_ratio"
                    f"{suffix}.{save_format}"
                ),
                dpi=300,
            )

        plt.show()

        figs.append(fig)
        axs.append(ax)

    return figs, axs


def plot_mean_and_std_occupancy_ratio_kde(
    distance_dict: dict[str, dict[str, np.ndarray]],
    kde_dict: dict[str, dict[str, dict[str, Any]]],
    packing_modes: list[str],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    distance_measure: str = "nucleus",
    method: str = "ratio",
    xlim: float | None = None,
    ylim: float | None = None,
    sample_size: int | None = None,
    save_format: str = "png",
) -> tuple[list[Any], list[Any]]:
    """
    Plot the mean occupancy ratio along with the confidence intervals.

    Parameters
    ----------
    distance_dict
        A dictionary containing distance information
    kde_dict
        A dictionary containing KDE information
    packing_modes
        A list of packing modes
    figures_dir
        The directory to save the figures
    suffix
        A suffix to add to the figure filename
    normalization
        Normalization method applied
    distance_measure
        The distance measure to plot
    method
        The ratio to plot ("density" or "ratio")
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    sample_size
        The number of samples to use
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing lists of figure and axis objects
    """
    log.info("Plotting occupancy with confidence intervals")

    figs = []
    axs = []
    cmap = plt.get_cmap("tab10")

    for ct, mode in enumerate(packing_modes):
        color = cmap(ct)
        log.info(f"Calculating occupancy for {mode}")
        fig, ax = plt.subplots(dpi=300, figsize=(9, 3))
        mode_dict = distance_dict[mode]
        all_mode_distances = list(mode_dict.values())
        all_mode_distances = create_padded_numpy_array(all_mode_distances)
        all_mode_distances[all_mode_distances < 0] = np.nan
        max_distance = np.nanmax(all_mode_distances)
        min_distance = np.nanmin(all_mode_distances)
        all_xvals = np.linspace(min_distance, max_distance, 1000)
        mode_yvals = np.full((len(mode_dict), len(all_xvals)), np.nan)
        seeds_to_use = mode_dict.keys()
        if sample_size is not None:
            seeds_to_use = np.random.choice(
                list(mode_dict.keys()), sample_size, replace=False
            )
        for rt, seed in tqdm(enumerate(seeds_to_use), total=len(seeds_to_use)):
            distances = mode_dict[seed]

            if seed not in kde_dict:
                continue

            kde_distance = kde_dict[seed][mode]["kde"]
            kde_available_space = kde_dict[seed]["available_distance"]["kde"]

            xmax = np.nanmax(distances)
            xmin = np.nanmin(distances)
            xmax_index = np.argmax(all_xvals >= xmax)
            xmin_index = np.argmax(all_xvals >= xmin)
            if xmax_index == xmin_index or xmax_index == 0:
                continue
            xvals = all_xvals[xmin_index:xmax_index]

            kde_distance_values = kde_distance(xvals)
            kde_available_space_values = kde_available_space(xvals)

            yvals, _, _ = get_pdf_ratio(
                xvals, kde_distance_values, kde_available_space_values, method
            )
            mode_yvals[rt, xmin_index:xmax_index] = yvals
            ax.plot(xvals, yvals, c=color, alpha=0.05, lw=0.5, zorder=0)

        mean_yvals = np.nanmean(mode_yvals, axis=0)
        std_yvals = np.nanstd(mode_yvals, axis=0)
        ax.plot(
            all_xvals,
            mean_yvals,
            c=color,
            label=MODE_LABELS.get(mode, mode),
            lw=3,
            zorder=2,
        )
        ax.fill_between(
            all_xvals,
            mean_yvals - std_yvals,
            mean_yvals + std_yvals,
            color=color,
            linewidth=0,
            alpha=0.5,
            zorder=1,
        )

        ax.xaxis.set_major_locator(MaxNLocator(5))
        if xlim is not None:
            ax.set_xlim((0, xlim))
        if ylim is not None:
            ax.set_ylim((0, ylim))
        ax.axhline(1, color="k", linestyle="--")
        distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
        if normalization is not None:
            distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
        else:
            distance_label = f"{distance_label} (\u03bcm)"
        ax.set_xlabel(distance_label)
        ylabel = "Occupancy Ratio"
        if method == "cumulative":
            ylabel = "Cumulative Ratio"
        ax.set_ylabel(ylabel)
        ax.set_title(f"{MODE_LABELS.get(mode, mode)}")

        fig.tight_layout()
        fname = f"{distance_measure}_{mode}_individual_{method}_occupancy_ratio_withCI{suffix}"
        if figures_dir is not None:
            fig.savefig(
                figures_dir / f"{fname}.{save_format}",
                dpi=300,
            )

        plt.show()
        figs.append(fig)
        axs.append(ax)

    return figs, axs


def plot_binned_occupancy_ratio(
    distance_dict: dict[str, dict[str, np.ndarray]],
    packing_modes: list[str],
    mesh_information_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str],
    figures_dir: Path | None = None,
    normalization: str | None = None,
    suffix: str = "",
    num_bins: int = 64,
    bin_width: float | None = None,
    distance_measure: str = "nucleus",
    xlim: float | None = None,
    ylim: float | None = None,
    sample_size: int | None = None,
    save_format: str = "png",
) -> tuple[Any, Any]:
    """
    Calculate the binned occupancy ratio based on the provided distance dictionary.

    Parameters
    ----------
    distance_dict
        A dictionary containing distance information for various entities
    packing_modes
        A list of packing modes to consider for the analysis
    mesh_information_dict
        Dictionary containing mesh information for each seed
    channel_map
        Dictionary mapping packing modes to channels
    figures_dir
        Directory to save the results
    normalization
        Method to normalize the data
    suffix
        Suffix to append to the result filenames
    num_bins
        Number of bins to use for histogram
    bin_width
        Width of bins (overrides num_bins if provided)
    distance_measure
        The measure of distance to use
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    sample_size
        The number of samples to consider
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))

    for mode in packing_modes:
        log.info(f"Calculating binned occupancy ratio for: {mode}")

        mode_dict = distance_dict[mode]
        cellids_to_use = sample_cellids_from_distance_dict(mode_dict, sample_size)
        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})

        # get max and min distances for this mode
        combined_available_distance_dict = {}
        for cellid in cellids_to_use:
            available_distances = mode_mesh_dict[cellid][
                GRID_DISTANCE_LABELS[distance_measure]
            ].flatten()
            available_distances = filter_invalid_distances(available_distances)
            normalization_factor = get_normalization_factor(
                normalization=normalization,
                mesh_information_dict=mesh_information_dict,
                cellid=cellid,
                distance_measure=distance_measure,
                distances=available_distances,
            )
            available_distances /= normalization_factor
            combined_available_distance_dict[cellid] = available_distances

        combined_available_distances = np.concatenate(
            list(combined_available_distance_dict.values())
        )
        min_distance = np.nanmin(combined_available_distances)
        max_distance = np.nanmax(combined_available_distances)
        if bin_width is not None:
            num_bins = int((max_distance - min_distance) / bin_width)
        bins = np.linspace(min_distance, max_distance, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        available_space_counts = {}
        occupied_space_counts = {}
        for cellid in tqdm(cellids_to_use):
            distances = mode_dict[cellid]
            distances = filter_invalid_distances(distances)
            if cellid not in mode_mesh_dict:
                continue
            available_distances = combined_available_distance_dict[cellid]
            available_space_counts[cellid] = np.histogram(
                available_distances, bins=bins, density=False
            )[0]
            occupied_space_counts[cellid] = np.histogram(
                distances, bins=bins, density=False
            )[0]
        occupied_space_counts = np.vstack(list(occupied_space_counts.values()))
        occupied_space_counts = occupied_space_counts + 1e-16

        available_space_counts = np.vstack(list(available_space_counts.values()))
        available_space_counts = available_space_counts + 1e-16

        # return occupied_space_counts, available_space_counts, bin_centers

        normalized_occupied_space_counts = (
            occupied_space_counts / np.sum(occupied_space_counts, axis=1)[:, np.newaxis]
        )
        normalized_available_space_counts = (
            available_space_counts
            / np.sum(available_space_counts, axis=1)[:, np.newaxis]
        )
        occupancy_ratio = (
            normalized_occupied_space_counts / normalized_available_space_counts
        )
        mean_occupancy_ratio = np.nanmean(occupancy_ratio, axis=0)
        std_occupancy_ratio = np.nanstd(occupancy_ratio, axis=0)

        ax.plot(
            bin_centers,
            mean_occupancy_ratio,
            label=MODE_LABELS.get(mode, mode),
            zorder=2,
            lw=3,
        )
        ax.fill_between(
            bin_centers,
            mean_occupancy_ratio - std_occupancy_ratio,
            mean_occupancy_ratio + std_occupancy_ratio,
            alpha=0.2,
            lw=0,
            label="_nolegend_",
            zorder=0,
        )

    ax.axhline(1, color="k", linestyle="--")

    if xlim is not None:
        ax.set_xlim((0, xlim))
    else:
        cur_xlim = ax.get_xlim()
        ax.set_xlim((0, cur_xlim[1]))

    if ylim is not None:
        ax.set_ylim((0, ylim))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    xlabel = DISTANCE_MEASURE_LABELS[distance_measure]
    if normalization is not None:
        xlabel = f"{xlabel} / {NORMALIZATION_LABELS[normalization]}"
    else:
        xlabel = f"{xlabel} (\u03bcm)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Occupancy Ratio")
    ax.legend()
    plt.tight_layout()
    plt.show()

    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_binned_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )

    return fig, ax


def plot_combined_occupancy_ratio(
    combined_kde_dict: dict[str, dict[str, Any]],
    packing_modes: list[str],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    aspect: float | None = None,
    save_format: str = "png",
    save_intermediates: bool = False,
    distance_measure: str = "nucleus",
    num_points: int = 100,
    method: str = "ratio",
    xlim: float | None = None,
    ylim: float | None = None,
) -> tuple[Any, Any]:
    """
    Plot combined occupancy ratio for all packing modes.

    Parameters
    ----------
    combined_kde_dict
        Dictionary containing combined KDE information
    packing_modes
        List of packing modes to plot
    figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    normalization
        Normalization method applied
    aspect
        Aspect ratio for the plot
    save_format
        Format to save the figures in
    save_intermediates
        Whether to save intermediate plots
    distance_measure
        Distance measure being plotted
    num_points
        Number of points for KDE evaluation
    method
        Method for ratio calculation
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    log.info("Plotting combined occupancy ratio")

    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)

    for ct, mode in enumerate(packing_modes):
        log.info(f"Calculating combined occupancy ratio for: {mode}")

        kde_mode_dict = combined_kde_dict[mode]
        mode_distances = kde_mode_dict["mode_distances"]
        mode_distances = filter_invalid_distances(mode_distances)

        kde_distance = kde_mode_dict["kde_distance"]
        kde_available_space = kde_mode_dict["kde_available_space"]

        min_distance = np.nanmin(mode_distances)
        max_distance = np.nanmax(mode_distances)
        xvals = np.linspace(min_distance, max_distance, num_points)

        kde_distance_values = kde_distance(xvals)
        kde_available_space_values = kde_available_space(xvals)

        yvals, _, _ = get_pdf_ratio(
            xvals, kde_distance_values, kde_available_space_values, method
        )

        ax.plot(xvals, yvals, label=MODE_LABELS.get(mode, mode))

        if xlim is not None:
            ax.set_xlim((0, xlim))
        else:
            cur_xlim = ax.get_xlim()
            ax.set_xlim((0, cur_xlim[1]))

        if ylim is not None:
            ax.set_ylim((0, ylim))

        xlabel = DISTANCE_MEASURE_LABELS[distance_measure]
        if normalization:
            xlabel = f"{xlabel} / {NORMALIZATION_LABELS[normalization]}"
        else:
            xlabel = f"{xlabel} (\u03bcm)"
        ax.set_xlabel(xlabel)
        ylabel = "Occupancy Ratio"
        if method == "cumulative":
            ylabel = "Cumulative Ratio"
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        plt.tight_layout()

        if save_intermediates and figures_dir is not None:
            used_suffix = f"{suffix}_{ct}"
            fig.savefig(
                figures_dir
                / (
                    f"{distance_measure}_combined_{method}_occupancy_ratio"
                    f"{used_suffix}.{save_format}"
                ),
                dpi=300,
            )
    if aspect is not None:
        ax.set_aspect(aspect)
        suffix = f"_aspect{suffix}"
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.tight_layout()
    plt.show()
    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_combined_{method}_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )
    return fig, ax


def plot_occupancy_emd_kdeplot(
    emd_occupancy_dict: dict[str, dict[str, float]],
    packing_modes: list[str],
    figures_dir: Path | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
) -> None:
    """
    Plot EMD occupancy distributions using kernel density estimation (KDE).

    Parameters
    ----------
    emd_occupancy_dict
        Dictionary containing EMD occupancy values for each packing mode and seed
    packing_modes
        List of packing modes to plot
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    suffix
        Suffix to append to the figure filenames
    distance_measure
        Distance measure used for the EMD calculation
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value
    save_format
        Format to save the figures in
    """
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    for ct, mode in enumerate(packing_modes):
        emd_values = list(emd_occupancy_dict[mode].values())
        mean_emd = np.mean(emd_values).item()
        sns.kdeplot(
            emd_values,
            ax=ax,
            label=MODE_LABELS.get(mode, mode),
            c=cmap(ct),
            bw_method=bandwidth,
            cut=0,
        )
        ax.axvline(mean_emd, color=cmap(ct), linestyle="--", label="_nolegend_")
    ax.set_xlabel("EMD")
    ax.legend()
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"{distance_measure}_occupancy_emd{suffix}.{save_format}",
            dpi=300,
        )
    plt.show()


def plot_occupancy_emd_boxplot(
    emd_occupancy_dict: dict[str, dict[str, float]],
    figures_dir: Path | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    save_format: str = "png",
) -> None:
    """
    Create and display a boxplot of Earth Mover's Distance (EMD) values for occupancy analysis.

    This function generates a boxplot visualization showing the distribution of EMD values
    across different modes/conditions for occupancy analysis. The plot can be saved to disk
    if a figures directory is provided.

    Parameters
    ----------
    emd_occupancy_dict
        Dictionary containing EMD values for each mode/condition, where keys are mode names
        and values are lists of EMD measurements
    figures_dir
        Directory path where the figure should be saved. If None, the figure is not saved
    suffix
        Additional suffix to append to the saved filename
    distance_measure
        Type of distance measurement used, incorporated into the filename when saving
    save_format
        File format for saving the figure

    Returns
    -------
    """
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    emd_df = pd.DataFrame(emd_occupancy_dict)
    sns.boxplot(data=emd_df, ax=ax)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    xticklabels = [MODE_LABELS[mode.get_text()] for mode in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_ylabel("EMD")
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_occupancy_emd_boxplot{suffix}.{save_format}",
            dpi=300,
        )
    plt.show()


def plot_occupancy_ks_test(
    ks_occupancy_dict: dict[str, dict[str, float]],
    figures_dir: Path | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    significance_level: float = 0.05,
    save_format: str = "png",
) -> None:
    """
    Plot results of KS test for occupancy analysis.

    Parameters
    ----------
    ks_occupancy_dict
        Dictionary containing KS test results
    figures_dir
        Directory to save the figures
    suffix
        Suffix to append to the figure filenames
    distance_measure
        Distance measure analyzed
    significance_level
        Significance level for the test
    save_format
        Format to save the figures in
    """
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    ks_df = pd.DataFrame(ks_occupancy_dict)
    ks_df_test = ks_df < significance_level
    ax = sns.barplot(data=ks_df_test, ax=ax)
    xticklabels = [MODE_LABELS[mode.get_text()] for mode in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_ylabel(
        f"Fraction with non space-filling occupancy\n(p < {significance_level})"
    )
    plt.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"{distance_measure}_occupancy_ks_test{suffix}.{save_format}",
            dpi=300,
        )
    plt.show()
