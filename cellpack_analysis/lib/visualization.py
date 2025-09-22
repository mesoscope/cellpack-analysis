import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from statannotations.Annotator import Annotator
from tqdm import tqdm

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import (
    filter_invalid_distances,
    get_normalization_factor,
    get_scaled_structure_radius,
)
from cellpack_analysis.lib.label_tables import (
    COLOR_PALETTE,
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
    MODE_LABELS,
    NORMALIZATION_LABELS,
)
from cellpack_analysis.lib.stats import create_padded_numpy_array, get_pdf_ratio, normalize_pdf

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
        cell_id_dict["cell_diameter"] * PIXEL_SIZE_IN_UM
        for _, cell_id_dict in mesh_information_dict.items()
    ]
    nuc_diameters = [
        cell_id_dict["nuc_diameter"] * PIXEL_SIZE_IN_UM
        for _, cell_id_dict in mesh_information_dict.items()
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
) -> tuple[Figure, dict[str, dict[str, Axes]]]:
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
    plt.rcParams.update({"font.size": 5})

    all_ax_dict = {}
    num_rows = len(packing_modes)
    num_cols = len(distance_measures)
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 1.2, num_rows * 0.5),
        dpi=300,
        squeeze=False,
        sharex="col",
        sharey="col",
    )
    for col, distance_measure in enumerate(distance_measures):
        distance_dict = all_distance_dict[distance_measure]
        if distance_measure not in all_ax_dict:
            all_ax_dict[distance_measure] = {}

        if normalization is not None:
            unit = ""
            distance_label = (
                f"{DISTANCE_MEASURE_LABELS[distance_measure]}"
                f" / {NORMALIZATION_LABELS[normalization]}"
            )
        elif "scaled" in distance_measure:
            unit = ""
            distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
        else:
            unit = "\u03bcm"
            distance_label = f"{DISTANCE_MEASURE_LABELS[distance_measure]} ({unit})"

        for row, mode in enumerate(packing_modes):
            log.info(f"Plotting distance distribution for: {distance_measure}, Mode: {mode}")

            ax = axs[row, col]
            mode_dict = distance_dict[mode]
            mode_color = COLOR_PALETTE.get(mode, "gray")
            mode_label = MODE_LABELS.get(mode, mode)

            # kde plots of individual distance distributions
            for _, distances in tqdm(mode_dict.items(), total=len(mode_dict)):
                sns.kdeplot(
                    distances,
                    ax=ax,
                    color=mode_color,
                    linewidth=0.15,
                    alpha=0.05,
                    bw_method=bandwidth,
                    cut=0,
                )
                # break

            # kde plot of combined distance distribution
            combined_mode_distances = np.concatenate(list(mode_dict.values()))
            combined_mode_distances = filter_invalid_distances(combined_mode_distances)
            if overlay:
                sns.kdeplot(
                    combined_mode_distances,
                    ax=ax,
                    color=mode_color,
                    linewidth=0.7,
                    label=mode_label,
                    bw_method=bandwidth,
                    cut=0,
                )

            # set axis limits
            if distance_limits is not None:
                ax.set_xlim(distance_limits.get(distance_measure, (0, 1)))
            else:
                min_xlim = np.nanmin(combined_mode_distances)
                max_xlim = np.nanmax(combined_mode_distances)
                min_xlim = min_xlim - 0.1 * (max_xlim - min_xlim)
                max_xlim = max_xlim + 0.1 * (max_xlim - min_xlim)
                ax.set_xlim(min_xlim, max_xlim)

            # remove top and right spines
            sns.despine(fig=fig)

            # plot mean distance and add annotation
            mean_distance = np.nanmean(combined_mode_distances).item()
            std_distance = np.nanstd(combined_mode_distances).item()
            ax.axvline(mean_distance, color="gray", linestyle="--", linewidth=0.3)
            ax.axvspan(
                mean_distance - std_distance,
                mean_distance + std_distance,
                edgecolor="none",
                facecolor="gray",
                alpha=0.1,
            )

            # set labels
            if col == 0:
                ax.set_ylabel(f"{mode_label}\nPDF")
            else:
                ax.set_ylabel("")
            if row == num_rows - 1:
                ax.set_xlabel(distance_label)

            all_ax_dict[distance_measure][row] = ax

            # set axes to use integer ticks
            ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(2, integer=True))

            # remove top and right spines
            # ax.spines["top"].set_visible(False)
            # ax.spines["right"].set_visible(False)

    if figures_dir is not None:
        fig.tight_layout()
        plt.show()
        fig.savefig(
            figures_dir / f"distance_distribution_kdeplot{suffix}.{save_format}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)
        # break

    return fig, all_ax_dict


def plot_ks_test_barplots(
    df_ks_bootstrap: pd.DataFrame,
    distance_measures: list[str],
    suffix: str = "",
    figures_dir: Path | None = None,
    baseline_mode: str | None = None,
    save_format: str = "png",
    annotate_significance: bool = False,
) -> tuple[Figure, np.ndarray]:
    """
    Plot KS test results as bar plots.

    Parameters
    ----------
    df_ks_bootstrap
        DataFrame containing KS test results
    distance_measures
        List of distance measures to plot
    significance_level
        Significance level for the KS test
    suffix
        Suffix to append to the figure filenames
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing lists of figure and axis objects for each distance measure
    """
    log.info("Plotting KS test barplots")
    plt.rcParams.update({"font.size": 6})

    num_cols = len(distance_measures)
    fig, axs = plt.subplots(
        1,
        num_cols,
        figsize=(1.25 * num_cols, 2.5),
        dpi=300,
        squeeze=False,
    )

    for col, distance_measure in enumerate(distance_measures):
        df_distance_measure = df_ks_bootstrap.query(f"distance_measure == '{distance_measure}'")

        ax = axs[0, col]

        plot_params = {
            "data": df_distance_measure,
            "x": "similar_fraction",
            "y": "packing_mode",
            "hue": "packing_mode",
            "legend": False,
            "orient": "h",
            "palette": COLOR_PALETTE,
        }
        sns.barplot(ax=ax, **plot_params)
        sns.despine(ax=ax)
        ax.set_xlim((0, 1))
        ax.set_xlabel("Similar fraction")
        ax.set_ylabel("")
        if col == 0:
            ax.set_yticks(
                ax.get_yticks(),
                labels=[
                    MODE_LABELS.get(label.get_text(), label.get_text())
                    for label in ax.get_yticklabels()
                ],
            )
        else:
            ax.set_yticks([])

        if annotate_significance:
            packing_modes = df_distance_measure["packing_mode"].unique()
            pairs = list(combinations(packing_modes, 2))
            if baseline_mode is not None:
                pairs = [(mode, baseline_mode) for mode in packing_modes if mode != baseline_mode]
            annotator = Annotator(ax=ax, pairs=pairs, plot="barplot", **plot_params)
            annotator.configure(
                test="Mann-Whitney",
                verbose=False,
                comparisons_correction="Bonferroni",
            ).apply_and_annotate()

    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"ks_test_vs_baseline{suffix}.{save_format}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
    return fig, axs


def plot_emd_comparisons(
    df_emd: pd.DataFrame,
    distance_measures: list[str],
    comparison_type: Literal["intra_mode", "baseline"] = "intra_mode",
    baseline_mode: str = "SLC25A17",
    suffix: str = "",
    figures_dir: Path | None = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
    annotate_significance: bool = False,
) -> tuple[Figure, np.ndarray, Figure, np.ndarray]:
    """
    Plot violinplots and barplots of Earth Mover's Distance (EMD) values.

    Parameters
    ----------
    df_emd
        DataFrame containing EMD values with columns 'distance_measure', 'packing_mode', and 'emd'
    distance_measures
        List of distance measures to plot
    comparison_type
        Type of comparison to plot:
        - "intra_mode": Compare EMD values within each packing mode
          (packing_mode_1 == packing_mode_2)
        - "baseline": Compare EMD values against a baseline mode
          (packing_mode_1 == baseline_mode)
    baseline_mode
        The baseline packing mode to use for comparison (used for both comparison types)
    suffix
        Suffix to append to the figure filenames
    figures_dir
        Directory to save the figures. If None, figures will not be saved
    save_format
        Format to save the figures in
    annotate_significance
        Whether to add statistical significance annotations to the plots

    Returns
    -------
    :
        Tuple containing (bar_figure, bar_axes, violin_figure, violin_axes)
    """
    if comparison_type == "intra_mode":
        log.info("Plotting within rule EMD plots")
        file_prefix = "intra_mode_emd"
    elif comparison_type == "baseline":
        log.info("Plotting baseline mode EMD plots")
        file_prefix = "baseline_mode_emd"
    else:
        raise ValueError(
            f"Invalid comparison_type: {comparison_type}. Must be 'intra_mode' or 'baseline'"
        )

    plt.rcParams.update({"font.size": 6})

    num_cols = len(distance_measures)
    fig_bar, axs_bar = plt.subplots(
        1,
        num_cols,
        figsize=(0.85 * num_cols, 2.5),
        dpi=300,
        squeeze=False,
    )
    fig_violin, axs_violin = plt.subplots(
        1,
        num_cols,
        figsize=(num_cols * 0.85, 2.5),
        dpi=300,
        squeeze=False,
    )

    for col, distance_measure in enumerate(distance_measures):
        # Build query and plot parameters based on comparison type
        if comparison_type == "intra_mode":
            query_str = (
                f"distance_measure == '{distance_measure}' and " "packing_mode_1 == packing_mode_2"
            )
            y_column = "packing_mode_1"
        else:  # baseline comparison
            query_str = (
                f"distance_measure == '{distance_measure}' and "
                f"packing_mode_1 == '{baseline_mode}' and packing_mode_2 != '{baseline_mode}'"
            )
            y_column = "packing_mode_2"

        df_plot = df_emd.query(query_str)

        ax_bar = axs_bar[0, col]
        ax_violin = axs_violin[0, col]

        plot_params = {
            "data": df_plot,
            "x": "emd",
            "y": y_column,
            "hue": y_column,
            "legend": False,
            "orient": "h",
            "palette": COLOR_PALETTE,
            "linewidth": 0.5,
        }
        sns.barplot(ax=ax_bar, **plot_params)
        sns.violinplot(ax=ax_violin, **plot_params)

        for ax, plot_type in zip([ax_bar, ax_violin], ["barplot", "violinplot"]):
            sns.despine(ax=ax)
            ax.set_xlabel("EMD")
            ax.set_ylabel("")
            ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            if col == 0:
                ax.set_yticks(
                    ax.get_yticks(),
                    labels=[
                        MODE_LABELS.get(label.get_text(), label.get_text())
                        for label in ax.get_yticklabels()
                    ],
                )
            else:
                ax.set_yticks([])

            if annotate_significance:
                if comparison_type == "intra_mode":
                    packing_modes = df_plot[y_column].unique()
                    pairs = list(combinations(packing_modes, 2))
                    pairs = [pair for pair in pairs if baseline_mode in pair]
                else:  # baseline comparison
                    packing_modes = [
                        mode for mode in df_plot[y_column].unique() if mode != baseline_mode
                    ]
                    pairs = list(combinations(packing_modes, 2))

                if pairs:  # Only annotate if there are pairs to compare
                    annotator = Annotator(ax=ax, pairs=pairs, plot=plot_type, **plot_params)
                    annotator.configure(
                        test="Mann-Whitney",
                        verbose=False,
                        comparisons_correction="Bonferroni",
                        loc="outside",
                    ).apply_and_annotate()

    if figures_dir is not None:
        final_suffix = suffix
        if annotate_significance:
            final_suffix += "_annotated"

        for fig, label in zip([fig_bar, fig_violin], ["barplot", "violinplot"]):
            fig.tight_layout()
            fig.savefig(
                figures_dir / f"{file_prefix}_{label}{final_suffix}.{save_format}",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()
            plt.close(fig)

    return fig_bar, axs_bar, fig_violin, axs_violin


def plot_occupancy_illustration(
    kde_dict: dict[str, dict[str, gaussian_kde]],
    baseline_mode: str = "random",
    figures_dir: Path | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    normalization: str | None = None,
    method: Literal["pdf", "cumulative"] = "pdf",
    xlim: float | None = None,
    num_points: int = 250,
    seed_index: int | None = None,
    save_format: str = "png",
    bandwidth: Literal["scott", "silverman"] | float | None = None,
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
    plt.rcParams.update({"font.size": 6})
    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=(2.5, 2.5), sharex=True)

    all_cell_ids = [cell_id for cell_id in kde_dict.keys() if baseline_mode in kde_dict[cell_id]]
    if seed_index is not None:
        seed = all_cell_ids[seed_index]
    else:
        seed = all_cell_ids[0]
    log.info(f"Using seed {seed} for occupancy illustration")

    occupied_kde = kde_dict[seed][baseline_mode]
    available_kde = kde_dict[seed]["available_distance"]
    xvals = np.linspace(0, occupied_kde.dataset.max(), num_points)
    if bandwidth is not None:
        occupied_kde.set_bandwidth(bandwidth)
        available_kde.set_bandwidth(bandwidth)

    pdf_occupied = normalize_pdf(xvals, occupied_kde.evaluate(xvals))
    pdf_available = normalize_pdf(xvals, available_kde.evaluate(xvals))

    yvals, distance_kde_values, available_space_kde_values = get_pdf_ratio(
        xvals, pdf_occupied, pdf_available, method
    )

    # plot occupied distance values
    ax = axs[0]
    sns.lineplot(x=xvals, y=distance_kde_values, ax=ax, color="r")
    if xlim is None:
        xlim = ax.get_xlim()[1]
    ax.set_xlim([0, xlim])
    ax.set_ylabel("Occupied \nPDF")
    # ax.set_title("Occupied Space")

    # plot available space values
    ax = axs[1]
    sns.lineplot(x=xvals, y=available_space_kde_values, ax=ax, color="b")
    ax.set_xlim([0, xlim])
    ax.set_ylabel("Available \nPDF")
    # ax.set_title("Available Space")

    ylims = [axs[i].get_ylim() for i in range(2)]
    ylim = [0, max([y[1] for y in ylims])]

    for i in range(2):
        axs[i].set_ylim(ylim)

    # plot ratio
    ax = axs[2]
    sns.lineplot(x=xvals, y=yvals, ax=ax, color="g")
    ax.set_ylim([0, 2])
    ax.set_xlim([0, xlim])
    ax.axhline(1, color="gray", linestyle="--")
    ax.set_ylabel("Occupancy ratio")

    for ax in axs:
        distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
        if normalization is not None:
            distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
        elif "scaled" not in distance_measure:
            distance_label = f"{distance_label} (\u03bcm)"
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.set_xlabel(distance_label)

    sns.despine(fig=fig)
    plt.tight_layout()
    plt.show()

    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_{method}_occupancy_ratio_illustration{suffix}.{save_format}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    return pdf_occupied, pdf_available, xvals, yvals, fig, axs


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


def plot_occupancy_ratio(
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    channel_map: dict[str, str],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    distance_measure: str = "nucleus",
    xlim: float | None = None,
    ylim: float | None = None,
    save_format: str = "png",
) -> tuple[Figure, Axes]:
    """
    Plot the occupancy ratio based on the given parameters.

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

    plt.rcParams.update({"font.size": 8})

    fig, ax = plt.subplots(dpi=300, figsize=(2.5, 2.5))

    for mode in channel_map.keys():

        for _cell_id, cell_id_dict in tqdm(
            occupancy_dict[mode]["individual"].items(), desc=f"Plotting {mode} occupancy"
        ):

            xvals = cell_id_dict["xvals"]
            occupancy = cell_id_dict["occupancy"]

            sns.lineplot(
                x=xvals,
                y=occupancy,
                ax=ax,
                color=COLOR_PALETTE.get(mode, "gray"),
                alpha=0.1,
                linewidth=0.1,
                label="_nolegend_",
                zorder=1,
            )

        # overlay occupancy for combined data
        combined_xvals = occupancy_dict[mode]["combined"]["xvals"]
        combined_occupancy = occupancy_dict[mode]["combined"]["occupancy"]

        sns.lineplot(
            x=combined_xvals,
            y=combined_occupancy,
            ax=ax,
            color=COLOR_PALETTE.get(mode, "gray"),
            alpha=1.0,
            linewidth=1.5,
            label=MODE_LABELS.get(mode, mode),
            zorder=2,
        )

    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    if xlim is not None:
        ax.set_xlim((0, xlim))
    if ylim is not None:
        ax.set_ylim((0, ylim))
    ax.axhline(1, color="gray", linestyle="--", zorder=0)
    distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
    if normalization is not None:
        distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
    else:
        distance_label = f"{distance_label} (\u03bcm)"
    ax.set_xlabel(distance_label)

    ax.set_ylabel("Occupancy Ratio")
    sns.despine(fig=fig)
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"{distance_measure}_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )

    plt.show()

    return fig, ax


def plot_mean_and_std_occupancy_ratio_kde(
    distance_dict: dict[str, dict[str, np.ndarray]],
    kde_dict: dict[str, dict[str, dict[str, Any]]],
    packing_modes: list[str],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    distance_measure: str = "nucleus",
    method: Literal["pdf", "cumulative"] = "pdf",
    xlim: float | None = None,
    ylim: float | None = None,
    sample_size: int | None = None,
    save_format: str = "png",
    num_points: int = 250,
    bandwidth: Literal["scott", "silverman"] | float | None = None,
) -> tuple[Figure, Axes]:
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

    lines = []
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    for ct, mode in enumerate(packing_modes):

        color = COLOR_PALETTE.get(mode, "gray")

        log.info(f"Calculating occupancy for {mode}")
        mode_dict = distance_dict[mode]

        all_mode_distances = list(mode_dict.values())
        all_mode_distances = create_padded_numpy_array(all_mode_distances)
        all_mode_distances[all_mode_distances <= 0] = np.nan

        min_distance = np.nanmin(all_mode_distances)
        max_distance = np.nanmax(all_mode_distances)

        all_xvals = np.linspace(min_distance, max_distance, num_points)
        mode_yvals = np.full((len(mode_dict), num_points), np.nan)
        seeds_to_use = mode_dict.keys()

        if sample_size is not None:
            seeds_to_use = np.random.choice(list(mode_dict.keys()), sample_size, replace=False)

        for rt, seed in tqdm(enumerate(seeds_to_use), total=len(seeds_to_use)):
            distances = mode_dict[seed]

            if seed not in kde_dict:
                continue

            distance_kde = kde_dict[seed][mode]["kde"]
            available_space_kde = kde_dict[seed]["available_distance"]["kde"]
            if bandwidth is not None:
                distance_kde.set_bandwidth(bandwidth)
                available_space_kde.set_bandwidth(bandwidth)

            xmax = np.nanmax(distances)
            xmin = np.nanmin(distances)
            xmax_index = np.argmax(all_xvals >= xmax)
            xmin_index = np.argmax(all_xvals >= xmin)
            if xmax_index == xmin_index or xmax_index == 0:
                continue
            xvals = all_xvals[xmin_index:xmax_index]

            distance_kde_values = distance_kde(xvals)
            available_space_kde_values = available_space_kde(xvals)

            yvals, _, _ = get_pdf_ratio(
                xvals, distance_kde_values, available_space_kde_values, method
            )
            mode_yvals[rt, xmin_index:xmax_index] = yvals
            ax.plot(xvals, yvals, c=color, alpha=0.05, lw=0.5, zorder=0)

        mean_yvals = np.nanmean(mode_yvals, axis=0)
        std_yvals = np.nanstd(mode_yvals, axis=0)
        line = ax.plot(
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
        lines.append(line[0])

    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    if xlim is not None:
        ax.set_xlim((0, xlim))
    if ylim is not None:
        ax.set_ylim((0, ylim))
    ax.axhline(1, color="gray", linestyle="--")
    distance_label = DISTANCE_MEASURE_LABELS[distance_measure]
    if normalization is not None:
        distance_label = f"{distance_label} / {NORMALIZATION_LABELS[normalization]}"
    else:
        distance_label = f"{distance_label} (\u03bcm)"
    ax.legend(handles=lines)
    ax.set_xlabel(distance_label)
    ylabel = "Occupancy Ratio"
    if method == "cumulative":
        ylabel = "Cumulative Ratio"
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    if figures_dir is not None:
        fname = f"{distance_measure}_individual_{method}_occupancy_ratio_withCI{suffix}"
        fig.savefig(
            figures_dir / f"{fname}.{save_format}",
            dpi=300,
        )

    plt.show()

    return fig, ax


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
        cell_ids_to_use = sample_cell_ids_from_distance_dict(mode_dict, sample_size)
        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})

        # get max and min distances for this mode
        combined_available_distance_dict = {}
        for cell_id in cell_ids_to_use:
            available_distances = mode_mesh_dict[cell_id][
                GRID_DISTANCE_LABELS[distance_measure]
            ].flatten()
            available_distances = filter_invalid_distances(available_distances)
            normalization_factor = get_normalization_factor(
                normalization=normalization,
                mesh_information_dict=mesh_information_dict,
                cell_id=cell_id,
                distance_measure=distance_measure,
                distances=available_distances,
            )
            available_distances /= normalization_factor
            combined_available_distance_dict[cell_id] = available_distances

        combined_available_distances = np.concatenate(
            list(combined_available_distance_dict.values())
        )
        max_distance = np.nanmax(combined_available_distances)
        if bin_width is not None:
            num_bins = int(max_distance / bin_width)
        bins = np.linspace(0, max_distance, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        available_space_counts = {}
        occupied_space_counts = {}
        for cell_id in tqdm(cell_ids_to_use):
            distances = mode_dict[cell_id]
            distances = filter_invalid_distances(distances)
            if cell_id not in mode_mesh_dict:
                continue
            available_distances = combined_available_distance_dict[cell_id]
            available_space_counts[cell_id] = np.histogram(
                available_distances, bins=bins, density=False
            )[0]
            occupied_space_counts[cell_id] = np.histogram(distances, bins=bins, density=False)[0]
        occupied_space_counts = np.vstack(list(occupied_space_counts.values()))
        occupied_space_counts = occupied_space_counts + 1e-16

        available_space_counts = np.vstack(list(available_space_counts.values()))
        available_space_counts = available_space_counts + 1e-16

        normalized_occupied_space_counts = (
            occupied_space_counts / np.sum(occupied_space_counts, axis=1)[:, np.newaxis]
        )
        normalized_available_space_counts = (
            available_space_counts / np.sum(available_space_counts, axis=1)[:, np.newaxis]
        )
        occupancy_ratio = normalized_occupied_space_counts / normalized_available_space_counts
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

    ax.axhline(1, color="gray", linestyle="--")

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
    fig.tight_layout()
    plt.show()

    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"{distance_measure}_binned_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )

    return fig, ax
