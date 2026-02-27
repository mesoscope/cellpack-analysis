import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from statannotations.Annotator import Annotator
from tqdm import tqdm

from cellpack_analysis.lib import label_tables
from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import filter_invalid_distances, get_scaled_structure_radius
from cellpack_analysis.lib.occupancy import get_cell_id_map_from_distance_kde_dict
from cellpack_analysis.lib.stats import create_padded_numpy_array, get_pdf_ratio, normalize_pdf

logger = logging.getLogger(__name__)


def plot_cell_diameter_distribution(
    mesh_information_dict: dict[Any, dict[str, float]],
) -> tuple[Figure, Axes]:
    """
    Plot distribution of cell and nucleus diameters as histograms.

    Parameters
    ----------
    mesh_information_dict
        Dictionary containing mesh information with cell_diameter and nuc_diameter keys
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

    return fig, ax


def plot_distance_distributions_kde(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    distance_limits: dict[str, tuple[float, float]] | None = None,
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
    minimum_distance: float | None = 0,
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
        {distance_measure: {mode: {cell_id: {seed:distances}}}}
    figures_dir
        Directory to save the figures, if None figures will not be saved
    suffix
        Suffix to append to figure filenames
    normalization
        Normalization method to apply to distance measures
    overlay
        If True, overlay the pooled KDE. Default is True
    distance_limits
        Dictionary containing limits for each distance measure
    bandwidth
        Bandwidth method for KDE
    save_format
        Format to save figures in
    minimum_distance
        Minimum distance threshold for filtering

    Returns
    -------
    :
        Tuple containing figure and dictionary mapping distance measures to axes
    """
    logger.info("Starting distance distribution kde plot")
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
                f"{label_tables.DISTANCE_MEASURE_LABELS[distance_measure]}"
                f" / {label_tables.NORMALIZATION_LABELS[normalization]}"
            )
        elif "scaled" in distance_measure:
            unit = ""
            distance_label = label_tables.DISTANCE_MEASURE_LABELS[distance_measure]
        else:
            unit = "\u03bcm"
            distance_label = f"{label_tables.DISTANCE_MEASURE_LABELS[distance_measure]} ({unit})"

        for row, mode in enumerate(packing_modes):
            logger.info(f"Plotting distance distribution for: {distance_measure}, Mode: {mode}")

            ax = axs[row, col]
            mode_dict = distance_dict[mode]
            mode_color = label_tables.COLOR_PALETTE.get(mode, "gray")
            mode_label = label_tables.MODE_LABELS.get(mode, mode)

            # kde plots of individual distance distributions
            for _, seed_distance_dict in tqdm(mode_dict.items(), total=len(mode_dict)):
                for _, distances in seed_distance_dict.items():
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
            combined_mode_distances = np.concatenate(
                [
                    distances
                    for seed_distance_dict in mode_dict.values()
                    for distances in seed_distance_dict.values()
                ]
            )
            combined_mode_distances = filter_invalid_distances(
                combined_mode_distances, minimum_distance=minimum_distance
            )

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

    return fig, all_ax_dict


def plot_ks_test_results(
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
    suffix
        Suffix to append to figure filenames
    figures_dir
        Directory to save figures, if None figures will not be saved
    baseline_mode
        Baseline packing mode for significance testing
    save_format
        Format to save figures in
    annotate_significance
        If True, add statistical significance annotations. Default is False

    Returns
    -------
    :
        Tuple containing figure and array of axes objects
    """
    logger.info("Plotting KS test barplots")
    plt.rcParams.update({"font.size": 6})

    num_cols = len(distance_measures)
    fig, axs = plt.subplots(
        1,
        num_cols,
        figsize=(1.25 * num_cols, 1.25),
        dpi=300,
        squeeze=False,
    )

    for col, distance_measure in enumerate(distance_measures):
        df_distance_measure = df_ks_bootstrap.query(f"distance_measure == '{distance_measure}'")

        ax = axs[0, col]

        plot_params = {
            "data": df_distance_measure,
            "x": "packing_mode",
            "y": "similar_fraction",
            "hue": "packing_mode",
            "legend": False,
            "orient": "v",
            "palette": label_tables.COLOR_PALETTE,
            "linewidth": 0.5,
            "errorbar": "sd",
            "err_kws": {"linewidth": 1},
        }
        # sns.boxplot(ax=ax, whis=(2.5, 97.5), fliersize=1, **plot_params)  # type: ignore
        sns.barplot(ax=ax, **plot_params)  # type: ignore
        sns.despine(ax=ax)
        ax.set_ylim((0, 1))
        ax.set_ylabel("Fraction")
        ax.set_xlabel("")
        ax.set_xticks(
            ax.get_xticks(),
            labels=[
                label_tables.MODE_LABELS.get(label.get_text(), label.get_text())
                for label in ax.get_xticklabels()
            ],
            rotation=45,
        )

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
        If True, add statistical significance annotations. Default is False

    Returns
    -------
    :
        Tuple containing (bar_figure, bar_axes, violin_figure, violin_axes)

    Raises
    ------
    ValueError
        If comparison_type is not 'intra_mode' or 'baseline'
    """
    if comparison_type == "intra_mode":
        logger.info("Plotting within rule EMD plots")
        file_prefix = "intra_mode_emd"
    elif comparison_type == "baseline":
        logger.info("Plotting baseline mode EMD plots")
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
        figsize=(1.25 * num_cols, 1.25),
        dpi=300,
        squeeze=False,
    )
    fig_violin, axs_violin = plt.subplots(
        1,
        num_cols,
        figsize=(1.25 * num_cols, 1.25),
        dpi=300,
        squeeze=False,
    )

    for col, distance_measure in enumerate(distance_measures):
        # Build query and plot parameters based on comparison type
        if comparison_type == "intra_mode":
            query_str = (
                f"distance_measure == '{distance_measure}' and " "packing_mode_1 == packing_mode_2"
            )
            x_column = "packing_mode_1"
        else:  # baseline comparison
            query_str = (
                f"distance_measure == '{distance_measure}' and "
                f"packing_mode_1 == '{baseline_mode}' and packing_mode_2 != '{baseline_mode}'"
            )
            x_column = "packing_mode_2"

        df_plot = df_emd.query(query_str)

        ax_bar = axs_bar[0, col]
        ax_violin = axs_violin[0, col]

        plot_params = {
            "data": df_plot,
            "x": x_column,
            "y": "emd",
            "hue": x_column,
            "legend": False,
            "orient": "v",
            "palette": label_tables.COLOR_PALETTE,
            "linewidth": 0.5,
        }
        sns.boxplot(ax=ax_bar, showfliers=False, whis=(2.5, 97.5), **plot_params)  # type: ignore
        sns.violinplot(ax=ax_violin, **plot_params)

        for ax, plot_type in zip([ax_bar, ax_violin], ["barplot", "violinplot"], strict=False):
            sns.despine(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("EMD")
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.set_xticks(
                ax.get_xticks(),
                labels=[
                    label_tables.MODE_LABELS.get(label.get_text(), label.get_text())
                    for label in ax.get_xticklabels()
                ],
                rotation=45,
            )

            if annotate_significance:
                if comparison_type == "intra_mode":
                    packing_modes = df_plot[x_column].unique()
                    pairs = list(combinations(packing_modes, 2))
                    pairs = [pair for pair in pairs if baseline_mode in pair]
                else:  # baseline comparison
                    packing_modes = [
                        mode for mode in df_plot[x_column].unique() if mode != baseline_mode
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

        for fig, label in zip([fig_bar, fig_violin], ["barplot", "violinplot"], strict=False):
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
    packing_mode: str = "random",
    figures_dir: Path | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    normalization: str | None = None,
    method: Literal["pdf", "cumulative"] = "pdf",
    xlim: float | None = None,
    num_points: int = 250,
    cellid_index: int | None = None,
    save_format: str = "png",
    bandwidth: Literal["scott", "silverman"] | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Figure, list[Axes]]:
    """
    Create an illustration of occupancy analysis showing occupied, available, and ratio plots.

    Parameters
    ----------
    kde_dict
        Dictionary containing KDE information
        {cell_id:{mode:gaussian_kde, "available":gaussian_kde}}
    packing_mode
        Packing mode to illustrate
    figures_dir
        Directory to save the figures
    suffix
        Suffix to add to the figure filename
    distance_measure
        The distance measure to plot
    normalization
        Normalization method applied
    method
        Method for ratio calculation
    xlim
        X-axis limit for plots
    num_points
        Number of points for KDE evaluation
    cellid_index
        Index of seed to use for illustration
    save_format
        Format to save the figures in
    bandwidth
        Bandwidth method for KDE

    Returns
    -------
    :
        Tuple containing (pdf_occupied, pdf_available, x_values, y_values, figure, axes)
    """
    plt.rcParams.update({"font.size": 6})
    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=(2.5, 2.5), sharex=True)

    all_cell_ids = [cell_id for cell_id in kde_dict.keys() if packing_mode in kde_dict[cell_id]]

    if cellid_index is not None:
        if str(cellid_index) in all_cell_ids:
            cell_id = str(cellid_index)
        elif cellid_index < len(all_cell_ids):
            cell_id = all_cell_ids[cellid_index]
        else:
            cell_id = all_cell_ids[0]
            logger.warning(
                f"Seed index {cellid_index} out of range. Using first cell ID {cell_id} instead."
            )
    else:
        cell_id = all_cell_ids[0]
    logger.info(f"Using cell ID {cell_id} for occupancy illustration")

    # Use first seed available for the baseline mode
    seeds = kde_dict[cell_id]["occupied"].keys()
    if not seeds:
        raise ValueError(f"No seeds found for cell ID {cell_id} in occupied KDE data")
    seed = next(iter(seeds))
    occupied_kde = kde_dict[cell_id]["occupied"][seed][packing_mode]
    available_kde = kde_dict[cell_id]["available"]
    xmax = occupied_kde.dataset.max()
    if xlim is not None and xlim < xmax:
        xmax = xlim
    xvals = np.linspace(0, xmax, num_points)
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
        distance_label = label_tables.DISTANCE_MEASURE_LABELS[distance_measure]
        if normalization is not None:
            distance_label = (
                f"{distance_label} / {label_tables.NORMALIZATION_LABELS[normalization]}"
            )
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
    ax: Axes,
    structure_id: str,
    mesh_information_dict: dict[str, dict[str, Any]],
    normalization: str | None = None,
) -> Axes:
    """
    Add the mean and standard deviation of structure radius to the plot.

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
    baseline_mode: str | None = None,
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    distance_measure: str = "nucleus",
    xlim: float | None = None,
    ylim: float | None = None,
    save_format: str = "png",
    fig_params: dict[str, Any] | None = None,
    plot_individual: bool = True,
    show_legend: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot occupancy ratio for individual and combined data across packing modes.

    Parameters
    ----------
    occupancy_dict
        Dictionary containing occupancy information
    channel_map
        Dictionary mapping packing modes to channels
    baseline_mode
        The baseline packing mode for normalization. Default is None
    figures_dir
        Directory to save the figures
    suffix
        Suffix to add to the figure filename
    normalization
        Normalization method applied
    distance_measure
        The distance measure to plot
    xlim
        X-axis limits
    ylim
        Y-axis limits
    save_format
        Format for saving the figure
    fig_params
        Additional figure parameters
    plot_individual
        Whether to plot individual cell data
    show_legend
        Whether to show the legend
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    save_format
        Format to save the figures in
    fig_params
        Dictionary of figure parameters (dpi, figsize). Default is None
    plot_individual
        Whether to plot individual cell data. Default is True
    show_legend
        Whether to show the legend. Default is False

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    logger.info("Plotting individual occupancy values")

    if fig_params is None:
        fig_params = {"dpi": 300, "figsize": (2.5, 2.5)}

    plt.rcParams.update({"font.size": 8})

    fig, ax = plt.subplots(**fig_params)

    for mode in channel_map.keys():
        if plot_individual:
            for _cell_id, cell_id_dict in tqdm(
                occupancy_dict[mode]["individual"].items(), desc=f"Plotting {mode} occupancy"
            ):

                xvals = cell_id_dict["xvals"]
                occupancy = cell_id_dict["occupancy"]

                sns.lineplot(
                    x=xvals,
                    y=occupancy,
                    ax=ax,
                    color=label_tables.COLOR_PALETTE.get(mode, "gray"),
                    alpha=0.1,
                    linewidth=0.1,
                    label="_nolegend_",
                    zorder=0,
                )

        # overlay occupancy for combined data
        combined_xvals = occupancy_dict[mode]["combined"]["xvals"]
        combined_occupancy = occupancy_dict[mode]["combined"]["occupancy"]

        plot_params = {
            "color": label_tables.COLOR_PALETTE.get(mode, "gray"),
            "alpha": 0.8,
            "linewidth": 1.5,
            "label": label_tables.MODE_LABELS.get(mode, mode),
            "zorder": 2,
        }
        if mode == baseline_mode:
            plot_params.update({"alpha": 1, "linewidth": 2.5, "zorder": 3})
        sns.lineplot(
            x=combined_xvals,
            y=combined_occupancy,
            ax=ax,
            **plot_params,
        )

    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    if xlim is not None:
        ax.set_xlim((0, xlim))
    if ylim is not None:
        ax.set_ylim((0, ylim))
    ax.axhline(1, linewidth=1, color="gray", linestyle="--", zorder=1)
    distance_label = label_tables.DISTANCE_MEASURE_LABELS[distance_measure]
    if normalization is not None:
        distance_label = f"{distance_label} / {label_tables.NORMALIZATION_LABELS[normalization]}"
    else:
        distance_label = f"{distance_label} (\u03bcm)"
    ax.set_xlabel(distance_label)
    ax.set_ylabel("Occupancy Ratio")
    if not show_legend:
        ax.legend().remove()
    sns.despine(fig=fig)
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"{distance_measure}_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )

    plt.show()

    return fig, ax


def plot_occupancy_ratio_interpolation(
    interpolated_occupancy_dict: dict[str, Any],
    baseline_mode: str = "SLC25A17",
    distance_measure: str = "nucleus",
    figures_dir: Path | None = None,
    suffix: str = "",
    xlim: float | None = None,
    ylim: float | None = None,
    save_format: str = "png",
    plot_type: str = "individual",
    fig_params: dict[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot combined occupancy ratio with interpolated reconstruction.

    Parameters
    ----------
    interpolated_occupancy_dict
        Dictionary containing interpolated occupancy information
    baseline_mode
        The baseline packing mode
    distance_measure
        The distance measure to plot
    figures_dir
        Directory to save the figures
    suffix
        Suffix to add to the figure filename
    xlim
        X-axis limits
    ylim
        Y-axis limits
    save_format
        Format for saving the figure
    plot_type
        Type of plot ("individual" or "joint")
    fig_params
        Additional figure parameters
        Additional figure parameters
        Baseline packing mode for interpolation
    distance_measure
        The distance measure to plot
    figures_dir
        Directory to save the figures
    suffix
        Suffix to add to the figure filename
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    save_format
        Format to save the figures in
    plot_type
        Type of plot to generate: "individual" or "joint"
    fig_params
        Dictionary of figure parameters (dpi, figsize). Default is None

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    if distance_measure not in interpolated_occupancy_dict["occupancy"]:
        raise ValueError(
            f"Distance measure {distance_measure} not found in interpolated_occupancy_dict"
        )

    logger.info("Plotting occupancy interpolation for %s", distance_measure)

    if fig_params is None:
        fig_params = {"dpi": 300, "figsize": (3.5, 2.5)}

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(**fig_params)

    occupancy_dict = interpolated_occupancy_dict["occupancy"][distance_measure]
    xvals = occupancy_dict["xvals"]
    modes_dict = occupancy_dict["modes"]

    # Plot combined occupancy for each mode
    for mode, mode_occupancy in modes_dict.items():
        plot_params = {
            "color": label_tables.COLOR_PALETTE.get(mode, "gray"),
            "alpha": 0.7,
            "linewidth": 1.5,
            "label": label_tables.MODE_LABELS.get(mode, mode),
            "zorder": 1,
        }
        if mode == baseline_mode:
            plot_params.update({"alpha": 1, "linewidth": 2.5, "zorder": 2})
        sns.lineplot(
            x=xvals,
            y=mode_occupancy,
            ax=ax,
            **plot_params,
        )

    # Plot interpolated reconstructions
    occupancy_key = None
    label = ""
    if plot_type == "individual":
        occupancy_key = "reconstructed_individual"
        label = "Interpolated (Individual)"
    elif plot_type == "joint":
        occupancy_key = "reconstructed_joint"
        label = "Interpolated (Joint)"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'individual' or 'joint'")

    sns.lineplot(
        x=xvals,
        y=occupancy_dict[occupancy_key],
        ax=ax,
        color=label_tables.COLOR_PALETTE.get(baseline_mode, "gray"),
        label=label,
        alpha=1,
        linewidth=2,
        zorder=3,
        linestyle="-.",
    )

    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    if xlim is not None:
        ax.set_xlim((0, xlim))
    if ylim is not None:
        ax.set_ylim((0, ylim))
    ax.axhline(1, linewidth=1, color="gray", linestyle="--", zorder=0)
    ax.set_xlabel(f"{label_tables.DISTANCE_MEASURE_LABELS[distance_measure]} (\u03bcm)")
    ax.set_ylabel("Occupancy Ratio")
    ax.set_title(f"{plot_type.capitalize()}, MSE: {occupancy_dict[f'mse_{plot_type}']:.4f}")
    relative_contribution_text = [
        f"{label_tables.MODE_LABELS[key]}: {value:.0%}"
        for key, value in occupancy_dict[f"relative_contribution_{plot_type}"].items()
    ]
    ax.text(
        0.95,
        0.95,
        "\n".join(relative_contribution_text),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    sns.despine(fig=fig)
    ax.legend().remove()
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_{plot_type}_interpolated_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )

    plt.show()

    return fig, ax


def add_baseline_occupancy_interpolation_to_plot(
    ax: Axes,
    interpolated_occupancy_dict: dict[str, Any],
    baseline_mode: str = "SLC25A17",
    distance_measure: str = "nucleus",
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: str = "png",
    plot_type: str = "individual",
) -> tuple[Figure, Axes]:
    """
    Add interpolated baseline occupancy reconstruction to an existing plot.

    Parameters
    ----------
    ax
        Matplotlib Axes object to add the plot to
    interpolated_occupancy_dict
        Dictionary containing interpolated occupancy information
    baseline_mode
        Baseline packing mode for interpolation
    distance_measure
        The distance measure to plot
    figures_dir
        Directory to save the figures
    suffix
        Suffix to add to the figure filename
    save_format
        Format to save the figures in
    plot_type
        Type of plot to generate: "individual" or "joint"

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    if distance_measure not in interpolated_occupancy_dict["occupancy"]:
        raise ValueError(
            f"Distance measure {distance_measure} not found in interpolated_occupancy_dict"
        )

    logger.info("Adding occupancy interpolation for %s", distance_measure)

    plt.rcParams.update({"font.size": 8})
    fig = ax.get_figure()
    if fig is None:
        raise ValueError("The axes object does not have an associated figure")
    if not isinstance(fig, Figure):
        raise ValueError("The axes object is not associated with a main Figure")

    occupancy_dict = interpolated_occupancy_dict["occupancy"][distance_measure]
    xvals = occupancy_dict["xvals"]

    # Plot interpolated reconstructions
    occupancy_key = None
    label = ""
    if plot_type == "individual":
        occupancy_key = "reconstructed_individual"
        label = "Interpolated (Individual)"
    elif plot_type == "joint":
        occupancy_key = "reconstructed_joint"
        label = "Interpolated (Joint)"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'individual' or 'joint'")

    sns.lineplot(
        x=xvals,
        y=occupancy_dict[occupancy_key],
        ax=ax,
        color=label_tables.COLOR_PALETTE.get(baseline_mode, "gray"),
        label=label,
        alpha=1,
        linewidth=2,
        zorder=3,
        linestyle="-.",
    )

    ax.set_title(f"{plot_type.capitalize()}, MSE: {occupancy_dict[f'mse_{plot_type}']:.4f}")

    x, y = 0.95, 0.95

    for i, (key, value) in enumerate(occupancy_dict[f"relative_contribution_{plot_type}"].items()):
        color = label_tables.COLOR_PALETTE.get(key, "gray")
        text = f"{label_tables.MODE_LABELS.get(key, key)}: {value:.0%}"
        ax.text(
            x,
            y - i * 0.05,
            text,
            color=color,
            fontsize=4,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
        )
    if ax.legend() is not None:
        ax.legend().remove()
    sns.despine(fig=fig)
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_{plot_type}_interpolated_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
            bbox_inches="tight",
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
    Plot mean occupancy ratio with confidence intervals across packing modes.

    Parameters
    ----------
    distance_dict
        Dictionary containing distance information
    kde_dict
        Dictionary containing KDE information
    packing_modes
        List of packing modes
    figures_dir
        Directory to save the figures
    suffix
        Suffix to add to the figure filename
    normalization
        Normalization method applied
    distance_measure
        The distance measure to plot
    method
        Method for ratio calculation
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    sample_size
        Number of samples to use
    save_format
        Format to save the figures in
    num_points
        Number of points for KDE evaluation
    bandwidth
        Bandwidth method for KDE

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    logger.info("Plotting occupancy with confidence intervals")

    lines = []
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    for _ct, mode in enumerate(packing_modes):

        color = label_tables.COLOR_PALETTE.get(mode, "gray")

        logger.info(f"Calculating occupancy for {mode}")
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
            label=label_tables.MODE_LABELS.get(mode, mode),
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
    ax.axhline(1, linewidth=1, color="gray", linestyle="--")
    distance_label = label_tables.DISTANCE_MEASURE_LABELS[distance_measure]
    if normalization is not None:
        distance_label = f"{distance_label} / {label_tables.NORMALIZATION_LABELS[normalization]}"
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


def plot_binned_occupancy_ratio_calc(
    kde_dict: dict[str, dict[str, gaussian_kde]],
    channel_map: dict[str, str],
    figures_dir: Path | None = None,
    normalization: str | None = None,
    suffix: str = "",
    num_bins: int = 64,
    num_cells: int | None = None,
    bin_width: float | None = None,
    distance_measure: str = "nucleus",
    xlim: float | None = None,
    ylim: float | None = None,
    save_format: str = "png",
) -> tuple[Figure, Axes]:
    """
    Calculate and plot binned occupancy ratio from KDE information.

    Parameters
    ----------
    kde_dict
        Dictionary containing KDE information for various entities
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
    num_cells
        Maximum number of cells to include per structure
    bin_width
        Width of bins (overrides num_bins if provided)
    distance_measure
        The measure of distance to use
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(dpi=300, figsize=(2.5, 2.5))

    cell_id_map = get_cell_id_map_from_distance_kde_dict(kde_dict, channel_map)
    if num_cells is not None:
        for structure_id, cell_ids in cell_id_map.items():
            if len(cell_ids) > num_cells:
                cell_id_map[structure_id] = np.random.choice(
                    cell_ids, num_cells, replace=False
                ).tolist()

    # Get all available distances to use for a combined_plot
    combined_available_distance_dict = {}
    for structure_id, cell_ids in cell_id_map.items():
        combined_available_distances = []
        for cell_id in cell_ids:
            combined_available_distances.extend(kde_dict[cell_id]["available_distance"].dataset)
        combined_available_distance_dict[structure_id] = np.concatenate(
            combined_available_distances
        )

    for mode, structure_id in channel_map.items():
        logger.info(f"Calculating binned occupancy ratio for: {mode}")

        max_distance = np.nanmax(combined_available_distance_dict[structure_id])
        if bin_width is not None:
            num_bins = int(max_distance / bin_width)
        bins = np.linspace(0, max_distance, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        available_space_counts = {}
        occupied_space_counts = {}
        for cell_id in tqdm(cell_id_map.get(structure_id, []), desc=f"Processing {mode} cells"):
            occupied_distances = kde_dict[cell_id][mode].dataset
            available_distances = kde_dict[cell_id]["available_distance"].dataset
            available_space_counts[cell_id] = (
                np.histogram(available_distances, bins=bins, density=True)[0] + 1e-16
            )
            occupied_space_counts[cell_id] = (
                np.histogram(occupied_distances, bins=bins, density=True)[0] + 1e-16
            )
            # sns.lineplot(
            #     x=bin_centers,
            #     y=occupied_space_counts[cell_id] / available_space_counts[cell_id],
            #     color=label_tables.COLOR_PALETTE.get(mode, "gray"),
            #     alpha=0.1,
            #     linewidth=0.1,
            #     ax=ax,
            #     label="_nolegend_",
            #     zorder=0,
            # )

        occupied_space_counts = np.vstack(list(occupied_space_counts.values()))
        available_space_counts = np.vstack(list(available_space_counts.values()))

        # normalized_occupied_space_counts = (
        #     occupied_space_counts / np.sum(occupied_space_counts, axis=1)[:, np.newaxis]
        # )
        # normalized_available_space_counts = (
        #     available_space_counts / np.sum(available_space_counts, axis=1)[:, np.newaxis]
        # )
        # occupancy_ratio = normalized_occupied_space_counts / normalized_available_space_counts
        occupancy_ratio = occupied_space_counts / available_space_counts
        mean_occupancy_ratio = np.nanmean(occupancy_ratio, axis=0)
        std_occupancy_ratio = np.nanstd(occupancy_ratio, axis=0)

        sns.lineplot(
            x=bin_centers,
            y=mean_occupancy_ratio,
            label=label_tables.MODE_LABELS.get(mode, mode),
            color=label_tables.COLOR_PALETTE.get(mode, "gray"),
            ax=ax,
            linewidth=1.5,
            zorder=2,
        )
        ax.fill_between(
            bin_centers,
            mean_occupancy_ratio - std_occupancy_ratio,
            mean_occupancy_ratio + std_occupancy_ratio,
            alpha=0.1,
            color=label_tables.COLOR_PALETTE.get(mode, "gray"),
            linewidth=0,
            label="_nolegend_",
            zorder=0,
        )

    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    if xlim is not None:
        ax.set_xlim((0, xlim))
    else:
        cur_xlim = ax.get_xlim()
        ax.set_xlim((0, cur_xlim[1]))
    if ylim is not None:
        ax.set_ylim((0, ylim))
    ax.axhline(1, linewidth=1, color="gray", linestyle="--", zorder=1)

    distance_label = label_tables.DISTANCE_MEASURE_LABELS[distance_measure]
    if normalization is not None:
        distance_label = f"{distance_label} / {label_tables.NORMALIZATION_LABELS[normalization]}"
    else:
        distance_label = f"{distance_label} (\u03bcm)"
    ax.set_xlabel(distance_label)
    ax.set_ylabel("Occupancy Ratio")
    sns.despine(fig=fig)
    fig.tight_layout()

    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"{distance_measure}_binned_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )
    plt.show()

    return fig, ax


def plot_binned_occupancy_ratio(
    binned_occupancy_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str],
    figures_dir: Path | None = None,
    normalization: str | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    xlim: float | None = None,
    ylim: float | None = None,
    save_format: str = "png",
) -> tuple[Any, Any]:
    """
    Plot binned occupancy ratio from precomputed binned occupancy dictionary.

    Parameters
    ----------
    binned_occupancy_dict
        Dictionary containing binned occupancy information for various entities
    channel_map
        Dictionary mapping packing modes to channels
    figures_dir
        Directory to save the results
    normalization
        Method to normalize the data
    suffix
        Suffix to append to the result filenames
    distance_measure
        The measure of distance to use
    xlim
        X-axis limit for plots
    ylim
        Y-axis limit for plots
    save_format
        Format to save the figures in

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    logger.info("Plotting binned occupancy ratio")
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(dpi=300, figsize=(2.5, 2.5))

    for mode in channel_map.keys():
        combined_xvals = binned_occupancy_dict[mode]["combined"]["xvals"]
        combined_occupancy = binned_occupancy_dict[mode]["combined"]["occupancy"]
        std_occupancy = binned_occupancy_dict[mode]["combined"]["std_occupancy"]
        sns.lineplot(
            x=combined_xvals,
            y=combined_occupancy,
            label=label_tables.MODE_LABELS.get(mode, mode),
            color=label_tables.COLOR_PALETTE.get(mode, "gray"),
            ax=ax,
            linewidth=1.5,
            zorder=2,
        )
        ax.fill_between(
            combined_xvals,
            combined_occupancy - std_occupancy,
            combined_occupancy + std_occupancy,
            alpha=0.1,
            color=label_tables.COLOR_PALETTE.get(mode, "gray"),
            linewidth=0,
            label="_nolegend_",
            zorder=0,
        )

    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    if xlim is not None:
        ax.set_xlim((0, xlim))
    else:
        cur_xlim = ax.get_xlim()
        ax.set_xlim((0, cur_xlim[1]))
    if ylim is not None:
        ax.set_ylim((0, ylim))
    ax.axhline(1, linewidth=1, color="gray", linestyle="--", zorder=1)

    distance_label = label_tables.DISTANCE_MEASURE_LABELS[distance_measure]
    if normalization is not None:
        distance_label = f"{distance_label} / {label_tables.NORMALIZATION_LABELS[normalization]}"
    else:
        distance_label = f"{distance_label} (\u03bcm)"
    ax.set_xlabel(distance_label)
    ax.set_ylabel("Occupancy Ratio")
    sns.despine(fig=fig)
    fig.tight_layout()

    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"{distance_measure}_binned_occupancy_ratio{suffix}.{save_format}",
            dpi=300,
        )
    plt.show()

    return fig, ax


def plot_grid_points_slice(
    grid_points_slice: np.ndarray,
    inside_mem_outside_nuc: np.ndarray,
    inside_nuc: np.ndarray,
    color_var: np.ndarray,
    cbar_label: str,
    dot_size: float = 2,
    projection_axis: str = "z",
    cmap: Colormap | str | None = None,
    reverse_cmap: bool = False,
    clim: tuple[float, float] | None = None,
):
    """
    Plot a slice of grid points with coloring based on a variable.

    Parameters
    ----------
    grid_points_slice
        Grid points for the slice in pixels
    inside_mem_outside_nuc
        Boolean mask for points inside membrane but outside nucleus
    inside_nuc
        Boolean mask for points inside nucleus
    color_var
        Variable to use for coloring the points
    cbar_label
        Label for the colorbar
    dot_size
        Size of the dots in the scatter plot
    projection_axis
        Axis along which to project ("x", "y", or "z")
    cmap
        Colormap to use for the plot
    reverse_cmap
        Whether to reverse the colormap
    clim
        Color limits for the plot
        Values to use for coloring the cytoplasm points
    cbar_label
        Label for the colorbar
    cell_id
        Cell ID for file naming
    figures_dir
        Directory to save the figure
    dot_size
        Size of scatter plot points (default: 2)
    projection_axis
        Axis of projection ('x', 'y', or 'z')
    cmap
        Colormap to use for coloring. Default is None
    reverse_cmap
        Whether to reverse the colormap
    clim
        Color limits as (min, max) tuple. Default is None

    Returns
    -------
    :
        Matplotlib figure and axis objects
    """
    grid_points_um = grid_points_slice * PIXEL_SIZE_IN_UM
    centroid = np.mean(grid_points_um, axis=0)
    # custom_cmap = LinearSegmentedColormap.from_list("cyan_to_magenta", ["cyan", "magenta"])
    # custom_cmap = LinearSegmentedColormap.from_list(
    #     "gray_cutoff", plt.cm.get_cmap("gray")(np.linspace(0, 0.9, 256))
    # )
    custom_cmap = LinearSegmentedColormap.from_list(
        "reds_cutoff", plt.cm.get_cmap("Reds")(np.linspace(0.3, 1, 256))
    )
    if cmap is not None:
        if isinstance(cmap, Colormap):
            custom_cmap = cmap
        else:
            custom_cmap = plt.get_cmap(cmap)
    if reverse_cmap:
        custom_cmap = custom_cmap.reversed()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    x_index, y_index = label_tables.PROJECTION_TO_INDEX_MAP[projection_axis]
    x_label, y_label = label_tables.PROJECTION_TO_LABEL_MAP[projection_axis]

    vmin, vmax = None, None
    if clim is not None:
        vmin, vmax = clim

    # Plot cytoplasm points with weight coloring
    ax.scatter(
        x=grid_points_um[inside_mem_outside_nuc, x_index] - centroid[x_index],
        y=grid_points_um[inside_mem_outside_nuc, y_index] - centroid[y_index],
        c=color_var,
        cmap=custom_cmap,
        s=dot_size,
        vmin=vmin,
        vmax=vmax,
        marker=".",
        edgecolor="none",
    )

    # Plot nucleus points
    ax.scatter(
        x=grid_points_um[inside_nuc, x_index] - centroid[x_index],
        y=grid_points_um[inside_nuc, y_index] - centroid[y_index],
        c="green",
        s=dot_size,
        marker=".",
        edgecolor="none",
    )

    ax.set_xlabel(f"{x_label} (\u03bcm)")
    ax.set_ylabel(f"{y_label} (\u03bcm)")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    sns.despine(ax=ax)

    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    return fig, ax


def plot_envelope_for_cell(
    cell_results: dict[str, dict],
    packing_mode: str,
    distance_measure: str,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot ECDF envelope for single cell, model, and distance measure.

    Displays observed ECDF against Monte Carlo envelope with mean curve.

    Parameters
    ----------
    cell_results
        Result dictionary for one cell from monte_carlo_per_cell
        {packing_mode: {"per_distance_measure": {distance_measure: {info}}}}
    packing_mode
        Name of packing mode to compare
    distance_measure
        Name of distance measure to plot
    title
        Optional custom plot title. If None, auto-generates from packing_mode/distance_measure
    ax
        Optional Matplotlib Axes to plot on. If None, creates a new figure.
    """
    info = cell_results[packing_mode]["per_distance_measure"][distance_measure]
    r = info["r"]
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(4, 4))
    else:
        fig = ax.get_figure(root=True)
        if fig is None:
            raise ValueError("The provided Axes object does not have an associated Figure")
    ax.fill_between(r, info["lo"], info["hi"], color="lightgray", alpha=0.7, label="MC envelope")
    ax.plot(r, info["mu"], "--", color="dimgray", label="Sim mean")
    ax.plot(r, info["obs_curve"], color="crimson", lw=2, label="Observed")
    ttl = title or (
        f"Cell: {distance_measure} vs {label_tables.MODE_LABELS.get(packing_mode, packing_mode)} "
        f"(p={info['pval']:.3f})"
    )
    ax.set_title(ttl)
    ax.set_xlabel("r")
    ax.set_ylabel("ECDF")
    # ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig, ax


def plot_rejection_bars(
    rej_rates_dict: pd.Series,
    title: str = "Rejection rates (q < alpha)",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot bar chart of rejection rates across packing modes or distance measures.

    Handles both joint rejection rates (single bars per packing mode) and
    per-distance measure rates (grouped bars per packing mode/distance measure combination).

    Parameters
    ----------
    rej_rates_dict: pd.Series
        Series with rejection rates, optionally with MultiIndex for per-distance measure rates
    title
        Plot title
    ax
        Optional Matplotlib Axes to plot on. If None, creates a new figure.

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    else:
        fig = ax.get_figure(root=True)
        if fig is None:
            raise ValueError("The provided Axes object does not have an associated Figure")
    if isinstance(rej_rates_dict, pd.Series) and isinstance(rej_rates_dict.index, pd.MultiIndex):
        # per-distance measure: make a grouped bar
        df = rej_rates_dict.unstack(level="distance_measure").reset_index()
        # pivot to a long format for seaborn
        df = df.melt(
            id_vars="packing_mode", var_name="distance_measure", value_name="rejection_rate"
        )

        df.rename(
            columns={"packing_mode": "Packing Mode", "distance_measure": "Distance Measure"},
            inplace=True,
        )
        ax = sns.barplot(
            data=df,
            x="Packing Mode",
            y="rejection_rate",
            hue="Distance Measure",
            palette=label_tables.COLOR_PALETTE,
            ax=ax,
        )
        ax.set_ylabel("Fraction of cells rejected")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    else:
        ser = pd.Series(rej_rates_dict)
        ax = sns.barplot(
            x=ser.index,
            y=ser.values,
            hue=ser.index,
            palette=label_tables.COLOR_PALETTE,
            ax=ax,
            legend=False,
        )
        ax.set_ylabel("Fraction of cells rejected")
        ax.set_title(title)
        ax.set_xlabel("Packing mode")
    ax.set_ylim(0, 1)
    ax.set_xticks(
        ax.get_xticks(),
        labels=[
            label_tables.MODE_LABELS.get(label.get_text(), label.get_text())
            for label in ax.get_xticklabels()
        ],
    )
    sns.despine(fig=fig)
    plt.tight_layout()
    return fig, ax


def plot_rejection_bars_by_sign(
    rej_positive: pd.Series,
    rej_negative: pd.Series,
    title: str | None = None,
    test_statistic: Literal["intdev", "supremum"] = "intdev",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot two-sided bar chart of rejection rates by test statistic deviation direction.

    Positive deviations extend up (observed > simulated mean),
    negative deviations extend down (observed < simulated mean).

    Parameters
    ----------
    rej_positive
        Fraction of cells rejected with positive deviations, indexed by packing_mode
    rej_negative
        Fraction of cells rejected with negative deviations, indexed by packing_mode
    title
        Plot title
    test_statistic
        Which test statistic to display in the title ("intdev" or "supremum")
    ax
        Optional Matplotlib Axes to plot on. If None, creates a new figure.

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    else:
        fig = ax.get_figure(root=True)
        if fig is None:
            raise ValueError("The provided Axes object does not have an associated Figure")
    if title is None:
        title = f"Rejection rates by deviation direction ({test_statistic})"
    packing_modes = rej_positive.index.tolist()
    x_pos = np.arange(len(packing_modes))

    # Convert to numpy arrays for plotting
    pos_values = np.asarray(rej_positive.values, dtype=float)
    neg_values = np.asarray(rej_negative.values, dtype=float)

    # Get colors for each packing mode from COLOR_PALETTE
    # For positive deviations: use saturated colors
    # For negative deviations: use desaturated colors (50% saturation)
    pos_colors = [
        label_tables.COLOR_PALETTE.get(mode, label_tables.COLOR_PALETTE.get("random", "gray"))
        for mode in packing_modes
    ]
    neg_colors = [
        label_tables.adjust_color_saturation(
            label_tables.COLOR_PALETTE.get(mode, label_tables.COLOR_PALETTE.get("random", "gray")),
            saturation=0.3,
        )
        for mode in packing_modes
    ]

    # Positive bars (upward) - use ax.bar for per-mode colors
    ax.bar(
        x_pos,
        pos_values,
        color=pos_colors,
        alpha=0.9,
        label="(obs > sim mean)",
    )

    # Negative bars (downward) - use desaturated colors
    ax.bar(
        x_pos,
        -neg_values,
        color=neg_colors,
        alpha=0.9,
        label="(obs < sim mean)",
    )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_ylabel("Fraction of cells rejected")
    ax.set_xlabel("Packing mode")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    mode_labels = [label_tables.MODE_LABELS.get(mode, mode) or mode for mode in packing_modes]
    ax.set_xticklabels(mode_labels)

    # Set y-axis limits symmetrically
    y_max = max(pos_values.max(), neg_values.max()) * 1.1
    if y_max > 0:
        ax.set_ylim(-y_max, y_max)

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 0.9))
    sns.despine(fig=fig, left=False)
    plt.tight_layout()
    return fig, ax


def plot_grouped_rejection_bars_by_sign(
    rej_positive: pd.Series,
    rej_negative: pd.Series,
    title: str | None = None,
    test_statistic: Literal["intdev", "supremum"] = "intdev",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot grouped bar chart of rejection rates by distance measure and deviation direction.

    Creates a grouped barplot where each packing mode has grouped bars for each distance
    measure, with positive deviations extending upward and negative deviations downward.

    Parameters
    ----------
    rej_positive
        Fraction of cells rejected with positive deviations,
        MultiIndex with (packing_mode, distance_measure)
    rej_negative
        Fraction of cells rejected with negative deviations,
        MultiIndex with (packing_mode, distance_measure)
    title
        Plot title
    test_statistic
        Which test statistic to display in the title ("intdev" or "supremum")
    ax
        Optional Matplotlib Axes to plot on. If None, creates a new figure.

    Returns
    -------
    :
        Tuple containing figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(8, 5))
    else:
        fig = ax.get_figure(root=True)
        if fig is None:
            raise ValueError("The provided Axes object does not have an associated Figure")

    if title is None:
        title = f"Rejection rates by deviation direction and distance measure ({test_statistic})"

    # Get packing modes and distance measures from MultiIndex
    packing_modes = rej_positive.index.get_level_values("packing_mode").unique().tolist()
    distance_measures = rej_positive.index.get_level_values("distance_measure").unique().tolist()

    # Number of groups and bars per group
    n_modes = len(packing_modes)
    n_measures = len(distance_measures)
    bar_width = 0.8 / n_measures
    x_pos = np.arange(n_modes)

    # Plot grouped bars for each distance measure
    for idx, measure in enumerate(distance_measures):
        # Get color for this distance measure
        color = label_tables.COLOR_PALETTE.get(measure, "#808080")
        color_neg = label_tables.adjust_color_saturation(color, saturation=0.3)

        # Extract positive and negative values for this measure across all packing modes
        pos_values = []
        neg_values = []
        for mode in packing_modes:
            pos_val = rej_positive.loc[(mode, measure)]
            neg_val = rej_negative.loc[(mode, measure)]
            pos_values.append(pos_val)
            neg_values.append(neg_val)

        pos_values = np.array(pos_values)
        neg_values = np.array(neg_values)

        # Calculate x positions for this measure
        x_offset = x_pos + (idx - n_measures / 2 + 0.5) * bar_width

        # Plot positive bars (upward)
        ax.bar(
            x_offset,
            pos_values,
            width=bar_width,
            color=color,
            alpha=0.9,
            label=label_tables.DISTANCE_MEASURE_LABELS.get(measure, measure),
        )

        # Plot negative bars (downward)
        ax.bar(
            x_offset,
            -neg_values,
            width=bar_width,
            color=color_neg,
            alpha=0.9,
        )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Fraction of cells rejected")
    ax.set_xlabel("Packing mode")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    mode_labels = [label_tables.MODE_LABELS.get(mode, mode) or mode for mode in packing_modes]
    ax.set_xticklabels(mode_labels)

    # Set y-axis limits symmetrically
    all_pos = np.asarray(rej_positive.values, dtype=float)
    all_neg = np.asarray(rej_negative.values, dtype=float)
    y_max = max(all_pos.max(), all_neg.max()) * 1.1
    if y_max > 0:
        ax.set_ylim(-y_max, y_max)

    ax.legend(
        frameon=False,
        loc="upper left",
        title="Distance measure",
        bbox_to_anchor=(1, 0.9),
    )
    sns.despine(fig=fig, left=False)
    plt.tight_layout()
    return fig, ax
