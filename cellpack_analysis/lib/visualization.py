import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from statannotations.Annotator import Annotator

from cellpack_analysis.lib import label_tables
from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import get_scaled_structure_radius
from cellpack_analysis.lib.occupancy import get_kde_occupancy_for_single_cell
from cellpack_analysis.lib.rule_interpolation import CVResult

logger = logging.getLogger(__name__)


def get_distance_label(distance_measure: str, normalization: str | None = None) -> str:
    """
    Get the label for a distance measure, optionally including normalization.

    Parameters
    ----------
    distance_measure
        The name of the distance measure (e.g. "nucleus", "membrane", "z", etc.)
    normalization
        The normalization method applied to the distance measure
        (e.g. "cell_diameter", "intracellular_radius", "max_distance"),
        or None if no normalization is applied

    Returns
    -------
    :
        The label for the distance measure, optionally including normalization
    """
    if normalization is not None and normalization not in label_tables.NORMALIZATION_LABELS:
        logger.warning(
            f"Normalization '{normalization}' not recognized. "
            f"Available normalizations: {list(label_tables.NORMALIZATION_LABELS.keys())}"
        )

    distance_label = label_tables.DISTANCE_MEASURE_LABELS.get(distance_measure, distance_measure)
    if normalization is not None:
        distance_label = (
            f"{distance_label} / "
            f"{label_tables.NORMALIZATION_LABELS.get(normalization, normalization)}"
        )
    return distance_label


def plot_cell_diameter_distribution(
    mesh_information_dict: dict[str, dict[str, Any]],
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


def plot_distance_distributions(
    distance_pdf_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    distance_measures: list[str],
    packing_modes: list[str],
    figures_dir: Path | None = None,
    suffix: str = "",
    normalization: str | None = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
    plot_individual_curves: bool = False,
    envelope_alpha: float = 0.05,
    production_mode: bool = False,
    overlay_mean_and_std: bool = False,
    figure_size: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, dict[int, Axes]]]:
    """Plot distance distributions from pre-computed PDFs.

    Accepts the unified ``distance_pdf_dict`` produced by
    :func:`~cellpack_analysis.lib.distance.compute_distance_pdfs` and renders
    mean curves with pointwise envelopes.  Works identically for histogram and
    KDE sources.

    Parameters
    ----------
    distance_pdf_dict
        ``{distance_measure: {mode: {"xvals", "individual_curves",
        "mean_pdf", "envelope_lo", "envelope_hi"}}}``
    distance_measures
        Distance measures to plot (columns).
    packing_modes
        Packing modes to plot (rows).
    figures_dir
        Directory to save the figure, or ``None`` to skip saving.
    suffix
        Suffix appended to the saved filename.
    normalization
        Normalization label for axis annotation.
    save_format
        Image format for saving.
    plot_individual_curves
        If ``True``, overlay every replicate curve (low alpha).
    envelope_alpha
        Only used for the legend label (envelope is pre-computed).
    production_mode
        If ``True``, use production-quality figure sizing and shared axis labels.
    overlay_mean_and_std
        If ``True``, annotate the weighted mean distance on each panel.
    figure_size
        Optional custom figure size (width, height in inches). Overrides default and
        production sizes when provided.

    Returns
    -------
    :
        ``(Figure, {distance_measure: {row_index: Axes}})``
    """
    logger.info("Starting unified distance distribution plot")
    plt.rcParams.update({"font.size": 5})

    all_ax_dict: dict[str, dict[int, Axes]] = {}
    num_rows = len(packing_modes)
    num_cols = len(distance_measures)
    figsize = (4.6, 2.25) if production_mode else (num_cols * 1.2, num_rows * 0.5)
    if figure_size is not None:
        figsize = figure_size
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
        dpi=300,
        squeeze=False,
        sharex="col",
        sharey="col",
    )

    for col, dm in enumerate(distance_measures):
        all_ax_dict.setdefault(dm, {})
        distance_label = get_distance_label(dm, normalization)

        y_max = -np.inf
        for row, mode in enumerate(packing_modes):
            ax = axs[row, col]
            mode_color = label_tables.COLOR_PALETTE.get(mode, "gray")
            mode_label = label_tables.MODE_LABELS.get(mode, mode)

            pdf_data = distance_pdf_dict[dm][mode]
            r_grid = pdf_data["xvals"]
            mean_pdf = pdf_data["mean_pdf"]
            lo_env = pdf_data["envelope_lo"]
            hi_env = pdf_data["envelope_hi"]

            # Individual replicate curves
            if plot_individual_curves:
                for curve in pdf_data["individual_curves"]:
                    ax.plot(r_grid, curve, color=mode_color, linewidth=0.15, alpha=0.05)

            # Envelope
            ax.fill_between(
                r_grid,
                lo_env,
                hi_env,
                color=mode_color,
                alpha=0.15,
                edgecolor="none",
                label=f"{int((1 - envelope_alpha) * 100)}% envelope",
            )

            # Mean curve
            ax.plot(
                r_grid,
                mean_pdf,
                color=mode_color,
                linewidth=0.7,
                label=f"{mode_label} mean",
            )

            # Mean distance annotation
            if overlay_mean_and_std:
                individual = pdf_data["individual_curves"]
                # Weighted mean: sum(xvals * pdf) / sum(pdf) averaged over replicates
                weighted_means = (individual * r_grid[np.newaxis, :]).sum(axis=1) / np.maximum(
                    individual.sum(axis=1), 1e-12
                )
                overall_mean = float(np.mean(weighted_means))
                ax.axvline(overall_mean, color=mode_color, linestyle="--", linewidth=0.7)
                ax.text(
                    overall_mean,
                    ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.9,
                    f"{overall_mean:.2f}",
                    color=mode_color,
                    fontsize=4,
                    ha="center",
                )

            # Axis limits
            ax.set_xlim(r_grid[0], r_grid[-1])
            y_max = max(y_max, float(np.nanmax(hi_env)) * 1.02)

            sns.despine(fig=fig)

            if col == 0:
                ax.set_ylabel(f"{mode_label}\nPDF" if not production_mode else "")
            else:
                ax.set_ylabel("")
            if row == num_rows - 1:
                ax.set_xlabel(distance_label if not production_mode else "")

            all_ax_dict[dm][row] = ax
            ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(2, integer=True))

        for ax in axs[:, col]:
            ax.set_ylim(0, y_max)

    if production_mode:
        fig.supylabel("Probability Density", fontsize=8)
        fig.supxlabel("Distance (\u03bcm)", fontsize=8)

    if figures_dir is not None:
        fig.tight_layout()
        plt.show()
        fig.savefig(
            figures_dir / f"distance_distribution{suffix}.{save_format}",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
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
            transparent=True,
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
                f"distance_measure == '{distance_measure}' and packing_mode_1 == packing_mode_2"
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
                transparent=True,
            )
            plt.show()
            plt.close(fig)

    return fig_bar, axs_bar, fig_violin, axs_violin


def plot_occupancy_illustration(
    distance_kde_dict: dict[str, dict[str, gaussian_kde]],
    packing_mode: str = "random",
    cell_id_or_index: str | int | None = None,
    figures_dir: Path | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    normalization: str | None = None,
    num_points: int = 100,
    bandwidth: float | None = None,
    xlim: float | None = None,
    ylim_ratio: float | None = None,
    save_format: str = "pdf",
) -> tuple[Figure, list[Axes]]:
    """
    Plot an occupancy illustration for one example cell.

    Shows three stacked panels sharing the x-axis:

    * **Top** — occupied-space PDF for the selected cell.
    * **Middle** — available-space PDF for the selected cell.
    * **Bottom** — occupancy ratio curve for the selected cell.

    Works with any occupancy dictionary produced by
    :func:`~cellpack_analysis.lib.occupancy.get_kde_occupancy_dict` or
    :func:`~cellpack_analysis.lib.occupancy.get_binned_occupancy_dict_from_distance_dict` —
    both share the same per-cell ``individual`` data structure.

    Parameters
    ----------
    distance_kde_dict
        ``{cell_id:{mode:gaussian_kde, "available":gaussian_kde}}`` for a single distance measure.
    packing_mode
        Packing mode to use for illustration.
    cell_id_or_index
        Specific cell to show.  When ``None`` the first available cell is used.
        An integer is treated as a positional index into the available cell list.
    figures_dir
        Directory to save the figure.  Skipped when ``None``.
    suffix
        Suffix appended to the saved filename.
    distance_measure
        Distance measure label used for the x-axis and filename.
    normalization
        Normalization method applied to distances (used for axis labelling).
    num_points
        Number of points to evaluate KDEs on.
    bandwidth
        KDE bandwidth to use when evaluating PDFs.  When ``None``, the default bandwidth of
        each KDE object is used.
    xlim
        Upper x-axis limit.
    ylim_ratio
        Upper y-axis limit for the ratio panel.
    save_format
        File format for saving.

    Returns
    -------
    :
        Tuple of ``(Figure, [ax_occupied, ax_available, ax_ratio])``.
    """
    if cell_id_or_index is None:
        cell_id = next(iter(distance_kde_dict))
    elif isinstance(cell_id_or_index, int):
        cell_ids = list(distance_kde_dict.keys())
        if cell_id_or_index < 0 or cell_id_or_index >= len(cell_ids):
            raise IndexError(
                f"cell_id_or_index {cell_id_or_index} out of range for mode '{packing_mode}'"
            )
        cell_id = cell_ids[cell_id_or_index]
    else:
        cell_id = str(cell_id_or_index)
        if cell_id not in distance_kde_dict:
            raise KeyError(f"cell_id '{cell_id}' not found for mode '{packing_mode}'")

    x_vals = np.linspace(0, xlim if xlim is not None else 1, num_points)
    result = get_kde_occupancy_for_single_cell(
        cell_id=cell_id,
        mode=packing_mode,
        distance_kde_dict=distance_kde_dict,
        x_vals=x_vals,
        bandwidth=bandwidth,
        num_points=num_points,
    )

    if result is None:
        raise ValueError(f"Mode '{packing_mode}' not found for cell '{cell_id}'")

    _, cell_result = result
    pdf_occupied = cell_result["pdf_occupied_common"]
    pdf_available = cell_result["pdf_available_common"]
    occupancy = cell_result["occupancy_common"]

    plt.rcParams.update({"font.size": 8})
    fig, axs = plt.subplots(3, 1, dpi=300, figsize=(3.5, 4), sharex=True)
    distance_label = get_distance_label(distance_measure, normalization)

    for ax, ylabel, ydata in zip(
        axs,
        ["Occupied PDF", "Available PDF", "Occupancy Ratio"],
        [pdf_occupied, pdf_available, occupancy],
        strict=False,
    ):
        sns.lineplot(x=x_vals, y=np.nan_to_num(ydata), ax=ax, color="k")
        ax.set_xlim(0, xlim if xlim is not None else float(x_vals.max()))
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        sns.despine(ax=ax)

    axs[-1].set_xlabel(distance_label)
    axs[-1].axhline(1, color="gray", linestyle="--")

    if ylim_ratio is not None:
        axs[2].set_ylim(0, ylim_ratio)

    mode_label = label_tables.MODE_LABELS.get(packing_mode, packing_mode)
    fig.suptitle(f"{mode_label} \u2014 cell {cell_id}", fontsize=8)
    fig.tight_layout()

    if figures_dir is not None:
        fname = f"{distance_measure}_occupancy_illustration_{packing_mode}{suffix}.{save_format}"
        fig.savefig(figures_dir / fname, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    return fig, list(axs)


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
    plot_individual: bool = False,
    show_legend: bool = False,
    show_envelope: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot occupancy ratio for each packing mode, with optional per-cell curves and pointwise
    envelope.

    Works with any occupancy dictionary produced by
    :func:`~cellpack_analysis.lib.occupancy.get_kde_occupancy_dict` or
    :func:`~cellpack_analysis.lib.occupancy.get_binned_occupancy_dict_from_distance_dict` —
    both produce the same unified format.

    Parameters
    ----------
    occupancy_dict
        ``{mode: {"individual": {cell_id: {"xvals", "occupancy", ...}},
        "combined": {"xvals", "occupancy", "envelope_lo", "envelope_hi", ...}}}``
    channel_map
        Mapping from packing modes to structure IDs.
    baseline_mode
        Mode rendered with a thicker, fully-opaque line.
    figures_dir
        Directory to save the figure.  Skipped when ``None``.
    suffix
        Suffix appended to the saved filename.
    normalization
        Normalization method for axis labelling.
    distance_measure
        Distance measure label used for axes and filename.
    xlim
        Upper x-axis limit.
    ylim
        Upper y-axis limit.
    save_format
        File format for saving.
    fig_params
        Optional matplotlib ``Figure`` keyword arguments (e.g.
        ``{"dpi": 300, "figsize": (3.5, 2.5)}``).
    plot_individual
        When ``True``, draw per-cell curves at low alpha behind the combined line.
    show_legend
        When ``True``, show the legend.
    show_envelope
        When ``True``, shade the pointwise envelope from
        ``combined["envelope_lo"]`` / ``combined["envelope_hi"]`` (if present).

    Returns
    -------
    :
        Tuple containing figure and axis objects.
    """
    logger.info("Plotting occupancy ratio")

    if fig_params is None:
        fig_params = {"dpi": 300, "figsize": (3.5, 2.5)}

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(**fig_params)

    for mode in channel_map.keys():
        color = label_tables.COLOR_PALETTE.get(mode, "gray")
        label = label_tables.MODE_LABELS.get(mode, mode)

        if plot_individual:
            for cell_id_dict in occupancy_dict[mode]["individual"].values():
                ax.plot(
                    cell_id_dict["xvals"],
                    cell_id_dict["occupancy"],
                    color=color,
                    alpha=0.1,
                    linewidth=0.1,
                    zorder=0,
                )

        combined = occupancy_dict[mode]["combined"]
        xvals = combined["xvals"]
        mean_occ = np.nan_to_num(combined["occupancy"])

        is_baseline = mode == baseline_mode
        ax.plot(
            xvals,
            mean_occ,
            color=color,
            alpha=1.0 if is_baseline else 0.8,
            linewidth=2.5 if is_baseline else 1.5,
            label=label,
            zorder=3 if is_baseline else 2,
        )

        if show_envelope and "envelope_lo" in combined and "envelope_hi" in combined:
            ax.fill_between(
                xvals,
                np.nan_to_num(combined["envelope_lo"]),
                np.nan_to_num(combined["envelope_hi"]),
                alpha=0.15,
                color=color,
                linewidth=0,
                label="_nolegend_",
                zorder=1,
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
    distance_label = get_distance_label(distance_measure, normalization)
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
            bbox_inches="tight",
            transparent=True,
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
            transparent=True,
        )

    plt.show()

    return fig, ax


def plot_grid_points_slice(
    grid_points_slice: np.ndarray,
    inside_mem_outside_nuc: np.ndarray,
    color_var: np.ndarray,
    cbar_label: str,
    inside_nuc: np.ndarray | None = None,
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
    if inside_nuc is not None:
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
    xvals = info["xvals"]
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(4, 4))
    else:
        fig = ax.get_figure(root=True)
        if fig is None:
            raise ValueError("The provided Axes object does not have an associated Figure")
    ax.fill_between(
        xvals,
        info["lo"],
        info["hi"],
        color="lightgray",
        alpha=0.7,
        label="MC envelope",
        edgecolor="none",
    )
    ax.plot(xvals, info["mu"], "--", color="dimgray", label="Sim mean")
    ax.plot(xvals, info["obs_curve"], color="crimson", lw=2, label="Observed")
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
        label="(ref > test)",
    )

    # Negative bars (downward) - use desaturated colors
    ax.bar(
        x_pos,
        -neg_values,
        color=neg_colors,
        alpha=0.9,
        label="(ref < test)",
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


def _compute_diag_xlim(
    modes: list[str],
    envelopes: dict[str, Any],
    distance_measure: str | None,
    pairwise_results: dict[str, Any],
) -> tuple[float, float] | None:
    """Compute shared x-axis limits across all diagonal envelope subplots."""
    diag_xmax = 0.0
    for mode in modes:
        if distance_measure is not None:
            if distance_measure in envelopes.get(mode, {}):
                xvals = envelopes[mode][distance_measure]["xvals"]
                diag_xmax = max(diag_xmax, float(xvals.max()))
        else:
            for dm in pairwise_results.get("distance_measures", []):
                if dm in envelopes.get(mode, {}):
                    xvals = envelopes[mode][dm]["xvals"]
                    diag_xmax = max(diag_xmax, float(xvals.max()))
    return (0, diag_xmax) if diag_xmax > 0 else None


def _draw_diagonal_cell_per_dm(
    ax: Axes,
    mode: str,
    envelopes: dict[str, Any],
    distance_measure: str,
    is_first_col: bool,
    is_last_row: bool,
) -> None:
    """Render the per-distance-measure ECDF envelope on a diagonal subplot."""
    env = envelopes[mode][distance_measure]
    mode_color = label_tables.COLOR_PALETTE.get(mode, "lightgray")
    ax.fill_between(
        env["xvals"],
        env["lo"],
        env["hi"],
        color=mode_color,
        alpha=0.7,
        label="Envelope",
        edgecolor="none",
    )
    ax.plot(env["xvals"], env["mu"], "--", color="dimgray", lw=1, label="Mean")
    ax.set_ylabel("ECDF" if is_first_col else "")
    label_str = label_tables.DISTANCE_MEASURE_LABELS.get(distance_measure, distance_measure)
    ax.set_xlabel(label_str if is_last_row else "")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    sns.despine(ax=ax)


def _draw_diagonal_cell_joint(
    ax: Axes,
    mode: str,
    envelopes: dict[str, Any],
    pairwise_results: dict[str, Any],
    is_first_col: bool,
    is_last_row: bool,
    font_scale: float = 1.0,
) -> None:
    """Render the joint (all-distance-measures overlaid) ECDF on a diagonal subplot."""
    for dm in pairwise_results["distance_measures"]:
        if dm in envelopes.get(mode, {}):
            env = envelopes[mode][dm]
            color = label_tables.COLOR_PALETTE.get(dm, "#808080")
            dm_label = label_tables.DISTANCE_MEASURE_TITLES.get(dm, dm)
            ax.plot(env["xvals"], env["mu"], color=color, lw=1, label=dm_label)
            # also plot the envelope for the joint test if available
            ax.fill_between(
                env["xvals"],
                env["lo"],
                env["hi"],
                color=color,
                alpha=0.3,
                edgecolor="none",
            )
    if is_last_row:
        ax.legend(fontsize=5 * font_scale, frameon=False, loc="lower right")
    ax.set_ylabel("ECDF" if is_first_col else "")
    ax.set_xlabel("Distance (µm)" if is_last_row else "")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    sns.despine(ax=ax)


def _draw_diagonal_cell_label(ax: Axes, mode_label: str, font_scale: float = 1.0) -> None:
    """Render a plain text mode label on a diagonal subplot (fallback when no envelope data)."""
    ax.text(
        0.5,
        0.5,
        mode_label,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12 * font_scale,
        fontweight="bold",
    )
    ax.axis("off")


def _draw_diagonal_cell(
    ax: Axes,
    mode: str,
    envelopes: dict[str, Any],
    distance_measure: str | None,
    pairwise_results: dict[str, Any],
    i: int,
    j: int,
    n: int,
    font_scale: float = 1.0,
) -> None:
    """Dispatch to the correct diagonal-cell renderer and set the subplot title."""
    mode_label = label_tables.MODE_LABELS.get(mode, mode) or mode
    is_first_col = j == 0
    is_last_row = i == n - 1

    if distance_measure is not None and distance_measure in envelopes.get(mode, {}):
        _draw_diagonal_cell_per_dm(ax, mode, envelopes, distance_measure, is_first_col, is_last_row)
    elif distance_measure is None:
        _draw_diagonal_cell_joint(
            ax, mode, envelopes, pairwise_results, is_first_col, is_last_row, font_scale
        )
    else:
        _draw_diagonal_cell_label(ax, mode_label, font_scale)

    ax.set_title(mode_label, fontsize=9 * font_scale, fontweight="bold")


def _draw_offdiagonal_cell(
    ax: Axes,
    pair: tuple[str, str],
    result_dict: dict[tuple[str, str], Any],
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    font_scale: float = 1.0,
) -> None:
    """Render an annotated heatmap cell for an off-diagonal subplot."""
    if pair in result_dict:
        val = result_dict[pair]["rejection_fraction"]
        positive_rejections = result_dict[pair].get("rejection_fraction_positive", 0)
        negative_rejections = result_dict[pair].get("rejection_fraction_negative", 0)

        rgba = (*cmap(norm(val))[:3], 0.7)
        ax.set_facecolor(rgba)

        # Adaptive text colour for readability
        r_c, g_c, b_c, _ = rgba
        luminance = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
        text_color = "white" if luminance < 0.5 else "black"

        ax.text(
            0.5,
            0.6,
            f"{val:.2f}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10 * font_scale,
            fontweight="bold",
            color=text_color,
        )
        ax.text(
            0.5,
            0.25,
            f"+:{positive_rejections:.2f}, -:{negative_rejections:.2f}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8 * font_scale,
            color=text_color,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "N/A",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10 * font_scale,
            color="gray",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_pairwise_envelope_matrix(
    pairwise_results: dict[str, Any],
    distance_measure: str | None = None,
    figure_size: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    cmap_name: str = "Reds",
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: Literal["svg", "png", "pdf"] = "pdf",
) -> tuple[Figure, np.ndarray]:
    """
    Plot NxN matrix summarizing pairwise Monte Carlo envelope test results.

    Rows correspond to the *test mode* (cells being evaluated) and columns
    to the *reference mode* (whose envelope is used).  Diagonal cells show
    the ECDF envelope for that mode; off-diagonal cells show an annotated
    heatmap of the BH-corrected rejection fraction.

    Parameters
    ----------
    pairwise_results
        Output dictionary from :func:`pairwise_envelope_test`
    distance_measure
        Specific distance measure to plot.  If ``None``, uses the joint
        (all-distance-measures-concatenated) test results.
    figure_size
        Figure size ``(width, height)`` in inches; auto-scaled if not given
    font_scale
        Multiplicative scale factor applied to all font sizes.  Values > 1
        enlarge text (useful for large figures); values < 1 shrink it.
        Defaults to ``1.0`` (base sizes: title 9 pt, labels 8--10 pt).
    cmap_name
        Matplotlib colormap name for heatmap cells
    figures_dir
        Directory to save figure; figure is not saved when ``None``
    suffix
        Suffix appended to the saved filename
    save_format
        File format for saving

    Returns
    -------
    :
        Tuple of (Figure, 2-D ndarray of Axes)
    """
    modes = pairwise_results["packing_modes"]
    envelopes = pairwise_results["envelopes"]
    n = len(modes)

    if distance_measure is not None:
        result_dict = pairwise_results["per_distance_measure"].get(distance_measure, {})
    else:
        result_dict = pairwise_results["joint"]

    if figure_size is None:
        figure_size = (n, n)

    fig, axs = plt.subplots(n, n, figsize=figure_size, dpi=300, squeeze=False)

    base_cmap = plt.get_cmap(cmap_name)
    cmap_colors = base_cmap(np.linspace(0, 1, 256))
    cmap_colors[:, 3] = 0.7
    cmap = LinearSegmentedColormap.from_list(f"{cmap_name}_alpha07", cmap_colors)
    norm = Normalize(vmin=0, vmax=1)

    diag_xlim = _compute_diag_xlim(modes, envelopes, distance_measure, pairwise_results)
    diagonal_axes: list[Axes] = []

    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            mode_i = modes[i]
            mode_j = modes[j]

            if i == j:
                _draw_diagonal_cell(
                    ax, mode_i, envelopes, distance_measure, pairwise_results, i, j, n, font_scale
                )
                diagonal_axes.append(ax)
            else:
                _draw_offdiagonal_cell(ax, (mode_i, mode_j), result_dict, cmap, norm, font_scale)

            # Row label on left column (skip diagonal to avoid redundancy)
            if j == 0 and i != 0:
                row_label = label_tables.MODE_LABELS.get(mode_i, mode_i) or mode_i
                ax.set_ylabel(row_label, fontsize=9 * font_scale)

    # Apply shared x-axis limits to all diagonal plots that have data
    if diag_xlim is not None:
        for diag_ax in diagonal_axes:
            if diag_ax.get_visible() and diag_ax.axison:
                diag_ax.set_xlim(diag_xlim)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes((1.02, 0.15, 0.02, 0.7))
    fig.colorbar(sm, cax=cbar_ax, label="Rejection fraction")

    # Axis description labels
    fig.text(
        0.45,
        -0.01,
        "\u2190 Reference condition (envelope source) \u2192",
        ha="center",
        fontsize=8 * font_scale,
    )
    fig.text(
        -0.01,
        0.45,
        "\u2190 Test condition \u2192",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8 * font_scale,
    )

    fig.tight_layout()
    plt.show()

    if figures_dir is not None:
        dm_label = distance_measure or "joint"
        filepath = figures_dir / f"pairwise_envelope_matrix_{dm_label}{suffix}.{save_format}"
        fig.savefig(filepath, format=save_format, bbox_inches="tight", transparent=True)
        logger.info("Saved pairwise envelope matrix to %s", filepath)

    return fig, axs


def plot_per_dm_rejection_bars(
    pairwise_results: dict[str, Any],
    reference_mode: str,
    joint_test: bool = False,
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: Literal["svg", "png", "pdf"] = "pdf",
) -> tuple[Figure, Axes]:
    """
    Plot per-distance-measure rejection rates against a fixed reference mode.

    Creates a single subplot with grouped horizontal bars.  Each y-axis group
    represents one *test* packing mode (``reference_mode`` is excluded).

    When ``joint_test=False`` (default): within each group there is one bar per
    distance measure, colored by distance measure.  Each bar is stacked with
    the positive rejection fraction (obs > sim mean) on the left and the
    negative rejection fraction (obs < sim mean) on the right.

    When ``joint_test=True``: a single bar per mode is plotted, colored by
    the mode color, using the joint (all-distance-measures combined) test
    rejection fractions.

    Parameters
    ----------
    pairwise_results
        Output dictionary from :func:`~cellpack_analysis.lib.stats.pairwise_envelope_test`.
    reference_mode
        The packing mode whose envelope was used as the reference.  This mode
        is excluded from the y-axis.
    joint_test
        If ``True``, plot a single bar per mode colored by the mode color using the
        joint (all-distance-measures combined) test rejection fractions instead of
        per-distance-measure grouped bars.
    figsize
        Figure size ``(width, height)`` in inches; auto-scaled if not given.
    font_scale
        Multiplicative scale factor applied to all font sizes.
    figures_dir
        Directory to save figure; figure is not saved when ``None``.
    suffix
        Suffix appended to the saved filename.
    save_format
        File format for saving.

    Returns
    -------
    :
        Tuple of (Figure, Axes).
    """
    distance_measures = pairwise_results["distance_measures"]
    packing_modes = pairwise_results["packing_modes"][::-1]  # reverse for better y-axis ordering
    per_dm = pairwise_results["per_distance_measure"]
    joint = pairwise_results["joint"]

    # Exclude the reference mode itself — no self-comparison entry exists in the dict
    test_modes = [m for m in packing_modes if m != reference_mode]
    n_modes = len(test_modes)
    n_dm = len(distance_measures)

    if figsize is None:
        figsize = (
            (5.0, max(2.0, 0.5 * n_modes)) if joint_test else (5.0, max(2.0, 0.6 * n_modes * n_dm))
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    if joint_test:
        bar_height = 0.6
        for mode_idx, mode in enumerate(test_modes):
            entry = joint.get((mode, reference_mode), {})
            pos_val = entry.get("rejection_fraction_positive", 0.0)
            neg_val = entry.get("rejection_fraction_negative", 0.0)

            mode_color = label_tables.COLOR_PALETTE.get(
                mode, label_tables.COLOR_PALETTE.get("random", "#808080")
            )
            mode_neg_color = label_tables.adjust_color_saturation(mode_color, saturation=0.3)

            ax.barh(mode_idx, pos_val, height=bar_height, color=mode_color, alpha=0.9)
            ax.barh(
                mode_idx, neg_val, height=bar_height, left=pos_val, color=mode_neg_color, alpha=0.9
            )

        legend_handles = [
            mpatches.Patch(color="gray", alpha=0.9, label="Test distances greater"),
            mpatches.Patch(color="gray", alpha=0.4, label="Test distances smaller"),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=7 * font_scale,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )
    else:
        bar_height = 0.8 / n_dm

        for mode_idx, mode in enumerate(test_modes):
            for dm_idx, dm in enumerate(distance_measures):
                dm_dict = per_dm.get(dm, {})
                pos_val = dm_dict.get((mode, reference_mode), {}).get(
                    "rejection_fraction_positive", 0.0
                )
                neg_val = dm_dict.get((mode, reference_mode), {}).get(
                    "rejection_fraction_negative", 0.0
                )

                dm_color = label_tables.COLOR_PALETTE.get(dm, "#808080")
                dm_neg_color = label_tables.adjust_color_saturation(dm_color, saturation=0.3)
                dm_label = label_tables.DISTANCE_MEASURE_TITLES.get(dm, dm)

                y = mode_idx + (dm_idx - n_dm / 2 + 0.5) * bar_height

                ax.barh(
                    y,
                    pos_val,
                    height=bar_height * 0.85,
                    color=dm_color,
                    alpha=0.9,
                    label=dm_label if mode_idx == 0 else "",
                )
                ax.barh(
                    y,
                    neg_val,
                    height=bar_height * 0.85,
                    left=pos_val,
                    color=dm_neg_color,
                    alpha=0.9,
                )

        dm_handles = [
            mpatches.Patch(
                color=label_tables.COLOR_PALETTE.get(dm, "#808080"),
                alpha=0.9,
                label=label_tables.DISTANCE_MEASURE_TITLES.get(dm, dm),
            )
            for dm in distance_measures
        ]
        sign_handles = [
            mpatches.Patch(color="gray", alpha=0.9, label="Test distances greater"),
            mpatches.Patch(color="gray", alpha=0.4, label="Test distances smaller"),
        ]
        ax.legend(
            handles=dm_handles + sign_handles,
            frameon=False,
            fontsize=7 * font_scale,
            bbox_to_anchor=(1, 1),
            loc="upper left",
            title="Distance measure",
            title_fontsize=7 * font_scale,
        )

    ax.set_xlim(0, 1)
    ax.set_yticks(np.arange(n_modes))
    ax.set_yticklabels(
        [label_tables.MODE_LABELS.get(tm, tm) or tm for tm in test_modes],
        fontsize=8 * font_scale,
    )
    ax.set_xlabel("Rejection fraction", fontsize=8 * font_scale)
    ref_label = label_tables.MODE_LABELS.get(reference_mode, reference_mode) or reference_mode
    ax.set_title(f"Rejection rates — reference: {ref_label}", fontsize=10 * font_scale)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()

    if figures_dir is not None:
        figname = "joint" if joint_test else "per_dm"
        filepath = (
            figures_dir / f"{figname}_rejection_bars_ref_{reference_mode}{suffix}.{save_format}"
        )
        fig.savefig(filepath, format=save_format, bbox_inches="tight", transparent=True)
        logger.info("Saved per-DM rejection bars to %s", filepath)

    return fig, ax


def plot_per_dm_envelopes_overlaid(
    pairwise_results: dict[str, Any],
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: Literal["svg", "png", "pdf"] = "pdf",
) -> tuple[Figure, np.ndarray]:
    """
    Overlay Monte Carlo ECDF envelopes for all modes on per-distance-measure subplots.

    Creates a 1 x N_dm row of subplots.  Each subplot overlays the simulated
    ECDF envelope (``fill_between`` with the mean line) for every packing mode,
    coloured by ``label_tables.COLOR_PALETTE``.

    Parameters
    ----------
    pairwise_results
        Output dictionary from :func:`~cellpack_analysis.lib.stats.pairwise_envelope_test`.
    figsize
        Figure size ``(width, height)`` in inches; auto-scaled if not given.
    font_scale
        Multiplicative scale factor applied to all font sizes.
    figures_dir
        Directory to save figure; figure is not saved when ``None``.
    suffix
        Suffix appended to the saved filename.
    save_format
        File format for saving.

    Returns
    -------
    :
        Tuple of (Figure, 2-D ndarray of Axes).
    """
    distance_measures = pairwise_results["distance_measures"]
    packing_modes = pairwise_results["packing_modes"]
    envelopes = pairwise_results["envelopes"]
    n_dm = len(distance_measures)

    if figsize is None:
        figsize = (4.0 * n_dm, 3.0)

    fig, axs = plt.subplots(1, n_dm, figsize=figsize, dpi=150, squeeze=False)

    for col, dm in enumerate(distance_measures):
        ax = axs[0, col]
        for mode in packing_modes:
            env = envelopes.get(mode, {}).get(dm)
            if env is None:
                continue
            mode_color = label_tables.COLOR_PALETTE.get(
                mode, label_tables.COLOR_PALETTE.get("random", "gray")
            )
            mode_label = label_tables.MODE_LABELS.get(mode, mode) or mode
            ax.fill_between(
                env["xvals"],
                env["lo"],
                env["hi"],
                color=mode_color,
                alpha=0.2,
                edgecolor="none",
            )
            ax.plot(env["xvals"], env["mu"], color=mode_color, lw=1.2, label=mode_label)

        ax.set_ylabel("ECDF" if col == 0 else "", fontsize=8 * font_scale)
        ax.set_title(
            label_tables.DISTANCE_MEASURE_TITLES.get(dm, dm),
            fontsize=9 * font_scale,
        )
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        sns.despine(ax=ax)

    fig.supxlabel("Distance (µm)", fontsize=8 * font_scale)
    # Single shared legend on the last subplot
    axs[0, -1].legend(
        frameon=False,
        fontsize=7 * font_scale,
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )

    fig.tight_layout()
    plt.show()

    if figures_dir is not None:
        filepath = figures_dir / f"per_dm_envelopes_overlaid{suffix}.{save_format}"
        fig.savefig(filepath, format=save_format, bbox_inches="tight", transparent=True)
        logger.info("Saved per-DM envelope overlay to %s", filepath)

    return fig, axs


def _draw_emd_violin_cell(
    ax: Axes,
    sub: pd.DataFrame,
    color: str,
    emd_xmax: float,
    ylabel: str,
    show_xlabel: bool,
    title: str | None = None,
    font_scale: float = 1.0,
) -> None:
    """Draw a violinplot of EMD values on *ax* (diagonal or upper-triangle cell)."""
    if len(sub) > 0:
        sns.violinplot(
            data=sub,
            x="emd",
            ax=ax,
            color=color,
            fill=True,
            alpha=0.4,
            linewidth=0.8,
            cut=0,
            orient="h",
        )
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=6 * font_scale)
    ax.set_xlim(0, emd_xmax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("EMD" if show_xlabel else "")
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    sns.despine(ax=ax)


def _draw_lower_distance_cell(
    ax: Axes,
    mode_i: str,
    mode_j: str,
    color_i: str,
    color_j: str,
    r_grid: np.ndarray,
    mode_mean: dict[str, np.ndarray],
    mode_lo: dict[str, np.ndarray],
    mode_hi: dict[str, np.ndarray],
    distance_limits: dict[str, tuple[float, float]] | None,
    distance_measure: str,
    distance_label: str,
    is_last_row: bool,
    is_first_col: bool,
) -> None:
    """Draw overlaid distance PDF curves for two modes on a lower-triangle cell."""
    for mode, color in [(mode_j, color_j), (mode_i, color_i)]:
        mode_lbl = label_tables.MODE_LABELS.get(mode, mode)
        ax.fill_between(
            r_grid, mode_lo[mode], mode_hi[mode], color=color, alpha=0.12, edgecolor="none"
        )
        ax.plot(r_grid, mode_mean[mode], color=color, linewidth=0.8, label=mode_lbl)

    if distance_limits is not None and distance_measure in distance_limits:
        ax.set_xlim(distance_limits[distance_measure])
    else:
        ax.set_xlim(r_grid[0], r_grid[-1])

    y_max = max(mode_hi[mode_i].max(), mode_hi[mode_j].max()) * 1.05
    ax.set_ylim(0, y_max)
    ax.set_ylabel("PDF" if is_first_col else "")
    ax.set_xlabel(distance_label if is_last_row else "")
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(2))
    sns.despine(ax=ax)


def _draw_lower_occupancy_cell(
    ax: Axes,
    mode_i: str,
    mode_j: str,
    color_i: str,
    color_j: str,
    binned_occupancy_dict: dict[str, dict[str, Any]],
    distance_label: str,
    xlim: float | None,
    ylim: float | None,
    is_last_row: bool,
    is_first_col: bool,
    font_scale: float = 1.0,
) -> None:
    """Draw overlaid occupancy ratio curves for two modes on a lower-triangle cell."""
    xvals: np.ndarray = np.array([])
    for mode, color in [(mode_j, color_j), (mode_i, color_i)]:
        mode_lbl = label_tables.MODE_LABELS.get(mode, mode)
        combined = binned_occupancy_dict.get(mode, {}).get("combined", {})
        if not combined:
            continue
        xvals = combined["xvals"]
        mean_occ = combined["occupancy"]

        if "envelope_lo" in combined and "envelope_hi" in combined:
            lo = combined["envelope_lo"]
            hi = combined["envelope_hi"]
        elif "std_occupancy" in combined:
            lo = mean_occ - combined["std_occupancy"]
            hi = mean_occ + combined["std_occupancy"]
        else:
            lo = hi = None

        if lo is not None and hi is not None:
            ax.fill_between(xvals, lo, hi, color=color, alpha=0.12, edgecolor="none")
        ax.plot(xvals, mean_occ, color=color, linewidth=0.8, label=mode_lbl)
        ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")

    _xlim = xlim if xlim is not None else (float(xvals[-1]) if len(xvals) else 1.0)
    ax.set_xlim(0, _xlim)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_ylabel("Occ. Ratio" if is_first_col else "")
    ax.set_xlabel(distance_label if is_last_row else "")
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.legend(fontsize=5 * font_scale, frameon=False, loc="upper right")
    sns.despine(ax=ax)


def plot_pairwise_emd_matrix(
    df_emd: pd.DataFrame,
    packing_modes: list[str],
    distance_measure: str,
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] | None = None,
    binned_occupancy_dict: dict[str, dict[str, Any]] | None = None,
    distance_pdf_dict: dict[str, dict[str, dict[str, np.ndarray]]] | None = None,
    normalization: str | None = None,
    distance_limits: dict[str, tuple[float, float]] | None = None,
    bin_width: float | dict[str, float] = 0.5,
    minimum_distance: float | None = 0,
    xlim: float | None = None,
    ylim: float | None = None,
    envelope_alpha: float = 0.05,
    figure_size: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: Literal["svg", "png", "pdf"] = "pdf",
) -> tuple[Figure, np.ndarray]:
    """Plot an NxN matrix of pairwise EMD comparisons.

    Accepts either raw distance arrays (via *all_distance_dict*), pre-computed
    distance PDFs (via *distance_pdf_dict*), or pre-binned occupancy data
    (via *binned_occupancy_dict*) — exactly one must be provided.

    * **Lower triangle** (row > col): distance PDF overlay when using
      *all_distance_dict* or *distance_pdf_dict*; occupancy ratio curves when
      using *binned_occupancy_dict*.
    * **Diagonal** (row == col): intra-mode EMD violinplot.
    * **Upper triangle** (row < col): cross-mode EMD violinplot.

    Parameters
    ----------
    df_emd
        DataFrame with columns ``distance_measure``, ``packing_mode_1``,
        ``packing_mode_2``, and ``emd``.
    packing_modes
        Ordered list of packing modes forming the matrix axes.
    distance_measure
        The distance measure to plot.
    all_distance_dict
        ``{distance_measure: {mode: {cell_id: {seed: distances}}}}``.
        Mutually exclusive with *binned_occupancy_dict* and *distance_pdf_dict*.
    binned_occupancy_dict
        ``{mode: {"combined": {"xvals", "occupancy", ...}}}``.
        Mutually exclusive with *all_distance_dict* and *distance_pdf_dict*.
    distance_pdf_dict
        ``{distance_measure: {mode: {"xvals", "mean_pdf", "envelope_lo",
        "envelope_hi"}}}``.  Pre-computed via ``compute_distance_pdfs()``.
        Mutually exclusive with *all_distance_dict* and *binned_occupancy_dict*.
    normalization
        Normalization method applied to distances (used for axis labelling).
    distance_limits
        Optional per-measure ``(lo, hi)`` used to fix axis ranges and bin
        edges (distance mode only).
    bin_width
        Histogram bin width (scalar or per-measure dict; distance mode only).
    minimum_distance
        Minimum distance for filtering invalid values (distance mode only).
    xlim
        Upper x-axis limit for occupancy ratio panels (occupancy mode only).
    ylim
        Upper y-axis limit for occupancy ratio panels (occupancy mode only).
    envelope_alpha
        Significance level for pointwise envelope shading.
    figure_size
        Figure size in inches; auto-scaled from *n* when ``None``.
    font_scale
        Multiplicative scale factor applied to all font sizes.  Values > 1
        enlarge text (useful for large figures); values < 1 shrink it.
        Defaults to ``1.0`` (base size 6 pt).
    figures_dir
        Directory to save the figure; skipped when ``None``.
    suffix
        Suffix appended to the saved filename.
    save_format
        Image format for saving.

    Returns
    -------
    :
        Tuple of ``(Figure, 2-D ndarray of Axes)``.
    """
    provided = sum(
        x is not None for x in (all_distance_dict, binned_occupancy_dict, distance_pdf_dict)
    )
    if provided != 1:
        raise ValueError(
            "Exactly one of all_distance_dict, binned_occupancy_dict, or "
            "distance_pdf_dict must be provided."
        )

    use_occupancy = binned_occupancy_dict is not None
    n = len(packing_modes)
    if figure_size is None:
        figure_size = (n * 1.8, n * 1.6)

    plt.rcParams.update({"font.size": 6 * font_scale})
    fig, axs = plt.subplots(n, n, figsize=figure_size, dpi=300, squeeze=False)

    dm_emd = df_emd.query(f"distance_measure == '{distance_measure}'")
    emd_xmax = dm_emd["emd"].quantile(0.99) * 1.1 if len(dm_emd) > 0 else 1.0
    distance_label = get_distance_label(distance_measure, normalization)

    # ── Pre-compute per-mode distance curves (distance mode only) ────────
    r_grid: np.ndarray = np.array([])
    mode_mean: dict[str, np.ndarray] = {}
    mode_lo: dict[str, np.ndarray] = {}
    mode_hi: dict[str, np.ndarray] = {}
    if not use_occupancy:
        if distance_pdf_dict is not None:
            # Use pre-computed PDFs
            for mode in packing_modes:
                pdf_data = distance_pdf_dict[distance_measure][mode]
                r_grid = pdf_data["xvals"]
                mode_mean[mode] = pdf_data["mean_pdf"]
                mode_lo[mode] = pdf_data["envelope_lo"]
                mode_hi[mode] = pdf_data["envelope_hi"]
        else:
            assert all_distance_dict is not None
            from cellpack_analysis.lib.distance import compute_distance_pdfs

            _pdf_dict = compute_distance_pdfs(
                all_distance_dict,
                [distance_measure],
                packing_modes,
                method="kde",
                bin_width=bin_width,
                distance_limits=distance_limits,
                minimum_distance=minimum_distance,
                envelope_alpha=envelope_alpha,
            )
            for mode in packing_modes:
                pdf_data = _pdf_dict[distance_measure][mode]
                r_grid = pdf_data["xvals"]
                mode_mean[mode] = pdf_data["mean_pdf"]
                mode_lo[mode] = pdf_data["envelope_lo"]
                mode_hi[mode] = pdf_data["envelope_hi"]

    lower_ylabel = "Occupancy Ratio" if use_occupancy else "PDF"

    # ── Fill matrix ──────────────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            mode_i = packing_modes[i]
            mode_j = packing_modes[j]
            color_i = label_tables.COLOR_PALETTE.get(mode_i, "gray")
            color_j = label_tables.COLOR_PALETTE.get(mode_j, "gray")
            label_i = label_tables.MODE_LABELS.get(mode_i, mode_i)
            label_j = label_tables.MODE_LABELS.get(mode_j, mode_j)

            if i == j:
                # ── Diagonal: intra-mode EMD violinplot ──────────────────
                sub = dm_emd.query(f"packing_mode_1 == '{mode_i}' and packing_mode_2 == '{mode_i}'")
                _draw_emd_violin_cell(
                    ax,
                    sub,
                    color_i,
                    emd_xmax,
                    ylabel="Density" if j == 0 else "",
                    show_xlabel=(i == n - 1),
                    title=label_i,
                    font_scale=font_scale,
                )

            elif i < j:
                # ── Upper triangle: cross-mode EMD violinplot ────────────
                sub = dm_emd.query(
                    "(packing_mode_1 == @mode_i and packing_mode_2 == @mode_j) or "
                    "(packing_mode_1 == @mode_j and packing_mode_2 == @mode_i)"
                )
                _draw_emd_violin_cell(
                    ax,
                    sub,
                    color_j,
                    emd_xmax,
                    ylabel="",
                    show_xlabel=(i == n - 1),
                    title=None,
                    font_scale=font_scale,
                )

            else:
                # ── Lower triangle: data curves ──────────────────────────
                if use_occupancy:
                    assert binned_occupancy_dict is not None
                    _draw_lower_occupancy_cell(
                        ax,
                        mode_i,
                        mode_j,
                        color_i,
                        color_j,
                        binned_occupancy_dict,
                        distance_label,
                        xlim,
                        ylim,
                        is_last_row=(i == n - 1),
                        is_first_col=(j == 0),
                        font_scale=font_scale,
                    )
                else:
                    _draw_lower_distance_cell(
                        ax,
                        mode_i,
                        mode_j,
                        color_i,
                        color_j,
                        r_grid,
                        mode_mean,
                        mode_lo,
                        mode_hi,
                        distance_limits,
                        distance_measure,
                        distance_label,
                        is_last_row=(i == n - 1),
                        is_first_col=(j == 0),
                    )

            # Column header on the first row
            if i == 0 and j != 0:
                ax.set_title(label_j, fontsize=6 * font_scale)

            # Row labels on leftmost column
            if j == 0 and i != 0:
                ax.set_ylabel(f"{label_i}\n{lower_ylabel if i > j else 'Density'}")

    fig.tight_layout()
    plt.show()

    if figures_dir is not None:
        prefix = "pairwise_occupancy_emd_matrix" if use_occupancy else "pairwise_emd_matrix"
        filepath = figures_dir / f"{prefix}_{distance_measure}{suffix}.{save_format}"
        fig.savefig(filepath, format=save_format, dpi=300, bbox_inches="tight", transparent=True)
        logger.info("Saved pairwise EMD matrix to %s", filepath)

    return fig, axs


def plot_rule_interpolation_fit(
    fit_result: Any,
    occupancy_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str],
    baseline_mode: str,
    distance_measure: str = "nucleus",
    plot_type: str = "joint",
    figures_dir: Path | None = None,
    xlim: float | None = None,
    ylim: float | None = None,
    suffix: str = "",
    save_format: Literal["svg", "png", "pdf"] = "pdf",
    fig_params: dict[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    """Plot all mode occupancy curves with a NNLS rule-interpolation reconstruction overlaid.

    Renders the same combined-occupancy curves and pointwise envelopes as
    :func:`plot_binned_occupancy_ratio`, then adds a dash-dot overlay for the
    NNLS reconstruction stored in *fit_result*.

    Parameters
    ----------
    fit_result
        :class:`~cellpack_analysis.analysis.rule_interpolation.FitResult` returned
        by :func:`~cellpack_analysis.analysis.rule_interpolation.fit_rule_interpolation`.
    occupancy_dict
        ``{distance_measure: {mode: {"individual": ..., "combined": {...}}}}``
        (same structure passed to ``fit_rule_interpolation``).
    channel_map
        Mapping from packing modes to structure IDs (used to determine draw order).
    baseline_mode
        Baseline packing mode key — used for the reconstruction overlay colour.
    distance_measure
        Which distance measure to plot.
    plot_type
        Scope of the reconstruction to overlay: ``"individual"`` (per-dm NNLS)
        or ``"joint"`` (stacked-across-dm NNLS).
    figures_dir
        Directory to save the figure; skipped when ``None``.
    xlim
        Upper x-axis limit.
    ylim
        Upper y-axis limit.
    suffix
        Suffix appended to the saved filename.
    save_format
        File format for saving.
    fig_params
        Optional matplotlib ``Figure`` keyword arguments.

    Returns
    -------
    :
        Tuple of ``(Figure, Axes)``.
    """
    plt.rcParams.update({"font.size": 8})
    _fig_params: dict[str, Any] = {"dpi": 300, "figsize": (3.5, 2.5)}
    if fig_params is not None:
        _fig_params.update(fig_params)
    fig, ax = plt.subplots(**_fig_params)

    dm_occ = occupancy_dict[distance_measure]

    # ── Draw all mode combined occupancy curves + pointwise envelopes ────────
    for mode in channel_map.keys():
        combined = dm_occ[mode]["combined"]
        xvals = combined["xvals"]
        mean_occ = np.nan_to_num(combined["occupancy"])
        color = label_tables.COLOR_PALETTE.get(mode, "gray")
        label = label_tables.MODE_LABELS.get(mode, mode)

        ax.plot(xvals, mean_occ, color=color, linewidth=2.0, label=label, zorder=2)

        if "envelope_lo" in combined and "envelope_hi" in combined:
            lo: np.ndarray | None = np.nan_to_num(combined["envelope_lo"])
            hi: np.ndarray | None = np.nan_to_num(combined["envelope_hi"])
        elif "std_occupancy" in combined:
            lo = mean_occ - np.nan_to_num(combined["std_occupancy"])
            hi = mean_occ + np.nan_to_num(combined["std_occupancy"])
        else:
            lo = hi = None

        if lo is not None and hi is not None:
            ax.fill_between(
                xvals,
                np.nan_to_num(lo),
                np.nan_to_num(hi),
                alpha=0.15,
                color=color,
                linewidth=0,
                label="_nolegend_",
                zorder=0,
            )

    # ── Overlay NNLS reconstruction ──────────────────────────────────────────
    recon: np.ndarray = fit_result.reconstructed_occupancy[distance_measure][plot_type]
    common_xvals: np.ndarray = dm_occ[baseline_mode]["combined"]["xvals"]

    mse: float = (
        fit_result.train_mse_individual[distance_measure]
        if plot_type == "individual"
        else fit_result.train_mse_joint[distance_measure]
    )
    relative_contribs: dict[str, float] = (
        fit_result.relative_contributions_individual[distance_measure]
        if plot_type == "individual"
        else fit_result.relative_contributions_joint
    )

    ax.plot(
        common_xvals,
        recon,
        color=label_tables.COLOR_PALETTE.get(baseline_mode, "gray"),
        linewidth=2,
        linestyle="-.",
        zorder=3,
        label=f"Interpolated ({plot_type})",
    )

    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    if xlim is not None:
        ax.set_xlim((0, xlim))
    else:
        ax.set_xlim((0, ax.get_xlim()[1]))
    if ylim is not None:
        ax.set_ylim((0, ylim))
    ax.axhline(1, linewidth=1, color="gray", linestyle="--", zorder=1)
    ax.set_xlabel(get_distance_label(distance_measure))
    ax.set_ylabel("Occupancy Ratio")
    ax.set_title(f"{plot_type.capitalize()}, MSE: {mse:.4f}")

    contrib_text = "\n".join(
        f"{label_tables.MODE_LABELS.get(k, k)}: {v:.0%}" for k, v in relative_contribs.items()
    )
    ax.text(
        0.95,
        0.95,
        contrib_text,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=7,
    )

    sns.despine(fig=fig)
    fig.tight_layout()

    if figures_dir is not None:
        fig.savefig(
            figures_dir
            / f"{distance_measure}_{plot_type}_rule_interpolation_fit{suffix}.{save_format}",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.show()

    return fig, ax


def plot_cv_mse_summary(
    cv_df: pd.DataFrame,
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: Literal["svg", "png", "pdf"] = "pdf",
    fig_params: dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot cross-validation MSE for each scope and distance measure.

    Draws a boxplot panel per *scope* (``"individual"``, ``"joint"``), with
    ``distance_measure`` on the x-axis and train/test split as the hue.

    Parameters
    ----------
    cv_df
        DataFrame returned by
        :func:`~cellpack_analysis.analysis.rule_interpolation.summarize_cv_results`.
        Expected columns: ``fold_idx``, ``scope``, ``distance_measure``, ``split``,
        ``mse``.
    figures_dir
        Directory to save the figure; skipped when ``None``.
    suffix
        Suffix appended to the saved filename.
    save_format
        File format for saving.
    fig_params
        Optional matplotlib ``Figure`` keyword arguments.

    Returns
    -------
    :
        Tuple of ``(Figure, 1-D ndarray of Axes)`` — one axis per scope.
    """
    scopes = list(cv_df["scope"].unique())
    n_scopes = len(scopes)
    plt.rcParams.update({"font.size": 8})
    _fig_params: dict[str, Any] = {"dpi": 300, "figsize": (max(3.0, n_scopes * 2.5), 2.5)}
    if fig_params is not None:
        _fig_params.update(fig_params)
    fig, axs = plt.subplots(1, n_scopes, squeeze=False, **_fig_params)

    for ax, scope in zip(axs[0], scopes, strict=False):
        scope_df = cv_df[cv_df["scope"] == scope]
        sns.boxplot(
            data=scope_df,
            x="distance_measure",
            y="mse",
            hue="split",
            ax=ax,
            palette={"train": "#4e9af1", "test": "#f97306"},
            linewidth=0.8,
            fliersize=2,
        )
        ax.set_title(scope.capitalize())
        ax.set_xlabel("Distance measure")
        ax.set_ylabel("MSE")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        sns.despine(ax=ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("")

    fig.suptitle("Cross-validation MSE", fontsize=9)
    fig.tight_layout()

    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"rule_interpolation_cv_mse_summary{suffix}.{save_format}",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.show()

    return fig, axs[0]


def plot_cv_coefficient_stability(
    cv_result: CVResult,
    figures_dir: Path | None = None,
    suffix: str = "",
    save_format: Literal["svg", "png", "pdf"] = "pdf",
    fig_params: dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot coefficient stability across cross-validation folds.

    Draws one bar-chart panel for the joint fit and one panel per distance
    measure for the individual fits.  Bar heights are the mean coefficient
    across folds; error bars show the standard deviation.

    Parameters
    ----------
    cv_result
        :class:`~cellpack_analysis.analysis.rule_interpolation.CVResult` returned
        by :func:`~cellpack_analysis.analysis.rule_interpolation.run_rule_interpolation_cv`.
    figures_dir
        Directory to save the figure; skipped when ``None``.
    suffix
        Suffix appended to the saved filename.
    save_format
        File format for saving.
    fig_params
        Optional matplotlib ``Figure`` keyword arguments.

    Returns
    -------
    :
        Tuple of ``(Figure, 1-D ndarray of Axes)`` — joint panel first, then
        one panel per distance measure.
    """
    distance_measures = list(cv_result.aggregated_coefficients_individual.keys())
    packing_modes = list(cv_result.aggregated_coefficients_joint.keys())
    n_subplots = 1 + len(distance_measures)

    plt.rcParams.update({"font.size": 8})
    _fig_params: dict[str, Any] = {"dpi": 300, "figsize": (n_subplots * 2.5, 2.5)}
    if fig_params is not None:
        _fig_params.update(fig_params)
    fig, axs = plt.subplots(1, n_subplots, squeeze=False, **_fig_params)

    x_pos = np.arange(len(packing_modes))
    colors = [label_tables.COLOR_PALETTE.get(mode, "gray") for mode in packing_modes]
    mode_labels = [label_tables.MODE_LABELS.get(mode, mode) for mode in packing_modes]

    def _draw_coeff_bars(ax: Axes, means: list[float], stds: list[float], title: str) -> None:
        ax.bar(
            x_pos,
            means,
            yerr=stds,
            color=colors,
            capsize=4,
            linewidth=0.8,
            error_kw={"linewidth": 1, "ecolor": "black"},
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(mode_labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Coefficient (mean ± std)")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        sns.despine(ax=ax)

    # Joint panel
    joint_means = [cv_result.aggregated_coefficients_joint[m][0] for m in packing_modes]
    joint_stds = [cv_result.aggregated_coefficients_joint[m][1] for m in packing_modes]
    _draw_coeff_bars(axs[0, 0], joint_means, joint_stds, "Joint")

    # Per-dm panels
    for i, dm in enumerate(distance_measures):
        dm_means = [cv_result.aggregated_coefficients_individual[dm][m][0] for m in packing_modes]
        dm_stds = [cv_result.aggregated_coefficients_individual[dm][m][1] for m in packing_modes]
        dm_label = label_tables.DISTANCE_MEASURE_LABELS.get(dm, dm)
        _draw_coeff_bars(axs[0, 1 + i], dm_means, dm_stds, dm_label)

    fig.suptitle("Coefficient stability across CV folds", fontsize=9)
    fig.tight_layout()

    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"rule_interpolation_cv_coefficients{suffix}.{save_format}",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.show()

    return fig, axs[0]
