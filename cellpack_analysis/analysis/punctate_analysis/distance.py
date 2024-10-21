import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from rtree.exceptions import RTreeError
from scipy import integrate
from scipy.spatial.distance import cdist, squareform
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm
from trimesh import proximity

from cellpack_analysis.analysis.punctate_analysis.stats_functions import ripley_k
from cellpack_analysis.lib.default_values import PIX_SIZE
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_structure_stats_dataframe import (
    get_structure_stats_dataframe,
)
from cellpack_analysis.lib.label_tables import (
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
    MODE_LABELS,
    NORMALIZATION_LABELS,
    STATIC_SHAPE_MODES,
)

log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 16})

PROJECT_ROOT = get_project_root()


def plot_cell_diameter_distribution(mesh_information_dict):
    """
    Plots the distribution of cell and nucleus diameters.

    This function calculates the cell and nucleus diameters from the mesh information dictionary,
    and then plots the histograms of the diameters. It also displays the mean cell and nucleus
    diameters in the plot title.

    Args:
    ----
        mesh_information_dict (dict): A dictionary containing mesh information.

    Returns:
    -------
        None
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


def calc_scaled_distance_to_nucleus_surface(
    position_list,
    nuc_mesh,
    mem_mesh,
    mem_distances=None,
):
    """
    Calculate the scaled distance of each point in position_list to the nucleus surface.

    Args:
    ----
        position_list (np.ndarray): A list of 3D coordinates of points.
        nuc_mesh (trimesh.Trimesh): A trimesh object representing the nucleus surface.
        mem_mesh (trimesh.Trimesh): A trimesh object representing the membrane surface.

    Returns:
    -------
        np.ndarray: A list of distances of each point to the nucleus surface scaled by
        the distance to the membrane surface.
    """
    if mem_distances is None:
        mem_distances = proximity.signed_distance(mem_mesh, position_list)

    nuc_query = proximity.ProximityQuery(nuc_mesh)

    # closest points in the inner mesh surface
    nuc_surface_positions, _, _ = nuc_query.on_surface(position_list)
    nuc_surface_distances = -nuc_query.signed_distance(position_list)

    # intersecting points on the outer surface
    mem_surface_positions = np.zeros(nuc_surface_positions.shape)
    failed_inds = []
    for ind, (
        position,
        nuc_surface_distance,
        mem_surface_distance,
    ) in enumerate(zip(position_list, nuc_surface_distances, mem_distances)):
        if (
            nuc_surface_distance < 0
            or mem_surface_distance < 0
            or position in nuc_surface_positions
        ):
            mem_surface_positions[ind] = np.nan
            failed_inds.append(ind)
            continue
        try:
            direction = position - nuc_surface_positions[ind]
            intersect_positions, _, _ = mem_mesh.ray.intersects_location(
                ray_origins=[nuc_surface_positions[ind]],
                ray_directions=[direction],
            )
            if len(intersect_positions) > 1:
                intersect_distances = np.linalg.norm(
                    intersect_positions - nuc_surface_positions[ind], axis=1
                )
                min_ind = np.argmin(intersect_distances)
                mem_surface_positions[ind] = intersect_positions[min_ind]
            else:
                mem_surface_positions[ind] = intersect_positions
        except ValueError as e:
            log.error(f"Value error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue
        except RTreeError as e:
            log.error(f"Rtree error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue
        except Exception as e:
            log.error(f"Unexpected error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue

    if len(failed_inds) > 0:
        log.debug(f"Failed {len(failed_inds)} out of {len(position_list)}")

    distance_between_surfaces = np.linalg.norm(
        mem_surface_positions - nuc_surface_positions, axis=1
    )
    scaled_nuc_distances = nuc_surface_distances / distance_between_surfaces
    scaled_nuc_distances[failed_inds] = np.nan
    scaled_nuc_distances[(scaled_nuc_distances < 0) | (scaled_nuc_distances > 1)] = (
        np.nan
    )

    return nuc_surface_distances, scaled_nuc_distances, distance_between_surfaces


def filter_nans_from_distance_distribution_dict(
    distance_distribution_dict: Dict[Any, Any],
) -> Dict[Any, Any]:
    """
    Filters out NaN values from the distance distribution dictionary.

    Parameters
    ----------
    distance_distribution_dict
        Dictionary containing distance distributions.
        Should have the form:
        {
            distance_measure: {
                packing_mode: {
                    seed: np.ndarray of distances
                }
            }
        }

    Returns
    -------
    :
        Filtered distance distribution dictionary.
    """
    for distance_measure, distance_measure_dict in distance_distribution_dict.items():
        for mode, mode_dict in distance_measure_dict.items():
            for seed, distances in mode_dict.items():
                # filter out NaN values
                mode_dict[seed] = distances[~np.isnan(distances)]
                if len(mode_dict[seed]) == 0:
                    log.warning(
                        f"All distances are NaN for {distance_measure}, {mode}, {seed}"
                    )
                    del mode_dict[seed]
                else:
                    log.debug(
                        f"Filtered {np.sum(np.isnan(distances))} NaN values from "
                        f"{distance_measure}, {mode}, {seed}"
                    )
    return distance_distribution_dict


def get_distance_dictionary(
    all_positions,
    distance_measures,
    mesh_information_dict,
    channel_map=None,
    results_dir=None,
    recalculate=False,
):
    """
    Calculate or load distance measures between particles in different modes.

    Parameters
    ----------
    all_positions
        A dictionary containing positions of particles in different packing modes.
    mesh_information_dict
        A dictionary containing mesh information.
    results_dir
        The directory to save or load distance dictionaries.
        Defaults to None.
    recalculate
        Whether to recalculate the distance measures.
        Defaults to False.

    Returns
    -------
    :
        A dictionary containing distance measures between particles in different modes.
    """
    log.info("Starting distance calculations")
    if channel_map is None:
        channel_map = {}
    if not recalculate and results_dir is not None:
        # load saved distance dictionary
        log.info("Loading saved distance dictionaries")
        all_distance_dict = {}
        for distance_measure in distance_measures:
            file_path = results_dir / f"{distance_measure}_distances.dat"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    all_distance_dict[distance_measure] = pickle.load(f)
            else:
                log.warning(f"File not found: {file_path.relative_to(PROJECT_ROOT)}")
        if len(all_distance_dict) == len(distance_measures):
            return all_distance_dict

    all_pairwise_distances = {}  # pairwise distance between particles
    all_nuc_distances = {}  # distance to nucleus surface
    all_nearest_distances = {}  # distance to nearest neighbor
    all_z_distances = {}  # distance from z-axis
    all_scaled_nuc_distances = {}  # scaled distance to nucleus surface
    all_membrane_distances = {}  # distance to membrane surface
    for mode, position_dict in all_positions.items():
        log.info(f"Calculating distances for mode: {MODE_LABELS.get(mode, mode)}")

        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})

        all_pairwise_distances[mode] = {}
        all_nuc_distances[mode] = {}
        all_nearest_distances[mode] = {}
        all_z_distances[mode] = {}
        all_scaled_nuc_distances[mode] = {}
        all_membrane_distances[mode] = {}

        for seed, positions in tqdm(position_dict.items()):
            # seed_to_use = seed.split("_")[0]
            if mode not in STATIC_SHAPE_MODES:
                seed_to_use = seed.split("_")[0]
                shape_key = seed_to_use
            else:
                seed_to_use = str(seed)
                shape_key = seed.split("_")[0]

            all_distances = cdist(positions, positions, metric="euclidean")

            # Distance from the nucleus surface
            if shape_key not in mode_mesh_dict:
                raise ValueError(f"Mesh information not found for cellid: {shape_key}")

            nuc_mesh = mode_mesh_dict[shape_key]["nuc_mesh"]
            mem_mesh = mode_mesh_dict[shape_key]["mem_mesh"]
            mem_distances = proximity.signed_distance(mem_mesh, positions)
            nuc_surface_distances, scaled_nuc_distances, _ = (
                calc_scaled_distance_to_nucleus_surface(
                    positions,
                    nuc_mesh,
                    mem_mesh,
                    mem_distances,
                )
            )
            good_inds = (nuc_surface_distances > 0) & (mem_distances > 0)
            log.debug(
                f"Fraction bad inds: {1 - np.sum(good_inds) / len(good_inds):.2f}"
            )
            all_nuc_distances[mode][seed_to_use] = nuc_surface_distances[good_inds]
            all_scaled_nuc_distances[mode][seed_to_use] = scaled_nuc_distances[
                good_inds
            ]
            all_membrane_distances[mode][seed_to_use] = mem_distances[good_inds]

            # Nearest neighbor distance
            nearest_distances = np.min(
                all_distances + np.eye(len(positions)) * 1e6, axis=1
            )
            all_nearest_distances[mode][seed_to_use] = nearest_distances

            # Pairwise distance
            pairwise_distances = squareform(all_distances)
            all_pairwise_distances[mode][seed_to_use] = pairwise_distances

            # Z distance
            z_min = mode_mesh_dict[shape_key]["cell_bounds"][:, 2].min()
            z_distances = positions[:, 2] - z_min
            all_z_distances[mode][seed_to_use] = z_distances

    # save distance dictionaries
    if results_dir is not None:
        for distance_dict, distance_measure in zip(
            [
                all_pairwise_distances,
                all_nuc_distances,
                all_scaled_nuc_distances,
                all_nearest_distances,
                all_z_distances,
                all_membrane_distances,
            ],
            ["pairwise", "nucleus", "scaled_nucleus", "nearest", "z", "membrane"],
        ):
            file_path = results_dir / f"{distance_measure}_distances.dat"
            with open(file_path, "wb") as f:
                pickle.dump(distance_dict, f)

    all_distance_dict = {
        "pairwise": all_pairwise_distances,
        "nucleus": all_nuc_distances,
        "scaled_nucleus": all_scaled_nuc_distances,
        "nearest": all_nearest_distances,
        "z": all_z_distances,
        "membrane": all_membrane_distances,
    }

    return all_distance_dict


def plot_distance_distributions_kde(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    figures_dir: Union[Path, None] = None,
    suffix: str = "",
    normalization: Union[str, None] = None,
    overlay: bool = False,
    distance_limits: Union[Dict[str, tuple[float, float]], None] = None,
    bandwidth: Union[Literal["scott", "silverman"], float] = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot distance distributions using kernel density estimation (KDE).

    Parameters
    ----------
    distance_measures
        List of distance measures to plot.
    packing_modes
        List of packing modes to plot.
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure.
        Should have the form:
        {
            distance_measure: {
                packing_mode: {
                    seed: np.ndarray of distances
                }
            }
        }
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    suffix
        Suffix to append to the figure filenames.
    normalization
        Normalization method to apply to the distance measures.
        If None, no normalization is applied.
    overlay
        If True, overlay the pooled KDE.
    distance_limits
        Dictionary containing limits for each distance measure.
        If None, limits will be automatically determined.
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value.
        Default is "scott".
    save_format
        Format to save the figures in. Default is "png".
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
    all_distance_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    figures_dir: Union[Path, None] = None,
    suffix: str = "",
    normalization: Union[str, None] = None,
    overlay: bool = False,
    distance_limits: Union[Dict[str, tuple[float, float]], None] = None,
    bandwidth: Union[Literal["scott", "silverman"], float] = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot distance distributions using kernel density estimation (KDE).

    Parameters
    ----------
    distance_measures
        List of distance measures to plot.
    packing_modes
        List of packing modes to plot.
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure.
        Should have the form:
        {
            distance_measure: {
                packing_mode: {
                    seed: np.ndarray of distances
                }
            }
        }
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    suffix
        Suffix to append to the figure filenames.
    normalization
        Normalization method to apply to the distance measures.
        If None, no normalization is applied.
    overlay
        If True, overlay the pooled KDE.
    distance_limits
        Dictionary containing limits for each distance measure.
        If None, limits will be automatically determined.
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value.
        Default is "scott".
    save_format
        Format to save the figures in. Default is "png".
    """
    log.info("Starting distance distribution kde plot")
    num_rows = len(packing_modes)

    fig_list, ax_list = [], []

    for col, distance_measure in enumerate(distance_measures):
        fig, axs = plt.subplots(
            num_rows,
            1,
            figsize=(5.5, num_rows * 1.5),
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
                # break

            # plot combined kde plot of distance distributions
            combined_mode_distances = np.concatenate(list(mode_dict.values()))
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
    all_distance_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ks_observed_dict: Dict[str, Dict[str, Dict[str, float]]],
    figures_dir: Union[Path, None] = None,
    suffix: str = "",
    normalization: Union[str, None] = None,
    baseline_mode: Union[str, None] = None,
    significance_level: float = 0.05,
    bandwidth: Union[Literal["scott", "silverman"], float] = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot distance distributions for packings with significant
    and non-significant KS test results.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot.
    packing_modes
        List of packing modes to plot.
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure.
        Should have the form:
        {
            distance_measure: {
                packing_mode: {
                    seed: np.ndarray of distances
                }
            }
        }
    ks_observed_dict
        Dictionary containing KS test results for each packing mode and distance measure.
        Should have the form:
        {
            distance_measure: {
                packing_mode: {
                    seed: p_value
                }
            }
        }
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    suffix
        Suffix to append to the figure filenames.
    normalization
        Normalization method to apply to the distance measures.
        If None, no normalization is applied.
    baseline_mode
        Packing mode to use as baseline for comparison.
    significance_level
        Significance level for the KS test. Default is 0.05.
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value.
        Default is "scott".
    save_format
        Format to save the figures in. Default is "png".
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
    all_distance_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    figures_dir: Union[Path, None] = None,
    suffix: str = "",
    normalization: Union[str, None] = None,
    bandwidth: Union[Literal["scott", "silverman"], float] = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot overlaid distance distributions using kernel density estimation (KDE).

    Parameters
    ----------
    distance_measures
        List of distance measures to plot.
    packing_modes
        List of packing modes to plot.
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure.
        Should have the form:
        {
            distance_measure: {
                packing_mode: {
                    seed: np.ndarray of distances
                }
            }
        }
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    suffix
        Suffix to append to the figure filenames.
    normalization
        Normalization method to apply to the distance measures.
        If None, no normalization is applied.
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value.
        Default is "scott".
    save_format
        Format to save the figures in. Default is "png".
    """
    log.info("Plotting overlaid distance distributions")

    for distance_measure in distance_measures:
        distance_dict = all_distance_dict[distance_measure]
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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
    distance_measures,
    packing_modes,
    all_distance_dict,
    figures_dir=None,
    suffix="",
    normalization=None,
    save_format="png",
):
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
            combined_mode_distances = combined_mode_distances[
                ~np.isnan(combined_mode_distances)
            ]

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


def get_ks_observed_combined_df(
    distance_measures,
    packing_modes,
    all_distance_dict,
    baseline_mode="SLC25A17",
    significance_level=0.05,
    save_dir=None,
    recalculate=True,
):
    """
    Perform KS test between distance distributions of different packing modes and combine results.

    Parameters
    ----------
    distance_measures: list
        List of distance measures to compare.
    packing_modes: list
        List of packing modes to compare.
    all_distance_dict: dict
        Dictionary containing distance distributions for each packing mode and distance measure.
        Should have the form:
        {
            distance_measure: {
                packing_mode: {
                    seed: np.ndarray of distances
                }
            }
        }
    baseline_mode: str
        The packing mode to use as the baseline for comparison.
        Default is "SLC25A17".
    significance_level: float
        Significance level for the KS test. Default is 0.05.

    Returns
    -------
    :
        DataFrame containing the KS observed results, with columns for cellid, distance_measure,
        packing_mode, and ks_observed.
    """
    file_name = "ks_observed_combined_df.parquet"
    if not recalculate and save_dir is not None:
        file_path = save_dir / file_name
        if file_path.exists():
            log.info(f"Loading saved KS observed DataFrame from {file_path}")
            return pd.read_parquet(file_path)
    df_list = []
    for distance_measure in distance_measures:
        distance_dict = all_distance_dict[distance_measure]
        ks_observed = {}
        for mode in packing_modes:
            if mode == baseline_mode:
                continue
            log.info(
                f"KS test between {baseline_mode} and {mode}, distance: {distance_measure}"
            )
            mode_dict = distance_dict[mode]
            ks_observed[mode] = {}
            for seed, distances in tqdm(mode_dict.items(), total=len(mode_dict)):
                observed_distances = distance_dict[baseline_mode][seed]
                _, p_val = ks_2samp(
                    observed_distances[~np.isnan(observed_distances)],
                    distances[~np.isnan(distances)],
                )
                ks_observed[mode][seed] = p_val
        ks_observed_df = pd.DataFrame(ks_observed)
        ks_observed_df = ks_observed_df >= significance_level
        ks_observed_df["distance_measure"] = distance_measure
        ks_observed_df = ks_observed_df.reset_index().rename(
            columns={"index": "cellid"}
        )
        df_list.append(ks_observed_df)
    ks_observed_combined_df = pd.concat(df_list, ignore_index=True)

    if save_dir is not None:
        save_path = save_dir / file_name
        log.info(f"Saving KS observed DataFrame to {save_path}")
        ks_observed_combined_df.to_parquet(save_path, index=False)

    return ks_observed_combined_df


def melt_df_for_plotting(df_plot):
    """
    Melt the DataFrame for plotting.

    Parameters
    ----------
    df_plot: DataFrame
        DataFrame containing the KS observed results with columns for cellid, distance_measure,
        and packing modes.
    Returns
    -------
    :
        Melted DataFrame with columns for cellid, distance_measure, packing_mode, and ks_observed.
    """
    df_melt = df_plot.melt(
        id_vars=["cellid", "distance_measure"],
        var_name="packing_mode",
        value_name="ks_observed",
    )
    # relabel values
    df_melt["packing_mode"] = (
        df_melt["packing_mode"].map(MODE_LABELS).fillna(df_melt["packing_mode"])
    )
    df_melt["distance_measure"] = (
        df_melt["distance_measure"]
        .map(DISTANCE_MEASURE_LABELS)
        .fillna(df_melt["distance_measure"])
    )
    return df_melt


def plot_ks_observed_barplots(
    df_melt,
    figures_dir=None,
    suffix="",
    significance_level=0.05,
    save_format="png",
):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
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


def get_pairwise_wasserstein_distance_dict(
    distribution_dict_1,
    distribution_dict_2=None,
):
    """
    distribution_dict is a dictionary with distances or other values for multiple seeds
    it has the form: {seed1: [value1, value2, ...], seed2: [value1, value2, ...], ...}

    The output has the form: {(seed1, seed2): distance, (seed1, seed3): distance, ...}
    """
    pairwise_wasserstein_distances = {}

    keys_1 = list(distribution_dict_1.keys())
    if distribution_dict_2 is None:
        for i in tqdm(range(len(keys_1))):
            for j in range(i + 1, len(keys_1)):
                seed_1 = keys_1[i]
                seed_2 = keys_1[j]
                pairwise_wasserstein_distances[(seed_1, seed_2)] = wasserstein_distance(
                    distribution_dict_1[seed_1], distribution_dict_1[seed_2]
                )
    else:
        keys_2 = list(distribution_dict_2.keys())
        for i in tqdm(range(len(keys_1))):
            for j in range(len(keys_2)):
                seed_1 = keys_1[i]
                seed_2 = keys_2[j]
                if seed_1 == seed_2:
                    continue
                pairwise_wasserstein_distances[(seed_1, seed_2)] = wasserstein_distance(
                    distribution_dict_1[seed_1], distribution_dict_2[seed_2]
                )

    return pairwise_wasserstein_distances


def get_distance_distribution_emd_df(
    all_distance_dict,
    packing_modes,
    distance_measures,
    results_dir=None,
    recalculate=False,
    baseline_mode=None,
    suffix="",
):
    """
    Calculate pairwise EMD between packing modes for each distance measure.

    Parameters
    ----------
    all_distance_dict: dict
        Dictionary containing distance measures for each packing mode.
    packing_modes: list
        List of packing modes to calculate pairwise EMD for.
    distance_measures: list
        List of distance measures to calculate pairwise EMD for.
    results_dir: str, optional
        Directory to save the EMD results.
    recalculate: bool, optional
        Whether to recalculate the EMD even if results already exist.
    baseline_mode: str, optional
        Baseline packing mode to compare against.
    suffix: str, optional
        Suffix to add to the saved EMD file name.

    Returns
    ----------
    all_pairwise_emd: dataframe
        DataFrame containing pairwise EMD for each distance measure.
    """
    if not recalculate and results_dir is not None:
        # load saved EMD df
        file_path = results_dir / f"pairwise_emd{suffix}.parquet"
        if file_path.exists():
            log.info(f"Loading pairwise EMD from {file_path.relative_to(PROJECT_ROOT)}")
            all_pairwise_emd = pd.read_parquet(file_path)
            return all_pairwise_emd

    log.info("Calculating pairwise EMDs")
    index_tuples = []
    for distance_measure in distance_measures:
        for mode in packing_modes:
            distributiion_dict = all_distance_dict[distance_measure][mode]
            for seed in distributiion_dict.keys():
                index_tuple = (distance_measure, mode, seed)
                if index_tuple not in index_tuples:
                    index_tuples.append(index_tuple)
    index = pd.MultiIndex.from_tuples(
        index_tuples, names=["distance_measure", "mode", "cellid"]
    )
    df_emd = pd.DataFrame(index=index, columns=index, dtype=float)
    df_emd.sort_index(axis=0, inplace=True)
    df_emd.sort_index(axis=1, inplace=True)
    if baseline_mode is not None:
        df_emd = df_emd[df_emd.index.get_level_values("mode") == baseline_mode]
    log.info(f"Created dataframe with shape {df_emd.shape}")
    for distance_measure in distance_measures:
        distribution_dict = all_distance_dict[distance_measure]
        log.info(f"Processing EMD for distance measure: {distance_measure}")
        for mode_1, mode_1_dict in distribution_dict.items():
            if baseline_mode is not None and mode_1 != baseline_mode:
                continue
            for mode_2, mode_2_dict in distribution_dict.items():
                log.info(f"Calculating EMD for {mode_1}, {mode_2}")
                for seed_1, distances_1 in tqdm(mode_1_dict.items()):
                    for seed_2, distances_2 in mode_2_dict.items():
                        # skip if already calculated
                        if not pd.isna(
                            df_emd.loc[
                                (distance_measure, mode_1, seed_1),
                                (distance_measure, mode_2, seed_2),
                            ]
                        ):
                            continue
                        # distance is zero for same mode and seed
                        if (mode_1 == mode_2) and (seed_1 == seed_2):
                            df_emd.loc[
                                (distance_measure, mode_1, seed_1),
                                (distance_measure, mode_2, seed_2),
                            ] = 0
                            continue

                        distance = wasserstein_distance(distances_1, distances_2)

                        # save symmetric distance
                        df_emd.loc[
                            (distance_measure, mode_2, seed_2),
                            (distance_measure, mode_1, seed_1),
                        ] = df_emd.loc[
                            (distance_measure, mode_1, seed_1),
                            (distance_measure, mode_2, seed_2),
                        ] = distance

    # Save EMD df
    if results_dir is not None:
        file_path = results_dir / f"pairwise_emd{suffix}.parquet"
        log.info(f"Saving pairwise EMD to {file_path.relative_to(PROJECT_ROOT)}")
        df_emd.to_parquet(file_path, index=True)

    return df_emd


def get_distance_distribution_emd_dictionary(
    all_distance_dict,
    packing_modes,
    distance_measures,
    results_dir=None,
    recalculate=False,
    baseline_mode=None,
    suffix="",
):
    """
    Description

    Parameters
    ----------
    all_distance_dict: dictionary
        Dictionary containing distance measures for each packing mode.
    packing_modes: list
        List of packing modes to calculate pairwise EMD for.
    distance_measures: list
        List of distance measures to calculate pairwise EMD for.
    results_dir: str, optional
        Directory to save the EMD results.
    recalculate: bool, optional
        Whether to recalculate the EMD even if results already exist.
    baseline_mode: str, optional
        Baseline packing mode to compare against.
    suffix: str, optional
        Suffix to add to the saved EMD file name.

    Returns
    ----------
    all_pairwise_emd: dictionary
        Dictionary containing pairwise EMD for each distance measure.
    """
    file_name = "distance_distribution_emd"
    if not recalculate and results_dir is not None:
        file_path = results_dir / f"{file_name}{suffix}.dat"
        # load saved EMD dictionary
        if file_path.exists():
            with open(file_path, "rb") as f:
                distance_distribution_emd = pickle.load(f)
            return distance_distribution_emd

    distance_distribution_emd = {}
    for distance_measure in distance_measures:
        distribution_dict = all_distance_dict[distance_measure]
        log.info(
            f"Calculating distance distribution EMD for distance measure: {distance_measure}"
        )
        measure_emd = {}
        for mode_1 in packing_modes:
            if baseline_mode is not None and mode_1 != baseline_mode:
                continue
            if mode_1 not in measure_emd:
                measure_emd[mode_1] = {}
            distribution_dict_1 = distribution_dict[mode_1]
            for mode_2 in packing_modes:
                if (
                    measure_emd.get(mode_2) is not None
                    and measure_emd[mode_2].get(mode_1) is not None
                ):
                    continue
                log.info(f"Calculating EMD for {mode_1}, {mode_2}")

                if mode_1 == mode_2:
                    distribution_dict_2 = None
                else:
                    distribution_dict_2 = distribution_dict[mode_2]
                measure_emd[mode_1][mode_2] = get_pairwise_wasserstein_distance_dict(
                    distribution_dict_1,
                    distribution_dict_2,
                )

                if mode_2 not in measure_emd:
                    measure_emd[mode_2] = {}
                measure_emd[mode_2][mode_1] = measure_emd[mode_1][mode_2]
        distance_distribution_emd[distance_measure] = measure_emd

    # Save EMD dict
    if results_dir is not None:
        file_path = results_dir / f"{file_name}{suffix}.dat"
        with open(file_path, "wb") as f:
            pickle.dump(distance_distribution_emd, f)
        log.info(f"Saved pairwise EMD to {file_path.relative_to(PROJECT_ROOT)}")

    return distance_distribution_emd


def plot_emd_heatmaps(
    distance_measures,
    all_pairwise_emd,
    figures_dir=None,
    suffix="",
    save_format="png",
):
    log.info("Plotting pairwise EMD heatmaps")
    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure]
        fig, axs = plt.subplots(len(emd_dict), len(emd_dict), figsize=(10, 10), dpi=300)
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


def get_average_emd_correlation(
    distance_measures,
    all_pairwise_emd,
    baseline_mode="SLC25A17",
):
    log.info("Calculating avg and std EMD")
    mode_names = list(all_pairwise_emd[distance_measures[0]].keys())
    corr_df_dict = {}
    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure]
        corr_df_dict[distance_measure] = {}
        df_corr = pd.DataFrame(columns=mode_names, index=mode_names)
        df_std = pd.DataFrame(columns=mode_names, index=mode_names)
        for mode_1, mode_1_dict in emd_dict.items():
            for mode_2, emd in mode_1_dict.items():
                if not np.isnan(df_corr.loc[mode_1, mode_2]):
                    continue
                if mode_1 == mode_2:
                    values = squareform(list(emd.values()))
                else:
                    values = list(emd.values())
                    values = np.array(values)
                    dim_len = int(np.sqrt(len(values)))
                    values = values.reshape((dim_len, -1))
                df_corr.loc[mode_1, mode_2] = np.mean(values).item()
                df_corr.loc[mode_2, mode_1] = df_corr.loc[mode_1, mode_2]
                df_std.loc[mode_1, mode_2] = np.std(values).item()
                df_std.loc[mode_2, mode_1] = df_std.loc[mode_1, mode_2]
        baseline_corr = df_corr.loc[baseline_mode, baseline_mode]
        baseline_std = df_std.loc[baseline_mode, baseline_mode]
        df_corr = (df_corr - baseline_corr) / baseline_corr  # type:ignore
        df_std = df_std / baseline_std  # type:ignore
        df_corr = df_corr.astype(float)
        df_std = df_std.astype(float)
        corr_df_dict[distance_measure]["mean"] = df_corr
        corr_df_dict[distance_measure]["std"] = df_std

    return corr_df_dict


def plot_average_emd_correlation_heatmap(
    distance_measures, corr_df_dict, pairwise_emd_dir=None, suffix=""
):
    """
    Plots the correlation heatmap for Earth Mover's Distance (EMD) values.

    Parameters
    ----------
    - distance_measures (list): List of distance measures.
    - df_corr (pandas.DataFrame): DataFrame containing correlation values.
    - pairwise_emd_dir (str or Path, optional): Directory to save the heatmap images.
        Defaults to None.
    - suffix (str, optional): Suffix to add to the heatmap image filenames. Defaults to "".
    """
    for distance_measure in distance_measures:
        fig, ax = plt.subplots(dpi=300)
        df_corr = corr_df_dict[distance_measure]["mean"]
        sns.heatmap(df_corr, cmap="PuOr", annot=True, ax=ax)
        plt.tight_layout()
        ax.set_title(f"{distance_measure} EMD")
        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir / f"{distance_measure}_emd_heatmap{suffix}.png",
                dpi=300,
            )


def plot_emd_correlation_circles(
    distance_measures,
    corr_df_dict,
    figures_dir=None,
    suffix="",
    save_format="png",
):
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
        circles = [Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
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
    all_pairwise_emd: Dict[str, Dict[str, Dict[str, float]]],
    baseline_mode: str = "SLC25A17",
    suffix: str = "",
    figures_dir: Union[Path, None] = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot boxplots of Earth Mover's Distance (EMD) values for different distance measures.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot.
    all_pairwise_emd
        Dictionary containing pairwise EMD values for each distance measure and mode.
        Should have the form:
        {
            distance_measure: {
                baseline_mode: {
                    mode: emd_value
                }
            }
        }
    baseline_mode
        The baseline packing mode to use for comparison. Default is "SLC25A17".
    suffix
        Suffix to append to the figure filenames.
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    save_format
        Format to save the figures in. Default is "png".
    """
    log.info("Plotting EMD variation boxplots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_dict = {
            k: v for k, v in emd_dict.items() if k != baseline_mode
        }  # exclude baseline mode for self comparison
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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
    all_pairwise_emd: Dict[str, Dict[str, Dict[str, float]]],
    baseline_mode: str = "SLC25A17",
    suffix: str = "",
    figures_dir: Union[Path, None] = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot barplots of Earth Mover's Distance (EMD) values for different distance measures.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot.
    all_pairwise_emd
        Dictionary containing pairwise EMD values for each distance measure and mode.
        Should have the form:
        {
            distance_measure: {
                baseline_mode: {
                    mode: emd_value
                }
            }
        }
    baseline_mode
        The baseline packing mode to use for comparison. Default is "SLC25A17".
    suffix
        Suffix to append to the figure filenames.
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    save_format
        Format to save the figures in. Default is "png".
    """
    log.info("Plotting EMD variation barplots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_dict = {
            k: v for k, v in emd_dict.items() if k != baseline_mode
        }  # exclude baseline mode for self comparison
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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
    all_pairwise_emd: Dict[str, Dict[str, Dict[str, float]]],
    baseline_mode: str = "SLC25A17",
    suffix: str = "",
    figures_dir: Union[Path, None] = None,
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot violinplots of Earth Mover's Distance (EMD) values for different distance measures.

    Parameters
    ----------
    distance_measures
        List of distance measures to plot.
    all_pairwise_emd
        Dictionary containing pairwise EMD values for each distance measure and mode.
        Should have the form:
        {
            distance_measure: {
                baseline_mode: {
                    mode: emd_value
                }
            }
        }
    baseline_mode
        The baseline packing mode to use for comparison. Default is "SLC25A17".
    suffix
        Suffix to append to the figure filenames.
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    save_format
        Format to save the figures in. Default is "png".
    """
    log.info("Plotting EMD variation violinplots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_dict = {
            k: v for k, v in emd_dict.items() if k != baseline_mode
        }  # exclude baseline mode for self comparison
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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
    distance_measures,
    all_pairwise_emd,
    baseline_mode="SLC25A17",
    pairwise_emd_dir=None,
    suffix="",
    save_format="png",
):
    log.info("Plotting EMD variation histograms")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_df = pd.DataFrame.from_dict(emd_dict)
        ncol = len(emd_df.columns)
        fig, axs = plt.subplots(
            1, ncol, dpi=300, sharey=True, sharex=True, figsize=(ncol * 5, 5)
        )
        for ct, col in enumerate(emd_df.columns):
            axs[ct].hist(emd_df[col], bins=50, density=True)
            axs[ct].set_title(
                f"{MODE_LABELS[col]}\nn={np.count_nonzero(~np.isnan(emd_df[col]))}"
            )
        fig.supxlabel(f"{distance_measure} EMD")
        fig.supylabel("Density")
        plt.tight_layout()

        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir
                / f"{distance_measure}_emd_boxplot{suffix}.{save_format}",
                dpi=300,
            )


def plot_emd_kdeplots(
    distance_measures,
    all_pairwise_emd,
    baseline_mode="SLC25A17",
    pairwise_emd_dir=None,
    suffix="",
    bandwidth="scott",
    save_format="png",
):
    log.info("Plotting EMD variation kde plots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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

        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir
                / f"{distance_measure}_emd_kdeplot{suffix}.{save_format}",
                dpi=300,
            )


def calculate_ripley_k(
    all_positions,
    mesh_information_dict,
):
    all_ripleyK = {}
    mean_ripleyK = {}
    ci_ripleyK = {}
    r_max = 0.5
    num_bins = 100
    r_values = np.linspace(0, r_max, num_bins)
    for mode, position_dict in all_positions.items():
        log.info(f"Calculating Ripley K for mode: {mode}")
        all_ripleyK[mode] = {}
        for seed, positions in tqdm(position_dict.items()):
            radius = mesh_information_dict[seed]["cell_diameter"] / 2
            volume = 4 / 3 * np.pi * radius**3
            mean_k_values, _ = ripley_k(
                positions, volume, r_values, norm_factor=(radius * 2)
            )
            all_ripleyK[mode][seed] = mean_k_values
        mean_ripleyK[mode] = np.mean(
            np.array([np.array(v) for v in all_ripleyK[mode].values()]), axis=0
        )
        ci_ripleyK[mode] = np.percentile(
            np.array([np.array(v) for v in all_ripleyK[mode].values()]),
            [2.5, 97.5],
            axis=0,
        )

    return all_ripleyK, mean_ripleyK, ci_ripleyK, r_values


def plot_ripley_k(
    mean_ripleyK,
    ci_ripleyK,
    r_values,
    figures_dir=None,
    suffix="",
    save_format="png",
):
    log.info("Plotting Ripley K")
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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


def get_normalization_factor(
    normalization,
    mesh_information_dict,
    cellid,
    distances=None,
    distance_measure="nucleus",
    pix_size=0.108,
):
    """
    Get the normalization factor for the distances based on the specified normalization method.

    Parameters
    ----------
    normalization: str
        Normalization method to use. Options are:
        - "intracellular_radius": Normalize by the intracellular radius.
        - "cell_diameter": Normalize by the cell diameter.
        - "max_distance": Normalize by the maximum distance.
        - None: No normalization.
    mesh_information_dict: dict
        Dictionary containing mesh information for each seed.
    cellid: str
        Seed/cellid for which to get the normalization factor.
    pix_size: float, optional
        Pixel size for the distances. Default is 0.108.
    Returns
    ----------
    normalization_factor: float
        Normalization factor for the distances.
    """
    if normalization == "intracellular_radius":
        normalization_factor = mesh_information_dict[cellid]["intracellular_radius"]
    elif normalization == "cell_diameter":
        normalization_factor = mesh_information_dict[cellid]["cell_diameter"]
    elif normalization == "max_distance" and distances is not None:
        # Get the maximum distance for the given distances
        distances = distances[~np.isnan(distances) & ~np.isinf(distances)]
        if len(distances) == 0:
            raise ValueError(
                f"No valid distances found for seed {cellid} and normalization {normalization}"
            )
        normalization_factor = np.nanmax(distances)
    elif "scaled" in distance_measure:
        normalization_factor = 1
    else:
        normalization_factor = 1 / pix_size

    if normalization_factor == 0 or np.isnan(normalization_factor):
        raise ValueError(
            f"Invalid normalization factor for seed {cellid} and normalization {normalization}"
        )
    return normalization_factor


def get_distance_distribution_kde(
    all_distance_dict,
    mesh_information_dict,
    channel_map,
    packing_modes,
    results_dir=None,
    recalculate=False,
    suffix="",
    normalization=None,
    distance_measure="nucleus",
    bandwidth="scott",
):
    """
    Calculate the kernel density estimation (KDE) for a given distance measure.
    This function computes the KDE for each packing mode and seed, and saves the results to a file.
    If the results already exist and recalculate is set to False, the function will load
    the existing results.
    The KDE is calculated using the Gaussian kernel density estimation method.
    The available space distances are also calculated and stored in the output dictionary.

    Parameters
    ----------
    all_distance_dict: dict
        Dictionary containing distance measures for each packing mode.
    mesh_information_dict: dict
        Dictionary containing mesh information for each seed.
    channel_map: dict
        Dictionary mapping packing modes to their corresponding channel names.
    packing_modes: list
        List of packing modes to calculate individual occupancy KDE for.
    results_dir: str, optional
        Directory to save the KDE results.
    recalculate: bool, optional
        Whether to recalculate the KDE even if results already exist.
    suffix: str, optional
        Suffix to add to the saved KDE file name.
    normalization: str, optional
        Normalization method to use for the distances. Options are:
        - "intracellular_radius": Normalize by the intracellular radius.
        - "cell_diameter": Normalize by the cell diameter.
        - "max_distance": Normalize by the maximum distance.
        - None: No normalization.
    distance_measure: str, optional
        Distance measure to use for the KDE calculation. Default is "nucleus".
    bandwidth: str, optional
        Bandwidth method for the Gaussian KDE. Default is "scott".
    pix_size: float, optional
        Pixel size for the distances. Default is 0.108.

    Returns
    ----------
    kde_dict: dict
        Dictionary containing the KDE for each packing mode and seed.
        The dictionary has the following structure:
        {
            seed: {
                mode: {
                    "distances": distances,
                    "kde": kde,
                },
                "available_distance": {
                    "distances": available_distances,
                    "kde": kde_available_space,
                },
            },
        }
    """
    # Set file path for saving/loading KDE results
    save_file_path = None
    if results_dir is not None:
        filename = f"{distance_measure}_distance_distribution_kde{suffix}.dat"
        save_file_path = results_dir / filename

    # Check if results already exist and recalculate is False
    # If so, load the existing results
    # Otherwise, calculate the KDE
    if not recalculate and save_file_path is not None and save_file_path.exists():
        with open(save_file_path, "rb") as f:
            kde_dict = pickle.load(f)
        return kde_dict

    # Initialize the KDE dictionary
    distance_dict = all_distance_dict[distance_measure]
    kde_dict = {}
    for mode in packing_modes:
        log.info(f"Calculating distance distribution kde for {mode}")

        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})
        mode_distances_dict = distance_dict[mode]

        for seed, distances in tqdm(
            mode_distances_dict.items(), total=len(mode_distances_dict)
        ):
            # Get the distances for a seed/cellid
            # These are already normalized
            distances = distances[~np.isnan(distances) & ~np.isinf(distances)]
            if len(distances) == 0:
                log.warning(f"No valid distances found for seed {seed} and mode {mode}")
                continue

            # Update available distances from mesh information if needed
            if seed not in kde_dict:
                kde_dict[seed] = {}
                available_distances = mode_mesh_dict[seed][
                    GRID_DISTANCE_LABELS[distance_measure]
                ].flatten()
                available_distances = available_distances[
                    ~np.isnan(available_distances) & ~np.isinf(available_distances)
                ]

                normalization_factor = get_normalization_factor(
                    normalization=normalization,
                    mesh_information_dict=mode_mesh_dict,
                    cellid=seed,
                    distance_measure=distance_measure,
                    distances=available_distances,
                )
                available_distances /= normalization_factor

                kde_available_space = gaussian_kde(
                    available_distances, bw_method=bandwidth
                )
                kde_dict[seed]["available_distance"] = {
                    "distances": available_distances,
                    "kde": kde_available_space,
                }

            # Calculate the KDE for the distances
            kde_distance = gaussian_kde(distances, bw_method=bandwidth)
            kde_dict[seed][mode] = {
                "distances": distances,
                "kde": kde_distance,
            }

    # save kde dictionary
    if save_file_path is not None:
        with open(save_file_path, "wb") as f:
            pickle.dump(kde_dict, f)

    return kde_dict


def normalize_density(xvals, density):
    return density / integrate.trapezoid(density, xvals)


def density_ratio(xvals, density1, density2):
    """
    Calculate the density ratio between two densities.

    Parameters
    ----------
    xvals (array-like): The x-values of the densities.
    density1 (array-like): The first density.
    density2 (array-like): The second density.

    Returns
    -------
    array-like: The density ratio between density1 and density2.
    array-like: The normalized density1.
    array-like: The normalized density2.
    """

    # regularize
    min_value = np.minimum(
        np.min(density1[density1 > 0]), np.min(density2[density2 > 0])
    )
    density1 = np.where(density1 <= min_value, min_value, density1)
    density2 = np.where(density2 <= min_value, min_value, density2)

    # normalize densities
    density1 = normalize_density(xvals, density1)
    density2 = normalize_density(xvals, density2)

    return density1 / density2, density1, density2


def cumulative_ratio(xvals, density1, density2):
    """
    Calculate the cumulative ratio between two density distributions.

    Parameters
    ----------
    - xvals (array-like): The x-values of the density distributions.
    - density1 (array-like): The first density distribution.
    - density2 (array-like): The second density distribution.

    Returns
    -------
    - cumulative_ratio (array-like): The cumulative ratio at each x-value.

    """
    cumulative_ratio = np.zeros(len(xvals))
    density1 = normalize_density(xvals, density1)
    density2 = normalize_density(xvals, density2)
    for ct in range(len(xvals)):
        cumulative_ratio[ct] = integrate.trapezoid(
            density1[: ct + 1], xvals[: ct + 1]
        ) / integrate.trapezoid(density2[: ct + 1], xvals[: ct + 1])
    return cumulative_ratio, density1, density2


def get_pdf_ratio(xvals, density1, density2, method="pdf"):
    """
    Calculate the ratio of two probability density functions (PDFs) based on the given method.

    Parameters
    ----------
        xvals (array-like): The x-values of the PDFs.
        density1 (array-like): The values of the first PDF.
        density2 (array-like): The values of the second PDF.
        method (str, optional): The method to calculate the ratio. Default is "pdf".

    Returns
    -------
        float: The ratio of the two PDFs based on the specified method.
    """
    if method == "pdf":
        return density_ratio(xvals, density1, density2)
    elif method == "cumulative":
        return cumulative_ratio(xvals, density1, density2)
    else:
        raise ValueError(f"Invalid ratio method: {method}")


def plot_occupancy_illustration(
    distance_dict,
    kde_dict,
    baseline_mode="random",
    figures_dir=None,
    suffix="",
    distance_measure="nucleus",
    normalization=None,
    method="pdf",
    xlim=None,
    seed_index=None,
    save_format="png",
):
    log.info("Plotting occupancy illustration")

    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=(7, 7))
    mode_dict = distance_dict[baseline_mode]
    all_cellids = list(mode_dict.keys())
    if seed_index is not None:
        seed = all_cellids[seed_index]
    else:
        seed = all_cellids[0]
    distances = mode_dict[seed]
    distances = distances[~np.isnan(distances) & ~np.isinf(distances)]

    kde_distance = kde_dict[seed][baseline_mode]["kde"]
    kde_available_space = kde_dict[seed]["available_distance"]["kde"]

    xvals = np.linspace(0, np.nanmax(distances), 100)

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
    ax, structure_id, mesh_information_dict, normalization=None
):
    """
    Adds the mean and std structure radius to the plot.

    Parameters
    ----------
        ax (matplotlib.axes.Axes): The axes object to add the structural diameter information to.
        a

    Returns
    -------
        matplotlib.axes.Axes: The modified axes object.
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


def get_scaled_structure_radius(
    structure_id, mesh_information_dict, normalization=None
):
    """
    Get the scaled structure radius based on the given structure ID.

    Parameters
    ----------
        structure_id (str): The structure ID.
        mesh_information_dict (dict): A dictionary containing mesh information.
        normalization (str, optional): The normalization method to use. Defaults to None.

    Returns
    -------
        float: The scaled structure radius.
        float: The scaled standard deviation of the structure radius.
    """
    df_struct_stats = get_structure_stats_dataframe(structure_id=structure_id)

    scaled_radius_list = []
    for seed, mesh_info in mesh_information_dict.items():
        if seed not in df_struct_stats.index:
            continue
        scaled_radius_list.append(
            df_struct_stats.loc[seed, "radius"]
            / mesh_info.get(normalization, 1 / PIX_SIZE)
        )

    avg_radius = np.mean(scaled_radius_list)
    std_radius = np.std(scaled_radius_list)

    return avg_radius, std_radius


def create_padded_numpy_array(lists, padding=np.nan):
    """
    Create a padded list with the specified padding value.
    """
    max_length = max([len(sublist) for sublist in lists])
    padded_array = np.zeros((len(lists), max_length))
    for ct, sublist in enumerate(lists):
        if len(sublist) < max_length:
            if isinstance(sublist, list):
                sublist += [padding] * (max_length - len(sublist))
            elif isinstance(sublist, np.ndarray):
                sublist = np.append(sublist, [padding] * (max_length - len(sublist)))
        padded_array[ct] = sublist[:]
    return padded_array


def plot_individual_occupancy_ratio(
    distance_dict,
    kde_dict,
    packing_modes,
    figures_dir=None,
    suffix="",
    normalization=None,
    distance_measure="nucleus",
    method="pdf",
    xlim=None,
    ylim=None,
    sample_size=None,
    save_format="png",
):
    """
    Plots the individual occupancy ratio based on the given parameters.

    Parameters
    ----------
        distance_dict (dict): A dictionary containing distance information.
        kde_dict (dict): A dictionary containing KDE information.
        packing_modes (list): A list of packing modes.
        figures_dir (str, optional): The directory to save the figures. Defaults to None.
        suffix (str, optional): A suffix to add to the figure filename. Defaults to "".
        mesh_information_dict (dict, optional): A dictionary containing mesh information.
            Defaults to None.
        struct_diameter (float, optional): The diameter of the structure. Defaults to None.
        distance_measure (str, optional): The distance measure to plot. Defaults to "nucleus".
        ratio_to_plot (str, optional): The ratio to plot ("pdf" or "cumulative").
            Defaults to "pdf".
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
            distances = distances[~np.isnan(distances) & ~np.isinf(distances)]
            if len(distances) == 0:
                log.warning(
                    f"No valid distances found for cellid {cellid} and mode {mode}"
                )
                continue
            kde_distance = kde_dict[cellid][mode]["kde"]
            kde_available_space = kde_dict[cellid]["available_distance"]["kde"]
            max_distance = np.max(distances)
            xvals = np.linspace(0, max_distance, 100)

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
    distance_dict,
    kde_dict,
    packing_modes,
    figures_dir=None,
    suffix="",
    normalization=None,
    distance_measure="nucleus",
    method="ratio",
    xlim=None,
    ylim=None,
    sample_size=None,
    save_format="png",
):
    """
    Plots the mean occupancy ratio along with the CI

    Parameters
    ----------
        distance_dict (dict): A dictionary containing distance information.
        kde_dict (dict): A dictionary containing KDE information.
        packing_modes (list): A list of packing modes.
        figures_dir (str, optional): The directory to save the figures. Defaults to None.
        suffix (str, optional): A suffix to add to the figure filename. Defaults to "".
        mesh_information_dict (dict, optional): A dictionary containing mesh information.
            Defaults to None.
        struct_diameter (float, optional): The diameter of the structure. Defaults to None.
        distance_measure (str, optional): The distance measure to plot. Defaults to "nucleus".
        ratio_to_plot (str, optional): The ratio to plot ("density" or "ratio").
            Defaults to "ratio".
        sample_size (int, optional): The number of samples to use. Defaults to None.
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


def sample_cellids_from_distance_dict(distance_dict, sample_size, rng_seed=42):
    """
    Sample seeds from the distance dictionary.

    Parameters
    ----------
    distance_dict (dict): A dictionary containing distance information.
    sample_size (int): The number of samples to use.

    Returns
    -------
    list: A list of sampled seeds.
    """
    rng = np.random.default_rng(rng_seed)
    if sample_size is not None:
        cellids_to_use = rng.choice(
            list(distance_dict.keys()), sample_size, replace=False
        )
    else:
        cellids_to_use = distance_dict.keys()

    return cellids_to_use


def plot_binned_occupancy_ratio(
    distance_dict,
    packing_modes,
    mesh_information_dict,
    channel_map,
    figures_dir=None,
    normalization=None,
    suffix="",
    num_bins=64,
    bin_width=None,
    distance_measure="nucleus",
    xlim=None,
    ylim=None,
    sample_size=None,
    save_format="png",
):
    """
    Calculate the binned occupancy ratio based on the provided distance dictionary.
    Parameters:
    all_distance_dict (dict): A dictionary containing distance information for various entities.
    packing_modes (list): A list of packing modes to consider for the analysis.
    results_dir (str, optional): Directory to save the results. Defaults to None.
    normalization (str, optional): Method to normalize the data. Defaults to None.
    suffix (str, optional): Suffix to append to the result filenames. Defaults to "".
    distance_measure (str, optional): The measure of distance to use (e.g., "nucleus").
        Defaults to "nucleus".
    sample_size (int, optional): The number of samples to consider. Defaults to None.
    Returns:
    None
    """

    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))

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
            available_distances = available_distances[
                ~np.isnan(available_distances) & ~np.isinf(available_distances)
            ]
            normalization_factor = get_normalization_factor(
                normalization=normalization,
                mesh_information_dict=mode_mesh_dict,
                cellid=cellid,
                distance_measure=distance_measure,
                distances=available_distances,
            )
            available_distances /= normalization_factor
            combined_available_distance_dict[cellid] = available_distances

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
        for cellid in tqdm(cellids_to_use):
            distances = mode_dict[cellid]
            distances = distances[~np.isnan(distances) & ~np.isinf(distances)]
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


def get_combined_occupancy_kde(
    all_distance_dict,
    mesh_information_dict,
    channel_map,
    packing_modes,
    results_dir=None,
    recalculate=False,
    normalization=None,
    suffix="",
    distance_measure="nucleus",
    sample_size=None,
    bandwidth="scott",
):
    """
    Calculate the combined distance distribution using kernel density estimation (KDE).

    Parameters
    ----------
    - all_distance_dict (dict): A dictionary containing distance information.
    - mesh_information_dict (dict): A dictionary containing mesh information.
    - packing_modes (list): A list of packing modes.
    - results_dir (str, optional): The directory to save the results. Default is None.
    - recalculate (bool, optional): Whether to recalculate the results. Default is False.
    - suffix (str, optional): A suffix to add to the saved file name. Default is an empty string.
    - distance_measure (str, optional): The distance measure to use. Default is "nucleus".
    - sample (float, optional): The fraction of samples to use. Default is None.

    Returns
    -------
    - combined_kde_dict (dict): A dictionary containing the combined KDE values.
    """
    file_path = None
    if results_dir is not None:
        filename = f"{distance_measure}_combined_occupancy_kde{suffix}.dat"
        if normalization is not None:
            filename = f"{distance_measure}_combined_occupancy_kde{suffix}.dat"
        file_path = results_dir / filename

    if not recalculate and file_path is not None and file_path.exists():
        # load combined space corrected kde values
        with open(file_path, "rb") as f:
            combined_kde_dict = pickle.load(f)
        return combined_kde_dict

    distance_dict = all_distance_dict[distance_measure]

    combined_kde_dict = {}
    for mode in packing_modes:
        log.info(f"Calculating combined occupancy kde for: {mode}")
        mode_dict = distance_dict[mode]

        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})

        cellids_to_use = sample_cellids_from_distance_dict(mode_dict, sample_size)

        if len(cellids_to_use) == 0:
            log.warning(f"No valid seeds found for mode {mode}")
            continue

        mode_dict = {
            cellid: mode_dict[cellid]
            for cellid in cellids_to_use
            if cellid in mode_mesh_dict
        }
        log.info(f"Using {len(mode_dict)} seeds for mode {mode}")

        combined_mode_distances = np.concatenate(list(mode_dict.values()))
        combined_mode_distances = combined_mode_distances[
            ~np.isnan(combined_mode_distances) & ~np.isinf(combined_mode_distances)
        ]
        kde_distance = gaussian_kde(combined_mode_distances, bw_method=bandwidth)

        combined_available_distances = []
        for seed in mode_mesh_dict:
            available_distances = mode_mesh_dict[seed][
                GRID_DISTANCE_LABELS[distance_measure]
            ].flatten()
            available_distances = available_distances[
                ~np.isnan(available_distances) & ~np.isinf(available_distances)
            ]
            normalization_factor = get_normalization_factor(
                normalization=normalization,
                mesh_information_dict=mode_mesh_dict,
                cellid=seed,
                distance_measure=distance_measure,
                distances=available_distances,
            )
            available_distances /= normalization_factor
            combined_available_distances.append(available_distances)

        combined_available_distances = np.concatenate(combined_available_distances)

        kde_available_space = gaussian_kde(
            combined_available_distances, bw_method=bandwidth
        )

        combined_kde_dict[mode] = {
            "mode_distances": combined_mode_distances,
            "kde_distance": kde_distance,
            "kde_available_space": kde_available_space,
        }
    if file_path is not None:
        with open(file_path, "wb") as f:
            pickle.dump(combined_kde_dict, f)

    return combined_kde_dict


def plot_combined_occupancy_ratio(
    combined_kde_dict,
    packing_modes,
    figures_dir=None,
    suffix="",
    normalization=None,
    aspect=None,
    save_format="png",
    save_intermediates=False,
    distance_measure="nucleus",
    num_points=100,
    method="ratio",
    xlim=None,
    ylim=None,
):
    log.info("Plotting combined occupancy ratio")

    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    ax.axhline(1, color="k", linestyle="--")

    for ct, mode in enumerate(packing_modes):
        log.info(f"Calculating combined occupancy ratio for: {mode}")

        kde_mode_dict = combined_kde_dict[mode]
        mode_distances = kde_mode_dict["mode_distances"]

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


def get_occupancy_emd(
    distance_dict,
    kde_dict,
    packing_modes,
    results_dir=None,
    recalculate=False,
    suffix="",
    distance_measure="nucleus",
):
    file_path = None
    if results_dir is not None:
        file_path = results_dir / f"{distance_measure}_occupancy_emd{suffix}.dat"

    if not recalculate and file_path is not None and file_path.exists():
        with open(file_path, "rb") as f:
            emd_occupancy_dict = pickle.load(f)
        return emd_occupancy_dict

    emd_occupancy_dict = {}
    for mode in packing_modes:
        log.info(mode)
        mode_dict = distance_dict[mode]
        emd_occupancy_dict[mode] = {}
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            occupied_distance = kde_dict[seed][mode]["distances"]
            available_distance = kde_dict[seed]["available_distance"]["distances"]
            emd_occupancy_dict[mode][seed] = wasserstein_distance(
                occupied_distance, available_distance
            )
    # save occupancy EMD dictionary
    if file_path is not None:
        with open(file_path, "wb") as f:
            pickle.dump(emd_occupancy_dict, f)

    return emd_occupancy_dict


def plot_occupancy_emd_kdeplot(
    emd_occupancy_dict: Dict[str, Dict[str, float]],
    packing_modes: list[str],
    figures_dir: Union[Path, None] = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    bandwidth: Union[Literal["scott", "silverman"], float] = "scott",
    save_format: Literal["svg", "png", "pdf"] = "png",
):
    """
    Plot EMD occupancy distributions using kernel density estimation (KDE).

    Parameters
    ----------
    emd_occupancy_dict
        Dictionary containing EMD occupancy values for each packing mode and seed.
        Should have the form:
        {
            packing_mode: {
                seed: emd_value
            }
        }
    packing_modes
        List of packing modes to plot.
    figures_dir
        Directory to save the figures. If None, figures will not be saved.
    suffix
        Suffix to append to the figure filenames.
    distance_measure
        Distance measure used for the EMD calculation. Default is "nucleus".
    bandwidth
        Bandwidth method for KDE. Can be "scott", "silverman", or a float value.
        Default is "scott".
    save_format
        Format to save the figures in. Default is "png".
    """
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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
    emd_occupancy_dict: Dict[str, List[float]],
    figures_dir: Optional[Path] = None,
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
    emd_occupancy_dict : Dict[str, List[float]]
        Dictionary containing EMD values for each mode/condition, where keys are mode names
        and values are lists of EMD measurements.
    figures_dir : Optional[Path], default=None
        Directory path where the figure should be saved. If None, the figure is not saved.
    suffix : str, default=""
        Additional suffix to append to the saved filename.
    distance_measure : str, default="nucleus"
        Type of distance measurement used, incorporated into the filename when saving.
    save_format : str, default="png"
        File format for saving the figure (e.g., 'png', 'pdf', 'svg').

    Returns
    -------
    None
        This function displays the plot and optionally saves it but does not return a value.

    Notes
    -----
    - The function uses seaborn's boxplot for visualization
    - X-axis labels are rotated 45 degrees for better readability
    - Figure is displayed using plt.show() regardless of save status
    - Requires MODE_LABELS dictionary to be defined for label mapping
    """
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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


def get_occupancy_ks_test_dict(
    distance_dict,
    kde_dict,
    packing_modes,
    results_dir=None,
    recalculate=False,
    suffix="",
    distance_measure="nucleus",
):
    file_path = None
    if results_dir is not None:
        file_path = results_dir / f"{distance_measure}_occupancy_ks{suffix}.dat"

    if not recalculate and file_path is not None and file_path.exists():
        # load ks test results
        with open(file_path, "rb") as f:
            ks_occupancy_dict = pickle.load(f)
        return ks_occupancy_dict

    ks_occupancy_dict = {}
    for mode in packing_modes:
        log.info(mode)
        mode_dict = distance_dict[mode]
        ks_occupancy_dict[mode] = {}
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            occupied_distance = kde_dict[seed][mode]["distances"]
            available_distance = kde_dict[seed]["available_distance"]["distances"]
            _, p_val = ks_2samp(occupied_distance, available_distance)
            ks_occupancy_dict[mode][seed] = p_val

    if file_path is not None:
        with open(file_path, "wb") as f:
            pickle.dump(ks_occupancy_dict, f)

    return ks_occupancy_dict


def plot_occupancy_ks_test(
    ks_occupancy_dict,
    figures_dir=None,
    suffix="",
    distance_measure="nucleus",
    significance_level=0.05,
    save_format="png",
):
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
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
    plt.show()
