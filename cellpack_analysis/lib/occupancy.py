import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.stats import normalize_pdf, pdf_ratio

logger = logging.getLogger(__name__)


def get_cell_id_map_from_distance_kde_dict(
    kde_dict: dict[str, dict[str, gaussian_kde]],
    channel_map: dict[str, str],
) -> dict[str, list[str]]:
    """
    Create mapping from structure IDs to cell IDs based on available KDE data.

    Parameters
    ----------
    kde_dict
        Dictionary with cell IDs as keys and mode-specific KDEs as values
    channel_map
        Mapping from packing modes to structure IDs

    Returns
    -------
    :
        Dictionary mapping structure IDs to lists of cell IDs
    """
    cell_id_map = {}
    for mode, structure_id in channel_map.items():
        if structure_id not in cell_id_map:
            cell_id_map[structure_id] = []
        new_cell_ids = [
            cell_id
            for cell_id in kde_dict.keys()
            if mode in kde_dict[cell_id] and cell_id not in cell_id_map[structure_id]
        ]
        cell_id_map[structure_id].extend(new_cell_ids)
    return cell_id_map


def get_kde_occupancy_dict(
    distance_kde_dict: dict[str, dict[str, gaussian_kde]],
    channel_map: dict[str, str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    distance_measure: str = "nucleus",
    bandwidth: str | float | None = None,
    num_cells: int | None = None,
    num_points: int = 100,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load or compute occupancy ratio from distance distribution KDEs.

    Parameters
    ----------
    distance_kde_dict
        Dictionary with cell IDs as keys and mode-specific KDEs as values
        Structure: {cell_id: {mode: gaussian_kde, "available_distance": gaussian_kde}}
    channel_map
        Mapping from packing modes to structure IDs
    results_dir
        Directory to save the results, by default None
    recalculate
        If True, recalculate even if results exist. Default is False
    suffix
        Suffix to add to the saved file name
    distance_measure
        Distance measure to analyze
    bandwidth
        Bandwidth for KDE smoothing, by default None
    num_cells
        Maximum number of cells to sample per structure, by default None
    num_points
        Number of points for KDE evaluation, by default 100

    Returns
    -------
    :
        Dictionary containing occupancy ratios and related data
        Has the structure:
        {
            "mode1": {
                "individual": {
                    "cell_id1": {
                        "xvals": np.ndarray,
                        "occupancy": np.ndarray,
                        "pdf_occupied": np.ndarray,
                        "pdf_available": np.ndarray
                    },
                    "cell_id2": { ... },
                    ...
                },
                "combined": { ... }
            },
            "mode2": { ... },
            ...
        }
    """
    # Set file path for saving/loading occupancy dict
    save_path = None
    if results_dir is not None:
        save_path = results_dir / f"{distance_measure}_occupancy{suffix}.dat"

    # Check if we need to recalculate or load existing data
    if not recalculate and save_path is not None and save_path.exists():
        with open(save_path, "rb") as f:
            kde_occupancy_dict = pickle.load(f)
        return kde_occupancy_dict

    # Initialize occupancy dictionary
    kde_occupancy_dict = {}

    # determine cell ids to use
    cell_id_map = get_cell_id_map_from_distance_kde_dict(distance_kde_dict, channel_map)
    if num_cells is not None:
        for structure_id, cell_ids in cell_id_map.items():
            if len(cell_ids) > num_cells:
                cell_id_map[structure_id] = np.random.choice(
                    cell_ids, size=num_cells, replace=False
                ).tolist()

    # Get all available distances to use for a combined_plot
    combined_available_distance_kde = {}

    x_min = np.inf
    x_max = -np.inf
    for structure_id, cell_ids in cell_id_map.items():
        combined_available_distances = []
        for cell_id in cell_ids:
            combined_available_distances.extend(
                distance_kde_dict[cell_id]["available_distance"].dataset
            )
            for mode in channel_map.keys():
                if mode in distance_kde_dict[cell_id]:
                    x_min = min(x_min, np.min(distance_kde_dict[cell_id][mode].dataset))
                    x_max = max(x_max, np.max(distance_kde_dict[cell_id][mode].dataset))
        combined_available_distance_kde[structure_id] = gaussian_kde(
            np.concatenate(combined_available_distances), bw_method=bandwidth
        )

    for mode, structure_id in channel_map.items():
        kde_occupancy_dict[mode] = {"individual": {}, "combined": {}}
        combined_occupied_distances = []

        for cell_id in tqdm(
            cell_id_map.get(structure_id, []), desc=f"Computing occupancy for {mode}"
        ):
            if mode not in distance_kde_dict[cell_id]:
                continue

            # Get occupied and available distances for cell_id
            occupied_kde = distance_kde_dict[cell_id][mode]
            occupied_distances = occupied_kde.dataset
            combined_occupied_distances.extend(occupied_distances)

            available_kde = distance_kde_dict[cell_id]["available_distance"]
            if bandwidth is not None:
                occupied_kde.set_bandwidth(bandwidth)
                available_kde.set_bandwidth(bandwidth)

            # Create xvals for evaluation
            x_vals = np.linspace(x_min, x_max, num_points)
            pdf_occupied = normalize_pdf(x_vals, occupied_kde.evaluate(x_vals))
            pdf_available = normalize_pdf(x_vals, available_kde.evaluate(x_vals))
            occupancy, pdf_occupied, pdf_available = pdf_ratio(x_vals, pdf_occupied, pdf_available)
            kde_occupancy_dict[mode]["individual"][cell_id] = {
                "xvals": x_vals,
                "occupancy": occupancy,
                "pdf_occupied": pdf_occupied,
                "pdf_available": pdf_available,
            }

        combined_occupied_distances = np.concatenate(combined_occupied_distances)
        # Create xvals for evaluation
        x_vals = np.linspace(x_min, x_max, num_points)

        pdf_combined_occupied = normalize_pdf(
            x_vals, gaussian_kde(combined_occupied_distances, bw_method=bandwidth).evaluate(x_vals)
        )
        pdf_combined_available = normalize_pdf(
            x_vals, combined_available_distance_kde[structure_id].evaluate(x_vals)
        )
        occupancy, pdf_combined_occupied, pdf_combined_available = pdf_ratio(
            x_vals, pdf_combined_occupied, pdf_combined_available
        )
        kde_occupancy_dict[mode]["combined"] = {
            "xvals": x_vals,
            "occupancy": occupancy,
            "pdf_occupied": pdf_combined_occupied,
            "pdf_available": pdf_combined_available,
        }

    # save occupancy dictionary
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(kde_occupancy_dict, f)

    return kde_occupancy_dict


def get_occupancy_emd(
    distance_dict: dict[str, dict[str, np.ndarray]],
    kde_dict: dict[str, dict[str, dict[str, Any]]],
    packing_modes: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    distance_measure: str = "nucleus",
) -> dict[str, dict[str, float]]:
    """
    Calculate Earth Mover's Distance (EMD) between occupied and available space distributions.

    Parameters
    ----------
    distance_dict
        Dictionary containing distance information
    kde_dict
        Dictionary containing KDE information
    packing_modes
        List of packing modes to analyze
    results_dir
        Directory to save the results
    recalculate
        If True, recalculate even if results exist. Default is False
    suffix
        Suffix to add to the saved file name
    distance_measure
        Distance measure to analyze

    Returns
    -------
    :
        Dictionary containing EMD values for each mode and seed
    """
    file_path = None
    if results_dir is not None:
        file_path = results_dir / f"{distance_measure}_occupancy_emd{suffix}.dat"

    if not recalculate and file_path is not None and file_path.exists():
        with open(file_path, "rb") as f:
            emd_occupancy_dict = pickle.load(f)
        return emd_occupancy_dict

    emd_occupancy_dict = {}
    for mode in packing_modes:
        logger.info(mode)
        mode_dict = distance_dict[mode]
        emd_occupancy_dict[mode] = {}
        for _k, (seed, _distances) in tqdm(enumerate(mode_dict.items()), total=len(mode_dict)):
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


def get_occupancy_ks_test_dict(
    distance_dict: dict[str, dict[str, np.ndarray]],
    kde_dict: dict[str, dict[str, dict[str, Any]]],
    packing_modes: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    distance_measure: str = "nucleus",
) -> dict[str, dict[str, float]]:
    """
    Perform KS test between occupied and available space distributions.

    Parameters
    ----------
    distance_dict
        Dictionary containing distance information
    kde_dict
        Dictionary containing KDE information
    packing_modes
        List of packing modes to analyze
    results_dir
        Directory to save the results
    recalculate
        If True, recalculate even if results exist. Default is False
    suffix
        Suffix to add to the saved file name
    distance_measure
        Distance measure to analyze

    Returns
    -------
    :
        Dictionary containing KS test p-values for each mode and seed
    """
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
        logger.info(mode)
        mode_dict = distance_dict[mode]
        ks_occupancy_dict[mode] = {}
        for _k, (seed, _distances) in tqdm(enumerate(mode_dict.items()), total=len(mode_dict)):
            occupied_distance = kde_dict[seed][mode]["distances"]
            available_distance = kde_dict[seed]["available_distance"]["distances"]
            _, p_val = ks_2samp(occupied_distance, available_distance)
            ks_occupancy_dict[mode][seed] = p_val

    if file_path is not None:
        with open(file_path, "wb") as f:
            pickle.dump(ks_occupancy_dict, f)

    return ks_occupancy_dict


def interpolate_occupancy_dict(
    occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    baseline_mode: str,
    results_dir: Path | None = None,
    suffix: str = "",
) -> dict[str, Any]:
    """
    Interpolate occupancy data using non-negative least squares fitting.

    Parameters
    ----------
    occupancy_dict
        Dictionary containing occupancy data for each packing mode
    baseline_mode
        The baseline packing mode used for interpolation
    results_dir
        Directory to save the results, by default None
    suffix
        Suffix to add to the saved file name

    Returns
    -------
    :
        Dictionary containing interpolated occupancy data and fit parameters
    """
    xvals = occupancy_dict[baseline_mode]["combined"]["xvals"]
    baseline_occupancy = occupancy_dict[baseline_mode]["combined"]["occupancy"]
    interpolated_occupancy_dict = {
        "xvals": xvals,
        "occupancy": {
            mode: occupancy_dict[mode]["combined"]["occupancy"] for mode in occupancy_dict
        },
    }

    simulated_occupancy_matrix = []
    packing_modes = []
    for packing_mode, mode_dict in occupancy_dict.items():
        if packing_mode == baseline_mode:
            continue
        simulated_occupancy = mode_dict["combined"]["occupancy"]
        simulated_occupancy_matrix.append(simulated_occupancy)
        packing_modes.append(packing_mode)

    simulated_occupancy_matrix = np.array(simulated_occupancy_matrix).T
    # coeffs, _, _, _ = np.linalg.lstsq(simulated_occupancy_matrix, baseline_occupancy, rcond=None)
    coeffs, _ = nnls(simulated_occupancy_matrix, baseline_occupancy)
    relative_contribution = coeffs / np.sum(coeffs)
    reconstructed_occupancy = simulated_occupancy_matrix @ coeffs

    interpolated_occupancy_dict["interpolation"] = {
        "occupancy": reconstructed_occupancy,
        "fit_params": {
            mode: {"coefficient": coeffs[i], "relative_contribution": relative_contribution[i]}
            for i, mode in enumerate(packing_modes)
        },
    }

    log_file_path = None
    if results_dir is not None:
        log_file_path = results_dir / f"{baseline_mode}_occupancy_interpolation_coeffs{suffix}.log"

    log_occupancy_interpolation_coeffs(
        interpolated_occupancy_dict, baseline_mode=baseline_mode, file_path=log_file_path
    )

    return interpolated_occupancy_dict


def log_occupancy_interpolation_coeffs(
    interpolated_occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    baseline_mode: str,
    file_path: Path | None = None,
) -> None:
    """
    Log interpolation coefficients for occupancy data.

    Parameters
    ----------
    interpolated_occupancy_dict
        Dictionary containing interpolated occupancy data and fit parameters
    baseline_mode
        The baseline packing mode used for interpolation
    file_path
        Optional file path to save the log output
    """
    if file_path is not None:
        dist_logger = add_file_handler_to_logger(logger, file_path)
    else:
        dist_logger = logger

    dist_logger.info(f"Baseline mode: {baseline_mode}")
    for mode, params in interpolated_occupancy_dict["interpolation"]["fit_params"].items():
        dist_logger.info(
            f"Mode: {mode}, Coefficient: {params['coefficient']:.4f}, "
            f"Relative Contribution: {params['relative_contribution']:.4f}"
        )
    remove_file_handler_from_logger(dist_logger, file_path)


def get_binned_occupancy_dict(
    distance_kde_dict: dict[str, dict[str, gaussian_kde]],
    channel_map: dict[str, str],
    num_cells: int | None = None,
    num_bins: int = 64,
    bin_width: float | None = None,
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    distance_measure: str = "nucleus",
    x_max: float | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Calculate binned occupancy ratios from distance KDE data.

    Parameters
    ----------
    distance_kde_dict
        Dictionary with cell IDs as keys and mode-specific KDEs as values
    channel_map
        Mapping from packing modes to structure IDs
    num_cells
        Maximum number of cells to sample per structure, by default None
    num_bins
        Number of histogram bins, by default 64
    bin_width
        Width of histogram bins. If provided, overrides num_bins
    results_dir
        Directory to save the results, by default None
    recalculate
        If True, recalculate even if results exist. Default is False
    suffix
        Suffix to add to the saved file name
    distance_measure
        Distance measure to analyze
    x_max
        Maximum distance for binning. If None, uses data maximum

    Returns
    -------
    :
        Dictionary containing binned occupancy ratios and statistics
    """
    # Set file path for saving/loading occupancy dict
    save_path = None
    if results_dir is not None:
        save_path = results_dir / f"{distance_measure}_binned_occupancy{suffix}.dat"

    # Check if we need to recalculate or load existing data
    if not recalculate and save_path is not None and save_path.exists():
        with open(save_path, "rb") as f:
            binned_occupancy_dict = pickle.load(f)
        return binned_occupancy_dict

    # Initialize occupancy dictionary
    binned_occupancy_dict = {}

    cell_id_map = get_cell_id_map_from_distance_kde_dict(distance_kde_dict, channel_map)
    if num_cells is not None:
        for structure_id, cell_ids in cell_id_map.items():
            if len(cell_ids) > num_cells:
                cell_id_map[structure_id] = np.random.choice(
                    cell_ids, size=num_cells, replace=False
                ).tolist()

    # Get all available distances to use for a combined_plot
    combined_available_distance_dict = {}
    for structure_id, cell_ids in cell_id_map.items():
        combined_available_distances = []
        for cell_id in cell_ids:
            combined_available_distances.extend(
                distance_kde_dict[cell_id]["available_distance"].dataset
            )
        combined_available_distance_dict[structure_id] = np.concatenate(
            combined_available_distances
        )

    for mode, structure_id in channel_map.items():
        binned_occupancy_dict[mode] = {"individual": {}, "combined": {}}

        max_distance = x_max or np.nanmax(combined_available_distance_dict[structure_id])
        if bin_width is not None:
            num_bins = int(np.ceil(max_distance / bin_width))
        bins = np.linspace(0, max_distance, num_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        available_space_counts = {}
        occupied_space_counts = {}
        for cell_id in tqdm(
            cell_id_map.get(structure_id, []), desc=f"Computing binned occupancy for {mode}"
        ):
            if mode not in distance_kde_dict[cell_id]:
                continue

            # Get occupied and available distances for cell_id
            occupied_distances = distance_kde_dict[cell_id][mode].dataset
            available_distances = distance_kde_dict[cell_id]["available_distance"].dataset

            available_space_counts[cell_id] = (
                np.histogram(available_distances, bins=bins, density=True)[0] + 1e-16
            )
            occupied_space_counts[cell_id] = (
                np.histogram(occupied_distances, bins=bins, density=True)[0] + 1e-16
            )
            occupancy = occupied_space_counts[cell_id] / available_space_counts[cell_id]
            binned_occupancy_dict[mode]["individual"][cell_id] = {
                "xvals": bin_centers,
                "occupancy": occupancy,
                "pdf_occupied": occupied_space_counts[cell_id],
                "pdf_available": available_space_counts[cell_id],
            }

        occupied_space_counts = np.vstack(list(occupied_space_counts.values()))
        available_space_counts = np.vstack(list(available_space_counts.values()))
        occupancy_ratio = occupied_space_counts / available_space_counts
        mean_occupancy_ratio = np.nanmean(occupancy_ratio, axis=0)
        std_occupancy_ratio = np.nanstd(occupancy_ratio, axis=0)
        binned_occupancy_dict[mode]["combined"] = {
            "xvals": bin_centers,
            "occupancy": mean_occupancy_ratio,
            "std_occupancy": std_occupancy_ratio,
            "pdf_occupied": np.nanmean(occupied_space_counts, axis=0),
            "pdf_available": np.nanmean(available_space_counts, axis=0),
        }

    # save occupancy dictionary
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(binned_occupancy_dict, f)

    return binned_occupancy_dict
