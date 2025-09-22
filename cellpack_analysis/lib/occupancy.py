import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.lib.stats_functions import pdf_ratio

log = logging.getLogger(__name__)


def get_cell_id_map_from_distance_kde_dict(
    kde_dict: dict[str, dict[str, gaussian_kde]],
    channel_map: dict[str, str],
) -> dict[str, list[str]]:
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


def compute_occupancy_from_kde_dict(
    kde_dict: dict[str, dict[str, gaussian_kde]],
    channel_map: dict[str, str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    distance_measure: str = "nucleus",
    bandwidth: str | float | None = None,
    num_cells: int | None = None,
    num_points: int = 100,
):
    # Set file path for saving/loading occupancy dict
    save_path = None
    if results_dir is not None:
        save_path = results_dir / f"{distance_measure}_occupancy{suffix}.dat"

    # Check if we need to recalculate or load existing data
    if not recalculate and save_path is not None and save_path.exists():
        with open(save_path, "rb") as f:
            occupancy_dict = pickle.load(f)
        return occupancy_dict

    # Initialize occupancy dictionary
    occupancy_dict = {}

    # determine cell ids to use
    cell_id_map = get_cell_id_map_from_distance_kde_dict(kde_dict, channel_map)
    if num_cells is not None:
        for structure_id, cell_ids in cell_id_map.items():
            if len(cell_ids) > num_cells:
                cell_id_map[structure_id] = np.random.choice(
                    cell_ids, size=num_cells, replace=False
                ).tolist()

    # Get all available distances to use for a combined_plot
    combined_available_distance_kde = {}
    for structure_id, cell_ids in cell_id_map.items():
        combined_available_distances = []
        for cell_id in cell_ids:
            combined_available_distances.extend(kde_dict[cell_id]["available_distance"].dataset)
        combined_available_distance_kde[structure_id] = gaussian_kde(
            np.array(combined_available_distances), bw_method=bandwidth
        )

    for mode, structure_id in channel_map.items():
        log.info(f"Calculating occupancy for: {mode}")

        occupancy_dict[mode] = {}
        combined_occupied_distances = []

        for cell_id in cell_id_map.get(structure_id, []):
            if mode not in kde_dict[cell_id]:
                continue

            # Get occupied and available distances for cell_id
            occupied_kde = kde_dict[cell_id][mode]
            occupied_distances = occupied_kde.dataset
            combined_occupied_distances.extend(occupied_distances)

            available_kde = kde_dict[cell_id]["available_distance"]
            if bandwidth is not None:
                occupied_kde.set_bandwidth(bandwidth)
                available_kde.set_bandwidth(bandwidth)

            # Create xvals for evaluation
            x_min, x_max = np.min(occupied_distances), np.max(occupied_distances)
            x_vals = np.linspace(x_min, x_max, num_points)
            pdf_occupied = occupied_kde.evaluate(x_vals)
            pdf_available = available_kde.evaluate(x_vals)
            occupancy, pdf_occupied, pdf_available = pdf_ratio(x_vals, pdf_occupied, pdf_available)
            occupancy_dict[mode][cell_id] = {
                "xvals": x_vals,
                "occupancy": occupancy,
                "pdf_occupied": pdf_occupied,
                "pdf_available": pdf_available,
            }

        combined_occupied_distances = np.array(combined_occupied_distances)
        x_min, x_max = np.min(combined_occupied_distances), np.max(combined_occupied_distances)
        x_vals = np.linspace(x_min, x_max, num_points)
        pdf_combined_occupied = gaussian_kde(
            combined_occupied_distances, bw_method=bandwidth
        ).evaluate(x_vals)
        pdf_combined_available = combined_available_distance_kde[structure_id].evaluate(x_vals)
        occupancy, pdf_combined_occupied, pdf_combined_available = pdf_ratio(
            x_vals, pdf_combined_occupied, pdf_combined_available
        )
        occupancy_dict[mode]["combined"] = {
            "xvals": x_vals,
            "occupancy": occupancy,
            "pdf_occupied": pdf_combined_occupied,
            "pdf_available": pdf_combined_available,
        }

    # save occupancy dictionary
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(occupancy_dict, f)

    return occupancy_dict


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
        Whether to recalculate even if results exist
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
        log.info(mode)
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
        Whether to recalculate even if results exist
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
        log.info(mode)
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
