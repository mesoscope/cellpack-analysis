import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.lib.distance import filter_invalid_distances, get_normalization_factor
from cellpack_analysis.lib.label_tables import GRID_DISTANCE_LABELS
from cellpack_analysis.lib.stats_functions import sample_cell_ids_from_distance_dict

log = logging.getLogger(__name__)


def get_combined_occupancy_kde(
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    mesh_information_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str],
    packing_modes: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    normalization: str | None = None,
    suffix: str = "",
    distance_measure: str = "nucleus",
    num_cells: int | None = None,
    num_available_points: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Calculate the combined distance distribution using kernel density estimation (KDE).

    Parameters
    ----------
    all_distance_dict
        A dictionary containing distance information
    mesh_information_dict
        A dictionary containing mesh information
    channel_map
        Dictionary mapping packing modes to channels
    packing_modes
        A list of packing modes
    results_dir
        The directory to save the results
    recalculate
        Whether to recalculate the results
    normalization
        Normalization method to apply
    suffix
        A suffix to add to the saved file name
    distance_measure
        The distance measure to use
    sample_size
        The fraction of samples to use
    bandwidth
        Bandwidth method for KDE

    Returns
    -------
    :
        A dictionary containing the combined KDE values
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

        cell_ids_to_use = sample_cell_ids_from_distance_dict(mode_dict, num_cells)

        if len(cell_ids_to_use) == 0:
            log.warning(f"No valid seeds found for mode {mode}")
            continue

        mode_dict = {
            cell_id: mode_dict[cell_id] for cell_id in cell_ids_to_use if cell_id in mode_mesh_dict
        }
        log.info(f"Using {len(mode_dict)} seeds for mode {mode}")

        combined_mode_distances = np.concatenate(list(mode_dict.values()))
        combined_mode_distances = filter_invalid_distances(combined_mode_distances)
        if num_available_points is not None:
            num_available_points = min(num_available_points, len(combined_mode_distances))
            combined_mode_distances = np.random.choice(
                combined_mode_distances, size=num_available_points, replace=False
            )
        kde_distance = gaussian_kde(combined_mode_distances)

        combined_available_distances = []
        for seed in mode_mesh_dict:
            available_distances = mode_mesh_dict[seed][
                GRID_DISTANCE_LABELS[distance_measure]
            ].flatten()
            available_distances = filter_invalid_distances(available_distances)
            normalization_factor = get_normalization_factor(
                normalization=normalization,
                mesh_information_dict=mode_mesh_dict,
                cell_id=seed,
                distance_measure=distance_measure,
                distances=available_distances,
            )
            available_distances /= normalization_factor
            combined_available_distances.append(available_distances)

        combined_available_distances = np.concatenate(combined_available_distances)

        kde_available_space = gaussian_kde(combined_available_distances)

        combined_kde_dict[mode] = {
            "xmin": np.nanmin(combined_mode_distances),
            "xmax": np.nanmax(combined_mode_distances),
            "n_points": len(combined_mode_distances),
            "n_cells": len(mode_dict),
            "kde_distance": kde_distance,
            "kde_available_space": kde_available_space,
        }
    if file_path is not None:
        with open(file_path, "wb") as f:
            pickle.dump(combined_kde_dict, f)

    return combined_kde_dict


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


def get_occupancy_file_path(
    figures_dir: Path,
    distance_measure: str,
    mode: str,
    normalization: str | None = None,
    method: Literal["pdf", "cumulative"] = "pdf",
    suffix: str = "",
) -> Path:
    """
    Generate the file path for saving occupancy ratio data.

    Parameters
    ----------
    figures_dir
        Directory to save the figures
    distance_measure
        Distance measure used
    mode
        Packing mode
    normalization
        Normalization method used
    method
        Method used for occupancy ratio calculation
    suffix
        Suffix to add to the file name

    Returns
    -------
    :
        Path to the .npz file
    """
    if normalization is None:
        normalization_str = ""
    else:
        normalization_str = f"_{normalization}"

    file_name = f"{distance_measure}_{mode}_{method}_occupancy_ratio{normalization_str}{suffix}.npz"
    return figures_dir / file_name


def save_occupancy_ratio_to_file(
    save_dir: Path,
    xvals: np.ndarray | Any,
    yvals: np.ndarray | Any,
    distance_measure: str,
    mode: str,
    normalization: str | None = None,
    method: Literal["pdf", "cumulative"] = "pdf",
    suffix: str = "",
) -> Path:
    """
    Save occupancy ratio data to a .npz file.

    Parameters
    ----------
    figures_dir
        Directory to save the figures
    xvals
        X values for the occupancy ratio
    yvals
        Y values for the occupancy ratio
    distance_measure
        Distance measure used
    mode
        Packing mode
    normalization
        Normalization method used
    method
        Method used for occupancy ratio calculation
    suffix
        Suffix to add to the file name

    Returns
    -------
    :
        Path to the saved .npz file
    """
    file_path = get_occupancy_file_path(
        save_dir, distance_measure, mode, normalization, method, suffix
    )

    np.savez(
        file_path,
        xvals=xvals,
        yvals=yvals,
        distance_measure=distance_measure,
        mode=mode,
        normalization=normalization if normalization is not None else "none",
        method=method,
        suffix=suffix,
    )
    log.info(f"Saved occupancy ratio to {file_path}")
    return file_path


def load_occupancy_ratio_from_file(file_path: Path) -> dict[str, Any]:
    """
    Load occupancy ratio data from a .npz file.

    Parameters
    ----------
    file_path
        Path to the .npz file

    Returns
    -------
    :
        Dictionary containing the loaded data
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    data = np.load(file_path, allow_pickle=True)
    return {
        "xvals": data["xvals"],
        "yvals": data["yvals"],
        "distance_measure": str(data["distance_measure"]),
        "mode": str(data["mode"]),
        "normalization": str(data["normalization"]),
        "method": str(data["method"]),
        "suffix": str(data["suffix"]),
    }
