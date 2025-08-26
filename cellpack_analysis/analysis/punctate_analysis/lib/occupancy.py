import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.analysis.punctate_analysis.lib.distance import (
    filter_invalid_distances,
    get_normalization_factor,
)
from cellpack_analysis.analysis.punctate_analysis.lib.stats_functions import (
    sample_cellids_from_distance_dict,
)
from cellpack_analysis.lib.label_tables import GRID_DISTANCE_LABELS

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
    sample_size: int | None = None,
    bandwidth: Literal["scott", "silverman"] | float = "scott",
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
        combined_mode_distances = filter_invalid_distances(combined_mode_distances)
        kde_distance = gaussian_kde(combined_mode_distances, bw_method=bandwidth)

        combined_available_distances = []
        for seed in mode_mesh_dict:
            available_distances = mode_mesh_dict[seed][
                GRID_DISTANCE_LABELS[distance_measure]
            ].flatten()
            available_distances = filter_invalid_distances(available_distances)
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
