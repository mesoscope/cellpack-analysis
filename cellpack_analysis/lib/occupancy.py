import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.lib.stats_functions import normalize_pdf, pdf_ratio

log = logging.getLogger(__name__)


def get_occupancy_dict(
    all_distance_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str],
    num_cells: int | None = None,
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    distance_measure: str = "nucleus",
    bandwidth: float = 0.1,
    num_points: int = 250,
):
    """
    Calculate occupancy dictionary based on distance distributions.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing all distance distributions with structure:
        {
            seed:: {
                "occupied": {
                    mode:: np.ndarray (distances of occupied points)
                },
            "available": {
                mode:: np.ndarray (distances of available points)
            }
        }
    channel_map
        Mapping between modes and channel names
    num_cells
        Number of cells to consider
    results_dir
        Directory to save or load results
    recalculate
        Whether to recalculate the occupancy
    suffix
        Suffix to add to the saved file name
    distance_measure
        Distance measure to use
    bandwidth
        Bandwidth for KDE

    Returns
    -------
    :
        Occupancy dictionary with structure:
        {
            "individual": {
                seed_1: {
                    "occupied": {
                        mode_1: {
                            "xvals": np.ndarray,
                            "pdf": np.ndarray,
                            "occupancy": np.ndarray,
                        },
                        ...
                    },
                    "available": {
                        "xvals": np.ndarray,
                        "pdf": np.ndarray,
                    },
                },
                ...
            },
            "combined": {
                "occupied": {
                    mode_1: {
                        "pdf": np.ndarray,
                        "occupancy": np.ndarray,
                    },
                    ...
                },
                "available": {
                # available distances are same for all modes of a structure
                    structure_id_1: {
                        "xvals": np.ndarray,
                        "pdf": np.ndarray,
                    },
                    ...
                },
            },
        }

    """
    # Set up file path for saving/loading
    file_path = None
    if results_dir is not None:
        file_path = results_dir / f"{distance_measure}_occupancy{suffix}.dat"

    # Check if we can load existing results
    if not recalculate and file_path is not None and file_path.exists():
        with open(file_path, "rb") as f:
            occupancy_dict = pickle.load(f)
        return occupancy_dict

    # Get cell id mapping
    cell_id_map = {}
    for cell_id, cell_id_dict in all_distance_dict.items():
        for mode in cell_id_dict["occupied"].keys():
            structure_id = channel_map.get(mode, mode)
            if structure_id not in cell_id_map:
                cell_id_map[structure_id] = []
            if cell_id not in cell_id_map[structure_id]:
                cell_id_map[structure_id].append(cell_id)

    # Check if num_cells is specified and limit the number of cells
    if num_cells is not None:
        for structure_id, cell_ids in cell_id_map.items():
            cell_id_map[structure_id] = cell_ids[:num_cells]

    occupancy_dict = {"individual": {}, "combined": {"occupied": {}, "available": {}}}
    combined_available_distances = {}  # per structure_id
    combined_occupied_distances = {}  # per mode
    # Loop through packing modes
    # Random, Nucleus bias, etc.
    for mode, structure_id in channel_map.items():
        log.info(f"Calculating occupancy for {mode}")

        # Keep track of available distances for the mode
        # Combined available only needs to be done once per structure_id
        if structure_id not in combined_available_distances:
            combined_available_distances[structure_id] = []

        # Keep track of occupied distances for the mode
        if mode not in combined_occupied_distances:
            combined_occupied_distances[mode] = []

        for cell_id in tqdm(cell_id_map[structure_id]):

            # Check if this mode has occupied distances for this cell
            # This can happen if a mode was not simuated for this cell ID
            # e.g., ER  struct gradient mode if not available for the observed peroxisome data
            if mode not in all_distance_dict[cell_id]["occupied"]:
                continue

            # Add individual available distances if not already added
            # 1. Add to the individual dictionary if not already added for the cell
            # 2. Extend the combined available distances for the structure
            # The check makes sure each cell only gets counted once across all modes
            if cell_id not in occupancy_dict["individual"]:
                cell_available_distances = all_distance_dict[cell_id]["available"]
                xvals = np.linspace(
                    np.min(cell_available_distances), np.max(cell_available_distances), num_points
                )
                available_kde = gaussian_kde(cell_available_distances, bw_method=bandwidth)
                available_pdf = available_kde(xvals)
                available_pdf = normalize_pdf(xvals, available_pdf)
                occupancy_dict["individual"][cell_id] = {
                    "occupied": {},
                    "available": {
                        "xvals": xvals,
                        "pdf": available_pdf,
                    },
                }
                combined_available_distances[structure_id].extend(cell_available_distances)

            # Get occupied distances for the current mode
            # 1. Extend the combined occupied distances for the mode
            # 2. Calculate individual occupied pdf and occupancy
            #    and add to individual occupancy dict
            cell_occupied_distances = all_distance_dict[cell_id]["occupied"][mode]
            combined_occupied_distances[mode].extend(cell_occupied_distances)

            # Calculate indicidual occupancy for the current cell and mode
            xvals = occupancy_dict["individual"][cell_id]["available"]["xvals"]
            occupied_kde = gaussian_kde(cell_occupied_distances, bw_method=bandwidth)
            occupied_pdf = occupied_kde(xvals)
            occupied_pdf = normalize_pdf(xvals, occupied_pdf)
            occupancy = pdf_ratio(
                xvals,
                occupied_pdf,
                occupancy_dict["individual"][cell_id]["available"]["pdf"],
            )[0]

            # Update individual occupancy dict
            occupancy_dict["individual"][cell_id]["occupied"][mode] = {
                "pdf": occupied_pdf,
                "occupancy": occupancy,
            }

        # Add combined available distances if not already added
        # in the first loop across all cellids for the structure. the available distances
        # were calculated
        if structure_id not in occupancy_dict["combined"]["available"]:
            structure_available_distances = combined_available_distances[structure_id]
            xvals = np.linspace(
                np.min(structure_available_distances),
                np.max(structure_available_distances),
                num_points,
            )
            available_kde = gaussian_kde(structure_available_distances, bw_method=bandwidth)
            available_pdf = available_kde(xvals)
            available_pdf = normalize_pdf(xvals, available_pdf)
            occupancy_dict["combined"]["available"][structure_id] = {
                "xvals": xvals,
                "pdf": available_pdf,
            }

        # Calculate combined occupancy for mode
        xvals = occupancy_dict["combined"]["available"][structure_id]["xvals"]
        mode_occupied_distances = combined_occupied_distances[mode]
        occupied_kde = gaussian_kde(mode_occupied_distances, bw_method=bandwidth)
        occupied_pdf = occupied_kde(xvals)
        occupied_pdf = normalize_pdf(xvals, occupied_pdf)
        available_pdf = occupancy_dict["combined"]["available"][structure_id]["pdf"]
        occupancy = pdf_ratio(xvals, occupied_pdf, available_pdf)[0]
        occupancy_dict["combined"]["occupied"][mode] = {
            "pdf": occupied_pdf,
            "occupancy": occupancy,
        }

    # Save occupancy dictionary
    if file_path is not None:
        with open(file_path, "wb") as f:
            pickle.dump(occupancy_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

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
