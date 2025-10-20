import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, squareform
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm
from trimesh import proximity

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    get_project_root,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.label_tables import GRID_DISTANCE_LABELS, MODE_LABELS
from cellpack_analysis.lib.mesh_tools import calc_scaled_distance_to_nucleus_surface
from cellpack_analysis.lib.stats import ripley_k

logger = logging.getLogger(__name__)

PROJECT_ROOT = get_project_root()


def filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    minimum_distance: float | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Filter out invalid values from distance distribution dictionary.

    Removes NaN, infinite, and optionally negative distance values from all
    distance arrays in the nested dictionary structure.

    Parameters
    ----------
    distance_distribution_dict
        Distance distributions with structure
        {distance_measure: {packing_mode: {cell_id: distances}}}
    minimum_distance
        Minimum distance to consider for filtering invalid distances

    Returns
    -------
    :
        Cleaned distance distribution dictionary with same structure as input
    """
    for distance_measure, distance_measure_dict in distance_distribution_dict.items():
        for mode, mode_dict in distance_measure_dict.items():
            for cell_id, distances in mode_dict.items():
                # filter out NaN and inf values
                mode_dict[cell_id] = filter_invalid_distances(
                    distances, minimum_distance=minimum_distance
                )
                num_valid = len(mode_dict[cell_id])
                if num_valid == 0:
                    logger.warning(
                        f"All distances are invalid for {distance_measure}, {mode}, {cell_id}"
                    )
                    del mode_dict[cell_id]
                else:
                    logger.debug(
                        f"Filtered {len(distances) - num_valid} values from "
                        f"{distance_measure}, {mode}, {cell_id}"
                    )
    return distance_distribution_dict


def filter_invalid_distances(
    distances: np.ndarray, minimum_distance: float | None = None
) -> np.ndarray:
    """
    Remove NaN, infinite, and optionally filter distance values.

    Parameters
    ----------
    distances
        Array of distance values to filter
    minimum_distance
        Minimum distance to consider for filtering invalid distances

    Returns
    -------
    :
        Array of valid distance values
    """
    condition = ~np.isnan(distances) & ~np.isinf(distances)
    if minimum_distance is not None:
        condition &= distances >= minimum_distance
    return distances[condition]


def _calculate_distances_for_cell_id(
    cell_id: str,
    positions: np.ndarray,
    mesh_dict: dict[str, Any],
) -> tuple[str, dict[str, np.ndarray]]:
    """
    Calculate various distance measures for particles in a single cell.

    Computes pairwise distances, nearest neighbor distances, and distances
    to cellular structures (nucleus, membrane) for the given particle positions.

    Parameters
    ----------
    cell_id
        Identifier for the cell
    positions
        3D coordinates of particles in the cell, shape (N, 3)
    mesh_dict
        Dictionary containing mesh information for the cell including
        'nuc_mesh', 'mem_mesh', and 'cell_bounds'

    Returns
    -------
    :
        Tuple of (cell_id, distance_dict) where distance_dict contains arrays
        for 'pairwise', 'nearest', 'nucleus', 'scaled_nucleus', 'membrane', 'z', 'scaled_z'

    Raises
    ------
    ValueError
        If mesh information not found for the specified cell_id
    """
    # Shape dependent distance measures
    if cell_id not in mesh_dict:
        raise ValueError(f"Mesh information not found for cell_id: {cell_id}")

    nuc_mesh = mesh_dict[cell_id]["nuc_mesh"]
    mem_mesh = mesh_dict[cell_id]["mem_mesh"]

    mem_distances = proximity.signed_distance(mem_mesh, positions)
    nuc_surface_distances, scaled_nuc_distances, _ = calc_scaled_distance_to_nucleus_surface(
        positions,
        nuc_mesh,
        mem_mesh,
        mem_distances,
    )

    # Nuc and membrane distances
    nuc_distances = filter_invalid_distances(nuc_surface_distances)
    scaled_nuc_distances = filter_invalid_distances(scaled_nuc_distances)
    membrane_distances = filter_invalid_distances(mem_distances)

    num_points = len(positions)
    inside_nucleus_mask = nuc_surface_distances < 0
    outside_membrane_mask = mem_distances < 0
    num_inside_nucleus = inside_nucleus_mask.sum()
    num_outside_membrane = outside_membrane_mask.sum()
    fraction_inside_nucleus = num_inside_nucleus / num_points
    fraction_outside_membrane = num_outside_membrane / num_points
    fraction_non_cytoplasm = fraction_inside_nucleus + fraction_outside_membrane

    logger.debug(f"Fraction non-cytoplasm positions: {fraction_non_cytoplasm:.2f}")
    if fraction_non_cytoplasm > 0.1:
        logger.warning(
            f"More than 10% of positions are outside cytoplasm for cell {cell_id}: "
            f"{fraction_non_cytoplasm:.2f}\n"
            f"Inside nucleus: {num_inside_nucleus} / {num_points} = {fraction_inside_nucleus:.2f}, "
            f"Outside membrane: {num_outside_membrane} / {num_points} = "
            f"{fraction_outside_membrane:.2f}"
        )

    # Distance from z-surface
    z_min = mesh_dict[cell_id]["cell_bounds"][:, 2].min()
    z_distances = positions[:, 2] - z_min
    z_distances = filter_invalid_distances(z_distances)

    # Shape independent distance measures
    all_distances = cdist(positions, positions, metric="euclidean")

    # Pairwise distance
    pairwise_distances = squareform(all_distances)
    pairwise_distances = filter_invalid_distances(pairwise_distances)

    # Nearest neighbor distance - mask diagonal to exclude self-distances
    np.fill_diagonal(all_distances, np.inf)
    nearest_distances = np.min(all_distances, axis=1)
    nearest_distances = filter_invalid_distances(nearest_distances)
    distance_dict = {
        "pairwise": pairwise_distances,
        "nearest": nearest_distances,
        "nucleus": nuc_distances,
        "scaled_nucleus": scaled_nuc_distances,
        "membrane": membrane_distances,
        "z": z_distances,
    }

    return cell_id, distance_dict


def get_distance_dictionary(
    all_positions: dict[str, dict[str, np.ndarray]],
    distance_measures: list[str],
    mesh_information_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str] | None = None,
    results_dir: Path | None = None,
    recalculate: bool = False,
    num_workers: int | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Calculate or load distance measures between particles in different modes.

    Parameters
    ----------
    all_positions
        A dictionary containing positions of particles in different packing modes
    distance_measures
        List of distance measures to calculate
    mesh_information_dict
        A dictionary containing mesh information
    channel_map
        Mapping between modes and channel names
    results_dir
        The directory to save or load distance dictionaries
    recalculate
        If True, recalculate the distance measures
    num_workers
        Number of parallel workers to use for distance calculations
    minimum_distance
        Minimum distance to consider for filtering invalid distances

    Returns
    -------
    :
        A dictionary containing distance measures between particles in different modes
    """
    if channel_map is None:
        channel_map = {}

    # Try to load cached data if not recalculating
    if not recalculate and results_dir is not None:
        logger.info(
            f"Loading saved distance dictionaries from {results_dir.relative_to(PROJECT_ROOT)}"
        )
        all_distance_dict = {}
        cache_valid = True

        # Check if all required distance measure files exist and are valid
        for distance_measure in distance_measures:
            file_path = results_dir / f"{distance_measure}_distances.dat"
            if not file_path.exists():
                logger.warning(f"File not found: {file_path.relative_to(PROJECT_ROOT)}")
                cache_valid = False
                break

            try:
                with open(file_path, "rb") as f:
                    distance_dict = pickle.load(f)

                # Validate that cached data matches requested packing modes
                cached_modes = set(distance_dict.keys())
                requested_modes = set(all_positions.keys())
                if cached_modes != requested_modes:
                    logger.warning(
                        f"Cached data in {file_path.relative_to(PROJECT_ROOT)} contains modes "
                        f"{cached_modes} but requested modes are {requested_modes}. "
                        f"Recalculating distances."
                    )
                    cache_valid = False
                    break

                all_distance_dict[distance_measure] = distance_dict
                if not all([len(v) > 0 for v in distance_dict.values()]):
                    logger.warning(
                        f"Cached data in {file_path.relative_to(PROJECT_ROOT)} is empty. "
                        f"Recalculating distances."
                    )
                    cache_valid = False
                    break
            except Exception as e:
                logger.warning(
                    f"Error loading cached data from {file_path.relative_to(PROJECT_ROOT)}: {e}"
                )
                cache_valid = False
                break

        # Return cached data if all checks passed
        if cache_valid:
            logger.info("Successfully loaded all cached distance dictionaries")
            return all_distance_dict
        else:
            logger.info("Cache validation failed, recalculating distances")

    logger.info("Calculating distance dictionaries")

    all_distance_dict = {
        distance_measure: {mode: {} for mode in all_positions.keys()}
        for distance_measure in distance_measures
    }
    for mode, position_dict in all_positions.items():
        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _calculate_distances_for_cell_id,
                    str(cell_id).split("_")[0],
                    positions,
                    mode_mesh_dict,
                ): (
                    str(cell_id).split("_")[0]
                )
                for cell_id, positions in position_dict.items()
            }
            for future in tqdm(
                as_completed(futures),
                desc=f"Calculating distances for {MODE_LABELS.get(mode, mode)}",
                total=len(futures),
            ):
                cell_id = futures[future]
                try:
                    _, distances_dict = future.result()
                except Exception as e:
                    logger.error(f"Error calculating distances for cell {cell_id}: {e}")
                    continue

                for distance_measure in distance_measures:
                    if distance_measure not in distances_dict:
                        raise ValueError(f"Distance measure {distance_measure} not found")
                    all_distance_dict[distance_measure][mode][cell_id] = distances_dict[
                        distance_measure
                    ]

    # save distance dictionaries
    if results_dir is not None:
        for distance_measure, distance_dict in all_distance_dict.items():
            file_path = results_dir / f"{distance_measure}_distances.dat"
            with open(file_path, "wb") as f:
                pickle.dump(distance_dict, f)

    return all_distance_dict


def get_distance_dictionary_serial(
    all_positions: dict[str, dict[str, np.ndarray]],
    distance_measures: list[str],
    mesh_information_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str] | None = None,
    results_dir: Path | None = None,
    recalculate: bool = False,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Calculate or load distance measures between particles in different modes.

    Parameters
    ----------
    all_positions
        A dictionary containing positions of particles in different packing modes
    distance_measures
        List of distance measures to calculate
    mesh_information_dict
        A dictionary containing mesh information
    channel_map
        Mapping between modes and channel names
    results_dir
        The directory to save or load distance dictionaries
    recalculate
        If True, recalculate the distance measures

    Returns
    -------
    :
        A dictionary containing distance measures between particles in different modes
    """
    if channel_map is None:
        channel_map = {}

    # Try to load cached data if not recalculating
    if not recalculate and results_dir is not None:
        logger.info(
            f"Loading saved distance dictionaries from {results_dir.relative_to(PROJECT_ROOT)}"
        )
        all_distance_dict = {}
        cache_valid = True

        # Check if all required distance measure files exist and are valid
        for distance_measure in distance_measures:
            file_path = results_dir / f"{distance_measure}_distances.dat"
            if not file_path.exists():
                logger.warning(f"File not found: {file_path.relative_to(PROJECT_ROOT)}")
                cache_valid = False
                break

            try:
                with open(file_path, "rb") as f:
                    distance_dict = pickle.load(f)

                # Validate that cached data matches requested packing modes
                cached_modes = set(distance_dict.keys())
                requested_modes = set(all_positions.keys())
                if cached_modes != requested_modes:
                    logger.warning(
                        f"Cached data in {file_path.relative_to(PROJECT_ROOT)} contains modes "
                        f"{cached_modes} but requested modes are {requested_modes}. "
                        f"Recalculating distances."
                    )
                    cache_valid = False
                    break

                all_distance_dict[distance_measure] = distance_dict
            except Exception as e:
                logger.warning(
                    f"Error loading cached data from {file_path.relative_to(PROJECT_ROOT)}: {e}"
                )
                cache_valid = False
                break

        # Return cached data if all checks passed
        if cache_valid:
            logger.info("Successfully loaded all cached distance dictionaries")
            return all_distance_dict
        else:
            logger.info("Cache validation failed, recalculating distances")
    logger.info("Calculating distance dictionaries")

    all_distance_dict = {}
    for mode, position_dict in all_positions.items():
        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})

        for cell_id, positions in tqdm(
            position_dict.items(),
            desc=f"Calculating distances for {MODE_LABELS.get(mode, mode)}",
            total=len(position_dict),
        ):
            cell_id = str(cell_id).split("_")[0]
            _, distances_dict = _calculate_distances_for_cell_id(
                cell_id=cell_id,
                positions=positions,
                mesh_dict=mode_mesh_dict,
            )
            for distance_measure in distance_measures:
                if distance_measure not in distances_dict:
                    raise ValueError(f"Distance measure {distance_measure} not found")
                if distance_measure not in all_distance_dict:
                    all_distance_dict[distance_measure] = {}
                if mode not in all_distance_dict[distance_measure]:
                    all_distance_dict[distance_measure][mode] = {}
                all_distance_dict[distance_measure][mode][cell_id] = distances_dict[
                    distance_measure
                ]

    # save distance dictionaries
    if results_dir is not None:
        for distance_measure, distance_dict in all_distance_dict.items():
            file_path = results_dir / f"{distance_measure}_distances.dat"
            with open(file_path, "wb") as f:
                pickle.dump(distance_dict, f)

    return all_distance_dict


def get_ks_test_df(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    baseline_mode: str = "SLC25A17",
    significance_level: float = 0.05,
    save_dir: Path | None = None,
    recalculate: bool = True,
) -> pd.DataFrame:
    """
    Perform KS test between distance distributions of different packing modes and combine results.

    Parameters
    ----------
    distance_measures
        List of distance measures to compare
    packing_modes
        List of packing modes to compare
    all_distance_dict
        Dictionary containing distance distributions for each packing mode and distance measure
    baseline_mode
        The packing mode to use as the baseline for comparison
    significance_level
        Significance level for the KS test
    save_dir
        Directory to save the results
    recalculate
        Whether to recalculate even if results exist

    Returns
    -------
    :
        DataFrame containing the KS observed results, with columns for cell_id, distance_measure,
        packing_mode, and ks_observed
    """
    file_name = "ks_observed_combined_df.parquet"
    if not recalculate and save_dir is not None:
        file_path = save_dir / file_name
        if file_path.exists():
            logger.info(f"Loading saved KS DataFrame from {file_path.relative_to(PROJECT_ROOT)}")
            return pd.read_parquet(file_path)
    record_list = []
    for distance_measure in distance_measures:
        logger.info(f"Calculating KS observed for distance measure: {distance_measure}")

        # Collect all (mode, cell_id) combinations
        all_pairs = []
        for mode in packing_modes:
            if mode == baseline_mode:
                continue
            for cell_id in all_distance_dict[distance_measure][mode].keys():
                all_pairs.append((mode, cell_id))
        for mode, cell_id in tqdm(all_pairs, desc=f"KS tests for {distance_measure}"):
            distances_1 = all_distance_dict[distance_measure][baseline_mode].get(cell_id, None)
            distances_2 = all_distance_dict[distance_measure][mode].get(cell_id, None)
            if distances_1 is None or distances_2 is None:
                logger.warning(f"Missing distances for {mode}, {cell_id}, skipping KS test")
                continue
            ks_result = ks_2samp(distances_1, distances_2)
            ks_stat, p_value = ks_result.statistic, ks_result.pvalue  # type:ignore
            record_list.append(
                {
                    "distance_measure": distance_measure,
                    "packing_mode": mode,
                    "cell_id": cell_id,
                    "ks_stat": ks_stat,
                    "p_value": p_value,
                    "different": p_value < significance_level,
                    "similar": p_value >= significance_level,
                }
            )
    ks_test_df = pd.DataFrame.from_records(record_list)
    if save_dir is not None:
        file_path = save_dir / file_name
        ks_test_df.to_parquet(file_path, index=False)
        logger.info(f"Saved KS observed DataFrame to {file_path.relative_to(PROJECT_ROOT)}")

    return ks_test_df


def bootstrap_ks_tests(
    ks_test_df: pd.DataFrame,
    distance_measures: list[str],
    packing_modes: list[str],
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Perform bootstrap KS tests to estimate the distribution of KS statistics.

    Parameters
    ----------
    ks_test_df
        DataFrame containing the KS observed results
    distance_measures
        List of distance measures to analyze
    packing_modes
        List of packing modes to analyze
    n_bootstrap
        Number of bootstrap samples to generate

    Returns
    -------
    :
        DataFrame containing the bootstrap KS statistics with columns for distance_measure,
        packing_mode, and similar_fraction
    """
    record_list = []
    cell_ids = ks_test_df["cell_id"].unique()
    for exp_num in tqdm(range(n_bootstrap), desc="Bootstrapping KS tests"):
        sampled_cell_ids = np.random.choice(cell_ids, size=len(cell_ids), replace=True)
        for distance_measure in distance_measures:
            for packing_mode in packing_modes:
                mode_df = ks_test_df.query(
                    "distance_measure == @distance_measure and packing_mode == @packing_mode"
                )
                similar_fraction = np.mean(
                    mode_df[mode_df["cell_id"].isin(sampled_cell_ids)]["similar"].to_numpy()
                )
                record_list.append(
                    {
                        "distance_measure": distance_measure,
                        "packing_mode": packing_mode,
                        "experiment_number": exp_num,
                        "similar_fraction": similar_fraction,
                    }
                )

    bootstrap_df = pd.DataFrame.from_records(record_list)
    return bootstrap_df


def get_distance_distribution_emd_df(
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    packing_modes: list[str],
    distance_measures: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Calculate pairwise EMD between packing modes for each distance measure.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing distance measures for each packing mode
    packing_modes
        List of packing modes to calculate pairwise EMD for
    distance_measures
        List of distance measures to calculate pairwise EMD for
    results_dir
        Directory to save the EMD results
    recalculate
        Whether to recalculate the EMD even if results already exist
    suffix
        Suffix to add to the saved EMD file name

    Returns
    -------
    :
        DataFrame containing pairwise EMD for each distance measure
    """
    file_name = f"pairwise_emd{suffix}.parquet"
    if not recalculate and results_dir is not None:
        file_path = results_dir / file_name
        if file_path.exists():
            logger.info(f"Loading pairwise EMD from {file_path.relative_to(PROJECT_ROOT)}")
            return pd.read_parquet(file_path)

    record_list = []
    for distance_measure in distance_measures:
        logger.info("Calculating EMD for %s", distance_measure)

        # Collect all (mode, cell_id) combinations
        all_pairs = []
        for mode in packing_modes:
            for cell_id in all_distance_dict[distance_measure][mode].keys():
                all_pairs.append((mode, cell_id))

        for i in tqdm(range(len(all_pairs)), desc=f"EMD calculations for {distance_measure}"):
            mode_1, cell_id_1 = all_pairs[i]
            distances_1 = all_distance_dict[distance_measure][mode_1][cell_id_1]
            for j in range(i + 1, len(all_pairs)):
                mode_2, cell_id_2 = all_pairs[j]
                distances_2 = all_distance_dict[distance_measure][mode_2][cell_id_2]
                emd = wasserstein_distance(distances_1, distances_2)
                record_list.append(
                    {
                        "distance_measure": distance_measure,
                        "packing_mode_1": mode_1,
                        "packing_mode_2": mode_2,
                        "cell_id_1": cell_id_1,
                        "cell_id_2": cell_id_2,
                        "emd": emd,
                    }
                )
    df_emd = pd.DataFrame.from_records(record_list)
    if results_dir is not None:
        file_path = results_dir / file_name
        df_emd.to_parquet(file_path, index=False)
        logger.info(f"Saved pairwise EMD to {file_path.relative_to(PROJECT_ROOT)}")

    return df_emd


def calculate_ripley_k(
    all_positions: dict[str, dict[str, np.ndarray]],
    mesh_information_dict: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    """
    Calculate Ripley's K function for spatial point patterns.

    Parameters
    ----------
    all_positions
        Dictionary containing positions for each mode and cell_id
    mesh_information_dict
        Dictionary containing mesh information for each cell_id

    Returns
    -------
    :
        Tuple containing all Ripley K values, mean values, confidence intervals, and r values
    """
    all_ripley_k = {}
    mean_ripley_k = {}
    ci_ripley_k = {}
    r_max = 0.5
    num_bins = 100
    r_values = np.linspace(0, r_max, num_bins)
    for mode, position_dict in all_positions.items():
        logger.info(f"Calculating Ripley K for mode: {mode}")
        all_ripley_k[mode] = {}
        for cell_id, positions in tqdm(position_dict.items(), desc=f"Ripley K for {mode}"):
            radius = mesh_information_dict[cell_id]["cell_diameter"] / 2
            volume = 4 / 3 * np.pi * radius**3
            mean_k_values, _ = ripley_k(positions, volume, r_values, norm_factor=(radius * 2))
            all_ripley_k[mode][cell_id] = mean_k_values
        mean_ripley_k[mode] = np.mean(
            np.array([np.array(v, dtype=float) for v in all_ripley_k[mode].values()], dtype=float),
            axis=0,
        )
        ci_ripley_k[mode] = np.percentile(
            np.array([np.array(v, dtype=float) for v in all_ripley_k[mode].values()], dtype=float),
            [2.5, 97.5],
            axis=0,
        )

    return all_ripley_k, mean_ripley_k, ci_ripley_k, r_values


def get_normalization_factor(
    normalization: str | None,
    mesh_information_dict: dict[str, dict[str, Any]],
    cell_id: str,
    distances: np.ndarray | None = None,
    distance_measure: str = "nucleus",
    pix_size: float = PIXEL_SIZE_IN_UM,
) -> float:
    """
    Get the normalization factor for the distances based on the specified normalization method.

    Parameters
    ----------
    normalization
        Normalization method to use. Options are: "intracellular_radius", "cell_diameter",
        "max_distance", or None
    mesh_information_dict
        Dictionary containing mesh information for each cell_id
    cell_id
        cell_id/cell_id for which to get the normalization factor
    distances
        Array of distances for max_distance normalization
    distance_measure
        Distance measure being normalized
    pix_size
        Pixel size for the distances

    Returns
    -------
    :
        Normalization factor for the distances
    """
    if normalization == "intracellular_radius":
        normalization_factor = mesh_information_dict[cell_id]["intracellular_radius"]
    elif normalization == "cell_diameter":
        normalization_factor = mesh_information_dict[cell_id]["cell_diameter"]
    elif normalization == "max_distance" and distances is not None:
        # Get the maximum distance for the given distances
        if len(distances) == 0:
            raise ValueError(
                f"No valid distances found for cell_id {cell_id} and normalization {normalization}"
            )
        normalization_factor = np.nanmax(distances)
    elif "scaled" in distance_measure:
        normalization_factor = 1
    else:
        normalization_factor = 1 / pix_size

    if normalization_factor == 0 or np.isnan(normalization_factor):
        raise ValueError(
            f"Invalid normalization factor for cell_id {cell_id} and normalization {normalization}"
        )
    return normalization_factor


def get_distance_distribution_kde(
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    mesh_information_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str],
    save_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    normalization: str | None = None,
    distance_measure: str = "nucleus",
    minimum_distance: float | None = 0,
) -> dict[str, dict[str, gaussian_kde]]:
    """
    Obtain KDE for distance distribution measures.

    This function computes the KDE for each packing mode and cell_id,
    and saves the results to a file.
    If the results already exist and recalculate is set to False, the function will load
    the existing results. The KDE is calculated using the Gaussian kernel density estimation method.
    The available space distances are also calculated and stored in the output dictionary.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing distance measures for each packing mode
    mesh_information_dict
        Dictionary containing mesh information for each cell_id
    channel_map
        Dictionary mapping packing modes to their corresponding channel names
    save_dir
        Directory to save the KDE results
    packing_modes
        List of packing modes to calculate distance distribution KDE for
    results_dir
        Directory to save the KDE results
    recalculate
        Whether to recalculate the KDE even if results already exist
    suffix
        Suffix to add to the saved KDE file name
    normalization
        Normalization method to use for the distances
    distance_measure
        Distance measure to use for the KDE calculation
    bandwidth
        Bandwidth method for the Gaussian KDE
    minimum_distance
        Minimum distance to consider for KDE calculation

    Returns
    -------
    :
        Dictionary containing the KDE for each packing mode and cell_id with structure:
        {
            cell_id:
                {
                    mode: gaussian_kde_object,
                    "available_distance": gaussian_kde_object,
                }
        }
    """
    # Set file path for saving/loading KDE results
    save_file_path = None
    if save_dir is not None:
        filename = f"{distance_measure}_distance_distribution_kde{suffix}.dat"
        save_file_path = save_dir / filename

    # Try to load cached KDE data if not recalculating
    if not recalculate and save_file_path is not None and save_file_path.exists():
        try:
            with open(save_file_path, "rb") as f:
                kde_dict = pickle.load(f)

            # Validate that cached data matches requested modes
            cached_modes = set()
            for cell_data in kde_dict.values():
                cached_modes.update(
                    mode for mode in cell_data.keys() if mode != "available_distance"
                )

            requested_modes = set(channel_map.keys())
            if cached_modes != requested_modes:
                logger.warning(
                    f"Cached KDE data contains modes {cached_modes} but requested modes are "
                    f"{requested_modes}. Recalculating KDE."
                )
            else:
                logger.info(
                    f"Successfully loaded cached KDE data from "
                    f"{save_file_path.relative_to(PROJECT_ROOT)}"
                )
                return kde_dict
        except Exception as e:
            logger.warning(
                f"Error loading cached KDE data from "
                f"{save_file_path.relative_to(PROJECT_ROOT)}: {e}. Recalculating KDE."
            )

    # Initialize the KDE dictionary
    distance_dict = all_distance_dict[distance_measure]
    kde_dict = {}
    for mode, structure_id in channel_map.items():
        mode_mesh_dict = mesh_information_dict.get(structure_id, {})
        mode_distances_dict = distance_dict[mode]

        for cell_id, distances in tqdm(
            mode_distances_dict.items(), total=len(mode_distances_dict), desc=f"KDE for {mode}"
        ):
            # Get the distances for a cell
            # These are already normalized
            if cell_id not in kde_dict:
                kde_dict[cell_id] = {}
            distances = filter_invalid_distances(
                distances, minimum_distance=minimum_distance
            ).astype(np.float32)
            if len(distances) == 0:
                logger.warning(f"No valid distances found for cell id {cell_id} and mode {mode}")
                continue

            # Calculate the KDE for the distances
            kde_dict[cell_id][mode] = gaussian_kde(distances)

            # Update available distances from mesh information if needed
            if "available_distance" not in kde_dict[cell_id]:
                available_distances = mode_mesh_dict[cell_id][
                    GRID_DISTANCE_LABELS[distance_measure]
                ].flatten()

                normalization_factor = get_normalization_factor(
                    normalization=normalization,
                    mesh_information_dict=mode_mesh_dict,
                    cell_id=cell_id,
                    distance_measure=distance_measure,
                    distances=available_distances,
                )
                available_distances /= normalization_factor
                available_distances = filter_invalid_distances(
                    available_distances, minimum_distance=minimum_distance
                ).astype(np.float32)

                kde_dict[cell_id]["available_distance"] = gaussian_kde(available_distances)

    # save kde dictionary
    if save_file_path is not None:
        with open(save_file_path, "wb") as f:
            pickle.dump(kde_dict, f)

    return kde_dict


def get_scaled_structure_radius(
    structure_id: str,
    mesh_information_dict: dict[str, dict[str, Any]],
    normalization: str | None = None,
) -> tuple[float, float]:
    """
    Get the scaled structure radius based on the given structure ID.

    Parameters
    ----------
    structure_id
        The structure ID
    mesh_information_dict
        A dictionary containing mesh information
    normalization
        The normalization method to use

    Returns
    -------
    :
        Tuple containing the scaled structure radius and standard deviation
    """
    df_struct_stats = get_structure_stats_dataframe(structure_id=structure_id).set_index("CellId")

    scaled_radius_list = []
    for cell_id, mesh_info in mesh_information_dict.items():
        if cell_id not in df_struct_stats.index:
            continue
        normalization_factor = (
            mesh_info.get(normalization, 1 / PIXEL_SIZE_IN_UM)
            if normalization is not None
            else 1 / PIXEL_SIZE_IN_UM
        )
        scaled_radius_list.append(
            df_struct_stats.loc[cell_id, "radius"] / normalization_factor  # type:ignore
        )

    avg_radius = np.mean(scaled_radius_list).item()
    std_radius = np.std(scaled_radius_list).item()

    return avg_radius, std_radius


def log_central_tendencies_for_emd(
    df_emd: pd.DataFrame,
    distance_measures: list[str],
    packing_modes: list[str],
    baseline_mode: str = "mean_count_and_size",
    comparison_type: str = "intra_mode",
    log_file_path: Path | None = None,
) -> None:
    """
    Log central tendencies of EMD values for within-rule and baseline comparisons.

    Parameters
    ----------
    df_emd
        DataFrame containing EMD values with columns for distance_measure, packing_mode_1,
        packing_mode_2, and emd
    distance_measures
        List of distance measures to analyze
    packing_modes
        List of packing modes to analyze
    baseline_mode
        The packing mode to use as the baseline for comparisons
    comparison_type
        Type of comparison: 'intra_mode' or 'baseline'
    log_file_path
        Optional file path to save the log output
    """
    if log_file_path is not None:
        emd_logger = add_file_handler_to_logger(logger, log_file_path)
    else:
        emd_logger = logger

    emd_logger.info("Comparison type: %s", comparison_type)
    for distance_measure in distance_measures:
        emd_logger.info(f"Distance measure: {distance_measure}")
        for packing_mode in packing_modes:
            if comparison_type == "baseline":
                if packing_mode == baseline_mode:
                    continue
                label = f"{baseline_mode} vs {packing_mode}"
                sub_df = df_emd.loc[
                    (df_emd["distance_measure"] == distance_measure)
                    & (
                        (
                            (df_emd["packing_mode_1"] == baseline_mode)
                            & (df_emd["packing_mode_2"] == packing_mode)
                        )
                        | (
                            (df_emd["packing_mode_1"] == packing_mode)
                            & (df_emd["packing_mode_2"] == baseline_mode)
                        )
                    ),
                    "emd",
                ]
            elif comparison_type == "intra_mode":
                label = packing_mode
                sub_df = df_emd.loc[
                    (df_emd["distance_measure"] == distance_measure)
                    & (df_emd["packing_mode_1"] == packing_mode)
                    & (df_emd["packing_mode_2"] == packing_mode),
                    "emd",
                ]
            else:
                raise ValueError(f"Invalid comparison type: {comparison_type}")

            mean = sub_df.mean()
            std = sub_df.std()
            median = sub_df.median()
            lower = sub_df.quantile(0.025)
            upper = sub_df.quantile(0.975)
            emd_logger.info(
                f"{label}: {mean:.2f} ± {std:.2f} "
                f"(median: {median:.2f}, 95% CI: {lower:.2f}, {upper:.2f})"
            )

    remove_file_handler_from_logger(emd_logger, log_file_path)


def log_central_tendencies_for_ks(
    df_ks_bootstrap: pd.DataFrame,
    distance_measures: list[str],
    file_path: Path | None = None,
) -> None:
    """
    Log central tendencies of KS test results.

    Parameters
    ----------
    df_ks_bootstrap
        DataFrame containing bootstrap KS test results with columns for distance_measure,
        packing_mode, and similar_fraction
    distance_measures
        List of distance measures to analyze
    file_path
        Optional file path to save the log output
    """
    if file_path is not None:
        ks_logger = add_file_handler_to_logger(logger, file_path)
    else:
        ks_logger = logger

    for distance_measure in distance_measures:
        ks_logger.info(f"Distance measure: {distance_measure}")
        sub_df = df_ks_bootstrap.loc[df_ks_bootstrap["distance_measure"] == distance_measure]
        for packing_mode in sub_df["packing_mode"].unique():
            mode_df = sub_df.loc[sub_df["packing_mode"] == packing_mode]
            mean = mode_df["similar_fraction"].mean()
            std = mode_df["similar_fraction"].std()
            median = mode_df["similar_fraction"].median()
            lower = mode_df["similar_fraction"].quantile(0.025)
            upper = mode_df["similar_fraction"].quantile(0.975)
            logger.info(
                f"{packing_mode}: {mean:.2f} ± {std:.2f} "
                f"(median: {median:.2f}, 95% CI: {lower:.2f}, {upper:.2f})"
            )

    remove_file_handler_from_logger(ks_logger, file_path)


def log_central_tendencies_for_distance_distributions(
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    distance_measures: list[str],
    packing_modes: list[str],
    file_path: Path | None = None,
    minimum_distance: float | None = 0,
) -> None:
    """
    Log central tendencies of distance distributions.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing distance measures for each packing mode
    distance_measures
        List of distance measures to analyze
    packing_modes
        List of packing modes to analyze
    file_path
        Optional file path to save the log output
    minimum_distance
        Minimum distance to consider for filtering invalid distances
    """
    if file_path is not None:
        dist_logger = add_file_handler_to_logger(logger, file_path)
    else:
        dist_logger = logger

    for distance_measure in distance_measures:
        dist_logger.info(f"Distance measure: {distance_measure}")
        distance_dict = all_distance_dict[distance_measure]
        for packing_mode in packing_modes:
            mode_dict = distance_dict[packing_mode]
            all_distances = np.concatenate(list(mode_dict.values()))
            all_distances = filter_invalid_distances(
                all_distances, minimum_distance=minimum_distance
            )
            if len(all_distances) == 0:
                dist_logger.warning(f"No valid distances found for mode {packing_mode}")
                continue
            mean = np.mean(all_distances)
            std = np.std(all_distances)
            median = np.median(all_distances)
            lower = np.percentile(all_distances, 2.5)
            upper = np.percentile(all_distances, 97.5)
            dist_logger.info(
                f"{packing_mode}: {mean:.2f} ± {std:.2f} "
                f"(median: {median:.2f}, 95% CI: {lower:.2f}, {upper:.2f})"
            )

    remove_file_handler_from_logger(dist_logger, file_path)
