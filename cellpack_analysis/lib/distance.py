import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
from cellpack_analysis.lib.label_tables import GRID_DISTANCE_LABELS, MODE_LABELS, STATIC_SHAPE_MODES
from cellpack_analysis.lib.mesh_tools import calc_scaled_distance_to_nucleus_surface
from cellpack_analysis.lib.stats_functions import ripley_k

log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 16})

PROJECT_ROOT = get_project_root()


def filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict: dict[Any, Any],
    filter_negatives: bool = True,
) -> dict[Any, Any]:
    """
    Filters out invalid values from the distance distribution dictionary.

    Parameters
    ----------
    distance_distribution_dict
        Dictionary containing distance distributions with the form:
        {distance_measure: {packing_mode: {seed: np.ndarray of distances}}}
    filter_negatives
        If True, filter out negative distances

    Returns
    -------
    :
        Filtered distance distribution dictionary
    """
    for distance_measure, distance_measure_dict in distance_distribution_dict.items():
        for mode, mode_dict in distance_measure_dict.items():
            for seed, distances in mode_dict.items():
                # filter out NaN and inf values
                mode_dict[seed] = filter_invalid_distances(
                    distances, filter_negatives=filter_negatives
                )
                num_valid = len(mode_dict[seed])
                if num_valid == 0:
                    log.warning(f"All distances are invalid for {distance_measure}, {mode}, {seed}")
                    del mode_dict[seed]
                else:
                    log.debug(
                        f"Filtered {len(distances) - num_valid} values from "
                        f"{distance_measure}, {mode}, {seed}"
                    )
    return distance_distribution_dict


def filter_invalid_distances(distances: np.ndarray, filter_negatives: bool = True) -> np.ndarray:
    """
    Remove nan, inf and negative distances.

    Parameters
    ----------
    distances
        A numpy array of distances to filter
    filter_negatives
        If True, filter out negative distances

    Returns
    -------
    :
        A numpy array of filtered distances
    """
    condition = ~np.isnan(distances) & ~np.isinf(distances)
    if filter_negatives:
        condition &= distances > 0
    return distances[condition]


def get_distance_dictionary(
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
    if not recalculate and results_dir is not None:
        # load saved distance dictionary
        log.info(
            f"Loading saved distance dictionaries from {results_dir.relative_to(PROJECT_ROOT)}"
        )
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
    log.info("Calculating distance dictionaries")
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
            if mode not in STATIC_SHAPE_MODES:
                seed_to_use = seed.split("_")[0]
                shape_key = seed_to_use
            else:
                seed_to_use = str(seed)
                shape_key = seed.split("_")[0]

            all_distances = cdist(positions, positions, metric="euclidean")

            # Distance from the nucleus surface
            if shape_key not in mode_mesh_dict:
                raise ValueError(f"Mesh information not found for cell_id: {shape_key}")

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
            log.debug(f"Fraction bad inds: {1 - np.sum(good_inds) / len(good_inds):.2f}")
            all_nuc_distances[mode][seed_to_use] = filter_invalid_distances(
                nuc_surface_distances[good_inds]
            )
            all_scaled_nuc_distances[mode][seed_to_use] = filter_invalid_distances(
                scaled_nuc_distances[good_inds]
            )
            all_membrane_distances[mode][seed_to_use] = filter_invalid_distances(
                mem_distances[good_inds]
            )

            # Nearest neighbor distance
            nearest_distances = np.min(all_distances + np.eye(len(positions)) * 1e6, axis=1)
            all_nearest_distances[mode][seed_to_use] = filter_invalid_distances(nearest_distances)

            # Pairwise distance
            pairwise_distances = squareform(all_distances)
            all_pairwise_distances[mode][seed_to_use] = filter_invalid_distances(pairwise_distances)

            # Z distance
            z_min = mode_mesh_dict[shape_key]["cell_bounds"][:, 2].min()
            z_distances = positions[:, 2] - z_min
            all_z_distances[mode][seed_to_use] = filter_invalid_distances(z_distances)

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
            strict=False,
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
            log.info(f"Loading saved KS DataFrame from {file_path.relative_to(PROJECT_ROOT)}")
            return pd.read_parquet(file_path)
    record_list = []
    for distance_measure in distance_measures:
        log.info(f"Calculating KS observed for distance measure: {distance_measure}")

        # Collect all (mode, seed) combinations
        all_pairs = []
        for mode in packing_modes:
            if mode == baseline_mode:
                continue
            for seed in all_distance_dict[distance_measure][mode].keys():
                all_pairs.append((mode, seed))
        for mode, seed in tqdm(all_pairs):
            distances_1 = all_distance_dict[distance_measure][baseline_mode].get(seed, None)
            distances_2 = all_distance_dict[distance_measure][mode].get(seed, None)
            if distances_1 is None or distances_2 is None:
                log.warning(f"Missing distances for {mode}, {seed}, skipping KS test")
                continue
            ks_result = ks_2samp(distances_1, distances_2)
            ks_stat, p_value = ks_result.statistic, ks_result.pvalue  # type:ignore
            record_list.append(
                {
                    "distance_measure": distance_measure,
                    "packing_mode": mode,
                    "cell_id": seed,
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
        log.info(f"Saved KS observed DataFrame to {file_path.relative_to(PROJECT_ROOT)}")

    return ks_test_df


def bootstrap_ks_tests(
    ks_test_df: pd.DataFrame,
    distance_measures: list[str],
    packing_modes: list[str],
    n_bootstrap: int = 1000,
):
    """
    Perform bootstrap KS tests to estimate the distribution of KS statistics.

    Parameters
    ----------
    ks_test_df
        DataFrame containing the KS observed results
    n_bootstrap
        Number of bootstrap samples to generate

    Returns
    -------
    :
        DataFrame containing the bootstrap KS statistics with columns for distance_measure,
        packing_mode, and ks_stat
    """
    record_list = []
    cell_ids = ks_test_df["cell_id"].unique()
    for exp_num in tqdm(range(n_bootstrap)):
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
            log.info(f"Loading pairwise EMD from {file_path.relative_to(PROJECT_ROOT)}")
            return pd.read_parquet(file_path)

    record_list = []
    for distance_measure in distance_measures:
        log.info("Calculating EMD for %s", distance_measure)

        # Collect all (mode, seed) combinations
        all_pairs = []
        for mode in packing_modes:
            for seed in all_distance_dict[distance_measure][mode].keys():
                all_pairs.append((mode, seed))

        for i in tqdm(range(len(all_pairs))):
            mode_1, seed_1 = all_pairs[i]
            distances_1 = all_distance_dict[distance_measure][mode_1][seed_1]
            for j in range(i + 1, len(all_pairs)):
                mode_2, seed_2 = all_pairs[j]
                distances_2 = all_distance_dict[distance_measure][mode_2][seed_2]
                emd = wasserstein_distance(distances_1, distances_2)
                record_list.append(
                    {
                        "distance_measure": distance_measure,
                        "packing_mode_1": mode_1,
                        "packing_mode_2": mode_2,
                        "seed_1": seed_1,
                        "seed_2": seed_2,
                        "emd": emd,
                    }
                )
    df_emd = pd.DataFrame.from_records(record_list)
    if results_dir is not None:
        file_path = results_dir / file_name
        df_emd.to_parquet(file_path, index=False)
        log.info(f"Saved pairwise EMD to {file_path.relative_to(PROJECT_ROOT)}")

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
        Dictionary containing positions for each mode and seed
    mesh_information_dict
        Dictionary containing mesh information for each seed

    Returns
    -------
    :
        Tuple containing all Ripley K values, mean values, confidence intervals, and r values
    """
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
            mean_k_values, _ = ripley_k(positions, volume, r_values, norm_factor=(radius * 2))
            all_ripleyK[mode][seed] = mean_k_values
        mean_ripleyK[mode] = np.mean(
            np.array([np.array(v, dtype=float) for v in all_ripleyK[mode].values()], dtype=float),
            axis=0,
        )
        ci_ripleyK[mode] = np.percentile(
            np.array([np.array(v, dtype=float) for v in all_ripleyK[mode].values()], dtype=float),
            [2.5, 97.5],
            axis=0,
        )

    return all_ripleyK, mean_ripleyK, ci_ripleyK, r_values


def get_normalization_factor(
    normalization: str | None,
    mesh_information_dict: dict[str, dict[str, Any]],
    cell_id: str,
    distances: np.ndarray | None = None,
    distance_measure: str = "nucleus",
    pix_size: float = 0.108,
) -> float:
    """
    Get the normalization factor for the distances based on the specified normalization method.

    Parameters
    ----------
    normalization
        Normalization method to use. Options are: "intracellular_radius", "cell_diameter",
        "max_distance", or None
    mesh_information_dict
        Dictionary containing mesh information for each seed
    cell_id
        Seed/cell_id for which to get the normalization factor
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
                f"No valid distances found for seed {cell_id} and normalization {normalization}"
            )
        normalization_factor = np.nanmax(distances)
    elif "scaled" in distance_measure:
        normalization_factor = 1
    else:
        normalization_factor = 1 / pix_size

    if normalization_factor == 0 or np.isnan(normalization_factor):
        raise ValueError(
            f"Invalid normalization factor for seed {cell_id} and normalization {normalization}"
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
) -> dict[str, dict[str, gaussian_kde]]:
    """
    Obtain KDE for distance distribution measures

    This function computes the KDE for each packing mode and seed, and saves the results to a file.
    If the results already exist and recalculate is set to False, the function will load
    the existing results. The KDE is calculated using the Gaussian kernel density estimation method.
    The available space distances are also calculated and stored in the output dictionary.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing distance measures for each packing mode
    mesh_information_dict
        Dictionary containing mesh information for each seed
    channel_map
        Dictionary mapping packing modes to their corresponding channel names
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

    Returns
    -------
    :
        Dictionary containing the KDE for each packing mode and seed with structure:
        {
            seed:
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
    for mode, structure_id in channel_map.items():
        log.info(f"Calculating distance distribution kde for {mode}")

        mode_mesh_dict = mesh_information_dict.get(structure_id, {})
        mode_distances_dict = distance_dict[mode]

        for seed, distances in tqdm(mode_distances_dict.items(), total=len(mode_distances_dict)):
            # Get the distances for a seed/cell_id
            # These are already normalized
            if seed not in kde_dict:
                kde_dict[seed] = {}
            distances = filter_invalid_distances(distances).astype(np.float32)
            if len(distances) == 0:
                log.warning(f"No valid distances found for seed {seed} and mode {mode}")
                continue

            # Calculate the KDE for the distances
            kde_dict[seed][mode] = gaussian_kde(distances)

            # Update available distances from mesh information if needed
            if "available_distance" not in kde_dict[seed]:
                available_distances = (
                    mode_mesh_dict[seed][GRID_DISTANCE_LABELS[distance_measure]]
                    .flatten()
                    .astype(np.float32)
                )
                available_distances = filter_invalid_distances(available_distances)

                normalization_factor = get_normalization_factor(
                    normalization=normalization,
                    mesh_information_dict=mode_mesh_dict,
                    cell_id=seed,
                    distance_measure=distance_measure,
                    distances=available_distances,
                )
                available_distances /= normalization_factor

                kde_dict[seed]["available_distance"] = gaussian_kde(available_distances)

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
    df_struct_stats = get_structure_stats_dataframe(structure_id=structure_id)

    scaled_radius_list = []
    for seed, mesh_info in mesh_information_dict.items():
        if seed not in df_struct_stats.index:
            continue
        normalization_factor = (
            mesh_info.get(normalization, 1 / PIXEL_SIZE_IN_UM)
            if normalization is not None
            else 1 / PIXEL_SIZE_IN_UM
        )
        scaled_radius_list.append(
            df_struct_stats.loc[seed, "radius"] / normalization_factor  # type:ignore
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
):
    """
    Print central tendencies of EMD values for within-rule and baseline comparisons.

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
    """
    if log_file_path is not None:
        emd_log = add_file_handler_to_logger(log, log_file_path)
    else:
        emd_log = log

    emd_log.info("Comparison type: %s", comparison_type)
    for distance_measure in distance_measures:
        emd_log.info(f"Distance measure: {distance_measure}")
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
            emd_log.info(
                f"{label}: {mean:.2f} ± {std:.2f} "
                f"(median: {median:.2f}, 95% CI: {lower:.2f}, {upper:.2f})"
            )

    remove_file_handler_from_logger(emd_log, log_file_path)


def log_central_tendencies_for_ks(
    df_ks_bootstrap: pd.DataFrame,
    distance_measures: list[str],
    file_path: Path | None = None,
):
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
        ks_log = add_file_handler_to_logger(log, file_path)
    else:
        ks_log = log

    for distance_measure in distance_measures:
        ks_log.info(f"Distance measure: {distance_measure}")
        sub_df = df_ks_bootstrap.loc[df_ks_bootstrap["distance_measure"] == distance_measure]
        for packing_mode in sub_df["packing_mode"].unique():
            mode_df = sub_df.loc[sub_df["packing_mode"] == packing_mode]
            mean = mode_df["similar_fraction"].mean()
            std = mode_df["similar_fraction"].std()
            median = mode_df["similar_fraction"].median()
            lower = mode_df["similar_fraction"].quantile(0.025)
            upper = mode_df["similar_fraction"].quantile(0.975)
            log.info(
                f"{packing_mode}: {mean:.2f} ± {std:.2f} "
                f"(median: {median:.2f}, 95% CI: {lower:.2f}, {upper:.2f})"
            )

    remove_file_handler_from_logger(ks_log, file_path)


def log_central_tendencies_for_distance_distributions(
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    distance_measures: list[str],
    packing_modes: list[str],
    file_path: Path | None = None,
):
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
    """
    if file_path is not None:
        dist_log = add_file_handler_to_logger(log, file_path)
    else:
        dist_log = log

    for distance_measure in distance_measures:
        dist_log.info(f"Distance measure: {distance_measure}")
        distance_dict = all_distance_dict[distance_measure]
        for packing_mode in packing_modes:
            mode_dict = distance_dict[packing_mode]
            all_distances = np.concatenate(list(mode_dict.values()))
            all_distances = filter_invalid_distances(all_distances)
            if len(all_distances) == 0:
                dist_log.warning(f"No valid distances found for mode {packing_mode}")
                continue
            mean = np.mean(all_distances)
            std = np.std(all_distances)
            median = np.median(all_distances)
            lower = np.percentile(all_distances, 2.5)
            upper = np.percentile(all_distances, 97.5)
            dist_log.info(
                f"{packing_mode}: {mean:.2f} ± {std:.2f} "
                f"(median: {median:.2f}, 95% CI: {lower:.2f}, {upper:.2f})"
            )

    remove_file_handler_from_logger(dist_log, file_path)
