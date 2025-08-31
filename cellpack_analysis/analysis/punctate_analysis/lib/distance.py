import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rtree.exceptions import RTreeError
from scipy.spatial.distance import cdist, squareform
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm
from trimesh import proximity

from cellpack_analysis.analysis.punctate_analysis.lib.stats_functions import ripley_k
from cellpack_analysis.lib.default_values import PIX_SIZE
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.label_tables import (
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
    MODE_LABELS,
    STATIC_SHAPE_MODES,
)

log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 16})

PROJECT_ROOT = get_project_root()


def calc_scaled_distance_to_nucleus_surface(
    position_list: np.ndarray,
    nuc_mesh: Any,
    mem_mesh: Any,
    mem_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the scaled distance of each point in position_list to the nucleus surface.

    Parameters
    ----------
    position_list
        A list of 3D coordinates of points
    nuc_mesh
        A trimesh object representing the nucleus surface
    mem_mesh
        A trimesh object representing the membrane surface
    mem_distances
        Pre-computed distances to membrane surface

    Returns
    -------
    :
        Tuple containing nucleus surface distances, scaled nucleus distances,
        and distance between surfaces
    """
    if mem_distances is None:
        mem_distances = np.array(proximity.signed_distance(mem_mesh, position_list))

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
    ) in enumerate(zip(position_list, nuc_surface_distances, mem_distances, strict=False)):
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
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_nuc_distances = np.divide(nuc_surface_distances, distance_between_surfaces)
    scaled_nuc_distances[failed_inds] = np.nan
    scaled_nuc_distances[(scaled_nuc_distances < 0) | (scaled_nuc_distances > 1)] = np.nan

    return nuc_surface_distances, scaled_nuc_distances, distance_between_surfaces


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


def get_ks_observed_combined_df(
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
            log.info(f"Loading saved KS observed DataFrame from {file_path}")
            return pd.read_parquet(file_path)
    df_list = []
    for distance_measure in distance_measures:
        distance_dict = all_distance_dict[distance_measure]
        ks_observed = {}
        for mode in packing_modes:
            if mode == baseline_mode:
                continue
            log.info(f"KS test between {baseline_mode} and {mode}, distance: {distance_measure}")
            mode_dict = distance_dict[mode]
            ks_observed[mode] = {}
            for seed, distances in tqdm(mode_dict.items(), total=len(mode_dict)):
                observed_distances = distance_dict[baseline_mode][seed]
                _, p_val = ks_2samp(
                    observed_distances,
                    distances,
                )
                ks_observed[mode][seed] = p_val
        ks_observed_df = pd.DataFrame(ks_observed)
        ks_observed_df = ks_observed_df >= significance_level
        ks_observed_df["distance_measure"] = distance_measure
        ks_observed_df = ks_observed_df.reset_index().rename(columns={"index": "cell_id"})
        df_list.append(ks_observed_df)
    ks_observed_combined_df = pd.concat(df_list, ignore_index=True)

    if save_dir is not None:
        save_path = save_dir / file_name
        log.info(f"Saving KS observed DataFrame to {save_path}")
        ks_observed_combined_df.to_parquet(save_path, index=False)

    return ks_observed_combined_df


def melt_df_for_plotting(df_plot: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the DataFrame for plotting.

    Parameters
    ----------
    df_plot
        DataFrame containing the KS observed results with columns for cell_id, distance_measure,
        and packing modes

    Returns
    -------
    :
        Melted DataFrame with columns for cell_id, distance_measure, packing_mode, and ks_observed
    """
    df_melt = df_plot.melt(
        id_vars=["cell_id", "distance_measure"],
        var_name="packing_mode",
        value_name="ks_observed",
    )
    # relabel values
    df_melt["packing_mode"] = (
        df_melt["packing_mode"].map(MODE_LABELS).fillna(df_melt["packing_mode"])
    )
    df_melt["distance_measure"] = (
        df_melt["distance_measure"].map(DISTANCE_MEASURE_LABELS).fillna(df_melt["distance_measure"])
    )
    return df_melt


def get_pairwise_wasserstein_distance_dict(
    distribution_dict_1: dict[str, np.ndarray],
    distribution_dict_2: dict[str, np.ndarray] | None = None,
) -> dict[tuple[str, str], float]:
    """
    Calculate pairwise Wasserstein distances between distributions.

    Parameters
    ----------
    distribution_dict_1
        Dictionary with distances or other values for multiple seeds
    distribution_dict_2
        Optional second dictionary for cross-comparisons

    Returns
    -------
    :
        Dictionary with pairwise Wasserstein distances
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
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    packing_modes: list[str],
    distance_measures: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    baseline_mode: str | None = None,
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
    baseline_mode
        Baseline packing mode to compare against
    suffix
        Suffix to add to the saved EMD file name

    Returns
    -------
    :
        DataFrame containing pairwise EMD for each distance measure
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
    index = pd.MultiIndex.from_tuples(index_tuples, names=["distance_measure", "mode", "cell_id"])
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
    all_distance_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    packing_modes: list[str],
    distance_measures: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    baseline_mode: str | None = None,
    suffix: str = "",
) -> dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]]:
    """
    Calculate pairwise EMD dictionary between packing modes for each distance measure.

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
    baseline_mode
        Baseline packing mode to compare against
    suffix
        Suffix to add to the saved EMD file name

    Returns
    -------
    :
        Dictionary containing pairwise EMD for each distance measure
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
        log.info(f"Calculating distance distribution EMD for distance measure: {distance_measure}")
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


def get_average_emd_correlation(
    distance_measures: list[str],
    all_pairwise_emd: dict[str, dict[str, dict[str, dict[tuple[str, str], float]]]],
    baseline_mode: str = "SLC25A17",
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Calculate average and standard deviation of EMD values.

    Parameters
    ----------
    distance_measures
        List of distance measures to analyze
    all_pairwise_emd
        Dictionary containing pairwise EMD values
    baseline_mode
        Baseline packing mode for normalization

    Returns
    -------
    :
        Dictionary containing mean and std DataFrames for each distance measure
    """
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
                if not pd.isna(df_corr.loc[mode_1, mode_2]):
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
            np.array([np.array(v) for v in all_ripleyK[mode].values()]), axis=0
        )
        ci_ripleyK[mode] = np.percentile(
            np.array([np.array(v) for v in all_ripleyK[mode].values()]),
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
    packing_modes: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    normalization: str | None = None,
    distance_measure: str = "nucleus",
    bandwidth: Literal["scott", "silverman"] | float = "scott",
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Calculate the kernel density estimation (KDE) for a given distance measure.

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
                    mode:
                        {
                            "distances": distances,
                            "kde": kde
                        },
                    "available_distance":
                        {
                            "distances": available_distances,
                            "kde": kde_available_space
                        }
                    }
                }
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

        for seed, distances in tqdm(mode_distances_dict.items(), total=len(mode_distances_dict)):
            # Get the distances for a seed/cell_id
            # These are already normalized
            distances = filter_invalid_distances(distances)
            if len(distances) == 0:
                log.warning(f"No valid distances found for seed {seed} and mode {mode}")
                continue

            # Update available distances from mesh information if needed
            if seed not in kde_dict:
                kde_dict[seed] = {}
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

                kde_available_space = gaussian_kde(available_distances, bw_method=bandwidth)
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
            mesh_info.get(normalization, 1 / PIX_SIZE)
            if normalization is not None
            else 1 / PIX_SIZE
        )
        scaled_radius_list.append(
            df_struct_stats.loc[seed, "radius"] / normalization_factor  # type:ignore
        )

    avg_radius = np.mean(scaled_radius_list).item()
    std_radius = np.std(scaled_radius_list).item()

    return avg_radius, std_radius
