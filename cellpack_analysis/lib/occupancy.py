import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.load_data import PROJECT_ROOT
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
    x_min: float | None = 0,
    x_max: float | None = None,
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
    x_min
        Minimum x value for occupancy evaluation. Default is 0
    x_max
        Maximum x value for occupancy evaluation. Default is None

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

    # Try to load cached occupancy data if not recalculating
    if not recalculate and save_path is not None and save_path.exists():
        try:
            with open(save_path, "rb") as f:
                kde_occupancy_dict = pickle.load(f)

            # Validate that cached data matches requested modes
            cached_modes = set(kde_occupancy_dict.keys())
            requested_modes = set(channel_map.keys())
            if cached_modes != requested_modes:
                logger.warning(
                    f"Cached occupancy data contains modes {cached_modes} but requested modes are "
                    f"{requested_modes}. Recalculating occupancy."
                )
            else:
                logger.info(f"Successfully loaded cached occupancy data from {save_path}")
                return kde_occupancy_dict
        except Exception as e:
            logger.warning(
                f"Error loading cached occupancy data from {save_path}: {e}. "
                f"Recalculating occupancy."
            )

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

    x_min_calc = np.inf
    x_max_calc = -np.inf
    for structure_id, cell_ids in cell_id_map.items():
        combined_available_distances = []
        for cell_id in cell_ids:
            combined_available_distances.extend(
                distance_kde_dict[cell_id]["available_distance"].dataset
            )
            for mode in channel_map.keys():
                if mode in distance_kde_dict[cell_id]:
                    x_min_calc = min(x_min_calc, np.min(distance_kde_dict[cell_id][mode].dataset))
                    x_max_calc = max(x_max_calc, np.max(distance_kde_dict[cell_id][mode].dataset))
        combined_available_distance_kde[structure_id] = gaussian_kde(
            np.concatenate(combined_available_distances), bw_method=bandwidth
        )
    if x_min is None:
        x_min = x_min_calc
    if x_max is None:
        x_max = x_max_calc
    # Create xvals for evaluation
    x_vals = np.linspace(x_min, x_max, num_points)
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
            cell_xvals = np.linspace(
                np.min(occupied_distances), np.max(occupied_distances), num_points
            )

            available_kde = distance_kde_dict[cell_id]["available_distance"]
            if bandwidth is not None:
                occupied_kde.set_bandwidth(bandwidth)
                available_kde.set_bandwidth(bandwidth)

            # Evaluate KDEs on x_vals
            pdf_occupied = normalize_pdf(cell_xvals, occupied_kde.evaluate(cell_xvals))
            pdf_available = normalize_pdf(cell_xvals, available_kde.evaluate(cell_xvals))
            occupancy, pdf_occupied, pdf_available = pdf_ratio(
                cell_xvals, pdf_occupied, pdf_available
            )
            kde_occupancy_dict[mode]["individual"][cell_id] = {
                "xvals": cell_xvals,
                "occupancy": occupancy,
                "pdf_occupied": pdf_occupied,
                "pdf_available": pdf_available,
            }

        combined_occupied_distances = np.concatenate(combined_occupied_distances)

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


def get_occupancy_ks_test_df(
    distance_measures: list[str],
    packing_modes: list[str],
    combined_occupancy_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
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
    combined_occupancy_dict
        Dictionary containing occupancy distributions for each packing mode and distance measure
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
    file_name = "occupancy_ks_observed_combined_df.parquet"
    if not recalculate and save_dir is not None:
        file_path = save_dir / file_name
        if file_path.exists():
            logger.info(f"Loading saved KS DataFrame from {file_path.relative_to(PROJECT_ROOT)}")
            return pd.read_parquet(file_path)
    record_list = []
    for distance_measure in distance_measures:
        occupancy_dict = combined_occupancy_dict[distance_measure]
        logger.info(f"Calculating KS observed for distance measure: {distance_measure}")

        # Collect all (mode, cell_id) combinations
        all_pairs = []
        for mode in packing_modes:
            if mode == baseline_mode:
                continue
            for cell_id in occupancy_dict[mode]["individual"].keys():
                all_pairs.append((mode, cell_id))

        for mode, cell_id in tqdm(all_pairs, desc=f"KS tests for {distance_measure}"):
            if (
                cell_id not in occupancy_dict[baseline_mode]["individual"]
                or cell_id not in occupancy_dict[mode]["individual"]
            ):
                logger.warning(f"Cell ID {cell_id} does not match in both modes, skipping KS test")
                continue
            occupancy_1 = occupancy_dict[baseline_mode]["individual"][cell_id]["occupancy"]
            occupancy_2 = occupancy_dict[mode]["individual"][cell_id]["occupancy"]
            ks_result = ks_2samp(occupancy_1, occupancy_2)
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


def get_occupancy_emd_df(
    combined_occupancy_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    packing_modes: list[str],
    distance_measures: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Calculate pairwise EMD between packing mode occupancy for each distance measure.

    Parameters
    ----------
    combined_occupancy_dict
        Dictionary containing occupancy measures for each packing mode
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
        DataFrame containing occupancy EMD for each distance measure
    """
    file_name = f"occupancy_emd{suffix}.parquet"
    if not recalculate and results_dir is not None:
        file_path = results_dir / file_name
        if file_path.exists():
            logger.info(f"Loading occupancy EMD from {file_path.relative_to(PROJECT_ROOT)}")
            return pd.read_parquet(file_path)

    record_list = []
    for distance_measure in distance_measures:
        occupancy_dict = combined_occupancy_dict[distance_measure]
        logger.info("Calculating EMD for %s", distance_measure)
        # Collect all (mode, cell_id) combinations
        all_pairs = []
        for mode in packing_modes:
            for cell_id in occupancy_dict[mode]["individual"].keys():
                all_pairs.append((mode, cell_id))

        for i in tqdm(range(len(all_pairs)), desc=f"EMD calculations for {distance_measure}"):
            mode_1, cell_id_1 = all_pairs[i]
            occupancy_1 = occupancy_dict[mode_1]["individual"][cell_id_1]["occupancy"]
            for j in range(i + 1, len(all_pairs)):
                mode_2, cell_id_2 = all_pairs[j]
                occupancy_2 = occupancy_dict[mode_2]["individual"][cell_id_2]["occupancy"]
                emd = wasserstein_distance(occupancy_1, occupancy_2)
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
    occupancy_dict: dict[str, dict[str, dict[str, dict[str, Any]]]],
    channel_map: dict[str, str],
    baseline_mode: str,
    results_dir: Path | None = None,
    suffix: str = "",
) -> dict[str, Any]:
    """
    Interpolate occupancy data using non-negative least squares fitting.

    Performs two types of interpolation:
    1. Individual: Separate optimization for each distance measure
    2. Joint: Combined optimization across all distance measures

    Parameters
    ----------
    occupancy_dict
        Dictionary containing occupancy data for each packing mode
        Has the structure:
        {distance_measure:{mode:{"individual":{ ... },"combined": { ... }}}}
    channel_map
        Mapping from packing modes to structure IDs
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
    # Validate that baseline mode exists in all distance measures
    for distance_measure in occupancy_dict:
        if baseline_mode not in occupancy_dict[distance_measure]:
            raise ValueError(
                f"Baseline mode '{baseline_mode}' not found in distance measure "
                f"'{distance_measure}'"
            )

    # Get all packing modes (excluding baseline) from first distance measure
    packing_modes = [
        mode for mode in channel_map.keys() if mode not in [baseline_mode, "interpolated"]
    ]

    # Initialize result dictionary
    interp_dict = {
        "occupancy": {},
        "interpolation": {
            "individual": {},
            "joint": {},
        },
    }

    # Stack data across all distance measures for joint optimization
    stacked_baseline_occupancy = []
    stacked_simulated_occupancy_matrix = []
    stacked_xvals = []
    distance_measures = []

    for distance_measure, distance_data in occupancy_dict.items():
        # Get baseline occupancy for this distance measure
        baseline_occupancy = distance_data[baseline_mode]["combined"]["occupancy"]
        xvals = distance_data[baseline_mode]["combined"]["xvals"]

        stacked_baseline_occupancy.append(baseline_occupancy)
        stacked_xvals.append(xvals)
        distance_measures.append(distance_measure)

        # Get simulated occupancy for each packing mode
        simulated_occupancy_for_distance = []
        for packing_mode in packing_modes:
            simulated_occupancy = distance_data[packing_mode]["combined"]["occupancy"]
            simulated_occupancy_for_distance.append(simulated_occupancy)

        simulated_occupancy_matrix = np.array(simulated_occupancy_for_distance).T
        stacked_simulated_occupancy_matrix.append(simulated_occupancy_matrix)

        # Perform individual optimization for this distance measure
        coeffs_individual, _ = nnls(simulated_occupancy_matrix, baseline_occupancy)
        relative_contribution_individual = coeffs_individual / np.sum(coeffs_individual)
        reconstructed_occupancy_individual = simulated_occupancy_matrix @ coeffs_individual
        mse_individual = np.mean((baseline_occupancy - reconstructed_occupancy_individual) ** 2)

        # Store individual interpolation results
        interp_dict["interpolation"]["individual"][distance_measure] = {
            "fit_params": {
                mode: {
                    "coefficient": coeffs_individual[i],
                    "relative_contribution": relative_contribution_individual[i],
                }
                for i, mode in enumerate(packing_modes)
            },
            "mse": mse_individual,
        }

    # Concatenate all data for joint optimization
    stacked_baseline_occupancy = np.concatenate(stacked_baseline_occupancy)
    stacked_simulated_occupancy_matrix = np.vstack(stacked_simulated_occupancy_matrix)

    # Perform joint non-negative least squares fitting
    coeffs_joint, _ = nnls(stacked_simulated_occupancy_matrix, stacked_baseline_occupancy)
    relative_contribution_joint = coeffs_joint / np.sum(coeffs_joint)

    # Store global fit parameters
    interp_dict["interpolation"]["joint"]["fit_params"] = {
        mode: {
            "coefficient": coeffs_joint[i],
            "relative_contribution": relative_contribution_joint[i],
        }
        for i, mode in enumerate(packing_modes)
    }

    # Reconstruct occupancy for each distance measure using both methods
    start_idx = 0
    for i, distance_measure in enumerate(distance_measures):
        distance_data = occupancy_dict[distance_measure]
        xvals = stacked_xvals[i]
        end_idx = start_idx + len(xvals)

        # Get the portion of the stacked matrix for this distance measure
        distance_matrix = stacked_simulated_occupancy_matrix[start_idx:end_idx]

        # Reconstruct using joint coefficients
        reconstructed_occupancy_joint = distance_matrix @ coeffs_joint
        mse_joint = np.mean(
            (distance_data[baseline_mode]["combined"]["occupancy"] - reconstructed_occupancy_joint)
            ** 2
        )

        # Reconstruct using individual coefficients
        coeffs_individual = np.array(
            [
                interp_dict["interpolation"]["individual"][distance_measure]["fit_params"][mode][
                    "coefficient"
                ]
                for mode in packing_modes
            ]
        )
        reconstructed_occupancy_individual = distance_matrix @ coeffs_individual

        interp_dict["occupancy"][distance_measure] = {
            "xvals": xvals,
            "baseline": distance_data[baseline_mode]["combined"]["occupancy"],
            "reconstructed_joint": reconstructed_occupancy_joint,
            "reconstructed_individual": reconstructed_occupancy_individual,
            "mse_joint": mse_joint,
            "mse_individual": interp_dict["interpolation"]["individual"][distance_measure]["mse"],
            "modes": {
                mode: distance_data[mode]["combined"]["occupancy"] for mode in distance_data.keys()
            },
            "coeffs_individual": dict(zip(packing_modes, coeffs_individual, strict=False)),
            "coeffs_joint": dict(zip(packing_modes, coeffs_joint, strict=False)),
            "relative_contribution_individual": {
                mode: params["relative_contribution"]
                for mode, params in interp_dict["interpolation"]["individual"][distance_measure][
                    "fit_params"
                ].items()
            },
            "relative_contribution_joint": {
                mode: params["relative_contribution"]
                for mode, params in interp_dict["interpolation"]["joint"]["fit_params"].items()
            },
        }

        start_idx = end_idx

    log_file_path = None
    if results_dir is not None:
        log_file_path = results_dir / f"{baseline_mode}_occupancy_interpolation_coeffs{suffix}.log"

    log_occupancy_interpolation_coeffs(
        interp_dict, baseline_mode=baseline_mode, file_path=log_file_path
    )

    return interp_dict


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
    dist_logger.info("=" * 80)

    # Log joint interpolation results
    dist_logger.info("\nJoint Optimization (across all distance measures):")
    dist_logger.info("-" * 80)
    for mode, params in interpolated_occupancy_dict["interpolation"]["joint"]["fit_params"].items():
        dist_logger.info(
            f"Mode: {mode}, Coefficient: {params['coefficient']:.4f}, "
            f"Relative Contribution: {params['relative_contribution']:.4f}"
        )

    # Log individual interpolation results for each distance measure
    dist_logger.info("\n" + "=" * 80)
    dist_logger.info("\nIndividual Optimization (per distance measure):")
    dist_logger.info("-" * 80)
    individual_interp = interpolated_occupancy_dict["interpolation"]["individual"]
    for distance_measure, distance_data in individual_interp.items():
        dist_logger.info(f"\nDistance Measure: {distance_measure}")
        dist_logger.info(f"  MSE: {distance_data['mse']:.6f}")
        for mode, params in distance_data["fit_params"].items():
            dist_logger.info(
                f"  Mode: {mode}, Coefficient: {params['coefficient']:.4f}, "
                f"Relative Contribution: {params['relative_contribution']:.4f}"
            )

    # Log MSE comparison for each distance measure
    dist_logger.info("\n" + "=" * 80)
    dist_logger.info("\nMSE Comparison:")
    dist_logger.info("-" * 80)
    for distance_measure in interpolated_occupancy_dict["occupancy"].keys():
        occupancy_data = interpolated_occupancy_dict["occupancy"][distance_measure]
        mse_individual = occupancy_data["mse_individual"]
        mse_joint = occupancy_data["mse_joint"]
        dist_logger.info(
            f"{distance_measure}: Individual MSE = {mse_individual:.6f}, "
            f"Joint MSE = {mse_joint:.6f}"
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

    # Try to load cached occupancy data if not recalculating
    if not recalculate and save_path is not None and save_path.exists():
        try:
            with open(save_path, "rb") as f:
                binned_occupancy_dict = pickle.load(f)

            # Validate that cached data matches requested modes
            cached_modes = set(binned_occupancy_dict.keys())
            requested_modes = set(channel_map.keys())
            if cached_modes != requested_modes:
                logger.warning(
                    f"Cached binned occupancy data contains modes {cached_modes} but requested "
                    f"modes are {requested_modes}. Recalculating binned occupancy."
                )
            else:
                logger.info(f"Successfully loaded cached binned occupancy data from {save_path}")
                return binned_occupancy_dict
        except Exception as e:
            logger.warning(
                f"Error loading cached binned occupancy data from {save_path}: {e}. "
                f"Recalculating binned occupancy."
            )

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
