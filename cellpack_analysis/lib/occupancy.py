import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.label_tables import GRID_DISTANCE_LABELS
from cellpack_analysis.lib.load_data import PROJECT_ROOT
from cellpack_analysis.lib.stats import (
    EnvelopeType,
    _pairwise_test_on_curves,
    ecdf,
    make_r_grid_from_pooled,
    normalize_pdf,
    pdf_ratio,
    pointwise_envelope,
)

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
    cell_id_map: dict[str, list[str]] = {}
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
        Structure: {cell_id: {mode: gaussian_kde, "available": gaussian_kde}}
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
            combined_available_distances.extend(distance_kde_dict[cell_id]["available"].dataset)
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

            available_kde = distance_kde_dict[cell_id]["available"]
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
            ks_stat, p_value = ks_result.statistic, ks_result.pvalue  # type: ignore
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
            occ_1_raw = occupancy_dict[mode_1]["individual"][cell_id_1]["occupancy"]
            occupancy_1 = occ_1_raw[np.isfinite(occ_1_raw)]
            for j in range(i + 1, len(all_pairs)):
                mode_2, cell_id_2 = all_pairs[j]
                occ_2_raw = occupancy_dict[mode_2]["individual"][cell_id_2]["occupancy"]
                occupancy_2 = occ_2_raw[np.isfinite(occ_2_raw)]
                if len(occupancy_1) == 0 or len(occupancy_2) == 0:
                    emd = np.nan
                else:
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
            available_distance = kde_dict[seed]["available"]["distances"]
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
            available_distance = kde_dict[seed]["available"]["distances"]
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
    interp_dict: dict[str, Any] = {
        "occupancy": {},
        "interpolation": {
            "individual": {},
            "joint": {},
        },
    }

    # Stack data across all distance measures for joint optimization
    stacked_baseline_occupancy: list[np.ndarray] = []
    stacked_simulated_occupancy_matrix: list[np.ndarray] = []
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
    stacked_baseline_occupancy_array = np.concatenate(stacked_baseline_occupancy)
    stacked_simulated_occupancy_matrix_array = np.vstack(stacked_simulated_occupancy_matrix)

    # Perform joint non-negative least squares fitting
    coeffs_joint, _ = nnls(
        stacked_simulated_occupancy_matrix_array, stacked_baseline_occupancy_array
    )
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
            combined_available_distances.extend(distance_kde_dict[cell_id]["available"].dataset)
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
            available_distances = distance_kde_dict[cell_id]["available"].dataset

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

        occupied_space_counts_array = np.vstack(list(occupied_space_counts.values()))
        available_space_counts_array = np.vstack(list(available_space_counts.values()))
        occupancy_ratio = occupied_space_counts_array / available_space_counts_array
        mean_occupancy_ratio = np.nanmean(occupancy_ratio, axis=0)
        std_occupancy_ratio = np.nanstd(occupancy_ratio, axis=0)
        binned_occupancy_dict[mode]["combined"] = {
            "xvals": bin_centers,
            "occupancy": mean_occupancy_ratio,
            "std_occupancy": std_occupancy_ratio,
            "pdf_occupied": np.nanmean(occupied_space_counts_array, axis=0),
            "pdf_available": np.nanmean(available_space_counts_array, axis=0),
        }

    # save occupancy dictionary
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(binned_occupancy_dict, f)

    return binned_occupancy_dict


def _compute_single_cell_occupancy(
    occupied: np.ndarray,
    available: np.ndarray,
    bin_width: float,
    pseudocount: float = 1.0,
    min_count: int = 5,
) -> dict[str, np.ndarray]:
    """
    Compute the histogram-based occupancy ratio for a single cell.

    Builds a cell-specific bin grid from the cell's own available-space
    distances so that bins never extend beyond the range where the cell has
    data.  This avoids the tail instability that arises when a shared pooled
    grid creates near-empty bins for individual cells with shorter distance
    ranges.

    Parameters
    ----------
    occupied
        1-D array of finite occupied distances (already in final units).
    available
        1-D array of finite available-space distances (already in final units).
    bin_width
        Histogram bin width (µm or unitless, matching the distance arrays).
    pseudocount
        Laplace smoothing constant added to raw histogram counts before
        normalization.
    min_count
        Minimum raw count in the available-space histogram for a bin to be
        retained.  Bins below this threshold have their ratio set to ``NaN``.

    Returns
    -------
    :
        ``{"xvals": r_grid, "occupancy": ratio, "pdf_occupied": pdf_occ,
        "pdf_available": pdf_avail, "raw_occupied": raw_occ,
        "raw_available": raw_avail}``
    """
    # Build per-cell grid from the cell's own available-space distances
    r_grid = make_r_grid_from_pooled([available], bin_width=bin_width)
    bin_edges = np.concatenate([r_grid - bin_width / 2, [r_grid[-1] + bin_width / 2]])

    raw_occ, _ = np.histogram(occupied, bins=bin_edges)
    raw_avail, _ = np.histogram(available, bins=bin_edges)

    smoothed_occ = raw_occ.astype(float) + pseudocount
    smoothed_avail = raw_avail.astype(float) + pseudocount

    pdf_occ = normalize_pdf(r_grid, smoothed_occ)
    pdf_avail = normalize_pdf(r_grid, smoothed_avail)

    # Log-space ratio for numerical stability
    reg = 1e-10
    pdf_occ_reg = np.maximum(pdf_occ, reg)
    pdf_avail_reg = np.maximum(pdf_avail, reg)
    log_ratio = np.log(pdf_occ_reg) - np.log(pdf_avail_reg)
    ratio = np.exp(log_ratio)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

    if min_count > 0:
        ratio[raw_avail < min_count] = np.nan

    return {
        "xvals": r_grid,
        "occupancy": ratio,
        "pdf_occupied": pdf_occ,
        "pdf_available": pdf_avail,
        "raw_occupied": raw_occ,
        "raw_available": raw_avail,
    }


def get_binned_occupancy_dict_from_distance_dict(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    combined_mesh_information_dict: dict[str, dict[str, dict[str, Any]]],
    channel_map: dict[str, str],
    distance_measure: str = "nucleus",
    bin_width: float = 0.2,
    x_min: float = 0.0,
    x_max: float | None = None,
    num_cells: int | None = None,
    pseudocount: float = 1.0,
    min_count: int = 5,
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
) -> dict[str, dict[str, Any]]:
    """
    Calculate binned occupancy ratios directly from distance and mesh dictionaries.

    Each cell's occupancy ratio is computed on a **per-cell bin grid** derived
    solely from that cell's own available-space distances.  This eliminates the
    tail instability that arises when a shared pooled grid creates near-empty
    bins for individual cells whose distance range is shorter than the pooled
    maximum.

    The ``"combined"`` result is reported on a **common bin grid** (derived
    from explicit ``x_max`` or from pooled available-space distances) and
    contains two flavors of the combined occupancy:

    * ``"occupancy"`` — mean of per-cell ratio curves interpolated onto the
      common grid (preserves per-cell envelope structure).
    * ``"occupancy_pooled"`` — single ratio computed from the pooled
      occupied and available distances of all cells (the global view).

    Parameters
    ----------
    all_distance_dict
        ``{distance_measure: {mode: {cell_id: {seed: distances}}}}``
    combined_mesh_information_dict
        ``{structure_id: {cell_id: {nuc_grid_distances: ..., ...}}}``
    channel_map
        Mapping from packing modes to structure IDs.
    distance_measure
        Distance measure to analyze (must be a key in ``GRID_DISTANCE_LABELS``).
    bin_width
        Histogram bin width in µm.
    x_min
        Minimum x value for the common bin grid.
    x_max
        Maximum x value for the common bin grid.  When ``None`` it is inferred
        from the data using ``make_r_grid_from_pooled``.
    num_cells
        Maximum number of cells to sample per structure.  When ``None`` all
        cells are used.
    pseudocount
        Laplace smoothing constant added to raw histogram counts before
        normalization.  Default 1.0.
    min_count
        Minimum raw count required in the *available-space* histogram for a bin
        to be included.  Bins below this threshold have their ratio set to
        ``NaN``.  Set to 0 to disable masking.  Default 5.
    results_dir
        Directory to save the cached result.  No caching when ``None``.
    recalculate
        If ``True``, bypass any cached result.
    suffix
        Suffix appended to the cache filename.

    Returns
    -------
    :
        ``{mode: {"individual": {cell_id: {"xvals", "occupancy",
        "pdf_occupied", "pdf_available", "raw_occupied", "raw_available"}},
        "combined": {"xvals", "occupancy", "occupancy_pooled",
        "std_occupancy", "envelope_lo", "envelope_hi", "pdf_occupied",
        "pdf_available", "all_occupancy"}}}``

        Each cell in ``"individual"`` has its own ``"xvals"`` (per-cell grid).
        ``"combined"`` reports everything on a shared common grid.
    """
    save_path = None
    if results_dir is not None:
        save_path = results_dir / f"{distance_measure}_binned_occupancy_from_dist{suffix}.dat"

    if not recalculate and save_path is not None and save_path.exists():
        try:
            with open(save_path, "rb") as f:
                result = pickle.load(f)
            cached_modes = set(result.keys())
            requested_modes = set(channel_map.keys())
            if cached_modes == requested_modes:
                logger.info(
                    "Loaded cached binned occupancy (from distance dict) from %s", save_path
                )
                return result
            logger.warning(
                "Cached binned occupancy modes %s differ from requested %s — recalculating.",
                cached_modes,
                requested_modes,
            )
        except Exception as exc:
            logger.warning("Could not load cached occupancy (%s) — recalculating.", exc)

    grid_key = GRID_DISTANCE_LABELS.get(distance_measure)
    if grid_key is None:
        raise ValueError(
            f"distance_measure '{distance_measure}' not found in GRID_DISTANCE_LABELS. "
            f"Available: {list(GRID_DISTANCE_LABELS.keys())}"
        )

    distance_dict_dm = all_distance_dict[distance_measure]

    # Build cell_id_map: {structure_id: [cell_ids present in distance_dict]}
    cell_id_map: dict[str, list[str]] = {}
    for mode, structure_id in channel_map.items():
        if structure_id not in cell_id_map:
            cell_id_map[structure_id] = []
        for cell_id in distance_dict_dm.get(mode, {}):
            if cell_id not in cell_id_map[structure_id]:
                cell_id_map[structure_id].append(cell_id)

    if num_cells is not None:
        rng = np.random.default_rng(0)
        for structure_id, cell_ids in cell_id_map.items():
            if len(cell_ids) > num_cells:
                cell_id_map[structure_id] = rng.choice(
                    cell_ids, size=num_cells, replace=False
                ).tolist()

    # Scaled measures (e.g. scaled_nucleus) are unitless; others are in
    # pixel units and must be converted to µm to match the occupied distances
    # which have already been through ``normalize_distance_dictionary``.
    is_scaled_measure = "scaled" in distance_measure

    # ------------------------------------------------------------------
    # Build a common bin grid (for combined results and interpolation)
    # ------------------------------------------------------------------
    if x_max is None:
        all_available: list[np.ndarray] = []
        for structure_id, cell_ids in cell_id_map.items():
            mesh_info = combined_mesh_information_dict.get(structure_id, {})
            for cell_id in cell_ids:
                cell_info = mesh_info.get(cell_id, {})
                avail = cell_info.get(grid_key)
                if avail is not None:
                    a = np.asarray(avail)
                    if not is_scaled_measure:
                        a = a * PIXEL_SIZE_IN_UM
                    all_available.append(a)
        common_r_grid = make_r_grid_from_pooled(all_available, bin_width=bin_width)
    else:
        n_bins = max(1, round((x_max - x_min) / bin_width))
        common_r_grid = np.linspace(x_min + bin_width / 2, x_max - bin_width / 2, n_bins)

    common_bin_edges = np.concatenate(
        [common_r_grid - bin_width / 2, [common_r_grid[-1] + bin_width / 2]]
    )

    occupancy_result: dict[str, dict[str, Any]] = {}

    for mode, structure_id in channel_map.items():
        occupancy_result[mode] = {"individual": {}, "combined": {}}

        mesh_info = combined_mesh_information_dict.get(structure_id, {})
        valid_cell_ids = cell_id_map.get(structure_id, [])

        # Collectors for pooled combined occupancy
        all_occupied_distances: list[np.ndarray] = []
        all_available_distances: list[np.ndarray] = []

        # Collectors for interpolated per-cell ratios on common grid
        interp_ratios: list[np.ndarray] = []

        for cell_id in tqdm(valid_cell_ids, desc=f"Binned occupancy ({mode})"):
            if cell_id not in distance_dict_dm.get(mode, {}):
                continue
            cell_info = mesh_info.get(cell_id, {})
            avail_raw = cell_info.get(grid_key)
            if avail_raw is None:
                logger.warning("No available-space grid for cell %s / %s", cell_id, structure_id)
                continue

            # Prepare occupied and available distance arrays
            occupied_arrays = list(distance_dict_dm[mode][cell_id].values())
            occupied_concat = np.concatenate(
                [np.asarray(a) for a in occupied_arrays],
            )
            occupied_concat = occupied_concat[np.isfinite(occupied_concat)]
            available_concat = np.asarray(avail_raw, dtype=float)
            if not is_scaled_measure:
                available_concat = available_concat * PIXEL_SIZE_IN_UM
            available_concat = available_concat[np.isfinite(available_concat)]

            if np.max(available_concat) < np.max(occupied_concat):
                logger.warning(
                    "Cell %s, mode %s has max occupied distance %.2f greater than max available "
                    "distance %.2f — discarding invalid points.",
                    cell_id,
                    mode,
                    np.max(occupied_concat),
                    np.max(available_concat),
                )
                occupied_concat = occupied_concat[occupied_concat <= np.max(available_concat)]

            # Per-cell occupancy on the cell's own grid
            cell_result = _compute_single_cell_occupancy(
                occupied=occupied_concat,
                available=available_concat,
                bin_width=bin_width,
                pseudocount=pseudocount,
                min_count=min_count,
            )
            occupancy_result[mode]["individual"][cell_id] = cell_result

            # Collect raw distances for pooled computation
            all_occupied_distances.append(occupied_concat)
            all_available_distances.append(available_concat)

            # Interpolate per-cell ratio onto common grid (NaN outside cell range)
            interp_ratio = np.interp(
                common_r_grid,
                cell_result["xvals"],
                cell_result["occupancy"],
                left=np.nan,
                right=np.nan,
            )
            interp_ratios.append(interp_ratio)

        # ------------------------------------------------------------------
        # Combined results on common grid
        # ------------------------------------------------------------------
        if interp_ratios:
            ratio_arr = np.vstack(interp_ratios)
            mean_ratio = np.nanmean(ratio_arr, axis=0)
            std_ratio = np.nanstd(ratio_arr, axis=0)
            env_lo, env_hi, _, _ = pointwise_envelope(
                np.where(np.isnan(ratio_arr), 0.0, ratio_arr), alpha=0.05
            )

            # Pooled occupancy: single ratio from all cells' distances combined
            pooled_occupied = np.concatenate(all_occupied_distances)
            pooled_available = np.concatenate(all_available_distances)
            raw_occ_pooled, _ = np.histogram(pooled_occupied, bins=common_bin_edges)
            raw_avail_pooled, _ = np.histogram(pooled_available, bins=common_bin_edges)
            smoothed_occ_p = raw_occ_pooled.astype(float) + pseudocount
            smoothed_avail_p = raw_avail_pooled.astype(float) + pseudocount
            pdf_occ_pooled = normalize_pdf(common_r_grid, smoothed_occ_p)
            pdf_avail_pooled = normalize_pdf(common_r_grid, smoothed_avail_p)

            reg = 1e-10
            log_ratio_pooled = np.log(np.maximum(pdf_occ_pooled, reg)) - np.log(
                np.maximum(pdf_avail_pooled, reg)
            )
            ratio_pooled = np.exp(log_ratio_pooled)
            ratio_pooled = np.nan_to_num(ratio_pooled, nan=0.0, posinf=0.0, neginf=0.0)
            if min_count > 0:
                ratio_pooled[raw_avail_pooled < min_count] = np.nan

            occupancy_result[mode]["combined"] = {
                "xvals": common_r_grid,
                "occupancy": mean_ratio,
                "occupancy_pooled": ratio_pooled,
                "std_occupancy": std_ratio,
                "envelope_lo": env_lo,
                "envelope_hi": env_hi,
                "pdf_occupied": pdf_occ_pooled,
                "pdf_available": pdf_avail_pooled,
                "all_occupancy": ratio_arr,
            }
        else:
            occupancy_result[mode]["combined"] = {
                "xvals": common_r_grid,
                "occupancy": np.zeros_like(common_r_grid),
                "occupancy_pooled": np.zeros_like(common_r_grid),
                "std_occupancy": np.zeros_like(common_r_grid),
                "envelope_lo": np.zeros_like(common_r_grid),
                "envelope_hi": np.zeros_like(common_r_grid),
                "pdf_occupied": np.zeros_like(common_r_grid),
                "pdf_available": np.zeros_like(common_r_grid),
                "all_occupancy": np.empty((0, len(common_r_grid))),
            }

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(occupancy_result, f)
        logger.info("Saved binned occupancy (from distance dict) to %s", save_path)

    return occupancy_result


def pairwise_envelope_test_occupancy(
    combined_binned_occupancy_dict: dict[str, dict[str, dict[str, Any]]],
    packing_modes: list[str],
    alpha: float = 0.05,
    statistic: Literal["supremum", "intdev"] = "intdev",
    envelope_type: EnvelopeType = "pointwise",
    joint_r_grid_size: int | None = None,
    comparison_type: Literal["ratio", "ecdf"] = "ratio",
) -> dict[str, Any]:
    """
    Run a pairwise Monte Carlo envelope test on per-cell occupancy ratio curves.

    For each ordered pair ``(mode_a, mode_b)`` the per-cell occupancy *ratio
    curves* from ``mode_b`` are used as the simulation envelope; each per-cell
    curve from ``mode_a`` is then tested against that envelope.  BH FDR
    correction is applied within each pair.

    Unlike :func:`~cellpack_analysis.lib.stats.pairwise_envelope_test`, the
    occupancy ratio curves are compared *directly* as spatial functions — each
    curve encodes how the occupancy ratio varies across distance bins, and the
    test detects differences in that spatial pattern.  No ECDF construction is
    performed.

    Parameters
    ----------
    combined_binned_occupancy_dict
        ``{distance_measure: {mode: {"individual": {cell_id: {"occupancy":
        np.ndarray, "xvals": np.ndarray}}, "combined": {...}}}}`` — the output
        of :func:`get_binned_occupancy_dict_from_distance_dict` wrapped in a
        distance-measure outer key.
    packing_modes
        Ordered list of modes forming the pairwise matrix.
    alpha
        Significance level for BH-corrected rejection.
    statistic
        Test statistic: ``"intdev"`` or ``"supremum"``.
        Ignored when ``envelope_type="rank"``.
    envelope_type
        ``"pointwise"``: pointwise quantile envelope (default).
        ``"rank"``: simultaneous rank envelope test (Myllymäki et al. 2017).
        The visualization envelopes (``envelopes`` key) always use pointwise
        quantiles regardless of this setting.
    joint_r_grid_size
        Points per distance measure when resampling curves to an equal-length
        grid before the joint test.  Defaults to the bin count of the first
        available occupancy curve, giving each distance measure equal weight.
    comparison_type
        ``"ratio"`` (default): compare per-cell occupancy ratio curves directly
        as spatial functions over distance.
        ``"ecdf"``: convert each per-cell ratio curve to its ECDF over ratio
        space before comparison.  The ECDF treats the bin-wise ratio values as
        a 1-D sample and evaluates the cumulative distribution on a shared grid
        spanning the range of all ratio values for that distance measure.

    Returns
    -------
    :
        Dictionary with the same structure as
        :func:`~cellpack_analysis.lib.stats.pairwise_envelope_test`:

        .. code-block:: python

            {
                "per_distance_measure": {
                    dm: {
                        (mode_a, mode_b): {
                            "pvals", "qvals", "signs",
                            "rejection_fraction",
                            "rejection_fraction_positive",
                            "rejection_fraction_negative",
                        }
                    }
                },
                "joint": { (mode_a, mode_b): {...} },
                "envelopes": {
                    mode: {
                        dm: {"xvals", "lo", "hi", "mu"}
                    }
                },
                "packing_modes": list[str],
                "distance_measures": list[str],
                "alpha": float,
                "statistic": str,
                "envelope_type": str,
                "comparison_type": str,
            }
    """
    distance_measures = list(combined_binned_occupancy_dict.keys())

    envelopes: dict[str, dict[str, Any]] = {}
    mode_curves: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in packing_modes}
    mode_xvals: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in packing_modes}

    for dm in distance_measures:
        occ_dm = combined_binned_occupancy_dict[dm]
        for mode in packing_modes:
            combined = occ_dm.get(mode, {}).get("combined", {})
            xvals = combined.get("xvals", np.array([]))
            mode_xvals[mode][dm] = xvals

            # Use the pre-interpolated per-cell ratios on the common grid
            arr = combined.get("all_occupancy", np.empty((0, max(1, len(xvals)))))
            # Replace NaN with 0 for stacking/envelope operations
            arr = np.where(np.isnan(arr), 0.0, arr)
            mode_curves[mode][dm] = arr

            # Build pointwise envelope for visualization (always pointwise regardless of
            # envelope_type, matching the behaviour of pairwise_envelope_test's envelopes key)
            if arr.shape[0] > 1:
                lo, hi, mu, _ = pointwise_envelope(arr, alpha=alpha)
            else:
                mu = combined.get("occupancy", np.zeros_like(xvals))
                lo = mu.copy()
                hi = mu.copy()
            envelopes.setdefault(mode, {})[dm] = {
                "xvals": xvals,
                "lo": lo,
                "hi": hi,
                "mu": mu,
            }

    # Optionally convert ratio curves to their ECDFs over ratio space
    if comparison_type == "ecdf":
        for dm in distance_measures:
            all_ratio_vals = np.concatenate(
                [
                    mode_curves[mode][dm].ravel()
                    for mode in packing_modes
                    if mode_curves[mode][dm].size > 0
                ]
            )
            if all_ratio_vals.size == 0:
                continue
            # Use the same number of grid points as the distance bins
            n_grid = len(mode_xvals[packing_modes[0]][dm])
            ratio_grid = np.linspace(
                np.nanmin(all_ratio_vals), np.nanmax(all_ratio_vals), max(n_grid, 2)
            )
            for mode in packing_modes:
                arr = mode_curves[mode][dm]
                if arr.shape[0] > 0:
                    mode_curves[mode][dm] = np.vstack([ecdf(row, ratio_grid) for row in arr])
                mode_xvals[mode][dm] = ratio_grid
                # Rebuild envelopes from the ECDF-transformed curves
                ecdf_arr = mode_curves[mode][dm]
                if ecdf_arr.shape[0] > 1:
                    lo, hi, mu, _ = pointwise_envelope(ecdf_arr, alpha=alpha)
                else:
                    mu = (
                        np.mean(ecdf_arr, axis=0)
                        if ecdf_arr.shape[0] == 1
                        else np.zeros_like(ratio_grid)
                    )
                    lo = mu.copy()
                    hi = mu.copy()
                envelopes.setdefault(mode, {})[dm] = {
                    "xvals": ratio_grid,
                    "lo": lo,
                    "hi": hi,
                    "mu": mu,
                }

    per_dm_results, joint_results = _pairwise_test_on_curves(
        mode_curves=mode_curves,
        mode_xvals=mode_xvals,
        packing_modes=packing_modes,
        distance_measures=distance_measures,
        alpha=alpha,
        statistic=statistic,
        envelope_type=envelope_type,
        joint_r_grid_size=joint_r_grid_size,
    )

    return {
        "per_distance_measure": per_dm_results,
        "joint": joint_results,
        "envelopes": envelopes,
        "packing_modes": packing_modes,
        "distance_measures": distance_measures,
        "alpha": alpha,
        "statistic": statistic,
        "envelope_type": envelope_type,
        "comparison_type": comparison_type,
    }
