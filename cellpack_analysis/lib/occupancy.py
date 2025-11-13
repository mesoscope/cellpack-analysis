import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize, nnls
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from sklearn.linear_model import ElasticNet, Ridge
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


def explore_regularized_solutions(
    simulated_occupancy_matrix: np.ndarray,
    baseline_occupancy: np.ndarray,
    packing_modes: list[str],
    alpha_range: np.ndarray = np.logspace(-4, 2, 20),
) -> dict[str, Any]:
    """
    Explore solutions using L1, L2, and elastic net regularization.

    Parameters
    ----------
    simulated_occupancy_matrix
        Matrix of simulated occupancy data
    baseline_occupancy
        Target baseline occupancy to fit
    packing_modes
        List of packing mode names
    alpha_range
        Range of regularization parameters to explore

    Returns
    -------
    :
        Dictionary containing regularized solutions
    """
    solutions = {"ridge": [], "elastic_net": [], "alphas": alpha_range}

    for alpha in alpha_range:
        # Ridge (L2) regularization
        ridge = Ridge(alpha=alpha, positive=True, fit_intercept=False)
        ridge.fit(simulated_occupancy_matrix, baseline_occupancy)
        ridge_coeffs = ridge.coef_
        ridge_mse = np.mean((baseline_occupancy - simulated_occupancy_matrix @ ridge_coeffs) ** 2)

        solutions["ridge"].append(
            {
                "coefficients": ridge_coeffs,
                "relative_contribution": ridge_coeffs / np.sum(ridge_coeffs),
                "mse": ridge_mse,
                "alpha": alpha,
            }
        )

        # Elastic Net (combines L1 and L2)
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, positive=True, fit_intercept=False)
        elastic.fit(simulated_occupancy_matrix, baseline_occupancy)
        elastic_coeffs = elastic.coef_
        elastic_mse = np.mean(
            (baseline_occupancy - simulated_occupancy_matrix @ elastic_coeffs) ** 2
        )

        solutions["elastic_net"].append(
            {
                "coefficients": elastic_coeffs,
                "relative_contribution": elastic_coeffs / np.sum(elastic_coeffs),
                "mse": elastic_mse,
                "alpha": alpha,
            }
        )

    return solutions


def find_pareto_optimal_solutions(
    simulated_occupancy_matrix: np.ndarray,
    baseline_occupancy: np.ndarray,
    packing_modes: list[str],
    num_solutions: int = 20,
    objectives: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Find Pareto optimal solutions using multi-objective optimization.

    Parameters
    ----------
    simulated_occupancy_matrix
        Matrix of simulated occupancy data
    baseline_occupancy
        Target baseline occupancy to fit
    packing_modes
        List of packing mode names
    num_solutions
        Number of solutions to find along Pareto front
    objectives
        List of objectives to optimize: ['mse', 'sparsity', 'entropy', 'smoothness']

    Returns
    -------
    :
        List of Pareto optimal solutions
    """
    if objectives is None:
        objectives = ["mse", "sparsity", "entropy"]

    def calculate_objectives(coeffs):
        """Calculate multiple objectives for given coefficients."""
        # Normalize coefficients
        rel_contrib = coeffs / (np.sum(coeffs) + 1e-16)

        objectives_dict = {}

        # MSE (minimize)
        mse = np.mean((baseline_occupancy - simulated_occupancy_matrix @ coeffs) ** 2)
        objectives_dict["mse"] = mse

        # Sparsity (minimize - prefer fewer active modes)
        sparsity = np.sum(coeffs < 1e-6) / len(coeffs)
        objectives_dict["sparsity"] = -sparsity  # Negative for minimization

        # Entropy (maximize diversity)
        entropy = -np.sum(rel_contrib * np.log(rel_contrib + 1e-16))
        objectives_dict["entropy"] = -entropy  # Negative for minimization

        # Smoothness (minimize coefficient variation)
        smoothness = np.std(coeffs) / (np.mean(coeffs) + 1e-16)
        objectives_dict["smoothness"] = smoothness

        # Dominance (prefer balanced solutions)
        dominance = np.max(rel_contrib) - np.min(rel_contrib)
        objectives_dict["dominance"] = dominance

        return {obj: objectives_dict[obj] for obj in objectives}

    solutions = []

    # Generate solutions with different weightings
    for i in range(num_solutions):
        # Random weights for scalarization
        weights = np.random.dirichlet(np.ones(len(objectives)))

        def multi_objective(coeffs):
            obj_vals = calculate_objectives(coeffs)
            return sum(w * obj_vals[obj] for w, obj in zip(weights, objectives))

        # Random initialization
        init_coeffs = np.random.exponential(1.0, len(packing_modes))

        result = minimize(
            multi_objective,
            init_coeffs,
            method="L-BFGS-B",
            bounds=[(0, None) for _ in range(len(packing_modes))],
        )

        if result.success:
            coeffs = result.x
            obj_vals = calculate_objectives(coeffs)

            solutions.append(
                {
                    "coefficients": coeffs,
                    "relative_contribution": coeffs / np.sum(coeffs),
                    "objectives": obj_vals,
                    "weights": weights,
                    "method": f"pareto_{i}",
                }
            )

    # Filter for Pareto optimal solutions
    pareto_solutions = []

    for i, sol_i in enumerate(solutions):
        is_dominated = False

        for j, sol_j in enumerate(solutions):
            if i == j:
                continue

            # Check if sol_j dominates sol_i
            dominates = True
            for obj in objectives:
                if sol_j["objectives"][obj] > sol_i["objectives"][obj]:
                    dominates = False
                    break

            # At least one objective must be strictly better
            if dominates:
                strictly_better = False
                for obj in objectives:
                    if sol_j["objectives"][obj] < sol_i["objectives"][obj]:
                        strictly_better = True
                        break

                if strictly_better:
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_solutions.append(sol_i)

    # Sort by MSE for consistency
    return sorted(pareto_solutions, key=lambda x: x["objectives"]["mse"])


def find_constrained_solutions(
    simulated_occupancy_matrix: np.ndarray,
    baseline_occupancy: np.ndarray,
    packing_modes: list[str],
    constraints: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Find solutions under different biological constraints.

    Parameters
    ----------
    simulated_occupancy_matrix
        Matrix of simulated occupancy data
    baseline_occupancy
        Target baseline occupancy to fit
    packing_modes
        List of packing mode names
    constraints
        Dictionary of constraint types and parameters

    Returns
    -------
    :
        List of constrained solutions
    """
    if constraints is None:
        constraints = {
            "max_single_contribution": [0.7, 0.8, 0.9],  # Maximum any single mode
            "min_modes_active": [1, 2, 3],  # Minimum number of active modes
            "entropy_threshold": [0.5, 1.0, 1.5],  # Minimum entropy
        }

    solutions = []

    # MSE objective function
    def mse_objective(coeffs):
        return np.mean((baseline_occupancy - simulated_occupancy_matrix @ coeffs) ** 2)

    for constraint_type, values in constraints.items():
        for value in values:
            try:
                if constraint_type == "max_single_contribution":
                    # No single mode can contribute more than 'value'
                    def constraint_fun(coeffs):
                        rel_contrib = coeffs / (np.sum(coeffs) + 1e-16)
                        return value - np.max(rel_contrib)

                    constraint = {"type": "ineq", "fun": constraint_fun}

                elif constraint_type == "min_modes_active":
                    # At least 'value' modes must be significantly active
                    def constraint_fun(coeffs):
                        active_modes = np.sum(coeffs > 0.01 * np.sum(coeffs))
                        return active_modes - value

                    constraint = {"type": "ineq", "fun": constraint_fun}

                elif constraint_type == "entropy_threshold":
                    # Entropy must be at least 'value'
                    def constraint_fun(coeffs):
                        rel_contrib = coeffs / (np.sum(coeffs) + 1e-16)
                        entropy = -np.sum(rel_contrib * np.log(rel_contrib + 1e-16))
                        return entropy - value

                    constraint = {"type": "ineq", "fun": constraint_fun}

                else:
                    continue

                # Initial guess
                init_coeffs = np.ones(len(packing_modes)) / len(packing_modes)

                result = minimize(
                    mse_objective,
                    init_coeffs,
                    method="SLSQP",
                    bounds=[(0, None) for _ in range(len(packing_modes))],
                    constraints=constraint,
                )

                if result.success:
                    coeffs = result.x
                    mse = mse_objective(coeffs)

                    solutions.append(
                        {
                            "coefficients": coeffs,
                            "relative_contribution": coeffs / np.sum(coeffs),
                            "mse": mse,
                            "constraint_type": constraint_type,
                            "constraint_value": value,
                            "method": f"constrained_{constraint_type}_{value}",
                        }
                    )

            except Exception as e:
                logger.warning(f"Constraint optimization failed for {constraint_type}={value}: {e}")
                continue

    return sorted(solutions, key=lambda x: x["mse"])


def find_robust_solutions(
    simulated_occupancy_matrix: np.ndarray,
    baseline_occupancy: np.ndarray,
    packing_modes: list[str],
    uncertainty_levels: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Find solutions that are robust to uncertainty in the data.

    Parameters
    ----------
    simulated_occupancy_matrix
        Matrix of simulated occupancy data
    baseline_occupancy
        Target baseline occupancy to fit
    packing_modes
        List of packing mode names
    uncertainty_levels
        List of uncertainty levels to consider

    Returns
    -------
    :
        List of robust solutions
    """
    if uncertainty_levels is None:
        uncertainty_levels = [0.01, 0.05, 0.1]

    solutions = []

    for uncertainty in uncertainty_levels:

        def robust_objective(coeffs):
            """Minimize worst-case MSE over uncertainty set."""
            base_mse = np.mean((baseline_occupancy - simulated_occupancy_matrix @ coeffs) ** 2)

            # Add penalty for sensitivity to perturbations
            n_perturbations = 10
            max_perturbed_mse = base_mse

            for _ in range(n_perturbations):
                # Perturb the matrix
                noise_matrix = np.random.normal(0, uncertainty, simulated_occupancy_matrix.shape)
                perturbed_matrix = simulated_occupancy_matrix + noise_matrix

                # Perturb baseline
                noise_baseline = np.random.normal(
                    0, uncertainty * np.std(baseline_occupancy), baseline_occupancy.shape
                )
                perturbed_baseline = baseline_occupancy + noise_baseline

                perturbed_mse = np.mean((perturbed_baseline - perturbed_matrix @ coeffs) ** 2)
                max_perturbed_mse = max(max_perturbed_mse, perturbed_mse)

            return max_perturbed_mse

        # Multiple random starts for robust solutions
        for i in range(5):
            init_coeffs = np.random.exponential(1.0, len(packing_modes))

            result = minimize(
                robust_objective,
                init_coeffs,
                method="L-BFGS-B",
                bounds=[(0, None) for _ in range(len(packing_modes))],
            )

            if result.success:
                coeffs = result.x
                base_mse = np.mean((baseline_occupancy - simulated_occupancy_matrix @ coeffs) ** 2)
                robust_mse = robust_objective(coeffs)

                solutions.append(
                    {
                        "coefficients": coeffs,
                        "relative_contribution": coeffs / np.sum(coeffs),
                        "mse": base_mse,
                        "robust_mse": robust_mse,
                        "uncertainty_level": uncertainty,
                        "method": f"robust_uncertainty_{uncertainty}_init_{i}",
                    }
                )

    return sorted(solutions, key=lambda x: x["robust_mse"])


def find_diverse_solutions(
    simulated_occupancy_matrix: np.ndarray,
    baseline_occupancy: np.ndarray,
    packing_modes: list[str],
    num_solutions: int = 10,
    diversity_weight: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Find multiple diverse solutions by adding diversity constraints.

    Parameters
    ----------
    simulated_occupancy_matrix
        Matrix of simulated occupancy data
    baseline_occupancy
        Target baseline occupancy to fit
    packing_modes
        List of packing mode names
    num_solutions
        Number of diverse solutions to find
    diversity_weight
        Weight for diversity penalty term

    Returns
    -------
    :
        List of diverse solutions sorted by MSE
    """
    solutions = []

    def objective_with_diversity(coeffs, existing_solutions, mse_weight=1.0):
        # MSE term
        mse = np.mean((baseline_occupancy - simulated_occupancy_matrix @ coeffs) ** 2)

        # Diversity term - penalize similarity to existing solutions
        diversity_penalty = 0
        for existing_coeffs in existing_solutions:
            # Use negative cosine similarity as diversity measure
            similarity = np.dot(coeffs, existing_coeffs) / (
                np.linalg.norm(coeffs) * np.linalg.norm(existing_coeffs)
            )
            diversity_penalty += similarity**2

        return mse_weight * mse + diversity_weight * diversity_penalty

    # Generate multiple solutions with different initializations
    for i in range(num_solutions):
        # Random initialization biased toward different modes
        init_coeffs = np.random.exponential(1.0, len(packing_modes))
        init_coeffs[i % len(packing_modes)] *= 3  # Bias toward different modes

        existing_coeffs = [sol["coefficients"] for sol in solutions]

        result = minimize(
            lambda c: objective_with_diversity(c, existing_coeffs),
            init_coeffs,
            method="L-BFGS-B",
            bounds=[(0, None) for _ in range(len(packing_modes))],
        )

        if result.success:
            coeffs = result.x
            mse = np.mean((baseline_occupancy - simulated_occupancy_matrix @ coeffs) ** 2)
            solutions.append(
                {
                    "coefficients": coeffs,
                    "relative_contribution": coeffs / np.sum(coeffs),
                    "mse": mse,
                    "method": f"diverse_init_{i}",
                }
            )

    return sorted(solutions, key=lambda x: x["mse"])


def bootstrap_solutions(
    simulated_occupancy_matrix: np.ndarray,
    baseline_occupancy: np.ndarray,
    packing_modes: list[str],
    n_bootstrap: int = 100,
    noise_level: float = 0.01,
) -> dict[str, Any]:
    """
    Generate bootstrap samples and add noise to explore solution stability.

    Parameters
    ----------
    simulated_occupancy_matrix
        Matrix of simulated occupancy data
    baseline_occupancy
        Target baseline occupancy to fit
    packing_modes
        List of packing mode names
    n_bootstrap
        Number of bootstrap samples
    noise_level
        Level of noise to add relative to data std

    Returns
    -------
    :
        Dictionary containing bootstrap solutions and statistics
    """
    bootstrap_solutions = []
    n_points = len(baseline_occupancy)

    for i in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_points, n_points, replace=True)
        boot_matrix = simulated_occupancy_matrix[indices]
        boot_baseline = baseline_occupancy[indices]

        # Add small amount of noise
        boot_baseline += np.random.normal(0, noise_level * np.std(boot_baseline), n_points)

        # Solve NNLS
        coeffs, _ = nnls(boot_matrix, boot_baseline)
        mse = np.mean((boot_baseline - boot_matrix @ coeffs) ** 2)

        bootstrap_solutions.append(
            {
                "coefficients": coeffs,
                "relative_contribution": coeffs / np.sum(coeffs) if np.sum(coeffs) > 0 else coeffs,
                "mse": mse,
                "bootstrap_idx": i,
            }
        )

    # Analyze bootstrap statistics
    all_coeffs = np.array([sol["coefficients"] for sol in bootstrap_solutions])
    all_contributions = np.array([sol["relative_contribution"] for sol in bootstrap_solutions])

    return {
        "solutions": bootstrap_solutions,
        "coefficient_stats": {
            mode: {
                "mean": np.mean(all_coeffs[:, i]),
                "std": np.std(all_coeffs[:, i]),
                "percentile_25": np.percentile(all_coeffs[:, i], 25),
                "percentile_75": np.percentile(all_coeffs[:, i], 75),
            }
            for i, mode in enumerate(packing_modes)
        },
        "contribution_stats": {
            mode: {
                "mean": np.mean(all_contributions[:, i]),
                "std": np.std(all_contributions[:, i]),
                "percentile_25": np.percentile(all_contributions[:, i], 25),
                "percentile_75": np.percentile(all_contributions[:, i], 75),
            }
            for i, mode in enumerate(packing_modes)
        },
    }


def visualize_solution_space(
    interp_dict: dict[str, Any],
    packing_modes: list[str],
    save_dir: Path | None = None,
    suffix: str = "",
) -> None:
    """
    Create comprehensive visualizations of the solution space.

    Parameters
    ----------
    interp_dict
        Dictionary containing interpolated occupancy data and alternatives
    packing_modes
        List of packing mode names
    save_dir
        Directory to save plots
    suffix
        Suffix for output files
    """
    if (
        "alternatives" not in interp_dict["interpolation"]
        or interp_dict["interpolation"]["alternatives"] is None
    ):
        logger.warning("No alternative solutions to visualize")
        return

    alternatives = interp_dict["interpolation"]["alternatives"]

    # Collect all solutions
    all_solutions = []
    solution_labels = []

    # Original NNLS
    orig_coeffs = alternatives["original_nnls"]["coefficients"]
    all_solutions.append(orig_coeffs)
    solution_labels.append("NNLS")

    # Regularized solutions
    if "regularized" in alternatives:
        for method in ["ridge", "elastic_net"]:
            if method in alternatives["regularized"]:
                for sol in alternatives["regularized"][method][::2]:  # Sample every other
                    all_solutions.append(sol["coefficients"])
                    solution_labels.append(f"{method.title()}")

    # Bootstrap solutions
    if "bootstrap" in alternatives:
        bootstrap_sols = alternatives["bootstrap"]["solutions"][::5]  # Sample every 5th
        for sol in bootstrap_sols:
            all_solutions.append(sol["coefficients"])
            solution_labels.append("Bootstrap")

    # Diverse solutions
    if "diverse" in alternatives:
        for i, sol in enumerate(alternatives["diverse"][:10]):  # Top 10
            all_solutions.append(sol["coefficients"])
            solution_labels.append(f"Diverse-{i+1}")

    # Pareto optimal solutions
    if "pareto_optimal" in alternatives:
        for i, sol in enumerate(alternatives["pareto_optimal"][:8]):  # Top 8
            all_solutions.append(sol["coefficients"])
            solution_labels.append(f"Pareto-{i+1}")

    # Constrained solutions
    if "constrained" in alternatives:
        for i, sol in enumerate(alternatives["constrained"][:5]):  # Top 5
            all_solutions.append(sol["coefficients"])
            solution_labels.append(f"Constrained-{i+1}")

    # Robust solutions
    if "robust" in alternatives:
        for i, sol in enumerate(alternatives["robust"][:5]):  # Top 5
            all_solutions.append(sol["coefficients"])
            solution_labels.append(f"Robust-{i+1}")

    solutions_array = np.array(all_solutions)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    fig.suptitle("Solution Space Exploration", fontsize=16, fontweight="bold")

    # 1. Coefficient space visualization (2D projection)
    if len(packing_modes) >= 2:
        ax = axes[0, 0]
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(set(solution_labels))))
        label_colors = {label: colors[i] for i, label in enumerate(set(solution_labels))}

        for sol, label in zip(solutions_array, solution_labels):
            # Only add label if not already in legend
            legend_labels = [t.get_text() for t in ax.legend_.get_texts()] if ax.legend_ else []
            show_label = label if label not in legend_labels else ""
            ax.scatter(sol[0], sol[1], c=[label_colors[label]], label=show_label, alpha=0.7, s=50)

        ax.set_xlabel(packing_modes[0])
        ax.set_ylabel(packing_modes[1] if len(packing_modes) > 1 else packing_modes[0])
        ax.set_title(
            f"{packing_modes[0]} vs {packing_modes[1] if len(packing_modes) > 1 else packing_modes[0]}"  # noqa: E501
        )
        ax.legend()

    # 2. Relative Contribution Visualization
    ax = axes[0, 1]
    relative_contribs = solutions_array / solutions_array.sum(axis=1, keepdims=True)

    # Stacked bar chart of relative contributions
    x_pos = np.arange(len(solutions_array))
    bottom = np.zeros(len(solutions_array))

    for i, mode in enumerate(packing_modes):
        ax.bar(x_pos, relative_contribs[:, i], bottom=bottom, label=mode, alpha=0.8)
        bottom += relative_contribs[:, i]

    ax.set_xlabel("Solution Index")
    ax.set_ylabel("Relative Contribution")
    ax.set_title("Relative Contributions Across Solutions")
    ax.legend()
    ax.set_xticks(x_pos[:: max(1, len(x_pos) // 10)])  # Show every 10th tick

    # 3. Solution diversity analysis
    ax = axes[1, 0]
    diversity_scores = []

    for sol in all_solutions:
        # Use coefficient of variation as diversity metric
        diversity = np.std(sol) / (np.mean(sol) + 1e-16)
        diversity_scores.append(diversity)

    # Color by solution type
    colors_by_type = []
    for label in solution_labels:
        if "NNLS" in label:
            colors_by_type.append("red")
        elif "Ridge" in label:
            colors_by_type.append("blue")
        elif "Bootstrap" in label:
            colors_by_type.append("green")
        elif "Diverse" in label:
            colors_by_type.append("orange")
        elif "Pareto" in label:
            colors_by_type.append("purple")
        elif "Constrained" in label:
            colors_by_type.append("brown")
        elif "Robust" in label:
            colors_by_type.append("pink")
        else:
            colors_by_type.append("gray")

    ax.scatter(diversity_scores, range(len(diversity_scores)), c=colors_by_type, alpha=0.7, s=50)
    ax.set_xlabel("Solution Diversity (CV of coefficients)")
    ax.set_ylabel("Solution Index")
    ax.set_title("Solution Diversity Distribution")

    # 4. MSE comparison
    ax = axes[1, 1]
    mse_values = []
    method_names = []

    # Get MSE values for each solution type
    mse_values.append(alternatives["original_nnls"]["mse"])
    method_names.append("NNLS")

    if "bootstrap" in alternatives:
        bootstrap_mse = [sol["mse"] for sol in alternatives["bootstrap"]["solutions"]]
        mse_values.extend(bootstrap_mse[:5])  # Limit for clarity
        method_names.extend(["Bootstrap"] * min(5, len(bootstrap_mse)))

    if "diverse" in alternatives and alternatives["diverse"]:
        diverse_mse = [sol["mse"] for sol in alternatives["diverse"][:5]]
        mse_values.extend(diverse_mse)
        method_names.extend(["Diverse" for i in range(len(diverse_mse))])

    # Box plot of MSE by method type
    method_types = list(set(method_names))
    mse_by_type = [[] for _ in method_types]

    for mse, method in zip(mse_values, method_names):
        type_idx = method_types.index(method)  # Group diverse solutions
        mse_by_type[type_idx].append(mse)

    ax.boxplot(
        [mse_list for mse_list in mse_by_type if mse_list],
        labels=[method_types[i] for i, mse_list in enumerate(mse_by_type) if mse_list],
    )
    ax.set_ylabel("MSE")
    ax.set_title("MSE Distribution by Method")
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / f"solution_space_overview{suffix}.png", dpi=300, bbox_inches="tight")
        logger.info(
            f"Solution space visualization saved to "
            f"{save_dir / f'solution_space_overview{suffix}.png'}"
        )

    plt.show()


def plot_pareto_front(
    interp_dict: dict[str, Any],
    packing_modes: list[str],
    save_dir: Path | None = None,
    suffix: str = "",
) -> None:
    """
    Create visualizations of the Pareto front for multi-objective solutions.

    Parameters
    ----------
    interp_dict
        Dictionary containing interpolated occupancy data and alternatives
    packing_modes
        List of packing mode names
    save_dir
        Directory to save plots
    suffix
        Suffix for output files
    """
    if (
        "alternatives" not in interp_dict["interpolation"]
        or interp_dict["interpolation"]["alternatives"] is None
        or "pareto_optimal" not in interp_dict["interpolation"]["alternatives"]
    ):
        logger.warning("No Pareto optimal solutions to visualize")
        return

    pareto_solutions = interp_dict["interpolation"]["alternatives"]["pareto_optimal"]

    if not pareto_solutions:
        logger.warning("Empty Pareto solution set")
        return

    # Extract objective values
    mse_vals = [sol["objectives"]["mse"] for sol in pareto_solutions]
    sparsity_vals = [
        -sol["objectives"]["sparsity"] for sol in pareto_solutions
    ]  # Convert back to positive
    entropy_vals = [
        -sol["objectives"]["entropy"] for sol in pareto_solutions
    ]  # Convert back to positive

    # Create Pareto front visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Multi-Objective Pareto Front Analysis", fontsize=14, fontweight="bold")

    # MSE vs Sparsity
    axes[0, 0].scatter(mse_vals, sparsity_vals, alpha=0.7, s=60, c="blue")
    axes[0, 0].set_xlabel("MSE (reconstruction error)")
    axes[0, 0].set_ylabel("Sparsity (fraction of inactive modes)")
    axes[0, 0].set_title("MSE vs Sparsity Trade-off")
    axes[0, 0].grid(True, alpha=0.3)

    # MSE vs Entropy
    axes[0, 1].scatter(mse_vals, entropy_vals, alpha=0.7, s=60, c="green")
    axes[0, 1].set_xlabel("MSE (reconstruction error)")
    axes[0, 1].set_ylabel("Entropy (solution diversity)")
    axes[0, 1].set_title("MSE vs Entropy Trade-off")
    axes[0, 1].grid(True, alpha=0.3)

    # Sparsity vs Entropy
    axes[1, 0].scatter(sparsity_vals, entropy_vals, alpha=0.7, s=60, c="red")
    axes[1, 0].set_xlabel("Sparsity (fraction of inactive modes)")
    axes[1, 0].set_ylabel("Entropy (solution diversity)")
    axes[1, 0].set_title("Sparsity vs Entropy Trade-off")
    axes[1, 0].grid(True, alpha=0.3)

    # Solution composition heatmap
    coeffs_matrix = np.array(
        [sol["relative_contribution"] for sol in pareto_solutions[:15]]
    )  # Top 15

    im = axes[1, 1].imshow(coeffs_matrix.T, aspect="auto", cmap="viridis")
    axes[1, 1].set_xlabel("Pareto Solution Index")
    axes[1, 1].set_ylabel("Packing Mode")
    axes[1, 1].set_title("Solution Composition Matrix")
    axes[1, 1].set_yticks(range(len(packing_modes)))
    axes[1, 1].set_yticklabels(packing_modes)

    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1], label="Relative Contribution")

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / f"pareto_front_analysis{suffix}.png", dpi=300, bbox_inches="tight")
        logger.info(
            f"Pareto front visualization saved to "
            f"{save_dir / f'pareto_front_analysis{suffix}.png'}"
        )

    plt.show()


def plot_solution_heatmap(
    interp_dict: dict[str, Any],
    packing_modes: list[str],
    save_dir: Path | None = None,
    suffix: str = "",
) -> None:
    """
    Create a heatmap of all solutions with hierarchical clustering.

    Parameters
    ----------
    interp_dict
        Dictionary containing interpolated occupancy data and alternatives
    packing_modes
        List of packing mode names
    save_dir
        Directory to save plots
    suffix
        Suffix for output files
    """
    if (
        "alternatives" not in interp_dict["interpolation"]
        or interp_dict["interpolation"]["alternatives"] is None
    ):
        logger.warning("No alternative solutions to create heatmap")
        return

    alternatives = interp_dict["interpolation"]["alternatives"]

    # Collect solutions and metadata
    solutions_data = []

    # Add original solution
    orig_sol = alternatives["original_nnls"]
    solutions_data.append(
        {
            "coefficients": orig_sol["coefficients"],
            "relative_contribution": orig_sol["relative_contribution"],
            "mse": orig_sol["mse"],
            "method": "NNLS",
            "alpha": None,
        }
    )

    # Add regularized solutions
    if "regularized" in alternatives:
        for method in ["ridge", "elastic_net"]:
            if method in alternatives["regularized"]:
                for sol in alternatives["regularized"][method]:
                    solutions_data.append(
                        {
                            "coefficients": sol["coefficients"],
                            "relative_contribution": sol["relative_contribution"],
                            "mse": sol["mse"],
                            "method": method.title(),
                            "alpha": sol["alpha"],
                        }
                    )

    # Add diverse solutions
    if "diverse" in alternatives:
        for i, sol in enumerate(alternatives["diverse"][:10]):
            solutions_data.append(
                {
                    "coefficients": sol["coefficients"],
                    "relative_contribution": sol["relative_contribution"],
                    "mse": sol["mse"],
                    "method": f"Diverse-{i+1}",
                    "alpha": None,
                }
            )

    # Create DataFrame for easier manipulation
    df_data = []
    for i, sol_data in enumerate(solutions_data):
        row = {
            "solution_id": i,
            "method": sol_data["method"],
            "mse": sol_data["mse"],
            "alpha": sol_data.get("alpha"),
        }
        for j, mode in enumerate(packing_modes):
            row[f"{mode}_coeff"] = sol_data["coefficients"][j]
            row[f"{mode}_contrib"] = sol_data["relative_contribution"][j]
        df_data.append(row)

    df_solutions = pd.DataFrame(df_data)

    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Solution Space Heatmaps", fontsize=16, fontweight="bold")

    # 1. Coefficients heatmap
    coeff_cols = [f"{mode}_coeff" for mode in packing_modes]
    coeff_data = df_solutions[coeff_cols].values

    sns.heatmap(
        coeff_data.T,
        xticklabels=df_solutions["method"].tolist(),
        yticklabels=[mode for mode in packing_modes],
        annot=False,
        cmap="viridis",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Raw Coefficients")
    axes[0, 0].set_xlabel("Solutions")

    # 2. Relative contributions heatmap
    contrib_cols = [f"{mode}_contrib" for mode in packing_modes]
    contrib_data = df_solutions[contrib_cols].values

    sns.heatmap(
        contrib_data.T,
        xticklabels=df_solutions["method"].tolist(),
        yticklabels=[mode for mode in packing_modes],
        annot=False,
        cmap="RdYlBu_r",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Relative Contributions")
    axes[0, 1].set_xlabel("Solutions")

    # 3. MSE vs Alpha (for regularized solutions)
    reg_data = df_solutions[df_solutions["alpha"].notna()]
    if not reg_data.empty:
        for method in reg_data["method"].unique():
            method_data = reg_data[reg_data["method"] == method]
            axes[1, 0].semilogx(
                method_data["alpha"], method_data["mse"], "o-", label=method, alpha=0.7
            )

        axes[1, 0].set_xlabel("Regularization Parameter (α)")
        axes[1, 0].set_ylabel("MSE")
        axes[1, 0].set_title("MSE vs Regularization")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Solution similarity matrix using pairwise distances
    from scipy.spatial.distance import pdist, squareform

    # Calculate pairwise distances between solutions
    distances = pdist(contrib_data, metric="euclidean")
    distance_matrix = squareform(distances)

    sns.heatmap(
        distance_matrix,
        xticklabels=df_solutions["method"].tolist(),
        yticklabels=df_solutions["method"].tolist(),
        annot=False,
        cmap="Blues",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Solution Similarity Matrix")

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / f"solution_heatmaps{suffix}.png", dpi=300, bbox_inches="tight")
        logger.info(f"Solution heatmaps saved to {save_dir / f'solution_heatmaps{suffix}.png'}")

    plt.show()


def create_solution_summary_report(
    interp_dict: dict[str, Any],
    packing_modes: list[str],
    save_dir: Path | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Create a comprehensive summary report of all solutions.

    Parameters
    ----------
    interp_dict
        Dictionary containing interpolated occupancy data and alternatives
    packing_modes
        List of packing mode names
    save_dir
        Directory to save report
    suffix
        Suffix for output files

    Returns
    -------
    :
        DataFrame summarizing all solutions
    """
    alternatives = interp_dict["interpolation"]["alternatives"]

    summary_data = []

    # Original NNLS
    orig_sol = alternatives["original_nnls"]
    summary_data.append(
        {
            "Method": "NNLS",
            "Parameter": "N/A",
            "MSE": orig_sol["mse"],
            **{
                f"{mode}_coeff": orig_sol["coefficients"][i] for i, mode in enumerate(packing_modes)
            },
            **{
                f"{mode}_contrib": orig_sol["relative_contribution"][i]
                for i, mode in enumerate(packing_modes)
            },
            "Sparsity": np.sum(orig_sol["coefficients"] < 1e-6),
            "Max_contrib": np.max(orig_sol["relative_contribution"]),
            "Entropy": -np.sum(
                orig_sol["relative_contribution"]
                * np.log(orig_sol["relative_contribution"] + 1e-16)
            ),
        }
    )

    # Regularized solutions - keep best few
    if "regularized" in alternatives:
        for method in ["ridge", "elastic_net"]:
            if method in alternatives["regularized"]:
                solutions = alternatives["regularized"][method]
                # Keep best 5 solutions by MSE
                best_solutions = sorted(solutions, key=lambda x: x["mse"])[:5]

                for sol in best_solutions:
                    summary_data.append(
                        {
                            "Method": method.title(),
                            "Parameter": f"α={sol['alpha']:.1e}",
                            "MSE": sol["mse"],
                            **{
                                f"{mode}_coeff": sol["coefficients"][i]
                                for i, mode in enumerate(packing_modes)
                            },
                            **{
                                f"{mode}_contrib": sol["relative_contribution"][i]
                                for i, mode in enumerate(packing_modes)
                            },
                            "Sparsity": np.sum(sol["coefficients"] < 1e-6),
                            "Max_contrib": np.max(sol["relative_contribution"]),
                            "Entropy": -np.sum(
                                sol["relative_contribution"]
                                * np.log(sol["relative_contribution"] + 1e-16)
                            ),
                        }
                    )

    # Bootstrap statistics
    if "bootstrap" in alternatives:
        bootstrap_stats = alternatives["bootstrap"]["contribution_stats"]
        summary_data.append(
            {
                "Method": "Bootstrap",
                "Parameter": f"n={len(alternatives['bootstrap']['solutions'])}",
                "MSE": np.mean([s["mse"] for s in alternatives["bootstrap"]["solutions"]]),
                **{f"{mode}_coeff": bootstrap_stats[mode]["mean"] for mode in packing_modes},
                **{f"{mode}_contrib": bootstrap_stats[mode]["mean"] for mode in packing_modes},
                "Sparsity": "N/A",
                "Max_contrib": "N/A",
                "Entropy": "N/A",
            }
        )

    # Top diverse solutions
    if "diverse" in alternatives:
        for i, sol in enumerate(alternatives["diverse"][:5]):
            summary_data.append(
                {
                    "Method": "Diverse",
                    "Parameter": f"rank_{i+1}",
                    "MSE": sol["mse"],
                    **{
                        f"{mode}_coeff": sol["coefficients"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    **{
                        f"{mode}_contrib": sol["relative_contribution"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    "Sparsity": np.sum(sol["coefficients"] < 1e-6),
                    "Max_contrib": np.max(sol["relative_contribution"]),
                    "Entropy": -np.sum(
                        sol["relative_contribution"] * np.log(sol["relative_contribution"] + 1e-16)
                    ),
                }
            )

    # Pareto optimal solutions
    if "pareto_optimal" in alternatives:
        for i, sol in enumerate(alternatives["pareto_optimal"][:5]):
            summary_data.append(
                {
                    "Method": "Pareto",
                    "Parameter": f"rank_{i+1}",
                    "MSE": sol["objectives"]["mse"],
                    **{
                        f"{mode}_coeff": sol["coefficients"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    **{
                        f"{mode}_contrib": sol["relative_contribution"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    "Sparsity": np.sum(sol["coefficients"] < 1e-6),
                    "Max_contrib": np.max(sol["relative_contribution"]),
                    "Entropy": -np.sum(
                        sol["relative_contribution"] * np.log(sol["relative_contribution"] + 1e-16)
                    ),
                }
            )

    # Constrained solutions
    if "constrained" in alternatives:
        for i, sol in enumerate(alternatives["constrained"][:3]):
            summary_data.append(
                {
                    "Method": "Constrained",
                    "Parameter": f"{sol['constraint_type']}={sol['constraint_value']}",
                    "MSE": sol["mse"],
                    **{
                        f"{mode}_coeff": sol["coefficients"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    **{
                        f"{mode}_contrib": sol["relative_contribution"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    "Sparsity": np.sum(sol["coefficients"] < 1e-6),
                    "Max_contrib": np.max(sol["relative_contribution"]),
                    "Entropy": -np.sum(
                        sol["relative_contribution"] * np.log(sol["relative_contribution"] + 1e-16)
                    ),
                }
            )

    # Robust solutions
    if "robust" in alternatives:
        for i, sol in enumerate(alternatives["robust"][:3]):
            summary_data.append(
                {
                    "Method": "Robust",
                    "Parameter": f"uncertainty={sol['uncertainty_level']}",
                    "MSE": sol["mse"],
                    **{
                        f"{mode}_coeff": sol["coefficients"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    **{
                        f"{mode}_contrib": sol["relative_contribution"][j]
                        for j, mode in enumerate(packing_modes)
                    },
                    "Sparsity": np.sum(sol["coefficients"] < 1e-6),
                    "Max_contrib": np.max(sol["relative_contribution"]),
                    "Entropy": -np.sum(
                        sol["relative_contribution"] * np.log(sol["relative_contribution"] + 1e-16)
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_data)

    if save_dir:
        summary_df.to_csv(save_dir / f"solution_summary{suffix}.csv", index=False)
        logger.info(f"Solution summary saved to {save_dir / f'solution_summary{suffix}.csv'}")

    return summary_df


def analyze_solution_space(interp_dict: dict[str, Any], packing_modes: list[str]) -> None:
    """
    Analyze and log insights about the alternative solutions.

    Parameters
    ----------
    interp_dict
        Dictionary containing interpolated occupancy data and alternatives
    packing_modes
        List of packing mode names
    """
    if (
        "alternatives" not in interp_dict["interpolation"]
        or interp_dict["interpolation"]["alternatives"] is None
    ):
        logger.warning("No alternative solutions found to analyze")
        return

    alternatives = interp_dict["interpolation"]["alternatives"]

    # Print solution diversity analysis
    logger.info("Solution Space Analysis:")
    logger.info("=" * 50)

    # Compare regularized solutions
    if "regularized" in alternatives:
        ridge_solutions = alternatives["regularized"]["ridge"]
        best_ridge = min(ridge_solutions, key=lambda x: x["mse"])
        logger.info(f"Best Ridge solution (α={best_ridge['alpha']:.4f}):")
        for i, mode in enumerate(packing_modes):
            logger.info(f"  {mode}: {best_ridge['relative_contribution'][i]:.4f}")
        logger.info(f"  MSE: {best_ridge['mse']:.6f}")

    # Bootstrap confidence intervals
    if "bootstrap" in alternatives:
        bootstrap_stats = alternatives["bootstrap"]["contribution_stats"]
        logger.info("\nBootstrap Confidence Intervals (25%-75%):")
        for mode in packing_modes:
            stats = bootstrap_stats[mode]
            logger.info(f"  {mode}: {stats['percentile_25']:.4f} - {stats['percentile_75']:.4f}")

    # Diverse solutions summary
    if "diverse" in alternatives:
        diverse_sols = alternatives["diverse"][:5]  # Top 5 diverse solutions
        logger.info(f"\nTop {len(diverse_sols)} diverse solutions:")
        for i, sol in enumerate(diverse_sols):
            logger.info(f"  Solution {i+1} (MSE: {sol['mse']:.6f}):")
            for j, mode in enumerate(packing_modes):
                logger.info(f"    {mode}: {sol['relative_contribution'][j]:.4f}")

    # Pareto optimal solutions summary
    if "pareto_optimal" in alternatives:
        pareto_sols = alternatives["pareto_optimal"][:3]  # Top 3 Pareto solutions
        logger.info(f"\nTop {len(pareto_sols)} Pareto optimal solutions:")
        for i, sol in enumerate(pareto_sols):
            logger.info(f"  Solution {i+1} (MSE: {sol['objectives']['mse']:.6f}):")
            logger.info(f"    Objectives: {sol['objectives']}")
            for j, mode in enumerate(packing_modes):
                logger.info(f"    {mode}: {sol['relative_contribution'][j]:.4f}")

    # Constrained solutions summary
    if "constrained" in alternatives:
        const_sols = alternatives["constrained"][:3]  # Top 3 constrained solutions
        logger.info(f"\nTop {len(const_sols)} constrained solutions:")
        for i, sol in enumerate(const_sols):
            logger.info(f"  Solution {i+1} (MSE: {sol['mse']:.6f}):")
            logger.info(f"    Constraint: {sol['constraint_type']} = {sol['constraint_value']}")
            for j, mode in enumerate(packing_modes):
                logger.info(f"    {mode}: {sol['relative_contribution'][j]:.4f}")

    # Robust solutions summary
    if "robust" in alternatives:
        robust_sols = alternatives["robust"][:3]  # Top 3 robust solutions
        logger.info(f"\nTop {len(robust_sols)} robust solutions:")
        for i, sol in enumerate(robust_sols):
            logger.info(
                f"  Solution {i+1} (Base MSE: {sol['mse']:.6f}, "
                f"Robust MSE: {sol['robust_mse']:.6f}):"
            )
            logger.info(f"    Uncertainty level: {sol['uncertainty_level']}")
            for j, mode in enumerate(packing_modes):
                logger.info(f"    {mode}: {sol['relative_contribution'][j]:.4f}")


def interpolate_occupancy_dict(
    occupancy_dict: dict[str, dict[str, dict[str, dict[str, Any]]]],
    channel_map: dict[str, str],
    baseline_mode: str,
    results_dir: Path | None = None,
    suffix: str = "",
    explore_alternatives: bool = True,
    n_bootstrap: int = 50,
    alpha_range: np.ndarray = np.logspace(-4, 2, 10),
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
    explore_alternatives
        Whether to explore alternative solutions beyond NNLS
    n_bootstrap
        Number of bootstrap samples for solution stability analysis
    alpha_range
        Range of regularization parameters for Ridge/ElasticNet

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

    # Explore alternative solutions if requested
    if explore_alternatives:
        logger.info("Exploring alternative solutions...")

        # Store original NNLS solution
        original_mse = np.mean(
            (stacked_baseline_occupancy - stacked_simulated_occupancy_matrix @ coeffs_joint) ** 2
        )
        mse_threshold = original_mse * 1.05  # 5% tolerance

        logger.info(f"NNLS baseline MSE: {original_mse:.6f}")
        logger.info(f"MSE threshold (5% tolerance): {mse_threshold:.6f}")

        # Regularized solutions
        reg_solutions = explore_regularized_solutions(
            stacked_simulated_occupancy_matrix,
            stacked_baseline_occupancy,
            packing_modes,
            alpha_range,
        )
        # Filter regularized solutions by MSE threshold
        reg_solutions["ridge"] = [
            sol for sol in reg_solutions["ridge"] if sol["mse"] <= mse_threshold
        ]
        reg_solutions["elastic_net"] = [
            sol for sol in reg_solutions["elastic_net"] if sol["mse"] <= mse_threshold
        ]

        # Bootstrap analysis
        bootstrap_results = bootstrap_solutions(
            stacked_simulated_occupancy_matrix,
            stacked_baseline_occupancy,
            packing_modes,
            n_bootstrap,
            noise_level=0,
        )
        # Filter bootstrap solutions by MSE threshold
        bootstrap_results["solutions"] = [
            sol for sol in bootstrap_results["solutions"] if sol["mse"] <= mse_threshold
        ]

        # Recalculate bootstrap statistics after filtering
        if bootstrap_results["solutions"]:
            filtered_coeffs = np.array(
                [sol["coefficients"] for sol in bootstrap_results["solutions"]]
            )
            filtered_contributions = np.array(
                [sol["relative_contribution"] for sol in bootstrap_results["solutions"]]
            )

            bootstrap_results["coefficient_stats"] = {
                mode: {
                    "mean": np.mean(filtered_coeffs[:, i]),
                    "std": np.std(filtered_coeffs[:, i]),
                    "percentile_25": np.percentile(filtered_coeffs[:, i], 25),
                    "percentile_75": np.percentile(filtered_coeffs[:, i], 75),
                }
                for i, mode in enumerate(packing_modes)
            }
            bootstrap_results["contribution_stats"] = {
                mode: {
                    "mean": np.mean(filtered_contributions[:, i]),
                    "std": np.std(filtered_contributions[:, i]),
                    "percentile_25": np.percentile(filtered_contributions[:, i], 25),
                    "percentile_75": np.percentile(filtered_contributions[:, i], 75),
                }
                for i, mode in enumerate(packing_modes)
            }

        # Diverse solutions
        diverse_solutions = find_diverse_solutions(
            stacked_simulated_occupancy_matrix, stacked_baseline_occupancy, packing_modes
        )
        # Filter diverse solutions by MSE threshold
        diverse_solutions = [sol for sol in diverse_solutions if sol["mse"] <= mse_threshold]

        # Multi-objective Pareto optimal solutions
        pareto_solutions = find_pareto_optimal_solutions(
            stacked_simulated_occupancy_matrix,
            stacked_baseline_occupancy,
            packing_modes,
            num_solutions=25,
            objectives=["mse", "sparsity", "entropy"],
        )
        # Filter Pareto solutions by MSE threshold
        pareto_solutions = [
            sol for sol in pareto_solutions if sol["objectives"]["mse"] <= mse_threshold
        ]

        # Constrained solutions
        constrained_solutions = find_constrained_solutions(
            stacked_simulated_occupancy_matrix, stacked_baseline_occupancy, packing_modes
        )
        # Filter constrained solutions by MSE threshold
        constrained_solutions = [
            sol for sol in constrained_solutions if sol["mse"] <= mse_threshold
        ]

        # Robust solutions
        robust_solutions = find_robust_solutions(
            stacked_simulated_occupancy_matrix, stacked_baseline_occupancy, packing_modes
        )
        # Filter robust solutions by MSE threshold (use base MSE, not robust MSE)
        robust_solutions = [sol for sol in robust_solutions if sol["mse"] <= mse_threshold]

        # Log filtering results
        logger.info("Filtered solution counts:")
        logger.info(f"  Ridge solutions: {len(reg_solutions['ridge'])}")
        logger.info(f"  Elastic Net solutions: {len(reg_solutions['elastic_net'])}")
        logger.info(f"  Bootstrap solutions: {len(bootstrap_results['solutions'])}")
        logger.info(f"  Diverse solutions: {len(diverse_solutions)}")
        logger.info(f"  Pareto optimal solutions: {len(pareto_solutions)}")
        logger.info(f"  Constrained solutions: {len(constrained_solutions)}")
        logger.info(f"  Robust solutions: {len(robust_solutions)}")

        interp_dict["interpolation"]["alternatives"] = {
            "regularized": reg_solutions,
            "bootstrap": bootstrap_results,
            "diverse": diverse_solutions,
            "pareto_optimal": pareto_solutions,
            "constrained": constrained_solutions,
            "robust": robust_solutions,
            "original_nnls": {
                "coefficients": coeffs_joint,
                "relative_contribution": relative_contribution_joint,
                "mse": original_mse,
            },
        }
    else:
        interp_dict["interpolation"]["alternatives"] = None

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

    # Create comprehensive visualizations if alternatives were explored
    if explore_alternatives and interp_dict["interpolation"]["alternatives"]:
        logger.info("Creating solution space visualizations...")
        try:
            # Create all visualizations
            visualize_solution_space(interp_dict, packing_modes, results_dir, suffix)
            plot_solution_heatmap(interp_dict, packing_modes, results_dir, suffix)
            plot_pareto_front(interp_dict, packing_modes, results_dir, suffix)

            # Create summary report
            create_solution_summary_report(interp_dict, packing_modes, results_dir, suffix)

            # Log key insights
            analyze_solution_space(interp_dict, packing_modes)

        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")
            raise e

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
