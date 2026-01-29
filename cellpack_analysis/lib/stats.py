from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import integrate

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two independent samples.

    Parameters
    ----------
    x
        First sample array
    y
        Second sample array

    Returns
    -------
    :
        Cohen's d effect size (absolute value)
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return np.abs(np.mean(x) - np.mean(y)) / pooled_std


def normalize_distances(
    all_distance_dict: dict[str, Any],
    mesh_information_dict: dict[str, Any],
    channel_map: dict[str, str],
    normalization: str | None = None,
    pixel_size_in_um: float = PIXEL_SIZE_IN_UM,
) -> dict[str, Any]:
    """
    Normalize distances using specified normalization method.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing distance measurements by measure and mode
    mesh_information_dict
        Dictionary containing mesh information for normalization
    normalization
        Normalization method: 'intracellular_radius', 'cell_diameter', 'max_distance',
        or None for pixel size
    channel_map
        Mapping between modes and mesh information keys
    pixel_size_in_um
        Pixel size for default normalization

    Returns
    -------
    :
        Dictionary with normalized distances
    """
    for measure, mode_distance_dict in all_distance_dict.items():
        if "scaled" in measure:
            continue
        for mode, distance_dict in mode_distance_dict.items():
            mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, ""), {})
            for cell_id, distance in distance_dict.items():
                mesh_info = mode_mesh_dict.get(
                    cell_id,
                    mode_mesh_dict.get("mean", {"intracellular_radius": 1}),
                )

                if normalization == "intracellular_radius":
                    normalization_factor = mesh_info["intracellular_radius"]
                elif normalization == "cell_diameter":
                    normalization_factor = mesh_info["cell_diameter"]
                elif normalization == "max_distance":
                    normalization_factor = distance.max()
                else:
                    normalization_factor = 1 / pixel_size_in_um

                distance_dict[cell_id] = distance / normalization_factor

    return all_distance_dict


def ripley_k(
    positions: np.ndarray,
    volume: float,
    r_values: np.ndarray,
    norm_factor: float = 1,
    edge_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Ripley's K metric for spatial point patterns.

    Parameters
    ----------
    positions
        Array of shape (n, 3) representing n points in 3D space
    volume
        Volume of the space
    r_values
        Array of distances at which to calculate K(r)
    norm_factor
        Normalization factor for distance calculation
    edge_correction
        If True, apply border correction

    Returns
    -------
    :
        Tuple containing K(r) values and input r_values array
    """
    num_positions = positions.shape[0]

    if num_positions < 2:
        return np.zeros_like(r_values), r_values

    # Calculate all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2) / norm_factor

    # Get upper triangular part to avoid double counting and self-distances
    triu_indices = np.triu_indices(num_positions, k=1)
    distances_upper = distances[triu_indices]

    ripley_k_values = np.zeros_like(r_values, dtype=float)

    # For each radius r, count pairs within distance r
    for i, r in enumerate(r_values):
        pairs_within_r = np.sum(distances_upper <= r)

        # Basic Ripley's K formula: K(r) = V * (number of pairs within distance r) / (n * (n-1) / 2)
        ripley_k_values[i] = volume * pairs_within_r / (num_positions * (num_positions - 1) / 2)

    return ripley_k_values, r_values


def normalize_pdf(xvals: np.ndarray, density: np.ndarray) -> np.ndarray:
    """
    Normalize density to integrate to 1.

    Parameters
    ----------
    xvals
        The x-values of the density
    density
        The density values

    Returns
    -------
    :
        Normalized density
    """
    integral = integrate.trapezoid(density, xvals)
    return density / integral if integral != 0 else density


def pdf_ratio(
    xvals: np.ndarray, density1: np.ndarray, density2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the density ratio between two densities.

    Parameters
    ----------
    xvals
        The x-values of the densities
    density1
        The first density
    density2
        The second density

    Returns
    -------
    :
        Tuple containing the density ratio, normalized density1, and normalized density2
    """
    # regularize
    reg = 1e-10
    density1_reg = np.maximum(density1, reg)
    density2_reg = np.maximum(density2, reg)

    # normalize densities
    density1_norm = normalize_pdf(xvals, density1_reg)
    density2_norm = normalize_pdf(xvals, density2_reg)

    # Calculate ratio in log space
    log_ratio = np.log(density1_norm) - np.log(density2_norm)
    density_ratio = np.exp(log_ratio)

    density_ratio = np.nan_to_num(density_ratio, nan=0.0, posinf=0.0, neginf=0.0)

    return density_ratio, density1_norm, density2_norm


def cpdf_ratio(
    xvals: np.ndarray, density1: np.ndarray, density2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the cumulative ratio between two density distributions.

    Parameters
    ----------
    xvals
        The x-values of the density distributions
    density1
        The first density distribution
    density2
        The second density distribution

    Returns
    -------
    :
        Tuple containing the cumulative ratio, normalized density1, and normalized density2
    """
    cumulative_ratio = np.zeros(len(xvals))
    density1 = normalize_pdf(xvals, density1)
    density2 = normalize_pdf(xvals, density2)
    for ct in range(len(xvals)):
        cumulative_ratio[ct] = integrate.trapezoid(
            density1[: ct + 1], xvals[: ct + 1]
        ) / integrate.trapezoid(density2[: ct + 1], xvals[: ct + 1])
    return cumulative_ratio, density1, density2


def get_pdf_ratio(
    xvals: np.ndarray,
    density_numerator: np.ndarray,
    density_denominator: np.ndarray,
    method: Literal["pdf", "cumulative"] = "pdf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the ratio of two probability density functions (PDFs) based on the given method.

    Parameters
    ----------
    xvals
        The x-values of the PDFs
    density_numerator
        Density values of the numerator PDF
    density_denominator
        Density values of the denominator PDF
    method
        The method to calculate the ratio

    Returns
    -------
    :
        Tuple containing the ratio and normalized densities based on the specified method
    """
    if method == "pdf":
        return pdf_ratio(xvals, density_numerator, density_denominator)
    elif method == "cumulative":
        return cpdf_ratio(xvals, density_numerator, density_denominator)
    else:
        raise ValueError(f"Invalid ratio method: {method}")


def create_padded_numpy_array(
    lists: Sequence[list[float] | np.ndarray], padding: float = np.nan
) -> np.ndarray:
    """
    Create a padded array with the specified padding value.

    Parameters
    ----------
    lists
        List of arrays or lists to pad
    padding
        Value to use for padding

    Returns
    -------
    :
        Padded numpy array with all sublists having the same length
    """
    max_length = max([len(sublist) for sublist in lists])
    padded_array = np.zeros((len(lists), max_length))
    for ct, sublist in enumerate(lists):
        if len(sublist) < max_length:
            if isinstance(sublist, list):
                sublist += [padding] * (max_length - len(sublist))
            elif isinstance(sublist, np.ndarray):
                sublist = np.append(sublist, [padding] * (max_length - len(sublist)))
        padded_array[ct] = sublist[:]
    return padded_array


def ecdf(values: np.ndarray | list, r_grid: np.ndarray) -> np.ndarray:
    """
    Calculate empirical cumulative distribution function at specified grid points.

    Parameters
    ----------
    values
        Array of observed values to compute ECDF from
    r_grid
        Grid points at which to evaluate the ECDF

    Returns
    -------
    :
        ECDF values evaluated at each grid point
    """
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(r_grid, dtype=float)
    v_sorted = np.sort(v)
    return np.searchsorted(v_sorted, r_grid, side="right") / v_sorted.size


def make_r_grid_from_pooled(
    list_of_arrays: Sequence[np.ndarray | None],
    n: int = 100,
    extend: float = 1.05,
    qmax: float = 99.5,
) -> np.ndarray:
    """
    Build r-grid from pooled values over observed and simulation data.

    Parameters
    ----------
    list_of_arrays
        List of arrays containing distance values to pool
    n
        Number of grid points to generate
    extend
        Extension factor beyond maximum value
    qmax
        Percentile to use as maximum value before extension

    Returns
    -------
    :
        Linearly spaced grid from 0 to extended maximum
    """
    vals = np.concatenate(
        [np.asarray(a)[np.isfinite(a)] for a in list_of_arrays if a is not None and len(a) > 0],
        axis=0,
    )
    if vals.size == 0:
        return np.linspace(0.0, 1.0, n)
    vmax = np.percentile(vals, qmax)
    vmax = max(vmax, np.finfo(float).eps)
    return np.linspace(0.0, vmax * extend, n)


def pointwise_envelope(
    sim_curves: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pointwise Monte Carlo envelope from simulation curves.

    Parameters
    ----------
    sim_curves
        Array of shape (M, L) containing M simulation ECDF curves
    alpha
        Significance level for envelope bounds

    Returns
    -------
    :
        Tuple containing (lower_bound, upper_bound, mean, std_dev)
    """
    lo = np.quantile(sim_curves, alpha / 2, axis=0)
    hi = np.quantile(sim_curves, 1 - alpha / 2, axis=0)
    mu = sim_curves.mean(axis=0)
    sd = sim_curves.std(axis=0, ddof=1)
    sd[sd == 0] = 1e-9
    return lo, hi, mu, sd


def global_sup_deviation_pvalue(
    obs_curve: np.ndarray, sim_curves: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """
    Calculate global test statistic and p-value using supremum deviation.

    Computes test statistic T = max_j |(obs - mu_j)/sd_j| and Monte Carlo
    p-value as (1 + #{T_sim >= T_obs}) / (M + 1).

    Parameters
    ----------
    obs_curve
        Observed ECDF curve
    sim_curves
        Array of shape (M, L) containing M simulation ECDF curves

    Returns
    -------
    :
        Tuple containing (p_value, T_observed, T_simulated_array)
    """
    mu = sim_curves.mean(axis=0)
    sd = sim_curves.std(axis=0, ddof=1)
    sd[sd == 0] = 1e-9
    T_obs = np.max(np.abs((obs_curve - mu) / sd))
    T_sim = np.max(np.abs((sim_curves - mu) / sd), axis=1)
    p = (np.sum(T_sim >= T_obs) + 1) / (T_sim.size + 1)
    return float(p), float(T_obs), T_sim


def analyze_cell_from_metrics(
    obs_metrics: dict[str, np.ndarray],
    sims_metrics_per_model: dict[str, list[dict[str, np.ndarray]]],
    alpha: float = 0.05,
    metrics_order: list[str] | None = None,
    r_grid_size: int = 150,
) -> dict[str, dict]:
    """
    Analyze single cell by comparing observed metrics against simulated null models.

    Performs per-metric envelope tests and joint test across all metrics
    for each null model using ECDF-based Monte Carlo testing.

    Parameters
    ----------
    obs_metrics
        Dictionary mapping metric names to observed distance arrays
    sims_metrics_per_model
        Dictionary mapping model names to lists of R replicate metric dictionaries
    alpha
        Significance level for envelope construction
    metrics_order
        Ordered list of metric names to analyze. If None, uses obs_metrics keys
    r_grid_size
        Number of points in r-grid for ECDF evaluation

    Returns
    -------
    :
        Nested dictionary with structure: model -> {
            'per_metric': {metric -> {'r', 'obs_curve', 'lo', 'hi', 'mu', 'sd', 'pval', 'T_obs'}},
            'joint': {'pval', 'T_obs', 'metric_order'}
        }
    """
    models = list(sims_metrics_per_model.keys())
    R = len(next(iter(sims_metrics_per_model.values())))  # number of replicates per model
    if metrics_order is None:
        metrics_order = list(obs_metrics.keys())

    results = {}
    for m in models:
        reps = sims_metrics_per_model[m]  # list of dicts
        per_metric = {}

        # First pass: build r-grids per metric based on pooled obs + sims
        r_grids = {}
        for metric in metrics_order:
            pooled = [obs_metrics.get(metric, np.array([]))] + [
                rep.get(metric, np.array([])) for rep in reps
            ]
            r_grids[metric] = make_r_grid_from_pooled(pooled, n=r_grid_size)

        # Second pass: compute curves, envelopes, p-values
        sim_curves_by_metric = {}
        obs_curves_by_metric = {}
        for metric in metrics_order:
            r = r_grids[metric]
            # observed curve
            obs_curve = ecdf(obs_metrics.get(metric, np.array([])), r)
            # simulation curves
            sim_mat = np.vstack([ecdf(rep.get(metric, np.array([])), r) for rep in reps])  # (R, L)
            lo, hi, mu, sd = pointwise_envelope(sim_mat, alpha=alpha)
            pval, Tobs, _ = global_sup_deviation_pvalue(obs_curve, sim_mat)

            per_metric[metric] = {
                "r": r,
                "obs_curve": obs_curve,
                "lo": lo,
                "hi": hi,
                "mu": mu,
                "sd": sd,
                "pval": pval,
                "T_obs": Tobs,
            }
            sim_curves_by_metric[metric] = sim_mat
            obs_curves_by_metric[metric] = obs_curve

        # Joint test: concatenate curves
        sim_concat = np.vstack(
            [
                np.concatenate(
                    [sim_curves_by_metric[metric][i] for metric in metrics_order], axis=0
                )
                for i in range(R)
            ]
        )  # (R, sum L_m)
        obs_concat = np.concatenate(
            [obs_curves_by_metric[metric] for metric in metrics_order], axis=0
        )

        # Now compute joint p-value in standardized sup-deviation sense
        p_joint, Tobs_joint, _ = global_sup_deviation_pvalue(obs_concat, sim_concat)

        results[m] = {
            "per_metric": per_metric,
            "joint": {"pval": p_joint, "T_obs": Tobs_joint, "metric_order": metrics_order},
        }

    return results


def benjamini_hochberg(pvals: np.ndarray | list) -> np.ndarray:
    """
    Apply Benjamini-Hochberg procedure for multiple testing correction.

    Parameters
    ----------
    pvals
        Array of p-values to adjust

    Returns
    -------
    :
        Array of adjusted q-values in same order as input, clipped to [0, 1]
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # ensure monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_vals = np.empty_like(q_sorted)
    q_vals[order] = q_sorted
    return np.clip(q_vals, 0, 1)


def summarize_across_cells(
    all_results: list[dict[str, dict]], alpha: float = 0.05
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Aggregate CSR test results across multiple cells with BH correction.

    Applies Benjamini-Hochberg correction separately for each model/metric
    combination and computes rejection rates at specified alpha level.

    Parameters
    ----------
    all_results
        List of per-cell result dictionaries from analyze_cell_from_metrics
    alpha
        Significance level for rejection rate calculation

    Returns
    -------
    :
        Dictionary containing:
        - 'per_metric_pvals': DataFrame with rows=cells, columns=(model, metric)
        - 'per_metric_qvals': BH-adjusted q-values
        - 'joint_pvals': DataFrame with rows=cells, columns=models
        - 'joint_qvals': BH-adjusted joint test q-values
        - 'rejection_rates': {'per_metric': pd.Series, 'joint': pd.Series} with
          fraction of cells rejected per model/metric
    """
    # Discover models and metrics from first cell
    first_cell = all_results[0]
    models = list(first_cell.keys())
    metrics = list(first_cell[models[0]]["per_metric"].keys())

    C = len(all_results)
    # Collect p-values
    per_metric_p = pd.DataFrame(
        index=np.arange(C),
        columns=pd.MultiIndex.from_product([models, metrics], names=["model", "metric"]),
        dtype=float,
    )
    joint_p = pd.DataFrame(index=np.arange(C), columns=models, dtype=float)

    for c in range(C):
        for m in models:
            for metric in metrics:
                per_metric_p.loc[c, (m, metric)] = all_results[c][m]["per_metric"][metric]["pval"]
            joint_p.loc[c, m] = all_results[c][m]["joint"]["pval"]

    # BH across cells per (model, metric)
    per_metric_q = per_metric_p.copy()
    for m in models:
        for metric in metrics:
            per_metric_q[(m, metric)] = benjamini_hochberg(
                np.asarray(per_metric_p[(m, metric)].values)
            )

    # BH across cells for joint p-values per model
    joint_q = joint_p.copy()
    for m in models:
        joint_q[m] = benjamini_hochberg(np.asarray(joint_p[m].values))

    # Rejection rates (q < alpha)
    rej_metric = (per_metric_q < alpha).mean(axis=0)  # fraction of cells rejected
    rej_joint = (joint_q < alpha).mean(axis=0)

    return {
        "per_metric_pvals": per_metric_p,
        "per_metric_qvals": per_metric_q,
        "joint_pvals": joint_p,
        "joint_qvals": joint_q,
        "rejection_per_metric": rej_metric,
        "rejection_joint": rej_joint,
    }
