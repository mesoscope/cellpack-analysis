from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import false_discovery_control


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
    bin_width: float | None = None,
) -> np.ndarray:
    """
    Build r-grid from pooled values over observed and simulation data.

    Parameters
    ----------
    list_of_arrays
        List of arrays containing distance values to pool
    n
        Number of grid points to generate (used if bin_width is None)
    extend
        Extension factor beyond maximum value
    qmax
        Percentile to use as maximum value before extension
    bin_width
        If provided, determines number of bins based on data range / bin_width.
        Takes precedence over n parameter.

    Returns
    -------
    :
        Linearly spaced grid of r values covering the range of the pooled data with
        specified extension
    """
    vals = np.concatenate(
        [np.asarray(a)[np.isfinite(a)] for a in list_of_arrays if a is not None and len(a) > 0],
        axis=0,
    )
    if vals.size == 0:
        return np.linspace(0.0, 1.0, n)
    vmax = np.percentile(vals, qmax)
    vmax = max(vmax, np.finfo(float).eps)

    # Use bin_width to calculate number of bins if provided
    if bin_width is not None:
        # Extend grid by half bin_width on either side to ensure 0 and vmax lie within bins
        n = int((vmax * extend) / bin_width) + 2
        r_grid = np.linspace(-bin_width / 2, vmax * extend + bin_width / 2, n)
    else:
        r_grid = np.linspace(0.0, vmax * extend, n)

    return r_grid


def pointwise_envelope(
    sim_curves: np.ndarray, alpha: float = 0.1
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


def get_test_statistic_and_pvalue(
    obs_curve: np.ndarray,
    sim_curves: np.ndarray,
    statistic: Literal["supremum", "intdev"] = "supremum",
) -> tuple[float, float, np.ndarray]:
    """
    Calculate global test statistic and p-value using supremum deviation.

    Computes test statistic
        T = max_j |(obs - mu_j)/sd_j| for "supremum" or
        T = integrate |(obs - mu)/sd| for "intdev"
    Monte Carlo p-value is (1 + #{T_sim >= T_obs}) / (M + 1).

    Parameters
    ----------
    obs_curve
        Observed ECDF curve
    sim_curves
        Array of shape (M, L) containing M simulation ECDF curves
    statistic
        Test statistic to compute: "supremum" or "intdev"

    Returns
    -------
    :
        Tuple containing (p_value, T_observed, T_simulated_array)
    """
    mu = sim_curves.mean(axis=0)
    sd = sim_curves.std(axis=0, ddof=1)
    sd[sd == 0] = 1e-9
    if statistic == "supremum":
        t_obs = np.max(np.abs((obs_curve - mu) / sd))
        t_sim = np.max(np.abs((sim_curves - mu) / sd), axis=1)
    elif statistic == "intdev":
        t_obs = integrate.trapezoid(np.abs((obs_curve - mu) / sd))
        t_sim = np.array(
            [
                integrate.trapezoid(np.abs((sim_curves[i] - mu) / sd))
                for i in range(sim_curves.shape[0])
            ]
        )
    else:
        raise ValueError(f"Invalid statistic: {statistic}")
    p = (np.sum(t_sim >= t_obs) + 1) / (t_sim.size + 1)
    return float(p), float(t_obs), t_sim


def monte_carlo_per_cell(
    observed_distances: dict[str, np.ndarray],
    simulated_distances_by_mode: dict[str, list[dict[str, np.ndarray]]],
    alpha: float = 0.05,
    distance_measures: list[str] | None = None,
    r_grid_size: int = 150,
    statistic: Literal["supremum", "intdev"] = "supremum",
) -> dict[str, dict]:
    """
    Analyze single cell by comparing observed distances against simulated null models.

    Performs per-distance measure envelope tests and joint test across all distance measures
    for each null model using ECDF-based Monte Carlo testing.

    Parameters
    ----------
    observed_distances
        Dictionary mapping metric names to observed distance arrays
        {distance_measure: num_cellxnum_points np.ndarray of distances}
    simulated_distances_by_mode
        Dictionary mapping packing mode names to lists of simulated distance dictionaries
        {packing_mode:[{distance_measure: num_cellxnum_points np.ndarray of distances}xreplicates]}
    alpha
        Significance level for envelope construction
    distance_measures
        List of distance measures to analyze; if None, use all from observed_distances
    r_grid_size
        Number of points in r-grid for ECDF evaluation
    statistic
        Test statistic to use for joint test.
        "supremum": use maximum standardized deviation across r-grid points
        "intdev": use integrated standardized deviation across r-grid points

    Returns
    -------
    :
        Nested dictionary with structure: packing_mode -> {
            'per_distance_measure': {distance_measure ->
                {'r', 'obs_curve', 'lo', 'hi', 'mu', 'sd', 'pval', 'T_obs'}},
            'joint': {'pval', 'T_obs', 'distance_measures'}
        }
    """
    packing_modes = list(simulated_distances_by_mode.keys())
    num_replicates = len(
        next(iter(simulated_distances_by_mode.values()))
    )  # number of replicates per model
    if distance_measures is None:
        distance_measures = list(observed_distances.keys())

    results = {}
    for packing_mode in packing_modes:
        reps = simulated_distances_by_mode[packing_mode]  # list of dicts
        per_distance_measure = {}

        # First pass: build r-grids per distance measure based on pooled obs + sims
        r_grids = {}
        for distance_measure in distance_measures:
            pooled = [observed_distances.get(distance_measure, np.array([]))] + [
                rep.get(distance_measure, np.array([])) for rep in reps
            ]
            r_grids[distance_measure] = make_r_grid_from_pooled(pooled, n=r_grid_size)

        # Second pass: compute curves, envelopes, p-values
        sim_curves_by_distance_measure = {}
        obs_curves_by_distance_measure = {}
        for distance_measure in distance_measures:
            r = r_grids[distance_measure]
            # observed curve
            obs_curve = ecdf(observed_distances.get(distance_measure, np.array([])), r)
            # simulation curves
            sim_mat = np.vstack(
                [ecdf(rep.get(distance_measure, np.array([])), r) for rep in reps]
            )  # (R, L)
            lo, hi, mu, sd = pointwise_envelope(sim_mat, alpha=alpha)
            pval, tobs, _ = get_test_statistic_and_pvalue(obs_curve, sim_mat, statistic=statistic)

            per_distance_measure[distance_measure] = {
                "r": r,
                "obs_curve": obs_curve,
                "lo": lo,
                "hi": hi,
                "mu": mu,
                "sd": sd,
                "pval": pval,
                "T_obs": tobs,
            }
            sim_curves_by_distance_measure[distance_measure] = sim_mat
            obs_curves_by_distance_measure[distance_measure] = obs_curve

        # Joint test: concatenate curves
        sim_concat = np.vstack(
            [
                np.concatenate(
                    [
                        sim_curves_by_distance_measure[distance_measure][i]
                        for distance_measure in distance_measures
                    ],
                    axis=0,
                )
                for i in range(num_replicates)
            ]
        )  # (R, sum L_m)
        obs_concat = np.concatenate(
            [
                obs_curves_by_distance_measure[distance_measure]
                for distance_measure in distance_measures
            ],
            axis=0,
        )

        # Now compute joint p-value in standardized sup-deviation sense
        p_joint, tobs_joint, _ = get_test_statistic_and_pvalue(obs_concat, sim_concat)

        results[packing_mode] = {
            "per_distance_measure": per_distance_measure,
            "joint": {"pval": p_joint, "T_obs": tobs_joint, "distance_measures": distance_measures},
        }

    return results


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
        - 'per_distance_measure_pvals': DataFrame with rows=cells,
            columns=(packing_mode, distance_measure)
        - 'per_distance_measure_qvals': BH-adjusted q-values
        - 'joint_pvals': DataFrame with rows=cells, columns=packing_mode
        - 'joint_qvals': BH-adjusted joint test q-values
        - 'rejection_per_distance_measure': {'per_distance_measure': pd.Series, 'joint': pd.Series}
            with fraction of cells rejected per packing_mode/distance_measure
    """
    # Discover models and metrics from first cell
    first_cell = all_results[0]
    packing_modes = list(first_cell.keys())
    distance_measures = list(first_cell[packing_modes[0]]["per_distance_measure"].keys())

    num_cells = len(all_results)
    # Collect p-values
    per_distance_measure_p = pd.DataFrame(
        index=np.arange(num_cells),
        columns=pd.MultiIndex.from_product(
            [packing_modes, distance_measures], names=["packing_mode", "distance_measure"]
        ),
        dtype=float,
    )
    joint_p = pd.DataFrame(index=np.arange(num_cells), columns=packing_modes, dtype=float)

    for cell_idx in range(num_cells):
        for packing_mode in packing_modes:
            for distance_measure in distance_measures:
                per_distance_measure_p.loc[cell_idx, (packing_mode, distance_measure)] = (
                    all_results[cell_idx][packing_mode]["per_distance_measure"][distance_measure][
                        "pval"
                    ]
                )
            joint_p.loc[cell_idx, packing_mode] = all_results[cell_idx][packing_mode]["joint"][
                "pval"
            ]

    # BH across cells per (model, distance_measure)
    per_distance_measure_q = per_distance_measure_p.copy()
    for packing_mode in packing_modes:
        for distance_measure in distance_measures:
            per_distance_measure_q[(packing_mode, distance_measure)] = false_discovery_control(
                np.asarray(per_distance_measure_p[(packing_mode, distance_measure)].values),
                method="bh",
            )

    # BH across cells for joint p-values per model
    joint_q = joint_p.copy()
    for packing_mode in packing_modes:
        joint_q[packing_mode] = false_discovery_control(
            np.asarray(joint_p[packing_mode].values), method="bh"
        )

    # Rejection rates (q < alpha)
    rej_distance_measure = (per_distance_measure_q < alpha).mean(
        axis=0
    )  # fraction of cells rejected
    rej_joint = (joint_q < alpha).mean(axis=0)

    return {
        "per_distance_measure_pvals": per_distance_measure_p,
        "per_distance_measure_qvals": per_distance_measure_q,
        "joint_pvals": joint_p,
        "joint_qvals": joint_q,
        "rejection_per_distance_measure": rej_distance_measure,
        "rejection_joint": rej_joint,
    }
