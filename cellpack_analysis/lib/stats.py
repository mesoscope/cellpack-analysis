import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import false_discovery_control

EnvelopeType = Literal["pointwise", "rank"]


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
    if integral == 0:
        return np.full_like(density, np.nan, dtype=float)
    return density / integral


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
    reg = 1e-10

    # normalize first so regularization doesn't bias the normalization integral
    density1_norm = normalize_pdf(xvals, density1)
    density2_norm = normalize_pdf(xvals, density2)

    # Calculate ratio in log space, applying floor only inside the log
    log_ratio = np.log(np.maximum(density1_norm, reg)) - np.log(np.maximum(density2_norm, reg))
    density_ratio = np.exp(log_ratio)

    # nan=nan: NaN arises when normalize_pdf returns all-NaN (zero-integral input);
    #   propagate honestly so downstream nanmean ignores these cells.
    # posinf=nan: denominator→0 with non-zero numerator → unreliable, propagate NaN
    density_ratio = np.nan_to_num(density_ratio, nan=np.nan, posinf=np.nan, neginf=0.0)

    return density_ratio, density1_norm, density2_norm


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


def compute_rank_envelope(
    obs_curve: np.ndarray,
    sim_curves: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """
    Compute a simultaneous rank envelope test (Myllymäki et al. 2017).

    Pools the observed curve with M simulation curves (pool size s = M+1) and
    assigns 1-indexed ranks at each grid point. Two tail statistics are computed:

    - ``u_i = min_j rank_{ij}``        — how extreme from *below* (1 = most extreme)
    - ``v_i = min_j (s+1 - rank_{ij})``— how extreme from *above* (1 = most extreme)

    The signed direction of the observed curve determines which tail produces the
    exact Monte Carlo p-value::

        p = (1 + #{v_sim <= v_obs}) / (M + 1)   if obs tends above simulations
        p = (1 + #{u_sim <= u_obs}) / (M + 1)   if obs tends below simulations

    This gives the minimum achievable p-value of ``1/(M+1)`` when the observed
    curve is the most extreme in its direction at every grid point.

    The envelope bounds are the ``k``-th smallest and largest order statistics
    where ``k = min(u_obs, v_obs)`` gives the tightest band consistent with the
    observed depth.

    Parameters
    ----------
    obs_curve
        Observed ECDF curve of shape (L,)
    sim_curves
        Simulation ECDF curves of shape (M, L)
    alpha
        Significance level; used only for display (lo/hi) — the p-value is exact
        regardless of alpha

    Returns
    -------
    :
        Tuple of (lo, hi, p_val, sign, rank_stat_obs) where:

        - ``lo``, ``hi``: simultaneous envelope bounds of shape (L,)
        - ``p_val``: exact Monte Carlo p-value
        - ``sign``: +1 if observed curve tends above mean, -1 if below
        - ``rank_stat_obs``: the relevant tail statistic (u_obs or v_obs)
    """
    num_sims = sim_curves.shape[0]
    num_total = num_sims + 1  # total pool size

    # Pool observed curve with simulations: shape (s, L)
    all_curves = np.vstack([obs_curve[np.newaxis, :], sim_curves])

    # 1-indexed ranks at each grid point across the pool of s curves
    ranks = np.argsort(np.argsort(all_curves, axis=0), axis=0) + 1  # shape (s, L)

    # Tail statistics per curve (shape s,):
    #   u_i = min_j rank_{ij}          small → extreme from below
    #   v_i = min_j (s+1 - rank_{ij}) small → extreme from above
    u = np.min(ranks, axis=1)
    v = np.min(num_total + 1 - ranks, axis=1)

    u_obs, v_obs = int(u[0]), int(v[0])
    u_sims, v_sims = u[1:], v[1:]

    # Direction of obs relative to simulations
    sign = int(np.sign(np.mean(obs_curve - sim_curves.mean(axis=0))))
    if sign == 0:
        sign = 1

    # Exact p-value from the relevant tail
    if sign > 0:
        # obs tends above sims → test upper tail via v
        r_obs = v_obs
        p_val = float((np.sum(v_sims <= v_obs) + 1) / (num_sims + 1))
    else:
        # obs tends below sims → test lower tail via u
        r_obs = u_obs
        p_val = float((np.sum(u_sims <= u_obs) + 1) / (num_sims + 1))

    # Envelope bounds at depth k = min(u_obs, v_obs)
    k = max(min(u_obs, v_obs), 1)
    sorted_all = np.sort(all_curves, axis=0)  # shape (s, L)
    lo = sorted_all[k - 1]  # k-th smallest at each grid point
    hi = sorted_all[num_total - k]  # k-th largest at each grid point

    return lo, hi, p_val, sign, r_obs


def _informative_columns(sim_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mu, sd, keep)`` for a set of simulated curves.

    ``keep`` marks grid points where the simulations vary.  Where ``sd`` is zero the
    standardized deviation is undefined, and flooring ``sd`` instead would swamp the
    statistic.  Callers must apply ``keep`` to the observed *and* simulated curves alike.
    """
    mu = sim_curves.mean(axis=0)
    sd = sim_curves.std(axis=0, ddof=1)
    keep = sd > 0
    return mu, sd, keep


def get_test_statistic_and_pvalue(
    obs_curve: np.ndarray,
    sim_curves: np.ndarray,
    statistic: Literal["supremum", "intdev", "rank"] = "supremum",
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray, int]:
    """
    Calculate global test statistic and p-value.

    Computes test statistic:

    - ``"supremum"``: T = max_j |(obs - mu_j)/sd_j|
    - ``"intdev"``: T = integrate |(obs - mu)/sd|
    - ``"rank"``: T = extreme rank statistic (see :func:`compute_rank_envelope`)

    Monte Carlo p-value is ``(1 + #{T_sim >= T_obs}) / (M + 1)`` for supremum/intdev,
    or the exact rank envelope p-value for ``"rank"``.

    Parameters
    ----------
    obs_curve
        Observed ECDF curve
    sim_curves
        Array of shape (M, L) containing M simulation ECDF curves
    statistic
        Test statistic to compute: ``"supremum"``, ``"intdev"``, or ``"rank"``
    alpha
        Significance level; only used when ``statistic="rank"``

    Returns
    -------
    :
        Tuple containing (p_value, T_observed, T_simulated_array, supremum_sign)
        where supremum_sign is +1 for positive and -1 for negative supremum direction
    """
    mu, sd, keep = _informative_columns(sim_curves)
    if statistic in ("supremum", "intdev"):
        # Restrict to informative grid points; see _informative_columns.
        obs_k = obs_curve[keep]
        sim_k = sim_curves[:, keep]
        mu_k, sd_k = mu[keep], sd[keep]
        if obs_k.size == 0:
            # Nothing varies anywhere, so there is no evidence against the null.
            return 1.0, 0.0, np.zeros(sim_curves.shape[0]), 1

    if statistic == "supremum":
        standardized_dev = (obs_k - mu_k) / sd_k
        t_obs = np.max(np.abs(standardized_dev))
        # Determine the sign of the supremum deviation
        supremum_idx = np.argmax(np.abs(standardized_dev))
        supremum_sign = int(np.sign(standardized_dev[supremum_idx]))
        if supremum_sign == 0:
            supremum_sign = 1  # Default to positive if exactly zero
        t_sim = np.max(np.abs((sim_k - mu_k) / sd_k), axis=1)
    elif statistic == "intdev":
        t_obs = integrate.trapezoid(np.abs((obs_k - mu_k) / sd_k))
        t_sim = np.array(
            [integrate.trapezoid(np.abs((sim_k[i] - mu_k) / sd_k)) for i in range(sim_k.shape[0])]
        )
        # For intdev, compute sign based on the dominant direction
        standardized_dev_integral = np.mean((obs_k - mu_k) / sd_k)
        supremum_sign = int(np.sign(standardized_dev_integral))
        if supremum_sign == 0:
            supremum_sign = 1  # Default to positive if exactly zero
    elif statistic == "rank":
        _, _, p_val, supremum_sign, r_obs = compute_rank_envelope(obs_curve, sim_curves, alpha)
        num_sims = sim_curves.shape[0]
        num_total = num_sims + 1
        # Reconstruct the relevant directional tail statistic for each sim curve
        all_curves = np.vstack([obs_curve[np.newaxis, :], sim_curves])
        ranks = np.argsort(np.argsort(all_curves, axis=0), axis=0) + 1
        if supremum_sign > 0:
            # upper tail: v_i = min_j (s+1 - rank_{ij})
            t_sim = np.min(num_total + 1 - ranks, axis=1)[1:].astype(float)
        else:
            # lower tail: u_i = min_j rank_{ij}
            t_sim = np.min(ranks, axis=1)[1:].astype(float)
        return float(p_val), float(r_obs), t_sim, supremum_sign
    else:
        raise ValueError(f"Invalid statistic: {statistic}")
    p = (np.sum(t_sim >= t_obs) + 1) / (t_sim.size + 1)
    return float(p), float(t_obs), t_sim, supremum_sign


def monte_carlo_per_cell(
    observed_distances: dict[str, np.ndarray],
    simulated_distances_by_mode: dict[str, list[dict[str, np.ndarray]]],
    alpha: float = 0.05,
    distance_measures: list[str] | None = None,
    r_grid_size: int = 150,
    bin_width: float | None = None,
    statistic: Literal["supremum", "intdev"] = "supremum",
    envelope_type: EnvelopeType = "pointwise",
    joint_r_grid_size: int | None = None,
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
    bin_width
        Optional bin width for r-grid construction; if None, uses r_grid_size to determine grid
        Overrides r_grid_size if provided, ensuring grid points are spaced by bin_width up to the
        extended max
    statistic
        Test statistic to use; ignored when ``envelope_type="rank"``.
        ``"supremum"``: use maximum standardized deviation across r-grid points
        ``"intdev"``: use integrated standardized deviation across r-grid points
    envelope_type
        ``"pointwise"``: pointwise quantile envelope (original behaviour).
        ``"rank"``: simultaneous rank envelope test (Myllymäki et al. 2017), which
        provides exact simultaneous coverage and a valid global p-value.
    joint_r_grid_size
        Number of points per distance measure when resampling each ECDF onto a
        uniform [0, 1] quantile grid before concatenating for the joint test.
        Defaults to ``r_grid_size``, giving each measure equal representation
        regardless of physical range (e.g. pairwise distances span 0-25 µm
        while scaled_nucleus spans 0-1).

    Returns
    -------
    :
        Nested dictionary with structure: packing_mode -> {
            'per_distance_measure': {distance_measure ->
                {'r', 'obs_curve', 'lo', 'hi', 'mu', 'sd', 'pval', 'T_obs', 'supremum_sign'}},
            'joint': {'pval', 'T_obs', 'distance_measures', 'supremum_sign'}
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
            r_grids[distance_measure] = make_r_grid_from_pooled(
                pooled, n=r_grid_size, bin_width=bin_width
            )

        # Second pass: compute curves, envelopes, p-values
        sim_curves_by_distance_measure = {}
        obs_curves_by_distance_measure = {}
        for distance_measure in distance_measures:
            xvals = r_grids[distance_measure]
            # observed curve
            obs_curve = ecdf(observed_distances.get(distance_measure, np.array([])), xvals)
            # simulation curves
            sim_mat = np.vstack(
                [ecdf(rep.get(distance_measure, np.array([])), xvals) for rep in reps]
            )  # (R, L)

            # tobs is int from the rank envelope but float from the other statistics
            tobs: float
            if envelope_type == "rank":
                lo, hi, pval, supremum_sign, tobs = compute_rank_envelope(
                    obs_curve, sim_mat, alpha=alpha
                )
                mu = sim_mat.mean(axis=0)
                sd = sim_mat.std(axis=0, ddof=1)
                sd[sd == 0] = 1e-9
            else:
                lo, hi, mu, sd = pointwise_envelope(sim_mat, alpha=alpha)
                pval, tobs, _, supremum_sign = get_test_statistic_and_pvalue(
                    obs_curve, sim_mat, statistic=statistic
                )

            per_distance_measure[distance_measure] = {
                "xvals": xvals,
                "obs_curve": obs_curve,
                "lo": lo,
                "hi": hi,
                "mu": mu,
                "sd": sd,
                "pval": pval,
                "T_obs": tobs,
                "supremum_sign": supremum_sign,
            }
            sim_curves_by_distance_measure[distance_measure] = sim_mat
            obs_curves_by_distance_measure[distance_measure] = obs_curve

        # Joint test: resample each measure's curves to a uniform joint_r_grid_size-point
        # grid on [0, 1] so every distance measure contributes equally regardless of range.
        _joint_r_grid_size = r_grid_size if joint_r_grid_size is None else joint_r_grid_size
        joint_u = np.linspace(0.0, 1.0, _joint_r_grid_size)

        def _resample(curve: np.ndarray, xvals: np.ndarray, joint_u: np.ndarray) -> np.ndarray:
            xvals_norm = xvals - xvals.min()
            xvals_max = xvals_norm.max()
            if xvals_max == 0:
                return np.interp(joint_u, np.array([0.0, 1.0]), np.array([curve[0], curve[-1]]))
            return np.interp(joint_u, xvals_norm / xvals_max, curve)

        sim_concat = np.vstack(
            [
                np.concatenate(
                    [
                        _resample(sim_curves_by_distance_measure[dm][i], r_grids[dm], joint_u)
                        for dm in distance_measures
                    ],
                    axis=0,
                )
                for i in range(num_replicates)
            ]
        )  # (num_replicates, joint_r_grid_size * len(distance_measures))
        obs_concat = np.concatenate(
            [
                _resample(obs_curves_by_distance_measure[dm], r_grids[dm], joint_u)
                for dm in distance_measures
            ],
            axis=0,
        )

        tobs_joint: float
        if envelope_type == "rank":
            _, _, p_joint, supremum_sign_joint, tobs_joint = compute_rank_envelope(
                obs_concat, sim_concat, alpha=alpha
            )
        else:
            p_joint, tobs_joint, _, supremum_sign_joint = get_test_statistic_and_pvalue(
                obs_concat, sim_concat, statistic=statistic
            )

        results[packing_mode] = {
            "per_distance_measure": per_distance_measure,
            "joint": {
                "pval": p_joint,
                "T_obs": tobs_joint,
                "distance_measures": distance_measures,
                "supremum_sign": supremum_sign_joint,
            },
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
    # Collect p-values and supremum signs
    per_distance_measure_p = pd.DataFrame(
        index=np.arange(num_cells),
        columns=pd.MultiIndex.from_product(
            [packing_modes, distance_measures], names=["packing_mode", "distance_measure"]
        ),
        dtype=float,
    )
    per_distance_measure_sign = pd.DataFrame(
        index=np.arange(num_cells),
        columns=pd.MultiIndex.from_product(
            [packing_modes, distance_measures], names=["packing_mode", "distance_measure"]
        ),
        dtype=int,
    )
    joint_p = pd.DataFrame(index=np.arange(num_cells), columns=packing_modes, dtype=float)
    joint_sign = pd.DataFrame(index=np.arange(num_cells), columns=packing_modes, dtype=int)

    for cell_idx in range(num_cells):
        for packing_mode in packing_modes:
            for distance_measure in distance_measures:
                per_distance_measure_p.loc[cell_idx, (packing_mode, distance_measure)] = (
                    all_results[cell_idx][packing_mode]["per_distance_measure"][distance_measure][
                        "pval"
                    ]
                )
                per_distance_measure_sign.loc[cell_idx, (packing_mode, distance_measure)] = (
                    all_results[cell_idx][packing_mode]["per_distance_measure"][distance_measure][
                        "supremum_sign"
                    ]
                )
            joint_p.loc[cell_idx, packing_mode] = all_results[cell_idx][packing_mode]["joint"][
                "pval"
            ]
            joint_sign.loc[cell_idx, packing_mode] = all_results[cell_idx][packing_mode]["joint"][
                "supremum_sign"
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

    # Rejection rates (q < alpha) overall and by sign
    rej_distance_measure = (per_distance_measure_q < alpha).mean(
        axis=0
    )  # fraction of cells rejected
    rej_joint = (joint_q < alpha).mean(axis=0)

    # Rejection rates by supremum sign
    rejected_distance_measure = per_distance_measure_q < alpha
    rej_distance_measure_positive = (
        (rejected_distance_measure) & (per_distance_measure_sign == 1)
    ).mean(axis=0)
    rej_distance_measure_negative = (
        (rejected_distance_measure) & (per_distance_measure_sign == -1)
    ).mean(axis=0)

    rejected_joint = joint_q < alpha
    rej_joint_positive = (rejected_joint & (joint_sign == 1)).mean(axis=0)
    rej_joint_negative = (rejected_joint & (joint_sign == -1)).mean(axis=0)

    return {
        "per_distance_measure_pvals": per_distance_measure_p,
        "per_distance_measure_qvals": per_distance_measure_q,
        "per_distance_measure_signs": per_distance_measure_sign,
        "joint_pvals": joint_p,
        "joint_qvals": joint_q,
        "joint_signs": joint_sign,
        "rejection_per_distance_measure": rej_distance_measure,
        "rejection_joint": rej_joint,
        "rejection_per_distance_measure_positive": rej_distance_measure_positive,
        "rejection_per_distance_measure_negative": rej_distance_measure_negative,
        "rejection_joint_positive": rej_joint_positive,
        "rejection_joint_negative": rej_joint_negative,
    }


def per_cell_envelope_test(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    baseline_mode: str,
    packing_modes: list[str],
    distance_measures: list[str],
    alpha: float = 0.05,
    r_grid_size: int = 150,
    bin_width: float | None = None,
    statistic: Literal["supremum", "intdev"] = "intdev",
    envelope_type: EnvelopeType = "rank",
    joint_r_grid_size: int | None = None,
    min_replicates: int = 2,
    calibration_holdout_index: int | None = None,
) -> dict[str, Any]:
    """
    Per-cell Monte Carlo envelope (CSR) test: baseline vs each null model.

    For every cell present in both ``baseline_mode`` and every null model, the
    baseline (experimental) distance distribution is compared against the Monte
    Carlo envelope built from that *same cell's* simulated replicates, separately
    for each null model and distance measure.  Per-cell p-values are aggregated
    across cells with Benjamini-Hochberg correction.

    This is the per-cell counterpart to :func:`pairwise_envelope_test`: instead of
    pooling every cell and seed into one cross-cell envelope, each cell is paired
    with its own replicates, which is what a replicate-per-cell packing design
    (e.g. 99 seeds/cell) is built to support.  With ``envelope_type="rank"`` the
    exact simultaneous rank-envelope p-value has resolution ``1/(R + 1)`` (e.g.
    0.01 for 99 replicates).

    Parameters
    ----------
    all_distance_dict
        ``{distance_measure: {mode: {cell_id: {seed: distances}}}}``.
    baseline_mode
        Mode treated as the observed data (typically the experimental structure).
    packing_modes
        All modes; every mode other than ``baseline_mode`` is a null model.
    distance_measures
        Distance measures to test (per-measure and joint).
    alpha
        Significance level for BH correction across cells.
    r_grid_size, bin_width, statistic, envelope_type, joint_r_grid_size
        Forwarded to :func:`monte_carlo_per_cell`.  ``envelope_type`` defaults to
        ``"rank"`` (``statistic`` is ignored when ``envelope_type="rank"``).
    min_replicates
        Minimum simulated replicates a cell must have, in every null model, to be
        included.
    calibration_holdout_index
        When set, run the hold-one-replicate-out coverage test instead: each null
        model's own replicate at this index becomes the pseudo-observed curve and the
        envelope is built from the rest, so a calibrated test rejects at ``alpha``.

    Returns
    -------
    :
        Dictionary containing:

        - ``"summary"``: output of :func:`summarize_across_cells` (BH q-values and
          rejection fractions per null model / distance measure and joint).
        - ``"per_cell_results"``: list of per-cell :func:`monte_carlo_per_cell`
          outputs, aligned with ``"cell_ids"``.
        - ``"cell_ids"``: cell IDs actually tested (row order of the summary
          DataFrames).
        - ``"baseline_mode"``, ``"null_modes"``, ``"distance_measures"``,
          ``"alpha"``, ``"statistic"``, ``"envelope_type"``.
    """
    logger = logging.getLogger(__name__)

    null_modes = [m for m in packing_modes if m != baseline_mode]
    if not null_modes:
        raise ValueError(
            "per_cell_envelope_test needs at least one null model besides "
            f"baseline_mode={baseline_mode!r}"
        )

    if calibration_holdout_index is not None and calibration_holdout_index < 0:
        # A negative index would leave the held-out replicate inside its own
        # envelope -- a silent false pass of the calibration.
        raise ValueError(
            f"calibration_holdout_index must be non-negative, got {calibration_holdout_index}"
        )

    def _cells_with_all_dms(mode: str) -> set[str]:
        cell_sets = [
            set(all_distance_dict.get(dm, {}).get(mode, {}).keys()) for dm in distance_measures
        ]
        return set.intersection(*cell_sets) if cell_sets else set()

    shared_cells = _cells_with_all_dms(baseline_mode)
    for mode in null_modes:
        shared_cells = shared_cells & _cells_with_all_dms(mode)

    logger.info(
        "per_cell_envelope_test: %d cells shared across baseline %r + %d null models %s",
        len(shared_cells),
        baseline_mode,
        len(null_modes),
        null_modes,
    )
    if not shared_cells:
        raise ValueError("No cell_ids shared between baseline and all null models")

    all_results: list[dict[str, dict]] = []
    cell_ids: list[str] = []
    skipped = 0

    for cell_id in sorted(shared_cells):
        # Observed = all baseline points for this cell (pool any seeds).
        observed: dict[str, np.ndarray] = {}
        obs_ok = True
        for dm in distance_measures:
            seed_dict = all_distance_dict[dm][baseline_mode][cell_id]
            if seed_dict:
                obs_arr = np.concatenate([np.asarray(a).ravel() for a in seed_dict.values()])
            else:
                obs_arr = np.array([])
            observed[dm] = obs_arr
            if obs_arr.size == 0 or not np.any(np.isfinite(obs_arr)):
                obs_ok = False
        if not obs_ok:
            skipped += 1
            continue

        # Per null model, build a replicate list keyed by the seeds available for
        # every requested distance measure.
        sim_by_mode: dict[str, list[dict[str, np.ndarray]]] = {}
        modes_ok = True
        for mode in null_modes:
            seed_sets = [
                set(all_distance_dict[dm][mode][cell_id].keys()) for dm in distance_measures
            ]
            seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
            reps = [
                {
                    dm: np.asarray(all_distance_dict[dm][mode][cell_id][seed])
                    for dm in distance_measures
                }
                for seed in seeds
            ]
            if len(reps) < min_replicates:
                modes_ok = False
                break
            sim_by_mode[mode] = reps
        if not modes_ok:
            skipped += 1
            continue

        # monte_carlo_per_cell's joint test assumes equal replicate counts across
        # modes; truncate every mode to the common minimum for this cell.
        min_reps = min(len(reps) for reps in sim_by_mode.values())
        if min_reps < min_replicates:
            skipped += 1
            continue
        sim_by_mode = {mode: reps[:min_reps] for mode, reps in sim_by_mode.items()}

        if calibration_holdout_index is None:
            cell_result = monte_carlo_per_cell(
                observed_distances=observed,
                simulated_distances_by_mode=sim_by_mode,
                alpha=alpha,
                distance_measures=distance_measures,
                r_grid_size=r_grid_size,
                bin_width=bin_width,
                statistic=statistic,
                envelope_type=envelope_type,
                joint_r_grid_size=joint_r_grid_size,
            )
        else:
            # Each null model supplies its own pseudo-observed replicate, so run one
            # comparison per mode and merge (output is keyed by mode).
            if min_reps - 1 < min_replicates or calibration_holdout_index >= min_reps:
                skipped += 1
                continue
            cell_result = {}
            for mode, reps in sim_by_mode.items():
                held_out = reps[calibration_holdout_index]
                rest = [r for i, r in enumerate(reps) if i != calibration_holdout_index]
                cell_result.update(
                    monte_carlo_per_cell(
                        observed_distances=held_out,
                        simulated_distances_by_mode={mode: rest},
                        alpha=alpha,
                        distance_measures=distance_measures,
                        r_grid_size=r_grid_size,
                        bin_width=bin_width,
                        statistic=statistic,
                        envelope_type=envelope_type,
                        joint_r_grid_size=joint_r_grid_size,
                    )
                )
        all_results.append(cell_result)
        cell_ids.append(cell_id)

    if not all_results:
        if calibration_holdout_index is not None:
            raise ValueError(
                "No cells had sufficient data for the per-cell envelope test with "
                f"calibration_holdout_index={calibration_holdout_index}; every cell "
                "has fewer replicates than that index requires."
            )
        raise ValueError("No cells had sufficient data for the per-cell envelope test")

    logger.info(
        "per_cell_envelope_test: tested %d cells (%d skipped for insufficient data)",
        len(all_results),
        skipped,
    )

    summary = summarize_across_cells(all_results, alpha=alpha)

    return {
        "summary": summary,
        "per_cell_results": all_results,
        "cell_ids": cell_ids,
        "baseline_mode": baseline_mode,
        "null_modes": null_modes,
        "distance_measures": distance_measures,
        "alpha": alpha,
        "statistic": statistic,
        "envelope_type": envelope_type,
        "calibration_holdout_index": calibration_holdout_index,
    }


def summarize_per_cell_calibration(
    per_cell_results: dict[str, Any], n_envelope_replicates: int, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Summarize a calibration run of ``per_cell_envelope_test`` against its expected values.

    In a calibration run every null replicate stands in for the observed one, so with
    ``R`` replicates in the envelope each p-value is uniform on multiples of ``1/(R+1)``.
    A calibrated test then rejects at the raw level ``expected_raw`` and, after BH across
    cells, essentially never.

    Parameters
    ----------
    per_cell_results
        Return value of ``per_cell_envelope_test`` run with ``calibration_holdout_index``.
    n_envelope_replicates
        Replicates left in each cell's envelope after the held-out one is removed (``R``).
    alpha
        Significance level for the raw and BH rejection rates.

    Returns
    -------
    :
        One row per (null mode, scope), scope being each distance measure then ``"joint"``.
        Columns: ``null_mode``, ``scope``, ``raw_rejection`` (fraction of cells with
        p < alpha), ``bh_rejection`` (fraction with q < alpha), ``at_floor`` (fraction at
        p = 1/(R+1)), ``min_p`` and ``expected_raw`` (the calibrated raw rate).
    """
    summary = per_cell_results["summary"]
    p_floor = 1.0 / (n_envelope_replicates + 1)
    # p < alpha on multiples of 1/(R+1); the epsilon guards float error in alpha * (R + 1)
    n_below = int(np.ceil(alpha * (n_envelope_replicates + 1) - 1e-9)) - 1
    expected_raw = n_below / (n_envelope_replicates + 1)

    rows = []
    for mode in per_cell_results["null_modes"]:
        for scope in [*per_cell_results["distance_measures"], "joint"]:
            if scope == "joint":
                p = summary["joint_pvals"][mode].to_numpy(dtype=float)
                q = summary["joint_qvals"][mode].to_numpy(dtype=float)
            else:
                p = summary["per_distance_measure_pvals"][(mode, scope)].to_numpy(dtype=float)
                q = summary["per_distance_measure_qvals"][(mode, scope)].to_numpy(dtype=float)
            rows.append(
                {
                    "null_mode": mode,
                    "scope": scope,
                    "raw_rejection": np.mean(p < alpha),
                    "bh_rejection": np.mean(q < alpha),
                    "at_floor": np.mean(p <= p_floor + 1e-12),
                    "min_p": p.min(),
                    "expected_raw": expected_raw,
                }
            )
    return pd.DataFrame(rows)


def _batch_test_obs_vs_sim(
    obs_curves: np.ndarray,
    sim_curves: np.ndarray,
    statistic: Literal["supremum", "intdev"] = "intdev",
    envelope_type: EnvelopeType = "pointwise",
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Test many observed curves against one fixed simulation envelope.

    Returns ``(pvals, signs)`` for each row of ``obs_curves``.  For the pointwise
    (``supremum``/``intdev``) path the simulation-side statistic (``mu``, ``sd``
    and the sorted ``t_sim`` distribution) is computed **once** and reused across
    all observed curves, turning the pooled pairwise test from
    ``O(M_obs * M_sim)`` into ``O(M_sim log M_sim + M_obs log M_sim)``.  The
    ``rank`` path pools each observed curve with the simulations, so it is still
    evaluated per observed curve via :func:`compute_rank_envelope`.

    Results match :func:`get_test_statistic_and_pvalue` applied per row.
    """
    n_obs = obs_curves.shape[0]
    if n_obs == 0:
        return np.empty(0), np.empty(0, dtype=int)

    if envelope_type == "rank":
        pvals = np.empty(n_obs)
        signs = np.empty(n_obs, dtype=int)
        for i in range(n_obs):
            _, _, p, s, _ = compute_rank_envelope(obs_curves[i], sim_curves, alpha=alpha)
            pvals[i] = p
            signs[i] = s
        return pvals, signs

    mu, sd, keep = _informative_columns(sim_curves)
    if not np.any(keep):
        # Nothing varies anywhere: no evidence against the null for any observation.
        return np.ones(n_obs), np.ones(n_obs, dtype=int)
    # Restrict to informative grid points; see _informative_columns.
    obs_k = obs_curves[:, keep]
    sim_k = sim_curves[:, keep]
    mu_k, sd_k = mu[keep], sd[keep]
    dev = (obs_k - mu_k) / sd_k  # (n_obs, L_keep)

    if statistic == "supremum":
        abs_dev = np.abs(dev)
        t_obs = np.max(abs_dev, axis=1)
        signs = np.sign(dev[np.arange(n_obs), np.argmax(abs_dev, axis=1)]).astype(int)
        t_sim = np.max(np.abs((sim_k - mu_k) / sd_k), axis=1)
    elif statistic == "intdev":
        t_obs = integrate.trapezoid(np.abs(dev), axis=1)
        signs = np.sign(np.mean(dev, axis=1)).astype(int)
        t_sim = integrate.trapezoid(np.abs((sim_k - mu_k) / sd_k), axis=1)
    else:
        raise ValueError(f"Invalid statistic: {statistic}")

    signs[signs == 0] = 1
    sorted_t_sim = np.sort(t_sim)
    m = sorted_t_sim.size
    # #{t_sim >= t_obs} via searchsorted (left => count strictly less than t_obs).
    n_ge = m - np.searchsorted(sorted_t_sim, t_obs, side="left")
    pvals = (n_ge + 1) / (m + 1)
    return pvals, signs


def _pairwise_test_on_curves(
    mode_curves: dict[str, dict[str, np.ndarray]],
    mode_xvals: dict[str, dict[str, np.ndarray]],
    packing_modes: list[str],
    distance_measures: list[str],
    alpha: float = 0.05,
    statistic: Literal["supremum", "intdev"] = "intdev",
    envelope_type: EnvelopeType = "pointwise",
    joint_r_grid_size: int | None = None,
) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    """
    Run pairwise Monte Carlo envelope tests on pre-computed per-cell curve arrays.

    Shared inner loop used by both :func:`pairwise_envelope_test` (after ECDF
    construction) and
    :func:`~cellpack_analysis.lib.occupancy.pairwise_envelope_test_occupancy`
    (with occupancy ratio curves).

    Parameters
    ----------
    mode_curves
        ``{mode: {dm: ndarray of shape (n_cells, n_bins)}}`` — one curve per
        cell (e.g. ECDF or occupancy ratio curve), pre-computed.
    mode_xvals
        ``{mode: {dm: ndarray of shape (n_bins,)}}`` — x-axis values for each
        curve; used only for equal-weight resampling in the joint test.
    packing_modes
        Ordered list of modes forming the pairwise comparison matrix.
    distance_measures
        Ordered list of distance measures.
    alpha
        Significance level for BH correction.
    statistic
        Test statistic; ignored when ``envelope_type="rank"``.
    envelope_type
        ``"pointwise"`` or ``"rank"``.
    joint_r_grid_size
        Points per distance measure after resampling onto a uniform [0, 1]
        grid for the joint test.  Defaults to the bin count of the first
        available curve.

    Returns
    -------
    :
        ``(per_dm_results, joint_results)`` where each is a dict keyed by
        ``(mode_a, mode_b)`` pairs containing ``"pvals"``, ``"qvals"``,
        ``"signs"``, ``"rejection_fraction"``,
        ``"rejection_fraction_positive"``, and
        ``"rejection_fraction_negative"``.
    """
    logger = logging.getLogger(__name__)

    if joint_r_grid_size is None:
        joint_r_grid_size = next(
            (
                mode_curves[m][d].shape[1]
                for m in packing_modes
                for d in distance_measures
                if mode_curves[m][d].shape[0] > 0
            ),
            50,
        )
        if not isinstance(joint_r_grid_size, int) or joint_r_grid_size <= 0:
            raise ValueError(f"Invalid joint_r_grid_size: {joint_r_grid_size}")
    joint_u = np.linspace(0.0, 1.0, joint_r_grid_size)

    def _resample(curve: np.ndarray, xvals: np.ndarray) -> np.ndarray:
        """Interpolate ``curve`` to a uniform [0, 1] grid of ``joint_r_grid_size`` points."""
        x_norm = xvals - xvals.min()
        x_max = x_norm.max()
        if x_max == 0:
            return np.interp(joint_u, np.array([0.0, 1.0]), np.array([curve[0], curve[-1]]))
        return np.interp(joint_u, x_norm / x_max, curve)

    per_dm_results: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        dm: {} for dm in distance_measures
    }
    joint_results: dict[tuple[str, str], dict[str, Any]] = {}

    for mode_a in packing_modes:
        for mode_b in packing_modes:
            if mode_a == mode_b:
                continue

            logger.info("Testing %s against %s envelope", mode_a, mode_b)

            # Per-dm tests
            for dm in distance_measures:
                sim_curves = mode_curves[mode_b][dm]
                obs_curves = mode_curves[mode_a][dm]
                if sim_curves.shape[0] < 2 or obs_curves.shape[0] == 0:
                    continue

                pvals_arr, signs_arr = _batch_test_obs_vs_sim(
                    obs_curves, sim_curves, statistic, envelope_type, alpha
                )
                qvals_arr = false_discovery_control(pvals_arr, method="bh")
                rejected = qvals_arr < alpha
                per_dm_results[dm][(mode_a, mode_b)] = {
                    "pvals": pvals_arr,
                    "qvals": qvals_arr,
                    "signs": signs_arr,
                    "rejection_fraction": float(rejected.mean()),
                    "rejection_fraction_positive": float((rejected & (signs_arr == 1)).mean()),
                    "rejection_fraction_negative": float((rejected & (signs_arr == -1)).mean()),
                }

            # Joint test: resample each dm to equal-length grid for equal weight in hstack
            valid_dms = [
                dm
                for dm in distance_measures
                if mode_curves[mode_b][dm].shape[0] >= 2 and mode_curves[mode_a][dm].shape[0] >= 1
            ]
            if not valid_dms:
                logger.warning(
                    "Skipping joint test for %s vs %s: no valid distance measures",
                    mode_a,
                    mode_b,
                )
                continue

            num_ref = min(mode_curves[mode_b][dm].shape[0] for dm in valid_dms)
            num_test = min(mode_curves[mode_a][dm].shape[0] for dm in valid_dms)
            sim_concat = np.hstack(
                [
                    np.vstack(
                        [
                            _resample(mode_curves[mode_b][dm][i], mode_xvals[mode_b][dm])
                            for i in range(num_ref)
                        ]
                    )
                    for dm in valid_dms
                ]
            )

            obs_concat_all = np.vstack(
                [
                    np.concatenate(
                        [
                            _resample(mode_curves[mode_a][dm][obs_idx], mode_xvals[mode_a][dm])
                            for dm in valid_dms
                        ]
                    )
                    for obs_idx in range(num_test)
                ]
            )
            jp, js = _batch_test_obs_vs_sim(
                obs_concat_all, sim_concat, statistic, envelope_type, alpha
            )
            jq = false_discovery_control(jp, method="bh")
            rej = jq < alpha
            joint_results[(mode_a, mode_b)] = {
                "pvals": jp,
                "qvals": jq,
                "signs": js,
                "rejection_fraction": float(rej.mean()),
                "rejection_fraction_positive": float((rej & (js == 1)).mean()),
                "rejection_fraction_negative": float((rej & (js == -1)).mean()),
            }

    return per_dm_results, joint_results


def pairwise_envelope_test(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    packing_modes: list[str],
    distance_measures: list[str],
    alpha: float = 0.05,
    r_grid_size: int = 150,
    bin_width: float | None = None,
    statistic: Literal["supremum", "intdev"] = "intdev",
    envelope_type: EnvelopeType = "pointwise",
    joint_r_grid_size: int | None = None,
) -> dict[str, Any]:
    """
    Pairwise Monte Carlo envelope test between all ordered pairs of packing modes.

    For each ordered pair (mode_a, mode_b) and each distance measure, builds an
    ECDF-based Monte Carlo envelope from mode_b's cells, tests each mode_a cell
    against it, and reports BH-corrected rejection fractions.

    Comparisons are asymmetric: testing mode_a against mode_b's envelope differs
    from the reverse because the envelope captures mode_b's specific variability.
    A mode with high internal variability produces a wide envelope that is harder
    to reject, while a tight envelope from a homogeneous mode is easier to reject.

    Parameters
    ----------
    all_distance_dict
        Distance data with structure
        {distance_measure: {mode: {cell_id: {seed: distances_array}}}}
    packing_modes
        List of packing modes to compare pairwise
    distance_measures
        List of distance measures to analyze
    alpha
        Significance level for BH correction and envelope construction
    r_grid_size
        Number of points in the ECDF evaluation grid
    bin_width
        Optional bin width for r-grid construction; if None, uses r_grid_size to determine grid
        Overrides r_grid_size if provided, ensuring grid points are spaced by bin_width up to the
        extended max
    statistic
        Test statistic; ignored when ``envelope_type="rank"``.
        ``"supremum"``: max standardized deviation, ``"intdev"``: integrated standardized deviation
    envelope_type
        ``"pointwise"``: pointwise quantile envelope (original behaviour).
        ``"rank"``: simultaneous rank envelope test (Myllymäki et al. 2017).
        The diagonal display envelopes (``envelopes`` key in result) always use
        the pointwise quantile method for visualization purposes.
    joint_r_grid_size
        Number of points per distance measure when resampling each ECDF onto a
        uniform [0, 1] quantile grid before concatenating for the joint test.
        Defaults to ``r_grid_size``, giving each measure equal representation
        regardless of physical range.

    Returns
    -------
    :
        Dictionary containing:
        - 'per_distance_measure': {dm: {(mode_a, mode_b): {pvals, qvals, signs,
          rejection_fraction, rejection_fraction_positive, rejection_fraction_negative}}}
        - 'joint': {(mode_a, mode_b): {same keys as above}}
        - 'envelopes': {mode: {dm: {xvals, lo, hi, mu, sd}}}
        - 'packing_modes', 'distance_measures', 'alpha', 'statistic', 'envelope_type': input params
    """
    logger = logging.getLogger(__name__)

    # Step 1: Flatten distance arrays per (mode, distance_measure)
    flat_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    for mode in packing_modes:
        for dm in distance_measures:
            arrays: list[np.ndarray] = []
            mode_dict = all_distance_dict.get(dm, {}).get(mode, {})
            for _cell_id, seed_dict in mode_dict.items():
                for _seed, distances in seed_dict.items():
                    arr = np.asarray(distances)
                    if arr.size > 0 and np.any(np.isfinite(arr)):
                        arrays.append(arr)
            flat_arrays[(mode, dm)] = arrays
    logger.info(
        "Flattened arrays: %s",
        {k: len(v) for k, v in flat_arrays.items()},
    )

    # Step 2: Build pointwise envelopes for each mode (used for diagonal display)
    envelopes: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for mode in packing_modes:
        envelopes[mode] = {}
        for dm in distance_measures:
            arrays = flat_arrays[(mode, dm)]
            if len(arrays) < 2:
                logger.warning(
                    "Mode %s, dm %s has < 2 arrays (%d), skipping envelope",
                    mode,
                    dm,
                    len(arrays),
                )
                continue
            r_grid = make_r_grid_from_pooled(arrays, n=r_grid_size, bin_width=bin_width)
            curves = np.vstack([ecdf(arr, r_grid) for arr in arrays])
            lo, hi, mu, sd = pointwise_envelope(curves, alpha=alpha)
            envelopes[mode][dm] = {
                "xvals": r_grid,
                "lo": lo,
                "hi": hi,
                "mu": mu,
                "sd": sd,
            }

    # Step 3: Build ECDF curves on a shared per-dm r-grid, then run pairwise tests.
    # The r-grid is pooled from all modes so the representation is consistent across
    # the full pairwise matrix (rather than recomputing a different grid per pair).
    r_grids_shared: dict[str, np.ndarray] = {}
    for dm in distance_measures:
        all_dm_arrays = [arr for mode in packing_modes for arr in flat_arrays[(mode, dm)]]
        r_grids_shared[dm] = make_r_grid_from_pooled(
            all_dm_arrays, n=r_grid_size, bin_width=bin_width
        )

    mode_ecdf_curves: dict[str, dict[str, np.ndarray]] = {}
    mode_ecdf_xvals: dict[str, dict[str, np.ndarray]] = {}
    for mode in packing_modes:
        mode_ecdf_curves[mode] = {}
        mode_ecdf_xvals[mode] = {}
        for dm in distance_measures:
            r_grid = r_grids_shared[dm]
            arrays = flat_arrays[(mode, dm)]
            if arrays:
                mode_ecdf_curves[mode][dm] = np.vstack([ecdf(arr, r_grid) for arr in arrays])
            else:
                mode_ecdf_curves[mode][dm] = np.empty((0, len(r_grid)))
            mode_ecdf_xvals[mode][dm] = r_grid

    # Default joint_r_grid_size to r_grid_size to preserve original behaviour
    effective_joint_r_grid_size = (
        joint_r_grid_size if joint_r_grid_size is not None else r_grid_size
    )

    per_dm_results, joint_results = _pairwise_test_on_curves(
        mode_curves=mode_ecdf_curves,
        mode_xvals=mode_ecdf_xvals,
        packing_modes=packing_modes,
        distance_measures=distance_measures,
        alpha=alpha,
        statistic=statistic,
        envelope_type=envelope_type,
        joint_r_grid_size=effective_joint_r_grid_size,
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
    }
