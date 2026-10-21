import json
import logging
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance
from tqdm import tqdm

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
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


def _nanmean_nanstd(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Column-wise ``nanmean``/``nanstd`` that silences benign all-NaN-slice warnings.

    Grid points outside every cell's available-space support form all-NaN
    columns (an expected consequence of restricting each cell to its own
    support); numpy would otherwise emit ``Mean of empty slice`` and
    ``Degrees of freedom <= 0`` ``RuntimeWarning``s. The result is still NaN
    for those columns, which is the intended "no coverage" value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    return mean, std


def _nan_pointwise_envelope(
    curves: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pointwise ``[alpha/2, 1-alpha/2]`` envelope over the non-NaN entries per column.

    Each cell contributes only where it has available-space support (its curve is
    NaN elsewhere). Substituting NaN with 0 before quantiling would drag the lower
    bound to 0 wherever coverage falls below ``1 - alpha/2`` — the band-collapse
    artifact seen at the tails of the z occupancy. ``nanquantile`` computes the
    band over just the covering cells; fully-uncovered columns stay NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanquantile(curves, alpha / 2, axis=0)
        hi = np.nanquantile(curves, 1 - alpha / 2, axis=0)
    return lo, hi


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


def _occupancy_cache_meta_path(save_path: Path) -> Path:
    """Sidecar path recording the grid parameters a cached occupancy dict was built with.

    A sidecar rather than a key inside the pickle: ``_load_cached_occupancy``
    compares ``set(occupancy_dict.keys())`` against the requested modes, and every
    consumer iterates those keys as modes, so an extra key would break both.

    Scope -- this is a *parameter* guard, not a provenance guard. The stamp holds
    only the grid fields (``distance_measure``, ``num_points``, ``x_min``,
    ``x_max``, ``bandwidth``, ``min_coverage_fraction``) and the distance floors, so three kinds of
    staleness still pass it and must be invalidated by hand via
    ``recalculate["run_occupancy_analysis"]``:

    1. **Cell cohort.** A cache built over a different set of cells reloads, since
       the cohort is not stamped.
    2. **Code version.** A cache built by an earlier revision of this module -- or
       from upstream inputs that have since been regenerated, e.g. repacked
       positions -- reloads, since nothing identifies the producing code or its
       inputs.
    3. **The binned path is unstamped entirely.**
       ``get_binned_occupancy_dict_from_distance_dict`` pickles its result without
       calling ``_write_occupancy_cache_meta``, so only the KDE path
       (``get_kde_occupancy_dict``) is guarded at all.

    Note the upstream KDE cache written by
    ``distance.get_distance_distribution_kde`` is likewise unstamped, and stores
    all modes in one file -- so there is no per-mode invalidation there either.
    """
    return save_path.with_suffix(".meta.json")


def _write_occupancy_cache_meta(save_path: Path, meta: dict[str, Any]) -> None:
    """Record the grid parameters alongside a freshly written occupancy cache.

    Deliberately called *after* the pickle, and deliberately non-fatal. Writing
    the sidecar first would be worse: the cache filename does not encode
    ``num_points``, so a crash between the two writes would leave a stale pickle
    under a sidecar matching the *new* parameters, and the next load would accept
    the wrong grid -- exactly the silent reload the stamp exists to prevent. In
    this order a crash leaves a stampless pickle, which ``_load_cached_occupancy``
    rejects, so it fails safe.

    ``TypeError`` is caught alongside ``OSError`` for the same reason: a
    non-serializable value in *meta* (a numpy scalar, say) must not propagate
    after the pickle has been written. A missing sidecar only costs a recompute;
    an unhandled raise here would abort the caller *after* the pickle landed.
    Serializing before opening the file also means a non-serializable value can
    never leave a half-written sidecar behind (``_load_cached_occupancy`` would
    reject one, but not creating it is better).
    """
    try:
        payload = json.dumps(meta, indent=2, sort_keys=True)
        _occupancy_cache_meta_path(save_path).write_text(payload)
    except (OSError, TypeError, ValueError) as e:  # pragma: no cover - bookkeeping
        logger.warning(f"Could not write occupancy cache metadata for {save_path}: {e}")


def _load_cached_occupancy(
    save_path: Path,
    channel_map: dict[str, str],
    expected_meta: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]] | None:
    """
    Load and validate a cached KDE occupancy dict from disk.

    Returns the cached dict when it is valid, or ``None`` when the cache is
    absent, outdated, or corrupt (logging a warning in the latter cases).

    When ``expected_meta`` is given, the cache is also rejected unless its sidecar
    records the same grid parameters. Without this the cache key is effectively
    only the file name, so changing ``num_points``, ``x_max`` or ``bandwidth``
    silently reloads a grid built with the old settings.
    """
    if not save_path.exists():
        return None
    if expected_meta is not None:
        meta_path = _occupancy_cache_meta_path(save_path)
        if not meta_path.exists():
            logger.warning(
                f"Cached occupancy data at {save_path} has no {meta_path.name} sidecar, so "
                "the grid it was built with is unknown. Recalculating occupancy."
            )
            return None
        try:
            with open(meta_path) as f:
                cached_meta = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {meta_path}: {e}. Recalculating occupancy.")
            return None
        differing = {
            key: (cached_meta.get(key), value)
            for key, value in expected_meta.items()
            if cached_meta.get(key) != value
        }
        if differing:
            logger.warning(
                f"Cached occupancy data at {save_path} was built with different grid "
                f"parameters {differing} (cached, requested). Recalculating occupancy."
            )
            return None
    try:
        with open(save_path, "rb") as f:
            occupancy_dict = pickle.load(f)
        cached_modes = set(occupancy_dict.keys())
        requested_modes = set(channel_map.keys())
        if cached_modes != requested_modes:
            logger.warning(
                f"Cached occupancy data contains modes {cached_modes} but requested modes are "
                f"{requested_modes}. Recalculating occupancy."
            )
            return None
        first_mode = next(iter(cached_modes))
        if "all_occupancy" in occupancy_dict.get(first_mode, {}).get("combined", {}):
            logger.info(f"Successfully loaded cached occupancy data from {save_path}")
            return occupancy_dict
        logger.warning("Cached occupancy data is missing unified format keys — recalculating.")
        return None
    except Exception as e:
        logger.warning(
            f"Error loading cached occupancy data from {save_path}: {e}. Recalculating occupancy."
        )
        return None


def load_occupancy_cache(
    results_dir: Path,
    distance_measure: str,
    suffix: str = "",
    expected_meta: dict[str, Any] | None = None,
    channel_map: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load a KDE occupancy cache written by ``get_kde_occupancy_dict``, or raise.

    Unlike ``_load_cached_occupancy`` this never falls back to recomputing. The
    ``.meta.json`` sidecar must exist, and every key in ``expected_meta`` must match it.

    Parameters
    ----------
    results_dir
        Directory holding ``{distance_measure}_occupancy{suffix}.dat``.
    distance_measure
        Distance measure the cache was built for.
    suffix
        File-name suffix of the cache.
    expected_meta
        Grid parameters the sidecar must record; ``None`` checks only that it exists.
    channel_map
        When given, the cached mode keys must equal ``set(channel_map)``. Fit and ladder
        caches key by ``cv_channel_map``, so pass that here.

    Returns
    -------
    :
        ``{mode: {"individual": {...}, "combined": {...}}}`` as the cache holds it.

    Raises
    ------
    FileNotFoundError
        If the pickle or its sidecar is missing.
    ValueError
        If the sidecar is unreadable or differs from ``expected_meta``, the mode keys
        differ from ``channel_map``, or the cache lacks the unified format keys.
    """
    save_path = results_dir / f"{distance_measure}_occupancy{suffix}.dat"
    if not save_path.exists():
        raise FileNotFoundError(f"Occupancy cache not found: {save_path}")
    meta_path = _occupancy_cache_meta_path(save_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Occupancy cache {save_path} has no {meta_path.name} sidecar")
    try:
        cached_meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read {meta_path}: {e}") from e
    differing = {
        key: (cached_meta.get(key), value)
        for key, value in (expected_meta or {}).items()
        if cached_meta.get(key) != value
    }
    if differing:
        raise ValueError(
            f"Occupancy cache {save_path} was built with different parameters "
            f"{differing} (cached, expected)"
        )
    with open(save_path, "rb") as f:
        occupancy_dict = pickle.load(f)
    if channel_map is not None and set(occupancy_dict) != set(channel_map):
        raise ValueError(
            f"Occupancy cache {save_path} has modes {sorted(occupancy_dict)}, "
            f"expected {sorted(channel_map)}"
        )
    first_mode = next(iter(occupancy_dict), None)
    if "all_occupancy" not in occupancy_dict.get(first_mode, {}).get("combined", {}):
        raise ValueError(f"Occupancy cache {save_path} is missing the unified format keys")
    return occupancy_dict


def get_kde_occupancy_for_single_cell(
    cell_id: str,
    mode: str,
    distance_kde_dict: dict[str, dict[str, gaussian_kde]],
    common_xvals: np.ndarray | None = None,
    bandwidth: str | float | None = None,
    num_points: int = 100,
) -> tuple[str, dict[str, Any]] | None:
    """Compute KDE occupancy for a single cell; returns ``(cell_id, result)`` or ``None``."""
    if mode not in distance_kde_dict[cell_id]:
        return None

    occupied_kde = distance_kde_dict[cell_id][mode]
    available_kde = distance_kde_dict[cell_id]["available"]

    if bandwidth is not None:
        # Rebuild copies so the cached KDE objects are never mutated in place
        # (set_bandwidth mutates state shared across modes and cells).
        occupied_kde = gaussian_kde(occupied_kde.dataset, bw_method=bandwidth)
        available_kde = gaussian_kde(available_kde.dataset, bw_method=bandwidth)

    available_distances = np.asarray(available_kde.dataset, dtype=float).ravel()
    available_distances = available_distances[np.isfinite(available_distances)]

    # The occupancy ratio is only defined where the cell actually has available
    # space. Restrict evaluation to the available-data support
    # ``[avail_min, avail_max]``; beyond it the denominator KDE is pure
    # extrapolation and the ratio blows up. Every packed object is retained (it
    # still informs the occupied KDE) — we simply never report a ratio outside
    # the support, so no object filtering is needed.
    avail_min = float(np.min(available_distances))
    avail_max = float(np.max(available_distances))

    cell_lo = max(0.0, avail_min)
    # Round-1 end: the sparse tail past q95 is noise
    cell_hi = float(np.percentile(available_distances, 95))

    cell_xvals = np.linspace(cell_lo, cell_hi, num_points)
    occupancy, pdf_occupied, pdf_available = pdf_ratio(
        cell_xvals, occupied_kde.evaluate(cell_xvals), available_kde.evaluate(cell_xvals)
    )
    result: dict[str, Any] = {
        "xvals": cell_xvals,
        "occupancy": occupancy,
        "pdf_occupied": pdf_occupied,
        "pdf_available": pdf_available,
    }

    if common_xvals is None:
        common_xvals = cell_xvals

    occ_common, pdf_occ_common, pdf_avail_common = pdf_ratio(
        common_xvals, occupied_kde.evaluate(common_xvals), available_kde.evaluate(common_xvals)
    )
    # Mask x_vals outside this cell's available-data support to NaN so that
    # nanmean across cells is naturally weighted by coverage and KDE tail
    # extrapolation beyond the data range cannot corrupt the aggregate.
    out_of_support = (common_xvals < avail_min) | (common_xvals > avail_max)
    occ_common = occ_common.copy()
    occ_common[out_of_support] = np.nan
    pdf_occ_common = pdf_occ_common.copy()
    pdf_occ_common[out_of_support] = np.nan
    pdf_avail_common = pdf_avail_common.copy()
    pdf_avail_common[out_of_support] = np.nan

    result["occupancy_common"] = occ_common
    result["pdf_occupied_common"] = pdf_occ_common
    result["pdf_available_common"] = pdf_avail_common
    return cell_id, result


def _apply_coverage_mask(
    all_curves: np.ndarray,
    min_coverage_fraction: float,
    *arrays: np.ndarray,
) -> None:
    """
    Blank the sparse high-x tail of the combined aggregate (in place).

    Cells drop out of the average at their own ``avail_max``, so in the sparse
    high-x tail the mean and envelope are computed over a shrinking, changing
    subset — producing a dropout "sawtooth". Each array in ``arrays`` is set to
    NaN for grid points *beyond the last point* whose per-column coverage (the
    fraction of non-NaN rows in ``all_curves``) is at least
    ``min_coverage_fraction``. The leading low-x edge is deliberately left
    intact so the reported range still starts at ``x_min`` (e.g. z = 0); the
    per-cell data in ``all_curves`` is untouched.
    """
    n = all_curves.shape[0]
    if n == 0 or min_coverage_fraction <= 0:
        return
    coverage = np.isfinite(all_curves).sum(axis=0)
    ok = coverage >= min_coverage_fraction * n
    if not ok.any():
        return
    last_ok = int(np.max(np.nonzero(ok)[0]))
    tail = np.arange(coverage.shape[0]) > last_ok
    for arr in arrays:
        arr[tail] = np.nan


def _make_task_kde_dict(
    cell_id: str,
    mode: str,
    distance_kde_dict: dict[str, dict[str, gaussian_kde]],
    use_processes: bool,
) -> dict[str, dict[str, gaussian_kde]]:
    """Return the KDE dict to hand to a single ``get_kde_occupancy_for_single_cell`` task.

    For the thread backend the full dict is shared in-memory at no cost, so it is
    returned unchanged. For the process backend only this cell's ``mode`` and
    ``"available"`` KDEs are included: the payload is pickled once per submitted task,
    so passing the whole dict would ship every cell's KDEs to every worker and make a
    process pool *slower* than threads rather than faster.

    A missing cell yields an empty dict and a missing mode is simply absent, so the
    caller's "mode not present" early return behaves identically on both backends.
    """
    if not use_processes:
        return distance_kde_dict
    cell_entry = distance_kde_dict.get(cell_id)
    if cell_entry is None:
        return {}
    task_entry = {k: cell_entry[k] for k in (mode, "available") if k in cell_entry}
    return {cell_id: task_entry}


def _compute_mode_kde_occupancy(
    mode: str,
    structure_id: str,
    cell_id_map: dict[str, list[str]],
    distance_kde_dict: dict[str, dict[str, gaussian_kde]],
    # combined_available_kde: dict[str, gaussian_kde],
    common_xvals: np.ndarray,
    bandwidth: str | float | None,
    num_points: int,
    num_workers: int | None = None,
    min_coverage_fraction: float = 0.95,
    parallel_backend: str = "thread",
) -> dict[str, Any]:
    """
    Compute individual and combined KDE occupancy for one packing mode.

    Returns
    -------
    :
        ``{"individual": {cell_id: {...}}, "combined": {...}}`` ready to be
        stored directly under ``kde_occupancy_dict[mode]``.
    """
    individual: dict[str, dict[str, Any]] = {}
    # combined_occupied_distances: list[np.ndarray] = []
    all_occupancy_list: list[np.ndarray] = []  # shape: (n_cells, n_points)

    cell_ids = cell_id_map.get(structure_id, [])

    # ``num_workers=None`` means "unbounded" for threads, which is fine, but a process
    # pool must be bounded or it would fork one interpreter per CPU by default; fall
    # back to the ProcessPoolExecutor default rather than an unbounded count.
    use_processes = parallel_backend == "process"
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    max_workers = num_workers if num_workers and num_workers > 0 else None

    with executor_cls(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                get_kde_occupancy_for_single_cell,
                cell_id,
                mode,
                _make_task_kde_dict(cell_id, mode, distance_kde_dict, use_processes),
                common_xvals,
                bandwidth,
                num_points,
            ): cell_id
            for cell_id in cell_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Computing occupancy for {mode}",
        ):
            result = future.result()
            if result is None:
                continue
            cell_id, cell_data = result
            individual[cell_id] = cell_data

    individual = dict(sorted(individual.items()))
    all_occupancy_list = [cell_data["occupancy_common"] for cell_data in individual.values()]

    # Pooled combined occupancy across all cells
    # pdf_combined_occupied = normalize_pdf(
    #     x_vals,
    #     gaussian_kde(np.concatenate(combined_occupied_distances), bw_method=bandwidth).evaluate(
    #         x_vals
    #     ),
    # )
    # pdf_combined_available = normalize_pdf(
    #     x_vals, combined_available_kde[structure_id].evaluate(x_vals)
    # )
    # occupancy_pooled, pdf_combined_occupied, pdf_combined_available = pdf_ratio(
    #     x_vals, pdf_combined_occupied, pdf_combined_available
    # )

    if all_occupancy_list:
        all_occ_arr = np.vstack(all_occupancy_list)
        mean_occ, std_occ = _nanmean_nanstd(all_occ_arr)
        env_lo, env_hi = _nan_pointwise_envelope(all_occ_arr, alpha=0.05)
        # Trim the sparse high-x tail (dropout sawtooth) while keeping the low-x
        # start (e.g. z = 0); the per-cell all_occupancy is left untouched.
        _apply_coverage_mask(all_occ_arr, min_coverage_fraction, mean_occ, std_occ, env_lo, env_hi)
    else:
        all_occ_arr = np.empty((0, len(common_xvals)))
        mean_occ = np.zeros_like(common_xvals)
        std_occ = np.zeros_like(common_xvals)
        env_lo = np.zeros_like(common_xvals)
        env_hi = np.zeros_like(common_xvals)

    return {
        "individual": individual,
        "combined": {
            "xvals": common_xvals,
            "occupancy": mean_occ,
            # "occupancy_pooled": occupancy_pooled,
            "std_occupancy": std_occ,
            "envelope_lo": env_lo,
            "envelope_hi": env_hi,
            # "pdf_occupied": pdf_combined_occupied,
            # "pdf_available": pdf_combined_available,
            "all_occupancy": all_occ_arr,
        },
    }


def _get_common_xvals_from_kde(
    distance_kde_dict: dict[str, dict[str, gaussian_kde]],
    cell_id_map: dict[str, list[str]],
    num_points: int = 100,
    pct_min_cells: float = 95,
    x_min: float = 0,
    x_max: float | None = None,
) -> np.ndarray:
    """
    Determine a common x-axis grid for occupancy evaluation based on the data range of cells.

    Parameters
    ----------
    distance_kde_dict
        Dictionary with cell IDs as keys and mode-specific KDEs as values
    cell_id_map
        Mapping from structure IDs to lists of cell IDs
    num_points
        Number of points for the x-axis grid
    pct_min_cells
        Deprecated and ignored; retained for backward-compatible signatures.
    x_min
        Minimum x value for the grid (default 0)
    x_max
        Optional cap on the grid end.

    Returns
    -------
    :
        A common x-axis grid ending at the smallest cell's maximum available
        distance (capped at ``x_max``), so every cell covers the whole grid.
    """
    # Round-1 grid; the float32 dataset max keeps the grid float32, as in round 1.
    x_max_to_use = min(
        distance_kde_dict[cell_id]["available"].dataset.max()
        for cell_ids in cell_id_map.values()
        for cell_id in cell_ids
    )
    if x_max is not None:
        x_max_to_use = min(x_max_to_use, x_max)

    return np.linspace(x_min, x_max_to_use, num_points)


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
    x_min: float = 0,
    x_max: float | None = None,
    pct_min_cells: float = 0.05,
    num_workers: int | None = None,
    min_coverage_fraction: float = 0.95,
    parallel_backend: str = "thread",
    floors_um: dict[str, float | None] | None = None,
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
        Cap on the grid end; the grid ends at the smallest cell's maximum
        available distance. Default is None
    pct_min_cells
        Deprecated and ignored; retained for backward-compatible call signatures.
    num_workers
        Number of worker threads for parallel per-cell KDE evaluation.
        ``None`` lets :class:`~concurrent.futures.ThreadPoolExecutor` choose
        (defaults to ``min(32, os.cpu_count() + 4)``).
    min_coverage_fraction
        Minimum fraction of cells that must cover a grid point for the combined
        mean/envelope to be reported there. The sparse high-x tail beyond the
        last point meeting this threshold is blanked (NaN) to remove the
        cell-dropout sawtooth; the low-x start is preserved. Default 0.95.
        Set to 0 to disable.
    parallel_backend
        Executor used for the per-cell KDE evaluation: ``"thread"`` (default) or
        ``"process"``. Threads are cheaper and suffice when the KDE releases the
        GIL; ``"process"`` pays pickling cost per cell but scales past it on
        large cohorts. Any other value is treated as ``"thread"``.
    floors_um
        Distance floors (microns) the KDEs were built with; stamped into the cache
        sidecar so a floor change is not silently reloaded.

    Returns
    -------
    :
        ``{mode: {"individual": {cell_id: {"xvals", "occupancy",
        "pdf_occupied", "pdf_available", "occupancy_common",
        "pdf_occupied_common", "pdf_available_common"}},
        "combined": {"xvals", "occupancy", "std_occupancy",
        "envelope_lo", "envelope_hi", "all_occupancy"}}}``

        ``combined["occupancy"]`` is the ``nanmean`` of the per-cell ratio
        curves evaluated on the shared ``x_vals`` grid (each cell contributes
        only within its own available-space support). Pooled-KDE ratios and
        pooled PDFs are intentionally not computed here — they live on the
        discrete/binned path.
        ``combined["all_occupancy"]`` has shape ``(n_cells, n_points)``.
    """
    save_path = (
        results_dir / f"{distance_measure}_occupancy{suffix}.dat"
        if results_dir is not None
        else None
    )

    cache_meta: dict[str, Any] = {
        "distance_measure": distance_measure,
        "num_points": num_points,
        "x_min": x_min,
        "x_max": x_max,
        "bandwidth": bandwidth,
        "min_coverage_fraction": min_coverage_fraction,
        **{k: None if v is None else float(v) for k, v in (floors_um or {}).items()},
    }

    if not recalculate and save_path is not None:
        cached = _load_cached_occupancy(save_path, channel_map, expected_meta=cache_meta)
        if cached is not None:
            return cached

    cell_id_map = get_cell_id_map_from_distance_kde_dict(distance_kde_dict, channel_map)
    if num_cells is not None:
        for structure_id, cell_ids in cell_id_map.items():
            if len(cell_ids) > num_cells:
                cell_id_map[structure_id] = np.random.choice(
                    cell_ids, size=num_cells, replace=False
                ).tolist()

    common_x_vals = _get_common_xvals_from_kde(
        distance_kde_dict, cell_id_map, num_points, pct_min_cells, x_min, x_max
    )

    kde_occupancy_dict = {
        mode: _compute_mode_kde_occupancy(
            mode,
            structure_id,
            cell_id_map,
            distance_kde_dict,
            # combined_available_kde,
            common_x_vals,
            bandwidth,
            num_points,
            num_workers=num_workers,
            min_coverage_fraction=min_coverage_fraction,
            parallel_backend=parallel_backend,
        )
        for mode, structure_id in channel_map.items()
    }

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(kde_occupancy_dict, f)
        _write_occupancy_cache_meta(save_path, cache_meta)

    return kde_occupancy_dict


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
    # Align with the KDE path (pdf_ratio): an unreliable +inf ratio (denominator
    # -> 0 with non-zero numerator) propagates as NaN rather than 0.
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=np.nan, neginf=0.0)

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


def _build_cell_id_map_from_distance_dict(
    distance_dict_dm: dict[str, dict[str, dict[str, np.ndarray]]],
    channel_map: dict[str, str],
    num_cells: int | None,
) -> dict[str, list[str]]:
    """
    Build ``{structure_id: [cell_ids]}`` from the per-mode distance dict.

    Collects every cell ID that appears under any mode in ``distance_dict_dm``
    and groups them by the corresponding structure ID taken from
    ``channel_map``.  Optionally sub-samples each structure to at most
    ``num_cells`` cells using a fixed random seed for reproducibility.

    Parameters
    ----------
    distance_dict_dm
        ``{mode: {cell_id: {seed: distances}}}`` for a single distance measure.
    channel_map
        Mapping from packing modes to structure IDs.
    num_cells
        Maximum number of cells to retain per structure.  When ``None`` all
        cells are kept.

    Returns
    -------
    :
        ``{structure_id: [cell_id, ...]}`` with at most ``num_cells`` entries
        per structure.
    """
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

    return cell_id_map


def _infer_common_r_grid(
    cell_id_map: dict[str, list[str]],
    combined_mesh_information_dict: dict[str, dict[str, dict[str, Any]]],
    grid_key: str,
    is_scaled_measure: bool,
    x_min: float,
    x_max: float | None,
    bin_width: float,
) -> np.ndarray:
    """
    Return the shared bin-centre grid used for combined occupancy.

    When ``x_max`` is ``None`` the range is inferred from the pooled
    available-space distances of every cell in ``cell_id_map`` via
    :func:`make_r_grid_from_pooled`.  Otherwise an evenly-spaced grid
    spanning ``[x_min, x_max]`` with step ``bin_width`` is constructed
    directly.

    Parameters
    ----------
    cell_id_map
        ``{structure_id: [cell_id, ...]}`` as returned by
        :func:`_build_cell_id_map_from_distance_dict`.
    combined_mesh_information_dict
        ``{structure_id: {cell_id: {grid_key: distances, ...}}}``.
    grid_key
        Key used to look up available-space distances inside each cell's mesh
        info dict (e.g. ``"nuc_grid_distances"``).
    is_scaled_measure
        When ``False``, raw distances are multiplied by ``PIXEL_SIZE_IN_UM``
        before being passed to :func:`make_r_grid_from_pooled`.
    x_min
        Lower bound of the grid (used only when ``x_max`` is not ``None``).
    x_max
        Upper bound of the grid.  ``None`` triggers automatic inference.
    bin_width
        Histogram bin width in µm (or unitless for scaled measures).

    Returns
    -------
    :
        1-D array of bin-centre positions on the common grid.
    """
    if x_max is None:
        all_available: list[np.ndarray] = []
        for structure_id, cell_ids in cell_id_map.items():
            mesh_info = combined_mesh_information_dict.get(structure_id, {})
            for cell_id in cell_ids:
                avail = mesh_info.get(cell_id, {}).get(grid_key)
                if avail is not None:
                    a = np.asarray(avail)
                    if not is_scaled_measure:
                        a = a * PIXEL_SIZE_IN_UM  # type: ignore[operator]
                    all_available.append(a)
        return make_r_grid_from_pooled(all_available, bin_width=bin_width)

    n_bins = max(1, round((x_max - x_min) / bin_width))
    return np.linspace(x_min + bin_width / 2, x_max - bin_width / 2, n_bins)


def _prepare_cell_distances(
    distance_dict_dm: dict[str, dict[str, dict[str, np.ndarray]]],
    mesh_info: dict[str, dict[str, Any]],
    cell_id: str,
    mode: str,
    grid_key: str,
    is_scaled_measure: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract and validate per-cell occupied/available distance arrays.

    Reads the available-space distances from ``mesh_info`` and the occupied
    distances from ``distance_dict_dm``, converts pixel units to µm when
    required, filters out non-finite values, and clips occupied distances to
    the maximum available distance when the occupied range exceeds it
    (logging a warning in that case).

    Parameters
    ----------
    distance_dict_dm
        ``{mode: {cell_id: {seed: distances}}}`` for a single distance measure.
    mesh_info
        ``{cell_id: {grid_key: distances, ...}}`` for one structure.
    cell_id
        Cell to extract distances for.
    mode
        Packing mode whose occupied distances are used.
    grid_key
        Key used to look up available-space distances inside ``mesh_info``.
    is_scaled_measure
        When ``False``, available-space distances are multiplied by
        ``PIXEL_SIZE_IN_UM`` to convert from pixel units to µm.

    Returns
    -------
    :
        ``(occupied_concat, available_concat)`` both as finite 1-D arrays in
        final units, or ``None`` when no available-space data exists for the
        cell.
    """
    avail_raw = mesh_info.get(cell_id, {}).get(grid_key)
    if avail_raw is None:
        logger.warning("No available-space grid for cell %s / %s", cell_id, mode)
        return None

    occupied_concat = np.concatenate(
        [np.asarray(a) for a in distance_dict_dm[mode][cell_id].values()]
    )
    occupied_concat = occupied_concat[np.isfinite(occupied_concat)]

    available_concat = np.asarray(avail_raw, dtype=float)
    if not is_scaled_measure:
        available_concat = available_concat * PIXEL_SIZE_IN_UM
    available_concat = available_concat[np.isfinite(available_concat)]

    # No object filtering: packed objects beyond the available-space range are
    # retained. They fall outside the per-cell histogram grid (built from the
    # available distances) and are simply not counted, rather than discarded.
    return occupied_concat, available_concat


def _collect_mode_cell_results(
    mode: str,
    distance_dict_dm: dict[str, dict[str, dict[str, np.ndarray]]],
    mesh_info: dict[str, dict[str, Any]],
    valid_cell_ids: list[str],
    grid_key: str,
    is_scaled_measure: bool,
    bin_width: float,
    pseudocount: float,
    min_count: int,
    common_r_grid: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """
    Iterate over cells for one mode and compute per-cell occupancy.

    For each valid cell: extracts distances via :func:`_prepare_cell_distances`,
    computes the per-cell histogram occupancy ratio via
    :func:`_compute_single_cell_occupancy`, and interpolates that ratio onto
    ``common_r_grid`` for later aggregation.

    Parameters
    ----------
    mode
        Packing mode being processed.
    distance_dict_dm
        ``{mode: {cell_id: {seed: distances}}}`` for a single distance measure.
    mesh_info
        ``{cell_id: {grid_key: distances, ...}}`` for the structure associated
        with ``mode``.
    valid_cell_ids
        Ordered list of cell IDs to iterate over.
    grid_key
        Key used to look up available-space distances inside ``mesh_info``.
    is_scaled_measure
        Forwarded to :func:`_prepare_cell_distances` to control unit conversion.
    bin_width
        Histogram bin width passed to :func:`_compute_single_cell_occupancy`.
    pseudocount
        Laplace smoothing constant passed to :func:`_compute_single_cell_occupancy`.
    min_count
        Minimum raw available-space count for a bin to be retained.
    common_r_grid
        Shared bin-centre grid onto which per-cell ratios are interpolated.

    Returns
    -------
    :
        A 4-tuple ``(individual_dict, all_occupied_distances,
        all_available_distances, interp_ratios)`` where

        * ``individual_dict`` — ``{cell_id: per-cell occupancy result}``
        * ``all_occupied_distances`` — list of per-cell occupied distance arrays
        * ``all_available_distances`` — list of per-cell available distance arrays
        * ``interp_ratios`` — list of per-cell ratio curves on ``common_r_grid``
    """
    individual: dict[str, dict[str, np.ndarray]] = {}
    all_occupied_distances: list[np.ndarray] = []
    all_available_distances: list[np.ndarray] = []
    interp_ratios: list[np.ndarray] = []

    for cell_id in tqdm(valid_cell_ids, desc=f"Binned occupancy ({mode})"):
        if cell_id not in distance_dict_dm.get(mode, {}):
            continue

        distances = _prepare_cell_distances(
            distance_dict_dm, mesh_info, cell_id, mode, grid_key, is_scaled_measure
        )
        if distances is None:
            continue
        occupied_concat, available_concat = distances

        cell_result = _compute_single_cell_occupancy(
            occupied=occupied_concat,
            available=available_concat,
            bin_width=bin_width,
            pseudocount=pseudocount,
            min_count=min_count,
        )
        individual[cell_id] = cell_result

        all_occupied_distances.append(occupied_concat)
        all_available_distances.append(available_concat)

        occ_common = np.interp(
            common_r_grid, cell_result["xvals"], cell_result["occupancy"], left=np.nan, right=np.nan
        )
        pdf_occ_common = np.interp(
            common_r_grid,
            cell_result["xvals"],
            cell_result["pdf_occupied"],
            left=np.nan,
            right=np.nan,
        )
        pdf_avail_common = np.interp(
            common_r_grid,
            cell_result["xvals"],
            cell_result["pdf_available"],
            left=np.nan,
            right=np.nan,
        )
        individual[cell_id]["occupancy_common"] = occ_common
        individual[cell_id]["pdf_occupied_common"] = pdf_occ_common
        individual[cell_id]["pdf_available_common"] = pdf_avail_common
        interp_ratios.append(occ_common)

    return individual, all_occupied_distances, all_available_distances, interp_ratios


def _aggregate_combined_occupancy(
    interp_ratios: list[np.ndarray],
    all_occupied_distances: list[np.ndarray],
    all_available_distances: list[np.ndarray],
    common_r_grid: np.ndarray,
    common_bin_edges: np.ndarray,
    pseudocount: float,
    min_count: int,
    min_coverage_fraction: float = 0.95,
) -> dict[str, Any]:
    """
    Aggregate per-cell results into combined occupancy statistics.

    Produces both the mean-of-interpolated-ratios occupancy (``"occupancy"``) and
    the pooled ratio from all cells' concatenated distances
    (``"occupancy_pooled"``), along with standard deviation, pointwise
    envelope bounds, and pooled PDFs — all reported on ``common_r_grid``.  When
    ``interp_ratios`` is empty a zero-filled fallback dict is returned.

    Parameters
    ----------
    interp_ratios
        Per-cell ratio curves already interpolated onto ``common_r_grid``
        (``NaN`` outside each cell's own distance range).
    all_occupied_distances
        Per-cell occupied distance arrays in final units.
    all_available_distances
        Per-cell available-space distance arrays in final units.
    common_r_grid
        Bin-centre positions of the shared grid.
    common_bin_edges
        Bin edges corresponding to ``common_r_grid``.
    pseudocount
        Laplace smoothing constant added to pooled histogram counts before
        normalization.
    min_count
        Minimum raw pooled available-space count for a bin to be retained in
        ``"occupancy_pooled"``.
    min_coverage_fraction
        Minimum fraction of cells that must cover a grid point for the combined
        mean/envelope to be reported there. The sparse high-x tail beyond the
        last qualifying point is blanked; the low-x start is preserved.

    Returns
    -------
    :
        ``{"xvals", "occupancy", "occupancy_pooled", "std_occupancy",
        "envelope_lo", "envelope_hi", "pdf_occupied", "pdf_available",
        "all_occupancy"}`` — all arrays on ``common_r_grid``.
    """
    if not interp_ratios:
        return {
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

    ratio_arr = np.vstack(interp_ratios)
    mean_ratio, std_ratio = _nanmean_nanstd(ratio_arr)
    env_lo, env_hi = _nan_pointwise_envelope(ratio_arr, alpha=0.05)
    # Trim the sparse high-x tail (dropout sawtooth); keep the low-x start.
    _apply_coverage_mask(ratio_arr, min_coverage_fraction, mean_ratio, std_ratio, env_lo, env_hi)

    pooled_occupied = np.concatenate(all_occupied_distances)
    pooled_available = np.concatenate(all_available_distances)
    raw_occ_pooled, _ = np.histogram(pooled_occupied, bins=common_bin_edges)
    raw_avail_pooled, _ = np.histogram(pooled_available, bins=common_bin_edges)
    pdf_occ_pooled = normalize_pdf(common_r_grid, raw_occ_pooled.astype(float) + pseudocount)
    pdf_avail_pooled = normalize_pdf(common_r_grid, raw_avail_pooled.astype(float) + pseudocount)

    reg = 1e-10
    ratio_pooled = np.exp(
        np.log(np.maximum(pdf_occ_pooled, reg)) - np.log(np.maximum(pdf_avail_pooled, reg))
    )
    # Align with the KDE path: unreliable +inf pooled ratio propagates as NaN.
    ratio_pooled = np.nan_to_num(ratio_pooled, nan=0.0, posinf=np.nan, neginf=0.0)
    if min_count > 0:
        ratio_pooled[raw_avail_pooled < min_count] = np.nan

    return {
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
    min_coverage_fraction: float = 0.95,
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
    min_coverage_fraction
        Minimum fraction of cells that must cover a grid point for the combined
        mean/envelope to be reported there. The sparse high-x tail beyond the
        last qualifying point is blanked; the low-x start is preserved.
        Default 0.95. Set to 0 to disable.
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
    # Scaled measures (e.g. scaled_nucleus) are unitless; others are in pixel units
    # and must be converted to µm to match the occupied distances.
    is_scaled_measure = "scaled" in distance_measure

    cell_id_map = _build_cell_id_map_from_distance_dict(distance_dict_dm, channel_map, num_cells)
    common_r_grid = _infer_common_r_grid(
        cell_id_map,
        combined_mesh_information_dict,
        grid_key,
        is_scaled_measure,
        x_min,
        x_max,
        bin_width,
    )
    common_bin_edges = np.concatenate(
        [
            common_r_grid - bin_width / 2,
            [common_r_grid[-1] + bin_width / 2],
        ]
    )

    occupancy_result: dict[str, dict[str, Any]] = {}
    for mode, structure_id in channel_map.items():
        mesh_info = combined_mesh_information_dict.get(structure_id, {})
        valid_cell_ids = cell_id_map.get(structure_id, [])

        individual, all_occupied, all_available, interp_ratios = _collect_mode_cell_results(
            mode=mode,
            distance_dict_dm=distance_dict_dm,
            mesh_info=mesh_info,
            valid_cell_ids=valid_cell_ids,
            grid_key=grid_key,
            is_scaled_measure=is_scaled_measure,
            bin_width=bin_width,
            pseudocount=pseudocount,
            min_count=min_count,
            common_r_grid=common_r_grid,
        )
        occupancy_result[mode] = {
            "individual": individual,
            "combined": _aggregate_combined_occupancy(
                interp_ratios,
                all_occupied,
                all_available,
                common_r_grid,
                common_bin_edges,
                pseudocount,
                min_count,
                min_coverage_fraction,
            ),
        }

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(occupancy_result, f)
        logger.info("Saved binned occupancy (from distance dict) to %s", save_path)

    return occupancy_result


def pairwise_envelope_test_occupancy(
    combined_occupancy_dict: dict[str, dict[str, dict[str, Any]]],
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
    combined_occupancy_dict
        ``{distance_measure: {mode: {"individual": {cell_id: {"occupancy":
        np.ndarray, "xvals": np.ndarray}}, "combined": {...}}}}`` — the output
        of :func:`get_kde_occupancy_dict` or
        :func:`get_binned_occupancy_dict_from_distance_dict` wrapped in a
        distance-measure outer key.  Both functions share the same unified
        format and are accepted here.
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
    distance_measures = list(combined_occupancy_dict.keys())

    envelopes: dict[str, dict[str, Any]] = {}
    mode_curves: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in packing_modes}
    mode_xvals: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in packing_modes}

    for dm in distance_measures:
        occ_dm = combined_occupancy_dict[dm]
        for mode in packing_modes:
            combined = occ_dm.get(mode, {}).get("combined", {})
            xvals = combined.get("xvals", np.array([]))

            # Use the pre-interpolated per-cell ratios on the common grid
            arr = combined.get("all_occupancy", np.empty((0, max(1, len(xvals)))))
            # Replace NaN with 0 for stacking/envelope operations
            arr = np.where(np.isnan(arr), 0.0, arr)

            mode_xvals[mode][dm] = xvals
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
            ratio_grid = np.linspace(0, np.nanpercentile(all_ratio_vals, 95), max(n_grid, 2))
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
