import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, squareform
from scipy.stats import gaussian_kde, ks_2samp
from tqdm import tqdm

from cellpack_analysis.lib import label_tables
from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import (
    add_file_handler_to_logger,
    get_project_root,
    remove_file_handler_from_logger,
)
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.mesh_tools import _compute_distances_for_points
from cellpack_analysis.lib.stats import (
    make_r_grid_from_pooled,
    normalize_pdf,
    pointwise_envelope,
    ripley_k,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = get_project_root()


def _load_pickle_with_validation(
    file_path: Path,
    validator: Any = None,
    context: str = "",
) -> Any | None:
    """
    Load pickled data with optional validation.

    Parameters
    ----------
    file_path
        Path to the pickle file
    validator
        Optional callable that takes loaded data and returns (is_valid, message)
    context
        Description of what's being loaded for logging

    Returns
    -------
    :
        Loaded data if successful and valid, None otherwise
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if validator is not None:
            is_valid, message = validator(data)
            if not is_valid:
                logger.warning(
                    f"Validation failed for {file_path.relative_to(PROJECT_ROOT)}: {message}"
                )
                return None

        logger.info(
            f"Successfully loaded cached {context} from {file_path.relative_to(PROJECT_ROOT)}"
        )
        return data
    except Exception as e:
        logger.warning(
            f"Error loading cached {context} from {file_path.relative_to(PROJECT_ROOT)}: {e}"
        )
        return None


def _load_parquet_if_exists(
    file_path: Path,
    recalculate: bool = False,
    context: str = "",
) -> pd.DataFrame | None:
    """
    Load parquet file if it exists and recalculate is False.

    Parameters
    ----------
    file_path
        Path to the parquet file
    recalculate
        Whether to skip loading and force recalculation
    context
        Description of what's being loaded for logging

    Returns
    -------
    :
        DataFrame if successfully loaded, None otherwise
    """
    if not recalculate and file_path.exists():
        logger.info(f"Loading {context} from {file_path.relative_to(PROJECT_ROOT)}")
        return pd.read_parquet(file_path)
    return None


def _validate_modes_match(
    cached_modes: set[str],
    requested_modes: set[str],
) -> tuple[bool, str]:
    """
    Validate that cached modes match requested modes.

    Parameters
    ----------
    cached_modes
        Set of modes found in cached data
    requested_modes
        Set of modes that were requested

    Returns
    -------
    :
        Tuple of (is_valid, error_message)
    """
    if cached_modes != requested_modes:
        return False, f"modes mismatch: cached {cached_modes} != requested {requested_modes}"
    return True, ""


def _validate_single_mode_distance_dict(
    data: dict[str, Any],
) -> tuple[bool, str]:
    """
    Validate a single-mode cached distance file (``{dm}_{mode}_distances.dat``).

    Parameters
    ----------
    data
        Dictionary ``{cell_id: {seed: distances}}`` loaded from one per-mode file.

    Returns
    -------
    :
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict) or len(data) == 0:
        return False, "empty or invalid distance data"
    return True, ""


def _validate_kde_dict(
    data: dict[str, Any],
    requested_modes: set[str],
) -> tuple[bool, str]:
    """
    Validate cached KDE dictionary.

    Extracts modes from nested structure and checks they match requested modes.

    Parameters
    ----------
    data
        KDE dictionary to validate. Has structure:
        {cell_id:{mode:gaussian_kde, "available":gaussian_kde}}
    requested_modes
        Set of modes that were requested

    Returns
    -------
    :
        Tuple of (is_valid, error_message)
    """
    cached_modes = set()
    for _, cell_dict in data.items():
        cached_modes.update(set(cell_dict.keys()) - {"available"})

    return _validate_modes_match(cached_modes, requested_modes)


_PDF_DICT_REQUIRED_KEYS = {"xvals", "individual_curves", "mean_pdf", "envelope_lo", "envelope_hi"}


def _validate_distance_pdfs_dict(
    data: dict[str, Any],
    distance_measures: list[str],
    packing_modes: list[str],
) -> tuple[bool, str]:
    """
    Validate a cached distance PDFs dictionary.

    Checks that all requested distance measures and packing modes are present
    and that each inner entry contains the required array keys.

    Parameters
    ----------
    data
        PDFs dictionary to validate.  Expected structure::

            {distance_measure: {mode: {"xvals": …, "individual_curves": …,
            "mean_pdf": …, "envelope_lo": …, "envelope_hi": …}}}

    distance_measures
        Distance measures that must be present.
    packing_modes
        Packing modes that must be present for every distance measure.

    Returns
    -------
    :
        Tuple of (is_valid, error_message)
    """
    requested_dms = set(distance_measures)
    cached_dms = set(data.keys())
    is_valid, message = _validate_modes_match(cached_dms, requested_dms)
    if not is_valid:
        return False, f"distance measures {message}"

    requested_modes = set(packing_modes)
    for dm, mode_dict in data.items():
        is_valid, message = _validate_modes_match(set(mode_dict.keys()), requested_modes)
        if not is_valid:
            return False, f"{dm}: packing modes {message}"
        for mode, inner in mode_dict.items():
            missing = _PDF_DICT_REQUIRED_KEYS - set(inner.keys())
            if missing:
                return False, f"{dm}/{mode}: missing keys {missing}"

    return True, ""


def filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    minimum_distance: float | None = None,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """
    Filter out invalid values from distance distribution dictionary.

    Removes NaN, infinite, and optionally negative distance values from all
    distance arrays in the nested dictionary structure.

    Parameters
    ----------
    distance_distribution_dict
        Distance distributions with structure
        {distance_measure: {packing_mode: {cell_id:{seed: distances}}}}
    minimum_distance
        Minimum distance to consider for filtering invalid distances

    Returns
    -------
    :
        Cleaned distance distribution dictionary with same structure as input
        {distance_measure: {packing_mode: {cell_id:{seed: valid_distances}}}}
    """
    for distance_measure, distance_measure_dict in distance_distribution_dict.items():
        for mode, mode_dict in distance_measure_dict.items():
            for cell_id, seed_dict in mode_dict.items():
                for seed, distances in seed_dict.items():
                    # filter out NaN and inf values
                    seed_dict[seed] = filter_invalid_distances(
                        distances, minimum_distance=minimum_distance
                    )
                    num_valid = len(seed_dict[seed])
                    if num_valid == 0:
                        logger.warning(
                            "All distances are invalid for %s, %s, %s, %s",
                            distance_measure,
                            mode,
                            cell_id,
                            seed,
                        )
                    else:
                        logger.debug(
                            "Filtered distances for %s, %s, %s, %s: %d valid distances",
                            distance_measure,
                            mode,
                            cell_id,
                            seed,
                            num_valid,
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


def _normalize_and_filter_distances(
    distances: np.ndarray,
    normalization: str | None,
    mesh_information_dict: dict[str, dict[str, Any]],
    cell_id: str,
    distance_measure: str,
    minimum_distance: float | None = None,
) -> np.ndarray:
    """
    Normalize and filter distances for KDE calculation.

    Parameters
    ----------
    distances
        Array of distance values to normalize and filter
    normalization
        Normalization method to use
    mesh_information_dict
        Dictionary containing mesh information for each cell_id
    cell_id
        Cell identifier
    distance_measure
        Distance measure being processed
    minimum_distance
        Minimum distance to consider for filtering

    Returns
    -------
    :
        Normalized and filtered distance array as float32
    """
    normalization_factor = get_normalization_factor(
        normalization=normalization,
        mesh_information_dict=mesh_information_dict,
        cell_id=cell_id,
        distance_measure=distance_measure,
        distances=distances,
    )
    normalized_distances = distances / normalization_factor
    filtered_distances = filter_invalid_distances(
        normalized_distances, minimum_distance=minimum_distance
    )

    if len(filtered_distances) == 0:
        raise ValueError(
            f"All distances are invalid after normalization for cell_id {cell_id}, "
            f"distance_measure {distance_measure}."
        )
    return filtered_distances.astype(np.float32)


def _calculate_distances_for_cell_id(
    cell_id: str,
    positions: np.ndarray,
    mesh_dict: dict[str, Any],
    distance_measures: list[str] | None = None,
) -> tuple[str, dict[str, np.ndarray]]:
    """
    Calculate various distance measures for particles in a single cell.

    Computes pairwise distances, nearest neighbor distances, and distances
    to cellular structures (nucleus, membrane) for the given particle positions.
    When ``distance_measures`` is provided only the requested measures are
    computed, skipping expensive operations that are not needed.

    Parameters
    ----------
    cell_id
        Identifier for the cell
    positions
        3D coordinates of particles in the cell, shape (N, 3)
    mesh_dict
        Dictionary containing mesh information for the cell including
        'nuc_mesh', 'mem_mesh', and 'mem_bounds'
    distance_measures
        List of distance measures to compute. When ``None`` all measures are
        computed. Supported values: 'pairwise', 'nearest', 'nucleus',
        'scaled_nucleus', 'membrane', 'z', 'scaled_z'

    Returns
    -------
    :
        Tuple of (cell_id, distance_dict) where distance_dict contains arrays
        for the requested distance measures.

    Raises
    ------
    ValueError
        If mesh information not found for the specified cell_id
    """
    requested: frozenset[str] = (
        frozenset(distance_measures) if distance_measures is not None else frozenset()
    )
    compute_all = distance_measures is None

    def _needed(measure: str) -> bool:
        return compute_all or measure in requested

    distance_dict: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Shape-dependent + z measures via the shared primitive                #
    # ------------------------------------------------------------------ #
    # SHAPE_DEPENDENT_MEASURES includes z and scaled_z: all five measures
    # require the mesh (nucleus/scaled_nucleus/membrane for proximity queries;
    # z/scaled_z for mem_mesh.bounds).
    need_shape = compute_all or bool(requested & label_tables.SHAPE_DEPENDENT_MEASURES)

    if need_shape:
        if cell_id not in mesh_dict:
            raise ValueError(f"Mesh information not found for cell_id: {cell_id}")

        nuc_mesh = mesh_dict[cell_id]["nuc_mesh"]
        mem_mesh = mesh_dict[cell_id]["mem_mesh"]

        # Proximity-based measures (nucleus, scaled_nucleus, membrane) require
        # mesh signed-distance queries and enable the cytoplasm-fraction warning.
        # z / scaled_z only need mem_mesh.bounds and are handled separately so
        # we don't pay for an unnecessary membrane proximity query in z-only calls.
        need_proximity = compute_all or bool(
            requested & frozenset({"nucleus", "scaled_nucleus", "membrane"})
        )

        prim_measures: set[str] = set()
        if need_proximity:
            # Always include 'membrane' so we can emit the cytoplasm-fraction
            # warning even when 'membrane' was not explicitly requested.
            prim_measures.add("membrane")
            if _needed("nucleus"):
                prim_measures.add("nucleus")
            if _needed("scaled_nucleus"):
                prim_measures.add("scaled_nucleus")
        if _needed("z"):
            prim_measures.add("z")
        if _needed("scaled_z"):
            prim_measures.add("scaled_z")

        prim_result = _compute_distances_for_points(
            positions,
            nuc_mesh,
            mem_mesh,
            prim_measures,
        )

        # Cytoplasm-fraction warning (uses raw, unfiltered arrays).
        # Only meaningful for proximity-based measures.
        if need_proximity:
            mem_distances_raw = prim_result["membrane"]
            # nucleus distances are positive outside the nucleus surface.
            nuc_distances_raw = prim_result.get("nucleus", prim_result.get("scaled_nucleus"))
            num_points = len(positions)
            outside_membrane_mask = mem_distances_raw < 0
            num_outside_membrane = int(outside_membrane_mask.sum())
            fraction_outside_membrane = num_outside_membrane / num_points

            if nuc_distances_raw is not None:
                inside_nucleus_mask = nuc_distances_raw < 0
                num_inside_nucleus = int(inside_nucleus_mask.sum())
                fraction_inside_nucleus = num_inside_nucleus / num_points
            else:
                num_inside_nucleus = 0
                fraction_inside_nucleus = 0.0

            fraction_non_cytoplasm = fraction_inside_nucleus + fraction_outside_membrane
            logger.debug(f"Fraction non-cytoplasm positions: {fraction_non_cytoplasm:.2f}")
            if fraction_non_cytoplasm > 0.1:
                logger.warning(
                    f"More than 10% of positions are outside cytoplasm for cell {cell_id}: "
                    f"{fraction_non_cytoplasm:.2f}\n"
                    f"Inside nucleus: {num_inside_nucleus} / {num_points} = "
                    f"{fraction_inside_nucleus:.2f}, "
                    f"Outside membrane: {num_outside_membrane} / {num_points} = "
                    f"{fraction_outside_membrane:.2f}"
                )

        # Store filtered results.
        for dm in ("nucleus", "scaled_nucleus", "membrane", "z", "scaled_z"):
            if dm in prim_result and _needed(dm):
                arr = prim_result[dm]
                if dm == "scaled_z" and len(arr) == 0:
                    # z_range was near-zero; warn and skip.
                    logger.warning(f"Zero z-range for cell {cell_id}, skipping scaled_z distances")
                    continue
                distance_dict[dm] = filter_invalid_distances(arr)
    # ------------------------------------------------------------------ #
    # Shape-independent distance measures: pairwise / nearest              #
    # ------------------------------------------------------------------ #
    need_pairwise = _needed("pairwise")
    need_nearest = _needed("nearest")

    if need_pairwise:
        # Build full NxN matrix; reuse for nearest-neighbor if also needed
        all_distances = cdist(positions, positions, metric="euclidean")
        pairwise_distances = squareform(all_distances)
        distance_dict["pairwise"] = filter_invalid_distances(pairwise_distances)

        if need_nearest:
            np.fill_diagonal(all_distances, np.inf)
            nearest_distances = np.min(all_distances, axis=1)
            distance_dict["nearest"] = filter_invalid_distances(nearest_distances)
    elif need_nearest:
        # Fast path: O(N log N) kd-tree, avoids materialising the NxN matrix
        tree = KDTree(positions)
        nn_dist, _ = tree.query(positions, k=2)  # k=2: column 0 is self (dist=0)
        distance_dict["nearest"] = filter_invalid_distances(nn_dist[:, 1])

    logger.debug(
        "Calculated distances for cell %s: %s",
        cell_id,
        ", ".join(f"{k}={len(v)}" for k, v in distance_dict.items()),
    )

    return cell_id, distance_dict


def process_cell(
    cell_id: str,
    seed_positions_dict: dict[str, np.ndarray],
    mode: str,
    distance_measures: list[str],
    mode_mesh_dict: dict[str, Any],
):
    """
    Process a single cell ID with all its seeds.

    Parameters
    ----------
    cell_id
        Identifier for the cell
    seed_positions_dict
        Dictionary of seed to positions for the cell
        {seed: positions}
    mode
        Packing mode name
    distance_measures
        List of distance measures to calculate
    mode_mesh_dict
        Mesh information dictionary for the current mode

    Returns
    -------
    :
        Tuple of (cell_id, cell_results) where cell_results is
        {distance_measure: {seed: distances}}
    """
    cell_results = {dm: {} for dm in distance_measures}

    for seed, seed_positions in seed_positions_dict.items():
        try:
            _, distances_dict = _calculate_distances_for_cell_id(
                cell_id=cell_id,
                positions=seed_positions,
                mesh_dict=mode_mesh_dict,
                distance_measures=distance_measures,
            )
        except Exception as e:
            logger.error(
                "Error calculating distances for cell %s, mode %s, seed %s: %s",
                cell_id,
                mode,
                seed,
                e,
            )
            continue

        for distance_measure in distance_measures:
            if distance_measure not in distances_dict:
                raise ValueError(f"Distance measure {distance_measure} not found")
            cell_results[distance_measure][seed] = distances_dict[distance_measure]

    return cell_id, cell_results


def _run_calculation_for_mode(
    mode: str,
    dms_for_mode: list[str],
    position_dict: dict[str, dict[str, np.ndarray]],
    mode_mesh_dict: dict[str, Any],
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    num_workers: int,
) -> None:
    """Compute distances for *mode* for the given subset of distance measures.

    Results are written in-place into ``all_distance_dict``.
    """
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_cell,
                    cell_id,
                    seed_positions_dict,
                    mode,
                    dms_for_mode,
                    mode_mesh_dict,
                ): cell_id
                for cell_id, seed_positions_dict in position_dict.items()
            }

            with tqdm(
                total=len(futures),
                desc=f"Calculating distances for {label_tables.MODE_LABELS.get(mode, mode)}",
                unit="cell IDs",
            ) as pbar:
                for future in as_completed(futures):
                    cell_id = futures[future]
                    try:
                        cell_id_result, cell_results = future.result()
                        for dm in dms_for_mode:
                            all_distance_dict[dm][mode][cell_id_result] = cell_results[dm]
                    except Exception as e:
                        logger.error(
                            "Error processing cell %s for mode %s: %s",
                            cell_id,
                            mode,
                            e,
                        )
                    pbar.update(1)
    else:
        for cell_id, seed_positions_dict in tqdm(
            position_dict.items(),
            desc=f"Calculating distances for {label_tables.MODE_LABELS.get(mode, mode)}",
            total=len(position_dict),
            unit="cell IDs",
        ):
            cell_id_result, cell_results = process_cell(
                cell_id, seed_positions_dict, mode, dms_for_mode, mode_mesh_dict
            )
            for dm in dms_for_mode:
                all_distance_dict[dm][mode][cell_id_result] = cell_results[dm]


def get_distance_dictionary(
    all_positions: dict[str, dict[str, dict[str, np.ndarray]]],
    distance_measures: list[str],
    mesh_information_dict: dict[str, dict[str, Any]],
    channel_map: dict[str, str] | None = None,
    results_dir: Path | None = None,
    recalculate: bool = False,
    num_workers: int = 1,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """
    Calculate or load distance measures between particles in different modes.

    Cache files are stored one per ``(distance_measure, mode)`` pair::

        {results_dir}/{distance_measure}_{mode}_distances.dat

    Each file contains ``{cell_id: {seed: distances_array}}``.  When only some
    ``(distance_measure, mode)`` pairs are missing from the cache, only those
    pairs are recalculated and their files are written; existing files for
    fully-cached pairs are left untouched.

    Parameters
    ----------
    all_positions
        A dictionary containing positions of particles in different packing modes
        {mode: {cell_id: {seed: positions}}}
    distance_measures
        List of distance measures to calculate
    mesh_information_dict
        A dictionary containing mesh information
    channel_map
        Mapping between modes and channel names
    results_dir
        The directory to save or load distance dictionaries
    recalculate
        If True, recalculate all distance measures regardless of cache
    num_workers
        Number of parallel workers for distance calculations

    Returns
    -------
    :
        A dictionary containing distance measures between particles in different modes
        {distance_measure: {mode: {cell_id: {seed:distances}}}
    """
    if channel_map is None:
        channel_map = {}

    all_modes = list(all_positions.keys())

    # Initialise output structure (will be filled from cache or calculation)
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {
        dm: {mode: {} for mode in all_modes} for dm in distance_measures
    }

    # Track which (dm, mode) pairs still need to be computed.
    # maps dm -> set of modes that need (re)calculation
    modes_to_calculate: dict[str, set[str]] = {dm: set(all_modes) for dm in distance_measures}

    # ------------------------------------------------------------------ #
    # Cache loading pass                                                   #
    # ------------------------------------------------------------------ #
    if not recalculate and results_dir is not None:
        logger.info(
            f"Loading cached distance dictionaries from {results_dir.relative_to(PROJECT_ROOT)}"
        )
        for dm in distance_measures:
            for mode in all_modes:
                file_path = results_dir / f"{dm}_{mode}_distances.dat"
                if not file_path.exists():
                    logger.debug(f"Cache miss (file absent): {file_path.relative_to(PROJECT_ROOT)}")
                    continue

                cell_dict = _load_pickle_with_validation(
                    file_path,
                    validator=_validate_single_mode_distance_dict,
                    context=f"{dm}/{mode} distances",
                )
                if cell_dict is None:
                    logger.debug(f"Cache miss (invalid data): {dm}/{mode}")
                    continue
                logger.info(
                    "Loaded %d cells for %s/%s from cache",
                    len(cell_dict),
                    dm,
                    mode,
                )

                all_distance_dict[dm][mode] = cell_dict
                modes_to_calculate[dm].discard(mode)

        total_pairs = len(distance_measures) * len(all_modes)
        cached_pairs = sum(len(all_modes) - len(modes_to_calculate[dm]) for dm in distance_measures)
        missing_pairs = total_pairs - cached_pairs
        logger.info(
            "Cache: %d/%d (dm, mode) pairs loaded; %d need calculation",
            cached_pairs,
            total_pairs,
            missing_pairs,
        )

        if missing_pairs == 0:
            logger.info("All distance dictionaries loaded from cache")
            return all_distance_dict
    else:
        if recalculate:
            logger.info("recalculate=True: skipping cache, computing all distances")
        else:
            logger.info("No results_dir provided: computing all distances")

    # ------------------------------------------------------------------ #
    # Identify which modes need work across *any* distance measure         #
    # ------------------------------------------------------------------ #
    all_modes_needing_work: set[str] = set()
    for dm_modes in modes_to_calculate.values():
        all_modes_needing_work.update(dm_modes)

    logger.info(
        "Computing distances for modes: %s",
        sorted(all_modes_needing_work),
    )

    # ------------------------------------------------------------------ #
    # Calculation loop — one mode at a time, only missing dms per mode    #
    # ------------------------------------------------------------------ #
    for mode in all_modes_needing_work:
        dms_for_mode = [dm for dm in distance_measures if mode in modes_to_calculate[dm]]
        if not dms_for_mode:
            continue

        # Pre-initialise cell_id entries for this mode
        position_dict = all_positions[mode]
        for dm in dms_for_mode:
            all_distance_dict[dm][mode] = {cell_id: {} for cell_id in position_dict.keys()}

        mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, mode), {})
        _run_calculation_for_mode(
            mode=mode,
            dms_for_mode=dms_for_mode,
            position_dict=position_dict,
            mode_mesh_dict=mode_mesh_dict,
            all_distance_dict=all_distance_dict,
            num_workers=num_workers,
        )

        # Write only the newly computed (dm, mode) files
        if results_dir is not None:
            for dm in dms_for_mode:
                file_path = results_dir / f"{dm}_{mode}_distances.dat"
                with open(file_path, "wb") as f:
                    pickle.dump(all_distance_dict[dm][mode], f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(
                    "Saved %s/%s distances to %s",
                    dm,
                    mode,
                    file_path.relative_to(PROJECT_ROOT),
                )

    return all_distance_dict


def get_ks_test_df(
    distance_measures: list[str],
    packing_modes: list[str],
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    baseline_mode: str = "SLC25A17",
    significance_level: float = 0.05,
    save_dir: Path | None = None,
    recalculate: bool = True,
) -> pd.DataFrame:
    """
    Perform KS test between distance distributions of different packing modes and combine results.

    For each non-baseline mode, iterates over all ``(cell_id, seed)``
    combinations and compares the test-mode distances against the
    baseline-mode distances for the same ``(cell_id, seed)``.  When a
    matching seed is not found in the baseline, the ``(cell_id, seed)`` pair
    is skipped with a warning.

    Parameters
    ----------
    distance_measures
        List of distance measures to compare
    packing_modes
        List of packing modes to compare
    all_distance_dict
        Distance distributions with structure
        ``{distance_measure: {mode: {cell_id: {seed: distances}}}}``
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
        DataFrame containing the KS test results with columns
        ``distance_measure``, ``packing_mode``, ``cell_id``, ``seed``,
        ``ks_stat``, ``p_value``, ``different``, ``similar``
    """
    file_name = "ks_observed_combined_df.parquet"
    if save_dir is not None:
        file_path = save_dir / file_name
        cached_df = _load_parquet_if_exists(
            file_path, recalculate=recalculate, context="KS observed DataFrame"
        )
        if cached_df is not None:
            return cached_df
    record_list = []
    for distance_measure in distance_measures:
        logger.info(f"Calculating KS observed for distance measure: {distance_measure}")

        # Collect all (mode, cell_id, seed) combinations
        all_triples: list[tuple[str, str, str]] = []
        for mode in packing_modes:
            if mode == baseline_mode:
                continue
            for cell_id, seed_dict in all_distance_dict[distance_measure][mode].items():
                for seed in seed_dict:
                    all_triples.append((mode, cell_id, seed))

        for mode, cell_id, seed in tqdm(all_triples, desc=f"KS tests for {distance_measure}"):
            baseline_cell = all_distance_dict[distance_measure][baseline_mode].get(cell_id, None)
            if baseline_cell is None:
                logger.warning(f"Missing baseline cell_id {cell_id} for {mode}, skipping KS test")
                continue
            distances_1 = baseline_cell.get(seed, None)
            if distances_1 is None:
                # Fall back to the first available seed in the baseline
                first_seed = next(iter(baseline_cell))
                distances_1 = baseline_cell[first_seed]
                logger.debug(f"Seed {seed} not in baseline for {cell_id}; using seed {first_seed}")
            distances_2 = all_distance_dict[distance_measure][mode][cell_id][seed]
            if len(distances_1) == 0 or len(distances_2) == 0:
                logger.warning(
                    f"Empty distances for {mode}, {cell_id}, seed {seed}, skipping KS test"
                )
                continue
            ks_result = ks_2samp(distances_1, distances_2)
            ks_stat, p_value = ks_result.statistic, ks_result.pvalue  # type: ignore
            record_list.append(
                {
                    "distance_measure": distance_measure,
                    "packing_mode": mode,
                    "cell_id": cell_id,
                    "seed": seed,
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

    Resampling is done at the **cell_id** level (with replacement) so that
    all seeds belonging to the same cell are included or excluded together,
    preserving within-cell correlation.  When there is only one seed per
    cell_id this is equivalent to resampling individual observations.

    Parameters
    ----------
    ks_test_df
        DataFrame containing per-observation KS test results.  Must have
        columns ``cell_id``, ``distance_measure``, ``packing_mode``, and
        ``similar``.  A ``seed`` column is allowed but not required.
    distance_measures
        List of distance measures to analyze
    packing_modes
        List of packing modes to analyze
    n_bootstrap
        Number of bootstrap samples to generate

    Returns
    -------
    :
        DataFrame containing the bootstrap KS statistics with columns
        ``distance_measure``, ``packing_mode``, ``experiment_number``, and
        ``similar_fraction``
    """
    record_list = []
    cell_ids = ks_test_df["cell_id"].unique()
    n_cells = len(cell_ids)

    for exp_num in tqdm(range(n_bootstrap), desc="Bootstrapping KS tests"):
        sampled_cell_ids = np.random.choice(cell_ids, size=n_cells, replace=True)
        # Concatenate rows for each drawn cell_id, preserving duplicates so that
        # a cell drawn k times contributes k copies of its rows. isin() would
        # silently deduplicate repeated draws, underestimating variance.
        chunks: list[pd.DataFrame] = [
            ks_test_df[ks_test_df["cell_id"] == cid] for cid in sampled_cell_ids
        ]
        sampled_df = pd.concat(chunks, ignore_index=True)
        for distance_measure in distance_measures:
            for packing_mode in packing_modes:
                mode_df = sampled_df.loc[
                    (sampled_df["distance_measure"] == distance_measure)
                    & (sampled_df["packing_mode"] == packing_mode)
                ]
                if mode_df.empty:
                    similar_fraction = np.nan
                else:
                    similar_fraction = mode_df["similar"].mean()
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


def _wasserstein_1d_presorted(u_sorted: np.ndarray, v_sorted: np.ndarray) -> float:
    """
    Compute 1D Wasserstein distance between two pre-sorted distributions.

    Equivalent to ``scipy.stats.wasserstein_distance`` but skips the internal
    sort, which is the dominant cost when the same arrays are reused across
    many pairwise comparisons.
    """
    n_u = len(u_sorted)
    n_v = len(v_sorted)
    # mergesort on two concatenated sorted runs is O(n_u + n_v)
    all_values = np.concatenate((u_sorted, v_sorted))
    all_values.sort(kind="mergesort")

    deltas = np.diff(all_values)
    u_cdf = np.searchsorted(u_sorted, all_values[:-1], side="right") / n_u
    v_cdf = np.searchsorted(v_sorted, all_values[:-1], side="right") / n_v

    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def _compute_emd_chunk(
    pair_indices: list[tuple[int, int]],
    sorted_arrays: list[np.ndarray],
) -> list[tuple[int, int, float]]:
    """Compute EMD for a chunk of index pairs using pre-sorted arrays."""
    results: list[tuple[int, int, float]] = []
    for i, j in pair_indices:
        emd = _wasserstein_1d_presorted(sorted_arrays[i], sorted_arrays[j])
        results.append((i, j, emd))
    return results


def get_distance_distribution_emd_df(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    packing_modes: list[str],
    distance_measures: list[str],
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
    num_workers: int = 1,
) -> pd.DataFrame:
    """
    Calculate pairwise EMD between packing modes for each distance measure.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing distance measures for each packing mode
        {distance_measure: {mode: {cell_id: {seed: distances}}}}
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
    num_workers
        Number of parallel threads to use for EMD computation.
        NumPy/SciPy operations release the GIL, so threading gives real
        speedup here.  Defaults to 1 (sequential).

    Returns
    -------
    :
        DataFrame containing pairwise EMD for each distance measure
    """
    file_name = f"distance_pairwise_emd{suffix}.parquet"
    if results_dir is not None:
        file_path = results_dir / file_name
        cached_df = _load_parquet_if_exists(
            file_path, recalculate=recalculate, context="pairwise EMD"
        )
        if cached_df is not None:
            return cached_df

    # Pre-allocate column lists for efficient DataFrame construction
    col_distance_measure: list[str] = []
    col_mode_1: list[str] = []
    col_mode_2: list[str] = []
    col_cell_id_1: list[str] = []
    col_cell_id_2: list[str] = []
    col_seed_1: list[str] = []
    col_seed_2: list[str] = []
    col_emd: list[float] = []

    for distance_measure in distance_measures:
        logger.info("Calculating EMD for %s", distance_measure)

        # Collect all (mode, cell_id, seed) combinations and pre-sort arrays
        # once so that sorting cost is O(n log n) per array instead of
        # O(n log n) per *pair*.
        metadata: list[tuple[str, str, str]] = []
        sorted_arrays: list[np.ndarray] = []
        for mode in packing_modes:
            for cell_id in all_distance_dict[distance_measure][mode]:
                for seed in all_distance_dict[distance_measure][mode][cell_id]:
                    metadata.append((mode, cell_id, seed))
                    sorted_arrays.append(
                        np.sort(all_distance_dict[distance_measure][mode][cell_id][seed])
                    )

        n = len(metadata)
        n_pairs = n * (n - 1) // 2

        # Build flat list of (i, j) pair indices
        pair_indices: list[tuple[int, int]] = [(i, j) for i in range(n) for j in range(i + 1, n)]

        if num_workers > 1 and n_pairs > 0:
            # Split pairs into chunks for parallel processing.
            # NumPy/SciPy functions release the GIL, so ThreadPoolExecutor
            # provides real parallelism for this workload.
            chunk_size = max(1, n_pairs // (num_workers * 4))
            chunks = [pair_indices[k : k + chunk_size] for k in range(0, n_pairs, chunk_size)]

            emd_results: list[tuple[int, int, float]] = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_compute_emd_chunk, chunk, sorted_arrays): idx
                    for idx, chunk in enumerate(chunks)
                }
                with tqdm(
                    total=len(chunks),
                    desc=f"EMD calculations for {distance_measure}",
                    unit="chunks",
                ) as pbar:
                    for future in as_completed(futures):
                        emd_results.extend(future.result())
                        pbar.update(1)
        else:
            # Sequential path (also used when n_pairs == 0)
            emd_results = []
            for i in tqdm(range(n), desc=f"EMD calculations for {distance_measure}"):
                u = sorted_arrays[i]
                for j in range(i + 1, n):
                    emd_results.append((i, j, _wasserstein_1d_presorted(u, sorted_arrays[j])))

        # Append results using column lists (faster than list-of-dicts)
        for i, j, emd in emd_results:
            m1, c1, s1 = metadata[i]
            m2, c2, s2 = metadata[j]
            col_distance_measure.append(distance_measure)
            col_mode_1.append(m1)
            col_mode_2.append(m2)
            col_cell_id_1.append(c1)
            col_cell_id_2.append(c2)
            col_seed_1.append(s1)
            col_seed_2.append(s2)
            col_emd.append(emd)

    df_emd = pd.DataFrame(
        {
            "distance_measure": col_distance_measure,
            "packing_mode_1": col_mode_1,
            "packing_mode_2": col_mode_2,
            "cell_id_1": col_cell_id_1,
            "cell_id_2": col_cell_id_2,
            "seed_1": col_seed_1,
            "seed_2": col_seed_2,
            "emd": col_emd,
        }
    )
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


def normalize_distance_dictionary(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    mesh_information_dict: dict[str, Any],
    channel_map: dict[str, str],
    normalization: str | None = None,
    pixel_size_in_um: float = PIXEL_SIZE_IN_UM,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """
    Normalize distances using specified normalization method.

    Parameters
    ----------
    all_distance_dict
        Dictionary containing distance measurements by measure and mode
        {distance_measure: {mode: {cell_id: {seed:distances}}}}
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
        {distance_measure: {mode: {cell_id: {seed:normalized_distances}}}}
    """
    for measure, mode_distance_dict in all_distance_dict.items():
        if "scaled" in measure:
            continue
        for mode, distance_dict in mode_distance_dict.items():
            channel_key = channel_map.get(mode, "")
            mode_mesh_dict = mesh_information_dict.get(channel_key, {})
            for cell_id, seed_distance_dict in distance_dict.items():
                if cell_id not in mode_mesh_dict:
                    logger.warning(
                        f"Mesh information not found for cell_id {cell_id} in mode {mode}, "
                        f"using default normalization factor"
                    )
                mesh_info = mode_mesh_dict.get(
                    cell_id,
                    {
                        "intracellular_radius": 1,
                        "cell_diameter": 1,
                    },
                )
                # Compute the per-cell normalization factor once outside the seed loop.
                # Only 'max_distance' varies per seed; all other methods are cell-level constants.
                if normalization == "intracellular_radius":
                    cell_factor: float | None = mesh_info["intracellular_radius"]
                elif normalization == "cell_diameter":
                    cell_factor = mesh_info["cell_diameter"]
                elif normalization == "max_distance":
                    cell_factor = None  # computed per seed below
                else:
                    cell_factor = 1 / pixel_size_in_um

                for seed, distance in seed_distance_dict.items():
                    normalization_factor = distance.max() if cell_factor is None else cell_factor
                    seed_distance_dict[seed] = distance / normalization_factor

    return all_distance_dict


def get_distance_distribution_kde(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
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
        {distance_measure: {mode: {cell_id: {seed: distances}}}}
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
        {cell_id:{mode:gaussian_kde, "available":gaussian_kde}}
    """
    # Set file path for saving/loading KDE results
    save_file_path = None
    if save_dir is not None:
        filename = f"{distance_measure}_distance_distribution_kde{suffix}.dat"
        save_file_path = save_dir / filename

    # Try to load cached KDE data if not recalculating
    if not recalculate and save_file_path is not None and save_file_path.exists():
        requested_modes = set(channel_map.keys())
        kde_dict = _load_pickle_with_validation(
            save_file_path,
            validator=lambda data: _validate_kde_dict(data, requested_modes),
            context="KDE data",
        )
        if kde_dict is not None:
            return kde_dict

    # Initialize the KDE dictionary
    distance_dict = all_distance_dict[distance_measure]
    kde_dict = {}
    for mode, structure_id in channel_map.items():
        mode_mesh_dict = mesh_information_dict.get(structure_id, {})
        mode_distances_dict = distance_dict[mode]
        for cell_id, seed_dict in tqdm(
            mode_distances_dict.items(), total=len(mode_distances_dict), desc=f"KDE for {mode}"
        ):
            # Get the distances for a cell
            # These are already normalized
            if cell_id not in kde_dict:
                kde_dict[cell_id] = {}
            all_cell_id_distances = []
            for _, distances in seed_dict.items():
                distances = filter_invalid_distances(
                    distances, minimum_distance=minimum_distance
                ).astype(np.float32)
                if len(distances) == 0:
                    logger.warning(
                        f"No valid distances found for cell id {cell_id} and mode {mode}"
                    )
                    continue
                all_cell_id_distances.extend(distances)
            # Calculate the KDE for the distances
            kde_dict[cell_id][mode] = gaussian_kde(all_cell_id_distances)

            # Update available distances from mesh information if needed
            if "available" not in kde_dict[cell_id]:
                available_distances = mode_mesh_dict[cell_id][
                    label_tables.GRID_DISTANCE_LABELS[distance_measure]
                ].flatten()

                available_distances = _normalize_and_filter_distances(
                    available_distances,
                    normalization=normalization,
                    mesh_information_dict=mode_mesh_dict,
                    cell_id=cell_id,
                    distance_measure=distance_measure,
                    minimum_distance=minimum_distance,
                )

                kde_dict[cell_id]["available"] = gaussian_kde(available_distances)

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
            df_struct_stats.loc[cell_id, "radius"] / normalization_factor  # type: ignore
        )

    avg_radius = np.mean(scaled_radius_list).item()
    std_radius = np.std(scaled_radius_list).item()

    return avg_radius, std_radius


def log_pairwise_emd_central_tendencies(
    df_emd: pd.DataFrame,
    distance_measures: list[str],
    packing_modes: list[str],
    log_file_path: Path | None = None,
) -> None:
    """Log central tendencies for all pairwise EMD comparisons.

    For every ordered pair of packing modes (including intra-mode), reports
    per-distance-measure stats followed by stats pooled across all distance
    measures.  Each row reports mean, std, median, and 95% CI of the EMD
    distribution.

    Parameters
    ----------
    df_emd
        DataFrame with columns ``distance_measure``, ``packing_mode_1``,
        ``packing_mode_2``, and ``emd``.
    distance_measures
        Distance measures to report.
    packing_modes
        Packing modes forming the matrix rows/columns.
    log_file_path
        Optional file to which the log is also written.
    """
    if log_file_path is not None:
        emd_logger = add_file_handler_to_logger(logger, log_file_path)
    else:
        emd_logger = logger

    emd_logger.info("=== Pairwise EMD central tendencies ===")
    for mode_i in packing_modes:
        for mode_j in packing_modes:
            # Build the mode-pair mask once; reuse across distance measures.
            mode_pair_mask = (
                (df_emd["packing_mode_1"] == mode_i) & (df_emd["packing_mode_2"] == mode_j)
            ) | ((df_emd["packing_mode_1"] == mode_j) & (df_emd["packing_mode_2"] == mode_i))

            # Per-distance-measure stats
            for distance_measure in distance_measures:
                sub_df = df_emd.loc[
                    (df_emd["distance_measure"] == distance_measure) & mode_pair_mask,
                    "emd",
                ]
                if sub_df.empty:
                    continue
                mean = sub_df.mean()
                std = sub_df.std()
                median = sub_df.median()
                lower = sub_df.quantile(0.025)
                upper = sub_df.quantile(0.975)
                emd_logger.info(
                    "  %s vs %s [%s]: %.2f ± %.2f (median: %.2f, 95%% CI: %.2f, %.2f)",
                    mode_i,
                    mode_j,
                    distance_measure,
                    mean,
                    std,
                    median,
                    lower,
                    upper,
                )

            # Pooled stats across all distance measures
            pooled_df = df_emd.loc[mode_pair_mask, "emd"]
            if pooled_df.empty:
                continue
            mean = pooled_df.mean()
            std = pooled_df.std()
            median = pooled_df.median()
            lower = pooled_df.quantile(0.025)
            upper = pooled_df.quantile(0.975)
            emd_logger.info(
                "  %s vs %s [pooled]: %.2f ± %.2f (median: %.2f, 95%% CI: %.2f, %.2f)",
                mode_i,
                mode_j,
                mean,
                std,
                median,
                lower,
                upper,
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
            ks_logger.info(
                f"{packing_mode}: {mean:.2f} ± {std:.2f} "
                f"(median: {median:.2f}, 95% CI: {lower:.2f}, {upper:.2f})"
            )

    remove_file_handler_from_logger(ks_logger, file_path)


def log_central_tendencies_for_distance_distributions(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
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
        {distance_measure: {mode: {cell_id: {seed:distances}}}}
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
            all_distances = np.concatenate(
                [
                    distances
                    for cell_id in mode_dict.keys()
                    for distances in mode_dict[cell_id].values()
                ]
            )
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


def compute_distance_pdfs(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    distance_measures: list[str],
    packing_modes: list[str],
    method: str = "histogram",
    bin_width: float | dict[str, float] = 0.2,
    bandwidth: float | str = "scott",
    distance_limits: dict[str, tuple[float, float]] | None = None,
    minimum_distance: float | None = 0,
    envelope_alpha: float = 0.05,
    n_grid: int = 100,
    results_dir: Path | None = None,
    recalculate: bool = False,
    suffix: str = "",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Compute distance PDFs on a shared r-grid for each measure and mode.

    Produces a unified dictionary structure that can be consumed by
    :func:`~cellpack_analysis.lib.visualization.plot_distance_distributions`
    and :func:`~cellpack_analysis.lib.visualization.plot_pairwise_emd_matrix`
    regardless of whether the underlying estimation used histograms or KDE.

    Parameters
    ----------
    all_distance_dict
        ``{distance_measure: {mode: {cell_id: {seed: distances}}}}``
    distance_measures
        Distance measures to compute PDFs for.
    packing_modes
        Packing modes to compute PDFs for.
    method
        ``"histogram"`` (Laplace-smoothed histogram) or ``"kde"`` (Gaussian KDE).
    bin_width
        Bin width for histogram method (single float or per-measure dict).
        Also used to determine the r-grid spacing for KDE.
    bandwidth
        Bandwidth for ``gaussian_kde`` (only used when *method* = ``"kde"``).
        Accepts ``"scott"``, ``"silverman"``, or a float.
    distance_limits
        Optional ``{measure: (lo, hi)}`` to fix the r-grid range.
    minimum_distance
        Distances below this value are dropped before PDF estimation.
    envelope_alpha
        Significance level for pointwise envelopes (default 0.05 → 95 %).
    n_grid
        Number of grid points when *bin_width* is ``None`` and
        *distance_limits* is not provided.
    results_dir
        Path to results directory, or ``None`` to skip saving.
    recalculate
        If ``True``, recompute PDFs even if they already exist.
    suffix
        Suffix appended to the saved filename.


    Returns
    -------
    :
        ``{distance_measure: {mode: {"xvals": …, "individual_curves": …,
        "mean_pdf": …, "envelope_lo": …, "envelope_hi": …}}}``
    """
    if method not in ("histogram", "kde"):
        raise ValueError(f"method must be 'histogram' or 'kde', got {method!r}")

    if results_dir is not None:
        filename = f"distance_pdfs_{method}{suffix}.dat"
        save_file_path = results_dir / filename
        if not recalculate and save_file_path.exists():
            cached_result = _load_pickle_with_validation(
                save_file_path,
                validator=lambda data: _validate_distance_pdfs_dict(
                    data, distance_measures, packing_modes
                ),
                context="distance PDFs",
            )
            if cached_result is not None:
                return cached_result

    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for dm in distance_measures:
        distance_dict = all_distance_dict[dm]
        measure_bin_width = bin_width[dm] if isinstance(bin_width, dict) else bin_width

        # ── Shared r-grid across all modes ──
        all_arrays: list[np.ndarray] = []
        for mode in packing_modes:
            for seed_dict in distance_dict[mode].values():
                for arr in seed_dict.values():
                    all_arrays.append(np.asarray(arr))

        if distance_limits is not None and dm in distance_limits:
            lo_lim, hi_lim = distance_limits[dm]
            n_bins = int((hi_lim - lo_lim) / measure_bin_width) + 1
            r_grid = np.linspace(lo_lim, hi_lim, n_bins)
        else:
            r_grid = make_r_grid_from_pooled(all_arrays, n=n_grid, bin_width=measure_bin_width)

        bin_edges = np.concatenate(
            [r_grid - measure_bin_width / 2, [r_grid[-1] + measure_bin_width / 2]]
        )

        result[dm] = {}
        for mode in packing_modes:
            curves: list[np.ndarray] = []
            for seed_dict in tqdm(
                distance_dict[mode].values(),
                desc=f"Calculating PDF for mode: {mode}, distance measure: {dm}",
            ):
                for distances in seed_dict.values():
                    filtered = filter_invalid_distances(
                        np.asarray(distances), minimum_distance=minimum_distance
                    )

                    if method == "histogram":
                        raw_counts, _ = np.histogram(filtered, bins=bin_edges)
                        smoothed = raw_counts.astype(float) + 1.0  # Laplace pseudocount
                        pdf = normalize_pdf(r_grid, smoothed)
                    else:  # kde
                        if filtered.size <= 2:
                            pdf = np.zeros_like(r_grid)
                        else:
                            kde = gaussian_kde(filtered, bw_method=bandwidth)
                            pdf = kde.evaluate(r_grid)
                            pdf = normalize_pdf(r_grid, pdf)

                    curves.append(pdf)

            curves_array = np.vstack(curves)
            lo, hi, mu, _ = pointwise_envelope(curves_array, alpha=envelope_alpha)

            result[dm][mode] = {
                "xvals": r_grid,
                "individual_curves": curves_array,
                "mean_pdf": mu,
                "envelope_lo": lo,
                "envelope_hi": hi,
            }

    return result
