"""Virtual-mixture diagnostics for the rule-mixing track (D2 of the mixing handoff).

A *virtual mixture* draws, for every cell and seed, ``round(c_i * N)`` puncta from rule
*i*'s existing single-rule packing of that same cell and seed (largest-remainder
apportionment, sampling without replacement) and unions them. Because occupancy is
linear in placement density, the expected occupancy of this union equals
``sum(c_i * occ_i)`` exactly — so running the standard occupancy estimator on the
virtual mixture previews the achievable packed-vs-expected match (the "ceiling") and
quantifies the seed-noise/steric floor without any engine change or new packing.
"""

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def allocate_mixture_counts(n_total: int, coefficients: dict[str, float]) -> dict[str, int]:
    """
    Apportion ``n_total`` puncta among rules by largest-remainder (Hamilton) allocation.

    Coefficients are normalized internally, so they do not need to sum to one. The
    result is deterministic: remainder ties are broken by descending quota and then
    by rule name.

    Parameters
    ----------
    n_total
        Total number of puncta to allocate.
    coefficients
        Mapping of rule name to (possibly unnormalized) mixture coefficient.

    Returns
    -------
    :
        Mapping of rule name to integer count; counts sum to ``n_total``.
    """
    total = sum(coefficients.values())
    if total <= 0:
        raise ValueError(f"Mixture coefficients must have positive sum, got {total}")

    quotas = {rule: n_total * c / total for rule, c in coefficients.items()}
    counts = {rule: int(np.floor(q)) for rule, q in quotas.items()}
    remaining = n_total - sum(counts.values())

    by_remainder = sorted(
        coefficients.keys(),
        key=lambda rule: (-(quotas[rule] - counts[rule]), -quotas[rule], rule),
    )
    for rule in by_remainder[:remaining]:
        counts[rule] += 1
    return counts


def _rng_for(base_seed: int, cell_id: str, seed: str) -> np.random.Generator:
    """Build a reproducible per-(cell, seed) generator from stable string hashes."""
    digest = hashlib.sha256(f"{cell_id}|{seed}".encode()).digest()
    entropy = [base_seed] + [int.from_bytes(digest[i : i + 4], "little") for i in (0, 4, 8)]
    return np.random.default_rng(entropy)


def build_virtual_mixture_distances(
    all_distance_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    coefficients: dict[str, float],
    distance_measures: list[str],
    base_seed: int = 42,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Build per-cell, per-seed virtual-mixture distance arrays from single-rule packings.

    For every cell and seed shared by all source rules, ``allocate_mixture_counts``
    splits that cell's puncta count among the rules; the allocated number of point
    indices is then drawn without replacement from each rule's packing, and the same
    indices are applied to every distance measure so the joint structure of each
    drawn punctum is preserved.

    Parameters
    ----------
    all_distance_dict
        ``{distance_measure: {mode: {cell_id: {seed: distances}}}}`` — the standard
        analysis-pipeline distance dictionary (each rule's arrays for a given cell
        and seed are index-aligned across distance measures).
    coefficients
        Mapping of source rule name to mixture coefficient (normalized internally).
    distance_measures
        Distance measures to build; all must be present for every source rule.
    base_seed
        Base seed for the reproducible per-(cell, seed) subsampling generators.

    Returns
    -------
    :
        ``{distance_measure: {cell_id: {seed: mixed_distances}}}`` for the virtual
        mixture, with each array the same length as the source arrays.
    """
    rules = sorted(coefficients.keys())
    for dm in distance_measures:
        missing = [rule for rule in rules if rule not in all_distance_dict.get(dm, {})]
        if missing:
            raise ValueError(
                f"Rules {missing} not present in all_distance_dict[{dm!r}] "
                f"(available: {sorted(all_distance_dict.get(dm, {}).keys())})"
            )

    dm0 = distance_measures[0]
    shared_cells = set(all_distance_dict[dm0][rules[0]].keys())
    for dm in distance_measures:
        for rule in rules:
            shared_cells &= set(all_distance_dict[dm][rule].keys())
    if not shared_cells:
        raise ValueError("No cell_ids shared by all source rules and distance measures")

    mixed: dict[str, dict[str, dict[str, np.ndarray]]] = {dm: {} for dm in distance_measures}
    for cell_id in sorted(shared_cells):
        shared_seeds = set(all_distance_dict[dm0][rules[0]][cell_id].keys())
        for dm in distance_measures:
            for rule in rules:
                shared_seeds &= set(all_distance_dict[dm][rule][cell_id].keys())

        for dm in distance_measures:
            mixed[dm][cell_id] = {}

        for seed in sorted(shared_seeds):
            lengths = {
                rule: len(np.asarray(all_distance_dict[dm0][rule][cell_id][seed])) for rule in rules
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"Unequal point counts across rules for cell {cell_id} seed {seed}: "
                    f"{lengths} — conditioned replicates should share the observed count"
                )
            n_total = lengths[rules[0]]

            counts = allocate_mixture_counts(n_total, coefficients)
            rng = _rng_for(base_seed, cell_id, seed)
            indices = {
                rule: rng.choice(lengths[rule], size=counts[rule], replace=False) for rule in rules
            }

            for dm in distance_measures:
                parts = [
                    np.asarray(all_distance_dict[dm][rule][cell_id][seed])[indices[rule]]
                    for rule in rules
                    if counts[rule] > 0
                ]
                mixed[dm][cell_id][seed] = (
                    np.concatenate(parts) if parts else np.array([], dtype=float)
                )
    return mixed


def load_mixture_coefficients(cv_results_path: Path, scope: str = "joint") -> dict[str, float]:
    """
    Load NNLS mixture coefficients from a rule-interpolation CV result pickle.

    Parameters
    ----------
    cv_results_path
        Path to a ``rule_interpolation_cv*.pkl`` written by the fit phase
        (a :class:`cellpack_analysis.lib.rule_interpolation.CVResult`).
    scope
        Which fitted scope to read (e.g. ``"joint"``, ``"nucleus"``, ``"z"``).

    Returns
    -------
    :
        Mapping of rule name to coefficient normalized to sum to one (the mean
        across CV repeats; rules fitted to exactly zero are kept with weight 0).
    """
    with open(cv_results_path, "rb") as f:
        cv_result = pickle.load(f)

    coefficients = {
        rule: float(scopes[scope][0]) for rule, scopes in cv_result.aggregated_coefficients.items()
    }
    total = sum(coefficients.values())
    if total <= 0:
        raise ValueError(f"Coefficients for scope {scope!r} sum to {total} in {cv_results_path}")
    normalized = {rule: c / total for rule, c in coefficients.items()}
    logger.info(
        "Loaded mixture coefficients from %s (scope=%s, raw sum=%.4f): %s",
        cv_results_path,
        scope,
        total,
        {rule: round(c, 4) for rule, c in normalized.items()},
    )
    return normalized


def expected_mixture_occupancy(
    occupancy_dict: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    coefficients: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Compute the per-cell expected mixture occupancy ``sum(c_i * occ_i)``.

    Uses each rule's per-cell occupancy on the shared x grid
    (``individual[cell]["occupancy_common"]``). NaNs (outside a cell's
    available-space support) propagate rather than being treated as zero.

    Parameters
    ----------
    occupancy_dict
        One distance measure's occupancy dictionary as returned by
        :func:`cellpack_analysis.lib.occupancy.get_kde_occupancy_dict` —
        ``{mode: {"individual": {cell_id: {...}}, "combined": {...}}}``.
    coefficients
        Mapping of rule name to mixture coefficient (assumed normalized).

    Returns
    -------
    :
        ``{cell_id: expected_occupancy_curve}`` for every cell present in all
        source rules.
    """
    rules = sorted(coefficients.keys())
    missing = [rule for rule in rules if rule not in occupancy_dict]
    if missing:
        raise ValueError(f"Rules {missing} not present in occupancy_dict")

    shared_cells: set[str] = set(occupancy_dict[rules[0]]["individual"].keys())
    for rule in rules[1:]:
        shared_cells &= set(occupancy_dict[rule]["individual"].keys())

    expected: dict[str, np.ndarray] = {}
    for cell_id in sorted(shared_cells):
        expected[cell_id] = np.sum(
            [
                coefficients[rule]
                * np.asarray(occupancy_dict[rule]["individual"][cell_id]["occupancy_common"])
                for rule in rules
            ],
            axis=0,
        )
    return expected
