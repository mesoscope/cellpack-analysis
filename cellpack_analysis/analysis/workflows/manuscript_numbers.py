"""Numbers sheets for the manuscript: the values behind each figure, one markdown sheet each.

The ``figure_numbers`` step of ``run_manuscript_workflow`` calls :func:`write_figure_numbers`.
It reads the pipeline caches through :class:`PaperSources` and never recomputes. Before it
writes, it asserts that the caches agree wherever the paper relies on it: the Fig 3 run and the
validate run share their single-rule arms, the fit and the Fig 4d curves share the observed
curve, each per-cell occupancy curve ends at the same distance in the Fig 3 run, fit and
validation caches, and validate's KS bootstrap is the seeded bootstrap of Fig 3c.

It writes ``<results_dir>/<Sheet>/<Sheet>_numbers.md`` for each sheet in
``config.numbers["sheets"]`` and, with ``config.numbers["write_json"]``,
``<results_dir>/numbers.json``, which holds one record per printed value (see
:func:`parse_records`). The ``Methods`` sheet belongs to ``manuscript_calibration`` and is
skipped here.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cellpack_analysis.analysis.workflows.manuscript_config import ManuscriptConfig
from cellpack_analysis.analysis.workflows.manuscript_sources import (
    PaperSources,
    check_canonical_apical_lambda,
)
from cellpack_analysis.lib import label_tables
from cellpack_analysis.lib import rule_interpolation as ri
from cellpack_analysis.lib import visualization as vis
from cellpack_analysis.lib.file_io import code_version, engine_commit

logger = logging.getLogger(__name__)

JSON_NAME = "numbers.json"
SKIPPED_SHEETS = ("Methods",)  # written by manuscript_calibration
STRUCTURE_SHEETS = ("Fig3", "Fig4", "Fig5")
MEASURES = ("nucleus", "z")  # the sheets are written for these two
REFERENCE_RULE = "random"  # the unbiased rule the text compares against
SHORT = {"peroxisome": "pex", "endosome": "endo"}
EMD_LOG_TOL = 0.0051  # the log prints 2 dp
KS_LOG_TOL = 0.01
STATISTIC_NAMES = {"intdev": "integrated deviation", "supremum": "supremum deviation"}
COUNT_WORDS = {3: "three", 4: "four", 5: "five", 6: "six"}
NUMBER = re.compile(r"[+−-]?\d+(?:[.,]\d+)*")
PARAGRAPH_REF = re.compile(r"¶\d+\s*(?:— )?")
LOG_PREFIX = re.compile(r"^.*\| INFO \| ")
LABEL = re.compile(r"^\*\*(.+?)\*\*")
INTRO = (
    "Values are read from the pipeline caches listed below; nothing is typed by hand. Mean ± SD "
    "over cells (SD with ddof = 1), median in parentheses. Text rows print fractions to 3 dp; "
    "the drop-in tables use the supplement's 2 dp."
)


# ── formatting ───────────────────────────────────────────────────────────────────


def msd(x: Sequence[float] | np.ndarray, nd: int = 2) -> str:
    """Format ``mean ± SD`` (ddof 1) at ``nd`` decimals."""
    a = np.asarray(x, dtype=float)
    return f"{a.mean():.{nd}f} ± {a.std(ddof=1):.{nd}f}"


def msdm(x: Sequence[float] | np.ndarray, nd: int = 2) -> str:
    """Format ``mean ± SD (median)`` at ``nd`` decimals."""
    return f"{msd(x, nd)} ({np.median(np.asarray(x, dtype=float)):.{nd}f})"


def fr(r: tuple[float, float, float], nd: int = 3) -> str:
    """Format a rejection fraction with its + and - sign parts."""
    return f"{r[0]:.{nd}f} (+{r[1]:.{nd}f} / −{r[2]:.{nd}f})"


def order(values: Mapping[str, float], labels: Mapping[str, str], reverse: bool = False) -> str:
    """Join the row labels of ``values`` sorted by value (ascending unless ``reverse``)."""
    sep = " > " if reverse else " < "
    return sep.join(labels[m] for m in sorted(values, key=values.__getitem__, reverse=reverse))


def rank(values: Mapping[str, float], mode: str, reverse: bool = False) -> int:
    """1-based rank of ``mode`` in ``values`` (1 = smallest unless ``reverse``)."""
    return sorted(values, key=values.__getitem__, reverse=reverse).index(mode) + 1


def ladder_name(mode: str, labels: Mapping[str, str]) -> str:
    """Name a ladder rule, ``nucleus_gradient_0pt1`` as ``Nucleus λ = 0.1``."""
    if "_gradient_" not in mode:
        return labels.get(mode, mode)
    family, lam = mode.split("_gradient_")
    return f"{family.capitalize()} λ = {lam.replace('pt', '.')}"


def model_label(name: str, labels: Mapping[str, str], mixed: str) -> str:
    """Row label of an AIC model name (``mixed_rule``, ``null`` or ``single:<mode>``)."""
    special = {"mixed_rule": labels[mixed], "null": "Null"}
    return special.get(name, labels.get(name.removeprefix("single:"), name))


def count_word(n: int) -> str:
    """Spell small counts as the sheets do ("five rules")."""
    return COUNT_WORDS.get(n, str(n))


def _signed(x: float) -> str:
    return str(float(x)).replace("-", "−")


def _rel(config: ManuscriptConfig, path: Path) -> str:
    try:
        return path.relative_to(config.project_root).as_posix()
    except ValueError:
        return str(path)


# ── log parsers ──────────────────────────────────────────────────────────────────


def log_lines(lines: Sequence[str]) -> list[str]:
    """Strip the logger prefix from raw log lines."""
    return [LOG_PREFIX.sub("", ln) for ln in lines]


def parse_dm_log(lines: Sequence[str]) -> dict[tuple[str, str], tuple[float, float]]:
    """Parse a central-tendencies log into ``{(measure, mode): (mean, sd)}``."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    dm = ""
    for ln in log_lines(lines):
        if m := re.match(r"\s*Distance measure: (\w+)", ln):
            dm = m.group(1)
        elif m := re.match(r"\s*(\w+): ([-\d.]+) ± ([\d.]+)", ln):
            out[(dm, m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def parse_emd_log(lines: Sequence[str]) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Parse an EMD central-tendencies log into ``{(mode 1, mode 2, measure): (mean, sd)}``."""
    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for ln in log_lines(lines):
        if m := re.match(r"\s*(\w+) vs (\w+) \[(\w+)\]: ([\d.]+) ± ([\d.]+)", ln):
            out[(m.group(1), m.group(2), m.group(3))] = (float(m.group(4)), float(m.group(5)))
    return out


def recon_mse(lines: Sequence[str]) -> dict[tuple[str, str], float]:
    """Parse the last CV summary of a log into ``{(scope, measure): full-data fit MSE}``.

    Parameters
    ----------
    lines
        Raw lines of ``<structure>_rule_interpolation_cv_summary.log``.

    Returns
    -------
    :
        The full-data reconstruction MSE per scope (``joint``, ``individual``) and measure.

    Raises
    ------
    ValueError
        If the log holds no ``Full reconstruction MSE`` block.
    """
    clean = log_lines(lines)
    starts = [i for i, ln in enumerate(clean) if ln.startswith("Full reconstruction MSE")]
    if not starts:
        raise ValueError("No 'Full reconstruction MSE' block in the CV summary log")
    out: dict[tuple[str, str], float] = {}
    for ln in clean[starts[-1] + 1 :]:
        if m := re.match(r"\s+(\w+)\s+(individual|joint)\s+([\d.]+)\s*$", ln):
            out[(m.group(2), m.group(1))] = float(m.group(3))
    return out


# ── cache checks ─────────────────────────────────────────────────────────────────


def same_cell_emd(
    df: pd.DataFrame, baseline: str, n_cells: int, measures: Sequence[str]
) -> dict[tuple[str, str], np.ndarray]:
    """Per-cell EMD of each mode against the observed cell, keyed ``(mode, measure)``.

    Keeps the rows that pair an observed cell with its own simulations. ``(mode, "pooled")``
    concatenates ``measures``.

    Parameters
    ----------
    df
        Pairwise EMD rows (``distance_measure``, ``packing_mode_1/2``, ``cell_id_1/2``, ``emd``).
    baseline
        The observed mode.
    n_cells
        Expected cells per mode and measure.
    measures
        Measures pooled into ``"pooled"``.

    Returns
    -------
    :
        Arrays sorted by cell id.

    Raises
    ------
    ValueError
        On duplicate pairs, or a mode and measure without exactly ``n_cells`` cells.
    """
    e = df[df["cell_id_1"] == df["cell_id_2"]]
    a = e[(e["packing_mode_1"] == baseline) & (e["packing_mode_2"] != baseline)]
    b = e[(e["packing_mode_2"] == baseline) & (e["packing_mode_1"] != baseline)]
    v = pd.concat(
        [
            a.rename(columns={"packing_mode_2": "mode"}),
            b.rename(columns={"packing_mode_1": "mode"}),
        ],
        ignore_index=True,
    )
    if v.duplicated(["mode", "distance_measure", "cell_id_1"]).any():
        raise ValueError("Duplicate same-cell EMD pairs")
    out: dict[tuple[str, str], np.ndarray] = {}
    for (mode, dm), g in v.groupby(["mode", "distance_measure"]):
        if len(g) != n_cells:
            raise ValueError(f"EMD ({mode}, {dm}): {len(g)} cells, expected {n_cells}")
        out[(str(mode), str(dm))] = g.sort_values("cell_id_1")["emd"].to_numpy(dtype=float)
    for mode in sorted({m for m, _ in out}):
        out[(mode, "pooled")] = np.concatenate([out[(mode, dm)] for dm in measures])
    return out


def rej(summary: Mapping[str, Any], mode: str, dm: str = "joint") -> tuple[float, float, float]:
    """Return (rejected, rejected with + sign, rejected with - sign), BH across cells."""
    base, idx = (
        ("rejection_joint", mode)
        if dm == "joint"
        else ("rejection_per_distance_measure", (mode, dm))
    )
    t, p, n = (float(summary[base + k][idx]) for k in ("", "_positive", "_negative"))
    return t, p, n


def alpha_index(alpha: float, n_replicates: int) -> int:
    """Smallest lattice index ``k`` with ``k / (R + 1) >= alpha``."""
    return math.ceil(round(alpha * (n_replicates + 1), 9))


def expected_uncorrected_rate(alpha: float, n_replicates: int) -> float:
    """Null rate of p < alpha when p is a multiple of ``1 / (R + 1)`` (4/100 for R = 99)."""
    return (alpha_index(alpha, n_replicates) - 1) / (n_replicates + 1)


def bh_floor_fraction(alpha: float, n_replicates: int) -> float:
    """Fraction of cells that must reach the smallest p for BH to reject (0.2 for R = 99)."""
    return 1 / ((n_replicates + 1) * alpha)


def uncorrected_rejections(pvals: np.ndarray, n_replicates: int, alpha: float) -> np.ndarray:
    """Mask of p < alpha, after checking p lies on the ``1 / (R + 1)`` lattice.

    Parameters
    ----------
    pvals
        Per-cell envelope-test p-values.
    n_replicates
        Replicates per cell (``R``).
    alpha
        Significance level.

    Returns
    -------
    :
        Boolean mask.

    Raises
    ------
    ValueError
        If a p-value is off the lattice, so ``R`` is not the test's replicate count.
    """
    p = np.asarray(pvals, dtype=float)
    k = p * (n_replicates + 1)
    if not np.allclose(k, np.rint(k), rtol=0, atol=1e-6):
        raise ValueError(f"p-values are not multiples of 1/(R + 1) for R = {n_replicates}")
    mask = np.rint(k) < alpha_index(alpha, n_replicates)
    if not np.array_equal(mask, p < alpha):
        raise ValueError("Lattice and direct p < alpha counts differ")
    return mask


@dataclass(frozen=True)
class EnvelopeCounts:
    """Cell counts behind one rejection fraction."""

    n: int
    bh: int
    bh_pos: int
    bh_neg: int
    raw: int
    raw_pos: int
    raw_neg: int


def envelope_counts(
    result: Mapping[str, Any], mode: str, dm: str, n_replicates: int
) -> EnvelopeCounts:
    """Count BH and uncorrected rejections (with signs) for one mode and measure.

    Parameters
    ----------
    result
        A ``per_cell_envelope_test`` return.
    mode
        Null mode.
    dm
        Distance measure, or ``"joint"``.
    n_replicates
        Replicates per cell (``R``), checked against the p-value lattice.

    Returns
    -------
    :
        The counts.

    Raises
    ------
    ValueError
        If the BH counts do not reproduce the summary's rejection fractions.
    """
    sm, alpha = result["summary"], float(result["alpha"])
    if dm == "joint":
        p, q, sg = (sm[f"joint_{k}"][mode] for k in ("pvals", "qvals", "signs"))
    else:
        p, q, sg = (
            sm[f"per_distance_measure_{k}"][(mode, dm)] for k in ("pvals", "qvals", "signs")
        )
    p, q, sg = (np.asarray(x, dtype=float) for x in (p, q, sg))
    bh, raw = q < alpha, uncorrected_rejections(p, n_replicates, alpha)
    counts = EnvelopeCounts(
        n=len(p),
        bh=int(bh.sum()),
        bh_pos=int((bh & (sg > 0)).sum()),
        bh_neg=int((bh & (sg < 0)).sum()),
        raw=int(raw.sum()),
        raw_pos=int((raw & (sg > 0)).sum()),
        raw_neg=int((raw & (sg < 0)).sum()),
    )
    got = (counts.bh / counts.n, counts.bh_pos / counts.n, counts.bh_neg / counts.n)
    if not np.allclose(got, rej(sm, mode, dm)):
        raise ValueError(f"BH counts {got} != summary {rej(sm, mode, dm)} for ({mode}, {dm})")
    return counts


def check_envelope_params(result: Mapping[str, Any], expected: Mapping[str, Any], n: int) -> None:
    """Check a per-cell envelope result's cell count and test parameters."""
    if len(result["cell_ids"]) != n:
        raise ValueError(f"Envelope test has {len(result['cell_ids'])} cells, expected {n}")
    got = {k: result.get(k) for k in expected}
    if got != dict(expected):
        raise ValueError(f"Envelope test parameters {got} != expected {dict(expected)}")


def check_ks_counts(df: pd.DataFrame, n_tests: int, what: str) -> None:
    """Check every (measure, mode) of a KS frame holds ``n_tests`` tests."""
    counts = df.groupby(["distance_measure", "packing_mode"]).size()
    bad = counts[counts != n_tests]
    if not bad.empty:
        raise ValueError(f"{what}: test counts {bad.to_dict()} != {n_tests}")


def ks_point(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Fraction of tests with ``similar`` per ``(measure, mode)``."""
    return {
        (str(k[0]), str(k[1])): float(v)
        for k, v in df.groupby(["distance_measure", "packing_mode"])["similar"].mean().items()
    }


def ks_boot(df: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """Bootstrap mean and SD (ddof 1) of ``similar_fraction`` per ``(measure, mode)``."""
    agg = df.groupby(["distance_measure", "packing_mode"])["similar_fraction"].agg(["mean", "std"])
    return {(str(k[0]), str(k[1])): (float(r["mean"]), float(r["std"])) for k, r in agg.iterrows()}


def check_ks_tests_match(fig3: pd.DataFrame, val: pd.DataFrame, modes: Sequence[str]) -> None:
    """Check validate's per-cell KS outcomes equal the Fig 3 run's for ``modes``.

    Raises
    ------
    ValueError
        If any (measure, mode, cell) count of ``similar`` tests differs, or is missing.
    """
    keys = ["distance_measure", "packing_mode", "cell_id"]
    a = fig3[fig3["packing_mode"].isin(modes)].groupby(keys)["similar"].sum()
    b = val[val["packing_mode"].isin(modes)].groupby(keys)["similar"].sum()
    if not a.index.equals(b.index) or not (a == b).all():
        diff = a.index.symmetric_difference(b.index)
        raise ValueError(
            f"Validate's KS tests differ from Fig 3's: {len(diff)} keys unmatched, "
            f"{int((a.reindex(b.index) != b).sum())} counts differ"
        )


def check_ks_bootstrap_seeded(
    fig3: pd.DataFrame, val: pd.DataFrame, modes: Sequence[str], seed: int | None
) -> None:
    """Check validate's KS bootstrap equals Fig 3c's seeded bootstrap for ``modes``.

    Both runs draw cells with ``RandomState(seed)`` over the same sorted cell ids, so the
    single-rule arms agree exactly only when both used the seed.

    Parameters
    ----------
    fig3
        Fig 3c's ``ks_bootstrap`` frame.
    val
        The validation result's ``distance_ks_bootstrap_df``.
    modes
        The single-rule modes both runs hold.
    seed
        ``ks_bootstrap_seed`` of the configs, for the message.

    Raises
    ------
    ValueError
        If an arm is missing or any ``similar_fraction`` differs.
    """
    keys = ["distance_measure", "packing_mode"]
    a = {k: g.sort_values("experiment_number") for k, g in fig3.groupby(keys)}
    b = {k: g.sort_values("experiment_number") for k, g in val.groupby(keys)}
    bad = []
    for key in sorted(k for k in a if k[1] in modes):
        x = a[key]["similar_fraction"].to_numpy(dtype=float)
        y = b[key]["similar_fraction"].to_numpy(dtype=float) if key in b else None
        if y is None or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
            bad.append(key)
    if bad or not any(k[1] in modes for k in a):
        raise ValueError(
            f"Validate's KS bootstrap differs from Fig 3c's for {bad or 'every arm'}: the "
            f"validation result was not produced with ks_bootstrap_seed = {seed}; rerun "
            "--phase validate"
        )


def check_curve_ends_consistent(curves: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    """Check each cell's own-grid occupancy curve ends at one distance in every cache.

    Parameters
    ----------
    curves
        ``{cache name: {mode: {"individual": {cell: {"xvals": ...}}}}}`` for one measure.

    Raises
    ------
    ValueError
        If the cell sets differ between caches or modes, or a cell's last ``xvals`` differ.
    """
    cells: set[str] | None = None
    first = ""
    ends: dict[str, float] = {}
    for name, occ in curves.items():
        for mode, o in occ.items():
            ind = o["individual"]
            here = set(map(str, ind))
            if cells is None:
                cells, first = here, f"{name}/{mode}"
            elif here != cells:
                raise ValueError(f"{name}/{mode}: {len(here ^ cells)} cells differ from {first}")
            for c, d in ind.items():
                end = float(np.asarray(d["xvals"], dtype=float)[-1])
                if ends.setdefault(str(c), end) != end:
                    raise ValueError(f"{name}/{mode} cell {c}: ends at {end}, not {ends[str(c)]}")


# ── loading ──────────────────────────────────────────────────────────────────────


@contextmanager
def _group(sources: PaperSources, inputs: dict[str, list[Path]], name: str) -> Iterator[None]:
    """Add the files the block reads to ``inputs[name]``."""
    with sources.record_inputs() as used:
        yield
    group = inputs.setdefault(name, [])
    group.extend(p for p in used if p not in group)


@dataclass
class StructureData:
    """Everything the Fig 3-5 and S6-S9 sheets print for one structure, checked."""

    structure: str
    baseline: str
    n_cells: int
    rules: list[str]
    mixed: str
    lam: float
    obs_dist: dict[tuple[str, str], tuple[float, float]]
    emd3: dict[tuple[str, str], np.ndarray]
    emdv: dict[tuple[str, str], np.ndarray]
    occ3: dict[tuple[str, str], np.ndarray]
    occv: dict[tuple[str, str], np.ndarray]
    env3: dict[str, Any]
    envv: dict[str, Any]
    counts3: dict[tuple[str, str], EnvelopeCounts]
    n_tests: dict[tuple[str, str], int]
    n_bootstrap: int
    ks_seed: int | None
    ks_seeded: bool
    ks3: dict[tuple[str, str], float]
    ks3_boot: dict[tuple[str, str], tuple[float, float]]
    ksv: dict[tuple[str, str], float]
    ksv_boot: dict[tuple[str, str], tuple[float, float]]
    aic: ri.AICComparisonResult
    cv: ri.CVResult
    recon: dict[tuple[str, str], float]
    occ_obs: dict[str, dict[str, Any]]
    inputs: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def modes(self) -> list[str]:
        """The mixed rule, then the single rules."""
        return [self.mixed, *self.rules]


def _split_modes(
    config: ManuscriptConfig, s: str, dist: Any, vcfg: Any
) -> tuple[list[str], list[str]]:
    """Return (mixed, rules): validate adds exactly one mode to the Fig 3 run's rules."""
    baseline = dist.baseline_mode
    mixed = [m for m in vcfg.packing_modes if m not in dist.packing_modes]
    rules = [m for m in vcfg.packing_modes if m != baseline and m not in mixed]
    if vcfg.baseline_mode != baseline or len(mixed) != 1:
        raise ValueError(f"{s}: validation modes {vcfg.packing_modes} do not add one mixed rule")
    if set(rules) != set(dist.packing_modes) - {baseline} or REFERENCE_RULE not in rules:
        raise ValueError(f"{s}: rules {rules} != Fig 3 modes {dist.packing_modes}")
    missing = [m for m in [*mixed, *rules] if m not in config.row_labels]
    if missing:
        raise ValueError(f"row_labels lacks {missing}")
    return mixed, rules


EMDPair = tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]


def _check_single_rule_arms(
    s: str,
    rules: Sequence[str],
    dms: Sequence[str],
    baseline: str,
    emd_logged: Mapping[tuple[str, str, str], tuple[float, float]],
    distance: EMDPair,
    occupancy: EMDPair,
) -> None:
    """Check the Fig 3 run's EMD equals validate's per rule, and the Fig 3b log."""
    for m in rules:
        for dm in (*dms, "pooled"):
            if abs(distance[0][(m, dm)].mean() - emd_logged[(baseline, m, dm)][0]) >= EMD_LOG_TOL:
                raise ValueError(f"{s} distance EMD ({m}, {dm}) does not match the Fig 3b log")
            for what, (fig3, val) in (("distance", distance), ("occupancy", occupancy)):
                if not np.allclose(fig3[(m, dm)], val[(m, dm)]):
                    raise ValueError(f"{s} {what} EMD ({m}, {dm}): Fig 3 != validate")


def load_structure(
    config: ManuscriptConfig,
    sources: PaperSources,
    structure: str,
    *,
    require_seeded_ks: bool = True,
) -> StructureData:
    """Load and cross-check one structure's caches for the Fig 3-5 and S6-S9 sheets.

    Parameters
    ----------
    config
        The manuscript config.
    sources
        Its sources.
    structure
        Structure (packing id).
    require_seeded_ks
        Raise if validate's KS bootstrap is not Fig 3c's seeded one; ``False`` only warns
        and the sheets say the KS SD is not seed-checked.

    Returns
    -------
    :
        The checked values.

    Raises
    ------
    ValueError
        If any cross-check fails.
    """
    s = structure
    n, reps = int(config.expected["n_cells"][s]), int(config.expected["n_replicates"])
    dms = list(config.distance_measures)
    inputs: dict[str, list[Path]] = {}
    with _group(sources, inputs, "condition"):
        pack = sources.packing_config("condition", s)
        if pack.get("number_of_replicates") != reps:
            raise ValueError(
                f"{s}: number_of_replicates {pack.get('number_of_replicates')} != {reps}"
            )
        lam = check_canonical_apical_lambda(sources.packing_config_path("condition", s))
    with _group(sources, inputs, "distance"):
        dist = sources.distance_config(s)
        baseline = dist.baseline_mode
        obs_dist = parse_dm_log(
            sources.read_log(dist.results_dir, f"{s}_distance_distribution_central_tendencies.log")
        )
        emd_logged = parse_emd_log(
            sources.read_log(dist.results_dir, f"{s}_emd_pairwise_central_tendencies.log")
        )
        ks_logged = parse_dm_log(
            sources.read_log(
                dist.results_dir, f"{s}_pairwise_ks_central_tendencies_ref_{baseline}.log"
            )
        )
        emd3 = same_cell_emd(sources.distance_emd(s), baseline, n, dms)
        env3 = sources.per_cell_envelope(s)
        ks3_df, kb3_df = sources.ks_tests(s), sources.ks_bootstrap(s)
    with _group(sources, inputs, "validation"):
        vcfg = sources.validation_config(s)
        v = sources.validation(s)
    with _group(sources, inputs, "occupancy"):
        occ3 = same_cell_emd(sources.occupancy_emd(s), baseline, n, dms)
        occ_curves = {dm: sources.occupancy(s, dm) for dm in dms}
    with _group(sources, inputs, "fit"):
        fcfg = sources.fit_config("rule_interpolation", s)
        fit_occ = sources.fit_occupancy("rule_interpolation", s)
        cv = sources.cv_result("rule_interpolation", s)
        recon = recon_mse(
            sources.read_log(fcfg.results_dir, f"{s}_rule_interpolation_cv_summary.log")
        )
    with _group(sources, inputs, "validation_occupancy"):
        val_occ = sources.validation_occupancy(s)

    mixed, rules = _split_modes(config, s, dist, vcfg)
    for name in ("distance_per_cell_envelope_test", "distance_ks_df", "aic_result"):
        if getattr(v, name) is None:
            raise ValueError(f"{s}: the validation result has no {name}")
    if v.distance_ks_bootstrap_df is None or v.distance_emd_df is None:
        raise ValueError(f"{s}: the validation result has no KS bootstrap or distance EMD")
    envv = v.distance_per_cell_envelope_test
    assert envv is not None and v.distance_ks_df is not None and v.aic_result is not None
    for env in (env3, envv):
        check_envelope_params(env, config.expected["envelope"], n)
    emdv = same_cell_emd(v.distance_emd_df, baseline, n, dms)
    occv = same_cell_emd(v.emd_df, baseline, n, dms)

    # KS: test counts, Fig 3 == validate per cell, seeded bootstrap, log
    n_tests = n * reps
    check_ks_counts(ks3_df, n_tests, f"{s} Fig 3 KS tests")
    check_ks_counts(v.distance_ks_df, n_tests, f"{s} validate KS tests")
    check_ks_tests_match(ks3_df, v.distance_ks_df, rules)
    n_boot = {int(x) for x in kb3_df.groupby(["distance_measure", "packing_mode"]).size()}
    if n_boot != {dist.n_bootstrap} or vcfg.n_bootstrap != dist.n_bootstrap:
        raise ValueError(f"{s}: bootstrap sizes {n_boot}, configs {dist.n_bootstrap}")
    seed = dist.ks_bootstrap_seed
    if seed is None or vcfg.ks_bootstrap_seed != seed:
        raise ValueError(f"{s}: ks_bootstrap_seed {seed} (Fig 3) / {vcfg.ks_bootstrap_seed}")
    try:
        check_ks_bootstrap_seeded(kb3_df, v.distance_ks_bootstrap_df, rules, seed)
        seeded = True
    except ValueError as e:
        if require_seeded_ks:
            raise ValueError(f"{s}: {e}") from e
        logger.warning("%s: %s (continuing: require_seeded_ks=False)", s, e)
        seeded = False
    ks3_boot = ks_boot(kb3_df)
    for key in ((dm, m) for m in rules for dm in dms):
        if abs(ks3_boot[key][0] - ks_logged[key][0]) > KS_LOG_TOL:
            raise ValueError(f"{s} KS {key}: cache {ks3_boot[key]} vs log {ks_logged[key]}")

    _check_single_rule_arms(s, rules, dms, baseline, emd_logged, (emd3, emdv), (occ3, occv))
    for key in ((m, dm) for m in rules for dm in ("joint", *dms)):
        if rej(env3["summary"], *key) != rej(envv["summary"], *key):
            raise ValueError(f"{s} rejection {key}: Fig 3 != validate")
    for dm in dms:
        a = np.asarray(occ_curves[dm][baseline]["combined"]["occupancy"], dtype=float)
        b = np.asarray(fit_occ[dm][baseline]["combined"]["occupancy"], dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
            raise ValueError(f"{s} {dm}: observed occupancy differs between Fig 4d and the fit")
        check_curve_ends_consistent(
            {"Fig 3 run": occ_curves[dm], "fit": fit_occ[dm], "validation": val_occ[dm]}
        )

    counts3 = {(m, dm): envelope_counts(env3, m, dm, reps) for m in rules for dm in ("joint", *dms)}
    return StructureData(
        structure=s,
        baseline=baseline,
        n_cells=n,
        rules=rules,
        mixed=mixed[0],
        lam=lam,
        obs_dist=obs_dist,
        emd3=emd3,
        emdv=emdv,
        occ3=occ3,
        occv=occv,
        env3=env3,
        envv=envv,
        counts3=counts3,
        n_tests=dict.fromkeys(ks_point(ks3_df), n_tests),
        n_bootstrap=int(dist.n_bootstrap),
        ks_seed=seed,
        ks_seeded=seeded,
        ks3=ks_point(ks3_df),
        ks3_boot=ks3_boot,
        ksv=ks_point(v.distance_ks_df),
        ksv_boot=ks_boot(v.distance_ks_bootstrap_df),
        aic=v.aic_result,
        cv=cv,
        recon=recon,
        occ_obs={dm: occ_curves[dm][baseline]["combined"] for dm in dms},
        inputs=inputs,
    )


# ── sheets ───────────────────────────────────────────────────────────────────────


@dataclass
class Table:
    """A markdown table."""

    header: list[str]
    rows: list[list[str]]


@dataclass
class NumbersSheet:
    """One numbers sheet: ordered headings, paragraphs, bullets and tables."""

    name: str
    title: str
    items: list[tuple[str, Any]] = field(default_factory=list)

    def h2(self, text: str) -> None:
        """Add a ``##`` heading."""
        self.items.append(("h2", text))

    def h3(self, text: str) -> None:
        """Add a ``###`` heading."""
        self.items.append(("h3", text))

    def para(self, text: str) -> None:
        """Add a paragraph."""
        self.items.append(("para", text))

    def bullet(self, text: str) -> None:
        """Add a bullet."""
        self.items.append(("bullet", text))

    def table(self, header: list[str], rows: list[list[str]]) -> None:
        """Add a table; cells may not hold ``|`` or newlines."""
        for row in [header, *rows]:
            if len(row) != len(header) or any("|" in c or "\n" in c for c in row):
                raise ValueError(f"{self.name}: bad table row {row}")
        self.items.append(("table", Table(header, rows)))

    def to_markdown(self) -> str:
        """Render the sheet."""
        out = [f"# {self.title}", ""]
        prev = ""
        for kind, value in self.items:
            if kind == "bullet":
                out.append(f"- {value}")
            else:
                if prev == "bullet":
                    out.append("")
                if kind == "h2":
                    out += [f"## {value}", ""]
                elif kind == "h3":
                    out += [f"### {value}", ""]
                elif kind == "para":
                    out += [value, ""]
                else:
                    t = value
                    out += ["| " + " | ".join(t.header) + " |", "|" + "---|" * len(t.header)]
                    out += ["| " + " | ".join(r) + " |" for r in t.rows]
                    out.append("")
            prev = kind
        return "\n".join(out).rstrip() + "\n"

    def records(self) -> dict[str, str]:
        """Return the records :func:`parse_records` reads back from the sheet."""
        return parse_records(self.name, self.to_markdown())


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _put(records: dict[str, str], key: str, value: str) -> None:
    k, i = key, 1
    while k in records:
        i += 1
        k = f"{key} #{i}"
    records[k] = value


def parse_records(sheet: str, markdown: str) -> dict[str, str]:
    """Read one record per printed value from a numbers sheet.

    Keys are ``sheet|context|label|row|column``. The context is the ``##`` heading without a
    leading paragraph reference (plus `` / `` and the ``###`` heading). A table's label is the
    bold text opening the last non-blank line before it, if any. Its row key is the first
    cell, or the first two cells when first cells repeat. A non-table line with digits is one
    record: row = the line with numbers replaced by ``#``, column ``text``, value = the
    numbers. Repeated keys get `` #2``, `` #3``.

    Parameters
    ----------
    sheet
        Sheet name (the key prefix).
    markdown
        The sheet text.

    Returns
    -------
    :
        ``{key: value}``.
    """
    records: dict[str, str] = {}
    h2 = h3 = prev = ""
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        ctx = h2 + (f" / {h3}" if h3 else "")
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            header, rows = _cells(block[0]), [_cells(r) for r in block[2:]]
            m = LABEL.match(prev)
            label = m.group(1).strip().rstrip(":") if m else ""
            firsts = [r[0] for r in rows]
            two = len(set(firsts)) < len(firsts)
            for r in rows:
                row = " / ".join(r[:2]) if two else r[0]
                for j in range(2 if two else 1, len(header)):
                    _put(records, "|".join([sheet, ctx, label, row, header[j]]), r[j])
            prev = ""
            continue
        if ln.startswith("## "):
            h2, h3, prev = PARAGRAPH_REF.sub("", ln[3:]).strip(), "", ""
        elif ln.startswith("### "):
            h3, prev = ln[4:].strip(), ""
        elif ln.strip():
            text = PARAGRAPH_REF.sub("", ln).strip()
            nums = NUMBER.findall(text)
            if nums:
                _put(
                    records,
                    "|".join([sheet, ctx, "", NUMBER.sub("#", text), "text"]),
                    " ".join(nums),
                )
            prev = ln
        i += 1
    return records


def _start(
    name: str,
    title: str,
    config: ManuscriptConfig,
    commit: str,
    inputs: Sequence[Path],
    cells: str,
) -> NumbersSheet:
    sheet = NumbersSheet(name, f"{title} — numbers for the text ({config.condition})")
    sheet.para(INTRO)
    sheet.para(
        f"**Provenance.** Condition `{config.condition}`; engine commit `{commit[:8]}` "
        f"(recorded, not asserted); analysis code `{code_version()}`; {cells}"
    )
    sheet.para("Sources:")
    for p in dict.fromkeys(inputs):
        sheet.bullet(f"`{_rel(config, p)}`")
    return sheet


def _floors(config: ManuscriptConfig) -> str:
    meta = config.expected["occupancy_meta"]
    return (
        f"distance floors {_signed(meta['minimum_distance_um'])} µm (puncta) / "
        f"{_signed(meta['available_minimum_distance_um'])} µm (available space); occupancy grid "
        f"{meta['num_points']} points, KDE bandwidth factor {meta['bandwidth']:g} (asserted from "
        "every cache's metadata)"
    )


def _cells_text(config: ManuscriptConfig, per_rule: bool = True) -> str:
    n = config.expected["n_cells"]
    text = "n = " + " / ".join(f"{n[s]} {s}" for s in config.structures) + " cells"
    if per_rule:
        text += f", {config.expected['n_replicates']} replicates per cell per rule"
    return text + " (asserted)"


def _structure_header(config: ManuscriptConfig, data: Mapping[str, StructureData]) -> str:
    lam = {d.lam for d in data.values()}
    return (
        f"{_floors(config)}; {_cells_text(config)}; apical λ "
        + " / ".join(f"{x:g}" for x in sorted(lam))
        + " (canonical, asserted)."
    )


def _inputs(data: Mapping[str, StructureData], *groups: str) -> list[Path]:
    return [p for d in data.values() for g in groups for p in d.inputs.get(g, [])]


def _col(s: str, what: str) -> str:
    return f"{SHORT.get(s, s)} {what}"


def _ks_cell(value: float, boot: tuple[float, float]) -> str:
    return f"{value:.2f} ± {boot[1]:.2f}"


def _seed_note(data: Mapping[str, StructureData]) -> str:
    d = next(iter(data.values()))
    if all(x.ks_seeded for x in data.values()):
        return f"seed {d.ks_seed}; single-rule arms equal Fig 3c's bootstrap, asserted"
    return (
        f"**not seed-checked**: validate's bootstrap is not Fig 3c's seed-{d.ks_seed} one, so "
        "the ± may differ from a seeded run at 2 dp"
    )


def section_fig3(
    config: ManuscriptConfig, data: Mapping[str, StructureData], commit: str
) -> NumbersSheet:
    """Fig 3 with Tables S2 and S3 (and the Fig S5 pair counts)."""
    lab, dms, reps = (
        config.row_labels,
        list(config.distance_measures),
        config.expected["n_replicates"],
    )
    sh = _start(
        "Fig3",
        "Figure 3 (with Tables S2 and S3)",
        config,
        commit,
        _inputs(data, "condition", "distance", "validation"),
        _structure_header(config, data),
    )
    d0 = next(iter(data.values()))
    rules = d0.rules

    sh.h2("observed distances (Fig 3a)")
    rows = []
    for dm in dms:
        for s, d in data.items():
            mu, sd = d.obs_dist[(dm, d.baseline)]
            rows.append([f"{s}, {dm}", f"{mu:.2f} ± {sd:.2f} µm"])
    sh.table(["Observed", config.tag], rows)

    sh.h2("distance EMD (Fig 3b; Table S2)")
    sh.para(
        f"Each observed cell against its own {reps} replicates, pooled into one distribution: "
        "n = "
        + " / ".join(str(len(d.emd3[(rules[0], dms[0])])) for d in data.values())
        + " comparisons per rule, one per cell."
    )
    for s, d in data.items():
        sh.para(f"**{s}** (Fig 3b, µm):")
        sh.table(
            ["Rule", *dms, "pooled"],
            [[lab[m], *(msdm(d.emd3[(m, dm)]) for dm in (*dms, "pooled"))] for m in rules],
        )
    for s, d in data.items():
        pooled = {m: d.emd3[(m, "pooled")].mean() for m in rules}
        per = {dm: {m: d.emd3[(m, dm)].mean() for m in rules} for dm in dms}
        sh.bullet(
            f"{s}: {lab[REFERENCE_RULE].lower()} pooled EMD "
            f"{msd(d.emd3[(REFERENCE_RULE, 'pooled')])}; order (pooled): {order(pooled, lab)}; "
            + "; ".join(f"{dm}: {order(per[dm], lab)}" for dm in dms)
            + "."
        )
    sh.para(
        "**Table S2 drop-in** (validate run, adds the mixed rule; single-rule arms identical to "
        "Fig 3b — asserted):"
    )
    sh.table(
        ["Rule", *(_col(s, x) for s in data for x in (*dms, "pooled"))],
        [
            [lab[m], *(msd(d.emdv[(m, x)]) for d in data.values() for x in (*dms, "pooled"))]
            for m in [*rules, d0.mixed]
        ],
    )
    sh.para(
        "Fig S5 (all-pairs EMD matrices): "
        + " / ".join(f"{math.comb(d.n_cells, 2):,} {s}" for s, d in data.items())
        + " cell pairs, n(n − 1)/2."
    )

    sh.h2("KS (Fig 3c; Table S3)")
    sh.para(
        f"Fraction of the {reps} × n per-cell KS tests with p ≥ 0.05 ("
        + " / ".join(f"{d.n_tests[(dms[0], rules[0])]:,} {s}" for s, d in data.items())
        + f" tests per rule); ± is the SD over {d0.n_bootstrap:,} cell-level bootstrap "
        f"resamples (seed {d0.ks_seed})."
    )
    sh.para("**Fig 3c** (Fig 3 run, seeded bootstrap cache):")
    sh.table(
        ["Rule", *(_col(s, dm) for s in data for dm in dms)],
        [
            [
                lab[m],
                *(
                    f"{d.ks3_boot[(dm, m)][0]:.2f} ± {d.ks3_boot[(dm, m)][1]:.2f}"
                    for d in data.values()
                    for dm in dms
                ),
            ]
            for m in rules
        ],
    )
    for s, d in data.items():
        for dm in dms:
            vals = {m: d.ks3[(dm, m)] for m in rules}
            sh.bullet(
                f"{s} {dm}: "
                + ", ".join(f"{lab[m]} {vals[m]:.3f} ± {d.ks3_boot[(dm, m)][1]:.3f}" for m in rules)
                + f"; order: {order(vals, lab, reverse=True)}."
            )
    sh.para(
        "**Table S3 drop-in** (validate run): the exact fraction of the "
        f"{reps}n tests with p ≥ 0.05, ± bootstrap SD ({_seed_note(data)})."
    )
    sh.table(
        ["Rule", *(_col(s, dm) for s in data for dm in dms)],
        [
            [
                lab[m],
                *(
                    _ks_cell(d.ksv[(dm, m)], d.ksv_boot[(dm, m)])
                    for d in data.values()
                    for dm in dms
                ),
            ]
            for m in [*rules, d0.mixed]
        ],
    )

    env = config.expected["envelope"]
    alpha = float(env["alpha"])
    stat = STATISTIC_NAMES.get(env["statistic"], env["statistic"])
    sh.h2("per-cell Monte Carlo envelope test (Fig 3d)")
    sh.para(
        f"Fraction of cells rejected at **BH q < {alpha:g} across cells**, each cell against "
        f"its own {reps} replicates ({env['envelope_type']} envelope, {stat}). In parentheses: "
        "rejected with + sign (observed ECDF above the envelope, i.e. shorter observed distances "
        "— the darker bars) / − sign."
    )
    for s, d in data.items():
        sh.para(f"**{s}** (n = {d.n_cells}):")
        sh.table(
            ["Rule", "joint", *dms],
            [
                [lab[m], *(fr(rej(d.env3["summary"], m, dm)) for dm in ("joint", *dms))]
                for m in rules
            ],
        )
    joint = {(s, m): rej(d.env3["summary"], m)[0] for s, d in data.items() for m in rules}
    for s in data:
        js = {m: joint[(s, m)] for m in rules}
        sh.bullet(
            f"{s} joint: "
            + ", ".join(f"{lab[m]} {js[m]:.3f}" for m in rules)
            + f". Order: {order(js, lab)}."
        )
    lo, hi = min(joint, key=joint.__getitem__), max(joint, key=joint.__getitem__)
    sh.bullet(
        f"Joint rejection, single rules: {joint[lo]:.3f} ({lab[lo[1]]}, {lo[0]}) to "
        f"{joint[hi]:.3f} ({lab[hi[1]]}, {hi[0]})."
    )
    for s, d in data.items():
        t, p, n = rej(d.env3["summary"], REFERENCE_RULE, "nucleus")
        share = f"{p / t:.0%} of rejections are +" if t else "no cell is rejected"
        sh.bullet(
            '"almost all the rejections occurred due to excess distances near the nucleus" '
            f"({lab[REFERENCE_RULE].lower()}, nucleus, {s}): {fr((t, p, n))} → {share}."
        )
    sh.bullet(
        "z rejection, single rules: "
        + "; ".join(
            f"{s}: " + ", ".join(f"{lab[m]} {rej(d.env3['summary'], m, 'z')[0]:.3f}" for m in rules)
            for s, d in data.items()
        )
        + "."
    )

    sh.h3("Counts")
    pct = expected_uncorrected_rate(alpha, reps)
    sh.para(
        f"Cells, not fractions. p is a multiple of 1/(R + 1) = {1 / (reps + 1):g} (R = {reps}, "
        f"checked on every p-value), so uncorrected p < {alpha:g} is expected in {pct:.0%} of "
        f"cells under the null, and BH at q < {alpha:g} can reject only if at least "
        f"{bh_floor_fraction(alpha, reps):.0%} of cells reach p = {1 / (reps + 1):g}."
    )
    for s, d in data.items():
        sh.para(f"**{s}** (n = {d.n_cells}):")
        rows = []
        for m in rules:
            for dm in ("joint", *dms):
                c = d.counts3[(m, dm)]
                rows.append(
                    [
                        lab[m],
                        dm,
                        f"{c.bh} of {c.n}",
                        str(c.bh_pos),
                        str(c.bh_neg),
                        f"{c.raw} of {c.n} ({c.raw / c.n:.0%})",
                        str(c.raw_pos),
                        str(c.raw_neg),
                    ]
                )
        sh.table(
            [
                "Rule",
                "Measure",
                f"BH q < {alpha:g}",
                "BH, + sign",
                "BH, − sign",
                f"p < {alpha:g}",
                "p, + sign",
                "p, − sign",
            ],
            rows,
        )
    for s, d in data.items():
        c, cz = d.counts3[(REFERENCE_RULE, "nucleus")], d.counts3[(REFERENCE_RULE, "z")]
        ref = lab[REFERENCE_RULE]
        sh.bullet(
            f"{s}, {ref}, nucleus: BH q < {alpha:g} rejects {c.bh} of {c.n}, {c.bh_pos} of these "
            f"with + sign; uncorrected p < {alpha:g} in {c.raw / c.n:.0%} ({c.raw} of {c.n}), "
            f"{c.raw_pos} of {c.raw} with + sign ({pct:.0%} expected)."
        )
        sh.bullet(
            f"{s}, {ref}, z: BH q < {alpha:g} rejects {cz.bh} of {cz.n}, {cz.bh_neg} of these "
            "with − sign."
        )
    return sh


def section_fig4(
    config: ManuscriptConfig, data: Mapping[str, StructureData], commit: str
) -> NumbersSheet:
    """Fig 4d: observed occupancy, joint NNLS shares and the panel-title MSE."""
    lab = config.row_labels
    sh = _start(
        "Fig4",
        "Figure 4",
        config,
        commit,
        _inputs(data, "condition", "occupancy", "fit"),
        _structure_header(config, data),
    )
    d0 = next(iter(data.values()))
    sh.h2("observed occupancy (Fig 4d), qualitative check")
    rows = []
    for s, d in data.items():
        o = d.occ_obs
        x, y = np.asarray(o["nucleus"]["xvals"]), np.asarray(o["nucleus"]["occupancy"], float)
        i = int(np.nanargmax(y))
        xz, yz = np.asarray(o["z"]["xvals"]), np.asarray(o["z"]["occupancy"], float)
        ok = np.flatnonzero(np.isfinite(yz))
        rows.append(
            [
                s,
                f"{y[i]:.2f} at {x[i]:.2f} µm",
                f"{y[np.isfinite(y)][-1]:.2f}",
                f"{yz[ok[0]]:.2f} → {yz[ok[-1]]:.2f} (z {xz[ok[0]]:.2f} → {xz[ok[-1]]:.2f} µm)",
            ]
        )
    sh.table(["Observed", "nucleus peak", "nucleus at grid end", "z, grid start → end"], rows)

    sh.h2("joint NNLS relative contributions (Fig 4d)")
    shares = {}
    for s, d in data.items():
        c = d.cv.aggregated_coefficients_joint
        tot = sum(v[0] for v in c.values())
        shares[s] = {m: 100 * c[m][0] / tot for m in d.rules}
    sh.table(
        ["Rule", *(_col(s, config.tag) for s in data)],
        [[lab[m], *(f"{shares[s][m]:.0f} %" for s in data)] for m in d0.rules],
    )
    for s, d in data.items():
        c = d.cv.aggregated_coefficients_joint
        sh.bullet(f"{s}: largest contribution {lab[max(c, key=lambda m: c[m][0])]}.")
    sh.bullet(
        "Rounded shares sum to "
        + " / ".join(f"{sum(round(v) for v in shares[s].values())} % ({s})" for s in data)
        + "."
    )
    figs = [config.template_vars[s]["val_fig"] for s in data]
    figs_text = "/".join([figs[0], *(f.removeprefix("Fig") for f in figs[1:])])
    sh.bullet(
        '"These contributions were stable under cross-validation repeats": fold SDs are in '
        f"`{figs_text}_numbers.md` (c)."
    )
    sh.h2("Fig 4d panel titles (joint, fit-phase NNLS MSE)")
    dms = list(config.distance_measures)
    sh.table(
        ["", *dms], [[s, *(f"{d.recon[('joint', dm)]:.4f}" for dm in dms)] for s, d in data.items()]
    )
    sh.para("The per-rule panels (`occupancy_ratio_per_rule`) carry no numbers.")
    return sh


def section_fig5(
    config: ManuscriptConfig, data: Mapping[str, StructureData], commit: str
) -> NumbersSheet:
    """Fig 5 with Table S4: occupancy EMD and the orthogonal comparisons."""
    lab, dms = config.row_labels, list(config.distance_measures)
    sh = _start(
        "Fig5",
        "Figure 5 (with Table S4)",
        config,
        commit,
        _inputs(data, "condition", "validation", "occupancy", "validation_occupancy"),
        _structure_header(config, data),
    )
    d0 = next(iter(data.values()))
    modes, rules, mixed = d0.modes, d0.rules, d0.mixed
    sh.h2("Fig 5b")
    sh.para(
        "The dashed expected curve uses the sum-normalized shares of Fig 4d (the joint NNLS "
        "table in `Fig4/Fig4_numbers.md`); the panel titles repeat the Fig 4d fit-phase MSE."
    )
    sh.h2("occupancy EMD (Fig 5c; Table S4)")
    sh.para(
        "Each observed cell's occupancy curve against its own replicates' (n = "
        + " / ".join(str(d.n_cells) for d in data.values())
        + " per rule). Each curve ends at the cell's 95th-percentile available distance "
        "(Methods, Occupancy ratio calculations); each cell's end is the same in the Fig 3 run, "
        "fit and validation caches (asserted)."
    )
    for s, d in data.items():
        sh.para(f"**{s}:**")
        sh.table(
            ["Rule", *dms, "pooled"],
            [[lab[m], *(msdm(d.occv[(m, x)]) for x in (*dms, "pooled"))] for m in modes],
        )
    for s, d in data.items():
        parts = []
        for dm in dms:
            means = {m: d.occv[(m, dm)].mean() for m in modes}
            best = min(rules, key=means.__getitem__)
            beat = float((d.occv[(mixed, dm)] < d.occv[(best, dm)]).mean())
            parts.append(
                f"{dm} {msd(d.occv[(mixed, dm)])}, rank {rank(means, mixed)} of {len(modes)}, "
                f"below the best single rule ({lab[best]}) in {beat:.0%} of cells"
            )
        sh.bullet(f"{lab[mixed].lower()}, {s}: " + "; ".join(parts) + ".")
    parts = []
    for s, d in data.items():
        dist = {m: d.emdv[(m, "pooled")].mean() for m in modes}
        ks = {dm: {m: d.ksv[(dm, m)] for m in modes} for dm in dms}
        mc = {m: rej(d.envv["summary"], m)[0] for m in modes}
        parts.append(
            f"{s}: distance EMD (pooled) rank {rank(dist, mixed)} of {len(modes)}; KS rank "
            + " / ".join(f"{rank(ks[dm], mixed, True)}" for dm in dms)
            + f" ({' / '.join(dms)}); joint MC rejection rank {rank(mc, mixed)} "
            f"({mc[mixed]:.3f})"
        )
    vf = [config.template_vars[s]["val_fig"].removeprefix("Fig") for s in data]
    sh.bullet(
        '"Orthogonal comparisons … indicated that the mixed rule provides a closer match '
        'than any single rule (Tables S3 and S4)" → ' + "; ".join(parts) + ". The MC result "
        f"is in {'/'.join(f + 'd' for f in vf)}, not in Tables S3/S4."
    )
    sh.para("**Table S4 drop-in:**")
    sh.table(
        ["Rule", *(_col(s, x) for s in data for x in (*dms, "pooled"))],
        [
            [lab[m], *(msd(d.occv[(m, x)]) for d in data.values() for x in (*dms, "pooled"))]
            for m in [*rules, mixed]
        ],
    )
    return sh


def section_fig_s67(
    config: ManuscriptConfig, d: StructureData, name: str, commit: str
) -> NumbersSheet:
    """S6 / S7: per-cell rejection by measure for every rule, including the mixed rule."""
    lab, dms = config.row_labels, list(config.distance_measures)
    f = name.removeprefix("Fig")
    data = {d.structure: d}
    sh = _start(
        name,
        f"Figure {f} ({d.structure})",
        config,
        commit,
        _inputs(data, "validation"),
        _structure_header(config, data),
    )
    alpha = float(config.expected["envelope"]["alpha"])
    sh.h2(f"{f}a — per-cell rejection by distance measure, {count_word(len(d.modes))} rules")
    sh.para(
        f"BH q < {alpha:g} across cells (n = {d.n_cells}); (+ / −) as Fig 3d. Single-rule arms are "
        "identical to Fig 3d (asserted)."
    )
    sh.table(
        ["Rule", *dms, "joint"],
        [[lab[m], *(fr(rej(d.envv["summary"], m, dm)) for dm in (*dms, "joint"))] for m in d.modes],
    )
    sh.h2(f"{f}b/c — example cells")
    sh.para(
        "Each panel's title carries its cell id and per-cell p (b: median-p cell, c: smallest-p "
        "cell, per rule and measure)."
    )
    return sh


def section_fig_s89(
    config: ManuscriptConfig, d: StructureData, name: str, commit: str
) -> NumbersSheet:
    """S8 / S9: CV MSE, AIC / BIC, NNLS coefficients and the validation summary."""
    lab, dms = config.row_labels, list(config.distance_measures)
    f = name.removeprefix("Fig")
    data = {d.structure: d}
    cv, a = d.cv, d.aic
    sh = _start(
        name,
        f"Figure {f} ({d.structure})",
        config,
        commit,
        _inputs(data, "fit", "validation"),
        _structure_header(config, data),
    )
    sh.h2(f"{f}a — cross-validated MSE ({cv.n_folds}-fold × {cv.n_repeats} repeats)")
    rows = []
    for scope in ("joint", "individual"):
        for dm in dms:
            rows.append(
                [
                    scope,
                    dm,
                    f"{cv.mean_train_mse[scope][dm]:.4f} ± {cv.std_train_mse[scope][dm]:.4f}",
                    f"{cv.mean_test_mse[scope][dm]:.4f} ± {cv.std_test_mse[scope][dm]:.4f}",
                    f"{d.recon[(scope, dm)]:.4f}",
                ]
            )
    sh.table(["Scope", "Measure", "Train", "Test", "Full-data fit"], rows)
    sh.para("The Fig 4d and Fig 5b panel titles print the joint full-data fit.")

    sh.h2(f"{f}b — AIC / BIC")
    sh.para(
        "k = number of non-zero NNLS coefficients (1 for a single rule and for the null); "
        "n = common-grid points with a finite target."
    )
    for scope, scope_dms in (("joint", ["joint"]), ("individual", dms)):
        for dm in scope_dms:
            rows = [
                [
                    model_label(r.model_name, lab, d.mixed),
                    str(r.k),
                    str(r.n),
                    f"{r.rss:.3f}",
                    f"{float(a.delta_aic[scope][dm][r.model_name]):.1f}",
                    f"{float(a.akaike_weights[scope][dm][r.model_name]):.4f}",
                    f"{float(a.delta_bic[scope][dm][r.model_name]):.1f}",
                    f"{float(a.bic_weights[scope][dm][r.model_name]):.4f}",
                ]
                for r in a.comparisons[scope][dm]
            ]
            title = f"Joint fit ({' + '.join(dms)})" if scope == "joint" else f"{dm} only"
            best_aic = model_label(a.best_model_aic[scope][dm], lab, d.mixed)
            best_bic = model_label(a.best_model_bic[scope][dm], lab, d.mixed)
            sh.para(f"**{title}** — best by AIC: {best_aic}; by BIC: {best_bic}")
            sh.table(["Model", "k", "n", "RSS", "ΔAIC", "AIC weight", "ΔBIC", "BIC weight"], rows)

    sh.h2(f"{f}c — NNLS coefficients (raw, mean ± SD across folds) and shares")
    cj = cv.aggregated_coefficients_joint
    tot = sum(v[0] for v in cj.values())
    rows = []
    for m in d.rules:
        row = [lab[m], f"{cj[m][0]:.3f} ± {cj[m][1]:.3f}", f"{100 * cj[m][0] / tot:.0f} %"]
        for dm in dms:
            ci = cv.aggregated_coefficients_individual[dm]
            row.append(f"{ci[m][0]:.3f} ± {ci[m][1]:.3f}")
        rows.append(row)
    sh.table(["Rule", "joint", "joint share", *(f"{dm} only" for dm in dms)], rows)
    sh.para(
        f"Raw joint coefficients sum to {tot:.4f}; the shares (and the mixed-rule recipe) are "
        "sum-normalized."
    )

    sh.h2(f"{f}d — validation summary ({count_word(len(d.modes))} rules)")
    sh.table(
        [
            "Rule",
            *(f"occ. EMD {dm}" for dm in dms),
            *(f"dist. EMD {dm}" for dm in dms),
            *(f"KS {dm}" for dm in dms),
            "MC joint rejection",
        ],
        [
            [
                lab[m],
                *(msd(d.occv[(m, dm)]) for dm in dms),
                *(msd(d.emdv[(m, dm)]) for dm in dms),
                *(_ks_cell(d.ksv[(dm, m)], d.ksv_boot[(dm, m)]) for dm in dms),
                fr(rej(d.envv["summary"], m)),
            ]
            for m in d.modes
        ],
    )
    sh.para(
        "Occupancy EMD: each cell's curve ends at its 95th-percentile available distance. KS: "
        f"fraction of tests with p ≥ 0.05 (point estimate; ± bootstrap SD, {_seed_note(data)}). "
        "MC: per-cell, BH across cells."
    )
    return sh


def section_fig_s10(config: ManuscriptConfig, sources: PaperSources, commit: str) -> NumbersSheet:
    """S10: the decay-length ladder fit (one packing per cell per decay length)."""
    lab, dms = config.row_labels, list(config.distance_measures)
    reps = config.expected["ladder_replicates"]
    inputs: dict[str, list[Path]] = {}
    parts = {}
    for s in config.structures:
        n = int(config.expected["n_cells"][s])
        with _group(sources, inputs, "ladder"):
            fcfg = sources.fit_config("ladder", s)
            pack = sources.packing_config("ladder", s)
            cv = sources.cv_result("ladder", s)
            occ = sources.fit_occupancy("ladder", s)
            rec = recon_mse(
                sources.read_log(fcfg.results_dir, f"{s}_rule_interpolation_cv_summary.log")
            )
        if pack.get("number_of_replicates") != reps:
            raise ValueError(f"{s} ladder: number_of_replicates {pack.get('number_of_replicates')}")
        base, c = fcfg.baseline_mode, cv.aggregated_coefficients_joint
        n_set = {len(occ[dm][m]["individual"]) for dm in dms for m in [base, *c]}
        if n_set != {n}:
            raise ValueError(f"{s} ladder: cell counts {n_set} != {{{n}}}")
        parts[s] = (pack, cv, occ, rec, base)
    pack0 = parts[config.structures[0]][0]
    sh = _start(
        "FigS10",
        "Figure S10 (decay-length ladder)",
        config,
        commit,
        inputs["ladder"],
        f"{_floors(config)}; {_cells_text(config, per_rule=False)}, {reps} packing per cell per "
        f"λ (`{pack0.get('condition')}`, `number_of_replicates` {reps}; not the "
        f"{config.expected['n_replicates']}-replicate set).",
    )
    for s, (_, cv, occ, rec, base) in parts.items():
        c = cv.aggregated_coefficients_joint
        tot = sum(v[0] for v in c.values())

        def rmse(dm: str, m: str, occ: dict[str, Any] = occ, base: str = base) -> float:
            target = vis._mean_occupancy_curve(occ[dm], base)[1]
            return vis._occupancy_rmse(occ[dm], m, target)

        sh.h2(s.capitalize())
        mse = ", ".join(f"{dm} {rec[('joint', dm)]:.4f}" for dm in dms)
        cvtxt = ", ".join(
            f"{dm} {cv.mean_train_mse['joint'][dm]:.4f} / {cv.mean_test_mse['joint'][dm]:.4f}"
            for dm in dms
        )
        sh.para(f"Joint fit-panel titles (full-data fit MSE): **{mse}**. CV train / test: {cvtxt}.")
        sh.table(
            ["Model (λ)", "Joint share", "Raw coefficient"],
            [
                [ladder_name(m, lab), f"{100 * v[0] / tot:.1f} %", f"{v[0]:.4f} ± {v[1]:.4f}"]
                for m, v in c.items()
                if v[0] > 0
            ],
        )
        sh.para("All other ladder models get 0 %.")
        sh.h3("Apical λ")
        rows = [[lab[REFERENCE_RULE], "", *(f"{rmse(dm, REFERENCE_RULE):.3f}" for dm in dms)]]
        rows += [
            [
                ladder_name(m, lab),
                f"{100 * c[m][0] / tot:.1f} ± {100 * c[m][1] / tot:.1f} %",
                *(f"{rmse(dm, m):.3f}" for dm in dms),
            ]
            for m in c
            if m.startswith("apical_gradient_")
        ]
        sh.table(["Rule", "Joint share ± fold SD", *(f"RMSE {dm}" for dm in dms)], rows)
        sh.para(
            "Joint share: NNLS relative contribution in the joint fit, ± SD across folds. RMSE "
            "is a rule's mean occupancy curve against the observed curve, over the bins both "
            "cover (the profile's right-hand panels)."
        )
    sh.h2("Ladder")
    for s, (_, cv, _, _, _) in parts.items():
        ladder: dict[str, list[str]] = {}
        for m in cv.aggregated_coefficients_joint:
            if "_gradient_" in m:
                fam, lam = m.split("_gradient_")
                ladder.setdefault(fam, []).append(lam.replace("pt", "."))
        others = [m for m in cv.aggregated_coefficients_joint if "_gradient_" not in m]
        sh.bullet(
            f"{s}: "
            + "; ".join(f"{fam} λ = {', '.join(sorted(v, key=float))}" for fam, v in ladder.items())
            + f"; plus {', '.join(others)} ({len(cv.aggregated_coefficients_joint)} models)."
        )
    return sh


def section_fig2(config: ManuscriptConfig, sources: PaperSources, commit: str) -> NumbersSheet:
    """Fig 2f, S3 and Table S1: the bio-variation envelope test and distance EMD."""
    inputs: dict[str, list[Path]] = {}
    with _group(sources, inputs, "biovar"):
        cfg = sources.biovar_config()
        res = sources.pairwise_envelope()
        emd = parse_emd_log(
            sources.read_log(
                cfg.results_dir, f"{cfg.packing_id}_emd_pairwise_central_tendencies.log"
            )
        )
    base, modes, dms = cfg.baseline_mode, list(res["packing_modes"]), list(res["distance_measures"])
    others = [m for m in modes if m != base]

    def label(m: str) -> str:
        return label_tables.MODE_LABELS.get(m, m)

    def frac(entry: Mapping[str, Any], nd: int) -> str:
        t, p, n = (float(entry[f"rejection_fraction{k}"]) for k in ("", "_positive", "_negative"))
        return fr((t, p, n), nd)

    sizes = {len(e["pvals"]) for e in res["joint"].values()}
    if len(sizes) != 1:
        raise ValueError(f"Bio-variation test sizes differ: {sizes}")
    n = sizes.pop()
    sh = _start(
        "Fig2",
        "Figure 2 (with Fig S3 and Table S1)",
        config,
        commit,
        inputs["biovar"],
        f"bio-variation run `{_rel(config, cfg.results_dir)}`; pooled envelope test "
        f"({STATISTIC_NAMES.get(res['statistic'], res['statistic'])}, {res['envelope_type']} "
        f"envelope, BH q < {res['alpha']:g}); n = {n} simulations per variation source.",
    )
    sh.h2("Fig 2f — joint rejection against the baseline envelope (Fig S3e)")
    sh.para(
        f"Fraction of each source's {n} simulations rejected by the {label(base).lower()} "
        f"envelope (joint test over {', '.join(dms)}); (+ / −) as Fig 3d."
    )
    sh.table(
        ["Variation source", "joint", "2 dp (S3e)"],
        [
            [label(m), frac(res["joint"][(m, base)], 3), frac(res["joint"][(m, base)], 2)]
            for m in others
        ],
    )
    sh.h2("Fig S3 — pooled envelope matrices")
    sh.para("Rows: tested source; columns: reference envelope; 2 dp, as the matrix annotations.")
    for letter, dm in zip("abcdefgh", [*dms, "joint"], strict=False):
        d = res["joint"] if dm == "joint" else res["per_distance_measure"][dm]
        sh.para(f"**{dm}** (S3{letter}):")
        sh.table(
            ["Tested", *(label(m) for m in modes)],
            [[label(t), *("—" if t == r else frac(d[(t, r)], 2) for r in modes)] for t in modes],
        )
    sh.h2("Table S1 — distance EMD (Fig 2c–e)")
    within, between = math.comb(n, 2), n * n
    sh.para(
        f"Mean ± SD over pairs, from the bio-variation EMD log. Pairs per measure: within a "
        f"source n(n − 1)/2 = {within:,}; to the {label(base).lower()} n² = {between:,} "
        f"(n = {n})."
    )

    def cell(key: tuple[str, str, str]) -> str:
        return f"{emd[key][0]:.2f} ± {emd[key][1]:.2f}" if key in emd else "N/A"

    sh.table(
        [
            "Source",
            *(f"within {dm}" for dm in dms),
            *(f"to {label(base).lower()} {dm}" for dm in dms),
        ],
        [
            [
                label(m),
                *(cell((m, m, dm)) for dm in dms),
                *("N/A" if m == base else cell((base, m, dm)) for dm in dms),
            ]
            for m in modes
        ],
    )
    return sh


# ── step ─────────────────────────────────────────────────────────────────────────


def _per_structure_sheets(config: ManuscriptConfig) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for s in config.structures:
        tv = config.template_vars.get(s, {})
        for kind, key in (("mc", "mc_fig"), ("val", "val_fig")):
            if key in tv:
                out[tv[key]] = (kind, s)
    return out


def write_figure_numbers(
    config: ManuscriptConfig, sources: PaperSources, *, require_seeded_ks: bool = True
) -> list[Path]:
    """Write the numbers sheets (and ``numbers.json``) under ``config.results_dir``.

    Parameters
    ----------
    config
        The manuscript config; ``numbers["sheets"]`` selects the sheets.
    sources
        Its sources.
    require_seeded_ks
        Raise unless validate's KS bootstrap equals Fig 3c's seeded one (the default, used by
        the runner). ``False`` warns and marks the KS SD as not seed-checked.

    Returns
    -------
    :
        The files written.

    Raises
    ------
    ValueError
        If a cross-check fails or a sheet is unknown.
    """
    if set(config.distance_measures) != set(MEASURES):
        raise ValueError(f"The numbers sheets need distance_measures {MEASURES}")
    sheets = [s for s in config.numbers["sheets"] if s not in SKIPPED_SHEETS]
    for skipped in sorted(set(config.numbers["sheets"]) - set(sheets)):
        logger.info("Skipping sheet %s (written by its own step)", skipped)
    per_structure = _per_structure_sheets(config)
    unknown = [
        s
        for s in sheets
        if s not in (*STRUCTURE_SHEETS, "Fig2", "FigS10") and s not in per_structure
    ]
    if unknown:
        raise ValueError(f"No builder for numbers sheets {unknown}")
    data: dict[str, StructureData] = {}
    if any(s in STRUCTURE_SHEETS or s in per_structure for s in sheets):
        data = {
            s: load_structure(config, sources, s, require_seeded_ks=require_seeded_ks)
            for s in config.structures
        }
    commit = engine_commit()
    built: list[NumbersSheet] = []
    for name in sheets:
        if name == "Fig2":
            built.append(section_fig2(config, sources, commit))
        elif name == "Fig3":
            built.append(section_fig3(config, data, commit))
        elif name == "Fig4":
            built.append(section_fig4(config, data, commit))
        elif name == "Fig5":
            built.append(section_fig5(config, data, commit))
        elif name == "FigS10":
            built.append(section_fig_s10(config, sources, commit))
        else:
            kind, s = per_structure[name]
            builder = section_fig_s67 if kind == "mc" else section_fig_s89
            built.append(builder(config, data[s], name, commit))

    written: list[Path] = []
    records: dict[str, str] = {}
    paths: dict[str, str] = {}
    for sheet in built:
        path = config.results_dir / sheet.name / f"{sheet.name}_numbers.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sheet.to_markdown())
        records.update(sheet.records())
        paths[sheet.name] = path.relative_to(config.results_dir).as_posix()
        written.append(path)
        logger.info("Wrote %s", path)
    if config.numbers.get("write_json", True):
        path = config.results_dir / JSON_NAME
        payload = {
            "condition": config.condition,
            "engine_commit": commit,
            "code_version": code_version(),
            "require_seeded_ks": require_seeded_ks,
            "ks_seed_checked": {s: d.ks_seeded for s, d in data.items()},
            "sheets": paths,
            "records": dict(sorted(records.items())),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        written.append(path)
    return written
