"""Methods calibration numbers: the per-cell envelope test's coverage check on its own nulls.

The ``calibration_summary`` step of ``run_manuscript_workflow`` reads each structure's
calibration cache (``per_cell_envelope_results_calibration.dat``, written by
``run_per_cell_envelope_test`` with ``calibration_holdout_index`` set) and writes
``Methods/Methods_numbers.md`` and ``Methods/calibration_summary.csv``.

In that run one replicate of every cell and null model is the pseudo-observed curve and the
envelope comes from the other ``R`` replicates. Each p-value is then a multiple of ``1/(R+1)``,
so a calibrated test rejects at p < alpha at the rate ``expected_raw`` (4/99 for R = 98,
alpha = 0.05) and, after Benjamini-Hochberg correction across cells, essentially never.

``R`` is not stored in the cache. It is measured from the lattice the p-values lie on and
checked against the config's ``expected.n_replicates``; a wrong ``R`` would shift the
at-floor rate silently.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cellpack_analysis.analysis.workflows.manuscript_config import ManuscriptConfig, check_under
from cellpack_analysis.analysis.workflows.manuscript_sources import PaperSources
from cellpack_analysis.lib.file_io import code_version, engine_commit
from cellpack_analysis.lib.stats import summarize_per_cell_calibration

logger = logging.getLogger(__name__)

CALIBRATION_SUFFIX = "_calibration"
OUTPUT_DIR = "Methods"
SHEET_NAME = "Methods_numbers.md"
CSV_NAME = "calibration_summary.csv"
MAX_ENVELOPE_REPLICATES = 1000
LATTICE_ATOL = 1e-9
SCOPE_SETS = ("all scopes", "per measure", "joint")
TABLE_COLUMNS = [
    "structure",
    "null_mode",
    "scope",
    "n_cells",
    "n_envelope_replicates",
    "holdout_index",
    "raw_rejection",
    "bh_rejection",
    "at_floor",
    "min_p",
    "expected_raw",
    "n_raw_rejected",
    "n_bh_rejected",
]


def measure_envelope_replicates(per_cell_results: dict[str, Any]) -> int:
    """Measure the envelope's replicate count ``R`` from the lattice of the p-values.

    Every Monte Carlo p-value is ``(1 + #{T_sim >= T_obs}) / (R + 1)``, so all of them are
    multiples of ``1/(R+1)``. The result is the smallest such ``R`` over every per-measure and
    joint p-value, which equals the true ``R`` unless every count shares a factor.

    Parameters
    ----------
    per_cell_results
        Return value of ``per_cell_envelope_test`` (as pickled by the pipeline).

    Returns
    -------
    :
        The measured ``R``.

    Raises
    ------
    ValueError
        If a p-value is missing or outside (0, 1], or the p-values do not lie on one lattice
        (cells with different replicate counts).
    """
    summary = per_cell_results["summary"]
    p = np.concatenate(
        [
            np.asarray(summary["per_distance_measure_pvals"], dtype=float).ravel(),
            np.asarray(summary["joint_pvals"], dtype=float).ravel(),
        ]
    )
    if p.size == 0 or not np.all(np.isfinite(p)) or p.min() <= 0 or p.max() > 1:
        raise ValueError("calibration p-values must be finite and in (0, 1]")
    for n in range(2, MAX_ENVELOPE_REPLICATES + 2):
        scaled = p * n
        if np.all(np.abs(scaled - np.rint(scaled)) < LATTICE_ATOL):
            return n - 1
    raise ValueError(
        f"calibration p-values are not multiples of one 1/(R+1) with R <= "
        f"{MAX_ENVELOPE_REPLICATES}; the cells' replicate counts differ"
    )


def check_calibration_result(
    config: ManuscriptConfig, structure: str, result: dict[str, Any]
) -> int:
    """Check a calibration result against the config and return its measured ``R``.

    Parameters
    ----------
    config
        The manuscript config (``expected`` holds alpha, replicates and cell counts).
    structure
        Structure (packing id) the result belongs to.
    result
        Return value of ``per_cell_envelope_test`` run with ``calibration_holdout_index``.

    Returns
    -------
    :
        The measured replicate count ``R`` of each envelope.

    Raises
    ------
    ValueError
        If the result is not a calibration run, or its alpha, cell count, null modes, distance
        measures or measured ``R`` differ from the config's expected values.
    """
    what = f"{structure} calibration result"
    expected = config.expected
    if result.get("calibration_holdout_index") is None:
        raise ValueError(f"{what} is not a hold-one-out calibration run")
    alpha = expected["envelope"]["alpha"]
    if result.get("alpha") != alpha:
        raise ValueError(f"{what}: alpha {result.get('alpha')!r} != expected {alpha!r}")
    n_cells = len(result["cell_ids"])
    if n_cells != expected["n_cells"][structure]:
        raise ValueError(f"{what}: {n_cells} cells != expected {expected['n_cells'][structure]}")
    if sorted(result["distance_measures"]) != sorted(config.distance_measures):
        raise ValueError(
            f"{what}: distance measures {result['distance_measures']} != {config.distance_measures}"
        )
    if not result["null_modes"]:
        raise ValueError(f"{what} has no null models")
    measured = measure_envelope_replicates(result)
    # one replicate is held out of its own envelope
    want = expected["n_replicates"] - 1
    if measured != want:
        raise ValueError(
            f"{what}: measured R = {measured} (p-values are multiples of 1/{measured + 1}), "
            f"expected {want} = {expected['n_replicates']} replicates - 1 held out"
        )
    return measured


def calibration_table(
    config: ManuscriptConfig, structure: str, result: dict[str, Any]
) -> pd.DataFrame:
    """Tidy calibration rows for one structure (one row per null model and scope).

    Parameters
    ----------
    config
        The manuscript config.
    structure
        Structure (packing id) the result belongs to.
    result
        Return value of ``per_cell_envelope_test`` run with ``calibration_holdout_index``.

    Returns
    -------
    :
        ``stats.summarize_per_cell_calibration`` rows plus ``structure``, ``n_cells``,
        ``n_envelope_replicates``, ``holdout_index`` and the integer cell counts
        ``n_raw_rejected`` and ``n_bh_rejected``; columns are ``TABLE_COLUMNS``.
    """
    r = check_calibration_result(config, structure, result)
    table = summarize_per_cell_calibration(result, r, alpha=result["alpha"])
    n_cells = len(result["cell_ids"])
    table["structure"] = structure
    table["n_cells"] = n_cells
    table["n_envelope_replicates"] = r
    table["holdout_index"] = int(result["calibration_holdout_index"])
    for column, count in (("raw_rejection", "n_raw_rejected"), ("bh_rejection", "n_bh_rejected")):
        scaled = table[column].to_numpy(dtype=float) * n_cells
        if not np.allclose(scaled, np.rint(scaled), atol=1e-6):
            raise ValueError(f"{structure}: {column} is not a fraction of {n_cells} cells")
        table[count] = np.rint(scaled).astype(int)
    return table[TABLE_COLUMNS].reset_index(drop=True)


def calibration_tables(config: ManuscriptConfig, sources: PaperSources) -> pd.DataFrame:
    """Calibration rows for every structure of the config, from the calibration caches.

    Parameters
    ----------
    config
        The manuscript config.
    sources
        Cache resolver; ``per_cell_envelope(structure, "_calibration")`` is read.

    Returns
    -------
    :
        The structures' :func:`calibration_table` rows, in config order.
    """
    tables = []
    for structure in config.structures:
        result = sources.per_cell_envelope(structure, CALIBRATION_SUFFIX)
        tables.append(calibration_table(config, structure, result))
    return pd.concat(tables, ignore_index=True)


def _in_set(scope: pd.Series, scope_set: str) -> pd.Series:
    if scope_set == "all scopes":
        return pd.Series(True, index=scope.index)
    return (scope == "joint") if scope_set == "joint" else (scope != "joint")


def _row_label(config: ManuscriptConfig, row: pd.Series) -> str:
    mode = config.row_labels.get(row["null_mode"], row["null_mode"])
    return f"{row['structure']}, {mode}, {row['scope']}"


def range_table(config: ManuscriptConfig, table: pd.DataFrame) -> pd.DataFrame:
    """Min, max and mean of the raw rejection rate over each organelle and scope set.

    Parameters
    ----------
    config
        The manuscript config (row labels for the argmin and argmax).
    table
        Output of :func:`calibration_tables`.

    Returns
    -------
    :
        One row per organelle (each structure, then ``"both"``) and scope set (``SCOPE_SETS``):
        ``organelles``, ``scopes``, ``n_rows``, ``min``, ``min_at``, ``max``, ``max_at``, ``mean``.
    """
    groups = [*dict.fromkeys(table["structure"]), "both"]
    rows = []
    for organelles in groups:
        in_group = table if organelles == "both" else table[table["structure"] == organelles]
        for scope_set in SCOPE_SETS:
            sub = in_group[_in_set(in_group["scope"], scope_set)]
            raw = sub["raw_rejection"]
            rows.append(
                {
                    "organelles": organelles,
                    "scopes": scope_set,
                    "n_rows": len(sub),
                    "min": raw.min(),
                    "min_at": _row_label(config, sub.loc[raw.idxmin()]),
                    "max": raw.max(),
                    "max_at": _row_label(config, sub.loc[raw.idxmax()]),
                    "mean": raw.mean(),
                }
            )
    return pd.DataFrame(rows)


def _g(x: float) -> str:
    """Two significant figures, as the manuscript prints the range."""
    return f"{x:.2g}"


def _md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]


def methods_markdown(
    config: ManuscriptConfig, table: pd.DataFrame, inputs: list[Path] | None = None
) -> str:
    """Render the Methods calibration numbers sheet.

    Parameters
    ----------
    config
        The manuscript config.
    table
        Output of :func:`calibration_tables`.
    inputs
        Files the numbers were read from (listed with their modification times).

    Returns
    -------
    :
        The markdown text of ``Methods/Methods_numbers.md``.
    """
    alpha = config.expected["envelope"]["alpha"]
    r = int(table["n_envelope_replicates"].iloc[0])
    expected_raw = float(table["expected_raw"].iloc[0])
    n_below = round(expected_raw * (r + 1))
    ranges = range_table(config, table)
    both = ranges[ranges["organelles"] == "both"].set_index("scopes")
    n_rows = len(table)
    n_bh_rows = int((table["n_bh_rejected"] == 0).sum())
    n_bh_cells = int(table["n_bh_rejected"].sum())
    n_tests = int(table["n_cells"].sum())
    lo, hi = both.loc["all scopes", "min"], both.loc["all scopes", "max"]
    per_measure = both.loc["per measure"]
    same = (_g(per_measure["min"]), _g(per_measure["max"])) == (_g(lo), _g(hi))
    structures = list(dict.fromkeys(table["structure"]))
    cells = " / ".join(
        f"{int(table[table['structure'] == s]['n_cells'].iloc[0])}" for s in structures
    )
    measures = [s for s in dict.fromkeys(table["scope"]) if s != "joint"]
    holdout = int(table["holdout_index"].iloc[0])
    holdout_note = (
        f"Replicate {holdout} of each cell and null model is the pseudo-observed curve and is "
        f"tested against the other R = {r} replicates "
        f"({config.expected['n_replicates']} per cell and rule, one held out)."
    )
    after = (
        "no cells were rejected in this calibration after correction"
        if n_bh_cells == 0
        else f"{n_bh_cells} of {n_tests} cell-level tests were rejected after correction"
    )
    envelope = config.expected["envelope"]
    lines = [
        f"# Methods (envelope calibration) — numbers for the text ({config.condition})",
        "",
        "Written by `run_manuscript_workflow` (step `calibration_summary`) from the calibration "
        "caches listed below; nothing is typed by hand. Rates are fractions of cells.",
        "",
        f"**Provenance.** Engine pin `{engine_commit()[:8]}`; analysis code `{code_version()}`; "
        f"per-cell envelope test, statistic {envelope['statistic']}, "
        f"{envelope['envelope_type']} envelope, "
        f"alpha = {alpha:g}, run as a hold-one-out calibration. {holdout_note} "
        f"{' / '.join(structures)}: n = {cells} cells (asserted); R is measured from the "
        f"p-value lattice (every p is a multiple of 1/{r + 1}) and asserted equal to the "
        f"configured replicates minus one.",
        "",
    ]
    if inputs:
        rows = [[f"`{_rel(config, p)}`", _modified(p)] for p in inputs]
        lines += _md_table(["Source", "Modified (UTC)"], rows)
    lines += [
        "## Envelope calibration (Methods)",
        "",
        f"Uncorrected rejection rates at p < {alpha:g} were {_g(lo)}\N{EN DASH}{_g(hi)} across "
        f"modeled rules, distance measures, and organelles (expected: {_g(expected_raw)}), and "
        f"{after}.",
        "",
        f"Expected under a calibrated test: raw rate {expected_raw:.4f} "
        f"({n_below}/{r + 1}, p in {{1/{r + 1}, ..., {n_below}/{r + 1}}}), BH rate 0, "
        f"at-floor rate {1 / (r + 1):.4f} (1/{r + 1}).",
        "",
        f"**Scopes.** The range above spans all {n_rows} rows: {len(structures)} organelles "
        f"\N{MULTIPLICATION SIGN} {table['null_mode'].nunique()} rules \N{MULTIPLICATION SIGN} "
        f"{len(measures) + 1} scopes ({', '.join(measures)} and joint). "
        + (
            f"The {int(per_measure['n_rows'])} per-measure rows alone give the same range."
            if same
            else f"The {int(per_measure['n_rows'])} per-measure rows alone give "
            f"{_g(per_measure['min'])}\N{EN DASH}{_g(per_measure['max'])}."
        ),
        "",
        "### Raw rejection rate by scope set",
        "",
    ]
    rows = []
    for rec in ranges.itertuples():
        rows.append(
            [
                str(rec.organelles),
                str(rec.scopes),
                str(rec.n_rows),
                f"{rec.min:.4f}",
                rec.min_at,
                f"{rec.max:.4f}",
                rec.max_at,
                f"{rec.mean:.4f}",
            ]
        )
    lines += _md_table(
        ["Organelles", "Scopes", "Rows", "Min", "Min at", "Max", "Max at", "Mean"], rows
    )
    lines += [
        "### Rejection after BH correction",
        "",
        f"BH q < {alpha:g} rejected {n_bh_cells} of {n_tests} cell-level tests; the rate is 0 in "
        f"{n_bh_rows} of {n_rows} rows. The smallest p-value anywhere is "
        f"{table['min_p'].min():.4f} (floor {1 / (r + 1):.4f}). Because p >= {1 / (r + 1):.4f}, "
        f"any BH rejection at alpha {alpha:g} would reject at least "
        f"{1 / (r + 1) / alpha:.0%} of a row's cells.",
        "",
        "### All rows",
        "",
    ]
    for structure in structures:
        sub = table[table["structure"] == structure]
        lines += [f"**{structure} (n = {int(sub['n_cells'].iloc[0])}):**", ""]
        rows = [
            [
                config.row_labels.get(rec.null_mode, rec.null_mode),
                str(rec.scope),
                f"{rec.raw_rejection:.4f} ({rec.n_raw_rejected})",
                f"{rec.bh_rejection:.4f} ({rec.n_bh_rejected})",
                f"{rec.at_floor:.4f}",
                f"{rec.min_p:.4f}",
            ]
            for rec in sub.itertuples()
        ]
        lines += _md_table(
            [
                "Null model",
                "Scope",
                f"Raw p < {alpha:g} (cells)",
                f"BH q < {alpha:g} (cells)",
                "At floor",
                "Min p",
            ],
            rows,
        )
    return "\n".join(lines).rstrip() + "\n"


def _rel(config: ManuscriptConfig, path: Path) -> str:
    try:
        return path.relative_to(config.project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _modified(path: Path) -> str:
    stamp = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.UTC)
    return stamp.strftime("%Y-%m-%d %H:%M")


def write_calibration_summary(config: ManuscriptConfig, sources: PaperSources) -> list[Path]:
    """Write ``Methods/Methods_numbers.md`` and ``Methods/calibration_summary.csv``.

    This is the ``calibration_summary`` step of ``run_manuscript_workflow``.

    Parameters
    ----------
    config
        The manuscript config.
    sources
        Cache resolver; the calibration caches are read through it.

    Returns
    -------
    :
        The two files written, under ``config.results_dir``.
    """
    with sources.record_inputs() as inputs:
        table = calibration_tables(config, sources)
    out_dir = check_under(config.results_dir / OUTPUT_DIR, config.results_dir, "Methods folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / SHEET_NAME
    csv = out_dir / CSV_NAME
    sheet.write_text(methods_markdown(config, table, inputs), encoding="utf-8")
    table.to_csv(csv, index=False, lineterminator="\n")
    logger.info("Wrote %s and %s (%d rows)", sheet, csv, len(table))
    return [sheet, csv]
