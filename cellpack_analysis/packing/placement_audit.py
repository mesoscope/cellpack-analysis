"""Audit whether cellPACK actually placed every punctum a recipe asked for.

Shortfall is silent: the engine gives up after 30 consecutive failed candidates and
still writes the run out as a success.  The true counts are in ``occurences_*.json``
(engine spelling, one 'r').
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: ``occurences_{packing_id}_{condition}_{rule}_{cell_id}.json``
_OCCURRENCE_CELL_ID = re.compile(r"_(\d+)\.json$")


def read_occurrences(occurrences_path: str | Path) -> dict[str, int]:
    """Read one ``occurences_*.json`` into ``{seed: total placed}``.

    Parameters
    ----------
    occurrences_path
        Path to a single engine-written occurrences file. Its shape is
        ``{seed: {ingredient_key: [placed_count, ...]}}``.

    Returns
    -------
    :
        Placed count per seed, summed over every ingredient key and every entry
        within a key.
    """
    with open(occurrences_path) as fh:
        raw: dict[str, dict[str, list[int]]] = json.load(fh)

    return {
        seed: sum(int(n) for counts in ingredients.values() for n in counts)
        for seed, ingredients in raw.items()
    }


def read_requested_count(recipe_path: str | Path, object_name: str) -> int:
    """Return the count a recipe requests for ``object_name``.

    Parameters
    ----------
    recipe_path
        Path to a generated cellPACK recipe.
    object_name
        Key in the recipe's ``objects`` block, e.g. ``"peroxisome"``.

    Returns
    -------
    :
        The requested count.

    Raises
    ------
    KeyError
        If no composition entry places ``object_name``.
    """
    with open(recipe_path) as fh:
        recipe: dict[str, Any] = json.load(fh)

    for entry in recipe.get("composition", {}).values():
        for region_contents in entry.get("regions", {}).values():
            for item in region_contents:
                if isinstance(item, dict) and item.get("object") == object_name:
                    return int(item["count"])

    raise KeyError(f"{recipe_path} has no composition entry placing '{object_name}'")


def audit_placement(
    packing_output_dir: str | Path,
    recipe_dir: str | Path,
    packing_id: str,
) -> pd.DataFrame:
    """Compare placed counts against requested counts for one rule of one condition.

    Parameters
    ----------
    packing_output_dir
        Directory holding the engine's output for one rule, i.e. the ``spheresSST``
        directory that contains the ``occurences_*.json`` files.
    recipe_dir
        Directory holding the generated recipes for that same rule.
    packing_id
        Object name in the recipe, e.g. ``"peroxisome"``.

    Returns
    -------
    :
        One row per (cell_id, seed) with columns ``cell_id``, ``seed``,
        ``requested``, ``placed`` and ``shortfall`` (requested minus placed, so a
        positive value means puncta went unplaced). Empty if no occurrences files
        are present.
    """
    packing_output_dir = Path(packing_output_dir)
    recipe_dir = Path(recipe_dir)

    rows: list[dict[str, Any]] = []
    occurrence_files = sorted(packing_output_dir.glob("occurences_*.json"))
    if not occurrence_files:
        logger.warning("No occurences_*.json found in %s", packing_output_dir)
        return pd.DataFrame(columns=["cell_id", "seed", "requested", "placed", "shortfall"])

    for occurrences_path in occurrence_files:
        match = _OCCURRENCE_CELL_ID.search(occurrences_path.name)
        if match is None:
            logger.warning("Cannot parse a cell id from %s", occurrences_path.name)
            continue
        cell_id = match.group(1)

        recipe_matches = sorted(recipe_dir.glob(f"*_{cell_id}.json"))
        if not recipe_matches:
            logger.warning("No recipe for cell %s in %s", cell_id, recipe_dir)
            continue
        requested = read_requested_count(recipe_matches[0], packing_id)

        for seed, placed in read_occurrences(occurrences_path).items():
            rows.append(
                {
                    "cell_id": cell_id,
                    "seed": seed,
                    "requested": requested,
                    "placed": placed,
                    "shortfall": requested - placed,
                }
            )

    return pd.DataFrame(rows)


def summarize_placement_audit(audit_df: pd.DataFrame) -> dict[str, Any]:
    """Reduce :func:`audit_placement` output to the numbers worth reporting.

    Parameters
    ----------
    audit_df
        Output of :func:`audit_placement`.

    Returns
    -------
    :
        ``n_packings``, ``n_short`` (packings that placed fewer than requested),
        ``fraction_short``, ``worst_shortfall`` and ``mean_placed_fraction``.
    """
    if audit_df.empty:
        return {
            "n_packings": 0,
            "n_short": 0,
            "fraction_short": float("nan"),
            "worst_shortfall": 0,
            "mean_placed_fraction": float("nan"),
        }

    short = audit_df["shortfall"] > 0
    return {
        "n_packings": len(audit_df),
        "n_short": int(short.sum()),
        "fraction_short": float(short.mean()),
        "worst_shortfall": int(audit_df["shortfall"].max()),
        "mean_placed_fraction": float((audit_df["placed"] / audit_df["requested"]).mean()),
    }
