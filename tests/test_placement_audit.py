"""Tests for the occurences_*.json placement audit."""

import json

import pandas as pd
import pytest

from cellpack_analysis.packing.placement_audit import (
    audit_placement,
    read_occurrences,
    read_requested_count,
    summarize_placement_audit,
)

PACKING_ID = "peroxisome"


def _write_occurrences(path, per_seed_counts):
    """Write an engine-shaped occurrences file: {seed: {ingredient: [count]}}."""
    path.write_text(
        json.dumps(
            {
                str(seed): {f"membrane_interior_{PACKING_ID}": [count]}
                for seed, count in per_seed_counts.items()
            }
        )
    )


def _write_recipe(path, count):
    """Write a recipe with the same composition nesting the generator produces."""
    path.write_text(
        json.dumps(
            {
                "objects": {PACKING_ID: {"type": "single_sphere", "radius": 2.37}},
                "composition": {
                    "bounding_area": {"regions": {"interior": ["membrane"]}},
                    "membrane": {
                        "object": "membrane_mesh",
                        "count": 1,
                        "regions": {
                            "interior": ["nucleus", {"object": PACKING_ID, "count": count}]
                        },
                    },
                },
            }
        )
    )


def test_read_occurrences_sums_over_ingredient_keys(tmp_path):
    """A seed's placed count is the total across every ingredient entry."""
    path = tmp_path / "occurences_peroxisome_cond_random_743916.json"
    path.write_text(json.dumps({"0": {"a": [3], "b": [4, 5]}, "1": {"a": [1]}}))

    assert read_occurrences(path) == {"0": 12, "1": 1}


def test_read_requested_count_walks_the_composition(tmp_path):
    """The requested count lives inside composition -> regions -> interior."""
    recipe = tmp_path / "peroxisome_random_743916.json"
    _write_recipe(recipe, count=59)

    assert read_requested_count(recipe, PACKING_ID) == 59


def test_read_requested_count_raises_when_absent(tmp_path):
    """A recipe that never places the object is an error, not a silent zero."""
    recipe = tmp_path / "peroxisome_random_743916.json"
    _write_recipe(recipe, count=59)

    with pytest.raises(KeyError, match="endosome"):
        read_requested_count(recipe, "endosome")


def test_audit_placement_flags_a_shortfall(tmp_path):
    """The audit must surface a packing that placed fewer puncta than requested.

    This is the failure the engine reports as success: ``reject`` gives up after 30
    consecutive candidate failures and nothing downstream notices.
    """
    packings = tmp_path / "spheresSST"
    recipes = tmp_path / "recipes"
    packings.mkdir()
    recipes.mkdir()

    _write_recipe(recipes / "peroxisome_random_743916.json", count=59)
    _write_recipe(recipes / "peroxisome_random_743920.json", count=40)
    _write_occurrences(packings / "occurences_peroxisome_cond_random_743916.json", {0: 59})
    _write_occurrences(packings / "occurences_peroxisome_cond_random_743920.json", {0: 31})

    audit = audit_placement(packings, recipes, PACKING_ID).set_index("cell_id")

    assert audit.loc["743916", "shortfall"] == 0
    assert audit.loc["743920", "shortfall"] == 9

    summary = summarize_placement_audit(audit.reset_index())
    assert summary["n_packings"] == 2
    assert summary["n_short"] == 1
    assert summary["worst_shortfall"] == 9
    assert summary["fraction_short"] == pytest.approx(0.5)


def test_audit_placement_is_empty_without_occurrences(tmp_path):
    """An absent occurrences set returns an empty frame with the right columns."""
    packings = tmp_path / "spheresSST"
    recipes = tmp_path / "recipes"
    packings.mkdir()
    recipes.mkdir()

    audit = audit_placement(packings, recipes, PACKING_ID)

    assert audit.empty
    assert list(audit.columns) == ["cell_id", "seed", "requested", "placed", "shortfall"]
    assert summarize_placement_audit(audit)["n_packings"] == 0


def test_summarize_handles_a_fully_placed_run():
    """Every punctum placed -> zero shortfall, placed fraction 1.0."""
    audit = pd.DataFrame(
        {
            "cell_id": ["743916", "743916"],
            "seed": ["0", "1"],
            "requested": [59, 59],
            "placed": [59, 59],
            "shortfall": [0, 0],
        }
    )

    summary = summarize_placement_audit(audit)

    assert summary["n_short"] == 0
    assert summary["mean_placed_fraction"] == pytest.approx(1.0)
