"""Tests for the seedable cell-level KS bootstrap."""

import numpy as np
import pandas as pd
import pytest

from cellpack_analysis.lib.distance import bootstrap_ks_tests

DMS = ["nucleus", "z"]
MODES = ["random", "nucleus_gradient", "apical_gradient"]


def _make_ks_df(n_cells: int = 40, n_seeds: int = 99, rng_seed: int = 7) -> pd.DataFrame:
    """Synthetic KS frame in the real layout, one cell missing from one mode."""
    rng = np.random.default_rng(rng_seed)
    cell_ids = [f"{1000 + 7 * i}" for i in range(n_cells)]
    records = []
    for dm in DMS:
        for mode in MODES:
            for cid in cell_ids:
                if (dm, mode, cid) == ("z", "apical_gradient", cell_ids[3]):
                    continue
                p_similar = rng.uniform(0.1, 0.9)
                similar = rng.random(n_seeds) < p_similar
                for seed, sim in enumerate(similar):
                    records.append(
                        {
                            "distance_measure": dm,
                            "packing_mode": mode,
                            "cell_id": cid,
                            "seed": str(seed),
                            "similar": bool(sim),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _reference_ks_bootstrap(
    ks: pd.DataFrame, dms: list[str], modes: list[str], n: int, seed: int
) -> pd.DataFrame:
    """Count-matrix bootstrap (private copy of the manuscript replot algorithm)."""
    cell_ids = ks["cell_id"].unique()
    g = ks.groupby(["distance_measure", "packing_mode", "cell_id"])["similar"]
    keys = [(dm, m) for dm in dms for m in modes]
    sums = (
        g.sum().unstack("cell_id").reindex(index=keys, columns=cell_ids).fillna(0).to_numpy(float)
    )
    counts = (
        g.size().unstack("cell_id").reindex(index=keys, columns=cell_ids).fillna(0).to_numpy(float)
    )
    pos = pd.Index(cell_ids)
    np.random.seed(seed)
    rows = []
    for i in range(n):
        draw = np.random.choice(cell_ids, size=len(cell_ids), replace=True)
        k = np.bincount(pos.get_indexer(draw), minlength=len(cell_ids))
        s, t = sums @ k, counts @ k
        rows += [
            {
                "distance_measure": dm,
                "packing_mode": m,
                "experiment_number": i,
                "similar_fraction": s[j] / t[j] if t[j] else np.nan,
            }
            for j, (dm, m) in enumerate(keys)
        ]
    return pd.DataFrame.from_records(rows)


@pytest.fixture(scope="module")
def ks_df() -> pd.DataFrame:
    return _make_ks_df()


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return _make_ks_df(n_cells=12, n_seeds=5, rng_seed=3)


def test_seed_equals_global_seed(small_df: pd.DataFrame) -> None:
    seeded = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20, seed=5)
    np.random.seed(5)
    global_stream = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20)
    pd.testing.assert_frame_equal(seeded, global_stream, check_exact=True)


def test_seeded_result_is_row_order_invariant(small_df: pd.DataFrame) -> None:
    shuffled = small_df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    a = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20, seed=5)
    b = bootstrap_ks_tests(shuffled, DMS, MODES, n_bootstrap=20, seed=5)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_seed_does_not_touch_global_stream(small_df: pd.DataFrame) -> None:
    np.random.seed(11)
    expected = np.random.random(3)
    np.random.seed(11)
    bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=5, seed=5)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_seed_none_uses_global_stream(small_df: pd.DataFrame) -> None:
    np.random.seed(5)
    a = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20)
    np.random.seed(5)
    b = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20)
    np.random.seed(6)
    c = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert not a["similar_fraction"].equals(c["similar_fraction"])


def test_different_seeds_differ(small_df: pd.DataFrame) -> None:
    a = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20, seed=1)
    b = bootstrap_ks_tests(small_df, DMS, MODES, n_bootstrap=20, seed=2)
    assert not a["similar_fraction"].equals(b["similar_fraction"])


def test_matches_replot_count_matrix_bootstrap_at_full_n(ks_df: pd.DataFrame) -> None:
    """The count-matrix algorithm behind the published Fig 3c equals the lib bootstrap."""
    assert ks_df["cell_id"].nunique() == 40 and ks_df["seed"].nunique() == 99
    assert list(ks_df["cell_id"].unique()) == sorted(ks_df["cell_id"].unique())
    lib = bootstrap_ks_tests(ks_df, DMS, MODES, n_bootstrap=1000, seed=0)
    ref = _reference_ks_bootstrap(ks_df, DMS, MODES, n=1000, seed=0)
    assert len(lib) == 1000 * len(DMS) * len(MODES)
    pd.testing.assert_frame_equal(lib[ref.columns], ref, check_exact=True)


def test_matches_reference_when_a_mode_is_missing_in_some_draws(small_df: pd.DataFrame) -> None:
    """A mode held by one cell gives NaN in draws that miss it, in both algorithms."""
    keep = small_df["packing_mode"] != "apical_gradient"
    only_one = (small_df["packing_mode"] == "apical_gradient") & (
        small_df["cell_id"] == small_df["cell_id"].iloc[0]
    )
    sparse = small_df[keep | only_one].reset_index(drop=True)
    lib = bootstrap_ks_tests(sparse, DMS, MODES, n_bootstrap=200, seed=0)
    ref = _reference_ks_bootstrap(sparse, DMS, MODES, n=200, seed=0)
    assert 0 < ref["similar_fraction"].isna().sum() < len(ref)
    pd.testing.assert_frame_equal(lib[ref.columns], ref, check_exact=True)
