"""Tests for the distance-PDF cache: sidecar-stamped persistence and the strict loader."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cellpack_analysis.lib import distance
from cellpack_analysis.lib.distance import compute_distance_pdfs, load_distance_pdfs

_LOGGER_NAME = "cellpack_analysis.lib.distance"
DMS = ["nucleus", "z"]
MODES = ["random", "apical_gradient"]
LEAF_KEYS = ("xvals", "individual_curves", "mean_pdf", "envelope_lo", "envelope_hi")
LIMITS = {"nucleus": (0.0, 1.0), "z": (0.0, 2.0), "pairwise": (0.0, 25.0)}


def _make_distances(rng_seed: int = 3) -> dict[str, Any]:
    """``{dm: {mode: {cell: {seed: distances}}}}`` with 3 cells x 4 seeds of 40 points."""
    rng = np.random.default_rng(rng_seed)
    return {
        dm: {
            mode: {
                f"cell{c}": {str(s): rng.uniform(0.0, 1.0, 40) for s in range(4)} for c in range(3)
            }
            for mode in MODES
        }
        for dm in DMS
    }


def _kwargs(**over: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "distance_measures": DMS,
        "packing_modes": MODES,
        "method": "histogram",
        "bin_width": 0.1,
        "distance_limits": LIMITS,
        "minimum_distance": None,
        "envelope_alpha": 0.05,
    }
    kw.update(over)
    return kw


def _assert_same(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert list(a) == list(b)
    for dm in a:
        assert list(a[dm]) == list(b[dm])
        for mode in a[dm]:
            assert set(a[dm][mode]) == set(LEAF_KEYS)
            for key in LEAF_KEYS:
                assert np.array_equal(a[dm][mode][key], b[dm][mode][key])


def _expected_meta(kw: dict[str, Any]) -> dict[str, Any]:
    """The meta a caller would expect for these kwargs, with tuple limits as passed in."""
    return {
        "method": kw["method"],
        "bin_width": kw["bin_width"],
        "bandwidth": kw.get("bandwidth", "scott"),
        "distance_limits": {dm: LIMITS[dm] for dm in kw["distance_measures"]},
        "minimum_distance": kw["minimum_distance"],
        "envelope_alpha": kw["envelope_alpha"],
        "n_grid": 100,
        "distance_measures": kw["distance_measures"],
        "packing_modes": kw["packing_modes"],
    }


@pytest.fixture()
def dist_dict() -> dict[str, Any]:
    return _make_distances()


@pytest.mark.parametrize("method", ["histogram", "kde"])
def test_round_trip_is_exact(tmp_path: Path, dist_dict: dict[str, Any], method: str) -> None:
    kw = _kwargs(method=method, bandwidth=0.2)
    computed = compute_distance_pdfs(dist_dict, results_dir=tmp_path, **kw)
    assert (tmp_path / f"distance_pdfs_{method}.dat").exists()
    loaded = load_distance_pdfs(tmp_path, method)
    _assert_same(computed, loaded)
    # the strict loader also accepts the parameters it was built with
    _assert_same(computed, load_distance_pdfs(tmp_path, method, expected_meta=_expected_meta(kw)))


def test_sidecar_records_parameters_after_pickle(tmp_path: Path, dist_dict: dict[str, Any]) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, suffix="_s", **_kwargs())
    assert (tmp_path / "distance_pdfs_histogram_s.dat").exists()
    meta = json.loads((tmp_path / "distance_pdfs_histogram_s.meta.json").read_text())
    assert meta == {
        "method": "histogram",
        "bin_width": 0.1,
        "bandwidth": "scott",
        "distance_limits": {"nucleus": [0.0, 1.0], "z": [0.0, 2.0]},  # measures used only
        "minimum_distance": None,
        "envelope_alpha": 0.05,
        "n_grid": 100,
        "distance_measures": ["nucleus", "z"],
        "packing_modes": ["apical_gradient", "random"],  # stored sorted
    }


def test_per_measure_bin_width_dict_is_restricted_to_measures_used(
    tmp_path: Path, dist_dict: dict[str, Any]
) -> None:
    kw = _kwargs(bin_width={"nucleus": 0.1, "z": 0.2, "pairwise": 1.0})
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **kw)
    meta = json.loads((tmp_path / "distance_pdfs_histogram.meta.json").read_text())
    assert meta["bin_width"] == {"nucleus": 0.1, "z": 0.2}


def test_no_write_when_results_dir_is_none(
    tmp_path: Path, dist_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("cache write reached with results_dir=None")

    monkeypatch.setattr(distance, "_write_distance_pdf_cache_meta", _boom)
    monkeypatch.setattr(distance.pickle, "dump", _boom)
    monkeypatch.chdir(tmp_path)
    result = compute_distance_pdfs(dist_dict, **_kwargs())
    assert set(result) == set(DMS)
    assert list(tmp_path.iterdir()) == []


def test_cache_hit_returns_cached_without_recomputing(
    tmp_path: Path, dist_dict: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    first = compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    other = _make_distances(rng_seed=99)  # different data, same parameters
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        second = compute_distance_pdfs(other, results_dir=tmp_path, **_kwargs())
    _assert_same(first, second)
    assert not caplog.records


def test_meta_mismatch_recomputes_with_warning(
    tmp_path: Path, dist_dict: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    first = compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        second = compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs(bin_width=0.05))
    assert any("bin_width" in r.getMessage() for r in caplog.records)
    assert second["nucleus"]["random"]["xvals"].size > first["nucleus"]["random"]["xvals"].size
    # the cache now carries the new parameters, so the strict loader accepts them
    _assert_same(
        second, load_distance_pdfs(tmp_path, "histogram", expected_meta={"bin_width": 0.05})
    )
    with pytest.raises(ValueError, match="bin_width"):
        load_distance_pdfs(tmp_path, "histogram", expected_meta={"bin_width": 0.1})


def test_missing_sidecar_recomputes_with_warning(
    tmp_path: Path, dist_dict: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    meta_path = tmp_path / "distance_pdfs_histogram.meta.json"
    meta_path.unlink()
    other = _make_distances(rng_seed=99)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        second = compute_distance_pdfs(other, results_dir=tmp_path, **_kwargs())
    assert any("sidecar" in r.getMessage() for r in caplog.records)
    assert meta_path.exists()
    _assert_same(second, load_distance_pdfs(tmp_path, "histogram"))
    # recomputed from the new data, not served from the stampless pickle
    fresh = compute_distance_pdfs(other, **_kwargs())
    _assert_same(second, fresh)


def test_corrupt_pickle_recomputes_with_warning(
    tmp_path: Path, dist_dict: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    first = compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    (tmp_path / "distance_pdfs_histogram.dat").write_bytes(b"not a pickle")
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        second = compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    assert any("Recomputing" in r.getMessage() for r in caplog.records)
    _assert_same(first, second)
    _assert_same(first, load_distance_pdfs(tmp_path, "histogram"))


def test_recalculate_overwrites_cache(tmp_path: Path, dist_dict: dict[str, Any]) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    other = _make_distances(rng_seed=99)
    redone = compute_distance_pdfs(other, results_dir=tmp_path, recalculate=True, **_kwargs())
    _assert_same(redone, load_distance_pdfs(tmp_path, "histogram"))
    _assert_same(redone, compute_distance_pdfs(other, **_kwargs()))


def test_results_dir_is_created(tmp_path: Path, dist_dict: dict[str, Any]) -> None:
    target = tmp_path / "a" / "b"
    compute_distance_pdfs(dist_dict, results_dir=target, **_kwargs())
    assert (target / "distance_pdfs_histogram.meta.json").exists()


def test_failed_pickle_write_leaves_no_stale_sidecar(
    tmp_path: Path, dist_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    meta_path = tmp_path / "distance_pdfs_histogram.meta.json"
    assert meta_path.exists()

    def _fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(distance.pickle, "dump", _fail)
    with pytest.raises(OSError, match="disk full"):
        compute_distance_pdfs(
            dist_dict, results_dir=tmp_path, recalculate=True, **_kwargs(bin_width=0.05)
        )
    assert not meta_path.exists()  # a stampless pickle fails safe
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_distance_pdfs(tmp_path, "histogram")


# ── strict loader ──────────────────────────────────────────────────────────────────


def test_loader_raises_on_missing_pickle(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_distance_pdfs(tmp_path, "histogram")


def test_loader_raises_on_missing_sidecar(tmp_path: Path, dist_dict: dict[str, Any]) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    (tmp_path / "distance_pdfs_histogram.meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_distance_pdfs(tmp_path, "histogram")


def test_loader_raises_on_mismatched_sidecar_and_names_the_key(
    tmp_path: Path, dist_dict: dict[str, Any]
) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    for key, value in [
        ("method", "kde"),
        ("bandwidth", 0.2),
        ("minimum_distance", 0.0),
        ("envelope_alpha", 0.1),
        ("n_grid", 50),
        ("distance_limits", {"nucleus": (0.0, 2.0), "z": (0.0, 2.0)}),
        ("distance_measures", ["nucleus"]),
        ("packing_modes", ["random"]),
    ]:
        with pytest.raises(ValueError, match=key):
            load_distance_pdfs(tmp_path, "histogram", expected_meta={key: value})


def test_loader_ignores_tuple_vs_list_and_list_order(
    tmp_path: Path, dist_dict: dict[str, Any]
) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    expected = {
        "distance_limits": {"nucleus": (0.0, 1.0), "z": (0.0, 2.0)},
        "distance_measures": ["z", "nucleus"],
        "packing_modes": ["random", "apical_gradient"],
    }
    assert set(load_distance_pdfs(tmp_path, "histogram", expected_meta=expected)) == set(DMS)


def test_loader_raises_when_expected_key_absent_from_sidecar(
    tmp_path: Path, dist_dict: dict[str, Any]
) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    meta_path = tmp_path / "distance_pdfs_histogram.meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["minimum_distance"]  # recorded None must not be matched by absence
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="minimum_distance"):
        load_distance_pdfs(tmp_path, "histogram", expected_meta={"minimum_distance": None})


def test_loader_raises_on_unreadable_sidecar(tmp_path: Path, dist_dict: dict[str, Any]) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    (tmp_path / "distance_pdfs_histogram.meta.json").write_text("{not json")
    with pytest.raises(ValueError, match="Could not read"):
        load_distance_pdfs(tmp_path, "histogram")


def test_loader_raises_when_pickle_disagrees_with_sidecar(
    tmp_path: Path, dist_dict: dict[str, Any]
) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    save_path = tmp_path / "distance_pdfs_histogram.dat"
    data = pickle.loads(save_path.read_bytes())
    del data["z"]
    save_path.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match="does not match its sidecar"):
        load_distance_pdfs(tmp_path, "histogram")


def test_loader_raises_when_sidecar_lacks_measures_and_modes(
    tmp_path: Path, dist_dict: dict[str, Any]
) -> None:
    compute_distance_pdfs(dist_dict, results_dir=tmp_path, **_kwargs())
    (tmp_path / "distance_pdfs_histogram.meta.json").write_text(json.dumps({"method": "histogram"}))
    with pytest.raises(ValueError, match="does not record"):
        load_distance_pdfs(tmp_path, "histogram")


# ── structure consumed by the replot ───────────────────────────────────────────────


def test_per_measure_and_mode_assembly_equals_the_full_result(
    dist_dict: dict[str, Any],
) -> None:
    """replot_figure_assets builds the dict leaf by leaf; with fixed limits it is identical."""
    full = compute_distance_pdfs(dist_dict, **_kwargs())
    got = {
        (dm, m): compute_distance_pdfs(
            dist_dict, **_kwargs(distance_measures=[dm], packing_modes=[m])
        )[dm][m]
        for dm in DMS
        for m in MODES
    }
    assembled = {dm: {m: got[dm, m] for m in MODES} for dm in DMS}
    _assert_same(full, assembled)
