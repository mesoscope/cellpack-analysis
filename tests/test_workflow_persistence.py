"""Tests for what the analysis workflows persist for the manuscript runner.

The runner loads caches and recomputes nothing, so each pipeline step it reads must write its
result: the distance PDFs, the seeded KS frames, the pairwise envelope dict, and the fit-phase
configs. All fixtures live in ``tmp_path``; the project root is patched to it.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cellpack_analysis.analysis.workflows import run_analysis_workflow as raw
from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw
from cellpack_analysis.analysis.workflows.configs import defaults
from cellpack_analysis.lib import distance
from cellpack_analysis.lib.rule_interpolation import CVResult

BASELINE = "SLC25A17"
MODES = [BASELINE, "random"]
MEASURES = ["nucleus", "z"]
CONFIGS_DIR = Path(raw.__file__).parent / "configs"
LADDER_CONFIGS = [
    CONFIGS_DIR / "rule_interpolation" / f"rule_interpolation_tune_gradients_v2_{s}.json"
    for s in ("peroxisome", "endosome")
]


def _distance_dict() -> dict[str, Any]:
    """Tiny ``{dm: {mode: {cell: {seed: distances}}}}`` in microns, inside DISTANCE_LIMITS."""
    rng = np.random.default_rng(1)
    out: dict[str, Any] = {}
    for dm, hi in (("nucleus", 5.0), ("z", 9.0)):
        out[dm] = {}
        for mode, seeds in ((BASELINE, ["0"]), ("random", ["0", "1"])):
            out[dm][mode] = {
                cell: {seed: rng.uniform(0.1, hi, 30) for seed in seeds}
                for cell in ("101", "102", "103", "104")
            }
    return out


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch both workflow modules to resolve data and results under ``tmp_path``."""
    monkeypatch.setattr(raw, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(riw, "get_project_root", lambda: tmp_path)
    return tmp_path


def _write_config(root: Path, name: str = "config.json", **extra: Any) -> Path:
    config = {
        "name": "persistence_test",
        "packing_id": "peroxisome",
        "structure_id": BASELINE,
        "channel_map": dict.fromkeys(MODES, BASELINE),
        "baseline_mode": BASELINE,
        "distance_measures": MEASURES,
        "output_path": "persist/peroxisome",
        "analysis_steps": [],
        **extra,
    }
    path = root / name
    path.write_text(json.dumps(config))
    return path


def _runner(config: raw.AnalysisConfig) -> raw.AnalysisRunner:
    runner = raw.AnalysisRunner()
    runner.shared_data["all_distance_dict"] = _distance_dict()
    return runner


# --- run provenance ----------------------------------------------------------


def test_main_writes_one_provenance_line(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_config(project)
    monkeypatch.setattr(raw, "setup_logging", lambda level: None)
    monkeypatch.setattr(sys, "argv", ["run_analysis_workflow", "-c", str(path)])
    raw.main()
    log = project / "results" / "persist" / "peroxisome" / "logs" / "run_provenance.jsonl"
    (line,) = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert line["argv"] == ["run_analysis_workflow", "-c", str(path)]
    assert line["config_file"] == path.as_posix() and line["config_sha256"]
    # a nested run (the rule-interpolation phases call it) records nothing itself
    raw.AnalysisRunner().run_analysis(str(path))
    assert len(log.read_text().splitlines()) == 1


# --- AnalysisConfig ----------------------------------------------------------


def test_create_dirs_false_creates_nothing(project: Path) -> None:
    raw.AnalysisConfig(str(_write_config(project)), create_dirs=False)
    assert not (project / "results").exists()


def test_create_dirs_default_creates_the_three_dirs(project: Path) -> None:
    config = raw.AnalysisConfig(str(_write_config(project)))
    assert config.results_dir.is_dir()
    assert config.figures_dir.is_dir()
    assert config.logs_dir.is_dir()


def test_ks_bootstrap_seed_defaults_to_unseeded(project: Path) -> None:
    assert defaults.KS_BOOTSTRAP_SEED is None
    default = raw.AnalysisConfig(str(_write_config(project)), create_dirs=False)
    seeded = raw.AnalysisConfig(
        str(_write_config(project, "seeded.json", ks_bootstrap_seed=0)), create_dirs=False
    )
    assert default.ks_bootstrap_seed is None
    assert seeded.ks_bootstrap_seed == 0


# --- plot_distance_distributions + run_ks_analysis (smoke) -------------------


@pytest.mark.parametrize("suffix", ["", "_x"])
def test_distance_steps_write_pdf_cache_and_ks_frames(project: Path, suffix: str) -> None:
    """A reduced distance config writes the PDF cache and the two KS parquets."""
    extra: dict[str, Any] = {"n_bootstrap": 10, "ks_bootstrap_seed": 0}
    if suffix:
        extra["suffix"] = suffix
    config = raw.AnalysisConfig(str(_write_config(project, **extra)))
    runner = _runner(config)
    runner._execute_step("plot_distance_distributions", config)
    runner._execute_step("run_ks_analysis", config)

    rd = config.results_dir
    assert (rd / f"distance_pdfs_kde{suffix}.dat").exists()
    assert (rd / f"distance_pdfs_kde{suffix}.meta.json").exists()
    loaded = distance.load_distance_pdfs(rd, "kde", suffix=suffix)
    assert set(loaded) == set(MEASURES)
    for dm in MEASURES:
        np.testing.assert_array_equal(
            loaded[dm]["random"]["mean_pdf"],
            runner.shared_data["distance_pdf_dict"][dm]["random"]["mean_pdf"],
        )

    ks_df = pd.read_parquet(rd / f"ks_test_vs_{BASELINE}{suffix}.parquet")
    boot_df = pd.read_parquet(rd / f"ks_bootstrap_vs_{BASELINE}{suffix}.parquet")
    assert set(ks_df["packing_mode"]) == {"random"}
    assert (ks_df["baseline_mode"] == BASELINE).all()
    assert len(boot_df) == 10 * len(MEASURES)
    pd.testing.assert_frame_equal(
        boot_df, runner.shared_data["pairwise_ks_bootstrap_df"], check_dtype=False
    )

    # Seeded: the persisted bootstrap is the direct seed-0 draw
    expected = distance.bootstrap_ks_tests(
        ks_test_df=ks_df,
        distance_measures=MEASURES,
        packing_modes=["random"],
        n_bootstrap=10,
        seed=0,
    )
    pd.testing.assert_series_equal(
        boot_df["similar_fraction"], expected["similar_fraction"], check_names=False
    )


def test_ks_seed_is_forwarded(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    real = distance.bootstrap_ks_tests

    def _spy(**kwargs: Any) -> pd.DataFrame:
        seen.append(kwargs.get("seed"))
        return real(**kwargs)

    monkeypatch.setattr(raw.distance, "bootstrap_ks_tests", _spy)
    config = raw.AnalysisConfig(str(_write_config(project, n_bootstrap=2, ks_bootstrap_seed=7)))
    _runner(config)._execute_step("run_ks_analysis", config)
    assert seen == [7]


# --- run_pairwise_envelope_test ----------------------------------------------


@pytest.fixture
def no_envelope_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "plot_pairwise_envelope_matrix",
        "plot_per_dm_rejection_bars",
        "plot_per_dm_envelopes_overlaid",
    ):
        monkeypatch.setattr(raw.visualization, name, lambda **kwargs: None)


def _assert_same(a: Any, b: Any) -> None:
    """Deep equality over dicts, sequences, arrays (NaN-aware) and scalars."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, list | tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_same(x, y)
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b)
    else:
        assert a == b


def test_pairwise_envelope_pickle_round_trips_the_raw_return(
    project: Path, no_envelope_plots: None
) -> None:
    """The pickle is the exact ``pairwise_envelope_test`` return, with its own keys."""
    config = raw.AnalysisConfig(str(_write_config(project)))
    runner = _runner(config)
    runner._execute_step("run_pairwise_envelope_test", config)

    path = config.results_dir / "pairwise_envelope_results.dat"
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    in_memory = runner.shared_data["pairwise_envelope_results"]
    assert {"joint", "packing_modes", "per_distance_measure", "envelopes"} <= set(loaded)
    assert loaded["packing_modes"] == MODES
    _assert_same(loaded, in_memory)

    direct = raw.pairwise_envelope_test(
        all_distance_dict=_distance_dict(),
        packing_modes=MODES,
        distance_measures=MEASURES,
        **config.envelope_test_params,
    )
    _assert_same(loaded, direct)


def test_pairwise_envelope_cache_is_reused_unless_recalculated(
    project: Path, no_envelope_plots: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = raw.AnalysisConfig(str(_write_config(project)))
    _runner(config)._execute_step("run_pairwise_envelope_test", config)
    path = config.results_dir / "pairwise_envelope_results.dat"
    with open(path, "rb") as f:
        first = pickle.load(f)

    def _boom(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("recomputed despite a usable cache")

    monkeypatch.setattr(raw, "pairwise_envelope_test", _boom)
    runner = _runner(config)
    runner._execute_step("run_pairwise_envelope_test", config)
    _assert_same(runner.shared_data["pairwise_envelope_results"], first)

    calls: list[int] = []

    def _fresh(**kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return {**first, "joint": {}}

    monkeypatch.setattr(raw, "pairwise_envelope_test", _fresh)
    forced = raw.AnalysisConfig(
        str(_write_config(project, "forced.json", recalculate={"run_pairwise_envelope_test": True}))
    )
    _runner(forced)._execute_step("run_pairwise_envelope_test", forced)
    assert calls == [1]
    with open(path, "rb") as f:
        assert pickle.load(f)["joint"] == {}


def _spy_on_pairwise_test(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count calls to the real ``pairwise_envelope_test``."""
    calls: list[int] = []
    real = raw.pairwise_envelope_test

    def _spy(**kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return real(**kwargs)

    monkeypatch.setattr(raw, "pairwise_envelope_test", _spy)
    return calls


def test_pairwise_envelope_sidecar_records_every_test_parameter(
    project: Path, no_envelope_plots: None
) -> None:
    config = raw.AnalysisConfig(str(_write_config(project)))
    _runner(config)._execute_step("run_pairwise_envelope_test", config)
    meta_path = config.results_dir / "pairwise_envelope_results.meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta == raw._pairwise_envelope_cache_meta(config)
    assert meta == {
        "alpha": 0.05,
        "bin_width": 0.2,
        "distance_measures": MEASURES,
        "envelope_type": "pointwise",
        "joint_r_grid_size": None,
        "packing_modes": MODES,
        "r_grid_size": 150,
        "statistic": "intdev",
    }


def test_pairwise_envelope_cache_recomputes_on_mode_and_measure_order(
    project: Path, no_envelope_plots: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = raw.AnalysisConfig(str(_write_config(project)))
    _runner(config)._execute_step("run_pairwise_envelope_test", config)
    calls = _spy_on_pairwise_test(monkeypatch)
    reordered = raw.AnalysisConfig(
        str(
            _write_config(
                project,
                "reordered.json",
                channel_map=dict.fromkeys(MODES[::-1], BASELINE),
                distance_measures=MEASURES[::-1],
            )
        )
    )
    _runner(reordered)._execute_step("run_pairwise_envelope_test", reordered)
    assert len(calls) == 1


def _change_alpha(project: Path, config: raw.AnalysisConfig) -> raw.AnalysisConfig:
    return raw.AnalysisConfig(
        str(_write_config(project, "alpha.json", envelope_test_params={"alpha": 0.1}))
    )


def _drop_sidecar(project: Path, config: raw.AnalysisConfig) -> raw.AnalysisConfig:
    (config.results_dir / "pairwise_envelope_results.meta.json").unlink()
    return config


def _corrupt_pickle(project: Path, config: raw.AnalysisConfig) -> raw.AnalysisConfig:
    (config.results_dir / "pairwise_envelope_results.dat").write_bytes(b"not a pickle")
    return config


def _pickle_a_list(project: Path, config: raw.AnalysisConfig) -> raw.AnalysisConfig:
    with open(config.results_dir / "pairwise_envelope_results.dat", "wb") as f:
        pickle.dump(["not", "a", "dict"], f)
    return config


@pytest.mark.parametrize(
    "spoil",
    [_change_alpha, _drop_sidecar, _corrupt_pickle, _pickle_a_list],
    ids=["alpha_differs", "no_sidecar", "corrupt_pickle", "not_a_dict"],
)
def test_unusable_pairwise_envelope_cache_is_recomputed_and_rewritten(
    project: Path,
    no_envelope_plots: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    spoil: Any,
) -> None:
    config = raw.AnalysisConfig(str(_write_config(project)))
    _runner(config)._execute_step("run_pairwise_envelope_test", config)
    rerun = spoil(project, config)
    calls = _spy_on_pairwise_test(monkeypatch)
    with caplog.at_level("WARNING", logger=raw.logger.name):
        runner = _runner(rerun)
        runner._execute_step("run_pairwise_envelope_test", rerun)

    assert calls == [1]
    assert "unusable" in caplog.text
    path = rerun.results_dir / "pairwise_envelope_results.dat"
    with open(path, "rb") as f:
        _assert_same(pickle.load(f), runner.shared_data["pairwise_envelope_results"])
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta == raw._pairwise_envelope_cache_meta(rerun)
    assert meta["alpha"] == rerun.envelope_test_params["alpha"]


def test_pairwise_envelope_cache_with_other_modes_is_recomputed(
    project: Path, no_envelope_plots: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-sidecar pickle, even one naming other modes, is never trusted."""
    config = raw.AnalysisConfig(str(_write_config(project)))
    path = config.results_dir / "pairwise_envelope_results.dat"
    with open(path, "wb") as f:
        pickle.dump({"packing_modes": ["other"], "distance_measures": MEASURES}, f)
    calls = _spy_on_pairwise_test(monkeypatch)
    _runner(config)._execute_step("run_pairwise_envelope_test", config)
    assert calls == [1]
    with open(path, "rb") as fh:
        assert pickle.load(fh)["packing_modes"] == MODES


# --- rule-interpolation fit/validate configs ---------------------------------


def _e2e_config(**extra: Any) -> dict[str, Any]:
    return {
        "name": "ri_test",
        "structure_id": BASELINE,
        "packing_id": "peroxisome",
        "channel_map": dict.fromkeys(MODES, BASELINE),
        "output_path": "ri_test/peroxisome",
        "validation_output_path": "ri_test/peroxisome/validation",
        "validation_config_out_path": "rule_interpolation_validation/ri_test",
        **extra,
    }


def test_fit_config_keeps_ks_bootstrap_seed() -> None:
    assert riw._build_fit_config_dict(_e2e_config(ks_bootstrap_seed=0))["ks_bootstrap_seed"] == 0


def test_validation_output_path_is_results_relative(project: Path) -> None:
    path = riw._write_validation_analysis_config(_e2e_config(), project)
    written = json.loads(Path(path).read_text())
    assert written["output_path"] == "ri_test/peroxisome/validation"
    config = raw.AnalysisConfig(str(path), create_dirs=False)
    assert config.results_dir == project / "results" / "ri_test/peroxisome/validation"


def test_loaders_raise_before_fit(project: Path) -> None:
    e2e = project / "e2e.json"
    e2e.write_text(json.dumps(_e2e_config()))
    with pytest.raises(FileNotFoundError, match="run --phase fit"):
        riw.load_fit_phase_config(e2e)
    with pytest.raises(FileNotFoundError, match="run --phase fit"):
        riw.load_validation_config(e2e)
    assert not (project / "results").exists()


def test_loaders_return_the_written_configs_without_creating_dirs(project: Path) -> None:
    raw_config = _e2e_config(ks_bootstrap_seed=0)
    e2e = project / "e2e.json"
    e2e.write_text(json.dumps(raw_config))
    results_dir = riw._resolve_results_dir(raw_config)
    results_dir.mkdir(parents=True)
    fit_config = {**riw._build_fit_config_dict(raw_config), "analysis_steps": []}
    (results_dir / "_fit_phase_config.json").write_text(json.dumps(fit_config))
    riw._write_validation_analysis_config(raw_config, results_dir)

    fit_cfg = riw.load_fit_phase_config(e2e)
    val_cfg = riw.load_validation_config(e2e)
    assert fit_cfg.results_dir == results_dir
    assert val_cfg.results_dir == project / "results" / "ri_test/peroxisome/validation"
    assert fit_cfg.ks_bootstrap_seed == val_cfg.ks_bootstrap_seed == 0
    assert "interpolated" in val_cfg.channel_map
    assert not (results_dir / "figures").exists()
    assert not val_cfg.results_dir.exists()


# --- Fig S10: the ladder fit draws the profile -------------------------------


@pytest.mark.parametrize("config_path", LADDER_CONFIGS, ids=lambda p: p.stem)
def test_ladder_fit_steps_end_with_the_profile(config_path: Path) -> None:
    steps = riw._build_fit_config_dict(json.loads(config_path.read_text()))["analysis_steps"]
    assert steps == [*riw._REQUIRED_FIT_ANALYSIS_STEPS, "plot_lambda_ladder_figures"]


def test_ladder_fit_phase_writes_the_profile(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--phase fit`` on the real ladder config runs the profile step after CV."""
    raw_config = json.loads(LADDER_CONFIGS[0].read_text())
    modes = list(raw_config["channel_map"])
    baseline = raw_config["structure_id"]
    xvals = np.linspace(0, 6, 40)
    order: list[str] = []

    def _occupancy(self: raw.AnalysisRunner, config: raw.AnalysisConfig) -> None:
        order.append("run_occupancy_analysis")
        self.shared_data["occupancy_dict"] = {
            dm: {
                mode: {
                    "individual": {},
                    "combined": {"xvals": xvals, "occupancy": np.linspace(2.0, 0.5 + i, 40)},
                }
                for i, mode in enumerate(modes)
            }
            for dm in config.occupancy_distance_measures
        }

    def _cv(self: raw.AnalysisRunner, config: raw.AnalysisConfig) -> None:
        order.append("run_rule_interpolation_cv")
        mse: dict[str, Any] = {"individual": {"nucleus": 0.0, "z": 0.0}, "joint": {"joint": 0.0}}
        self.shared_data["rule_interpolation_cv_result"] = CVResult(
            folds=[],
            aggregated_coefficients={
                mode: {"nucleus": (0.1, 0.01), "z": (0.1, 0.01), "joint": (0.1, 0.01)}
                for mode in modes
                if mode != baseline
            },
            mean_train_mse=mse,
            std_train_mse=mse,
            mean_test_mse=mse,
            std_test_mse=mse,
            n_folds=5,
            baseline_mode=baseline,
        )

    real_ladder = raw.AnalysisRunner._plot_lambda_ladder_figures

    def _ladder(self: raw.AnalysisRunner, config: raw.AnalysisConfig) -> None:
        order.append("plot_lambda_ladder_figures")
        real_ladder(self, config)

    monkeypatch.setattr(raw.AnalysisRunner, "_run_occupancy_analysis", _occupancy)
    monkeypatch.setattr(raw.AnalysisRunner, "_run_rule_interpolation_cv", _cv)
    monkeypatch.setattr(raw.AnalysisRunner, "_plot_lambda_ladder_figures", _ladder)
    monkeypatch.setattr(
        riw.rule_interpolation, "generate_mixed_rule_packing_configs", lambda **kwargs: []
    )

    riw.fit(str(LADDER_CONFIGS[0]))

    results_dir = project / "results" / raw_config["output_path"]
    assert order == [*riw._REQUIRED_FIT_ANALYSIS_STEPS, "plot_lambda_ladder_figures"]
    assert (results_dir / "figures" / "lambda_ladder" / "lambda_ladder_profile.pdf").exists()


# --- shipped configs -----------------------------------------------------------

SEEDED_CONFIGS = {
    f"distance_analysis/distance_analysis_{kind}{s}_replicate_99_v4.json"
    for kind in ("", "supp_")
    for s in ("peroxisome", "endosome")
} | {
    f"rule_interpolation_replicate_99_v4/rule_interpolation_replicate_99_v4_{s}.json"
    for s in ("peroxisome", "endosome")
}
PAPER_CONFIG_DIRS = [
    "distance_analysis",
    "occupancy_analysis",
    "rule_interpolation",
    "rule_interpolation_replicate_99_v4",
    "rule_interpolation_validation",
    "biological_variation",
]


def test_ks_bootstrap_seed_is_set_in_exactly_the_v4_paper_configs() -> None:
    """Hand-written configs only; fit-generated validation configs may inherit the seed."""
    seeded = {
        path.relative_to(CONFIGS_DIR).as_posix(): json.loads(path.read_text())["ks_bootstrap_seed"]
        for path in CONFIGS_DIR.rglob("*.json")
        if "ks_bootstrap_seed" in json.loads(path.read_text())
    }
    generated = {k for k in seeded if k.startswith("rule_interpolation_validation/")}
    assert {k: seeded[k] for k in seeded.keys() - generated} == dict.fromkeys(SEEDED_CONFIGS, 0)
    assert all(seeded[k] == 0 for k in generated)


def test_paper_analysis_configs_carry_no_absolute_paths() -> None:
    hits = [
        path.relative_to(CONFIGS_DIR).as_posix()
        for d in PAPER_CONFIG_DIRS
        for path in (CONFIGS_DIR / d).rglob("*.json")
        if "/allen" in path.read_text()
    ]
    assert hits == []
