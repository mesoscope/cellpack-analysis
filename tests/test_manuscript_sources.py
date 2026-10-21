# mypy: allow-untyped-defs
"""Tests for the manuscript source resolver (``manuscript_sources``) on a fake project root.

The real v4 source and packing configs are copied into ``tmp_path``; every cache is written
there by the pipeline's own writers, so nothing reads or writes the live ``results/``.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

from cellpack_analysis.analysis.workflows import manuscript_config as mc
from cellpack_analysis.analysis.workflows import manuscript_sources as ms
from cellpack_analysis.analysis.workflows import run_analysis_workflow as raw
from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw
from cellpack_analysis.lib import occupancy
from cellpack_analysis.lib import rule_interpolation as ri
from cellpack_analysis.packing.rule_repository import GRADIENTS

TREE = Path(__file__).parents[1]
CONFIGS = "cellpack_analysis/analysis/workflows/configs"
MANUSCRIPT = f"{CONFIGS}/manuscript/manuscript_replicate_99_v4.json"
STRUCTURES = ("peroxisome", "endosome")
S = "endosome"
FLOORS: dict[str, float | None] = {
    "minimum_distance_um": -0.108,
    "available_minimum_distance_um": -1.0,
}
X_MAX = {"nucleus": 6, "z": 8}


def _copied_paths() -> list[str]:
    raw_config = json.loads((TREE / MANUSCRIPT).read_text())
    paths = [MANUSCRIPT]
    for template in raw_config["sources"].values():
        for s in STRUCTURES if "{structure}" in template else [None]:
            paths.append(f"{CONFIGS}/{template.format(structure=s)}")
    for template in raw_config["packing_configs"].values():
        paths += [template.format(structure=s) for s in STRUCTURES]
    validation = f"{CONFIGS}/rule_interpolation_validation/replicate_99_v4"
    paths += [f"{validation}/mixed_rule_validation_config_{s}.json" for s in STRUCTURES]
    return paths


@pytest.fixture()
def proj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "proj"
    for rel in _copied_paths():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((TREE / rel).read_bytes())  # no mode bits copied
    for module in (mc, raw, riw):
        monkeypatch.setattr(module, "get_project_root", lambda: root)
    monkeypatch.setattr(mc, "get_results_path", lambda: root / "results")
    return root


@pytest.fixture()
def sources(proj: Path) -> ms.PaperSources:
    return ms.PaperSources(mc.ManuscriptConfig(proj / MANUSCRIPT))


def _edit(path: Path, **changes: Any) -> None:
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def _config(path: Path) -> raw.AnalysisConfig:
    return raw.AnalysisConfig(str(path), create_dirs=False)


def _write_occupancy(
    results_dir: Path,
    channel_map: dict[str, str],
    suffix: str = "",
    num_points: int = 250,
    measures: tuple[str, ...] = ("nucleus", "z"),
) -> dict[str, Any]:
    """Write KDE occupancy caches (and sidecars) as ``run_occupancy_analysis`` does."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rng, rng_available = np.random.default_rng(0), np.random.default_rng(1)
    kdes: dict[str, dict[str, gaussian_kde]] = {}
    for i in range(4):
        cell = {"available": gaussian_kde(rng_available.uniform(0.0, 9.0, 200))}
        cell.update({mode: gaussian_kde(rng.uniform(0.2, 5.0, 60)) for mode in channel_map})
        kdes[f"c{i}"] = cell
    return {
        dm: occupancy.get_kde_occupancy_dict(
            kdes,
            channel_map,
            results_dir=results_dir,
            recalculate=True,
            suffix=suffix,
            distance_measure=dm,
            bandwidth=0.2,
            num_points=num_points,
            x_min=0,
            x_max=X_MAX[dm],
            num_workers=1,
            floors_um=FLOORS,
        )
        for dm in measures
    }


def _write_fit(sources: ms.PaperSources, structure: str = S) -> raw.AnalysisConfig:
    """Write the fit phase's config, occupancy caches and CV pickle."""
    e2e = json.loads(sources.source_path("rule_interpolation", structure).read_text())
    fit_dir = riw._resolve_results_dir(e2e)
    fit_dir.mkdir(parents=True)
    path = riw._fit_phase_config_path(fit_dir)
    path.write_text(json.dumps(riw._build_fit_config_dict(e2e)))
    cfg = _config(path)
    occ = _write_occupancy(fit_dir, cfg.cv_channel_map, cfg.suffix)
    ri.run_rule_interpolation_cv(
        occ,
        cfg.cv_channel_map,
        cfg.baseline_mode,
        n_folds=2,
        n_repeats=1,
        random_state=0,
        distance_measures=cfg.occupancy_distance_measures,
        results_dir=fit_dir,
        recalculate=True,
        suffix=cfg.suffix,
        grouping="combined",
    )
    return cfg


def _validation_cfg(sources: ms.PaperSources, structure: str = S) -> raw.AnalysisConfig:
    e2e = json.loads(sources.source_path("rule_interpolation", structure).read_text())
    return _config(riw._resolve_validation_config_path(e2e))


EMD_ROWS = pd.DataFrame(
    {
        "cell_id_1": ["c0", "c0", "c0"],
        "cell_id_2": ["c0", "c1", "c1"],
        "packing_mode_1": ["RAB5A", "RAB5A", "RAB5A"],
        "packing_mode_2": ["interpolated", "interpolated", "RAB5A"],
        "emd": [0.1, 0.2, 0.3],
    }
)


def _write_validation(sources: ms.PaperSources, fit: raw.AnalysisConfig) -> raw.AnalysisConfig:
    """Write the validate phase's occupancy caches, result pickle and EMD parquet."""
    v = _validation_cfg(sources)
    _write_occupancy(v.results_dir, v.channel_map, v.suffix)
    ri._save_validation_result(ri.ValidationResult(emd_df=EMD_ROWS), fit.results_dir)
    EMD_ROWS.to_parquet(fit.results_dir / "occupancy_emd_mixed_rule_validation.parquet")
    return v


def _envelope_result(**changes: Any) -> dict[str, Any]:
    result = {
        "summary": {},
        "statistic": "intdev",
        "envelope_type": "pointwise",
        "alpha": 0.05,
        "calibration_holdout_index": None,
    }
    return {**result, **changes}


def _write_pickle(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    return path


def _write_pairwise(sources: ms.PaperSources) -> Path:
    cfg = _config(sources.source_path("biological_variation"))
    path = cfg.results_dir / f"pairwise_envelope_results{cfg.suffix or ''}.dat"
    result = {"joint": {}, "packing_modes": cfg.packing_modes}
    raw._save_pairwise_envelope_cache(path, result, raw._pairwise_envelope_cache_meta(cfg))
    return path


class TestConfigs:
    def test_resolving_every_config_creates_nothing(self, proj, sources):
        for s in STRUCTURES:
            assert sources.distance_config(s).results_dir.name == s
            assert sources.distance_config(s, supp=True).results_dir.name == "supp"
            assert sources.occupancy_config(s).packing_id == s
            assert sources.packing_config("condition", s)["packing_id"] == s
        assert sources.biovar_config().minimum_distance_um is None
        assert not (proj / "results").exists()
        assert all(path.is_relative_to(proj) and path.exists() for path in sources.inputs)
        assert len(sources.inputs) == 3 * 2 + 2 + 1

    def test_paths(self, proj, sources):
        assert sources.source_path("distance", S) == (
            proj / CONFIGS / "distance_analysis/distance_analysis_endosome_replicate_99_v4.json"
        )
        assert sources.source_path("biological_variation") == (
            proj / CONFIGS / "biological_variation/biological_variation_config.json"
        )
        assert sources.packing_config_path("ladder", S) == (
            proj / "cellpack_analysis/packing/configs/tune_gradients_v2/"
            "endosome_tune_gradients_v2.json"
        )

    def test_bad_structure_and_key(self, sources):
        with pytest.raises(ValueError, match=r"Unknown structure 'golgi'"):
            sources.distance_config("golgi")
        with pytest.raises(ValueError, match=r"per structure"):
            sources.source_path("distance")
        with pytest.raises(KeyError, match=r"'nope' is not configured"):
            sources.source_path("nope", S)

    def test_a_source_left_out_fails_only_on_use(self, proj):
        path = proj / MANUSCRIPT
        data = json.loads(path.read_text())
        del data["sources"]["ladder"]
        path.write_text(json.dumps(data))
        sources = ms.PaperSources(mc.ManuscriptConfig(path))
        assert sources.distance_config(S).packing_id == S
        with pytest.raises(KeyError, match=r"'ladder' is not configured"):
            sources.fit_config("ladder", S)

    def test_missing_source_config(self, sources):
        sources.source_path("occupancy", S).unlink()
        with pytest.raises(FileNotFoundError, match=r"Source config 'occupancy' not found"):
            sources.occupancy_config(S)

    def test_results_dir_must_stay_under_the_results_root(self, sources):
        _edit(sources.source_path("distance", S), output_path="../../outside/endosome")
        with pytest.raises(ValueError, match=r"not under the results root"):
            sources.distance_config(S)

    def test_packing_id_must_match_the_structure(self, sources):
        pero = sources.source_path("distance", "peroxisome").read_text()
        sources.source_path("distance", S).write_text(pero)
        with pytest.raises(ValueError, match=r"packing_id 'peroxisome' != 'endosome'"):
            sources.distance_config(S)

    def test_floors_must_match_expected(self, sources):
        _edit(sources.source_path("occupancy", S), minimum_distance_um=-0.28)
        with pytest.raises(ValueError, match=r"floors .*-0.28"):
            sources.occupancy_config(S)

    def test_packing_config_checks(self, sources):
        sources.packing_config_path("condition", S).unlink()
        with pytest.raises(FileNotFoundError, match=r"Packing config 'condition'"):
            sources.packing_config("condition", S)
        _edit(sources.packing_config_path("ladder", S), packing_id="peroxisome")
        with pytest.raises(ValueError, match=r"packing_id 'peroxisome'"):
            sources.packing_config("ladder", S)


class TestDistanceCaches:
    def test_distance_pdfs_load_the_pipeline_cache(self, sources, monkeypatch):
        cfg = _config(sources.source_path("distance", S))
        rng = np.random.default_rng(0)
        runner = raw.AnalysisRunner()
        runner.shared_data["all_distance_dict"] = {
            dm: {
                mode: {f"c{i}": {"0": rng.uniform(0.5, 5.0, 40)} for i in range(3)}
                for mode in cfg.packing_modes
            }
            for dm in cfg.distance_measures
        }
        monkeypatch.setattr(raw.visualization, "plot_distance_distributions", lambda **_: None)
        monkeypatch.setattr(
            raw.distance, "log_central_tendencies_for_distance_distributions", lambda **_: None
        )
        runner._plot_distance_distributions(cfg)
        pdfs = sources.distance_pdfs(S)
        assert set(pdfs) == set(cfg.distance_measures)
        cache = cfg.results_dir / "distance_pdfs_kde.dat"
        assert sources.inputs[-2:] == [cache, cache.with_suffix(".meta.json")]

    def test_distance_pdfs_reject_a_stale_sidecar(self, sources):
        cfg = sources.distance_config(S)
        cache = _write_pickle(cfg.results_dir / "distance_pdfs_kde.dat", {})
        with pytest.raises(FileNotFoundError, match=r"plot_distance_distributions"):
            sources.distance_pdfs(S)
        meta = {**sources.distance_pdf_meta(S), "n_grid": 7}
        cache.with_suffix(".meta.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError):
            sources.distance_pdfs(S)

    def test_ks_frames(self, sources):
        cfg = sources.distance_config(S)
        cfg.results_dir.mkdir(parents=True)
        df = pd.DataFrame({"baseline_mode": ["RAB5A"] * 2, "packing_mode": ["random", "x"]})
        df.to_parquet(cfg.results_dir / "ks_test_vs_RAB5A.parquet")
        pd.testing.assert_frame_equal(sources.ks_tests(S), df)
        with pytest.raises(FileNotFoundError, match=r"run_ks_analysis .*stage 'distance'"):
            sources.ks_bootstrap(S)
        df.assign(baseline_mode="random").to_parquet(
            cfg.results_dir / "ks_bootstrap_vs_RAB5A.parquet"
        )
        with pytest.raises(ValueError, match=r"baseline_mode \['random'\]"):
            sources.ks_bootstrap(S)

    def test_distance_emd_is_same_cell_filtered(self, sources):
        cfg = sources.distance_config(S, supp=True)
        cfg.results_dir.mkdir(parents=True)
        EMD_ROWS.to_parquet(cfg.results_dir / "distance_pairwise_emd.parquet")
        assert sources.distance_emd(S, supp=True)["emd"].tolist() == [0.1, 0.3]
        with pytest.raises(FileNotFoundError, match=r"stage 'distance'"):
            sources.distance_emd(S)

    def test_occupancy_emd_is_the_raw_cached_frame(self, sources):
        cfg = sources.occupancy_config(S)
        path = cfg.results_dir / f"occupancy_emd{cfg.suffix or ''}.parquet"
        with pytest.raises(FileNotFoundError, match=r"run_occupancy_emd_analysis .*'occupancy'"):
            sources.occupancy_emd(S)
        cfg.results_dir.mkdir(parents=True)
        EMD_ROWS.to_parquet(path)
        df = sources.occupancy_emd(S)
        pd.testing.assert_frame_equal(df, EMD_ROWS)  # not same-cell filtered
        assert path in sources.inputs
        path.unlink()
        assert sources.occupancy_emd(S) is df  # memoized

    def test_per_cell_envelope(self, sources):
        cfg = sources.distance_config(S)
        path = cfg.results_dir / "per_cell_envelope_results.dat"
        with pytest.raises(FileNotFoundError, match=r"run_per_cell_envelope_test"):
            sources.per_cell_envelope(S)
        _write_pickle(path, _envelope_result())
        assert sources.per_cell_envelope(S)["summary"] == {}
        _write_pickle(path, _envelope_result(statistic="supremum"))
        with pytest.raises(ValueError, match=r"test parameters"):
            ms.PaperSources(sources.config).per_cell_envelope(S)
        _write_pickle(path, {"statistic": "intdev"})
        with pytest.raises(ValueError, match=r"not a per_cell_envelope_test result"):
            ms.PaperSources(sources.config).per_cell_envelope(S)

    def test_calibration_envelope(self, sources):
        cal = sources.source_path("distance_calibration", S)
        path = _config(cal).results_dir / "per_cell_envelope_results_calibration.dat"
        with pytest.raises(FileNotFoundError, match=r"stage 'calibration'"):
            sources.per_cell_envelope(S, "_calibration")
        _write_pickle(path, _envelope_result(calibration_holdout_index=0))
        assert sources.per_cell_envelope(S, "_calibration")["calibration_holdout_index"] == 0
        assert cal in sources.inputs
        with pytest.raises(ValueError, match=r"suffix '_calibration' != '_other'"):
            sources.per_cell_envelope(S, "_other")
        _write_pickle(path, _envelope_result())
        with pytest.raises(ValueError, match=r"test parameters"):
            ms.PaperSources(sources.config).per_cell_envelope(S, "_calibration")

    def test_pairwise_envelope_needs_a_matching_sidecar(self, sources):
        path = _write_pairwise(sources)
        meta_path = path.with_suffix(".meta.json")
        assert set(ms.PaperSources(sources.config).pairwise_envelope()) == {
            "joint",
            "packing_modes",
        }
        meta = json.loads(meta_path.read_text())
        for bad in ({**meta, "bin_width": 0.2}, {**meta, "extra": 1}):
            meta_path.write_text(json.dumps(bad))
            with pytest.raises(ValueError, match=r"other parameters"):
                ms.PaperSources(sources.config).pairwise_envelope()
        meta_path.write_text("{not json")
        with pytest.raises(ValueError, match=r"Could not read"):
            ms.PaperSources(sources.config).pairwise_envelope()
        meta_path.unlink()
        with pytest.raises(FileNotFoundError, match=r"meta.json not found.*stage 'biovar'"):
            ms.PaperSources(sources.config).pairwise_envelope()

    def test_pairwise_envelope_checks_the_pickle(self, sources):
        path = _write_pairwise(sources)
        _write_pickle(path, {"joint": {}})
        with pytest.raises(ValueError, match=r"not a pairwise_envelope_test result"):
            sources.pairwise_envelope()


class TestOccupancyAndFit:
    def test_occupancy_and_memo_recording(self, sources):
        cfg = sources.occupancy_config(S)
        with pytest.raises(FileNotFoundError, match=r"run_occupancy_analysis .*'occupancy'"):
            sources.occupancy(S, "z")
        _write_occupancy(cfg.results_dir, cfg.channel_map, measures=("z",))
        first = sources.occupancy(S, "z")
        assert set(first) == set(cfg.channel_map)
        with sources.record_inputs() as used:
            assert sources.occupancy(S, "z") is first
        cache = cfg.results_dir / "z_occupancy.dat"
        assert used == [cfg.config_file, cache, cache.with_suffix(".meta.json")]
        with pytest.raises(ValueError, match=r"no occupancy for 'pairwise'"):
            sources.occupancy(S, "pairwise")

    def test_occupancy_with_other_parameters_raises(self, sources):
        cfg = sources.occupancy_config(S)
        _write_occupancy(cfg.results_dir, cfg.channel_map, num_points=100, measures=("z",))
        with pytest.raises(ValueError, match=r"different parameters .*num_points"):
            sources.occupancy(S, "z")

    def test_fit_config_is_required(self, sources):
        with pytest.raises(FileNotFoundError, match=r"--phase fit \(run_paper.sh stage 'fit'\)"):
            sources.fit_config("rule_interpolation", S)
        with pytest.raises(FileNotFoundError, match=r"stage 'ladder'"):
            sources.fit_config("ladder", S)
        with pytest.raises(ValueError, match=r"kind must be one of"):
            sources.fit_config("validation", S)

    def test_rule_interpolation_chain(self, sources):
        fit = _write_fit(sources)
        assert sources.fit_config("rule_interpolation", S).results_dir == fit.results_dir
        fit_occ = sources.fit_occupancy("rule_interpolation", S)
        assert {dm: set(occ) for dm, occ in fit_occ.items()} == {
            dm: set(fit.cv_channel_map) for dm in ("nucleus", "z")
        }
        assert isinstance(sources.cv_result("rule_interpolation", S), ri.CVResult)
        cv_fit = sources.cv_fit("rule_interpolation", S)
        assert sources.cv_fit("rule_interpolation", S) is cv_fit
        assert isinstance(sources.cv_fit("rule_interpolation", S, occupancy=fit_occ), ri.FitResult)
        with pytest.raises(FileNotFoundError, match=r"--phase validate .*stage 'validate'"):
            sources.validation_occupancy(S)
        with pytest.raises(FileNotFoundError, match=r"stage 'validate'"):
            sources.validation(S)

        v = _write_validation(sources, fit)
        merged = sources.validation_occupancy(S)
        assert set(merged["z"]) == set(v.channel_map)
        for mode in fit.cv_channel_map:  # pinned to the fit cache
            np.testing.assert_array_equal(
                merged["z"][mode]["combined"]["occupancy"],
                fit_occ["z"][mode]["combined"]["occupancy"],
            )
        result = sources.validation(S)
        assert result.emd_df["emd"].tolist() == [0.1, 0.3]

        e2e = sources.source_path("rule_interpolation", S)
        expected = {e2e, fit.config_file, v.config_file}
        expected |= {fit.results_dir / "rule_interpolation_cv.pkl"}
        for dm in ("nucleus", "z"):
            for d in (fit.results_dir, v.results_dir):
                expected |= {d / f"{dm}_occupancy.dat", d / f"{dm}_occupancy.meta.json"}
        expected |= {
            fit.results_dir / "validation_result_mixed_rule_validation.dat",
            fit.results_dir / "occupancy_emd_mixed_rule_validation.parquet",
        }
        assert set(sources.inputs) == expected
        assert len(sources.inputs) == len(expected)

    def test_validation_records_the_distance_emd(self, sources):
        fit = _write_fit(sources)
        _write_validation(sources, fit)
        distance_emd = fit.results_dir / "distance_pairwise_emd_mixed_rule_validation.parquet"
        EMD_ROWS.to_parquet(distance_emd)
        assert sources.validation(S).distance_emd_df["emd"].tolist() == [0.1, 0.3]
        assert distance_emd in sources.inputs

    def test_fit_config_must_live_in_its_results_dir(self, sources):
        fit = _write_fit(sources)
        _edit(fit.config_file, output_path="rule_interpolation_replicate_99_v4/elsewhere")
        with pytest.raises(ValueError, match=r"is elsewhere"):
            sources.fit_config("rule_interpolation", S)


class TestLogsAndModuleFunctions:
    def test_read_log(self, proj, sources):
        cfg = sources.distance_config(S)
        log = cfg.results_dir / "logs" / "x.log"
        with pytest.raises(FileNotFoundError, match=r"Log not found"):
            sources.read_log(cfg.results_dir, "x.log")
        log.parent.mkdir(parents=True)
        log.write_text("INFO | a\rb\nc\n")
        assert sources.read_log(cfg.results_dir, "x.log") == ["INFO | a", "b", "c"]
        assert sources.inputs[-1] == log
        with pytest.raises(ValueError, match=r"not under the results root"):
            sources.read_log(proj / "data", "x.log")

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_v4_packing_configs_use_the_canonical_apical_lambda(self, sources, structure):
        path = sources.packing_config_path("condition", structure)
        assert ms.check_canonical_apical_lambda(path) == 0.7

    def test_off_canonical_apical_lambda_raises(self, sources, tmp_path):
        data = json.loads(sources.packing_config_path("condition", S).read_text())
        assert "apical_gradient_1" in GRADIENTS
        cases: dict[str, Any] = {
            "canonical": ["apical_gradient_1"],
            "unknown gradients": ["apical_gradient_typo"],
            "expected one": ["apical_gradient", "apical_gradient_1"],
        }
        inline = json.loads(json.dumps(GRADIENTS["apical_gradient"]))
        inline["weight_mode_settings"] = {**inline["weight_mode_settings"], "decay_length": 1}
        cases["apical λ 1 != canonical 0.7"] = {"apical_gradient": inline}
        for match, gradients in cases.items():
            data["recipe_data"]["apical_gradient"]["gradients"] = gradients
            path = tmp_path / "packing.json"
            path.write_text(json.dumps(data))
            with pytest.raises(ValueError, match=match):
                ms.check_canonical_apical_lambda(path)
        ladder = sources.packing_config_path("ladder", S)
        with pytest.raises(ValueError, match=r"no recipe_data.apical_gradient.gradients"):
            ms.check_canonical_apical_lambda(ladder)
