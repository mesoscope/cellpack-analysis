# mypy: allow-untyped-defs
"""Tests for the Methods calibration step (``manuscript_calibration``) on a fake project root.

The real v4 manuscript config and calibration source configs are copied into ``tmp_path``.
Calibration caches are built with the real ``summarize_across_cells`` from hand-chosen p-value
counts ``k`` (``p = k / (R + 1)``), so nothing reads or writes the live ``results/``.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cellpack_analysis.analysis.workflows import manuscript_calibration as cal
from cellpack_analysis.analysis.workflows import manuscript_config as mc
from cellpack_analysis.analysis.workflows import manuscript_panels as mp
from cellpack_analysis.analysis.workflows import manuscript_sources as ms
from cellpack_analysis.analysis.workflows import run_analysis_workflow as raw
from cellpack_analysis.analysis.workflows import run_manuscript_workflow as rmw
from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw
from cellpack_analysis.lib.stats import summarize_across_cells

TREE = Path(__file__).parents[1]
CONFIGS = "cellpack_analysis/analysis/workflows/configs"
MANUSCRIPT = f"{CONFIGS}/manuscript/manuscript_replicate_99_v4.json"
STRUCTURES = ("peroxisome", "endosome")
MODES = ("random", "apical_gradient")
MEASURES = ("nucleus", "z")
N_CELLS = 6
R = 98


def _calibration_config(structure: str) -> str:
    name = f"distance_analysis_{structure}_replicate_99_v4_calibration.json"
    return f"{CONFIGS}/distance_analysis/{name}"


@pytest.fixture()
def proj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "proj"
    for rel in (MANUSCRIPT, *(_calibration_config(s) for s in STRUCTURES)):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((TREE / rel).read_bytes())  # no mode bits copied
    for module in (mc, raw, riw):
        monkeypatch.setattr(module, "get_project_root", lambda: root)
    monkeypatch.setattr(mc, "get_results_path", lambda: root / "results")
    return root


@pytest.fixture()
def config(proj: Path) -> mc.ManuscriptConfig:
    config = mc.ManuscriptConfig(proj / MANUSCRIPT, panels=set(mp.PANELS), edits=set(mp.EDITS))
    config.expected["n_cells"] = dict.fromkeys(STRUCTURES, N_CELLS)
    return config


@pytest.fixture()
def sources(config: mc.ManuscriptConfig) -> ms.PaperSources:
    return ms.PaperSources(config)


def _ks(n: int = N_CELLS, seed: int = 0) -> dict[tuple[str, str], list[int]]:
    """Counts ``k`` per (mode, scope); each column holds a 1 so the lattice is 1/(R+1)."""
    rng = np.random.default_rng(seed)
    out = {}
    for mode in MODES:
        for scope in (*MEASURES, "joint"):
            ks = [int(k) for k in rng.integers(6, 100, size=n)]
            ks[0], ks[1] = 1, 7
            out[(mode, scope)] = ks
    return out


def _result(
    ks: dict[tuple[str, str], list[int]] | None = None, r: int = R, **changes: Any
) -> dict[str, Any]:
    """A calibration result shaped like ``per_cell_envelope_test``'s return."""
    ks = ks or _ks()
    modes = list(dict.fromkeys(mode for mode, _ in ks))
    measures = [s for s in dict.fromkeys(scope for _, scope in ks) if s != "joint"]
    n_cells = len(next(iter(ks.values())))
    cells = []
    for i in range(n_cells):
        cell = {}
        for mode in modes:
            cell[mode] = {
                "per_distance_measure": {
                    dm: {"pval": ks[(mode, dm)][i] / (r + 1), "supremum_sign": 1} for dm in measures
                },
                "joint": {"pval": ks[(mode, "joint")][i] / (r + 1), "supremum_sign": 1},
            }
        cells.append(cell)
    result = {
        "summary": summarize_across_cells(cells, alpha=0.05),
        "per_cell_results": cells,
        "cell_ids": [f"c{i}" for i in range(n_cells)],
        "baseline_mode": "baseline",
        "null_modes": modes,
        "distance_measures": measures,
        "alpha": 0.05,
        "statistic": "intdev",
        "envelope_type": "pointwise",
        "calibration_holdout_index": 0,
    }
    return {**result, **changes}


def _cache_path(sources: ms.PaperSources, structure: str) -> Path:
    cfg = raw.AnalysisConfig(str(sources.source_path("distance_calibration", structure)), False)
    return cfg.results_dir / "per_cell_envelope_results_calibration.dat"


def _write(sources: ms.PaperSources, structure: str, result: dict[str, Any]) -> Path:
    path = _cache_path(sources, structure)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return path


def _write_both(sources: ms.PaperSources) -> None:
    for i, structure in enumerate(STRUCTURES):
        _write(sources, structure, _result(_ks(seed=i)))


class TestConfigAndCacheNames:
    def test_v4_config_gives_r_98(self):
        config = json.loads((TREE / MANUSCRIPT).read_text())
        assert config["condition"] == "replicate_99_v4"
        assert config["expected"]["n_replicates"] - 1 == 98
        assert config["expected"]["envelope"]["alpha"] == 0.05

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_calibration_config_matches_the_suffix(self, sources, structure):
        path = sources.source_path("distance_calibration", structure)
        assert path.name == f"distance_analysis_{structure}_replicate_99_v4_calibration.json"
        cfg = raw.AnalysisConfig(str(path), create_dirs=False)
        assert cfg.suffix == cal.CALIBRATION_SUFFIX
        assert cfg.envelope_test_params["calibration_holdout_index"] == 0

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_calibration_shares_the_main_output_folder(self, structure):
        folder = f"{CONFIGS}/distance_analysis/distance_analysis_{structure}_replicate_99_v4"
        main = json.loads((TREE / f"{folder}.json").read_text())
        calibration = json.loads((TREE / f"{folder}_calibration.json").read_text())
        assert calibration["output_path"] == main["output_path"]
        assert calibration["suffix"] == cal.CALIBRATION_SUFFIX
        assert main.get("suffix", "") == ""


class TestMeasureEnvelopeReplicates:
    @pytest.mark.parametrize("r", [98, 99, 199])
    def test_reads_r_off_the_lattice(self, r):
        assert cal.measure_envelope_replicates(_result(_ks(), r=r)) == r

    def test_one_odd_count_pins_the_lattice(self):
        ks = {key: [k * 2 for k in v] for key, v in _ks().items()}
        ks[("random", "z")][2] = 3  # one odd count is enough to pin the lattice
        assert cal.measure_envelope_replicates(_result(ks, r=198)) == 198

    def test_p_values_off_the_lattice_raise(self):
        result = _result()
        result["summary"]["joint_pvals"].iloc[0, 0] = 0.1234567
        with pytest.raises(ValueError, match="not multiples of one"):
            cal.measure_envelope_replicates(result)

    def test_mixed_replicate_counts_raise(self):
        # one cell tested against 96 replicates, the rest against 98
        result = _result()
        result["summary"]["joint_pvals"].iloc[0, 0] = 5 / 97
        with pytest.raises(ValueError, match="replicate counts differ"):
            cal.measure_envelope_replicates(result)

    @pytest.mark.parametrize("bad", [np.nan, 0.0, 1.5, -0.1])
    def test_invalid_p_values_raise(self, bad):
        result = _result()
        result["summary"]["per_distance_measure_pvals"].iloc[0, 0] = bad
        with pytest.raises(ValueError, match=r"finite and in \(0, 1\]"):
            cal.measure_envelope_replicates(result)


class TestCheckCalibrationResult:
    def test_returns_measured_r(self, config):
        assert cal.check_calibration_result(config, "peroxisome", _result()) == 98

    def test_not_a_calibration_run(self, config):
        result = _result(calibration_holdout_index=None)
        with pytest.raises(ValueError, match="not a hold-one-out calibration"):
            cal.check_calibration_result(config, "peroxisome", result)

    def test_alpha_must_equal_the_config(self, config):
        with pytest.raises(ValueError, match=r"alpha 0\.01 != expected 0\.05"):
            cal.check_calibration_result(config, "peroxisome", _result(alpha=0.01))

    def test_cell_count(self, config):
        config.expected["n_cells"]["peroxisome"] = 305
        with pytest.raises(ValueError, match=f"{N_CELLS} cells != expected 305"):
            cal.check_calibration_result(config, "peroxisome", _result())

    def test_distance_measures(self, config):
        with pytest.raises(ValueError, match="distance measures"):
            cal.check_calibration_result(
                config, "peroxisome", _result(distance_measures=["nucleus"])
            )

    def test_wrong_r_is_an_error_not_a_shifted_floor(self, config):
        # p on the 1/100 lattice means R = 99: 99 replicates, none held out
        with pytest.raises(ValueError, match=r"measured R = 99 .*expected 98"):
            cal.check_calibration_result(config, "peroxisome", _result(_ks(), r=99))

    def test_r_follows_the_configured_replicates(self, config):
        config.expected["n_replicates"] = 100
        assert cal.check_calibration_result(config, "peroxisome", _result(_ks(), r=99)) == 99


class TestCalibrationTable:
    @pytest.fixture()
    def table(self, config) -> pd.DataFrame:
        ks = {
            ("random", "nucleus"): [1, 1, 4, 5, 50, 60],
            ("random", "z"): [2, 3, 90, 91, 92, 93],
            ("random", "joint"): [30, 31, 32, 33, 34, 35],
            ("apical_gradient", "nucleus"): [1, 1, 1, 1, 1, 1],
            ("apical_gradient", "z"): [10, 20, 30, 40, 50, 60],
            ("apical_gradient", "joint"): [5, 6, 7, 8, 9, 99],
        }
        return cal.calibration_table(config, "peroxisome", _result(ks))

    def test_columns_and_order(self, table):
        assert list(table.columns) == cal.TABLE_COLUMNS
        assert list(zip(table.null_mode, table.scope, strict=True)) == [
            (mode, scope) for mode in MODES for scope in (*MEASURES, "joint")
        ]
        assert set(table.structure) == {"peroxisome"}
        assert set(table.n_cells) == {N_CELLS}
        assert set(table.n_envelope_replicates) == {R}
        assert set(table.holdout_index) == {0}

    def test_raw_counts_and_rates(self, table):
        # p < 0.05 at R = 98 means k <= 4
        row = table.iloc[0]
        assert (row.n_raw_rejected, row.raw_rejection) == (3, 3 / 6)
        assert row.at_floor == 2 / 6
        assert row.min_p == 1 / 99
        assert table.iloc[1].n_raw_rejected == 2
        assert table.iloc[2].n_raw_rejected == 0
        assert set(table.expected_raw) == {4 / 99}

    def test_bh_counts(self, table):
        # six cells at the floor: q = 1/99 < 0.05 for all of them
        row = table.iloc[3]
        assert (row.n_bh_rejected, row.bh_rejection) == (N_CELLS, 1.0)
        # row 0 has two floor cells: p_(2) = 1/99 <= 2 * 0.05 / 6, so BH rejects both
        assert table.n_bh_rejected.tolist() == [2, 0, 0, 6, 0, 0]

    def test_counts_are_integers_of_the_rates(self, table):
        assert table.n_raw_rejected.dtype.kind == "i"
        assert np.allclose(table.raw_rejection * table.n_cells, table.n_raw_rejected)


class TestRangeTable:
    def test_groups_scopes_and_labels(self, config):
        tables = [
            cal.calibration_table(config, s, _result(_ks(seed=i))) for i, s in enumerate(STRUCTURES)
        ]
        table = pd.concat(tables, ignore_index=True)
        ranges = cal.range_table(config, table)
        assert ranges[["organelles", "scopes"]].values.tolist() == [
            [o, s] for o in (*STRUCTURES, "both") for s in cal.SCOPE_SETS
        ]
        n_rows = ranges.set_index(["organelles", "scopes"]).n_rows
        assert n_rows[("peroxisome", "all scopes")] == 6
        assert n_rows[("peroxisome", "per measure")] == 4
        assert n_rows[("peroxisome", "joint")] == 2
        assert n_rows[("both", "all scopes")] == 12
        both = ranges[ranges.organelles == "both"].set_index("scopes")
        assert both.loc["all scopes", "min"] == table.raw_rejection.min()
        assert both.loc["all scopes", "max"] == table.raw_rejection.max()
        # the sub-ranges nest inside the overall one
        for scope_set in ("per measure", "joint"):
            assert both.loc["all scopes", "min"] <= both.loc[scope_set, "min"]
            assert both.loc[scope_set, "max"] <= both.loc["all scopes", "max"]
        # labels use the config's row names; the argmax row really holds the maximum
        label = both.loc["all scopes", "max_at"]
        structure, rule, scope = label.split(", ")
        mode = next(k for k, v in config.row_labels.items() if v == rule)
        hit = table[(table.structure == structure) & (table.null_mode == mode)]
        assert hit[hit.scope == scope].raw_rejection.iloc[0] == both.loc["all scopes", "max"]


class TestMethodsMarkdown:
    @pytest.fixture()
    def md(self, config) -> str:
        ks = {
            ("random", "nucleus"): [1, 2, 50, 60, 70, 80],
            ("random", "z"): [30, 31, 32, 33, 34, 35],
            ("random", "joint"): [30, 31, 32, 33, 34, 99],
            ("apical_gradient", "nucleus"): [1, 2, 3, 4, 50, 60],
            ("apical_gradient", "z"): [30, 31, 32, 33, 34, 99],
            ("apical_gradient", "joint"): [30, 31, 32, 33, 34, 99],
        }
        table = pd.concat(
            [cal.calibration_table(config, s, _result(ks)) for s in STRUCTURES], ignore_index=True
        )
        return cal.methods_markdown(config, table)

    def test_methods_sentence(self, md):
        # raw rates run from 0 to 4/6 over the 12 rows; BH rejects nothing at n = 6
        assert "were 0\u20130.67 across modeled rules, distance measures, and organelles" in md
        assert (
            "(expected: 0.04), and no cells were rejected in this calibration after correction."
            in md
        )
        assert "Uncorrected rejection rates at p < 0.05" in md

    def test_states_scopes_and_r(self, md):
        times = "\u00d7"
        assert f"spans all 12 rows: 2 organelles {times} 2 rules {times} 3 scopes" in md
        assert "(nucleus, z and joint)" in md
        assert "The 8 per-measure rows alone give the same range." in md
        assert "Replicate 0 of each cell and null model is the pseudo-observed curve" in md
        assert "R = 98 replicates (99 per cell and rule, one held out)" in md
        assert "every p is a multiple of 1/99" in md
        assert "Expected under a calibrated test: raw rate 0.0404 (4/99" in md
        assert "BH q < 0.05 rejected 0 of 72 cell-level tests; the rate is 0 in 12 of 12 rows" in md
        assert "at least 20% of a row's cells" in md

    def test_scope_note_when_joint_widens_the_range(self, config):
        # the per-measure rows stay at 1/6, the joint rows reach 4/6
        ks = _ks()
        ks[("random", "joint")] = [1, 2, 3, 4, 50, 60]
        table = pd.concat(
            [cal.calibration_table(config, s, _result(ks)) for s in STRUCTURES], ignore_index=True
        )
        md = cal.methods_markdown(config, table)
        assert "The 8 per-measure rows alone give" in md
        assert "alone give the same range" not in md

    def test_bh_rejections_are_reported_not_hidden(self, config):
        ks = _ks()
        ks[("random", "nucleus")] = [1] * N_CELLS
        table = pd.concat(
            [cal.calibration_table(config, s, _result(ks)) for s in STRUCTURES], ignore_index=True
        )
        md = cal.methods_markdown(config, table)
        assert "12 of 72 cell-level tests were rejected after correction" in md
        assert "no cells were rejected" not in md
        assert (
            "BH q < 0.05 rejected 12 of 72 cell-level tests; the rate is 0 in 10 of 12 rows" in md
        )

    def test_all_rows_tables_use_row_labels(self, md):
        assert "**peroxisome (n = 6):**" in md
        assert "| Unbiased | nucleus | 0.3333 (2) | 0.0000 (0) |" in md
        assert "| Apical bias | joint |" in md
        assert md.endswith("\n") and not md.endswith("\n\n")

    def test_inputs_table(self, config, tmp_path):
        path = tmp_path / "x.dat"
        path.write_bytes(b"")
        table = cal.calibration_table(config, "peroxisome", _result())
        md = cal.methods_markdown(config, table, [path])
        assert "| Source | Modified (UTC) |" in md
        assert f"`{path.as_posix()}`" in md


class TestWriteCalibrationSummary:
    def test_writes_both_files_under_results_dir(self, config, sources):
        _write_both(sources)
        written = cal.write_calibration_summary(config, sources)
        assert [p.name for p in written] == [cal.SHEET_NAME, cal.CSV_NAME]
        for path in written:
            assert path.is_file()
            assert path.is_relative_to(config.results_dir / cal.OUTPUT_DIR)
        md = written[0].read_text(encoding="utf-8")
        assert md.startswith("# Methods (envelope calibration)")
        assert "(replicate_99_v4)" in md
        assert "Engine pin `" in md
        assert "peroxisome / endosome: n = 6 / 6 cells (asserted)" in md

    def test_csv_holds_every_row(self, config, sources):
        _write_both(sources)
        _, csv = cal.write_calibration_summary(config, sources)
        frame = pd.read_csv(csv, float_precision="round_trip")
        assert list(frame.columns) == cal.TABLE_COLUMNS
        assert len(frame) == len(STRUCTURES) * len(MODES) * 3
        pd.testing.assert_frame_equal(
            frame, cal.calibration_tables(config, sources), check_dtype=False
        )

    def test_reads_only_the_calibration_caches_and_records_them(self, config, sources):
        _write_both(sources)
        cal.write_calibration_summary(config, sources)
        names = {p.name for p in sources.inputs}
        assert "per_cell_envelope_results_calibration.dat" in names
        assert len([p for p in sources.inputs if p.suffix == ".dat"]) == len(STRUCTURES)
        md = (config.results_dir / cal.OUTPUT_DIR / cal.SHEET_NAME).read_text(encoding="utf-8")
        for structure in STRUCTURES:
            assert f"distance_analysis_{structure}_replicate_99_v4_calibration.json" in md
        assert md.count("per_cell_envelope_results_calibration.dat") == len(STRUCTURES)

    def test_is_deterministic(self, config, sources):
        _write_both(sources)
        first = [p.read_bytes() for p in cal.write_calibration_summary(config, sources)]
        again = [
            p.read_bytes() for p in cal.write_calibration_summary(config, ms.PaperSources(config))
        ]
        assert first == again

    def test_missing_cache_names_the_stage(self, config, sources):
        _write(sources, "peroxisome", _result())
        with pytest.raises(FileNotFoundError, match=r"run_paper.sh stage 'calibration'"):
            cal.write_calibration_summary(config, sources)

    def test_wrong_replicate_count_fails_before_writing(self, config, sources):
        _write(sources, "peroxisome", _result(_ks(), r=99))
        _write(sources, "endosome", _result(_ks(seed=1)))
        with pytest.raises(ValueError, match=r"measured R = 99"):
            cal.write_calibration_summary(config, sources)
        assert not (config.results_dir / cal.OUTPUT_DIR).exists()

    def test_alpha_mismatch_in_the_cache_is_refused(self, config, sources):
        _write(sources, "peroxisome", _result(alpha=0.01))
        _write(sources, "endosome", _result())
        with pytest.raises(ValueError, match="test parameters"):
            cal.write_calibration_summary(config, sources)

    def test_a_non_calibration_cache_is_refused(self, config, sources):
        _write(sources, "peroxisome", _result(calibration_holdout_index=None))
        _write(sources, "endosome", _result())
        with pytest.raises(ValueError, match="test parameters"):
            cal.write_calibration_summary(config, sources)


class TestRunnerStep:
    def test_calibration_summary_step_records_its_outputs(self, config, sources):
        _write_both(sources)
        runner = rmw.ManuscriptRunner(config, sources)
        runner.run(steps=["calibration_summary", "manifest"])
        manifest = json.loads((runner.out_dir / rmw.MANIFEST_JSON).read_text())
        assert manifest["outputs"]["calibration_summary"] == sorted(
            [f"{cal.OUTPUT_DIR}/{cal.SHEET_NAME}", f"{cal.OUTPUT_DIR}/{cal.CSV_NAME}"]
        )
        text = (runner.out_dir / rmw.MANIFEST_MD).read_text(encoding="utf-8")
        assert f"`{cal.OUTPUT_DIR}/{cal.SHEET_NAME}`" in text
