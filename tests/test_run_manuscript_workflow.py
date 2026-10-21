# mypy: allow-untyped-defs
"""Tests for the manuscript runner (``run_manuscript_workflow``) on a fake project root.

The real v4 manuscript, source and packing configs are copied into ``tmp_path``. Tests about
the runner stub the few panels they draw, so they need no caches; the builders themselves are
tested in ``test_manuscript_panels.py``.
"""

import importlib
import json
import logging
import re
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest

from cellpack_analysis.analysis.workflows import manuscript_config as mc
from cellpack_analysis.analysis.workflows import manuscript_panels as mp
from cellpack_analysis.analysis.workflows import run_analysis_workflow as raw
from cellpack_analysis.analysis.workflows import run_manuscript_workflow as rmw
from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw
from cellpack_analysis.analysis.workflows.manuscript_config import DEFAULT_STEPS

TREE = Path(__file__).parents[1]
CONFIGS = "cellpack_analysis/analysis/workflows/configs"
MANUSCRIPT = f"{CONFIGS}/manuscript/manuscript_replicate_99_v4.json"
STRUCTURES = ("peroxisome", "endosome")
FIG3D = "Fig3/Fig3d_endosome_envelope_rejection_joint.pdf"
S3C = "FigS3/FigS3c_peroxisome_nucleus_envelope_matrix.pdf"
TARGET = "Fig4/Fig4d_endosome_z_interpolated_target.pdf"
STUBBED = ("per_cell_rejection_joint", "biovar_envelope_matrix", "cv_fit")
EPOCH = "946684800"


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


def _runner(root: Path) -> rmw.ManuscriptRunner:
    config = mc.ManuscriptConfig(root / MANUSCRIPT, panels=set(mp.PANELS), edits=set(mp.EDITS))
    return rmw.ManuscriptRunner(config)


@pytest.fixture()
def stubbed(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Stub panels that draw one line and read one config; record what they were called with."""
    calls: dict[str, list[Any]] = {"panel": [], "inputs": []}

    def read(ctx: mp.PanelContext) -> None:
        if ctx.structure is None:
            ctx.sources.biovar_config()
        else:
            ctx.sources.distance_config(ctx.structure)

    def panel(ctx: mp.PanelContext, **args: Any) -> tuple[Any, list[Any]]:
        calls["panel"].append((args, ctx.style is None))
        read(ctx)
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0.2, 0.6])
        ax.set_xlabel("x label")
        ax.set_title("A title")
        return fig, [ax]

    def inputs(ctx: mp.PanelContext, **args: Any) -> dict[str, Any]:
        calls["inputs"].append((args, ctx.style is None))
        read(ctx)
        return {}

    for name in STUBBED:
        monkeypatch.setitem(mp.PANELS, name, panel)
        monkeypatch.setitem(mp.PANEL_INPUTS, name, inputs)
    return calls


def _manifest(runner: rmw.ManuscriptRunner) -> dict[str, Any]:
    return json.loads((runner.out_dir / rmw.MANIFEST_JSON).read_text())


def _only(*paths: str) -> str:
    return "|".join(map(re.escape, paths))


class TestSteps:
    def test_step_table(self, proj):
        runner = _runner(proj)
        assert list(runner.steps) == DEFAULT_STEPS == runner.config.steps
        assert set(rmw.EXTERNAL_STEPS) == {"figure_numbers", "calibration_summary"}

    def test_steps_run_in_workflow_order(self, proj):
        runner = _runner(proj)
        order: list[str] = []

        def recorder(name: str) -> Callable[[], None]:
            return lambda: order.append(name)

        runner.steps = {k: recorder(k) for k in runner.steps}
        runner.run(steps=["manifest", "check_inputs", "figure_assets"])
        assert order == ["check_inputs", "figure_assets", "manifest"]
        order.clear()
        runner.run()
        assert order == DEFAULT_STEPS
        with pytest.raises(ValueError, match=r"Unknown steps \['nope'\]"):
            runner.run(steps=["nope"])

    def test_run_appends_provenance(self, proj):
        runner = _runner(proj)
        runner.steps = {k: (lambda: None) for k in runner.steps}
        runner.run(steps=["manifest"])
        runner.run(steps=["manifest"])
        path = runner.out_dir / "run_provenance.jsonl"
        lines = [json.loads(ln) for ln in path.read_text().splitlines()]
        assert len(lines) == 2
        assert lines[0]["config_file"] == runner.config.config_file.as_posix()
        assert lines[0]["engine_commit"] and lines[0]["config_sha256"]
        assert runner._report_unmanaged(set()) == []  # not an unmanaged file

    def test_a_check_only_run_writes_no_provenance(self, proj):
        runner = _runner(proj)
        runner.steps = {k: (lambda: None) for k in runner.steps}
        runner.run(steps=["check_inputs"])
        assert not runner.out_dir.exists()

    def test_only_selects_assets(self, proj):
        runner = _runner(proj)
        runner.steps = {k: (lambda: None) for k in runner.steps}
        runner.run(only=r"Fig5c_endosome")
        assert [s.path for s in runner.selected] == [
            "Fig5/Fig5c_endosome_nucleus_occupancy_EMD.pdf",
            "Fig5/Fig5c_endosome_z_occupancy_EMD.pdf",
        ]
        with pytest.raises(ValueError):
            runner.run(only="NoSuchAsset")

    def test_unset_source_date_epoch_warns(self, proj, monkeypatch, caplog):
        monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
        caplog.set_level(logging.WARNING, logger=rmw.__name__)
        _runner(proj).run(steps=[])
        assert "SOURCE_DATE_EPOCH is unset" in caplog.text


class TestCheckInputs:
    def test_packing_guards_are_collected(self, proj, stubbed):
        runner = _runner(proj)
        pero = runner.sources.packing_config_path("condition", "peroxisome")
        data = json.loads(pero.read_text())
        data["recipe_data"]["apical_gradient"]["gradients"] = ["apical_gradient_1"]
        pero.write_text(json.dumps(data))
        endo = runner.sources.packing_config_path("condition", "endosome")
        endo.write_text(json.dumps({**json.loads(endo.read_text()), "number_of_replicates": 98}))
        with pytest.raises(RuntimeError) as err:
            runner.run(steps=["check_inputs"], only=_only(FIG3D))
        text = str(err.value)
        assert text.startswith("check_inputs found 2 problem(s):")
        assert "- peroxisome packing configs: " in text and "canonical" in text
        assert "endosome packing configs: " in text
        assert "number_of_replicates 98 != expected 99" in text

    def test_missing_caches_name_their_steps(self, proj):
        runner = _runner(proj)
        only = r"Fig5c_endosome_z|Fig4d_endosome_z_occupancy|FigS10_endosome_lambda"
        with pytest.raises(RuntimeError) as err:
            runner.run(steps=["check_inputs"], only=only)
        text = str(err.value)
        assert "check_inputs found 3 problem(s)" in text
        for needed in ("stage 'fit'", "run_occupancy_analysis", "stage 'ladder'"):
            assert needed in text

    def test_panel_arguments_are_bound(self, proj, stubbed, monkeypatch):
        def needs(ctx: mp.PanelContext, extra: int) -> None:
            raise AssertionError("never drawn")

        monkeypatch.setitem(mp.PANELS, "per_cell_rejection_joint", needs)
        with pytest.raises(RuntimeError, match=r"missing a required argument: 'extra'"):
            _runner(proj).run(steps=["check_inputs"], only=_only(FIG3D, S3C))
        assert len(stubbed["inputs"]) == 1  # S3c still checked

    def test_native_arguments_are_checked_too(self, proj, stubbed, tmp_path):
        runner = _runner(proj)
        runner.run(steps=["check_inputs"], only=_only(TARGET))
        slot = ({"source": "rule_interpolation", "observed_only": True}, False)
        assert stubbed["inputs"] == [slot]
        stubbed["inputs"].clear()
        runner.run(steps=["check_inputs"], only=_only(TARGET), native_dir=tmp_path / "native")
        native = ({"source": "rule_interpolation", "observed_only": False}, True)
        assert stubbed["inputs"] == [slot, native]

    def test_bugs_are_not_collected(self, proj, stubbed, monkeypatch):
        def broken(ctx: mp.PanelContext) -> dict[str, Any]:
            raise RuntimeError("a bug")

        monkeypatch.setitem(mp.PANEL_INPUTS, "per_cell_rejection_joint", broken)
        with pytest.raises(RuntimeError, match=r"^a bug$"):
            _runner(proj).run(steps=["check_inputs"], only=_only(FIG3D))


class TestFigureAssets:
    def test_slot_and_native_records(self, proj, stubbed, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", EPOCH)
        runner = _runner(proj)
        runner.run(steps=["figure_assets", "manifest"], only=_only(FIG3D, S3C))
        assert [style_none for _, style_none in stubbed["panel"]] == [True, False]
        data = _manifest(runner)
        assert data["version"] == rmw.MANIFEST_VERSION and data["outputs"] == {}
        slot, native = data["assets"][FIG3D], data["assets"][S3C]
        assert slot["render"] == "slot" and slot["fonts"].startswith("tick 6")
        assert slot["frames"] == [[pytest.approx(1.633, abs=5e-3), pytest.approx(0.909, abs=5e-3)]]
        assert slot["removed"] == ["x label", "A title"]
        distance = f"{CONFIGS}/distance_analysis/distance_analysis_endosome_replicate_99_v4.json"
        assert slot["inputs"] == [distance]
        assert (native["render"], native["frames"], native["fonts"]) == ("native", [], "base 4.8")
        biovar = f"{CONFIGS}/biological_variation/biological_variation_config.json"
        assert native["inputs"] == [biovar]
        for path in (FIG3D, S3C):
            assert b"D:20000101000000" in (runner.out_dir / path).read_bytes()

        text = (runner.out_dir / rmw.MANIFEST_MD).read_text()
        assert text.startswith("# Manuscript figure assets (replicate_99_v4)\n")
        assert "round 1" not in text
        page = " \N{MULTIPLICATION SIGN} ".join(f"{x:g}" for x in slot["page"])
        frames = " \N{MULTIPLICATION SIGN} ".join(["1", "1.633", "0.909"])
        assert f"| `{FIG3D}` | {page} | {frames} | tick 6" in text
        assert f"| `{S3C}` | " in text and "| native | base 4.8 | — |" in text
        assert f"| `{FIG3D}` | `{distance}` |" in text
        missing = text.split("## Not drawn yet\n\n")[1].splitlines()
        assert len(missing) == len(runner.config.assets) - 2

    def test_pdfs_are_byte_reproducible(self, proj, stubbed, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", EPOCH)
        runner = _runner(proj)
        runner.run(steps=["figure_assets"], only=_only(FIG3D))
        first = (runner.out_dir / FIG3D).read_bytes()
        _runner(proj).run(steps=["figure_assets"], only=_only(FIG3D))
        assert (runner.out_dir / FIG3D).read_bytes() == first

    def test_native_copies(self, proj, stubbed, tmp_path):
        native_dir = tmp_path / "native"
        _runner(proj).run(steps=["figure_assets"], only=_only(TARGET), native_dir=native_dir)
        assert (native_dir / TARGET).read_bytes().startswith(b"%PDF")
        assert [args["observed_only"] for args, _ in stubbed["panel"]] == [True, False]

    def test_manifest_merges_runs_and_drops_stale_records(self, proj, stubbed, caplog):
        runner = _runner(proj)
        runner.run(steps=["figure_assets"], only=_only(FIG3D))
        runner.run(steps=["figure_assets"], only=_only(S3C))
        assert set(_manifest(runner)["assets"]) == {FIG3D, S3C}
        data = _manifest(runner)
        data["assets"]["Old/gone.pdf"] = data["assets"][S3C]
        (runner.out_dir / rmw.MANIFEST_JSON).write_text(json.dumps(data))
        caplog.set_level(logging.WARNING, logger=rmw.__name__)
        runner.run(steps=["figure_assets"], only=_only(FIG3D))
        assert set(_manifest(runner)["assets"]) == {FIG3D, S3C}
        assert "dropping Old/gone.pdf" in caplog.text

    def test_manifest_version_is_checked(self, proj):
        runner = _runner(proj)
        runner.out_dir.mkdir(parents=True)
        (runner.out_dir / rmw.MANIFEST_JSON).write_text(json.dumps({"version": 0}))
        with pytest.raises(ValueError, match=r"version 0 != 1"):
            runner.run(steps=["manifest"])


class TestExternalSteps:
    @pytest.mark.parametrize("step", list(rmw.EXTERNAL_STEPS))
    def test_absent_module_raises_naming_it(self, proj, monkeypatch, step):
        module, func = rmw.EXTERNAL_STEPS[step]
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(NotImplementedError, match=re.escape(f"{module}.{func}")):
            _runner(proj).run(steps=[step])

    def test_other_import_errors_propagate(self, proj, monkeypatch):
        def fail(name: str, package: str | None = None) -> None:
            raise ModuleNotFoundError("No module named 'missing_dep'", name="missing_dep")

        runner = _runner(proj)
        monkeypatch.setattr(importlib, "import_module", fail)
        with pytest.raises(ModuleNotFoundError, match=r"missing_dep"):
            runner.run(steps=["figure_numbers"])

    @pytest.mark.parametrize("step", list(rmw.EXTERNAL_STEPS))
    def test_outputs_are_recorded(self, proj, monkeypatch, caplog, step):
        module_name, func = rmw.EXTERNAL_STEPS[step]
        runner = _runner(proj)
        seen = []

        def write(config: mc.ManuscriptConfig, sources: Any) -> list[Any]:
            seen.append((config, sources))
            config.results_dir.mkdir(parents=True, exist_ok=True)
            for name in ("Fig3_numbers.md", "numbers.json"):
                (config.results_dir / name).write_text("x")
            return [config.results_dir / "Fig3_numbers.md", "numbers.json"]

        module = types.ModuleType(module_name)
        setattr(module, func, write)
        monkeypatch.setitem(sys.modules, module_name, module)
        caplog.set_level(logging.WARNING, logger=rmw.__name__)
        runner.run(steps=[step, "manifest"])
        assert seen == [(runner.config, runner.sources)]
        assert _manifest(runner)["outputs"] == {step: ["Fig3_numbers.md", "numbers.json"]}
        text = (runner.out_dir / rmw.MANIFEST_MD).read_text()
        assert f"| `{step}` | `Fig3_numbers.md`<br>`numbers.json` |" in text
        assert "Unmanaged" not in caplog.text

    def test_outputs_must_stay_in_the_asset_folder(self, proj, monkeypatch):
        module_name, func = rmw.EXTERNAL_STEPS["figure_numbers"]
        module = types.ModuleType(module_name)
        setattr(module, func, lambda config, sources: ["../elsewhere.md"])
        monkeypatch.setitem(sys.modules, module_name, module)
        with pytest.raises(ValueError, match=r"outside"):
            _runner(proj).run(steps=["figure_numbers"])


class TestUnmanaged:
    def test_only_strays_are_reported(self, proj, stubbed, caplog):
        runner = _runner(proj)
        runner.run(steps=["figure_assets"], only=_only(FIG3D))
        for rel in ("Fig3/old_panel.pdf", "Fig3_numbers.md", "notes/x_numbers.md", "numbers.json"):
            (runner.out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (runner.out_dir / rel).write_text("x")
        caplog.set_level(logging.WARNING, logger=rmw.__name__)
        runner.run(steps=["manifest"])
        assert runner._report_unmanaged(set()) == ["Fig3/old_panel.pdf"]
        assert "Unmanaged file in" in caplog.text and "old_panel.pdf" in caplog.text


class TestMain:
    def test_cli(self, proj, stubbed, monkeypatch, tmp_path):
        monkeypatch.setattr(rmw, "setup_logging", lambda level: None)
        argv = ["run_manuscript_workflow", "-c", str(proj / MANUSCRIPT), "--only", _only(TARGET)]
        argv += ["--steps", "figure_assets", "manifest", "--native", str(tmp_path / "native")]
        monkeypatch.setattr(sys, "argv", argv)
        rmw.main()
        out = proj / "results" / "manuscript_figure_assets_v4"
        assert (out / TARGET).exists() and (out / rmw.MANIFEST_MD).exists()
        assert (tmp_path / "native" / TARGET).exists()
        monkeypatch.setattr(sys, "argv", [*argv[:3], "--steps", "nope"])
        with pytest.raises(SystemExit):
            rmw.main()
