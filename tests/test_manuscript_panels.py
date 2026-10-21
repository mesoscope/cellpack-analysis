# mypy: allow-untyped-defs
"""Tests for the manuscript panel builders and slot edits (``manuscript_panels``).

The real v4 manuscript, source and packing configs are copied into ``tmp_path``. The caches the
distance and bio-variation panels read are written there by the pipeline's own steps from
synthetic distances, then drawn through the runner; nothing touches the live ``results/``.
"""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cellpack_analysis.analysis.workflows import manuscript_config as mc
from cellpack_analysis.analysis.workflows import manuscript_panels as mp
from cellpack_analysis.analysis.workflows import manuscript_sources as ms
from cellpack_analysis.analysis.workflows import run_analysis_workflow as raw
from cellpack_analysis.analysis.workflows import run_manuscript_workflow as rmw
from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw
from cellpack_analysis.lib.figure_layout import TextStyle

TREE = Path(__file__).parents[1]
CONFIGS = "cellpack_analysis/analysis/workflows/configs"
MANUSCRIPT = f"{CONFIGS}/manuscript/manuscript_replicate_99_v4.json"
STRUCTURES = ("peroxisome", "endosome")
S = "endosome"
N_CELLS = 3
# Every asset drawn from the T5 caches (plus the per-cell test), for one structure.
T5_ONLY = r"Fig2f|FigS3|(Fig3|FigS4|FigS5|FigS7[bc])\w*_endosome"
T5_ASSETS = 14
QUIET_PLOTS = (
    "plot_distance_distributions",
    "plot_emd_comparisons",
    "plot_pairwise_emd_matrix",
    "plot_ks_test_results",
    "plot_per_cell_rejection_bars",
    "plot_per_cell_envelope_examples",
    "plot_pairwise_envelope_matrix",
    "plot_per_dm_rejection_bars",
    "plot_per_dm_envelopes_overlaid",
)
QUIET_LOGS = (
    "log_central_tendencies_for_distance_distributions",
    "log_pairwise_emd_central_tendencies",
    "log_central_tendencies_for_ks",
)


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


def _edit(path: Path, **changes: Any) -> None:
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def _set_n_cells(root: Path, n: int) -> None:
    path = root / MANUSCRIPT
    data = json.loads(path.read_text())
    data["expected"]["n_cells"] = dict.fromkeys(STRUCTURES, n)
    path.write_text(json.dumps(data))


@pytest.fixture()
def proj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "proj"
    for rel in _copied_paths():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((TREE / rel).read_bytes())  # no mode bits copied
    for rel in ("distance", "distance_supp", "biological_variation"):
        for s in STRUCTURES if rel != "biological_variation" else [None]:
            template = json.loads((root / MANUSCRIPT).read_text())["sources"][rel]
            _edit(root / CONFIGS / template.format(structure=s), num_workers=1, n_bootstrap=20)
    _set_n_cells(root, N_CELLS)
    for module in (mc, raw, riw):
        monkeypatch.setattr(module, "get_project_root", lambda: root)
    monkeypatch.setattr(mc, "get_results_path", lambda: root / "results")
    monkeypatch.setattr(raw.distance, "PROJECT_ROOT", root)  # its log lines need it
    return root


def _manuscript(root: Path) -> mc.ManuscriptConfig:
    return mc.ManuscriptConfig(root / MANUSCRIPT, panels=set(mp.PANELS), edits=set(mp.EDITS))


def _source_config(root: Path, key: str, structure: str | None = None) -> raw.AnalysisConfig:
    template = json.loads((root / MANUSCRIPT).read_text())["sources"][key]
    path = root / CONFIGS / template.format(structure=structure)
    return raw.AnalysisConfig(str(path), create_dirs=False)


def _distances(cfg: raw.AnalysisConfig, observed: str | None, n_seeds: int = 5) -> dict:
    """Synthetic ``{measure: {mode: {cell: {seed: distances}}}}``; one seed for ``observed``."""
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for dm in cfg.distance_measures:
        out[dm] = {}
        for mode in cfg.packing_modes:
            seeds = ["0"] if mode == observed else [str(i) for i in range(n_seeds)]
            out[dm][mode] = {
                f"c{i}": {k: rng.uniform(0.5, 5.0, 40) for k in seeds} for i in range(N_CELLS)
            }
    return out


def _run_steps(cfg: raw.AnalysisConfig, steps: list[str], observed: str | None) -> None:
    """Run pipeline steps with their plots and logs silenced; they write the caches."""
    runner = raw.AnalysisRunner()
    runner.shared_data["all_distance_dict"] = _distances(cfg, observed)
    with pytest.MonkeyPatch.context() as patch:
        for name in QUIET_PLOTS:
            patch.setattr(raw.visualization, name, lambda **_: (None, None))
        for name in QUIET_LOGS:
            patch.setattr(raw.distance, name, lambda **_: None)
        for step in steps:
            getattr(runner, f"_{step}")(cfg)


def _write_distance_caches(root: Path, structure: str = S) -> None:
    cfg = _source_config(root, "distance", structure)
    steps = ["plot_distance_distributions", "run_emd_analysis", "run_ks_analysis"]
    _run_steps(cfg, [*steps, "run_per_cell_envelope_test"], cfg.baseline_mode)
    supp = _source_config(root, "distance_supp", structure)
    _run_steps(supp, steps, supp.baseline_mode)


def _write_biovar_cache(root: Path) -> None:
    _run_steps(_source_config(root, "biological_variation"), ["run_pairwise_envelope_test"], None)


def _write_per_cell_cache(root: Path, structure: str = S) -> None:
    cfg = _source_config(root, "distance", structure)
    _run_steps(cfg, ["run_per_cell_envelope_test"], cfg.baseline_mode)


# ── Registries ───────────────────────────────────────────────────────────────────────


class TestRegistries:
    def test_names(self):
        assert len(mp.PANELS) == 18
        assert len(mp.EDITS) == 16
        assert set(mp.PANEL_INPUTS) == set(mp.PANELS)
        assert set(mp.EDIT_INPUTS) <= set(mp.EDITS)

    def test_config_names_resolve(self, proj):
        config = _manuscript(proj)
        assert len(config.assets) == 57
        config.check_registries(mp.PANELS, mp.EDITS)
        assert {spec.panel for spec in config.assets} == set(mp.PANELS)
        assert {e for spec in config.assets for e in spec.edits} <= set(mp.EDITS)

    def test_every_asset_binds_its_arguments(self, proj):
        config = mc.ManuscriptConfig(proj / MANUSCRIPT)
        ctx = mp.PanelContext(None, "endosome", "z")  # type: ignore[arg-type]
        for spec in config.assets:
            for args in (spec.panel_args, spec.native_panel_args):
                inspect.signature(mp.PANELS[spec.panel]).bind(ctx, **args)
                inspect.signature(mp.PANEL_INPUTS[spec.panel]).bind(ctx, **args)
            for name in spec.edits:
                params = spec.edit_params.get(name, {})
                inspect.signature(mp.EDITS[name]).bind(None, [], ctx, **params)
                if name in mp.EDIT_INPUTS:
                    inspect.signature(mp.EDIT_INPUTS[name]).bind(ctx, **params)

    def test_context_guards(self):
        ctx = mp.PanelContext(None, None, None)  # type: ignore[arg-type]
        for attr, match in (("s", "structure"), ("m", "measure"), ("text_style", "slot style")):
            with pytest.raises(ValueError, match=match):
                getattr(ctx, attr)
        with pytest.raises(ValueError, match=r"source must be one of"):
            mp.in_distance_pdfs(ctx, source="occupancy")


# ── Edits on plain figures ───────────────────────────────────────────────────────────


def _axes(n: int = 2) -> tuple[Any, list[Any]]:
    fig, axs = plt.subplots(1, n)
    axes = list(np.ravel(axs))
    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1.8])
        ax.set_xlabel(f"Distance to nucleus {i} (\N{MICRO SIGN}m)")
        ax.set_ylabel(f"Rule\n{i}")
        ax.set_title(f"t{i}")
        ax.text(0.5, 0.5, f"p = 0.0{i}")
    return fig, axes


CTX = mp.PanelContext(None, S, "z", TextStyle(legend=6.0))  # type: ignore[arg-type]


class TestEdits:
    def teardown_method(self):
        plt.close("all")

    def test_fig3a_labels(self):
        fig, axes = _axes()
        removed = mp.fig3a_labels(fig, axes, CTX)
        assert removed[:2] == ["Rule 0", "Rule 1"]
        assert {"p = 0.00", "t0", "p = 0.01", "t1"} <= set(removed)
        assert [ax.get_ylabel() for ax in axes] == ["", ""]
        assert {ax.get_xlabel() for ax in axes} == {"Distance (\N{GREEK SMALL LETTER MU}m)"}

    def test_fig3b_fig3c_fig3d_labels(self):
        fig, axes = _axes()
        assert mp.fig3b_labels(fig, axes, CTX)[0] == "x-axis rule names"
        assert [ax.get_ylabel() for ax in axes] == ["Rule\n0", ""]
        fig, axes = _axes()
        mp.fig3c_labels(fig, axes, CTX)
        assert [ax.get_ylabel() for ax in axes] == ["Fraction similar", ""]
        fig, axes = _axes(1)
        removed = mp.fig3d_labels(fig, axes, CTX)
        assert removed[0].startswith("Distance to nucleus 0")
        assert list(axes[0].get_xticks()) == [0, 0.5, 1.0]
        assert axes[0].get_xlabel() == ""

    def test_measure_title_and_target_labels(self):
        fig, axes = _axes(1)
        mp.measure_title(fig, axes, CTX)
        assert axes[0].get_title() == "Z"
        mp.target_labels(fig, axes, CTX, drop_xlabel=True)
        assert (axes[0].get_xlabel(), axes[0].get_ylabel()) == ("", "")

    def test_generic_wrappers(self):
        fig, axes = _axes()
        assert mp.e_fit_ylim_top(fig, axes, CTX, top=1.0, axis_index=1) == ["y range 0\N{EN DASH}2"]
        assert axes[1].get_ylim() == (0, 2) and axes[0].get_ylim()[1] < 2
        mp.e_drop_ylabels(fig, axes, CTX, keep_first=True)
        assert [ax.get_ylabel() for ax in axes] == ["Rule\n0", ""]
        assert "t0" in mp.e_pull_texts(fig, axes, CTX)
        axes[0].fill_between([0, 1], [0, 0], [1, 1])
        assert mp.e_remove_collections(fig, axes, CTX) == ["envelopes"]
        assert not axes[0].collections
        axes[0].plot([0, 1], [0, 1], label="mode")
        axes[0].legend()
        assert mp.e_legend_above(fig, axes, CTX) == []
        legend = axes[0].get_legend()
        assert not legend.get_frame_on() and legend.get_texts()[0].get_fontsize() == 6.0
        with pytest.raises(ValueError, match=r"slot style"):
            mp.e_legend_above(fig, axes, mp.PanelContext(None, S, "z"))  # type: ignore[arg-type]


class FakeSources:
    """The few ``PaperSources`` calls the S10 edit makes."""

    def __init__(self, n_cells: int) -> None:
        self.config = SimpleNamespace(expected={"n_cells": {S: 3}})
        self.n_cells = n_cells
        self.calls: list[str] = []

    def fit_config(self, kind, structure):
        self.calls.append(kind)
        return SimpleNamespace(baseline_mode="RAB5A")

    def fit_occupancy(self, kind, structure):
        cells: dict[str, dict] = {f"c{i}": {} for i in range(self.n_cells)}
        return {"z": {"RAB5A": {"individual": cells}}}

    def cv_result(self, kind, structure):
        return None

    def cv_fit(self, kind, structure):
        return SimpleNamespace(relative_contributions_joint={"random": 0.6, "apical_gradient": 0.4})


class TestSharesToSide:
    def test_shares_are_listed_beside_the_panel(self):
        fig, axes = _axes(1)
        sources = FakeSources(3)
        ctx = mp.PanelContext(sources, S, "z", TextStyle(legend=6.0))  # type: ignore[arg-type]
        removed = mp.shares_to_side(fig, axes, ctx, show=True)
        assert removed == ["p = 0.00"] and axes[0].get_title() == "t0"
        notes = [t.get_text() for t in axes[0].texts]
        assert notes == ["Unbiased: 60%", "Apical: 40%"]
        assert sources.calls == ["ladder"]
        fig, axes = _axes(1)
        assert mp.shares_to_side(fig, axes, ctx, show=False) == ["p = 0.00"]
        assert not axes[0].texts and sources.calls == ["ladder"]

    def test_cell_count_is_checked(self):
        ctx = mp.PanelContext(FakeSources(2), S, "z")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"ladder z occupancy for endosome holds 2 cells"):
            mp.in_shares_to_side(ctx)


# ── Builders on caches the pipeline wrote ────────────────────────────────────────────


class TestBuildersOnPipelineCaches:
    def test_missing_caches_name_their_steps(self, proj):
        runner = rmw.ManuscriptRunner(_manuscript(proj))
        with pytest.raises(RuntimeError) as err:
            runner.run(steps=["check_inputs"], only=T5_ONLY)
        text = str(err.value)
        for step in (
            "plot_distance_distributions",
            "run_emd_analysis",
            "run_ks_analysis",
            "run_per_cell_envelope_test",
            "stage 'biovar'",
        ):
            assert step in text
        assert f"check_inputs found {T5_ASSETS} problem(s)" in text
        assert not (proj / "results" / "manuscript_figure_assets_v4").exists()

    def test_per_cell_cell_count_is_checked(self, proj):
        _write_per_cell_cache(proj)
        _set_n_cells(proj, N_CELLS + 1)
        runner = rmw.ManuscriptRunner(_manuscript(proj))
        with pytest.raises(RuntimeError, match=r"holds 3 cells; expected 4") as err:
            runner.run(steps=["check_inputs"], only=r"(Fig3d|FigS7[bc])_endosome")
        assert "3 problem(s)" in str(err.value)

    def test_draws_every_t5_asset(self, proj, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "946684800")
        _write_distance_caches(proj)
        _write_biovar_cache(proj)
        config = _manuscript(proj)
        runner = rmw.ManuscriptRunner(config, ms.PaperSources(config))
        runner.run(steps=["check_inputs", "figure_assets", "manifest"], only=T5_ONLY)
        assert len(runner.selected) == T5_ASSETS

        out = config.results_dir
        records = json.loads((out / rmw.MANIFEST_JSON).read_text())["assets"]
        assert set(records) == {spec.path for spec in runner.selected}
        for spec in runner.selected:
            pdf = (out / spec.path).read_bytes()
            assert pdf.startswith(b"%PDF") and b"D:20000101000000" in pdf
            r = records[spec.path]
            if spec.render == "native":
                assert (r["frames"], r["fonts"]) == ([], "base 4.8")
            else:
                assert r["frames"] and all(len(f) == 2 for f in r["frames"])
            assert all(not p.startswith("/") for p in r["inputs"])

        def inputs(name: str) -> set[str]:
            return {Path(p).name for p in records[name]["inputs"]}

        pdfs = {"distance_pdfs_kde.dat", "distance_pdfs_kde.meta.json"}
        assert pdfs <= inputs("Fig3/Fig3a_endosome_distance_distributions.pdf")
        assert pdfs <= inputs("FigS4/FigS4a_endosome_distance_distributions.pdf")
        assert "distance_pairwise_emd.parquet" in inputs("Fig3/Fig3b_endosome_distance_EMD.pdf")
        assert "ks_bootstrap_vs_RAB5A.parquet" in inputs("Fig3/Fig3c_endosome_distance_KS.pdf")
        s5 = inputs("FigS5/FigS5_endosome_z_pairwise_EMD_matrix.pdf")
        assert {"distance_pairwise_emd.parquet", *pdfs} <= s5
        per_cell = "per_cell_envelope_results.dat"
        assert per_cell in inputs("FigS7/FigS7b_endosome_envelope_example_median_cell.pdf")
        biovar = {"pairwise_envelope_results.dat", "pairwise_envelope_results.meta.json"}
        assert biovar <= inputs("Fig2/Fig2f_peroxisome_envelope_rejection_joint.pdf")
        supp = records["FigS4/FigS4c_endosome_distance_KS.pdf"]["inputs"]
        assert any("/supp/ks_bootstrap_vs_RAB5A.parquet" in p for p in supp)

        fig2f = records["Fig2/Fig2f_peroxisome_envelope_rejection_joint.pdf"]
        assert fig2f["removed"] and len(fig2f["frames"]) == 1
        assert len(records["FigS5/FigS5_endosome_nucleus_pairwise_EMD_matrix.pdf"]["frames"]) == 25
        text = (out / rmw.MANIFEST_MD).read_text()
        assert "## Not drawn yet" in text
        assert "`Fig5/Fig5c_endosome_z_occupancy_EMD.pdf`" in text
