# mypy: allow-untyped-defs
"""Tests for the manuscript workflow config (``manuscript_config``) and its v4 JSON."""

import copy
import json
from pathlib import Path

import pytest

from cellpack_analysis.analysis.workflows import manuscript_config as mc
from cellpack_analysis.lib.figure_layout import TextStyle

CONFIG = (
    Path(__file__).parents[1]
    / "cellpack_analysis/analysis/workflows/configs/manuscript/manuscript_replicate_99_v4.json"
)

EXPECTED_PATHS = [
    "Fig2/Fig2f_peroxisome_envelope_rejection_joint.pdf",
    "FigS3/FigS3c_peroxisome_nucleus_envelope_matrix.pdf",
    "FigS3/FigS3e_peroxisome_joint_envelope_matrix.pdf",
    "Fig3/Fig3a_peroxisome_distance_distributions.pdf",
    "Fig3/Fig3a_endosome_distance_distributions.pdf",
    "Fig3/Fig3b_peroxisome_distance_EMD.pdf",
    "Fig3/Fig3b_endosome_distance_EMD.pdf",
    "Fig3/Fig3c_peroxisome_distance_KS.pdf",
    "Fig3/Fig3c_endosome_distance_KS.pdf",
    "Fig3/Fig3d_peroxisome_envelope_rejection_joint.pdf",
    "Fig3/Fig3d_endosome_envelope_rejection_joint.pdf",
    "FigS4/FigS4a_peroxisome_distance_distributions.pdf",
    "FigS4/FigS4a_endosome_distance_distributions.pdf",
    "FigS4/FigS4b_peroxisome_distance_EMD.pdf",
    "FigS4/FigS4b_endosome_distance_EMD.pdf",
    "FigS4/FigS4c_peroxisome_distance_KS.pdf",
    "FigS4/FigS4c_endosome_distance_KS.pdf",
    "Fig4/Fig4d_peroxisome_nucleus_occupancy_ratio_per_rule.pdf",
    "Fig4/Fig4d_peroxisome_z_occupancy_ratio_per_rule.pdf",
    "Fig4/Fig4d_endosome_nucleus_occupancy_ratio_per_rule.pdf",
    "Fig4/Fig4d_endosome_z_occupancy_ratio_per_rule.pdf",
    "Fig4/Fig4d_peroxisome_nucleus_interpolated_target.pdf",
    "Fig4/Fig4d_peroxisome_z_interpolated_target.pdf",
    "Fig4/Fig4d_endosome_nucleus_interpolated_target.pdf",
    "Fig4/Fig4d_endosome_z_interpolated_target.pdf",
    "Fig5/Fig5b_peroxisome_nucleus_occupancy_mixed_rule.pdf",
    "Fig5/Fig5b_peroxisome_z_occupancy_mixed_rule.pdf",
    "Fig5/Fig5b_endosome_nucleus_occupancy_mixed_rule.pdf",
    "Fig5/Fig5b_endosome_z_occupancy_mixed_rule.pdf",
    "Fig5/Fig5c_peroxisome_nucleus_occupancy_EMD.pdf",
    "Fig5/Fig5c_peroxisome_z_occupancy_EMD.pdf",
    "Fig5/Fig5c_endosome_nucleus_occupancy_EMD.pdf",
    "Fig5/Fig5c_endosome_z_occupancy_EMD.pdf",
    "FigS5/FigS5_peroxisome_nucleus_pairwise_EMD_matrix.pdf",
    "FigS5/FigS5_peroxisome_z_pairwise_EMD_matrix.pdf",
    "FigS5/FigS5_endosome_nucleus_pairwise_EMD_matrix.pdf",
    "FigS5/FigS5_endosome_z_pairwise_EMD_matrix.pdf",
    "FigS6/FigS6a_peroxisome_envelope_rejection_by_measure.pdf",
    "FigS7/FigS7a_endosome_envelope_rejection_by_measure.pdf",
    "FigS6/FigS6b_peroxisome_envelope_example_median_cell.pdf",
    "FigS7/FigS7b_endosome_envelope_example_median_cell.pdf",
    "FigS6/FigS6c_peroxisome_envelope_example_most_rejected_cell.pdf",
    "FigS7/FigS7c_endosome_envelope_example_most_rejected_cell.pdf",
    "FigS8/FigS8a_peroxisome_CV_MSE.pdf",
    "FigS9/FigS9a_endosome_CV_MSE.pdf",
    "FigS8/FigS8b_peroxisome_AIC_BIC_weights.pdf",
    "FigS9/FigS9b_endosome_AIC_BIC_weights.pdf",
    "FigS8/FigS8c_peroxisome_NNLS_coefficients.pdf",
    "FigS9/FigS9c_endosome_NNLS_coefficients.pdf",
    "FigS8/FigS8d_peroxisome_validation_summary.pdf",
    "FigS9/FigS9d_endosome_validation_summary.pdf",
    "FigS10/FigS10_peroxisome_nucleus_joint_fit.pdf",
    "FigS10/FigS10_peroxisome_z_joint_fit.pdf",
    "FigS10/FigS10_endosome_nucleus_joint_fit.pdf",
    "FigS10/FigS10_endosome_z_joint_fit.pdf",
    "FigS10/FigS10_peroxisome_lambda_ladder_profile.pdf",
    "FigS10/FigS10_endosome_lambda_ladder_profile.pdf",
]

PANELS = {
    "biovar_rejection_joint",
    "biovar_envelope_matrix",
    "distance_pdfs",
    "distance_emd",
    "distance_ks",
    "per_cell_rejection_joint",
    "occupancy_per_rule",
    "cv_fit",
    "validation_occupancy",
    "occupancy_emd",
    "pairwise_emd_matrix",
    "rejection_by_measure",
    "envelope_examples",
    "cv_mse",
    "aic_weights",
    "cv_coefficients",
    "validation_summary",
    "ladder_profile",
}
EDITS = {
    "biovar_rejection_bars",
    "fig3a_labels",
    "fig3b_labels",
    "fig3c_labels",
    "fig3d_labels",
    "pull_texts",
    "drop_ylabels",
    "sentence_case_ylabels",
    "target_labels",
    "fit_ylim_top",
    "measure_title",
    "emd_axis_per_pair",
    "legend_above",
    "set_ylim_headroom",
    "shares_to_side",
    "remove_collections",
}


@pytest.fixture(autouse=True)
def roots(tmp_path, monkeypatch):
    """Point the project and results roots at tmp so nothing touches the tree."""
    monkeypatch.setattr(mc, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(mc, "get_results_path", lambda: tmp_path / "results")
    return tmp_path


@pytest.fixture()
def raw() -> dict:
    return json.loads(CONFIG.read_text())


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "manuscript.json"
    path.write_text(json.dumps(config))
    return path


def _load(tmp_path: Path, config: dict) -> mc.ManuscriptConfig:
    return mc.ManuscriptConfig(_write(tmp_path, config))


def _by_path(cfg: mc.ManuscriptConfig) -> dict[str, mc.AssetSpec]:
    return {spec.path: spec for spec in cfg.assets}


class TestV4Config:
    def test_expands_to_the_57_published_assets_in_order(self):
        cfg = mc.ManuscriptConfig(CONFIG)
        assert [spec.path for spec in cfg.assets] == EXPECTED_PATHS
        assert len(cfg.assets) == 57 and len(cfg.config["assets"]) == 24

    def test_identity_paths_and_defaults(self, roots):
        cfg = mc.ManuscriptConfig(CONFIG)
        assert cfg.tag == "v4"
        assert cfg.results_dir == roots / "results" / "manuscript_figure_assets_v4"
        assert cfg.config_root == roots / mc.DEFAULT_CONFIG_ROOT
        assert cfg.steps == mc.DEFAULT_STEPS
        assert cfg.render == mc.DEFAULT_RENDER
        assert cfg.numbers == {"sheets": mc.DEFAULT_SHEETS, "write_json": True}
        assert set(cfg.sources) == set(mc.SOURCE_KEYS)
        assert cfg.expected["n_cells"] == {"peroxisome": 305, "endosome": 442}
        assert cfg.expected["occupancy_meta"] == {
            "minimum_distance_um": -0.108,
            "available_minimum_distance_um": -1.0,
            "num_points": 250,
            "bandwidth": 0.2,
        }

    def test_axes_of_expanded_assets(self):
        specs = _by_path(mc.ManuscriptConfig(CONFIG))
        fig2f = specs["Fig2/Fig2f_peroxisome_envelope_rejection_joint.pdf"]
        assert (fig2f.structure, fig2f.measure) == (None, None)
        s6a = specs["FigS6/FigS6a_peroxisome_envelope_rejection_by_measure.pdf"]
        assert (s6a.structure, s6a.measure) == ("peroxisome", None)
        s5 = specs["FigS5/FigS5_endosome_z_pairwise_EMD_matrix.pdf"]
        assert (s5.structure, s5.measure) == ("endosome", "z")

    def test_frame_overrides_land(self):
        specs = _by_path(mc.ManuscriptConfig(CONFIG))
        assert specs["Fig3/Fig3b_peroxisome_distance_EMD.pdf"].frames == {
            "kind": "grid",
            "w": 0.838,
            "h": 0.450,
            "xpitch": 1.334,
        }
        assert specs["Fig3/Fig3b_endosome_distance_EMD.pdf"].frames == {
            "kind": "grid",
            "w": 0.781,
            "h": 0.450,
            "xpitch": 1.334,
        }
        pero = specs["FigS4/FigS4b_peroxisome_distance_EMD.pdf"].frames
        endo = specs["FigS4/FigS4b_endosome_distance_EMD.pdf"].frames
        assert pero == {"kind": "grid", "w": 1.167, "h": 0.870, "xpitch": 1.673}
        assert endo == {"kind": "grid", "w": 1.127, "h": 0.884, "xpitch": 1.710}

    def test_edit_param_overrides_land(self):
        specs = _by_path(mc.ManuscriptConfig(CONFIG))
        for s in ("peroxisome", "endosome"):
            target = specs[f"Fig4/Fig4d_{s}_nucleus_interpolated_target.pdf"]
            assert target.edit_params["target_labels"] == {"drop_xlabel": s == "peroxisome"}
            assert target.edit_params["fit_ylim_top"] == {"top": 2.0}
            for m in ("nucleus", "z"):
                fig5b = specs[f"Fig5/Fig5b_{s}_{m}_occupancy_mixed_rule.pdf"]
                keep = (s, m) == ("peroxisome", "nucleus")
                assert fig5b.edit_params["drop_ylabels"] == {"keep_first": keep}
                assert fig5b.edit_params["fit_ylim_top"] == {"top": 1.5 if m == "z" else 2.0}
                s10 = specs[f"FigS10/FigS10_{s}_{m}_joint_fit.pdf"]
                assert s10.edit_params["shares_to_side"] == {"show": m == "z"}
                assert s10.edits == ["shares_to_side", "remove_collections"]

    def test_native_panel_args(self):
        specs = _by_path(mc.ManuscriptConfig(CONFIG))
        target = specs["Fig4/Fig4d_endosome_z_interpolated_target.pdf"]
        assert target.panel_args == {"source": "rule_interpolation", "observed_only": True}
        assert target.native_panel_args == {"source": "rule_interpolation", "observed_only": False}
        fig3a = specs["Fig3/Fig3a_endosome_distance_distributions.pdf"]
        assert fig3a.native_panel_args == fig3a.panel_args == {"source": "distance"}
        assert fig3a.native_panel_args is not fig3a.panel_args

    def test_native_assets(self):
        specs = _by_path(mc.ManuscriptConfig(CONFIG))
        s3e = specs["FigS3/FigS3e_peroxisome_joint_envelope_matrix.pdf"]
        assert s3e.render == "native"
        assert (s3e.style, s3e.frames, s3e.edits, s3e.edit_params) == (None, None, [], {})
        assert s3e.panel_args["distance_measure"] is None
        assert s3e.rc == {"font.size": 4.8}
        assert s3e.savefig == {
            "format": "pdf",
            "bbox_inches": "tight",
            "dpi": "figure",
            "transparent": False,
        }
        assert [s.path for s in specs.values() if s.render == "native"] == EXPECTED_PATHS[1:3]

    def test_biovar_edit_params(self):
        fig2f = _by_path(mc.ManuscriptConfig(CONFIG))[EXPECTED_PATHS[0]]
        params = fig2f.edit_params["biovar_rejection_bars"]
        assert params["labels"]["variable_count"] == "Count\nvaried"
        assert params["bar_height"] == 0.72
        assert params["light_colors"] == {"variable_count": "#e5e4cf", "shape": "#d1c3c2"}
        assert params["xticks"] == [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert fig2f.frames == {"kind": "grid", "w": 1.534, "h": 0.797}

    def test_row_frames(self):
        s9d = _by_path(mc.ManuscriptConfig(CONFIG))["FigS9/FigS9d_endosome_validation_summary.pdf"]
        assert s9d.frames == {
            "kind": "row",
            "xs": [0, 1.742, 3.551, 5.290],
            "ws": [1.354, 1.354, 1.354, 1.334],
            "h": 1.153,
        }
        assert s9d.edit_params == {"set_ylim_headroom": {"axis_index": 3, "top": 1.4}}

    def test_specs_share_no_mutable_state(self):
        cfg = mc.ManuscriptConfig(CONFIG)
        specs = _by_path(cfg)
        pero = specs["Fig5/Fig5b_peroxisome_nucleus_occupancy_mixed_rule.pdf"]
        endo = specs["Fig5/Fig5b_endosome_nucleus_occupancy_mixed_rule.pdf"]
        assert pero.frames is not endo.frames
        assert pero.edit_params["drop_ylabels"] is not endo.edit_params["drop_ylabels"]
        assert pero.edits is not endo.edits
        assert "w" in cfg.config["assets"][12]["frames"]  # raw entry untouched

    def test_styles(self):
        styles = mc.ManuscriptConfig(CONFIG).styles
        assert styles["base"] == TextStyle()
        assert styles["supp"] == styles["base"]
        assert styles["fig5c"] == TextStyle(
            tick=6,
            label=8,
            title=6.5,
            bold_title=True,
            lw=1.0,
            spine=0.5,
            ticklen=2.0,
            tickpad=1.5,
            labelpad=2.0,
            titlepad=2.0,
        )
        assert styles["s10_fit"].title == 10 and styles["s10_profile"].title == 8
        assert styles["s8a"] == TextStyle(legend=7, suptitle=7.5)

    def test_every_name_resolves_in_the_registries(self):
        cfg = mc.ManuscriptConfig(CONFIG, panels=PANELS, edits=EDITS)
        cfg.check_registries(PANELS, EDITS)
        assert {spec.panel for spec in cfg.assets} == PANELS
        assert {e for spec in cfg.assets for e in spec.edits} == EDITS
        assert {spec.style for spec in cfg.assets} - {None} <= set(cfg.styles)

    def test_unknown_registry_names_raise(self):
        cfg = mc.ManuscriptConfig(CONFIG)
        with pytest.raises(ValueError, match=r"panel 'cv_fit'.*edit 'legend_above'"):
            cfg.check_registries(PANELS - {"cv_fit"}, EDITS - {"legend_above"})
        with pytest.raises(ValueError, match=r"panel 'ladder_profile'"):
            mc.ManuscriptConfig(CONFIG, panels=PANELS - {"ladder_profile"}, edits=EDITS)

    def test_select_assets(self):
        cfg = mc.ManuscriptConfig(CONFIG)
        selected = cfg.select_assets("Fig5c_peroxisome_nucleus")
        assert [s.path for s in selected] == ["Fig5/Fig5c_peroxisome_nucleus_occupancy_EMD.pdf"]
        assert len(cfg.select_assets()) == 57
        assert len(cfg.select_assets(r"^FigS10/")) == 6
        with pytest.raises(ValueError, match=r"matches no asset"):
            cfg.select_assets("Fig9")


class TestConfigValidation:
    @pytest.mark.parametrize(
        "output_path", ["/abs/manuscript", "../escape", "a/../../b", "", "a\\b"]
    )
    def test_bad_output_path_is_rejected(self, tmp_path, raw, output_path):
        raw["output_path"] = output_path
        with pytest.raises(ValueError, match=r"output_path"):
            _load(tmp_path, raw)

    def test_output_path_defaults_to_name(self, tmp_path, raw):
        del raw["output_path"]
        assert _load(tmp_path, raw).results_dir == tmp_path / "results" / raw["name"]

    def test_unknown_step_is_fatal(self, tmp_path, raw):
        raw["steps"] = ["check_inputs", "figure_asset"]
        with pytest.raises(ValueError, match=r"Unknown steps"):
            _load(tmp_path, raw)

    def test_duplicate_step_is_fatal(self, tmp_path, raw):
        raw["steps"] = ["manifest", "manifest"]
        with pytest.raises(ValueError, match=r"duplicates"):
            _load(tmp_path, raw)

    def test_steps_keep_their_order(self, tmp_path, raw):
        raw["steps"] = ["figure_numbers", "check_inputs"]
        assert _load(tmp_path, raw).steps == ["figure_numbers", "check_inputs"]

    def test_unknown_top_level_key_is_fatal(self, tmp_path, raw):
        raw["asset"] = []
        with pytest.raises(ValueError, match=r"unknown keys \['asset'\]"):
            _load(tmp_path, raw)

    @pytest.mark.parametrize("key", ["condition", "sources", "expected", "styles", "assets"])
    def test_missing_required_key_is_fatal(self, tmp_path, raw, key):
        del raw[key]
        with pytest.raises(ValueError, match=rf"missing required keys \['{key}'\]"):
            _load(tmp_path, raw)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mc.ManuscriptConfig(tmp_path / "absent.json")

    def test_unknown_source_key_is_fatal(self, tmp_path, raw):
        raw["sources"]["distances"] = "x/{structure}.json"
        with pytest.raises(ValueError, match=r"sources: unknown keys"):
            _load(tmp_path, raw)

    def test_source_templates_take_only_structure(self, tmp_path, raw):
        raw["sources"]["distance"] = "distance_analysis/{structure}_{measure}.json"
        with pytest.raises(ValueError, match=r"only \{structure\}"):
            _load(tmp_path, raw)

    def test_absolute_source_path_is_rejected(self, tmp_path, raw):
        raw["packing_configs"]["ladder"] = "/allen/x/{structure}.json"
        with pytest.raises(ValueError, match=r"packing_configs.ladder"):
            _load(tmp_path, raw)

    def test_absolute_config_root_is_rejected(self, tmp_path, raw):
        raw["config_root"] = "/allen/configs"
        with pytest.raises(ValueError, match=r"config_root"):
            _load(tmp_path, raw)

    def test_n_cells_must_cover_the_structures(self, tmp_path, raw):
        raw["expected"]["n_cells"] = {"peroxisome": 305}
        with pytest.raises(ValueError, match=r"n_cells"):
            _load(tmp_path, raw)

    def test_expected_envelope_needs_all_keys(self, tmp_path, raw):
        del raw["expected"]["envelope"]["alpha"]
        with pytest.raises(ValueError, match=r"expected.envelope"):
            _load(tmp_path, raw)

    def test_unknown_expected_key(self, tmp_path, raw):
        raw["expected"]["n_seeds"] = 99
        with pytest.raises(ValueError, match=r"expected: unknown keys"):
            _load(tmp_path, raw)

    def test_unknown_style_field_is_fatal(self, tmp_path, raw):
        raw["styles"]["fig3"]["fontsize"] = 8
        with pytest.raises(ValueError, match=r"styles.fig3.*fontsize"):
            _load(tmp_path, raw)

    def test_styles_need_a_base(self, tmp_path, raw):
        del raw["styles"]["base"]
        with pytest.raises(ValueError, match=r"'base'"):
            _load(tmp_path, raw)

    def test_unknown_style_name_in_an_asset(self, tmp_path, raw):
        raw["assets"][3]["style"] = "fig3x"
        with pytest.raises(ValueError, match=r"style 'fig3x'"):
            _load(tmp_path, raw)

    def test_unknown_render_key(self, tmp_path, raw):
        raw["render"] = {"dpi": 600, "format": "png"}
        with pytest.raises(ValueError, match=r"render: unknown keys"):
            _load(tmp_path, raw)

    def test_render_merges_over_defaults(self, tmp_path, raw):
        raw["render"] = {"dpi": 600}
        assert _load(tmp_path, raw).render == {**mc.DEFAULT_RENDER, "dpi": 600}

    def test_unknown_sheet(self, tmp_path, raw):
        raw["numbers"]["sheets"] = ["Fig3", "FigS11"]
        with pytest.raises(ValueError, match=r"Unknown numbers sheets"):
            _load(tmp_path, raw)

    def test_template_vars_keys_must_be_structures(self, tmp_path, raw):
        raw["template_vars"]["golgi"] = {"mc_fig": "FigS11"}
        with pytest.raises(ValueError, match=r"template_vars: unknown keys"):
            _load(tmp_path, raw)

    def test_template_var_may_not_shadow_an_axis(self, tmp_path, raw):
        raw["template_vars"]["peroxisome"]["measure"] = "z"
        with pytest.raises(ValueError, match=r"bad variable 'measure'"):
            _load(tmp_path, raw)


def _entry(**kwargs) -> dict:
    base = {
        "path": "F/F_{structure}_{measure}.pdf",
        "panel": "p",
        "style": "s",
        "frames": {"kind": "grid", "w": 1.0, "h": 0.5},
    }
    return {**base, **kwargs}


def _expand(*entries: dict, **kwargs) -> list[mc.AssetSpec]:
    return mc.expand_assets(
        list(entries),
        ["a", "b"],
        ["m", "n"],
        kwargs.pop("template_vars", {"a": {"v": "A"}, "b": {"v": "B"}}),
        **kwargs,
    )


class TestExpandAssets:
    def test_expansion_order_and_axes(self):
        specs = _expand(_entry(), _entry(path="G/G.pdf"), _entry(path="H/{v}_{structure}.pdf"))
        assert [(s.path, s.structure, s.measure) for s in specs] == [
            ("F/F_a_m.pdf", "a", "m"),
            ("F/F_a_n.pdf", "a", "n"),
            ("F/F_b_m.pdf", "b", "m"),
            ("F/F_b_n.pdf", "b", "n"),
            ("G/G.pdf", None, None),
            ("H/A_a.pdf", "a", None),
            ("H/B_b.pdf", "b", None),
        ]

    def test_override_order_structure_then_pair_then_wildcard(self):
        entry = _entry(
            edits=["e"],
            edit_params={"e": {"x": 0, "y": 0, "z": 0}},
            overrides={
                "a": {"edit_params": {"e": {"x": 1}}},
                "a/m": {"edit_params": {"e": {"y": 2}}},
                "*/m": {"edit_params": {"e": {"z": 3}}, "frames": {"h": 0.25}},
            },
        )
        got = {s.path: (s.edit_params["e"], (s.frames or {})["h"]) for s in _expand(entry)}
        assert got == {
            "F/F_a_m.pdf": ({"x": 1, "y": 2, "z": 3}, 0.25),
            "F/F_a_n.pdf": ({"x": 1, "y": 0, "z": 0}, 0.5),
            "F/F_b_m.pdf": ({"x": 0, "y": 0, "z": 3}, 0.25),
            "F/F_b_n.pdf": ({"x": 0, "y": 0, "z": 0}, 0.5),
        }

    def test_edits_override_replaces_the_list(self):
        entry = _entry(edits=["e1", "e2"], overrides={"b": {"edits": ["e3"]}})
        assert [s.edits for s in _expand(entry)] == [["e1", "e2"]] * 2 + [["e3"]] * 2

    def test_overrides_setting_one_value_twice_raise(self):
        entry = _entry(
            overrides={"a": {"frames": {"w": 2.0}}, "*/m": {"frames": {"w": 3.0}}},
        )
        with pytest.raises(ValueError, match=r"both set \('frames', 'w'\)"):
            _expand(entry)

    @pytest.mark.parametrize(
        ("path", "key"),
        [
            ("F/F_{measure}.pdf", "a"),
            ("F/F_{structure}.pdf", "a/m"),
            ("F/F_{structure}.pdf", "*/m"),
            ("F/F_{structure}_{measure}.pdf", "c"),
            ("F/F_{structure}_{measure}.pdf", "a/q"),
            ("F/F_{structure}_{measure}.pdf", "*"),
            ("F/F_{structure}_{measure}.pdf", "a/*"),
        ],
    )
    def test_inapplicable_override_keys_raise(self, path, key):
        with pytest.raises(ValueError, match=r"does not apply"):
            _expand(_entry(path=path, overrides={key: {"frames": {"w": 2.0}}}))

    def test_unknown_override_field_raises(self):
        with pytest.raises(ValueError, match=r"unknown keys \['style'\]"):
            _expand(_entry(overrides={"a": {"style": "t"}}))

    def test_duplicate_paths_raise(self):
        with pytest.raises(ValueError, match=r"Duplicate asset paths"):
            _expand(_entry(path="F/x.pdf"), _entry(path="F/x.pdf"))

    def test_template_var_needs_structure(self):
        with pytest.raises(ValueError, match=r"need a \{structure\} field"):
            _expand(_entry(path="{v}/x.pdf"))

    def test_missing_template_var_raises(self):
        with pytest.raises(ValueError, match=r"template_vars\['b'\] lacks \['v'\]"):
            _expand(_entry(path="{v}/{structure}.pdf"), template_vars={"a": {"v": "A"}})

    def test_unknown_placeholder_raises(self):
        with pytest.raises(ValueError, match=r"lacks \['figure'\]"):
            _expand(_entry(path="{figure}/{structure}.pdf"))

    @pytest.mark.parametrize("path", ["F/{structure!r}.pdf", "F/{structure:>5}.pdf", "F/{}.pdf"])
    def test_formatted_placeholders_raise(self, path):
        with pytest.raises(ValueError, match=r"plain named placeholders"):
            _expand(_entry(path=path))

    @pytest.mark.parametrize("path", ["/F/x.pdf", "../x.pdf", "F/x.png", "F//x.pdf", "./x.pdf"])
    def test_bad_asset_paths_raise(self, path):
        with pytest.raises(ValueError, match=r"path"):
            _expand(_entry(path=path))

    def test_unknown_asset_key_raises(self):
        with pytest.raises(ValueError, match=r"unknown keys \['frame'\]"):
            _expand(_entry(frame={}))

    def test_path_and_panel_are_required(self):
        bad = _entry()
        del bad["panel"]
        with pytest.raises(ValueError, match=r"'panel' is required"):
            _expand(bad)

    @pytest.mark.parametrize(
        "frames",
        [
            {"kind": "grid", "w": 1.0},
            {"kind": "grid", "w": 1.0, "h": 0.5, "pitch": 1.0},
            {"kind": "grid", "w": 1.0, "h": 0.5, "ncols": 0},
            {"kind": "grid", "w": -1.0, "h": 0.5},
            {"kind": "grid", "w": True, "h": 0.5},
            {"kind": "row", "xs": [0, 1], "ws": [1], "h": 1.0},
            {"kind": "row", "xs": [], "ws": [], "h": 1.0},
            {"kind": "box", "w": 1.0, "h": 0.5},
        ],
    )
    def test_bad_frames_raise(self, frames):
        with pytest.raises(ValueError, match=r"frames"):
            _expand(_entry(frames=frames))

    def test_slot_needs_style_and_frames(self):
        with pytest.raises(ValueError, match=r"need a 'style'"):
            _expand(_entry(style=None))
        no_frames = _entry()
        del no_frames["frames"]
        with pytest.raises(ValueError, match=r"'frames' object"):
            _expand(no_frames)

    def test_slot_takes_no_rc_or_savefig(self):
        with pytest.raises(ValueError, match=r"native assets only"):
            _expand(_entry(rc={"font.size": 5}))

    @pytest.mark.parametrize(
        "extra",
        [
            {"style": "s"},
            {"frames": {"kind": "grid", "w": 1.0, "h": 0.5}},
            {"edits": ["e"]},
        ],
    )
    def test_native_takes_no_slot_settings(self, extra):
        entry = {"path": "N/n.pdf", "panel": "p", "render": "native", **extra}
        with pytest.raises(ValueError, match=r"native assets take no"):
            _expand(entry)

    def test_native_frames_override_raises(self):
        entry = {
            "path": "N/{structure}.pdf",
            "panel": "p",
            "render": "native",
            "overrides": {"a": {"frames": {"w": 1.0}}},
        }
        with pytest.raises(ValueError, match=r"entry without frames"):
            _expand(entry)

    def test_bad_render_raises(self):
        with pytest.raises(ValueError, match=r"render must be"):
            _expand(_entry(render="png"))

    def test_edit_params_must_name_listed_edits(self):
        with pytest.raises(ValueError, match=r"not in 'edits': \['f'\]"):
            _expand(_entry(edits=["e"], edit_params={"f": {}}))

    def test_duplicate_edits_raise(self):
        with pytest.raises(ValueError, match=r"duplicates"):
            _expand(_entry(edits=["e", "e"]))

    def test_registry_checks_are_optional_and_strict(self):
        entry = _entry(path="F/x.pdf", edits=["e"])
        assert _expand(entry)[0].panel == "p"
        assert _expand(entry, panels={"p"}, edits={"e"}, styles={"s"})
        with pytest.raises(ValueError, match=r"edit 'e'"):
            _expand(entry, panels={"p"}, edits=set(), styles={"s"})
        with pytest.raises(ValueError, match=r"style 's'"):
            _expand(entry, styles={"t"})

    def test_entries_are_not_mutated(self):
        entry = _entry(overrides={"a": {"frames": {"w": 2.0}}})
        before = copy.deepcopy(entry)
        _expand(entry)
        assert entry == before


class TestCheckUnder:
    def test_lexical_containment(self, tmp_path):
        root = tmp_path / "results"
        assert mc.check_under(root / "a" / "b", root, "x") == root / "a" / "b"
        for bad in (root, root / ".." / "other", tmp_path / "elsewhere", root / "a/../../b"):
            with pytest.raises(ValueError, match=r"not under the results root"):
                mc.check_under(bad, root, "x")

    def test_symlinks_are_not_resolved(self, tmp_path):
        target = tmp_path / "real_results"
        target.mkdir()
        root = tmp_path / "results"
        root.symlink_to(target)
        assert mc.check_under(root / "a", root, "x") == root / "a"
        with pytest.raises(ValueError):
            mc.check_under(target / "a", root, "x")
