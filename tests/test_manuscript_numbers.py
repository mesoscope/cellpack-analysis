# mypy: allow-untyped-defs
# ruff: noqa: RUF001
"""Tests for the numbers sheets (``manuscript_numbers``) on synthetic caches.

Pure helpers are tested directly. The sheet builders run on synthetic ``StructureData`` and a
stub source, and ``write_figure_numbers`` writes into a fake project root, so nothing reads or
writes the live ``results/``.
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from cellpack_analysis.analysis.workflows import manuscript_config as mc
from cellpack_analysis.analysis.workflows import manuscript_numbers as mn
from cellpack_analysis.analysis.workflows import run_analysis_workflow as raw
from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw
from cellpack_analysis.lib import rule_interpolation as ri

TREE = Path(__file__).parents[1]
MANUSCRIPT = (
    "cellpack_analysis/analysis/workflows/configs/manuscript/manuscript_replicate_99_v4.json"
)
RULES = ["random", "nucleus_gradient", "membrane_gradient", "apical_gradient"]
MIXED = "interpolated"
DMS = ("nucleus", "z")
R = 99


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mc.ManuscriptConfig:
    root = tmp_path / "proj"
    dst = root / MANUSCRIPT
    dst.parent.mkdir(parents=True)
    dst.write_bytes((TREE / MANUSCRIPT).read_bytes())
    for module in (mc, raw, riw):
        monkeypatch.setattr(module, "get_project_root", lambda: root)
    monkeypatch.setattr(mc, "get_results_path", lambda: root / "results")
    return mc.ManuscriptConfig(dst)


# ── helpers ──────────────────────────────────────────────────────────────────────


def test_formatters():
    assert mn.msd([1.0, 2.0, 3.0]) == "2.00 ± 1.00"
    assert mn.msdm([1.0, 2.0, 6.0], nd=1) == "3.0 ± 2.6 (2.0)"
    assert mn.fr((0.5, 0.25, 0.25)) == "0.500 (+0.250 / −0.250)"
    labels = {"a": "A", "b": "B", "c": "C"}
    vals = {"a": 2.0, "b": 1.0, "c": 3.0}
    assert mn.order(vals, labels) == "B < A < C"
    assert mn.order(vals, labels, reverse=True) == "C > A > B"
    assert mn.rank(vals, "a") == 2 and mn.rank(vals, "c", reverse=True) == 1
    assert mn.ladder_name("nucleus_gradient_0pt1", {}) == "Nucleus λ = 0.1"
    assert mn.ladder_name("random", {"random": "Unbiased"}) == "Unbiased"
    lab = {"random": "Unbiased", MIXED: "Mixed rule"}
    assert mn.model_label("single:random", lab, MIXED) == "Unbiased"
    assert mn.model_label("mixed_rule", lab, MIXED) == "Mixed rule"
    assert mn.model_label("null", lab, MIXED) == "Null"
    assert mn._signed(-1.0) == "−1.0" and mn._signed(-0.108) == "−0.108"
    assert mn.count_word(5) == "five" and mn.count_word(12) == "12"


def test_log_parsers():
    prefix = "2026-09-29 03:35:00 | x | INFO | "
    dm_log = [
        prefix + "Distance measure: nucleus",
        prefix + "  SLC25A17: 1.81 ± 1.87",
        prefix + "Distance measure: z",
        prefix + "  SLC25A17: -0.50 ± 2.41",
    ]
    assert mn.parse_dm_log(dm_log) == {
        ("nucleus", "SLC25A17"): (1.81, 1.87),
        ("z", "SLC25A17"): (-0.5, 2.41),
    }
    emd_log = [prefix + "  baseline vs shape [z]: 2.24 ± 0.92 (median: 2.1, 95% CI: x)"]
    assert mn.parse_emd_log(emd_log) == {("baseline", "shape", "z"): (2.24, 0.92)}
    cv_log = [
        prefix + "Full reconstruction MSE",
        prefix + "  nucleus  joint  0.5",
        prefix + "other",
        prefix + "Full reconstruction MSE",
        prefix + "  nucleus  joint  0.0076",
        prefix + "  z  individual  0.0190",
    ]
    assert mn.recon_mse(cv_log) == {("joint", "nucleus"): 0.0076, ("individual", "z"): 0.019}
    with pytest.raises(ValueError, match="Full reconstruction MSE"):
        mn.recon_mse([prefix + "nothing"])


def _emd_frame(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for dm in DMS:
            rows.append(("obs", "random", f"c{i}", f"c{i}", dm, 0.1 * i))
            rows.append(("apical", "obs", f"c{i}", f"c{i}", dm, 1 + i))  # baseline on side 2
            rows.append(("obs", "random", f"c{i}", f"c{(i + 1) % n}", dm, 9.0))  # cross-cell
            rows.append(("random", "random", f"c{i}", f"c{i}", dm, 9.0))  # intra-mode
    return pd.DataFrame(
        rows,
        columns=[
            "packing_mode_1",
            "packing_mode_2",
            "cell_id_1",
            "cell_id_2",
            "distance_measure",
            "emd",
        ],
    )


def test_same_cell_emd():
    out = mn.same_cell_emd(_emd_frame(), "obs", 3, DMS)
    np.testing.assert_allclose(out[("random", "nucleus")], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(out[("apical", "z")], [1, 2, 3])
    assert len(out[("random", "pooled")]) == 6
    with pytest.raises(ValueError, match="expected 4"):
        mn.same_cell_emd(_emd_frame(), "obs", 4, DMS)
    df = _emd_frame()
    with pytest.raises(ValueError, match="Duplicate"):
        mn.same_cell_emd(pd.concat([df, df.iloc[:1]]), "obs", 3, DMS)


def test_lattice_and_rates():
    assert mn.alpha_index(0.05, R) == 5
    assert mn.expected_uncorrected_rate(0.05, R) == pytest.approx(0.04)
    assert mn.bh_floor_fraction(0.05, R) == pytest.approx(0.2)
    p = np.array([0.01, 0.04, 0.05, 0.5, 1.0])
    np.testing.assert_array_equal(mn.uncorrected_rejections(p, R, 0.05), [1, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="multiples"):
        mn.uncorrected_rejections(np.array([0.015]), R, 0.05)


def _summary(n: int, rejected: dict[tuple[str, str], tuple[int, int]]) -> dict:
    """Synthetic per-cell envelope result: ``(mode, dm) -> (cells BH-rejected, + of those)``."""
    sm: dict = {
        k: {}
        for k in (
            "joint_pvals",
            "joint_qvals",
            "joint_signs",
            "per_distance_measure_pvals",
            "per_distance_measure_qvals",
            "per_distance_measure_signs",
        )
    }
    for (mode, dm), (bh, pos) in rejected.items():
        p = np.full(n, 0.5)
        p[: bh + 1] = 0.01  # one extra cell with p < 0.05 but q >= 0.05
        q = np.where(np.arange(n) < bh, 0.01, 0.6)
        sg = np.where(np.arange(n) < pos, 1.0, -1.0)
        key, idx = ("joint", mode) if dm == "joint" else ("per_distance_measure", (mode, dm))
        base = "rejection_joint" if dm == "joint" else "rejection_per_distance_measure"
        sm[f"{key}_pvals"][idx], sm[f"{key}_qvals"][idx], sm[f"{key}_signs"][idx] = p, q, sg
        for suffix, value in (("", bh), ("_positive", pos), ("_negative", bh - pos)):
            sm.setdefault(base + suffix, {})[idx] = value / n
    return {
        "summary": sm,
        "alpha": 0.05,
        "cell_ids": [f"c{i}" for i in range(n)],
        "statistic": "intdev",
        "envelope_type": "pointwise",
    }


def test_envelope_counts():
    res = _summary(10, {("random", "nucleus"): (3, 2)})
    c = mn.envelope_counts(res, "random", "nucleus", R)
    assert (c.n, c.bh, c.bh_pos, c.bh_neg) == (10, 3, 2, 1)
    assert (c.raw, c.raw_pos, c.raw_neg) == (4, 2, 2)
    res["summary"]["rejection_per_distance_measure"][("random", "nucleus")] = 0.9
    with pytest.raises(ValueError, match="BH counts"):
        mn.envelope_counts(res, "random", "nucleus", R)


def test_check_envelope_params():
    res = _summary(4, {("random", "joint"): (1, 1)})
    expected = {"statistic": "intdev", "envelope_type": "pointwise", "alpha": 0.05}
    mn.check_envelope_params(res, expected, 4)
    with pytest.raises(ValueError, match="cells"):
        mn.check_envelope_params(res, expected, 5)
    with pytest.raises(ValueError, match="parameters"):
        mn.check_envelope_params(res, {**expected, "statistic": "supremum"}, 4)


def _ks_frames(n_exp: int = 5, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    boot = pd.DataFrame(
        [(dm, m, e, float(rng.uniform())) for dm in DMS for m in RULES for e in range(n_exp)],
        columns=["distance_measure", "packing_mode", "experiment_number", "similar_fraction"],
    )
    tests = pd.DataFrame(
        [
            (dm, m, f"c{c}", str(s), bool(rng.uniform() > 0.5))
            for dm in DMS
            for m in RULES
            for c in range(3)
            for s in range(2)
        ],
        columns=["distance_measure", "packing_mode", "cell_id", "seed", "similar"],
    )
    return boot, tests


def test_ks_checks():
    boot, tests = _ks_frames()
    mixed = boot[boot["packing_mode"] == "random"].assign(packing_mode=MIXED)
    val = pd.concat([mixed, boot.sample(frac=1, random_state=1)], ignore_index=True)
    mn.check_ks_bootstrap_seeded(boot, val, RULES, 0)  # order and extra arms do not matter
    val.loc[val.index[-1], "similar_fraction"] += 1e-9
    with pytest.raises(ValueError, match="rerun --phase validate"):
        mn.check_ks_bootstrap_seeded(boot, val, RULES, 0)
    mn.check_ks_counts(tests, 6, "t")
    with pytest.raises(ValueError, match="test counts"):
        mn.check_ks_counts(tests.iloc[1:], 6, "t")
    mn.check_ks_tests_match(tests, tests.sample(frac=1, random_state=2), RULES)
    flipped = tests.copy()
    flipped.loc[0, "similar"] = not flipped.loc[0, "similar"]
    with pytest.raises(ValueError, match="differ"):
        mn.check_ks_tests_match(tests, flipped, RULES)
    point = mn.ks_point(tests)
    assert point[("nucleus", "random")] == pytest.approx(
        tests.query("distance_measure == 'nucleus' and packing_mode == 'random'")["similar"].mean()
    )
    mean, sd = mn.ks_boot(boot)[("z", "random")]
    sel = boot.query("distance_measure == 'z' and packing_mode == 'random'")["similar_fraction"]
    assert (mean, sd) == pytest.approx((sel.mean(), sel.std(ddof=1)))


def test_check_curve_ends_consistent():
    def occ(ends: list[float]) -> dict:
        return {
            "individual": {f"c{i}": {"xvals": np.linspace(0, e, 5)} for i, e in enumerate(ends)}
        }

    good = {"a": {"m1": occ([1.0, 2.0]), "m2": occ([1.0, 2.0])}, "b": {"m1": occ([1.0, 2.0])}}
    mn.check_curve_ends_consistent(good)
    with pytest.raises(ValueError, match="ends at"):
        mn.check_curve_ends_consistent({**good, "c": {"m1": occ([1.0, 2.5])}})
    with pytest.raises(ValueError, match="cells differ"):
        mn.check_curve_ends_consistent({**good, "c": {"m1": occ([1.0])}})


def test_records_round_trip():
    sh = mn.NumbersSheet("FigX", "Title 1")
    sh.h2("¶12 — section")
    sh.para("**pex** (n = 305):")
    sh.table(["Rule", "Measure", "v"], [["A", "joint", "1"], ["A", "z", "2"]])
    sh.h3("sub")
    sh.para("Plain text without digits.")
    sh.table(["Rule", "v"], [["B", "0.50 ± 0.10"]])
    sh.bullet("x 1.5 and −2")
    sh.bullet("x 3 and 4")
    sh.bullet("x 3 and 4")
    rec = sh.records()
    assert rec["FigX|section|pex|A / joint|v"] == "1"
    assert rec["FigX|section|pex|A / z|v"] == "2"
    assert rec["FigX|section / sub||B|v"] == "0.50 ± 0.10"
    assert rec["FigX|section / sub||- x # and #|text"] == "1.5 −2"
    assert rec["FigX|section / sub||- x # and #|text #2"] == "3 4"
    assert rec["FigX|section / sub||- x # and #|text #3"] == "3 4"
    assert rec["FigX|||# Title #|text"] == "1"
    assert mn.parse_records("FigX", sh.to_markdown()) == rec
    with pytest.raises(ValueError, match="bad table row"):
        sh.table(["a"], [["b | c"]])


# ── sheets ───────────────────────────────────────────────────────────────────────


def _cv() -> ri.CVResult:
    coef = {
        m: {"joint": (0.1 * (i + 1), 0.01), **dict.fromkeys(DMS, (0.2, 0.02))}
        for i, m in enumerate(RULES)
    }
    mse = {scope: dict.fromkeys(DMS, 0.01) for scope in ("joint", "individual")}
    return ri.CVResult(
        folds=[],
        aggregated_coefficients=coef,
        mean_train_mse=mse,
        std_train_mse=mse,
        mean_test_mse=mse,
        std_test_mse=mse,
        n_folds=5,
        baseline_mode="obs",
    )


def _aic() -> ri.AICComparisonResult:
    names = ["mixed_rule", *(f"single:{m}" for m in RULES), "null"]
    models = [ri.AICModelResult(n, 1, 500, 0.5, 1.0, 1.0, 1.0) for n in names]
    scopes = {"joint": ["joint"], "individual": list(DMS)}
    per = {s: {dm: dict.fromkeys(names, 0.0) for dm in d} for s, d in scopes.items()}
    best = {s: dict.fromkeys(d, "mixed_rule") for s, d in scopes.items()}
    return ri.AICComparisonResult(
        comparisons={s: dict.fromkeys(d, models) for s, d in scopes.items()},
        delta_aic=per,
        delta_bic=per,
        akaike_weights=per,
        bic_weights=per,
        best_model_aic=best,
        best_model_bic=best,
    )


def _structure(s: str, n: int = 6) -> mn.StructureData:
    modes = [MIXED, *RULES]
    rng = np.random.default_rng(len(s))
    emd = {(m, dm): rng.uniform(0, 1, n) for m in modes for dm in DMS}
    emd.update({(m, "pooled"): np.concatenate([emd[(m, dm)] for dm in DMS]) for m in modes})
    rejected = {(m, dm): (n // 2, n // 3) for m in modes for dm in ("joint", *DMS)}
    env = _summary(n, rejected)
    ks = {(dm, m): 0.5 for dm in DMS for m in modes}
    boot = {(dm, m): (0.5, 0.02) for dm in DMS for m in modes}
    curve = {"xvals": np.linspace(0, 5, 11), "occupancy": np.linspace(1.5, 0.5, 11)}
    return mn.StructureData(
        structure=s,
        baseline="obs",
        n_cells=n,
        rules=list(RULES),
        mixed=MIXED,
        lam=0.7,
        obs_dist={(dm, "obs"): (1.0, 0.5) for dm in DMS},
        emd3=emd,
        emdv=emd,
        occ3=emd,
        occv=emd,
        env3=env,
        envv=env,
        counts3={
            (m, dm): mn.envelope_counts(env, m, dm, R) for m in RULES for dm in ("joint", *DMS)
        },
        n_tests={(dm, m): n * R for dm in DMS for m in RULES},
        n_bootstrap=1000,
        ks_seed=0,
        ks_seeded=s == "peroxisome",
        ks3=ks,
        ks3_boot=boot,
        ksv=ks,
        ksv_boot=boot,
        aic=_aic(),
        cv=_cv(),
        recon={(scope, dm): 0.0076 for scope in ("joint", "individual") for dm in DMS},
        occ_obs=dict.fromkeys(DMS, curve),
        inputs={"distance": [Path("/x/a.parquet")]},
    )


def test_structure_sheets(config):
    data = {s: _structure(s) for s in ("peroxisome", "endosome")}
    sheets = [
        mn.section_fig3(config, data, "abcdef0123"),
        mn.section_fig4(config, data, "abcdef0123"),
        mn.section_fig5(config, data, "abcdef0123"),
        mn.section_fig_s67(config, data["peroxisome"], "FigS6", "abcdef0123"),
        mn.section_fig_s89(config, data["endosome"], "FigS9", "abcdef0123"),
    ]
    rec = {k: v for sh in sheets for k, v in sh.records().items()}
    key = "Fig3|per-cell Monte Carlo envelope test (Fig 3d) / Counts|peroxisome|Unbiased / nucleus|"
    assert rec[key + "BH q < 0.05"] == "3 of 6"
    assert rec[key + "p < 0.05"] == "4 of 6 (67%)"
    md = sheets[0].to_markdown()
    assert "engine commit `abcdef01`" in md and "**not seed-checked**" in md
    assert "4% of cells under the null" in md and "at least 20% of cells" in md
    assert rec["Fig4|joint NNLS relative contributions (Fig 4d)||Apical bias|pex v4"] == "40 %"
    assert "S8d/S9d" in sheets[2].to_markdown()
    assert "## S6a — per-cell rejection by distance measure, five rules" in sheets[3].to_markdown()
    assert "Raw joint coefficients sum to 1.0000" in sheets[4].to_markdown()


class StubSources:
    """The four ``PaperSources`` calls the Fig 2 sheet makes."""

    def __init__(self, root: Path):
        self.root = root
        self.cfg = SimpleNamespace(
            results_dir=root / "results/biological_variation",
            packing_id="peroxisome",
            baseline_mode="baseline",
        )

    @contextmanager
    def record_inputs(self):
        used: list[Path] = []
        yield used
        used.append(self.cfg.results_dir / "pairwise_envelope_results.dat")

    def biovar_config(self):
        return self.cfg

    def pairwise_envelope(self):
        modes, dms = ["baseline", "variable_count"], ["nearest", "z"]

        def entry(r: float) -> dict:
            return {
                "pvals": np.full(5, 0.5),
                "rejection_fraction": r,
                "rejection_fraction_positive": r / 4,
                "rejection_fraction_negative": 3 * r / 4,
            }

        pairs = {(a, b): entry(0.2 if a != b else 0.0) for a in modes for b in modes}
        return {
            "packing_modes": modes,
            "distance_measures": dms,
            "joint": pairs,
            "per_distance_measure": dict.fromkeys(dms, pairs),
            "statistic": "intdev",
            "envelope_type": "pointwise",
            "alpha": 0.05,
        }

    def read_log(self, results_dir: Path, name: str) -> list[str]:
        return [
            "t | x | INFO |   baseline vs baseline [nearest]: 0.10 ± 0.04 (median: 0.1)",
            "t | x | INFO |   baseline vs variable_count [z]: 0.51 ± 0.34 (median: 0.4)",
        ]


def test_write_figure_numbers_fig2(config, caplog):
    config.numbers = {"sheets": ["Fig2", "Methods"], "write_json": True}
    with caplog.at_level(logging.INFO, logger=mn.__name__):
        paths = mn.write_figure_numbers(config, cast(Any, StubSources(config.project_root)))
    assert "Skipping sheet Methods" in caplog.text
    out = config.results_dir
    assert paths == [out / "Fig2/Fig2_numbers.md", out / mn.JSON_NAME]
    assert all(p.is_relative_to(out) for p in paths)
    assert not (out / "Methods").exists()
    md = paths[0].read_text()
    assert "| Count | 0.200 (+0.050 / −0.150) | 0.20 (+0.05 / −0.15) |" in md
    assert "n(n − 1)/2 = 10; to the baseline n² = 25 (n = 5)" in md
    payload = json.loads(paths[1].read_text())
    assert payload["sheets"] == {"Fig2": "Fig2/Fig2_numbers.md"}
    assert payload["records"] == mn.parse_records("Fig2", md)
    key = "Fig2|Table S1 — distance EMD (Fig 2c–e)||Count|to baseline z"
    assert payload["records"][key] == "0.51 ± 0.34"
    assert (
        payload["records"]["Fig2|Table S1 — distance EMD (Fig 2c–e)||Baseline|to baseline z"]
        == "N/A"
    )


def test_write_figure_numbers_needs_both_measures(config):
    config.distance_measures = ["nucleus"]
    with pytest.raises(ValueError, match="distance_measures"):
        mn.write_figure_numbers(config, cast(Any, StubSources(config.project_root)))
