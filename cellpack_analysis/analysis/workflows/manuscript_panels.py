"""Manuscript panel builders and slot edits, keyed by the names the manuscript config uses.

A builder draws one panel with the pipeline's own plot function from the persisted caches
(:class:`~cellpack_analysis.analysis.workflows.manuscript_sources.PaperSources`) and returns
``(figure, axes)``, the axes in the order the asset's frames expect. It recomputes nothing.

An edit adapts a built panel to its slot before it is restyled, and returns the texts it removed
so the MANIFEST can list them for typesetting.

``PANEL_INPUTS`` and ``EDIT_INPUTS`` load what a builder or edit reads. The builders and edits
call them too, so ``check_inputs`` checks exactly the caches an asset will read.
"""

import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cellpack_analysis.analysis.workflows.manuscript_sources import PaperSources
from cellpack_analysis.lib import figure_layout as fl
from cellpack_analysis.lib import label_tables
from cellpack_analysis.lib import rule_interpolation as ri
from cellpack_analysis.lib import visualization as vis
from cellpack_analysis.lib.figure_layout import TextStyle

DISTANCE_SOURCES = ("distance", "distance_supp")


@dataclass(frozen=True)
class PanelContext:
    """What a builder or edit may read besides its own arguments.

    Parameters
    ----------
    sources
        The cache resolver.
    structure
        Structure (packing id) the asset was expanded for, or ``None``.
    measure
        Distance measure the asset was expanded for, or ``None``.
    style
        The slot's text style; ``None`` for native renders.
    """

    sources: PaperSources
    structure: str | None
    measure: str | None
    style: TextStyle | None = None

    @property
    def s(self) -> str:
        """The structure; raises if the asset has none."""
        if self.structure is None:
            raise ValueError("This panel needs a {structure} asset")
        return self.structure

    @property
    def m(self) -> str:
        """The distance measure; raises if the asset has none."""
        if self.measure is None:
            raise ValueError("This panel needs a {measure} asset")
        return self.measure

    @property
    def text_style(self) -> TextStyle:
        """The slot style; raises for native renders."""
        if self.style is None:
            raise ValueError("This edit needs a slot style")
        return self.style


PanelFn = Callable[..., tuple[Figure, list[Axes]]]  # (ctx, **panel_args)
EditFn = Callable[..., list[str] | None]  # (fig, axes, ctx, **params)
InputFn = Callable[..., dict[str, Any]]  # (ctx, **args)


# ── Inputs: one loader per panel (and per data-reading edit) ─────────────────────────


def _supp(source: str) -> bool:
    if source not in DISTANCE_SOURCES:
        raise ValueError(f"source must be one of {DISTANCE_SOURCES}, got {source!r}")
    return source == "distance_supp"


def _check_n_cells(ctx: PanelContext, n: int, what: str) -> None:
    want = ctx.sources.config.expected["n_cells"][ctx.s]
    if n != want:
        raise ValueError(f"{what} for {ctx.s} holds {n} cells; expected {want}")


def _check_occupancy_cells(
    ctx: PanelContext, occ: Mapping[str, Any], baseline: str, what: str
) -> None:
    _check_n_cells(ctx, len(occ[baseline]["individual"]), what)


def _per_cell(ctx: PanelContext) -> dict[str, Any]:
    res = ctx.sources.per_cell_envelope(ctx.s)
    _check_n_cells(ctx, len(res["cell_ids"]), "per_cell_envelope_results.dat")
    return res


def _validation(ctx: PanelContext) -> ri.ValidationResult:
    vr = ctx.sources.validation(ctx.s)
    test = vr.distance_per_cell_envelope_test
    if test is not None:
        _check_n_cells(ctx, len(test["cell_ids"]), "validation per-cell envelope test")
    return vr


def _fit_inputs(ctx: PanelContext, kind: str) -> dict[str, Any]:
    cfg = ctx.sources.fit_config(kind, ctx.s)
    occ = ctx.sources.fit_occupancy(kind, ctx.s)
    for dm, by_mode in occ.items():
        _check_occupancy_cells(ctx, by_mode, cfg.baseline_mode, f"{kind} {dm} occupancy")
    return {"config": cfg, "occupancy": occ, "cv": ctx.sources.cv_result(kind, ctx.s)}


def in_biovar(ctx: PanelContext, **_: Any) -> dict[str, Any]:
    """Fig 2f / S3: the pooled bio-variation envelope test."""
    return {"result": ctx.sources.pairwise_envelope()}


def in_distance_pdfs(ctx: PanelContext, source: str) -> dict[str, Any]:
    """Fig 3a / S4a: the distance config and its PDF cache."""
    supp = _supp(source)
    return {
        "config": ctx.sources.distance_config(ctx.s, supp),
        "pdfs": ctx.sources.distance_pdfs(ctx.s, supp),
    }


def in_distance_emd(ctx: PanelContext, source: str, **_: Any) -> dict[str, Any]:
    """Fig 3b / S4b: the distance config and its EMD frame."""
    supp = _supp(source)
    return {
        "config": ctx.sources.distance_config(ctx.s, supp),
        "emd": ctx.sources.distance_emd(ctx.s, supp),
    }


def in_distance_ks(ctx: PanelContext, source: str) -> dict[str, Any]:
    """Fig 3c / S4c: the distance config and its seeded KS bootstrap."""
    supp = _supp(source)
    return {
        "config": ctx.sources.distance_config(ctx.s, supp),
        "ks_bootstrap": ctx.sources.ks_bootstrap(ctx.s, supp),
    }


def in_per_cell(ctx: PanelContext, **_: Any) -> dict[str, Any]:
    """Fig 3d / S6/S7 b, c: the per-cell envelope test."""
    return {"result": _per_cell(ctx)}


def in_occupancy_per_rule(ctx: PanelContext) -> dict[str, Any]:
    """Fig 4d per-rule panels: the occupancy config and one measure's cache."""
    cfg = ctx.sources.occupancy_config(ctx.s)
    occ = ctx.sources.occupancy(ctx.s, ctx.m)
    _check_occupancy_cells(ctx, occ, cfg.baseline_mode, f"{ctx.m} occupancy")
    return {"config": cfg, "occupancy": occ}


def in_cv_fit(ctx: PanelContext, source: str, **_: Any) -> dict[str, Any]:
    """Fig 4d target / S10 fits: the fit config, its occupancy and its CV result."""
    return _fit_inputs(ctx, source)


def in_validation_occupancy(ctx: PanelContext, **_: Any) -> dict[str, Any]:
    """Fig 5b: the validation config, the merged occupancy and the fit's CV result."""
    cfg = ctx.sources.validation_config(ctx.s)
    occ = ctx.sources.validation_occupancy(ctx.s)
    for dm, by_mode in occ.items():
        _check_occupancy_cells(ctx, by_mode, cfg.baseline_mode, f"validation {dm} occupancy")
    _fit_inputs(ctx, "rule_interpolation")
    return {"config": cfg, "occupancy": occ}


def in_validation(ctx: PanelContext) -> dict[str, Any]:
    """Fig 5c / S6/S7 a / S8/S9 b, d: the validation config and result."""
    return {"config": ctx.sources.validation_config(ctx.s), "validation": _validation(ctx)}


def in_pairwise_emd_matrix(ctx: PanelContext) -> dict[str, Any]:
    """S5: the Fig 3 distance config, its EMD frame and its PDFs."""
    return {
        "config": ctx.sources.distance_config(ctx.s),
        "emd": ctx.sources.distance_emd(ctx.s),
        "pdfs": ctx.sources.distance_pdfs(ctx.s),
    }


def in_cv_result(ctx: PanelContext) -> dict[str, Any]:
    """S8/S9 a, c: the rule-interpolation CV result."""
    return {"cv": ctx.sources.cv_result("rule_interpolation", ctx.s)}


def in_ladder_profile(ctx: PanelContext) -> dict[str, Any]:
    """S10 profile: the ladder fit config, occupancy and CV result."""
    return _fit_inputs(ctx, "ladder")


# ── Panels ───────────────────────────────────────────────────────────────────────────


def biovar_rejection_joint(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """Fig 2f: joint rejection fractions of each bio-variation source against the baseline."""
    res = in_biovar(ctx)["result"]
    fig, ax = vis.plot_per_dm_rejection_bars(
        pairwise_results=res, compare_mode="baseline", joint_test=True
    )
    return fig, [ax]


def biovar_envelope_matrix(
    ctx: PanelContext,
    distance_measure: str | None,
    figure_size: Sequence[float],
    font_scale: float,
) -> tuple[Figure, list[Axes]]:
    """S3c / S3e: the pooled envelope-test matrix for one measure (``None``: joint)."""
    res = in_biovar(ctx)["result"]
    fig, _ = vis.plot_pairwise_envelope_matrix(
        pairwise_results=res,
        distance_measure=distance_measure,
        figure_size=(float(figure_size[0]), float(figure_size[1])),
        font_scale=font_scale,
    )
    return fig, list(fig.axes)


def distance_pdfs(ctx: PanelContext, source: str) -> tuple[Figure, list[Axes]]:
    """Fig 3a / S4a: distance PDFs, axes row-major (rule rows, measure columns)."""
    data = in_distance_pdfs(ctx, source)
    c = data["config"]
    fig, axd = vis.plot_distance_distributions(
        distance_pdf_dict=data["pdfs"],
        distance_measures=c.distance_measures,
        packing_modes=c.packing_modes,
        normalization=c.normalization,
        **c.distance_plot_params,
    )
    return fig, [axd[dm][r] for r in range(len(c.packing_modes)) for dm in c.distance_measures]


def distance_emd(ctx: PanelContext, source: str, violins: bool) -> tuple[Figure, list[Axes]]:
    """Fig 3b (boxplots) / S4b (``violins``): distance EMD of each rule against the baseline."""
    data = in_distance_emd(ctx, source)
    c = data["config"]
    fig, axs, fig_v, axs_v = vis.plot_emd_comparisons(
        df_emd=data["emd"],
        distance_measures=c.distance_measures,
        comparison_type="baseline",
        baseline_mode=c.baseline_mode,
        packing_modes=c.packing_modes,
        annotate_significance=False,
    )
    if violins:
        fig, axs = fig_v, axs_v
    else:
        plt.close(fig_v)
    return fig, list(axs.ravel())


def distance_ks(ctx: PanelContext, source: str) -> tuple[Figure, list[Axes]]:
    """Fig 3c / S4c: bootstrapped fraction of KS-similar cells per rule."""
    data = in_distance_ks(ctx, source)
    fig, axs = vis.plot_ks_test_results(
        df_ks_bootstrap=data["ks_bootstrap"], distance_measures=data["config"].distance_measures
    )
    return fig, list(axs.ravel())


def per_cell_rejection_joint(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """Fig 3d: joint per-cell envelope rejection fractions."""
    fig, ax = vis.plot_per_cell_rejection_bars(
        per_cell_results=in_per_cell(ctx)["result"], joint_test=True
    )
    return fig, [ax]


def occupancy_per_rule(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """Fig 4d: occupancy ratio of each rule against the observed curve, one axes per rule."""
    data = in_occupancy_per_rule(ctx)
    c = data["config"]
    fig, axs = vis.plot_occupancy_ratio_per_rule(
        occupancy_dict=data["occupancy"],
        modes=[x for x in c.channel_map if x != c.structure_id],
        normalization=c.normalization,
        distance_measure=ctx.m,
        ylim=c.occupancy_params[ctx.m]["ylim"],
    )
    return fig, list(axs)


def cv_fit(ctx: PanelContext, source: str, observed_only: bool) -> tuple[Figure, list[Axes]]:
    """Fig 4d target / S10 joint fit: the CV-mean NNLS fit over the fit-phase occupancy."""
    data = in_cv_fit(ctx, source)
    c = data["config"]
    cm = {c.baseline_mode: c.channel_map[c.baseline_mode]} if observed_only else c.channel_map
    fig, ax = vis.plot_rule_interpolation_fit(
        fit_result=ctx.sources.cv_fit(source, ctx.s),
        occupancy_dict=data["occupancy"],
        channel_map=cm,
        baseline_mode=c.baseline_mode,
        distance_measure=ctx.m,
        plot_type="joint",
        ylim=c.occupancy_params[ctx.m]["ylim"],
    )
    return fig, [ax]


def validation_occupancy(ctx: PanelContext, mixed_only: bool) -> tuple[Figure, list[Axes]]:
    """Fig 5b: observed vs mixed-rule occupancy (``mixed_only``) or every validated mode."""
    data = in_validation_occupancy(ctx)
    v, occ = data["config"], data["occupancy"]
    keep = (v.baseline_mode, "interpolated")
    cm = {k: x for k, x in v.channel_map.items() if k in keep} if mixed_only else v.channel_map
    fig, ax = vis.plot_validation_occupancy(
        fit_result=ctx.sources.cv_fit("rule_interpolation", ctx.s, occ),
        occupancy_dict=occ,
        channel_map=cm,
        baseline_mode=v.baseline_mode,
        distance_measure=ctx.m,
        ylim=v.occupancy_params.get(ctx.m, {}).get("ylim"),
    )
    return fig, [ax]


def occupancy_emd(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """Fig 5c: occupancy EMD of each mode against the observed curve, as validate draws it."""
    data = in_validation(ctx)
    v = data["config"]
    df = data["validation"].emd_df
    df = vis._filter_emd_vs_baseline(df[df["distance_measure"] == ctx.m], v.baseline_mode)
    fig, ax = plt.subplots(dpi=300, figsize=(2.5, 2.5))
    plt.rcParams.update({"font.size": 8})
    title = label_tables.DISTANCE_MEASURE_TITLES.get(ctx.m, ctx.m)
    vis._draw_emd_boxplot(
        ax,
        df,
        f"Occupancy EMD vs baseline\n({title})",
        packing_modes=[x for x in v.packing_modes if x != v.baseline_mode],
    )
    fig.tight_layout()
    return fig, [ax]


def pairwise_emd_matrix(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """S5: the pairwise distance-EMD matrix, fed the Fig 3a PDFs."""
    data = in_pairwise_emd_matrix(ctx)
    c = data["config"]
    fig, axs = vis.plot_pairwise_emd_matrix(
        df_emd=data["emd"],
        packing_modes=c.packing_modes,
        distance_measure=ctx.m,
        normalization=c.normalization,
        distance_pdf_dict=data["pdfs"],
        **c.emd_plot_params,
    )
    return fig, list(axs.ravel())


def rejection_by_measure(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """S6a / S7a: per-measure per-cell rejection fractions, mixed rule included."""
    vr = in_validation(ctx)["validation"]
    fig, ax = vis.plot_per_cell_rejection_bars(per_cell_results=vr.distance_per_cell_envelope_test)
    return fig, [ax]


def envelope_examples(
    ctx: PanelContext, select: Literal["most_rejected", "median"]
) -> tuple[Figure, list[Axes]]:
    """S6/S7 b, c: one cell's curves against its envelopes (median or most-rejected cell)."""
    fig, axs = vis.plot_per_cell_envelope_examples(
        per_cell_results=in_per_cell(ctx)["result"], select=select
    )
    return fig, list(axs.ravel())


def cv_mse(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """S8a / S9a: cross-validated MSE per scope."""
    fig, _ = vis.plot_cv_mse_summary(cv_df=ri.summarize_cv_results(in_cv_result(ctx)["cv"]))
    return fig, list(fig.axes)


def cv_coefficients(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """S8c / S9c: NNLS coefficients across CV folds."""
    fig, _ = vis.plot_cv_coefficient_stability(cv_result=in_cv_result(ctx)["cv"])
    return fig, list(fig.axes)


def aic_weights(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """S8b / S9b: AIC and BIC model weights."""
    fig, ax = vis.plot_aic_model_weights(aic_result=in_validation(ctx)["validation"].aic_result)
    return fig, [ax]


def validation_summary(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """S8d / S9d: the validation summary (EMDs, KS, mixed-rule envelope test)."""
    data = in_validation(ctx)
    v = data["config"]
    fig, axs = vis.plot_rule_interpolation_validation_summary(
        validation_result=data["validation"],
        packing_modes=v.packing_modes,
        baseline_mode=v.baseline_mode,
        distance_measures=v.occupancy_distance_measures,
    )
    return fig, [ax for ax in axs if ax.get_visible()]


def ladder_profile(ctx: PanelContext) -> tuple[Figure, list[Axes]]:
    """S10 b, d: CV MSE and coefficient profile along the decay-length ladder."""
    data = in_ladder_profile(ctx)
    c = data["config"]
    fig, axes = vis.plot_lambda_ladder_profile(
        cv_result=data["cv"],
        occupancy_dicts=data["occupancy"],
        channel_map=c.cv_channel_map,
        baseline_mode=c.baseline_mode,
    )
    return fig, [ax for ax in np.ravel(axes) if ax.get_visible()]


PANELS: dict[str, PanelFn] = {
    "biovar_rejection_joint": biovar_rejection_joint,
    "biovar_envelope_matrix": biovar_envelope_matrix,
    "distance_pdfs": distance_pdfs,
    "distance_emd": distance_emd,
    "distance_ks": distance_ks,
    "per_cell_rejection_joint": per_cell_rejection_joint,
    "occupancy_per_rule": occupancy_per_rule,
    "cv_fit": cv_fit,
    "validation_occupancy": validation_occupancy,
    "occupancy_emd": occupancy_emd,
    "pairwise_emd_matrix": pairwise_emd_matrix,
    "rejection_by_measure": rejection_by_measure,
    "envelope_examples": envelope_examples,
    "cv_mse": cv_mse,
    "aic_weights": aic_weights,
    "cv_coefficients": cv_coefficients,
    "validation_summary": validation_summary,
    "ladder_profile": ladder_profile,
}

PANEL_INPUTS: dict[str, InputFn] = {
    "biovar_rejection_joint": in_biovar,
    "biovar_envelope_matrix": in_biovar,
    "distance_pdfs": in_distance_pdfs,
    "distance_emd": in_distance_emd,
    "distance_ks": in_distance_ks,
    "per_cell_rejection_joint": in_per_cell,
    "occupancy_per_rule": in_occupancy_per_rule,
    "cv_fit": in_cv_fit,
    "validation_occupancy": in_validation_occupancy,
    "occupancy_emd": in_validation,
    "pairwise_emd_matrix": in_pairwise_emd_matrix,
    "rejection_by_measure": in_validation,
    "envelope_examples": in_per_cell,
    "cv_mse": in_cv_result,
    "aic_weights": in_validation,
    "cv_coefficients": in_cv_result,
    "validation_summary": in_validation,
    "ladder_profile": in_ladder_profile,
}


# ── Edits: generic wrappers over figure_layout ───────────────────────────────────────


def e_pull_texts(
    fig: Figure, axes: list[Axes], ctx: PanelContext, titles: bool = True
) -> list[str]:
    """Remove in-panel texts (and titles) the slot does not carry."""
    return fl.pull_texts(axes, titles=titles)


def e_drop_ylabels(
    fig: Figure, axes: list[Axes], ctx: PanelContext, keep_first: bool = False
) -> None:
    """Clear the y labels, optionally keeping the first."""
    fl.drop_ylabels(axes, keep_first=keep_first)


def e_sentence_case_ylabels(
    fig: Figure, axes: list[Axes], ctx: PanelContext, mapping: Mapping[str, str] | None = None
) -> None:
    """Set y labels in sentence case ("Occupancy ratio")."""
    fl.sentence_case_ylabels(axes, mapping)


def e_fit_ylim_top(
    fig: Figure,
    axes: list[Axes],
    ctx: PanelContext,
    top: float,
    step: float = 0.5,
    fill: float = 0.97,
    axis_index: int = 0,
) -> list[str]:
    """Set one axes' y range to ``0..top``, raised in steps until its lines fit."""
    return fl.fit_ylim_top(axes[axis_index], top, step=step, fill=fill)


def e_set_ylim_headroom(
    fig: Figure,
    axes: list[Axes],
    ctx: PanelContext,
    top: float = 1.4,
    tick_step: float = 0.2,
    axis_index: int = 0,
) -> list[str]:
    """Extend one axes' 0 to 1 range to ``top`` so a legend clears the bars."""
    return fl.set_ylim_headroom(axes[axis_index], top=top, tick_step=tick_step)


def e_legend_above(fig: Figure, axes: list[Axes], ctx: PanelContext, dy: float = 0.33) -> list[str]:
    """Move the one in-panel legend above the grid, at the slot's legend size."""
    return fl.legend_above(axes, ctx.text_style.legend, dy=dy)


def e_remove_collections(fig: Figure, axes: list[Axes], ctx: PanelContext) -> list[str]:
    """Draw the mean curves without envelope bands."""
    return fl.remove_collections(axes)


# ── Edits: paper-specific label policy ───────────────────────────────────────────────


def fig3a_labels(fig: Figure, axes: list[Axes], ctx: PanelContext) -> list[str]:
    """Rule names are typeset beside the rows; both x labels read Distance in microns."""
    removed = list(dict.fromkeys(a.get_ylabel().replace("\n", " ") for a in axes if a.get_ylabel()))
    for ax in axes:
        ax.set_ylabel("")
        if ax.get_xlabel():
            ax.set_xlabel("Distance (\N{GREEK SMALL LETTER MU}m)")
    return removed + fl.pull_texts(axes)


def fig3b_labels(fig: Figure, axes: list[Axes], ctx: PanelContext) -> list[str]:
    """No rule names (3c carries them) and one "EMD" label."""
    for ax in axes:
        ax.tick_params(axis="x", labelbottom=False)
    fl.drop_ylabels(axes, keep_first=True)
    return ["x-axis rule names", *fl.pull_texts(axes)]


def fig3c_labels(fig: Figure, axes: list[Axes], ctx: PanelContext) -> list[str]:
    """One y label, "Fraction similar"."""
    axes[0].set_ylabel("Fraction similar")
    fl.drop_ylabels(axes, keep_first=True)
    return fl.pull_texts(axes)


def fig3d_labels(fig: Figure, axes: list[Axes], ctx: PanelContext) -> list[str]:
    """Title and x label are typeset; x ticks at 0, 0.5, 1."""
    ax = axes[0]
    removed = [ax.get_xlabel(), *fl.pull_texts(axes)]
    ax.set_xlabel("")
    ax.set_xticks([0, 0.5, 1.0])
    return removed


def measure_title(fig: Figure, axes: list[Axes], ctx: PanelContext) -> None:
    """Title names the measure only; the two-line title overhangs a small frame."""
    axes[0].set_title(label_tables.DISTANCE_MEASURE_TITLES.get(ctx.m, ctx.m))


def target_labels(
    fig: Figure, axes: list[Axes], ctx: PanelContext, drop_xlabel: bool = False
) -> list[str]:
    """Fig 4d targets: no y label, and optionally no x label."""
    fl.drop_ylabels(axes)
    if drop_xlabel:
        axes[0].set_xlabel("")
    return fl.pull_texts(axes)


def in_emd_axis_per_pair(ctx: PanelContext, **_: Any) -> dict[str, Any]:
    """S5 edit: the Fig 3 distance EMD frame."""
    return {"emd": ctx.sources.distance_emd(ctx.s)}


def emd_axis_per_pair(fig: Figure, axes: list[Axes], ctx: PanelContext) -> list[str]:
    """Shared EMD axis to the largest per-pair q99, times 1.1.

    The pipeline pools all pairs, which the diagonal's cross-cell pairs dominate, and clips
    the upper-triangle violins.
    """
    df = in_emd_axis_per_pair(ctx)["emd"]
    d = df[df["distance_measure"] == ctx.m]
    a, b = d["packing_mode_1"].to_numpy(), d["packing_mode_2"].to_numpy()
    key = np.where(a < b, a + "|" + b, b + "|" + a)
    top = float(d.groupby(key)["emd"].quantile(0.99).max()) * 1.1
    n = round(len(axes) ** 0.5)
    for i in range(n):
        for j in range(i, n):
            axes[i * n + j].set_xlim(0, top)
    for ax in axes:  # one line would run into the next column's label
        t = ax.get_xlabel()
        if len(t) > 20:
            ax.set_xlabel(textwrap.fill(t, width=len(t) // 2 + 1))
    return []


def in_shares_to_side(ctx: PanelContext, source: str = "ladder", **_: Any) -> dict[str, Any]:
    """S10 edit: the fit whose joint shares are listed."""
    _fit_inputs(ctx, source)
    return {"fit": ctx.sources.cv_fit(source, ctx.s)}


def shares_to_side(
    fig: Figure, axes: list[Axes], ctx: PanelContext, show: bool, source: str = "ladder"
) -> list[str]:
    """S10: joint shares as a colored column right of the panel (when ``show``)."""
    ax = axes[0]
    removed = fl.pull_texts(axes, titles=False)
    if show:
        legend = ctx.text_style.legend
        fit = in_shares_to_side(ctx, source)["fit"]
        for i, (mode, v) in enumerate(fit.relative_contributions_joint.items()):
            ax.annotate(
                f"{label_tables.MODE_LABELS.get(mode, mode)}: {v:.0%}",
                xy=(1.03, 1.0),
                xycoords="axes fraction",
                xytext=(0, -i * legend * 1.3),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=legend,
                annotation_clip=False,
                color=label_tables.COLOR_PALETTE.get(mode, "gray"),
            )
    return removed


def biovar_rejection_bars(
    fig: Figure,
    axes: list[Axes],
    ctx: PanelContext,
    labels: Mapping[str, str],
    bar_height: float,
    light_colors: Mapping[str, str],
    xticks: Sequence[float],
) -> list[str]:
    """Fig 2f: title and legend are typeset; row labels, bar height, tints and x ticks set."""
    res = in_biovar(ctx)["result"]
    ax = axes[0]
    legend = ax.get_legend()
    removed = [ax.get_title()] + ([t.get_text() for t in legend.get_texts()] if legend else [])
    ax.set_title("")
    if legend is not None:
        legend.remove()
    for p in ax.patches:
        c = p.get_y() + p.get_height() / 2  # type: ignore[attr-defined]
        p.set_y(c - bar_height / 2)  # type: ignore[attr-defined]
        p.set_height(bar_height)  # type: ignore[attr-defined]
    ax.relim()
    ax.autoscale_view(scalex=False)
    # same order as plot_per_dm_rejection_bars draws the rows
    ref_modes = [m for m in res["packing_modes"][::-1] if m != "baseline"]
    ax.set_yticklabels([labels[m] for m in ref_modes])
    for m, p in zip(ref_modes, ax.patches[1::2], strict=True):
        if m in light_colors:
            p.set_facecolor(light_colors[m])
            p.set_alpha(1.0)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    return removed


EDITS: dict[str, EditFn] = {
    "pull_texts": e_pull_texts,
    "drop_ylabels": e_drop_ylabels,
    "sentence_case_ylabels": e_sentence_case_ylabels,
    "fit_ylim_top": e_fit_ylim_top,
    "set_ylim_headroom": e_set_ylim_headroom,
    "legend_above": e_legend_above,
    "remove_collections": e_remove_collections,
    "fig3a_labels": fig3a_labels,
    "fig3b_labels": fig3b_labels,
    "fig3c_labels": fig3c_labels,
    "fig3d_labels": fig3d_labels,
    "measure_title": measure_title,
    "target_labels": target_labels,
    "emd_axis_per_pair": emd_axis_per_pair,
    "shares_to_side": shares_to_side,
    "biovar_rejection_bars": biovar_rejection_bars,
}

EDIT_INPUTS: dict[str, InputFn] = {
    "emd_axis_per_pair": in_emd_axis_per_pair,
    "shares_to_side": in_shares_to_side,
    "biovar_rejection_bars": in_biovar,
}
