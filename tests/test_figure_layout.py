"""Tests for ``cellpack_analysis.lib.figure_layout``."""

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cellpack_analysis.lib import figure_layout as fl

DASH = "\N{EN DASH}"
SOURCE_DATE_EPOCH = "1577836800"  # 2020-01-01T00:00:00Z


def build_panel(nrows: int = 2, ncols: int = 3, n: int | None = None) -> tuple[Figure, list[Axes]]:
    """A deterministic synthetic panel with every element ``restyle`` touches."""
    fig, grid = plt.subplots(nrows, ncols, figsize=(6.0, 4.0))
    axes = list(np.ravel(grid))
    x = np.linspace(0.0, 1.0, 40)
    for i, ax in enumerate(axes):
        ax.plot(x, (i + 1) * np.sin(3 * x), marker="o", ms=3, label="mean")
        ax.plot(x, (i + 1) * np.cos(3 * x), label="other")
        ax.fill_between(x, 0, x * (i + 1), alpha=0.3)
        ax.bar([0.2, 0.5, 0.8], [0.3, 0.6, 0.4], width=0.2, hatch="//", edgecolor="k")
        ax.text(0.5, 0.5, f"n = {i}")
        ax.set_title(f"Title {i}")
        ax.set_title("left", loc="left")
        ax.set_xlabel("Distance (µm)")
        ax.set_ylabel("Occupancy Ratio")
        ax.minorticks_on()
        ax.legend(title="rule", loc="upper right")
    fig.text(0.5, 0.01, "figure text")
    fig.suptitle("Super title")
    if n is not None:
        for ax in axes[n:]:
            ax.remove()
        axes = axes[:n]
    return fig, axes


def build_plain() -> tuple[Figure, list[Axes]]:
    """A panel without suptitle, figure text or legends."""
    fig, grid = plt.subplots(1, 3, figsize=(5.0, 2.0))
    x = np.linspace(0.0, 2.0, 30)
    for i, ax in enumerate(np.ravel(grid)):
        ax.plot(x, x ** (i + 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    return fig, list(np.ravel(grid))


def grid_slot(axes: list[Axes]) -> fl.FrameMap:
    return fl.grid_frames(axes, 1.4, 1.1, ncols=3, xpitch=2.0, ypitch=1.8)


def style_for_tests() -> fl.TextStyle:
    return fl.TextStyle(tick=5, label=8, title=7, bold_title=True, lw=1.0, hatch_lw=0.7)


@pytest.fixture(autouse=True)
def _isolate_matplotlib() -> Iterator[None]:
    """Close figures and undo rcParams the tests' own restyle calls set."""
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", SOURCE_DATE_EPOCH)


def _rc_snapshot() -> dict[str, Any]:
    return {k: v for k, v in mpl.rcParams.items() if k != "backend"}


# ── Frames ──────────────────────────────────────────────────────────────────────────


def _laid_out(frames_fn: Callable[[list[Axes]], fl.FrameMap]) -> tuple[Figure, fl.FrameMap]:
    fig, axes = build_panel()
    fl.restyle(fig, style_for_tests())
    frames = frames_fn(axes)
    fl.fit_page(fig, frames)
    return fig, frames


def _assert_frames_land(fig: Figure, frames: fl.FrameMap, atol: float = 1e-9) -> None:
    sizes = fl.check_frames(fig, frames, atol=atol)
    assert sizes == [(round(w, 3), round(h, 3)) for _, _, w, h in frames.values()]
    # relative placement in inches follows the frames, whatever the page crop offset is
    fig_w, fig_h = fig.get_size_inches()
    boxes = [ax.get_position() for ax in frames]
    (x_ref, y_ref, _, _), p_ref = next(iter(frames.values())), boxes[0]
    for (x, y, w, h), p in zip(frames.values(), boxes, strict=True):
        assert (p.x0 - p_ref.x0) * fig_w == pytest.approx(x - x_ref, abs=atol)
        assert (p.y0 - p_ref.y0) * fig_h == pytest.approx(y - y_ref, abs=atol)
        assert p.width * fig_w == pytest.approx(w, abs=atol)
        assert p.height * fig_h == pytest.approx(h, abs=atol)


def test_grid_frames_values() -> None:
    _, axes = build_panel(n=5)
    frames = fl.grid_frames(axes, 1.0, 0.5, ncols=3, xpitch=2.0, ypitch=1.5)
    assert list(frames.values()) == [
        (0.0, 1.5, 1.0, 0.5),
        (2.0, 1.5, 1.0, 0.5),
        (4.0, 1.5, 1.0, 0.5),
        (0.0, 0.0, 1.0, 0.5),
        (2.0, 0.0, 1.0, 0.5),
    ]
    assert fl.grid_frames(axes[:3], 1.0, 0.5, xpitch=1.5)[axes[2]] == (3.0, 0.0, 1.0, 0.5)


def test_row_frames_values() -> None:
    _, axes = build_panel(n=3)
    frames = fl.row_frames(axes, xs=[0.0, 1.5, 3.1], ws=[1.0, 1.2, 0.9], h=0.8)
    assert list(frames.values()) == [
        (0.0, 0.0, 1.0, 0.8),
        (1.5, 0.0, 1.2, 0.8),
        (3.1, 0.0, 0.9, 0.8),
    ]
    with pytest.raises(ValueError):
        fl.row_frames(axes, xs=[0.0], ws=[1.0, 1.0, 1.0], h=0.8)


def test_grid_frames_land_exactly() -> None:
    fig, frames = _laid_out(grid_slot)
    _assert_frames_land(fig, frames)


def test_row_frames_land_exactly() -> None:
    fig, frames = _laid_out(
        lambda axes: fl.row_frames(
            axes, xs=[0, 1.7, 3.0, 4.9, 6.2, 7.5], ws=[1.0, 0.9, 1.3, 1.1, 0.8, 1.2], h=0.95
        )
    )
    _assert_frames_land(fig, frames)


def test_check_frames_raises_on_deviation() -> None:
    fig, frames = _laid_out(grid_slot)
    ax = next(iter(frames))
    p = ax.get_position()
    ax.set_position((p.x0, p.y0, p.width * 1.01, p.height))
    with pytest.raises(AssertionError):
        fl.check_frames(fig, frames)


def test_check_neighbours_raises_on_overlap() -> None:
    fig, axes = build_panel(n=2)
    fl.restyle(fig, style_for_tests())
    # frames closer than the decorations of the left axes
    frames = fl.row_frames(axes, xs=[0.0, 1.05], ws=[1.0, 1.0], h=1.0)
    fl.fit_page(fig, frames)
    with pytest.raises(AssertionError):
        fl.check_neighbours(fig, frames)


def test_fit_page_crops_to_content() -> None:
    fig, _ = _laid_out(grid_slot)
    w, h = fig.get_size_inches()
    r = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    bb = fig.get_tightbbox(r)
    assert (bb.x0, bb.y0) == pytest.approx((0.02, 0.02), abs=1e-6)
    assert (bb.x1, bb.y1) == pytest.approx((w - 0.02, h - 0.02), abs=1e-6)


# ── Rendering ───────────────────────────────────────────────────────────────────────


def test_render_to_slot_report(tmp_path: Path, epoch: None) -> None:
    out = tmp_path / "sub" / "panel.pdf"
    report = fl.render_to_slot(build_panel, out, style_for_tests(), grid_slot)
    assert out.is_file()
    assert len(report.frames) == 6
    assert all(f == (1.4, 1.1) for f in report.frames)
    assert report.removed == []
    assert "tick 5" in report.fonts and "label 8" in report.fonts and "suptitle 6/8" in report.fonts


def test_render_to_slot_collects_removed_texts(tmp_path: Path, epoch: None) -> None:
    def edit(fig: Figure, axes: list[Axes]) -> list[str]:
        return fl.pull_texts(axes)

    report = fl.render_to_slot(
        lambda: build_panel(n=2),
        tmp_path / "p.pdf",
        style_for_tests(),
        lambda axes: fl.row_frames(axes, [0.0, 2.0], [1.4, 1.4], 1.1),
        edit=edit,
    )
    assert report.removed == ["n = 0", "Title 0", "n = 1", "Title 1"]


def test_render_to_slot_is_byte_deterministic(tmp_path: Path, epoch: None) -> None:
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    fl.render_to_slot(build_panel, a, style_for_tests(), grid_slot)
    plt.close("all")
    fl.render_to_slot(build_panel, b, style_for_tests(), grid_slot)
    assert a.read_bytes() == b.read_bytes()
    assert b"/CreationDate (D:20200101000000Z)" in a.read_bytes()  # SOURCE_DATE_EPOCH honoured


def test_render_to_slot_leaves_no_rcparams(tmp_path: Path, epoch: None) -> None:
    before = _rc_snapshot()
    style = style_for_tests()
    assert style.hatch_lw != before["hatch.linewidth"]  # a leak would be visible
    base = before["font.size"] + 1.0
    fl.render_to_slot(build_panel, tmp_path / "p.pdf", style, grid_slot, base_font_size=base)
    assert _rc_snapshot() == before
    # control: restyle alone does set the global (which is why render_to_slot contains it)
    with plt.rc_context():
        fig, _ = build_panel()
        fl.restyle(fig, style)
        assert mpl.rcParams["hatch.linewidth"] == style.hatch_lw


def test_render_to_slot_sets_base_font_size_for_build(tmp_path: Path, epoch: None) -> None:
    seen: list[float] = []

    def build() -> tuple[Figure, list[Axes]]:
        seen.append(mpl.rcParams["font.size"])
        return build_plain()

    fl.render_to_slot(
        build,
        tmp_path / "p.pdf",
        fl.TextStyle(),
        lambda axes: fl.row_frames(axes, [0, 1.8, 3.6], [1.4] * 3, 1.0),
        base_font_size=9.0,
    )
    assert seen == [9.0]


def test_render_native_writes_and_leaves_no_rcparams(tmp_path: Path, epoch: None) -> None:
    before = _rc_snapshot()
    seen: list[float] = []

    def build() -> tuple[Figure, list[Axes]]:
        seen.append(mpl.rcParams["font.size"])
        mpl.rcParams["hatch.linewidth"] = 3.0  # what a panel's own code may do
        return build_plain()

    out = tmp_path / "n" / "native.pdf"
    fl.render_native(build, out)
    assert out.is_file() and out.stat().st_size > 0
    fl.render_native(build, tmp_path / "n" / "native2.pdf", rc={"font.size": 11.0})
    assert seen == [8.0, 11.0]
    assert _rc_snapshot() == before


def test_render_native_savefig_kwargs(tmp_path: Path, epoch: None) -> None:
    tight = tmp_path / "tight.pdf"
    loose = tmp_path / "loose.pdf"
    fl.render_native(build_plain, tight)
    fl.render_native(build_plain, loose, savefig_kwargs={"bbox_inches": None})
    (tw, th), (lw, lh) = fl.pdf_mediabox(tight), fl.pdf_mediabox(loose)
    assert (lw, lh) == pytest.approx((5.0, 2.0), abs=0.01)
    assert tw < lw or th < lh  # tight crops the margins


# ── Page size ───────────────────────────────────────────────────────────────────────


def test_pdf_mediabox_matches_fit_page(tmp_path: Path, epoch: None) -> None:
    fig, frames = _laid_out(grid_slot)
    page = fl.fit_page(fig, frames)
    out = tmp_path / "p.pdf"
    fig.savefig(out, format="pdf", dpi=300, transparent=True)
    mw, mh = fl.pdf_mediabox(out)
    assert abs(mw - page[0]) < 0.01 and abs(mh - page[1]) < 0.01


def test_render_to_slot_page_matches_mediabox(tmp_path: Path, epoch: None) -> None:
    out = tmp_path / "p.pdf"
    report = fl.render_to_slot(build_panel, out, style_for_tests(), grid_slot)
    mw, mh = fl.pdf_mediabox(out)
    assert abs(mw - report.page[0]) < 0.01 and abs(mh - report.page[1]) < 0.01


def test_pdf_mediabox_rejects_non_pdf(tmp_path: Path) -> None:
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"not a pdf")
    with pytest.raises(ValueError):
        fl.pdf_mediabox(junk)


# ── Style and generic edits ─────────────────────────────────────────────────────────


def test_text_style_from_dict() -> None:
    base = {"tick": 5, "label": 8, "bold_title": True}
    style = fl.TextStyle.from_dict(base, {"label": 9, "lw": 1.0})
    assert (style.tick, style.label, style.lw, style.bold_title) == (5, 9, 1.0, True)
    assert style.hatch_lw == 0.5  # default kept
    assert fl.TextStyle.from_dict(base) == fl.TextStyle(tick=5, label=8, bold_title=True)
    with pytest.raises(ValueError, match="nope"):
        fl.TextStyle.from_dict(base, {"nope": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        style.tick = 9  # type: ignore[misc]


def test_restyle_sets_sizes_by_role() -> None:
    fig, axes = build_panel(n=2)
    style = fl.TextStyle(tick=5, label=8, title=7, legend=4, text=3, suptitle=9, bold_title=True)
    with plt.rc_context():
        fl.restyle(fig, style)
    ax = axes[0]
    assert ax.xaxis.label.get_fontsize() == 8
    assert all(t.get_fontsize() == 5 for t in ax.get_xticklabels())
    assert ax.title.get_fontsize() == 7 and ax.title.get_fontweight() == "bold"
    assert ax.texts[0].get_fontsize() == 3
    legend = ax.get_legend()
    assert legend is not None and legend.get_texts()[0].get_fontsize() == 4
    assert fig._suptitle.get_fontsize() == 9  # type: ignore[attr-defined]


def test_font_report() -> None:
    fig, _ = build_panel(n=2)
    with plt.rc_context():
        fl.restyle(fig, fl.TextStyle(tick=5, label=8, title=7, legend=4, text=3, suptitle=9))
    fig.canvas.draw()  # tick labels exist after a draw
    # fig.text("figure text") is reported with the suptitle and keeps the text size
    assert fl.font_report(fig) == "tick 5, label 8, title 7, text 3, legend 4, suptitle 3/9"


def test_drop_ylabels() -> None:
    _, axes = build_panel(n=3)
    fl.drop_ylabels(axes, keep_first=True)
    assert [a.get_ylabel() for a in axes] == ["Occupancy Ratio", "", ""]
    fl.drop_ylabels(axes)
    assert [a.get_ylabel() for a in axes] == ["", "", ""]


def test_pull_texts() -> None:
    _, axes = build_panel(n=2)
    assert fl.pull_texts(axes, titles=False) == ["n = 0", "n = 1"]
    assert all(not a.texts and a.get_title() for a in axes)
    assert fl.pull_texts(axes) == ["Title 0", "Title 1"]
    assert all(a.get_title() == "" for a in axes)


def test_sentence_case_ylabels() -> None:
    _, axes = build_panel(n=2)
    axes[1].set_ylabel("Other")
    fl.sentence_case_ylabels(axes)
    assert [a.get_ylabel() for a in axes] == ["Occupancy ratio", "Other"]
    fl.sentence_case_ylabels(axes, {"Other": "Changed"})
    assert axes[1].get_ylabel() == "Changed"


def test_fit_ylim_top() -> None:
    _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1.3])
    assert fl.fit_ylim_top(ax, 1.0) == [f"y range 0{DASH}1.5"]  # 1.3 > 0.97 * 1.0 but < 0.97 * 1.5
    assert ax.get_ylim() == (0, 1.5)
    assert fl.fit_ylim_top(ax, 4.0) == [f"y range 0{DASH}4"]
    with pytest.raises(ValueError):
        fl.fit_ylim_top(ax, 1.0, step=0.0)


def test_set_ylim_headroom() -> None:
    _, ax = plt.subplots()
    assert fl.set_ylim_headroom(ax) == []
    assert ax.get_ylim() == (0, 1.4)
    assert list(ax.get_yticks()) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fl.set_ylim_headroom(ax, top=1.2, tick_step=0.5)
    assert ax.get_ylim() == (0, 1.2)
    assert list(ax.get_yticks()) == pytest.approx([0, 0.5, 1.0])


def test_remove_collections() -> None:
    _, axes = build_panel(n=2)
    assert fl.remove_collections(axes) == ["envelopes"]
    assert all(not a.collections and a.get_lines() for a in axes)


def test_legend_above(tmp_path: Path, epoch: None) -> None:
    _, axes = build_panel(n=2)
    other = axes[1].get_legend()
    assert other is not None
    other.remove()
    assert fl.legend_above(axes, fontsize=5) == []
    legend = axes[0].get_legend()
    assert legend is not None and legend.get_texts()[0].get_fontsize() == 5
    assert legend._ncols == 2  # type: ignore[attr-defined]  # one row
    assert axes[1].get_legend() is None
