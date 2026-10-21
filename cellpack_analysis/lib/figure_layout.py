"""Place matplotlib panels into exact page slots, set every text by role, and check the result.

``render_to_slot`` draws one panel with a caller-supplied builder, applies slot edits, restyles
it by text role (:class:`TextStyle`), gives every axes an exact frame (spine to spine, inches),
crops the page to the drawn content and saves a PDF whose size is checked against the layout.
A placed PDF needs no rescaling.

The layout calls draw the figure to measure it and so need an Agg canvas (``MPLBACKEND=Agg``).
Output is byte-reproducible when ``SOURCE_DATE_EPOCH`` is set, since matplotlib's PDF backend
then takes ``CreationDate`` from it.
"""

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

logger = logging.getLogger(__name__)

Frame = tuple[float, float, float, float]  # x, y, w, h in inches
FrameMap = dict[Axes, Frame]
EditFn = Callable[[Figure, list[Axes]], list[str] | None]
BuildFn = Callable[[], tuple[Figure, list[Axes]]]


@dataclass(frozen=True)
class TextStyle:
    """Point sizes by text role, a line-width scale and the axes weights of one slot."""

    tick: float = 7
    label: float = 7
    title: float = 8
    legend: float = 6
    text: float = 6
    suptitle: float = 8
    bold_title: bool = False
    lw: float = 0.75
    spine: float = 0.6
    ticklen: float = 2.5
    tickpad: float = 2.0
    labelpad: float = 3.0
    titlepad: float = 4.0
    hatch_lw: float = 0.5

    @classmethod
    def from_dict(
        cls, base: Mapping[str, Any], override: Mapping[str, Any] | None = None
    ) -> "TextStyle":
        """Build a style from ``base`` updated by ``override``; unknown keys raise."""
        merged = {**base, **(override or {})}
        unknown = sorted(set(merged) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown TextStyle fields: {unknown}")
        return cls(**merged)


@dataclass
class SlotReport:
    """What ``render_to_slot`` drew: page size, frame sizes, font sizes and removed texts."""

    page: tuple[float, float]
    frames: list[tuple[float, float]]
    fonts: str
    removed: list[str]


def _ensure(ok: bool | np.bool_, *detail: object) -> None:
    """Raise AssertionError (also under ``python -O``) when a layout invariant fails."""
    if not ok:
        raise AssertionError(detail)


def _fig_texts(fig: Figure) -> list[Text]:
    return [t for t in [fig._suptitle, *fig.texts] if t is not None and t.get_text()]  # type: ignore[attr-defined]


def restyle(fig: Figure, style: TextStyle) -> None:
    """Set every text by role and scale line weights; axes weights are set outright."""
    k = style.lw
    for ax in fig.axes:
        ax.tick_params(
            which="both",
            labelsize=style.tick,
            width=style.spine,
            length=style.ticklen,
            pad=style.tickpad,
        )
        ax.tick_params(which="minor", length=style.ticklen * 0.6)
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontsize(style.label)
            axis.labelpad = style.labelpad
            axis.get_offset_text().set_fontsize(style.tick)
        for t in (ax.title, ax._left_title, ax._right_title):  # type: ignore[attr-defined]
            t.set_fontsize(style.title)
            if style.bold_title:
                t.set_fontweight("bold")
        ax._set_title_offset_trans(float(style.titlepad))  # type: ignore[attr-defined]
        for t in ax.texts:
            t.set_fontsize(style.text)
        for sp in ax.spines.values():
            sp.set_linewidth(style.spine)
        for ln in ax.get_lines():
            ln.set_linewidth(ln.get_linewidth() * k)
            ln.set_markersize(ln.get_markersize() * k)
            ln.set_markeredgewidth(ln.get_markeredgewidth() * k)
        for p in ax.patches:
            p.set_linewidth(p.get_linewidth() * k)
        for c in ax.collections:
            c.set_linewidth(np.asarray(c.get_linewidth(), float) * k)  # type: ignore[arg-type]
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(style.legend)
            leg.get_title().set_fontsize(style.legend)
            for h in leg.legend_handles:
                if hasattr(h, "set_linewidth"):
                    h.set_linewidth(h.get_linewidth() * k)  # type: ignore[union-attr]
    for t in fig.texts:
        t.set_fontsize(style.text)
    if fig._suptitle is not None:  # type: ignore[attr-defined]  # also in fig.texts
        fig._suptitle.set_fontsize(style.suptitle)  # type: ignore[attr-defined]
    plt.rcParams["hatch.linewidth"] = style.hatch_lw


def grid_frames(
    axes: Sequence[Axes],
    w: float,
    h: float,
    ncols: int | None = None,
    xpitch: float = 0.0,
    ypitch: float = 0.0,
) -> FrameMap:
    """Frames (x, y, w, h) in inches for axes in row-major order, top row first."""
    ncols = ncols or len(axes)
    nrows = -(-len(axes) // ncols)
    return {
        ax: ((i % ncols) * xpitch, (nrows - 1 - i // ncols) * ypitch, w, h)
        for i, ax in enumerate(axes)
    }


def row_frames(
    axes: Sequence[Axes], xs: Sequence[float], ws: Sequence[float], h: float
) -> FrameMap:
    """One row of frames at explicit x offsets and widths (hand-placed panels)."""
    return {ax: (x, 0.0, w, h) for ax, x, w in zip(axes, xs, ws, strict=True)}


def _tight_inches(ax: Axes, renderer: Any, **kwargs: Any) -> Bbox:
    """Return the axes' tight bounding box in inches."""
    bb = ax.get_tightbbox(renderer, **kwargs)
    if bb is None:
        raise ValueError("axes has no drawn extent")
    return bb.transformed(ax.figure.dpi_scale_trans.inverted())


def fit_page(fig: Figure, frames: FrameMap, pad: float = 0.02) -> tuple[float, float]:
    """Give each axes its exact frame and crop the page to the drawn content (inches)."""
    m = 2.0  # working margin around the frames while the decorations are measured
    x0 = min(x for x, _, _, _ in frames.values()) - m
    y0 = min(y for _, y, _, _ in frames.values()) - m
    page_w = max(x + w for x, _, w, _ in frames.values()) - x0 + m
    page_h = max(y + h for _, y, _, h in frames.values()) - y0 + m
    fig.set_size_inches(page_w, page_h)
    for ax, (x, y, w, h) in frames.items():
        ax.set_position(((x - x0) / page_w, (y - y0) / page_h, w / page_w, h / page_h))
    r = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    suptitle = fig._suptitle  # type: ignore[attr-defined]
    if suptitle is not None and suptitle.get_text():
        fig.canvas.draw()
        top = max(_tight_inches(ax, r).y1 for ax in frames)
        xmin = min(x for x, _, _, _ in frames.values())
        xmax = max(x + w for x, _, w, _ in frames.values())
        suptitle.set_position((((xmin + xmax) / 2 - x0) / page_w, (top + 0.04) / page_h))
        suptitle.set_verticalalignment("bottom")
        suptitle.set_horizontalalignment("center")
    fig.canvas.draw()
    bb = fig.get_tightbbox(r)
    texts = [(t, t.get_position()) for t in _fig_texts(fig)]
    w2, h2 = bb.width + 2 * pad, bb.height + 2 * pad
    dx, dy = bb.x0 - pad, bb.y0 - pad
    fig.set_size_inches(w2, h2)
    for ax, (x, y, w, h) in frames.items():
        ax.set_position(((x - x0 - dx) / w2, (y - y0 - dy) / h2, w / w2, h / h2))
    for t, (px, py) in texts:
        t.set_position(((px * page_w - dx) / w2, (py * page_h - dy) / h2))
    fig.canvas.draw()
    bb2 = fig.get_tightbbox(r)
    _ensure(
        bb2.x0 > -0.005 and bb2.y0 > -0.005 and bb2.x1 < w2 + 0.005 and bb2.y1 < h2 + 0.005,
        "content outside the cropped page",
        bb2,
        w2,
        h2,
    )
    return float(w2), float(h2)


def check_frames(fig: Figure, frames: FrameMap, atol: float = 1e-6) -> list[tuple[float, float]]:
    """Check each axes is exactly its frame size; return the sizes in inches (3 dp)."""
    fig_w, fig_h = fig.get_size_inches()
    out = []
    for ax, (_, _, w, h) in frames.items():
        p = ax.get_position()
        got = (p.width * fig_w, p.height * fig_h)
        _ensure(np.allclose(got, (w, h), atol=atol), "frame size off", got, (w, h))
        out.append((round(float(got[0]), 3), round(float(got[1]), 3)))
    return out


def check_neighbours(fig: Figure, frames: FrameMap, tol: float = 0.01) -> None:
    """No axes' decorations (or its legend) may reach into another axes' frame."""
    r = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    inv = fig.dpi_scale_trans.inverted()
    items = list(frames)
    for a in items:
        boxes = [_tight_inches(a, r, bbox_extra_artists=[])]
        legend = a.get_legend()
        if legend is not None:
            boxes.append(legend.get_window_extent(r).transformed(inv))
        for b in items:
            if a is b:
                continue
            fb = b.get_window_extent(r).transformed(inv)
            for tb in boxes:
                ix = min(tb.x1, fb.x1) - max(tb.x0, fb.x0)
                iy = min(tb.y1, fb.y1) - max(tb.y0, fb.y0)
                _ensure(
                    not (ix > tol and iy > tol), "overlap", a.get_title(), b.get_title(), ix, iy
                )


def font_report(fig: Figure) -> str:
    """Font sizes in use by role, e.g. ``"tick 6, label 8, title 7"``."""
    sizes: dict[str, set[float]] = {}
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        legend = ax.get_legend()
        for role, ts in (
            ("tick", ax.get_xticklabels() + ax.get_yticklabels()),
            ("label", [ax.xaxis.label, ax.yaxis.label]),
            ("title", [ax.title]),
            ("text", ax.texts),
            ("legend", legend.get_texts() if legend else []),
        ):
            for t in ts:
                if t.get_text() and t.get_visible():
                    sizes.setdefault(role, set()).add(float(t.get_fontsize()))
    for t in _fig_texts(fig):
        sizes.setdefault("suptitle", set()).add(float(t.get_fontsize()))
    return ", ".join(f"{k} {'/'.join(f'{v:g}' for v in sorted(s))}" for k, s in sizes.items())


def pdf_mediabox(path: Path) -> tuple[float, float]:
    """Page size in inches from the PDF's MediaBox."""
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", path.read_bytes())
    if m is None:
        raise ValueError(f"no MediaBox in {path}")
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def render_to_slot(
    build: BuildFn,
    out: Path,
    style: TextStyle,
    frames: Callable[[list[Axes]], FrameMap],
    edit: EditFn | None = None,
    base_font_size: float = 8.0,
    dpi: int = 300,
    transparent: bool = True,
) -> SlotReport:
    """Draw one panel into its slot and save it as a PDF cropped to the drawn content.

    Parameters
    ----------
    build : Callable
        Returns ``(figure, axes)`` in the order ``frames`` expects.
    out : Path
        PDF path; parent directories are created.
    style : TextStyle
        Text sizes and line weights applied after ``edit``.
    frames : Callable
        Maps the built axes to their exact frames in inches.
    edit : EditFn, optional
        Slot edits run before ``restyle`` (``set_title`` resets the size ``restyle`` sets);
        returns the texts it removed.
    base_font_size : float
        ``font.size`` while building, for text no role covers.
    dpi : int
        Passed to ``savefig`` (rasterized parts only).
    transparent : bool
        Passed to ``savefig``.

    Returns
    -------
    SlotReport
        Page size, frame sizes, font sizes by role and the removed texts.
    """
    with plt.rc_context():  # restyle's rcParams must not leak into the next panel
        plt.close("all")
        plt.rcParams["font.size"] = base_font_size
        fig, axes = build()
        removed = (edit(fig, axes) if edit else None) or []
        restyle(fig, style)
        slot = frames(axes)
        page_w, page_h = fit_page(fig, slot)
        got = check_frames(fig, slot)
        check_neighbours(fig, slot)
        fonts = font_report(fig)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="pdf", dpi=dpi, transparent=transparent)
        plt.close(fig)
    mw, mh = pdf_mediabox(out)
    _ensure(
        abs(mw - page_w) < 0.01 and abs(mh - page_h) < 0.01,
        "MediaBox differs from page",
        out,
        (mw, mh),
        (page_w, page_h),
    )
    logger.debug("%s: page %.2f x %.2f in; %s", out.name, page_w, page_h, fonts)
    return SlotReport(
        page=(round(page_w, 2), round(page_h, 2)), frames=got, fonts=fonts, removed=removed
    )


def render_native(
    build: BuildFn,
    out: Path,
    rc: Mapping[str, Any] | None = None,
    savefig_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Save a panel exactly as the pipeline draws it, for a pixel check of the data feed."""
    with plt.rc_context():
        plt.close("all")
        plt.rcParams["font.size"] = 8.0
        plt.rcParams.update(rc or {})
        fig, _ = build()
        out.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"dpi": 300, "bbox_inches": "tight", "transparent": True, **(savefig_kwargs or {})}
        fig.savefig(out, **kwargs)
        plt.close(fig)


# ── Generic slot edits ─────────────────────────────────────────────────────────────


def drop_ylabels(axes: Sequence[Axes], keep_first: bool = False) -> None:
    """Clear the y labels, optionally keeping the first axes' label."""
    for i, ax in enumerate(axes):
        if not (keep_first and i == 0):
            ax.set_ylabel("")


def pull_texts(axes: Sequence[Axes], titles: bool = True) -> list[str]:
    """Remove in-panel texts (and titles) a slot does not carry; return them for the manifest."""
    out = []
    for ax in axes:
        for t in list(ax.texts):
            out.append(t.get_text().replace("\n", "; "))
            t.remove()
        if titles and ax.get_title():
            out.append(ax.get_title().replace("\n", " "))
            ax.set_title("")
    return out


def sentence_case_ylabels(axes: Sequence[Axes], mapping: Mapping[str, str] | None = None) -> None:
    """Replace y labels found in ``mapping`` (default ``Occupancy Ratio`` to sentence case)."""
    mapping = {"Occupancy Ratio": "Occupancy ratio"} if mapping is None else mapping
    for ax in axes:
        label = ax.get_ylabel()
        if label in mapping:
            ax.set_ylabel(mapping[label])


def fit_ylim_top(ax: Axes, top: float, step: float = 0.5, fill: float = 0.97) -> list[str]:
    """Set the y range to ``0..top``, raised in ``step`` until the highest drawn line fits."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    peak = max(np.nanmax(np.asarray(ln.get_ydata(), float)) for ln in ax.get_lines())
    while peak > fill * top:
        top += step
    ax.set_ylim(0, top)
    return [f"y range 0\N{EN DASH}{top:g}"]


def set_ylim_headroom(ax: Axes, top: float = 1.4, tick_step: float = 0.2) -> list[str]:
    """Add headroom above 1 for a legend; ticks stay on 0 to 1."""
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, 1.01, tick_step))
    return []


def legend_above(axes: Sequence[Axes], fontsize: float, dy: float = 0.33) -> list[str]:
    """Move the one in-panel legend above the grid, in one row, so it covers no curve.

    It ends at the right edge of the axes that carries it, ``dy`` inches above that frame.
    """
    src = next(ax for ax in axes if ax.get_legend() is not None)
    h, lab = src.get_legend_handles_labels()
    legend = src.get_legend()
    assert legend is not None  # chosen above for having one
    legend.remove()
    up = src.transAxes + mtransforms.ScaledTranslation(0, dy, src.figure.dpi_scale_trans)
    src.legend(
        h,
        lab,
        ncol=len(h),
        frameon=False,
        fontsize=fontsize,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=up,
        borderaxespad=0.0,
        handlelength=1.5,
        columnspacing=1.0,
    )
    return []


def remove_collections(axes: Sequence[Axes]) -> list[str]:
    """Remove every collection (envelope bands) and keep the lines."""
    for ax in axes:
        for c in list(ax.collections):
            c.remove()
    return ["envelopes"]
