r"""Draw the manuscript figure assets from the pipeline caches and write their MANIFEST.

One config, ordered steps::

    python -m cellpack_analysis.analysis.workflows.run_manuscript_workflow \
        -c cellpack_analysis/analysis/workflows/configs/manuscript/<config>.json \
        [--only REGEX] [--native DIR] [--steps STEP ...]

Steps, always run in this order:

- ``check_inputs``: λ guard, replicate counts, panel arguments, and every cache the selected
  assets read (with its parameters and cell count). All problems are reported at once.
- ``figure_assets``: draw each selected asset into its slot (or at native size) and record its
  page, frames, fonts, removed texts and inputs in ``MANIFEST.json``.
- ``figure_numbers`` / ``calibration_summary``: the numbers sheets and the Methods calibration
  summary (``manuscript_numbers`` / ``manuscript_calibration``).
- ``manifest``: write ``MANIFEST.md`` from ``MANIFEST.json`` and log unmanaged files.

Nothing is recomputed: a missing or mismatched cache raises, naming the step that writes it.
Set ``SOURCE_DATE_EPOCH`` for byte-reproducible PDFs.
"""

import argparse
import functools
import importlib
import inspect
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # fit_page measures on an Agg canvas

from cellpack_analysis import setup_logging
from cellpack_analysis.analysis.workflows import manuscript_panels as mp
from cellpack_analysis.analysis.workflows.manuscript_config import (
    DEFAULT_STEPS,
    AssetSpec,
    ManuscriptConfig,
)
from cellpack_analysis.analysis.workflows.manuscript_sources import (
    PaperSources,
    check_canonical_apical_lambda,
)
from cellpack_analysis.lib import figure_layout as fl
from cellpack_analysis.lib.file_io import PROVENANCE_FILE, write_run_provenance

logger = logging.getLogger(__name__)

MANIFEST_JSON = "MANIFEST.json"
MANIFEST_MD = "MANIFEST.md"
MANIFEST_VERSION = 1
TIMES = "\N{MULTIPLICATION SIGN}"
# Files other steps write into the asset folder; never reported as unmanaged.
UNMANAGED_IGNORE = ("*_numbers.md", "numbers.json", "MANIFEST.*", PROVENANCE_FILE)
# (module, function) behind the steps T8 and T9 provide.
EXTERNAL_STEPS = {
    "figure_numbers": (
        "cellpack_analysis.analysis.workflows.manuscript_numbers",
        "write_figure_numbers",
    ),
    "calibration_summary": (
        "cellpack_analysis.analysis.workflows.manuscript_calibration",
        "write_calibration_summary",
    ),
}
# Errors check_inputs collects; anything else is a bug and propagates.
INPUT_ERRORS = (FileNotFoundError, ValueError, KeyError, TypeError)


def _frames(frames: dict[str, Any]) -> Callable[[list[Any]], fl.FrameMap]:
    """Frame layout for an asset's ``frames`` spec."""
    args = {k: v for k, v in frames.items() if k != "kind"}
    if frames["kind"] == "grid":
        return lambda axes: fl.grid_frames(axes, **args)
    return lambda axes: fl.row_frames(axes, **args)


def _fmt(x: float) -> str:
    return f"{x:g}"


class ManuscriptRunner:
    """Run the manuscript workflow steps for one config.

    Parameters
    ----------
    config
        The manuscript config.
    sources
        Cache resolver; a new one when ``None``.
    """

    def __init__(self, config: ManuscriptConfig, sources: PaperSources | None = None) -> None:
        config.check_registries(mp.PANELS, mp.EDITS)
        self.config = config
        self.sources = sources if sources is not None else PaperSources(config)
        self.out_dir = config.results_dir
        self.steps: dict[str, Callable[[], None]] = {
            "check_inputs": self._check_inputs,
            "figure_assets": self._figure_assets,
            "figure_numbers": self._figure_numbers,
            "calibration_summary": self._calibration_summary,
            "manifest": self._write_manifest,
        }
        if list(self.steps) != DEFAULT_STEPS:
            raise RuntimeError(f"Step table {list(self.steps)} != {DEFAULT_STEPS}")
        self.selected: list[AssetSpec] = list(config.assets)
        self.native_dir: Path | None = None

    # ── driver ───────────────────────────────────────────────────────────────────

    def run(
        self,
        steps: Sequence[str] | None = None,
        only: str | None = None,
        native_dir: Path | None = None,
    ) -> None:
        """Run ``steps`` (default: the config's) in workflow order.

        Before the first step that writes, one line is appended to ``run_provenance.jsonl`` in
        the results directory.

        Parameters
        ----------
        steps
            Step names; ``None`` runs the config's ``steps``.
        only
            Regular expression; only assets whose path matches are checked and drawn.
        native_dir
            Also save each selected asset at native size under this folder.
        """
        requested = list(self.config.steps if steps is None else steps)
        unknown = [s for s in requested if s not in self.steps]
        if unknown:
            raise ValueError(f"Unknown steps {unknown}; known {list(self.steps)}")
        self.selected = self.config.select_assets(only)
        self.native_dir = native_dir
        if "SOURCE_DATE_EPOCH" not in os.environ:
            logger.warning("SOURCE_DATE_EPOCH is unset; PDFs carry the current CreationDate")
        recorded = False
        for name in self.steps:
            if name in requested:
                if name != "check_inputs" and not recorded:  # a failed check writes nothing
                    write_run_provenance(self.out_dir, self.config.config_file)
                    recorded = True
                logger.info("Step %s (%d assets selected)", name, len(self.selected))
                self.steps[name]()

    def _context(self, spec: AssetSpec, native: bool = False) -> mp.PanelContext:
        style = None if native or spec.style is None else self.config.styles[spec.style]
        return mp.PanelContext(self.sources, spec.structure, spec.measure, style)

    def _rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.config.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    # ── check_inputs ─────────────────────────────────────────────────────────────

    def _check_inputs(self) -> None:
        """Check guards, arguments and every cache the selected assets read; raise once."""
        errors: dict[str, None] = {}

        def attempt(what: str, fn: Callable[[], Any]) -> None:
            try:
                fn()
            except INPUT_ERRORS as e:
                errors.setdefault(f"{what}: {e}", None)

        for s in self.config.structures:
            attempt(f"{s} packing configs", functools.partial(self._check_packing, s))
        for spec in self.selected:
            attempt(spec.path, functools.partial(self._check_asset, spec))
        if errors:
            raise RuntimeError(
                f"check_inputs found {len(errors)} problem(s):\n- " + "\n- ".join(errors)
            )
        logger.info(
            "check_inputs: %d assets OK; %d input files",
            len(self.selected),
            len(self.sources.inputs),
        )

    def _check_packing(self, structure: str) -> None:
        """λ guard on the condition packing config, and replicate counts of both."""
        expected = self.config.expected
        path = self.sources.packing_config_path("condition", structure)
        check_canonical_apical_lambda(path)
        for key, want in (
            ("condition", expected["n_replicates"]),
            ("ladder", expected["ladder_replicates"]),
        ):
            got = self.sources.packing_config(key, structure).get("number_of_replicates")
            if got != want:
                raise ValueError(
                    f"{self.sources.packing_config_path(key, structure)}: "
                    f"number_of_replicates {got} != expected {want}"
                )

    def _check_asset(self, spec: AssetSpec) -> None:
        """Bind the asset's arguments, then load what its panel and edits read."""
        variants = [(spec.panel_args, spec.render == "native")]
        if self.native_dir is not None and spec.render == "slot":
            variants.append((spec.native_panel_args, True))
        for args, native in variants:
            ctx = self._context(spec, native)
            inspect.signature(mp.PANELS[spec.panel]).bind(ctx, **args)
            mp.PANEL_INPUTS[spec.panel](ctx, **args)
        ctx = self._context(spec)
        for name in spec.edits:
            params = spec.edit_params.get(name, {})
            inspect.signature(mp.EDITS[name]).bind(None, [], ctx, **params)
            if name in mp.EDIT_INPUTS:
                mp.EDIT_INPUTS[name](ctx, **params)

    # ── figure_assets ────────────────────────────────────────────────────────────

    def _edit(self, spec: AssetSpec, ctx: mp.PanelContext) -> fl.EditFn | None:
        if not spec.edits:
            return None

        def edit(fig: Any, axes: list[Any]) -> list[str]:
            removed: list[str] = []
            for name in spec.edits:
                removed += mp.EDITS[name](fig, axes, ctx, **spec.edit_params.get(name, {})) or []
            return removed

        return edit

    def _render(self, spec: AssetSpec) -> dict[str, Any]:
        """Draw one asset; return its MANIFEST record (without inputs)."""
        out = self.out_dir / spec.path
        panel = mp.PANELS[spec.panel]
        if spec.render == "slot":
            assert spec.style is not None and spec.frames is not None  # checked by the config
            ctx = self._context(spec)
            report = fl.render_to_slot(
                build=lambda: panel(ctx, **spec.panel_args),
                out=out,
                style=self.config.styles[spec.style],
                frames=_frames(spec.frames),
                edit=self._edit(spec, ctx),
                base_font_size=self.config.render["base_font_size"],
                dpi=self.config.render["dpi"],
                transparent=self.config.render["transparent"],
            )
            record = {
                "render": "slot",
                "page": list(report.page),
                "frames": [list(f) for f in report.frames],
                "fonts": report.fonts,
                "removed": list(report.removed),
            }
        else:
            ctx = self._context(spec, native=True)
            fl.render_native(
                lambda: panel(ctx, **spec.panel_args), out, rc=spec.rc, savefig_kwargs=spec.savefig
            )
            w, h = fl.pdf_mediabox(out)
            record = {
                "render": "native",
                "page": [round(w, 2), round(h, 2)],
                "frames": [],
                "fonts": f"base {_fmt(spec.rc.get('font.size', 8.0))}",
                "removed": [],
            }
        if self.native_dir is not None:
            args = spec.native_panel_args if spec.render == "slot" else spec.panel_args
            native_ctx = self._context(spec, native=True)
            rc, savefig = (spec.rc, spec.savefig) if spec.render == "native" else (None, None)
            fl.render_native(
                lambda: panel(native_ctx, **args),
                self.native_dir / spec.path,
                rc=rc,
                savefig_kwargs=savefig,
            )
        return record

    def _figure_assets(self) -> None:
        """Draw every selected asset and merge its record into ``MANIFEST.json``."""
        records: dict[str, dict[str, Any]] = {}
        for spec in self.selected:
            with self.sources.record_inputs() as used:
                record = self._render(spec)
            record["inputs"] = [self._rel(p) for p in used]
            records[spec.path] = record
            logger.info(
                "%s: page %s in; %s%s",
                spec.path,
                f" {TIMES} ".join(_fmt(x) for x in record["page"]),
                record["fonts"],
                f"; removed: {' | '.join(record['removed'])}" if record["removed"] else "",
            )
        self._update_manifest(assets=records)
        logger.info("Drew %d assets into %s", len(records), self.out_dir)

    # ── external steps ───────────────────────────────────────────────────────────

    def _external(self, step: str) -> None:
        module_name, func = EXTERNAL_STEPS[step]
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name != module_name:
                raise
            raise NotImplementedError(
                f"Step {step!r} needs {module_name}.{func}, which does not exist yet"
            ) from e
        written = getattr(module, func)(self.config, self.sources)
        rels = []
        for p in map(Path, written):
            path = Path(os.path.normpath(p if p.is_absolute() else self.out_dir / p))
            if not path.is_relative_to(self.out_dir):
                raise ValueError(f"{step} wrote {p} outside {self.out_dir}")
            rels.append(path.relative_to(self.out_dir).as_posix())
        self._update_manifest(outputs={step: sorted(rels)})
        logger.info("%s wrote %d files", step, len(rels))

    def _figure_numbers(self) -> None:
        """Numbers sheets: ``manuscript_numbers.write_figure_numbers(config, sources)``."""
        self._external("figure_numbers")

    def _calibration_summary(self) -> None:
        """Calibration: ``manuscript_calibration.write_calibration_summary(config, sources)``."""
        self._external("calibration_summary")

    # ── MANIFEST ─────────────────────────────────────────────────────────────────

    def _read_manifest(self) -> dict[str, Any]:
        path = self.out_dir / MANIFEST_JSON
        if not path.exists():
            return {"version": MANIFEST_VERSION, "assets": {}, "outputs": {}}
        data = json.loads(path.read_text())
        if data.get("version") != MANIFEST_VERSION:
            raise ValueError(f"{path}: version {data.get('version')} != {MANIFEST_VERSION}")
        return data

    def _update_manifest(
        self,
        assets: dict[str, dict[str, Any]] | None = None,
        outputs: dict[str, list[str]] | None = None,
    ) -> None:
        """Merge records into ``MANIFEST.json`` by path; drop assets no longer configured."""
        data = self._read_manifest()
        data["assets"].update(assets or {})
        data["outputs"].update(outputs or {})
        known = {spec.path for spec in self.config.assets}
        stale = sorted(set(data["assets"]) - known)
        for path in stale:
            logger.warning("MANIFEST.json: dropping %s (not in the config)", path)
            del data["assets"][path]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        (self.out_dir / MANIFEST_JSON).write_text(text, encoding="utf-8")

    def _write_manifest(self) -> None:
        """Write ``MANIFEST.md`` from ``MANIFEST.json`` and log unmanaged files."""
        data = self._read_manifest()
        (self.out_dir / MANIFEST_MD).write_text(self._manifest_markdown(data), encoding="utf-8")
        written = set(data["assets"]).union(*data["outputs"].values())
        self._report_unmanaged(written)

    def _manifest_markdown(self, data: dict[str, Any]) -> str:
        config_rel = self._rel(self.config.config_file.resolve())
        lines = [
            f"# Manuscript figure assets ({self.config.condition})",
            "",
            f"Drawn by `run_manuscript_workflow -c {config_rel}` from the pipeline caches with "
            "the pipeline's own plot functions; nothing is recomputed.",
            "Each slot PDF drops in at 100 %: every axes frame (spine to spine) matches its slot, "
            "every text is set by role in points, and the page is cropped to the drawn content.",
            "Native PDFs keep the plot function's size. Texts a slot does not carry are listed "
            "under Removed, so they can be typeset.",
            "",
            "## Drop-in sizes",
            "",
            f"| Asset | Page (in) | Frames w {TIMES} h (in) | Fonts (pt) | Removed from panel |",
            "|---|---|---|---|---|",
        ]
        missing = []
        for spec in self.config.assets:
            r = data["assets"].get(spec.path)
            if r is None:
                missing.append(spec.path)
                continue
            page = f" {TIMES} ".join(_fmt(x) for x in r["page"])
            if r["render"] == "slot":
                sizes = sorted({f"{_fmt(w)} {TIMES} {_fmt(h)}" for w, h in r["frames"]})
                frames = f"{len(r['frames'])} {TIMES} {', '.join(sizes)}"
            else:
                frames = "native"
            removed = "; ".join(r["removed"]) or "—"
            lines.append(f"| `{spec.path}` | {page} | {frames} | {r['fonts']} | {removed} |")
        lines += ["", "## Inputs", "", "| Asset | Files read |", "|---|---|"]
        for spec in self.config.assets:
            r = data["assets"].get(spec.path)
            if r is not None:
                files = "<br>".join(f"`{p}`" for p in r["inputs"]) or "—"
                lines.append(f"| `{spec.path}` | {files} |")
        if data["outputs"]:
            lines += ["", "## Other outputs", "", "| Step | Files |", "|---|---|"]
            for step in DEFAULT_STEPS:
                if step in data["outputs"]:
                    files = "<br>".join(f"`{p}`" for p in data["outputs"][step]) or "—"
                    lines.append(f"| `{step}` | {files} |")
        if missing:
            lines += ["", "## Not drawn yet", ""] + [f"- `{p}`" for p in missing]
        return "\n".join(lines) + "\n"

    def _report_unmanaged(self, written: set[str]) -> list[str]:
        """Log files in the asset folder that no step manages (stale panels, strays)."""
        managed = {spec.path for spec in self.config.assets} | written
        stray = []
        for path in sorted(self.out_dir.rglob("*")):
            rel = path.relative_to(self.out_dir).as_posix()
            if path.is_dir() or rel in managed:
                continue
            if any(path.match(pattern) for pattern in UNMANAGED_IGNORE):
                continue
            stray.append(rel)
        for rel in stray:
            logger.warning("Unmanaged file in %s: %s", self.out_dir, rel)
        return stray


def main() -> None:
    """Run the manuscript workflow from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config_file", "-c", required=True, help="Manuscript JSON config")
    parser.add_argument("--only", help="Only assets whose path matches this regex")
    parser.add_argument("--native", type=Path, help="Also save native-size copies here")
    parser.add_argument("--steps", nargs="+", choices=DEFAULT_STEPS, help="Steps to run")
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))
    config = ManuscriptConfig(args.config_file, panels=set(mp.PANELS), edits=set(mp.EDITS))
    ManuscriptRunner(config).run(steps=args.steps, only=args.only, native_dir=args.native)


if __name__ == "__main__":
    main()
