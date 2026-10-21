"""Config for the manuscript workflow: source configs, expected values, styles and assets.

``run_manuscript_workflow.py`` reads one JSON config through :class:`ManuscriptConfig`.
Asset entries are templated over structures and distance measures, and
:func:`expand_assets` turns them into one :class:`AssetSpec` per published PDF.
Validation is strict: an unknown key, step, sheet, source or override is an error.
"""

from __future__ import annotations

import copy
import json
import os
import re
import string
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from cellpack_analysis.lib.figure_layout import TextStyle
from cellpack_analysis.lib.file_io import get_project_root, get_results_path

DEFAULT_CONFIG_ROOT = "cellpack_analysis/analysis/workflows/configs"
DEFAULT_STEPS = [
    "check_inputs",
    "figure_assets",
    "figure_numbers",
    "calibration_summary",
    "manifest",
]
DEFAULT_RENDER: dict[str, Any] = {"base_font_size": 8.0, "dpi": 300, "transparent": True}
DEFAULT_SHEETS = [
    "Fig2",
    "Fig3",
    "Fig4",
    "Fig5",
    "FigS6",
    "FigS7",
    "FigS8",
    "FigS9",
    "FigS10",
    "Methods",
]
SOURCE_KEYS = (
    "distance",
    "distance_supp",
    "distance_calibration",
    "occupancy",
    "rule_interpolation",
    "ladder",
    "biological_variation",
)
PACKING_CONFIG_KEYS = ("condition", "ladder")
EXPECTED_KEYS = ("occupancy_meta", "n_cells", "n_replicates", "ladder_replicates", "envelope")
ENVELOPE_KEYS = ("statistic", "envelope_type", "alpha")

_REQUIRED_KEYS = (
    "name",
    "condition",
    "structures",
    "distance_measures",
    "sources",
    "packing_configs",
    "expected",
    "row_labels",
    "styles",
    "assets",
)
_OPTIONAL_KEYS = (
    "output_path",
    "steps",
    "config_root",
    "template_vars",
    "render",
    "numbers",
)
_ASSET_KEYS = frozenset(
    {
        "path",
        "panel",
        "panel_args",
        "native_panel_args",
        "render",
        "style",
        "frames",
        "edits",
        "edit_params",
        "overrides",
        "rc",
        "savefig",
    }
)
_OVERRIDE_KEYS = frozenset({"frames", "edit_params", "edits"})
# kind -> (required keys, optional keys)
_FRAME_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "grid": (frozenset({"w", "h"}), frozenset({"ncols", "xpitch", "ypitch"})),
    "row": (frozenset({"xs", "ws", "h"}), frozenset()),
}
_AXIS_FIELDS = ("structure", "measure")


@dataclass(frozen=True)
class AssetSpec:
    """One published PDF: the panel that draws it and how it is fitted to its slot.

    Parameters
    ----------
    path
        Output path relative to the manuscript ``results_dir``.
    structure
        Structure the entry was expanded for, or ``None``.
    measure
        Distance measure the entry was expanded for, or ``None``.
    panel
        Name of the panel builder.
    panel_args
        Keyword arguments for the builder.
    native_panel_args
        Builder arguments for a native-size render (defaults to ``panel_args``).
    render
        ``"slot"`` (fitted to frames) or ``"native"`` (the builder's own size).
    style
        Text style name; slot assets only.
    frames
        Frame spec ``{"kind": "grid" | "row", ...}``; slot assets only.
    edits
        Slot edits, applied in order.
    edit_params
        Keyword arguments per edit.
    rc
        Matplotlib rc overrides; native assets only.
    savefig
        ``savefig`` keyword arguments; native assets only.
    """

    path: str
    structure: str | None
    measure: str | None
    panel: str
    panel_args: dict[str, Any]
    native_panel_args: dict[str, Any]
    render: Literal["slot", "native"]
    style: str | None
    frames: dict[str, Any] | None
    edits: list[str]
    edit_params: dict[str, dict[str, Any]]
    rc: dict[str, Any]
    savefig: dict[str, Any]


def check_under(path: Path, root: Path, what: str) -> Path:
    """Return ``path`` if it lies strictly under ``root``, compared lexically.

    Parameters
    ----------
    path
        Path to check.
    root
        Containing directory. Symlinks are not resolved, on either side.
    what
        Description used in the error message.

    Returns
    -------
    :
        ``path`` unchanged.

    Raises
    ------
    ValueError
        If ``path`` is ``root`` itself or lies outside it.
    """
    norm_root = Path(os.path.normpath(root))
    norm_path = Path(os.path.normpath(path))
    if norm_path == norm_root or not norm_path.is_relative_to(norm_root):
        raise ValueError(f"{what} {path} is not under the results root {root}")
    return path


def _relative_path(value: Any, what: str, suffix: str | None = None) -> str:
    """Check ``value`` is a clean relative POSIX path (no ``..``) and return it."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{what} must be a relative path without '..', got {value!r}")
    if suffix is not None and (path.suffix != suffix or value != path.as_posix()):
        raise ValueError(f"{what} must be a normalized relative '{suffix}' path, got {value!r}")
    return value


def _template_fields(template: str, what: str) -> set[str]:
    """Named placeholders in a ``str.format`` template; positional or formatted ones raise."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:
        raise ValueError(f"{what}: bad template {template!r}: {e}") from e
    names: set[str] = set()
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if not name.isidentifier() or spec or conversion:
            raise ValueError(f"{what}: only plain named placeholders allowed, got {template!r}")
        names.add(name)
    return names


def _check_keys(mapping: Any, allowed: Collection[str], what: str) -> dict[str, Any]:
    """Check ``mapping`` is a dict whose keys are all in ``allowed``."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{what} must be an object, got {type(mapping).__name__}")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{what}: unknown keys {unknown}; allowed {sorted(allowed)}")
    return mapping


def _string_list(value: Any, what: str) -> list[str]:
    """Check ``value`` is a non-empty list of unique non-empty strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{what} must be a non-empty list")
    if not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{what} must hold non-empty strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{what} has duplicates: {value!r}")
    return list(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _positive_int(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return value


def _check_frames(frames: Any, what: str) -> None:
    """Validate a slot frame spec against the ``grid_frames`` / ``row_frames`` arguments."""
    if not isinstance(frames, dict):
        raise ValueError(f"{what}: slot assets need a 'frames' object")
    kind = frames.get("kind")
    if kind not in _FRAME_KEYS:
        raise ValueError(f"{what}: frames.kind must be one of {sorted(_FRAME_KEYS)}, got {kind!r}")
    required, optional = _FRAME_KEYS[kind]
    keys = set(frames) - {"kind"}
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing or unknown:
        raise ValueError(f"{what}: {kind} frames missing {missing}, unknown {unknown}")
    for key in sorted(keys):
        value = frames[key]
        if key in ("xs", "ws"):
            ok = isinstance(value, list) and bool(value) and all(_is_number(v) for v in value)
        elif key == "ncols":
            ok = isinstance(value, int) and not isinstance(value, bool) and value >= 1
        else:
            ok = _is_number(value) and (key not in ("w", "h") or value > 0)
        if not ok:
            raise ValueError(f"{what}: bad frames.{key} {value!r}")
    if kind == "row" and len(frames["xs"]) != len(frames["ws"]):
        raise ValueError(f"{what}: frames.xs and frames.ws differ in length")


def _check_overrides(
    overrides: Any,
    by_structure: bool,
    by_measure: bool,
    structures: list[str],
    distance_measures: list[str],
    what: str,
) -> dict[str, dict[str, Any]]:
    """Validate override keys (``<s>``, ``<s>/<m>``, ``*/<m>``) and their values."""
    if not isinstance(overrides, dict):
        raise ValueError(f"{what}.overrides must be an object")
    for key, value in overrides.items():
        where = f"{what}.overrides[{key!r}]"
        structure, _, measure = key.partition("/")
        if measure:
            if measure not in distance_measures or not by_measure:
                raise ValueError(f"{where}: measure {measure!r} does not apply to this entry")
            ok = structure == "*" or (structure in structures and by_structure)
        else:
            ok = structure in structures and by_structure
        if not ok:
            raise ValueError(f"{where}: structure {structure!r} does not apply to this entry")
        _check_keys(value, _OVERRIDE_KEYS, where)
        if "frames" in value and not isinstance(value["frames"], dict):
            raise ValueError(f"{where}.frames must be an object")
        if "edit_params" in value:
            params = value["edit_params"]
            if not isinstance(params, dict) or not all(
                isinstance(p, dict) for p in params.values()
            ):
                raise ValueError(f"{where}.edit_params must map edit names to objects")
        if "edits" in value:
            _check_edit_list(value["edits"], f"{where}.edits")
    return overrides


def _check_edit_list(edits: Any, what: str) -> list[str]:
    if not isinstance(edits, list) or not all(isinstance(e, str) and e for e in edits):
        raise ValueError(f"{what} must be a list of edit names")
    if len(set(edits)) != len(edits):
        raise ValueError(f"{what} has duplicates: {edits!r}")
    return edits


def _apply_overrides(
    entry: dict[str, Any], overrides: dict[str, dict[str, Any]], keys: list[str], what: str
) -> None:
    """Merge the applicable overrides into ``entry``; two setting one value is an error."""
    owner: dict[tuple[str, ...], str] = {}

    def claim(leaf: tuple[str, ...], key: str) -> None:
        if leaf in owner:
            raise ValueError(f"{what}: overrides {owner[leaf]!r} and {key!r} both set {leaf}")
        owner[leaf] = key

    for key in keys:
        override = overrides[key]
        if "edits" in override:
            claim(("edits",), key)
            entry["edits"] = copy.deepcopy(override["edits"])
        if "frames" in override:
            frames = entry.get("frames")
            if not isinstance(frames, dict):
                raise ValueError(f"{what}: override {key!r} sets frames on an entry without frames")
            for name, value in override["frames"].items():
                claim(("frames", name), key)
                frames[name] = copy.deepcopy(value)
        for edit, params in override.get("edit_params", {}).items():
            merged = entry.setdefault("edit_params", {}).setdefault(edit, {})
            for name, value in params.items():
                claim(("edit_params", edit, name), key)
                merged[name] = copy.deepcopy(value)


def _make_spec(
    entry: dict[str, Any], path: str, structure: str | None, measure: str | None, what: str
) -> AssetSpec:
    """Validate one expanded entry and freeze it into an :class:`AssetSpec`."""
    render = entry.get("render", "slot")
    if render not in ("slot", "native"):
        raise ValueError(f"{what}: render must be 'slot' or 'native', got {render!r}")
    dicts = {}
    for key in ("panel_args", "rc", "savefig", "edit_params"):
        value = entry.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{what}.{key} must be an object")
        dicts[key] = value
    native_args = entry.get("native_panel_args", copy.deepcopy(dicts["panel_args"]))
    if not isinstance(native_args, dict):
        raise ValueError(f"{what}.native_panel_args must be an object")
    edits = _check_edit_list(entry.get("edits", []), f"{what}.edits")
    edit_params = dicts["edit_params"]
    if not all(isinstance(p, dict) for p in edit_params.values()):
        raise ValueError(f"{what}.edit_params must map edit names to objects")
    stray = sorted(set(edit_params) - set(edits))
    if stray:
        raise ValueError(f"{what}: edit_params for edits not in 'edits': {stray}")
    style = entry.get("style")
    frames = entry.get("frames")
    if render == "slot":
        if not isinstance(style, str) or not style:
            raise ValueError(f"{what}: slot assets need a 'style'")
        _check_frames(frames, what)
        if dicts["rc"] or dicts["savefig"]:
            raise ValueError(f"{what}: rc and savefig apply to native assets only")
    elif style is not None or frames is not None or edits or edit_params:
        raise ValueError(f"{what}: native assets take no style, frames, edits or edit_params")
    return AssetSpec(
        path=path,
        structure=structure,
        measure=measure,
        panel=entry["panel"],
        panel_args=dicts["panel_args"],
        native_panel_args=native_args,
        render=render,
        style=style,
        frames=frames,
        edits=edits,
        edit_params=edit_params,
        rc=dicts["rc"],
        savefig=dicts["savefig"],
    )


def _expand_entry(
    raw: Any,
    index: int,
    structures: list[str],
    distance_measures: list[str],
    template_vars: Mapping[str, Mapping[str, str]],
) -> list[AssetSpec]:
    """Expand one asset entry over its ``{structure}`` / ``{measure}`` placeholders."""
    what = f"assets[{index}]"
    _check_keys(raw, _ASSET_KEYS, what)
    for key in ("path", "panel"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"{what}: '{key}' is required and must be a string")
    template = raw["path"]
    what = f"{what} ({template})"
    fields = _template_fields(template, what)
    var_names = fields - set(_AXIS_FIELDS)
    by_structure = "structure" in fields
    by_measure = "measure" in fields
    if var_names and not by_structure:
        raise ValueError(f"{what}: template vars {sorted(var_names)} need a {{structure}} field")
    overrides = _check_overrides(
        raw.get("overrides", {}), by_structure, by_measure, structures, distance_measures, what
    )
    base = {k: v for k, v in raw.items() if k != "overrides"}
    s_axis: list[str | None] = list(structures) if by_structure else [None]
    m_axis: list[str | None] = list(distance_measures) if by_measure else [None]
    specs = []
    for structure in s_axis:
        for measure in m_axis:
            values: dict[str, str] = {}
            if structure is not None:
                svars = template_vars.get(structure, {})
                missing = sorted(var_names - set(svars))
                if missing:
                    raise ValueError(f"{what}: template_vars[{structure!r}] lacks {missing}")
                values.update({name: svars[name] for name in var_names})
                values["structure"] = structure
            if measure is not None:
                values["measure"] = measure
            path = _relative_path(template.format(**values), f"{what} path", suffix=".pdf")
            keys = [
                key
                for key in (structure, f"{structure}/{measure}", f"*/{measure}")
                if key in overrides
            ]
            entry = copy.deepcopy(base)
            _apply_overrides(entry, overrides, keys, f"{what} -> {path}")
            specs.append(_make_spec(entry, path, structure, measure, f"{what} -> {path}"))
    return specs


def check_asset_names(
    assets: list[AssetSpec],
    panels: Collection[str] | None = None,
    edits: Collection[str] | None = None,
    styles: Collection[str] | None = None,
) -> None:
    """Check every panel, edit and style name an asset uses exists in its registry.

    Parameters
    ----------
    assets
        Expanded assets.
    panels, edits, styles
        Registry names; ``None`` skips that registry.

    Raises
    ------
    ValueError
        Listing every unknown name with the first asset that uses it.
    """
    unknown: dict[str, str] = {}
    for spec in assets:
        used = [("panel", spec.panel, panels)]
        used += [("edit", name, edits) for name in spec.edits]
        if spec.style is not None:
            used.append(("style", spec.style, styles))
        for kind, name, registry in used:
            if registry is not None and name not in registry:
                unknown.setdefault(f"{kind} {name!r}", spec.path)
    if unknown:
        listed = "; ".join(f"{name} (in {path})" for name, path in unknown.items())
        raise ValueError(f"Unknown registry names: {listed}")


def expand_assets(
    raw_assets: list[dict[str, Any]],
    structures: list[str],
    distance_measures: list[str],
    template_vars: Mapping[str, Mapping[str, str]],
    panels: Collection[str] | None = None,
    edits: Collection[str] | None = None,
    styles: Collection[str] | None = None,
) -> list[AssetSpec]:
    """Expand templated asset entries into one :class:`AssetSpec` per output PDF.

    A path with ``{structure}`` expands over ``structures`` and one with ``{measure}``
    over ``distance_measures``; other placeholders come from ``template_vars[structure]``.
    Override keys ``"<structure>"``, ``"<structure>/<measure>"`` and ``"*/<measure>"``
    apply in that order. They merge shallowly into ``frames`` and into
    ``edit_params[name]``, or replace ``edits``. Two overrides setting the same value
    for one asset is an error, so the order never decides a value.

    Parameters
    ----------
    raw_assets
        Asset entries from the config.
    structures
        Structures (packing ids) to expand over.
    distance_measures
        Distance measures to expand over.
    template_vars
        Per-structure path variables.
    panels, edits, styles
        Registry names to check against; ``None`` skips that check.

    Returns
    -------
    :
        Assets in entry order, then structure order, then measure order.

    Raises
    ------
    ValueError
        On any malformed entry, inapplicable override, unknown name or duplicate path.
    """
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("assets must be a non-empty list")
    specs: list[AssetSpec] = []
    for index, raw in enumerate(raw_assets):
        specs.extend(_expand_entry(raw, index, structures, distance_measures, template_vars))
    counts = Counter(spec.path for spec in specs)
    dupes = sorted(path for path, n in counts.items() if n > 1)
    if dupes:
        raise ValueError(f"Duplicate asset paths: {dupes}")
    check_asset_names(specs, panels, edits, styles)
    return specs


class ManuscriptConfig:
    """Configuration for the manuscript workflow (assets, numbers, calibration).

    Parameters
    ----------
    config_file
        Path to the JSON config.
    panels, edits
        Panel and edit registry names to check the assets against; ``None`` defers the
        check to :meth:`check_registries`.
    """

    def __init__(
        self,
        config_file: str | Path,
        panels: Collection[str] | None = None,
        edits: Collection[str] | None = None,
    ) -> None:
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._setup_run()
        self._setup_paths()
        self._setup_sources()
        self._setup_expected()
        self._setup_styles()
        self._setup_assets(panels, edits)
        self._setup_numbers()

    def _load_config(self) -> dict[str, Any]:
        """Load the JSON config and check its top-level keys."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        with open(self.config_file) as f:
            config = json.load(f)
        _check_keys(config, _REQUIRED_KEYS + _OPTIONAL_KEYS, str(self.config_file))
        missing = [key for key in _REQUIRED_KEYS if key not in config]
        if missing:
            raise ValueError(f"{self.config_file}: missing required keys {missing}")
        return config

    def _setup_run(self) -> None:
        """Set name, condition, structures, measures, steps and render settings."""
        c = self.config
        if not isinstance(c["name"], str) or not c["name"]:
            raise ValueError("name must be a non-empty string")
        self.name: str = c["name"]
        if not isinstance(c["condition"], str) or not c["condition"]:
            raise ValueError("condition must be a non-empty string")
        self.condition: str = c["condition"]
        self.tag = self.condition.removeprefix("replicate_99_")
        self.structures = _string_list(c["structures"], "structures")
        self.distance_measures = _string_list(c["distance_measures"], "distance_measures")
        self.steps = _string_list(c.get("steps", list(DEFAULT_STEPS)), "steps")
        unknown = [step for step in self.steps if step not in DEFAULT_STEPS]
        if unknown:
            raise ValueError(f"Unknown steps {unknown}; known {DEFAULT_STEPS}")
        render = {**DEFAULT_RENDER, **_check_keys(c.get("render", {}), DEFAULT_RENDER, "render")}
        if not (
            _is_number(render["base_font_size"])
            and _is_number(render["dpi"])
            and isinstance(render["transparent"], bool)
        ):
            raise ValueError(f"render: bad values {render}")
        self.render: dict[str, Any] = render

    def _setup_paths(self) -> None:
        """Resolve the output directory under the results root (lexically contained)."""
        self.project_root = get_project_root()
        self.results_root = get_results_path()
        self.output_path = _relative_path(self.config.get("output_path", self.name), "output_path")
        self.results_dir = check_under(
            self.results_root / self.output_path, self.results_root, "output_path"
        )

    def _setup_sources(self) -> None:
        """Validate the source-config and packing-config path templates."""
        root = _relative_path(self.config.get("config_root", DEFAULT_CONFIG_ROOT), "config_root")
        self.config_root = self.project_root / root
        self.sources = self._templates("sources", SOURCE_KEYS)
        self.packing_configs = self._templates("packing_configs", PACKING_CONFIG_KEYS)

    def _templates(self, key: str, allowed: tuple[str, ...]) -> dict[str, str]:
        templates = _check_keys(self.config[key], allowed, key)
        for name, template in templates.items():
            what = f"{key}.{name}"
            _relative_path(template, what)
            extra = sorted(_template_fields(template, what) - {"structure"})
            if extra:
                raise ValueError(f"{what}: only {{structure}} may be templated, got {extra}")
        return dict(templates)

    def _setup_expected(self) -> None:
        """Validate the values every source and cache must carry."""
        expected = _check_keys(self.config["expected"], EXPECTED_KEYS, "expected")
        missing = [key for key in EXPECTED_KEYS if key not in expected]
        if missing:
            raise ValueError(f"expected: missing keys {missing}")
        meta = expected["occupancy_meta"]
        if not isinstance(meta, dict) or not meta:
            raise ValueError("expected.occupancy_meta must be a non-empty object")
        n_cells = expected["n_cells"]
        if not isinstance(n_cells, dict) or set(n_cells) != set(self.structures):
            raise ValueError(f"expected.n_cells keys must be the structures {self.structures}")
        for structure, n in n_cells.items():
            _positive_int(n, f"expected.n_cells[{structure!r}]")
        _positive_int(expected["n_replicates"], "expected.n_replicates")
        _positive_int(expected["ladder_replicates"], "expected.ladder_replicates")
        envelope = _check_keys(expected["envelope"], ENVELOPE_KEYS, "expected.envelope")
        if set(envelope) != set(ENVELOPE_KEYS):
            raise ValueError(f"expected.envelope needs exactly {list(ENVELOPE_KEYS)}")
        self.expected: dict[str, Any] = copy.deepcopy(expected)

    def _setup_styles(self) -> None:
        """Build ``styles["base"]`` and each named override as a ``TextStyle``."""
        styles = self.config["styles"]
        if not isinstance(styles, dict) or not isinstance(styles.get("base"), dict):
            raise ValueError("styles must be an object with a 'base' object")
        self.styles: dict[str, TextStyle] = {}
        for name, override in styles.items():
            if not isinstance(override, dict):
                raise ValueError(f"styles.{name} must be an object")
            try:
                self.styles[name] = TextStyle.from_dict(
                    styles["base"], None if name == "base" else override
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"styles.{name}: {e}") from e

    def _setup_assets(self, panels: Collection[str] | None, edits: Collection[str] | None) -> None:
        """Validate ``template_vars`` and expand the asset entries."""
        template_vars = _check_keys(
            self.config.get("template_vars", {}), self.structures, "template_vars"
        )
        for structure, svars in template_vars.items():
            what = f"template_vars[{structure!r}]"
            if not isinstance(svars, dict):
                raise ValueError(f"{what} must be an object")
            for name, value in svars.items():
                if not name.isidentifier() or name in _AXIS_FIELDS or not isinstance(value, str):
                    raise ValueError(f"{what}: bad variable {name!r}: {value!r}")
        self.template_vars: dict[str, dict[str, str]] = copy.deepcopy(template_vars)
        self.assets = expand_assets(
            self.config["assets"],
            self.structures,
            self.distance_measures,
            self.template_vars,
            panels=panels,
            edits=edits,
            styles=self.styles,
        )

    def _setup_numbers(self) -> None:
        """Validate the numbers-sheet settings and row labels."""
        row_labels = self.config["row_labels"]
        if not isinstance(row_labels, dict) or not row_labels:
            raise ValueError("row_labels must be a non-empty object")
        if not all(isinstance(v, str) for v in row_labels.values()):
            raise ValueError("row_labels values must be strings")
        self.row_labels: dict[str, str] = dict(row_labels)
        numbers = _check_keys(self.config.get("numbers", {}), ("sheets", "write_json"), "numbers")
        sheets = _string_list(numbers.get("sheets", list(DEFAULT_SHEETS)), "numbers.sheets")
        unknown = [sheet for sheet in sheets if sheet not in DEFAULT_SHEETS]
        if unknown:
            raise ValueError(f"Unknown numbers sheets {unknown}; known {DEFAULT_SHEETS}")
        write_json = numbers.get("write_json", True)
        if not isinstance(write_json, bool):
            raise ValueError("numbers.write_json must be true or false")
        self.numbers: dict[str, Any] = {"sheets": sheets, "write_json": write_json}

    def check_registries(self, panels: Collection[str], edits: Collection[str]) -> None:
        """Check every asset's panel and edit names against the runner's registries.

        Parameters
        ----------
        panels
            Panel builder names.
        edits
            Slot edit names.
        """
        check_asset_names(self.assets, panels, edits, self.styles)

    def select_assets(self, only: str | None = None) -> list[AssetSpec]:
        """Assets whose path matches the regular expression ``only`` (all when ``None``).

        Parameters
        ----------
        only
            Regular expression searched in each asset path.

        Returns
        -------
        :
            The matching assets, in config order.

        Raises
        ------
        ValueError
            If ``only`` matches no asset.
        """
        if only is None:
            return list(self.assets)
        pattern = re.compile(only)
        selected = [spec for spec in self.assets if pattern.search(spec.path)]
        if not selected:
            raise ValueError(f"--only {only!r} matches no asset")
        return selected
