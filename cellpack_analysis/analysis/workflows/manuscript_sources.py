"""Resolve and load the pipeline caches that the manuscript workflow draws from.

:class:`PaperSources` builds every source config with ``create_dirs=False``, checks it
against the manuscript config's ``expected`` values and loads caches with the strict
loaders. It never recomputes: a missing cache raises ``FileNotFoundError`` naming the
pipeline step and the ``run_paper.sh`` stage that write it, and a cache stamped with
other parameters raises ``ValueError``.
"""

from __future__ import annotations

import inspect
import json
import logging
import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from cellpack_analysis.analysis.workflows import run_rule_interpolation_workflow as riw
from cellpack_analysis.analysis.workflows.manuscript_config import ManuscriptConfig, check_under
from cellpack_analysis.analysis.workflows.run_analysis_workflow import (
    AnalysisConfig,
    _normalize_envelope_cache_meta,
    _pairwise_envelope_cache_meta,
)
from cellpack_analysis.lib import distance, label_tables, occupancy
from cellpack_analysis.lib import rule_interpolation as ri
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
from cellpack_analysis.packing.rule_repository import GRADIENTS

logger = logging.getLogger(__name__)

# run_paper.sh stage that runs each source config
SOURCE_STAGES = {
    "distance": "distance",
    "distance_supp": "distance-supp",
    "distance_calibration": "calibration",
    "occupancy": "occupancy",
    "rule_interpolation": "fit",
    "ladder": "ladder",
    "biological_variation": "biovar",
}
FIT_KINDS = ("rule_interpolation", "ladder")
FLOOR_KEYS = ("minimum_distance_um", "available_minimum_distance_um")
_MISSING = "<missing>"
_PDF_META_ARGS = (
    "method",
    "bin_width",
    "bandwidth",
    "distance_limits",
    "minimum_distance",
    "envelope_alpha",
    "n_grid",
)


class PaperSources:
    """Source configs and caches for one manuscript config, loaded lazily and memoized.

    Parameters
    ----------
    config
        The manuscript config.
    """

    def __init__(self, config: ManuscriptConfig) -> None:
        self.config = config
        self._cache: dict[tuple[Any, ...], tuple[Any, list[Path]]] = {}
        self._inputs: dict[Path, None] = {}
        self._recorders: list[dict[Path, None]] = []

    # ── bookkeeping ──────────────────────────────────────────────────────────────

    @property
    def inputs(self) -> list[Path]:
        """Every file read so far (configs and caches), in first-read order."""
        return list(self._inputs)

    @contextmanager
    def record_inputs(self) -> Iterator[list[Path]]:
        """Collect the files one block depends on, memoized loads included.

        Yields
        ------
        :
            A list that holds the block's inputs, in first-read order, once it exits.
        """
        used: dict[Path, None] = {}
        out: list[Path] = []
        self._recorders.append(used)
        try:
            yield out
        finally:
            self._recorders = [r for r in self._recorders if r is not used]
            out.extend(used)

    def _record(self, path: Path) -> None:
        self._inputs.setdefault(path, None)
        for used in self._recorders:
            used.setdefault(path, None)

    def _get(self, key: tuple[Any, ...], make: Callable[[], Any]) -> Any:
        """Memoize ``make()``; a cache hit still records the inputs it was built from."""
        if key not in self._cache:
            with self.record_inputs() as used:
                value = make()
            self._cache[key] = (value, used)
        value, used = self._cache[key]
        for path in used:
            self._record(path)
        return value

    def _require(
        self, path: Path, step: str, key: str, structure: str | None, stage: str | None = None
    ) -> Path:
        """Record ``path``, or raise naming the step and stage that write it."""
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. It is written by {step} for "
                f"{self.source_path(key, structure)} "
                f"(run_paper.sh stage '{stage or SOURCE_STAGES[key]}')."
            )
        self._record(path)
        return path

    # ── paths and configs ────────────────────────────────────────────────────────

    def _format(self, templates: dict[str, str], key: str, structure: str | None) -> str:
        if key not in templates:
            raise KeyError(f"{key!r} is not configured in {self.config.config_file}")
        template = templates[key]
        if structure is not None and structure not in self.config.structures:
            raise ValueError(f"Unknown structure {structure!r}; known {self.config.structures}")
        if "{structure}" in template and structure is None:
            raise ValueError(f"{key!r} is per structure ({template}); pass a structure")
        return template.format(structure=structure)

    def source_path(self, key: str, structure: str | None = None) -> Path:
        """Path of the source config ``key`` (for ``structure``), under ``config_root``.

        Parameters
        ----------
        key
            A key of the config's ``sources``.
        structure
            Structure (packing id) for per-structure sources.

        Returns
        -------
        :
            The config path.
        """
        return self.config.config_root / self._format(self.config.sources, key, structure)

    def packing_config_path(self, key: str, structure: str) -> Path:
        """Path of the tracked packing config ``key`` for ``structure``.

        Parameters
        ----------
        key
            A key of the config's ``packing_configs``.
        structure
            Structure (packing id).

        Returns
        -------
        :
            The packing-config path, relative to the project root.
        """
        return self.config.project_root / self._format(self.config.packing_configs, key, structure)

    def packing_config(self, key: str, structure: str) -> dict[str, Any]:
        """Load the tracked packing config ``key`` for ``structure``.

        Parameters
        ----------
        key
            A key of the config's ``packing_configs``.
        structure
            Structure (packing id).

        Returns
        -------
        :
            The parsed JSON.
        """

        def make() -> dict[str, Any]:
            path = self.packing_config_path(key, structure)
            if not path.exists():
                raise FileNotFoundError(f"Packing config {key!r} not found: {path}")
            self._record(path)
            data = json.loads(path.read_text())
            if data.get("packing_id") != structure:
                raise ValueError(f"{path}: packing_id {data.get('packing_id')!r} != {structure!r}")
            return data

        return self._get(("packing_config", key, structure), make)

    def _check_config(
        self, cfg: AnalysisConfig, what: str, structure: str | None, floors: bool = True
    ) -> AnalysisConfig:
        """Containment, structure and floor checks shared by every source config."""
        check_under(cfg.results_dir, self.config.results_root, f"{what} results_dir")
        if structure is not None and cfg.packing_id != structure:
            raise ValueError(f"{what}: packing_id {cfg.packing_id!r} != {structure!r}")
        if floors:
            meta = self.config.expected["occupancy_meta"]
            got = {k: getattr(cfg, k) for k in FLOOR_KEYS if k in meta}
            want = {k: meta[k] for k in got}
            if got != want:
                raise ValueError(f"{what}: floors {got} != expected {want}")
        return cfg

    def _source_file(self, key: str, structure: str | None) -> Path:
        path = self.source_path(key, structure)
        if not path.exists():
            raise FileNotFoundError(f"Source config {key!r} not found: {path}")
        self._record(path)
        return path

    def _analysis_config(self, key: str, structure: str | None) -> AnalysisConfig:
        def make() -> AnalysisConfig:
            path = self._source_file(key, structure)
            cfg = AnalysisConfig(str(path), create_dirs=False)
            floors = key != "biological_variation"
            return self._check_config(cfg, f"{key} config {path}", structure, floors)

        return self._get(("config", key, structure), make)

    def distance_config(self, structure: str, supp: bool = False) -> AnalysisConfig:
        """Fig 3's distance config, or with ``supp`` S4's (nearest and pairwise).

        Parameters
        ----------
        structure
            Structure (packing id).
        supp
            Use the ``distance_supp`` source.

        Returns
        -------
        :
            The config, built with ``create_dirs=False`` and checked.
        """
        return self._analysis_config("distance_supp" if supp else "distance", structure)

    def occupancy_config(self, structure: str) -> AnalysisConfig:
        """Fig 4d's occupancy config (the per-rule curves).

        Parameters
        ----------
        structure
            Structure (packing id).

        Returns
        -------
        :
            The config, built with ``create_dirs=False`` and checked.
        """
        return self._analysis_config("occupancy", structure)

    def biovar_config(self) -> AnalysisConfig:
        """Return the biological-variation config behind Fig 2f and S3.

        Returns
        -------
        :
            The config, built with ``create_dirs=False``; its floors are not checked.
        """
        return self._analysis_config("biological_variation", None)

    def _fit_kind(self, kind: str) -> None:
        if kind not in FIT_KINDS:
            raise ValueError(f"kind must be one of {FIT_KINDS}, got {kind!r}")

    def fit_config(self, kind: str, structure: str) -> AnalysisConfig:
        """Return the analysis config ``--phase fit`` wrote for a rule-interpolation source.

        Parameters
        ----------
        kind
            ``"rule_interpolation"`` (Fig 4d, 5, S8/S9) or ``"ladder"`` (S10).
        structure
            Structure (packing id).

        Returns
        -------
        :
            The fit phase's config, built with ``create_dirs=False`` and checked.
        """
        self._fit_kind(kind)

        def make() -> AnalysisConfig:
            e2e = self._source_file(kind, structure)
            try:
                cfg = riw.load_fit_phase_config(e2e, create_dirs=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"{e} (run_paper.sh stage '{SOURCE_STAGES[kind]}')") from e
            self._record(cfg.config_file)
            if cfg.results_dir != cfg.config_file.parent:
                raise ValueError(f"{cfg.config_file}: results_dir {cfg.results_dir} is elsewhere")
            return self._check_config(cfg, f"{kind} fit config {cfg.config_file}", structure)

        return self._get(("fit_config", kind, structure), make)

    def validation_config(self, structure: str) -> AnalysisConfig:
        """Return the validation analysis config ``--phase fit`` wrote (Fig 5, S8, S9).

        Parameters
        ----------
        structure
            Structure (packing id).

        Returns
        -------
        :
            The validate phase's config, built with ``create_dirs=False`` and checked.
        """

        def make() -> AnalysisConfig:
            e2e = self._source_file("rule_interpolation", structure)
            try:
                cfg = riw.load_validation_config(e2e, create_dirs=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"{e} (run_paper.sh stage 'fit')") from e
            self._record(cfg.config_file)
            return self._check_config(cfg, f"validation config {cfg.config_file}", structure)

        return self._get(("validation_config", structure), make)

    # ── distance caches ──────────────────────────────────────────────────────────

    def distance_pdf_meta(self, structure: str, supp: bool = False) -> dict[str, Any]:
        """Build the PDF-cache sidecar ``plot_distance_distributions`` writes for this config.

        Parameters
        ----------
        structure
            Structure (packing id).
        supp
            Use the ``distance_supp`` source.

        Returns
        -------
        :
            The expected sidecar, built with the pipeline's kwargs and builder.
        """
        cfg = self.distance_config(structure, supp)
        defaults = {
            name: p.default
            for name, p in inspect.signature(distance.compute_distance_pdfs).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        # same kwargs as _plot_distance_distributions
        kwargs = {
            **defaults,
            "method": cfg.distribution_method,
            "distance_limits": DISTANCE_LIMITS,
            "minimum_distance": None,
            **cfg.distance_pdf_params,
        }
        return distance._build_distance_pdf_cache_meta(
            **{name: kwargs[name] for name in _PDF_META_ARGS},
            distance_measures=cfg.distance_measures,
            packing_modes=cfg.packing_modes,
        )

    def distance_pdfs(self, structure: str, supp: bool = False) -> dict[str, Any]:
        """Fig 3a / S4a PDFs from the ``plot_distance_distributions`` cache.

        Parameters
        ----------
        structure
            Structure (packing id).
        supp
            Use the ``distance_supp`` source.

        Returns
        -------
        :
            ``{measure: {mode: {xvals, individual_curves, mean_pdf, envelope_lo, ...}}}``.
        """
        key = "distance_supp" if supp else "distance"
        cfg = self.distance_config(structure, supp)

        def make() -> dict[str, Any]:
            method, suffix = cfg.distribution_method, cfg.suffix or ""
            path = cfg.results_dir / f"distance_pdfs_{method}{suffix}.dat"
            self._require(path, "plot_distance_distributions", key, structure)
            self._require(
                path.with_suffix(".meta.json"), "plot_distance_distributions", key, structure
            )
            return distance.load_distance_pdfs(
                cfg.results_dir, method, suffix, self.distance_pdf_meta(structure, supp)
            )

        return self._get(("distance_pdfs", key, structure), make)

    def _ks_frame(self, structure: str, supp: bool, kind: str) -> pd.DataFrame:
        key = "distance_supp" if supp else "distance"
        cfg = self.distance_config(structure, supp)

        def make() -> pd.DataFrame:
            name = f"ks_{kind}_vs_{cfg.baseline_mode}{cfg.suffix or ''}.parquet"
            path = self._require(cfg.results_dir / name, "run_ks_analysis", key, structure)
            df = pd.read_parquet(path)
            refs = sorted(set(df["baseline_mode"])) if "baseline_mode" in df else []
            if refs != [cfg.baseline_mode]:
                raise ValueError(f"{path}: baseline_mode {refs} != [{cfg.baseline_mode!r}]")
            return df

        return self._get((f"ks_{kind}", key, structure), make)

    def ks_tests(self, structure: str, supp: bool = False) -> pd.DataFrame:
        """Per-cell KS tests of each rule against the observed data (``run_ks_analysis``).

        Parameters
        ----------
        structure
            Structure (packing id).
        supp
            Use the ``distance_supp`` source.

        Returns
        -------
        :
            The persisted ``ks_test_vs_<baseline>`` frame.
        """
        return self._ks_frame(structure, supp, "test")

    def ks_bootstrap(self, structure: str, supp: bool = False) -> pd.DataFrame:
        """Fig 3c / S4c: the seeded cell bootstrap of the KS tests (``run_ks_analysis``).

        Parameters
        ----------
        structure
            Structure (packing id).
        supp
            Use the ``distance_supp`` source.

        Returns
        -------
        :
            The persisted ``ks_bootstrap_vs_<baseline>`` frame.
        """
        return self._ks_frame(structure, supp, "bootstrap")

    def distance_emd(self, structure: str, supp: bool = False) -> pd.DataFrame:
        """Fig 3b / S4b / S5: the distance EMD parquet, filtered as ``run_emd_analysis`` does.

        Parameters
        ----------
        structure
            Structure (packing id).
        supp
            Use the ``distance_supp`` source.

        Returns
        -------
        :
            Pairwise EMD rows (same-cell only when the config sets ``emd_same_cell``).
        """
        key = "distance_supp" if supp else "distance"
        cfg = self.distance_config(structure, supp)

        def make() -> pd.DataFrame:
            name = f"distance_pairwise_emd{cfg.suffix or ''}.parquet"
            path = self._require(cfg.results_dir / name, "run_emd_analysis", key, structure)
            df = pd.read_parquet(path)
            return distance.same_cell_emd_pairs(df) if cfg.emd_same_cell else df

        return self._get(("distance_emd", key, structure), make)

    def per_cell_envelope(self, structure: str, suffix: str = "") -> dict[str, Any]:
        """Fig 3d / S6 / S7: the per-cell envelope test, or its calibration run.

        Parameters
        ----------
        structure
            Structure (packing id).
        suffix
            ``""`` reads the ``distance`` source; any other value reads
            ``distance_calibration``, whose suffix must equal it.

        Returns
        -------
        :
            The raw ``per_cell_envelope_test`` return.
        """
        key = "distance_calibration" if suffix else "distance"
        cfg = self._analysis_config(key, structure)
        if (cfg.suffix or "") != suffix:
            raise ValueError(f"{cfg.config_file}: suffix {cfg.suffix!r} != {suffix!r}")

        def make() -> dict[str, Any]:
            name = f"per_cell_envelope_results{suffix}.dat"
            path = self._require(
                cfg.results_dir / name, "run_per_cell_envelope_test", key, structure
            )
            with open(path, "rb") as f:
                result = pickle.load(f)
            if not isinstance(result, dict) or "summary" not in result:
                raise ValueError(f"{path} is not a per_cell_envelope_test result")
            want = dict(self.config.expected["envelope"])
            want["calibration_holdout_index"] = cfg.envelope_test_params.get(
                "calibration_holdout_index"
            )
            got = {k: result.get(k, _MISSING) for k in want}
            if got != want:
                raise ValueError(f"{path}: test parameters {got} != expected {want}")
            return result

        return self._get(("per_cell_envelope", structure, suffix), make)

    def pairwise_envelope(self) -> dict[str, Any]:
        """Fig 2f / S3: the pooled bio-variation envelope test (``run_pairwise_envelope_test``).

        The ``.meta.json`` sidecar must exist and match the bio-variation config.

        Returns
        -------
        :
            The raw ``pairwise_envelope_test`` return.
        """
        key = "biological_variation"
        cfg = self.biovar_config()

        def make() -> dict[str, Any]:
            path = cfg.results_dir / f"pairwise_envelope_results{cfg.suffix or ''}.dat"
            meta_path = path.with_suffix(".meta.json")
            self._require(path, "run_pairwise_envelope_test", key, None)
            self._require(meta_path, "run_pairwise_envelope_test", key, None)
            try:
                cached = _normalize_envelope_cache_meta(json.loads(meta_path.read_text()))
            except (OSError, ValueError) as e:
                raise ValueError(f"Could not read {meta_path}: {e}") from e
            expected = _pairwise_envelope_cache_meta(cfg)
            differing = {
                k: (cached.get(k, _MISSING), expected.get(k, _MISSING))
                for k in sorted(cached.keys() | expected.keys())
                if cached.get(k, _MISSING) != expected.get(k, _MISSING)
            }
            if differing:
                raise ValueError(
                    f"{path} was built with other parameters {differing} (cached, now)"
                )
            with open(path, "rb") as f:
                result = pickle.load(f)
            if not isinstance(result, dict) or not {"joint", "packing_modes"} <= set(result):
                raise ValueError(f"{path} is not a pairwise_envelope_test result")
            return result

        return self._get(("pairwise_envelope",), make)

    # ── occupancy and rule interpolation ─────────────────────────────────────────

    def _occupancy_meta(self, distance_measure: str) -> dict[str, Any]:
        return {**self.config.expected["occupancy_meta"], "distance_measure": distance_measure}

    def _load_occupancy(
        self,
        cfg: AnalysisConfig,
        distance_measure: str,
        channel_map: dict[str, str],
        step: str,
        key: str,
        structure: str,
        stage: str | None = None,
    ) -> dict[str, Any]:
        if distance_measure not in cfg.occupancy_distance_measures:
            raise ValueError(f"{cfg.config_file}: no occupancy for {distance_measure!r}")
        suffix = cfg.suffix or ""
        path = cfg.results_dir / f"{distance_measure}_occupancy{suffix}.dat"
        self._require(path, step, key, structure, stage)
        self._require(path.with_suffix(".meta.json"), step, key, structure, stage)
        return occupancy.load_occupancy_cache(
            cfg.results_dir,
            distance_measure,
            suffix,
            expected_meta=self._occupancy_meta(distance_measure),
            channel_map=channel_map,
        )

    def occupancy(self, structure: str, distance_measure: str) -> dict[str, Any]:
        """Fig 4d per-rule occupancy for one measure (``run_occupancy_analysis``).

        Parameters
        ----------
        structure
            Structure (packing id).
        distance_measure
            Occupancy distance measure.

        Returns
        -------
        :
            ``{mode: {"individual": ..., "combined": ...}}``.
        """
        cfg = self.occupancy_config(structure)
        return self._get(
            ("occupancy", structure, distance_measure),
            lambda: self._load_occupancy(
                cfg,
                distance_measure,
                cfg.channel_map,
                "run_occupancy_analysis",
                "occupancy",
                structure,
            ),
        )

    def occupancy_emd(self, structure: str) -> pd.DataFrame:
        """Pairwise occupancy EMD rows from the ``run_occupancy_emd_analysis`` cache.

        Parameters
        ----------
        structure
            Structure (packing id).

        Returns
        -------
        :
            The cached frame as written, with every cell paired with every mode (no same-cell
            filtering).
        """
        cfg = self.occupancy_config(structure)

        def make() -> pd.DataFrame:
            name = f"occupancy_emd{cfg.suffix or ''}.parquet"
            path = self._require(
                cfg.results_dir / name, "run_occupancy_emd_analysis", "occupancy", structure
            )
            return pd.read_parquet(path)

        return self._get(("occupancy_emd", structure), make)

    def fit_occupancy(self, kind: str, structure: str) -> dict[str, dict[str, Any]]:
        """Load the occupancy curves the fit phase used, for every occupancy measure.

        Parameters
        ----------
        kind
            ``"rule_interpolation"`` or ``"ladder"``.
        structure
            Structure (packing id).

        Returns
        -------
        :
            ``{measure: {mode: ...}}`` keyed by the fit's ``cv_channel_map``.
        """
        cfg = self.fit_config(kind, structure)
        return self._get(
            ("fit_occupancy", kind, structure),
            lambda: {
                dm: self._load_occupancy(
                    cfg, dm, cfg.cv_channel_map, "--phase fit", kind, structure
                )
                for dm in cfg.occupancy_distance_measures
            },
        )

    def cv_result(self, kind: str, structure: str) -> ri.CVResult:
        """Load the fit phase's cross-validation result (``run_rule_interpolation_cv``).

        Parameters
        ----------
        kind
            ``"rule_interpolation"`` or ``"ladder"``.
        structure
            Structure (packing id).

        Returns
        -------
        :
            The pickled :class:`~cellpack_analysis.lib.rule_interpolation.CVResult`.
        """
        cfg = self.fit_config(kind, structure)

        def make() -> ri.CVResult:
            path = cfg.results_dir / f"rule_interpolation_cv{cfg.suffix or ''}.pkl"
            self._require(path, "--phase fit (run_rule_interpolation_cv)", kind, structure)
            with open(path, "rb") as f:
                result = pickle.load(f)
            if not isinstance(result, ri.CVResult):
                raise ValueError(f"{path} holds {type(result).__name__}, not CVResult")
            return result

        return self._get(("cv_result", kind, structure), make)

    def cv_fit(
        self, kind: str, structure: str, occupancy: dict[str, Any] | None = None
    ) -> ri.FitResult:
        """Build the fit from the CV mean coefficients, as the fit (or validate) phase does.

        Parameters
        ----------
        kind
            ``"rule_interpolation"`` or ``"ladder"``.
        structure
            Structure (packing id).
        occupancy
            Occupancy to apply the coefficients to; defaults to :meth:`fit_occupancy`.

        Returns
        -------
        :
            The :class:`~cellpack_analysis.lib.rule_interpolation.FitResult`.
        """
        cfg = self.fit_config(kind, structure)

        def make(occ: dict[str, Any]) -> ri.FitResult:
            return ri.fit_result_from_cv(
                cv_result=self.cv_result(kind, structure),
                occupancy_dict=occ,
                channel_map=cfg.cv_channel_map,
                baseline_mode=cfg.baseline_mode,
                distance_measures=cfg.occupancy_distance_measures,
            )

        if occupancy is not None:
            return make(occupancy)
        return self._get(
            ("cv_fit", kind, structure), lambda: make(self.fit_occupancy(kind, structure))
        )

    def validation_occupancy(self, structure: str) -> dict[str, Any]:
        """Fig 5b occupancy: the fit's component curves plus the validation mixed rule.

        Parameters
        ----------
        structure
            Structure (packing id).

        Returns
        -------
        :
            ``{measure: {mode: ...}}`` as ``--phase validate`` assembles it.
        """
        v = self.validation_config(structure)
        fit = self.fit_config("rule_interpolation", structure)

        def make() -> dict[str, Any]:
            self.fit_occupancy("rule_interpolation", structure)  # checks the fit sidecars
            val_occ = {
                dm: self._load_occupancy(
                    v,
                    dm,
                    v.channel_map,
                    "--phase validate (run_occupancy_analysis)",
                    "rule_interpolation",
                    structure,
                    stage="validate",
                )
                for dm in v.occupancy_distance_measures
            }
            return riw._merge_fit_and_validation_occupancy(
                results_dir=fit.results_dir,
                fit_suffix=fit.suffix or "",
                distance_measures=v.occupancy_distance_measures,
                validation_occupancy_dict=val_occ,
                baseline_mode=v.baseline_mode,
            )

        return self._get(("validation_occupancy", structure), make)

    def validation(self, structure: str) -> ri.ValidationResult:
        """Fig 5c / S8 / S9: the persisted validation result, EMD frames restored.

        Parameters
        ----------
        structure
            Structure (packing id).

        Returns
        -------
        :
            The :class:`~cellpack_analysis.lib.rule_interpolation.ValidationResult`.
        """
        v = self.validation_config(structure)
        fit_dir = self.fit_config("rule_interpolation", structure).results_dir

        def make() -> ri.ValidationResult:
            step = "--phase validate"
            for name in (
                "validation_result_mixed_rule_validation.dat",
                "occupancy_emd_mixed_rule_validation.parquet",
            ):
                self._require(fit_dir / name, step, "rule_interpolation", structure, "validate")
            distance_emd = fit_dir / "distance_pairwise_emd_mixed_rule_validation.parquet"
            if distance_emd.exists():
                self._record(distance_emd)
            return ri.load_validation_result(fit_dir, same_cell_emd=v.emd_same_cell)

        return self._get(("validation", structure), make)

    # ── logs ─────────────────────────────────────────────────────────────────────

    def read_log(self, results_dir: Path, name: str) -> list[str]:
        """Lines of a pipeline log, carriage returns split as line breaks.

        Parameters
        ----------
        results_dir
            A source config's ``results_dir``.
        name
            Log file name under ``results_dir / "logs"``.

        Returns
        -------
        :
            The raw lines (log prefixes kept).
        """
        check_under(results_dir, self.config.results_root, "log results_dir")
        path = results_dir / "logs" / name
        if not path.exists():
            raise FileNotFoundError(f"Log not found: {path}; rerun the step that writes it")
        self._record(path)
        return path.read_text().replace("\r", "\n").splitlines()


def check_canonical_apical_lambda(packing_config: Path) -> float:
    """Check the apical rule of a tracked packing config uses the canonical λ.

    Parameters
    ----------
    packing_config
        Packing config whose ``recipe_data.apical_gradient.gradients`` names the rule.

    Returns
    -------
    :
        The apical decay length (0.7 for the dataset of record).

    Raises
    ------
    ValueError
        If the apical entry is missing, ambiguous or off the canonical λ in
        ``label_tables.CANONICAL_DECAY_LENGTHS``.
    """
    data = json.loads(Path(packing_config).read_text())
    entry = data.get("recipe_data", {}).get("apical_gradient", {}).get("gradients")
    gradients: list[Any]
    if isinstance(entry, list):
        unknown = [name for name in entry if name not in GRADIENTS]
        if unknown:
            raise ValueError(f"{packing_config}: unknown gradients {unknown}")
        gradients = [GRADIENTS[name] for name in entry]
    elif isinstance(entry, dict):
        gradients = list(entry.values())
    else:
        raise ValueError(f"{packing_config}: no recipe_data.apical_gradient.gradients")
    if len(gradients) != 1:
        raise ValueError(f"{packing_config}: expected one apical gradient, got {len(gradients)}")
    lam = float(gradients[0]["weight_mode_settings"]["decay_length"])
    canonical = label_tables.CANONICAL_DECAY_LENGTHS["apical_gradient"]
    if lam != canonical:
        raise ValueError(f"{packing_config}: apical λ {lam:g} != canonical {canonical:g}")
    return lam
