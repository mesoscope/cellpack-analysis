#!/usr/bin/env python3
"""
Workflow runner for cellpack analysis.

This script runs configurable analysis workflows based on configuration files.
The workflow is controlled by the ``analysis_steps`` field in the config, which
specifies which steps to execute in order.

Available analysis steps
------------------------
Data loading / preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``load_common_data``       — Load position data and mesh information.
- ``calculate_distances``    — Calculate distance measures and (optionally) normalize.

Distance distribution analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``plot_distance_distributions`` — Plot per-cell distance distributions.
  Uses discrete histograms when ``distribution_method`` is ``"discrete"``
  or KDE curves when it is ``"kde"`` (default). 
- ``run_emd_analysis``       — Earth Mover's Distance analysis across packing modes,
  including pairwise EMD matrix plots.
- ``run_ks_analysis``        — Pairwise Kolmogorov-Smirnov tests with bootstrap CIs
  for every ordered pair of packing modes.
- ``run_pairwise_envelope_test`` — Pairwise Monte Carlo rank-envelope tests comparing
  every ordered pair of packing modes.
- ``run_per_cell_envelope_test`` — Per-cell Monte Carlo rank-envelope (CSR) test comparing
  the baseline (experimental) distribution against each null model's own per-cell replicate
  envelope, aggregated across cells with BH correction. This is the per-cell counterpart to
  ``run_pairwise_envelope_test`` and is the correct test for replicate-per-cell packing designs.

Occupancy analysis
~~~~~~~~~~~~~~~~~~
- ``run_occupancy_analysis`` — Occupancy ratio analysis (discrete histogram or KDE,
  controlled by ``distribution_method``).
- ``run_occupancy_emd_analysis``  — EMD analysis on occupancy ratio curves, including
  pairwise EMD matrix plots.
- ``run_occupancy_pairwise_envelope_test`` — Pairwise Monte Carlo rank-envelope test
  on per-cell occupancy ratio curves.
- ``run_occupancy_ks_analysis``   — Pairwise Kolmogorov-Smirnov tests with bootstrap CIs
  on per-cell occupancy ratio distributions.
- ``run_rule_interpolation_cv`` — K-fold cross-validation for NNLS rule-mixing
  coefficients (works with both KDE and discrete occupancy).
- ``build_virtual_mixture_mode`` — Inject a virtual-mixture mode into the shared distance
  data: per cell and seed, draw ``round(c_i * N)`` puncta from rule *i*'s existing
  single-rule packing (coefficients from a rule-interpolation CV pickle, configured via
  ``virtual_mixture_params``). Must run after ``calculate_distances`` and before
  ``run_occupancy_analysis``.
- ``compare_virtual_mixture_occupancy`` — Compare the virtual-mixture mode's occupancy
  against the expected NNLS mixture ``sum(c_i * occ_i)`` computed per cell from the same
  run's single-rule occupancy curves (mixing-handoff diagnostic D2: the achievable
  packed-vs-expected ceiling). Must run after ``run_occupancy_analysis``.

Config key: ``distribution_method``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``"discrete"`` (default) — histogram-based plots and occupancy.
- ``"kde"`` — kernel-density-estimate plots and occupancy (legacy behaviour).

Example workflows
-----------------
Biological variation::

    ["load_common_data", "calculate_distances",
     "plot_distance_distributions", "run_emd_analysis",
     "run_pairwise_envelope_test"]

Distance analysis::

    ["load_common_data", "calculate_distances",
     "plot_distance_distributions", "run_emd_analysis",
     "run_ks_analysis", "run_pairwise_envelope_test"]

Occupancy analysis::

    ["load_common_data", "calculate_distances",
     "run_occupancy_analysis", "run_occupancy_emd_analysis",
     "run_occupancy_pairwise_envelope_test"]

Usage::

    python run_analysis_workflow.py --config_file path/to/config.json
    python run_analysis_workflow.py --config_file configs/distance_analysis_config.json \
        --log_level DEBUG
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellpack_analysis import setup_logging
from cellpack_analysis.analysis.workflows.configs import defaults
from cellpack_analysis.lib import (
    distance,
    mixture_diagnostics,
    occupancy,
    rule_interpolation,
    visualization,
)
from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_project_root, get_results_path
from cellpack_analysis.lib.io import format_time
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats import pairwise_envelope_test, per_cell_envelope_test

mpl.use("Agg")  # Use non-interactive backend for plotting
plt.ioff()  # Disable interactive mode to prevent figures from displaying during analysis

logger = logging.getLogger(__name__)


class AnalysisConfig:
    """Configuration class for analysis workflows."""

    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._setup_packing_params()
        self._setup_analysis_params()
        self._setup_ks_params()
        self._setup_occupancy_params()
        self._setup_envelope_params()
        self._setup_distance_pdf_params()
        self._setup_distance_plot_params()
        self._setup_emd_plot_params()
        self._setup_recalculate()
        self._setup_rule_interpolation_cv_params()
        self._setup_paths()

    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file) as f:
            return json.load(f)

    def _setup_packing_params(self) -> None:
        """Set structure, packing, and data-loading parameters."""
        self.structure_id = self.config.get("structure_id", defaults.STRUCTURE_ID)
        self.packing_id = self.config.get("packing_id", defaults.PACKING_ID)
        self.structure_name = self.config.get("structure_name", defaults.STRUCTURE_NAME)
        self.channel_map = self.config.get("channel_map", {self.structure_id: self.structure_id})
        self.packing_modes = list(self.channel_map.keys())
        self.packing_output_folder = self.config.get(
            "packing_output_folder", defaults.PACKING_OUTPUT_FOLDER
        )
        self.baseline_mode = self.config.get("baseline_mode", self.structure_id)
        self.ingredient_key = self.config.get(
            "ingredient_key", f"membrane_interior_{self.structure_name}"
        )
        self.num_workers = self.config.get("num_workers", defaults.NUM_WORKERS)
        self.distance_backend = self.config.get("distance_backend", defaults.DISTANCE_BACKEND)
        self.occupancy_backend = self.config.get("occupancy_backend", defaults.OCCUPANCY_BACKEND)

    def _setup_analysis_params(self) -> None:
        """Set top-level analysis parameters."""
        self.name = self.config["name"]
        self.suffix = self.config.get("suffix", "")
        self.normalization = self.config.get("normalization")
        if self.normalization:
            self.suffix += f"_norm_{self.normalization}"
        self.distance_measures = self.config.get("distance_measures", defaults.DISTANCE_MEASURES)
        self.emd_pool_seeds = self.config.get("emd_pool_seeds", defaults.EMD_POOL_SEEDS)
        self.emd_same_cell = self.config.get("emd_same_cell", defaults.EMD_SAME_CELL)
        self.virtual_mixture_params = self.config.get("virtual_mixture_params", {})
        self.save_format = self.config.get("save_format", defaults.SAVE_FORMAT)
        self.distribution_method = self.config.get(
            "distribution_method", defaults.DISTRIBUTION_METHOD
        )
        if self.distribution_method not in ("discrete", "kde"):
            raise ValueError(
                f"Invalid distribution_method '{self.distribution_method}'. "
                "Must be 'discrete' or 'kde'."
            )
        # Merge user-provided bin_width_map over defaults
        self.bin_width_map: dict[str, float] = defaults.BIN_WIDTH_MAP.copy()
        self.bin_width_map.update(self.config.get("bin_width_map", {}))
        # analysis_steps is the primary driver of execution
        self.analysis_steps: list[str] = self.config["analysis_steps"]
        for legacy_key in ("filter_minimum_distance", "available_minimum_distance"):
            if legacy_key in self.config:
                raise ValueError(
                    f"'{legacy_key}' is replaced by 'minimum_distance_um' (puncta) and "
                    "'available_minimum_distance_um' (available space), both in microns"
                )
        self.minimum_distance_um: float | None = self.config.get(
            "minimum_distance_um", defaults.MINIMUM_DISTANCE_UM
        )
        self.available_minimum_distance_um: float | None = self.config.get(
            "available_minimum_distance_um", defaults.AVAILABLE_MINIMUM_DISTANCE_UM
        )
        # Same floors in raw (pixel) distance units.
        self.minimum_distance: float | None = None
        if self.minimum_distance_um is not None:
            self.minimum_distance = self.minimum_distance_um / PIXEL_SIZE_IN_UM
        self.available_minimum_distance: float | None = None
        if self.available_minimum_distance_um is not None:
            self.available_minimum_distance = self.available_minimum_distance_um / PIXEL_SIZE_IN_UM

    def _setup_ks_params(self) -> None:
        """Set Kolmogorov-Smirnov test parameters."""
        self.ks_significance_level = self.config.get(
            "ks_significance_level", defaults.KS_SIGNIFICANCE_LEVEL
        )
        self.n_bootstrap = self.config.get("n_bootstrap", defaults.N_BOOTSTRAP)
        self.bandwidth = self.config.get("bandwidth", defaults.BANDWIDTH)

    def _setup_occupancy_params(self) -> None:
        """Set occupancy analysis parameters."""
        self.occupancy_distance_measures = self.config.get(
            "occupancy_distance_measures", defaults.OCCUPANCY_DISTANCE_MEASURES
        )
        self.occupancy_params: dict = defaults.OCCUPANCY_PARAMS.copy()
        self.occupancy_params.update(self.config.get("occupancy_params", {}))
        self.discrete_occupancy_params: dict = defaults.DISCRETE_OCCUPANCY_PARAMS.copy()
        self.discrete_occupancy_params.update(self.config.get("discrete_occupancy_params", {}))

    def _setup_envelope_params(self) -> None:
        """Set Monte Carlo envelope test parameters."""
        self.envelope_test_params: dict = defaults.ENVELOPE_TEST_PARAMS.copy()
        self.envelope_test_params.update(self.config.get("envelope_test_params", {}))
        self.envelope_plot_params: dict = defaults.ENVELOPE_PLOT_PARAMS.copy()
        self.envelope_plot_params.update(self.config.get("envelope_plot_params", {}))

    def _setup_distance_pdf_params(self) -> None:
        """Set distance PDF computation parameters."""
        self.distance_pdf_params: dict = defaults.DISTANCE_PDF_PARAMS.copy()
        self.distance_pdf_params.update(self.config.get("distance_pdf_params", {}))

    def _setup_distance_plot_params(self) -> None:
        """Set distance distribution plot parameters."""
        self.distance_plot_params: dict = defaults.DISTANCE_PLOT_PARAMS.copy()
        self.distance_plot_params.update(self.config.get("distance_plot_params", {}))

    def _setup_emd_plot_params(self) -> None:
        """Set EMD matrix plot parameters."""
        self.emd_plot_params: dict = defaults.EMD_PLOT_PARAMS.copy()
        self.emd_plot_params.update(self.config.get("emd_plot_params", {}))

    def _setup_rule_interpolation_cv_params(self) -> None:
        """Set rule interpolation cross-validation parameters."""
        self.rule_interpolation_cv_params: dict = defaults.RULE_INTERPOLATION_CV_PARAMS.copy()
        self.rule_interpolation_cv_params.update(
            self.config.get("rule_interpolation_cv_params", {})
        )
        self.base_packing_config_path: str | None = self.config.get(
            "base_packing_config_path", None
        )
        self.mode_to_gradient_name: dict[str, str] = self.config.get("mode_to_gradient_name", {})
        # cv_channel_map contains only the component modes used for CV fitting.
        # When set in the config (e.g. by the validation phase), it excludes
        # "interpolated" to prevent circular fitting with the mixed-rule mode.
        # Defaults to the full channel_map when not explicitly specified.
        self.cv_channel_map: dict[str, str] = self.config.get("cv_channel_map", self.channel_map)

    def _setup_recalculate(self) -> None:
        """Parse and validate the recalculate config field."""
        recalculate_config = self.config.get("recalculate")
        if isinstance(recalculate_config, bool):
            self.recalculate = dict.fromkeys(defaults.RECALCULATE.keys(), recalculate_config)
        elif isinstance(recalculate_config, dict):
            self.recalculate = defaults.RECALCULATE.copy()
            self.recalculate.update(recalculate_config)
        else:
            self.recalculate = defaults.RECALCULATE.copy()
            if recalculate_config is not None:
                logger.warning(
                    "Invalid 'recalculate' config."
                    " Using default recalculation settings (all False)."
                )
        logger.info(f"Recalculation settings: {self.recalculate}")

    def _setup_paths(self) -> None:
        """Set directory paths and create output folders."""
        self.project_root = get_project_root()
        self.base_datadir = self.project_root / "data"
        self.base_results_dir = self.project_root / "results"
        output_path = self.config.get("output_path")
        if output_path:
            self.results_dir = self.base_results_dir / output_path
        else:
            self.results_dir = self.base_results_dir / self.name / self.packing_id
        self.results_dir.mkdir(exist_ok=True, parents=True)
        figures_dir = self.config.get("figures_dir")
        if figures_dir is None:
            self.figures_dir = self.results_dir / "figures"
        else:
            self.figures_dir = self.results_dir / figures_dir
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        logs_dir = self.config.get("logs_dir")
        if logs_dir is None:
            self.logs_dir = self.results_dir / "logs"
        else:
            self.logs_dir = self.results_dir / logs_dir
        self.logs_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Results directory: {self.results_dir.relative_to(self.project_root)}")
        logger.info(f"Figures directory: {self.figures_dir.relative_to(self.project_root)}")
        logger.info(f"Logs directory: {self.logs_dir.relative_to(self.project_root)}")


class AnalysisRunner:
    """Main class for running different types of analysis workflows."""

    def __init__(self) -> None:
        self.shared_data: dict = {}

    def run_analysis(self, config_file: str) -> None:
        """Run analysis based on the configuration file."""
        config = AnalysisConfig(config_file)

        logger.info(
            f"Starting '{config.name}' analysis (distribution_method={config.distribution_method})"
        )
        for step in config.analysis_steps:
            self._execute_step(step, config)

    def _execute_step(self, step: str, config: AnalysisConfig) -> None:
        """Execute a single analysis step."""
        logger.info(f"Executing step: {step}")

        step_method_map = {
            "load_common_data": self._load_common_data,
            "calculate_distances": self._calculate_distances,
            "plot_distance_distributions": self._plot_distance_distributions,
            "run_emd_analysis": self._run_emd_analysis,
            "run_ks_analysis": self._run_ks_analysis,
            "run_pairwise_envelope_test": self._run_pairwise_envelope_test,
            "run_per_cell_envelope_test": self._run_per_cell_envelope_test,
            "run_occupancy_analysis": self._run_occupancy_analysis,
            "run_occupancy_emd_analysis": self._run_occupancy_emd_analysis,
            "run_occupancy_pairwise_envelope_test": self._run_occupancy_pairwise_envelope_test,
            "run_occupancy_ks_analysis": self._run_occupancy_ks_analysis,
            "run_rule_interpolation_cv": self._run_rule_interpolation_cv,
            "build_virtual_mixture_mode": self._build_virtual_mixture_mode,
            "compare_virtual_mixture_occupancy": self._compare_virtual_mixture_occupancy,
        }

        if step in step_method_map:
            step_method_map[step](config)
        else:
            logger.warning(f"Unknown step: '{step}'. Available steps: {list(step_method_map)}")

    def _load_common_data(self, config: AnalysisConfig):
        """Load position data and mesh information common to all analyses."""
        if (
            "all_positions" in self.shared_data
            and "combined_mesh_information_dict" in self.shared_data
            and not config.recalculate["load_common_data"]
        ):
            logger.info("Using cached position data and mesh information")
            return

        logger.info("Loading position data")
        all_positions = get_position_data_from_outputs(
            structure_id=config.structure_id,
            packing_id=config.packing_id,
            packing_modes=config.packing_modes,
            base_datadir=config.base_datadir,
            results_dir=config.results_dir,
            packing_output_folder=config.packing_output_folder,
            ingredient_key=config.ingredient_key,
            recalculate=config.recalculate["load_common_data"],
        )

        logger.info("Loading mesh information")
        all_structures = list(set(config.channel_map.values()))
        combined_mesh_information_dict = {}
        for structure_id in all_structures:
            mesh_information_dict = get_mesh_information_dict_for_structure(
                structure_id=structure_id,
                base_datadir=config.base_datadir,
                recalculate=config.recalculate["load_common_data"],
            )
            combined_mesh_information_dict[structure_id] = mesh_information_dict

        self.shared_data["all_positions"] = all_positions
        self.shared_data["combined_mesh_information_dict"] = combined_mesh_information_dict

    def _calculate_distances(self, config: AnalysisConfig):
        """Calculate distance measures and normalize."""
        if (
            "all_distance_dict" in self.shared_data
            and not config.recalculate["calculate_distances"]
        ):
            logger.info("Using cached distance calculations")
            return

        if (
            "all_positions" not in self.shared_data
            or "combined_mesh_information_dict" not in self.shared_data
        ):
            logger.warning("Position data not loaded. Loading common data first.")
            self._load_common_data(config)

        logger.info("Calculating distance measures")
        all_distance_dict = distance.get_distance_dictionary(
            all_positions=self.shared_data["all_positions"],
            distance_measures=config.distance_measures,
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            results_dir=config.results_dir,
            recalculate=config.recalculate["calculate_distances"],
            num_workers=config.num_workers,
            parallel_backend=config.distance_backend,
        )

        all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
            distance_distribution_dict=all_distance_dict,
            minimum_distance=config.minimum_distance,
        )

        all_distance_dict = distance.normalize_distance_dictionary(
            all_distance_dict=all_distance_dict,
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            normalization=config.normalization,
        )

        self.shared_data["all_distance_dict"] = all_distance_dict

    def _plot_distance_distributions(self, config: AnalysisConfig) -> None:
        """Plot distance distributions (discrete histograms or KDE curves)."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info(f"Plotting distance distributions ({config.distribution_method})")
        distance_figures_dir = config.figures_dir / "distance" / "distributions"
        distance_figures_dir.mkdir(exist_ok=True, parents=True)

        method = config.distribution_method
        pdf_kwargs: dict = {
            "all_distance_dict": self.shared_data["all_distance_dict"],
            "distance_measures": config.distance_measures,
            "packing_modes": config.packing_modes,
            "method": method,
            "distance_limits": DISTANCE_LIMITS,
            "results_dir": config.results_dir,
            "recalculate": config.recalculate["plot_distance_distributions"],
            "minimum_distance": None,  # floored in _calculate_distances
            **config.distance_pdf_params,
        }
        distance_pdf_dict = distance.compute_distance_pdfs(**pdf_kwargs)
        self.shared_data["distance_pdf_dict"] = distance_pdf_dict

        _ = visualization.plot_distance_distributions(
            distance_pdf_dict=distance_pdf_dict,
            distance_measures=config.distance_measures,
            packing_modes=config.packing_modes,
            figures_dir=distance_figures_dir,
            suffix=config.suffix,
            normalization=config.normalization,
            save_format=config.save_format,
            **config.distance_plot_params,
        )

        log_file_path = (
            config.logs_dir
            / f"{config.structure_name}_distance_distribution_central_tendencies{config.suffix}.log"
        )
        distance.log_central_tendencies_for_distance_distributions(
            all_distance_dict=self.shared_data["all_distance_dict"],
            distance_measures=config.distance_measures,
            packing_modes=config.packing_modes,
            file_path=log_file_path,
        )

    def _run_emd_analysis(self, config: AnalysisConfig) -> None:
        """Run EMD analysis including pairwise EMD matrix plots."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info("Running EMD analysis")
        emd_figures_dir = config.figures_dir / "distance" / "emd"
        emd_figures_dir.mkdir(exist_ok=True, parents=True)

        df_emd = distance.get_distance_distribution_emd_df(
            all_distance_dict=self.shared_data["all_distance_dict"],
            packing_modes=config.packing_modes,
            distance_measures=config.distance_measures,
            results_dir=config.results_dir,
            recalculate=config.recalculate["run_emd_analysis"],
            suffix=config.suffix,
            num_workers=config.num_workers,
            pool_seeds=config.emd_pool_seeds,
        )
        if config.emd_same_cell:
            df_emd = distance.same_cell_emd_pairs(df_emd)

        # Per-comparison-type plots
        for comparison_type in ["intra_mode", "baseline"]:
            _ = visualization.plot_emd_comparisons(
                df_emd=df_emd,
                distance_measures=config.distance_measures,
                comparison_type=comparison_type,  # type: ignore
                baseline_mode=config.baseline_mode,
                packing_modes=config.packing_modes,
                figures_dir=emd_figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
                annotate_significance=False,
            )

        # Pairwise EMD matrix per distance measure
        distance_pdf_dict = self.shared_data.get("distance_pdf_dict")
        for dm in config.distance_measures:
            emd_matrix_kwargs: dict = {
                "df_emd": df_emd,
                "packing_modes": config.packing_modes,
                "distance_measure": dm,
                "normalization": config.normalization,
                "figures_dir": emd_figures_dir,
                "suffix": config.suffix,
                "save_format": config.save_format,
                **config.emd_plot_params,
            }
            if distance_pdf_dict is not None:
                emd_matrix_kwargs["distance_pdf_dict"] = distance_pdf_dict
            else:
                emd_matrix_kwargs["all_distance_dict"] = self.shared_data["all_distance_dict"]
                emd_matrix_kwargs["distance_limits"] = DISTANCE_LIMITS
                emd_matrix_kwargs["bin_width"] = config.distance_pdf_params.get("bin_width", 0.01)
            _ = visualization.plot_pairwise_emd_matrix(**emd_matrix_kwargs)

        pairwise_emd_log_file_path = (
            config.logs_dir
            / f"{config.packing_id}_emd_pairwise_central_tendencies{config.suffix}.log"
        )
        distance.log_pairwise_emd_central_tendencies(
            df_emd=df_emd,
            distance_measures=config.distance_measures,
            packing_modes=config.packing_modes,
            log_file_path=pairwise_emd_log_file_path,
        )

        self.shared_data["df_emd"] = df_emd

    def _run_ks_analysis(self, config: AnalysisConfig) -> None:
        """Run pairwise KS test analysis across all ordered pairs of packing modes."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info("Running pairwise KS test analysis")
        ks_figures_dir = config.figures_dir / "distance" / "ks_test"
        ks_figures_dir.mkdir(exist_ok=True, parents=True)

        # Collect per-ref-mode KS results
        pairwise_ks_dfs: list[pd.DataFrame] = []
        for ref_mode in config.packing_modes:
            ks_test_df = distance.get_ks_test_df(
                distance_measures=config.distance_measures,
                packing_modes=config.packing_modes,
                all_distance_dict=self.shared_data["all_distance_dict"],
                baseline_mode=ref_mode,
                significance_level=config.ks_significance_level,
                save_dir=None,
                recalculate=config.recalculate["run_ks_analysis"],
            )
            ks_test_df["baseline_mode"] = ref_mode
            pairwise_ks_dfs.append(ks_test_df)

        pairwise_ks_test_df = pd.concat(pairwise_ks_dfs, ignore_index=True)

        # Bootstrap per ref_mode and collect results
        pairwise_ks_bootstrap_dfs: list[pd.DataFrame] = []
        ref_mode = config.baseline_mode
        ks_log_file_path = (
            config.logs_dir / f"{config.structure_name}_pairwise_ks_central_tendencies"
            f"_ref_{ref_mode}{config.suffix}.log"
        )
        other_modes = [m for m in config.packing_modes if m != ref_mode]
        ref_ks_df = pairwise_ks_test_df.query("baseline_mode == @ref_mode")
        df_boot = distance.bootstrap_ks_tests(
            ks_test_df=ref_ks_df,
            distance_measures=config.distance_measures,
            packing_modes=other_modes,
            n_bootstrap=config.n_bootstrap,
        )
        df_boot["baseline_mode"] = ref_mode
        pairwise_ks_bootstrap_dfs.append(df_boot)

        _ = visualization.plot_ks_test_results(
            df_ks_bootstrap=df_boot,
            distance_measures=config.distance_measures,
            figures_dir=ks_figures_dir,
            suffix=f"{config.suffix}_vs_{ref_mode}",
            save_format=config.save_format,
        )
        distance.log_central_tendencies_for_ks(
            df_ks_bootstrap=df_boot,
            distance_measures=config.distance_measures,
            file_path=ks_log_file_path,
        )

        self.shared_data["pairwise_ks_bootstrap_df"] = pd.concat(
            pairwise_ks_bootstrap_dfs, ignore_index=True
        )

    def _run_pairwise_envelope_test(self, config: AnalysisConfig) -> None:
        """Run pairwise Monte Carlo envelope test across all ordered mode pairs."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info("Running pairwise envelope test")
        envelope_figures_dir = config.figures_dir / "distance" / "pairwise_envelope"
        envelope_figures_dir.mkdir(exist_ok=True, parents=True)

        pairwise_results = pairwise_envelope_test(
            all_distance_dict=self.shared_data["all_distance_dict"],
            packing_modes=config.packing_modes,
            distance_measures=config.distance_measures,
            **config.envelope_test_params,
        )

        ep = config.envelope_plot_params
        for dm in config.distance_measures:
            _ = visualization.plot_pairwise_envelope_matrix(
                pairwise_results=pairwise_results,
                distance_measure=dm,
                figures_dir=envelope_figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
                figure_size=tuple(ep["per_dm_matrix_figsize"]),
                font_scale=ep["per_dm_matrix_font_scale"],
            )

        # Joint test across all distance measures
        _ = visualization.plot_pairwise_envelope_matrix(
            pairwise_results=pairwise_results,
            distance_measure=None,
            figures_dir=envelope_figures_dir,
            suffix=config.suffix,
            save_format=config.save_format,
            figure_size=tuple(ep["joint_matrix_figsize"]),
            font_scale=ep["joint_matrix_font_scale"],
        )

        # Per-DM rejection bars (per test mode)
        for test_mode in config.packing_modes:
            for joint_test in [False, True]:
                _ = visualization.plot_per_dm_rejection_bars(
                    pairwise_results=pairwise_results,
                    compare_mode=test_mode,
                    joint_test=joint_test,
                    figures_dir=envelope_figures_dir,
                    figsize=tuple(ep["rejection_bars_figsize"]),
                    font_scale=ep["rejection_bars_font_scale"],
                    suffix=config.suffix,
                    save_format=config.save_format,
                )

        # Overlay all mode envelopes per distance measure
        _ = visualization.plot_per_dm_envelopes_overlaid(
            pairwise_results=pairwise_results,
            figures_dir=envelope_figures_dir,
            suffix=config.suffix,
            figsize=tuple(ep["overlaid_figsize"]),
            save_format=config.save_format,
        )

        self.shared_data["pairwise_envelope_results"] = pairwise_results

    def _run_per_cell_envelope_test(self, config: AnalysisConfig) -> None:
        """Run the per-cell Monte Carlo rank-envelope (CSR) test: baseline vs each null model."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info("Running per-cell envelope test (baseline=%s)", config.baseline_mode)
        figures_dir = config.figures_dir / "distance" / "per_cell_envelope"
        figures_dir.mkdir(exist_ok=True, parents=True)

        per_cell_results = per_cell_envelope_test(
            all_distance_dict=self.shared_data["all_distance_dict"],
            baseline_mode=config.baseline_mode,
            packing_modes=config.packing_modes,
            distance_measures=config.distance_measures,
            **config.envelope_test_params,
        )
        self.shared_data["per_cell_envelope_results"] = per_cell_results

        # Persist the BH-corrected per-cell q-values (rows = cells) for the record.
        summary = per_cell_results["summary"]
        qvals = summary["per_distance_measure_qvals"].copy()
        qvals.index = per_cell_results["cell_ids"]
        csv_path = config.results_dir / f"per_cell_envelope_qvals{config.suffix}.csv"
        qvals.to_csv(csv_path)
        logger.info("Saved per-cell envelope q-values to %s", csv_path)

        # Persist the full per-cell results (envelopes + p-values) for later inspection.
        results_path = config.results_dir / f"per_cell_envelope_results{config.suffix}.dat"
        with open(results_path, "wb") as f:
            pickle.dump(per_cell_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved per-cell envelope results to %s", results_path)

        # Rejection-fraction bars (per distance measure and joint across measures).
        for joint_test in (False, True):
            _ = visualization.plot_per_cell_rejection_bars(
                per_cell_results=per_cell_results,
                joint_test=joint_test,
                figures_dir=figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
            )

        # Example per-cell envelope overlays (experimental ECDF vs each null band).
        for select in ("most_rejected", "median"):
            _ = visualization.plot_per_cell_envelope_examples(
                per_cell_results=per_cell_results,
                select=select,
                figures_dir=figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
            )

    def _run_occupancy_analysis(self, config: AnalysisConfig) -> None:
        """Run occupancy analysis (discrete histogram or KDE, per distribution_method)."""
        if (
            "all_distance_dict" not in self.shared_data
            or "combined_mesh_information_dict" not in self.shared_data
        ):
            logger.warning(
                "Required data not available. Loading common data and calculating distances first."
            )
            self._load_common_data(config)
            self._calculate_distances(config)

        logger.info(f"Running occupancy analysis ({config.distribution_method})")

        self.shared_data["occupancy_dict"] = {}
        if config.distribution_method != "discrete":
            self.shared_data.setdefault("distance_kde_dict", {})
        for occupancy_distance_measure in config.occupancy_distance_measures:
            if config.distribution_method == "discrete":
                self._run_single_occupancy_analysis_discrete(config, occupancy_distance_measure)
            else:
                self._run_single_occupancy_analysis(config, occupancy_distance_measure)

    def _run_single_occupancy_analysis_discrete(
        self,
        config: AnalysisConfig,
        occupancy_distance_measure: str,
    ) -> None:
        """Run discrete histogram occupancy analysis for one distance measure."""
        logger.info(
            f"Discrete occupancy analysis for distance measure: {occupancy_distance_measure}"
        )
        occupancy_figures_dir = config.figures_dir / occupancy_distance_measure
        occupancy_figures_dir.mkdir(exist_ok=True, parents=True)

        bin_width = config.bin_width_map.get(occupancy_distance_measure, 0.2)
        xlim = config.occupancy_params.get(occupancy_distance_measure, {}).get("xlim", 8)

        binned_occupancy_dict = occupancy.get_binned_occupancy_dict_from_distance_dict(
            all_distance_dict=self.shared_data["all_distance_dict"],
            combined_mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            distance_measure=occupancy_distance_measure,
            bin_width=bin_width,
            x_max=xlim,
            x_min=config.discrete_occupancy_params["x_min"],
            results_dir=config.results_dir,
            pseudocount=config.discrete_occupancy_params["pseudocount"],
            min_count=config.discrete_occupancy_params["min_count"],
            recalculate=config.recalculate["run_occupancy_analysis"],
            suffix=config.suffix,
        )

        self.shared_data["occupancy_dict"][occupancy_distance_measure] = binned_occupancy_dict

        # Illustration for one example cell
        # _ = visualization.plot_occupancy_illustration(
        #     occupancy_dict=binned_occupancy_dict,
        #     packing_mode=config.baseline_mode,
        #     figures_dir=occupancy_figures_dir,
        #     suffix=config.suffix,
        #     distance_measure=occupancy_distance_measure,
        #     normalization=config.normalization,
        #     save_format=config.save_format,
        # )

        # Occupancy ratio: mean + 95% pointwise envelope
        ylim = config.occupancy_params.get(occupancy_distance_measure, {}).get("ylim", 3)
        _ = visualization.plot_occupancy_ratio(
            occupancy_dict=binned_occupancy_dict,
            channel_map=config.channel_map,
            figures_dir=occupancy_figures_dir,
            normalization=config.normalization,
            suffix=config.suffix,
            distance_measure=occupancy_distance_measure,
            # xlim=xlim,
            ylim=ylim,
            save_format=config.save_format,
            fig_params=config.occupancy_params.get(
                "fig_params", {"dpi": 300, "figsize": (3.5, 2.5)}
            ),
        )

    def _run_single_occupancy_analysis(
        self,
        config: AnalysisConfig,
        occupancy_distance_measure: str,
    ) -> None:
        """Run KDE occupancy analysis for a single distance measure."""
        logger.info(
            f"Running occupancy analysis for distance measure: {occupancy_distance_measure}"
        )

        occupancy_figures_dir = config.figures_dir / "occupancy" / occupancy_distance_measure
        occupancy_figures_dir.mkdir(exist_ok=True, parents=True)

        # Create KDE dictionary
        distance_kde_dict = distance.get_distance_distribution_kde(
            all_distance_dict=self.shared_data["all_distance_dict"],
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            save_dir=config.results_dir,
            recalculate=config.recalculate["run_occupancy_analysis"],
            suffix=config.suffix,
            normalization=config.normalization,
            distance_measure=occupancy_distance_measure,
            minimum_distance=config.available_minimum_distance,
        )
        self.shared_data["distance_kde_dict"][occupancy_distance_measure] = distance_kde_dict

        # Compute and store occupancy ratios
        occupancy_dict = occupancy.get_kde_occupancy_dict(
            distance_kde_dict=distance_kde_dict,
            channel_map=config.channel_map,
            results_dir=config.results_dir,
            recalculate=config.recalculate["run_occupancy_analysis"],
            suffix=config.suffix,
            distance_measure=occupancy_distance_measure,
            bandwidth=config.occupancy_params[occupancy_distance_measure]["bandwidth"],
            # .get(), not subscript: the occupancy_params merge is shallow, so a
            # config that overrides any key of occupancy_params[dm] replaces the
            # whole inner dict and drops num_points.
            num_points=config.occupancy_params[occupancy_distance_measure].get(
                "num_points", defaults.OCCUPANCY_NUM_POINTS
            ),
            x_min=0,
            x_max=config.occupancy_params[occupancy_distance_measure]["xlim"],
            num_workers=config.num_workers,
            parallel_backend=config.occupancy_backend,
            floors_um={
                "minimum_distance_um": config.minimum_distance_um,
                "available_minimum_distance_um": config.available_minimum_distance_um,
            },
        )

        self.shared_data["occupancy_dict"][occupancy_distance_measure] = occupancy_dict

        # Illustration for one example cell
        # _ = visualization.plot_occupancy_illustration(
        #     distance_kde_dict=distance_kde_dict,
        #     packing_mode="random",
        #     figures_dir=occupancy_figures_dir,
        #     suffix=config.suffix,
        #     distance_measure=occupancy_distance_measure,
        #     normalization=config.normalization,
        #     save_format=config.save_format,
        #     # xlim=config.occupancy_params[occupancy_distance_measure]["xlim"],
        #     num_points=250,
        # )

        # Plot occupancy ratio: mean + pointwise envelope
        _ = visualization.plot_occupancy_ratio(
            occupancy_dict=self.shared_data["occupancy_dict"][occupancy_distance_measure],
            channel_map=config.channel_map,
            baseline_mode=config.baseline_mode,
            figures_dir=occupancy_figures_dir,
            suffix=config.suffix,
            normalization=config.normalization,
            distance_measure=occupancy_distance_measure,
            save_format=config.save_format,
            # xlim=config.occupancy_params[occupancy_distance_measure]["xlim"],
            ylim=config.occupancy_params[occupancy_distance_measure]["ylim"],
            fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
            plot_individual=config.occupancy_params.get("plot_individual", False),
            show_legend=config.occupancy_params.get("show_legend", True),
        )

    def _run_occupancy_emd_analysis(self, config: AnalysisConfig) -> None:
        """Run occupancy EMD analysis including pairwise matrix plots."""
        if "occupancy_dict" not in self.shared_data:
            logger.warning("Occupancy data not available. Running occupancy analysis first.")
            self._run_occupancy_analysis(config)

        logger.info("Running occupancy EMD analysis")
        occupancy_emd_figures_dir = config.figures_dir / "occupancy" / "emd"
        occupancy_emd_figures_dir.mkdir(exist_ok=True, parents=True)

        occupancy_emd_df = occupancy.get_occupancy_emd_df(
            combined_occupancy_dict=self.shared_data["occupancy_dict"],
            packing_modes=config.packing_modes,
            distance_measures=config.occupancy_distance_measures,
            results_dir=config.results_dir,
            recalculate=config.recalculate["run_occupancy_emd_analysis"],
            suffix=config.suffix,
        )
        if config.emd_same_cell:
            occupancy_emd_df = distance.same_cell_emd_pairs(occupancy_emd_df)

        _ = visualization.plot_emd_comparisons(
            df_emd=occupancy_emd_df,
            distance_measures=config.occupancy_distance_measures,
            comparison_type="baseline",
            baseline_mode=config.baseline_mode,
            packing_modes=config.packing_modes,
            figures_dir=occupancy_emd_figures_dir,
            suffix=config.suffix,
            save_format=config.save_format,
        )

        occupancy_emd_log_file_path = (
            config.logs_dir
            / f"{config.packing_id}_occupancy_emd_pairwise_central_tendencies{config.suffix}.log"
        )
        distance.log_pairwise_emd_central_tendencies(
            df_emd=occupancy_emd_df,
            distance_measures=config.occupancy_distance_measures,
            packing_modes=config.packing_modes,
            log_file_path=occupancy_emd_log_file_path,
        )

        # Pairwise occupancy EMD matrix per distance measure
        for dm in config.occupancy_distance_measures:
            dm_figures_dir = config.figures_dir / "occupancy" / dm
            dm_figures_dir.mkdir(exist_ok=True, parents=True)
            xlim = config.occupancy_params.get(dm, {}).get("xlim", 8)
            ylim = config.occupancy_params.get(dm, {}).get("ylim", 3)
            _ = visualization.plot_pairwise_emd_matrix(
                df_emd=occupancy_emd_df,
                binned_occupancy_dict=self.shared_data["occupancy_dict"][dm],
                packing_modes=config.packing_modes,
                distance_measure=dm,
                normalization=config.normalization,
                xlim=xlim,
                ylim=ylim,
                figures_dir=dm_figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
            )

        self.shared_data["occupancy_emd_df"] = occupancy_emd_df

    def _run_occupancy_pairwise_envelope_test(self, config: AnalysisConfig) -> None:
        """Run pairwise Monte Carlo rank-envelope test on per-cell occupancy ratio curves."""
        if "occupancy_dict" not in self.shared_data:
            logger.warning("Occupancy data not available. Running occupancy analysis first.")
            self._run_occupancy_analysis(config)

        logger.info("Running occupancy pairwise envelope test")
        envelope_figures_dir = config.figures_dir / "occupancy" / "pairwise_envelope"
        envelope_figures_dir.mkdir(exist_ok=True, parents=True)

        occ_pairwise_results = occupancy.pairwise_envelope_test_occupancy(
            combined_occupancy_dict=self.shared_data["occupancy_dict"],
            packing_modes=config.packing_modes,
            alpha=config.envelope_test_params["alpha"],
            statistic=config.envelope_test_params["statistic"],
            comparison_type="ecdf",
        )

        ep = config.envelope_plot_params
        for dm in config.occupancy_distance_measures:
            _ = visualization.plot_pairwise_envelope_matrix(
                pairwise_results=occ_pairwise_results,
                distance_measure=dm,
                figures_dir=envelope_figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
                figure_size=tuple(ep["per_dm_matrix_figsize"]),
                font_scale=ep["per_dm_matrix_font_scale"],
            )

        # Joint test across all occupancy distance measures
        _ = visualization.plot_pairwise_envelope_matrix(
            pairwise_results=occ_pairwise_results,
            distance_measure=None,
            figures_dir=envelope_figures_dir,
            suffix=config.suffix,
            save_format=config.save_format,
            figure_size=tuple(ep["joint_matrix_figsize"]),
            font_scale=ep["joint_matrix_font_scale"],
        )

        # Per-DM rejection bars (per test mode)
        for test_mode in config.packing_modes:
            for joint_test in [False, True]:
                _ = visualization.plot_per_dm_rejection_bars(
                    pairwise_results=occ_pairwise_results,
                    compare_mode=test_mode,
                    joint_test=joint_test,
                    figures_dir=envelope_figures_dir,
                    figsize=tuple(ep["rejection_bars_figsize"]),
                    suffix=config.suffix,
                    save_format=config.save_format,
                )

        # Overlay all mode envelopes per distance measure
        _ = visualization.plot_per_dm_envelopes_overlaid(
            pairwise_results=occ_pairwise_results,
            figures_dir=envelope_figures_dir,
            suffix=config.suffix,
            figsize=tuple(ep["overlaid_figsize"]),
            save_format=config.save_format,
        )

        self.shared_data["occupancy_pairwise_envelope_results"] = occ_pairwise_results

    def _run_rule_interpolation_cv(self, config: AnalysisConfig) -> None:
        """Run k-fold cross-validated NNLS rule interpolation."""
        occupancy_key = "occupancy_dict"
        if occupancy_key not in self.shared_data:
            logger.warning("Occupancy data not available. Running occupancy analysis first.")
            self._run_occupancy_analysis(config)

        logger.info("Running rule interpolation cross-validation")
        cv_params = config.rule_interpolation_cv_params
        cv_result = rule_interpolation.run_rule_interpolation_cv(
            occupancy_dict=self.shared_data[occupancy_key],
            channel_map=config.cv_channel_map,
            baseline_mode=config.baseline_mode,
            n_folds=cv_params["n_folds"],
            n_repeats=cv_params.get("n_repeats", 10),
            random_state=cv_params["random_state"],
            distance_measures=config.occupancy_distance_measures,
            results_dir=config.results_dir,
            recalculate=config.recalculate["run_rule_interpolation_cv"],
            suffix=config.suffix,
            grouping=cv_params.get("grouping", "combined"),
        )
        self.shared_data["rule_interpolation_cv_result"] = cv_result

        # CV visualizations
        cv_figures_dir = config.figures_dir / "cross_validation"
        cv_figures_dir.mkdir(exist_ok=True, parents=True)

        cv_df = rule_interpolation.summarize_cv_results(cv_result)
        _ = visualization.plot_cv_mse_summary(
            cv_df=cv_df,
            figures_dir=cv_figures_dir,
            suffix=config.suffix,
            save_format=config.save_format,
        )
        _ = visualization.plot_cv_coefficient_stability(
            cv_result=cv_result,
            figures_dir=cv_figures_dir,
            suffix=config.suffix,
            save_format=config.save_format,
        )

        # CV mean-coefficient fit overlay
        cv_fit_result = rule_interpolation.fit_result_from_cv(
            cv_result=cv_result,
            occupancy_dict=self.shared_data[occupancy_key],
            channel_map=config.cv_channel_map,
            baseline_mode=config.baseline_mode,
            distance_measures=config.occupancy_distance_measures,
        )
        self.shared_data["rule_interpolation_cv_fit_result"] = cv_fit_result

        # Log CV summary to file (after cv_fit_result so both MSE types can be logged)
        cv_log_path = (
            config.logs_dir
            / f"{config.structure_name}_rule_interpolation_cv_summary{config.suffix}.log"
        )
        rule_interpolation.log_cv_summary(
            cv_result=cv_result,
            file_path=cv_log_path,
            reconstruction_fit_result=cv_fit_result,
        )

        cv_fit_figures_dir = config.figures_dir / "cv_fit_overlay"
        cv_fit_figures_dir.mkdir(exist_ok=True, parents=True)
        for dm in config.occupancy_distance_measures:
            dm_cv_fit_figures_dir = cv_fit_figures_dir / dm
            dm_cv_fit_figures_dir.mkdir(exist_ok=True, parents=True)
            for plot_type in ("individual", "joint"):
                _ = visualization.plot_rule_interpolation_fit(
                    fit_result=cv_fit_result,
                    occupancy_dict=self.shared_data[occupancy_key],
                    channel_map=config.channel_map,
                    baseline_mode=config.baseline_mode,
                    distance_measure=dm,
                    plot_type=plot_type,
                    figures_dir=dm_cv_fit_figures_dir,
                    # xlim=config.occupancy_params[dm]["xlim"],
                    ylim=config.occupancy_params[dm]["ylim"],
                    suffix=f"{config.suffix}_cv_mean",
                    save_format=config.save_format,
                )

        # Optionally generate packing configs for held-out cells
        if cv_params.get("generate_packing_configs") and config.base_packing_config_path:
            packing_configs_dir = config.results_dir / "mixed_rule_packing_configs"
            written = rule_interpolation.generate_mixed_rule_packing_configs(
                cv_result=cv_result,
                base_config_path=config.project_root / config.base_packing_config_path,
                output_config_dir=packing_configs_dir,
                mode_to_gradient_name=config.mode_to_gradient_name,
                scope=cv_params.get("packing_config_scope", "joint"),
            )
            logger.info(
                f"Generated {len(written)} mixed-rule packing config(s) in {packing_configs_dir}"
            )
            self.shared_data["mixed_rule_packing_configs"] = written

        # Run mixed-rule validation if packing outputs exist
        mixed_rule_results_dir = config.results_dir / "mixed_rule_packings"
        if mixed_rule_results_dir.exists():
            logger.info("Running mixed-rule validation")
            validation_result = rule_interpolation.run_mixed_rule_validation(
                combined_occupancy_dict=self.shared_data[occupancy_key],
                channel_map=config.channel_map,
                baseline_mode=config.baseline_mode,
                packing_modes=config.packing_modes,
                distance_measures=config.occupancy_distance_measures,
                results_dir=config.results_dir,
                recalculate=config.recalculate["run_rule_interpolation_cv"],
            )
            self.shared_data["mixed_rule_validation_result"] = validation_result

    def _run_occupancy_ks_analysis(self, config: AnalysisConfig) -> None:
        """Run pairwise KS test analysis across all ordered mode pairs for occupancy."""
        if "occupancy_dict" not in self.shared_data:
            logger.warning("Occupancy data not available. Running occupancy analysis first.")
            self._run_occupancy_analysis(config)

        logger.info("Running pairwise occupancy KS test analysis")
        ks_figures_dir = config.figures_dir / "occupancy" / "ks_test"
        ks_figures_dir.mkdir(exist_ok=True, parents=True)

        # Collect per-ref-mode KS results
        pairwise_occ_ks_dfs: list[pd.DataFrame] = []
        for ref_mode in config.packing_modes:
            occ_ks_df = occupancy.get_occupancy_ks_test_df(
                distance_measures=config.occupancy_distance_measures,
                packing_modes=config.packing_modes,
                combined_occupancy_dict=self.shared_data["occupancy_dict"],
                baseline_mode=ref_mode,
                significance_level=config.ks_significance_level,
                results_dir=None,
                recalculate=config.recalculate["run_occupancy_ks_analysis"],
            )
            occ_ks_df["baseline_mode"] = ref_mode
            pairwise_occ_ks_dfs.append(occ_ks_df)

        pairwise_occ_ks_test_df = pd.concat(pairwise_occ_ks_dfs, ignore_index=True)

        # Bootstrap per ref_mode and collect results
        pairwise_occ_ks_bootstrap_dfs: list[pd.DataFrame] = []
        ks_log_file_path = (
            config.logs_dir
            / f"{config.structure_name}_pairwise_occupancy_ks_central_tendencies{config.suffix}.log"
        )
        for ref_mode in config.packing_modes:
            other_modes = [m for m in config.packing_modes if m != ref_mode]
            ref_ks_df = pairwise_occ_ks_test_df.query("baseline_mode == @ref_mode")
            df_boot = distance.bootstrap_ks_tests(
                ks_test_df=ref_ks_df,
                distance_measures=config.occupancy_distance_measures,
                packing_modes=other_modes,
                n_bootstrap=config.n_bootstrap,
            )
            df_boot["baseline_mode"] = ref_mode
            pairwise_occ_ks_bootstrap_dfs.append(df_boot)

            _ = visualization.plot_ks_test_results(
                df_ks_bootstrap=df_boot,
                distance_measures=config.occupancy_distance_measures,
                figures_dir=ks_figures_dir,
                suffix=f"{config.suffix}_vs_{ref_mode}",
                save_format=config.save_format,
            )
            distance.log_central_tendencies_for_ks(
                df_ks_bootstrap=df_boot,
                distance_measures=config.occupancy_distance_measures,
                file_path=ks_log_file_path,
            )

        self.shared_data["pairwise_occupancy_ks_bootstrap_df"] = pd.concat(
            pairwise_occ_ks_bootstrap_dfs, ignore_index=True
        )

    def _build_virtual_mixture_mode(self, config: AnalysisConfig) -> None:
        """Inject a virtual-mixture mode into the shared distance data (mixing D2)."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        params = config.virtual_mixture_params
        cv_path_value = params.get("cv_results_path")
        if not cv_path_value:
            raise ValueError(
                "build_virtual_mixture_mode requires 'virtual_mixture_params.cv_results_path' "
                "(a rule_interpolation_cv pickle, relative to the results root or absolute)"
            )
        cv_path = Path(cv_path_value)
        if not cv_path.is_absolute():
            cv_path = get_results_path() / cv_path
        scope = params.get("scope", "joint")
        mode_name = params.get("mode_name", "virtual_mixed")
        base_seed = params.get("base_seed", 42)

        coefficients = mixture_diagnostics.load_mixture_coefficients(cv_path, scope=scope)
        missing_nonzero = [
            rule for rule, c in coefficients.items() if c > 0 and rule not in config.channel_map
        ]
        if missing_nonzero:
            raise ValueError(
                f"Mixture rules {missing_nonzero} have nonzero coefficients but are not "
                f"in this config's channel_map ({list(config.channel_map)})"
            )
        coefficients = {rule: c for rule, c in coefficients.items() if rule in config.channel_map}

        mixed = mixture_diagnostics.build_virtual_mixture_distances(
            all_distance_dict=self.shared_data["all_distance_dict"],
            coefficients=coefficients,
            distance_measures=config.distance_measures,
            base_seed=base_seed,
        )
        for distance_measure in config.distance_measures:
            self.shared_data["all_distance_dict"][distance_measure][mode_name] = mixed[
                distance_measure
            ]
        config.channel_map[mode_name] = config.structure_id
        config.packing_modes = list(config.channel_map.keys())
        self.shared_data["virtual_mixture"] = {
            "mode_name": mode_name,
            "coefficients": coefficients,
            "cv_results_path": str(cv_path),
            "scope": scope,
            "base_seed": base_seed,
        }
        n_cells = len(mixed[config.distance_measures[0]])
        logger.info(
            "Injected virtual-mixture mode %r (%d cells, coefficients %s)",
            mode_name,
            n_cells,
            {rule: round(c, 4) for rule, c in coefficients.items()},
        )

    def _compare_virtual_mixture_occupancy(self, config: AnalysisConfig) -> None:
        """Compare virtual-mixture occupancy with the expected NNLS mixture (mixing D2)."""
        if "virtual_mixture" not in self.shared_data:
            raise ValueError(
                "Run build_virtual_mixture_mode (and run_occupancy_analysis) before "
                "compare_virtual_mixture_occupancy"
            )
        if not self.shared_data.get("occupancy_dict"):
            logger.warning("Occupancy not computed. Running occupancy analysis first.")
            self._run_occupancy_analysis(config)

        info = self.shared_data["virtual_mixture"]
        mode_name = info["mode_name"]
        coefficients = {rule: c for rule, c in info["coefficients"].items() if rule != mode_name}
        figures_dir = config.figures_dir / "occupancy" / "virtual_mixture"
        figures_dir.mkdir(exist_ok=True, parents=True)

        comparison_summaries = {}
        for distance_measure, occupancy_dict in self.shared_data["occupancy_dict"].items():
            expected_by_cell = mixture_diagnostics.expected_mixture_occupancy(
                occupancy_dict, coefficients
            )
            virtual_individual = occupancy_dict[mode_name]["individual"]
            shared_cells = sorted(set(expected_by_cell) & set(virtual_individual))
            if not shared_cells:
                logger.warning(
                    "No shared cells between virtual mixture and expected mixture for %s",
                    distance_measure,
                )
                continue

            xvals = np.asarray(occupancy_dict[mode_name]["combined"]["xvals"])
            virtual_all = np.vstack(
                [np.asarray(virtual_individual[cell]["occupancy_common"]) for cell in shared_cells]
            )
            expected_all = np.vstack([expected_by_cell[cell] for cell in shared_cells])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                virtual_mean = np.nanmean(virtual_all, axis=0)
                expected_mean = np.nanmean(expected_all, axis=0)
                virtual_lo = np.nanpercentile(virtual_all, 2.5, axis=0)
                virtual_hi = np.nanpercentile(virtual_all, 97.5, axis=0)
                per_cell_rmse = np.sqrt(np.nanmean((virtual_all - expected_all) ** 2, axis=1))

            mean_diff = virtual_mean - expected_mean
            summary = {
                "n_cells": len(shared_cells),
                "max_abs_mean_diff": float(np.nanmax(np.abs(mean_diff))),
                "mean_abs_mean_diff": float(np.nanmean(np.abs(mean_diff))),
                "median_per_cell_rmse": float(np.nanmedian(per_cell_rmse)),
            }
            comparison_summaries[distance_measure] = summary
            logger.info("Virtual-mixture vs expected occupancy (%s): %s", distance_measure, summary)

            comparison_df = pd.DataFrame(
                {
                    "xvals": xvals,
                    "virtual_mean": virtual_mean,
                    "virtual_envelope_lo": virtual_lo,
                    "virtual_envelope_hi": virtual_hi,
                    "expected_mean": expected_mean,
                    "mean_diff": mean_diff,
                }
            )
            csv_path = (
                config.results_dir
                / f"{distance_measure}_virtual_mixture_comparison{config.suffix}.csv"
            )
            comparison_df.to_csv(csv_path, index=False)
            logger.info("Saved virtual-mixture comparison to %s", csv_path)

            _ = visualization.plot_virtual_mixture_comparison(
                xvals=xvals,
                virtual_mean=virtual_mean,
                virtual_envelope_lo=virtual_lo,
                virtual_envelope_hi=virtual_hi,
                expected_mean=expected_mean,
                coefficients=coefficients,
                distance_measure=distance_measure,
                figures_dir=figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
            )

        self.shared_data["virtual_mixture_comparison"] = comparison_summaries


def main():
    """Run the analysis workflow based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run cellpack analysis workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis_workflow.py --config_file configs/distance_analysis_config.json
    python run_analysis_workflow.py --config_file configs/biological_variation_config.json
    python run_analysis_workflow.py --config_file configs/occupancy_analysis_config.json
        """,
    )

    parser.add_argument(
        "--config_file", "-c", type=str, required=True, help="Path to the JSON configuration file"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    setup_logging(level=level_map[args.log_level])

    # Run analysis
    start_time = time.time()
    runner = AnalysisRunner()
    runner.run_analysis(args.config_file)
    logger.info(f"Total time taken: {format_time(time.time() - start_time)}")


if __name__ == "__main__":
    main()
