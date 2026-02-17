#!/usr/bin/env python3
"""
Workflow runner for cellpack analysis.

This script runs configurable analysis workflows based on configuration files.
The workflow is controlled by the 'analysis_steps' field in the config, which
specifies which steps to execute in order.

Example analyses:
- biological_variation: Analysis of biological variation factors
- distance_analysis: Distance analysis with EMD and KS tests
- occupancy_analysis: Occupancy analysis with spatial statistics

Available analysis steps:
- load_common_data: Load position data and mesh information
- calculate_distances: Calculate distance measures and normalize
- plot_distance_distributions: Plot distance distributions with KDE
- run_emd_analysis: Run Earth Mover's Distance analysis
- run_ks_analysis: Run Kolmogorov-Smirnov test analysis
- run_occupancy_analysis: Run occupancy analysis with spatial statistics
- run_occupancy_emd_analysis: Run EMD analysis on occupancy data
- run_occupancy_interpolation_analysis: Run interpolation analysis for occupancy data

Example workflows:
- Biological variation: ["load_common_data", "calculate_distances",
  "plot_distance_distributions", "run_emd_analysis"]
- Distance analysis: ["load_common_data", "calculate_distances",
  "plot_distance_distributions", "run_emd_analysis", "run_ks_analysis"]
- Occupancy analysis: ["load_common_data", "calculate_distances", "run_occupancy_analysis"]

Usage:
    python run_analysis_workflow.py --config_file path/to/config.json
    python run_analysis_workflow.py --config_file configs/distance_analysis_config.json
    python run_analysis_workflow.py --config_file configs/biological_variation_config.json
    python run_analysis_workflow.py --config_file configs/occupancy_analysis_config.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

from cellpack_analysis import setup_logging
from cellpack_analysis.analysis.workflows.configs import defaults
from cellpack_analysis.lib import distance, occupancy, visualization
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.io import format_time
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats import normalize_distances

logger = logging.getLogger(__name__)


class AnalysisConfig:
    """Configuration class for analysis workflows."""

    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._setup_parameters()
        self._setup_paths()

    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file) as f:
            return json.load(f)

    def _setup_paths(self):
        """Set directory paths."""
        self.project_root = get_project_root()
        self.base_datadir = self.project_root / "data"
        self.base_results_dir = self.project_root / "results"

        # Setup results directory
        self.results_dir = self.base_results_dir / self.name / self.packing_id
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Setup figures directory
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True, parents=True)

    def _setup_parameters(self):
        """Set analysis parameters from config."""
        # Structure parameters
        self.structure_id = self.config.get("structure_id", defaults.STRUCTURE_ID)
        self.packing_id = self.config.get("packing_id", defaults.PACKING_ID)
        self.structure_name = self.config.get("structure_name", defaults.STRUCTURE_NAME)

        # Analysis parameters
        self.name = self.config["name"]
        self.channel_map = self.config.get("channel_map", {self.structure_id: self.structure_id})
        self.packing_modes = list(self.channel_map.keys())
        self.packing_output_folder = self.config.get(
            "packing_output_folder", defaults.PACKING_OUTPUT_FOLDER
        )
        self.baseline_mode = self.config.get("baseline_mode", self.structure_id)

        # Naming
        self.suffix = self.config.get("suffix", "")

        # Distance measures
        self.distance_measures = self.config.get("distance_measures", defaults.DISTANCE_MEASURES)

        # Normalization
        self.normalization = self.config.get("normalization")
        if self.normalization:
            self.suffix += f"_norm_{self.normalization}"

        # Visualization
        self.save_format = self.config.get("save_format", defaults.SAVE_FORMAT)

        # Analysis-specific parameters
        self.ks_significance_level = self.config.get(
            "ks_significance_level", defaults.KS_SIGNIFICANCE_LEVEL
        )
        self.n_bootstrap = self.config.get("n_bootstrap", defaults.N_BOOTSTRAP)
        self.bandwidth = self.config.get("bandwidth", defaults.BANDWIDTH)

        # Occupancy analysis specific
        self.occupancy_distance_measures = self.config.get(
            "occupancy_distance_measures", defaults.OCCUPANCY_DISTANCE_MEASURES
        )
        self.occupancy_params = defaults.OCCUPANCY_PARAMS.copy()
        self.occupancy_params.update(self.config.get("occupancy_params", {}))

        # Ingredient key for position data
        self.ingredient_key = self.config.get(
            "ingredient_key", f"membrane_interior_{self.structure_name}"
        )

        recalculate_config = self.config.get("recalculate")
        if isinstance(recalculate_config, bool):
            # If recalculate is a boolean, apply to all steps
            self.recalculate = dict.fromkeys(defaults.RECALCULATE.keys(), recalculate_config)
        elif isinstance(recalculate_config, dict):
            # If recalculate is a dict, update defaults with provided values
            self.recalculate = defaults.RECALCULATE.copy()
            self.recalculate.update(recalculate_config)
        else:
            # If recalculate is None or invalid type, use defaults
            self.recalculate = defaults.RECALCULATE
            if recalculate_config is not None:
                logger.warning(
                    "Invalid 'recalculate' config."
                    " Using default recalculation settings (all False)."
                )
        logger.info(f"Recalculation settings: {self.recalculate}")

        # Parallel processing
        self.num_workers = self.config.get("num_workers", defaults.NUM_WORKERS)

        # Analysis steps
        self.analysis_steps = self.config["analysis_steps"]


class AnalysisRunner:
    """Main class for running different types of analysis workflows."""

    def __init__(self):
        self.shared_data = {}

    def run_analysis(self, config_file: str):
        """Run analysis based on the configuration file."""
        config = AnalysisConfig(config_file)

        logger.info(f"Starting {config.name} analysis")
        # Execute each step in the analysis_steps list
        for step in config.analysis_steps:
            self._execute_step(step, config)

    def _execute_step(self, step: str, config: AnalysisConfig):
        """Execute a single analysis step."""
        logger.info(f"Executing step: {step}")

        step_method_map = {
            "load_common_data": self._load_common_data,
            "calculate_distances": self._calculate_distances,
            "plot_distance_distributions": self._plot_distance_distributions,
            "run_emd_analysis": self._run_emd_analysis,
            "run_ks_analysis": self._run_ks_analysis,
            "run_occupancy_analysis": self._run_occupancy_analysis,
            "run_occupancy_emd_analysis": self._run_occupancy_emd_analysis,
            "run_occupancy_interpolation_analysis": self._run_occupancy_interpolation_analysis,
        }

        if step in step_method_map:
            step_method_map[step](config)
        else:
            logger.warning(f"Unknown step: {step}")

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
        )

        all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
            distance_distribution_dict=all_distance_dict, minimum_distance=None
        )

        all_distance_dict = normalize_distances(
            all_distance_dict=all_distance_dict,
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            normalization=config.normalization,
        )

        self.shared_data["all_distance_dict"] = all_distance_dict

    def _plot_distance_distributions(self, config: AnalysisConfig):
        """Plot distance distributions."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info("Plotting distance distributions")
        distance_figures_dir = config.figures_dir / "distance_distributions"
        distance_figures_dir.mkdir(exist_ok=True, parents=True)

        # Plot KDE distributions
        _ = visualization.plot_distance_distributions_kde(
            distance_measures=config.distance_measures,
            packing_modes=config.packing_modes,
            all_distance_dict=self.shared_data["all_distance_dict"],
            figures_dir=distance_figures_dir,
            suffix=config.suffix,
            normalization=config.normalization,
            distance_limits=DISTANCE_LIMITS,
            bandwidth=config.bandwidth,
            save_format=config.save_format,
        )

        # Log central tendencies
        log_file_path = (
            config.results_dir
            / f"{config.structure_name}_distance_distribution_central_tendencies{config.suffix}.log"
        )
        distance.log_central_tendencies_for_distance_distributions(
            all_distance_dict=self.shared_data["all_distance_dict"],
            distance_measures=config.distance_measures,
            packing_modes=config.packing_modes,
            file_path=log_file_path,
        )

    def _run_emd_analysis(self, config: AnalysisConfig):
        """Run EMD analysis."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info("Running EMD analysis")
        emd_figures_dir = config.figures_dir / "emd"
        emd_figures_dir.mkdir(exist_ok=True, parents=True)

        # Get EMD distances
        df_emd = distance.get_distance_distribution_emd_df(
            all_distance_dict=self.shared_data["all_distance_dict"],
            packing_modes=config.packing_modes,
            distance_measures=config.distance_measures,
            results_dir=config.results_dir,
            recalculate=config.recalculate["run_emd_analysis"],
            suffix=config.suffix,
        )

        # Plot EMD comparisons
        emd_log_file_path = (
            config.results_dir / f"{config.packing_id}_emd_central_tendencies{config.suffix}.log"
        )
        for comparison_type in ["intra_mode", "baseline"]:
            _ = visualization.plot_emd_comparisons(
                df_emd=df_emd,
                distance_measures=config.distance_measures,
                comparison_type=comparison_type,  # type: ignore
                baseline_mode=config.baseline_mode,
                figures_dir=emd_figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
                annotate_significance=False,
            )
            distance.log_central_tendencies_for_emd(
                df_emd=df_emd,
                distance_measures=config.distance_measures,
                packing_modes=config.packing_modes,
                baseline_mode=config.baseline_mode,
                log_file_path=emd_log_file_path,
                comparison_type=comparison_type,
            )

        self.shared_data["df_emd"] = df_emd

    def _run_ks_analysis(self, config: AnalysisConfig):
        """Run KS test analysis."""
        if "all_distance_dict" not in self.shared_data:
            logger.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        logger.info("Running KS test analysis")
        ks_figures_dir = config.figures_dir / "ks_test"
        ks_figures_dir.mkdir(exist_ok=True, parents=True)

        # Run KS tests
        ks_test_df = distance.get_ks_test_df(
            distance_measures=config.distance_measures,
            packing_modes=config.packing_modes,
            all_distance_dict=self.shared_data["all_distance_dict"],
            baseline_mode=config.baseline_mode,
            significance_level=config.ks_significance_level,
            save_dir=config.results_dir,
            recalculate=config.recalculate["run_ks_analysis"],
        )

        # Bootstrap KS tests
        df_ks_bootstrap = distance.bootstrap_ks_tests(
            ks_test_df=ks_test_df,
            distance_measures=config.distance_measures,
            packing_modes=[pm for pm in config.packing_modes if pm != config.baseline_mode],
            n_bootstrap=config.n_bootstrap,
        )

        # Plot KS results
        _ = visualization.plot_ks_test_results(
            df_ks_bootstrap=df_ks_bootstrap,
            distance_measures=config.distance_measures,
            figures_dir=ks_figures_dir,
            suffix=config.suffix,
            save_format=config.save_format,
        )

        # Log KS statistics
        ks_log_file_path = (
            config.results_dir
            / f"{config.structure_name}_ks_test_central_tendencies{config.suffix}.log"
        )
        distance.log_central_tendencies_for_ks(
            df_ks_bootstrap=df_ks_bootstrap,
            distance_measures=config.distance_measures,
            file_path=ks_log_file_path,
        )

        self.shared_data["ks_test_df"] = ks_test_df
        self.shared_data["df_ks_bootstrap"] = df_ks_bootstrap

    def _run_occupancy_analysis(self, config: AnalysisConfig):
        """Run occupancy analysis workflow."""
        if (
            "all_distance_dict" not in self.shared_data
            or "combined_mesh_information_dict" not in self.shared_data
        ):
            logger.warning(
                "Required data not available. Loading common data and calculating distances first."
            )
            self._load_common_data(config)
            self._calculate_distances(config)

        logger.info("Running occupancy analysis")

        # Run occupancy analysis for each specified distance measure
        self.shared_data["occupancy_dict"] = {}
        for occupancy_distance_measure in config.occupancy_distance_measures:
            self._run_single_occupancy_analysis(
                config,
                occupancy_distance_measure,
            )

    def _run_single_occupancy_analysis(
        self,
        config: AnalysisConfig,
        occupancy_distance_measure: str,
    ):
        """Run occupancy analysis for a single distance measure."""
        logger.info(
            f"Running occupancy analysis for distance measure: {occupancy_distance_measure}"
        )

        occupancy_figures_dir = config.figures_dir / occupancy_distance_measure
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
            minimum_distance=-1,
        )

        # Plot illustration for occupancy distribution
        _ = visualization.plot_occupancy_illustration(
            kde_dict=distance_kde_dict,
            baseline_mode="random",
            suffix=config.suffix,
            distance_measure=occupancy_distance_measure,
            normalization=config.normalization,
            method="pdf",
            seed_index=743916,
            figures_dir=occupancy_figures_dir,
            save_format=config.save_format,
            xlim=config.occupancy_params[occupancy_distance_measure]["xlim"],
            bandwidth=config.occupancy_params[occupancy_distance_measure]["bandwidth"],
        )

        # Compute and store occupancy ratios
        occupancy_dict = occupancy.get_kde_occupancy_dict(
            distance_kde_dict=distance_kde_dict,
            channel_map=config.channel_map,
            results_dir=config.results_dir,
            recalculate=config.recalculate["run_occupancy_analysis"],
            suffix=config.suffix,
            distance_measure=occupancy_distance_measure,
            bandwidth=config.occupancy_params[occupancy_distance_measure]["bandwidth"],
            num_points=250,
            x_min=0,
            x_max=config.occupancy_params[occupancy_distance_measure]["xlim"],
        )

        self.shared_data["occupancy_dict"][occupancy_distance_measure] = occupancy_dict

        # Plot individual occupancy ratio
        _ = visualization.plot_occupancy_ratio(
            occupancy_dict=self.shared_data["occupancy_dict"][occupancy_distance_measure],
            channel_map=config.channel_map,
            baseline_mode=config.baseline_mode,
            figures_dir=occupancy_figures_dir,
            suffix=config.suffix,
            normalization=config.normalization,
            distance_measure=occupancy_distance_measure,
            save_format=config.save_format,
            xlim=config.occupancy_params[occupancy_distance_measure]["xlim"],
            ylim=config.occupancy_params[occupancy_distance_measure]["ylim"],
            fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
            plot_individual=config.occupancy_params.get("plot_individual", True),
            show_legend=config.occupancy_params.get("show_legend", True),
        )

    def _run_occupancy_emd_analysis(self, config: AnalysisConfig):
        """Run occupancy EMD analysis."""
        if "occupancy_dict" not in self.shared_data:
            logger.warning("Occupancy data not available. Running occupancy analysis first.")
            self._run_occupancy_analysis(config)

        occupancy_emd_figures_dir = config.figures_dir / "occupancy_emd"
        occupancy_emd_figures_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Running occupancy EMD analysis")

        # Compute occupancy EMD
        self.shared_data["occupancy_emd_df"] = occupancy.get_occupancy_emd_df(
            combined_occupancy_dict=self.shared_data["occupancy_dict"],
            packing_modes=config.packing_modes,
            distance_measures=config.occupancy_distance_measures,
            results_dir=config.results_dir,
            recalculate=config.recalculate["run_occupancy_emd_analysis"],
            suffix=config.suffix,
        )

        # Plot occupancy EMD comparisons
        _ = visualization.plot_emd_comparisons(
            df_emd=self.shared_data["occupancy_emd_df"],
            distance_measures=config.occupancy_distance_measures,
            comparison_type="baseline",
            baseline_mode=config.baseline_mode,
            figures_dir=occupancy_emd_figures_dir,
            suffix=config.suffix,
            save_format=config.save_format,
        )

        occupancy_emd_log_file_path = (
            config.results_dir
            / f"{config.packing_id}_occupancy_emd_central_tendencies{config.suffix}.log"
        )
        distance.log_central_tendencies_for_emd(
            df_emd=self.shared_data["occupancy_emd_df"],
            distance_measures=config.occupancy_distance_measures,
            packing_modes=config.packing_modes,
            baseline_mode=config.baseline_mode,
            log_file_path=occupancy_emd_log_file_path,
            comparison_type="baseline",
        )

    def _run_occupancy_interpolation_analysis(self, config: AnalysisConfig):
        """Run interpolation analysis for occupancy data."""
        if "occupancy_dict" not in self.shared_data:
            logger.warning("Occupancy data not available. Running occupancy analysis first.")
            self._run_occupancy_analysis(config)

        logger.info("Running occupancy interpolation analysis")

        interpolation_figures_dir = config.figures_dir / "interpolation"
        interpolation_figures_dir.mkdir(exist_ok=True, parents=True)

        # Interpolate occupancy ratio and plot
        self.shared_data["interp_occupancy_dict"] = occupancy.interpolate_occupancy_dict(
            occupancy_dict=self.shared_data["occupancy_dict"],
            channel_map=config.channel_map,
            baseline_mode=config.baseline_mode,
            results_dir=config.results_dir,
            suffix=config.suffix,
        )

        for occupancy_distance_measure in config.occupancy_distance_measures:
            # for plot_type in ["individual", "joint"]:
            plot_type = "joint"
            _, ax = visualization.plot_occupancy_ratio(
                occupancy_dict=self.shared_data["occupancy_dict"][occupancy_distance_measure],
                channel_map=config.channel_map,
                baseline_mode=config.baseline_mode,
                suffix=config.suffix,
                normalization=config.normalization,
                distance_measure=occupancy_distance_measure,
                xlim=config.occupancy_params[occupancy_distance_measure]["xlim"],
                ylim=config.occupancy_params[occupancy_distance_measure]["ylim"],
                fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
                plot_individual=True,
                show_legend=config.occupancy_params.get("show_legend", True),
            )
            _ = visualization.add_baseline_occupancy_interpolation_to_plot(
                ax=ax,
                interpolated_occupancy_dict=self.shared_data["interp_occupancy_dict"],
                baseline_mode=config.baseline_mode,
                distance_measure=occupancy_distance_measure,
                figures_dir=interpolation_figures_dir,
                suffix=config.suffix,
                save_format=config.save_format,
                plot_type=plot_type,
            )


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
