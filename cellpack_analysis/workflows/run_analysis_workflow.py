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
from typing import Dict

from cellpack_analysis import setup_logging
from cellpack_analysis.lib import distance, occupancy, visualization
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats_functions import normalize_distances

log = logging.getLogger(__name__)


class AnalysisConfig:
    """Configuration class for analysis workflows."""

    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._setup_parameters()
        self._setup_paths()

    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, "r") as f:
            return json.load(f)

    def _setup_paths(self):
        """Setup directory paths."""
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
        """Setup analysis parameters from config."""
        # Structure parameters
        self.structure_id = self.config.get("structure_id", "SLC25A17")
        self.packing_id = self.config.get("packing_id", "peroxisome")
        self.structure_name = self.config.get("structure_name", "peroxisome")

        # Analysis parameters
        self.name = self.config.get("name", "analysis")
        self.channel_map = self.config.get("channel_map", {self.structure_id: self.structure_id})
        self.packing_modes = list(self.channel_map.keys())
        self.packing_output_folder = self.config.get(
            "packing_output_folder", "packing_outputs/8d_sphere_data/rules_shape/"
        )
        self.baseline_mode = self.config.get("baseline_mode", self.structure_id)

        # Distance measures
        self.distance_measures = self.config.get(
            "distance_measures", ["nearest", "pairwise", "nucleus", "z"]
        )

        # Normalization
        self.normalization = self.config.get("normalization", None)
        self.suffix = f"_normalized_{self.normalization}" if self.normalization else ""

        # Visualization
        self.save_format = self.config.get("save_format", "svg")

        # Analysis-specific parameters
        self.ks_significance_level = self.config.get("ks_significance_level", 0.05)
        self.n_bootstrap = self.config.get("n_bootstrap", 1000)
        self.bandwidth = self.config.get("bandwidth", 0.4)

        # Occupancy analysis specific
        self.occupancy_distance_measures = self.config.get(
            "occupancy_distance_measures", ["nucleus", "z"]
        )
        self.xlim = self.config.get("xlim", {"nucleus": 6, "z": 8})

        # Ingredient key for position data
        self.ingredient_key = self.config.get(
            "ingredient_key", f"membrane_interior_{self.structure_name}"
        )

        # Recalculation flags
        self.recalculate = self.config.get("recalculate", False)

        # Analysis steps
        self.analysis_steps = self.config.get("analysis_steps", [])


class AnalysisRunner:
    """Main class for running different types of analysis workflows."""

    def __init__(self):
        self.shared_data = {}

    def run_analysis(self, config_file: str):
        """Run analysis based on the configuration file."""
        config = AnalysisConfig(config_file)

        log.info(f"Starting {config.name} analysis")
        start_time = time.time()

        # Execute each step in the analysis_steps list
        for step in config.analysis_steps:
            self._execute_step(step, config)

        log.info(f"Workflow completed in {time.time() - start_time:.2f} seconds")

    def _execute_step(self, step: str, config: AnalysisConfig):
        """Execute a single analysis step."""
        log.info(f"Executing step: {step}")

        step_method_map = {
            "load_common_data": self._load_common_data,
            "calculate_distances": self._calculate_distances,
            "plot_distance_distributions": self._plot_distance_distributions,
            "run_emd_analysis": self._run_emd_analysis,
            "run_ks_analysis": self._run_ks_analysis,
            "run_occupancy_analysis": self._run_occupancy_analysis,
        }

        if step in step_method_map:
            step_method_map[step](config)
        else:
            log.warning(f"Unknown step: {step}")

    def _load_common_data(self, config: AnalysisConfig):
        """Load position data and mesh information common to all analyses."""
        if (
            "all_positions" in self.shared_data
            and "combined_mesh_information_dict" in self.shared_data
            and not config.recalculate
        ):
            log.info("Using cached position data and mesh information")
            return

        log.info("Loading position data")
        all_positions = get_position_data_from_outputs(
            structure_id=config.structure_id,
            structure_name=config.packing_id,
            packing_modes=config.packing_modes,
            base_datadir=config.base_datadir,
            results_dir=config.results_dir,
            packing_output_folder=config.packing_output_folder,
            ingredient_key=config.ingredient_key,
            recalculate=config.recalculate,
        )

        log.info("Loading mesh information")
        all_structures = list(set(config.channel_map.values()))
        combined_mesh_information_dict = {}
        for structure_id in all_structures:
            mesh_information_dict = get_mesh_information_dict_for_structure(
                structure_id=structure_id,
                base_datadir=config.base_datadir,
                recalculate=config.recalculate,
            )
            combined_mesh_information_dict[structure_id] = mesh_information_dict

        self.shared_data["all_positions"] = all_positions
        self.shared_data["combined_mesh_information_dict"] = combined_mesh_information_dict

    def _calculate_distances(self, config: AnalysisConfig):
        """Calculate distance measures and normalize."""
        if "all_distance_dict" in self.shared_data and not config.recalculate:
            log.info("Using cached distance calculations")
            return

        if (
            "all_positions" not in self.shared_data
            or "combined_mesh_information_dict" not in self.shared_data
        ):
            log.warning("Position data not loaded. Loading common data first.")
            self._load_common_data(config)

        log.info("Calculating distance measures")
        all_distance_dict = distance.get_distances_from_positions(
            all_positions=self.shared_data["all_positions"],
            distance_measures=config.distance_measures,
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            results_dir=config.results_dir,
            recalculate=config.recalculate,
        )

        all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
            distance_distribution_dict=all_distance_dict,
        )

        all_distance_dict = normalize_distances(
            occupied_distance_dict=all_distance_dict,
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            normalization=config.normalization,
        )

        self.shared_data["all_distance_dict"] = all_distance_dict

    def _plot_distance_distributions(self, config: AnalysisConfig):
        """Plot distance distributions."""
        if "all_distance_dict" not in self.shared_data:
            log.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        log.info("Plotting distance distributions")
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
            log.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        log.info("Running EMD analysis")
        emd_figures_dir = config.figures_dir / "emd"
        emd_figures_dir.mkdir(exist_ok=True, parents=True)

        # Get EMD distances
        df_emd = distance.get_distance_distribution_emd_df(
            all_distance_dict=self.shared_data["all_distance_dict"],
            packing_modes=config.packing_modes,
            distance_measures=config.distance_measures,
            results_dir=config.results_dir,
            recalculate=config.recalculate,
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
            log.warning("Distance data not calculated. Calculating distances first.")
            self._calculate_distances(config)

        log.info("Running KS test analysis")
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
            recalculate=config.recalculate,
        )

        # Bootstrap KS tests
        df_ks_bootstrap = distance.bootstrap_ks_tests(
            ks_test_df=ks_test_df,
            distance_measures=config.distance_measures,
            packing_modes=[pm for pm in config.packing_modes if pm != config.baseline_mode],
            n_bootstrap=config.n_bootstrap,
        )

        # Plot KS results
        _ = visualization.plot_ks_test_barplots(
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
            log.warning(
                "Required data not available. Loading common data and calculating distances first."
            )
            self._load_common_data(config)
            self._calculate_distances(config)

        log.info("Running occupancy analysis")

        # Run occupancy analysis for each specified distance measure
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
        log.info(f"Running occupancy analysis for distance measure: {occupancy_distance_measure}")

        occupancy_figures_dir = config.figures_dir / "occupancy"
        occupancy_figures_dir.mkdir(exist_ok=True, parents=True)

        occupancy_distance_figures_dir = occupancy_figures_dir / occupancy_distance_measure
        occupancy_distance_figures_dir.mkdir(exist_ok=True, parents=True)

        # Create KDE dictionary
        distance_kde_dict = distance.get_occupied_and_available_distance_distribution(
            occupied_distance_dict=self.shared_data["all_distance_dict"],
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            save_dir=config.results_dir,
            recalculate=config.recalculate,
            suffix=config.suffix,
            normalization=config.normalization,
            distance_measure=occupancy_distance_measure,
        )

        # Plot illustration for occupancy distribution
        _ = visualization.plot_occupancy_illustration(
            distance_dict=self.shared_data["all_distance_dict"][occupancy_distance_measure],
            occupancy_dict=distance_kde_dict,
            baseline_mode="random",
            suffix=config.suffix,
            distance_measure=occupancy_distance_measure,
            normalization=config.normalization,
            method="pdf",
            seed_index=0,
            xlim=config.xlim.get(occupancy_distance_measure, 6),
            figures_dir=occupancy_distance_figures_dir,
            save_format=config.save_format,
        )

        # Plot individual occupancy ratio
        figs_ind, axs_ind = visualization.plot_occupancy_ratio(
            distance_dict=self.shared_data["all_distance_dict"][occupancy_distance_measure],
            occupancy_dict=distance_kde_dict,
            packing_modes=config.packing_modes,
            suffix=config.suffix,
            method="pdf",
            normalization=config.normalization,
            distance_measure=occupancy_distance_measure,
            xlim=config.xlim.get(occupancy_distance_measure, 6),
            num_cells=5,
            bandwidth=config.bandwidth,
            figures_dir=occupancy_distance_figures_dir,
            save_format=config.save_format,
            num_points=100,
        )

        # Plot mean and std of occupancy ratio
        figs_ci, axs_ci = visualization.plot_mean_and_std_occupancy_ratio_kde(
            distance_dict=self.shared_data["all_distance_dict"][occupancy_distance_measure],
            kde_dict=distance_kde_dict,
            packing_modes=config.packing_modes,
            suffix=config.suffix,
            normalization=config.normalization,
            distance_measure=occupancy_distance_measure,
            method="pdf",
            xlim=config.xlim.get(occupancy_distance_measure, 6),
            bandwidth=config.bandwidth,
            figures_dir=occupancy_distance_figures_dir,
            save_format=config.save_format,
        )

        # Get combined space corrected KDE
        combined_kde_dict = occupancy.get_combined_occupancy_kde(
            all_distance_dict=self.shared_data["all_distance_dict"],
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            packing_modes=config.packing_modes,
            results_dir=config.results_dir,
            recalculate=config.recalculate,
            suffix=config.suffix,
            distance_measure=occupancy_distance_measure,
        )

        # Plot combined space corrected KDE
        fig_combined, ax_combined = visualization.plot_combined_occupancy_ratio(
            combined_kde_dict=combined_kde_dict,
            packing_modes=config.packing_modes,
            suffix=config.suffix,
            normalization=config.normalization,
            aspect=None,
            distance_measure=occupancy_distance_measure,
            num_points=100,
            method="pdf",
            xlim=6,
            figures_dir=occupancy_distance_figures_dir,
            bandwidth=config.bandwidth,
            save_format=config.save_format,
            recalculate=config.recalculate,
        )

        # Plot binned occupancy ratio
        fig_binned, ax_binned = visualization.plot_binned_occupancy_ratio(
            distance_dict=self.shared_data["all_distance_dict"][occupancy_distance_measure],
            packing_modes=config.packing_modes,
            mesh_information_dict=self.shared_data["combined_mesh_information_dict"],
            channel_map=config.channel_map,
            normalization=config.normalization,
            num_bins=40,
            suffix=config.suffix,
            distance_measure=occupancy_distance_measure,
            xlim=config.xlim.get(occupancy_distance_measure, 6),
            figures_dir=occupancy_distance_figures_dir,
            save_format=config.save_format,
        )


def main():
    """Main entry point for command line interface."""
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

    # Setup logging
    if args.log_level == "DEBUG":
        setup_logging(level=logging.DEBUG)
    elif args.log_level == "INFO":
        setup_logging(level=logging.INFO)
    elif args.log_level == "WARNING":
        setup_logging(level=logging.WARNING)
    elif args.log_level == "ERROR":
        setup_logging(level=logging.ERROR)

    # Run analysis
    runner = AnalysisRunner()
    runner.run_analysis(args.config_file)


if __name__ == "__main__":
    main()
