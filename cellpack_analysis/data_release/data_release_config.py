"""Workflow configuration for data release to BFF."""

import json
import logging
from pathlib import Path

from cellpack_analysis.lib.file_io import get_datadir_path, get_results_path

logger = logging.getLogger(__name__)


# Default values
DEFAULT_S3_BUCKET = "cellpack-analysis-data"
DEFAULT_BASE_S3_URL = "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/"
DEFAULT_DATASET = "8d_sphere_data"
DEFAULT_CONDITION = "rules_shape"
DEFAULT_EXPERIMENT = "norm_weights"
DEFAULT_RULES = ["random", "nucleus_gradient", "membrane_gradient", "apical_gradient"]
DEFAULT_BASE_CHANNEL_COLORS = {
    "nucleus": (0.18, 0.32, 0.32),
    "membrane": (0.31, 0.19, 0.31),
}
DEFAULT_MAX_WORKERS = 8


class DataReleaseConfig:
    """Configuration class for data release workflow."""

    def __init__(self, config_file: Path):
        """
        Initialize data release configuration.

        Parameters
        ----------
        config_file
            Path to the configuration JSON file
        """
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

    def _setup_parameters(self):
        """Set workflow parameters from config."""
        # S3 configuration
        self.s3_bucket = self.config.get("s3_bucket", DEFAULT_S3_BUCKET)
        self.base_s3_url = self.config.get("base_s3_url", DEFAULT_BASE_S3_URL)

        # Dataset configuration
        self.dataset = self.config.get("dataset", DEFAULT_DATASET)
        self.condition = self.config.get("condition", DEFAULT_CONDITION)
        self.experiment = self.config.get("experiment", DEFAULT_EXPERIMENT)
        self.rules = self.config.get("rules", DEFAULT_RULES)

        # Structure configuration
        self.structures = self.config.get("structures", [])
        if not self.structures:
            raise ValueError("No structures specified in config")

        # Channel colors
        self.base_channel_colors = self.config.get(
            "base_channel_colors", DEFAULT_BASE_CHANNEL_COLORS
        )

        # Workflow step toggles
        self.upload_meshes_to_s3 = self.config.get("upload_meshes_to_s3", True)
        self.update_simularium_files = self.config.get("update_simularium_files", True)
        self.upload_simularium_to_s3 = self.config.get("upload_simularium_to_s3", True)
        self.generate_thumbnails = self.config.get("generate_thumbnails", True)
        self.create_csv = self.config.get("create_csv", True)
        self.update_csv_stats = self.config.get("update_csv_stats", False)
        self.create_metadata_csv = self.config.get("create_metadata_csv", True)
        self.upload_csv_to_s3 = self.config.get("upload_csv_to_s3", True)
        self.upload_metadata_csv = self.config.get("upload_metadata_csv", True)

        # Mesh upload configuration
        self.use_inverted_meshes = self.config.get("use_inverted_meshes", False)
        self.reinvert_meshes = self.config.get("reinvert_meshes", False)

        # Processing configuration
        self.reupload_simularium_files = self.config.get("reupload_simularium_files", False)
        self.reupload_thumbnails = self.config.get("reupload_thumbnails", False)
        self.reupload_csv_files = self.config.get("reupload_csv_files", False)
        self.max_workers = self.config.get("max_workers", DEFAULT_MAX_WORKERS)

        # Output name
        self.output_name = self.config.get("output_name", "cellpack_simularium")

    def _setup_paths(self):
        """Set directory paths."""
        self.base_datadir = get_datadir_path()
        self.base_results_dir = get_results_path()

        # Output directory for CSV files
        self.csv_output_dir = self.base_results_dir / "data_release" / self.output_name
        self.csv_output_dir.mkdir(parents=True, exist_ok=True)

        # Thumbnail directory
        self.thumbnail_dir = self.csv_output_dir / "thumbnails"
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def get_structure_mesh_url(self, structure_id: str) -> str:
        """
        Get the S3 URL for structure meshes.

        Parameters
        ----------
        structure_id
            Structure identifier

        Returns
        -------
        str
            S3 URL for structure meshes
        """
        return f"{self.base_s3_url}structure_data/{structure_id}/meshes/"

    def get_channel_colors(self, structure_color: tuple[float, float, float]) -> dict:
        """
        Get channel colors for a structure.

        Parameters
        ----------
        structure_color
            RGB color tuple for the structure

        Returns
        -------
        dict
            Dictionary mapping channel names to RGB colors
        """
        return {**self.base_channel_colors, "structure": structure_color}

    def __repr__(self) -> str:
        """String representation of the config."""
        return (
            f"DataReleaseConfig(dataset={self.dataset}, "
            f"experiment={self.experiment}, "
            f"structures={len(self.structures)})"
        )
