"""Unit tests for run_packing_workflow module."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cellpack_analysis.packing.run_packing_workflow import _run_packing_workflow
from cellpack_analysis.packing.workflow_config import WorkflowConfig

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def mock_workflow_config(tmp_path):
    """Create a mock workflow configuration."""
    config = MagicMock(spec=WorkflowConfig)
    config.generate_recipes = True
    config.generate_configs = True
    config.dry_run = False
    config.packing_id = "peroxisome"
    config.condition = "test"
    config.output_path = tmp_path / "mock_output"
    config.output_path.mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary workflow config file for testing."""
    config_data = {
        "packing_id": "peroxisome",
        "structure_id": "SLC25A17",
        "condition": "test",
        "dry_run": True,
        "generate_recipes": True,
        "generate_configs": True,
        "get_counts_from_data": False,
        "get_size_from_data": False,
        "get_bounding_box_from_mesh": False,
        "multiple_replicates": False,
        "num_cells": 2,
        "use_mean_cell": False,
        "use_cells_in_8d_sphere": False,
        "use_additional_struct": False,
        "gradient_structure_name": "peroxisome",
        "num_processes": 1,
        "skip_completed": False,
        "result_type": "image",
        "datadir": str(tmp_path / "data"),
        "recipe_data": {"random": {}},
        "packings_to_run": {"rules": ["random"]},
    }

    config_path = tmp_path / "test_config.json"
    import json

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    return config_path


class TestRunPackingWorkflowUnit:
    """Unit tests for _run_packing_workflow function with mocked dependencies."""

    @patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_configs")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_all_steps_enabled(
        self,
        mock_config_class,
        mock_generate_recipes,
        mock_generate_configs,
        mock_pack_recipes,
        mock_workflow_config,
        temp_config_file,
    ):
        """Test workflow with all steps (generate recipes, configs, and packing) enabled."""
        # Setup
        mock_config_class.return_value = mock_workflow_config
        mock_pack_recipes.return_value = 0

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify all steps were called
        mock_generate_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_generate_configs.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_pack_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        assert result == 0

    @patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_configs")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_skip_recipe_generation(
        self,
        mock_config_class,
        mock_generate_recipes,
        mock_generate_configs,
        mock_pack_recipes,
        mock_workflow_config,
        temp_config_file,
    ):
        """Test workflow when recipe generation is disabled."""
        # Setup
        mock_workflow_config.generate_recipes = False
        mock_config_class.return_value = mock_workflow_config
        mock_pack_recipes.return_value = 0

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify recipe generation was skipped
        mock_generate_recipes.assert_not_called()
        mock_generate_configs.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_pack_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        assert result == 0

    @patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_configs")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_skip_config_generation(
        self,
        mock_config_class,
        mock_generate_recipes,
        mock_generate_configs,
        mock_pack_recipes,
        mock_workflow_config,
        temp_config_file,
    ):
        """Test workflow when config generation is disabled."""
        # Setup
        mock_workflow_config.generate_configs = False
        mock_config_class.return_value = mock_workflow_config
        mock_pack_recipes.return_value = 0

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify config generation was skipped
        mock_generate_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_generate_configs.assert_not_called()
        mock_pack_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        assert result == 0

    @patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_configs")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_dry_run(
        self,
        mock_config_class,
        mock_generate_recipes,
        mock_generate_configs,
        mock_pack_recipes,
        mock_workflow_config,
        temp_config_file,
    ):
        """Test workflow in dry run mode (dry_run is handled inside pack_recipes)."""
        # Setup
        mock_workflow_config.dry_run = True
        mock_config_class.return_value = mock_workflow_config
        mock_pack_recipes.return_value = 0

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify all steps were called (dry_run is handled inside pack_recipes)
        mock_generate_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_generate_configs.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_pack_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        assert result == 0

    @patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_configs")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_skip_all_generation(
        self,
        mock_config_class,
        mock_generate_recipes,
        mock_generate_configs,
        mock_pack_recipes,
        mock_workflow_config,
        temp_config_file,
    ):
        """Test workflow when both recipe and config generation are disabled."""
        # Setup
        mock_workflow_config.generate_recipes = False
        mock_workflow_config.generate_configs = False
        mock_config_class.return_value = mock_workflow_config
        mock_pack_recipes.return_value = 0

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify only packing was called
        mock_generate_recipes.assert_not_called()
        mock_generate_configs.assert_not_called()
        mock_pack_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        assert result == 0

    @patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_configs")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_with_failures(
        self,
        mock_config_class,
        mock_generate_recipes,
        mock_generate_configs,
        mock_pack_recipes,
        mock_workflow_config,
        temp_config_file,
    ):
        """Test workflow returns failure count when packing fails."""
        # Setup
        mock_config_class.return_value = mock_workflow_config
        mock_pack_recipes.return_value = 5  # 5 failed packings

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify failure count is returned
        assert result == 5

    @patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_configs")
    @patch("cellpack_analysis.packing.run_packing_workflow.generate_recipes")
    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_dry_run_skips_only_packing(
        self,
        mock_config_class,
        mock_generate_recipes,
        mock_generate_configs,
        mock_pack_recipes,
        mock_workflow_config,
        temp_config_file,
    ):
        """Test that dry run mode still generates recipes, configs, and calls pack_recipes."""
        # Setup
        mock_workflow_config.dry_run = True
        mock_workflow_config.generate_recipes = True
        mock_workflow_config.generate_configs = True
        mock_config_class.return_value = mock_workflow_config
        mock_pack_recipes.return_value = 0

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify all steps were called (dry_run is handled inside pack_recipes)
        mock_generate_recipes.assert_called_once()
        mock_generate_configs.assert_called_once()
        mock_pack_recipes.assert_called_once()
        assert result == 0

    @patch("cellpack_analysis.packing.run_packing_workflow.WorkflowConfig")
    def test_run_packing_workflow_config_initialization(self, mock_config_class, temp_config_file):
        """Test that WorkflowConfig is properly initialized with the config file path."""
        # Setup
        mock_config = MagicMock()
        mock_config.generate_recipes = False
        mock_config.generate_configs = False
        mock_config.dry_run = True
        mock_config_class.return_value = mock_config

        # Execute
        _run_packing_workflow(temp_config_file)

        # Verify WorkflowConfig was called with the correct path
        mock_config_class.assert_called_once_with(workflow_config_path=temp_config_file)


class TestWorkflowConfig:
    """Comprehensive tests for WorkflowConfig class."""

    def test_workflow_config_loads_from_file(self, temp_config_file):
        """Test that WorkflowConfig can load from a real config file."""
        # Execute
        config = WorkflowConfig(workflow_config_path=temp_config_file)

        # Verify config attributes
        assert config.packing_id == "peroxisome"
        assert config.structure_id == "SLC25A17"
        assert config.condition == "test"
        assert config.dry_run is True
        assert config.generate_recipes is True
        assert config.generate_configs is True
        assert config.num_cells == 2
        assert config.num_processes == 1

    def test_workflow_config_creates_directories(self, temp_config_file, tmp_path):
        """Test that WorkflowConfig creates necessary directories."""
        # Execute
        config = WorkflowConfig(workflow_config_path=temp_config_file)

        # Verify directories were created
        assert config.generated_recipe_path.exists()
        assert config.generated_config_path.exists()

    def test_workflow_config_defaults(self, tmp_path):
        """Test that WorkflowConfig uses default values when keys are missing."""
        # Create minimal config
        minimal_config = {"packing_id": "test_structure"}

        config_path = tmp_path / "minimal_config.json"
        import json

        with open(config_path, "w") as f:
            json.dump(minimal_config, f)

        # Execute
        config = WorkflowConfig(workflow_config_path=config_path)

        # Verify defaults are applied
        assert config.packing_id == "test_structure"
        # Other values should use defaults from default_values module
        assert hasattr(config, "dry_run")
        assert hasattr(config, "generate_recipes")
        assert hasattr(config, "generate_configs")

    def test_invalid_json_config(self, tmp_path):
        """Test that invalid JSON config raises appropriate error."""
        config_path = tmp_path / "invalid.json"
        with open(config_path, "w") as f:
            f.write("{ invalid json }")

        import json

        with pytest.raises(json.JSONDecodeError):
            WorkflowConfig(workflow_config_path=config_path)

    def test_nonexistent_config_file(self, tmp_path):
        """Test that nonexistent config file raises appropriate error."""
        config_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            WorkflowConfig(workflow_config_path=config_path)

    def test_config_with_custom_paths(self, tmp_path):
        """Test workflow configuration with custom template paths."""
        import json

        custom_config_data = {
            "packing_id": "peroxisome",
            "structure_id": "SLC25A17",
            "condition": "custom_paths",
            "datadir": str(tmp_path),
            "recipe_template_path": str(tmp_path / "custom_recipe.json"),
            "config_template_path": str(tmp_path / "custom_config.json"),
            "generated_recipe_path": str(tmp_path / "custom_recipes"),
            "generated_config_path": str(tmp_path / "custom_configs"),
        }

        config_path = tmp_path / "custom_config.json"
        with open(config_path, "w") as f:
            json.dump(custom_config_data, f)

        # Execute
        config = WorkflowConfig(workflow_config_path=config_path)

        # Verify custom paths were used
        assert config.recipe_template_path == Path(custom_config_data["recipe_template_path"])
        assert config.config_template_path == Path(custom_config_data["config_template_path"])
        assert config.generated_recipe_path == Path(custom_config_data["generated_recipe_path"])
        assert config.generated_config_path == Path(custom_config_data["generated_config_path"])


class TestRunPackingWorkflowSimulariumOutput:
    """Integration test: run the packing workflow on real data and verify .simularium output."""

    # Paths to existing recipe and config files used as test inputs.
    RECIPE_DIR = PROJECT_ROOT / "data/recipes/peroxisome/rules_shape_with_seed/random"
    CONFIG_PATH = (
        PROJECT_ROOT
        / "data/configs/peroxisome/rules_shape_with_seed/random/peroxisome_random_config.json"
    )

    def _build_workflow(
        self,
        tmp_path: Path,
        cell_ids: list[str],
        num_processes: int = 1,
    ) -> tuple[Path, Path]:
        """
        Build a workflow config that reuses existing recipe/config files
        but redirects the packing output to a temporary directory.

        Parameters
        ----------
        tmp_path
            Temporary directory provided by pytest
        cell_ids
            Cell IDs whose recipes will be copied into the temp layout
        num_processes
            Number of parallel worker processes for packing

        Returns
        -------
        :
            Tuple of (workflow_config_path, packing_output_dir)
        """
        # --- set up directory layout expected by pack_recipes ---
        generated_recipe_dir = tmp_path / "recipes" / "random"
        generated_recipe_dir.mkdir(parents=True, exist_ok=True)
        generated_config_dir = tmp_path / "configs" / "random"
        generated_config_dir.mkdir(parents=True, exist_ok=True)

        # Copy all requested recipe files
        for cell_id in cell_ids:
            src_recipe = self.RECIPE_DIR / f"peroxisome_random_{cell_id}.json"
            dest_recipe = generated_recipe_dir / f"peroxisome_random_{cell_id}.json"
            shutil.copy(src_recipe, dest_recipe)

        # Copy the real config file and redirect its "out" to a temp location
        packing_output_dir = tmp_path / "packing_output"
        packing_output_dir.mkdir(exist_ok=True)

        with open(self.CONFIG_PATH) as f:
            config_data = json.load(f)
        config_data["out"] = str(packing_output_dir)

        dest_config = generated_config_dir / "peroxisome_random_config.json"
        with open(dest_config, "w") as f:
            json.dump(config_data, f, indent=2)

        # --- build workflow-level config ---
        workflow_data = {
            "packing_id": "peroxisome",
            "structure_id": "SLC25A17",
            "condition": "test_simularium",
            "datadir": str(tmp_path),
            "generate_recipes": False,
            "generate_configs": False,
            "dry_run": False,
            "num_processes": num_processes,
            "skip_completed": False,
            "result_type": "simularium",
            "use_mean_cell": False,
            "use_cells_in_8d_sphere": False,
            "use_additional_struct": False,
            "gradient_structure_name": "peroxisome",
            "generated_recipe_path": str(tmp_path / "recipes"),
            "generated_config_path": str(tmp_path / "configs"),
            "packings_to_run": {
                "rules": ["random"],
                "cell_ids": cell_ids,
            },
        }

        workflow_config_path = tmp_path / "workflow_config.json"
        with open(workflow_config_path, "w") as f:
            json.dump(workflow_data, f, indent=2)

        return workflow_config_path, packing_output_dir

    @pytest.mark.slow
    def test_simularium_output_created(self, tmp_path):
        """Run the full packing workflow on real data and verify a .simularium file is produced.

        Expected output location:
            {config["out"]}/{packing_id}/spheresSST/results_*.simularium

        where config["out"] is the "out" field in the cellPACK config JSON.
        """
        cell_ids = ["743916"]
        workflow_config_path, packing_output_dir = self._build_workflow(
            tmp_path,
            cell_ids=cell_ids,
            num_processes=1,
        )

        # Mock get_cell_id_list_for_structure to avoid loading the cell-id parquet
        with patch(
            "cellpack_analysis.packing.pack_recipes.get_cell_id_list_for_structure"
        ) as mock_cell_ids:
            mock_cell_ids.return_value = cell_ids
            result = _run_packing_workflow(workflow_config_path)

        # The packing should succeed with zero failures
        assert result == 0, f"Packing workflow reported {result} failure(s)"

        # Verify the .simularium output file was created
        simularium_dir = packing_output_dir / "peroxisome" / "spheresSST"
        assert (
            simularium_dir.is_dir()
        ), f"Expected output directory does not exist: {simularium_dir}"

        simularium_files = list(simularium_dir.glob("results_*.simularium"))
        assert len(simularium_files) > 0, (
            f"No .simularium files found in {simularium_dir}. "
            f"Contents: {list(simularium_dir.iterdir())}"
        )

    @pytest.mark.slow
    def test_simularium_output_multiprocess(self, tmp_path):
        """Run the packing workflow with num_processes > 1 and multiple recipes.

        Verifies that parallel execution still produces the expected .simularium
        files for every recipe submitted.
        """
        cell_ids = ["743916", "743920"]
        workflow_config_path, packing_output_dir = self._build_workflow(
            tmp_path,
            cell_ids=cell_ids,
            num_processes=2,
        )

        with patch(
            "cellpack_analysis.packing.pack_recipes.get_cell_id_list_for_structure"
        ) as mock_cell_ids:
            mock_cell_ids.return_value = cell_ids
            result = _run_packing_workflow(workflow_config_path)

        assert result == 0, f"Packing workflow reported {result} failure(s)"

        simularium_dir = packing_output_dir / "peroxisome" / "spheresSST"
        assert (
            simularium_dir.is_dir()
        ), f"Expected output directory does not exist: {simularium_dir}"

        simularium_files = list(simularium_dir.glob("results_*.simularium"))
        assert len(simularium_files) == len(cell_ids), (
            f"Expected {len(cell_ids)} .simularium files but found {len(simularium_files)} "
            f"in {simularium_dir}. Contents: {list(simularium_dir.iterdir())}"
        )
