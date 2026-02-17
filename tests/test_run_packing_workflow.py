"""Unit tests for run_packing_workflow module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cellpack_analysis.packing.run_packing_workflow import _run_packing_workflow
from cellpack_analysis.packing.workflow_config import WorkflowConfig


@pytest.fixture
def mock_workflow_config():
    """Create a mock workflow configuration."""
    config = MagicMock(spec=WorkflowConfig)
    config.generate_recipes = True
    config.generate_configs = True
    config.dry_run = False
    config.structure_name = "peroxisome"
    config.condition = "test"
    return config


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary workflow config file for testing."""
    config_data = {
        "structure_name": "peroxisome",
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
        """Test workflow in dry run mode (no actual packing)."""
        # Setup
        mock_workflow_config.dry_run = True
        mock_config_class.return_value = mock_workflow_config

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify packing was skipped in dry run
        mock_generate_recipes.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_generate_configs.assert_called_once_with(workflow_config=mock_workflow_config)
        mock_pack_recipes.assert_not_called()
        assert result is None

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
        """Test that dry run mode still generates recipes and configs."""
        # Setup
        mock_workflow_config.dry_run = True
        mock_workflow_config.generate_recipes = True
        mock_workflow_config.generate_configs = True
        mock_config_class.return_value = mock_workflow_config

        # Execute
        result = _run_packing_workflow(temp_config_file)

        # Verify generation steps were called but packing was skipped
        mock_generate_recipes.assert_called_once()
        mock_generate_configs.assert_called_once()
        mock_pack_recipes.assert_not_called()
        assert result is None

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
        mock_config_class.assert_called_once_with(config_file_path=temp_config_file)


class TestWorkflowConfig:
    """Comprehensive tests for WorkflowConfig class."""

    def test_workflow_config_loads_from_file(self, temp_config_file):
        """Test that WorkflowConfig can load from a real config file."""
        # Execute
        config = WorkflowConfig(config_file_path=temp_config_file)

        # Verify config attributes
        assert config.structure_name == "peroxisome"
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
        config = WorkflowConfig(config_file_path=temp_config_file)

        # Verify directories were created
        assert config.generated_recipe_path.exists()
        assert config.generated_config_path.exists()

    def test_workflow_config_defaults(self, tmp_path):
        """Test that WorkflowConfig uses default values when keys are missing."""
        # Create minimal config
        minimal_config = {"structure_name": "test_structure"}

        config_path = tmp_path / "minimal_config.json"
        import json

        with open(config_path, "w") as f:
            json.dump(minimal_config, f)

        # Execute
        config = WorkflowConfig(config_file_path=config_path)

        # Verify defaults are applied
        assert config.structure_name == "test_structure"
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
            WorkflowConfig(config_file_path=config_path)

    def test_nonexistent_config_file(self, tmp_path):
        """Test that nonexistent config file raises appropriate error."""
        config_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            WorkflowConfig(config_file_path=config_path)

    def test_config_with_custom_paths(self, tmp_path):
        """Test workflow configuration with custom template paths."""
        import json

        custom_config_data = {
            "structure_name": "peroxisome",
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
        config = WorkflowConfig(config_file_path=config_path)

        # Verify custom paths were used
        assert config.recipe_template_path == Path(custom_config_data["recipe_template_path"])
        assert config.config_template_path == Path(custom_config_data["config_template_path"])
        assert config.generated_recipe_path == Path(custom_config_data["generated_recipe_path"])
        assert config.generated_config_path == Path(custom_config_data["generated_config_path"])
