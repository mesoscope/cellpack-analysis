"""Tests for pack_recipes module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cellpack_analysis.packing.pack_recipes import (
    check_recipe_completed,
    get_cell_ids_to_pack,
    get_input_file_dictionary,
    pack_recipes,
    run_single_packing,
)


@pytest.fixture
def mock_workflow_config():
    """Create a mock workflow configuration."""
    config = MagicMock()
    config.structure_name = "peroxisome"
    config.structure_id = "SLC25A17"
    config.condition = "test"
    config.use_mean_cell = False
    config.use_cells_in_8d_sphere = False
    config.num_processes = 2
    config.skip_completed = False
    config.result_type = "image"
    config.generated_config_path = Path("/test/configs/peroxisome/test")
    config.generated_recipe_path = Path("/test/recipes/peroxisome/test")
    config.output_path = Path("/test/outputs")
    config.data = {
        "packings_to_run": {
            "rules": ["random"],
            "cell_ids": ["743916", "888888"],
            "number_of_packings": None,
        }
    }
    return config


@pytest.fixture
def sample_recipe_data():
    """Sample recipe data for testing."""
    return {
        "name": "peroxisome",
        "version": "random_743916",
        "randomness_seed": 743916,
        "bounding_box": [[-50, -50, -50], [50, 50, 50]],
        "objects": {"peroxisome": {"radius": 2.0}},
    }


@pytest.fixture
def sample_config_data():
    """Sample config data for testing."""
    return {"name": "test", "out": "/test/outputs/random", "number_of_packings": 1}


class TestCheckRecipeCompleted:
    """Tests for check_recipe_completed function."""

    def test_check_recipe_completed_image_single_seed_exists(
        self, tmp_path, sample_recipe_data, sample_config_data, mock_workflow_config
    ):
        """Test checking completed recipe with image output type - file exists."""
        # Setup
        recipe_path = tmp_path / "recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(sample_recipe_data, f)

        output_dir = tmp_path / "peroxisome" / "spheresSST" / "figures"
        output_dir.mkdir(parents=True)

        output_file = (
            output_dir / "voxelized_image_peroxisome_test_random_743916_seed_743916.ome.tiff"
        )
        output_file.touch()

        sample_config_data["out"] = str(tmp_path)
        mock_workflow_config.result_type = "image"

        # Execute
        result = check_recipe_completed(recipe_path, sample_config_data, mock_workflow_config)

        # Verify
        assert result is True

    def test_check_recipe_completed_image_single_seed_missing(
        self, tmp_path, sample_recipe_data, sample_config_data, mock_workflow_config
    ):
        """Test checking completed recipe with image output type - file missing."""
        # Setup
        recipe_path = tmp_path / "recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(sample_recipe_data, f)

        sample_config_data["out"] = str(tmp_path)
        mock_workflow_config.result_type = "image"

        # Execute (no output files created)
        result = check_recipe_completed(recipe_path, sample_config_data, mock_workflow_config)

        # Verify
        assert result is False

    def test_check_recipe_completed_simularium_type(
        self, tmp_path, sample_recipe_data, sample_config_data, mock_workflow_config
    ):
        """Test checking completed recipe with simularium output type."""
        # Setup
        recipe_path = tmp_path / "recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(sample_recipe_data, f)

        output_dir = tmp_path / "peroxisome" / "spheresSST"
        output_dir.mkdir(parents=True)

        output_file = output_dir / "results_peroxisome_test_random_743916_seed_743916.simularium"
        output_file.touch()

        sample_config_data["out"] = str(tmp_path)
        mock_workflow_config.result_type = "simularium"

        # Execute
        result = check_recipe_completed(recipe_path, sample_config_data, mock_workflow_config)

        # Verify
        assert result is True

    def test_check_recipe_completed_multiple_seeds(
        self, tmp_path, sample_recipe_data, sample_config_data, mock_workflow_config
    ):
        """Test checking completed recipe with multiple random seeds."""
        # Setup
        sample_recipe_data["randomness_seed"] = [743916, 888888, 999999]
        recipe_path = tmp_path / "recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(sample_recipe_data, f)

        output_dir = tmp_path / "peroxisome" / "spheresSST" / "figures"
        output_dir.mkdir(parents=True)

        sample_config_data["out"] = str(tmp_path)
        sample_config_data["number_of_packings"] = 3
        mock_workflow_config.result_type = "image"

        # Create only 2 of 3 required files
        for seed in [743916, 888888]:
            output_file = (
                output_dir / f"voxelized_image_peroxisome_test_random_743916_seed_{seed}.ome.tiff"
            )
            output_file.touch()

        # Execute
        result = check_recipe_completed(recipe_path, sample_config_data, mock_workflow_config)

        # Verify - should be False because 2 < 3
        assert result is False

    def test_check_recipe_completed_multiple_seeds_all_exist(
        self, tmp_path, sample_recipe_data, sample_config_data, mock_workflow_config
    ):
        """Test checking completed recipe with all multiple seeds present."""
        # Setup
        sample_recipe_data["randomness_seed"] = [743916, 888888]
        recipe_path = tmp_path / "recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(sample_recipe_data, f)

        output_dir = tmp_path / "peroxisome" / "spheresSST" / "figures"
        output_dir.mkdir(parents=True)

        sample_config_data["out"] = str(tmp_path)
        sample_config_data["number_of_packings"] = 2
        mock_workflow_config.result_type = "image"

        # Create all required files
        for seed in [743916, 888888]:
            output_file = (
                output_dir / f"voxelized_image_peroxisome_test_random_743916_seed_{seed}.ome.tiff"
            )
            output_file.touch()

        # Execute
        result = check_recipe_completed(recipe_path, sample_config_data, mock_workflow_config)

        # Verify
        assert result is True

    def test_check_recipe_completed_invalid_result_type(
        self, tmp_path, sample_recipe_data, sample_config_data, mock_workflow_config
    ):
        """Test that invalid result type raises ValueError."""
        # Setup
        recipe_path = tmp_path / "recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(sample_recipe_data, f)

        sample_config_data["out"] = str(tmp_path)
        mock_workflow_config.result_type = "invalid_type"

        # Execute & Verify
        with pytest.raises(ValueError, match="check_type must be 'image' or 'simularium'"):
            check_recipe_completed(recipe_path, sample_config_data, mock_workflow_config)


class TestGetCellIdsToPack:
    """Tests for get_cell_ids_to_pack function."""

    def test_get_cell_ids_mean_cell(self, mock_workflow_config):
        """Test getting cell IDs when use_mean_cell is True."""
        # Setup
        mock_workflow_config.use_mean_cell = True

        # Execute
        result = get_cell_ids_to_pack(mock_workflow_config)

        # Verify
        assert result == ["mean"]

    @patch("cellpack_analysis.packing.pack_recipes.get_cell_id_list_for_structure")
    def test_get_cell_ids_from_structure(self, mock_get_cell_ids, mock_workflow_config):
        """Test getting cell IDs from structure data."""
        # Setup
        mock_get_cell_ids.return_value = ["743916", "888888", "999999"]
        mock_workflow_config.use_mean_cell = False
        # Remove cell_ids constraint to get all IDs
        mock_workflow_config.data["packings_to_run"].pop("cell_ids", None)

        # Execute
        result = get_cell_ids_to_pack(mock_workflow_config)

        # Verify
        assert result == ["743916", "888888", "999999"]
        mock_get_cell_ids.assert_called_once_with(
            structure_id=mock_workflow_config.structure_id,
            df_cell_id=None,
            dsphere=mock_workflow_config.use_cells_in_8d_sphere,
            load_local=True,
        )

    @patch("cellpack_analysis.packing.pack_recipes.get_cell_id_list_for_structure")
    def test_get_cell_ids_with_specified_ids(self, mock_get_cell_ids, mock_workflow_config):
        """Test getting cell IDs when specific IDs are provided in config."""
        # Setup
        mock_get_cell_ids.return_value = ["743916", "888888", "999999", "111111"]
        mock_workflow_config.use_mean_cell = False
        mock_workflow_config.data["packings_to_run"]["cell_ids"] = ["743916", "888888"]

        # Execute
        result = get_cell_ids_to_pack(mock_workflow_config)

        # Verify
        assert result == ["743916", "888888"]

    @patch("cellpack_analysis.packing.pack_recipes.get_cell_id_list_for_structure")
    def test_get_cell_ids_with_number_limit(self, mock_get_cell_ids, mock_workflow_config):
        """Test getting limited number of cell IDs."""
        # Setup
        mock_get_cell_ids.return_value = ["743916", "888888", "999999", "111111"]
        mock_workflow_config.use_mean_cell = False
        mock_workflow_config.data["packings_to_run"]["cell_ids"] = [
            "743916",
            "888888",
            "999999",
            "111111",
        ]
        mock_workflow_config.data["packings_to_run"]["number_of_packings"] = 2

        # Execute
        result = get_cell_ids_to_pack(mock_workflow_config)

        # Verify
        assert result == ["743916", "888888"]
        assert len(result) == 2


class TestGetInputFileDictionary:
    """Tests for get_input_file_dictionary function."""

    @patch("cellpack_analysis.packing.pack_recipes.get_cell_ids_to_pack")
    def test_get_input_file_dictionary_single_rule(
        self, mock_get_cell_ids, mock_workflow_config, tmp_path
    ):
        """Test getting input file dictionary for a single rule."""
        # Setup
        mock_get_cell_ids.return_value = ["743916", "888888"]
        mock_workflow_config.generated_config_path = tmp_path / "configs"
        mock_workflow_config.generated_recipe_path = tmp_path / "recipes"

        # Create recipe files
        recipe_dir = tmp_path / "recipes" / "random"
        recipe_dir.mkdir(parents=True)

        for cell_id in ["743916", "888888"]:
            recipe_file = recipe_dir / f"peroxisome_random_{cell_id}.json"
            recipe_file.touch()

        # Execute
        result = get_input_file_dictionary(mock_workflow_config)

        # Verify
        assert "random" in result
        assert "config_path" in result["random"]
        assert "recipe_paths" in result["random"]
        assert len(result["random"]["recipe_paths"]) == 2
        assert str(result["random"]["config_path"]).endswith("peroxisome_random_config.json")

    @patch("cellpack_analysis.packing.pack_recipes.get_cell_ids_to_pack")
    def test_get_input_file_dictionary_multiple_rules(
        self, mock_get_cell_ids, mock_workflow_config, tmp_path
    ):
        """Test getting input file dictionary for multiple rules."""
        # Setup
        mock_get_cell_ids.return_value = ["743916"]
        mock_workflow_config.generated_config_path = tmp_path / "configs"
        mock_workflow_config.generated_recipe_path = tmp_path / "recipes"
        mock_workflow_config.data["packings_to_run"]["rules"] = ["random", "gradient"]

        # Create recipe files for both rules
        for rule in ["random", "gradient"]:
            recipe_dir = tmp_path / "recipes" / rule
            recipe_dir.mkdir(parents=True)
            recipe_file = recipe_dir / "peroxisome_random_743916.json"
            recipe_file.touch()

        # Execute
        result = get_input_file_dictionary(mock_workflow_config)

        # Verify
        assert len(result) == 2
        assert "random" in result
        assert "gradient" in result

    @patch("cellpack_analysis.packing.pack_recipes.get_cell_ids_to_pack")
    def test_get_input_file_dictionary_missing_recipes(
        self, mock_get_cell_ids, mock_workflow_config, tmp_path
    ):
        """Test that missing recipe files are skipped."""
        # Setup
        mock_get_cell_ids.return_value = ["743916", "888888", "999999"]
        mock_workflow_config.generated_config_path = tmp_path / "configs"
        mock_workflow_config.generated_recipe_path = tmp_path / "recipes"

        # Create recipe files for only some cell IDs
        recipe_dir = tmp_path / "recipes" / "random"
        recipe_dir.mkdir(parents=True)
        recipe_file = recipe_dir / "peroxisome_random_743916.json"
        recipe_file.touch()
        # Don't create files for 888888 and 999999

        # Execute
        result = get_input_file_dictionary(mock_workflow_config)

        # Verify - should only include existing recipe
        assert len(result["random"]["recipe_paths"]) == 1


class TestRunSinglePacking:
    """Tests for run_single_packing function."""

    @patch("cellpack_analysis.packing.pack_recipes.subprocess.run")
    def test_run_single_packing_success(self, mock_subprocess):
        """Test successful single packing execution."""
        # Setup
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        recipe_path = "/test/recipe.json"
        config_path = "/test/config.json"

        # Execute
        result = run_single_packing(recipe_path, config_path)

        # Verify
        assert result is True
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "python" in args
        assert "-r" in args
        assert recipe_path in args
        assert "-c" in args
        assert config_path in args

    @patch("cellpack_analysis.packing.pack_recipes.subprocess.run")
    def test_run_single_packing_failure(self, mock_subprocess):
        """Test failed single packing execution."""
        # Setup
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")

        recipe_path = "/test/recipe.json"
        config_path = "/test/config.json"

        # Execute
        result = run_single_packing(recipe_path, config_path)

        # Verify
        assert result is False

    @patch("cellpack_analysis.packing.pack_recipes.subprocess.run")
    def test_run_single_packing_exception(self, mock_subprocess):
        """Test single packing with unexpected exception."""
        # Setup
        mock_subprocess.side_effect = Exception("Unexpected error")

        recipe_path = "/test/recipe.json"
        config_path = "/test/config.json"

        # Execute
        result = run_single_packing(recipe_path, config_path)

        # Verify
        assert result is False


class TestPackRecipes:
    """Tests for pack_recipes function."""

    @patch("cellpack_analysis.packing.pack_recipes.ProcessPoolExecutor")
    @patch("cellpack_analysis.packing.pack_recipes.get_input_file_dictionary")
    def test_pack_recipes_all_success(
        self, mock_get_input, mock_executor_class, mock_workflow_config, tmp_path
    ):
        """Test pack_recipes with all successful packings."""
        # Setup
        mock_workflow_config.output_path = tmp_path / "outputs"
        mock_workflow_config.num_processes = 1

        mock_get_input.return_value = {
            "random": {
                "config_path": "/test/config.json",
                "recipe_paths": ["/test/recipe1.json", "/test/recipe2.json"],
            }
        }

        # Mock the executor and futures
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        mock_future1 = MagicMock()
        mock_future1.result.return_value = True
        mock_future2 = MagicMock()
        mock_future2.result.return_value = True

        mock_executor.submit.side_effect = [mock_future1, mock_future2]

        with patch("cellpack_analysis.packing.pack_recipes.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2]

            with patch("cellpack_analysis.packing.pack_recipes.read_json") as mock_read:
                mock_read.return_value = {"name": "test", "out": "/test/out"}

                # Execute
                result = pack_recipes(mock_workflow_config)

        # Verify
        assert result == 0  # No failures
        assert mock_executor.submit.call_count == 2

    @patch("cellpack_analysis.packing.pack_recipes.ProcessPoolExecutor")
    @patch("cellpack_analysis.packing.pack_recipes.get_input_file_dictionary")
    def test_pack_recipes_with_failures(
        self, mock_get_input, mock_executor_class, mock_workflow_config, tmp_path
    ):
        """Test pack_recipes with some failed packings."""
        # Setup
        mock_workflow_config.output_path = tmp_path / "outputs"
        mock_workflow_config.num_processes = 1

        mock_get_input.return_value = {
            "random": {
                "config_path": "/test/config.json",
                "recipe_paths": ["/test/recipe1.json", "/test/recipe2.json", "/test/recipe3.json"],
            }
        }

        # Mock the executor and futures - first two succeed, third fails
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        mock_future1 = MagicMock()
        mock_future1.result.return_value = True
        mock_future2 = MagicMock()
        mock_future2.result.return_value = True
        mock_future3 = MagicMock()
        mock_future3.result.return_value = False

        mock_executor.submit.side_effect = [mock_future1, mock_future2, mock_future3]

        with patch("cellpack_analysis.packing.pack_recipes.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2, mock_future3]

            with patch("cellpack_analysis.packing.pack_recipes.read_json") as mock_read:
                mock_read.return_value = {"name": "test", "out": "/test/out"}

                # Execute
                result = pack_recipes(mock_workflow_config)

        # Verify
        assert result == 1  # One failure
        assert mock_executor.submit.call_count == 3

    @patch("cellpack_analysis.packing.pack_recipes.check_recipe_completed")
    @patch("cellpack_analysis.packing.pack_recipes.ProcessPoolExecutor")
    @patch("cellpack_analysis.packing.pack_recipes.get_input_file_dictionary")
    def test_pack_recipes_skip_completed(
        self,
        mock_get_input,
        mock_executor_class,
        mock_check_completed,
        mock_workflow_config,
        tmp_path,
    ):
        """Test pack_recipes with skip_completed enabled."""
        # Setup
        mock_workflow_config.output_path = tmp_path / "outputs"
        mock_workflow_config.skip_completed = True
        mock_workflow_config.num_processes = 1

        mock_get_input.return_value = {
            "random": {
                "config_path": "/test/config.json",
                "recipe_paths": ["/test/recipe1.json", "/test/recipe2.json", "/test/recipe3.json"],
            }
        }
        # First recipe is completed, others are not
        mock_check_completed.side_effect = [True, False, False]

        # Mock the executor and futures
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        mock_future1 = MagicMock()
        mock_future1.result.return_value = True
        mock_future2 = MagicMock()
        mock_future2.result.return_value = True

        mock_executor.submit.side_effect = [mock_future1, mock_future2]

        with patch("cellpack_analysis.packing.pack_recipes.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2]

            with patch("cellpack_analysis.packing.pack_recipes.read_json") as mock_read:
                mock_read.return_value = {"name": "test", "out": "/test/out"}

                # Execute
                result = pack_recipes(mock_workflow_config)

        # Verify
        assert result == 0  # No failures
        # Should only submit 2 recipes (1 was skipped)
        assert mock_executor.submit.call_count == 2

    @patch("cellpack_analysis.packing.pack_recipes.ProcessPoolExecutor")
    @patch("cellpack_analysis.packing.pack_recipes.get_input_file_dictionary")
    def test_pack_recipes_multiple_rules(
        self, mock_get_input, mock_executor_class, mock_workflow_config, tmp_path
    ):
        """Test pack_recipes with multiple rules."""
        # Setup
        mock_workflow_config.output_path = tmp_path / "outputs"
        mock_workflow_config.num_processes = 1

        mock_get_input.return_value = {
            "random": {
                "config_path": "/test/random_config.json",
                "recipe_paths": ["/test/recipe1.json"],
            },
            "gradient": {
                "config_path": "/test/gradient_config.json",
                "recipe_paths": ["/test/recipe2.json", "/test/recipe3.json"],
            },
        }

        # Mock the executor and futures
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        mock_future1 = MagicMock()
        mock_future1.result.return_value = True
        mock_future2 = MagicMock()
        mock_future2.result.return_value = True
        mock_future3 = MagicMock()
        mock_future3.result.return_value = True

        mock_executor.submit.side_effect = [mock_future1, mock_future2, mock_future3]

        with patch("cellpack_analysis.packing.pack_recipes.as_completed") as mock_as_completed:
            # Return futures for both rules
            mock_as_completed.side_effect = [[mock_future1], [mock_future2, mock_future3]]

            with patch("cellpack_analysis.packing.pack_recipes.read_json") as mock_read:
                mock_read.return_value = {"name": "test", "out": "/test/out"}

                # Execute
                result = pack_recipes(mock_workflow_config)

        # Verify
        assert result == 0  # No failures
        assert mock_executor.submit.call_count == 3  # 1 + 2 recipes

    def test_pack_recipes_creates_log_folder(self, mock_workflow_config, tmp_path):
        """Test that pack_recipes creates log folder."""
        # Setup
        mock_workflow_config.output_path = tmp_path / "outputs"

        with patch(
            "cellpack_analysis.packing.pack_recipes.get_input_file_dictionary"
        ) as mock_get_input:
            mock_get_input.return_value = {}  # No rules to run

            # Execute
            pack_recipes(mock_workflow_config)

        # Verify log folder was created
        assert (tmp_path / "outputs" / "logs").exists()
