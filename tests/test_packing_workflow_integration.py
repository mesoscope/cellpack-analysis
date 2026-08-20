"""Integration tests for the packing workflow.

These tests verify end-to-end functionality with real file I/O and dependencies.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cellpack_analysis.lib.file_io import get_datadir_path, get_project_root
from cellpack_analysis.packing.run_packing_workflow import _run_packing_workflow
from cellpack_analysis.packing.workflow_config import WorkflowConfig

# Cell whose meshes/grid/stats are expected to be available locally for the
# real-data packing test below.
REAL_PACKING_CELL_ID = "743916"


def _real_data_assets(cell_id: str = REAL_PACKING_CELL_ID) -> dict[str, Path]:
    """Resolve the local assets a real (non-dry) peroxisome packing needs."""
    datadir = get_datadir_path()
    sd = datadir / "structure_data" / "SLC25A17"
    return {
        "recipe_template": datadir
        / "templates/recipes/rules_shape/peroxisome_recipe_template.json",
        "config_template": datadir
        / "templates/configs/rules_shape/peroxisome_config_template.json",
        "membrane_mesh": sd / "meshes" / f"mem_mesh_{cell_id}.obj",
        "nucleus_mesh": sd / "meshes" / f"nuc_mesh_{cell_id}.obj",
        "grid": sd / "grids" / f"{cell_id}_grid.dat",
        "stats": datadir / "structure_stats.parquet",
    }


def _missing_real_data_assets() -> list[str]:
    """Return the names of any local assets missing for the real packing test."""
    return [name for name, path in _real_data_assets().items() if not path.exists()]


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return get_project_root() / "tests/test_data"


@pytest.fixture
def integration_test_dir(tmp_path):
    """Create a temporary directory for integration tests."""
    test_dir = tmp_path / "integration_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def integration_config(integration_test_dir, test_data_dir):
    """Create an integration test configuration."""
    config_data = {
        "packing_id": "peroxisome",
        "structure_id": "SLC25A17",
        "condition": "test_integration",
        "datadir": str(integration_test_dir),
        "generate_recipes": True,
        "generate_configs": True,
        "use_mean_cell": True,
        "use_cells_in_8d_sphere": False,
        "get_counts_from_data": False,
        "get_size_from_data": False,
        "get_bounding_box_from_mesh": False,
        "num_processes": 1,
        "skip_completed": False,
        "dry_run": True,
        "multiple_replicates": False,
        "result_type": "image",
        "use_additional_struct": False,
        "gradient_structure_name": "peroxisome",
        "recipe_template_path": str(
            test_data_dir / "templates" / "peroxisome_recipe_template_test.json"
        ),
        "config_template_path": str(
            test_data_dir / "templates" / "peroxisome_config_template_test.json"
        ),
        "recipe_data": {"random": {}},
        "packings_to_run": {"rules": ["random"]},
    }

    config_path = integration_test_dir / "integration_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    return config_path


@pytest.fixture
def mock_cell_id_data():
    """Mock cell ID data retrieval functions."""
    with patch(
        "cellpack_analysis.packing.generate_cellpack_input_files.sample_cell_ids_for_structure"
    ) as mock_sample:
        mock_sample.return_value = ["mean"]
        yield mock_sample


@pytest.fixture
def mock_stats_dataframe():
    """Mock the structure stats dataframe."""
    import pandas as pd

    mock_df = pd.DataFrame({"CellId": ["mean"], "count": [10], "radius": [2.5]})

    with patch(
        "cellpack_analysis.packing.generate_cellpack_input_files.get_structure_stats_dataframe"
    ) as mock_get_stats:
        mock_get_stats.return_value = mock_df
        yield mock_get_stats


class TestPackingWorkflowIntegration:
    """Integration tests for the complete packing workflow."""

    def test_workflow_creates_directory_structure(self, integration_config, integration_test_dir):
        """Test that workflow creates necessary directory structure."""
        # Execute workflow config initialization
        config = WorkflowConfig(workflow_config_path=integration_config)

        # Verify directories were created
        assert config.generated_recipe_path.exists()
        assert config.generated_config_path.exists()
        assert (
            config.generated_recipe_path.parent == integration_test_dir / "recipes" / "peroxisome"
        )
        assert (
            config.generated_config_path.parent == integration_test_dir / "configs" / "peroxisome"
        )

    def test_workflow_dry_run_no_packing(
        self, integration_config, integration_test_dir, mock_cell_id_data, mock_stats_dataframe
    ):
        """Test that dry run mode generates files but doesn't run packing."""
        # Execute
        result = _run_packing_workflow(integration_config)

        # Verify dry run returns 0 (pack_recipes is called but skips actual packing)
        assert result == 0

        # Verify recipe files were generated
        recipe_dir = integration_test_dir / "recipes" / "peroxisome" / "test_integration" / "random"
        assert recipe_dir.exists(), f"Recipe directory not found: {recipe_dir}"

        recipe_files = list(recipe_dir.glob("*.json"))
        assert len(recipe_files) > 0, "No recipe files were generated"

        # Verify config files were generated
        config_dir = integration_test_dir / "configs" / "peroxisome" / "test_integration" / "random"
        assert config_dir.exists(), f"Config directory not found: {config_dir}"

        config_files = list(config_dir.glob("*.json"))
        assert len(config_files) > 0, "No config files were generated"

    def test_workflow_generated_recipe_content(
        self, integration_config, integration_test_dir, mock_cell_id_data, mock_stats_dataframe
    ):
        """Test that generated recipe files have correct content."""
        # Execute
        _run_packing_workflow(integration_config)

        # Find generated recipe file
        recipe_dir = integration_test_dir / "recipes" / "peroxisome" / "test_integration" / "random"
        recipe_files = list(recipe_dir.glob("*.json"))
        assert len(recipe_files) > 0

        # Read and verify recipe content
        with open(recipe_files[0]) as f:
            recipe = json.load(f)

        # Verify key fields were updated
        assert "version" in recipe
        assert "random" in recipe["version"]  # Should include rule name
        assert "bounding_box" in recipe
        assert "objects" in recipe
        assert "peroxisome" in recipe["objects"]
        assert "membrane_mesh" in recipe["objects"]
        assert "nucleus_mesh" in recipe["objects"]

    def test_workflow_generated_config_content(
        self, integration_config, integration_test_dir, mock_cell_id_data, mock_stats_dataframe
    ):
        """Test that generated config files have correct content."""
        # Execute
        _run_packing_workflow(integration_config)

        # Find generated config file
        config_dir = integration_test_dir / "configs" / "peroxisome" / "test_integration" / "random"
        config_files = list(config_dir.glob("*.json"))
        assert len(config_files) > 0

        # Read and verify config content
        with open(config_files[0]) as f:
            config = json.load(f)

        # Verify key fields
        assert "name" in config
        assert config["name"] == "test_integration"
        assert "out" in config
        assert "image_export_options" in config

    def test_workflow_skip_recipe_generation(
        self, integration_config, integration_test_dir, mock_cell_id_data, mock_stats_dataframe
    ):
        """Test workflow with recipe generation disabled."""
        # Modify config to skip recipe generation
        with open(integration_config) as f:
            config_data = json.load(f)
        config_data["generate_recipes"] = False
        with open(integration_config, "w") as f:
            json.dump(config_data, f)

        # Execute
        _run_packing_workflow(integration_config)

        # Verify recipe files were NOT generated
        recipe_dir = integration_test_dir / "recipes" / "peroxisome" / "test_integration" / "random"
        if recipe_dir.exists():
            recipe_files = list(recipe_dir.glob("*.json"))
            assert len(recipe_files) == 0, "Recipe files should not be generated"

        # Verify config files WERE generated
        config_dir = integration_test_dir / "configs" / "peroxisome" / "test_integration" / "random"
        assert config_dir.exists()
        config_files = list(config_dir.glob("*.json"))
        assert len(config_files) > 0

    def test_workflow_skip_config_generation(
        self, integration_config, integration_test_dir, mock_cell_id_data, mock_stats_dataframe
    ):
        """Test workflow with config generation disabled."""
        # Modify config to skip config generation
        with open(integration_config) as f:
            config_data = json.load(f)
        config_data["generate_configs"] = False
        with open(integration_config, "w") as f:
            json.dump(config_data, f)

        # Mock pack_recipes since the config file won't exist when config generation is skipped
        with patch("cellpack_analysis.packing.run_packing_workflow.pack_recipes") as mock_pack:
            mock_pack.return_value = 0

            # Execute
            _run_packing_workflow(integration_config)

        # Verify recipe files WERE generated
        recipe_dir = integration_test_dir / "recipes" / "peroxisome" / "test_integration" / "random"
        assert recipe_dir.exists()
        recipe_files = list(recipe_dir.glob("*.json"))
        assert len(recipe_files) > 0

        # Verify config files were NOT generated
        config_dir = integration_test_dir / "configs" / "peroxisome" / "test_integration" / "random"
        if config_dir.exists():
            config_files = list(config_dir.glob("*.json"))
            assert len(config_files) == 0, "Config files should not be generated"

    def test_workflow_with_multiple_rules(
        self, integration_config, integration_test_dir, mock_cell_id_data, mock_stats_dataframe
    ):
        """Test workflow with multiple packing rules."""
        # Modify config to include multiple rules
        with open(integration_config) as f:
            config_data = json.load(f)
        config_data["recipe_data"] = {"random": {}, "test_rule": {"gradients": ["test_gradient"]}}
        config_data["packings_to_run"]["rules"] = ["random", "test_rule"]
        with open(integration_config, "w") as f:
            json.dump(config_data, f)

        # Mock gradient processing
        with patch(
            "cellpack_analysis.packing.generate_cellpack_input_files.process_gradient_data"
        ) as mock_gradient:
            mock_gradient.side_effect = lambda entry, recipe, struct: recipe

            # Execute
            _run_packing_workflow(integration_config)

        # Verify files were generated for both rules
        for rule in ["random", "test_rule"]:
            recipe_dir = integration_test_dir / "recipes" / "peroxisome" / "test_integration" / rule
            assert recipe_dir.exists(), f"Recipe directory not found for rule: {rule}"

            config_dir = integration_test_dir / "configs" / "peroxisome" / "test_integration" / rule
            assert config_dir.exists(), f"Config directory not found for rule: {rule}"

    def test_workflow_error_handling_missing_template(self, integration_test_dir):
        """Test that workflow handles missing template files gracefully."""
        config_data = {
            "packing_id": "peroxisome",
            "condition": "missing_template",
            "datadir": str(integration_test_dir),
            "recipe_template_path": str(integration_test_dir / "nonexistent_recipe.json"),
            "config_template_path": str(integration_test_dir / "nonexistent_config.json"),
            "generate_recipes": True,
            "generate_configs": True,
            "dry_run": True,
            "recipe_data": {"random": {}},
            "packings_to_run": {"rules": ["random"]},
        }

        config_path = integration_test_dir / "error_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Verify that attempting to run workflow raises appropriate error
        with pytest.raises(FileNotFoundError):
            _run_packing_workflow(config_path)

    def test_workflow_idempotency(
        self, integration_config, integration_test_dir, mock_cell_id_data, mock_stats_dataframe
    ):
        """Test that running workflow multiple times is idempotent."""
        # First run
        _run_packing_workflow(integration_config)

        # Get file count after first run
        recipe_dir = integration_test_dir / "recipes" / "peroxisome" / "test_integration" / "random"
        first_run_files = list(recipe_dir.glob("*.json"))

        # Second run
        _run_packing_workflow(integration_config)

        # Verify file count is the same (files were overwritten, not duplicated)
        second_run_files = list(recipe_dir.glob("*.json"))
        assert len(first_run_files) == len(second_run_files)


@pytest.mark.slow
class TestPackingWorkflowRealData:
    """End-to-end test that runs the *full* workflow (generation AND packing).

    Unlike the dry-run tests above (which only exercise recipe/config generation)
    and the slow tests in ``test_run_packing_workflow.py`` (which pre-supply
    recipes and skip generation), this runs the complete pipeline with
    ``dry_run=False``: generate recipes + configs from the real ``rules_shape``
    template, then actually pack them with cellPACK for both a plain ``random``
    rule and a gradient rule.

    It requires the local structure-data assets (meshes, grid, stats) that are
    not present in fresh clones, so it skips when they are missing and is gated
    behind the ``slow`` marker (run with ``pytest -m slow``).

    To run against a specific cellpack build (e.g. a candidate branch rather than
    the pinned revision), point ``CELLPACK`` at it AND add its repo root to
    ``PYTHONPATH`` so the ``pack.py`` subprocess imports that branch — setting
    ``CELLPACK`` alone only redirects the script, not ``import cellpack``. The
    cellpack build in use is logged by ``pack.py`` ("Using cellpack package from: ...").
    """

    @pytest.fixture
    def real_packing_config(self, tmp_path):
        """Full-pipeline workflow config (generation + packing) on real data."""
        missing = _missing_real_data_assets()
        if missing:
            pytest.skip(f"Missing local assets for real packing test: {missing}")

        assets = _real_data_assets()
        config_data = {
            "packing_id": "peroxisome",
            "structure_id": "SLC25A17",
            "condition": "integration_real",
            "datadir": str(get_datadir_path()),
            "generate_recipes": True,
            "generate_configs": True,
            "dry_run": False,
            "use_mean_cell": False,
            "use_cells_in_8d_sphere": False,
            "get_counts_from_data": True,
            "get_size_from_data": True,
            "get_bounding_box_from_mesh": True,
            "number_of_replicates": 1,
            "result_type": "simularium",
            "num_processes": 1,
            "skip_completed": False,
            "use_additional_struct": False,
            "gradient_structure_name": "peroxisome",
            "recipe_template_path": str(assets["recipe_template"]),
            "config_template_path": str(assets["config_template"]),
            # keep generated inputs and outputs inside the temp dir
            "generated_recipe_path": str(tmp_path / "recipes"),
            "generated_config_path": str(tmp_path / "configs"),
            "output_path": str(tmp_path / "outputs"),
            "recipe_data": {
                "random": {},
                "nucleus_gradient_strong": {"gradients": ["nucleus_gradient"]},
            },
            "packings_to_run": {"rules": ["random", "nucleus_gradient_strong"]},
        }
        config_path = tmp_path / "workflow_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        return config_path, tmp_path

    def test_full_workflow_generates_and_packs(self, real_packing_config):
        """Generate recipes/configs and pack them for a random and a gradient rule."""
        config_path, tmp_path = real_packing_config

        cell_ids = [REAL_PACKING_CELL_ID]
        # Pin cell selection on both the generation side (sample_cell_ids_for_structure)
        # and the packing side (get_cell_id_list_for_structure) to a known local cell.
        with (
            patch(
                "cellpack_analysis.packing.generate_cellpack_input_files."
                "sample_cell_ids_for_structure",
                return_value=cell_ids,
            ),
            patch(
                "cellpack_analysis.packing.pack_recipes.get_cell_id_list_for_structure",
                return_value=cell_ids,
            ),
        ):
            result = _run_packing_workflow(config_path)

        assert result == 0, f"Packing workflow reported {result} failure(s)"

        # Recipes were generated for both rules
        for rule in ["random", "nucleus_gradient_strong"]:
            recipe = tmp_path / "recipes" / rule / f"peroxisome_{rule}_{REAL_PACKING_CELL_ID}.json"
            assert recipe.exists(), f"Recipe not generated for rule {rule}: {recipe}"

        # A .simularium result was packed for each rule, with the expected naming
        for rule in ["random", "nucleus_gradient_strong"]:
            sim_dir = tmp_path / "outputs" / rule / "peroxisome" / "spheresSST"
            assert sim_dir.is_dir(), f"Output dir missing for rule {rule}: {sim_dir}"
            results = list(
                sim_dir.glob(f"results_*_{rule}_{REAL_PACKING_CELL_ID}_seed_*.simularium")
            )
            assert results, (
                f"No results_*.simularium for rule {rule} in {sim_dir}. "
                f"Contents: {list(sim_dir.iterdir())}"
            )
