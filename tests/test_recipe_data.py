from pathlib import Path
from unittest.mock import patch

import pytest

from cellpack_analysis.packing.generate_cellpack_input_files import update_and_save_recipe


@pytest.fixture
def sample_recipe_template():
    """Fixture providing a sample recipe template for testing."""
    return {
        "version": "original",
        "bounding_box": [[0, 0, 0], [100, 100, 100]],
        "randomness_seed": 12345,
        "grid_file_path": "original_grid.dat",
        "objects": {
            "nucleus_mesh": {
                "representations": {
                    "mesh": {"path": "original_path", "name": "original_nucleus.obj"}
                }
            },
            "membrane_mesh": {
                "representations": {
                    "mesh": {"path": "original_path", "name": "original_membrane.obj"}
                }
            },
            "peroxisome": {"radius": 2.0},
        },
        "composition": {
            "membrane": {"regions": {"interior": ["nucleus", {"object": "existing", "count": 5}]}}
        },
    }


@pytest.fixture
def sample_rule_dict():
    """Fixture providing a sample rule dictionary for testing."""
    return {
        "nucleus_gradient": {
            "description": "gradient based on distance from the surface of the nucleus mesh",
            "mode": "surface",
            "mode_settings": {"object": "nucleus", "scale_to_next_surface": False},
            "weight_mode": "exponential",
            "weight_mode_settings": {"decay_length": 0.1},
        }
    }


class TestUpdateAndSaveRecipe:
    """Test class for update_and_save_recipe function."""

    @patch("cellpack_analysis.packing.generate_cellpack_input_files.write_json")
    @patch("cellpack_analysis.packing.generate_cellpack_input_files.process_rule_dict")
    @patch("pathlib.Path.mkdir")
    def test_update_and_save_recipe_basic(
        self,
        mock_mkdir,
        mock_process_rule_dict,
        mock_write_json,
        sample_recipe_template,
        sample_rule_dict,
    ):
        """Test basic functionality of update_and_save_recipe."""

        # Setup mocks - process_rule_dict should return whatever it receives
        def mock_process_side_effect(recipe, rule_dict, structure_name):
            return recipe

        mock_process_rule_dict.side_effect = mock_process_side_effect

        # Define inputs
        cell_id = 743916
        structure_name = "peroxisome"
        rule_name = "test_rule"
        grid_path = Path("./grid_path")
        mesh_path = Path("./mesh_path")
        generated_recipe_path = Path("./generated_recipe_path")
        multiple_replicates = False
        count = 10
        radius = 5.0
        get_bounding_box_from_mesh = False

        # Call function
        result = update_and_save_recipe(
            cell_id=cell_id,
            structure_name=structure_name,
            recipe_template=sample_recipe_template,
            rule_name=rule_name,
            rule_dict=sample_rule_dict,
            grid_path=grid_path,
            mesh_path=mesh_path,
            generated_recipe_path=generated_recipe_path,
            multiple_replicates=multiple_replicates,
            count=count,
            radius=radius,
            get_bounding_box_from_mesh=get_bounding_box_from_mesh,
        )

        # Assertions
        assert isinstance(result, dict)
        assert result["version"] == f"{rule_name}_{cell_id}"
        assert result["grid_file_path"] == f"{grid_path}/{cell_id}_grid.dat"
        assert result["randomness_seed"] == cell_id
        assert result["objects"]["peroxisome"]["radius"] == radius
        composition_count = result["composition"]["membrane"]["regions"]["interior"][1]["count"]
        assert composition_count == count

        # Verify mesh paths were updated
        expected_mesh_path = str(mesh_path)
        nucleus_mesh_path = result["objects"]["nucleus_mesh"]["representations"]["mesh"]["path"]
        membrane_mesh_path = result["objects"]["membrane_mesh"]["representations"]["mesh"]["path"]
        nucleus_mesh_name = result["objects"]["nucleus_mesh"]["representations"]["mesh"]["name"]
        membrane_mesh_name = result["objects"]["membrane_mesh"]["representations"]["mesh"]["name"]

        assert nucleus_mesh_path == expected_mesh_path
        assert membrane_mesh_path == expected_mesh_path
        assert nucleus_mesh_name == f"nuc_mesh_{cell_id}.obj"
        assert membrane_mesh_name == f"mem_mesh_{cell_id}.obj"

        # Verify mocks were called correctly
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_process_rule_dict.assert_called_once_with(result, sample_rule_dict, structure_name)
        mock_write_json.assert_called_once()

        # Verify the file path passed to write_json
        write_json_call_args = mock_write_json.call_args[0]
        expected_path = (
            f"{generated_recipe_path}/{rule_name}/" f"{structure_name}_{rule_name}_{cell_id}.json"
        )
        assert write_json_call_args[0] == expected_path

    @patch("cellpack_analysis.packing.generate_cellpack_input_files.write_json")
    @patch("cellpack_analysis.packing.generate_cellpack_input_files.process_rule_dict")
    @patch("cellpack_analysis.packing.generate_cellpack_input_files.get_bounding_box")
    @patch("pathlib.Path.mkdir")
    def test_update_and_save_recipe_with_bounding_box(
        self,
        mock_mkdir,
        mock_get_bounding_box,
        mock_process_rule_dict,
        mock_write_json,
        sample_recipe_template,
        sample_rule_dict,
    ):
        """Test update_and_save_recipe with bounding box calculation from mesh."""
        # Setup mocks
        mock_bounding_box = [[0, 0, 0], [50, 50, 50]]
        mock_get_bounding_box.return_value.tolist.return_value = mock_bounding_box

        def mock_process_side_effect(recipe, rule_dict, structure_name):
            return recipe

        mock_process_rule_dict.side_effect = mock_process_side_effect

        # Define inputs
        cell_id = 743916
        structure_name = "peroxisome"
        rule_name = "test_rule"
        grid_path = Path("./grid_path")
        mesh_path = Path("./mesh_path")
        generated_recipe_path = Path("./generated_recipe_path")
        get_bounding_box_from_mesh = True

        # Call function
        result = update_and_save_recipe(
            cell_id=cell_id,
            structure_name=structure_name,
            recipe_template=sample_recipe_template,
            rule_name=rule_name,
            rule_dict=sample_rule_dict,
            grid_path=grid_path,
            mesh_path=mesh_path,
            generated_recipe_path=generated_recipe_path,
            multiple_replicates=False,
            get_bounding_box_from_mesh=get_bounding_box_from_mesh,
        )

        # Assertions
        assert result["bounding_box"] == mock_bounding_box

        # Verify get_bounding_box was called with correct parameters
        expected_mesh_file = Path(mesh_path) / f"mem_mesh_{cell_id}.obj"
        mock_get_bounding_box.assert_called_once_with(expected_mesh_file, expand=1.05)

    @patch("cellpack_analysis.packing.generate_cellpack_input_files.write_json")
    @patch("cellpack_analysis.packing.generate_cellpack_input_files.process_rule_dict")
    @patch("pathlib.Path.mkdir")
    def test_update_and_save_recipe_multiple_replicates(
        self,
        mock_mkdir,
        mock_process_rule_dict,
        mock_write_json,
        sample_recipe_template,
        sample_rule_dict,
    ):
        """Test update_and_save_recipe with multiple replicates (no seed update)."""

        # Setup mocks
        def mock_process_side_effect(recipe, rule_dict, structure_name):
            return recipe

        mock_process_rule_dict.side_effect = mock_process_side_effect

        # Define inputs
        cell_id = 743916
        structure_name = "peroxisome"
        rule_name = "test_rule"
        grid_path = Path("./grid_path")
        mesh_path = Path("./mesh_path")
        generated_recipe_path = Path("./generated_recipe_path")
        multiple_replicates = True

        # Call function
        result = update_and_save_recipe(
            cell_id=cell_id,
            structure_name=structure_name,
            recipe_template=sample_recipe_template,
            rule_name=rule_name,
            rule_dict=sample_rule_dict,
            grid_path=grid_path,
            mesh_path=mesh_path,
            generated_recipe_path=generated_recipe_path,
            multiple_replicates=multiple_replicates,
        )

        # Assertions - randomness_seed should remain unchanged when multiple_replicates=True
        assert result["randomness_seed"] == sample_recipe_template["randomness_seed"]

    @patch("cellpack_analysis.packing.generate_cellpack_input_files.write_json")
    @patch("cellpack_analysis.packing.generate_cellpack_input_files.process_rule_dict")
    @patch("pathlib.Path.mkdir")
    def test_update_and_save_recipe_with_additional_struct(
        self,
        mock_mkdir,
        mock_process_rule_dict,
        mock_write_json,
        sample_recipe_template,
        sample_rule_dict,
    ):
        """Test update_and_save_recipe with additional structure mesh."""
        # Setup recipe template with struct_mesh
        recipe_with_struct = sample_recipe_template.copy()
        recipe_with_struct["objects"]["struct_mesh"] = {
            "representations": {
                "mesh": {"path": "original_struct_path", "name": "original_struct.obj"}
            }
        }

        def mock_process_side_effect(recipe, rule_dict, structure_name):
            return recipe

        mock_process_rule_dict.side_effect = mock_process_side_effect

        # Define inputs
        cell_id = 743916
        structure_name = "peroxisome"
        rule_name = "test_rule"
        grid_path = Path("./grid_path")
        mesh_path = Path("./mesh_path")
        generated_recipe_path = Path("./generated_recipe_path")
        use_additional_struct = True

        # Call function
        result = update_and_save_recipe(
            cell_id=cell_id,
            structure_name=structure_name,
            recipe_template=recipe_with_struct,
            rule_name=rule_name,
            rule_dict=sample_rule_dict,
            grid_path=grid_path,
            mesh_path=mesh_path,
            generated_recipe_path=generated_recipe_path,
            multiple_replicates=False,
            use_additional_struct=use_additional_struct,
        )

        # Assertions
        expected_mesh_path = str(mesh_path)
        struct_mesh_path = result["objects"]["struct_mesh"]["representations"]["mesh"]["path"]
        struct_mesh_name = result["objects"]["struct_mesh"]["representations"]["mesh"]["name"]
        assert struct_mesh_path == expected_mesh_path
        assert struct_mesh_name == f"struct_mesh_{cell_id}.obj"

    @patch("cellpack_analysis.packing.generate_cellpack_input_files.write_json")
    @patch("cellpack_analysis.packing.generate_cellpack_input_files.process_rule_dict")
    @patch("pathlib.Path.mkdir")
    def test_update_and_save_recipe_with_gradient_structure_name(
        self,
        mock_mkdir,
        mock_process_rule_dict,
        mock_write_json,
        sample_recipe_template,
        sample_rule_dict,
    ):
        """Test update_and_save_recipe with custom gradient structure name."""

        # Setup mocks
        def mock_process_side_effect(recipe, rule_dict, structure_name):
            return recipe

        mock_process_rule_dict.side_effect = mock_process_side_effect

        # Define inputs
        cell_id = 743916
        structure_name = "peroxisome"
        gradient_structure_name = "custom_gradient_structure"
        rule_name = "test_rule"
        grid_path = Path("./grid_path")
        mesh_path = Path("./mesh_path")
        generated_recipe_path = Path("./generated_recipe_path")

        # Call function
        result = update_and_save_recipe(
            cell_id=cell_id,
            structure_name=structure_name,
            recipe_template=sample_recipe_template,
            rule_name=rule_name,
            rule_dict=sample_rule_dict,
            grid_path=grid_path,
            mesh_path=mesh_path,
            generated_recipe_path=generated_recipe_path,
            multiple_replicates=False,
            gradient_structure_name=gradient_structure_name,
        )

        # Verify process_rule_dict was called with custom gradient structure name
        mock_process_rule_dict.assert_called_once_with(
            result, sample_rule_dict, gradient_structure_name
        )
