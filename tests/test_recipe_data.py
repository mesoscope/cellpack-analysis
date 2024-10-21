from pathlib import Path

from cellpack_analysis.packing.generate_cellpack_input_files import (
    update_and_save_recipe,
)


def test_update_and_save_recipe():
    # Define the inputs
    cellid = 1
    structure_name = "test_structure"
    recipe_template = {}  
    rule_name = "test_rule"
    rule_dict = {}  # replace with actual rule_dict
    grid_path = Path("./grid_path")
    mesh_path = Path("./mesh_path")
    generated_recipe_path = Path("./generated_recipe_path")
    multiple_replicates = False
    count = 10
    radius = 5
    get_bounding_box_from_mesh = False

    # Call the function
    result = update_and_save_recipe(
        cellid,
        structure_name,
        recipe_template,
        rule_name,
        rule_dict,
        grid_path,
        mesh_path,
        generated_recipe_path,
        multiple_replicates,
        count,
        radius,
        get_bounding_box_from_mesh,
    )

    # Check the result
    assert isinstance(result, dict)
    assert result["version"] == f"{rule_name}_{cellid}"
    assert result["grid_file_path"] == f"{grid_path}/{cellid}_grid.dat"
    # Add more assertions based on your expectations
