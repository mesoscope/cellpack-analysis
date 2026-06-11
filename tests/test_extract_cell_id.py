import pytest

from cellpack_analysis.data_release.run_data_release_workflow import extract_cell_id


@pytest.mark.parametrize(
    "file_stem,packing_id,condition,expected",
    [
        (
            "results_peroxisome_rules_shape_random_743916_seed_0",
            "peroxisome",
            "rules_shape_random",
            "743916",
        ),
        (
            "results_peroxisome_rules_shape_with_seed_743916",
            "peroxisome",
            "rules_shape_with_seed",
            "743916",
        ),
        (
            "results_peroxisome_rules_shape_743916_seed_0",
            "peroxisome",
            "rules_shape",
            "743916",
        ),
        (
            "results_peroxisome_rules_shape_743916",
            "peroxisome",
            "rules_shape",
            "743916",
        ),
    ],
)
def test_extract_cell_id(file_stem: str, packing_id: str, condition: str, expected: str) -> None:
    assert extract_cell_id(file_stem, packing_id, condition) == expected
