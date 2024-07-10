import json
from pathlib import Path
from typing import Optional

from cellpack_analysis.lib import default_values


class WorkflowConfig:
    """Class to hold the configuration of the packing workflow."""

    def __init__(self, config_file_path: Optional[Path] = None):

        if config_file_path is None:
            config_file_path = Path(__file__).parent / "configs/example.json"

        self.config_file_path = config_file_path
        self.data = self._read_config_file()
        self._setup()

    def _read_config_file(self):
        with open(self.config_file_path, "r") as f:
            config = json.load(f)
        return config

    def _setup(self):
        self.structure_name = self.data.get(
            "structure_name", default_values.STRUCTURE_NAME
        )
        self.structure_id = self.data.get("structure_id", default_values.STRUCTURE_ID)
        self.condition = self.data.get("condition", default_values.CONDITION)

        # Base level data directory
        self.datadir = self.data.get("datadir", default_values.DATADIR)

        # simulation settings
        self.generate_recipes = self.data.get(
            "generate_recipes", default_values.GENERATE_RECIPES
        )
        self.get_counts_from_data = self.data.get(
            "get_counts_from_data", default_values.GET_COUNTS_FROM_DATA
        )
        self.get_size_from_data = self.data.get(
            "get_size_from_data", default_values.GET_SIZE_FROM_DATA
        )
        self.get_bounding_box_from_mesh = self.data.get(
            "get_bounding_box_from_mesh", default_values.GET_BOUNDING_BOX_FROM_MESH
        )
        self.multiple_replicates = self.data.get(
            "multiple_replicates", default_values.MULTIPLE_REPLICATES
        )
        self.use_mean_cell = self.data.get(
            "use_mean_cell", default_values.USE_MEAN_CELL
        )
        self.use_cells_in_8d_sphere = self.data.get(
            "use_cells_in_8d_sphere", default_values.USE_CELLS_IN_8D_SPHERE
        )

        self.recipe_template_path = Path(
            self.data.get(
                "recipe_template_path",
                self.datadir / f"templates/{self.structure_name}_template.json",
            )
        )

        self.cellpack_config_path = Path(
            self.data.get(
                "cellpack_config_path",
                self.datadir / f"configs/{self.structure_name}_config.json",
            )
        )

        self.generated_recipe_path = Path(
            self.data.get(
                "generated_recipe_path",
                self.datadir / f"recipes/{self.structure_name}/{self.condition}",
            )
        )
        self.generated_recipe_path.mkdir(parents=True, exist_ok=True)

        self.grid_path = Path(
            self.data.get(
                "grid_path",
                self.datadir / f"structure_data/{self.structure_id}/grids",
            )
        )
        self.grid_path.mkdir(parents=True, exist_ok=True)

        self.mesh_path = Path(
            self.data.get(
                "mesh_path",
                self.datadir / f"structure_data/{self.structure_id}/meshes",
            )
        )
        self.mesh_path.mkdir(parents=True, exist_ok=True)

        subfolder = (
            "8d_sphere_data" if self.use_cells_in_8d_sphere else "full_variance_data"
        )
        self.output_path = Path(
            self.data.get(
                "output_path",
                self.datadir
                / f"packing_outputs/{subfolder}/{self.condition}/{self.structure_name}",
            )
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
