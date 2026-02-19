import json
import logging
from pathlib import Path

from cellpack_analysis.lib import default_values
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.io import is_url

logger = logging.getLogger(__name__)


class WorkflowConfig:
    """Class to hold the configuration of the packing workflow."""

    def __init__(self, workflow_config_path: Path | None = None):

        if workflow_config_path is None:
            workflow_config_path = Path(__file__).parent / "configs/peroxisome.json"

        self.workflow_config_path = workflow_config_path
        self.project_root = get_project_root()
        self.data = self._read_config_file()
        self._setup()

    def _read_config_file(self) -> dict:
        with open(self.workflow_config_path) as f:
            config = json.load(f)
        return config

    def _resolve_path(self, path: str | Path) -> Path:
        """
        Resolve a path entry from the config file.

        If the path is absolute, return it as-is.
        If the path is relative, resolve it relative to the project root.

        Parameters
        ----------
        path
            The path to resolve, either as a string or a Path object.

        Returns
        -------
        :
            The resolved Path object.
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            # Resolve relative to project root
            return self.project_root / path_obj

    def _setup(self) -> None:
        self.structure_name = self.data.get("structure_name", default_values.STRUCTURE_NAME)
        self.structure_id = self.data.get("structure_id", default_values.STRUCTURE_ID)
        if (
            self.data.get("use_mean_cell", default_values.USE_MEAN_CELL)
            and self.structure_id != "mean"
        ):
            logger.warning(
                f"Using mean cell but structure_id is {self.structure_id}. "
                f"Overriding structure_id to 'mean'."
            )
            self.structure_id = "mean"
        self.condition = self.data.get("condition", default_values.CONDITION)

        # Base level data directory
        # Resolve datadir: if provided in config, resolve it; otherwise use default
        if "datadir" in self.data:
            self.datadir = self._resolve_path(self.data["datadir"])
        else:
            self.datadir = self._resolve_path(default_values.DATADIR)

        # simulation settings
        self.dry_run = self.data.get("dry_run", default_values.DRY_RUN)
        self.generate_recipes = self.data.get("generate_recipes", default_values.GENERATE_RECIPES)
        self.generate_configs = self.data.get("generate_configs", default_values.GENERATE_CONFIGS)
        self.get_counts_from_data = self.data.get(
            "get_counts_from_data", default_values.GET_COUNTS_FROM_DATA
        )
        self.get_size_from_data = self.data.get(
            "get_size_from_data", default_values.GET_SIZE_FROM_DATA
        )
        self.get_bounding_box_from_mesh = self.data.get(
            "get_bounding_box_from_mesh", default_values.GET_BOUNDING_BOX_FROM_MESH
        )
        self.number_of_replicates = self.data.get(
            "number_of_replicates", default_values.NUM_REPLICATES
        )
        self.result_type = self.data.get("result_type", default_values.RESULT_TYPE)
        self.skip_completed = self.data.get("skip_completed", default_values.SKIP_COMPLETED)
        self.use_mean_cell = self.data.get("use_mean_cell", default_values.USE_MEAN_CELL)
        self.use_cells_in_8d_sphere = self.data.get(
            "use_cells_in_8d_sphere", default_values.USE_CELLS_IN_8D_SPHERE
        )
        self.num_cells = self.data.get("num_cells", default_values.NUM_CELLS)
        self.use_additional_struct = self.data.get(
            "use_additional_struct", default_values.USE_ADDITIONAL_STRUCT
        )
        self.gradient_structure_name = self.data.get(
            "gradient_structure_name", default_values.GRADIENT_STRUCTURE_NAME
        )

        # number of processes
        self.num_processes = self.data.get("num_processes", default_values.NUM_PROCESSES)

        # resolve paths
        if "recipe_template_path" in self.data:
            self.recipe_template_path = self._resolve_path(self.data["recipe_template_path"])
        else:
            self.recipe_template_path = (
                self.datadir / f"templates/recipes/{self.structure_name}_recipe_template.json"
            )

        if "config_template_path" in self.data:
            self.config_template_path = self._resolve_path(self.data["config_template_path"])
        else:
            self.config_template_path = (
                self.datadir / f"templates/configs/{self.structure_name}_config_template.json"
            )

        if "generated_recipe_path" in self.data:
            self.generated_recipe_path = self._resolve_path(self.data["generated_recipe_path"])
        else:
            self.generated_recipe_path = (
                self.datadir / f"recipes/{self.structure_name}/{self.condition}"
            )
        self.generated_recipe_path.mkdir(parents=True, exist_ok=True)

        if "generated_config_path" in self.data:
            self.generated_config_path = self._resolve_path(self.data["generated_config_path"])
        else:
            self.generated_config_path = (
                self.datadir / f"configs/{self.structure_name}/{self.condition}"
            )
        self.generated_config_path.mkdir(parents=True, exist_ok=True)

        # Handle grid_path (may be a URL)
        if "grid_path" in self.data:
            grid_path_value = self.data["grid_path"]
            if is_url(grid_path_value):
                self.grid_path = grid_path_value
            else:
                self.grid_path = self._resolve_path(grid_path_value)
                self.grid_path.mkdir(parents=True, exist_ok=True)
        else:
            self.grid_path = self.datadir / f"structure_data/{self.structure_id}/grids"
            self.grid_path.mkdir(parents=True, exist_ok=True)

        # Handle mesh_path (may be a URL)
        if "mesh_path" in self.data:
            mesh_path_value = self.data["mesh_path"]
            if is_url(mesh_path_value):
                self.mesh_path = mesh_path_value
            else:
                self.mesh_path = self._resolve_path(mesh_path_value)
                self.mesh_path.mkdir(parents=True, exist_ok=True)
        else:
            self.mesh_path = self.datadir / f"structure_data/{self.structure_id}/meshes"
            self.mesh_path.mkdir(parents=True, exist_ok=True)

        subfolder = "8d_sphere_data" if self.use_cells_in_8d_sphere else "full_variance_data"
        if "output_path" in self.data:
            self.output_path = self._resolve_path(self.data["output_path"])
        else:
            self.output_path = self.datadir / f"packing_outputs/{subfolder}/{self.condition}"
        self.output_path.mkdir(parents=True, exist_ok=True)
