# %% [markdown]
# # Visualize grid points
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import sample_cell_ids_for_structure
from cellpack_analysis.lib.mesh_tools import (
    get_distances_from_mesh,
    get_grid_points_slice,
    get_inside_outside_check,
    get_list_of_grid_points,
    get_weights_from_distances,
    round_away_from_zero,
)
from cellpack_analysis.lib.visualization import plot_grid_points_slice

logger = logging.getLogger(__name__)
plt.rcParams["font.size"] = 10

# %% [markdown]
# ## Set parameters and file paths
STRUCTURE_ID = "SLC25A17"  # SLC25A17: peroxisomes, RAB5A: early endosomes
STRUCTURE_NAME = "peroxisome"

project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

figures_dir = base_results_dir / "occupancy_analysis/available_space"
figures_dir.mkdir(exist_ok=True, parents=True)

grid_dir = base_datadir / f"structure_data/{STRUCTURE_ID}/grid_distances"
grid_dir.mkdir(exist_ok=True, parents=True)

logger.info(f"Figures directory: {figures_dir}")
logger.info(f"Grid directory: {grid_dir}")
# %% [markdown]
# ## Select cell_ids to use
mesh_folder = base_datadir / f"structure_data/{STRUCTURE_ID}/meshes/"
cell_ids_to_use = sample_cell_ids_for_structure(
    structure_id=STRUCTURE_ID, num_cells=10, dsphere=True
)
cell_id = "834350"
logger.info(f"Using cell id: {cell_id} for grid spacing illustration")

# %% [markdown]
# ## Get meshes for cell_ids used
nuc_mesh_path = mesh_folder / f"nuc_mesh_{cell_id}.obj"
mem_mesh_path = mesh_folder / f"mem_mesh_{cell_id}.obj"
nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
mem_mesh = trimesh.load_mesh(mem_mesh_path)

# %% [markdown]
# ## Helper functions for grid map creation


def create_axis_grid_data(
    projection_axis,
    all_grid_points,
    spacing,
    nuc_mesh,
    mem_mesh,
    results_dir=None,
    recalculate=False,
):
    """Create grid data for a single projection axis."""
    file_name = f"grid_data_{projection_axis}.pkl"
    if results_dir is not None:
        file_path = results_dir / file_name
        if not recalculate and file_path.exists():
            logger.info(f"Loading existing grid data from {file_path}")
            with open(file_path, "rb") as f:
                axis_data = pickle.load(f)
            return axis_data

    axis_data = {"labels": {"distances": {}, "weights": {}}, "distances": {}, "weights": {}}

    # Get grid points slice and inside/outside checks
    axis_data["grid_points_slice"] = get_grid_points_slice(
        all_grid_points=all_grid_points,
        projection_axis=projection_axis,
        spacing=spacing,
    )

    (
        axis_data["inside_nuc"],
        axis_data["inside_mem"],
        axis_data["inside_mem_outside_nuc"],
    ) = get_inside_outside_check(
        nuc_mesh=nuc_mesh,
        mem_mesh=mem_mesh,
        grid_points_slice=axis_data["grid_points_slice"],
    )

    # Save axis data to file
    if results_dir is not None:
        file_path = results_dir / file_name
        with open(file_path, "wb") as f:
            pickle.dump(axis_data, f)
        logger.info(f"Saved grid data to {file_path}")

    return axis_data


def add_distance_weights(axis_data, distance_type, distances, labels, decay_length):
    """Add distance and weight data to axis data."""
    axis_data["distances"][distance_type] = distances
    axis_data["weights"][distance_type] = get_weights_from_distances(
        distances, decay_length=decay_length
    )
    axis_data["labels"]["distances"][distance_type] = labels["distance"]
    axis_data["labels"]["weights"][distance_type] = labels["weight"]


# %% [markdown]
# ## Get grid points
# SPACING = 1.7063
SPACING = 2
bounds = mem_mesh.bounds
bounding_box = round_away_from_zero(bounds)
all_grid_points = get_list_of_grid_points(bounding_box, SPACING)
logger.info(f"Total grid points to check: {all_grid_points.shape[0]}")
recalculate = False
# %% [markdown]
# ## Set up distance calculation configurations
decay_length = 0.1

# Define distance calculation configurations
distance_configs = {
    "nuc": {
        "labels": {"distance": "Distance from nucleus (µm)", "weight": "Nucleus Weight"},
        "calc_func": lambda axis_data, nuc_mesh, mem_mesh: get_distances_from_mesh(
            points=axis_data["grid_points_slice"][axis_data["inside_mem_outside_nuc"]],
            mesh=nuc_mesh,
            invert=True,
        ),
    },
    "mem": {
        "labels": {"distance": "Distance from membrane (µm)", "weight": "Membrane Weight"},
        "calc_func": lambda axis_data, nuc_mesh, mem_mesh: get_distances_from_mesh(
            points=axis_data["grid_points_slice"][axis_data["inside_mem_outside_nuc"]],
            mesh=mem_mesh,
        ),
    },
    "z": {
        "labels": {"distance": "Z Distance (µm)", "weight": "Z Weight"},
        "calc_func": lambda axis_data, nuc_mesh, mem_mesh: np.abs(
            axis_data["grid_points_slice"][axis_data["inside_mem_outside_nuc"], 2]
        ),
    },
    "rnd": {
        "labels": {"distance": "Uniform Distance", "weight": "Uniform Weight"},
        "calc_func": lambda axis_data, nuc_mesh, mem_mesh: np.ones_like(
            axis_data["grid_points_slice"][axis_data["inside_mem_outside_nuc"], 0]
        ),
    },
}

# %% [markdown]
# ## Create grid map with distances and weights
grid_map = {}
for projection_axis in ["x", "y", "z"]:
    # Initialize axis data
    grid_map[projection_axis] = create_axis_grid_data(
        projection_axis,
        all_grid_points,
        SPACING,
        nuc_mesh,
        mem_mesh,
        results_dir=figures_dir,
        recalculate=recalculate,
    )

    # Calculate distances and weights for each type
    for distance_type, config in distance_configs.items():
        distances = config["calc_func"](grid_map[projection_axis], nuc_mesh, mem_mesh)
        add_distance_weights(
            grid_map[projection_axis], distance_type, distances, config["labels"], decay_length
        )
# %% [markdown]
# ## Calculate mixed weights and add to grid_map
mixed_weight_fraction = {
    "rnd": 0,
    "nuc": 0.33,
    "mem": 0.34,
    "z": 0.33,
}

# Add mixed weights to each projection axis in grid_map
for projection_axis in ["x", "y", "z"]:
    mixed_weights_axis = (
        mixed_weight_fraction["rnd"] * grid_map[projection_axis]["weights"]["rnd"]
        + mixed_weight_fraction["nuc"] * grid_map[projection_axis]["weights"]["nuc"]
        + mixed_weight_fraction["mem"] * grid_map[projection_axis]["weights"]["mem"]
        + mixed_weight_fraction["z"] * grid_map[projection_axis]["weights"]["z"]
    )
    mixed_weights_axis /= np.max(mixed_weights_axis)

    # Add mixed weights to grid_map
    grid_map[projection_axis]["weights"]["mixed"] = mixed_weights_axis
    grid_map[projection_axis]["labels"]["weights"]["mixed"] = "Mixed Weight"

# %% [markdown]
# ## Set dot size for plotting
dot_size = 6

# %% [markdown]
# ## Plot distance and weight maps for all combinations (including mixed weights)
for measure in ["weights", "distances"]:
    measure_dir = figures_dir / measure
    measure_dir.mkdir(exist_ok=True, parents=True)
    for projection_axis in ["x", "y", "z"]:
        # Get available types based on measure
        if measure == "distances":
            available_types = distance_configs.keys()
        else:  # weights
            available_types = [*list(distance_configs.keys()), "mixed"]

        for distance_type in available_types:
            # Skip mixed for distances since it doesn't exist
            if measure == "distances" and distance_type == "mixed":
                continue

            grid_points_slice = grid_map[projection_axis]["grid_points_slice"]
            inside_nuc = grid_map[projection_axis]["inside_nuc"]
            inside_mem_outside_nuc = grid_map[projection_axis]["inside_mem_outside_nuc"]
            color_var = grid_map[projection_axis][measure][distance_type]
            cbar_label = grid_map[projection_axis]["labels"][measure][distance_type]
            file_name = f"{distance_type}_{measure}_{projection_axis}_{cell_id}"

            fig, ax = plot_grid_points_slice(
                grid_points_slice=grid_points_slice,
                inside_mem_outside_nuc=inside_mem_outside_nuc,
                inside_nuc=inside_nuc,
                color_var=color_var,
                cbar_label=cbar_label,
                dot_size=dot_size,
                projection_axis=projection_axis,
                clim=(0, 1) if measure == "weights" else None,
            )
            plt.show()
            fig.savefig(measure_dir / f"{file_name}.pdf", bbox_inches="tight")

# %%
