import numpy as np
from tqdm import tqdm
import trimesh
import vtk
from pathlib import Path
import pickle

from trimesh import proximity

from cellpack_analysis.analyses.stochastic_variation_analysis.load_data import (
    get_cellid_list,
)


def get_mesh_vertices(mesh_file_path):
    """
    Given a mesh file path, returns the coordinates of the mesh vertices.
    """
    coordinates = []
    with open(mesh_file_path, "r") as mesh_file:
        for line in mesh_file:
            if line.startswith("v"):
                coordinates.append([float(x) for x in line.split()[1:]])
    coordinates = np.array(coordinates)
    return coordinates


def get_mesh_center(mesh_file_path):
    """
    Given a mesh file path, returns the center of the mesh.
    """
    coordinates = get_mesh_vertices(mesh_file_path)
    center = np.mean(coordinates, axis=0)
    return center


def get_mesh_boundaries(mesh_file_path):
    """
    Given a mesh file path, returns the coordinates:
    [max_x, max_y, max_z] , [min_x, min_y, min_z]
    """
    coordinates = get_mesh_vertices(mesh_file_path)
    max_coordinates = np.max(coordinates, axis=0)
    min_coordinates = np.min(coordinates, axis=0)
    return max_coordinates, min_coordinates


def calculate_scaled_distances_from_mesh(positions, inner_mesh, outer_mesh):
    """
    Calculates the scaled distances between the inner mesh and outer mesh surfaces.

    Parameters:
        positions (numpy.ndarray): Array of positions.
        inner_mesh (trimesh.Trimesh): Inner mesh surface.
        outer_mesh (trimesh.Trimesh): Outer mesh surface.

    Returns:
        tuple: A tuple containing the scaled distances between the surfaces, the distances between the surfaces,
               and the distances between the positions and the inner mesh surface.
    """
    query = proximity.ProximityQuery(inner_mesh)

    # closest points on the inner mesh surface
    inner_loc, inner_surface_distances, _ = query.on_surface(positions)
    ray_directions = positions - inner_loc

    # intersecting points on the outer surface
    outer_loc, _, _ = outer_mesh.ray.intersects_location(
        ray_origins=inner_loc, ray_directions=ray_directions
    )

    distance_between_surfaces = np.linalg.norm(outer_loc - inner_loc, axis=1)
    scaled_distance_between_surfaces = (
        inner_surface_distances / distance_between_surfaces
    )

    if any(scaled_distance_between_surfaces > 1) or any(
        scaled_distance_between_surfaces < 0
    ):
        raise ValueError("Check distances between surfaces")

    return (
        scaled_distance_between_surfaces,
        distance_between_surfaces,
        inner_surface_distances,
    )


def get_average_shape_mesh_objects(mesh_folder: Path):
    # Get the mesh objects for the average shape
    mesh_dict = {}
    for shape in ["nuc", "mem"]:
        reader = vtk.vtkOBJReader()
        reader.SetFileName(f"{mesh_folder}/{shape}_mesh_mean.obj")
        reader.Update()
        mesh_dict[shape] = reader.GetOutput()
    return mesh_dict


def get_mesh_information_dict(
    structure_id,
    base_datadir,
    cellid_list=None,
    recalculate=False,
):
    """
    Retrieves or calculates mesh information dictionary.

    Args:
        all_positions (dict): Dictionary containing positions of particles in different modes.
        structure_id (str): ID of the structure.
        base_datadir (str): Base directory path.
        results_dir (str, optional): Directory path to save/load mesh information. Defaults to None.
        recalculate (bool, optional): Flag to indicate whether to recalculate mesh information. Defaults to False.

    Returns:
        dict: Mesh information dictionary.
    """
    file_path = base_datadir / f"structure_data/{structure_id}/mesh_information.dat"
    if not recalculate and file_path.exists():
        print("Loading mesh information")
        with open(file_path, "rb") as f:
            mesh_information_dict = pickle.load(f)
        return mesh_information_dict

    print("Calculating mesh information")
    if cellid_list is None:
        cellid_list = get_cellid_list(structure_id, filter_8d=True)

    mesh_information_dict = {}
    for seed in tqdm([*cellid_list, "mean"], total=len(cellid_list) + 1):
        if seed == "mean":
            nuc_mesh_path = base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
            mem_mesh_path = base_datadir / "average_shape_meshes/mem_mesh_mean.obj"
        else:
            nuc_mesh_path = (
                base_datadir
                / f"structure_data/{structure_id}/meshes/nuc_mesh_{seed}.obj"
            )
            mem_mesh_path = (
                base_datadir
                / f"structure_data/{structure_id}/meshes/mem_mesh_{seed}.obj"
            )
        nuc_grid_distance_path = (
            base_datadir
            / f"structure_data/{structure_id}/grid_distances/nuc_distances_{seed}.npy"
        )
        mem_grid_distance_path = (
            base_datadir
            / f"structure_data/{structure_id}/grid_distances/mem_distances_{seed}.npy"
        )
        nuc_grid_distances = np.load(nuc_grid_distance_path)
        mem_grid_distances = np.load(mem_grid_distance_path)

        nuc_mesh = trimesh.load_mesh(str(nuc_mesh_path))
        mem_mesh = trimesh.load_mesh(str(mem_mesh_path))

        nuc_bounds = nuc_mesh.bounds
        cell_bounds = mem_mesh.bounds

        nuc_diameter = np.diff(nuc_bounds, axis=0).max()
        cell_diameter = np.diff(cell_bounds, axis=0).max()

        intracellular_radius = (cell_diameter - nuc_diameter) / 2

        mesh_information_dict[seed] = {
            "nuc_mesh": nuc_mesh,
            "mem_mesh": mem_mesh,
            "nuc_diameter": nuc_diameter,
            "cell_diameter": cell_diameter,
            "nuc_bounds": nuc_bounds,
            "cell_bounds": cell_bounds,
            "intracellular_radius": intracellular_radius,
            "nuc_grid_distances": nuc_grid_distances,
            "mem_grid_distances": mem_grid_distances,
        }

    # save mesh information dictionary
    with open(file_path, "wb") as f:
        pickle.dump(mesh_information_dict, f)

    return mesh_information_dict
