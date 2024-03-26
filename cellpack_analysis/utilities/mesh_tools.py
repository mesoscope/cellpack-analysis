import numpy as np
import trimesh
import vtk
from pathlib import Path


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
    query = trimesh.proximity.ProximityQuery(inner_mesh)

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