import numpy as np


def get_mesh_vertices(mesh_file_path):
    """
    Given a mesh file path, returns the coordinates of the mesh vertices.
    """
    coordinates = []
    with open(mesh_file_path, 'r') as mesh_file:
        for line in mesh_file:
            if line.startswith('v'):
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
