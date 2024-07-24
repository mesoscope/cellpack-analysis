import concurrent.futures
import gc
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import trimesh
import vtk
from tqdm import tqdm
from trimesh import proximity

from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure

log = logging.getLogger(__name__)


def round_away_from_zero(array):
    """
    Rounds the elements of the input array away from zero.

    Parameters:
        array (numpy.ndarray): Input array to be rounded.

    Returns:
        numpy.ndarray: Array with elements rounded away from zero.
    """
    return np.copysign(np.ceil(np.abs(array)), array)


def get_list_of_grid_points(bounding_box, spacing):
    """
    Generate a list of grid points within a given bounding box with a specified spacing.

    Parameters:
    bounding_box (numpy.ndarray): The bounding box defining the region of interest.
    spacing (float): The spacing between grid points.

    Returns:
    numpy.ndarray: A 2D array containing the list of grid points.
    """

    grid = np.mgrid[
        bounding_box[0, 0] : bounding_box[1, 0] + spacing : spacing,
        bounding_box[0, 1] : bounding_box[1, 1] + spacing : spacing,
        bounding_box[0, 2] : bounding_box[1, 2] + spacing : spacing,
    ]
    all_points = grid.reshape(3, -1).T

    return all_points


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


def get_bounding_box(mesh_path, expand=1.0):
    """
    Get the bounding box of a mesh.

    Parameters:
        mesh_path (Path): Path to the mesh file.
        expand (optional): Amount to expand the bounding box. Defaults to 0.

    Returns:
        numpy.ndarray: The bounding box of the mesh.
    """
    mesh = trimesh.load_mesh(mesh_path)
    return mesh.bounds * expand


def process_seed(seed, base_datadir, structure_id):
    if seed == "mean":
        nuc_mesh_path = base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
        mem_mesh_path = base_datadir / "average_shape_meshes/mem_mesh_mean.obj"
    else:
        nuc_mesh_path = (
            base_datadir / f"structure_data/{structure_id}/meshes/nuc_mesh_{seed}.obj"
        )
        mem_mesh_path = (
            base_datadir / f"structure_data/{structure_id}/meshes/mem_mesh_{seed}.obj"
        )
    nuc_grid_distance_path = (
        base_datadir
        / f"structure_data/{structure_id}/grid_distances/nuc_distances_{seed}.npy"
    )
    mem_grid_distance_path = (
        base_datadir
        / f"structure_data/{structure_id}/grid_distances/mem_distances_{seed}.npy"
    )
    z_grid_distance_path = (
        base_datadir
        / f"structure_data/{structure_id}/grid_distances/z_distances_{seed}.npy"
    )
    nuc_grid_distances = np.load(nuc_grid_distance_path)
    mem_grid_distances = np.load(mem_grid_distance_path)
    z_grid_distances = np.load(z_grid_distance_path)

    nuc_mesh = trimesh.load_mesh(str(nuc_mesh_path))
    mem_mesh = trimesh.load_mesh(str(mem_mesh_path))

    nuc_bounds = nuc_mesh.bounds
    cell_bounds = mem_mesh.bounds

    nuc_diameter = np.abs(np.diff(nuc_bounds, axis=0)).max()
    cell_diameter = np.abs(np.diff(cell_bounds, axis=0)).max()

    intracellular_radius = (cell_diameter - nuc_diameter) / 2

    return {
        "nuc_mesh": nuc_mesh,
        "mem_mesh": mem_mesh,
        "nuc_diameter": nuc_diameter,
        "cell_diameter": cell_diameter,
        "nuc_bounds": nuc_bounds,
        "cell_bounds": cell_bounds,
        "intracellular_radius": intracellular_radius,
        "nuc_grid_distances": nuc_grid_distances,
        "mem_grid_distances": mem_grid_distances,
        "z_grid_distances": z_grid_distances,
    }


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
        cellid_list = get_cellid_list_for_structure(
            structure_id=structure_id,
            dsphere=True,
            load_local=True,
        )

    mesh_information_dict = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    process_seed,
                    cellid_list,
                    [base_datadir] * len(cellid_list),
                    [structure_id] * len(cellid_list),
                ),
                total=len(cellid_list),
            )
        )

    for seed, result in zip(cellid_list, results):
        mesh_information_dict[str(seed)] = result

    # save mesh information dictionary
    with open(file_path, "wb") as f:
        pickle.dump(mesh_information_dict, f)

    return mesh_information_dict


def calculate_grid_distances(
    nuc_mesh_path,
    mem_mesh_path,
    cellid,
    spacing,
    save_dir=None,
    recalculate=True,
    calc_nuc_distances=True,
    calc_mem_distances=True,
    calc_z_distances=True,
    chunk_size=None,
):
    """
    Calculate distances to nucleus, membrane, z-coordinate for each point in points.

    Parameters
    ----------
    nuc_mesh_path: Path
        Path to nucleus mesh

    mem_mesh_path: Path
        Path to membrane mesh

    cellid: str
        Cellid for the mesh

    points: np.ndarray
        Array of points to calculate distances for

    save_dir: Path
        Directory to save distances

    recalculate: bool
        Recalculate distances

    calc_nuc_distances: bool
        Calculate distances to nucleus

    calc_mem_distances: bool
        Calculate distances to membrane

    calc_z_distance: bool
        Calculate z-coordinate

    Returns
    ----------
    nuc_distances: np.ndarray
        Array of distances to nucleus

    mem_distances: np.ndarray
        Array of distances to membrane

    z_distances: np.ndarray
        Array of z-coordinates

    chunk_size: int
        Size of chunks to process points
    """
    # check if distances already calculated
    if not recalculate and save_dir is not None:
        mem_file_name = save_dir / f"mem_distances_{cellid}.npy"
        if mem_file_name.exists():
            calc_mem_distances = False
            mem_distances = np.load(save_dir / f"mem_distances_{cellid}.npy")
            log.info(f"Loaded mem distances for {cellid}")

        nuc_file_name = save_dir / f"nuc_distances_{cellid}.npy"
        if nuc_file_name.exists():
            calc_nuc_distances = False
            nuc_distances = np.load(nuc_file_name)
            log.info(f"Loaded nuc distances for {cellid}")

        z_file_name = save_dir / f"z_distances_{cellid}.npy"
        if z_file_name.exists():
            calc_z_distances = False
            z_distances = np.load(z_file_name)
            log.info(f"Loaded z distances for {cellid}")

    # load meshes
    nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
    mem_mesh = trimesh.load_mesh(mem_mesh_path)

    # get bounding box
    bounding_box = round_away_from_zero(mem_mesh.bounds)

    # get grid points
    points = get_list_of_grid_points(bounding_box, spacing)

    if calc_mem_distances:
        # check if points are inside encapsulating sphere
        log.debug(f"Calculating points inside encapsulating sphere for {cellid}")

        mem_centroid = mem_mesh.centroid
        mem_radius = np.max(np.linalg.norm(mem_mesh.bounds[1] - mem_mesh.bounds[0]) / 2)

        outside_encapsulating_sphere = (
            np.linalg.norm(points - mem_centroid, axis=1) > mem_radius
        )
        num_to_check = np.sum(~outside_encapsulating_sphere)

        log.debug(
            f"Checking {num_to_check} "
            f"points inside encapsulating sphere for {cellid}"
        )
        log.debug(f"Fraction of points: {num_to_check / len(points):0.2g}")

        points_to_check_indices = np.where(~outside_encapsulating_sphere)[0]
        log.info(
            f"Calculating membrane distance for {len(points_to_check_indices)} points in {cellid}"
        )

        if chunk_size is None:
            chunk_size = int(len(points_to_check_indices) / 10)

        start_time = time.time()
        mem_distances = np.full(len(points), np.inf)
        for i in tqdm(
            range(0, len(points_to_check_indices), chunk_size),
            desc=f"Membrane distance chunks for {cellid}",
        ):
            chunk_indices = points_to_check_indices[i : i + chunk_size]
            chunk_points = points[chunk_indices]
            mem_distances[chunk_indices] = trimesh.proximity.signed_distance(
                mem_mesh, chunk_points
            )

        if save_dir is not None:
            np.save(save_dir / f"mem_distances_{cellid}.npy", mem_distances)

        log.info(
            f"Took {(time.time() - start_time):0.2g}s to calculate mem distances for {cellid}"
        )

    # get distances to nuc_mesh
    if calc_nuc_distances:
        log.info(f"Calculating nuc distances for {cellid}")

        start_time = time.time()

        # only check points insde the membrane
        points_to_check_indices = np.where(
            (mem_distances > 0) & ~np.isinf(mem_distances)
        )[0]

        nuc_distances = np.full(len(points_to_check_indices), np.inf)

        if chunk_size is None:
            chunk_size = int(len(points_to_check_indices) / 10)

        for i in tqdm(
            range(0, len(points_to_check_indices), chunk_size),
            desc=f"Nucleus distance chunks for {cellid}",
        ):
            chunk_indices = points_to_check_indices[i : i + chunk_size]
            chunk_points = points[chunk_indices]
            nuc_distances[i : i + chunk_size] = -trimesh.proximity.signed_distance(
                nuc_mesh, chunk_points
            )

        log.info(
            f"Took {(time.time() - start_time):0.2g}s to calculate nuc distances for {cellid}"
        )

        if save_dir is not None:
            np.save(save_dir / f"nuc_distances_{cellid}.npy", nuc_distances)

    # get z-coordinates
    if calc_z_distances:
        log.info(f"Calculating z distances for {cellid}")

        start_time = time.time()

        # only check points insde the membrane
        inside_mem_points = points[(mem_distances > 0) & ~np.isinf(mem_distances)]

        z_coords = inside_mem_points[:, 2]
        min_z = np.min(z_coords)
        z_distances = np.abs(z_coords - min_z)

        log.info(
            f"Took {(time.time() - start_time):0.2g}s to calculate z distances for {cellid}"
        )
        if save_dir is not None:
            np.save(save_dir / f"z_distances_{cellid}.npy", z_distances)

    del points
    gc.collect()

    return nuc_distances, mem_distances, z_distances
