import concurrent.futures
import gc
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import vtk
from rtree.exceptions import RTreeError
from tqdm import tqdm
from trimesh import proximity

from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure

log = logging.getLogger(__name__)


def round_away_from_zero(array):
    """
    Rounds the elements of the input array away from zero.

    Parameters
    ----------
        array (numpy.ndarray): Input array to be rounded.

    Returns
    -------
        numpy.ndarray: Array with elements rounded away from zero.
    """
    return np.copysign(np.ceil(np.abs(array)), array)


def get_list_of_grid_points(bounding_box, spacing):
    """
    Generate a list of grid points within a given bounding box with a specified spacing.

    Parameters
    ----------
    bounding_box (numpy.ndarray): The bounding box defining the region of interest.
    spacing (float): The spacing between grid points.

    Returns
    -------
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
    """Given a mesh file path, returns the coordinates of the mesh vertices."""
    coordinates = []
    with open(mesh_file_path) as mesh_file:
        for line in mesh_file:
            if line.startswith("v"):
                coordinates.append([float(x) for x in line.split()[1:]])
    coordinates = np.array(coordinates)
    return coordinates


def get_mesh_center(mesh_file_path):
    """Given a mesh file path, returns the center of the mesh."""
    coordinates = get_mesh_vertices(mesh_file_path)
    center = np.mean(coordinates, axis=0)
    return center


def get_mesh_boundaries(mesh_file_path):
    """
    Given a mesh file path, returns the coordinates:
    [max_x, max_y, max_z] , [min_x, min_y, min_z].
    """
    coordinates = get_mesh_vertices(mesh_file_path)
    max_coordinates = np.max(coordinates, axis=0)
    min_coordinates = np.min(coordinates, axis=0)
    return max_coordinates, min_coordinates


def calculate_scaled_distances_from_mesh(positions, inner_mesh, outer_mesh):
    """
    Calculates the scaled distances between the inner mesh and outer mesh surfaces.

    Parameters
    ----------
        positions (numpy.ndarray): Array of positions.
        inner_mesh (trimesh.Trimesh): Inner mesh surface.
        outer_mesh (trimesh.Trimesh): Outer mesh surface.

    Returns
    -------
        tuple: A tuple containing the scaled distances between the surfaces,
            the distances between the surfaces, and the distances between
            the positions and the inner mesh surface.
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
    scaled_distance_between_surfaces = inner_surface_distances / distance_between_surfaces

    if any(scaled_distance_between_surfaces > 1) or any(scaled_distance_between_surfaces < 0):
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


def expand_bounding_box(bounding_box, expand=1.2):
    """
    Expand the bounding box by a given factor.

    Parameters
    ----------
        bounding_box (numpy.ndarray): The bounding box to expand.
        expand (optional): Amount to expand the bounding box. Defaults to 0.

    Returns
    -------
        numpy.ndarray: The expanded bounding box.
    """
    center = np.mean(bounding_box, axis=0)
    sizes = np.abs(np.diff(bounding_box, axis=0).squeeze())
    new_sizes = sizes * expand
    new_bounding_box = np.array([center - new_sizes / 2, center + new_sizes / 2])
    return new_bounding_box


def get_bounding_box(mesh_path, expand=1.2):
    """
    Get the bounding box of a mesh.

    Parameters
    ----------
        mesh_path (Path): Path to the mesh file.
        expand (optional): Amount to expand the bounding box. Defaults to 0.

    Returns
    -------
        numpy.ndarray: The bounding box of the mesh.
    """
    mesh = trimesh.load_mesh(mesh_path)
    return expand_bounding_box(mesh.bounds, expand)


def get_mesh_information_for_shape(seed, base_datadir, structure_id):
    nuc_mesh_path = base_datadir / f"structure_data/{structure_id}/meshes/nuc_mesh_{seed}.obj"
    mem_mesh_path = base_datadir / f"structure_data/{structure_id}/meshes/mem_mesh_{seed}.obj"

    nuc_grid_distance_path = (
        base_datadir / f"structure_data/{structure_id}/grid_distances/nuc_distances_{seed}.npy"
    )
    mem_grid_distance_path = (
        base_datadir / f"structure_data/{structure_id}/grid_distances/mem_distances_{seed}.npy"
    )
    z_grid_distance_path = (
        base_datadir / f"structure_data/{structure_id}/grid_distances/z_distances_{seed}.npy"
    )
    scaled_nuc_grid_distance_path = (
        base_datadir
        / f"structure_data/{structure_id}/grid_distances/scaled_nuc_distances_{seed}.npy"
    )

    nuc_grid_distances = np.load(nuc_grid_distance_path)
    mem_grid_distances = np.load(mem_grid_distance_path)
    z_grid_distances = np.load(z_grid_distance_path)
    scaled_nuc_grid_distances = np.load(scaled_nuc_grid_distance_path)

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
        "scaled_nuc_grid_distances": scaled_nuc_grid_distances,
    }


def get_mesh_information_dict_for_structure(
    structure_id: str,
    base_datadir: Path,
    recalculate: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Retrieves or calculates mesh information dictionary.

    Args:
    ----
        all_positions (dict): Dictionary containing positions of particles
            in different modes.
        structure_id (str): ID of the structure.
        base_datadir (str): Base directory path.
        results_dir (str, optional): Directory path to save/load mesh information.
            Defaults to None.
        recalculate (bool, optional): Flag to indicate whether to recalculate
            mesh information. Defaults to False.

    Returns:
    -------
        dict: Mesh information dictionary.
    """
    file_path = base_datadir / f"structure_data/{structure_id}/mesh_information.dat"
    if not recalculate and file_path.exists():
        log.info(f"Loading mesh information for {structure_id} from {file_path}")
        with open(file_path, "rb") as f:
            mesh_information_dict = pickle.load(f)
        return mesh_information_dict

    log.info(f"Calculating mesh information for {structure_id}")
    if structure_id == "mean":
        cell_id_list = ["mean"]
    else:
        cell_id_list = get_cell_id_list_for_structure(
            structure_id=structure_id,
            dsphere=True,
            load_local=True,
        )

    mesh_information_dict = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    get_mesh_information_for_shape,
                    cell_id_list,
                    [base_datadir] * len(cell_id_list),
                    [structure_id] * len(cell_id_list),
                ),
                total=len(cell_id_list),
            )
        )

    for seed, result in zip(cell_id_list, results, strict=False):
        mesh_information_dict[str(seed)] = result

    # save mesh information dictionary
    with open(file_path, "wb") as f:
        pickle.dump(mesh_information_dict, f)

    return mesh_information_dict


def calc_scaled_distance_to_nucleus_surface(
    position_list: np.ndarray,
    nuc_mesh: Any,
    mem_mesh: Any,
    mem_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the scaled distance of each point in position_list to the nucleus surface.

    Parameters
    ----------
    position_list
        A list of 3D coordinates of points
    nuc_mesh
        A trimesh object representing the nucleus surface
    mem_mesh
        A trimesh object representing the membrane surface
    mem_distances
        Pre-computed distances to membrane surface

    Returns
    -------
    :
        Tuple containing nucleus surface distances, scaled nucleus distances,
        and distance between surfaces
    """
    if mem_distances is None:
        mem_distances = np.array(proximity.signed_distance(mem_mesh, position_list))

    nuc_query = proximity.ProximityQuery(nuc_mesh)

    # closest points in the inner mesh surface
    nuc_surface_positions, _, _ = nuc_query.on_surface(position_list)
    nuc_surface_distances = -nuc_query.signed_distance(position_list)

    # intersecting points on the outer surface
    mem_surface_positions = np.zeros(nuc_surface_positions.shape)
    failed_inds = []
    for ind, (
        position,
        nuc_surface_distance,
        mem_surface_distance,
    ) in enumerate(zip(position_list, nuc_surface_distances, mem_distances, strict=False)):
        if (
            nuc_surface_distance < 0
            or mem_surface_distance < 0
            or position in nuc_surface_positions
        ):
            mem_surface_positions[ind] = np.nan
            failed_inds.append(ind)
            continue
        try:
            direction = position - nuc_surface_positions[ind]
            intersect_positions, _, _ = mem_mesh.ray.intersects_location(
                ray_origins=[nuc_surface_positions[ind]],
                ray_directions=[direction],
            )
            if len(intersect_positions) > 1:
                intersect_distances = np.linalg.norm(
                    intersect_positions - nuc_surface_positions[ind], axis=1
                )
                min_ind = np.argmin(intersect_distances)
                mem_surface_positions[ind] = intersect_positions[min_ind]
            else:
                mem_surface_positions[ind] = intersect_positions
        except ValueError as e:
            log.error(f"Value error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue
        except RTreeError as e:
            log.error(f"Rtree error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue
        except Exception as e:
            log.error(f"Unexpected error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue

    if len(failed_inds) > 0:
        log.debug(f"Failed {len(failed_inds)} out of {len(position_list)}")

    distance_between_surfaces = np.linalg.norm(
        mem_surface_positions - nuc_surface_positions, axis=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_nuc_distances = np.divide(nuc_surface_distances, distance_between_surfaces)
    scaled_nuc_distances[failed_inds] = np.nan
    scaled_nuc_distances[(scaled_nuc_distances < 0) | (scaled_nuc_distances > 1)] = np.nan

    return nuc_surface_distances, scaled_nuc_distances, distance_between_surfaces


def calculate_grid_distances(
    nuc_mesh_path,
    mem_mesh_path,
    cell_id,
    spacing,
    save_dir=None,
    recalculate=True,
    calc_nuc_distances=True,
    calc_mem_distances=True,
    calc_z_distances=True,
    calc_scaled_nuc_distances=True,
    chunk_size=None,
    struct_mesh_path=None,
):
    """
    Calculate distances to nucleus, membrane, z-coordinate for each point in points.

    Parameters
    ----------
    nuc_mesh_path: Path
        Path to nucleus mesh

    mem_mesh_path: Path
        Path to membrane mesh

    cell_id: str
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
    -------
    nuc_distances: np.ndarray
        Array of distances to nucleus

    mem_distances: np.ndarray
        Array of distances to membrane

    z_distances: np.ndarray
        Array of z-coordinates

    scaled_nuc_distances: np.ndarray
        Array of scaled distances to nucleus

    chunk_size: int
        Size of chunks to process points
    """
    # Initialize variables
    mem_distances = None
    nuc_distances = None
    struct_distances = None
    z_distances = None
    scaled_nuc_distances = None

    # check if distances already calculated
    if not recalculate and save_dir is not None:
        mem_file_name = save_dir / f"mem_distances_{cell_id}.npy"
        if mem_file_name.exists():
            calc_mem_distances = False
            mem_distances = np.load(save_dir / f"mem_distances_{cell_id}.npy")
            log.info(f"Loaded mem distances for {cell_id}")

        nuc_file_name = save_dir / f"nuc_distances_{cell_id}.npy"
        if nuc_file_name.exists():
            calc_nuc_distances = False
            nuc_distances = np.load(nuc_file_name)
            log.info(f"Loaded nuc distances for {cell_id}")

        z_file_name = save_dir / f"z_distances_{cell_id}.npy"
        if z_file_name.exists():
            calc_z_distances = False
            z_distances = np.load(z_file_name)
            log.info(f"Loaded z distances for {cell_id}")

        scaled_nuc_file_name = save_dir / f"scaled_nuc_distances_{cell_id}.npy"
        if scaled_nuc_file_name.exists():
            calc_scaled_nuc_distances = False
            scaled_nuc_distances = np.load(scaled_nuc_file_name)
            log.info(f"Loaded scaled nuc distances for {cell_id}")

    # load meshes
    nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
    mem_mesh = trimesh.load_mesh(mem_mesh_path)
    struct_mesh = None
    if struct_mesh_path is not None:
        struct_mesh = trimesh.load_mesh(struct_mesh_path)

    # get bounding box
    bounding_box = round_away_from_zero(mem_mesh.bounds)

    # get grid points
    points = get_list_of_grid_points(bounding_box, spacing)
    if save_dir is not None:
        np.save(save_dir / f"grid_points_{cell_id}.npy", points)

    if calc_mem_distances:
        log.info(f"Calculating mem distances for {cell_id}")
        if chunk_size is None:
            chunk_size = int(len(points) / 10)

        start_time = time.time()
        mem_distances = np.full(len(points), np.inf)

        if struct_mesh is not None:
            struct_distances = np.full(len(points), np.inf)

        for i in tqdm(
            range(0, len(points), chunk_size),
            desc=f"Membrane distance chunks for {cell_id}",
        ):
            chunk_points = points[i : (i + chunk_size)]
            mem_distances[i : (i + chunk_size)] = trimesh.proximity.signed_distance(
                mem_mesh, chunk_points
            )
            if struct_mesh is not None:
                struct_distances[i : (i + chunk_size)] = trimesh.proximity.signed_distance(
                    struct_mesh, chunk_points
                )

        if save_dir is not None:
            np.save(save_dir / f"mem_distances_{cell_id}.npy", mem_distances)

        log.info(
            f"Took {(time.time() - start_time):0.2g}s to calculate mem distances for {cell_id}"
        )

    if mem_distances is None:
        raise ValueError("Membrane distances must be calculated or loaded first")

    inside_mem_inds = np.where((mem_distances > 0) & ~np.isinf(mem_distances))[0]
    if struct_distances is not None:
        inside_mem_inds = inside_mem_inds[
            struct_distances[inside_mem_inds] < 0 & ~np.isinf(struct_distances[inside_mem_inds])
        ]
    # calculate scaled distances
    if calc_scaled_nuc_distances:
        log.info(f"Calculating scaled distances for {cell_id}")

        start_time = time.time()

        nuc_distances = np.full(len(inside_mem_inds), np.inf)
        scaled_nuc_distances = np.full(len(inside_mem_inds), np.inf)

        if chunk_size is None:
            chunk_size = int(len(inside_mem_inds) / 10)

        for i in tqdm(
            range(0, len(inside_mem_inds), chunk_size),
            desc=f"Scaled distance chunks for {cell_id}",
        ):
            chunk_indices = inside_mem_inds[i : (i + chunk_size)]
            chunk_points = points[chunk_indices]
            (
                nuc_distances[i : (i + chunk_size)],
                scaled_nuc_distances[i : (i + chunk_size)],
                _,
            ) = calc_scaled_distance_to_nucleus_surface(chunk_points, nuc_mesh, mem_mesh)

        log.info(
            f"Took {(time.time() - start_time):0.2g}s to calculate scaled distances for {cell_id}"
        )

        if save_dir is not None:
            np.save(save_dir / f"scaled_nuc_distances_{cell_id}.npy", scaled_nuc_distances)

        if calc_nuc_distances:
            if save_dir is not None:
                np.save(save_dir / f"nuc_distances_{cell_id}.npy", nuc_distances)
            calc_nuc_distances = False

    # get distances to nuc_mesh
    if calc_nuc_distances:
        log.info(f"Calculating nuc distances for {cell_id}")

        start_time = time.time()

        nuc_distances = np.full(len(inside_mem_inds), np.inf)

        if chunk_size is None:
            chunk_size = int(len(inside_mem_inds) / 10)

        for i in tqdm(
            range(0, len(inside_mem_inds), chunk_size),
            desc=f"Nucleus distance chunks for {cell_id}",
        ):
            chunk_indices = inside_mem_inds[i : (i + chunk_size)]
            chunk_points = points[chunk_indices]
            nuc_distances[i : (i + chunk_size)] = -trimesh.proximity.signed_distance(
                nuc_mesh, chunk_points
            )

        log.info(
            f"Took {(time.time() - start_time):0.2g}s to calculate nuc distances for {cell_id}"
        )

        if save_dir is not None:
            np.save(save_dir / f"nuc_distances_{cell_id}.npy", nuc_distances)

    # get z-coordinates
    if calc_z_distances and mem_distances is not None:
        log.info(f"Calculating z distances for {cell_id}")

        start_time = time.time()

        # only check points insde the membrane
        inside_mem_points = points[inside_mem_inds]

        z_coords = inside_mem_points[:, 2]
        min_z = np.min(z_coords)
        z_distances = np.abs(z_coords - min_z)

        log.info(f"Took {(time.time() - start_time):0.2g}s to calculate z distances for {cell_id}")
        if save_dir is not None:
            np.save(save_dir / f"z_distances_{cell_id}.npy", z_distances)

    del points
    gc.collect()

    return nuc_distances, mem_distances, z_distances, scaled_nuc_distances
