import concurrent.futures
import gc
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import skimage.filters as skfilters
import skimage.measure
import trimesh
from rtree.exceptions import RTreeError
from tqdm import tqdm
from trimesh import proximity
from vtkmodules.util import numpy_support as vtknp
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkIOGeometry import vtkOBJReader

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.io import format_time
from cellpack_analysis.lib.label_tables import AXIS_TO_INDEX_MAP

logger = logging.getLogger(__name__)

_EPSILON = 1e-8
_MAX_CHUNK_SIZE = 25_000
_TARGET_CHUNKS = 10


def _adaptive_chunk_size(n: int) -> int:
    """Return a chunk size that targets ``_TARGET_CHUNKS`` chunks but never exceeds
    ``_MAX_CHUNK_SIZE``, bounding per-chunk memory for proximity queries.
    """
    return min(_MAX_CHUNK_SIZE, max(1, -(-n // _TARGET_CHUNKS)))  # ceiling division


def round_away_from_zero(array: np.ndarray) -> np.ndarray:
    """
    Round array elements away from zero.

    Parameters
    ----------
    array
        Input array to be rounded

    Returns
    -------
    :
        Array with elements rounded away from zero
    """
    return np.copysign(np.ceil(np.abs(array)), array)


def get_list_of_grid_points(bounding_box: np.ndarray, spacing: float) -> np.ndarray:
    """
    Generate uniform grid points within a bounding box with specified spacing.

    Parameters
    ----------
    bounding_box
        Array defining the region of interest bounds with shape (2, 3)
        where bounding_box[0] is min coordinates and bounding_box[1] is max coordinates
    spacing
        Distance between adjacent grid points

    Returns
    -------
    :
        2D array of grid point coordinates with shape (N, 3)
    """
    if bounding_box.shape != (2, 3):
        raise ValueError(f"Expected bounding_box shape (2, 3), got {bounding_box.shape}")

    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")

    # Calculate number of points in each dimension more accurately
    # Use np.arange instead of mgrid slice notation for better control
    min_coords = bounding_box[0]
    max_coords = bounding_box[1]

    # Generate coordinate arrays for each dimension
    x_coords = np.arange(min_coords[0], max_coords[0] + spacing / 2, spacing)
    y_coords = np.arange(min_coords[1], max_coords[1] + spacing / 2, spacing)
    z_coords = np.arange(min_coords[2], max_coords[2] + spacing / 2, spacing)

    # Create meshgrid and reshape more efficiently
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    # Stack and reshape in one operation
    all_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    return all_points


def get_list_of_grid_points_mgrid(bounding_box: np.ndarray, spacing: float) -> np.ndarray:
    """
    Generate uniform grid points within a bounding box with specified spacing.

    Parameters
    ----------
    bounding_box
        Array defining the region of interest bounds
    spacing
        Distance between adjacent grid points

    Returns
    -------
    :
        2D array of grid point coordinates with shape (N, 3)
    """

    grid = np.mgrid[
        bounding_box[0, 0] : bounding_box[1, 0] + spacing : spacing,
        bounding_box[0, 1] : bounding_box[1, 1] + spacing : spacing,
        bounding_box[0, 2] : bounding_box[1, 2] + spacing : spacing,
    ]
    all_points = grid.reshape(3, -1).T

    return all_points


def get_mesh_vertices(mesh_file_path: str | Path) -> np.ndarray:
    """Extract vertex coordinates from mesh file."""
    coordinates = []
    with open(mesh_file_path) as mesh_file:
        for line in mesh_file:
            if line.startswith("v"):
                coordinates.append([float(x) for x in line.split()[1:]])
    coordinates = np.array(coordinates)
    return coordinates


def get_mesh_center(mesh_file_path: str | Path) -> np.ndarray:
    """Calculate center coordinates of mesh vertices."""
    coordinates = get_mesh_vertices(mesh_file_path)
    center = np.mean(coordinates, axis=0)
    return center


def get_mesh_boundaries(mesh_file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate maximum and minimum coordinates of mesh vertices.

    Parameters
    ----------
    mesh_file_path
        Path to the mesh file

    Returns
    -------
    :
        Tuple of (max_coordinates, min_coordinates) arrays
    """
    coordinates = get_mesh_vertices(mesh_file_path)
    max_coordinates = np.max(coordinates, axis=0)
    min_coordinates = np.min(coordinates, axis=0)
    return max_coordinates, min_coordinates


def calculate_scaled_distances_from_mesh(
    positions: np.ndarray, inner_mesh: trimesh.Trimesh, outer_mesh: trimesh.Trimesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate scaled distances between inner and outer mesh surfaces.

    Parameters
    ----------
    positions
        Array of 3D positions
    inner_mesh
        Inner mesh surface
    outer_mesh
        Outer mesh surface

    Returns
    -------
    :
        Tuple containing (scaled distances, distance between surfaces,
        distances to inner surface)

    Raises
    ------
    ValueError
        If calculated distances are outside valid range [0, 1]
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


def get_average_shape_mesh_objects(mesh_folder: Path) -> dict[str, Any]:
    """
    Load mesh objects for the average shape.

    Parameters
    ----------
    mesh_folder
        Path to folder containing mesh files

    Returns
    -------
    :
        Dictionary mapping shape names to VTK mesh objects
    """
    mesh_dict = {}
    for shape in ["nuc", "mem"]:
        reader = vtkOBJReader()
        reader.SetFileName(f"{mesh_folder}/{shape}_mesh_mean.obj")
        reader.Update()
        mesh_dict[shape] = reader.GetOutput()
    return mesh_dict


def expand_bounding_box(bounding_box: np.ndarray, expand: float = 1.2) -> np.ndarray:
    """
    Expand bounding box by specified factor.

    Parameters
    ----------
    bounding_box
        Bounding box to expand
    expand
        Expansion factor, defaults to 1.2

    Returns
    -------
    :
        Expanded bounding box array
    """
    center = np.mean(bounding_box, axis=0)
    sizes = np.abs(np.diff(bounding_box, axis=0).squeeze())
    new_sizes = sizes * expand
    new_bounding_box = np.array([center - new_sizes / 2, center + new_sizes / 2])
    return new_bounding_box


def get_bounding_box(mesh_path: str | Path, expand: float = 1.2) -> np.ndarray:
    """
    Get expanded bounding box of a mesh.

    Parameters
    ----------
    mesh_path
        Path to the mesh file
    expand
        Expansion factor, defaults to 1.2

    Returns
    -------
    :
        Expanded bounding box array
    """
    mesh = trimesh.load_mesh(mesh_path)
    return expand_bounding_box(mesh.bounds, expand)


def get_mesh_information_for_shape(
    seed: str, base_datadir: Path, structure_id: str
) -> dict[str, Any]:
    """
    Get mesh information for a specific shape.

    Parameters
    ----------
    seed
        Seed identifier for the mesh
    base_datadir
        Base directory containing data
    structure_id
        Identifier for the structure

    Returns
    -------
    :
        Dictionary containing mesh and distance information for the shape
    """
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
    # scaled_nuc_grid_distance_path = base_datadir / (
    #     f"structure_data/{structure_id}/grid_distances/scaled_nuc_distances_{seed}.npy"
    # )
    # scaled_z_grid_distance_path = base_datadir / (
    #     f"structure_data/{structure_id}/grid_distances/scaled_z_distances_{seed}.npy"
    # )

    nuc_grid_distances = np.load(nuc_grid_distance_path)
    mem_grid_distances = np.load(mem_grid_distance_path)
    z_grid_distances = np.load(z_grid_distance_path)
    # scaled_nuc_grid_distances = np.load(scaled_nuc_grid_distance_path)
    # scaled_z_grid_distances = np.load(scaled_z_grid_distance_path)

    inside_mem_inds = np.where((mem_grid_distances > 0) & ~np.isinf(mem_grid_distances))[0]
    mem_grid_distances = mem_grid_distances[inside_mem_inds]
    if not (
        len(nuc_grid_distances)
        == len(mem_grid_distances)
        == len(z_grid_distances)
        # == len(scaled_nuc_grid_distances)
        # == len(scaled_z_grid_distances)
    ):
        raise ValueError(
            f"Grid distances have different lengths:\n"
            f"nuc: {len(nuc_grid_distances)},\n"
            f"mem: {len(mem_grid_distances)},\n"
            f"z: {len(z_grid_distances)}, \n"
            # f"scaled_nuc: {len(scaled_nuc_grid_distances)}, \n"
            # f"scaled_z: {len(scaled_z_grid_distances)}"
        )

    cytoplasm_point_inds = np.where(nuc_grid_distances > 0)[0]
    z_grid_distances = z_grid_distances[cytoplasm_point_inds]

    nuc_mesh = trimesh.load_mesh(str(nuc_mesh_path))
    mem_mesh = trimesh.load_mesh(str(mem_mesh_path))

    nuc_bounds = nuc_mesh.bounds
    mem_bounds = mem_mesh.bounds

    nuc_diameter = np.abs(np.diff(nuc_bounds, axis=0)).max()
    mem_diameter = np.abs(np.diff(mem_bounds, axis=0)).max()

    intracellular_radius = (mem_diameter - nuc_diameter) / 2

    mem_sphericity = (np.pi ** (1 / 3) * (6 * mem_mesh.volume) ** (2 / 3)) / mem_mesh.area
    nuc_sphericity = (np.pi ** (1 / 3) * (6 * nuc_mesh.volume) ** (2 / 3)) / nuc_mesh.area

    return {
        "nuc_mesh": nuc_mesh,
        "mem_mesh": mem_mesh,
        "nuc_diameter": nuc_diameter,
        "mem_diameter": mem_diameter,
        "nuc_bounds": nuc_bounds,
        "mem_bounds": mem_bounds,
        "nuc_volume": nuc_mesh.volume,
        "mem_volume": mem_mesh.volume,
        "mem_area": mem_mesh.area,
        "nuc_area": nuc_mesh.area,
        "mem_sphericity": mem_sphericity,
        "nuc_sphericity": nuc_sphericity,
        "intracellular_radius": intracellular_radius,
        "nuc_grid_distances": nuc_grid_distances,
        # "mem_grid_distances": mem_grid_distances,
        "z_grid_distances": z_grid_distances,
        # "scaled_nuc_grid_distances": scaled_nuc_grid_distances,
    }


def get_mesh_from_image(
    image: np.ndarray,
    sigma: float = 0,
    lcc: bool = True,
    translate_to_origin: bool = True,
) -> tuple[Any, np.ndarray, tuple[float, ...]]:
    """
    Convert numpy array to 3D mesh using VTK contour filter.

    Parameters
    ----------
    image
        Binary input array for mesh computation
    sigma
        Gaussian smoothing degree, defaults to 0 (no smoothing)
    lcc
        If True, use only largest connected component. Default is True
    translate_to_origin
        If True, translate mesh to origin (0,0,0). Default is True

    Returns
    -------
    :
        VTK mesh object
    :
        Preprocessed input image array
    :
        Mesh centroid coordinates (x, y, z)

    Raises
    ------
    ValueError
        If no foreground voxels found after preprocessing or object not centered

    Notes
    -----
    Adapted from the aicsshparam package. Input assumed binary with
    isosurface value 0.5. Image borders set to zero to ensure manifold mesh.
    """

    img = image.copy()

    # VTK requires YXZ
    img = np.swapaxes(img, 0, 2)

    # Extracting the largest connected component
    if lcc is True:
        labeled_img = skimage.measure.label(img.astype(np.uint8))
        labeled_img = np.asarray(labeled_img)

        counts = np.bincount(labeled_img.flatten())

        lcc_val = 1 + np.argmax(counts[1:])

        img = np.zeros_like(labeled_img, dtype=np.uint8)
        img[labeled_img == lcc_val] = 1

    # Smooth binarize the input image and binarize
    if sigma > 0:
        img = skfilters.gaussian(img.astype(np.float32), sigma=sigma)

        img[img < 1.0 / np.exp(1.0)] = 0
        img[img > 0] = 1

        if img.sum() == 0:
            raise ValueError("No foreground voxels found after pre-processing. Try using sigma=0.")

    # Set image border to 0 so that the mesh forms a manifold
    img[[0, -1], :, :] = 0
    img[:, [0, -1], :] = 0
    img[:, :, [0, -1]] = 0
    img = img.astype(np.float32)

    if img.sum() == 0:
        raise ValueError(
            "No foreground voxels found after pre-processing." "Is the object of interest centered?"
        )

    # Create vtkImageData
    imgdata = vtkImageData()
    imgdata.SetDimensions(img.shape)

    img = img.transpose(2, 1, 0)
    img_output = img.copy()
    img = img.flatten()
    arr = vtknp.numpy_to_vtk(img, array_type=VTK_FLOAT)
    arr.SetName("Scalar")
    imgdata.GetPointData().SetScalars(arr)

    # Create 3d mesh
    cf = vtkContourFilter()
    cf.SetInputData(imgdata)
    cf.SetValue(0, 0.5)
    cf.Update()

    mesh = cf.GetOutput()

    # Calculate the mesh centroid
    coords = vtknp.vtk_to_numpy(mesh.GetPoints().GetData())
    centroid = coords.mean(axis=0, keepdims=True)

    if translate_to_origin:
        coords -= centroid
        mesh.GetPoints().SetData(vtknp.numpy_to_vtk(coords))

    return mesh, img_output, tuple(centroid.squeeze())


def get_mesh_information_dict_for_structure(
    structure_id: str,
    base_datadir: Path | None = None,
    recalculate: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Retrieve or calculate mesh information dictionary for structure.

    Parameters
    ----------
    structure_id
        ID of the structure
    base_datadir
        Base data directory path. If None, the default data directory is used.
    recalculate
        If True, recalculate mesh information. Default is False

    Returns
    -------
    :
        Dictionary mapping cell IDs to mesh information
    """
    if base_datadir is None:
        base_datadir = get_datadir_path()

    file_path = base_datadir / f"structure_data/{structure_id}/mesh_information.dat"
    if not recalculate and file_path.exists():
        logger.info(f"Loading mesh information for {structure_id} from {file_path}")
        with open(file_path, "rb") as f:
            mesh_information_dict = pickle.load(f)
        return mesh_information_dict

    logger.info(f"Calculating mesh information for {structure_id}")
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
                desc=f"Calculating mesh information for {structure_id}",
            )
        )

    for seed, result in zip(cell_id_list, results, strict=False):
        mesh_information_dict[str(seed)] = result

    # save mesh information dictionary
    with open(file_path, "wb") as f:
        pickle.dump(mesh_information_dict, f)

    return mesh_information_dict


def calc_scaled_distance_to_nucleus_surface_serial(
    position_list: np.ndarray,
    nuc_mesh: Any,
    mem_mesh: Any,
    mem_distances: np.ndarray | None = None,
    nuc_query: proximity.ProximityQuery | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the scaled distance of each point in position_list to the nucleus surface.

    This is the serial (per-ray loop) implementation, kept as a reference and
    fallback. See ``calc_scaled_distance_to_nucleus_surface`` for the vectorized
    version.

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

    if nuc_query is None:
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
            or np.linalg.norm(position - nuc_surface_positions[ind]) < _EPSILON
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
            logger.error(f"Value error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue
        except RTreeError as e:
            logger.error(f"Rtree error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue
        except Exception as e:
            logger.error(f"Unexpected error in scaled distance calculation: {e}")
            failed_inds.append(ind)
            continue

    if len(failed_inds) > 0:
        logger.debug(f"Failed {len(failed_inds)} out of {len(position_list)}")

    distance_between_surfaces = np.linalg.norm(
        mem_surface_positions - nuc_surface_positions, axis=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_nuc_distances = np.divide(nuc_surface_distances, distance_between_surfaces)
    scaled_nuc_distances[failed_inds] = np.nan
    scaled_nuc_distances[(scaled_nuc_distances < 0) | (scaled_nuc_distances > 1)] = np.nan

    return nuc_surface_distances, scaled_nuc_distances, distance_between_surfaces


def calc_scaled_distance_to_nucleus_surface(
    position_list: np.ndarray,
    nuc_mesh: Any,
    mem_mesh: Any,
    mem_distances: np.ndarray | None = None,
    nuc_query: proximity.ProximityQuery | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the scaled distance of each point in position_list to the nucleus surface.

    Uses a vectorized batch ray-cast against the membrane mesh to find the
    cytoplasm thickness along the nucleus-to-particle direction. Falls back to
    ``calc_scaled_distance_to_nucleus_surface_serial`` when the batch ray call
    raises an unexpected exception.

    Parameters
    ----------
    position_list
        A list of 3D coordinates of points
    nuc_mesh
        A trimesh object representing the nucleus surface
    mem_mesh
        A trimesh object representing the membrane surface
    mem_distances
        Pre-computed distances to membrane surface. If None, will be computed.

    Returns
    -------
    :
        Tuple containing nucleus surface distances, scaled nucleus distances,
        and distance between surfaces
    """
    if mem_distances is None:
        mem_distances = np.array(proximity.signed_distance(mem_mesh, position_list))

    if nuc_query is None:
        nuc_query = proximity.ProximityQuery(nuc_mesh)

    # Closest points on the nucleus surface (batch, no loop)
    nuc_surface_positions, _, _ = nuc_query.on_surface(position_list)
    nuc_surface_distances = -nuc_query.signed_distance(position_list)

    n = len(position_list)
    mem_surface_positions = np.full((n, 3), np.nan)

    # Valid: not inside nucleus, not outside membrane, and not coincident with nuc surface
    ray_vectors = position_list - nuc_surface_positions
    ray_lengths = np.linalg.norm(ray_vectors, axis=1)
    valid_mask = (nuc_surface_distances >= 0) & (mem_distances >= 0) & (ray_lengths > _EPSILON)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        try:
            ray_origins = nuc_surface_positions[valid_indices]
            ray_dirs = ray_vectors[valid_indices] / ray_lengths[valid_indices, np.newaxis]

            locations, index_ray, _ = mem_mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_dirs,
                multiple_hits=True,
            )

            if len(locations) > 0:
                # For each ray that hit, pick the intersection closest to its origin
                hit_origins = ray_origins[index_ray]
                hit_dists = np.linalg.norm(locations - hit_origins, axis=1)

                # Per-ray minimum: sort by (ray_idx, dist) so the first occurrence
                # of each ray index is its nearest intersection — no Python loop.
                sort_order = np.lexsort((hit_dists, index_ray))
                sorted_rays = index_ray[sort_order]
                unique_rays, first_idx = np.unique(sorted_rays, return_index=True)

                # Scatter nearest hit position back into mem_surface_positions
                mem_surface_positions[valid_indices[unique_rays]] = locations[sort_order][first_idx]

                missed = len(valid_indices) - len(unique_rays)
                if missed > 0:
                    logger.debug(
                        "Batch ray cast: %d rays missed the membrane out of %d valid rays",
                        missed,
                        len(valid_indices),
                    )
            else:
                logger.debug("Batch ray cast returned no intersections; all results set to nan")

        except Exception as e:
            logger.warning(
                "Vectorized ray cast failed (%s), falling back to serial implementation", e
            )
            return calc_scaled_distance_to_nucleus_surface_serial(
                position_list, nuc_mesh, mem_mesh, mem_distances, nuc_query
            )

    failed_inds = np.where(np.isnan(mem_surface_positions[:, 0]))[0].tolist()
    if len(failed_inds) > 0:
        logger.debug(f"Failed {len(failed_inds)} out of {len(position_list)}")

    distance_between_surfaces = np.linalg.norm(
        mem_surface_positions - nuc_surface_positions, axis=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_nuc_distances = np.divide(nuc_surface_distances, distance_between_surfaces)
    scaled_nuc_distances[failed_inds] = np.nan
    scaled_nuc_distances[(scaled_nuc_distances < 0) | (scaled_nuc_distances > 1)] = np.nan

    return nuc_surface_distances, scaled_nuc_distances, distance_between_surfaces


def _load_existing_distances(
    save_dir: Path, cell_id: str
) -> dict[str, tuple[bool, np.ndarray | None]]:
    """Load existing distance arrays if they exist."""
    distance_types = {
        "mem": "mem_distances",
        "nuc": "nuc_distances",
        "z": "z_distances",
        "scaled_nuc": "scaled_nuc_distances",
        "scaled_z": "scaled_z_distances",
    }

    results = {}
    for key, prefix in distance_types.items():
        file_path = save_dir / f"{prefix}_{cell_id}.npy"
        if file_path.exists():
            results[key] = (False, np.load(file_path))  # (should_calc, data)
            logger.info(f"Loaded {prefix} for {cell_id}")
        else:
            results[key] = (True, None)  # Need to calculate

    return results


def _compute_distances_for_points(
    points: np.ndarray,
    nuc_mesh: trimesh.Trimesh,
    mem_mesh: trimesh.Trimesh,
    distance_measures: set[str],
    mem_distances: np.ndarray | None = None,
    nuc_query: proximity.ProximityQuery | None = None,
) -> dict[str, np.ndarray]:
    """Compute one or more distance measures for an arbitrary batch of points.

    This is the single source of truth for the physical distance calculations
    shared between the grid pipeline (``calculate_grid_distances``) and the
    particle pipeline (``_calculate_distances_for_cell_id`` in
    ``distance.py``).  It is purely computational — no chunking, no I/O, no
    filtering of invalid values; those responsibilities belong to the callers.

    Parameters
    ----------
    points
        3-D coordinates to evaluate, shape ``(N, 3)``.
    nuc_mesh
        Nucleus surface mesh.
    mem_mesh
        Cell membrane surface mesh.
    distance_measures
        Which measures to compute.  Supported values: ``'membrane'``,
        ``'nucleus'``, ``'scaled_nucleus'``, ``'z'``, ``'scaled_z'``.
    mem_distances
        Pre-computed signed membrane distances for *points* (positive inside).
        When supplied the membrane proximity query is skipped, saving time for
        callers that already need ``mem_distances`` for other purposes.
    nuc_query
        Pre-built :class:`trimesh.proximity.ProximityQuery` for *nuc_mesh*.
        When supplied the BVH is reused, saving construction time over many
        consecutive chunk calls.

    Returns
    -------
    :
        Dictionary mapping each requested measure name to its distance array.
        Arrays are the same length as *points* and may contain ``NaN`` /
        ``inf`` for degenerate inputs — callers are responsible for filtering.
    """
    result: dict[str, np.ndarray] = {}

    need_mem = "membrane" in distance_measures
    need_nuc = "nucleus" in distance_measures
    need_scaled_nuc = "scaled_nucleus" in distance_measures
    need_z = "z" in distance_measures
    need_scaled_z = "scaled_z" in distance_measures

    # ------------------------------------------------------------------ #
    # Membrane distances                                                   #
    # ------------------------------------------------------------------ #
    if mem_distances is None:
        mem_distances = trimesh.proximity.signed_distance(mem_mesh, points)

    if need_mem and mem_distances is not None:
        result["membrane"] = mem_distances

    # ------------------------------------------------------------------ #
    # Nucleus / scaled_nucleus                                             #
    # ------------------------------------------------------------------ #
    if need_nuc or need_scaled_nuc:
        nuc_surface_distances, scaled_nuc_distances, _ = calc_scaled_distance_to_nucleus_surface(
            points,
            nuc_mesh,
            mem_mesh,
            mem_distances,
            nuc_query,
        )
        if need_nuc:
            result["nucleus"] = nuc_surface_distances
        if need_scaled_nuc:
            result["scaled_nucleus"] = scaled_nuc_distances

    # ------------------------------------------------------------------ #
    # Z / scaled_z                                                         #
    # ------------------------------------------------------------------ #
    if need_z or need_scaled_z:
        z_min = float(mem_mesh.bounds[0, 2])
        z_distances = np.abs(points[:, 2] - z_min)
        if need_z:
            result["z"] = z_distances
        if need_scaled_z:
            z_range = float(mem_mesh.bounds[1, 2] - z_min)
            if z_range > _EPSILON:
                result["scaled_z"] = z_distances / z_range
            else:
                result["scaled_z"] = np.full(len(points), np.nan)

    return result


def _compute_all_interior_distances(
    interior_points: np.ndarray,
    nuc_mesh: trimesh.Trimesh,
    mem_mesh: trimesh.Trimesh,
    cell_id: str,
    chunk_size: int,
    distance_measures: set[str],
    save_dir: Path | None = None,
    mem_distances_interior: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute nucleus / scaled-nucleus / z distances for interior grid points.

    Delegates per-chunk math to :func:`_compute_distances_for_points` and
    assembles full-length result arrays.

    Parameters
    ----------
    interior_points
        Points already filtered to lie inside the membrane, shape ``(M, 3)``.
    nuc_mesh
        Nucleus surface mesh.
    mem_mesh
        Cell membrane surface mesh.
    cell_id
        Cell identifier used for progress-bar labels and file names.
    chunk_size
        Number of points per proximity-query chunk.
    distance_measures
        Subset of ``{'nucleus', 'scaled_nucleus', 'z', 'scaled_z'}`` to
        compute.  ``'membrane'`` is handled upstream and should not be
        passed here.
    save_dir
        If given, each computed array is saved as a ``.npy`` file using the
        same naming convention as the old individual helpers
        (e.g. ``nuc_distances_{cell_id}.npy``).
    mem_distances_interior
        Pre-computed membrane signed-distances for *interior_points*.  Passed
        through to :func:`_compute_distances_for_points` so the membrane BVH
        is not rebuilt per chunk.

    Returns
    -------
    :
        ``{measure_name: array}`` for every requested measure.
    """
    n = len(interior_points)

    # Pre-allocate full arrays so chunk writes are simple index assignments.
    arrays: dict[str, np.ndarray] = {dm: np.full(n, np.nan) for dm in distance_measures}

    # Build the nucleus BVH once and reuse across all chunks.
    nuc_query = proximity.ProximityQuery(nuc_mesh)

    logger.info(f"Computing interior distances {sorted(distance_measures)} for {cell_id}")
    start_time = time.time()

    for i in tqdm(
        range(0, n, chunk_size),
        desc=f"Interior distance chunks for {cell_id}",
    ):
        chunk_slice = slice(i, i + chunk_size)
        chunk_points = interior_points[chunk_slice]
        chunk_mem = (
            mem_distances_interior[chunk_slice] if mem_distances_interior is not None else None
        )

        chunk_result = _compute_distances_for_points(
            chunk_points,
            nuc_mesh,
            mem_mesh,
            distance_measures,
            mem_distances=chunk_mem,
            nuc_query=nuc_query,
        )
        for dm, arr in chunk_result.items():
            arrays[dm][chunk_slice] = arr

    time_taken = time.time() - start_time
    formatted_time = format_time(time_taken)
    logger.info(f"Took {formatted_time} to compute interior distances for {cell_id}")

    if save_dir is not None:
        _SAVE_NAMES = {
            "nucleus": "nuc_distances",
            "scaled_nucleus": "scaled_nuc_distances",
            "z": "z_distances",
            "scaled_z": "scaled_z_distances",
        }
        for dm, arr in arrays.items():
            prefix = _SAVE_NAMES.get(dm)
            if prefix is not None:
                np.save(save_dir / f"{prefix}_{cell_id}.npy", arr)

    return arrays


def _calculate_membrane_distances(
    points: np.ndarray,
    mem_mesh: trimesh.Trimesh,
    struct_mesh: trimesh.Trimesh | None,
    cell_id: str,
    chunk_size: int,
    save_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Calculate membrane and optional structure distances."""
    logger.info(f"Calculating mem distances for {cell_id}")
    start_time = time.time()

    mem_distances = np.full(len(points), np.inf)
    struct_distances = None

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
        if struct_mesh is not None and struct_distances is not None:
            struct_distances[i : (i + chunk_size)] = trimesh.proximity.signed_distance(
                struct_mesh, chunk_points
            )

    if save_dir is not None:
        np.save(save_dir / f"mem_distances_{cell_id}.npy", mem_distances)

    time_taken = time.time() - start_time
    formatted_time = format_time(time_taken)
    logger.info(f"Took {formatted_time} to calculate mem distances for {cell_id}")
    return mem_distances, struct_distances


def calculate_grid_distances(
    nuc_mesh_path: str | Path,
    mem_mesh_path: str | Path,
    cell_id: str,
    spacing: float,
    save_dir: Path | None = None,
    recalculate: bool = True,
    calc_nuc_distances: bool = True,
    calc_mem_distances: bool = True,
    calc_z_distances: bool = True,
    calc_scaled_nuc_distances: bool = True,
    calc_scaled_z_distances: bool = True,
    chunk_size: int | None = None,
    struct_mesh_path: str | Path | None = None,
) -> tuple[
    np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None
]:
    """
    Calculate distances to nucleus, membrane, and z-coordinate for grid points.

    Parameters
    ----------
    nuc_mesh_path
        Path to nucleus mesh file
    mem_mesh_path
        Path to membrane mesh file
    cell_id
        Cell identifier for the mesh
    spacing
        Grid spacing for point generation
    save_dir
        Directory to save distance arrays
    recalculate
        If True, recalculate existing distances. Default is True
    calc_nuc_distances
        If True, calculate nucleus distances. Default is True
    calc_mem_distances
        If True, calculate membrane distances. Default is True
    calc_z_distances
        If True, calculate z-coordinates. Default is True
    calc_scaled_nuc_distances
        If True, calculate scaled nucleus distances. Default is True
    calc_scaled_z_distances
        If True, calculate scaled z distances (z / cell height). Default is True
    chunk_size
        Number of points to process per chunk
    struct_mesh_path
        Optional path to structure mesh file

    Returns
    -------
    :
        Nucleus distances array or None
    :
        Membrane distances array or None
    :
        Z-coordinate distances array or None
    :
        Scaled nucleus distances array or None
    :
        Scaled z distances array or None
    """
    # Load existing distances if not recalculating
    distance_flags = {
        "mem": calc_mem_distances,
        "nuc": calc_nuc_distances,
        "z": calc_z_distances,
        "scaled_nuc": calc_scaled_nuc_distances,
        "scaled_z": calc_scaled_z_distances,
    }

    if not recalculate and save_dir is not None:
        existing = _load_existing_distances(save_dir, cell_id)
        for key in distance_flags:
            if not existing[key][0]:  # If file exists, don't calculate
                distance_flags[key] = False
        mem_distances = existing["mem"][1]
        nuc_distances = existing["nuc"][1]
        z_distances = existing["z"][1]
        scaled_nuc_distances = existing["scaled_nuc"][1]
        scaled_z_distances = existing["scaled_z"][1]
    else:
        mem_distances = nuc_distances = z_distances = scaled_nuc_distances = scaled_z_distances = (
            None
        )

    # Load meshes
    nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
    mem_mesh = trimesh.load_mesh(mem_mesh_path)
    struct_mesh = trimesh.load_mesh(struct_mesh_path) if struct_mesh_path else None

    # Generate grid points
    bounding_box = round_away_from_zero(mem_mesh.bounds)
    points = get_list_of_grid_points(bounding_box, spacing)
    if save_dir is not None:
        np.save(save_dir / f"grid_points_{cell_id}.npy", points)

    # Compute per-stage chunk sizes: honour an explicit user override, otherwise
    # use the adaptive cap (_MAX_CHUNK_SIZE) to bound per-chunk memory usage.
    _user_chunk_size = chunk_size
    mem_chunk_size = (
        _user_chunk_size if _user_chunk_size is not None else _adaptive_chunk_size(len(points))
    )

    # Calculate membrane distances
    struct_distances = None
    if distance_flags["mem"]:
        mem_distances, struct_distances = _calculate_membrane_distances(
            points, mem_mesh, struct_mesh, cell_id, mem_chunk_size, save_dir
        )

    if mem_distances is None:
        raise ValueError("Membrane distances must be calculated or loaded first")

    # Find points inside membrane
    inside_mem_inds = np.where((mem_distances > 0) & ~np.isinf(mem_distances))[0]
    if struct_distances is not None:
        inside_mem_inds = inside_mem_inds[
            (struct_distances[inside_mem_inds] < 0) & ~np.isinf(struct_distances[inside_mem_inds])
        ]

    inner_chunk_size = (
        _user_chunk_size
        if _user_chunk_size is not None
        else _adaptive_chunk_size(len(inside_mem_inds))
    )

    # Build the set of interior distance measures that still need computing.
    # 'scaled_nuc' flag covers both nucleus and scaled_nucleus simultaneously
    # (the primitive computes both in one pass via calc_scaled_distance_to_nucleus_surface).
    interior_measures: set[str] = set()
    if distance_flags["scaled_nuc"]:
        interior_measures.add("nucleus")
        interior_measures.add("scaled_nucleus")
        distance_flags["nuc"] = False  # _compute_all_interior_distances covers this
    elif distance_flags["nuc"]:
        interior_measures.add("nucleus")
    if distance_flags["z"]:
        interior_measures.add("z")
    if distance_flags["scaled_z"]:
        interior_measures.add("scaled_z")

    if interior_measures:
        interior_points = points[inside_mem_inds]
        mem_distances_interior = mem_distances[inside_mem_inds]
        interior_results = _compute_all_interior_distances(
            interior_points=interior_points,
            nuc_mesh=nuc_mesh,
            mem_mesh=mem_mesh,
            cell_id=cell_id,
            chunk_size=inner_chunk_size,
            distance_measures=interior_measures,
            save_dir=save_dir,
            mem_distances_interior=mem_distances_interior,
        )
        nuc_distances = interior_results.get("nucleus")
        scaled_nuc_distances = interior_results.get("scaled_nucleus")
        z_distances = interior_results.get("z")
        scaled_z_distances = interior_results.get("scaled_z")

    # Raise error if any requested distance measure is still None at this point (should only happen if save_dir was None or files were missing when recalculate=False)
    for key in distance_flags:
        if distance_flags[key] and locals().get(f"{key}_distances") is None:
            raise ValueError(f"{key} distances must be calculated or loaded first")

    # Cleanup
    del points
    gc.collect()

    return nuc_distances, mem_distances, z_distances, scaled_nuc_distances, scaled_z_distances


def get_grid_points_slice(
    all_grid_points: np.ndarray,
    projection_axis: str,
    spacing: float,
) -> np.ndarray:
    """
    Select a slice of grid points at the median coordinate along the projection axis.

    Parameters
    ----------
    all_grid_points : np.ndarray
        All grid points in pixels
    projection_axis : str
        Axis to project along ('x', 'y', or 'z')
    spacing : float
        Spacing between grid points in pixels

    Returns
    -------
    grid_points_slice : np.ndarray
        Grid points for the selected slice in pixels
    """
    coord_values = all_grid_points[:, AXIS_TO_INDEX_MAP[projection_axis]]
    median_coord = np.median(coord_values)
    point_indexes = np.isclose(coord_values, median_coord, atol=spacing / 2)
    logger.info(
        f"Selected {projection_axis} slice at "
        f"{projection_axis}={median_coord * PIXEL_SIZE_IN_UM:.2f} (\u03bcm) "
        f"with {np.sum(point_indexes)} points"
    )
    grid_points_slice = all_grid_points[point_indexes]
    return grid_points_slice


def get_inside_outside_check(
    nuc_mesh: trimesh.Trimesh,
    mem_mesh: trimesh.Trimesh,
    grid_points_slice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get inside-outside check for grid points with respect to a mesh.

    Parameters
    ----------
    nuc_mesh
        Nucleus mesh
    mem_mesh
        Membrane mesh
    grid_points_slice : np.ndarray
        Grid points to check in voxels

    Returns
    -------
    :
        Tuple of boolean arrays indicating points inside nucleus, inside membrane,
        and inside membrane but outside nucleus
    """

    logger.info("Calculating nuc inside check")
    inside_nuc = nuc_mesh.contains(grid_points_slice)
    logger.info("Calculating mem inside check")
    inside_mem = mem_mesh.contains(grid_points_slice)

    inside_mem_outside_nuc = inside_mem & ~inside_nuc

    return inside_nuc, inside_mem, inside_mem_outside_nuc


def get_distances_from_mesh(
    points: np.ndarray, mesh: trimesh.Trimesh, invert: bool = False
) -> np.ndarray:
    """
    Calculate distances from points to a mesh and compute weights based on an exponential decay.

    Parameters
    ----------
    points
        Points to calculate distances for
    mesh
        Mesh to calculate distances to
    invert
        If True, invert the sign of the distances.

    Returns
    -------
    :
        Distances in micrometers

    """
    distances = mesh.nearest.signed_distance(points)
    if invert:
        distances = -distances
    distances_um = distances * PIXEL_SIZE_IN_UM
    return distances_um


def get_weights_from_distances(
    distances_um: np.ndarray, decay_length: float | None = None
) -> np.ndarray:
    """
    Calculate weights based on distances using an exponential decay.

    Parameters
    ----------
    distances_um
        Distances in micrometers
    decay_length
        Decay length for the exponential weight calculation

    Returns
    -------
    :
        Weights based on exponential decay
    """
    scaled_distances = distances_um / np.max(distances_um)
    if decay_length is not None:
        weights = np.exp(-scaled_distances / decay_length)
        weights /= np.max(weights)
    else:
        weights = 1 - scaled_distances
    return weights


def invert_mesh_faces(input_mesh_path: str | Path, output_mesh_path: str | Path) -> None:
    """
    Invert the faces of a mesh and save to a new file.

    Parameters
    ----------
    input_mesh_path
        Path to the input mesh file
    output_mesh_path
        Path to save the inverted mesh file
    """
    mesh = trimesh.load_mesh(input_mesh_path)
    mesh.invert()
    mesh.export(output_mesh_path)
