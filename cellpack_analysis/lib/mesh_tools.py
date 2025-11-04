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
import vtk
from rtree.exceptions import RTreeError
from tqdm import tqdm
from trimesh import proximity
from vtkmodules.util import numpy_support as vtknp

from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure

logger = logging.getLogger(__name__)


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
        reader = vtk.vtkOBJReader()
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
    scaled_nuc_grid_distance_path = base_datadir / (
        f"structure_data/{structure_id}/grid_distances/scaled_nuc_distances_{seed}.npy"
    )

    nuc_grid_distances = np.load(nuc_grid_distance_path)
    mem_grid_distances = np.load(mem_grid_distance_path)
    z_grid_distances = np.load(z_grid_distance_path)
    scaled_nuc_grid_distances = np.load(scaled_nuc_grid_distance_path)

    inside_mem_inds = np.where((mem_grid_distances > 0) & ~np.isinf(mem_grid_distances))[0]
    mem_grid_distances = mem_grid_distances[inside_mem_inds]
    if not (
        len(nuc_grid_distances)
        == len(mem_grid_distances)
        == len(z_grid_distances)
        == len(scaled_nuc_grid_distances)
    ):
        raise ValueError(
            f"Grid distances have different lengths:\n"
            f"nuc: {len(nuc_grid_distances)}, mem: {len(mem_grid_distances)},\n"
            f"z: {len(z_grid_distances)}, scaled_nuc: {len(scaled_nuc_grid_distances)}"
        )

    cytoplasm_point_inds = np.where(nuc_grid_distances > 0)[0]
    z_grid_distances = z_grid_distances[cytoplasm_point_inds]

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
        "nuc_volume": nuc_mesh.volume,
        "cell_volume": mem_mesh.volume,
        "intracellular_radius": intracellular_radius,
        "nuc_grid_distances": nuc_grid_distances,
        "mem_grid_distances": mem_grid_distances,
        "z_grid_distances": z_grid_distances,
        "scaled_nuc_grid_distances": scaled_nuc_grid_distances,
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
    imgdata = vtk.vtkImageData()
    imgdata.SetDimensions(img.shape)

    img = img.transpose(2, 1, 0)
    img_output = img.copy()
    img = img.flatten()
    arr = vtknp.numpy_to_vtk(img, array_type=vtk.VTK_FLOAT)
    arr.SetName("Scalar")
    imgdata.GetPointData().SetScalars(arr)

    # Create 3d mesh
    cf = vtk.vtkContourFilter()
    cf.SetInputData(imgdata)
    cf.SetValue(0, 0.5)
    cf.Update()

    mesh = cf.GetOutput()

    # Calculate the mesh centroid
    coords = vtknp.vtk_to_numpy(mesh.GetPoints().GetData())
    centroid = coords.mean(axis=0, keepdims=True)

    if translate_to_origin is True:
        # Translate to origin
        coords -= centroid
        mesh.GetPoints().SetData(vtknp.numpy_to_vtk(coords))

    return mesh, img_output, tuple(centroid.squeeze())


def get_mesh_information_dict_for_structure(
    structure_id: str,
    base_datadir: Path,
    recalculate: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Retrieve or calculate mesh information dictionary for structure.

    Parameters
    ----------
    structure_id
        ID of the structure
    base_datadir
        Base directory path
    recalculate
        If True, recalculate mesh information. Default is False

    Returns
    -------
    :
        Dictionary mapping cell IDs to mesh information
    """
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


def _load_existing_distances(
    save_dir: Path, cell_id: str
) -> dict[str, tuple[bool, np.ndarray | None]]:
    """Load existing distance arrays if they exist."""
    distance_types = {
        "mem": "mem_distances",
        "nuc": "nuc_distances",
        "z": "z_distances",
        "scaled_nuc": "scaled_nuc_distances",
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

    logger.info(f"Took {(time.time() - start_time):0.2g}s to calculate mem distances for {cell_id}")
    return mem_distances, struct_distances


def _calculate_scaled_nucleus_distances(
    points: np.ndarray,
    inside_mem_inds: np.ndarray,
    nuc_mesh: trimesh.Trimesh,
    mem_mesh: trimesh.Trimesh,
    cell_id: str,
    chunk_size: int,
    save_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate scaled nucleus distances and return both nuc and scaled distances."""
    logger.info(f"Calculating scaled distances for {cell_id}")
    start_time = time.time()

    nuc_distances = np.full(len(inside_mem_inds), np.inf)
    scaled_nuc_distances = np.full(len(inside_mem_inds), np.inf)

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

    logger.info(
        f"Took {(time.time() - start_time):0.2g}s to calculate scaled distances for {cell_id}"
    )

    if save_dir is not None:
        np.save(save_dir / f"scaled_nuc_distances_{cell_id}.npy", scaled_nuc_distances)
        np.save(save_dir / f"nuc_distances_{cell_id}.npy", nuc_distances)

    return nuc_distances, scaled_nuc_distances


def _calculate_nucleus_distances(
    points: np.ndarray,
    inside_mem_inds: np.ndarray,
    nuc_mesh: trimesh.Trimesh,
    cell_id: str,
    chunk_size: int,
    save_dir: Path | None = None,
) -> np.ndarray:
    """Calculate nucleus distances."""
    logger.info(f"Calculating nuc distances for {cell_id}")
    start_time = time.time()

    nuc_distances = np.full(len(inside_mem_inds), np.inf)

    for i in tqdm(
        range(0, len(inside_mem_inds), chunk_size),
        desc=f"Nucleus distance chunks for {cell_id}",
    ):
        chunk_indices = inside_mem_inds[i : (i + chunk_size)]
        chunk_points = points[chunk_indices]
        nuc_distances[i : (i + chunk_size)] = -trimesh.proximity.signed_distance(
            nuc_mesh, chunk_points
        )

    logger.info(f"Took {(time.time() - start_time):0.2g}s to calculate nuc distances for {cell_id}")

    if save_dir is not None:
        np.save(save_dir / f"nuc_distances_{cell_id}.npy", nuc_distances)

    return nuc_distances


def _calculate_z_distances(
    points: np.ndarray,
    inside_mem_inds: np.ndarray,
    cell_id: str,
    save_dir: Path | None = None,
) -> np.ndarray:
    """Calculate z-coordinate distances."""
    logger.info(f"Calculating z distances for {cell_id}")
    start_time = time.time()

    inside_mem_points = points[inside_mem_inds]
    z_coords = inside_mem_points[:, 2]
    min_z = np.min(z_coords)
    z_distances = np.abs(z_coords - min_z)

    logger.info(f"Took {(time.time() - start_time):0.2g}s to calculate z distances for {cell_id}")

    if save_dir is not None:
        np.save(save_dir / f"z_distances_{cell_id}.npy", z_distances)

    return z_distances


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
    chunk_size: int | None = None,
    struct_mesh_path: str | Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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
    """
    # Load existing distances if not recalculating
    distance_flags = {
        "mem": calc_mem_distances,
        "nuc": calc_nuc_distances,
        "z": calc_z_distances,
        "scaled_nuc": calc_scaled_nuc_distances,
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
    else:
        mem_distances = nuc_distances = z_distances = scaled_nuc_distances = None

    # Load meshes
    nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
    mem_mesh = trimesh.load_mesh(mem_mesh_path)
    struct_mesh = trimesh.load_mesh(struct_mesh_path) if struct_mesh_path else None

    # Generate grid points
    bounding_box = round_away_from_zero(mem_mesh.bounds)
    points = get_list_of_grid_points(bounding_box, spacing)
    if save_dir is not None:
        np.save(save_dir / f"grid_points_{cell_id}.npy", points)

    # Set default chunk size
    if chunk_size is None:
        chunk_size = max(1, int(len(points) / 10))

    # Calculate membrane distances
    struct_distances = None
    if distance_flags["mem"]:
        mem_distances, struct_distances = _calculate_membrane_distances(
            points, mem_mesh, struct_mesh, cell_id, chunk_size, save_dir
        )

    if mem_distances is None:
        raise ValueError("Membrane distances must be calculated or loaded first")

    # Find points inside membrane
    inside_mem_inds = np.where((mem_distances > 0) & ~np.isinf(mem_distances))[0]
    if struct_distances is not None:
        inside_mem_inds = inside_mem_inds[
            (struct_distances[inside_mem_inds] < 0) & ~np.isinf(struct_distances[inside_mem_inds])
        ]

    # Update chunk size for interior points
    if chunk_size is None or len(inside_mem_inds) < chunk_size:
        chunk_size = max(1, int(len(inside_mem_inds) / 10))

    # Calculate scaled nucleus distances (includes regular nuc distances)
    if distance_flags["scaled_nuc"]:
        nuc_distances, scaled_nuc_distances = _calculate_scaled_nucleus_distances(
            points, inside_mem_inds, nuc_mesh, mem_mesh, cell_id, chunk_size, save_dir
        )
        distance_flags["nuc"] = False  # Already calculated

    # Calculate nucleus distances separately if needed
    if distance_flags["nuc"]:
        nuc_distances = _calculate_nucleus_distances(
            points, inside_mem_inds, nuc_mesh, cell_id, chunk_size, save_dir
        )

    # Calculate z distances
    if distance_flags["z"]:
        z_distances = _calculate_z_distances(points, inside_mem_inds, cell_id, save_dir)

    # Cleanup
    del points
    gc.collect()

    return nuc_distances, mem_distances, z_distances, scaled_nuc_distances
