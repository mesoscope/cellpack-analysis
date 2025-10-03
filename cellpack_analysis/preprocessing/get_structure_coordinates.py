#!/usr/bin/env python
"""
Script to extract 3D coordinates of cellular structures from segmented images.

Usage:
    python get_structure_coordinates.py --structure-id SLC25A17

Example:
    python get_structure_coordinates.py --structure-id RAB5A --full --subfolder sample_8d

Available structures:
    - SLC25A17 (peroxisomes)
    - RAB5A (early endosomes)  
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import io, measure
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.label_tables import STRUCTURE_NAME_DICT

logger = logging.getLogger(__name__)

def get_positions_from_single_image(
    file_path: Path,
    mem_channel_index: int = 1,
    nuc_channel_index: int = 0,
    struct_channel_index: int = 3,
) -> tuple[str, list[list[float]], list[float], list[float], list[float], list[float]]:
    """
    Get the positions of structures from a single image.

    Parameters
    ----------
    file_path
        The path to the image file.
    mem_channel_index
        Index of the membrane channel.
    nuc_channel_index
        Index of the nucleus channel.
    struct_channel_index
        Index of the structure channel.

    Returns
    -------
    cell_id
        The cell ID extracted from the file name.
    positions
        A list of positions of the structures.
    nuc_centroid
        The centroid of the nucleus.
    mem_centroid
        The centroid of the membrane.
    struct_nuc_distances
        A list of distances from each structure to the nucleus.
    struct_mem_distances
        A list of distances from each structure to the membrane.
    """
    cell_id = file_path.stem.split("_")[1]
    img = io.imread(file_path)
    img_pex = img[:, struct_channel_index]
    img_nuc = img[:, nuc_channel_index]
    img_mem = img[:, mem_channel_index]
    label_img_pex, n_pex = measure.label(img_pex, return_num=True)  # type: ignore
    label_img_nuc, n_nuc = measure.label(img_nuc, return_num=True)  # type: ignore
    label_img_mem, n_mem = measure.label(img_mem, return_num=True)  # type: ignore

    nuc_positions = []
    nuc_sizes = []
    if n_nuc > 10:
        logger.info(f"Warning: {n_nuc} nuclei detected in cell {cell_id}")
    for i in range(1, n_nuc + 1):
        zcoords, ycoords, xcoords = np.where(label_img_nuc == i)
        nuc_positions.append(
            [
                np.mean(xcoords),
                np.mean(ycoords),
                np.mean(zcoords),
            ]
        )
        nuc_sizes.append(len(xcoords))
    nuc_lcc = np.argmax(nuc_sizes)
    nuc_distances = ndimage.distance_transform_edt(
        ~(label_img_nuc == (nuc_lcc + 1)), return_indices=False
    )
    nuc_centroid = nuc_positions[nuc_lcc]

    mem_positions = []
    mem_sizes = []
    if n_mem > 10:
        logger.info(f"Warning: {n_mem} membranes detected in cell {cell_id}")
    for i in range(1, n_mem + 1):
        zcoords, ycoords, xcoords = np.where(label_img_mem == i)
        mem_positions.append(
            [
                np.mean(xcoords),
                np.mean(ycoords),
                np.mean(zcoords),
            ]
        )
        mem_sizes.append(len(xcoords))
    mem_lcc = np.argmax(mem_sizes)
    mem_distances = ndimage.distance_transform_edt(
        ~(label_img_mem == (mem_lcc + 1)), return_indices=False
    )
    mem_centroid = mem_positions[mem_lcc]

    positions = []
    struct_nuc_distances = []
    struct_mem_distances = []
    for i in range(1, n_pex + 1):
        zcoords, ycoords, xcoords = np.where(label_img_pex == i)
        centroid = [np.mean(xcoords), np.mean(ycoords), np.mean(zcoords)]
        centroid_inds = np.round(centroid).astype(int)
        positions.append(
            centroid,
        )
        struct_nuc_distances.append(
            nuc_distances[centroid_inds[2], centroid_inds[1], centroid_inds[0]]  # type: ignore
        )
        struct_mem_distances.append(
            mem_distances[centroid_inds[2], centroid_inds[1], centroid_inds[0]]  # type: ignore
        )
    return (
        cell_id,
        positions,
        nuc_centroid,
        mem_centroid,
        struct_nuc_distances,
        struct_mem_distances,
    )

def extract_structure_coordinates(
    file_path_list: list[Path],
    structure_id: str,
    structure_name: str,
    num_processes: int,
    datadir: Path,
):
    """
    Extract structure coordinates from segmented images and save to JSON files.

    Parameters
    ----------
    file_path_list
        List of paths to segmented image files.
    structure_id
        The structure ID (e.g., SLC25A17 for peroxisomes, RAB5A for early endosomes).
    structure_name
        The structure name corresponding to the structure ID.
    num_processes
        Number of parallel processes to use.
    datadir
        Directory to save the output JSON files.
    """
    positions_dict = {}
    centroids_dict = {}
    distances_dict = {}

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for _, (cell_id, positions, nuc_centroid, mem_centroid, nuc_distances, mem_distances) in tqdm(
            zip(
                file_path_list,
                executor.map(get_positions_from_single_image, file_path_list),
                strict=False,
            ),
            total=len(file_path_list),
        ):
            positions_dict[cell_id] = {}
            positions_dict[cell_id][f"membrane_interior_{structure_name}"] = positions
            centroids_dict[cell_id] = {}
            centroids_dict[cell_id]["nucleus"] = nuc_centroid
            centroids_dict[cell_id]["membrane"] = mem_centroid
            distances_dict[cell_id] = {}
            distances_dict[cell_id]["nucleus"] = nuc_distances
            distances_dict[cell_id]["membrane"] = mem_distances

    # Save positions, centroids, and distances to JSON files
    save_path = datadir / f"positions_{structure_id}.json"
    with open(save_path, "w") as f:
        json.dump(positions_dict, f, indent=4, sort_keys=True)
    logger.info(f"Saved positions to {save_path}")

    save_path = datadir / f"centroids/centroids_{structure_id}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(centroids_dict, f, indent=4, sort_keys=True)
    logger.info(f"Saved centroids to {save_path}")

    save_path = datadir / f"distances/distances_{structure_id}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(distances_dict, f, indent=4, sort_keys=True)
    logger.info(f"Saved distances to {save_path}")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract 3D coordinates of cellular structures from segmented images."
    )
    parser.add_argument(
        "--structure-id",
        type=str,
        required=True,
        help="The structure ID (e.g., SLC25A17 for peroxisomes, RAB5A for early endosomes).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full variance dataset instead of 8D sphere sample.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Custom subfolder name within structure_data (default: sample_8d or full).",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of parallel processes to use (default: 4).",
    )
    args = parser.parse_args()

    structure_id = args.structure_id
    structure_name = STRUCTURE_NAME_DICT.get(structure_id)
    if structure_name is None:
        logger.error(f"Unknown structure ID: {structure_id}. Exiting.")
        return 1

    dsphere = not args.full
    if args.subfolder:
        subfolder = args.subfolder
    else:
        subfolder = "sample_8d" if dsphere else "full"

    structure_data_dir = get_datadir_path() / f"structure_data/{structure_id}/{subfolder}"
    if not structure_data_dir.exists():
        logger.error(f"Structure data directory {structure_data_dir} does not exist.")
        logger.error("Please run get_structure_images.py first to download images.")
        return 1
    logger.info(f"Results will be saved to {structure_data_dir}")

    img_path = structure_data_dir / "segmented"
    if not img_path.exists(): 
        logger.error(f"Segmented image directory {img_path} does not exist.")
        logger.error("Please run get_structure_images.py first to download images.")
        return 1

    file_path_list = [f for f in img_path.glob("*.tiff") if not f.name.startswith(".")]
    if len(file_path_list) == 0:
        logger.error(f"No .tiff files found in {img_path}.")
        logger.error("Please run get_structure_images.py first to download images.")
        return 1

    extract_structure_coordinates(
        file_path_list=file_path_list,
        structure_id=structure_id,
        structure_name=structure_name,
        num_processes=args.num_processes,
        datadir=structure_data_dir,
    )

    return 0

if __name__ == "__main__":
    exit(main())