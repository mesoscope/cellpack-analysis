"""Utilities for working with simularium files."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors

from cellpack_analysis.lib.img_io import generate_composite_thumbnail
from cellpack_analysis.lib.s3_utils import should_upload_to_s3, upload_to_s3

logger = logging.getLogger(__name__)


def update_simularium_mesh_urls(file_path: Path, s3_mesh_url: str, cell_id: str) -> None:
    """
    Update mesh URLs in simularium file to point to S3.

    Parameters
    ----------
    file_path
        Path to the simularium file to update
    s3_mesh_url
        Base URL for the meshes on S3 (should end with /)
    cell_id
        Cell ID to construct the full mesh filename (e.g. "nuc_mesh_{cell_id}.obj")
    """
    with open(file_path) as f:
        sim_data = json.load(f)

    # Ensure s3_mesh_url ends with /
    if not s3_mesh_url.endswith("/"):
        s3_mesh_url += "/"

    for mapping in sim_data["trajectoryInfo"]["typeMapping"].values():
        mapping_name = mapping.get("name", "")

        # Update mesh URLs based on geometry type
        if mapping_name == "nucleus":
            mapping["geometry"]["url"] = f"{s3_mesh_url}nuc_mesh_{cell_id}.obj"
        elif mapping_name == "membrane":
            mapping["geometry"]["url"] = f"{s3_mesh_url}mem_mesh_{cell_id}.obj"
        elif "structure" in mapping_name:
            mapping["geometry"]["url"] = f"{s3_mesh_url}struct_mesh_{cell_id}.obj"

    with open(file_path, "w") as f:
        json.dump(sim_data, f)
    logger.debug(f"Updated mesh URLs in {file_path}")


def update_simularium_colors(
    file_path: Path, color_mapping: dict[str, tuple[float, float, float]], structure_name: str
) -> None:
    """
    Update colors in simularium file based on provided color mapping.

    Parameters
    ----------
    file_path
        Path to the simularium file to update
    color_mapping
        Dictionary mapping component names to RGB colors
    structure_name
        Name of the structure being packed
    """
    with open(file_path) as f:
        sim_data = json.load(f)

    for mapping in sim_data["trajectoryInfo"]["typeMapping"].values():
        mapping_name = mapping["name"]
        if mapping_name in color_mapping:
            mapping["geometry"]["color"] = mcolors.to_hex(color_mapping[mapping_name])
        elif structure_name in mapping_name and "structure" in color_mapping:
            mapping["geometry"]["color"] = mcolors.to_hex(color_mapping["structure"])

    with open(file_path, "w") as f:
        json.dump(sim_data, f)
    logger.debug(f"Updated colors in {file_path}")


def generate_and_upload_thumbnail(
    s3_client: Any,
    figure_path: Path,
    file_stem: str,
    packing_id: str,
    condition: str,
    rule: str,
    cell_id: str,
    thumbnail_dir: Path,
    base_datadir: Path,
    file_path: Path,
    channel_colors: dict[str, tuple[float, float, float]],
    bucket: str,
    base_s3_url: str,
    reupload: bool = False,
) -> str:
    """
    Generate thumbnail and upload to S3.

    Parameters
    ----------
    s3_client
        Boto3 S3 client
    figure_path
        Path to directory containing figure files
    file_stem
        Stem of the file name
    packing_id
        Packing identifier
    condition
        Experimental condition
    rule
        Packing rule
    cell_id
        Cell identifier
    thumbnail_dir
        Directory to save thumbnails
    base_datadir
        Base data directory
    file_path
        Path to the simularium file
    channel_colors
        Dictionary mapping channels to RGB colors
    bucket
        S3 bucket name
    base_s3_url
        Base S3 URL
    reupload
        If True, force reupload even if file exists

    Returns
    -------
    str
        S3 URL of the uploaded thumbnail, or empty string if failed
    """
    figure_file_glob = f"voxelized_image_{packing_id}_{condition}_{rule}_{cell_id}_*.ome.tiff"
    figure_file_path = next(figure_path.glob(figure_file_glob), None)

    if figure_file_path is None:
        logger.warning(f"Figure file not found for {file_stem}")
        return ""

    thumbnail_path = thumbnail_dir / f"{file_stem}_thumbnail.png"
    if reupload or not thumbnail_path.exists():
        try:
            generate_composite_thumbnail(
                tiff_path=figure_file_path,
                output_png_path=thumbnail_path,
                image_type="cellpack",
                channel_colors=channel_colors,
            )
            logger.debug(f"Generated thumbnail: {thumbnail_path}")
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {figure_file_path}: {e}")
            return ""

    thumbnail_s3_key = (
        file_path.relative_to(base_datadir).parent.as_posix() + f"/thumbnails/{thumbnail_path.name}"
    )

    if should_upload_to_s3(s3_client, bucket, thumbnail_s3_key, reupload):
        upload_to_s3(s3_client, thumbnail_path, bucket, thumbnail_s3_key)

    return f"{base_s3_url}{thumbnail_s3_key}"


def _agent_records(data: list[float]) -> list[list[float]]:
    """Split a frame's flat agent data into one record per agent."""
    records = []
    start = 0
    while start < len(data):
        end = start + 11 + int(data[start + 10])
        records.append(data[start:end])
        start = end
    return records


def combine_seed_simulariums(seed_paths: list[Path], output_path: Path) -> None:
    """
    Write one simularium file with each single-seed file as a frame.

    Every frame keeps its compartments and all of its agents.

    Parameters
    ----------
    seed_paths
        Single-frame simularium files, one per seed, in frame order
    output_path
        Path to save the combined simularium file
    """
    seed_data = []
    for seed_path in seed_paths:
        with open(seed_path) as f:
            seed_data.append(json.load(f))

    combined = seed_data[0]
    type_mapping = combined["trajectoryInfo"]["typeMapping"]
    frame_records = []
    for seed_path, sim_data in zip(seed_paths, seed_data, strict=True):
        if sim_data["trajectoryInfo"]["typeMapping"] != type_mapping:
            raise ValueError(f"Type mapping differs in {seed_path}")
        (frame,) = sim_data["spatialData"]["bundleData"]
        frame_records.append(_agent_records(frame["data"]))

    # Offset non-mesh agent IDs so each frame's agents are distinct
    id_stride = 1 + int(max(record[1] for records in frame_records for record in records))
    frames = []
    for frame_index, records in enumerate(frame_records):
        data: list[float] = []
        for record in records:
            if type_mapping[str(int(record[2]))]["geometry"]["displayType"] != "OBJ":
                record[1] += frame_index * id_stride
            data.extend(record)
        frames.append({"frameNumber": frame_index, "time": float(frame_index), "data": data})

    combined["trajectoryInfo"]["totalSteps"] = len(frames)
    combined["trajectoryInfo"]["timeStepSize"] = 1.0
    combined["spatialData"]["bundleSize"] = len(frames)
    combined["spatialData"]["bundleData"] = frames
    with open(output_path, "w") as f:
        json.dump(combined, f)
    logger.debug(f"Combined {len(frames)} seed files into {output_path}")
