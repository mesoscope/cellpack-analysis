"""Utilities for working with simularium files."""

import json
import logging
from pathlib import Path

import matplotlib.colors as mcolors

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
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
    with open(file_path, "r") as f:
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
        json.dump(sim_data, f, indent=2)
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
    with open(file_path, "r") as f:
        sim_data = json.load(f)

    for mapping in sim_data["trajectoryInfo"]["typeMapping"].values():
        mapping_name = mapping["name"]
        if mapping_name in color_mapping:
            mapping["geometry"]["color"] = mcolors.to_hex(color_mapping[mapping_name])
        elif structure_name in mapping_name and "structure" in color_mapping:
            mapping["geometry"]["color"] = mcolors.to_hex(color_mapping["structure"])

    with open(file_path, "w") as f:
        json.dump(sim_data, f, indent=2)
    logger.debug(f"Updated colors in {file_path}")


def generate_and_upload_thumbnail(
    s3_client,
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


def process_simularium_file(
    file_path: Path,
    structure_id: str,
    structure_name: str,
    packing_id: str,
    rule: str,
    dataset: str,
    condition: str,
    experiment: str,
    base_datadir: Path,
    base_s3_mesh_url: str,
    figure_path: Path,
    thumbnail_dir: Path,
    channel_colors: dict[str, tuple[float, float, float]],
    s3_client,
    bucket: str,
    base_s3_url: str,
    structure_stats_df,
    reupload_simularium: bool,
    reupload_thumbnails: bool,
) -> dict:
    """
    Process a single simularium file: update mesh URLs, colors, upload to S3, generate thumbnail.

    Parameters
    ----------
    file_path
        Path to the simularium file
    structure_id
        Structure identifier (e.g., "SLC25A17")
    structure_name
        Structure name (e.g., "peroxisome")
    packing_id
        Packing identifier
    rule
        Packing rule
    dataset
        Dataset name
    condition
        Experimental condition
    experiment
        Experiment name
    base_datadir
        Base data directory
    base_s3_mesh_url
        Base S3 URL for meshes
    figure_path
        Path to directory containing figure files
    thumbnail_dir
        Directory to save thumbnails
    channel_colors
        Dictionary mapping channels to RGB colors
    s3_client
        Boto3 S3 client
    bucket
        S3 bucket name
    base_s3_url
        Base S3 URL
    structure_stats_df
        DataFrame containing structure statistics
    reupload_simularium
        If True, force reupload simularium files
    reupload_thumbnails
        If True, force reupload thumbnails

    Returns
    -------
    dict
        Record containing metadata about the processed file
    """
    cell_id = file_path.stem.split("_seed")[0].split("_")[-1]

    # Update mesh paths in simularium file
    update_simularium_mesh_urls(file_path, base_s3_mesh_url, cell_id)
    # Update colors in simularium file
    update_simularium_colors(file_path, channel_colors, structure_name)

    # Upload simularium file to S3
    simularium_s3_key = file_path.relative_to(base_datadir).as_posix()
    if should_upload_to_s3(s3_client, bucket, simularium_s3_key, reupload_simularium):
        upload_to_s3(s3_client, file_path, bucket, simularium_s3_key)
    simularium_s3_path = f"{base_s3_url}{simularium_s3_key}"

    # Get count if available
    cellid_structure_stats = structure_stats_df[
        (structure_stats_df["CellId"] == cell_id)
        & (structure_stats_df["structure_name"] == structure_id)
    ]
    count = cellid_structure_stats["count"].values[0] if not cellid_structure_stats.empty else None
    cell_volume = (
        cellid_structure_stats["cell_volume"].values[0] * PIXEL_SIZE_IN_UM**3
        if not cellid_structure_stats.empty
        else None
    )
    nuc_volume = (
        cellid_structure_stats["nuc_volume"].values[0] * PIXEL_SIZE_IN_UM**3
        if not cellid_structure_stats.empty
        else None
    )
    cell_height = (
        cellid_structure_stats["cell_height"].values[0] * PIXEL_SIZE_IN_UM
        if not cellid_structure_stats.empty
        else None
    )
    nuc_height = (
        cellid_structure_stats["nuc_height"].values[0] * PIXEL_SIZE_IN_UM
        if not cellid_structure_stats.empty
        else None
    )
    sphericity = (
        cellid_structure_stats["sphericity"].values[0] if not cellid_structure_stats.empty else None
    )

    # Generate and upload thumbnail
    thumbnail_s3_path = generate_and_upload_thumbnail(
        s3_client=s3_client,
        figure_path=figure_path,
        file_stem=file_path.stem,
        packing_id=packing_id,
        condition=condition,
        rule=rule,
        cell_id=cell_id,
        thumbnail_dir=thumbnail_dir,
        base_datadir=base_datadir,
        file_path=file_path,
        channel_colors=channel_colors,
        bucket=bucket,
        base_s3_url=base_s3_url,
        reupload=reupload_thumbnails,
    )

    return {
        "File Name": f"{packing_id}_{rule}_{cell_id}",
        "Cell ID": cell_id,
        "Rule": rule,
        "Packing ID": packing_id,
        "Structure ID": structure_id,
        "Structure Name": structure_name,
        "Count": count,
        "Dataset": dataset,
        "Condition": condition,
        "Experiment": experiment,
        "Cell Volume": cell_volume,
        "Nucleus Volume": nuc_volume,
        "Cell Height": cell_height,
        "Nucleus Height": nuc_height,
        "Cell Sphericity": sphericity,
        "File Type": "simularium",
        "File Path": simularium_s3_path,
        "Thumbnail": thumbnail_s3_path,
    }
