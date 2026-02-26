# %%
# # cellPACK BFF exporter
# Test notebook to export simularium paths as CSV files
# Need to use s3 uploaded data
# Steps:
# 1. Update simularium files to use mesh paths from s3
# 2. Upload simularium outputs to s3
# 3. Create metadata CSV for BFF (optional)
# 4. Create thumbnails? (tricky, maybe use MIP of tiff outputs)
# 5. Export s3 file paths as CSV for put into BFF


# TODO: Thumbnails?
# TODO: inverted meshes?
# %%
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import matplotlib.colors as mcolors
import pandas as pd
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path, get_results_path
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.img_io import generate_composite_thumbnail
from cellpack_analysis.notebooks.data_release.csv_metadata_config import get_metadata_dict

# %%
# Constants
S3_BUCKET = "cellpack-analysis-data"
BASE_S3_URL = "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/"

structure_stats_df = get_structure_stats_dataframe()


# %%
# Helper functions
def should_upload_to_s3(s3_client, s3_key: str, force_upload: bool = False) -> bool:
    """Check if a file should be uploaded to S3."""
    if force_upload:
        return True
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return False
    except s3_client.exceptions.ClientError:
        return True


def upload_to_s3(s3_client, local_path: Path, s3_key: str) -> None:
    """Upload a file to S3 and set it as public-read."""
    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
    s3_client.put_object_acl(ACL="public-read", Bucket=S3_BUCKET, Key=s3_key)


def update_simularium_mesh_urls(file_path: Path, s3_mesh_url: str, cell_id: str) -> None:
    """Update mesh URLs in simularium file to point to S3.

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


def update_simularium_colors(
    file_path: Path, color_mapping: dict[str, tuple[float, float, float]], structure_name: str
) -> None:
    """Update colors in simularium file based on provided color mapping."""
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
    reupload: bool = False,
) -> str:
    """Generate thumbnail and upload to S3. Returns S3 URL or empty string."""
    figure_file_glob = f"voxelized_image_{packing_id}_{condition}_{rule}_{cell_id}_*.ome.tiff"
    figure_file_path = next(figure_path.glob(figure_file_glob), None)

    if figure_file_path is None:
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
        except Exception as e:
            print(f"Failed to generate thumbnail for {figure_file_path}: {e}")
            return ""

    thumbnail_s3_key = (
        file_path.relative_to(base_datadir).parent.as_posix() + f"/thumbnails/{thumbnail_path.name}"
    )

    if should_upload_to_s3(s3_client, thumbnail_s3_key, reupload):
        upload_to_s3(s3_client, thumbnail_path, thumbnail_s3_key)

    return f"{BASE_S3_URL}{thumbnail_s3_key}"


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
    reupload_simularium: bool,
    reupload_thumbnails: bool,
) -> dict:
    """
    Process a single simularium file: update mesh URLs, colors, upload to S3, generate thumbnail.
    """
    cell_id = file_path.stem.split("_seed")[0].split("_")[-1]

    # Update mesh paths in simularium file
    update_simularium_mesh_urls(file_path, base_s3_mesh_url, cell_id)
    # Update colors in simularium file
    update_simularium_colors(file_path, channel_colors, structure_name)

    # Upload simularium file to S3
    simularium_s3_key = file_path.relative_to(base_datadir).as_posix()
    if should_upload_to_s3(s3_client, simularium_s3_key, reupload_simularium):
        upload_to_s3(s3_client, file_path, simularium_s3_key)
    simularium_s3_path = f"{BASE_S3_URL}{simularium_s3_key}"

    # Get count if available
    cellid_structure_stats = structure_stats_df[
        (structure_stats_df["CellId"] == cell_id)
        & (structure_stats_df["structure_name"] == structure_id)
    ]
    count = cellid_structure_stats["count"].values[0] if not cellid_structure_stats.empty else None
    cell_volume = (
        cellid_structure_stats["cell_volume"].values[0]
        if not cellid_structure_stats.empty
        else None
    )
    nuc_volume = (
        cellid_structure_stats["nuc_volume"].values[0] if not cellid_structure_stats.empty else None
    )
    cell_height = (
        cellid_structure_stats["cell_height"].values[0]
        if not cellid_structure_stats.empty
        else None
    )
    nuc_height = (
        cellid_structure_stats["nuc_height"].values[0] if not cellid_structure_stats.empty else None
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


# %%
base_datadir = get_datadir_path()
base_results_dir = get_results_path()

# %%
csv_output_dir = base_results_dir / "data_release" / "test_bff"
csv_output_dir.mkdir(parents=True, exist_ok=True)

thumbnail_dir = csv_output_dir / "thumbnails"
thumbnail_dir.mkdir(parents=True, exist_ok=True)

# %%
# Structure configuration
structures = [
    {
        "structure_id": "SLC25A17",
        "structure_name": "peroxisome",
        "packing_id": "peroxisome",
        "color": (0.12, 1.0, 0.12),  # rgb(44, 160, 44)
    },
    {
        "structure_id": "RAB5A",
        "structure_name": "endosome",
        "packing_id": "endosome",
        "color": (1.0, 0.78, 0.20),  # rgb(255, 127, 14)
    },
]

# Experiment configuration
dataset = "8d_sphere_data"
condition = "rules_shape"
experiment = "norm_weights"
rules = ["random", "nucleus_gradient", "membrane_gradient", "apical_gradient"]

# Base channel colors (nucleus and membrane)
base_channel_colors = {
    "nucleus": (0.18, 0.32, 0.32),
    "membrane": (0.31, 0.19, 0.31),
}
# %%
records = []
s3_client = boto3.client("s3")

# %%
# Set to True to reupload and overwrite existing files on s3
reupload_simularium_files = True
reupload_thumbnails = True

# %%
# Collect all files to process
files_to_process = []
for structure_config in structures:
    structure_id = structure_config["structure_id"]
    structure_name = structure_config["structure_name"]
    packing_id = structure_config["packing_id"]
    structure_color = structure_config["color"]

    # Create channel colors with the correct structure color (copy for thread safety)
    channel_colors = {**base_channel_colors, "structure": structure_color}

    # Create structure-specific mesh URL
    structure_mesh_url = f"{BASE_S3_URL}structure_data/{structure_id}/meshes/"

    for rule in rules:
        simularium_path = (
            base_datadir
            / "packing_outputs"
            / dataset
            / experiment
            / rule
            / packing_id
            / "spheresSST"
        )
        figure_path = simularium_path / "figures"

        for file_path in simularium_path.rglob("*.simularium"):
            files_to_process.append(
                {
                    "file_path": file_path,
                    "structure_id": structure_id,
                    "structure_name": structure_name,
                    "packing_id": packing_id,
                    "rule": rule,
                    "figure_path": figure_path,
                    "channel_colors": channel_colors.copy(),  # Copy for thread safety
                    "mesh_url": structure_mesh_url,
                }
            )

# Process files in parallel using ThreadPoolExecutor
max_workers = 8  # Adjust based on your system and S3 rate limits
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(
            process_simularium_file,
            file_info["file_path"],
            file_info["structure_id"],
            file_info["structure_name"],
            file_info["packing_id"],
            file_info["rule"],
            dataset,
            condition,
            experiment,
            base_datadir,
            file_info["mesh_url"],
            file_info["figure_path"],
            thumbnail_dir,
            file_info["channel_colors"],
            s3_client,
            reupload_simularium_files,
            reupload_thumbnails,
        ): file_info
        for file_info in files_to_process
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        try:
            record = future.result()
            records.append(record)
        except Exception as e:
            file_info = futures[future]
            print(f"Error processing {file_info['file_path']}: {e}")
# %%
df = pd.DataFrame.from_records(records)
df.to_csv(csv_output_dir / "cellpack_simularium_paths.csv", index=False)

# %%
# Construct metadata CSV for BFF
metadata_dict = get_metadata_dict()
metadata_df = pd.DataFrame(list(metadata_dict.items()), columns=["Column Name", "Description"])
metadata_df.to_csv(csv_output_dir / "cellpack_simularium_metadata.csv", index=False)

# # %%
# df = pd.read_csv(csv_output_dir / "cellpack_simularium_paths.csv")
# df["File Name"] = df.apply(
#     lambda row: f"{row['Packing ID']}_{row['Rule']}_{row['Cell ID']}", axis=1
# )
# # %%
# df.to_csv(csv_output_dir / "cellpack_simularium_paths.csv", index=False)

# # %%
