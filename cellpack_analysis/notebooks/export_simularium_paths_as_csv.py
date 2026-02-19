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

import boto3
import pandas as pd
from bioio_conversion.converters import OmeZarrConverter
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path, get_results_path

# %%
base_datadir = get_datadir_path()
base_results_dir = get_results_path()

# %%
csv_output_dir = base_results_dir / "data_release" / "test_bff"
csv_output_dir.mkdir(parents=True, exist_ok=True)

thumbnail_dir = csv_output_dir / "thumbnails"
thumbnail_dir.mkdir(parents=True, exist_ok=True)

# %%
structure_id = "SLC25A17"
structure_name = "peroxisome"
packing_id = "peroxisome"
dataset = "8d_sphere_data"
condition = "rules_shape"
experiment = "norm_weights"
rules = ["random", "nucleus_gradient", "membrane_gradient", "apical_gradient"]
# %%
records = []
base_s3_mesh_url = (
    "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/structure_data/SLC25A17/meshes/"
)
base_s3_simularium_url = "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/"
s3_client = boto3.client("s3")

# %%
reupload = False
# Update simularium files and upload to s3
for rule in rules:
    simularium_path = (
        base_datadir / "packing_outputs" / dataset / experiment / rule / packing_id / "spheresSST"
    )
    figure_path = simularium_path / "figures"
    for file_path in tqdm(simularium_path.rglob("*.simularium")):

        cell_id = file_path.stem.split("_seed")[0].split("_")[-1]

        # update mesh paths in simularium file
        with open(file_path, "r") as f:
            sim_data = json.load(f)
        for index, mapping in sim_data["trajectoryInfo"]["typeMapping"].items():
            if mapping["name"] in ["nucleus", "membrane", "structure"]:
                mapping["geometry"]["url"] = f"{base_s3_mesh_url}{mapping['geometry']['url']}"
        with open(file_path, "w") as f:
            json.dump(sim_data, f, indent=2)

        simularium_s3_key = file_path.relative_to(base_datadir).as_posix()

        should_upload = reupload
        if not reupload:
            try:
                s3_client.head_object(Bucket="cellpack-analysis-data", Key=simularium_s3_key)
                should_upload = False
            except s3_client.exceptions.ClientError:
                should_upload = True

        if should_upload:
            s3_client.upload_file(str(file_path), "cellpack-analysis-data", simularium_s3_key)
            s3_client.put_object_acl(
                ACL="public-read",
                Bucket="cellpack-analysis-data",
                Key=simularium_s3_key,
            )

        simularium_s3_path = f"{base_s3_simularium_url}{simularium_s3_key}"

        # Generate thumbnail from figure path (use MIP)
        figure_file_glob = f"voxelized_image_{packing_id}_{condition}_{rule}_{cell_id}_*.ome.tiff"
        figure_file_path = next(figure_path.glob(figure_file_glob), None)
        if figure_file_path is not None:
            thumbnail_path = thumbnail_dir / f"{file_path.stem}_thumbnail.ome.zarr"
            if not thumbnail_path.exists():
                conv = OmeZarrConverter(
                    source=figure_file_path.as_posix(),
                    destination=thumbnail_path.as_posix(),
                )
                conv.convert()
            # upload thumbnail to s3
            thumbnail_s3_key = (
                file_path.relative_to(base_datadir).parent.as_posix()
                + f"/thumbnails/{thumbnail_path.name}"
            )
            # Recursively upload zarr directory
            for local_file_path in thumbnail_path.rglob("*"):
                if local_file_path.is_file():
                    relative_path = local_file_path.relative_to(thumbnail_path.parent).as_posix()
                    s3_key = thumbnail_s3_key.rstrip("/") + "/" + relative_path
                    s3_client.upload_file(str(local_file_path), "cellpack-analysis-data", s3_key)
                    s3_client.put_object_acl(
                        ACL="public-read",
                        Bucket="cellpack-analysis-data",
                        Key=s3_key,
                    )
            thumbnail_s3_path = f"{base_s3_simularium_url}{thumbnail_s3_key}"
        else:
            thumbnail_s3_path = ""

        record = {
            "File Path": str(simularium_s3_path),
            "Cell ID": cell_id,
            "File Name": file_path.stem,
            "Structure ID": structure_id,
            "Structure Name": structure_name,
            "Packing ID": packing_id,
            "Rule": rule,
            "Dataset": dataset,
            "Condition": condition,
            "Experiment": experiment,
            "File Type": "simularium",
            "Thumbnail Path": thumbnail_s3_path,
        }
        records.append(record)
# %%
df = pd.DataFrame.from_records(records)
df.to_csv(csv_output_dir / "simularium_paths.csv", index=False)

# %%
# Construct metadata CSV for BFF
metadata_dict = {
    "File Path": "Path to simularium file on s3",
    "Cell ID": "Unique identifier for each cell/packing output",
    "File Name": "Name of the simularium file",
    "Structure ID": "Identifier for the structure (e.g. SLC25A17)",
    "Structure Name": "Name of the structure (e.g. peroxisome)",
    "Rule": "Packing rule used (e.g. random, nucleus_gradient, etc.)",
    "Dataset": "Dataset name (e.g. 8d_sphere_data)",
    "Condition": "Experimental condition (e.g. norm_weights)",
    "File Type": "Type of file (e.g. simularium)",
    "Thumbnail Path": "Path to thumbnail file on s3",
}
metadata_df = pd.DataFrame(list(metadata_dict.items()), columns=["Column Name", "Description"])
metadata_df.to_csv(csv_output_dir / "simularium_metadata.csv", index=False)

# %%
