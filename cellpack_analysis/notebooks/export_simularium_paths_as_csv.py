# %%
# # cellPACK BFF exporter
# Test notebook to export simularium paths as CSV files
# Need to use s3 uploaded data
# Steps:
# 1. Update simularium files to use mesh paths from s3
# 2. Fix meshes in s3 bucket (if needed)
# 2. Upload simularium outputs to s3
# 3. Create metadata CSV for BFF (optional)
# 4. Create thumbnails? (tricky, maybe use MIP of tiff outputs)
# 5. Export s3 file paths as CSV for put into BFF


# TODO: Thumbnails?
# TODO: inverted meshes?
# %%
import json
import shutil

import boto3
import pandas as pd
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path, get_results_path
from cellpack_analysis.lib.mesh_tools import invert_mesh_faces

# %%
base_datadir = get_datadir_path()
base_results_dir = get_results_path()

# %%
csv_output_dir = base_results_dir / "data_release" / "test_bff"
csv_output_dir.mkdir(parents=True, exist_ok=True)

# %%
structure_id = "SLC25A17"
structure_name = "peroxisome"
dataset = "8d_sphere_data"
condition = "norm_weights"
rule = "random"
simularium_path = (
    base_datadir / "packing_outputs" / dataset / condition / rule / structure_name / "spheresSST"
)
figure_path = simularium_path / "figures"
# %%
records = []
base_s3_mesh_url = (
    "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/structure_data/SLC25A17/meshes/"
)
base_s3_simularium_url = "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/"
s3_client = boto3.client("s3")
# %%
# Invert and upload meshes to s3 bucket
invert_meshes = False
mesh_dir = base_datadir / "structure_data" / structure_id / "meshes"
for file_path in tqdm(mesh_dir.rglob("*")):
    if file_path.is_file():
        # invert mem meshes
        if invert_meshes and "mem" in file_path.name and "original" not in file_path.name:
            orig_mesh_path = file_path.parent / f"{file_path.stem}_original.obj"
            # copy original file with suffix
            shutil.copy(
                file_path,
                orig_mesh_path,
            )
            invert_mesh_faces(input_mesh_path=orig_mesh_path, output_mesh_path=file_path)

        # upload to s3
        s3_key = file_path.relative_to(base_datadir).as_posix()
        s3_client.upload_file(str(file_path), "cellpack-analysis-data", s3_key)
        s3_client.put_object_acl(
            ACL="public-read",
            Bucket="cellpack-analysis-data",
            Key=s3_key,
        )
# %%
reupload = False
# Update simularium files and upload to s3
for file_path in tqdm(simularium_path.rglob("*.simularium")):
    # update mesh paths in simularium file
    with open(file_path, "r") as f:
        sim_data = json.load(f)
    for index, mapping in sim_data["trajectoryInfo"]["typeMapping"].items():
        if mapping["name"] in ["nucleus", "membrane", "structure"]:
            mapping["geometry"]["url"] = f"{base_s3_mesh_url}{mapping['geometry']['url']}"
    with open(file_path, "w") as f:
        json.dump(sim_data, f, indent=2)

    s3_key = file_path.relative_to(base_datadir).as_posix()

    should_upload = reupload
    if not reupload:
        try:
            s3_client.head_object(Bucket="cellpack-analysis-data", Key=s3_key)
            should_upload = False
        except s3_client.exceptions.ClientError:
            should_upload = True

    if should_upload:
        s3_client.upload_file(str(file_path), "cellpack-analysis-data", s3_key)
        s3_client.put_object_acl(
            ACL="public-read",
            Bucket="cellpack-analysis-data",
            Key=s3_key,
        )

    s3_path = f"{base_s3_simularium_url}{s3_key}"
    record = {
        "File Path": str(s3_path),
        "File Type": "simularium",
        "Structure ID": structure_id,
        "Structure Name": structure_name,
        "Dataset": dataset,
        "Condition": condition,
        "Rule": rule,
    }
    records.append(record)
# %%
df = pd.DataFrame.from_records(records)
df.to_csv(csv_output_dir / "simularium_paths.csv", index=False)

# %%
