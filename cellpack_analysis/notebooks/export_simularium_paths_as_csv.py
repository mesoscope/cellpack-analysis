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

import boto3
import pandas as pd
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path, get_results_path

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
# %%
records = []
base_s3_mesh_url = (
    "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/structure_data/SLC25A17/meshes/"
)
base_s3_simularium_url = "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/"
s3_client = boto3.client("s3")
# %%
# Update meshes in s3 bucket
mesh_path = base_datadir / "structure_data" / structure_id / "meshes"
for file_path in tqdm(mesh_path.rglob("*")):
    if file_path.is_file():
        s3_key = file_path.relative_to(base_datadir).as_posix()
        s3_client.upload_file(str(file_path), "cellpack-analysis-data", s3_key)
        s3_client.put_object_acl(
            ACL="public-read",
            Bucket="cellpack-analysis-data",
            Key=s3_key,
        )
# %%
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
    # try:
    #     s3_client.head_object(Bucket="cellpack-analysis-data", Key=s3_key)
    # except s3_client.exceptions.ClientError:
    # upload to s3
    s3_key = file_path.relative_to(base_datadir).as_posix()
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
