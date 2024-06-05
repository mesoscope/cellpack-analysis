# %%
import pandas as pd
import quilt3
from pathlib import Path
from tqdm.notebook import tqdm

# %% [markdown]
# ### Load variance dataset from quilt and save locally

# %%
pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")

# %%
meta_df = pkg["metadata.csv"]()
meta_df.set_index("CellId", inplace=True)

# %%
meta_df.to_csv("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/variance_dataset.csv")

# %%
meta_df = pd.read_csv("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/variance_dataset.csv", index_col="CellId")

# %%
print(meta_df.structure_name.unique())

# %% [markdown]
# ### Set structure of interest
# - `SLC25A17` is peroxisomes
# - `RAB5A` is early endosomes

# %%
structure_id = "SLC25A17"

# %%
struct_data = meta_df[meta_df["structure_name"] == structure_id]

# %% [markdown]
# ### Get cellIDs within 8D sphere in shape space (cells shaped close to average)

# %%
df_cellID_path = "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/8dsphere_ids.csv"

# %%
df_cellID = pd.read_csv(df_cellID_path)

# %%
df_cellID.set_index("structure", inplace=True)

# %%
str_cellid = df_cellID.loc[structure_id, "CellIds"].split(",")

# %%
cellid_list = []
for cellid in str_cellid:
    cellid_list.append(int(cellid.replace("[", "").replace("]", "")))


# %%
print(*cellid_list)

# %% [markdown]
# ### Select cellIDs in 8d sphere from the dataframe

# %%
data = struct_data[struct_data.index.isin(cellid_list)].reset_index()
data.shape

# %% [markdown]
# ### Alternatively select all the cellIDs

# %%
data = struct_data.reset_index()
data.structure_name.unique()
data.shape

# %% [markdown]
# ### Prepare file paths

# %%
save_path = Path(f"/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/structure_data/{structure_id}/sample_8d/")
save_path.mkdir(exist_ok=True, parents=True)
raw_path = save_path / Path("unsegmented_imgs")
raw_path.mkdir(exist_ok=True, parents=True)

# %%
for row in tqdm(data.itertuples()):
    subdir_name = row.crop_raw.split("/")[0]
    file_name = row.crop_raw.split("/")[1]
    local_fn = raw_path / f"{row.structure_name}_{row.CellId}_ch_{row.ChannelNumberStruct}_crop_seg_original.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)
print("Done")


