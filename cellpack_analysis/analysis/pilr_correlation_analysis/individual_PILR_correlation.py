# %%
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from scipy.stats import pearsonr
from tqdm import tqdm

from cellpack_analysis.utilities.PILR_tools import get_cell_id_list
from cellpack_analysis.utilities.plotting_tools import save_PILR_image


# %%
def load_individual_pilr_dict(base_folder):
    print(f"Loading PILR data from {base_folder}")
    with open(base_folder / "individual_PILR.json") as f:
        individual_PILR_dict = json.load(f)
    for ch in individual_PILR_dict:
        individual_PILR_dict[ch] = np.array(individual_PILR_dict[ch])
    return individual_PILR_dict


# %% Get cell ids
def get_cell_id_list_for_struct(struct):
    datadir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data")
    df_path = datadir / "8dsphere_ids.csv"
    df_cellID = pd.read_csv(df_path)
    df_cellID.set_index("structure", inplace=True)
    cellid_list = get_cell_id_list(df_cellID, struct)
    return cellid_list


# %% [markdown]
# ### Save individual PILR images
def save_individual_PILR_images(individual_PILR_dict, cellid_list, base_folder):
    save_dir = base_folder / "individual_PILR/"
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = OmeTiffWriter()
    for ch, pilr_list in individual_PILR_dict.items():
        print(f"Saving individual pilr images for {ch}")
        for cellid, pilr in tqdm(zip(cellid_list, pilr_list), total=len(cellid_list)):
            cellid = str(cellid)
            # save pilr as ome.tiff
            writer.save(
                pilr.astype(np.float32),
                str(save_dir / f"{ch}_{cellid}.ome.tiff"),
            )
            save_PILR_image(
                pilr,
                save_dir=save_dir,
                label=f"{ch}_{cellid}",
                vmin=0,
                vmax=1,
            )


# %% create new dataframe
def create_df_for_correlations(individual_PILR_dict, cellid_list):
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [individual_PILR_dict.keys(), cellid_list], names=["channel", "cellid"]
        ),
        columns=pd.MultiIndex.from_product(
            [individual_PILR_dict.keys(), cellid_list], names=["channel", "cellid"]
        ),
    )
    print("Created new dataframe for correlations")
    return df


# %% [markdown]
# #### try to load previously calculated correlations
def initialize_corr_df(individual_PILR_dict, cellid_list, base_folder, create_new=True):
    if create_new:
        df = create_df_for_correlations(individual_PILR_dict, cellid_list)
    else:
        df_path = base_folder / "individual_PILR_corr.csv"
        if df_path.exists() and not create_new:
            df = pd.read_csv(
                base_folder / "individual_PILR_corr.csv",
                index_col=[0, 1],
                header=[0, 1],
            )
            # set 0 values to nan
            df[df == 0] = np.nan
            print("Loaded previously calculated correlations")
        else:
            print("No previously calculated correlations found, creating new dataframe")
            df = create_df_for_correlations(individual_PILR_dict, cellid_list)

    df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
    df.columns = df.columns.set_levels(df.columns.levels[1].astype(str), level=1)

    return df


# %% [markdown]
# #### calculate cross correlations between pilrs
def calculate_cross_correlations(individual_PILR_dict, cellid_list, base_folder):
    df = initialize_corr_df(
        individual_PILR_dict, cellid_list, base_folder, create_new=True
    )
    cellid_list_str = [str(cellid) for cellid in cellid_list]
    for ch1, pilr_list1 in individual_PILR_dict.items():
        for ch2, pilr_list2 in individual_PILR_dict.items():
            print(f"Processing {ch1}, {ch2}")
            for cellid1, pilr1 in tqdm(
                zip(cellid_list_str, pilr_list1), total=len(cellid_list_str)
            ):
                masked_pilr1 = pilr1[(pilr1.shape[0] // 2) :, :].flatten()
                for cellid2, pilr2 in zip(cellid_list_str, pilr_list2):
                    if df.loc[(ch1, cellid1), (ch2, cellid2)] is np.nan:
                        if (ch1 == ch2) and (cellid1 == cellid2):
                            df.loc[(ch1, cellid1), (ch2, cellid2)] = 1
                            df.loc[(ch2, cellid2), (ch1, cellid1)] = 1
                        else:
                            masked_pilr2 = pilr2[(pilr2.shape[0] // 2) :, :].flatten()
                            corr_val = pearsonr(masked_pilr1, masked_pilr2)[0]
                            df.loc[(ch1, cellid1), (ch2, cellid2)] = corr_val
                            df.loc[(ch2, cellid2), (ch1, cellid1)] = corr_val
        df.to_csv(base_folder / "individual_PILR_corr.csv")
    df.to_csv(base_folder / "individual_PILR_corr.csv")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PILR correlations")

    parser.add_argument(
        "--structure_id",
        type=str,
        help="Structure id",
    )
    parser.add_argument(
        "--structure_name",
        type=str,
        help="Structure name",
    )
    parser.add_argument(
        "--base_folder",
        default="/allen/aics/animated-cell/Saurabh/cellpack-analysis/results/SLC25A17/RS",
        type=str,
        help="Base folder",
    )
    parser.add_argument(
        "--save_individual_images",
        action="store_true",
        default=False,
        help="Save individual images",
    )
    parser.add_argument(
        "--get_correlations",
        action="store_true",
        default=False,
        help="Get correlations",
    )

    args = parser.parse_args()
    base_folder = Path(args.base_folder)
    base_folder.mkdir(parents=True, exist_ok=True)

    individual_pilr_dict = load_individual_pilr_dict(base_folder)
    cellid_list = get_cell_id_list_for_struct(args.structure_id)

    if args.save_individual_images:
        save_individual_PILR_images(individual_pilr_dict, cellid_list, base_folder)

    if args.get_correlations:
        calculate_cross_correlations(individual_pilr_dict, cellid_list, base_folder)
