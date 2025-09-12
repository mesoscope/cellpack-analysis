# %%
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from scipy.stats import pearsonr
from tqdm import tqdm

from cellpack_analysis.lib.plotting_tools import plot_PILR


# %%
def load_individual_pilr_dict(base_folder):
    print(f"Loading PILR data from {base_folder}")
    with open(base_folder / "individual_PILR.json") as f:
        individual_PILR_dict = json.load(f)
    for ch in individual_PILR_dict:
        individual_PILR_dict[ch] = np.array(individual_PILR_dict[ch])
    return individual_PILR_dict


# %% [markdown]
# ### Save individual PILR images
def save_individual_PILR_images(individual_PILR_dict, cell_id_list, base_folder):
    save_dir = base_folder / "individual_PILR/"
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = OmeTiffWriter()
    for ch, pilr_list in individual_PILR_dict.items():
        print(f"Saving individual pilr images for {ch}")
        for cell_id, pilr in tqdm(
            zip(cell_id_list, pilr_list, strict=False), total=len(cell_id_list)
        ):
            cell_id = str(cell_id)
            # save pilr as ome.tiff
            writer.save(
                pilr.astype(np.float32),
                str(save_dir / f"{ch}_{cell_id}.ome.tif"),
            )
            plot_PILR(
                pilr,
                save_dir=save_dir,
                label=f"{ch}_{cell_id}",
                vmin=0,
                vmax=1,
            )


# %% [markdown]
# #### try to load previously calculated correlations
def initialize_corr_df(individual_PILR_dict, cell_id_list, base_folder):
    try:
        df = pd.read_csv(base_folder / "individual_PILR_corr.csv", index_col=[0, 1], header=[0, 1])
        # set 0 values to nan
        df[df == 0] = np.nan
        print("Loaded previously calculated correlations")
    except FileNotFoundError:
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [individual_PILR_dict.keys(), cell_id_list], names=["channel", "cell_id"]
            ),
            columns=pd.MultiIndex.from_product(
                [individual_PILR_dict.keys(), cell_id_list], names=["channel", "cell_id"]
            ),
        )
        print("Created new dataframe for correlations")
    df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
    df.columns = df.columns.set_levels(df.columns.levels[1].astype(str), level=1)

    return df


# %% [markdown]
# #### calculate cross correlations between pilrs
def calculate_cross_correlations(individual_PILR_dict, cell_id_list, base_folder):
    df = initialize_corr_df(individual_PILR_dict, cell_id_list, base_folder)
    cell_id_list_str = [str(cell_id) for cell_id in cell_id_list]
    for ch1, pilr_list1 in individual_PILR_dict.items():
        for ch2, pilr_list2 in individual_PILR_dict.items():
            print(f"Processing {ch1}, {ch2}")
            for cell_id1, pilr1 in tqdm(
                zip(cell_id_list_str, pilr_list1, strict=False), total=len(cell_id_list_str)
            ):
                pilr1 = pilr1[pilr1.shape[0] // 2 :, :].ravel()
                for cell_id2, pilr2 in zip(cell_id_list_str, pilr_list2, strict=False):
                    if df.loc[(ch1, cell_id1), (ch2, cell_id2)] is np.nan:
                        if ch1 == ch2 and cell_id1 == cell_id2:
                            df.loc[(ch1, cell_id1), (ch2, cell_id2)] = 1
                            df.loc[(ch2, cell_id2), (ch1, cell_id1)] = 1
                            continue
                        pilr2 = pilr2[pilr2.shape[0] // 2 :, :].ravel()
                        corr_val = pearsonr(pilr1, pilr2)[0]
                        df.loc[(ch1, cell_id1), (ch2, cell_id2)] = corr_val
                        df.loc[(ch2, cell_id2), (ch1, cell_id1)] = corr_val
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

    args = parser.parse_args()
    base_folder = Path(args.base_folder)
    base_folder.mkdir(parents=True, exist_ok=True)

    individual_pilr_dict = load_individual_pilr_dict(base_folder)
    cell_id_list = get_cell_id_list_for_struct(args.structure_id)

    if args.save_individual_images:
        save_individual_PILR_images(individual_pilr_dict, cell_id_list, base_folder)

    calculate_cross_correlations(individual_pilr_dict, cell_id_list, base_folder)
