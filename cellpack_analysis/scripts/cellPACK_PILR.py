# Import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path
import pandas as pd
import seaborn as sns
import json

from cellpack_analysis.utilities.plotting_tools import save_PILR_image
from cellpack_analysis.utilities.PILR_tools import (
    average_over_dimension,
    get_pilr_for_single_image,
)

import argparse

import multiprocessing
import concurrent.futures

# This was used for the original SAC talk 2023
# CHANNEL_NAME_DICT = {
#     "SLC25A17": "Peroxisome",
#     "membrane": "Membrane",
#     "random": "Random",
#     "nucleus_cube": "Nucleus Cube",
#     "nucleus_linear": "Nucleus Linear",
#     "nucleus_square": "Nucleus Square",
#     "nucleus_twelve": "Nucleus Twelve",
# }

RAW_IMAGE_CHANNEL = "RAB5A"

# Newer packings in September 2023
CHANNEL_NAME_DICT = {
    "SLC25A17": "Peroxisome",
    "RAB5A": "Endosome",
    "random": "Random",
    "membrane_moderate": "Membrane Moderate",
    "nucleus_moderate": "Nucleus Moderate",
    # "membrane_moderate_invert": "Membrane Moderate inverted",
    # "nucleus_moderate_invert": "Nucleus Moderate inverted",
    "planar_gradient_Z": "Planar Gradient Z",
}


def test_io():
    # Test io
    image_path = Path("./raw_images_for_PILR/")

    with open("./results/filenames.txt", "w") as f:
        for file in image_path.glob("*.tiff"):
            f.write(f"{file.name}\n")


def run_PILR_analysis(
    raw_image_path=Path("./raw_images_for_PILR/"),
    simulated_image_path=Path("./simulated_images_for_PILR/"),
    num_cores=80,
    save_dir="./results",
    raw_image_channel=RAW_IMAGE_CHANNEL,
    channel_name_dict=CHANNEL_NAME_DICT,
):
    channel_names = list(channel_name_dict.keys())

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Run PILR on all the images
    individual_pilr_dict = {}
    average_pilr_dict = {}

    mpl.use("agg")

    for ch_name in channel_names:
        print(f"Processing {ch_name} channel")
        if ch_name == raw_image_channel:
            image_path = raw_image_path
        else:
            image_path = simulated_image_path

        file_list = []
        for file in image_path.glob(f"*{ch_name}*.tiff"):
            if "invert" in file.name and "invert" not in ch_name:
                continue
            file_list.append(file)

        num_files = len(file_list)

        if num_files == 0:
            print(f"No files found for {ch_name} channel")
            continue
        gfp_representations = []
        if num_cores > 1:
            num_processes = np.min(
                [
                    num_cores,
                    int(np.floor(0.9 * multiprocessing.cpu_count())),
                    num_files,
                ]
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_processes
            ) as executor:
                for file, gfp_representation in tqdm(
                    zip(
                        file_list,
                        executor.map(
                            get_pilr_for_single_image,
                            file_list,
                            [ch_name] * num_files,
                            [raw_image_channel] * num_files,
                        ),
                    ),
                    total=num_files,
                ):
                    # print("Processing file", file, "with channel", ch_name)
                    gfp_representations.append(gfp_representation)

        else:
            for file in tqdm(file_list, total=num_files):
                img_pilr = get_pilr_for_single_image(file, ch_name)
                gfp_representations.append(img_pilr)
        individual_pilr_dict[ch_name] = np.array(gfp_representations)

        average_pilr_dict[ch_name] = np.mean(
            individual_pilr_dict[ch_name], axis=0
        )  # Average over all the images

        # Save the PILR as an image
        avg_gfp = average_pilr_dict[ch_name]
        if ch_name == raw_image_channel:
            vmin, vmax = np.percentile(avg_gfp, [1, 90])
        else:
            vmin, vmax = None, None
        save_PILR_image(
            avg_gfp,
            ch_name,
            save_dir=save_dir,
            suffix="",
            aspect=20,
            vmin=vmin,
            vmax=vmax,
        )

        for ind, dim in enumerate(["R", "phi", "theta"]):
            dim_dir = save_dir / dim
            dim_dir.mkdir(exist_ok=True)
            avg_dim = average_over_dimension(avg_gfp, dim=ind)
            save_PILR_image(
                avg_dim,
                ch_name,
                save_dir=dim_dir,
                suffix=f"_{dim}",
                aspect=1,
                vmin=vmin,
                vmax=vmax,
            )

    # Save the PILR as json
    print("Saving average PILR")
    write_pilr_avg = {}
    for ch in average_pilr_dict:
        write_pilr_avg[ch] = average_pilr_dict[ch].astype(np.float16).tolist()

    with open(save_dir / "avg_PILR.json", "w") as f:
        json.dump(write_pilr_avg, f, indent=4)

    # Save individual PILRs
    print("Saving individual PILR")
    write_pilr_ind = {}
    for ch in individual_pilr_dict:
        write_pilr_ind[ch] = individual_pilr_dict[ch].astype(int).tolist()

    with open(save_dir / "individual_PILR.json", "w") as f:
        json.dump(write_pilr_ind, f, indent=4)

    # Get correlations between average PILRs
    df = pd.DataFrame(
        index=channel_name_dict.values(),
        columns=channel_name_dict.values(),
        dtype=float,
    )

    for ch_ind, ch_name in channel_name_dict.items():
        if ch_ind not in average_pilr_dict:
            continue
        pilr1 = average_pilr_dict[ch_ind].ravel()
        pilr1 = pilr1 / np.max(pilr1)
        for ch_ind2, ch_name2 in channel_name_dict.items():
            if ch_ind2 not in average_pilr_dict:
                continue
            pilr2 = average_pilr_dict[ch_ind2].ravel()
            pilr2 = pilr2 / np.max(pilr2)
            df.loc[ch_name, ch_name2] = np.corrcoef(pilr1, pilr2)[0, 1]

    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.close("all")

    # exp_channel_name = channel_name_dict[raw_image_channel]
    # exp_vals = df.loc[exp_channel_name].values
    # if len(exp_vals):
    #     min_val = np.min(exp_vals[exp_vals > 0])
    #     max_val = np.max(exp_vals[exp_vals < 0.9])
    # else:
    #     min_val = 0
    #     max_val = 1

    sns.heatmap(
        df,
        annot=True,
        cmap="cool",
        # vmin=min_val,
        # vmax=max_val,
        mask=mask,
    )
    plt.savefig(
        save_dir / "PILR_correlation_bias.png",
        dpi=300,
        bbox_inches="tight",
    )


def test_parser(args):
    print(args)
    print(args.raw_image_path)
    print(args.simulated_image_path)
    print(args.num_cores)
    print(args.save_dir)
    print(args.raw_image_channel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PILR analysis on cellPACK output")

    parser.add_argument(
        "--raw_image_path",
        "-r",
        type=str,
        default="./raw_images_for_PILR/",
        help="Path to raw images",
    )

    parser.add_argument(
        "--simulated_image_path",
        "-s",
        type=str,
        default="./simulated_images_for_PILR/",
        help="Path to simulated images",
    )

    parser.add_argument(
        "--num_cores",
        "-n",
        type=int,
        default=80,
        help="Number of cores to use for PILR calculation",
    )

    parser.add_argument(
        "--save_dir",
        "-d",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--raw_image_channel",
        "-c",
        type=str,
        default="SLC25A17",
        help="Channel name for raw images",
    )

    args = parser.parse_args()

    # test parser
    # test_parser(args)

    # Run PILR analysis
    run_PILR_analysis(
        raw_image_path=Path(args.raw_image_path),
        simulated_image_path=Path(args.simulated_image_path),
        num_cores=args.num_cores,
        save_dir=Path(args.save_dir),
        raw_image_channel=args.raw_image_channel,
    )
