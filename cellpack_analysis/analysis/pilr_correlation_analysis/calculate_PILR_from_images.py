# Import required packages
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path

import matplotlib as mpl
import numpy as np
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from tqdm import tqdm

from cellpack_analysis.utilities.PILR_tools import (
    average_over_dimension,
    get_pilr_for_single_image,
)
from cellpack_analysis.utilities.plotting_tools import (
    save_PILR_as_tiff,
    save_PILR_image,
)

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
    "nucleus_moderate": "Nucleus Moderate",
    "membrane_moderate": "Membrane Moderate",
    # "membrane_moderate_invert": "Membrane Moderate inverted",
    # "nucleus_moderate_invert": "Nucleus Moderate inverted",
    "planar_gradient_Z": "Planar Gradient Z",
    # "planar_gradient_Y": "Planar Gradient Y",
}


def PILR_calculation_workflow(
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
        gfp_representations = []
        for file in image_path.glob(f"*{ch_name}*.tiff"):
            if "invert" in file.name and "invert" not in ch_name:
                continue
            file_list.append(file)

        num_files = len(file_list)

        if num_files == 0:
            print(f"No files found for {ch_name} channel")
            continue

        writer = OmeTiffWriter()
        individual_pilr_dir = save_dir / "individual_PILR"
        individual_pilr_dir.mkdir(exist_ok=True, parents=True)
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
                    save_PILR_as_tiff(
                        gfp_representation,
                        individual_pilr_dir,
                        f"{file.stem.split('.')[0]}_pilr",
                        writer,
                    )

        else:
            for file in tqdm(file_list, total=num_files):
                img_pilr = get_pilr_for_single_image(file, ch_name)
                gfp_representations.append(img_pilr)
        individual_pilr_dict[ch_name] = np.array(gfp_representations)

        average_pilr_dict[ch_name] = np.mean(
            individual_pilr_dict[ch_name], axis=0
        )  # Average over all the images

        # Save the average PILR as an image
        avg_gfp = average_pilr_dict[ch_name]
        save_PILR_image(
            avg_gfp,
            ch_name,
            save_dir=save_dir,
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

    # Run PILR analysis
    PILR_calculation_workflow(
        raw_image_path=Path(args.raw_image_path),
        simulated_image_path=Path(args.simulated_image_path),
        num_cores=args.num_cores,
        save_dir=Path(args.save_dir),
        raw_image_channel=args.raw_image_channel,
    )
