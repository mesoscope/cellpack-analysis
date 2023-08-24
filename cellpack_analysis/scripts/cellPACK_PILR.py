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

import multiprocessing
import concurrent.futures


def test_io():
    # Test io
    image_path = Path("./raw_imgs_for_PILR/")

    with open("./results/filenames.txt", "w") as f:
        for file in image_path.glob("*.tiff"):
            f.write(f"{file.name}\n")


def run_PILR_analysis(
    image_path=Path("./raw_imgs_for_PILR/"),
    ch_names=[
        "SLC25A17",
        "nucleus_cube",
        "random",
        "membrane",
        "nucleus_linear",
        "nucleus_square",
        "nucleus_twelve",
    ],
    parallel=True,
    save_dir="./results"
):
    # Run PILR on all the images
    ch_pilrs = {}
    avg_pilr = {}

    mpl.use("agg")

    for ch_name in ch_names:
        print(f"Processing {ch_name} channel")
        num_files = len([file.name for file in image_path.glob(f"*{ch_name}*.tiff")])
        if num_files == 0:
            print(f"No files found for {ch_name} channel")
            continue
        gfp_representations = []
        if parallel:
            num_processes = np.min(
                [
                    int(np.floor(0.9 * multiprocessing.cpu_count())),
                    num_files,
                ]
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_processes
            ) as executor:
                for file, gfp_representation in tqdm(
                    zip(
                        image_path.glob(f"*{ch_name}*.tiff"),
                        executor.map(
                            get_pilr_for_single_image,
                            image_path.glob(f"*{ch_name}*.tiff"),
                            [ch_name] * num_files,
                        ),
                    ),
                    total=num_files,
                ):
                    gfp_representations.append(gfp_representation)

        else:
            for file in tqdm(image_path.glob(f"*{ch_name}*.tiff"), total=num_files):
                img_pilr = get_pilr_for_single_image(file, ch_name)
                gfp_representations.append(img_pilr)
        ch_pilrs[ch_name] = np.array(gfp_representations)

        avg_pilr[ch_name] = np.mean(
            ch_pilrs[ch_name], axis=0
        )  # Average over all the images

        # Save the PILR as an image
        avg_gfp = avg_pilr[ch_name]
        save_PILR_image(avg_gfp, ch_name, save_dir=save_dir, suffix="", aspect=20)

        for ind, dim in enumerate(["R", "phi", "theta"]):
            avg_dim = average_over_dimension(avg_gfp, dim=ind)
            save_PILR_image(
                avg_dim,
                ch_name,
                save_dir="./results",
                suffix=f"_{dim}",
                aspect=1,
            )

    # Save the PILR as json
    print("Saving average PILR")
    write_pilr_avg = {}
    for ch in avg_pilr:
        write_pilr_avg[ch] = avg_pilr[ch].tolist()

    with open("./results/avg_PILR.json", "w") as f:
        json.dump(write_pilr_avg, f, indent=4)

    # Save individual PILRs
    print("Saving individual PILR")
    write_pilr_ind = {}
    for ch in ch_pilrs:
        write_pilr_ind[ch] = ch_pilrs[ch].astype(int).tolist()

    with open("./results/individual_PILR.json", "w") as f:
        json.dump(write_pilr_ind, f, indent=4)

    # Get correlations between average PILRs
    ch_names_dict = {
        "SLC25A17": "Peroxisome",
        "membrane": "Membrane",
        "random": "Random",
        "nucleus_cube": "Nucleus Cube",
        "nucleus_linear": "Nucleus Linear",
        "nucleus_square": "Nucleus Square",
        "nucleus_twelve": "Nucleus Twelve",
    }

    df = pd.DataFrame(
        index=ch_names_dict.values(), columns=ch_names_dict.values(), dtype=float
    )

    for ch_ind, ch_name in ch_names_dict.items():
        if ch_ind not in avg_pilr:
            continue
        pilr1 = avg_pilr[ch_ind].ravel()
        # pilr1 = pilr1 / np.max(pilr1)
        for ch_ind2, ch_name2 in ch_names_dict.items():
            if ch_ind2 not in avg_pilr:
                continue
            pilr2 = avg_pilr[ch_ind2].ravel()
            # pilr2 = pilr2 / np.max(pilr2)
            df.loc[ch_name, ch_name2] = np.corrcoef(pilr1, pilr2)[0, 1]

    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.close("all")
    pex_vals = df.loc["Peroxisome"].values
    min_val = np.min(pex_vals[pex_vals > 0])
    max_val = np.max(pex_vals[pex_vals < 0.9])
    sns.heatmap(
        df,
        annot=True,
        cmap="cool",
        vmin=min_val,
        vmax=max_val,
        mask=mask,
    )
    plt.savefig(
        "./results/PILR_correlation_bias.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    run_PILR_analysis()
