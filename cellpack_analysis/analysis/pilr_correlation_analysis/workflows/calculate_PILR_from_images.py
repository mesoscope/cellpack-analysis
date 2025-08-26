# %% Import required packages
import argparse
import concurrent.futures
import multiprocessing
from logging import getLogger
from pathlib import Path
from time import time

import matplotlib as mpl
import numpy as np
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from tqdm import tqdm

from cellpack_analysis.lib.label_tables import (
    DUAL_STRUCTURE_SIM_CHANNEL_MAP,
    RAW_CHANNEL_MAP,
    SIM_CHANNEL_MAP,
)
from cellpack_analysis.lib.PILR_tools import (
    average_over_dimension,
    get_pilr_for_single_image,
)
from cellpack_analysis.lib.plotting_tools import plot_PILR, save_PILR_as_tiff

log = getLogger(__name__)


# %%
def PILR_calculation_workflow(
    channel_names,
    raw_image_path,
    simulated_image_path=None,
    num_cores=80,
    save_dir=None,
    raw_image_channel="SLC25A17",
    raw_channel_map=RAW_CHANNEL_MAP,
    simulated_channel_map=SIM_CHANNEL_MAP,
):
    """
    Perform PILR (Parameterized Intracellular Localization Representation)
    calculation workflow for raw and simulated images across multiple channels.
    This function processes images, calculates PILR for each image, averages
    the PILR values, and saves the results in various formats including TIFF,
    JSON, and NumPy arrays. It supports parallel processing for efficiency.

    Parameters
    ----------
    raw_image_path : str or Path
        Path to the directory containing raw images.
    simulated_image_path : str or Path, optional
        Path to the directory containing simulated images. If None, only raw
        images are processed. Default is None.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. Default is 80.
    save_dir : str or Path, optional
        Directory where results will be saved. If None, results are saved in
        "./results". Default is None.
    raw_image_channel : str, optional
        Name of the channel corresponding to raw images. Default is "SLC25A17".
    channel_names : list of str, optional
        List of channel names to process. Default is None.
    raw_channel_map : dict
        Mapping of raw image channel indices to labeled structures.
    simulated_channel_map : dict
        Mapping of simulated image channel indices to labeled structures.

    Returns
    -------
    None
        This function does not return anything. It saves the results to the
        specified directory.

    Notes
    -----
    - The function supports both single-threaded and multi-threaded processing
      based on the `num_cores` parameter.
    - Results include individual PILR values, average PILR values, and
      dimensional averages saved as images and NumPy arrays.
    - The function uses the `OmeTiffWriter` for saving TIFF files and `numpy`
      for saving arrays.

    Examples
    --------
    >>> PILR_calculation_workflow(
    ...     raw_image_path="/path/to/raw/images",
    ...     simulated_image_path="/path/to/simulated/images",
    ...     num_cores=4,
    ...     save_dir="/path/to/save/results",
    ...     raw_image_channel="SLC25A17",
    ...     channel_names=["Channel1", "Channel2"],
    ... )
    """
    if save_dir is None:
        save_dir = "./results"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    mpl.use("agg")
    for ch_name in channel_names:
        log.info(f"Processing {ch_name} channel")
        image_path = None
        channel_map = simulated_channel_map
        if ch_name == raw_image_channel:
            image_path = raw_image_path
            channel_map = raw_channel_map
        elif simulated_image_path is not None:
            image_path = simulated_image_path.replace("{rule}", ch_name)

        if image_path is None:
            log.info(f"No image path given for {ch_name} channel")
            continue
        else:
            image_path = Path(image_path)

        log.info(f"Getting images from {image_path}")

        file_list = []
        gfp_representations = []
        for file in image_path.glob(f"*{ch_name}*.tiff"):
            if "invert" in file.name and "invert" not in ch_name:
                continue
            file_list.append(file)

        num_files = len(file_list)

        if num_files == 0:
            log.info(f"No files found for {ch_name} channel")
            continue
        ch_save_dir = save_dir / ch_name
        ch_save_dir.mkdir(exist_ok=True, parents=True)

        writer = OmeTiffWriter()
        individual_pilr_dir = ch_save_dir / "individual_PILR"
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
                            [channel_map] * num_files,
                        ), strict=False,
                    ),
                    desc=f"Processing {ch_name} files",
                    total=num_files,
                ):
                    # log.info(f"Processing file {file} with channel {ch_name}")
                    if gfp_representation is None:
                        log.info(f"File {file} returned None")
                        continue
                    gfp_representations.append(gfp_representation)
                    save_PILR_as_tiff(
                        gfp_representation,
                        individual_pilr_dir,
                        f"{file.stem.split('.')[0]}_pilr",
                        writer,
                    )

        else:
            for file in tqdm(file_list, total=num_files):
                img_pilr = get_pilr_for_single_image(
                    file, ch_name, raw_image_channel, channel_map
                )
                gfp_representations.append(img_pilr)
                save_PILR_as_tiff(
                    img_pilr,
                    individual_pilr_dir,
                    f"{file.stem.split('.')[0]}_pilr",
                    writer,
                )
        individual_pilrs = np.array(gfp_representations, dtype=np.float16)

        average_pilr = np.mean(individual_pilrs, axis=0)  # Average over all the images

        # Save the average PILR as an image
        plot_PILR(
            average_pilr,
            ch_name,
            save_dir=ch_save_dir,
        )

        for ind, dim in enumerate(["R", "phi", "theta"]):
            dim_dir = ch_save_dir / f"dimensional_average_PILR/{dim}"
            dim_dir.mkdir(parents=True, exist_ok=True)
            avg_dim = average_over_dimension(average_pilr, dim=ind)
            plot_PILR(
                avg_dim,
                ch_name,
                save_dir=dim_dir,
                suffix=f"_{dim}",
                aspect=1,
            )

        # Save the PILR as json
        avg_save_path = ch_save_dir / f"{ch_name}_average_PILR.npy"
        log.info(f"Saving average PILR to {avg_save_path}")
        np.save(avg_save_path, average_pilr)

        # Save individual PILRs
        individual_save_path = ch_save_dir / f"{ch_name}_individual_PILR.npy"
        log.info(f"Saving individual PILR to {individual_save_path}")
        np.save(individual_save_path, individual_pilrs)


if __name__ == "__main__":
    start_time = time()
    parser = argparse.ArgumentParser(description="Run PILR analysis on cellPACK output")

    parser.add_argument(
        "--channel_names",
        "-c",
        type=str,
        nargs="+",
        required=True,
        default=[
            "SLC25A17",
            "random",
            "nucleus_gradient_strong",
            "membrane_gradient_strong",
        ],
        help="List of channel names to process, e.g. 'SLC25A17 random'",
    )
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
        default=None,
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
        "-rc",
        type=str,
        default="SLC25A17",
        help="Channel name for raw images",
    )
    parser.add_argument(
        "--dual_structure",
        "-ds",
        action="store_true",
        help="Use dual structure for PILR calculation",
        default=False,
    )

    args = parser.parse_args()
    log.info("Starting PILR calculation workflow with the following parameters:")
    log.info(f"Channel names: {args.channel_names}")
    log.info(f"Raw image path: {args.raw_image_path}")
    log.info(f"Simulated image path: {args.simulated_image_path}")
    log.info(f"Number of cores: {args.num_cores}")
    log.info(f"Save directory: {args.save_dir}")
    log.info(f"Raw image channel: {args.raw_image_channel}")
    log.info(f"Dual structure: {args.dual_structure}")

    simulated_channel_map = SIM_CHANNEL_MAP
    if args.dual_structure:
        simulated_channel_map = DUAL_STRUCTURE_SIM_CHANNEL_MAP

    PILR_calculation_workflow(
        raw_image_path=args.raw_image_path,
        channel_names=args.channel_names,
        simulated_image_path=args.simulated_image_path,
        save_dir=args.save_dir,
        raw_image_channel=args.raw_image_channel,
        simulated_channel_map=simulated_channel_map,
        num_cores=args.num_cores,
    )

    log.info(f"Finished PILR calculation workflow in {time() - start_time:.2f} seconds")
    log.info(f"Results saved in {args.save_dir}")
