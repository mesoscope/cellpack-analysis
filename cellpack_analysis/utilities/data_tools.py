import json
from pathlib import Path
import numpy as np

from tqdm import tqdm
from scipy.stats import wasserstein_distance


def combine_multiple_seeds_to_dictionary(
    data_folder,
    ingredient_key="membrane_interior_peroxisome",
    search_prefix="positions_",
    save_name="positions_peroxisome_analyze_random_mean",
):
    data_folder = Path(data_folder)
    output_dict = {}
    for file in data_folder.glob(f"{search_prefix}*.json"):
        if save_name in file.name:
            continue
        seed = int(file.stem.split("_")[-1])
        with open(file) as j:
            raw_data = json.load(j)
        for seed_key, ingr_dict in raw_data.items():
            output_dict[f"{seed}_{seed_key}"] = {}
            for ingr_key, positions in ingr_dict.items():
                if ingr_key == ingredient_key:
                    output_dict[f"{seed}_{seed_key}"][ingr_key] = positions

    with open(data_folder / f"{save_name}.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return output_dict


def get_positions_dictionary_from_file(
    filename, ingredient_key="membrane_interior_peroxisome"
):
    """
    Retrieve positions dictionary from a file.

    Args:
        filename (str): The path to the file containing the positions data.
        ingredient_key (str, optional): The key for the ingredient in the raw data. Defaults to "membrane_interior_peroxisome".

    Returns:
        dict: A dictionary containing the positions data, where the keys are integers and the values are NumPy arrays.
    """
    with open(filename) as j:
        raw_data = json.load(j)
    positions = {k: np.array(v[ingredient_key]) for k, v in raw_data.items()}

    return positions


def get_pairwise_wasserstein_distance_dict(
    distribution_dict_1,
    distribution_dict_2=None,
):
    """
    distribution_dict is a dictionary with distances or other values for multiple seeds
    it has the form: {seed1: [value1, value2, ...], seed2: [value1, value2, ...], ...}

    The output has the form: {(seed1, seed2): distance, (seed1, seed3): distance, ...}
    """
    pairwise_wasserstein_distances = {}

    keys_1 = list(distribution_dict_1.keys())
    if distribution_dict_2 is None:
        for i in tqdm(range(len(keys_1))):
            for j in range(i + 1, len(keys_1)):
                seed_1 = keys_1[i]
                seed_2 = keys_1[j]
                pairwise_wasserstein_distances[(seed_1, seed_2)] = wasserstein_distance(
                    distribution_dict_1[seed_1], distribution_dict_1[seed_2]
                )
    else:
        keys_2 = list(distribution_dict_2.keys())
        for i in tqdm(range(len(keys_1))):
            for j in range(len(keys_2)):
                seed_1 = keys_1[i]
                seed_2 = keys_2[j]
                pairwise_wasserstein_distances[(seed_1, seed_2)] = wasserstein_distance(
                    distribution_dict_1[seed_1], distribution_dict_2[seed_2]
                )

    return pairwise_wasserstein_distances
