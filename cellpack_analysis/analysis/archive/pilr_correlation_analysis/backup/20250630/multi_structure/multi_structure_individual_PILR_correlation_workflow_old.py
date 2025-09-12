# %% [markdown]
# # Workflow to calculate correlation between individual PILRs
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure

log = logging.getLogger(__name__)
# %% [markdown]
# ## Define channel names
EXPERIMENT = "rules_shape"

STRUCTURE_ID = "RAB5A"
STRUCTURE_NAME = "endosome"
COMPARE_STRUCTURE_FOLDER = "golgi_endosome"

CHANNEL_STRUCTURE_MAP = {  # map channel name to structure ID it's calculated from
    "peroxisome": "SLC25A17",
    "ER_peroxisome": "SEC61B",
    "ER_peroxisome_no_struct": "SEC61B",
    # "observed": "RAB5A",
    "endosome": "RAB5A",
    "golgi_endosome": "ST6GAL1",
    "golgi_endosome_no_struct": "ST6GAL1",
}

RULE_MAP = {
    "peroxisome": [
        "observed",
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
    ],
    "ER_peroxisome": [
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
        "struct_gradient",
    ],
    "ER_peroxisome_no_struct": [
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
    ],
    "endosome": [
        "observed",
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
        "apical_gradient",
    ],
    "golgi_endosome": [
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
        "struct_gradient",
    ],
    "golgi_endosome_no_struct": [
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
    ],
}

# %% [markdown]
# ## Set up folders
dsphere = True
subset_folder = "sample_8d" if dsphere else "full"

base_datadir = Path(__file__).parents[4] / "data"
base_result_dir = Path(__file__).parents[4] / "results/PILR_correlation_analysis"

observed_pilr_path = (
    base_datadir
    / f"structure_data/{STRUCTURE_ID}/{subset_folder}/pilr/{STRUCTURE_ID}_individual_PILR.npy"
)

result_dir = base_result_dir / "multi_structure_colocalization" / COMPARE_STRUCTURE_FOLDER
result_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load individual PILRs and combine
individual_pilr_dict = {}
for structure_name, rule_list in RULE_MAP.items():
    individual_pilr_dict[structure_name] = {}
    simulated_pilr_folder = base_result_dir / f"{structure_name}/{EXPERIMENT}"
    for rule in rule_list:
        log.info(f"Loading PILR for {structure_name} with rule {rule}")
        if "observed" in rule:
            pilr_path = observed_pilr_path
        else:
            pilr_path = simulated_pilr_folder / f"{rule}/{rule}_individual_PILR.npy"
        individual_pilr_dict[structure_name][rule] = np.load(pilr_path)
        log.info(f"Loaded PILR with shape {individual_pilr_dict[structure_name][rule].shape}")
# %% [markdown]
# ## Get cell_ids
cell_ids = {}
for structure_name in individual_pilr_dict:
    log.info(f"Getting cell_ids for {structure_name}")
    cell_ids[structure_name] = get_cell_id_list_for_structure(
        CHANNEL_STRUCTURE_MAP[structure_name], dsphere=True
    )
# %% [markdown]
# ## Calculate correlations
recalculate = True
df_corr_path = result_dir / "individual_PILR_correlations.parquet"

if (not recalculate) and df_corr_path.exists():
    log.info(f"Loading correlations from {df_corr_path}")
    df_corr = pd.read_parquet(df_corr_path)
else:
    log.info("Calculating correlations")
    index_tuples = []
    for structure_name, rule_dict in individual_pilr_dict.items():
        for rule_name in rule_dict:
            for cell_id in cell_ids[structure_name]:
                index_tuples.append((structure_name, rule_name, cell_id))
    index = pd.MultiIndex.from_tuples(index_tuples, names=["channel", "rule", "cell_id"])
    df_corr = pd.DataFrame(
        columns=index,
        index=index,
        dtype=float,
    )
    df_corr.sort_index(axis=0, inplace=True)
    df_corr.sort_index(axis=1, inplace=True)
    log.info(f"Created dataframe with shape {df_corr.shape}")

    for channel_1, rule_dict_1 in individual_pilr_dict.items():
        for rule_1, pilr_list_1 in rule_dict_1.items():
            cell_id_list_1 = cell_ids[channel_1]
            for channel_2, rule_dict_2 in individual_pilr_dict.items():
                for rule_2, pilr_list_2 in rule_dict_2.items():
                    cell_id_list_2 = cell_ids[channel_2]
                    log.info(
                        f"Calculating correlation between {channel_1} with rule {rule_1}"
                        f" and {channel_2} with rule {rule_2}"
                    )
                    for cell_id1, pilr1 in tqdm(
                        zip(cell_id_list_1, pilr_list_1, strict=False), total=len(cell_id_list_1)
                    ):
                        masked_pilr1 = pilr1[(pilr1.shape[0] // 2) :, :].flatten()
                        for cell_id2, pilr2 in zip(cell_id_list_2, pilr_list_2, strict=False):
                            # only calculate if not already calculated
                            if pd.isna(
                                df_corr.loc[
                                    (channel_1, rule_1, cell_id1),
                                    (channel_2, rule_2, cell_id2),
                                ]
                            ):
                                if (
                                    (channel_1 == channel_2)
                                    and (rule_1 == rule_2)
                                    and (cell_id1 == cell_id2)
                                ):
                                    df_corr.loc[
                                        (channel_1, rule_1, cell_id1),
                                        (channel_2, rule_2, cell_id2),
                                    ] = 1
                                    df_corr.loc[
                                        (channel_2, rule_2, cell_id2),
                                        (channel_1, rule_1, cell_id1),
                                    ] = 1
                                else:
                                    masked_pilr2 = pilr2[(pilr2.shape[0] // 2) :, :].flatten()
                                    corr_val = pearsonr(masked_pilr1, masked_pilr2).correlation
                                    df_corr.loc[
                                        (channel_1, rule_1, cell_id1),
                                        (channel_2, rule_2, cell_id2),
                                    ] = corr_val
                                    df_corr.loc[
                                        (channel_2, rule_2, cell_id2),
                                        (channel_1, rule_1, cell_id1),
                                    ] = corr_val
        df_corr.to_parquet(df_corr_path)
log.info(f"Correlations calculated and saved to {df_corr_path}")
