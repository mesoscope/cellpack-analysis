# %% [markdown]
# # Workflow to calculate correlation between individual PILRs
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure

log = logging.getLogger(__name__)
# %% [markdown]
# ## Define channel names
STRUCTURE_ID = "RAB5A"
STRUCTURE_NAME = "golgi_endosome"
EXPERIMENT = "rules_shape"

CHANNEL_IDS = {
    # "SLC25A17": "Peroxisome",
    "RAB5A": "Endosome",
    "random": "Random",
    "nucleus_gradient_strong": "Nucleus",
    "membrane_gradient_strong": "Membrane",
    "struct_gradient": "Structure",
    # "apical_gradient": "Apical gradient",
}

CHANNEL_STRUCTURE_MAP = {  # map channel name to structure ID it's calculated from
    # "SLC25A17": "SLC25A17",
    "RAB5A": "RAB5A",
    # "random": "SEC61B",
    # "nucleus_gradient_strong": "SEC61B",
    # "membrane_gradient_strong": "SEC61B",
    # "struct_gradient": "SEC61B",
    "random": "ST6GAL1",
    "nucleus_gradient_strong": "ST6GAL1",
    "membrane_gradient_strong": "ST6GAL1",
    "struct_gradient": "ST6GAL1",
}

# %% [markdown]
# ## Set up folders
base_datadir = Path(__file__).parents[4] / "data"
base_result_dir = Path(__file__).parents[4] / "results/PILR_correlation_analysis"

result_dir = base_result_dir / f"{STRUCTURE_NAME}/{EXPERIMENT}/"
result_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load individual PILRs and combine
individual_pilr_path = result_dir / "individual_PILR.json"
log.info(f"Loading individual PILRs from {individual_pilr_path}")
with open(individual_pilr_path) as f:
    individual_pilr_dict = json.load(f)

# %% [markdown]
# ## Convert to numpy arrays
cellids = {}
for channel, pilr_list in individual_pilr_dict.items():
    log.info(f"Converting PILRs for {channel} to numpy arrays")
    individual_pilr_dict[channel] = np.array(pilr_list, dtype=np.float16)
    log.info(f"{channel} PILR shape {individual_pilr_dict[channel].shape}")
    cellids[channel] = get_cellid_list_for_structure(
        CHANNEL_STRUCTURE_MAP[channel], dsphere=True
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
    for channel, cellid_list in cellids.items():
        index_tuples.extend([(channel, cellid) for cellid in cellid_list])
    index = pd.MultiIndex.from_tuples(index_tuples, names=["channel", "cellid"])
    df_corr = pd.DataFrame(
        columns=index,
        index=index,
        dtype=float,
    )
    log.info(f"Created dataframe with shape {df_corr.shape}")
    for channel_1, pilr_list_1 in individual_pilr_dict.items():
        cellid_list_1 = df_corr.loc[channel_1].index
        for channel_2, pilr_list_2 in individual_pilr_dict.items():
            cellid_list_2 = df_corr.loc[channel_2].index
            log.info(f"Calculating correlation between {channel_1} and {channel_2}")
            for cellid1, pilr1 in tqdm(
                zip(cellid_list_1, pilr_list_1), total=len(cellid_list_1)
            ):
                masked_pilr1 = pilr1[(pilr1.shape[0] // 2) :, :].flatten()  # noqa
                for cellid2, pilr2 in zip(cellid_list_2, pilr_list_2):
                    # only calculate if not already calculated
                    if pd.isna(df_corr.loc[(channel_1, cellid1), (channel_2, cellid2)]):
                        if (channel_1 == channel_2) and (cellid1 == cellid2):
                            df_corr.loc[(channel_1, cellid1), (channel_2, cellid2)] = 1
                            df_corr.loc[(channel_2, cellid2), (channel_1, cellid1)] = 1
                        else:
                            masked_pilr2 = pilr2[
                                (pilr2.shape[0] // 2) :, :  # noqa
                            ].flatten()
                            corr_val = pearsonr(masked_pilr1, masked_pilr2).correlation
                            df_corr.loc[(channel_1, cellid1), (channel_2, cellid2)] = (
                                corr_val
                            )
                            df_corr.loc[(channel_2, cellid2), (channel_1, cellid1)] = (
                                corr_val
                            )
            # df.to_csv(result_dir / "individual_PILR_corr.csv")
        df_corr.to_parquet(result_dir / "individual_PILR_correlations.parquet")
log.info(f"Correlations calculated and saved to {df_corr_path}")
# %% [markdown]
# ## Plot correlations
sns.heatmap(df_corr)
# %%
