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

from cellpack_analysis.analysis.pilr_correlation_analysis import (
    individual_PILR_correlation,
)

log = logging.getLogger(__name__)
# %% [markdown]
# ## Define channel names
STRUCTURE_IDS = {
    "SLC25A17": "Peroxisome",
    "RAB5A": "Endosome",
    "SEC61B": "ER",
    "ST6GAL1": "Golgi",
    "TOMM20": "Mitochondria",
}

FOLDER_ID = "raw_data"

# %% [markdown]
# ## Set up folders
base_datadir = Path(__file__).parents[3] / "data"
base_result_dir = Path(__file__).parents[3] / "results/PILR_correlation_analysis"

result_dir = base_result_dir / "ER_colocalization/individual_PILR_correlations"
result_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load individual PILRs and combine
recalculate = False
individual_pilr_path = result_dir / "individual_PILR.json"

if (not recalculate) and individual_pilr_path.exists():
    log.info(f"Loading individual PILRs from {individual_pilr_path}")
    with open(individual_pilr_path) as f:
        individual_pilr_dict = json.load(f)
else:
    log.info("Reading individual PILRs")
    individual_pilr_dict = {}
    for structure_id in STRUCTURE_IDS:
        structure_folder = base_result_dir / structure_id / FOLDER_ID
        with open(structure_folder / "individual_PILR.json") as f:
            struct_pilr = json.load(f)
        individual_pilr_dict[structure_id] = struct_pilr[structure_id]

    with open(individual_pilr_path, "w") as f:
        json.dump(individual_pilr_dict, f)

# %% [markdown]
# ## Convert to numpy arrays
for struct, pilr_list in individual_pilr_dict.items():
    log.info(f"Converting PILRs for {struct} to numpy arrays")
    individual_pilr_dict[struct] = np.array(pilr_list)

# %% [markdown]
# ## Calculate correlations
recalculate = False
df_corr_path = result_dir / "individual_PILR_correlations.csv"

if (not recalculate) and df_corr_path.exists():
    log.info(f"Loading correlations from {df_corr_path}")
    df_corr = pd.read_csv(df_corr_path, index_col=[0, 1], header=[0, 1])
else:
    log.info("Calculating correlations")
    df_corr = individual_PILR_correlation.get_structure_cellid_dataframe(
        structure_ids=STRUCTURE_IDS,
        use_8d_sphere=True,
    )
    for struct_1, pilr_list_1 in individual_pilr_dict.items():
        cellid_list_1 = df_corr.loc[struct_1].index
        for struct_2, pilr_list_2 in individual_pilr_dict.items():
            cellid_list_2 = df_corr.loc[struct_2].index
            log.info(f"Calculating correlation between {struct_1} and {struct_2}")
            for cellid1, pilr1 in tqdm(
                zip(cellid_list_1, pilr_list_1, strict=False), total=len(cellid_list_1)
            ):
                masked_pilr1 = pilr1[(pilr1.shape[0] // 2) :, :].flatten()
                for cellid2, pilr2 in zip(cellid_list_2, pilr_list_2, strict=False):
                    # only calculate if not already calculated
                    if df_corr.loc[(struct_1, cellid1), (struct_2, cellid2)] is np.nan:
                        if (struct_1 == struct_2) and (cellid1 == cellid2):
                            df_corr.loc[(struct_1, cellid1), (struct_2, cellid2)] = 1
                            df_corr.loc[(struct_2, cellid2), (struct_1, cellid1)] = 1
                        else:
                            masked_pilr2 = pilr2[(pilr2.shape[0] // 2) :, :].flatten()
                            corr_val = pearsonr(masked_pilr1, masked_pilr2)[0]
                            df_corr.loc[(struct_1, cellid1), (struct_2, cellid2)] = (
                                corr_val
                            )
                            df_corr.loc[(struct_2, cellid2), (struct_1, cellid1)] = (
                                corr_val
                            )
            # df.to_csv(result_dir / "individual_PILR_corr.csv")
        df_corr.to_csv(result_dir / "individual_PILR_correlations.csv")

# %% [markdown]
# ## Plot correlations
sns.heatmap(df_corr)
# %%
