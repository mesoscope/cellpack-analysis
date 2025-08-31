# %% [markdown]
# # Workflow to calculate correlation between individual PILRs
import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.label_tables import DATA_CONFIG

log = logging.getLogger(__name__)
# %% [markdown]
# ## Set up folders
dsphere = True
subset_folder = "sample_8d" if dsphere else "full"

project_root = get_project_root()
base_datadir = project_root / "data"
result_dir = project_root / "results/PILR_correlation_analysis/multi_structure_colocalization"
result_dir.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# ## Setup pilr dict
recalculate = False  # Set to False to load existing PILR dict
individual_pilr_dict = {}
index_tuples = []
pilr_dict_file = result_dir / "individual_PILR_dict.pkl"
index_tuples_file = result_dir / "index_tuples.pkl"
if pilr_dict_file.exists() and not recalculate:
    log.info(f"Loading existing PILR dict from {pilr_dict_file}")
    individual_pilr_dict = pd.read_pickle(pilr_dict_file)
    index_tuples = pd.read_pickle(index_tuples_file)
else:
    log.info("No existing PILR dict found, creating new one")
    for structure_name, structure_dict in DATA_CONFIG.items():
        individual_pilr_dict[structure_name] = {}
        for rule in structure_dict["rules"]:
            structure_id = structure_dict["structure_id"]
            individual_pilr_dict[structure_name][rule] = {}
            if "observed" in rule:
                pilr_path = (
                    base_datadir
                    / f"structure_data/{structure_id}/{subset_folder}/pilr/{structure_id}_individual_PILR.npy"
                )
            else:
                pilr_path = (
                    base_datadir
                    / f"PILR/{structure_name}/rules_shape/{rule}/{rule}_individual_PILR.npy"
                )
            log.info(f"Loading PILR for {structure_name} with rule {rule} from {pilr_path}")
            if not pilr_path.exists():
                log.warning(f"PILR file {pilr_path} does not exist")
                continue
            pilr_array = np.load(
                pilr_path,
            )
            log.info(f"Loaded PILR with shape {pilr_array.shape}")
            cell_ids = get_cell_id_list_for_structure(
                structure_id=structure_id,
                dsphere=dsphere,
            )
            for ct, cell_id in enumerate(cell_ids):
                index_tuples.append((structure_name, rule, cell_id))
                individual_pilr_dict[structure_name][rule][cell_id] = pilr_array[ct]
    # save combined pilr dict
    with open(pilr_dict_file, "wb") as f:
        pd.to_pickle(individual_pilr_dict, f)
    # save index tuples
    with open(index_tuples_file, "wb") as f:
        pd.to_pickle(index_tuples, f)
# %% [markdown]
# ## Get list of all structures and rules
all_struct_rule_pairs = []
cell_id_dict = {}
for structure in DATA_CONFIG.keys():
    structure_id = DATA_CONFIG[structure]["structure_id"]
    cell_id_dict[structure_id] = get_cell_id_list_for_structure(
        structure_id=structure_id,
        dsphere=dsphere,
    )
    for rule in DATA_CONFIG[structure]["rules"]:
        all_struct_rule_pairs.append((structure, rule))
print(all_struct_rule_pairs)

# %% [markdown]
# ## Create list of structures and rules to compare with each other
num_struct_rule = len(all_struct_rule_pairs)
restricted_tuples = []
# df_row = []
for (structure_1, rule_1), (
    structure_2,
    rule_2,
) in tqdm(
    itertools.combinations_with_replacement(all_struct_rule_pairs, 2),
    total=num_struct_rule * (num_struct_rule + 1) // 2,
    desc="Creating restricted tuples for correlation calculation",
):
    structure_id_1 = DATA_CONFIG[structure_1]["structure_id"]
    structure_id_2 = DATA_CONFIG[structure_2]["structure_id"]
    cell_ids_1 = cell_id_dict[structure_id_1]
    cell_ids_2 = cell_id_dict[structure_id_2]
    for cell_id_1, cell_id_2 in itertools.product(cell_ids_1, cell_ids_2):
        restricted_tuples.append(
            (
                (structure_1, rule_1, cell_id_1),
                (structure_2, rule_2, cell_id_2),
            )
        )

print(f"Number of restricted tuples: {len(restricted_tuples)}")

# %% [markdown]
# ## Calculate correlations
num_tuples = len(restricted_tuples)
index = np.arange(1, num_tuples + 1)
save_frequency = 10000000
df = pd.DataFrame(
    index=index,
    columns=[
        "cell_id_1",
        "cell_id_2",
        "rule_1",
        "rule_2",
        "structure_1",
        "structure_2",
        "correlation",
    ],
)
for row, (
    (structure_name_1, rule_1, cell_id_1),
    (structure_name_2, rule_2, cell_id_2),
) in tqdm(enumerate(restricted_tuples), total=len(df)):
    if structure_name_1 == structure_name_2 and rule_1 == rule_2 and cell_id_1 == cell_id_2:
        corr_val = 1.0
    else:
        pilr_1 = individual_pilr_dict[structure_name_1][rule_1][cell_id_1]
        pilr_2 = individual_pilr_dict[structure_name_2][rule_2][cell_id_2]
        masked_pilr_1 = pilr_1[(pilr_1.shape[0] // 2) :, :].flatten()
        masked_pilr_2 = pilr_2[(pilr_2.shape[0] // 2) :, :].flatten()
        corr_val = pearsonr(masked_pilr_1, masked_pilr_2).correlation
    df.loc[row] = [
        cell_id_1,
        cell_id_2,
        rule_1,
        rule_2,
        structure_name_1,
        structure_name_2,
        corr_val,
    ]
    if (row + 1) % save_frequency == 0:
        log.info(
            f"Saving intermediate results to {result_dir / 'individual_PILR_correlations.parquet'}"
        )
        df.to_parquet(result_dir / "individual_PILR_correlations.parquet")
log.info(
    "Finished calculating correlations, saving to"
    f" {result_dir / 'individual_PILR_correlations.parquet'}"
)
df.to_parquet(result_dir / "individual_PILR_correlations.parquet")

# %%
