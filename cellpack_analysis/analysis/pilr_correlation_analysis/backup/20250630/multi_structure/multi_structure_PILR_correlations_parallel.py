# %% [markdown]
# # Workflow to calculate correlation between individual PILRs (parallelized)
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure
from cellpack_analysis.lib.label_tables import DATA_CONFIG

log = logging.getLogger(__name__)

# %% [markdown]
# ## Set up folders
dsphere = True
subset_folder = "sample_8d" if dsphere else "full"

project_root = get_project_root()
base_datadir = project_root / "data"
result_dir = (
    project_root / "results/PILR_correlation_analysis/multi_structure_colocalization"
)
result_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load individual PILR dict and prepare index tuples
# %% [markdown]
# ## Setup pilr dict
recalculate = False  # Set to False to load existing PILR dict
individual_pilr_dict = {}
pilr_dict_file = result_dir / "individual_PILR_dict.pkl"
if pilr_dict_file.exists() and not recalculate:
    log.info(f"Loading existing PILR dict from {pilr_dict_file}")
    individual_pilr_dict = pd.read_pickle(pilr_dict_file)
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
            log.info(
                f"Loading PILR for {structure_name} with rule {rule} from {pilr_path}"
            )
            if not pilr_path.exists():
                log.warning(f"PILR file {pilr_path} does not exist")
                continue
            pilr_array = np.load(
                pilr_path,
            )
            log.info(f"Loaded PILR with shape {pilr_array.shape}")
            cellids = get_cellid_list_for_structure(
                structure_id=structure_id,
                dsphere=dsphere,
            )
            for ct, cellid in enumerate(cellids):
                individual_pilr_dict[structure_name][rule][cellid] = pilr_array[ct]
    # save combined pilr dict
    with open(pilr_dict_file, "wb") as f:
        pd.to_pickle(individual_pilr_dict, f)

# %% [markdown]
# ## Get list of all structures and rules
all_struct_rule_pairs = []
cellid_dict = {}
for structure in DATA_CONFIG.keys():
    structure_id = DATA_CONFIG[structure]["structure_id"]
    cellid_dict[structure_id] = get_cellid_list_for_structure(
        structure_id=structure_id,
        dsphere=dsphere,
    )
    for rule in DATA_CONFIG[structure]["rules"]:
        all_struct_rule_pairs.append((structure, rule))
print(all_struct_rule_pairs)

# %% [markdown]
# ## Create list of structures and rules to compare with each other
index_tuple_file = result_dir / "index_tuple_pairs.pkl"
recalculate = False
if index_tuple_file.exists() and not recalculate:
    log.info(f"Loading existing index tuples from {index_tuple_file}")
    index_tuple_pairs = pd.read_pickle(index_tuple_file)
else:
    num_struct_rule = len(all_struct_rule_pairs)
    index_tuple_pairs = []
    for (structure_1, rule_1), (
        structure_2,
        rule_2,
    ) in tqdm(
        itertools.combinations_with_replacement(all_struct_rule_pairs, 2),
        total=num_struct_rule * (num_struct_rule + 1) // 2,
        desc="Creating index tuple pairs for correlation calculation",
    ):
        structure_id_1 = DATA_CONFIG[structure_1]["structure_id"]
        structure_id_2 = DATA_CONFIG[structure_2]["structure_id"]
        cellids_1 = cellid_dict[structure_id_1]
        cellids_2 = cellid_dict[structure_id_2]
        for cellid_1, cellid_2 in itertools.product(cellids_1, cellids_2):
            index_tuple_pairs.append(
                (
                    (structure_1, rule_1, cellid_1),
                    (structure_2, rule_2, cellid_2),
                )
            )

    log.info(f"Number of index tuple pairs: {len(index_tuple_pairs)}")
    with open(index_tuple_file, "wb") as f:
        pd.to_pickle(index_tuple_pairs, f)
# %% [markdown]
# ## Define parallel correlation functions

_global_index_tuple_pairs = None
_global_individual_pilr_dict = None


def init_worker(index_tuple_pairs, individual_pilr_dict):
    """Set up shared data in worker process."""
    global _global_index_tuple_pairs, _global_individual_pilr_dict
    _global_index_tuple_pairs = index_tuple_pairs
    _global_individual_pilr_dict = individual_pilr_dict


def compute_correlation(index):
    """Compute Pearson correlation for a pair of cell rules."""
    t1, t2 = _global_index_tuple_pairs[index]

    structure_name_1, rule_1, cellid_1 = t1
    structure_name_2, rule_2, cellid_2 = t2

    if t1 == t2:
        correlation = 1.0
    else:
        pilr_1 = _global_individual_pilr_dict[structure_name_1][rule_1][cellid_1]
        pilr_2 = _global_individual_pilr_dict[structure_name_2][rule_2][cellid_2]

        masked_pilr_1 = pilr_1[(pilr_1.shape[0] // 2) :, :].flatten()
        masked_pilr_2 = pilr_2[(pilr_2.shape[0] // 2) :, :].flatten()

        correlation = pearsonr(masked_pilr_1, masked_pilr_2).correlation

    return (
        cellid_1,
        cellid_2,
        rule_1,
        rule_2,
        structure_name_1,
        structure_name_2,
        correlation,
    )


# %% [markdown]
# ## Parallel correlation calculation

pairs = list(index_tuple_pairs)
results = []

log.info(f"Starting correlation calculation for {len(pairs)} pairs")

with ProcessPoolExecutor(
    initializer=init_worker,
    initargs=(index_tuple_pairs, individual_pilr_dict),
    max_workers=64,
) as executor:
    futures = [executor.submit(compute_correlation, idx) for idx in range(len(pairs))]
    log.info("Executor initialized, starting correlation calculation")

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            log.warning(f"Correlation calculation failed: {e}")

# %% [markdown]
# ## Save final DataFrame
df = pd.DataFrame(
    results,
    columns=[
        "cellid_1",
        "cellid_2",
        "rule_1",
        "rule_2",
        "structure_1",
        "structure_2",
        "correlation",
    ],
)

output_path = result_dir / "individual_PILR_correlations.parquet"
df.to_parquet(output_path, index=False)
log.info(f"Finished all correlations. Results saved to {output_path}")

# %%
