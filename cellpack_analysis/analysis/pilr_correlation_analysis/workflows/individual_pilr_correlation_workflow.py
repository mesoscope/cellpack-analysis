"""
Calculate individual PILR correlations between structures and rules

This workflow calculates correlations between PILRs for the structures and rules
specified in the config file.
"""

import itertools
import logging
from pathlib import Path
from time import time

import fire
import numpy as np
import pandas as pd

from cellpack_analysis.lib.file_io import get_project_root, read_json
from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure
from cellpack_analysis.lib.PILR_tools import vectorized_pixelwise_correlation

log = logging.getLogger(__name__)

project_root = get_project_root()
DEFAULT_CONFIG_PATH = project_root / (
    "cellpack_analysis/analysis/pilr_correlation_analysis/"
    "configs/peroxisome_individual_pilr_correlation.json"
)


def calculate_and_save_correlations(config_path: Path = DEFAULT_CONFIG_PATH):
    """
    Calculate correlations between PILRs based on the provided configuration file.

    Parameters
    ----------
        config_path (Path): Path to the configuration file.
    """
    config = read_json(config_path)

    # Load PILRs
    pilr_list = []
    structure_rule_cellid_lookup = []
    workflow = config["workflow"]
    output_folder = project_root / f"results/PILR_correlation_analysis/{workflow}"
    output_folder.mkdir(parents=True, exist_ok=True)
    for structure_name, structure_info in config["data"].items():
        base_folder = project_root / f"data/PILR/{structure_name}"
        for rule, rule_info in structure_info["rules"].items():
            structure_id = rule_info["structure_id"]
            cellid_list = get_cellid_list_for_structure(
                structure_id, dsphere=rule_info["dsphere"]
            )
            pilr_path = base_folder / rule / f"{rule}_individual_PILR.npy"
            pilr_data = np.load(pilr_path)
            pilr_data[pilr_data > 0] = 1

            for i, cellid in enumerate(cellid_list):
                pilr_list.append(pilr_data[i])
                structure_rule_cellid_lookup.append((structure_name, rule, cellid))
    pilr_array = np.array(pilr_list)

    # Create pairs of structures and rules to compare
    index_pairs = list(
        itertools.combinations_with_replacement(range(len(pilr_array)), 2)
    )

    # Get correlation matrix
    corr_start_time = time()
    log.info("Calculating correlation matrix")
    corr_matrix = vectorized_pixelwise_correlation(
        pilr_array,
        flatten=False,
    )
    triu_indices = np.triu_indices(len(pilr_array), k=0)
    corr_values = corr_matrix[triu_indices]
    log.info(f"Correlation matrix calculated in {time() - corr_start_time:.2f} seconds")

    # Build dataframe from index pairs
    structure_rule_cellid_pairs = [
        [
            structure_rule_cellid_lookup[i][0],
            structure_rule_cellid_lookup[j][0],
            structure_rule_cellid_lookup[i][1],
            structure_rule_cellid_lookup[j][1],
            structure_rule_cellid_lookup[i][2],
            structure_rule_cellid_lookup[j][2],
        ]
        for i, j in index_pairs
    ]

    # Construct dataframe
    df = pd.DataFrame(
        structure_rule_cellid_pairs,
        columns=[
            "structure_1",
            "structure_2",
            "rule_1",
            "rule_2",
            "cellid_1",
            "cellid_2",
        ],
    )
    df["correlation"] = corr_values

    # Save results
    df_save_path = output_folder / "pilr_correlation.parquet"
    log.info(f"Saving results to {df_save_path}")
    df.to_parquet(df_save_path, index=False)


if __name__ == "__main__":
    start_time = time()
    log.info("Starting individual PILR correlation workflow")
    fire.Fire(calculate_and_save_correlations)
    log.info(f"Workflow completed in {time() - start_time:.2f} seconds")
