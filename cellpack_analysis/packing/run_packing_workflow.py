"""Workflow to generate simulated packed structures using cellPACK."""

import argparse
import time
from pathlib import Path

import numpy as np

from cellpack_analysis.lib import default_values
from cellpack_analysis.packing.generate_cellpack_input_files import generate_recipes
from cellpack_analysis.packing.workflow_config import WorkflowConfig

# set random seed
np.random.seed(42)


def _run_packing_workflow(workflow_config_path: Path):
    """
    Run the packing workflow.

    Args:
        workflow_config_path (Path): The path to the packing configuration file.

    Returns:
        None
    """
    if workflow_config_path is None:
        workflow_config_path = Path(__file__).parent / "configs/example.json"

    workflow_config = WorkflowConfig(config_file_path=workflow_config_path)

    # ## Generate cellPACK input files if needed
    if workflow_config.generate_recipes:
        generate_recipes(workflow_config=workflow_config)


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow_config_path",
        type=str,
        help="Path to the packing configuration file",
        default=default_values.WORKFLOW_CONFIG_PATH,
    )

    args = parser.parse_args()

    _run_packing_workflow(workflow_config_path=Path(args.workflow_config_path))
