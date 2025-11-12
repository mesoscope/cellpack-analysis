"""Workflow to generate simulated packed structures using cellPACK."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from cellpack_analysis.lib import default_values
from cellpack_analysis.lib.io import format_time
from cellpack_analysis.packing.generate_cellpack_input_files import (
    generate_configs,
    generate_recipes,
)
from cellpack_analysis.packing.pack_recipes import pack_recipes
from cellpack_analysis.packing.workflow_config import WorkflowConfig

# set random seed
np.random.seed(42)

logger = logging.getLogger(__name__)


def _run_packing_workflow(workflow_config_path: Path):
    """
    Run the packing workflow.

    Args:
    ----
        workflow_config_path (Path): The path to the packing configuration file.

    Returns:
    -------
        None
    """
    workflow_config = WorkflowConfig(config_file_path=workflow_config_path)

    # ## Generate cellPACK input files if needed
    if workflow_config.generate_recipes:
        logger.info("Generating cellPACK input files")
        generate_recipes(workflow_config=workflow_config)

    # ## update cellpack config file
    if workflow_config.generate_configs:
        logger.info("Updating cellPACK config file")
        generate_configs(workflow_config=workflow_config)

    # ## pack recipes
    if workflow_config.dry_run:
        logger.info("Dry run. Skipping packing.")
        return
    logger.info("Packing recipes")
    return pack_recipes(workflow_config=workflow_config)


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow_config_path",
        "-c",
        type=str,
        help="Path to the packing configuration file",
        default=default_values.WORKFLOW_CONFIG_PATH,
    )

    args = parser.parse_args()

    total_failed = _run_packing_workflow(workflow_config_path=Path(args.workflow_config_path))

    logger.info(f"Total time: {format_time(time.time() - start)}")
    if total_failed:
        logger.info(f"Total failed packings: {total_failed}")
        sys.exit(1)
    sys.exit(0)
