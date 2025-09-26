#!/usr/bin/env python3
"""
Validate configuration files for the unified analysis workflow runner.
"""

import json
import logging
from pathlib import Path

from cellpack_analysis import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def validate_config_file(config_path: Path):
    """Validate a single configuration file."""
    logger.info(f"Validating {config_path.name}...")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check required fields
        required_fields = ["analysis_type", "structure_id", "packing_modes"]
        for field in required_fields:
            if field not in config:
                logger.info(f"  ❌ Missing required field: {field}")
                return False

        # Check analysis type
        valid_analysis_types = ["distance_analysis", "biological_variation", "occupancy_analysis"]
        if config["analysis_type"] not in valid_analysis_types:
            logger.info(f"  ❌ Invalid analysis_type: {config['analysis_type']}")
            return False

        # Check packing modes is a list
        if not isinstance(config["packing_modes"], list):
            logger.info("  ❌ packing_modes must be a list")
            return False

        logger.info("  ✅ Valid configuration")
        logger.info(f"     Analysis type: {config['analysis_type']}")
        logger.info(f"     Structure: {config.get('structure_name', 'N/A')}")
        logger.info(f"     Packing modes: {len(config['packing_modes'])} modes")

        return True

    except json.JSONDecodeError as e:
        logger.info(f"  ❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        logger.info(f"  ❌ Error: {e}")
        return False


def main():
    """Validate all configuration files."""
    configs_dir = Path("cellpack_analysis/workflows/configs")

    if not configs_dir.exists():
        logger.info(f"❌ Configs directory not found: {configs_dir}")
        return

    config_files = [
        "distance_analysis_config.json",
        "biological_variation_config.json",
        "occupancy_analysis_config.json",
    ]

    logger.info("Validating configuration files...")
    all_valid = True

    for config_file in config_files:
        config_path = configs_dir / config_file
        if config_path.exists():
            valid = validate_config_file(config_path)
            all_valid = all_valid and valid
        else:
            logger.info(f"\n❌ Config file not found: {config_file}")
            all_valid = False

    status_message = (
        "✅ All configurations valid!" if all_valid else "❌ Some configurations have issues."
    )
    logger.info(f"\n{status_message}")


if __name__ == "__main__":
    main()
