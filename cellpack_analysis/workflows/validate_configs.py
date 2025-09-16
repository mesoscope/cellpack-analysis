#!/usr/bin/env python3
"""
Validate configuration files for the unified analysis workflow runner.
"""

import json
from pathlib import Path


def validate_config_file(config_path: Path):
    """Validate a single configuration file."""
    print(f"\nValidating {config_path.name}...")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check required fields
        required_fields = ["analysis_type", "structure_id", "packing_modes"]
        for field in required_fields:
            if field not in config:
                print(f"  ❌ Missing required field: {field}")
                return False

        # Check analysis type
        valid_analysis_types = ["distance_analysis", "biological_variation", "occupancy_analysis"]
        if config["analysis_type"] not in valid_analysis_types:
            print(f"  ❌ Invalid analysis_type: {config['analysis_type']}")
            return False

        # Check packing modes is a list
        if not isinstance(config["packing_modes"], list):
            print("  ❌ packing_modes must be a list")
            return False

        print("  ✅ Valid configuration")
        print(f"     Analysis type: {config['analysis_type']}")
        print(f"     Structure: {config.get('structure_name', 'N/A')}")
        print(f"     Packing modes: {len(config['packing_modes'])} modes")

        return True

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Validate all configuration files."""
    configs_dir = Path("cellpack_analysis/workflows/configs")

    if not configs_dir.exists():
        print(f"❌ Configs directory not found: {configs_dir}")
        return

    config_files = [
        "distance_analysis_config.json",
        "biological_variation_config.json",
        "occupancy_analysis_config.json",
    ]

    print("Validating configuration files...")
    all_valid = True

    for config_file in config_files:
        config_path = configs_dir / config_file
        if config_path.exists():
            valid = validate_config_file(config_path)
            all_valid = all_valid and valid
        else:
            print(f"\n❌ Config file not found: {config_file}")
            all_valid = False

    status_message = (
        "✅ All configurations valid!" if all_valid else "❌ Some configurations have issues."
    )
    print(f"\n{status_message}")


if __name__ == "__main__":
    main()
