# Analysis Workflow Runner

This script provides a unified interface to run different types of cellpack analysis workflows.

## Installation

Make sure you have the `cellpack-analysis` package installed with all dependencies:

```bash
uv sync
```

## Usage

### Basic Usage

Run an analysis using a configuration file:

```bash
python cellpack_analysis/workflows/run_analysis_workflow.py --config_file path/to/config.json
```

### Analysis Types

The script supports three types of analysis:

1. **Distance Analysis** - Compare distance distributions using EMD and KS tests
2. **Biological Variation** - Analyze variation in organization due to biological factors  
3. **Occupancy Analysis** - Analyze how punctate structures occupy available sapce

### Example Commands

```bash
# Run distance analysis
python cellpack_analysis/workflows/run_analysis_workflow.py --config_file cellpack_analysis/workflows/configs/distance_analysis_config.json

# Run biological variation analysis
python cellpack_analysis/workflows/run_analysis_workflow.py --config_file cellpack_analysis/workflows/configs/biological_variation_config.json

# Run occupancy analysis
python cellpack_analysis/workflows/run_analysis_workflow.py --config_file cellpack_analysis/workflows/configs/occupancy_analysis_config.json
```

## Logging

You can control the logging level:

```bash
python cellpack_analysis/workflows/run_analysis_argparse.py --config_file config.json --log_level DEBUG
```

## Configuration File Format

Configuration files are JSON files that specify analysis parameters. See the example config files in `cellpack_analysis/workflows/configs/` for detailed examples.

### Key Parameters

- `analysis_type`: Type of analysis ("distance_analysis", "biological_variation", or "occupancy_analysis")
- `structure_id`: ID of the structure to analyze (e.g., "SLC25A17")
- `structure_name`: Human-readable name of the structure (e.g., "peroxisome")
- `packing_modes`: List of packing modes to compare
- `distance_measures`: Distance measures to calculate
- `normalization`: Distance normalization method (null for no normalization)
- `recalculate`: Whether to recalculate existing results

### Distance Analysis Specific

- `ks_significance_level`: Significance level for KS tests (default: 0.05)
- `n_bootstrap`: Number of bootstrap samples (default: 1000)

### Occupancy Analysis Specific

- `occupancy_distance_measures`: Distance measures for occupancy analysis
- `xlim`: X-axis limits for plotting different distance measures

## Output

Results are saved to the `results/` directory in the project root, organized by analysis type and structure name. Figures are saved in SVG format by default.
