# Data Release Workflow

This workflow exports cellPACK simularium files and metadata to S3 for BFF (Biomedical File Finder) data release.

## Overview

The data release workflow provides modular execution of the following steps:
1. **Upload meshes to S3** - Upload structure meshes (with optional face inversion)
2. **Update simularium files locally** - Update mesh URLs and colors in simularium files
3. **Upload simularium files to S3** - Upload the updated simularium files
4. **Generate and upload thumbnails** - Create composite thumbnails and upload to S3
5. **Create CSV files** - Export S3 file paths and metadata as CSV

Each step can be run independently or together, controlled via configuration file or command-line arguments.

## Files

- `run_data_release_workflow.py` - Main workflow script with modular execution
- `data_release_config.py` - Configuration class
- `configs/data_release_config.json` - Example configuration file
- `export_simularium_paths_as_csv.py` - Original notebook (preserved for reference)

## Library Files

The workflow uses utility functions from:
- `cellpack_analysis/lib/s3_utils.py` - S3 upload/download and mesh processing utilities
- `cellpack_analysis/lib/simularium_utils.py` - Simularium file processing utilities
- `cellpack_analysis/lib/mesh_tools.py` - Mesh manipulation utilities (invert_mesh_faces)

## Usage

### Run All Enabled Steps

Run all steps that are enabled in the configuration file:

```bash
python cellpack_analysis/notebooks/data_release/run_data_release_workflow.py \
    --config_file cellpack_analysis/notebooks/data_release/configs/data_release_config.json
```

### Run Specific Steps

Run only specific steps using command-line flags:

```bash
# Upload meshes only
python run_data_release_workflow.py --config_file configs/data_release_config.json --upload_meshes

# Update simularium files only (local changes, no upload)
python run_data_release_workflow.py --config_file configs/data_release_config.json --update_simularium

# Upload simularium files to S3
python run_data_release_workflow.py --config_file configs/data_release_config.json --upload_simularium

# Generate and upload thumbnails
python run_data_release_workflow.py --config_file configs/data_release_config.json --upload_thumbnails

# Create CSV files only
python run_data_release_workflow.py --config_file configs/data_release_config.json --create_csv

# Combine multiple steps
python run_data_release_workflow.py --config_file configs/data_release_config.json \
    --update_simularium --create_csv
```

### Programmatic Usage

```python
from pathlib import Path
from cellpack_analysis.notebooks.data_release.run_data_release_workflow import (
    run_data_release_workflow
)

# Run all enabled steps
config_file = Path("configs/data_release_config.json")
run_data_release_workflow(config_file)

# Run specific steps
run_data_release_workflow(config_file, steps_to_run=["update_simularium", "create_csv"])
```

## Configuration

The workflow is configured using a JSON file. See `configs/data_release_config.json` for an example.

### Configuration Parameters

#### S3 Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `s3_bucket` | string | S3 bucket name | `"cellpack-analysis-data"` |
| `base_s3_url` | string | Base S3 URL | `"https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/"` |

#### Dataset Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `dataset` | string | Dataset name | `"8d_sphere_data"` |
| `condition` | string | Experimental condition | `"rules_shape"` |
| `experiment` | string | Experiment name | `"norm_weights"` |
| `rules` | array | List of packing rules to process | `["random", "nucleus_gradient", ...]` |
| `structures` | array | List of structure configurations | See below |
| `base_channel_colors` | object | RGB colors for nucleus and membrane | See example config |

#### Workflow Step Toggles

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `upload_meshes_to_s3` | boolean | Enable mesh upload step | `true` |
| `update_simularium_files` | boolean | Enable simularium file update step | `true` |
| `upload_simularium_to_s3` | boolean | Enable simularium upload step | `true` |
| `generate_thumbnails` | boolean | Enable thumbnail generation step | `true` |
| `create_csv` | boolean | Enable CSV creation step | `true` |

#### Mesh Upload Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `use_inverted_meshes` | boolean | Invert mesh faces before uploading | `false` |
| `reinvert_meshes` | boolean | Force re-inversion of existing meshes | `false` |

#### Processing Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `reupload_simularium_files` | boolean | Force reupload of simularium files | `false` |
| `reupload_thumbnails` | boolean | Force reupload of thumbnails | `false` |
| `max_workers` | integer | Number of parallel workers | `8` |
| `output_name` | string | Output file prefix | `"cellpack_simularium"` |

### Structure Configuration

Each structure in the `structures` array should have:

```json
{
  "structure_id": "SLC25A17",
  "structure_name": "peroxisome",
  "packing_id": "peroxisome",
  "color": [0.12, 1.0, 0.12]
}
```

- `structure_id` - Structure identifier (e.g., gene name)
- `structure_name` - Human-readable structure name
- `packing_id` - Directory name in packing outputs
- `color` - RGB color tuple for visualization (values 0-1)

## Output

The workflow creates the following files in `results/data_release/{output_name}/`:

- `{output_name}_paths.csv` - Main CSV with S3 paths and metadata
- `{output_name}_metadata.csv` - Description of CSV columns
- `thumbnails/` - Directory containing thumbnail images

### Output CSV Columns

| Column | Description |
|--------|-------------|
| File Path | Path to simularium file on S3 |
| Cell ID | Unique identifier for each cell/packing output |
| File Name | Name of the simularium file |
| Packing ID | Identifier for the packing directory |
| Structure ID | Identifier for the structure |
| Structure Name | Name of the structure |
| Rule | Packing rule used |
| Dataset | Dataset name |
| Condition | Experimental condition |
| Experiment | Experiment name |
| File Type | Type of file (simularium) |
| Thumbnail | Path to thumbnail file on S3 |
| Count | Count of structures packed in the cell |
| Cell Volume | Volume of the cell |
| Nucleus Volume | Volume of the nucleus |
| Cell Height | Height of the cell |
| Nucleus Height | Height of the nucleus |
| Sphericity | Sphericity of the cell |

## Typical Workflow

### Initial Setup (One-time)

1. Upload meshes to S3:
```bash
python run_data_release_workflow.py --config_file configs/data_release_config.json --upload_meshes
```

### Iterative Development

2. Update simularium files locally (test changes without uploading):
```bash
python run_data_release_workflow.py --config_file configs/data_release_config.json --update_simularium
```

3. Create CSV to verify metadata:
```bash
python run_data_release_workflow.py --config_file configs/data_release_config.json --create_csv
```

### Final Release

4. Upload everything to S3:
```bash
# Update config to enable all upload steps
# Then run:
python run_data_release_workflow.py --config_file configs/data_release_config.json
```

Or run specific upload steps:
```bash
python run_data_release_workflow.py --config_file configs/data_release_config.json \
    --upload_simularium --upload_thumbnails
```

## Requirements

- AWS credentials configured for S3 access
- boto3
- pandas
- tqdm
- matplotlib
- Access to cellpack-analysis data and results directories

