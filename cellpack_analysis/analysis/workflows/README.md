# Analysis Workflow Runner

This module provides a unified interface to run different types of cellpack analysis
workflows, driven entirely by JSON configuration files.

## Installation

```bash
uv sync
```

## Usage

Run an analysis using a configuration file:

```bash
python cellpack_analysis/analysis/workflows/run_analysis_workflow.py --config_file path/to/config.json
```

Control the logging level:

```bash
python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \
    --config_file path/to/config.json --log_level DEBUG
```

---

## Combined Packing + Analysis Workflow

`packing_and_analysis_workflow.sh` runs the full pipeline (packing → analysis) from a
single command.

### Local packing + analysis

```bash
bash cellpack_analysis/analysis/workflows/packing_and_analysis_workflow.sh \
    -p cellpack_analysis/packing/configs/peroxisome.json \
    -a cellpack_analysis/analysis/workflows/configs/analysis_config_peroxisome.json
```

### SLURM packing + local analysis (two-step)

Because SLURM packing is asynchronous (fire-and-forget), analysis **cannot** be chained
automatically. The script submits the packing jobs and then prints the analysis command
to run once the jobs finish.

```bash
# Step 1 — submit packing to SLURM
bash cellpack_analysis/analysis/workflows/packing_and_analysis_workflow.sh \
    -p cellpack_analysis/packing/configs/peroxisome.json \
    -a cellpack_analysis/analysis/workflows/configs/analysis_config_peroxisome.json \
    --slurm --batch-size 16 --partition my_partition

# Step 2 — after SLURM jobs finish, run analysis (command is printed by Step 1)
python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \
    --config_file cellpack_analysis/analysis/workflows/configs/analysis_config_peroxisome.json
```

### Selective execution

```bash
# Analysis only (packing already done)
bash cellpack_analysis/analysis/workflows/packing_and_analysis_workflow.sh \
    -a cellpack_analysis/analysis/workflows/configs/analysis_config_peroxisome.json \
    --skip-packing

# Packing only
bash cellpack_analysis/analysis/workflows/packing_and_analysis_workflow.sh \
    -p cellpack_analysis/packing/configs/peroxisome.json \
    --skip-analysis
```

### Full option reference

| Option | Description |
|---|---|
| `-p / --packing-config` | Packing workflow JSON config (required unless `--skip-packing`) |
| `-a / --analysis-config` | Analysis workflow JSON config (required unless `--skip-analysis`) |
| `--slurm` | Submit packing via SLURM; analysis is **not** run automatically |
| `--skip-packing` | Run analysis only |
| `--skip-analysis` | Run packing only |
| `--log-level` | Log level for analysis (default: `INFO`) |
| `-b / --batch-size` | Recipes per SLURM worker (default: 8) |
| `-v / --venv` | Path to virtualenv activate script (auto-detected) |
| `--partition` | SLURM partition |
| `-t / --time` | Worker wall-clock limit (default: `00:30:00`) |
| `-m / --mem` | Worker memory (default: `16G`) |
| `--cpus` | CPUs per worker (default: 4) |
| `--max-jobs` | Max concurrent SLURM workers (default: 16) |
| `--job-name` | SLURM job name prefix (default: `cellpack`) |
| `--dry-run` | Dry-run mode for SLURM orchestrator |

---

## Analysis Types

The script supports three types of analysis, selected via the `analysis_steps` field in
the config:

1. **Distance Analysis** — Compare distance distributions using EMD, pairwise KS tests,
   and pairwise Monte Carlo envelope tests.
2. **Biological Variation** — Analyze variation in spatial organization due to biological
   factors (size, count, shape).
3. **Occupancy Analysis** — Analyze how punctate structures occupy available space using
   discrete histogram or KDE methods.

---

## Available Analysis Steps

| Step | Description |
|---|---|
| `load_common_data` | Load position data and mesh information |
| `calculate_distances` | Calculate distance measures and (optionally) normalize |
| `plot_distance_distributions` | Plot per-cell distance histograms (discrete) or KDE curves |
| `run_emd_analysis` | EMD across packing modes + pairwise EMD matrix plots |
| `run_ks_analysis` | Pairwise KS tests with bootstrap CIs for every ordered mode pair |
| `run_pairwise_envelope_test` | Pairwise Monte Carlo rank-envelope test on distance distributions |
| `run_occupancy_analysis` | Occupancy ratio analysis (discrete histogram or KDE) |
| `run_occupancy_emd_analysis` | EMD on occupancy ratios + pairwise EMD matrix plots |
| `run_occupancy_pairwise_envelope_test` | Pairwise Monte Carlo rank-envelope test on occupancy ratio curves |

### Example step lists

```json
// Distance analysis
"analysis_steps": [
    "load_common_data", "calculate_distances",
    "plot_distance_distributions", "run_emd_analysis",
    "run_ks_analysis", "run_pairwise_envelope_test"
]

// Biological variation
"analysis_steps": [
    "load_common_data", "calculate_distances",
    "plot_distance_distributions", "run_emd_analysis",
    "run_pairwise_envelope_test"
]

// Occupancy analysis
"analysis_steps": [
    "load_common_data", "calculate_distances",
    "run_occupancy_analysis", "run_occupancy_emd_analysis",
    "run_occupancy_pairwise_envelope_test"
]
```

---

## Configuration File Format

Config files are JSON files in `cellpack_analysis/analysis/workflows/configs/`.

### Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `name` | Analysis identifier (used for output folder naming) | required |
| `structure_id` | ID of the punctate structure (e.g. `"SLC25A17"`) | `"SLC25A17"` |
| `packing_id` | Packing configuration ID (used for folder naming) | `"peroxisome"` |
| `structure_name` | Human-readable structure name | `"peroxisome"` |
| `channel_map` | Dict mapping packing mode to structure ID for mesh lookup | `{structure_id: structure_id}` |
| `packing_output_folder` | Relative path to packing outputs | see defaults |
| `baseline_mode` | Reference packing mode for comparisons | `structure_id` |
| `distance_measures` | List of distance measures to compute | `["nearest","pairwise","nucleus","z"]` |
| `analysis_steps` | Ordered list of steps to execute | required |
| `distribution_method` | `"discrete"` (histogram) or `"kde"` (kernel density). **Default: `"discrete"`** | `"discrete"` |
| `normalization` | Distance normalization: `null`, `"intracellular_radius"`, `"cell_diameter"`, `"max_distance"` | `null` |
| `recalculate` | `bool` or `dict` of per-step flags to force recomputation | all `false` |
| `num_workers` | Parallel workers for distance calculation | `16` |
| `save_format` | Figure format: `"pdf"`, `"svg"`, `"png"` | `"pdf"` |

### Distance distribution parameters

| Parameter | Description | Default |
|---|---|---|
| `bandwidth` | KDE bandwidth (KDE mode only) | `0.2` |
| `bin_width_map` | Dict of per-distance-measure histogram bin widths (discrete mode) | `{"nucleus": 0.2, "z": 0.2, ...}` |

### KS test parameters

| Parameter | Description | Default |
|---|---|---|
| `ks_significance_level` | Significance level for KS tests | `0.05` |
| `n_bootstrap` | Bootstrap samples for KS CIs | `1000` |

### Envelope test parameters

Configure via the `envelope_test_params` dict:

| Key | Description | Default |
|---|---|---|
| `alpha` | Significance level | `0.05` |
| `r_grid_size` | Grid size for rank-envelope test | `150` |
| `bin_width` | Bin width for ECDF grid | `0.2` |
| `statistic` | Test statistic: `"intdev"` or `"supremum"` | `"intdev"` |

```json
"envelope_test_params": {"alpha": 0.05, "r_grid_size": 150, "bin_width": 0.2, "statistic": "intdev"}
```

### Occupancy analysis parameters

| Parameter | Description | Default |
|---|---|---|
| `occupancy_distance_measures` | Distance measures for occupancy | `["nucleus", "z"]` |
| `occupancy_params` | Per-measure dict with `xlim`, `ylim`, `bandwidth` | see defaults |
| `discrete_occupancy_params` | Extra params for discrete mode: `pseudocount`, `min_count`, `x_min` | see defaults |

```json
"occupancy_params": {
    "nucleus": {"xlim": 6, "ylim": 3},
    "z": {"xlim": 8, "ylim": 2}
}
```

> **Breaking change note:** `distribution_method` defaults to `"discrete"` (histogram-based).
> Existing configs that relied on KDE-based `plot_distance_distributions` or `run_occupancy_analysis`
> should add `"distribution_method": "kde"` to preserve the previous behaviour.

---

## Output

Results are saved under `results/<name>/<packing_id>/`, organized by step. Figures are
saved in `figures/` subdirectories per step. Log files for central tendencies and
statistical tests are written alongside the figures.
