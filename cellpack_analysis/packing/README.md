# Packing

Generates cellPACK recipes and packing configs from each cell's measured geometry and puncta counts,
and runs the packings locally or on SLURM. The paper analysis uses the published packings (see
`cellpack_analysis/data_release/README.md`), so nothing here is needed to reproduce the figures. The
engine is the cellPACK revision pinned in `pyproject.toml`; it is run as a subprocess from the
virtual environment's copy (`pack_recipes.py`), so do not set `CELLPACK` or `PYTHONPATH`.

| File | Role |
|---|---|
| `run_packing_workflow.py` | local entry point |
| `run_packing_workflow_slurm.py` | SLURM entry point (orchestrator, worker and aggregator modes) |
| `submit_packing_slurm.sh` | bash launcher: submits the orchestrator job from the login node |
| `submit_replicate_99_slurm.sh` | launcher for the paper's `replicate_99_v4` packings |
| `generate_cellpack_input_files.py` | per-cell recipe and packing config generation |
| `pack_recipes.py` | runs one packing (`run_single_packing`) or a set (`pack_recipes`) |
| `workflow_config.py` | `WorkflowConfig`: reads the workflow JSON and fills defaults |
| `rule_repository.py` | gradient and modeling-rule definitions (decay lengths of Table 3) |
| `configs/replicate_99_v4/` | `{peroxisome,endosome}_replicate_99_v4.json` (four single rules) and `..._interpolated.json` (mixed rule) |
| `configs/tune_gradients_v2/` | `{peroxisome,endosome}_tune_gradients_v2.json` (decay-length ladder, Fig S10) |
| `configs/biological_variation/` | `biological_variation_{baseline,shape,variable_count,variable_size}.json` (Fig 2, Fig S3, Table S1) |

## Local run and recipe regeneration

```bash
uv run python -m cellpack_analysis.packing.run_packing_workflow \
    -c cellpack_analysis/packing/configs/replicate_99_v4/peroxisome_replicate_99_v4.json [--dry-run] [-v]
```

`-c` is required. `--dry-run` generates the
packing configs and recipes and skips every packing. The output goes to
`data/recipes/<packing_id>/<condition>/<rule>/` and `data/configs/<packing_id>/<condition>/`, the
locations the `packing_inputs` group extracts to. Regenerating needs the `tables`, `meshes_*` groups
(`paper_inputs.py fetch --groups tables,meshes_SLC25A17,meshes_RAB5A,meshes_mean`) and the tracked
templates in `data/templates/`. Regenerated recipes are byte-identical to the recipes of record
except for the absolute data directory prefix they embed.

## Workflow configuration

Keys of the workflow JSON (defaults in `cellpack_analysis/lib/default_values.py`):

| Key | Meaning |
|---|---|
| `packing_id`, `structure_id`, `structure_name` | structure to pack (`peroxisome` / `SLC25A17`, `endosome` / `RAB5A`) |
| `condition` | name of the modeling-rule variant; keys the output directories (default `replicate_99_v4`) |
| `datadir` | data root (default: repository `data/`) |
| `dry_run` | generate recipes and configs only |
| `generate_recipes`, `generate_configs` | write recipe and config files |
| `get_counts_from_data`, `get_size_from_data`, `get_bounding_box_from_mesh` | take puncta count, radius and bounding box from each cell's data |
| `number_of_replicates` | seeds per cell (99 in the paper) |
| `result_type` | engine output type |
| `skip_completed` | skip recipes whose outputs already exist |
| `use_mean_cell` | pack in the mean cell geometry |
| `use_cells_in_8d_sphere`, `num_cells` | restrict to the 8d-sphere cohort and/or the first N cells |
| `use_additional_struct`, `gradient_structure_name` | structure-based gradient options |
| `num_processes` | local worker processes |
| `recipe_template_path`, `config_template_path` | templates (default `<datadir>/templates/{recipes,configs}/<condition>/<packing_id>_{recipe,config}_template.json`) |
| `generated_recipe_path`, `generated_config_path` | generated files (default `<datadir>/recipes/...`, `<datadir>/configs/...`) |
| `grid_path`, `mesh_path` | grid cache and meshes (default `<datadir>/structure_data/<structure_id>/{grids,meshes}`) |
| `output_path` | packing outputs (default `<datadir>/packing_outputs/<8d_sphere_data or full_variance_data>/<condition>`) |
| `recipe_data` | per-rule recipe settings (`gradients` list) |
| `packings_to_run` | `rules`, `cell_ids`, `number_of_packings` |

Packing outputs land in `<output_path>/<rule>/<packing_id>/spheresSST/`.

## SLURM

`submit_packing_slurm.sh` submits an orchestrator job that generates recipes, submits worker jobs
(`--batch-size` recipes each, as a job array limited by `--max-jobs`) and an aggregator job that
collects the per-batch results into `<output_path>/slurm_staging/<packing_id>/aggregated_results.json`.
Per-job logs go to `<output_path>/logs/slurm/`.

```bash
bash cellpack_analysis/packing/submit_packing_slurm.sh -c CONFIG.json [-b N] [-p PARTITION] [-t TIME] \
    [-m MEM] [--cpus N] [--max-jobs N] [--no-wait]
bash cellpack_analysis/packing/submit_replicate_99_slurm.sh [-s peroxisome|endosome|both] \
    [--suffix S] [--condition C] [-b N] [--max-jobs N] [-p PARTITION]
```

Run either script with `-h` or `--help` for every option and its default. The paper's packings are
driven by `run_paper.sh pack`, `stage2-config` and `pack2`
(`cellpack_analysis/analysis/workflows/README.md`); `submit_replicate_99_slurm.sh` defaults to
partition `aics` and is sized for about 256 CPUs with both structures.

Notes:

- `--dry-run` on the SLURM launchers still submits the orchestrator job (it only suppresses the
  workers). To preview recipe generation, use the local `--dry-run` above.
- `run_packing_workflow_slurm.py` can be called directly (`-c`, `-b`, `--orchestrate`, `--worker`,
  `--aggregate FILE`, `--no-wait`, `--max-jobs`, `--venv-path`, `--slurm-*`); the bash launcher is
  the supported route.
- Repacking writes into the same `data/packing_outputs/` paths that `fetch` fills with the published
  positions. Use a separate clone.
- Packings of the sources-of-variation analysis (Fig 2, Fig S3, Table S1) were made with an older
  engine (`14225bcb` for shape, baseline and variable_size; `ae4f0aa6` or later for variable_count);
  repacking at the release pin is not expected to reproduce them exactly.
