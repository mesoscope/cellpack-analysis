# Packing

Generates cellPACK recipes and packing configs from each cell's geometry and puncta counts, and runs
the packings locally or on SLURM. The paper analysis uses the published packings, so nothing here is
needed to reproduce the figures.

| File | Use |
|---|---|
| `run_packing_workflow.py` | local entry point |
| `submit_packing_slurm.sh` | SLURM launcher |
| `submit_replicate_99_slurm.sh` | SLURM launcher for the paper's `replicate_99_v4` packings |
| `generate_cellpack_input_files.py` | recipe and packing config generation |
| `rule_repository.py` | modeling rules and decay lengths (Table 3) |
| `configs/replicate_99_v4/` | the four single rules and the mixed rule (`..._interpolated.json`) |
| `configs/tune_gradients_v2/` | decay-length ladder (Fig S10) |
| `configs/biological_variation/` | sources of variation (Fig 2, Fig S3, Table S1) |

## Local run

```bash
uv run python -m cellpack_analysis.packing.run_packing_workflow \
    -c cellpack_analysis/packing/configs/replicate_99_v4/peroxisome_replicate_99_v4.json
```

Add `--dry-run` to write the recipes and packing configs to `data/recipes/` and `data/configs/`
without packing. This needs the cell tables and meshes:

```bash
uv run python -m cellpack_analysis.data_release.paper_inputs fetch \
    --groups tables,meshes_SLC25A17,meshes_RAB5A,meshes_mean
```

Config keys and defaults are in `workflow_config.py` and `cellpack_analysis/lib/default_values.py`.
Packing outputs go to `data/packing_outputs/`.

## SLURM

```bash
bash cellpack_analysis/packing/submit_packing_slurm.sh -c CONFIG.json
bash cellpack_analysis/packing/submit_replicate_99_slurm.sh -s peroxisome
```

Run either script with `--help` for the options. The paper's packings run through `run_paper.sh pack`,
`stage2-config` and `pack2`. `--dry-run` on the SLURM launchers still submits the orchestrator job,
so use the local `--dry-run` to preview recipes.
