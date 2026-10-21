# Analysis workflows

Config-driven analysis of observed and packed puncta positions. `run_paper.sh` runs the analyses
for the paper; its stages are listed in the top-level README.

## Entry points

| Script | Use |
|---|---|
| `run_paper.sh` | paper stage driver, one stage per call |
| `run_analysis_workflow.py` | distance, EMD, KS, envelope-test and occupancy analyses |
| `run_rule_interpolation_workflow.py` | NNLS rule mixing (`--phase fit`) and mixed-rule validation (`--phase validate`) |
| `run_manuscript_workflow.py` | figure panels, numbers sheets and calibration summary from the cached results |

Run the Python entry points as modules with a config file. Use `--help` for all options.

```bash
uv run python -m cellpack_analysis.analysis.workflows.run_analysis_workflow -c CONFIG.json
```

## Configs

Each config is a JSON file under `configs/`. Results go to `results/<name>/<packing_id>/`. Defaults
for every key are in `configs/defaults.py`.

| Config | Stage |
|---|---|
| `distance_analysis/distance_analysis_{peroxisome,endosome}_replicate_99_v4_calibration.json` | `calibration` |
| `distance_analysis/distance_analysis_{peroxisome,endosome}_replicate_99_v4.json` | `distance` |
| `distance_analysis/distance_analysis_supp_{peroxisome,endosome}_replicate_99_v4.json` | `distance-supp` |
| `occupancy_analysis/occupancy_analysis_{peroxisome,endosome}_replicate_99_v4.json` | `occupancy` |
| `rule_interpolation_replicate_99_v4/rule_interpolation_replicate_99_v4_{peroxisome,endosome}.json` | `fit`, `validate` |
| `rule_interpolation_validation/replicate_99_v4/mixed_rule_validation_config_{peroxisome,endosome}.json` | written by `fit`, read by `validate` |
| `rule_interpolation/rule_interpolation_tune_gradients_v2_{peroxisome,endosome}.json` | `ladder` |
| `biological_variation/biological_variation_config.json` | `biovar` |
| `manuscript/manuscript_replicate_99_v4.json` | `manuscript` |

`calibration` writes the distance cache that `distance` reuses, so run it first.

## Analysis steps

`analysis_steps` in a config lists the steps of `run_analysis_workflow.py` to run, in order:

| Step | Action |
|---|---|
| `load_common_data` | load cell tables, meshes and available-space distances |
| `calculate_distances` | distances of observed and packed puncta |
| `plot_distance_distributions` | distance distributions per rule |
| `run_emd_analysis` | earth mover's distances between rules |
| `run_ks_analysis` | KS tests with a bootstrap over cells |
| `run_pairwise_envelope_test` | envelope test pooled across cells (sources of variation) |
| `run_per_cell_envelope_test` | envelope test of each cell against its own replicates |
| `run_occupancy_analysis` | occupancy curves |
| `run_occupancy_emd_analysis` | EMD between occupancy curves |
| `run_occupancy_pairwise_envelope_test` | envelope test on occupancy curves |
| `run_rule_interpolation_cv` | cross-validated NNLS fit |
| `plot_lambda_ladder_figures` | decay-length ladder figures |

`run_manuscript_workflow.py` recomputes nothing. A missing cache raises an error that names the
step that writes it. Each run appends a line to `run_provenance.jsonl` with the command, code
version, engine commit and config checksum.
