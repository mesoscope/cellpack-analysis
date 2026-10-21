# Analysis workflows

Config-driven analysis of observed and packed puncta positions. The paper is reproduced by
`run_paper.sh`, which calls the entry points below with the configs under `configs/`; the figure
and table map is in the top-level `README.md`.

## Entry points

| Script | Use | Arguments |
|---|---|---|
| `run_paper.sh` | one paper stage per call | `bash run_paper.sh <stage>`; env `STRUCTS` (default `"peroxisome endosome"`); no stage prints the stage list |
| `run_analysis_workflow.py` | distance, EMD, KS, envelope-test and occupancy analyses | `--config_file/-c FILE` (required), `--log_level DEBUG\|INFO\|WARNING\|ERROR` |
| `run_rule_interpolation_workflow.py` | NNLS rule mixing: fit, optional mixed-rule packing, validation | `-c FILE` (required), `--phase fit\|pack\|validate\|all` (default `fit`), `--local`, `--dry_run`, `--log_level` |
| `run_manuscript_workflow.py` | draw the figure panels, numbers sheets and calibration summary from the caches | `-c FILE` (required), `--only REGEX`, `--native DIR`, `--steps STEP ...`, `--log_level` |

Run the Python entry points with `uv run python -m cellpack_analysis.analysis.workflows.<module> ...`.

Stages of `run_paper.sh` (all use `replicate_99_v4`; `analysis` runs `calibration` through
`manuscript` in this order):

| Stage | Command it runs |
|---|---|
| `fetch` | `python -m cellpack_analysis.data_release.paper_inputs fetch --groups all` |
| `calibration`, `distance`, `distance-supp`, `occupancy`, `biovar` | `run_analysis_workflow -c <config>` per structure (biovar: one config) |
| `fit`, `validate`, `ladder` | `run_rule_interpolation_workflow -c <config> --phase fit` or `--phase validate` |
| `illustrations` | the five scripts in `cellpack_analysis/analysis/notebooks/`, run with `MPLBACKEND=Agg` |
| `manuscript` | `run_manuscript_workflow -c configs/manuscript/manuscript_replicate_99_v4.json` |
| `pack`, `pack2` | SLURM packing through `packing/submit_replicate_99_slurm.sh` (optional) |
| `stage2-config` | copy the fit's mixed-rule packing config into `packing/configs/replicate_99_v4/` (optional) |

`calibration` writes the distance cache that `distance` reuses, so run it first. `run_paper.sh`
requires `.venv/bin/python` (`uv sync --frozen`) and unsets `CELLPACK` and `PYTHONPATH`.

## Analysis steps

`analysis_steps` in a config selects and orders these steps of `run_analysis_workflow.py`:

| Step | Action |
|---|---|
| `load_common_data` | load cell tables, meshes and available-space distances |
| `calculate_distances` | distances of observed and packed puncta (nucleus, z, nearest, pairwise) |
| `plot_distance_distributions` | distance distributions per rule |
| `run_emd_analysis` | pairwise earth mover's distances between rules |
| `run_ks_analysis` | KS tests with a bootstrap over cells |
| `run_pairwise_envelope_test` | Monte Carlo envelope test pooled across cells (used for the sources-of-variation analysis) |
| `run_per_cell_envelope_test` | per-cell envelope test against each cell's own replicates, Benjamini-Hochberg across cells |
| `run_occupancy_analysis` | occupancy ratio curves with available-volume normalization |
| `run_occupancy_emd_analysis` | EMD between occupancy curves |
| `run_occupancy_pairwise_envelope_test` | envelope test on occupancy curves |
| `run_rule_interpolation_cv` | cross-validated NNLS fit (used by the rule-interpolation workflow) |
| `plot_lambda_ladder_figures` | decay-length ladder figures |

Manuscript steps (`run_manuscript_workflow.py`, always in this order): `check_inputs`,
`figure_assets`, `figure_numbers`, `calibration_summary`, `manifest`. Nothing is recomputed: a
missing or mismatched cache raises and names the step that writes it. Set `SOURCE_DATE_EPOCH` for
byte-reproducible PDFs.

Every run appends one JSON line to `run_provenance.jsonl`: in the logs directory of the analysis
workflow, in `logs/` of the results directory of each rule-interpolation phase, and in the results
directory of the manuscript workflow (the packing workflow writes it to the `logs/` of its output
path). The line holds the UTC time, `argv`, `git describe` of the checkout (`code_version`), the
engine commit, the Python version, the SHA-256 of `uv.lock`, and the config file with its SHA-256.

## Configuration

A JSON file; results go to `results/<name>/<packing_id>/` unless `output_path` is set. Keys of the
analysis configs (class `AnalysisConfig`; defaults in `configs/defaults.py`):

| Key | Default | Meaning |
|---|---|---|
| `name` | none | experiment name; keys the results directory |
| `structure_id`, `packing_id`, `structure_name` | `SLC25A17`, `peroxisome`, `peroxisome` | structure |
| `channel_map` | `{structure_id: structure_id}` | packing mode to structure; the keys are the modes analysed |
| `packing_output_folder` | `packing_outputs/8d_sphere_data/replicate_99_v4/` | packings to read, relative to `data/` |
| `baseline_mode`, `ingredient_key` | `structure_id`, `membrane_interior_<structure_name>` | reference (observed) mode and the ingredient key read from the packing outputs |
| `analysis_steps` | none | ordered list of steps above |
| `distance_measures` | `nearest`, `pairwise`, `nucleus`, `z` | distance measures to compute |
| `minimum_distance_um` | `-0.108` | floor (µm) on observed and simulated distances (-1 px) |
| `available_minimum_distance_um` | `-1.0` | floor (µm) on available-space distances |
| `distribution_method` | `kde` | `kde` or `discrete` (histogram) |
| `bandwidth` | `0.2` | KDE bandwidth |
| `bin_width_map` | 0.2 per measure | histogram bin widths (discrete mode) |
| `emd_pool_seeds` | `false` | pool each cell's seeds into one curve before the pairwise EMD |
| `emd_same_cell` | `false` | pair each cell only with itself in cross-mode EMD |
| `ks_significance_level`, `n_bootstrap`, `ks_bootstrap_seed` | `0.05`, `1000`, `null` | KS test; `null` leaves the bootstrap unseeded (the v4 configs set `0`) |
| `occupancy_distance_measures` | `nucleus`, `z` | measures for occupancy |
| `occupancy_params` | see `defaults.py` | per-measure plot limits, bandwidth, grid points (`num_points` 250) |
| `discrete_occupancy_params` | see `defaults.py` | pseudocount, `min_count`, `x_min` for discrete occupancy |
| `envelope_test_params` | `alpha` 0.05, `bin_width` 0.2, `statistic` `intdev` | envelope tests (below) |
| `num_workers` | `8` | parallel workers |
| `distance_backend`, `occupancy_backend` | `thread` | `thread` or `process` executors |
| `recalculate` | all `false` | per-step flags that force recomputation instead of loading the cache |
| `suffix` | empty | suffix for cache and figure file names (`normalization` appends `_norm_<normalization>`) |
| `save_format` | `pdf` | figure format |
| `output_path`, `figures_dir`, `logs_dir` | derived from `name` | output locations |

The legacy keys `filter_minimum_distance` and `available_minimum_distance` raise an error; use the
`..._um` keys.

Envelope-test options (`envelope_test_params`): `pairwise_envelope_test` takes `alpha`,
`r_grid_size` (150), `bin_width`, `statistic` (`intdev` or `supremum`), `envelope_type`
(`pointwise` by default, or `rank`) and `joint_r_grid_size`. `per_cell_envelope_test` adds
`min_replicates` (2) and `calibration_holdout_index`, and its default `envelope_type` is `rank`; the
v4 configs set `pointwise` with `intdev`. `calibration_holdout_index` goes inside
`envelope_test_params`.

Rule-interpolation configs (`run_rule_interpolation_workflow.py`) add `rule_interpolation_cv_params`
(`n_folds` 5, `n_repeats` 10, `random_state` 42, `grouping`, `packing_config_scope`),
`base_packing_config_path`, `mode_to_gradient_name`, `validation_analysis_steps`,
`validation_config_out_path`, `validation_output_path`, `slurm_kwargs` and
`skip_mixed_rule_validation`. The `fit` phase writes the mixed-rule packing config and the
validation configs in `configs/rule_interpolation_validation/replicate_99_v4/`; `validate` reads
them. `fit` runs locally. `--phase pack` submits SLURM jobs without a job limit (or runs locally
with `--local`); use `run_paper.sh stage2-config` and `pack2` instead.

## Files

| Path | Contents |
|---|---|
| `run_paper.sh` | paper stage driver |
| `run_analysis_workflow.py` | distance, EMD, KS, envelope and occupancy analyses |
| `run_rule_interpolation_workflow.py` | NNLS rule mixing (fit, pack, validate) |
| `run_manuscript_workflow.py` | figure assets, numbers sheets, calibration summary, manifest |
| `manuscript_config.py`, `manuscript_panels.py`, `manuscript_sources.py` | manuscript config schema, panel drawing, cache readers |
| `manuscript_numbers.py`, `manuscript_calibration.py` | numbers sheets and calibration summary |
| `configs/defaults.py` | defaults of `AnalysisConfig` |
| `configs/distance_analysis/` | `distance_analysis_{peroxisome,endosome}_replicate_99_v4{,_calibration}.json`, `distance_analysis_supp_*_replicate_99_v4.json` |
| `configs/occupancy_analysis/` | `occupancy_analysis_{peroxisome,endosome}_replicate_99_v4.json` |
| `configs/rule_interpolation_replicate_99_v4/` | `rule_interpolation_replicate_99_v4_{peroxisome,endosome}.json` |
| `configs/rule_interpolation_validation/replicate_99_v4/` | `mixed_rule_validation_config_{peroxisome,endosome}.json` (rewritten by `fit`) |
| `configs/rule_interpolation/` | `rule_interpolation_tune_gradients_v2_{peroxisome,endosome}.json` (decay-length ladder) |
| `configs/biological_variation/` | `biological_variation_config.json` |
| `configs/manuscript/` | `manuscript_replicate_99_v4.json` |
