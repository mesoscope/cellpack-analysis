# Data release

Two independent tools live here:

| Tool | Purpose | Needs credentials |
|---|---|---|
| `paper_inputs.py` | plan, build, fetch and verify the paper's input-data archives | no (fetch uses anonymous HTTPS) |
| `run_data_release_workflow.py` | export packings as Simularium files plus a BioFile Finder (BFF) CSV and upload them to S3 | yes, for uploads |

| File | Role |
|---|---|
| `paper_inputs.py` | plan / build / fetch / verify of the paper input archives |
| `configs/paper_inputs.json` | manifest of 26 groups (file globs, archive name, sha256, size, file count) |
| `run_data_release_workflow.py` | BFF / Simularium export workflow |
| `data_release_config.py` | configuration class and defaults of the export workflow |
| `csv_metadata_config.py` | column definitions of the export CSV and its metadata CSV |
| `configs/replicate_99_v4/data_release_config.json` | configuration used for the paper's BFF dataset |
| `upload_meshes_to_s3.py` | notebook-style script, not called by the workflow (use `--upload_meshes`) |

## Paper inputs (`paper_inputs.py`)

The analysis inputs of record are published as one deterministic `.tar.gz` per group, listed in
`configs/paper_inputs.json` (tag `v1.0.0`, base URL
`https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/release/v1.0.0/`). Archives extract into
`data/`, so paths mirror the repository. All 26 groups total 4.99 GB to download, 30.68 GB extracted,
43,876 files. Only packing positions are shipped (no Simularium files).

```bash
bash cellpack_analysis/analysis/workflows/run_paper.sh fetch        # all groups
uv run python -m cellpack_analysis.data_release.paper_inputs fetch --groups tables,meshes_RAB5A
uv run python -m cellpack_analysis.data_release.paper_inputs verify
```

| Command | Arguments | Action |
|---|---|---|
| `plan` | `--data-dir D` (required), `--out plan.tsv`, `--manifest P`, `--groups a,b` | resolve the globs against `D`, print file counts and sizes; exit 1 if flagged |
| `build` | `--data-dir D` and `--out-dir O` (required), `--workers N`, `--manifest P`, `--groups a,b` | write deterministic archives and a built manifest (sha256, bytes, n_files) to `O` |
| `fetch` | `--data-dir D`, `--manifest P`, `--groups all\|a,b`, `--source URL_OR_LOCAL_DIR`, `--keep-archives` | download, sha256-verify and extract into `D` (default: this repository's `data/`) |
| `verify` | `--data-dir D`, `--manifest P`, `--groups a,b` | check that stamped groups are still complete; exit 1 on problems |

Behavior of `fetch`:

- Downloads use anonymous HTTPS GET only (`s3://` URLs are refused). A sha256 mismatch deletes the
  download and fails.
- Each extracted group is stamped in `data/.paper_inputs/<group>.json`; a group whose stamp matches
  the manifest sha256 and tag is skipped. Downloads go to `data/.paper_inputs/downloads/` and are
  removed unless `--keep-archives` is given.
- `fetch` refuses a manifest that was not built (null sha256); the shipped manifest is the built one.
- Exit codes: 0 success, 1 plan flagged or verify found problems, 2 error.

Groups:

| Group | Contents |
|---|---|
| `tables` | `all_cell_ids`, `variance_dataset` and `structure_stats` parquet files |
| `observed_{SLC25A17,RAB5A}` | observed puncta positions of the 8d-sphere cohort (305 peroxisome, 442 endosome cells) |
| `meshes_{SLC25A17,RAB5A,mean}` | per-cell nucleus and membrane meshes; `mean` is the mean cell used by the baseline, variable_count and variable_size packings |
| `mesh_information_{SLC25A17,RAB5A,mean}` | per-cell mesh information |
| `grid_distances_{SLC25A17,RAB5A,mean}` | per-cell available-space grid points and their distances |
| `packings_replicate_99_v4_{peroxisome,endosome}_{random,nucleus_gradient,membrane_gradient,apical_gradient,interpolated}` | positions of 99 replicates per cell per modeling rule (random groups add `ENGINE_PROVENANCE.txt`) |
| `packings_tune_gradients_v2` | decay-length ladder packings (Fig S10) |
| `packings_biological_variation` | sources-of-variation packings (Fig 2, Fig S3, Table S1) |
| `packings_fig4c_illustration` | the single random-rule packing of cell 743916 shown in Fig 4c |
| `packing_inputs` | recipes and packing configs of record (provenance; embed the absolute data path of the machine that wrote them) |

Run `plan` and `build` only to produce a new release: `build` writes the archives and a manifest
with their checksums, which then replaces `configs/paper_inputs.json`.

## BFF / Simularium export (`run_data_release_workflow.py`)

Exports packings as Simularium files and writes the CSVs that describe them for BFF. It reads
`data/packing_outputs/<dataset>/<experiment>/<rule>/<packing_id>/spheresSST/*.simularium`, so it needs
per-seed Simularium outputs, which are not part of the paper inputs; generate them by repacking
(see `cellpack_analysis/packing/README.md`). The voxelized-image step needs only positions and
recipes.

Steps, each runnable alone:

| Order | Step | Flag | Config toggle |
|---|---|---|---|
| 1 | upload meshes to S3 | `--upload_meshes` | `upload_meshes_to_s3` |
| 1a | rebuild each cell's combined `_all` Simularium file from per-seed files (reads `source_experiment`, writes `experiment`) | `--rebuild_all_simularium` | `rebuild_all_simularium` |
| 1b | rebuild the voxelized image of each packing's first replicate from recipe and positions, and save its thumbnail locally | `--export_voxelized_images` | `export_voxelized_images` |
| 2 | update mesh URLs and colors in the Simularium files (local) | `--update_simularium` | `update_simularium_files` |
| 3 | upload Simularium files | `--upload_simularium` | `upload_simularium_to_s3` |
| 4 | generate and upload thumbnails | `--upload_thumbnails` | `generate_thumbnails` |
| 5 | write the paths CSV and metadata CSV | `--create_csv` | `create_csv`, `create_metadata_csv` |
| 5a | refresh cell statistics columns of an existing CSV | `--update_csv_stats` | `update_csv_stats` |
| 6 | upload the CSVs | none | `upload_csv_to_s3`, `upload_metadata_csv` |

With no step flag, every step enabled in the config runs. `--dry_run` logs what would be done
without uploading or modifying files.

```bash
uv run python -m cellpack_analysis.data_release.run_data_release_workflow \
    -c cellpack_analysis/data_release/configs/replicate_99_v4/data_release_config.json --dry_run
uv run python -m cellpack_analysis.data_release.run_data_release_workflow \
    -c cellpack_analysis/data_release/configs/replicate_99_v4/data_release_config.json --create_csv
```

Programmatic use:

```python
from pathlib import Path
from cellpack_analysis.data_release.run_data_release_workflow import run_data_release_workflow

run_data_release_workflow(Path("path/to/config.json"), steps_to_run=["update_simularium", "create_csv"])
```

AWS credentials from the standard boto3 chain (for example `AWS_PROFILE`) are needed only for the upload steps.

### Configuration

| Key | Default | Meaning |
|---|---|---|
| `s3_bucket` | `cellpack-analysis-data` | bucket |
| `base_s3_url` | `https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/` | URL prefix written into the CSV |
| `dataset` | `8d_sphere_data` | dataset folder under `data/packing_outputs/` |
| `condition` | `replicate_99_v4` | condition (modeling-rule variant) name |
| `experiment` | `norm_weights` | folder holding the Simularium files; the paper config uses `replicate_99_v4_release` |
| `source_experiment` | `null` | folder to rebuild combined files and images from; must differ from `experiment` for `rebuild_all_simularium` |
| `rules` | `random`, `nucleus_gradient`, `membrane_gradient`, `apical_gradient` | the paper config adds `interpolated` |
| `structures` | none | list of `{structure_id, structure_name, packing_id, color}` |
| `base_channel_colors` | see `data_release_config.py` | RGB (0-1) of nucleus and membrane |
| `use_inverted_meshes`, `reinvert_meshes` | `false` | mesh face inversion before upload |
| `reupload_simularium_files`, `reupload_thumbnails`, `reupload_csv_files` | `false` | force re-upload |
| `max_workers` | `8` | parallel workers |
| `output_name` | `cellpack_simularium` | prefix of the output files |
| `dry_run` | `false` | same as `--dry_run` |

Step toggles are listed in the step table above; their defaults are `upload_meshes_to_s3`,
`update_simularium_files`, `upload_simularium_to_s3`, `generate_thumbnails`, `create_csv`,
`create_metadata_csv`, `upload_csv_to_s3` and `upload_metadata_csv` true, and
`rebuild_all_simularium`, `export_voxelized_images` and `update_csv_stats` false.

### Output

`results/data_release/<output_name>/` holds `<output_name>_paths.csv`, `<output_name>_metadata.csv`
(column descriptions) and `thumbnails/`. Columns of the paths CSV: File Name, Cell ID, Rule,
Packing ID, Structure ID, Structure Name, Count, Dataset, Condition, Experiment, Cell Volume
(µm³), Nucleus Volume (µm³), Cell Height (µm), Nucleus Height (µm), Cell Sphericity, File Type,
File Path, Thumbnail.

The workflow uses `cellpack_analysis/lib/{s3_utils,simularium_utils,mesh_tools,img_io}.py`.
