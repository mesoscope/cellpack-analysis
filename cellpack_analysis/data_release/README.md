# Data release

| Script | Use |
|---|---|
| `paper_inputs.py` | download and verify the input data for the paper (no credentials needed) |
| `run_data_release_workflow.py` | export packings as Simularium files and a BioFile Finder CSV, and upload them to S3 (needs AWS credentials) |

## Paper inputs

The inputs are published as 26 `.tar.gz` archives (5 GB, 31 GB extracted) at
`https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/release/v1.0.0/`.
`configs/paper_inputs.json` lists each archive with its checksum. The archives extract into `data/`.

```bash
uv run python -m cellpack_analysis.data_release.paper_inputs fetch     # all groups
uv run python -m cellpack_analysis.data_release.paper_inputs fetch --groups tables,meshes_RAB5A
uv run python -m cellpack_analysis.data_release.paper_inputs verify
```

`fetch` checks each archive's sha256 and skips groups that are already extracted. `plan` and `build`
make the archives for a new release.

| Group | Contents |
|---|---|
| `tables` | `all_cell_ids`, `variance_dataset` and `structure_stats` parquet files |
| `observed_{SLC25A17,RAB5A}` | observed puncta positions (305 peroxisome, 442 endosome cells) |
| `meshes_{SLC25A17,RAB5A,mean}` | nucleus and membrane meshes; `mean` is the mean cell |
| `mesh_information_{SLC25A17,RAB5A,mean}` | mesh information |
| `grid_distances_{SLC25A17,RAB5A,mean}` | available-space grid points and their distances |
| `packings_replicate_99_v4_{peroxisome,endosome}_{rule}` | 99 packings per cell for each modeling rule and the mixed rule (`interpolated`) |
| `packings_tune_gradients_v2` | decay-length ladder packings (Fig S10) |
| `packings_biological_variation` | sources-of-variation packings (Fig 2, Fig S3, Table S1) |
| `packings_fig4c_illustration` | the packing shown in Fig 4c |
| `packing_inputs` | recipes and packing configs used for the packings |

## BioFile Finder / Simularium export

`run_data_release_workflow.py` exports packings as Simularium files, writes the CSVs that describe
them for BioFile Finder, and uploads both to S3. It needs the per-seed Simularium files from a
repack; they are not part of the paper inputs.

```bash
uv run python -m cellpack_analysis.data_release.run_data_release_workflow \
    -c cellpack_analysis/data_release/configs/replicate_99_v4/data_release_config.json --dry_run
```

With no step flag, every step enabled in the config runs. `--help` lists the step flags, and
`--dry_run` logs the actions without uploading or changing files. Config keys and defaults are in
`data_release_config.py`. The CSVs go to `results/data_release/<output_name>/`.
