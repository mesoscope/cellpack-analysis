# Packing Workflow

This package generates simulated packed structures using
[cellPACK](https://github.com/mesoscope/cellpack).  It provides three
entry-points: a **local** workflow (single machine, multiprocess), a
**SLURM bash launcher** (recommended — zero compute on the login node), and
a **SLURM Python** workflow (runs Python on whichever node you call it from).

---

## Quick start

### 1. Local execution

Run all recipes sequentially (or with local multiprocessing controlled by
`num_processes` in the workflow config):

```bash
python -m cellpack_analysis.packing.run_packing_workflow \
    -c path/to/workflow_config.json
```

### 2. SLURM execution (recommended)

Use the **bash launcher** — the only thing it runs on the login node is a
single `sbatch` call.  All Python work (generating configs, recipes,
submitting worker batches) happens on a compute node:

```bash
bash cellpack_analysis/packing/submit_packing_slurm.sh \
    -c path/to/workflow_config.json \
    -b 8
```

The launcher auto-detects your active virtualenv (via `$VIRTUAL_ENV`).
You can also pass `--venv /path/to/.venv/bin/activate` explicitly.

### 3. SLURM execution (direct Python)

If you prefer to run the orchestrator directly (Python runs on the
login node for config generation, then submits SLURM worker jobs):

```bash
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    -c path/to/workflow_config.json \
    --batch-size 8
```

Packing outputs land in the **same directory** regardless of which
entry-point you use — only the execution is parallelised.

---

## Module overview

| File | Description |
|---|---|
| `submit_packing_slurm.sh` | **Recommended** bash launcher — zero Python on login node |
| `run_packing_workflow.py` | Local entry-point — runs packings on a single machine |
| `run_packing_workflow_slurm.py` | SLURM Python entry-point — orchestrator, worker, and aggregator modes |
| `pack_recipes.py` | Core packing logic (`run_single_packing`, `pack_recipes`) |
| `generate_cellpack_input_files.py` | Generates per-cell recipe and config JSON files |
| `workflow_config.py` | `WorkflowConfig` class — reads the JSON config |
| `rule_repository.py` | Gradient and rule definitions |
| `configs/` | Workflow configuration JSON files |

---

## Workflow configuration

Both entry-points accept a **workflow config JSON** (`-c`).  Key fields:

| Field | Default | Description |
|---|---|---|
| `packing_id` | `"peroxisome"` | Identifier for this packing run |
| `structure_id` | `"SLC25A17"` | Structure to pack |
| `condition` | `"rules_shape"` | Simulation condition |
| `generate_recipes` | `true` | Generate recipe JSONs before packing |
| `generate_configs` | `true` | Generate cellPACK config JSONs before packing |
| `skip_completed` | `false` | Skip recipes whose outputs already exist |
| `dry_run` | `false` | Log what would be done without actually packing |
| `num_processes` | `1` | Local parallelism (used by `run_packing_workflow.py`) |
| `number_of_replicates` | `1` | Seeds / replicates per recipe |
| `result_type` | `"simularium"` | Output format (`"simularium"` or `"image"`) |
| `use_mean_cell` | `false` | Use mean cell instead of individual cells |
| `use_cells_in_8d_sphere` | `false` | Restrict to cells within 8D PCA sphere |
| `num_cells` | *all* | Limit total number of cells to process |
| `use_additional_struct` | `false` | Include additional structure (e.g. ER, Golgi) |
| `gradient_structure_name` | *none* | Structure to apply gradient packing to |
| `get_counts_from_data` | `false` | Derive molecule counts from experimental data |
| `get_size_from_data` | `false` | Derive molecule sizes from experimental data |
| `get_bounding_box_from_mesh` | `false` | Derive bounding box from mesh file |
| `datadir` | `data/` | Root data directory (absolute or project-relative) |
| `output_path` | *derived* | Override packing output directory |
| `recipe_template_path` | *derived* | Path to recipe template JSON |
| `config_template_path` | *derived* | Path to cellPACK config template JSON |
| `generated_recipe_path` | *derived* | Output directory for generated recipe JSONs |
| `generated_config_path` | *derived* | Output directory for generated config JSONs |
| `grid_path` | *derived* | Path or URL to grid files |
| `mesh_path` | *derived* | Path or URL to mesh files |
| `packings_to_run.rules` | `[]` | List of rule names to pack |
| `packings_to_run.cell_ids` | *all* | Subset of cell IDs (omit to use all) |
| `packings_to_run.number_of_packings` | *all* | Cap on number of packings |

---

## SLURM workflow in detail

### How it works

**With the bash launcher (recommended):**

```
                   login node               compute nodes
                  ┌──────────┐
                  │  bash     │
  user ────▸      │  submit_  │ ── sbatch ──▸ ┌──────────────────┐
                  │  packing_ │               │  Orchestrator    │
                  │  slurm.sh │               │  (--orchestrate) │
                  └──────────┘               │  generates       │
                   (only runs                │  configs/recipes │
                    sbatch)                  │  submits workers │
                                             └────────┬─────────┘
                                                      │ sbatch ×N
                                             ┌────────▼─────────┐
                                             │  Worker batch    │
                                             │  (--worker)      │
                                             │  run_single_     │
                                             │  packing() ×B    │
                                             └────────┬─────────┘
                                                      │ result JSONs
                                             ┌────────▼─────────┐
                                             │  Aggregate       │
                                             │  (--aggregate)   │
                                             │  collect results │
                                             └──────────────────┘
```

**With direct Python invocation:**

```
┌──────────────┐      sbatch ×N       ┌─────────────────┐
│ Orchestrator │ ──────────────────▸   │  SLURM Worker   │
│  (login node)│                       │  (compute node)  │
│              │  ◂── result JSONs ──  │  runs batch of   │
│  aggregates  │                       │  recipes via     │
│  results     │                       │  run_single_     │
│              │                       │  packing()       │
└──────────────┘                       └─────────────────┘
```

**Steps:**

1. The **orchestrator** reads the workflow config, optionally generates
   recipes/configs, then partitions recipes into batches of `--batch-size`.
2. For each batch it writes a **manifest JSON** and an **sbatch script**,
   then submits with `sbatch`.
3. Each SLURM job runs the same module in **worker mode** (`--worker`),
   reads its manifest, and calls `run_single_packing()` for each recipe.
4. Workers write per-batch result JSONs to `<output_path>/slurm_staging/<packing_id>/results/`.
5. The orchestrator polls `squeue` until all jobs finish (or use `--no-wait`),
   then aggregates all result JSONs into `slurm_staging/<packing_id>/aggregated_results.json`.

### Bash launcher flags (`submit_packing_slurm.sh`)

```
  -c, --config         Path to the workflow config JSON (required)
  -b, --batch-size     Recipes per worker SLURM job (default: 8)
  -v, --venv           Path to virtualenv activate script (auto-detected)
  -p, --partition      SLURM partition (default: cluster default)
  -t, --time           Wall-clock limit for worker jobs (default: 00:30:00)
  -m, --mem            Memory per worker job (default: 16G)
  --cpus               CPUs per worker task (default: 4)
  --job-name           Job name prefix (default: cellpack)
  --max-jobs N         Max concurrent worker jobs (uses SLURM job arrays)
  --orch-time          Wall-clock limit for orchestrator job (default: 1-00:00:00)
  --orch-mem           Memory for orchestrator job (default: 16G)
  --orch-cpus          CPUs for orchestrator job (default: same as --cpus)
  --orch-partition     Partition for orchestrator job (defaults to --partition)
  --dry-run            Write scripts but don't submit workers
```

### Python CLI flags (`run_packing_workflow_slurm.py`)

```
Orchestrator options:
  -c, --workflow-config-path   Path to the workflow config JSON
  -b, --batch-size             Recipes per SLURM job (default: 8)
  --orchestrate                Run orchestrator inside a SLURM job
  --dry-run                    Write sbatch scripts without submitting
  --no-wait                    Submit and exit; aggregate later with --aggregate
  --aggregate TRACKING_FILE    Re-aggregate results from a previous --no-wait run
  --venv-path                  Path to virtualenv activate script
  -v, --verbose                Debug logging on the console

SLURM resource options:
  --slurm-partition            Partition name (default: cluster default)
  --slurm-time                 Wall-clock limit (default: 00:30:00)
  --slurm-mem                  Memory per job (default: 16G)
  --slurm-cpus-per-task        CPUs per job (default: 4)
  --slurm-job-name             Job name prefix (default: cellpack)
  --max-jobs N                 Max concurrent workers (SLURM job array, default: 0 = unlimited)
```

### Examples

#### Bash launcher (recommended — no Python on login node)

```bash
# Basic — auto-detect venv, default batch size (8)
bash cellpack_analysis/packing/submit_packing_slurm.sh \
    -c data/configs/peroxisome.json

# Explicit venv and batch size
bash cellpack_analysis/packing/submit_packing_slurm.sh \
    -c data/configs/peroxisome.json \
    -b 16 \
    -v /path/to/.venv/bin/activate

# Custom SLURM resources
bash cellpack_analysis/packing/submit_packing_slurm.sh \
    -c data/configs/peroxisome.json \
    -p my_partition -t 24:00:00 -m 32G

# Dry run — generates scripts but doesn't submit workers
bash cellpack_analysis/packing/submit_packing_slurm.sh \
    -c data/configs/peroxisome.json --dry-run

# Limit to 20 concurrent worker jobs (uses SLURM job arrays)
bash cellpack_analysis/packing/submit_packing_slurm.sh \
    -c data/configs/peroxisome.json --max-jobs 20
```

#### Direct Python (orchestrator runs on login node)

```bash
# Submit with default settings (batches of 8)
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    -c data/configs/peroxisome.json

# Dry run — inspect generated scripts without submitting
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    -c data/configs/peroxisome.json --dry-run

# 16 recipes per job, custom partition and time limit
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    -c data/configs/peroxisome.json \
    --batch-size 16 \
    --slurm-partition my_partition \
    --slurm-time 24:00:00

# Submit and exit immediately; aggregate later
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    -c data/configs/peroxisome.json --no-wait

# Limit concurrency: max 20 workers at a time (job array)
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    -c data/configs/peroxisome.json --max-jobs 20

# ... after jobs finish ...
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    --aggregate path/to/slurm_staging/job_tracking.json
```

---

## Limiting resource usage

By default, the orchestrator submits all batches at once, which can
saturate the cluster. Use `--max-jobs N` to cap how many worker jobs run
concurrently.

Under the hood this uses a **SLURM job array** with a `%N` throttle
(e.g. `--array=0-152%20`). SLURM manages the queue natively — as soon as
one task finishes, the next starts — so there is no wasted idle time
between waves.

### Quick examples

```bash
# Bash launcher: at most 20 workers at a time
bash cellpack_analysis/packing/submit_packing_slurm.sh \
    -c data/configs/peroxisome.json --max-jobs 20

# Python CLI: same thing
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    -c data/configs/peroxisome.json --max-jobs 20
```

### Choosing a good value

| Scenario | Suggested `--max-jobs` |
|---|---|
| Large run, shared cluster | 10–30 (leaves nodes for others) |
| Off-hours / dedicated partition | 50–100 or omit (no limit) |
| Quick test (few recipes) | Omit (no limit needed) |

A good rule of thumb: check `sinfo -p <partition>` to see how many nodes
are in the partition and pick a `--max-jobs` that uses roughly half of them.

### How it works

Without `--max-jobs`, the orchestrator submits each batch as an independent
`sbatch` call — all jobs enter the SLURM queue immediately and compete for
resources.

With `--max-jobs N`, a single SLURM job array is submitted instead:

```
sbatch --array=0-152%20 job_array.sh   # 153 tasks, max 20 at a time
```

Each array task reads its batch manifest via `$SLURM_ARRAY_TASK_ID`.
Benefits:

- **Single job ID** — `scancel <array_job_id>` cancels everything
- **Native throttling** — SLURM enforces the concurrency cap
- **No idle gaps** — the next task starts as soon as a slot opens

### Other SLURM-level controls

Your SLURM administrator may also enforce limits via **QOS** policies
(e.g. `MaxSubmitJobsPerUser`). Check with:

```bash
sacctmgr show qos format=Name,MaxSubmitJobsPerUser,MaxJobsPerUser
```

---

## Monitoring SLURM jobs

Once jobs are submitted, use these standard SLURM commands to track
progress:

### Cluster and partition info

```bash
# Overview of all partitions and their node states
sinfo

# Detailed per-node info for a specific partition
sinfo -p <partition> -N -l
```

### Your jobs

```bash
# All your jobs (running + pending)
squeue -u $USER

# Only running jobs
squeue -u $USER -t RUNNING

# Only pending (queued) jobs
squeue -u $USER -t PENDING

# Filter by the cellpack job-name prefix
squeue -u $USER --name=cellpack

# Compact one-liner with estimated start times
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.10l %.6D %.4C %S"
```

### Detailed job inspection

```bash
# Full details for one job (state, node, time limit, memory, command)
scontrol show job <JOBID>
```

### Cancelling jobs

```bash
# Cancel a single job
scancel <JOBID>

# Cancel all your cellpack jobs at once
scancel -u $USER --name=cellpack

# Cancel all your pending jobs (keep running ones)
scancel -u $USER -t PENDING
```

### Reviewing completed jobs

```bash
# Summary for a specific job
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode

# All your jobs that started today
sacct -u $USER --starttime=today \
    --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode

# Only failed jobs today
sacct -u $USER --starttime=today --state=FAILED \
    --format=JobID,JobName,State,ExitCode,Elapsed
```

### Tailing live log output

SLURM stdout/stderr logs are written to `<output_path>/logs/slurm/`.  The
exact paths are printed when each batch is submitted.

```bash
# Follow a running job's output
tail -f <output_path>/logs/slurm/cellpack_batch0000_<JOBID>.out

# Check for errors
grep -i "error\|failed\|exception" <output_path>/logs/slurm/*.err
```

### Aggregated results

After all jobs complete, the orchestrator writes
`<output_path>/slurm_staging/<packing_id>/aggregated_results.json` with per-rule
success/failure/skip counts and the list of failed recipe paths.

```bash
# Pretty-print the summary
python -m json.tool <output_path>/slurm_staging/<packing_id>/aggregated_results.json
```

---

## Output directory structure

Both local and SLURM workflows write packing outputs to the same location
defined in the workflow config (typically
`data/packing_outputs/<subfolder>/<condition>/`).  The SLURM workflow adds
a `slurm_staging/` directory for its own bookkeeping:

```
<output_path>/
├── <packing_id>/              # cellPACK results (same as local)
│   └── spheresSST/
│       ├── results_*.simularium
│       └── figures/
│           └── voxelized_image_*.ome.tiff
├── logs/
│   ├── <packing_id>_<condition>.log          # local workflow log
│   ├── <packing_id>_<condition>_slurm.log    # orchestrator log
│   └── slurm/                                # per-job SLURM logs
│       ├── cellpack_orch_<JOBID>.out         # orchestrator job
│       ├── cellpack_batch0000_<JOBID>.out
│       └── cellpack_batch0000_<JOBID>.err
└── slurm_staging/
    └── <packing_id>/            # per-packing-id staging area
        ├── manifests/             # per-batch recipe lists
        ├── scripts/               # generated sbatch scripts
        ├── results/               # per-batch worker result JSONs
        ├── job_tracking.json      # (only with --no-wait)
        └── aggregated_results.json
```
