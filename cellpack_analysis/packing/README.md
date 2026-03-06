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
4. Workers write per-batch result JSONs to `<output_path>/slurm_staging/results/`.
5. The orchestrator polls `squeue` until all jobs finish (or use `--no-wait`),
   then aggregates all result JSONs into `slurm_staging/aggregated_results.json`.

### Bash launcher flags (`submit_packing_slurm.sh`)

```
  -c, --config         Path to the workflow config JSON (required)
  -b, --batch-size     Recipes per worker SLURM job (default: 8)
  -v, --venv           Path to virtualenv activate script (auto-detected)
  -p, --partition      SLURM partition (default: aics_gpu)
  -t, --time           Wall-clock limit for worker jobs (default: 1:00:00)
  -m, --mem            Memory per worker job (default: 16G)
  --cpus               CPUs per worker task (default: 4)
  --job-name           Job name prefix (default: cellpack)
  --orch-time          Wall-clock limit for orchestrator job (default: 1:00:00)
  --orch-mem           Memory for orchestrator job (default: 16G)
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
  --venv-path                  Path to virtualenv activate script
  -v, --verbose                Debug logging on the console

SLURM resource options:
  --slurm-partition            Partition name (default: aics_gpu)
  --slurm-time                 Wall-clock limit (default: 1:00:00)
  --slurm-mem                  Memory per job (default: 16G)
  --slurm-cpus-per-task        CPUs per job (default: 4)
  --slurm-job-name             Job name prefix (default: cellpack)
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

# ... after jobs finish ...
python -m cellpack_analysis.packing.run_packing_workflow_slurm \
    --aggregate path/to/slurm_staging/job_tracking.json
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
`<output_path>/slurm_staging/aggregated_results.json` with per-rule
success/failure/skip counts and the list of failed recipe paths.

```bash
# Pretty-print the summary
python -m json.tool <output_path>/slurm_staging/aggregated_results.json
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
    ├── manifests/             # per-batch recipe lists
    ├── scripts/               # generated sbatch scripts
    ├── results/               # per-batch worker result JSONs
    ├── job_tracking.json      # (only with --no-wait)
    └── aggregated_results.json
```
