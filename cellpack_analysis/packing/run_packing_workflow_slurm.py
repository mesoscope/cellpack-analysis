"""
SLURM-based workflow to generate simulated packed structures using cellPACK.

This module provides four modes of operation:

1. **Orchestrator mode** (default): Reads the workflow configuration, generates
   recipes/configs if needed, partitions recipes into batches, and submits each
   batch as a SLURM job via ``sbatch``. It then monitors the jobs and aggregates
   results into a single summary file.  This mode runs Python on **whichever
   node you invoke it from** (login or compute).

2. **Orchestrate mode** (``--orchestrate``): Identical to orchestrator mode,
   but intended to be called **inside** a SLURM job — typically by
   ``submit_packing_slurm.sh``.  This keeps all heavy Python work off the
   login node.

3. **Worker mode** (``--worker``): Executed inside each SLURM job. Receives a
   batch manifest (JSON) and runs the packings sequentially or in parallel within
   that job.

4. **Aggregate mode** (``--aggregate``): Re-collects results from a previous
   ``--no-wait`` submission.

Recommended usage (zero compute on the login node)
---------------------------------------------------
Use the bash launcher which only runs ``sbatch`` on the login node::

    bash cellpack_analysis/packing/submit_packing_slurm.sh \\
        -c path/to/workflow_config.json \\
        -b 8

This submits an orchestrator SLURM job that generates configs/recipes and
fans out worker batch jobs — all on compute nodes.

Direct usage (runs Python on the login node)
---------------------------------------------
::

    python -m cellpack_analysis.packing.run_packing_workflow_slurm \\
        -c path/to/workflow_config.json \\
        --batch-size 8

The ``--batch-size`` flag controls how many recipes are packed per SLURM job
(default: 8, valid range: 1-64).

SLURM resource defaults can be overridden with ``--slurm-*`` flags (see
``--help``).

Monitoring running jobs
-----------------------
Once jobs are submitted you can monitor them with standard SLURM commands:

**Check cluster/partition availability**::

    sinfo                              # overview of all partitions and node states
    sinfo -p <partition> -N -l         # detailed per-node info for a specific partition

**List your running and pending jobs**::

    squeue -u $USER                    # all your jobs
    squeue -u $USER -t RUNNING         # only running jobs
    squeue -u $USER -t PENDING         # only pending (queued) jobs
    squeue -u $USER --name=cellpack    # filter by job name prefix

**Detailed info on a specific job**::

    scontrol show job <JOBID>          # full job details (state, node, time, memory)

**Cancel jobs**::

    scancel <JOBID>                    # cancel a single job
    scancel -u $USER --name=cellpack   # cancel all cellpack jobs

**Review completed jobs (accounting)**::

    sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
    sacct -u $USER --starttime=today --format=JobID,JobName,State,Elapsed

**Tail live SLURM log output** (paths are printed at submission time)::

    tail -f <output_path>/logs/slurm/cellpack_batch0000_<JOBID>.out
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from cellpack_analysis.lib import default_values
from cellpack_analysis.lib.file_io import read_json, setup_workflow_logging
from cellpack_analysis.lib.io import format_time
from cellpack_analysis.packing.generate_cellpack_input_files import (
    generate_configs,
    generate_recipes,
)
from cellpack_analysis.packing.pack_recipes import (
    check_recipe_completed,
    get_input_file_dictionary,
    run_single_packing,
)
from cellpack_analysis.packing.workflow_config import WorkflowConfig

np.random.seed(42)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLURM_DEFAULTS: dict[str, str] = {
    "partition": "",
    "time": "1:00:00",
    "mem": "16G",
    "cpus_per_task": "4",
    "job_name": "cellpack",
}
"""Sensible SLURM defaults - override via CLI flags.

Partition defaults to empty string, which omits the ``--partition``
directive so SLURM uses the cluster's default partition."""

POLL_INTERVAL_SECONDS = 30
"""How often the orchestrator checks ``squeue`` for running jobs."""

MAX_ARRAY_SIZE = 10000
"""SLURM default MaxArraySize — check ``scontrol show config | grep MaxArraySize``."""


# ===================================================================
# Helper utilities
# ===================================================================


def _chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split *lst* into sublists of at most *chunk_size* elements."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _write_batch_manifest(
    manifest_path: Path,
    rule: str,
    config_path: str,
    recipe_paths: list[str | Path],
    workflow_config_path: str | Path,
) -> None:
    """Write a JSON manifest consumed by the worker."""
    data = {
        "rule": rule,
        "config_path": str(config_path),
        "recipe_paths": [str(p) for p in recipe_paths],
        "workflow_config_path": str(workflow_config_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(data, fh, indent=2)


def _write_worker_result(result_path: Path, results: dict[str, Any]) -> None:
    """Write the worker result summary to *result_path*."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as fh:
        json.dump(results, fh, indent=2)


def _read_worker_result(result_path: Path) -> dict[str, Any]:
    """Read a worker result JSON, returning an empty dict on failure."""
    try:
        with open(result_path) as fh:
            return json.load(fh)
    except Exception:
        return {}


# ===================================================================
# SLURM sbatch helpers
# ===================================================================


def _build_sbatch_script(
    manifest_path: Path,
    result_path: Path,
    log_dir: Path,
    batch_index: int,
    slurm_opts: dict[str, str],
    venv_path: str | None = None,
) -> str:
    """
    Build a bash script string suitable for ``sbatch --wrap`` or piping to
    ``sbatch``.

    Parameters
    ----------
    manifest_path
        Path to the JSON manifest for this batch.
    result_path
        Path where the worker should write its result summary.
    log_dir
        Directory for SLURM stdout/stderr logs.
    batch_index
        Numeric index used for log file naming.
    slurm_opts
        Dictionary of SLURM options (partition, time, mem, …).
    venv_path
        Optional path to a virtual-environment activate script.  If ``None``,
        the worker inherits the submitter's environment.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"{slurm_opts.get('job_name', 'cellpack')}_batch{batch_index:04d}"

    activate_cmd = ""
    if venv_path:
        activate_cmd = f"source {venv_path}"

    # Determine the Python that is running this orchestrator
    python_exec = sys.executable

    worker_cmd = (
        f"{python_exec} -m cellpack_analysis.packing.run_packing_workflow_slurm "
        f"--worker "
        f"--batch-manifest {manifest_path} "
        f"--result-path {result_path}"
    )

    partition = slurm_opts.get("partition", "")
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    script = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
{partition_line}
#SBATCH --time={slurm_opts.get("time", SLURM_DEFAULTS["time"])}
#SBATCH --mem={slurm_opts.get("mem", SLURM_DEFAULTS["mem"])}
#SBATCH --cpus-per-task={slurm_opts.get("cpus_per_task", SLURM_DEFAULTS["cpus_per_task"])}
#SBATCH --output={log_dir / f"{job_name}_%j.out"}
#SBATCH --error={log_dir / f"{job_name}_%j.err"}

set -euo pipefail

{activate_cmd}

echo "=== SLURM Job $SLURM_JOB_ID on $(hostname) ==="
echo "Manifest: {manifest_path}"
echo "Start: $(date)"

{worker_cmd}

echo "End: $(date)"
"""
    return script


def _build_array_sbatch_script(
    manifest_dir: Path,
    result_dir: Path,
    log_dir: Path,
    num_batches: int,
    max_concurrent: int,
    slurm_opts: dict[str, str],
    venv_path: str | None = None,
) -> str:
    """
    Build an ``sbatch`` script for a SLURM **job array**.

    Each array task uses ``$SLURM_ARRAY_TASK_ID`` to locate its manifest
    and result path.  The ``%max_concurrent`` suffix on the ``--array``
    directive limits how many tasks run simultaneously, preventing the
    workflow from monopolising all cluster nodes.

    Parameters
    ----------
    manifest_dir
        Directory containing ``batch_NNNN.json`` manifests.
    result_dir
        Directory where workers write ``batch_NNNN_result.json``.
    log_dir
        Directory for SLURM stdout/stderr logs.
    num_batches
        Total number of array tasks (0 … num_batches-1).
    max_concurrent
        Maximum number of array tasks running at the same time.
        Use ``0`` for no limit.
    slurm_opts
        Dictionary of SLURM options (partition, time, mem, …).
    venv_path
        Optional path to a virtual-environment activate script.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    job_name = slurm_opts.get("job_name", "cellpack")

    array_spec = f"0-{num_batches - 1}"
    if max_concurrent > 0:
        array_spec += f"%{max_concurrent}"

    activate_cmd = f"source {venv_path}" if venv_path else ""
    python_exec = sys.executable

    partition = slurm_opts.get("partition", "")
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    script = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
{partition_line}
#SBATCH --time={slurm_opts.get("time", SLURM_DEFAULTS["time"])}
#SBATCH --mem={slurm_opts.get("mem", SLURM_DEFAULTS["mem"])}
#SBATCH --cpus-per-task={slurm_opts.get("cpus_per_task", SLURM_DEFAULTS["cpus_per_task"])}
#SBATCH --array={array_spec}
#SBATCH --output={log_dir / f"{job_name}_%A_%a.out"}
#SBATCH --error={log_dir / f"{job_name}_%A_%a.err"}

set -euo pipefail

{activate_cmd}

BATCH_ID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
MANIFEST="{manifest_dir}/batch_${{BATCH_ID}}.json"
RESULT="{result_dir}/batch_${{BATCH_ID}}_result.json"

echo "=== SLURM Array Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "Manifest: $MANIFEST"
echo "Start: $(date)"

{python_exec} -m cellpack_analysis.packing.run_packing_workflow_slurm \\
    --worker \\
    --batch-manifest "$MANIFEST" \\
    --result-path "$RESULT"

echo "End: $(date)"
"""
    return script


def _submit_sbatch(script_content: str, script_path: Path) -> str | None:
    """
    Write *script_content* to *script_path* and submit via ``sbatch``.

    Returns the SLURM job ID string on success, ``None`` on failure.
    """
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as fh:
        fh.write(script_content)
    os.chmod(script_path, 0o755)

    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # sbatch output: "Submitted batch job 12345\n"
        job_id = result.stdout.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as exc:
        logger.error(f"sbatch submission failed: {exc.stderr}")
        return None


def _get_running_job_ids(job_ids: list[str]) -> list[str]:
    """Return the subset of *job_ids* that are still in the SLURM queue."""
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "-o", "%i", "--jobs", ",".join(job_ids)],
            capture_output=True,
            text=True,
        )
        active = {line.strip() for line in result.stdout.strip().splitlines() if line.strip()}
        return [jid for jid in job_ids if jid in active]
    except Exception:
        # If squeue fails (e.g. all jobs already finished), treat as empty
        return []


def _wait_for_jobs(
    job_ids: list[str],
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> None:
    """Block until none of *job_ids* appear in ``squeue``."""
    remaining = list(job_ids)
    while remaining:
        time.sleep(poll_interval)
        remaining = _get_running_job_ids(remaining)
        if remaining:
            logger.info(
                f"Waiting for {len(remaining)}/{len(job_ids)} SLURM jobs "
                f"({', '.join(remaining[:5])}{'…' if len(remaining) > 5 else ''})"
            )
    logger.info("All SLURM jobs have finished.")


# ===================================================================
# Worker entry-point
# ===================================================================


def _run_worker(manifest_path: Path, result_path: Path) -> None:
    """
    Execute packings described in the batch manifest.

    This function is called inside each SLURM job. It reads the manifest,
    runs each recipe packing, and writes a summary JSON.
    """
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    rule = manifest["rule"]
    config_path = manifest["config_path"]
    recipe_paths = manifest["recipe_paths"]
    workflow_config_path = manifest.get("workflow_config_path")

    # Load the workflow config so we can check for completed recipes
    workflow_config: WorkflowConfig | None = None
    if workflow_config_path:
        workflow_config = WorkflowConfig(workflow_config_path=Path(workflow_config_path))

    config_data = read_json(config_path)

    logger.info(
        f"Worker starting: rule={rule}, {len(recipe_paths)} recipes, " f"config={config_path}"
    )

    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for recipe_path in recipe_paths:
        # Skip completed recipes if applicable
        if (
            workflow_config is not None
            and workflow_config.skip_completed
            and check_recipe_completed(recipe_path, config_data, workflow_config)
        ):
            skipped.append(str(recipe_path))
            logger.info(f"Skipping completed recipe: {recipe_path}")
            continue

        ok = run_single_packing(recipe_path, config_path)
        if ok:
            succeeded.append(str(recipe_path))
            logger.info(f"Packing succeeded: {recipe_path}")
        else:
            failed.append(str(recipe_path))
            logger.error(f"Packing failed: {recipe_path}")

    summary = {
        "rule": rule,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "total": len(recipe_paths),
    }

    _write_worker_result(result_path, summary)

    logger.info(
        f"Worker finished: {len(succeeded)} succeeded, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )

    # Exit with non-zero if any packing failed
    if failed:
        sys.exit(1)


# ===================================================================
# Orchestrator entry-point
# ===================================================================


def _run_orchestrator(
    workflow_config_path: Path,
    batch_size: int = 8,
    slurm_opts: dict[str, str] | None = None,
    venv_path: str | None = None,
    no_wait: bool = False,
    dry_run: bool = False,
    max_jobs: int = 0,
) -> int:
    """
    Prepare recipes and submit batched SLURM jobs.

    Parameters
    ----------
    workflow_config_path
        Path to the workflow configuration JSON.
    batch_size
        Number of recipes per SLURM job (default 8).
    slurm_opts
        SLURM resource overrides.
    venv_path
        Path to ``activate`` script for the virtual environment.
    no_wait
        If ``True``, return immediately after submitting jobs without
        waiting for completion.
    dry_run
        If ``True``, write sbatch scripts but do not actually submit them.
    max_jobs
        Maximum number of SLURM jobs running concurrently.  When > 0 the
        orchestrator submits all batches as a single **SLURM job array**
        with a ``%max_jobs`` throttle (e.g. ``--array=0-152%20``).
        Set to 0 (the default) for no concurrency limit.

    Returns
    -------
    :
        Total number of failed packings (0 on success, or if ``no_wait``).
    """
    if slurm_opts is None:
        slurm_opts = dict(SLURM_DEFAULTS)

    workflow_config = WorkflowConfig(workflow_config_path=workflow_config_path)

    # Set up logging
    slurm_log_dir = workflow_config.output_path / "logs" / "slurm"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    workflow_log_file = setup_workflow_logging(
        workflow_config.output_path
        / "logs"
        / f"{workflow_config.packing_id}_{workflow_config.condition}_slurm.log"
    )
    logger.info(f"Logging all debug messages to {workflow_log_file}")
    logger.info(f"Batch size: {batch_size}")

    # Generate configs and recipes if workflow says so
    if workflow_config.generate_configs:
        logger.info("Updating cellPACK config files")
        generate_configs(workflow_config=workflow_config)

    if workflow_config.generate_recipes:
        logger.info("Generating cellPACK recipe files")
        generate_recipes(workflow_config=workflow_config)

    # Build recipe list per rule
    input_file_dict = get_input_file_dictionary(workflow_config)

    # Directories for batch manifests, sbatch scripts, and worker results
    staging_dir = workflow_config.output_path / "slurm_staging"
    manifest_dir = staging_dir / "manifests"
    script_dir = staging_dir / "scripts"
    result_dir = staging_dir / "results"
    for d in [manifest_dir, script_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Pre-filter completed recipes and build batches
    all_job_ids: list[str] = []
    all_result_paths: list[Path] = []
    batch_counter = 0
    total_submitted = 0
    total_skipped = 0

    for rule, input_files in input_file_dict.items():
        config_path = input_files["config_path"]
        config_data = read_json(config_path)

        # Filter out already-completed recipes at submission time
        recipes_to_pack: list[str | Path] = []
        for recipe_path in input_files["recipe_paths"]:
            if workflow_config.skip_completed and check_recipe_completed(
                recipe_path, config_data, workflow_config
            ):
                total_skipped += 1
                logger.debug(f"Skipping completed recipe at submission: {recipe_path}")
                continue
            recipes_to_pack.append(recipe_path)

        if not recipes_to_pack:
            logger.info(f"Rule {rule}: all recipes already completed, nothing to submit.")
            continue

        logger.info(
            f"Rule {rule}: {len(recipes_to_pack)} recipes to pack "
            f"({total_skipped} skipped so far)"
        )

        # Partition into batches
        batches = _chunk_list(recipes_to_pack, batch_size)
        logger.info(f"Rule {rule}: split into {len(batches)} batch(es)")

        for batch in batches:
            manifest_path = manifest_dir / f"batch_{batch_counter:04d}.json"
            result_path = result_dir / f"batch_{batch_counter:04d}_result.json"

            _write_batch_manifest(
                manifest_path=manifest_path,
                rule=rule,
                config_path=config_path,
                recipe_paths=batch,
                workflow_config_path=str(workflow_config_path),
            )

            all_result_paths.append(result_path)
            total_submitted += len(batch)
            batch_counter += 1

    logger.info(
        f"Manifest creation complete: {batch_counter} batches, "
        f"{total_submitted} recipes queued, {total_skipped} skipped"
    )

    if batch_counter == 0:
        logger.info("Nothing to submit.")
        return 0

    # ---- Submission strategy ----
    # When --max-jobs is set we use a single SLURM job array with a
    # %max_concurrent throttle.  Otherwise we submit individual jobs.
    use_job_array = max_jobs > 0

    if use_job_array:
        if batch_counter > MAX_ARRAY_SIZE:
            logger.warning(
                f"Total batches ({batch_counter}) exceeds SLURM MaxArraySize "
                f"({MAX_ARRAY_SIZE}).  Increase --batch-size to reduce the "
                f"number of array tasks, or check `scontrol show config | "
                f"grep MaxArraySize` if your cluster allows more."
            )

        array_script = _build_array_sbatch_script(
            manifest_dir=manifest_dir,
            result_dir=result_dir,
            log_dir=slurm_log_dir,
            num_batches=batch_counter,
            max_concurrent=max_jobs,
            slurm_opts=slurm_opts,
            venv_path=venv_path,
        )
        array_script_path = script_dir / "job_array.sh"

        if dry_run:
            array_script_path.parent.mkdir(parents=True, exist_ok=True)
            with open(array_script_path, "w") as fh:
                fh.write(array_script)
            logger.info(
                f"[DRY RUN] Would submit job array with {batch_counter} tasks "
                f"(max {max_jobs} concurrent) — script: {array_script_path}"
            )
        else:
            job_id = _submit_sbatch(array_script, array_script_path)
            if job_id:
                all_job_ids.append(job_id)
                logger.info(
                    f"Submitted job array {job_id} with {batch_counter} tasks "
                    f"(max {max_jobs} concurrent)"
                )
            else:
                logger.error("Failed to submit job array")
    else:
        # Individual job submission (original behaviour)
        for idx in range(batch_counter):
            manifest_path = manifest_dir / f"batch_{idx:04d}.json"
            result_path = result_dir / f"batch_{idx:04d}_result.json"
            script_path = script_dir / f"batch_{idx:04d}.sh"

            sbatch_script = _build_sbatch_script(
                manifest_path=manifest_path,
                result_path=result_path,
                log_dir=slurm_log_dir,
                batch_index=idx,
                slurm_opts=slurm_opts,
                venv_path=venv_path,
            )

            if dry_run:
                script_path.parent.mkdir(parents=True, exist_ok=True)
                with open(script_path, "w") as fh:
                    fh.write(sbatch_script)
                logger.info(f"[DRY RUN] Would submit batch {idx} — script: {script_path}")
            else:
                job_id = _submit_sbatch(sbatch_script, script_path)
                if job_id:
                    all_job_ids.append(job_id)
                    logger.info(f"Submitted batch {idx} as SLURM job {job_id}")
                else:
                    logger.error(f"Failed to submit batch {idx}")

    logger.info(
        f"Submission complete: {batch_counter} batches, "
        f"{total_submitted} recipes submitted, {total_skipped} skipped"
    )

    if dry_run:
        logger.info("[DRY RUN] No jobs were actually submitted.")
        return 0

    if no_wait:
        logger.info(
            "Not waiting for jobs to complete (--no-wait). "
            "Run the aggregation step manually later."
        )
        # Write a job-tracking file for later aggregation
        tracking_file = staging_dir / "job_tracking.json"
        with open(tracking_file, "w") as fh:
            json.dump(
                {
                    "job_ids": all_job_ids,
                    "result_paths": [str(p) for p in all_result_paths],
                    "workflow_config_path": str(workflow_config_path),
                    "is_job_array": use_job_array,
                },
                fh,
                indent=2,
            )
        logger.info(f"Job tracking info written to {tracking_file}")
        return 0

    # Wait for all SLURM jobs to finish
    if all_job_ids:
        logger.info(f"Waiting for {len(all_job_ids)} SLURM jobs to complete …")
        _wait_for_jobs(all_job_ids)

    # Aggregate results
    return _aggregate_results(all_result_paths, staging_dir)


def _aggregate_results(
    result_paths: list[Path],
    staging_dir: Path,
) -> int:
    """
    Read all worker result JSONs and produce a single aggregated summary.

    Parameters
    ----------
    result_paths
        Paths to individual worker result files.
    staging_dir
        Directory where the aggregated summary will be written.

    Returns
    -------
    :
        Total number of failed packings across all batches.
    """
    aggregated: dict[str, dict[str, list[str]]] = {}
    total_succeeded = total_failed = total_skipped = 0
    missing_results: list[str] = []

    for rp in result_paths:
        data = _read_worker_result(rp)
        if not data:
            missing_results.append(str(rp))
            logger.warning(f"Missing or unreadable result file: {rp}")
            continue

        rule = data.get("rule", "unknown")
        if rule not in aggregated:
            aggregated[rule] = {"succeeded": [], "failed": [], "skipped": []}
        aggregated[rule]["succeeded"].extend(data.get("succeeded", []))
        aggregated[rule]["failed"].extend(data.get("failed", []))
        aggregated[rule]["skipped"].extend(data.get("skipped", []))

        total_succeeded += len(data.get("succeeded", []))
        total_failed += len(data.get("failed", []))
        total_skipped += len(data.get("skipped", []))

    summary = {
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "missing_result_files": missing_results,
        "per_rule": {
            rule: {
                "succeeded": len(info["succeeded"]),
                "failed": len(info["failed"]),
                "skipped": len(info["skipped"]),
                "failed_recipes": info["failed"],
            }
            for rule, info in aggregated.items()
        },
    }

    summary_path = staging_dir / "aggregated_results.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(f"Aggregated results written to {summary_path}")
    logger.info(
        f"Summary — Succeeded: {total_succeeded}, Failed: {total_failed}, "
        f"Skipped: {total_skipped}, Missing results: {len(missing_results)}"
    )

    if total_failed > 0:
        logger.warning(f"{total_failed} packings failed. See aggregated_results.json for details.")

    return total_failed


def aggregate_from_tracking(tracking_file: Path) -> int:
    """
    Re-aggregate results using a previously written job tracking file.

    Useful when the orchestrator was run with ``--no-wait`` and you want to
    collect results later::

        python -m cellpack_analysis.packing.run_packing_workflow_slurm \\
            --aggregate path/to/slurm_staging/job_tracking.json
    """
    with open(tracking_file) as fh:
        tracking = json.load(fh)

    result_paths = [Path(p) for p in tracking["result_paths"]]
    staging_dir = tracking_file.parent

    return _aggregate_results(result_paths, staging_dir)


# ===================================================================
# CLI
# ===================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLURM-parallel packing workflow for cellPACK.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Recommended: zero compute on login node (bash launcher)
  bash cellpack_analysis/packing/submit_packing_slurm.sh -c config.json -b 8

# Direct: submit all recipes in batches of 8 (runs Python on login node)
  python -m cellpack_analysis.packing.run_packing_workflow_slurm -c config.json

# Dry run (writes scripts but does not submit):
  python -m cellpack_analysis.packing.run_packing_workflow_slurm -c config.json --dry-run

# Larger batches (16 recipes per job) on a specific partition:
  python -m cellpack_analysis.packing.run_packing_workflow_slurm -c config.json \\
      --batch-size 16 --slurm-partition my_partition --slurm-time 24:00:00

# Submit without waiting, then aggregate later:
  python -m cellpack_analysis.packing.run_packing_workflow_slurm -c config.json --no-wait
  # ... later ...
  python -m cellpack_analysis.packing.run_packing_workflow_slurm \\
      --aggregate path/to/slurm_staging/job_tracking.json

Monitoring jobs
---------------
  sinfo                               # partition & node availability
  squeue -u $USER                     # your running/pending jobs
  squeue -u $USER --name=cellpack     # filter by job name prefix
  scontrol show job <JOBID>           # detailed info for one job
  sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
  scancel <JOBID>                     # cancel a single job
  tail -f <output_path>/logs/slurm/cellpack_batch0000_<JOBID>.out
""",
    )

    # -- Mode flags (mutually exclusive) --
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--worker",
        action="store_true",
        help="Run in worker mode (used internally by SLURM jobs).",
    )
    mode_group.add_argument(
        "--orchestrate",
        action="store_true",
        help=(
            "Run the orchestrator inside a SLURM job (used by "
            "submit_packing_slurm.sh). Generates configs/recipes on the "
            "compute node, then submits worker batch jobs."
        ),
    )
    mode_group.add_argument(
        "--aggregate",
        type=str,
        default=None,
        metavar="TRACKING_FILE",
        help="Aggregate results from a previous --no-wait run.",
    )

    # -- Worker arguments --
    worker_group = parser.add_argument_group("Worker options")
    worker_group.add_argument(
        "--batch-manifest",
        type=str,
        help="Path to batch manifest JSON (worker mode only).",
    )
    worker_group.add_argument(
        "--result-path",
        type=str,
        help="Path to write worker result JSON (worker mode only).",
    )

    # -- Orchestrator arguments --
    orch_group = parser.add_argument_group("Orchestrator options")
    orch_group.add_argument(
        "--workflow-config-path",
        "-c",
        type=str,
        help="Path to the packing workflow configuration file.",
        default=str(default_values.WORKFLOW_CONFIG_PATH),
    )
    orch_group.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Number of recipes per SLURM job (default: 8).",
    )
    orch_group.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and exit without waiting for completion.",
    )
    orch_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Write sbatch scripts but do not submit them.",
    )
    orch_group.add_argument(
        "--venv-path",
        type=str,
        default=None,
        help=(
            "Path to the virtualenv activate script. If omitted, the worker "
            "inherits the submitter's environment."
        ),
    )
    orch_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging on the console.",
    )

    # -- SLURM resource overrides --
    slurm_group = parser.add_argument_group("SLURM resource options")
    slurm_group.add_argument(
        "--slurm-partition",
        type=str,
        default="",
        help="SLURM partition. Omit to use the cluster default.",
    )
    slurm_group.add_argument(
        "--slurm-time",
        type=str,
        default=SLURM_DEFAULTS["time"],
        help=f"SLURM time limit (default: {SLURM_DEFAULTS['time']}).",
    )
    slurm_group.add_argument(
        "--slurm-mem",
        type=str,
        default=SLURM_DEFAULTS["mem"],
        help=f"SLURM memory per job (default: {SLURM_DEFAULTS['mem']}).",
    )
    slurm_group.add_argument(
        "--slurm-cpus-per-task",
        type=str,
        default=SLURM_DEFAULTS["cpus_per_task"],
        help=f"CPUs per SLURM task (default: {SLURM_DEFAULTS['cpus_per_task']}).",
    )
    slurm_group.add_argument(
        "--slurm-job-name",
        type=str,
        default=SLURM_DEFAULTS["job_name"],
        help=f"SLURM job name prefix (default: {SLURM_DEFAULTS['job_name']}).",
    )
    slurm_group.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Maximum number of worker SLURM jobs running at the same time. "
            "When set, all batches are submitted as a single SLURM job array "
            "with a %%N concurrency throttle (e.g. --max-jobs 20 => "
            "--array=0-152%%20).  Set to 0 (default) for no limit."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.DEBUG)

    # ---------- Worker mode ----------
    if args.worker:
        if not args.batch_manifest or not args.result_path:
            parser.error("--worker requires --batch-manifest and --result-path")
        _run_worker(
            manifest_path=Path(args.batch_manifest),
            result_path=Path(args.result_path),
        )
        return

    # ---------- Aggregate mode ----------
    if args.aggregate:
        total_failed = aggregate_from_tracking(Path(args.aggregate))
        sys.exit(1 if total_failed else 0)

    # ---------- Orchestrate mode (inside SLURM) ----------
    # Identical to default orchestrator; the flag exists so the bash
    # launcher can be explicit about intent.

    # ---------- Orchestrator mode ----------
    start = time.time()

    slurm_opts = {
        "partition": args.slurm_partition,
        "time": args.slurm_time,
        "mem": args.slurm_mem,
        "cpus_per_task": args.slurm_cpus_per_task,
        "job_name": args.slurm_job_name,
    }

    total_failed = _run_orchestrator(
        workflow_config_path=Path(args.workflow_config_path),
        batch_size=args.batch_size,
        slurm_opts=slurm_opts,
        venv_path=args.venv_path,
        no_wait=args.no_wait,
        dry_run=args.dry_run,
        max_jobs=args.max_jobs,
    )

    logger.info(f"Total time: {format_time(time.time() - start)}")
    if total_failed:
        logger.info(f"Total failed packings: {total_failed}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
