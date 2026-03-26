r"""
SLURM-based workflow to calculate available intracellular space across many cells.

This module fans out ``calculate_grid_distances`` calls—one per cell—across a
SLURM cluster, mirroring the packing SLURM workflow.  It supports three modes:

1. **Orchestrate mode** (``--orchestrate``): Run inside a SLURM job submitted by
   ``submit_available_space_slurm.sh``.  Discovers valid meshes, partitions cells
   into batches, and submits a SLURM job array.  No Python runs on the login node.

2. **Worker mode** (``--worker``): Executed inside each array task.  Reads a batch
   manifest and calls ``calculate_grid_distances`` for each cell in parallel using
   ``ProcessPoolExecutor``.

3. **Aggregate mode** (``--aggregate``): Collect per-batch result JSONs from a
   previous submission into a single summary.

Recommended usage (zero compute on login node)
----------------------------------------------
::

    bash cellpack_analysis/preprocessing/submit_available_space_slurm.sh \
        -s <structure_id>

Direct usage (runs Python on the login node)
--------------------------------------------
::

    python -m cellpack_analysis.preprocessing.run_available_space_slurm \
        --orchestrate \
        --structure-id <structure_id>

Monitoring
----------
::

    squeue -u $USER --name=cellpack_avspace
    sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
"""

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.mesh_tools import calculate_grid_distances

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SLURM defaults (larger than packing — mesh + grid arrays need more memory)
# ---------------------------------------------------------------------------
SLURM_DEFAULTS: dict[str, str] = {
    "partition": "",
    "time": "02:00:00",
    "mem": "64G",
    "cpus_per_task": "4",
    "job_name": "cellpack_avspace",
}

POLL_INTERVAL_SECONDS = 120
MAX_ARRAY_SIZE = 10000


# ===================================================================
# Manifest / result helpers
# ===================================================================


def _write_batch_manifest(manifest_path: Path, cells: list[dict[str, Any]], **kwargs: Any) -> None:
    data: dict[str, Any] = {"cells": cells}
    data.update(kwargs)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(data, fh, indent=2)


def _write_worker_result(result_path: Path, results: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as fh:
        json.dump(results, fh, indent=2)


def _read_worker_result(result_path: Path) -> dict[str, Any]:
    try:
        with open(result_path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _chunk_list(lst: list, chunk_size: int) -> list[list]:
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ===================================================================
# SLURM submission helpers
# ===================================================================


def _build_array_sbatch_script(
    manifest_dir: Path,
    result_dir: Path,
    log_dir: Path,
    num_batches: int,
    max_concurrent: int,
    slurm_opts: dict[str, str],
    venv_path: str | None = None,
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    job_name = slurm_opts.get("job_name", SLURM_DEFAULTS["job_name"])

    array_spec = f"0-{num_batches - 1}"
    if max_concurrent > 0:
        array_spec += f"%{max_concurrent}"

    activate_cmd = f"source {venv_path}" if venv_path else ""
    python_exec = sys.executable
    partition = slurm_opts.get("partition", "")
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    return f"""#!/usr/bin/env bash
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

echo "=== Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "Manifest: $MANIFEST"
echo "Start: $(date)"

{python_exec} -m cellpack_analysis.preprocessing.run_available_space_slurm \\
    --worker \\
    --batch-manifest "$MANIFEST" \\
    --result-path "$RESULT"

echo "End: $(date)"
"""


def _submit_sbatch(script_content: str, script_path: Path) -> str | None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as fh:
        fh.write(script_content)
    os.chmod(script_path, 0o755)
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split()[-1]
    except subprocess.CalledProcessError as exc:
        logger.error(f"sbatch submission failed: {exc.stderr}")
        return None


def _get_running_job_ids(job_ids: list[str]) -> tuple[list[str], int]:
    """
    Query squeue for active tasks among *job_ids*.

    Returns a tuple of:
    - still-running base job IDs
    - total number of active array tasks (one squeue output line per task)
    """
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "-o", "%i", "--jobs", ",".join(job_ids)],
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        active_base_ids = {line.split("_")[0] for line in lines}
        return [jid for jid in job_ids if jid in active_base_ids], len(lines)
    except Exception:
        return [], 0


def _log_batch_status(result_paths: list[Path], total_batches: int) -> None:
    """Log how many batches have completed and their succeeded/failed counts."""
    done_ok = 0
    done_failed = 0
    total_cells_succeeded = 0
    total_cells_failed = 0
    for rp in result_paths:
        data = _read_worker_result(rp)
        if not data:
            continue
        if data.get("failed"):
            done_failed += 1
        else:
            done_ok += 1
        total_cells_succeeded += len(data.get("succeeded", []))
        total_cells_failed += len(data.get("failed", []))
    done_total = done_ok + done_failed
    pending = total_batches - done_total
    logger.info(
        f"Batch progress: {done_total}/{total_batches} finished "
        f"({done_ok} ok, {done_failed} with failures, {pending} pending) — "
        f"cells succeeded={total_cells_succeeded}, failed={total_cells_failed}"
    )


def _wait_for_jobs(
    job_ids: list[str],
    poll_interval: int = POLL_INTERVAL_SECONDS,
    result_paths: list[Path] | None = None,
) -> None:
    remaining = list(job_ids)
    total_batches = len(result_paths) if result_paths else 0
    while remaining:
        time.sleep(poll_interval)
        remaining, active_tasks = _get_running_job_ids(remaining)
        if remaining:
            logger.info(
                f"Waiting for {len(remaining)}/{len(job_ids)} SLURM jobs "
                f"({active_tasks} array tasks still active) "
                f"({', '.join(remaining[:5])}{'…' if len(remaining) > 5 else ''})"
            )
        if result_paths:
            _log_batch_status(result_paths, total_batches)
    logger.info("All SLURM jobs have finished.")
    if result_paths:
        _log_batch_status(result_paths, total_batches)


# ===================================================================
# Worker
# ===================================================================


def _run_single_cell(cell: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Call ``calculate_grid_distances`` for one cell; return cell_id on success."""
    cell_id: str = cell["cell_id"]
    calculate_grid_distances(
        nuc_mesh_path=cell["nuc_mesh_path"],
        mem_mesh_path=cell["mem_mesh_path"],
        cell_id=cell_id,
        spacing=manifest["spacing"],
        save_dir=Path(manifest["grid_dir"]),
        recalculate=manifest.get("recalculate", False),
        calc_mem_distances=cell.get("calc_mem_distances", True),
        calc_nuc_distances=cell.get("calc_nuc_distances", True),
        calc_scaled_nuc_distances=cell.get("calc_scaled_nuc_distances", True),
        calc_z_distances=cell.get("calc_z_distances", True),
        calc_scaled_z_distances=cell.get("calc_scaled_z_distances", True),
        chunk_size=manifest.get("chunk_size"),
        struct_mesh_path=cell.get("struct_mesh_path"),
    )
    return cell_id


def _run_worker(manifest_path: Path, result_path: Path) -> None:
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    cells: list[dict[str, Any]] = manifest["cells"]
    num_processes = int(manifest.get("num_processes", os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")

    logger.info(
        f"Worker starting: {len(cells)} cells, "
        f"num_processes={num_processes}, job={slurm_job_id}"
    )

    succeeded: list[str] = []
    failed: list[str] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_cell = {
            executor.submit(_run_single_cell, cell, manifest): cell["cell_id"] for cell in cells
        }
        for future in concurrent.futures.as_completed(future_to_cell):
            cell_id = future_to_cell[future]
            try:
                future.result()
                succeeded.append(cell_id)
                logger.info(f"Succeeded: {cell_id}")
            except Exception as exc:
                failed.append(cell_id)
                logger.error(f"Failed {cell_id}: {exc}")

    summary = {
        "slurm_job_id": slurm_job_id,
        "succeeded": succeeded,
        "failed": failed,
        "total": len(cells),
    }
    _write_worker_result(result_path, summary)
    logger.info(f"Worker finished: {len(succeeded)} succeeded, {len(failed)} failed")

    if failed:
        sys.exit(1)


# ===================================================================
# Orchestrator
# ===================================================================
_DISTANCE_FLAG_TO_PREFIX: dict[str, str] = {
    "calc_mem_distances": "mem_distances",
    "calc_nuc_distances": "nuc_distances",
    "calc_scaled_nuc_distances": "scaled_nuc_distances",
    "calc_z_distances": "z_distances",
    "calc_scaled_z_distances": "scaled_z_distances",
}


def _get_missing_distance_flags(cell_id: str, grid_dir: Path) -> dict[str, bool]:
    """Return calc_* flags set True only for distance files not yet on disk."""
    return {
        flag: not (grid_dir / f"{prefix}_{cell_id}.npy").exists()
        for flag, prefix in _DISTANCE_FLAG_TO_PREFIX.items()
    }


def _discover_mesh_data(
    structure_id: str,
    use_mean_shape: bool,
    use_struct_mesh: bool,
    mesh_dir: Path,
    grid_dir: Path,
    use_all_cells: bool = False,
    recalculate: bool = False,
) -> list[dict[str, Any]]:
    """Return a list of cell dicts with valid nuc+mem mesh pairs."""
    cell_ids = (
        ["mean"]
        if use_mean_shape
        else get_cell_id_list_for_structure(structure_id, dsphere=not use_all_cells)
    )
    mesh_data = []
    for cell_id in cell_ids:
        flags = (
            {flag: True for flag in _DISTANCE_FLAG_TO_PREFIX}
            if recalculate
            else _get_missing_distance_flags(cell_id, grid_dir)
        )
        if not any(flags.values()):
            logger.info(f"Skipping {cell_id} (all output files already exist)")
            continue
        nuc_mesh_path = mesh_dir / f"nuc_mesh_{cell_id}.obj"
        mem_mesh_path = mesh_dir / f"mem_mesh_{cell_id}.obj"
        if not (nuc_mesh_path.exists() and mem_mesh_path.exists()):
            logger.warning(f"Missing mesh for {cell_id}, skipping")
            continue
        struct_mesh_path = mesh_dir / f"struct_mesh_{cell_id}.obj"
        entry: dict[str, Any] = {
            "cell_id": str(cell_id),
            "nuc_mesh_path": str(nuc_mesh_path),
            "mem_mesh_path": str(mem_mesh_path),
            "struct_mesh_path": (
                str(struct_mesh_path) if (use_struct_mesh and struct_mesh_path.exists()) else None
            ),
            **flags,
        }
        mesh_data.append(entry)
    return mesh_data


def _run_orchestrator(
    structure_id: str,
    spacing: float = 2.0,
    use_struct_mesh: bool = False,
    use_mean_shape: bool = False,
    use_all_cells: bool = False,
    recalculate: bool = False,
    chunk_size: int | None = None,
    batch_size: int = 4,
    max_concurrent: int = 16,
    slurm_opts: dict[str, str] | None = None,
    venv_path: str | None = None,
    dry_run: bool = False,
    no_wait: bool = False,
) -> int:
    if slurm_opts is None:
        slurm_opts = dict(SLURM_DEFAULTS)

    base_datadir = get_project_root() / "data"
    mesh_dir = base_datadir / f"structure_data/{structure_id}/meshes"
    grid_dir = base_datadir / f"structure_data/{structure_id}/grid_distances"
    grid_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d")
    staging_dir = base_datadir / f"structure_data/{structure_id}/slurm_staging/{timestamp}"
    manifest_dir = staging_dir / "manifests"
    script_dir = staging_dir / "scripts"
    result_dir = staging_dir / "results"
    log_dir = base_datadir / f"structure_data/{structure_id}/slurm_logs/{timestamp}/workers"
    for d in [manifest_dir, script_dir, result_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mesh_data = _discover_mesh_data(
        structure_id,
        use_mean_shape,
        use_struct_mesh,
        mesh_dir,
        grid_dir,
        use_all_cells,
        recalculate,
    )
    if not mesh_data:
        if not recalculate:
            logger.info("All cells already have grid distance outputs. Nothing to do.")
            return 0
        logger.error(f"No valid meshes found for structure {structure_id}")
        return 1

    logger.info(f"Found {len(mesh_data)} cells for structure {structure_id}")

    num_processes = int(slurm_opts.get("cpus_per_task", SLURM_DEFAULTS["cpus_per_task"]))
    batches = _chunk_list(mesh_data, batch_size)
    logger.info(f"Partitioned into {len(batches)} batches of up to {batch_size} cells")

    if len(batches) > MAX_ARRAY_SIZE:
        logger.warning(
            f"Batch count ({len(batches)}) exceeds SLURM MaxArraySize ({MAX_ARRAY_SIZE}). "
            f"Increase --batch-size to reduce array tasks."
        )

    all_result_paths: list[Path] = []
    for idx, batch in enumerate(batches):
        manifest_path = manifest_dir / f"batch_{idx:04d}.json"
        result_path = result_dir / f"batch_{idx:04d}_result.json"
        _write_batch_manifest(
            manifest_path,
            cells=batch,
            structure_id=structure_id,
            spacing=spacing,
            recalculate=recalculate,
            chunk_size=chunk_size,
            grid_dir=str(grid_dir),
            num_processes=num_processes,
        )
        all_result_paths.append(result_path)

    array_script = _build_array_sbatch_script(
        manifest_dir=manifest_dir,
        result_dir=result_dir,
        log_dir=log_dir,
        num_batches=len(batches),
        max_concurrent=max_concurrent,
        slurm_opts=slurm_opts,
        venv_path=venv_path,
    )
    array_script_path = script_dir / "job_array.sh"

    if dry_run:
        with open(array_script_path, "w") as fh:
            fh.write(array_script)
        logger.info(
            f"[DRY RUN] Would submit {len(batches)} array tasks (max {max_concurrent} concurrent)"
            f" — script: {array_script_path}"
        )
        return 0

    job_id = _submit_sbatch(array_script, array_script_path)
    if not job_id:
        logger.error("Failed to submit job array")
        return 1

    logger.info(
        f"Submitted job array {job_id} with {len(batches)} tasks "
        f"(max {max_concurrent} concurrent)"
    )

    tracking_file = staging_dir / "job_tracking.json"
    with open(tracking_file, "w") as fh:
        json.dump(
            {
                "job_ids": [job_id],
                "result_paths": [str(p) for p in all_result_paths],
                "structure_id": structure_id,
            },
            fh,
            indent=2,
        )
    logger.info(f"Job tracking written to {tracking_file}")

    if no_wait:
        logger.info(
            "Not waiting for completion (--no-wait). " "Run --aggregate later to collect results."
        )
        return 0

    _wait_for_jobs([job_id], result_paths=all_result_paths)
    return _aggregate_results(all_result_paths, staging_dir)


# ===================================================================
# Aggregator
# ===================================================================


def _aggregate_results(result_paths: list[Path], staging_dir: Path) -> int:
    succeeded: list[str] = []
    failed: list[str] = []
    missing: list[str] = []

    for rp in result_paths:
        data = _read_worker_result(rp)
        if not data:
            missing.append(str(rp))
            logger.warning(f"Missing or unreadable result: {rp}")
            continue
        succeeded.extend(data.get("succeeded", []))
        failed.extend(data.get("failed", []))

    summary = {
        "total_succeeded": len(succeeded),
        "total_failed": len(failed),
        "missing_result_files": missing,
        "succeeded": succeeded,
        "failed": failed,
    }
    summary_path = staging_dir / "aggregated_results.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(f"Aggregated results written to {summary_path}")
    logger.info(
        f"Summary — Succeeded: {len(succeeded)}, Failed: {len(failed)}, "
        f"Missing result files: {len(missing)}"
    )
    return len(failed)


def _aggregate_from_tracking(tracking_file: Path) -> int:
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
        description="SLURM-parallel available-space calculation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Recommended: zero compute on login node (bash launcher)
  bash cellpack_analysis/preprocessing/submit_available_space_slurm.sh -s <structure_id>

# Direct orchestrate (runs Python on wherever you call it)
  python -m cellpack_analysis.preprocessing.run_available_space_slurm \\
      --orchestrate --structure-id <structure_id>

# Dry run
  python -m cellpack_analysis.preprocessing.run_available_space_slurm \\
      --orchestrate --structure-id <structure_id> --dry-run

# Aggregate after a no-wait submission
  python -m cellpack_analysis.preprocessing.run_available_space_slurm \\
      --aggregate data/structure_data/<id>/slurm_staging/job_tracking.json
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--orchestrate",
        action="store_true",
        help="Discover meshes, write manifests, and submit job array.",
    )
    mode.add_argument(
        "--worker",
        action="store_true",
        help="Execute distance calculations for one batch (called by SLURM).",
    )
    mode.add_argument(
        "--aggregate",
        metavar="TRACKING_FILE",
        help="Aggregate results from a previous --no-wait submission.",
    )

    # Orchestrate-mode args
    orch = parser.add_argument_group("orchestrate options")
    orch.add_argument("--structure-id", "-s", help="Structure ID to process.")
    orch.add_argument("--spacing", type=float, default=2.0, help="Grid spacing (default: 2.0).")
    orch.add_argument(
        "--use-struct-mesh",
        action="store_true",
        help="Exclude structure volume from available space.",
    )
    orch.add_argument(
        "--use-mean-shape",
        action="store_true",
        help="Use mean-shape cell only instead of individual cells.",
    )
    orch.add_argument(
        "--use-all-cells",
        action="store_true",
        help="Use all cells for the structure, including those outside the 8D sphere (default: False).",
    )
    orch.add_argument(
        "--recalculate",
        action="store_true",
        help="Recalculate even if output files already exist.",
    )
    orch.add_argument("--chunk-size", type=int, default=None, help="Override adaptive chunk size.")
    orch.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Cells per SLURM job (default: 4).",
    )
    orch.add_argument(
        "--max-jobs",
        type=int,
        default=16,
        help="Max concurrent SLURM array tasks (default: 16).",
    )
    orch.add_argument("--dry-run", action="store_true", help="Write scripts but do not submit.")
    orch.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after submission without waiting.",
    )
    orch.add_argument("--venv-path", help="Path to venv activate script.")

    # SLURM resource args
    slurm = parser.add_argument_group("SLURM resource options")
    slurm.add_argument("--slurm-partition", default="", help="SLURM partition.")
    slurm.add_argument("--slurm-time", default=SLURM_DEFAULTS["time"], help="Wall-clock limit.")
    slurm.add_argument("--slurm-mem", default=SLURM_DEFAULTS["mem"], help="Memory per job.")
    slurm.add_argument(
        "--slurm-cpus-per-task",
        type=int,
        default=int(SLURM_DEFAULTS["cpus_per_task"]),
        help="CPUs per job (also sets ProcessPoolExecutor workers).",
    )
    slurm.add_argument(
        "--slurm-job-name", default=SLURM_DEFAULTS["job_name"], help="SLURM job name."
    )

    # Worker-mode args
    worker = parser.add_argument_group("worker options")
    worker.add_argument("--batch-manifest", help="Path to batch manifest JSON.")
    worker.add_argument("--result-path", help="Path to write worker result JSON.")

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = _build_parser()
    args = parser.parse_args()

    if args.orchestrate:
        if not args.structure_id:
            parser.error("--structure-id is required with --orchestrate")

        slurm_opts: dict[str, str] = {
            "partition": args.slurm_partition,
            "time": args.slurm_time,
            "mem": args.slurm_mem,
            "cpus_per_task": str(args.slurm_cpus_per_task),
            "job_name": args.slurm_job_name,
        }

        rc = _run_orchestrator(
            structure_id=args.structure_id,
            spacing=args.spacing,
            use_struct_mesh=args.use_struct_mesh,
            use_mean_shape=args.use_mean_shape,
            use_all_cells=args.use_all_cells,
            recalculate=args.recalculate,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            max_concurrent=args.max_jobs,
            slurm_opts=slurm_opts,
            venv_path=args.venv_path,
            dry_run=args.dry_run,
            no_wait=args.no_wait,
        )
        sys.exit(rc)

    elif args.worker:
        if not args.batch_manifest or not args.result_path:
            parser.error("--batch-manifest and --result-path are required with --worker")
        _run_worker(Path(args.batch_manifest), Path(args.result_path))

    elif args.aggregate:
        rc = _aggregate_from_tracking(Path(args.aggregate))
        sys.exit(rc)


if __name__ == "__main__":
    main()
