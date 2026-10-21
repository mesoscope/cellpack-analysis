"""
Versioned, sha256-pinned input data for the paper (plan, build, fetch, verify).

The analysis inputs of record are published as one deterministic ``.tar.gz`` per group. The tracked
manifest ``configs/paper_inputs.json`` lists the groups, their file globs (relative to ``data/``)
and, once built, the archive name, sha256, size and file count. Archives extract into ``data/`` so
paths mirror the repository. Objects are fetched with plain anonymous HTTPS GETs (no credentials).

Usage
-----
::

    python -m cellpack_analysis.data_release.paper_inputs plan --data-dir DATA [--out plan.tsv]
    python -m cellpack_analysis.data_release.paper_inputs build --data-dir DATA --out-dir OUT
    python -m cellpack_analysis.data_release.paper_inputs fetch [--groups tables,meshes_RAB5A]
    python -m cellpack_analysis.data_release.paper_inputs verify
"""

import argparse
import copy
import fnmatch
import glob
import gzip
import hashlib
import json
import logging
import os
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
from typing import IO, Any

from cellpack_analysis.lib.file_io import get_datadir_path

logger = logging.getLogger(__name__)

DESCRIPTION = "Versioned, sha256-pinned input data for the paper (plan, build, fetch, verify)."
MANIFEST_PATH = Path(__file__).parent / "configs" / "paper_inputs.json"
STAMP_DIRNAME = ".paper_inputs"
DOWNLOAD_DIRNAME = "downloads"
MANIFEST_SHA_NAME = "MANIFEST.sha256"
GZIP_LEVEL = 6
FILE_MODE = 0o644
DIR_MODE = 0o755
CHUNK = 1 << 20
LOG_EVERY_S = 5.0
LOG_EVERY_BYTES = 1 << 30
MAX_ATTEMPTS = 3


class ChecksumError(ValueError):
    """Raised when a downloaded archive does not match the manifest sha256."""


@dataclass
class GroupPlan:
    """
    Resolved file list of one group.

    Attributes
    ----------
    name
        Group name.
    files
        Sorted data-relative POSIX paths.
    n_bytes
        Total size of the files in bytes.
    expected
        Declared expected file count, or None.
    """

    name: str
    files: list[str] = field(default_factory=list)
    n_bytes: int = 0
    expected: int | None = None

    @property
    def n_files(self) -> int:
        """Number of files resolved for the group."""
        return len(self.files)

    @property
    def flag(self) -> str:
        """Empty string when the group looks right, otherwise the reason it does not."""
        if self.n_files == 0:
            return "EMPTY"
        if self.expected is not None and self.n_files != self.expected:
            return f"COUNT(expected {self.expected})"
        return ""


# ---------------------------------------------------------------------------------------------
# manifest and glob resolution
# ---------------------------------------------------------------------------------------------


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict[str, Any]:
    """
    Load a paper-inputs manifest.

    Parameters
    ----------
    path
        Manifest JSON path. Defaults to the tracked manifest.

    Returns
    -------
    :
        The manifest with ``tag``, ``base_url`` and ``groups``.
    """
    with open(path) as f:
        manifest: dict[str, Any] = json.load(f)
    for key in ("tag", "base_url", "groups"):
        if key not in manifest:
            raise ValueError(f"Manifest {path} is missing the {key!r} key")
    names = [g["name"] for g in manifest["groups"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Manifest {path} has duplicate group names")
    return manifest


def save_manifest(manifest: dict[str, Any], path: Path | str) -> None:
    """Write a manifest as indented JSON."""
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")


def select_groups(manifest: dict[str, Any], names: Sequence[str] | None) -> list[dict[str, Any]]:
    """
    Select manifest groups by name.

    Parameters
    ----------
    manifest
        Loaded manifest.
    names
        Group names, or None / ``["all"]`` for every group.

    Returns
    -------
    :
        The selected group entries in manifest order.
    """
    groups: list[dict[str, Any]] = manifest["groups"]
    if not names or list(names) == ["all"]:
        return groups
    known = {g["name"] for g in groups}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown group(s) {unknown}; valid groups: {sorted(known)}")
    return [g for g in groups if g["name"] in set(names)]


def expand_braces(pattern: str) -> list[str]:
    """
    Expand ``{a,b}`` alternatives (which ``glob`` does not support), including nested ones.

    Parameters
    ----------
    pattern
        Glob pattern, possibly with brace groups.

    Returns
    -------
    :
        Patterns without braces, in alternative order.
    """
    start = pattern.find("{")
    if start < 0:
        return [pattern]
    depth, parts, last = 0, [], start + 1
    for i in range(start, len(pattern)):
        char = pattern[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parts.append(pattern[last:i])
                end = i
                break
        elif char == "," and depth == 1:
            parts.append(pattern[last:i])
            last = i + 1
    else:
        raise ValueError(f"Unbalanced braces in pattern {pattern!r}")
    head, tail = pattern[:start], pattern[end + 1 :]
    return [out for part in parts for out in expand_braces(head + part + tail)]


def _is_excluded(rel: str, patterns: Sequence[str]) -> bool:
    """Match a relative path against exclude patterns (basename patterns have no slash)."""
    base = rel.rsplit("/", 1)[-1]
    for pat in patterns:
        if "/" not in pat:
            if fnmatch.fnmatchcase(base, pat):
                return True
            continue
        candidates = [pat, pat[3:]] if pat.startswith("**/") else [pat]
        if any(fnmatch.fnmatchcase(rel, c) for c in candidates):
            return True
    return False


def resolve_group(data_dir: Path, group: dict[str, Any]) -> GroupPlan:
    """
    Resolve a group's include/exclude globs against ``data_dir``.

    Parameters
    ----------
    data_dir
        Data directory the globs are relative to.
    group
        Manifest group entry.

    Returns
    -------
    :
        Sorted regular files (symlinks followed) with their total size.
    """
    sizes: dict[str, int] = {}
    excludes = group.get("exclude", [])
    for include in group["include"]:
        for pattern in expand_braces(include):
            parts = pattern.split("/")
            if pattern.startswith("/") or ".." in parts:
                raise ValueError(f"Group {group['name']!r}: unsafe pattern {pattern!r}")
            for match in glob.glob(pattern, root_dir=data_dir, recursive=True):
                rel = Path(match).as_posix()
                if rel in sizes or _is_excluded(rel, excludes):
                    continue
                try:
                    st = os.stat(data_dir / rel)
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    sizes[rel] = st.st_size
    files = sorted(sizes)
    return GroupPlan(
        name=group["name"],
        files=files,
        n_bytes=sum(sizes.values()),
        expected=group.get("expected_n_files"),
    )


def plan_groups(data_dir: Path, groups: Sequence[dict[str, Any]]) -> list[GroupPlan]:
    """Resolve every group against ``data_dir`` (read-only)."""
    return [resolve_group(data_dir, g) for g in groups]


def format_plan_tsv(plans: Sequence[GroupPlan]) -> str:
    """Format resolved groups as a TSV (group, n_files, bytes, GB, expected, flag)."""
    lines = ["group\tn_files\tbytes\tGB\texpected_n_files\tflag"]
    for p in plans:
        expected = "" if p.expected is None else str(p.expected)
        lines.append(
            f"{p.name}\t{p.n_files}\t{p.n_bytes}\t{p.n_bytes / 1e9:.3f}\t{expected}\t{p.flag}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------


class _HashingWriter:
    """Write-through file wrapper that hashes and counts the bytes written."""

    def __init__(self, fh: IO[bytes]) -> None:
        self._fh = fh
        self._sha = hashlib.sha256()
        self.n_bytes = 0

    def write(self, data: bytes) -> int:
        self._sha.update(data)
        self.n_bytes += len(data)
        return self._fh.write(data)

    def flush(self) -> None:
        self._fh.flush()

    def hexdigest(self) -> str:
        return self._sha.hexdigest()


def _file_tarinfo(rel: str, st: os.stat_result) -> tarfile.TarInfo:
    """Build a normalized tar header for a regular file (mtime kept, owner and mode fixed)."""
    info = tarfile.TarInfo(rel)
    info.type = tarfile.REGTYPE
    info.size = st.st_size
    info.mtime = int(st.st_mtime)
    info.mode = FILE_MODE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _dir_tarinfo(rel: str) -> tarfile.TarInfo:
    """Build a normalized tar header for a directory."""
    info = tarfile.TarInfo(rel)
    info.type = tarfile.DIRTYPE
    info.mtime = 0
    info.mode = DIR_MODE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _ancestor_dirs(files: Sequence[str]) -> set[str]:
    """Return every ancestor directory of the given relative paths."""
    dirs: set[str] = set()
    for rel in files:
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    return dirs


def build_archive(data_dir: Path, plan: GroupPlan, out_path: Path) -> tuple[str, int]:
    """
    Write one deterministic ``.tar.gz`` for a resolved group.

    Members are sorted by path, modes are normalized (files 0o644, directories 0o755), uid and gid
    are 0, user and group names are empty, file mtimes are kept, and the gzip header carries no
    name or time. Identical inputs therefore give identical bytes.

    Parameters
    ----------
    data_dir
        Data directory the group paths are relative to.
    plan
        Resolved group (see `resolve_group`).
    out_path
        Archive path to write (written via a ``.partial`` file, then renamed).

    Returns
    -------
    :
        sha256 hex digest and size in bytes of the archive.
    """
    entries = sorted(set(plan.files) | _ancestor_dirs(plan.files))
    files = set(plan.files)
    partial = out_path.with_name(out_path.name + ".partial")
    start = time.time()
    done_bytes = 0
    next_log = LOG_EVERY_BYTES
    try:
        with open(partial, "wb") as raw:
            sink = _HashingWriter(raw)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=sink, compresslevel=GZIP_LEVEL, mtime=0
            ) as gz:
                with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for rel in entries:
                        if rel not in files:
                            tar.addfile(_dir_tarinfo(rel))
                            continue
                        with open(data_dir / rel, "rb") as src:
                            info = _file_tarinfo(rel, os.fstat(src.fileno()))
                            tar.addfile(info, src)
                        done_bytes += info.size
                        if done_bytes >= next_log:
                            next_log += LOG_EVERY_BYTES
                            logger.info(
                                "  %s: read %.1f/%.1f GB (%.0fs)",
                                plan.name,
                                done_bytes / 1e9,
                                plan.n_bytes / 1e9,
                                time.time() - start,
                            )
        os.replace(partial, out_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return sink.hexdigest(), sink.n_bytes


def _check_resolved(plans: Sequence[GroupPlan]) -> None:
    """Fail on empty groups and on groups whose count differs from the declared expectation."""
    bad = [f"{p.name}: {p.n_files} files, {p.flag}" for p in plans if p.flag]
    if bad:
        raise ValueError("Refusing to build; unexpected file counts:\n  " + "\n  ".join(bad))


def _build_one(data_dir: Path, plan: GroupPlan, out_dir: Path, archive: str) -> dict[str, Any]:
    logger.info("Building %s: %d files, %.2f GB", plan.name, plan.n_files, plan.n_bytes / 1e9)
    start = time.time()
    sha, size = build_archive(data_dir, plan, out_dir / archive)
    logger.info(
        "Built %s -> %s (%.2f GB, %.0fs)", archive, sha[:12], size / 1e9, time.time() - start
    )
    return {"archive": archive, "sha256": sha, "bytes": size, "n_files": plan.n_files}


def build(
    data_dir: Path,
    out_dir: Path,
    manifest: dict[str, Any],
    names: Sequence[str] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """
    Build archives for the selected groups and write the filled manifest and checksums.

    Parameters
    ----------
    data_dir
        Data directory the group globs are relative to.
    out_dir
        Output directory for the archives, ``paper_inputs.json`` and ``MANIFEST.sha256``.
    manifest
        Loaded spec manifest.
    names
        Groups to build (default all). Groups not rebuilt keep the fields of an existing
        ``out_dir/paper_inputs.json`` when their archive is still present.
    workers
        Number of groups built concurrently.

    Returns
    -------
    :
        The filled manifest that was written to ``out_dir/paper_inputs.json``.
    """
    groups = select_groups(manifest, names)
    plans = plan_groups(data_dir, groups)
    _check_resolved(plans)
    out_dir.mkdir(parents=True, exist_ok=True)

    filled = copy.deepcopy(manifest)
    previous_path = out_dir / "paper_inputs.json"
    if previous_path.exists():
        previous = {g["name"]: g for g in load_manifest(previous_path)["groups"]}
        for group in filled["groups"]:
            prev = previous.get(group["name"])
            if prev and prev.get("sha256") and (out_dir / prev["archive"]).exists():
                group.update({k: prev[k] for k in ("archive", "sha256", "bytes", "n_files")})

    by_name = {g["name"]: g for g in filled["groups"]}
    jobs = [(p, f"{p.name}.tar.gz") for p in plans]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_build_one, data_dir, p, out_dir, a) for p, a in jobs]
        for (plan, _), future in zip(jobs, futures, strict=True):
            by_name[plan.name].update(future.result())

    save_manifest(filled, previous_path)
    lines = sorted(f"{g['sha256']}  {g['archive']}" for g in filled["groups"] if g.get("sha256"))
    (out_dir / MANIFEST_SHA_NAME).write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s and %s in %s", previous_path.name, MANIFEST_SHA_NAME, out_dir)
    return filled


# ---------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------


def _is_http(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _download_http(url: str, dest: Path, label: str) -> tuple[str, int]:
    """Stream a URL to ``dest`` with progress; return sha256 and size."""
    request = urllib.request.Request(url, headers={"User-Agent": "cellpack-analysis-paper-inputs"})
    sha = hashlib.sha256()
    n_bytes = 0
    with urllib.request.urlopen(request, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        last = time.time()
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            sha.update(chunk)
            n_bytes += len(chunk)
            if time.time() - last >= LOG_EVERY_S:
                last = time.time()
                pct = f" ({100 * n_bytes / total:.0f}%)" if total else ""
                logger.info("  %s: %.2f GB downloaded%s", label, n_bytes / 1e9, pct)
        if total and n_bytes != total:
            raise IncompleteRead(b"", total - n_bytes)
    return sha.hexdigest(), n_bytes


def _download_file(src: Path, dest: Path) -> tuple[str, int]:
    """Copy a local archive to ``dest``; return sha256 and size."""
    sha = hashlib.sha256()
    n_bytes = 0
    with open(src, "rb") as fin, open(dest, "wb") as out:
        while chunk := fin.read(CHUNK):
            out.write(chunk)
            sha.update(chunk)
            n_bytes += len(chunk)
    return sha.hexdigest(), n_bytes


def _obtain_archive(source: str, archive: str, dest: Path, label: str) -> tuple[str, int]:
    """Get one archive from an https base URL or a local directory into ``dest``."""
    if source.startswith("s3://"):
        raise ValueError("Use the https URL of the bucket, not an s3:// URI (anonymous GET only)")
    if not _is_http(source):
        src = Path(source) / archive
        if not src.is_file():
            raise FileNotFoundError(f"Archive not found in local source: {src}")
        logger.info("Copying %s from %s", archive, source)
        return _download_file(src, dest)
    url = source.rstrip("/") + "/" + urllib.parse.quote(archive)
    logger.info("Downloading %s", url)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _download_http(url, dest, label)
        except urllib.error.HTTPError as err:
            raise RuntimeError(
                f"GET {url} failed with HTTP {err.code}; is the release uploaded and public?"
            ) from err
        except (urllib.error.URLError, OSError, IncompleteRead) as err:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"GET {url} failed after {attempt} attempts: {err}") from err
            logger.warning("Download failed (%s); retry %d/%d", err, attempt, MAX_ATTEMPTS - 1)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _check_member_name(member: tarfile.TarInfo) -> None:
    """Reject absolute names and ``..`` components (the data filter would only strip a slash)."""
    parts = PurePosixPath(member.name).parts
    if member.name.startswith("/") or ".." in parts:
        raise ValueError(f"Unsafe member name in archive: {member.name!r}")


def _make_dir(data_dir: Path, member: tarfile.TarInfo) -> None:
    """Create an archive directory entry (mode 0o755) without touching existing directories."""
    target = data_dir / member.name
    if target.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(DIR_MODE)


def extract_archive(archive_path: Path, data_dir: Path) -> list[tuple[str, int]]:
    """
    Extract a paper-inputs archive into ``data_dir`` in a single pass.

    Parameters
    ----------
    archive_path
        Local ``.tar.gz`` file.
    data_dir
        Extraction target.

    Returns
    -------
    :
        Sorted ``(relative path, size)`` of every regular file extracted.
    """
    members: list[tuple[str, int]] = []
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            _check_member_name(member)
            if member.isdir():
                _make_dir(data_dir, member)
            elif member.isreg():
                tar.extract(member, path=data_dir, filter="data")
                members.append((member.name, member.size))
            else:
                raise ValueError(f"Unsupported member type for {member.name!r} in {archive_path}")
    return sorted(members)


def _stamp_path(data_dir: Path, name: str) -> Path:
    return data_dir / STAMP_DIRNAME / f"{name}.json"


def _read_stamp(data_dir: Path, name: str) -> dict[str, Any] | None:
    path = _stamp_path(data_dir, name)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            stamp: dict[str, Any] = json.load(f)
    except (OSError, ValueError):
        return None
    return stamp


def _require_pinned(group: dict[str, Any]) -> None:
    sha = group.get("sha256")
    if not (group.get("archive") and isinstance(sha, str) and len(sha) == 64):
        raise ValueError(
            f"Group {group['name']!r} has no archive/sha256 in the manifest. The manifest is "
            "unbuilt: run 'build' and copy its paper_inputs.json over the tracked manifest."
        )


def fetch(
    manifest: dict[str, Any],
    data_dir: Path,
    names: Sequence[str] | None = None,
    source: str | None = None,
    keep_archives: bool = False,
) -> list[str]:
    """
    Fetch, verify and extract the selected groups into ``data_dir``.

    A group whose stamp (``data_dir/.paper_inputs/<group>.json``) matches the manifest sha256 is
    skipped. The sha256 is checked before extraction, and the stamp is written only after the
    extracted file count matches the manifest.

    Parameters
    ----------
    manifest
        Loaded manifest with pinned sha256 values.
    data_dir
        Extraction target (the repository ``data/`` directory).
    names
        Groups to fetch (default all).
    source
        Base URL or local directory holding the archives. Defaults to the manifest ``base_url``.
    keep_archives
        Keep the downloaded archives under ``data_dir/.paper_inputs/downloads``.

    Returns
    -------
    :
        Names of the groups that were downloaded and extracted.
    """
    groups = select_groups(manifest, names)
    for group in groups:
        _require_pinned(group)
    source = source or manifest["base_url"]
    download_dir = data_dir / STAMP_DIRNAME / DOWNLOAD_DIRNAME
    fetched: list[str] = []
    for i, group in enumerate(groups, start=1):
        name, sha = group["name"], group["sha256"]
        prefix = f"[{i}/{len(groups)}] {name}"
        stamp = _read_stamp(data_dir, name)
        if stamp and stamp.get("sha256") == sha and stamp.get("tag") == manifest["tag"]:
            logger.info("%s: up to date (stamp matches %s)", prefix, sha[:12])
            continue
        download_dir.mkdir(parents=True, exist_ok=True)
        dest = download_dir / group["archive"]
        start = time.time()
        logger.info("%s: fetching (%.2f GB archive)", prefix, group["bytes"] / 1e9)
        try:
            got_sha, got_bytes = _obtain_archive(source, group["archive"], dest, name)
            if got_sha != sha:
                dest.unlink(missing_ok=True)
                raise ChecksumError(
                    f"{prefix}: sha256 mismatch for {group['archive']} (expected {sha}, got "
                    f"{got_sha}, {got_bytes} bytes); the download was deleted"
                )
            logger.info("%s: sha256 verified, extracting into %s", prefix, data_dir)
            members = extract_archive(dest, data_dir)
        finally:
            if not keep_archives:
                dest.unlink(missing_ok=True)
        if len(members) != group["n_files"]:
            raise ValueError(
                f"{prefix}: extracted {len(members)} files, manifest says {group['n_files']}"
            )
        stamp = {
            "group": name,
            "tag": manifest["tag"],
            "archive": group["archive"],
            "sha256": sha,
            "bytes": group["bytes"],
            "n_files": group["n_files"],
            "members": [[rel, size] for rel, size in members],
        }
        path = _stamp_path(data_dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        tmp.write_text(json.dumps(stamp) + "\n")
        os.replace(tmp, path)
        fetched.append(name)
        logger.info("%s: done, %d files (%.0fs)", prefix, len(members), time.time() - start)
    if not keep_archives:
        try:
            download_dir.rmdir()
        except OSError:
            pass
    return fetched


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def verify(data_dir: Path, manifest: dict[str, Any] | None = None) -> list[str]:
    """
    Check every stamped group: each recorded file exists with its recorded size.

    Parameters
    ----------
    data_dir
        Data directory holding ``.paper_inputs`` stamps.
    manifest
        Optional manifest; stamps whose sha256 differs from it are reported as stale.

    Returns
    -------
    :
        Problem descriptions (empty when everything checks out).
    """
    stamp_dir = data_dir / STAMP_DIRNAME
    stamps = sorted(stamp_dir.glob("*.json")) if stamp_dir.is_dir() else []
    pinned = {g["name"]: g.get("sha256") for g in manifest["groups"]} if manifest else {}
    problems: list[str] = []
    if not stamps:
        problems.append(f"no stamped groups under {stamp_dir}; run 'fetch' first")
    for path in stamps:
        stamp = _read_stamp(data_dir, path.stem)
        if stamp is None:
            problems.append(f"{path.name}: unreadable stamp")
            continue
        name = stamp["group"]
        if pinned.get(name) and pinned[name] != stamp["sha256"]:
            problems.append(f"{name}: stamp sha256 differs from the manifest (stale; re-fetch)")
        n_bad = 0
        for rel, size in stamp["members"]:
            file_path = data_dir / rel
            if not file_path.is_file():
                problems.append(f"{name}: missing {rel}")
                n_bad += 1
            elif file_path.stat().st_size != size:
                problems.append(f"{name}: size {file_path.stat().st_size} != {size} for {rel}")
                n_bad += 1
        status = "OK" if n_bad == 0 else f"{n_bad} problem file(s)"
        logger.info("verify %s: %d files, %s", name, len(stamp["members"]), status)
    return problems


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _split_groups(value: str | None) -> list[str] | None:
    return [g for g in value.split(",") if g] if value else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper_inputs", description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, data_dir_required: bool = False) -> None:
        p.add_argument("--data-dir", type=Path, required=data_dir_required, help="Data directory")
        p.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Manifest JSON")
        p.add_argument("--groups", default=None, help="Comma-separated group names, or 'all'")

    plan = sub.add_parser("plan", help="Resolve group globs and print file counts and sizes")
    common(plan, data_dir_required=True)
    plan.add_argument("--out", type=Path, default=None, help="Write the TSV to this path")

    bld = sub.add_parser("build", help="Build deterministic per-group .tar.gz archives")
    common(bld, data_dir_required=True)
    bld.add_argument("--out-dir", type=Path, required=True, help="Archive output directory")
    bld.add_argument("--workers", type=int, default=1, help="Groups built concurrently")

    fch = sub.add_parser("fetch", help="Download, verify and extract groups into the data dir")
    common(fch)
    fch.add_argument("--source", default=None, help="Base URL or local directory of the archives")
    fch.add_argument("--keep-archives", action="store_true", help="Keep downloaded archives")

    ver = sub.add_parser("verify", help="Check that stamped groups are still complete")
    common(ver)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface; return the process exit code (2 on an error)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, RuntimeError, OSError, tarfile.TarError) as err:
        logger.error("%s: %s", type(err).__name__, err)
        return 2


def _run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    data_dir = (args.data_dir or get_datadir_path()).resolve()
    names = _split_groups(args.groups)

    if args.command == "plan":
        plans = plan_groups(data_dir, select_groups(manifest, names))
        tsv = format_plan_tsv(plans)
        logger.info("Plan for %s\n%s", data_dir, tsv.rstrip("\n"))
        logger.info(
            "Total: %d files, %.2f GB",
            sum(p.n_files for p in plans),
            sum(p.n_bytes for p in plans) / 1e9,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(tsv)
        return 1 if any(p.flag for p in plans) else 0
    if args.command == "build":
        build(data_dir, args.out_dir, manifest, names, args.workers)
        return 0
    if args.command == "fetch":
        fetch(manifest, data_dir, names, args.source, args.keep_archives)
        return 0
    problems = verify(data_dir, manifest)
    for problem in problems:
        logger.error(problem)
    logger.info("verify: %s", "all stamped groups complete" if not problems else "FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
