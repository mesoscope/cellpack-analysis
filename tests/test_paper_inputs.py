"""Tests for the versioned paper-input archives (plan, build, fetch, verify) and table loading."""

import copy
import functools
import hashlib
import http.server
import io
import json
import os
import stat
import tarfile
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cellpack_analysis.data_release import paper_inputs as pi
from cellpack_analysis.lib import get_structure_stats_dataframe as gssd
from cellpack_analysis.lib import io as cp_io

V4 = "packing_outputs/8d_sphere_data/replicate_99_v4"
FETCH_CMD = "python -m cellpack_analysis.data_release.paper_inputs fetch --groups tables"


def _write(
    path: Path, content: bytes | str, mode: int | None = None, mtime: int | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    if mode is not None:
        path.chmod(mode)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _group(
    name: str, include: list[str], exclude: list[str] | None = None, expected: int | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "description": name,
        "used_by": "test",
        "include": include,
        "exclude": ["**/all_positions_*", "._*", ".DS_Store"] + (exclude or []),
        "expected_n_files": expected,
        "archive": None,
        "sha256": None,
        "bytes": None,
        "n_files": None,
    }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A small synthetic data tree that mirrors the repository layout."""
    d = tmp_path / "data"
    _write(d / "all_cell_ids.parquet", b"cells" * 100, mtime=1_700_000_000)
    _write(d / "structure_stats.parquet", b"stats" * 100, mtime=1_700_000_001)
    _write(d / "structure_data/SID/meshes/mem_mesh_1.obj", "v 0 0 0\n" * 50, mode=0o600)
    _write(d / "structure_data/SID/meshes/nuc_mesh_1.obj", "v 1 1 1\n" * 50, mode=0o777)
    _write(d / "structure_data/SID/meshes/readme.txt", "not a mesh")
    spheres = d / V4 / "random/peroxisome/spheresSST"
    _write(spheres / "positions_a_1.json", '{"a": 1}', mode=0o444)
    _write(spheres / "positions_a_2.json", '{"a": 2}')
    _write(spheres / "all_positions_peroxisome_analyze_random.json", "derived")
    _write(spheres / "._positions_a_1.json", "appledouble")
    _write(spheres / ".DS_Store", "ds")
    _write(d / V4 / "ENGINE_PROVENANCE.txt", "engine 74a51050")
    _write(d / "recipes/peroxisome/replicate_99_v4/random/r_1.json", "{}")
    _write(d / "recipes/peroxisome/replicate_99_v4/interpolated_floor-0.28/r_1.json", "{}")
    _write(d / "configs/peroxisome/replicate_99_v4/random/c.json", "{}")
    return d


@pytest.fixture
def spec() -> dict[str, Any]:
    """A manifest spec (unbuilt) matching the `data_dir` fixture."""
    return {
        "tag": "v0.0.1",
        "base_url": "https://example.invalid/release/v0.0.1/",
        "groups": [
            _group("tables", ["all_cell_ids.parquet", "structure_stats.parquet"], expected=2),
            _group("meshes_SID", ["structure_data/SID/meshes/*.obj"], expected=2),
            _group(
                "packings_v4_random",
                [
                    f"{V4}/random/{{peroxisome,endosome}}/spheresSST/positions_*.json",
                    f"{V4}/ENGINE_PROVENANCE.txt",
                ],
                expected=3,
            ),
            _group(
                "packing_inputs",
                ["recipes/peroxisome/replicate_99_v4/**", "configs/peroxisome/replicate_99_v4/**"],
                exclude=["**/interpolated_floor-0.28/**"],
                expected=2,
            ),
        ],
    }


@pytest.fixture
def built(data_dir: Path, spec: dict[str, Any], tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Build all groups; return (filled manifest, out_dir)."""
    out_dir = tmp_path / "out"
    return pi.build(data_dir, out_dir, spec), out_dir


def _fetch_target(tmp_path: Path, name: str = "fresh") -> Path:
    target = tmp_path / name
    target.mkdir()
    return target


# ---------------------------------------------------------------------------------------------
# globs and plan
# ---------------------------------------------------------------------------------------------


def test_expand_braces() -> None:
    assert pi.expand_braces("a/b") == ["a/b"]
    assert pi.expand_braces("a/{x,y}/c") == ["a/x/c", "a/y/c"]
    assert pi.expand_braces("{a,b}/{x,y}") == ["a/x", "a/y", "b/x", "b/y"]
    assert pi.expand_braces("p/{a,b{1,2}}/q") == ["p/a/q", "p/b1/q", "p/b2/q"]
    with pytest.raises(ValueError, match="Unbalanced"):
        pi.expand_braces("a/{x,y")


def test_plan_counts_and_excludes(data_dir: Path, spec: dict[str, Any]) -> None:
    plans = {p.name: p for p in pi.plan_groups(data_dir, spec["groups"])}
    assert plans["tables"].n_files == 2
    assert plans["meshes_SID"].files == [
        "structure_data/SID/meshes/mem_mesh_1.obj",
        "structure_data/SID/meshes/nuc_mesh_1.obj",
    ]
    packings = plans["packings_v4_random"]
    assert packings.files == [
        f"{V4}/ENGINE_PROVENANCE.txt",
        f"{V4}/random/peroxisome/spheresSST/positions_a_1.json",
        f"{V4}/random/peroxisome/spheresSST/positions_a_2.json",
    ]
    inputs = plans["packing_inputs"]
    assert inputs.files == [
        "configs/peroxisome/replicate_99_v4/random/c.json",
        "recipes/peroxisome/replicate_99_v4/random/r_1.json",
    ]
    assert plans["tables"].n_bytes == 500 + 500
    assert all(p.flag == "" for p in plans.values())


def test_plan_flags_empty_and_count_mismatch(data_dir: Path) -> None:
    empty = _group("empty", ["nothing/*.json"])
    wrong = _group("wrong", ["all_cell_ids.parquet"], expected=5)
    plans = pi.plan_groups(data_dir, [empty, wrong])
    assert plans[0].flag == "EMPTY"
    assert plans[1].flag == "COUNT(expected 5)"
    tsv = pi.format_plan_tsv(plans).splitlines()
    assert tsv[0] == "group\tn_files\tbytes\tGB\texpected_n_files\tflag"
    assert tsv[1].split("\t")[:3] == ["empty", "0", "0"]


def test_unsafe_patterns_rejected(data_dir: Path) -> None:
    for pattern in ("../outside/*", "/abs/path/*"):
        with pytest.raises(ValueError, match="unsafe pattern"):
            pi.resolve_group(data_dir, _group("bad", [pattern]))


def test_plan_cli_writes_tsv_and_is_read_only(
    data_dir: Path, spec: dict[str, Any], tmp_path: Path
) -> None:
    manifest_path = tmp_path / "spec.json"
    manifest_path.write_text(json.dumps(spec))
    out = tmp_path / "plan.tsv"
    before = sorted(p.relative_to(data_dir) for p in data_dir.rglob("*"))
    code = pi.main(
        ["plan", "--data-dir", str(data_dir), "--manifest", str(manifest_path), "--out", str(out)]
    )
    assert code == 0
    assert out.read_text().startswith("group\tn_files")
    assert sorted(p.relative_to(data_dir) for p in data_dir.rglob("*")) == before


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------


def test_build_cli(data_dir: Path, spec: dict[str, Any], tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    out_dir = tmp_path / "cli_out"
    argv = ["build", "--data-dir", str(data_dir), "--manifest", str(spec_path)]
    assert pi.main([*argv, "--out-dir", str(out_dir), "--workers", "2", "--groups", "tables"]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "MANIFEST.sha256",
        "paper_inputs.json",
        "tables.tar.gz",
    ]


def test_tracked_manifest_is_well_formed() -> None:
    manifest = pi.load_manifest()
    names = [g["name"] for g in manifest["groups"]]
    assert manifest["tag"] == "v1.0.0" and manifest["base_url"].endswith("/release/v1.0.0/")
    assert "tables" in names and len(set(names)) == len(names)
    for group in manifest["groups"]:
        assert group["include"] and "**/all_positions_*" in group["exclude"]
        assert {"description", "used_by", "archive", "sha256", "bytes", "n_files"} <= set(group)


def test_build_fails_on_empty_group(data_dir: Path, spec: dict[str, Any], tmp_path: Path) -> None:
    spec["groups"].append(_group("empty", ["nothing/*.json"]))
    with pytest.raises(ValueError, match="empty: 0 files"):
        pi.build(data_dir, tmp_path / "out", spec)
    assert not list((tmp_path / "out").glob("*.tar.gz"))


def test_build_fails_on_unexpected_count(
    data_dir: Path, spec: dict[str, Any], tmp_path: Path
) -> None:
    spec["groups"][0]["expected_n_files"] = 3
    with pytest.raises(ValueError, match="tables: 2 files"):
        pi.build(data_dir, tmp_path / "out", spec)


def test_build_is_deterministic(data_dir: Path, spec: dict[str, Any], tmp_path: Path) -> None:
    first = pi.build(data_dir, tmp_path / "out1", spec, workers=2)
    second = pi.build(data_dir, tmp_path / "out2", spec)
    for g1, g2 in zip(first["groups"], second["groups"], strict=True):
        assert g1["sha256"] == g2["sha256"]
        assert (tmp_path / "out1" / g1["archive"]).read_bytes() == (
            tmp_path / "out2" / g2["archive"]
        ).read_bytes()


def test_build_manifest_and_checksums(built: tuple[dict[str, Any], Path], data_dir: Path) -> None:
    manifest, out_dir = built
    for group in manifest["groups"]:
        archive = out_dir / group["archive"]
        assert group["archive"] == f"{group['name']}.tar.gz"
        assert group["bytes"] == archive.stat().st_size
        assert group["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        assert group["n_files"] == len(pi.resolve_group(data_dir, group).files)
    assert json.loads((out_dir / "paper_inputs.json").read_text()) == manifest
    for line in (out_dir / "MANIFEST.sha256").read_text().splitlines():
        sha, name = line.split("  ")  # sha256sum -c format: two spaces
        assert hashlib.sha256((out_dir / name).read_bytes()).hexdigest() == sha


def test_build_archive_headers_are_normalized(built: tuple[dict[str, Any], Path]) -> None:
    manifest, out_dir = built
    group = next(g for g in manifest["groups"] if g["name"] == "meshes_SID")
    raw = (out_dir / group["archive"]).read_bytes()
    assert raw[3] == 0 and raw[4:8] == b"\0\0\0\0"  # no FNAME flag, gzip mtime 0
    with tarfile.open(out_dir / group["archive"]) as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == sorted(m.name for m in members)
    for m in members:
        assert (m.uid, m.gid, m.uname, m.gname) == (0, 0, "", "")
        assert m.mode == (0o755 if m.isdir() else 0o644)


def test_tarinfo_mode_000_is_normalized() -> None:
    st = os.stat_result((stat.S_IFREG | 0o000, 0, 0, 1, 1234, 5678, 42, 0, 1_700_000_123, 0))
    info = pi._file_tarinfo("a/b.json", st)
    assert (info.mode, info.uid, info.gid, info.uname, info.gname) == (0o644, 0, 0, "", "")
    assert (info.size, info.mtime) == (42, 1_700_000_123)


def test_partial_build_keeps_previous_groups(
    data_dir: Path, spec: dict[str, Any], tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    full = pi.build(data_dir, out_dir, spec)
    partial = pi.build(data_dir, out_dir, spec, names=["tables"])
    assert partial["groups"] == full["groups"]
    with pytest.raises(ValueError, match="Unknown group"):
        pi.build(data_dir, out_dir, spec, names=["nope"])


# ---------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------


def test_fetch_round_trip_from_local_source(
    built: tuple[dict[str, Any], Path], data_dir: Path, tmp_path: Path
) -> None:
    manifest, out_dir = built
    target = _fetch_target(tmp_path)
    fetched = pi.fetch(manifest, target, source=str(out_dir))
    assert fetched == [g["name"] for g in manifest["groups"]]
    for group in manifest["groups"]:
        for rel in pi.resolve_group(data_dir, group).files:
            assert (target / rel).read_bytes() == (data_dir / rel).read_bytes()
            assert (target / rel).stat().st_mtime == int((data_dir / rel).stat().st_mtime)
    assert not (
        target / V4 / "random/peroxisome/spheresSST/all_positions_peroxisome_analyze_random.json"
    ).exists()
    assert not list((target / pi.STAMP_DIRNAME).glob("downloads"))
    stamp = json.loads((target / pi.STAMP_DIRNAME / "tables.json").read_text())
    assert stamp["sha256"] == manifest["groups"][0]["sha256"]
    assert stamp["members"] == [["all_cell_ids.parquet", 500], ["structure_stats.parquet", 500]]


def test_fetch_normalizes_modes_even_under_strict_umask(
    built: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    manifest, out_dir = built
    target = _fetch_target(tmp_path)
    old = os.umask(0o077)
    try:
        pi.fetch(manifest, target, source=str(out_dir))
    finally:
        os.umask(old)
    assert (
        stat.S_IMODE((target / "structure_data/SID/meshes/mem_mesh_1.obj").stat().st_mode) == 0o644
    )
    assert (
        stat.S_IMODE((target / "structure_data/SID/meshes/nuc_mesh_1.obj").stat().st_mode) == 0o644
    )
    assert (
        stat.S_IMODE(
            (target / f"{V4}/random/peroxisome/spheresSST/positions_a_1.json").stat().st_mode
        )
        == 0o644
    )
    assert stat.S_IMODE((target / "structure_data/SID/meshes").stat().st_mode) == 0o755


def test_fetch_leaves_existing_directories_alone(
    built: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    manifest, out_dir = built
    target = _fetch_target(tmp_path)
    existing = target / "structure_data"
    existing.mkdir()
    existing.chmod(0o750)
    pi.fetch(manifest, target, names=["meshes_SID"], source=str(out_dir))
    assert stat.S_IMODE(existing.stat().st_mode) == 0o750


def test_fetch_skips_when_stamp_matches(built: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    manifest, out_dir = built
    target = _fetch_target(tmp_path)
    assert pi.fetch(manifest, target, names=["tables"], source=str(out_dir))
    (out_dir / "tables.tar.gz").unlink()  # a second fetch must not need the archive
    assert pi.fetch(manifest, target, names=["tables"], source=str(out_dir)) == []
    changed = copy.deepcopy(manifest)
    changed["groups"][0]["sha256"] = "0" * 64
    with pytest.raises(FileNotFoundError):  # a new sha invalidates the stamp
        pi.fetch(changed, target, names=["tables"], source=str(out_dir))


def test_fetch_keep_archives(built: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    manifest, out_dir = built
    target = _fetch_target(tmp_path)
    pi.fetch(manifest, target, names=["tables"], source=str(out_dir), keep_archives=True)
    kept = target / pi.STAMP_DIRNAME / pi.DOWNLOAD_DIRNAME / "tables.tar.gz"
    assert kept.read_bytes() == (out_dir / "tables.tar.gz").read_bytes()


def test_fetch_sha_mismatch_raises_and_leaves_no_stamp(
    built: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    manifest, out_dir = built
    source = tmp_path / "corrupt"
    source.mkdir()
    data = (out_dir / "tables.tar.gz").read_bytes()
    (source / "tables.tar.gz").write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    target = _fetch_target(tmp_path)
    with pytest.raises(pi.ChecksumError, match="sha256 mismatch"):
        pi.fetch(manifest, target, names=["tables"], source=str(source), keep_archives=True)
    assert not (target / pi.STAMP_DIRNAME / "tables.json").exists()
    assert not (target / pi.STAMP_DIRNAME / pi.DOWNLOAD_DIRNAME / "tables.tar.gz").exists()
    assert not (target / "all_cell_ids.parquet").exists()
    assert (source / "tables.tar.gz").exists()  # the source copy is never deleted


def test_fetch_requires_a_built_manifest(spec: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no archive/sha256"):
        pi.fetch(spec, _fetch_target(tmp_path), names=["tables"], source=str(tmp_path))


def test_fetch_checks_file_count(built: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    manifest, out_dir = built
    manifest = copy.deepcopy(manifest)
    manifest["groups"][0]["n_files"] = 7
    target = _fetch_target(tmp_path)
    with pytest.raises(ValueError, match="extracted 2 files, manifest says 7"):
        pi.fetch(manifest, target, names=["tables"], source=str(out_dir))
    assert not (target / pi.STAMP_DIRNAME / "tables.json").exists()


def _malicious_manifest(tmp_path: Path, member_name: str) -> tuple[dict[str, Any], Path]:
    source = tmp_path / "evil_src"
    source.mkdir(exist_ok=True)
    archive = source / "evil.tar.gz"
    payload = b"owned"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    manifest = {
        "tag": "v0",
        "base_url": "https://example.invalid/",
        "groups": [
            {
                "name": "evil",
                "archive": "evil.tar.gz",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "bytes": archive.stat().st_size,
                "n_files": 1,
            }
        ],
    }
    return manifest, source


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/tmp_evil_abs.txt"])
def test_fetch_rejects_path_traversal(tmp_path: Path, name: str) -> None:
    manifest, source = _malicious_manifest(tmp_path, name)
    target = _fetch_target(tmp_path, "victim")
    with pytest.raises(ValueError, match="Unsafe member name"):
        pi.fetch(manifest, target, source=str(source))
    assert not (tmp_path / "evil.txt").exists()
    assert not Path("/tmp_evil_abs.txt").exists()
    assert not (target / pi.STAMP_DIRNAME / "evil.json").exists()


def test_fetch_rejects_links(tmp_path: Path) -> None:
    source = tmp_path / "link_src"
    source.mkdir()
    archive = source / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    manifest = {
        "tag": "v0",
        "base_url": "https://example.invalid/",
        "groups": [
            {
                "name": "evil",
                "archive": "evil.tar.gz",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "bytes": archive.stat().st_size,
                "n_files": 0,
            }
        ],
    }
    with pytest.raises(ValueError, match="Unsupported member type"):
        pi.fetch(manifest, _fetch_target(tmp_path), source=str(source))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return None


def test_fetch_over_http(built: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    manifest, out_dir = built
    handler = functools.partial(_QuietHandler, directory=str(out_dir))
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError:
        pytest.skip("cannot bind a loopback port")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        target = _fetch_target(tmp_path)
        assert pi.fetch(manifest, target, names=["tables"], source=base) == ["tables"]
        assert (target / "all_cell_ids.parquet").exists()
        missing = copy.deepcopy(manifest)
        missing["groups"][1]["archive"] = "absent.tar.gz"
        with pytest.raises(RuntimeError, match="HTTP 404"):
            pi.fetch(missing, _fetch_target(tmp_path, "other"), names=["meshes_SID"], source=base)
    finally:
        server.shutdown()
        server.server_close()


def test_cli_reports_errors_with_exit_code_2(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"tag": "v0", "base_url": "x", "groups": [_group("g", ["a"])]}))
    assert pi.main(["fetch", "--data-dir", str(tmp_path), "--manifest", str(spec_path)]) == 2


def test_fetch_rejects_s3_uri(built: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    manifest, _ = built
    with pytest.raises(ValueError, match="https URL"):
        pi.fetch(manifest, _fetch_target(tmp_path), names=["tables"], source="s3://bucket/x/")


def test_fetch_cli(built: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    _, out_dir = built
    target = _fetch_target(tmp_path)
    manifest_arg = str(out_dir / "paper_inputs.json")
    argv = ["--data-dir", str(target), "--manifest", manifest_arg]
    code = pi.main(["fetch", *argv, "--groups", "tables,meshes_SID", "--source", str(out_dir)])
    assert code == 0
    assert (target / "all_cell_ids.parquet").exists()
    assert not (target / V4).exists()
    assert pi.main(["verify", *argv]) == 0


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def test_verify_detects_deleted_and_resized_files(
    built: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    manifest, out_dir = built
    target = _fetch_target(tmp_path)
    pi.fetch(manifest, target, source=str(out_dir))
    assert pi.verify(target, manifest) == []

    (target / "all_cell_ids.parquet").unlink()
    problems = pi.verify(target, manifest)
    assert problems == ["tables: missing all_cell_ids.parquet"]

    (target / "structure_data/SID/meshes/mem_mesh_1.obj").write_text("short")
    problems = pi.verify(target, manifest)
    assert len(problems) == 2 and any("meshes_SID: size 5 !=" in p for p in problems)
    argv = ["--data-dir", str(target), "--manifest", str(out_dir / "paper_inputs.json")]
    assert pi.main(["verify", *argv]) == 1


def test_verify_reports_nothing_fetched(tmp_path: Path) -> None:
    problems = pi.verify(tmp_path)
    assert len(problems) == 1 and "run 'fetch' first" in problems[0]


def test_verify_flags_stale_stamp(built: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    manifest, out_dir = built
    target = _fetch_target(tmp_path)
    pi.fetch(manifest, target, names=["tables"], source=str(out_dir))
    newer = copy.deepcopy(manifest)
    newer["groups"][0]["sha256"] = "1" * 64
    assert "stale" in pi.verify(target, newer)[0]


# ---------------------------------------------------------------------------------------------
# table loading and unit check
# ---------------------------------------------------------------------------------------------


def test_load_dataframe_missing_file_names_fetch_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cp_io, "get_datadir_path", lambda: tmp_path)

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("load_dataframe must not read anything remote")

    monkeypatch.setattr(pd, "read_parquet", no_network)
    with pytest.raises(FileNotFoundError) as err:
        cp_io.load_dataframe(prefix="all_cell_ids")
    assert FETCH_CMD in str(err.value)
    assert str(tmp_path / "all_cell_ids.parquet") in str(err.value)
    assert not list(tmp_path.iterdir())  # nothing is written back locally


def test_load_dataframe_reads_local_file_and_rejects_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cp_io, "get_datadir_path", lambda: tmp_path)
    pd.DataFrame({"CellId": [1, 2]}).to_parquet(tmp_path / "all_cell_ids.parquet")
    pd.DataFrame({"CellId": []}).to_parquet(tmp_path / "empty.parquet")
    assert len(cp_io.load_dataframe(prefix="all_cell_ids")) == 2
    with pytest.raises(ValueError, match="empty"):
        cp_io.load_dataframe(prefix="empty")


def _stats(
    radius_pex: float, radius_endo: float, legacy: bool = False, mem: bool = True
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "CellId": [1, 2, 3, 4],
            "structure_name": ["SLC25A17", "SLC25A17", "RAB5A", "RAB5A"],
            "radius": [radius_pex, radius_pex * 1.1, radius_endo, radius_endo * 0.9],
        }
    )
    if legacy:
        df["sphericity"] = 0.9
    if mem:
        df["mem_sphericity"] = 0.8
        df["nuc_sphericity"] = 0.85
    return df


def test_structure_stats_micron_table_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gssd, "load_dataframe", lambda prefix: _stats(0.2588, 0.2822))
    assert len(gssd.get_structure_stats_dataframe()) == 4
    assert len(gssd.get_structure_stats_dataframe("RAB5A")) == 2


@pytest.mark.parametrize(
    "frame",
    [
        _stats(2.378, 2.611, legacy=True, mem=False),  # the legacy pixel file
        _stats(2.378, 0.2822),  # one structure in pixels
        _stats(0.2588, 2.611),
        _stats(0.2588, 0.2822, mem=False),  # micron radii, old column layout
        _stats(0.2588, 0.2822, legacy=True),  # both sphericity layouts
    ],
)
def test_structure_stats_pixel_table_raises(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame
) -> None:
    monkeypatch.setattr(gssd, "load_dataframe", lambda prefix: frame)
    with pytest.raises(ValueError, match="pixel-unit structure_stats; fetch the release tables"):
        gssd.get_structure_stats_dataframe("SLC25A17")


def test_structure_stats_missing_structures_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _stats(0.2588, 0.2822).loc[lambda d: d["structure_name"] == "SLC25A17"]
    monkeypatch.setattr(gssd, "load_dataframe", lambda prefix: frame)
    assert len(gssd.get_structure_stats_dataframe()) == 2
