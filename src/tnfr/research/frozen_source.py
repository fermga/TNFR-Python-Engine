"""Inspect and restore a declared frozen source without executing its code.

Supported adapters are the class nonlinear-readout, comparison, collective
forward and neighbor-forward v1 schemas.
Checks associate retained bytes with a complete Git base; they do not validate
the experiment's mathematical premises, runtime dependencies or chronology.
Other freeze schemas need an explicit adapter, not guessed field semantics.
"""

from __future__ import annotations

import io
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import json_loads
from .artifact_io import (
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)

_RECORD_LIMIT = 32 * 1024**2
_ARCHIVE_LIMIT = 128 * 1024**2
_SCHEMAS = {
    "tnfr.sine-class-nonlinear-readout-freeze.v1": (
        "tnfr.sine-class-nonlinear-readout-protocol.v1",
        "tnfr.sine-class-nonlinear-readout-source-snapshot.v1",
    ),
    "tnfr.sine-class-comparison-freeze.v1": (
        "tnfr.sine-class-comparison-protocol.v1",
        "tnfr.sine-class-comparison-source-snapshot.v1",
    ),
    "tnfr.sine-class-collective-forward-freeze.v1": (
        "tnfr.sine-class-collective-forward-protocol.v1",
        "tnfr.sine-class-collective-forward-source-snapshot.v1",
    ),
    "tnfr.sine-class-neighbor-forward-freeze.v1": (
        "tnfr.sine-class-neighbor-forward-protocol.v1",
        "tnfr.sine-class-neighbor-forward-source-snapshot.v1",
    ),
}


@dataclass(frozen=True)
class FrozenSourceInspection:
    """Byte/base association only; no restored-runtime or scientific verdict."""

    source_base_commit: str
    receipt_path: str
    receipt_sha256: str
    archived_file_count: int
    prior_artifact_count: int
    future_evaluator: str
    existing_outcome_files: tuple[str, ...]


@dataclass(frozen=True)
class _Snapshot:
    inspection: FrozenSourceInspection
    files: dict[str, bytes]
    outcome_paths: tuple[str, ...]


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=root)


def _repository(root: str | Path) -> Path:
    root = Path(root).resolve(strict=True)
    actual = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    if actual.resolve() != root:
        raise ValueError("root must be the Git working-tree root")
    return root


def _local_path(root: Path, name: str) -> Path:
    name = validate_artifact_path(name)
    target = root / name
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink() or parent.resolve() != parent:
            raise ValueError(f"redirected artifact path: {name}")
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"artifact escapes repository: {name}")
    return target


def _receipt_bytes(root: Path, item: dict) -> tuple[str, bytes]:
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
        raise ValueError("invalid artifact receipt")
    name = validate_artifact_path(item["path"])
    size = item["bytes"]
    digest = item["sha256"]
    if type(size) is not int or not 0 <= size <= _ARCHIVE_LIMIT:
        raise ValueError("invalid artifact byte count")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid artifact digest")
    data = read_bytes_bounded(_local_path(root, name), max_bytes=size)
    if len(data) != size or sha256_bytes(data) != digest:
        raise ValueError(f"artifact receipt differs: {name}")
    return name, data


def _normalized(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _runtime_path(name: str) -> bool:
    if name.split("/", 1)[0].casefold() != "src":
        return False
    if not name.startswith("src/"):
        raise ValueError("runtime paths require the canonical src/ spelling")
    return True


def _snapshot(root: Path, receipt_path: str) -> _Snapshot:
    receipt_path = validate_artifact_path(receipt_path)
    if not receipt_path.endswith(".freeze.json"):
        raise ValueError("expected a .freeze.json receipt")
    receipt_data = read_bytes_bounded(
        _local_path(root, receipt_path), max_bytes=_RECORD_LIMIT
    )
    receipt = json_loads(receipt_data)
    if (
        not isinstance(receipt["schema"], str)
        or receipt["schema"] not in _SCHEMAS
        or receipt["evaluation_status_at_freeze"] != "not_evaluated"
        or receipt["runtime_overlays"] != []
    ):
        raise ValueError("unsupported freeze schema or source overlays")
    protocol_schema, source_schema = _SCHEMAS[receipt["schema"]]
    base = receipt["source_base_commit"]
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("expected a full source commit identifier")
    if _git(root, "cat-file", "-t", base).strip() != b"commit":
        raise ValueError("source base is not an available Git commit")
    stem = receipt_path[: -len(".freeze.json")]
    protocol_path, archive_path = stem + ".protocol.json", stem + ".source.zip"
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("freeze inventory must contain protocol and source")
    retained = dict(_receipt_bytes(root, item) for item in artifacts)
    if set(retained) != {protocol_path, archive_path}:
        raise ValueError("freeze inventory differs")
    retained[receipt_path] = receipt_data
    protocol = json_loads(retained[protocol_path])
    if (
        protocol["schema"] != protocol_schema
        or protocol["source_base_commit"] != base
        or protocol["runtime_overlays"] != []
    ):
        raise ValueError("protocol source association differs")
    with zipfile.ZipFile(io.BytesIO(retained[archive_path])) as archive:
        # Admit the manifest size before decompressing it. The shared verifier
        # then checks the complete inventory, expansion cap and member bytes.
        if archive.getinfo("source-manifest.json").file_size > _RECORD_LIMIT:
            raise ValueError("source manifest exceeds the record budget")
        manifest_data = archive.read("source-manifest.json")
        manifest = json_loads(manifest_data)
        if (
            manifest["schema"] != source_schema
            or manifest["source_base_commit"] != base
            or manifest["runtime_overlays"] != []
        ):
            raise ValueError("source manifest association differs")
        entries = manifest["files"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("empty source manifest")
        names = [validate_artifact_path(item["path"]) for item in entries]
        if len(names) != len(set(names)) or "source-manifest.json" in names:
            raise ValueError("duplicate source manifest entry")
        expected = {item["path"]: item["sha256"] for item in entries}
        expected["source-manifest.json"] = sha256_bytes(manifest_data)
        verify_archive_members(io.BytesIO(retained[archive_path]), expected)
        for item in entries:
            name = item["path"]
            data = archive.read(name)
            if type(item["bytes"]) is not int or len(data) != item["bytes"]:
                raise ValueError(f"archived byte count differs: {name}")
            if name in retained and retained[name] != data:
                raise ValueError(f"external and archived bytes differ: {name}")
            if _runtime_path(name):
                committed = _git(root, "show", f"{base}:{name}")
                if (
                    sha256_bytes(committed) != item["git_blob_sha256"]
                    or sha256_bytes(_normalized(data)) != item["normalized_lf_sha256"]
                    or _normalized(data) != _normalized(committed)
                ):
                    raise ValueError(f"runtime snapshot differs from Git base: {name}")
            retained[name] = data
    if protocol_path not in names or not any(_runtime_path(name) for name in names):
        raise ValueError("archive lacks protocol or runtime snapshots")
    evaluator = validate_artifact_path(receipt["future_evaluator"])
    if evaluator not in names or _runtime_path(evaluator):
        raise ValueError("archive lacks the declared supplemental evaluator")
    prior = protocol["source_specification"]["prior_artifact_receipts"]
    if not isinstance(prior, list):
        raise ValueError("invalid prior artifact inventory")
    prior_names = set()
    for item in prior:
        name, data = _receipt_bytes(root, item)
        if _runtime_path(name) or name in prior_names:
            raise ValueError("invalid prior artifact path")
        if name in retained and retained[name] != data:
            raise ValueError(f"conflicting retained bytes: {name}")
        prior_names.add(name)
        retained[name] = data
    # Check portable path aliases across all overlays, not just the ZIP.
    if len({name.casefold() for name in retained}) != len(retained):
        raise ValueError("retained paths collide on a case-insensitive filesystem")
    outcome_paths = (stem + ".attempt.json", stem + ".json")
    if receipt["schema"] in {
        "tnfr.sine-class-comparison-freeze.v1",
        "tnfr.sine-class-collective-forward-freeze.v1",
        "tnfr.sine-class-neighbor-forward-freeze.v1",
    }:
        outcome_paths += (stem + ".export-error.json",)
    if any(name in retained for name in outcome_paths):
        raise ValueError("source supplements cannot install attempt/outcome evidence")
    existing = tuple(name for name in outcome_paths if _local_path(root, name).exists())
    return _Snapshot(
        FrozenSourceInspection(
            base,
            receipt_path,
            sha256_bytes(receipt_data),
            len(entries),
            len(prior),
            evaluator,
            existing,
        ),
        retained,
        outcome_paths,
    )


def inspect_frozen_source(
    root: str | Path, receipt_path: str
) -> FrozenSourceInspection:
    """Read bytes and Git objects; import no archived module and run no producer."""
    return _snapshot(_repository(root), receipt_path).inspection


def restore_frozen_source(
    root: str | Path, receipt_path: str, destination: str | Path
) -> FrozenSourceInspection:
    """Create a new detached full-base worktree and restore admitted supplements.

    Refuse existing destinations and local first-attempt/outcome files. Failed
    preparation leaves its worktree for inspection; it is never erased or
    silently reused. This is not evaluation authorization or a global lock:
    the research gate still selects one workspace, checks its runtime and
    retains the first outcome. No dependency installation or code execution.
    """
    root = _repository(root)
    snapshot = _snapshot(root, receipt_path)
    if snapshot.inspection.existing_outcome_files:
        raise FileExistsError("preserve the existing first attempt or outcome")
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError("restoration requires a new destination")
    if not target.parent.is_dir() or target.parent.resolve() != target.parent:
        raise ValueError("destination needs an existing unredirected parent")
    if target.is_relative_to(root):
        raise ValueError("restore outside the active repository")
    # Hooks are not part of the admitted source-restoration operation.
    with tempfile.TemporaryDirectory(prefix="tnfr-no-hooks-") as hooks:
        _git(
            root,
            "-c",
            f"core.hooksPath={hooks}",
            "worktree",
            "add",
            "--detach",
            str(target),
            snapshot.inspection.source_base_commit,
        )
    for name in snapshot.outcome_paths:
        if _local_path(target, name).exists():
            raise FileExistsError(
                "source base already contains attempt/outcome evidence"
            )
    for name, data in snapshot.files.items():
        path = _local_path(target, name)
        if _runtime_path(name):
            if _normalized(path.read_bytes()) != _normalized(data):
                raise ValueError(f"checked-out runtime differs: {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    base = snapshot.inspection.source_base_commit
    if (
        _git(target, "diff", base, "--name-only", "--", "src").strip()
        or _git(
            target, "ls-files", "--others", "--exclude-standard", "--", "src"
        ).strip()
    ):
        raise ValueError("restored runtime contains changes outside the pinned base")
    restored = _snapshot(target, receipt_path).inspection
    if restored != snapshot.inspection:
        raise ValueError("restored source association differs")
    return restored
