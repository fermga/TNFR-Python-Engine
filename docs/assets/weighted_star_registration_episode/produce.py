"""Explicit, single-attempt transport for the frozen registration study.

This script does not run on import. ``prepare`` records the supplied source
and freezes its implementation; ``evaluate`` consumes that freeze once. Byte
receipts associate retained contents, not execution authenticity. Scientific
admission, certificates and interpretation belong to the shared research owner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MAX_BYTES = 256 * 1024**2
FROZEN_NAMES = (
    "protocol.json",
    "preparation.json",
    "source-manifest.json",
    "source.zip",
)
BRANCHES = ("nonlinear", "tangent")
SOURCE_OWNERS = (
    "AGENTS.md",
    ".github/agents/my-agent.md",
    "pyproject.toml",
    "theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md",
    "theory/nodal/SINE_WEIGHTED_STAR_FINITE_IDENTITY.md",
    "tests/operators/test_weighted_star_canonical_contact.py",
    "tests/research/test_weighted_star_episode_evidence.py",
    "tests/research/test_weighted_star_episode_summary.py",
    "tests/research/test_weighted_star_finite_episode.py",
    "tests/mathematics/test_validated_metric_taylor.py",
    "tests/mathematics/test_metric_taylor_observation_boxes.py",
    "docs/assets/weighted_star_registration_episode/produce.py",
)

sys.path.insert(0, str(ROOT / "src"))

from tnfr.research.artifact_io import (  # noqa: E402
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
    write_json_once,
)
from tnfr.utils.io import json_dumps, json_loads  # noqa: E402


def _owner():
    return import_module("tnfr.research.weighted_star_finite_episode")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _runtime():
    versions = {}
    for name in ("numpy", "networkx", "mpmath"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "versions": versions,
    }


def _path(base, name):
    name = validate_artifact_path(name)
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("artifact resolves outside its declared root")
    return path


def _write(name, value):
    write_json_once(_path(OUT, name), value, base_dir=OUT)


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(_path(OUT, name), max_bytes=MAX_BYTES))
    )


def _receipt(base, name):
    return file_receipt(_path(base, name), label=name, max_bytes=MAX_BYTES)


def _same_protocol(left, right):
    # Text comparison preserves scalar types: Python would equate True and 1.
    return json_dumps(encode_exact_tree(left), sort_keys=True) == json_dumps(
        encode_exact_tree(right), sort_keys=True
    )


def _verify_receipts(base, rows):
    names = []
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid artifact receipt fields")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("invalid artifact byte count")
        name = validate_artifact_path(row["path"])
        names.append(name)
        if _receipt(base, name) != row:
            raise ValueError(f"frozen content differs: {name}")
    if len(names) != len(set(name.casefold() for name in names)):
        raise ValueError("duplicate artifact receipt")
    return tuple(names)


def _source_paths():
    paths = set((ROOT / "src/tnfr").rglob("*.py"))
    paths.update(ROOT / name for name in SOURCE_OWNERS)
    for pattern in (
        "tests/research/test_weighted_star_episode*.py",
        "tests/mathematics/test_validated_taylor*.py",
    ):
        paths.update(ROOT.glob(pattern))
    return tuple(sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix()))


def _archive_sources():
    # Read each member once: its archive entry and digest refer to these exact
    # same bytes. A subsequent live-file check detects preparation-time edits.
    payloads = tuple(
        (
            validate_artifact_path(path.relative_to(ROOT).as_posix()),
            read_bytes_bounded(
                _path(ROOT, path.relative_to(ROOT).as_posix()), max_bytes=MAX_BYTES
            ),
        )
        for path in _source_paths()
    )
    if sum(len(data) for _, data in payloads) > MAX_BYTES:
        raise ValueError("source inventory exceeds expanded archive budget")
    rows = tuple(
        {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}
        for name, data in payloads
    )
    with ZipFile(OUT / "source.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, data in payloads:
            archive.writestr(name, data)
    _write("source-manifest.json", {"schema": "tnfr.source-manifest.v1", "files": rows})
    verify_archive_members(
        OUT / "source.zip",
        {row["path"]: row["sha256"] for row in rows},
        max_bytes=MAX_BYTES,
    )
    _verify_receipts(ROOT, rows)
    return rows


def prepare():
    """Create one preparation and freeze; never evaluate its continuation."""
    OUT.mkdir(parents=True, exist_ok=True)
    unexpected = [
        path.name
        for path in OUT.iterdir()
        if path.name not in {"produce.py", "__pycache__"}
    ]
    if unexpected:
        raise FileExistsError(
            "existing preparation or attempt prevents another prepare"
        )
    owner = _owner()
    # Exclusive protocol creation also blocks an unrecorded retry after a
    # partial source preparation. It is not a reserved trajectory response.
    _write("protocol.json", owner.frozen_protocol())
    try:
        rows = _archive_sources()
        _write("preparation.json", owner.prepare_source())
        _verify_receipts(ROOT, rows)
        if not _same_protocol(_read("protocol.json"), owner.frozen_protocol()):
            raise ValueError("protocol changed during source preparation")
        _write(
            "freeze.json",
            {
                "schema": "tnfr.weighted-star-episode-freeze.v1",
                "created_utc": _utc(),
                "evaluation_status_at_freeze": "not_evaluated",
                "artifacts": tuple(_receipt(OUT, name) for name in FROZEN_NAMES),
                "runtime": _runtime(),
                "archive_scope": (
                    "all src/tnfr Python plus declared instrument, proof, instructions, "
                    "tests and project metadata; not dependency binaries or "
                    "authentication of execution"
                ),
            },
        )
    except BaseException as exc:
        _write(
            "preparation-failure.json",
            {"created_utc": _utc(), "failure": _failure(exc)},
        )
        raise
    print("Prepared and frozen; no trajectory evaluated.", flush=True)
    return 0


def _verify_frozen_inputs():
    if any(
        (OUT / name).exists()
        for name in ("attempt.json", "response.json", "result.json", "records")
    ):
        raise FileExistsError(
            "a retained attempt or response prevents another evaluation"
        )
    frozen = _read("freeze.json")
    if (
        frozen.get("schema") != "tnfr.weighted-star-episode-freeze.v1"
        or frozen.get("evaluation_status_at_freeze") != "not_evaluated"
    ):
        raise ValueError("missing unevaluated source freeze")
    if not _same_protocol(frozen["runtime"], _runtime()):
        raise ValueError("runtime differs from the frozen preparation")
    names = _verify_receipts(OUT, frozen["artifacts"])
    if set(names) != set(FROZEN_NAMES):
        raise ValueError("frozen artifact inventory differs")
    rows = _read("source-manifest.json")["files"]
    _verify_receipts(ROOT, rows)
    expected = {row["path"]: row["sha256"] for row in rows}
    if set(expected) != {path.relative_to(ROOT).as_posix() for path in _source_paths()}:
        raise ValueError("live source inventory differs from its freeze")
    verify_archive_members(OUT / "source.zip", expected, max_bytes=MAX_BYTES)
    with ZipFile(OUT / "source.zip") as archive:
        if {info.filename: info.file_size for info in archive.infolist()} != {
            row["path"]: row["bytes"] for row in rows
        }:
            raise ValueError("source archive byte inventory differs")
    owner = _owner()
    protocol = _read("protocol.json")
    if not _same_protocol(protocol, owner.frozen_protocol()):
        raise ValueError("frozen protocol differs from the declared shared owner")
    return owner, _read("preparation.json"), protocol


def _failure(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def evaluate():
    """Consume the frozen source once; retain raw prefixes even on failure."""
    owner, preparation, protocol = _verify_frozen_inputs()
    _write(
        "attempt.json",
        {
            "schema": "tnfr.weighted-star-episode-attempt.v1",
            "created_utc": _utc(),
            "attempt_number": 1,
            "evaluation_status_at_start": "not_evaluated",
            "freeze": _receipt(OUT, "freeze.json"),
            "runtime": _runtime(),
        },
    )
    started = perf_counter()
    failure = summary = None

    def progress(branch, index, record):
        if type(branch) is not str or branch not in BRANCHES:
            raise ValueError("unexpected progress branch")
        if type(index) is not int or not 0 <= index <= 128:
            raise ValueError("unexpected progress record index")
        if type(record) is not dict:
            raise TypeError("progress record must be a detached record mapping")
        _write(f"records/{branch}-{index:04d}.json", record)

    try:
        (OUT / "records").mkdir(exist_ok=False)
        response = owner.evaluate_protocol(preparation, protocol, progress=progress)
        if type(response) is not dict:
            raise TypeError("scientific response must be a record mapping")
        _write("response.json", response)
        # Interpret the durable serialized response, never only a cached flag.
        summary = owner.summarize_response(_read("response.json"), protocol)
        if type(summary) is not dict:
            raise TypeError("scientific summary must be a record mapping")
    except BaseException as exc:
        failure = _failure(exc)
    records = tuple(
        path.relative_to(OUT).as_posix()
        for path in sorted((OUT / "records").glob("*"))
        if path.is_file()
    )
    artifacts = records + (
        ("response.json",) if (OUT / "response.json").exists() else ()
    )
    _write(
        "result.json",
        {
            "schema": "tnfr.weighted-star-episode-result.v1",
            "created_utc": _utc(),
            "status": "evaluated" if failure is None else "failed",
            "attempt": _receipt(OUT, "attempt.json"),
            "artifacts": tuple(_receipt(OUT, name) for name in artifacts),
            "elapsed_seconds": perf_counter() - started,
            "scientific_summary": summary,
            "failure": failure,
            "status_scope": "transport completion is separate from scientific verdicts",
        },
    )
    print(
        "Evaluated." if failure is None else "Failed; retained prefix preserved.",
        flush=True,
    )
    return int(failure is not None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    args = parser.parse_args(argv)
    return {"prepare": prepare, "evaluate": evaluate}[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
