"""Freeze, inspect or evaluate the single reserved mobile-pair endpoint.

Freeze and audit perform no scientific response calculation. Evaluation keeps
the first attempt and outcome, including failure, and never retries. Byte
receipts establish source association, not source or execution authentication.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import platform
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research.artifact_io import (  # noqa: E402
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
    write_json_once,
)
from tnfr.utils.io import json_dumps, json_loads, safe_write  # noqa: E402

PROTOCOL_REVISION = "14cbad44f92194774c4cd66a8763f2069c95270c"
PROOF = "theory/nodal/LOCALIZED_MOBILE_INTERACTION.md"
RESPONSE_PROOF = "theory/nodal/LOCALIZED_MOBILE_RESPONSE.md"
DRIVER = "scripts/run_localized_mobile_response.py"
RESPONSE_SOURCE = "src/tnfr/physics/_localized_mobile_response.py"
SUPPORTING_TESTS = (
    "tests/scripts/test_localized_mobile_response_lifecycle.py",
    "tests/physics/test_localized_mobile_response.py",
    "tests/physics/test_localized_mobile_spectral.py",
    "tests/physics/test_localized_mobile_interaction.py",
    "tests/physics/test_localized_identity_transport.py",
    "tests/test_gauss_legendre.py",
    "tests/test_rational_interval.py",
)
EVIDENCE = "docs/assets/localized_mobile_interaction"
MAX_SOURCE_BYTES = 128 * 1024**2
MAX_RECORD_BYTES = 32 * 1024**2
OUTCOMES = ("attempt.json", "report.json", "failure.json", "export-error.json")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git(root, *arguments):
    return subprocess.check_output(("git", *arguments), cwd=root)


def _json_bytes(payload):
    return (
        json_dumps(encode_exact_tree(payload), sort_keys=True, indent=2) + "\n"
    ).encode()


def _policy():
    """The one frozen source, clock, observation and numerical budget."""
    amplitude = Q(1, 2**12)
    return {
        "model": {
            "epi_weight": Q(1, 2**20),
            "phase_weight": 1 - Q(1, 2**20),
            "storage_scale": 1,
            "phase_domain": "regular",
        },
        "arguments": {
            "node_count": 1024,
            "fejer_order": 16,
            "amplitude": amplitude,
            "initial_state_error_bound": amplitude / 1000,
            "localization_radius": 96,
            "horizon": 256,
            "quadrature_order": 12,
            "max_panels": 65536,
        },
        "numerical_target": amplitude**3 / 1000,
        "quadrature_selection": "deterministic doubling from one using only the error bound",
        "quadrature_error_allowance": "numerical_target/10",
        "arithmetic": "shared 128-bit outward rational intervals",
        "node_order": tuple(range(1024)),
        "support": "fixed cycle C_1024; unit conductance, length and held capacity",
        "clock": "tau=exchange*t/pi in both complete evolution rows",
        "input_and_events": "none",
        "preparation": "opposite Fejer carriers at +128 and -128; AB=A+B exactly",
        "preparation_errors": "two joint error balls reused in all three histories",
        "pressure": "actual normalized-sine complete law; actual damped tangent proxy",
        "observation": "full-state joint-minus-separate norm at tau=256",
        "seed": None,
        "maximum_evaluations": 1,
        "failure_policy": "retain first outcome; no parameter change or retry",
    }


def _runtime():
    from tnfr.mathematics._rational_interval import INTERVAL_BITS, INTERVAL_METHOD

    if INTERVAL_BITS != 128:
        raise ValueError("the frozen arithmetic policy requires 128-bit intervals")
    packages = {}
    for name in ("tnfr", "numpy", "scipy", "networkx", "cachetools", "mpmath"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "interval_bits": INTERVAL_BITS,
        "interval_method": INTERVAL_METHOD,
    }


def _imported_sources(root):
    """Require loaded TNFR modules to come from the captured source tree."""
    sources = {}
    for name, module in tuple(sys.modules.items()):
        if name == "tnfr" or name.startswith("tnfr."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(root / "src"):
                    raise ValueError(
                        f"loaded TNFR module is outside the frozen tree: {name}"
                    )
                sources[name] = path.relative_to(root).as_posix()
    return sources


def _local(root, name):
    validate_artifact_path(name)
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("artifact path is redirected outside its declared source")
    return path


def _source_files(root):
    """Capture all current TNFR source bytes, excluding Python bytecode caches."""
    paths = []
    for path in (root / "src").rglob("*"):
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise ValueError("source snapshots require ordinary files")
        if path.is_file():
            paths.append(path.relative_to(root).as_posix())
    paths.extend((DRIVER, "pyproject.toml", PROOF, RESPONSE_PROOF, *SUPPORTING_TESTS))
    files = {}
    remaining = MAX_SOURCE_BYTES
    for name in sorted(set(paths)):
        data = read_bytes_bounded(_local(root, name), max_bytes=remaining)
        remaining -= len(data)
        files[name] = data
    if RESPONSE_SOURCE not in files:
        raise ValueError("the complete source requires the response implementation")
    return files


def _receipt(name, data):
    return {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}


def _read_record(path):
    return json_loads(read_bytes_bounded(path, max_bytes=MAX_RECORD_BYTES))


def _write_bytes_once(path, data):
    safe_write(
        path,
        lambda stream: stream.write(data),
        mode="xb",
        atomic=False,
        sync=True,
    )


def freeze(root=ROOT):
    """Create the whole-source freeze without importing the response owner."""
    root = Path(root).resolve()
    output = _local(root, EVIDENCE)
    if output.exists():
        raise FileExistsError("retain the existing or partial mobile response freeze")
    files = _source_files(root)
    imported = _imported_sources(root)
    proof = _git(root, "show", f"{PROTOCOL_REVISION}:{PROOF}")
    runtime = _runtime()
    protocol = {
        "schema": "tnfr.localized-mobile-response-protocol.v1",
        "created_utc": _now(),
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_proof": _receipt("protocol-proof.md", proof),
        "policy": _policy(),
        "runtime": runtime,
        "evaluation_status_at_freeze": "not_evaluated",
    }
    manifest = {
        "schema": "tnfr.localized-mobile-response-source.v1",
        "source_git_head": _git(root, "rev-parse", "HEAD").decode().strip(),
        "source_git_status": _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "src",
            DRIVER,
            "pyproject.toml",
            PROOF,
            RESPONSE_PROOF,
            *SUPPORTING_TESTS,
        ).decode(),
        "source_scope": "all current src files, driver, pyproject, current proofs and supporting tests",
        "excluded": ("__pycache__ directories", ".pyc and .pyo files"),
        "imported_sources_at_freeze": imported,
        "files": tuple(_receipt(name, data) for name, data in files.items()),
        "association_only_not_authentication": True,
    }
    members = dict(files)
    members.update(
        {
            "protocol-proof.md": proof,
            "protocol.json": _json_bytes(protocol),
            "source-manifest.json": _json_bytes(manifest),
        }
    )
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    packed = archive_bytes.getvalue()
    verify_archive_members(
        io.BytesIO(packed),
        {name: sha256_bytes(data) for name, data in members.items()},
        max_bytes=MAX_SOURCE_BYTES + MAX_RECORD_BYTES,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for name in ("protocol-proof.md", "protocol.json", "source-manifest.json"):
        _write_bytes_once(output / name, members[name])
    _write_bytes_once(output / "source.zip", packed)
    if _source_files(root) != files or _runtime() != runtime:
        raise RuntimeError(
            "source or runtime changed while freezing; preserve partial freeze"
        )
    write_json_once(
        output / "freeze.json",
        {
            "schema": "tnfr.localized-mobile-response-freeze.v1",
            "created_utc": _now(),
            "evaluation_status_at_freeze": "not_evaluated",
            "artifacts": tuple(
                file_receipt(output / name, label=name)
                for name in (
                    "protocol-proof.md",
                    "protocol.json",
                    "source-manifest.json",
                    "source.zip",
                )
            ),
            "archive_members": tuple(
                _receipt(name, data) for name, data in members.items()
            ),
            "association_only_not_authentication": True,
        },
    )
    return audit(root)


def _check_receipt(output, receipt):
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"path", "bytes", "sha256"}
        or type(receipt["bytes"]) is not int
        or receipt["bytes"] < 0
    ):
        raise ValueError("invalid artifact receipt")
    actual = file_receipt(
        _local(output, receipt["path"]),
        label=receipt["path"],
        max_bytes=MAX_SOURCE_BYTES + MAX_RECORD_BYTES,
    )
    if actual != receipt:
        raise ValueError("artifact bytes differ from their freeze receipt")


def audit(root=ROOT, *, require_current=False):
    """Verify the declared byte associations without replaying any computation."""
    root = Path(root).resolve()
    output = _local(root, EVIDENCE)
    receipt = _read_record(output / "freeze.json")
    if (
        receipt["schema"] != "tnfr.localized-mobile-response-freeze.v1"
        or receipt["evaluation_status_at_freeze"] != "not_evaluated"
    ):
        raise ValueError("unsupported mobile response freeze")
    expected = {
        "protocol-proof.md",
        "protocol.json",
        "source-manifest.json",
        "source.zip",
    }
    artifacts = receipt["artifacts"]
    if (
        len(artifacts) != len(expected)
        or {item["path"] for item in artifacts} != expected
    ):
        raise ValueError("freeze artifact inventory differs")
    for item in artifacts:
        _check_receipt(output, item)
    protocol, manifest = (
        _read_record(output / name)
        for name in (
            "protocol.json",
            "source-manifest.json",
        )
    )
    proof = _git(root, "show", f"{PROTOCOL_REVISION}:{PROOF}")
    if (
        protocol["schema"] != "tnfr.localized-mobile-response-protocol.v1"
        or manifest["schema"] != "tnfr.localized-mobile-response-source.v1"
        or protocol["protocol_revision"] != PROTOCOL_REVISION
        or protocol["protocol_proof"] != _receipt("protocol-proof.md", proof)
        or _json_bytes(protocol["policy"]) != _json_bytes(_policy())
        or protocol["evaluation_status_at_freeze"] != "not_evaluated"
    ):
        raise ValueError("the frozen scientific protocol differs")
    members = receipt["archive_members"]
    hashes = {item["path"]: item["sha256"] for item in members}
    if len(hashes) != len(members):
        raise ValueError("duplicate archive member receipt")
    source_hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    if len(source_hashes) != len(manifest["files"]):
        raise ValueError("duplicate source member receipt")
    external = {
        name: file_receipt(output / name, label=name)["sha256"]
        for name in (
            "protocol-proof.md",
            "protocol.json",
            "source-manifest.json",
        )
    }
    if hashes != source_hashes | external:
        raise ValueError("source manifest and archive inventory differ")
    verify_archive_members(
        output / "source.zip", hashes, max_bytes=MAX_SOURCE_BYTES + MAX_RECORD_BYTES
    )
    if require_current:
        actual = {
            name: sha256_bytes(data) for name, data in _source_files(root).items()
        }
        if actual != source_hashes or _runtime() != protocol["runtime"]:
            raise ValueError(
                "current complete source or runtime differs from the freeze"
            )
    for name in ("report.json", "failure.json", "export-error.json"):
        if (output / name).exists():
            outcome = _read_record(output / name)
            if (
                outcome["attempt"]["path"] != "attempt.json"
                or outcome["freeze"]["path"] != "freeze.json"
            ):
                raise ValueError("outcome names a different attempt or freeze")
            _check_receipt(output, outcome["attempt"])
            _check_receipt(output, outcome["freeze"])
    return {
        "status": "byte_associations_verified",
        "source_file_count": len(source_hashes),
        "existing_outcomes": tuple(
            name for name in OUTCOMES if (output / name).exists()
        ),
        "source_or_execution_authenticated": False,
    }


def _evaluate_response(progress):
    from tnfr.dynamics.relational import RelationalExchangeModel
    from tnfr.physics._localized_mobile_response import (
        _evaluate_localized_mobile_response,
    )

    policy = _policy()
    return _evaluate_localized_mobile_response(
        reference_model=RelationalExchangeModel(**policy["model"]),
        **policy["arguments"],
        progress=progress,
    )


def evaluate(root=ROOT):
    """Attempt the one endpoint exactly once, after the complete freeze audit."""
    root = Path(root).resolve()
    output = _local(root, EVIDENCE)
    if any((output / name).exists() for name in OUTCOMES):
        raise FileExistsError("preserve the first mobile response attempt and outcome")
    audit(root, require_current=True)
    _imported_sources(root)
    freeze_receipt = file_receipt(output / "freeze.json", label="freeze.json")
    write_json_once(
        output / "attempt.json",
        {
            "schema": "tnfr.localized-mobile-response-attempt.v1",
            "created_utc": _now(),
            "freeze": freeze_receipt,
            "evaluation_count": 1,
            "evaluation_status_at_start": "not_evaluated",
        },
    )
    association = {
        "attempt": file_receipt(output / "attempt.json", label="attempt.json"),
        "freeze": freeze_receipt,
    }
    count = 0

    def progress(record):
        nonlocal count
        count += 1
        if count == 1 or count % 8 == 0:
            print(json_dumps(encode_exact_tree({"progress": record})), flush=True)

    started = time.perf_counter()
    try:
        report = _evaluate_response(progress)
        audit(root, require_current=True)
        imported = _imported_sources(root)
        encoded = encode_exact_tree(asdict(report))
    except BaseException as error:
        write_json_once(
            output / "failure.json",
            {
                "schema": "tnfr.localized-mobile-response-failure.v1",
                "created_utc": _now(),
                **association,
                "elapsed_seconds": time.perf_counter() - started,
                "exception": type(error).__name__,
                "message": str(error),
                "no_retry": True,
            },
        )
        raise
    try:
        write_json_once(
            output / "report.json",
            {
                "schema": "tnfr.localized-mobile-response-outcome.v1",
                "created_utc": _now(),
                **association,
                "elapsed_seconds": time.perf_counter() - started,
                "report": encoded,
                "imported_sources_at_completion": imported,
                "source_or_execution_authenticated": False,
            },
        )
    except BaseException as error:
        write_json_once(
            output / "export-error.json",
            {
                "schema": "tnfr.localized-mobile-response-export-error.v1",
                "created_utc": _now(),
                **association,
                "exception": type(error).__name__,
                "message": str(error),
                "no_retry": True,
            },
        )
        raise
    return {"status": "first_report_retained", "path": str(output / "report.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    arguments = parser.parse_args(argv)
    run = freeze if arguments.freeze else audit if arguments.audit else evaluate
    print(json_dumps(encode_exact_tree(run()), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
