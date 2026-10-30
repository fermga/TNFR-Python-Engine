"""One fixed source-aware reference attempt with complete streamed evidence.

Importing this module performs no target admission, field or propagation.
The worker entry point must be launched by the reviewed hard supervisor.
This is a conditional mathematical certificate, not physical identification.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import asdict, fields
from fractions import Fraction as Q
from pathlib import Path

from ..utils.io import json_dumps, json_loads, safe_write
from .artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)

STEP_COUNT = 4096
STEP_SIZE = Q(1, 4)
ORDER = 8
STEP_BYTES = 256 * 1024**2
AUX_BYTES = 64 * 1024**2
SUPERVISOR_RESERVE = 2 * 1024**2
CALLER_RESERVE = 64 * 1024
TERMINAL_RESERVE = 1024**2
PROTOCOL_SCHEMA = "tnfr.formed-two-port-executable-protocol.v1"
_REQUIRED_SOURCES = {
    "src/tnfr/research/formed_two_port_reference.py",
    "src/tnfr/research/formed_two_port_run.py",
    "src/tnfr/research/bounded_worker.py",
    "theory/nodal/SINE_TWO_PORT_CAPTURE.md",
    "theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md",
    "theory/nodal/SINE_FORMED_CLASS_CONTACT.md",
    "theory/RESEARCH_PROGRAM.md",
    "theory/research_prerequisites.json",
}
_TARGET_EVIDENCE = {
    f"docs/assets/sine_formed_classes/two-port-compatibility-v1{suffix}"
    for suffix in (".json", ".manifest.json", ".protocol.json", ".source.zip")
}
_REQUIRED_SOURCES |= _TARGET_EVIDENCE
_SHARED_FIELDS = {"metric", "coordinate_projection_factors"}
_CERTIFICATE_METADATA = {
    "method",
    "conditional_premises",
    "growth_certificate",
    "prescribed_time_interval",
}


def _exact_math_tree(value):
    """Reject coerced scalars throughout the projected mathematical fields."""
    if isinstance(value, dict):
        for item in value.values():
            _exact_math_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _exact_math_tree(item)
    else:
        exact_record(value)


def declared_policy() -> dict:
    """The selected policy has no caller-controlled scientific parameters."""
    return {
        "reference_dimension": 18,
        "actual_dimension": 36,
        "source_phi": Q(1, 1000),
        "source_epsilon": Q(1, 10**32),
        "contact_tau": 100 + 10**13,
        "contact_edges": ((4, 13), (5, 14)),
        "step_count": STEP_COUNT,
        "step_size": STEP_SIZE,
        "order": ORDER,
        "picard_max_iterations": 16,
        "reference_horizon": 1024,
        "analytical_tail": 1,
        "reference_margin": Q(1, 2048),
        "arithmetic": "rational_outward_dyadic128_machin_trigonometry_v1",
        "growth_rate": 0,
        "wall_seconds": 3600,
        "process_memory_bytes": 1024**3,
        "memory_metric": "Windows Job process and job committed virtual memory",
        "step_bytes": STEP_BYTES,
        "auxiliary_bytes": AUX_BYTES,
        "supervisor_reserved_bytes": SUPERVISOR_RESERVE,
        "caller_reserved_bytes": CALLER_RESERVE,
        "workers": 1,
        "attempts": 1,
        "resume": False,
        "new_root_search": False,
    }


def _json_bytes(value) -> bytes:
    """Detach first, then serialize once with the shared exact JSON contract."""
    return (
        json_dumps(
            encode_exact_tree(value), sort_keys=True, indent=None, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _contained(root: Path, name: str) -> Path:
    name = validate_artifact_path(name)
    path = root / name
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink() or parent.resolve() != parent:
            raise ValueError("redirected evidence path")
    if not path.resolve().is_relative_to(root):
        raise ValueError("evidence path escapes root")
    return path


class EvidenceBudgetExceeded(ValueError):
    """A fixed artifact quota was reached; this attempt cannot retry."""


class EvidenceWriter:
    """Bound immutable bytes, keeping one complete certificate at a time.

    The supervisor owns at most two MiB, including its at-most-one-MiB log.
    A separate 64-KiB caller allowance is unavailable to this child writer.
    Existing auxiliary files count conservatively as well. A final one-MiB
    child allowance remains available for failure metadata after regular writes.
    The attempt directory is private; external writes are not an admitted input.
    """

    def __init__(self, directory: Path):
        self.root = Path(directory).resolve(strict=True)
        if (self.root / "worker-start.json").exists() or (self.root / "steps").exists():
            raise FileExistsError("reference attempt already started; no resume")
        self.step_used = 0
        self.aux_used = 0
        self.terminal_written = False
        for path in self.root.rglob("*"):
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("redirected attempt content")
            if path.is_file():
                self.aux_used += path.stat().st_size
        if (
            self.aux_used
            > AUX_BYTES - SUPERVISOR_RESERVE - CALLER_RESERVE - TERMINAL_RESERVE
        ):
            raise EvidenceBudgetExceeded("initial auxiliary evidence exceeds budget")
        (self.root / "steps").mkdir()

    def _write(self, name: str, data: bytes, *, step: bool, terminal: bool) -> dict:
        if type(data) is not bytes:
            raise TypeError("prepared evidence must be immutable bytes")
        if terminal and (step or self.terminal_written):
            raise ValueError("only one auxiliary terminal record is admitted")
        if step != name.startswith("steps/"):
            raise ValueError("evidence category differs from its path")
        path = _contained(self.root, name)
        allowance = (
            STEP_BYTES if step else AUX_BYTES - SUPERVISOR_RESERVE - CALLER_RESERVE
        )
        if not step and not terminal:
            allowance -= TERMINAL_RESERVE
        used = self.step_used if step else self.aux_used
        if used + len(data) > allowance:
            raise EvidenceBudgetExceeded("fixed evidence byte allowance exceeded")
        if terminal and len(data) > TERMINAL_RESERVE:
            raise EvidenceBudgetExceeded("terminal record exceeds reserved allowance")
        # Reserve bytes before opening: partial writes remain counted and immutable.
        if step:
            self.step_used += len(data)
        else:
            self.aux_used += len(data)
        if terminal:
            self.terminal_written = True
        safe_write(
            path,
            lambda stream: stream.write(data),
            mode="xb",
            atomic=False,
            sync=True,
            base_dir=self.root,
        )
        row = file_receipt(path, label=name, max_bytes=len(data))
        if row["bytes"] != len(data) or row["sha256"] != sha256_bytes(data):
            raise ValueError("written evidence differs from prepared bytes")
        return row

    def write(self, name: str, payload, *, step=False, terminal=False) -> dict:
        return self._write(name, _json_bytes(payload), step=step, terminal=terminal)

    def copy(self, name: str, data: bytes) -> dict:
        return self._write(name, data, step=False, terminal=False)


def _receipt(root: Path, row: dict) -> Path:
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
        raise ValueError("invalid source receipt")
    if type(row["bytes"]) is not int or not 0 <= row["bytes"] <= AUX_BYTES:
        raise ValueError("invalid source size")
    if not isinstance(row["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", row["sha256"]
    ):
        raise ValueError("invalid source digest")
    path = _contained(root, row["path"])
    if file_receipt(path, label=row["path"], max_bytes=row["bytes"]) != row:
        raise ValueError("frozen source receipt differs")
    return path


def _receipt_data(root: Path, row: dict) -> bytes:
    """Bind the bytes actually copied or parsed, not a preceding file read."""
    path = _receipt(root, row)
    data = read_bytes_bounded(path, max_bytes=row["bytes"])
    if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
        raise ValueError("consumed frozen bytes differ")
    return data


def admit_execution_inputs(
    root: Path, protocol_path: Path, writer: EvidenceWriter
) -> dict:
    """Re-admit source/runtime associations inside the bounded worker.

    This verifies content, not independent review authorship or chronology.
    The substantive reviewed source freeze is an external execution prerequisite.
    """
    root = Path(root).resolve(strict=True)
    path = Path(protocol_path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("protocol must belong to the selected repository")
    raw = read_bytes_bounded(path, max_bytes=1024**2)
    record = json_loads(raw)
    if not isinstance(record, dict) or set(record) != {
        "schema",
        "policy",
        "head",
        "runtime",
        "sources",
        "archive",
        "reviews",
        "controls",
        "worker_entry",
    }:
        raise ValueError("invalid executable protocol shape")
    if record["schema"] != PROTOCOL_SCHEMA or _json_bytes(
        record["policy"]
    ) != _json_bytes(declared_policy()):
        raise ValueError("protocol differs from the selected fixed policy")
    if not isinstance(record["head"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", record["head"]
    ):
        raise ValueError("invalid frozen Git base")
    expected_runtime = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": sys.version,
        "package_root": str((root / "src/tnfr").resolve()),
    }
    if record["runtime"] != expected_runtime:
        raise ValueError("worker runtime differs from reviewed interpreter/source")
    import tnfr

    if not Path(tnfr.__file__).resolve().is_relative_to(root / "src/tnfr"):
        raise ValueError("worker imported a different TNFR package")
    rows = record["sources"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("invalid frozen inventory")
    names = []
    for row in rows:
        _receipt(root, row)
        names.append(row["path"])
    if len(names) != len(set(names)) or not _REQUIRED_SOURCES <= set(names):
        raise ValueError("incomplete or duplicate frozen inventory")
    source_rows = {row["path"]: row for row in rows}
    entry = _receipt(root, record["worker_entry"])
    if (
        source_rows.get(record["worker_entry"]["path"]) != record["worker_entry"]
        or entry != Path(sys.argv[0]).resolve()
    ):
        raise ValueError("executed wrapper is not the frozen worker entry")
    attempt = json_loads(
        read_bytes_bounded(writer.root / "attempt.json", max_bytes=512 * 1024)
    )
    expected_arguments = [
        "--root",
        str(root),
        "--protocol",
        str(path),
        "--output",
        str(writer.root),
    ]
    if (
        type(attempt.get("wall_seconds")) is not int
        or attempt["wall_seconds"] != 3600
        or type(attempt.get("process_memory_bytes")) is not int
        or attempt["process_memory_bytes"] != 1024**3
        or type(attempt.get("log_limit_bytes")) is not int
        or attempt["log_limit_bytes"] != 1024**2
        or attempt.get("one_worker") is not True
        or attempt.get("script") != str(entry)
        or attempt.get("cwd") != str(root)
        or attempt.get("arguments") != expected_arguments
        or Path(attempt.get("interpreter", "")).resolve()
        != Path(sys.executable).resolve()
        or Path(attempt.get("prefix", "")).resolve() != Path(sys.prefix).resolve()
    ):
        raise ValueError("supervisor declaration differs from the fixed attempt")
    all_python = {
        p.relative_to(root).as_posix() for p in (root / "src/tnfr").rglob("*.py")
    }
    if not all_python <= set(names):
        raise ValueError("frozen inventory does not cover the installed TNFR source")
    archive_data = _receipt_data(root, record["archive"])
    verify_archive_members(
        io.BytesIO(archive_data),
        {row["path"]: row["sha256"] for row in rows},
        max_bytes=AUX_BYTES,
    )
    reviews = record["reviews"]
    if not isinstance(reviews, list) or len(reviews) < 2:
        raise ValueError("missing independent reviewed-byte associations")
    review_data = [_receipt_data(root, row) for row in reviews]
    if len({row["path"] for row in reviews}) != len(reviews):
        raise ValueError("duplicate independent review association")
    controls_data = _receipt_data(root, record["controls"])
    control_record = json_loads(controls_data)
    if (
        control_record.get("passed") is not True
        or type(control_record.get("reserved_reference_steps")) is not int
        or control_record["reserved_reference_steps"] != 0
    ):
        raise ValueError("non-transit controls not passed")
    protocol_receipt = writer.copy("protocol.json", raw)
    archive_receipt = writer.copy("source.zip", archive_data)
    for index, data in enumerate(review_data):
        writer.copy(f"review-{index:02d}.md", data)
    writer.copy("controls.json", controls_data)
    return {
        "protocol": protocol_receipt,
        "archive": archive_receipt,
        "head": record["head"],
        "worker_entry": record["worker_entry"],
        "target_evidence": [source_rows[name] for name in sorted(_TARGET_EVIDENCE)],
    }


def shared_certificate_record(certificate, reference) -> dict:
    """Share only exactly equal constants; retain all other fields per step."""
    metric = tuple(tuple(exact_record(v) for v in row) for row in certificate.metric)
    projection = tuple(
        exact_record(v) for v in certificate.coordinate_projection_factors
    )
    if metric != reference.metric or len(projection) != 18 or min(projection) <= 0:
        raise ValueError("invalid shared metric/projection")
    return {
        "schema": "tnfr.formed-two-port-certificate-fields.v1",
        "fields": {"metric": metric, "coordinate_projection_factors": projection},
    }


def certificate_record(
    certificate,
    reference,
    index: int,
    center,
    radius,
    source: dict,
    previous: dict | None,
    *,
    shared: dict,
    shared_receipt: dict,
) -> dict:
    """Retain every shared certificate field, including Taylor remainders."""
    from ..mathematics._validated_metric import (
        METRIC_TAYLOR_METHOD,
        ValidatedMetricTaylorStep,
    )

    if type(index) is not int or not 0 <= index < STEP_COUNT:
        raise ValueError("step index outside frozen schedule")
    if not isinstance(certificate, ValidatedMetricTaylorStep):
        raise TypeError("a complete shared metric certificate is required")
    if (
        exact_record(certificate.time) != index * STEP_SIZE
        or exact_record(certificate.duration) != STEP_SIZE
        or type(certificate.order) is not int
        or certificate.order != ORDER
        or certificate.metric != reference.metric
        or certificate.initial_center != center
        or certificate.initial_radius != radius
        or exact_record(certificate.growth_rate_upper_bound) != 0
        or exact_record(certificate.growth_factor_upper_bound) != 1
        or certificate.method != METRIC_TAYLOR_METHOD
        or certificate.growth_certificate is not None
        or certificate.prescribed_time_interval is not None
        or exact_record(certificate.picard_interior_margin) <= 0
        or len(certificate.domain_lower_bounds) != 20
        or min(exact_record(v) for v in certificate.domain_lower_bounds) <= 0
    ):
        raise ValueError("shared certificate does not meet the selected step contract")
    values = asdict(certificate)
    for name, value in values.items():
        if name not in _CERTIFICATE_METADATA:
            _exact_math_tree(value)
    if set(shared.get("fields", {})) != _SHARED_FIELDS:
        raise ValueError("invalid shared certificate field inventory")
    for name in _SHARED_FIELDS:
        if _json_bytes(values.pop(name)) != _json_bytes(shared["fields"][name]):
            raise ValueError("shared certificate field changed between steps")
    return {
        "schema": "tnfr.formed-two-port-full-step.v1",
        "index": index,
        "source": source,
        "previous_complete_step": previous,
        "domain_values": "strict excess above rho, not raw acute margins",
        "shared_certificate_fields": shared_receipt,
        "certificate": values,
    }


def read_step_certificate(directory: Path, index: int) -> dict:
    """Recover every stored field; this is not a new Taylor proof or audit."""
    from ..mathematics._validated_metric import ValidatedMetricTaylorStep

    root = Path(directory).resolve(strict=True)
    if type(index) is not int or not 0 <= index < STEP_COUNT:
        raise ValueError("invalid step index")
    record = json_loads(
        read_bytes_bounded(
            _contained(root, f"steps/{index:04d}.json"), max_bytes=STEP_BYTES
        )
    )
    if (
        record.get("schema") != "tnfr.formed-two-port-full-step.v1"
        or type(record.get("index")) is not int
        or record["index"] != index
    ):
        raise ValueError("step identity differs")
    receipt = record["shared_certificate_fields"]
    if receipt.get("path") != "steps/certificate-fields.json":
        raise ValueError("unexpected shared-field path")
    common = json_loads(_receipt_data(root, receipt))
    if (
        common.get("schema") != "tnfr.formed-two-port-certificate-fields.v1"
        or set(common.get("fields", {})) != _SHARED_FIELDS
    ):
        raise ValueError("shared certificate shape differs")
    values = record["certificate"]
    expected = {field.name for field in fields(ValidatedMetricTaylorStep)}
    if not isinstance(values, dict) or set(values) != expected - _SHARED_FIELDS:
        raise ValueError("incomplete or unexpected certificate fields")
    return decode_exact_tree({**values, **common["fields"]})


def run_reference_attempt(root: Path, protocol_path: Path, directory: Path) -> int:
    """Execute one fixed schedule only after the external supervisor's gate.

    Return zero only for a complete conditional capture certificate. Every
    unavailable mathematical or resource condition stops the attempt permanently.
    """
    from .bounded_worker import require_current_job_limits

    job = require_current_job_limits(process_memory_bytes=1024**3)
    writer = EvidenceWriter(directory)
    writer.write(
        "worker-start.json",
        {
            "schema": "tnfr.formed-two-port-attempt.v1",
            "policy": declared_policy(),
            "job": job,
        },
    )
    completed, previous, source = 0, None, None
    try:
        source = admit_execution_inputs(root, protocol_path, writer)
        from ..mathematics._validated_metric import validated_metric_taylor_step
        from .formed_two_port_reference import (
            admit_formed_two_port_endpoint,
            admit_formed_two_port_target,
            build_formed_two_port_reference,
            formed_two_port_comparison_bounds,
        )

        reference = build_formed_two_port_reference()
        bounds = formed_two_port_comparison_bounds()
        target = admit_formed_two_port_target(
            reference, Path(root) / "docs/assets/sine_formed_classes"
        )
        if len(target.evidence_receipts) != 4 or {
            row["path"]: row for row in target.evidence_receipts
        } != {row["path"]: row for row in source["target_evidence"]}:
            raise ValueError(
                "actually consumed target evidence differs from frozen source"
            )
        writer.write(
            "admission.json",
            {
                "source": source,
                "reference": asdict(reference),
                "bounds": asdict(bounds),
                "target": asdict(target),
                "actual_source_is_flow_image_family": True,
                "no_actual_state_reset": True,
            },
        )
        center, radius = reference.initial_center, reference.initial_radius
        shared = shared_receipt = None
        for index in range(STEP_COUNT):
            result, failed_tube, reason = validated_metric_taylor_step(
                center,
                radius,
                reference.metric,
                STEP_SIZE,
                reference.field,
                reference.domain,
                order=ORDER,
                time=index * STEP_SIZE,
                growth_rate=Q(0),
                domain_failure="formed_two_port_whole_tube_margin_unavailable",
            )
            if result is None:
                writer.write(
                    "failed-step.json",
                    {
                        "index": index,
                        "reason": reason,
                        "source": source,
                        "previous_complete_step": previous,
                        "failed_tube": (
                            None
                            if failed_tube is None
                            else [asdict(v) for v in failed_tube]
                        ),
                        "initial_center": center,
                        "initial_radius": radius,
                    },
                )
                raise ValueError(f"reference step {index} unavailable: {reason}")
            if shared is None:
                shared = shared_certificate_record(result, reference)
                shared_receipt = writer.write(
                    "steps/certificate-fields.json", shared, step=True
                )
            record = certificate_record(
                result,
                reference,
                index,
                center,
                radius,
                source,
                previous,
                shared=shared,
                shared_receipt=shared_receipt,
            )
            previous = writer.write(f"steps/{index:04d}.json", record, step=True)
            center, radius = result.endpoint_center, result.endpoint_radius
            completed = index + 1
            # No accumulated history; only current state and prior receipt survive.
            del record, result
        endpoint = admit_formed_two_port_endpoint(reference, target, center, radius)
        status = (
            "certified_capture"
            if endpoint.certified is True
            else "endpoint_unavailable"
        )
        writer.write(
            "worker-result.json",
            {
                "status": status,
                "completed_steps": completed,
                "source": source,
                "last_complete_step": previous,
                "endpoint": asdict(endpoint),
                "step_evidence_reserved_bytes": writer.step_used,
                "conditional_on_successful_post_write_supervisor_return": True,
                "boundary": "Conditional full-state mathematical entry; no autonomous event or physical identification.",
            },
            terminal=True,
        )
        return 0 if endpoint.certified is True else 2
    except Exception as exc:
        if not writer.terminal_written:
            writer.write(
                "worker-result.json",
                {
                    "status": "unavailable",
                    "completed_steps": completed,
                    "source": source,
                    "last_complete_step": previous,
                    "exception_type": type(exc).__name__,
                    "reason": str(exc)[:2048],
                    "step_evidence_reserved_bytes": writer.step_used,
                    "non_entry_not_proved": True,
                },
                terminal=True,
            )
        return 2


def worker_main(argv=None) -> int:
    """Entry for the frozen wrapper; use only after the Job/venv gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_reference_attempt(args.root, args.protocol, args.output)
