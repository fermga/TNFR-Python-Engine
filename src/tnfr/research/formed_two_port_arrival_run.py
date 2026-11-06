"""One prospective actual-phase inclusion attempt, with two terminal tests.

Scientific admission and propagation occur only inside the bounded worker.
This method neither resumes the consumed auxiliary-reference attempt nor
changes its source. Importing this module evaluates no scientific response.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path

from ..utils.io import json_loads
from .artifact_io import read_bytes_bounded, verify_archive_members
from .formed_two_port_run import (
    AUX_BYTES,
    STEP_BYTES,
    EvidenceWriter,
    _json_bytes,
    _receipt,
    _receipt_data,
)

STEP_SIZE = Q(1, 16)
START_TIME = Q(1)
END_TIME = Q(32)
STEP_COUNT = 496
ORDER = 8
RATE_BOUNDS = (Q(-1, 64), Q(2))
RATE_BISECTIONS = 12
SCHEMA = "tnfr.formed-two-port-actual-inclusion-protocol.v2"
_PREFIX = "artifacts/formed-two-port-attempt-20261029"
_TARGET = "docs/assets/sine_formed_classes"
_TARGET_FILES = {
    f"{_TARGET}/two-port-compatibility-v1{suffix}"
    for suffix in (".json", ".manifest.json", ".protocol.json", ".source.zip")
}
_PREFIX_FILES = {
    f"{_PREFIX}/{name}"
    for name in (
        "audit.json",
        "attempt/source.zip",
        "attempt/protocol.json",
        "attempt/steps/certificate-fields.json",
        *(f"attempt/steps/{i:04d}.json" for i in range(4)),
    )
}
_OWNERS = {
    "theory/nodal/SINE_TWO_PORT_CAPTURE.md",
    "theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md",
    "theory/nodal/SINE_FORMED_CLASS_CONTACT.md",
    "theory/nodal/SINE_FORM_PHASE_REDUCTION.md",
    "theory/RESEARCH_PROGRAM.md",
    "theory/research_prerequisites.json",
}


def declared_policy() -> dict:
    """A fixed finite discriminator, not a promised convergence deadline."""
    return {
        "actual_dimension": 36,
        "execution_revision": "chunked_artifact_reads_and_durable_phase_diagnostics_v2",
        "prior_consumed_protocol": "artifacts/formed-two-port-actual-arrival-20261101/protocol.json",
        "failure_diagnostics": "immutable_phase_journal_and_at_most_12_stack_frames_no_locals",
        "relative_phase_dimension": 17,
        "source_phi": Q(1, 1000),
        "source_epsilon": Q(1, 10**32),
        "contact_tau": 100 + 10**13,
        "contact_edges": ((4, 13), (5, 14)),
        "start_slow_time": START_TIME,
        "end_slow_time": END_TIME,
        "step_size": STEP_SIZE,
        "maximum_steps": STEP_COUNT,
        "order": ORDER,
        "picard_max_iterations_per_call": 16,
        "growth_rate_bounds": RATE_BOUNDS,
        "growth_bisections": RATE_BISECTIONS,
        "growth_certificate": "relative_signed_edge_cosine_laplacian",
        "forcing_picard_bound": Q(1, 7000),
        "forcing_endpoint_bound": "min(1/7000,tube_field_norm/20000)",
        "arithmetic": "rational_outward_dyadic128_machin_trigonometry_v1",
        "terminal_checkpoints": "initial_admitted_ball_and_every_complete_step",
        "early_terminal": ("certified_capture", "certified_nonarrival"),
        "capture_test": "actual_target_distance_upper<=1/1024_then_ST7",
        "nonarrival_test": "actual_total_storage_upper<target_storage_lower",
        "inconclusive": "neither_test_or_any_failed_enclosure_or_resource_limit",
        "wall_seconds": 3600,
        "process_memory_bytes": 1024**3,
        "memory_metric": "Windows Job process and job committed virtual memory",
        "step_bytes": STEP_BYTES,
        "auxiliary_bytes": AUX_BYTES,
        "supervisor_reserved_bytes": 2 * 1024**2,
        "caller_reserved_bytes": 64 * 1024,
        "workers": 1,
        "attempts": 1,
        "resume": False,
        "new_root_search": False,
        "adaptive_parameters": False,
        "physical_mapping": False,
    }


def admit_execution_inputs(root: Path, protocol_path: Path, writer: EvidenceWriter):
    """Bind the exact reviewed bytes and effective bounded-worker declaration.

    Byte checks do not authenticate review authorship or chronology. Independent
    substantive review of this method and its frozen sources is a prerequisite.
    """
    root = Path(root).resolve(strict=True)
    path = Path(protocol_path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("protocol must belong to the selected repository")
    raw = read_bytes_bounded(path, max_bytes=1024**2)
    record = json_loads(raw)
    if not isinstance(record, dict) or set(record) != {
        "schema", "policy", "head", "runtime", "sources", "archive",
        "reviews", "controls", "worker_entry",
    }:
        raise ValueError("invalid actual-inclusion protocol shape")
    if record["schema"] != SCHEMA or _json_bytes(record["policy"]) != _json_bytes(
        declared_policy()
    ):
        raise ValueError("protocol differs from the selected inclusion policy")
    if not isinstance(record["head"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", record["head"]
    ):
        raise ValueError("invalid frozen Git base")
    runtime = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": sys.version,
        "package_root": str((root / "src/tnfr").resolve()),
    }
    if record["runtime"] != runtime:
        raise ValueError("worker runtime differs from reviewed interpreter/source")
    import tnfr

    if not Path(tnfr.__file__).resolve().is_relative_to(root / "src/tnfr"):
        raise ValueError("worker imported a different TNFR package")
    rows = record["sources"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("invalid frozen source inventory")
    names = []
    for row in rows:
        _receipt(root, row)
        names.append(row["path"])
    all_python = {
        p.relative_to(root).as_posix() for p in (root / "src/tnfr").rglob("*.py")
    }
    if len(names) != len(set(names)) or not (
        all_python | _OWNERS | _PREFIX_FILES | _TARGET_FILES
    ) <= set(names):
        raise ValueError("incomplete or duplicate frozen source inventory")
    source_rows = {row["path"]: row for row in rows}
    entry = _receipt(root, record["worker_entry"])
    if source_rows.get(record["worker_entry"]["path"]) != record["worker_entry"] or (
        entry != Path(sys.argv[0]).resolve()
    ):
        raise ValueError("executed wrapper is not the frozen worker entry")
    attempt = json_loads(
        read_bytes_bounded(writer.root / "attempt.json", max_bytes=512 * 1024)
    )
    expected_arguments = [
        "--root", str(root), "--protocol", str(path), "--output", str(writer.root)
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
    archive = _receipt_data(root, record["archive"])
    verify_archive_members(
        io.BytesIO(archive),
        {row["path"]: row["sha256"] for row in rows},
        max_bytes=AUX_BYTES,
    )
    reviews = record["reviews"]
    if not isinstance(reviews, list) or len(reviews) < 2:
        raise ValueError("missing substantive independent reviews")
    review_data = [_receipt_data(root, row) for row in reviews]
    if len({row["path"] for row in reviews}) != len(reviews):
        raise ValueError("duplicate independent review association")
    controls = _receipt_data(root, record["controls"])
    control_record = json_loads(controls)
    if control_record.get("passed") is not True or (
        type(control_record.get("reserved_actual_steps")) is not int
        or control_record["reserved_actual_steps"] != 0
    ):
        raise ValueError("source-independent method controls not passed")
    association = {
        "protocol": writer.copy("protocol.json", raw),
        "archive": writer.copy("source.zip", archive),
        "head": record["head"],
        "worker_entry": record["worker_entry"],
        "target_evidence": [source_rows[name] for name in sorted(_TARGET_FILES)],
        "prefix_evidence": [source_rows[name] for name in sorted(_PREFIX_FILES)],
    }
    for index, data in enumerate(review_data):
        writer.copy(f"review-{index:02d}.md", data)
    writer.copy("controls.json", controls)
    return association


def _advance(model, center, radius, time):
    """One fixed inclusion step; never call this outside the admitted attempt."""
    from ..mathematics._validated_metric_inclusion import (
        validated_metric_inclusion_step,
    )

    return validated_metric_inclusion_step(
        center, radius, model.metric, STEP_SIZE, model.flow, model.domain,
        forcing_bound=Q(1, 7000),
        forcing_refinement=lambda tube: min(
            Q(1, 7000), model.field_norm_upper(tube) / 20000
        ),
        growth_certifier=lambda tube: model.certify_growth(
            tube, rate_bounds=RATE_BOUNDS, bisections=RATE_BISECTIONS
        ),
        order=ORDER, time=time,
    )


def _decision_record(decision):
    if decision.status not in {
        "certified_capture", "certified_nonarrival", "unavailable"
    }:
        raise ValueError("unknown mathematical terminal verdict")
    return asdict(decision)


def _exception_frames(exc):
    """Retain bounded location metadata without locals or linecache allocations."""
    frames, current = [], exc.__traceback__
    while current is not None and len(frames) < 12:
        code = current.tb_frame.f_code
        frames.append({"file": code.co_filename[-1024:],
                       "function": code.co_name[-256:], "line": current.tb_lineno})
        current = current.tb_next
    return frames, current is not None


def run_arrival_attempt(root: Path, protocol_path: Path, directory: Path) -> int:
    """Stop on the first declared proof, unavailable step, or fixed horizon.

    Both mathematical proofs return zero; inconclusive/failed work returns two.
    A successful worker return is still conditional on the supervisor's closing
    time/resource checks and the subsequent independent certificate audit.
    """
    from .bounded_worker import require_current_job_limits

    job = require_current_job_limits(process_memory_bytes=1024**3)
    writer = EvidenceWriter(directory)
    writer.write("worker-start.json", {"schema": SCHEMA, "job": job,
                                       "policy": declared_policy()})
    completed, previous, association = 0, None, None
    phase, phase_ordinal = "worker_started", 0
    # Released on failure to leave room for a small, bounded terminal record.
    emergency_reserve = bytearray(64 * 1024)

    def mark(label):
        nonlocal phase, phase_ordinal
        phase = label
        writer.write(f"phase-{phase_ordinal:04d}.json", {
            "ordinal": phase_ordinal, "phase": phase, "completed_steps": completed,
        })
        phase_ordinal += 1

    try:
        mark("input_admission_started")
        association = admit_execution_inputs(root, protocol_path, writer)
        mark("input_admission_completed")
        mark("scientific_import_started")
        from .formed_two_port_arrival import (
            admit_formed_two_port_arrival_source,
            admit_formed_two_port_arrival_target,
            assess_formed_two_port_arrival_terminal,
            build_formed_two_port_arrival_model,
        )

        mark("scientific_import_completed")
        mark("model_build_started")
        model = build_formed_two_port_arrival_model()
        mark("model_build_completed")
        mark("source_admission_started")
        source = admit_formed_two_port_arrival_source(model, Path(root) / _PREFIX)
        mark("source_admission_completed")
        mark("target_admission_started")
        target = admit_formed_two_port_arrival_target(model, Path(root) / _TARGET)
        mark("target_admission_completed")
        mark("consumed_receipts_check_started")
        prefix_rows = {
            f"{_PREFIX}/{row['path']}": {**row, "path": f"{_PREFIX}/{row['path']}"}
            for row in source.evidence_receipts
        }
        if prefix_rows != {row["path"]: row for row in association["prefix_evidence"]}:
            raise ValueError("consumed prefix evidence differs from frozen source")
        if {row["path"]: row for row in target.evidence_receipts} != {
            row["path"]: row for row in association["target_evidence"]
        }:
            raise ValueError("consumed target evidence differs from frozen source")
        if source.time != START_TIME:
            raise ValueError("actual source time differs from admitted prefix")
        center, radius, time = source.center, source.radius, source.time
        mark("consumed_receipts_check_completed")
        mark("initial_decision_started")
        decision = assess_formed_two_port_arrival_terminal(
            model, target, center, radius, time
        )
        mark("initial_decision_completed")
        mark("admission_write_started")
        writer.write("admission.json", {
            "source": association, "model": asdict(model),
            "actual_source": asdict(source), "target": asdict(target),
            "initial_decision": _decision_record(decision),
            "full_law_and_original_source_preserved": True,
        })
        mark("admission_write_completed")
        for index in range(STEP_COUNT):
            if decision.status != "unavailable":
                break
            mark("step_advance_started")
            result, failed_tube, reason = _advance(model, center, radius, time)
            mark("step_advance_returned")
            if result is None:
                mark("failed_step_write_started")
                writer.write("failed-step.json", {
                    "index": index, "time": time, "reason": reason,
                    "initial_center": center, "initial_radius": radius,
                    "failed_tube": None if failed_tube is None else [
                        asdict(value) for value in failed_tube
                    ],
                    "previous_complete_step": previous, "source": association,
                })
                raise ValueError(f"actual inclusion step {index} unavailable: {reason}")
            next_time = START_TIME + (index + 1) * STEP_SIZE
            mark("step_decision_started")
            decision = assess_formed_two_port_arrival_terminal(
                model, target, result.endpoint_center, result.endpoint_radius, next_time
            )
            mark("step_decision_completed")
            mark("step_certificate_write_started")
            record = {
                "schema": "tnfr.formed-two-port-actual-inclusion-step.v1",
                "index": index, "time": time, "endpoint_time": next_time,
                "source": association, "previous_complete_step": previous,
                "certificate": asdict(result), "decision": _decision_record(decision),
            }
            previous = writer.write(f"steps/{index:04d}.json", record, step=True)
            center, radius, time = result.endpoint_center, result.endpoint_radius, next_time
            completed = index + 1
            del record, result
            mark("step_certificate_write_completed")
        status = decision.status if decision.status != "unavailable" else "inconclusive"
        mark("terminal_write_started")
        writer.write("worker-result.json", {
            "status": status, "completed_steps": completed, "endpoint_time": time,
            "source": association, "last_complete_step": previous,
            "decision": _decision_record(decision),
            "step_evidence_reserved_bytes": writer.step_used,
            "conditional_on_successful_post_write_supervisor_return": True,
            "boundary": "Conditional unchanged-law mathematical result; no physical identification.",
        }, terminal=True)
        return 0 if status in {"certified_capture", "certified_nonarrival"} else 2
    except Exception as exc:
        emergency_reserve = None
        frames, truncated = _exception_frames(exc)
        diagnostic = {"exception_type": type(exc).__name__, "reason": str(exc)[:2048],
                      "failure_phase": phase, "traceback_frames": frames,
                      "traceback_truncated": truncated}
        # The independently bounded supervisor log survives a partial/failed
        # terminal-file write. Never overwrite that partial immutable file.
        sys.stderr.write(_json_bytes(diagnostic).decode("utf-8"))
        sys.stderr.flush()
        try:
            if not writer.terminal_written:
                writer.write("worker-result.json", {
                    "status": "unavailable", "completed_steps": completed,
                    "source": association, "last_complete_step": previous,
                    **diagnostic, "step_evidence_reserved_bytes": writer.step_used,
                    "nonarrival_not_proved": True,
                }, terminal=True)
        except Exception as finalization_error:
            frames, truncated = _exception_frames(finalization_error)
            sys.stderr.write(_json_bytes({
                "failure_phase": "failure_record_write",
                "exception_type": type(finalization_error).__name__,
                "traceback_frames": frames, "traceback_truncated": truncated,
                "original_failure_phase": phase,
            }).decode("utf-8"))
            sys.stderr.flush()
        return 2


def worker_main(argv=None):
    """Entry point for the separately frozen wrapper and hard supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_arrival_attempt(args.root, args.protocol, args.output)
