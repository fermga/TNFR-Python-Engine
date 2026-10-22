"""Freeze and evaluate one full eleven-node receiver prefix/tail certificate.

Uses the existing immutable evidence transport. A separate preparation call
archives source and numerical settings before any reserved response. Neither
source hashes nor archive consistency authenticate chronology or execution.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from benchmarks import relational_seeded_response as evidence
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.research.relational_receiver_barrier import (
    evaluate_receiver_barrier,
    prepare_receiver_barrier,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts/research/relational_receiver_barrier/response-v1.json"
)


def _source_files():
    files = evidence._source_files()
    files[Path(__file__).relative_to(ROOT).as_posix()] = Path(__file__).read_bytes()
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output
    protocol_path = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    evidence._verify_runtime_source()
    files = _source_files()
    if args.prepare:
        if protocol_path.exists() or archive.exists():
            raise FileExistsError("retain existing frozen protocol and source")
        declaration = {
            "schema": "tnfr.sine-receiver-barrier-frozen-protocol.v1",
            "protocol": prepare_receiver_barrier(),
            "runtime": evidence._runtime(),
            "source_sha256": evidence._manifest(files),
            "archive_scope": "all_tnfr_python_and_both_producers_and_pyproject",
        }
        evidence._archive(archive, files)
        _verify_archive(archive, declaration["source_sha256"])
        evidence._write(protocol_path, declaration)
        print(f"Frozen protocol: {protocol_path}")
        return 0
    evidence._require(
        protocol_path.stat().st_size <= 32 * 1024**2, "protocol exceeds byte budget"
    )
    raw = protocol_path.read_bytes()
    protocol = json_loads(raw)
    evidence._require(
        protocol["schema"] == "tnfr.sine-receiver-barrier-frozen-protocol.v1"
        and protocol["runtime"] == evidence._runtime()
        and protocol["source_sha256"] == evidence._manifest(files)
        and evidence._encoded(protocol["protocol"])
        == evidence._encoded(prepare_receiver_barrier()),
        "frozen declaration, runtime or source mismatch",
    )
    _verify_archive(archive, protocol["source_sha256"])
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    provenance = {
        "schema": "tnfr.sine-receiver-barrier-record.v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "source_archive_sha256": archive_digest,
    }
    payload = None
    try:
        response = evaluate_receiver_barrier()
        evidence._encoded(response)
        payload = response
        passed = response["assessment"][
            "all_future_receiver_potential_barrier_excluded"
        ]
        evidence._require(type(passed) is bool, "response requires a Boolean verdict")
        evidence._verify_runtime_source()
        evidence._require(
            protocol["source_sha256"] == evidence._manifest(_source_files()),
            "source changed during response evaluation",
        )
        evidence._require(
            protocol["runtime"] == evidence._runtime(),
            "runtime changed during evaluation",
        )
        evidence._require(
            protocol_path.read_bytes() == raw
            and hashlib.sha256(archive.read_bytes()).hexdigest() == archive_digest,
            "frozen evidence changed during response evaluation",
        )
        record = {**provenance, "response": payload, "evaluation_error": None}
    except Exception as error:
        record = {
            **provenance,
            "response": payload,
            "evaluation_error": {
                "error_type": type(error).__name__,
                "error": str(error),
            },
        }
        passed = False
    record["passed"] = passed
    evidence._write(output, record)
    print(f"Retained response: {output}; passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
