"""Freeze and evaluate one bounded regular source/receiver response.

Preparation archives the declared source bytes without evaluating the reserved
trajectory. Evaluation is a separate invocation and never overwrites a protocol,
archive or response, including a failed verdict. Archive and runtime checks
establish recorded consistency, not execution or chronology authentication.
Use ``--correction-of`` in both invocations to identify a numerical correction
of an existing response; it is not an independent blind replication.
The separate ``--study target-budget`` declaration requires an explicit output
path and the same study selection when evaluating its frozen protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import zipfile
from fractions import Fraction as Q
from pathlib import Path

from tnfr.mathematics._rational_interval import INTERVAL_METHOD
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.research.relational_seeded_response import (
    evaluate_relational_seeded_response,
    prepare_relational_seeded_response,
)
from tnfr.utils.io import json_dumps, json_loads, safe_write

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/research/relational_seeded_response/response-v1.json"
_MAX_PROTOCOL_BYTES = 32 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _encoded(value):
    return json_dumps(value, sort_keys=True, allow_nan=False)


def _write(path, value):
    """Create each evidence file once, preserving original negative outcomes."""
    text = _encoded(value) + "\n"
    safe_write(
        path, lambda stream: stream.write(text), mode="x", atomic=False, sync=True
    )


def _source_files():
    paths = sorted((ROOT / "src/tnfr").rglob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "pyproject.toml"]
    return {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}


def _manifest(files):
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def _verify_runtime_source():
    source = (ROOT / "src/tnfr").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "tnfr" or name.startswith("tnfr."):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                path = Path(origin).resolve()
                _require(
                    path.is_relative_to(source) and path.suffix == ".py",
                    f"executing source is outside the declared checkout: {name}",
                )


def _runtime():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "precision": "mathematical pi and outward exact rational dyadic128 intervals",
        "interval_method": INTERVAL_METHOD,
        "randomness": "none; no RNG or seed",
    }


def _prior_record_metadata(path):
    """Bind a correction to retained bytes, without evaluating the prior response."""
    path = Path(path).resolve(strict=True)
    _require(
        path.stat().st_size <= _MAX_PROTOCOL_BYTES,
        "prior response exceeds verification byte budget",
    )
    data = path.read_bytes()
    _require(
        len(data) <= _MAX_PROTOCOL_BYTES,
        "prior response exceeds verification byte budget",
    )
    record = json_loads(data)
    _require(
        isinstance(record, dict)
        and record.get("schema") == "tnfr.relational-seeded-response-record.v1"
        and type(record.get("passed")) is bool,
        "correction requires a retained seeded response record",
    )
    display = (
        path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
    )
    return {
        "prior_record": display,
        "prior_record_sha256": hashlib.sha256(data).hexdigest(),
        "prior_passed": record["passed"],
        "kind": "separately_identified_numerical_correction",
        "scope": "numerical_correction_not_independent_blind_replication",
    }


def _study_options(study, horizon):
    """Keep short/default calls unchanged; the shared owner admits time bounds."""
    if study == "short":
        _require(horizon is None, "a horizon override requires the target-budget study")
        return {}
    options = {"study": study}
    if horizon is not None:
        options["horizon"] = horizon
    return options


def prepare_protocol(*, files=None, correction_of=None, study="short", horizon=None):
    """Freeze the declaration and provenance without a reserved response."""
    _verify_runtime_source()
    protocol = {
        "schema": "tnfr.relational-seeded-response-protocol.v1",
        "protocol": prepare_relational_seeded_response(
            **_study_options(study, horizon)
        ),
        "runtime": _runtime(),
        "source_sha256": _manifest(_source_files() if files is None else files),
        "source_archive_scope": (
            "all project Python under src/tnfr, this producer and pyproject; "
            "excludes dependency binaries and does not authenticate chronology"
        ),
    }
    if correction_of is not None:
        protocol["correction"] = _prior_record_metadata(correction_of)
    return protocol


def _archive(path, files):
    def write(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in files.items():
                bundle.writestr(name, data)

    safe_write(path, write, mode="xb", atomic=False, sync=True)


def _parse_horizon(text):
    try:
        return Q(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(
            "horizon requires exact rational text"
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--study", choices=("short", "target-budget"), default="short")
    parser.add_argument(
        "--horizon",
        type=_parse_horizon,
        help="optional exact target-budget horizon, for example 9/8; default is 1",
    )
    parser.add_argument(
        "--correction-of",
        type=Path,
        help="retained prior response; supply for both preparation and evaluation",
    )
    args = parser.parse_args(argv)
    if args.horizon is not None and args.study != "target-budget":
        parser.error("--horizon requires --study target-budget")
    if args.study == "target-budget" and args.output is None:
        parser.error("--study target-budget requires an explicit --output")
    output = args.output if args.output is not None else DEFAULT_OUTPUT
    frozen = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    if args.prepare:
        if frozen.exists() or archive.exists():
            raise FileExistsError("retain existing protocol/source archive")
        files = _source_files()
        protocol = prepare_protocol(
            files=files,
            correction_of=args.correction_of,
            study=args.study,
            horizon=args.horizon,
        )
        _archive(archive, files)
        _verify_archive(archive, protocol["source_sha256"])
        _write(frozen, protocol)
        print(f"Frozen protocol: {frozen}")
        return 0

    _require(
        frozen.stat().st_size <= _MAX_PROTOCOL_BYTES,
        "protocol exceeds verification byte budget",
    )
    protocol_bytes = frozen.read_bytes()
    protocol = json_loads(protocol_bytes)
    _require(
        _encoded(protocol)
        == _encoded(
            prepare_protocol(
                correction_of=args.correction_of,
                study=args.study,
                horizon=args.horizon,
            )
        ),
        "frozen protocol/source/runtime mismatch",
    )
    _verify_archive(archive, protocol["source_sha256"])
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    provenance = {
        "schema": "tnfr.relational-seeded-response-record.v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_archive_sha256": archive_digest,
    }
    response_payload = None
    try:
        declaration = json_loads(_encoded(protocol["protocol"]))
        response = evaluate_relational_seeded_response(
            declaration, **_study_options(args.study, args.horizon)
        )
        # Validate JSON projection before retaining a returned payload. An
        # unserializable value must not prevent a failed evaluation record.
        _encoded(response)
        response_payload = response
        _require(
            isinstance(response, dict) and type(response.get("passed")) is bool,
            "response requires an explicit Boolean passed verdict",
        )
        _verify_runtime_source()
        _require(
            protocol["source_sha256"] == _manifest(_source_files()),
            "source changed during response evaluation",
        )
        _require(protocol["runtime"] == _runtime(), "runtime changed during evaluation")
        if args.correction_of is not None:
            _require(
                _prior_record_metadata(args.correction_of) == protocol["correction"],
                "prior response changed during correction evaluation",
            )
        _require(
            frozen.read_bytes() == protocol_bytes
            and hashlib.sha256(archive.read_bytes()).hexdigest() == archive_digest,
            "frozen evidence changed during response evaluation",
        )
        record = {**provenance, "response": response_payload, "evaluation_error": None}
        passed = response["passed"]
    except Exception as error:
        record = {
            **provenance,
            "response": response_payload,
            "evaluation_error": {
                "error_type": type(error).__name__,
                "error": str(error),
            },
        }
        passed = False
    record["passed"] = passed
    _write(output, record)
    print(f"Retained response: {output}; passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
