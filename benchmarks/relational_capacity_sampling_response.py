"""Frozen K3 phase-sampling software verification, with no adaptive retries.

Prepare a protocol/source bundle in one invocation, then evaluate its four
declared candidate/arm cases in another. Interval Taylor enclosures certify a
supplied software model; success does not select an unknown physical law.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import zipfile
from pathlib import Path

from tnfr.research.relational_capacity_discriminator import (
    evaluate_relational_capacity_response,
    prepare_relational_capacity_response,
)
from tnfr.utils.io import json_dumps, json_loads, safe_write

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts/research/relational_capacity_discriminator/response-v1.json"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _encoded(value):
    return json_dumps(value, sort_keys=True, allow_nan=False)


def _write(path, value):
    """An evidence file is created once, including an original failed verdict."""
    text = _encoded(value) + "\n"
    safe_write(
        path, lambda stream: stream.write(text), mode="x", atomic=False, sync=True
    )


def _source_files():
    paths = sorted((ROOT / "src/tnfr").rglob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "pyproject.toml"]
    return {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}


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


def prepare_protocol():
    """Freeze declarations and prior-domain bounds without response samples."""
    _verify_runtime_source()
    protocol = prepare_relational_capacity_response()
    return {
        "schema": "tnfr.relational-capacity-response-protocol.v1",
        "protocol": protocol.to_dict(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "precision": "exact rational inputs and outward dyadic128 intervals",
            "randomness": "none",
        },
        "source_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in _source_files().items()
        },
        "source_archive_scope": (
            "all project Python under src/tnfr, this producer and pyproject; "
            "excludes dependency binaries and does not authenticate chronology"
        ),
    }


def _archive(path, files):
    def write(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in files.items():
                bundle.writestr(name, data)

    safe_write(path, write, mode="xb", atomic=False, sync=True)


def _verify_archive(path, expected):
    """Check bytes without extraction, including a finite expanded-size budget."""
    with zipfile.ZipFile(path) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        _require(
            len(names) == len(set(names)) and set(names) == set(expected),
            "source archive inventory differs",
        )
        _require(
            len(entries) <= 10000
            and sum(entry.file_size for entry in entries) <= 128 * 1024**2,
            "source archive exceeds verification budget",
        )
        for entry in entries:
            _require(
                entry.file_size <= 8 * 1024**2
                and not entry.flag_bits & 1
                and entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "unsupported source archive member",
            )
            _require(
                hashlib.sha256(bundle.read(entry)).hexdigest()
                == expected[entry.filename],
                "source archive content differs",
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output
    frozen = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    if args.prepare:
        if frozen.exists() or archive.exists():
            raise FileExistsError("retain existing protocol/source archive")
        protocol = prepare_protocol()
        _archive(archive, _source_files())
        _verify_archive(archive, protocol["source_sha256"])
        _write(frozen, protocol)
        print(f"Frozen protocol: {frozen}")
        return 0

    protocol = json_loads(frozen.read_bytes())
    _require(
        _encoded(protocol) == _encoded(prepare_protocol()),
        "frozen protocol/source/runtime mismatch",
    )
    _verify_archive(archive, protocol["source_sha256"])
    provenance = {
        "schema": "tnfr.relational-capacity-response-record.v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    response_payload = None
    try:
        # Equality above binds this typed declaration to the separate frozen file.
        response = evaluate_relational_capacity_response(
            prepare_relational_capacity_response()
        )
        response_payload = response.to_dict()
        _verify_runtime_source()
        _require(
            protocol["source_sha256"]
            == {
                name: hashlib.sha256(data).hexdigest()
                for name, data in _source_files().items()
            },
            "source changed during response evaluation",
        )
        record = {**provenance, "response": response_payload, "evaluation_error": None}
        passed = response.passed
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
