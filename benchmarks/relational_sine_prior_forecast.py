"""Freeze prior evidence, issue a prediction, then evaluate one reserved response.

Three separate invocations retain immutable protocol/source, prediction and
response records. The predictor receives only the public declaration. Source
hashes bind recorded bytes; they do not authenticate chronology or establish
an adversarial blind experiment.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from benchmarks import relational_seeded_response as evidence
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.research.relational_sine_prior_forecast import (
    assess_sine_prior_response,
    predict_sine_prior_forecast,
    prepare_sine_prior_forecast,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts/research/relational_sine_prior_forecast/response-v1.json"
)
_MAX_BYTES = 32 * 1024**2

# Reuse the established immutable evidence transport without changing its
# already frozen producer or creating a second archive verification policy.
_encoded = evidence._encoded
_write = evidence._write
_require = evidence._require


def _source_files():
    files = evidence._source_files()
    files[Path(__file__).relative_to(ROOT).as_posix()] = Path(__file__).read_bytes()
    return files


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    _require(path.stat().st_size <= _MAX_BYTES, "evidence exceeds byte budget")
    data = path.read_bytes()
    _require(len(data) <= _MAX_BYTES, "evidence exceeds byte budget")
    return data, json_loads(data)


def _source_preparation():
    from tnfr.research.relational_sine_prior_source import prepare_sine_prior_source

    return prepare_sine_prior_source()


def _source_response(source):
    from tnfr.research.relational_sine_prior_source import evaluate_sine_prior_source

    return evaluate_sine_prior_source(source)


def _verify(protocol, archive):
    evidence._verify_runtime_source()
    _require(protocol["runtime"] == evidence._runtime(), "frozen runtime mismatch")
    _require(
        protocol["source_sha256"] == evidence._manifest(_source_files()),
        "frozen source mismatch",
    )
    declaration = protocol["declaration"]
    _require(
        _encoded(declaration)
        == _encoded(prepare_sine_prior_forecast(declaration["prior"])),
        "frozen declaration mismatch",
    )
    _verify_archive(archive, protocol["source_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare", action="store_true")
    stage.add_argument("--predict", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output
    frozen = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    source_path = output.with_suffix(".source-state.json")
    prediction_path = output.with_suffix(".prediction.json")
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    if args.prepare:
        if any(
            path.exists() for path in (frozen, archive, source_path, prediction_path)
        ):
            raise FileExistsError("retain existing preparation or prediction")
        evidence._verify_runtime_source()
        prior, source = _source_preparation()
        files = _source_files()
        protocol = {
            "schema": "tnfr.sine-prior-frozen-protocol.v1",
            "declaration": prepare_sine_prior_forecast(prior),
            "runtime": evidence._runtime(),
            "source_sha256": evidence._manifest(files),
            "archive_scope": "all_tnfr_python_both_producers_and_pyproject_no_dependency_binaries",
        }
        evidence._archive(archive, files)
        _verify_archive(archive, protocol["source_sha256"])
        _write(source_path, source)
        protocol["source_state_sha256"] = _digest(source_path.read_bytes())
        _write(frozen, protocol)
        print(f"Frozen prior protocol: {frozen}")
        return 0

    protocol_bytes, protocol = _read(frozen)
    _verify(protocol, archive)
    archive_digest = _digest(archive.read_bytes())
    provenance = {
        "protocol_sha256": _digest(protocol_bytes),
        "source_archive_sha256": archive_digest,
    }
    if args.predict:
        if prediction_path.exists():
            raise FileExistsError(
                "retain existing prediction, including failed admission"
            )
        target = prediction_path
        schema = "tnfr.sine-prior-prediction-record.v1"

        def action():
            return predict_sine_prior_forecast(protocol["declaration"])

        prediction_bytes = None
    else:
        prediction_bytes, prediction_record = _read(prediction_path)
        _require(
            prediction_record.get("schema") == "tnfr.sine-prior-prediction-record.v1"
            and prediction_record.get("passed") is True
            and all(
                prediction_record.get(key) == value for key, value in provenance.items()
            ),
            "an admitted prediction for this protocol must precede evaluation",
        )
        source_bytes, source = _read(source_path)
        _require(
            _digest(source_bytes) == protocol["source_state_sha256"],
            "source preparation changed",
        )
        provenance["prediction_sha256"] = _digest(prediction_bytes)
        provenance["source_state_sha256"] = _digest(source_bytes)
        target = output
        schema = "tnfr.sine-prior-response-record.v1"

        def action():
            return assess_sine_prior_response(
                prediction_record["result"], _source_response(source)
            )

    result, error, passed = None, None, False
    try:
        payload = action()
        _encoded(payload)
        result = payload
        _require(
            isinstance(result, dict) and type(result.get("passed")) is bool,
            "explicit Boolean verdict required",
        )
        _verify(protocol, archive)
        _require(
            frozen.read_bytes() == protocol_bytes, "protocol changed during execution"
        )
        _require(
            _digest(archive.read_bytes()) == archive_digest,
            "archive changed during execution",
        )
        if prediction_bytes is not None:
            _require(
                prediction_path.read_bytes() == prediction_bytes,
                "issued prediction changed",
            )
            _require(
                source_path.read_bytes() == source_bytes, "source preparation changed"
            )
        passed = result["passed"]
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
    _write(
        target,
        {
            "schema": schema,
            **provenance,
            "result": result,
            "execution_error": error,
            "passed": passed,
        },
    )
    print(
        f"Retained {'prediction' if args.predict else 'response'}: {target}; passed={passed}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
