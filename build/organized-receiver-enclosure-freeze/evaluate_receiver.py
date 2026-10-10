"""One receiver-enclosure attempt with immediately retained accepted steps."""

import platform
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research.artifact_io import (  # noqa: E402
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    write_json_once,
)
from tnfr.research.frozen_source import inspect_frozen_source  # noqa: E402
from tnfr.utils.io import json_dumps, json_loads  # noqa: E402

STEM = "docs/assets/sine_formed_classes/organized-receiver-enclosure-v1"


def _canonical(value):
    return json_dumps(encode_exact_tree(value), sort_keys=True)


def _runtime_environment():
    return {
        "python": sys.version,
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ("numpy", "scipy", "networkx", "mpmath")
        },
    }


def main():
    receipt_path = STEM + ".freeze.json"
    inspection = inspect_frozen_source(ROOT, receipt_path)
    if inspection.existing_outcome_files:
        raise FileExistsError("preserve the first receiver attempt and partial prefix")
    evaluator = ROOT / inspection.future_evaluator
    if Path(__file__).resolve() != evaluator.resolve():
        raise ValueError("receiver evaluator differs from its frozen location")
    with zipfile.ZipFile(ROOT / (STEM + ".source.zip")) as archive:
        if evaluator.read_bytes() != archive.read(inspection.future_evaluator):
            raise ValueError("receiver evaluator bytes differ from frozen source")
    base = inspection.source_base_commit
    subprocess.run(["git", "diff", "--quiet", base, "--", "src"], cwd=ROOT, check=True)
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src"], cwd=ROOT
    ).strip():
        raise ValueError("runtime has unrecorded source files")
    protocol = decode_exact_tree(
        json_loads(
            read_bytes_bounded(ROOT / (STEM + ".protocol.json"), max_bytes=32 * 1024**2)
        )
    )
    environment = _runtime_environment()
    if _canonical(protocol["runtime_environment"]) != _canonical(environment):
        raise ValueError("runtime environment differs from frozen declaration")
    from tnfr.research.organized_receiver_enclosure import (
        receiver_enclosure_policy,
        receiver_enclosure_source,
        run_receiver_enclosure,
    )

    policy = receiver_enclosure_policy()
    source = asdict(receiver_enclosure_source())
    for key, value in (("numerical_policy", policy), ("source_admission", source)):
        if _canonical(protocol[key]) != _canonical(value):
            raise ValueError(f"frozen primitive association differs: {key}")
    maximum_steps = protocol["reserved_attempt_policy"]["maximum_step_records"]
    if type(maximum_steps) is not int or maximum_steps < 1:
        raise ValueError("frozen step count must be a positive ordinary integer")
    attempt_path = ROOT / (STEM + ".attempt.json")
    write_json_once(
        attempt_path,
        {
            "schema": "tnfr.organized-receiver-enclosure-attempt.v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_base_commit": base,
            "freeze_sha256": inspection.receipt_sha256,
            "runtime_environment": environment,
            "seed": None,
            "producer": "tnfr.research.organized_receiver_enclosure.run_receiver_enclosure",
            "attempt_policy": "one_fixed_attempt_no_refinement_retry_or_alternative_route",
        },
    )
    started = time.monotonic()
    steps_path = ROOT / (STEM + ".steps")
    retained_steps = []

    def record_step(record):
        if not isinstance(record, dict):
            raise ValueError("attempted step must be an exact record dictionary")
        if (
            type(record.get("step_index")) is not int
            or record["step_index"] != len(retained_steps) + 1
            or type(record.get("accepted")) is not bool
        ):
            raise ValueError(
                "step records require sequential indices and Boolean admission"
            )
        if len(retained_steps) >= maximum_steps:
            raise ValueError("producer exceeded frozen step-record budget")
        label = f"{STEM}.steps/step-{len(retained_steps) + 1:04d}.json"
        path = ROOT / label
        write_json_once(path, record)
        retained_steps.append(file_receipt(path, label=label))
        if len(retained_steps) % 8 == 0 or not record["accepted"]:
            print(f"Retained receiver step record {len(retained_steps)}.", flush=True)

    report = None
    try:
        steps_path.mkdir()
        report = run_receiver_enclosure(record_step=record_step)
        if not isinstance(report, dict) or not isinstance(report.get("status"), str):
            raise ValueError("producer must return an explicit status record")
        result = {"status": report["status"], "report": report}
    except BaseException as error:
        result = {
            "status": "execution_or_assessment_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "report": report,
        }
    completed_paths = {item["path"] for item in retained_steps}
    incomplete_paths = tuple(
        path.relative_to(ROOT).as_posix()
        for path in sorted(steps_path.glob("step-*.json"))
        if path.relative_to(ROOT).as_posix() not in completed_paths
    )
    result.update(
        schema="tnfr.organized-receiver-enclosure-outcome.v1",
        source_base_commit=base,
        freeze_sha256=inspection.receipt_sha256,
        numerical_policy=policy,
        step_receipts=retained_steps,
        incomplete_step_paths=incomplete_paths,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        attempt=file_receipt(attempt_path, label=STEM + ".attempt.json"),
    )
    try:
        write_json_once(ROOT / (STEM + ".json"), result)
    except BaseException as error:
        write_json_once(
            ROOT / (STEM + ".export-error.json"),
            {
                "schema": "tnfr.organized-receiver-enclosure-export-error.v1",
                "error_type": type(error).__name__,
                "error": str(error),
                "result_status": result["status"],
                "step_receipts": retained_steps,
                "incomplete_step_paths": incomplete_paths,
                "first_attempt_retained": True,
                "retry_authorized": False,
            },
        )
        raise
    print(f"First receiver outcome retained: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
