"""Exclusive evidence lifecycle for one fixed analytical assessment.

Scientific premises and fixed choices belong to the committed declaration
and no-argument assessor. This runner records their source and exact report;
it neither proves the report nor authorizes a repeat or parameter search.
Previously frozen producers retain their own unchanged lifecycle code.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from ..mathematics._rational_interval import INTERVAL_BITS, INTERVAL_METHOD
from .artifact_io import (
    encode_exact_tree,
    file_receipt,
    sha256_bytes,
    validate_artifact_path,
    write_json_once,
)
from .core_manifests import current_git_source_provenance


def _source_snapshot(repository, declaration_path, evaluator_path):
    scope = ("src/tnfr", evaluator_path, declaration_path, "pyproject.toml")
    revision, dirty, _ = current_git_source_provenance(repository, scope)
    if dirty:
        raise RuntimeError(
            "commit the declaration and complete source before evaluation"
        )
    listing = subprocess.check_output(
        ("git", "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", *scope),
        cwd=repository,
    )
    inventory = {}
    for entry in listing.split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            mode, kind, blob = metadata.decode("ascii").split()
            if kind != "blob":
                raise RuntimeError(
                    "research source inventory requires ordinary Git blobs"
                )
            inventory[path.decode("utf-8")] = {"mode": mode, "blob": blob}
    if declaration_path not in inventory or evaluator_path not in inventory:
        raise RuntimeError(
            "the prospective declaration and evaluator must be committed"
        )
    tree = (
        subprocess.check_output(("git", "rev-parse", "HEAD:src"), cwd=repository)
        .decode("ascii")
        .strip()
    )
    return {
        "source_revision": revision,
        "source_scope": scope,
        "source_tree": tree,
        "source_inventory": inventory,
        "source_inventory_sha256": sha256_bytes(listing),
    }


def _imported_sources(repository):
    sources = {}
    for name, module in tuple(sys.modules.items()):
        if name == "tnfr" or name.startswith("tnfr."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(repository / "src"):
                    raise RuntimeError(
                        f"imported TNFR source is outside this tree: {name}"
                    )
                sources[name] = path.relative_to(repository).as_posix()
    return sources


def _run_single_analytic_assessment(
    *,
    repository,
    output,
    declaration_path,
    evaluator_path,
    schema_name,
    assess,
    admitted_verdict,
    unavailable_verdict,
):
    """Record one no-argument dataclass assessment and return its stated verdict.

    The source scope includes every tracked core dependency. A partial attempt
    or failed export deliberately prevents an unrecorded retry in that output.
    ``admitted`` must be an ordinary Boolean; its scientific meaning belongs
    to the assessor, while every other field is serialized without reinterpretation.
    """
    repository, output = Path(repository).resolve(), Path(output)
    declaration_path = validate_artifact_path(declaration_path)
    evaluator_path = validate_artifact_path(evaluator_path)
    if Path.cwd().resolve() != repository:
        raise RuntimeError("run from the declared repository root")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("evidence directory is not empty")
    source = _source_snapshot(repository, declaration_path, evaluator_path)
    imported = _imported_sources(repository)
    output.mkdir(parents=True, exist_ok=True)
    attempt_path = output / "attempt.json"
    write_json_once(
        attempt_path,
        {
            "schema": f"{schema_name}_attempt_v1",
            **source,
            "declaration": declaration_path,
            "evaluator": evaluator_path,
            "parameters": {},
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "arithmetic_method": INTERVAL_METHOD,
            "interval_bits": INTERVAL_BITS,
            "imported_sources": imported,
            "seed": None,
            "maximum_assessments": 1,
            "evaluation_status_at_start": "not_evaluated",
        },
    )
    encoded_report = None
    try:
        report = assess()
        encoded_report = encode_exact_tree(asdict(report))
        if type(report.admitted) is not bool:
            raise ValueError("assessment admission must be an ordinary Boolean")
        imported = _imported_sources(repository)
        if _source_snapshot(repository, declaration_path, evaluator_path) != source:
            raise RuntimeError("committed source changed during the assessment")
        verdict = admitted_verdict if report.admitted else unavailable_verdict
        write_json_once(
            output / "result.json",
            {
                "schema": f"{schema_name}_result_v1",
                "source_revision": source["source_revision"],
                "attempt": file_receipt(attempt_path, label="attempt.json"),
                "imported_sources": imported,
                "assessment_count": 1,
                "verdict": verdict,
                "report": encoded_report,
            },
        )
        return verdict
    except Exception as error:
        write_json_once(
            output / "incomplete.json",
            {
                "schema": f"{schema_name}_incomplete_v1",
                "source_revision": source["source_revision"],
                "attempt": file_receipt(attempt_path, label="attempt.json"),
                "exception": type(error).__name__,
                "message": str(error),
                "report_if_serialized": encoded_report,
            },
        )
        raise
