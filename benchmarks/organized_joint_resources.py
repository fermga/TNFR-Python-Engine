"""Evaluate the single declared joint-resource corner after committing its source.

This analytical instrument executes no trajectory, sweep or frozen producer.
An exclusive attempt record precedes the one assessment; existing evidence
blocks repetition. The declaration owns the meaning of an unavailable box.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.mathematics._rational_interval import (  # noqa: E402
    INTERVAL_BITS,
    INTERVAL_METHOD,
)
from tnfr.physics._organized_joint_resources import (  # noqa: E402
    _assess_organized_joint_resources,
)
from tnfr.research.artifact_io import (  # noqa: E402
    encode_exact_tree,
    file_receipt,
    sha256_bytes,
    write_json_once,
)
from tnfr.research.core_manifests import (  # noqa: E402
    current_git_source_provenance,
)

_DECLARATION = "theory/nodal/ORGANIZED_JOINT_RESOURCES.md"
_SOURCE_SCOPE = (
    "src/tnfr",
    "benchmarks/organized_joint_resources.py",
    _DECLARATION,
    "pyproject.toml",
)


def _corner():
    """Return the one prospectively declared primitive corner, without evaluation."""
    return dict(
        form_error_bounds=(Q(1, 10**6),) * 6,
        phase_error_bounds=(Q(1, 10**6),) * 6,
        quarter_dwell=Q(8 * 10**10),
        recovered_radius=Q(1, 10**8),
        decay_power=80,
        sequential_readout_error=Q(1, 10**14),
        contact_readout_error=Q(1, 10**9),
    )


def _git(*arguments):
    return subprocess.check_output(("git", *arguments), cwd=ROOT)


def _source_snapshot():
    revision, dirty, _ = current_git_source_provenance(ROOT, _SOURCE_SCOPE)
    if dirty:
        raise RuntimeError(
            "commit the declaration and complete source before evaluation"
        )
    listing = _git("ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", *_SOURCE_SCOPE)
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
    if _DECLARATION not in inventory:
        raise RuntimeError("the prospective declaration is not committed")
    return {
        "source_revision": revision,
        "source_scope": _SOURCE_SCOPE,
        "source_tree": _git("rev-parse", "HEAD:src").decode("ascii").strip(),
        "source_inventory": inventory,
        "source_inventory_sha256": sha256_bytes(listing),
    }


def _imported_sources():
    sources = {}
    for name, module in tuple(sys.modules.items()):
        if name == "tnfr" or name.startswith("tnfr."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(ROOT / "src"):
                    raise RuntimeError(
                        f"imported TNFR source is outside this tree: {name}"
                    )
                sources[name] = path.relative_to(ROOT).as_posix()
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh evidence directory"
    )
    arguments = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the declared repository root")
    output = arguments.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("evidence directory is not empty")
    source = _source_snapshot()
    imported = _imported_sources()
    corner = _corner()
    output.mkdir(parents=True, exist_ok=True)
    attempt_path = output / "attempt.json"
    write_json_once(
        attempt_path,
        {
            "schema": "organized_joint_resources_attempt_v1",
            **source,
            "declaration": _DECLARATION,
            "parameters": corner,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "arithmetic_method": INTERVAL_METHOD,
            "interval_bits": INTERVAL_BITS,
            "imported_sources": imported,
            "node_order": tuple(range(54)),
            "node_index_rule": "9*c+j; host c=0..4, external c=5; j=0..8",
            "support": "C9 square W6; 144 undirected unit edges",
            "clock": "tau=(1023/1024)*t; both rows have uniform quarter coefficient",
            "pressure_refresh": "live smooth sine in the complete analytical rows",
            "source_directions": 96,
            "seed": None,
            "maximum_assessments": 1,
            "evaluation_status_at_start": "not_evaluated",
        },
    )
    encoded_report = None
    try:
        report = _assess_organized_joint_resources(**corner)
        encoded_report = encode_exact_tree(asdict(report))
        imported = _imported_sources()
        if _source_snapshot() != source:
            raise RuntimeError("committed source changed during the assessment")
        verdict = (
            "joint_resource_box_admitted"
            if report.admitted
            else "specified_joint_resource_box_unavailable"
        )
        write_json_once(
            output / "result.json",
            {
                "schema": "organized_joint_resources_result_v1",
                "source_revision": source["source_revision"],
                "attempt": file_receipt(attempt_path, label="attempt.json"),
                "imported_sources": imported,
                "assessment_count": 1,
                "verdict": verdict,
                "report": encoded_report,
            },
        )
        print(verdict)
        return 0
    except Exception as error:
        write_json_once(
            output / "incomplete.json",
            {
                "schema": "organized_joint_resources_incomplete_v1",
                "source_revision": source["source_revision"],
                "attempt": file_receipt(attempt_path, label="attempt.json"),
                "exception": type(error).__name__,
                "message": str(error),
                "report_if_serialized": encoded_report,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
