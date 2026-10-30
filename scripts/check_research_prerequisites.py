#!/usr/bin/env python3
"""Guard declared physical-mapping work behind four reviewed foundations.

This is a workflow gate, not a proof checker or identity authenticator. Review
independence is an explicit attestation; hashes bind its reviewed evidence.
Final readiness binds the exact current program bytes, including task and mode.
Undeclared work and deliberate bypasses require repository review discipline.
The research program remains the sole task queue.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "src"):
    if str(directory) in sys.path:
        sys.path.remove(str(directory))
    sys.path.insert(0, str(directory))

from scripts.verify_internal_references import _anchors, _markdown_lines
from tnfr.utils.io import json_loads

REGISTRY = "theory/research_prerequisites.json"
PROGRAM = "theory/RESEARCH_PROGRAM.md"
REQUIREMENTS = (
    "operator_causality",
    "state_observation",
    "organizational_identity",
    "research_selection",
)
MAX_JSON_BYTES = 256 * 1024
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
_MARKER = re.compile(r"<!-- research-activity: (foundations|physical-mapping) -->")
_SHA = re.compile(r"[0-9a-f]{64}")


class PrerequisiteError(ValueError):
    """Missing, ambiguous or stale prerequisite information."""


@dataclass(frozen=True)
class PrerequisiteStatus:
    """Validated declarations, without a scientific or authentication seal."""

    activity: str
    unresolved: tuple[str, ...]
    physical_mapping_ready: bool


def _require(condition: bool, message: str) -> None:
    # These checks must remain active under python -O.
    if not condition:
        raise PrerequisiteError(message)


def _keys(value, expected, label):
    _require(
        isinstance(value, dict) and set(value) == set(expected),
        f"{label}: expected exactly {', '.join(expected)}",
    )


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()), f"{label}: missing text")
    return value.strip()


def _local_file(root, name):
    _text(name, "local evidence path")
    _require(
        name == name.strip()
        and not any(char in name for char in ("\\", ":", "\x00"))
        and not any(part in ("", ".", "..") for part in name.split("/"))
        and not PurePosixPath(name).is_absolute(),
        f"unsafe repository-relative path: {name}",
    )
    path = root / name
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        _require(
            not current.is_symlink()
            and not (hasattr(current, "is_junction") and current.is_junction()),
            f"linked evidence paths are not admitted: {name}",
        )
    _require(path.resolve().is_relative_to(root), f"path escapes repository: {name}")
    _require(path.is_file(), f"missing local evidence file: {name}")
    return path


def _bytes(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    _require(len(data) <= limit, f"file exceeds {limit}-byte admission: {path}")
    return data


def _json(data, label):
    try:
        return json_loads(data)
    except (ValueError, UnicodeError) as exc:
        raise PrerequisiteError(f"{label}: invalid strict JSON: {exc}") from exc


def _receipt(root, value, *, evidence=False):
    expected = ("path", "anchor", "sha256") if evidence else ("path", "sha256")
    _keys(value, expected, "evidence receipt" if evidence else "review receipt")
    digest = value["sha256"]
    _require(isinstance(digest, str) and _SHA.fullmatch(digest), "invalid SHA-256")
    path = _local_file(root, value["path"])
    _require(
        path.suffix == (".md" if evidence else ".json"), f"wrong evidence type: {path}"
    )
    data = _bytes(path, MAX_EVIDENCE_BYTES if evidence else MAX_JSON_BYTES)
    _require(
        sha256(data).hexdigest() == digest, f"stale evidence hash: {value['path']}"
    )
    if evidence:
        anchor = _text(value["anchor"], "evidence anchor")
        _require(anchor == value["anchor"], "evidence anchor has whitespace")
        try:
            document = data.decode("utf-8")
        except UnicodeError as exc:
            raise PrerequisiteError(f"non-UTF-8 evidence: {value['path']}") from exc
        _require(
            anchor in _anchors(path, document),
            f"missing evidence anchor: {value['path']}#{anchor}",
        )
    return data


def _approval(review, criteria):
    _require(review["verdict"] == "approved", "review verdict is not approved")
    _require(review["independent"] is True, "review lacks independent attestation")
    author = _text(review["author"], "review author")
    reviewer = _text(review["reviewer"], "independent reviewer")
    _require(
        author.casefold() != reviewer.casefold(), "reviewer must differ from author"
    )
    _keys(review["criteria_assessment"], criteria, "review criteria assessment")
    for criterion in criteria:
        _text(review["criteria_assessment"][criterion], criterion)


def _requirement_review(root, identifier, entry):
    review = _json(_receipt(root, entry["independent_review"]), identifier + " review")
    _keys(
        review,
        (
            "schema",
            "requirement",
            "verdict",
            "independent",
            "author",
            "reviewer",
            "criteria_assessment",
            "evidence",
        ),
        identifier + " review",
    )
    _require(
        review["schema"] == "tnfr.research-prerequisite-review.v1",
        "unknown prerequisite review schema",
    )
    _require(
        review["requirement"] == identifier, "review belongs to another prerequisite"
    )
    _require(
        review["evidence"] == entry["evidence"],
        "review does not bind the current evidence",
    )
    criteria = ["closure"]
    if identifier in ("operator_causality", "state_observation"):
        criteria.append("primitive_or_derived")
    _approval(review, criteria)


def _activity(root):
    path = _local_file(root, PROGRAM)
    data = _bytes(path, 96 * 1024)
    try:
        document = data.decode("utf-8")
    except UnicodeError as exc:
        raise PrerequisiteError("research program is not UTF-8") from exc
    # Exclude examples in code fences and the generated navigation tail.
    document = document.split("<!-- BEGIN RESEARCH CHAPTER ROUTES -->", 1)[0]
    declarations = [
        line.strip()
        for line, prose in _markdown_lines(document)
        if prose and "research-activity" in line
    ]
    _require(
        len(declarations) == 1, "program requires exactly one research-activity marker"
    )
    match = _MARKER.fullmatch(declarations[0])
    _require(
        match is not None, "program activity must be foundations or physical-mapping"
    )
    return match.group(1), sha256(data).hexdigest()


def validate_prerequisites(root=REPO_ROOT, *, require_physical_mapping=False):
    """Validate registry/provenance and deny physical work before reviewed closure."""
    root = Path(root).resolve()
    activity, program_sha256 = _activity(root)
    registry = _json(_bytes(_local_file(root, REGISTRY), MAX_JSON_BYTES), REGISTRY)
    _keys(
        registry,
        ("schema", "requirements", "physical_mapping_readiness_review"),
        "prerequisite registry",
    )
    _require(
        registry["schema"] == "tnfr.research-prerequisites.v1",
        "unknown prerequisite registry schema",
    )
    _keys(registry["requirements"], REQUIREMENTS, "requirements")
    unresolved = []
    for identifier in REQUIREMENTS:
        entry = registry["requirements"][identifier]
        _keys(entry, ("status", "evidence", "independent_review"), identifier)
        _require(
            entry["status"] in ("open", "resolved"), f"{identifier}: invalid status"
        )
        _require(
            isinstance(entry["evidence"], list),
            f"{identifier}: evidence must be a list",
        )
        if entry["status"] == "open":
            _require(
                entry["evidence"] == [] and entry["independent_review"] is None,
                f"{identifier}: open prerequisites cannot carry closure approvals",
            )
            unresolved.append(identifier)
            continue
        _require(bool(entry["evidence"]), f"{identifier}: resolved without evidence")
        seen = set()
        for evidence in entry["evidence"]:
            _receipt(root, evidence, evidence=True)
            key = evidence["path"], evidence["anchor"]
            _require(key not in seen, f"{identifier}: duplicate evidence")
            seen.add(key)
        _requirement_review(root, identifier, entry)
    readiness = registry["physical_mapping_readiness_review"]
    if readiness is not None:
        _require(
            not unresolved, "premature physical readiness: unresolved prerequisites"
        )
        review = _json(_receipt(root, readiness), "physical readiness review")
        _keys(
            review,
            (
                "schema",
                "verdict",
                "independent",
                "author",
                "reviewer",
                "criteria_assessment",
                "requirement_reviews",
                "program_sha256",
            ),
            "physical readiness review",
        )
        _require(
            review["schema"] == "tnfr.physical-mapping-readiness.v1",
            "unknown physical readiness review schema",
        )
        _require(
            review["program_sha256"] == program_sha256,
            "readiness does not bind the current research program",
        )
        expected = {
            identifier: registry["requirements"][identifier]["independent_review"][
                "sha256"
            ]
            for identifier in REQUIREMENTS
        }
        _require(
            review["requirement_reviews"] == expected,
            "readiness does not bind all current prerequisite reviews",
        )
        _approval(review, ("cross_pillar_consistency", "physical_mapping_scope"))
    ready = not unresolved and readiness is not None
    if activity == "physical-mapping" or require_physical_mapping:
        detail = (
            ", ".join(unresolved) if unresolved else "missing final readiness review"
        )
        _require(ready, "physical mapping blocked: " + detail)
    return PrerequisiteStatus(activity, tuple(unresolved), ready)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--require-physical-mapping", action="store_true")
    args = parser.parse_args(argv)
    try:
        status = validate_prerequisites(
            args.root, require_physical_mapping=args.require_physical_mapping
        )
    except (PrerequisiteError, OSError) as exc:
        print(f"Research prerequisite check failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Research prerequisites: {status.activity}; {4 - len(status.unresolved)}/4 resolved; physical mapping {'ready' if status.physical_mapping_ready else 'blocked'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
