"""Physical work requires all four evidence-bound prerequisite closures."""

import json
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from scripts import check_documentation as documentation
from scripts import check_research_prerequisites as gate


def _write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": name, "sha256": sha256(data).hexdigest()}


def _save(root, registry):
    _write_json(root, gate.REGISTRY, registry)


def _program(root, activity="foundations"):
    (root / gate.PROGRAM).write_text(
        "# Research program\n\n<!-- research-activity: " + activity + " -->\n",
        encoding="utf-8",
    )


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "theory").mkdir()
    _program(tmp_path)
    registry = {
        "schema": "tnfr.research-prerequisites.v1",
        "requirements": {
            identifier: {"status": "open", "evidence": [], "independent_review": None}
            for identifier in gate.REQUIREMENTS
        },
        "physical_mapping_readiness_review": None,
    }
    _save(tmp_path, registry)
    return tmp_path, registry


def _resolve(root, registry, identifier):
    path = root / "theory" / (identifier + ".md")
    data = b"# Synthetic protocol fixture\n\n## Closure\n\nA reviewed fixture claim.\n"
    path.write_bytes(data)
    evidence = [
        {
            "path": path.relative_to(root).as_posix(),
            "anchor": "closure",
            "sha256": sha256(data).hexdigest(),
        }
    ]
    criteria = {
        "closure": "The declared closure criteria have been individually assessed."
    }
    if identifier in ("operator_causality", "state_observation"):
        criteria["primitive_or_derived"] = (
            "The assumed and derived statements are explicitly separated."
        )
    review = {
        "schema": "tnfr.research-prerequisite-review.v1",
        "requirement": identifier,
        "verdict": "approved",
        "independent": True,
        "author": "fixture author",
        "reviewer": "independent fixture reviewer",
        "criteria_assessment": criteria,
        "evidence": deepcopy(evidence),
    }
    receipt = _write_json(root, "reviews/" + identifier + ".json", review)
    registry["requirements"][identifier] = {
        "status": "resolved",
        "evidence": evidence,
        "independent_review": receipt,
    }


def _ready(root, registry):
    for identifier in gate.REQUIREMENTS:
        _resolve(root, registry, identifier)
    review = {
        "schema": "tnfr.physical-mapping-readiness.v1",
        "verdict": "approved",
        "independent": True,
        "author": "fixture author",
        "reviewer": "independent readiness reviewer",
        "program_sha256": sha256((root / gate.PROGRAM).read_bytes()).hexdigest(),
        "criteria_assessment": {
            "cross_pillar_consistency": "All four closures use compatible premises and evidence.",
            "physical_mapping_scope": "The permitted scope and limits were independently reviewed.",
        },
        "requirement_reviews": {
            identifier: registry["requirements"][identifier]["independent_review"][
                "sha256"
            ]
            for identifier in gate.REQUIREMENTS
        },
    }
    registry["physical_mapping_readiness_review"] = _write_json(
        root, "reviews/readiness.json", review
    )
    _save(root, registry)


def _review(root, entry):
    return json.loads((root / entry["independent_review"]["path"]).read_bytes())


def _replace_review(root, entry, review):
    entry["independent_review"] = _write_json(
        root, entry["independent_review"]["path"], review
    )


def test_open_foundations_and_historical_physics_reading_remain_allowed(workspace):
    root, _ = workspace
    (root / "theory/physics_history.md").write_text(
        "Historical physical hypotheses remain readable.", encoding="utf-8"
    )
    status = gate.validate_prerequisites(root)
    assert status.activity == "foundations"
    assert status.unresolved == gate.REQUIREMENTS
    assert status.physical_mapping_ready is False


@pytest.mark.parametrize("physical_request", ("marker", "flag"))
def test_each_explicit_physical_request_is_blocked_while_open(
    workspace, physical_request
):
    root, _ = workspace
    if physical_request == "marker":
        _program(root, "physical-mapping")
    with pytest.raises(gate.PrerequisiteError, match="physical mapping blocked"):
        gate.validate_prerequisites(
            root, require_physical_mapping=physical_request == "flag"
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing-id", "unknown-id", "unknown-field", "unknown-status", "wrong-schema"),
)
def test_registry_schema_fails_closed(workspace, mutation):
    root, registry = workspace
    if mutation == "missing-id":
        registry["requirements"].pop("state_observation")
    elif mutation == "unknown-id":
        registry["requirements"]["fifth_pillar"] = registry["requirements"][
            "state_observation"
        ]
    elif mutation == "unknown-field":
        registry["requirements"]["operator_causality"]["done"] = True
    elif mutation == "unknown-status":
        registry["requirements"]["organizational_identity"]["status"] = "documented"
    else:
        registry["schema"] = "future-or-unknown"
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError):
        gate.validate_prerequisites(root)


def test_duplicate_json_keys_are_rejected(workspace):
    root, _ = workspace
    (root / gate.REGISTRY).write_text(
        '{"requirements": {}, "requirements": {}}', encoding="utf-8"
    )
    with pytest.raises(gate.PrerequisiteError, match="invalid strict JSON"):
        gate.validate_prerequisites(root)


@pytest.mark.parametrize(
    "marker",
    (
        "",
        "<!-- research-activity: physical -->",
        "<!-- research-activity: foundations -->\n<!-- research-activity: physical-mapping -->",
        "```markdown\n<!-- research-activity: foundations -->\n```",
    ),
)
def test_missing_ambiguous_or_example_only_activity_fails_closed(workspace, marker):
    root, _ = workspace
    (root / gate.PROGRAM).write_text("# Program\n" + marker + "\n", encoding="utf-8")
    with pytest.raises(gate.PrerequisiteError):
        gate.validate_prerequisites(root)


def test_navigation_tail_and_fenced_example_do_not_override_authored_activity(
    workspace,
):
    root, _ = workspace
    path = root / gate.PROGRAM
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n```markdown\n<!-- research-activity: physical-mapping -->\n```\n<!-- BEGIN RESEARCH CHAPTER ROUTES -->\n<!-- research-activity: physical-mapping -->\n",
        encoding="utf-8",
    )
    assert gate.validate_prerequisites(root).activity == "foundations"


def test_documentation_completion_and_status_flip_cannot_resolve_a_pillar(workspace):
    root, registry = workspace
    registry["requirements"]["operator_causality"]["status"] = "resolved"
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError, match="resolved without evidence"):
        gate.validate_prerequisites(root)


@pytest.mark.parametrize(
    "path",
    (
        "../outside.md",
        "/absolute.md",
        "C:/outside.md",
        "https://example.test/proof.md",
        "theory/../proof.md",
        "theory\\proof.md",
    ),
)
def test_evidence_paths_cannot_escape_or_fetch_remote_sources(workspace, path):
    root, registry = workspace
    _resolve(root, registry, "operator_causality")
    registry["requirements"]["operator_causality"]["evidence"][0]["path"] = path
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError, match="unsafe repository-relative path"):
        gate.validate_prerequisites(root)


def test_linked_evidence_path_is_rejected_before_reading(workspace, monkeypatch):
    root, registry = workspace
    _resolve(root, registry, "operator_causality")
    _save(root, registry)
    actual = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name == "operator_causality.md" or actual(path),
    )
    with pytest.raises(gate.PrerequisiteError, match="linked evidence paths"):
        gate.validate_prerequisites(root)


@pytest.mark.parametrize(
    "failure",
    (
        "missing-file",
        "missing-anchor",
        "stale-proof",
        "updated-proof-unreviewed",
        "stale-review",
    ),
)
def test_missing_or_changed_evidence_invalidates_its_old_approval(workspace, failure):
    root, registry = workspace
    _resolve(root, registry, "operator_causality")
    entry = registry["requirements"]["operator_causality"]
    evidence = entry["evidence"][0]
    path = root / evidence["path"]
    if failure == "missing-file":
        path.unlink()
    elif failure == "missing-anchor":
        evidence["anchor"] = "absent"
    elif failure in ("stale-proof", "updated-proof-unreviewed"):
        path.write_bytes(path.read_bytes() + b"Changed premises.\n")
        if failure == "updated-proof-unreviewed":
            evidence["sha256"] = sha256(path.read_bytes()).hexdigest()
    else:
        review_path = root / entry["independent_review"]["path"]
        review_path.write_bytes(review_path.read_bytes() + b"\n")
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError):
        gate.validate_prerequisites(root)


@pytest.mark.parametrize(
    "field,value",
    (
        ("independent", False),
        ("independent", "true"),
        ("verdict", "pending"),
        ("reviewer", "fixture author"),
        ("requirement", "state_observation"),
        (
            "criteria_assessment",
            {"closure": "Explained, but primitive/derived was omitted."},
        ),
    ),
)
def test_approval_requires_scoped_independent_and_explicit_criteria(
    workspace, field, value
):
    root, registry = workspace
    _resolve(root, registry, "operator_causality")
    entry = registry["requirements"]["operator_causality"]
    review = _review(root, entry)
    review[field] = value
    _replace_review(root, entry, review)
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError):
        gate.validate_prerequisites(root)


def test_one_reviewed_closure_allows_foundational_work_but_not_physical_work(workspace):
    root, registry = workspace
    _resolve(root, registry, "operator_causality")
    _save(root, registry)
    assert gate.validate_prerequisites(root).unresolved == gate.REQUIREMENTS[1:]
    with pytest.raises(gate.PrerequisiteError, match="physical mapping blocked"):
        gate.validate_prerequisites(root, require_physical_mapping=True)


def test_premature_readiness_is_invalid_even_in_foundations_mode(workspace):
    root, registry = workspace
    registry["physical_mapping_readiness_review"] = {
        "path": "reviews/premature.json",
        "sha256": "0" * 64,
    }
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError, match="premature physical readiness"):
        gate.validate_prerequisites(root)


def test_four_closed_pillars_still_need_a_final_readiness_review(workspace):
    root, registry = workspace
    for identifier in gate.REQUIREMENTS:
        _resolve(root, registry, identifier)
    _save(root, registry)
    assert gate.validate_prerequisites(root).unresolved == ()
    with pytest.raises(gate.PrerequisiteError, match="missing final readiness review"):
        gate.validate_prerequisites(root, require_physical_mapping=True)


def test_complete_review_chain_allows_physical_mode_and_is_read_only(workspace):
    root, registry = workspace
    _program(root, "physical-mapping")
    _ready(root, registry)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    status = gate.validate_prerequisites(root, require_physical_mapping=True)
    assert status.physical_mapping_ready and status.unresolved == ()
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("change", ("question", "mode", "navigation"))
def test_changed_program_requires_a_new_readiness_review(workspace, change):
    root, registry = workspace
    _ready(root, registry)
    assert gate.validate_prerequisites(root, require_physical_mapping=True)
    path = root / gate.PROGRAM
    if change == "mode":
        _program(root, "physical-mapping")
    else:
        extra = "\nA different active physical question.\n"
        if change == "navigation":
            extra = "\n<!-- BEGIN RESEARCH CHAPTER ROUTES -->\nChanged routes.\n"
        path.write_bytes(path.read_bytes() + extra.encode("utf-8"))
    with pytest.raises(
        gate.PrerequisiteError,
        match="readiness does not bind the current research program",
    ):
        gate.validate_prerequisites(root, require_physical_mapping=True)


def test_readiness_must_bind_replacement_pillar_reviews(workspace):
    root, registry = workspace
    _ready(root, registry)
    entry = registry["requirements"]["state_observation"]
    review = _review(root, entry)
    review["criteria_assessment"][
        "closure"
    ] += " Additional reviewed domain restriction."
    _replace_review(root, entry, review)
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError, match="readiness does not bind"):
        gate.validate_prerequisites(root)


@pytest.mark.parametrize(
    "bad_readiness",
    (
        "stale-hash",
        "false-attestation",
        "missing-pillar",
        "missing-criteria",
        "missing-program",
    ),
)
def test_final_readiness_itself_is_checked(workspace, bad_readiness):
    root, registry = workspace
    _ready(root, registry)
    receipt = registry["physical_mapping_readiness_review"]
    path = root / receipt["path"]
    if bad_readiness == "stale-hash":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        review = json.loads(path.read_bytes())
        if bad_readiness == "false-attestation":
            review["independent"] = False
        elif bad_readiness == "missing-pillar":
            review["requirement_reviews"].pop("research_selection")
        elif bad_readiness == "missing-program":
            review.pop("program_sha256")
        else:
            review["criteria_assessment"].pop("cross_pillar_consistency")
        registry["physical_mapping_readiness_review"] = _write_json(
            root, receipt["path"], review
        )
    _save(root, registry)
    with pytest.raises(gate.PrerequisiteError):
        gate.validate_prerequisites(root, require_physical_mapping=True)


def test_documentation_gate_blocks_before_generation_or_examples(
    workspace, monkeypatch
):
    root, _ = workspace
    _program(root, "physical-mapping")
    monkeypatch.setattr(documentation, "REPO_ROOT", root)
    monkeypatch.setattr(sys, "argv", ["check_documentation.py", "--write-generated"])
    reached = []
    monkeypatch.setattr(
        documentation, "update_glossary_index", lambda: reached.append("generation")
    )
    monkeypatch.setattr(
        documentation, "check_documented_examples", lambda: reached.append("examples")
    )
    with pytest.raises(gate.PrerequisiteError, match="physical mapping blocked"):
        documentation.main()
    assert reached == []


def test_cli_is_fail_closed_even_under_optimization(workspace):
    root, _ = workspace
    command = [sys.executable, "-O", str(Path(gate.__file__)), "--root", str(root)]
    ordinary = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert ordinary.returncode == 0 and "0/4 resolved" in ordinary.stdout
    blocked = subprocess.run(
        command + ["--require-physical-mapping"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert blocked.returncode == 1 and "physical mapping blocked" in blocked.stderr
