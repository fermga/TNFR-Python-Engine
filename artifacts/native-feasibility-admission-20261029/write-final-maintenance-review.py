"""Independently audit documentation preservation and write a final review only."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from copy import deepcopy
import subprocess
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once, verify_archive_members

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ALLOWED = {"theory/RESEARCH_PROGRAM.md", "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md", "theory/research_prerequisites.json"}
CHAPTER = "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md"
PROGRAM = "theory/RESEARCH_PROGRAM.md"
REGISTRY = "theory/research_prerequisites.json"


def read(name):
    return json_loads((OUT / name).read_bytes())


def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())


def verify(item):
    actual = receipt(ROOT / item["path"])
    assert actual["sha256"] == item["sha256"], item["path"]
    if "bytes" in item:
        assert actual["bytes"] == item["bytes"], item["path"]


def stored_receipts(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for child in value.values():
            yield from stored_receipts(child)
    elif isinstance(value, list):
        for child in value:
            yield from stored_receipts(child)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


# Final approval is forbidden until all documentation checks have completed.
docs = read("documentation-validation.json")
links = read("rendered-links.json")
assert docs["passed"] is True and len(docs["checks"]) == 5
assert all(check["exit_code"] == 0 for check in docs["checks"])
assert not links["failures"] and links["pages"] > 0

scope = read("scope.json")
baseline = read("baseline.json")
assert set(scope["allowed_maintained_changes"]) == ALLOWED
assert set(baseline["allowed_changed_paths"]) == ALLOWED
assert scope["new_scientific_evaluations"] == scope["new_trajectories"] == scope["historical_producer_replays"] == 0
assert scope["external_source_searches"] == 0
assert git("rev-parse", "HEAD").strip() == baseline["head"]
assert git("status", "--porcelain=v1", "--untracked-files=all") == baseline["dirty_paths_before"]
names = {n for n in git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0") if n and (ROOT / n).is_file()}
original = baseline["source_sha256"]
assert len(names) == len(original) == 4626 and names == set(original)
current = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
changed = {name for name in names if current[name] != original[name]}
assert changed == ALLOWED
verify(baseline["prior_final_state"])
prior = json_loads((ROOT / baseline["prior_final_state"]["path"]).read_bytes())
assert prior["head"] == baseline["head"] and prior["source_sha256"] == original
historical = {item["path"]: item for item in stored_receipts(prior) if item["path"].startswith("artifacts/")}
for item in historical.values():
    verify(item)
verify_archive_members(OUT / "before-edit.zip", {name: original[name] for name in ALLOWED})
with ZipFile(OUT / "before-edit.zip") as archive:
    old_sources = {name: archive.read(name) for name in ALLOWED}
assert len(old_sources[CHAPTER]) == scope["chapter_original_prefix_bytes"] == 183408
draft = (OUT / "admission-section.md").read_text(encoding="utf-8").replace("\n", "\r\n").encode()
assert (ROOT / CHAPTER).read_bytes() == old_sources[CHAPTER] + draft
expected = old_sources[PROGRAM]
for start, end, name in (
    (b"## Continuation point:", b'<a id="foundational-pillars"></a>', "program-introduction.md"),
    (b"The [native foundation account]", b'<a id="foundation-refactoring-plan">', "program-decision.md"),
):
    assert expected.count(start) == 1
    left = expected.index(start)
    right = expected.index(end, left)
    fragment = (OUT / name).read_text(encoding="utf-8").replace("\n", "\r\n").encode()
    expected = expected[:left] + fragment + b"\r\n" + expected[right:]
assert (ROOT / PROGRAM).read_bytes() == expected
assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / ".github/agents/my-agent.md").read_bytes()
assert (ROOT / PROGRAM).stat().st_size <= 96 * 1024
assert len((ROOT / CHAPTER).read_bytes().splitlines()) <= 4000
assert (ROOT / PROGRAM).read_bytes().count(b"<!-- research-activity: physical-mapping -->") == 1

applied = read("applied-decision.json")
assert applied["chapter_prefix_preserved_bytes"] == 183408
assert applied["new_scientific_evaluations"] == 0
for key in ("chapter", "program"):
    verify(applied[key])
for key in ("drafts", "independent_reviews"):
    for item in applied[key]:
        verify(item)
for name in ("foundation-review-final.json", "protocol-review.json", "scope-review-final.json"):
    review = read(name)
    assert review["verdict"] == "PASS" and review["independent"] is True
    assert review["author"] == "/root" and review["reviewer"] != review["author"]
    assert review["reviewed_files"] == applied["drafts"]
    for item in review["reviewed_files"]:
        verify(item)
    for item in review.get("consulted_sources", review.get("source_associations", [])):
        if item["path"] in old_sources:
            source = old_sources[item["path"]]
            assert len(source) == item["bytes"] and sha256(source).hexdigest() == item["sha256"]
        else:
            verify(item)
    if "receipt_writer" in review:
        verify(review["receipt_writer"])

registry = json_loads((ROOT / REGISTRY).read_bytes())
old_registry = json_loads(old_sources[REGISTRY])


def normalize_registry(value):
    value = deepcopy(value)
    value.pop("physical_mapping_readiness_review")
    for requirement in value["requirements"].values():
        requirement.pop("independent_review")
        for item in requirement["evidence"]:
            item.pop("sha256")
    return value


assert normalize_registry(registry) == normalize_registry(old_registry)
assert registry["physical_mapping_readiness_review"] is not None
keys = {"schema", "requirement", "verdict", "independent", "author", "reviewer", "criteria_assessment", "evidence"}
for identifier, requirement in registry["requirements"].items():
    assert requirement["status"] == "resolved"
    verify(requirement["independent_review"])
    review = json_loads((ROOT / requirement["independent_review"]["path"]).read_bytes())
    assert set(review) == keys and review["schema"] == "tnfr.research-prerequisite-review.v1"
    assert review["requirement"] == identifier and review["verdict"] == "approved"
    assert review["independent"] is True and review["author"] != review["reviewer"]
    assert review["evidence"] == requirement["evidence"]
    old_item = old_registry["requirements"][identifier]["independent_review"]
    verify(old_item)
    old_review = json_loads((ROOT / old_item["path"]).read_bytes())
    assert [(e["path"], e["anchor"]) for e in review["evidence"]] == [(e["path"], e["anchor"]) for e in old_review["evidence"]]
    assert set(review["criteria_assessment"]) == set(old_review["criteria_assessment"])
    for key, value in review["criteria_assessment"].items():
        previous = old_review["criteria_assessment"][key]
        if key == "closure":
            assert value.split("The first ", 1)[0] == previous.split("The first ", 1)[0]
        else:
            assert value == previous
    for item in review["evidence"]:
        verify(item)

prerequisites = read("prerequisite-validation.json")
assert prerequisites["passed"] is True
assert [check["exit_code"] for check in prerequisites["checks"]] == [0, 0]
assert prerequisites["empirical_comparison_authorized"] is False
assert prerequisites["external_source_searches_performed"] == 0
verify(prerequisites["registry"])
for record in (prerequisites, docs):
    for check in record["checks"]:
        verify(check["log"])
for item in docs["validated_sources"]:
    verify(item)

required_final = ALLOWED | {
    (OUT / name).relative_to(ROOT).as_posix()
    for name in (
        "applied-decision.json", "foundation-review-final.json", "protocol-review.json",
        "scope-review-final.json", "prerequisite-validation.json",
        "documentation-validation.json", "rendered-links.json", "finalize.py",
        "operator_causality-review.json", "state_observation-review.json",
        "organizational_identity-review.json", "research_selection-review.json",
        "physical-readiness-review.json",
    )
}
assert len(required_final) == 16


# The unchanged checker and exact final readiness bind only the documentary task.
checker = "scripts/check_research_prerequisites.py"
assert current[checker] == original[checker] == "95829b94bb851690b733d16772a8a38dd94de51be04313a4312f15d1c3117e3c"
assert old_registry["physical_mapping_readiness_review"] is None
pre_gate = read("pre-admission-gate.json")
assert pre_gate["exit_code"] == 1
assert "missing final readiness review" in pre_gate["output"]
assert pre_gate["registry"]["sha256"] == sha256(old_sources[REGISTRY]).hexdigest()
verify(registry["physical_mapping_readiness_review"])
ready = read("physical-readiness-review.json")
assert set(ready) == {"schema", "verdict", "independent", "author", "reviewer", "criteria_assessment", "requirement_reviews", "program_sha256"}
assert ready["schema"] == "tnfr.physical-mapping-readiness.v1"
assert ready["verdict"] == "approved" and ready["independent"] is True
assert ready["author"] == "/root" and ready["reviewer"] != ready["author"]
assert set(ready["criteria_assessment"]) == {"cross_pillar_consistency", "physical_mapping_scope"}
assert ready["program_sha256"] == current[PROGRAM]
assert ready["requirement_reviews"] == {identifier: item["independent_review"]["sha256"] for identifier, item in registry["requirements"].items()}
assert registry["physical_mapping_readiness_review"]["path"] == (OUT / "physical-readiness-review.json").relative_to(ROOT).as_posix()
verify(prerequisites["readiness"])
assert prerequisites["readiness"]["sha256"] == registry["physical_mapping_readiness_review"]["sha256"]

# Both superseded reviews remain bound to their archived draft bytes.
clarification = read("clarification-record.json")
verify(clarification["archive"])
with ZipFile(ROOT / clarification["archive"]["path"]) as archive:
    for item in clarification["superseded_reviews"]:
        verify(item)
        old_review = json_loads((ROOT / item["path"]).read_bytes())
        for item in old_review["reviewed_files"]:
            data = archive.read(Path(item["path"]).name)
            assert len(data) == item["bytes"] and sha256(data).hexdigest() == item["sha256"]
    for name in ("admission-section.md", "program-introduction.md"):
        assert archive.read(name) == (OUT / name).read_bytes()
    earlier = archive.read("program-decision.md").decode("utf-8")
    insertion_point = "URL, family and access-based selection reason before detailed assessment.\n"
    insertion = "Record access date and version/DOI/revision/commit when available, and the\nsection or page supporting each extracted fact. Unknown metadata remain\nunknown rather than being inferred from another source.\n"
    assert earlier.count(insertion_point) == 1
    earlier = earlier.replace(insertion_point, insertion_point + insertion)
    before = "Stop when the selected cases are classified or either source/query limit\nis reached. Close the audit even if no candidate survives. Retain the source"
    after = "Stop new searches and document selections when either source/query limit\nis reached; finish classifying only the material already admitted, without\nextra retrievals or queries. Close the audit when those cases are classified,\neven if no candidate survives. Retain the source"
    assert earlier.count(before) == 1
    assert earlier.replace(before, after) == (OUT / "program-decision.md").read_text(encoding="utf-8")

payload = {
    "schema": "tnfr.native-feasibility-admission-final-maintenance-review.v1",
    "author": "/root",
    "reviewer": "/root/joint_review_controls",
    "independent": True,
    "verdict": "PASS",
    "reviewed_files": [receipt(ROOT / path) for path in sorted(required_final)],
    "scope": "Independent maintenance, evidence-association, preservation and authorization-boundary review of the final documentary feasibility admission. This is not a candidate physical assessment, empirical comparison or additional historical mathematical proof.",
    "contribution_disclosure": "The reviewer previously audited maintenance of the prepared, finite-observation and causal-access stages. The reviewer did not author or edit this admission decision, the program fragments, the four renewed approvals or the physical-readiness assessment. The substantive exact-task authorization is supplied by the independent physical-readiness review; this receipt audits its binding and maintained limits.",
    "assessments": {
        "preservation": "The baseline equals the preceding final-state source inventory. All 4626 maintained paths remain present, with exactly three authorized changes and 4623 unchanged sources. HEAD and dirty-path inventory are preserved. Prior immutable artifact receipts and the pre-edit archive verify.",
        "canonical_application": "The original 183408 chapter bytes are unchanged before the exact reviewed admission append. PROGRAM contains exactly the two reviewed replacement fragments, preserving every other byte. The three final reviews bind the same draft bytes, and historical canonical source associations match the pre-edit archive.",
        "clarification": "The archived old drafts and their superseded review receipts verify. Admission section and introduction are unchanged by clarification. The only two program edits add source/version/location provenance and distinguish stopping new acquisition from completing classification of already admitted material. Neither expands the three-family, six-document or six-query ceilings.",
        "prerequisites": "The four original prepared approval prefixes, primitive/derived assessments, statuses and evidence paths/anchors are unchanged. Source hashes and review associations are renewed. The only additional registry contract change is the separately reviewed exact-task readiness receipt; no autonomous physical actuator or experimental bridge is added to the prepared pillar scopes.",
        "strict_gate": "The prerequisite checker remains byte-identical to the baseline. The preserved pre-admission strict run failed for missing readiness. The final exact eight-field readiness record binds current PROGRAM bytes and the SHA-256 of all four current reviews. The registered receipt and validation receipt agree. Both final ordinary and strict checks pass. The checker verifies integrity, not semantic execution of documentary limits.",
        "limited_permission": "The readiness text and program authorize only the frozen primary-source access-feasibility audit: three ordered terrestrial apparatus families, at most one apparatus per family, six primary documents and six targeted queries. Inaccessible selected documents count. Source selection, provenance and supporting locations must be recorded, and new retrieval/search stops at either ceiling.",
        "excluded_work": "No physical realization, variable identification, empirical comparison, FO threshold on data, raw-response retrieval, fitting, simulation, controller installation or apparatus construction is admitted. The strongest positive outcome is a documented candidate for a separate bridge review. Missing documentation and failure to establish a sufficient native error budget are not universal physical exclusions.",
        "transition": "Physical activity now denotes the limited documentary task, not unrestricted physics. Any changed program, pillar review, broader budget, new consumer, bridge derivation or empirical follow-up requires a new substantive exact-task admission. The completed audit cannot authorize its own follow-up implicitly.",
        "validation": "Five documentation checks exit zero, all current source and log receipts verify, and rendered links contain no failures. The finalizer's exact sixteen-file evidence set was inspected. This reviewer did not execute the finalizer, external searches, model evaluations, trajectories, scientific tests or producer replays."
    },
    "verified_counts": {
        "maintained_files": len(names),
        "unchanged_files": len(names) - len(changed),
        "changed_paths": sorted(changed),
        "historical_artifact_receipts": len(historical),
        "reviewed_files": len(required_final),
        "documentation_checks": len(docs["checks"]),
        "rendered_pages": links["pages"],
        "rendered_fragment_links": links["local_fragment_links"]
    },
    "authorized_scope": "Only the frozen bounded primary-source access-feasibility documentary audit.",
    "empirical_comparison_authorized": False,
    "physical_readiness_binding": registry["physical_mapping_readiness_review"],
    "performed": {
        "metadata_and_source_audit": True,
        "external_source_searches": 0,
        "new_scientific_evaluations": 0,
        "scientific_tests": 0,
        "trajectory_runs": 0,
        "historical_producer_replays": 0,
        "documentation_checks_rerun": 0,
        "canonical_edits": 0,
        "checker_edits": 0,
        "finalizer_runs": 0
    },
    "limitations": [
        "Hash and metadata checks prove source association and preservation, not scientific truth or empirical validity.",
        "The independent readiness assessment supplies the substantive documentary-task admission; this maintenance receipt is not another physical study.",
        "Strict readiness does not authorize any empirical comparison or physical realization beyond the frozen documentary task."
    ],
    "receipt_writer": receipt(Path(__file__).resolve())
}
write_json_once(OUT / "final-admission-review.json", payload)
print(receipt(OUT / "final-admission-review.json"))
