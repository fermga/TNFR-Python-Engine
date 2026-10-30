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
assert scope["physical_mapping_readiness"] is None
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
assert len(old_sources[CHAPTER]) == scope["chapter_original_prefix_bytes"] == 159069
draft = (OUT / "finite-observation-section.md").read_text(encoding="utf-8").replace("\n", "\r\n").encode()
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
assert (ROOT / PROGRAM).read_bytes().count(b"<!-- research-activity: foundations -->") == 1

applied = read("applied-decision.json")
assert applied["chapter_prefix_preserved_bytes"] == 159069
assert applied["new_scientific_evaluations"] == 0
for key in ("chapter", "program"):
    verify(applied[key])
for key in ("drafts", "independent_reviews"):
    for item in applied[key]:
        verify(item)
for name in ("mathematical-review.json", "independent-review.json", "field-and-contract-review.json"):
    review = read(name)
    assert review["verdict"] == "PASS" and review["independent"] is True
    assert review["author"] == "/root" and review["reviewer"] != review["author"]
    assert review["reviewed_files"] == applied["drafts"]
    for item in review["reviewed_files"]:
        verify(item)
    for item in review["consulted_sources"]:
        if item["path"] in old_sources:
            source = old_sources[item["path"]]
            assert len(source) == item["bytes"] and sha256(source).hexdigest() == item["sha256"]
        else:
            verify(item)
    verify(review["receipt_writer"])
independent = read("independent-review.json")
assert independent["reviewer"] == "/root/finite_independent_review"
assert "did not derive, author or edit" in independent["contribution_disclosure"]

registry = json_loads((ROOT / REGISTRY).read_bytes())
old_registry = json_loads(old_sources[REGISTRY])


def normalize_registry(value):
    value = deepcopy(value)
    for requirement in value["requirements"].values():
        requirement.pop("independent_review")
        for item in requirement["evidence"]:
            item.pop("sha256")
    return value


assert normalize_registry(registry) == normalize_registry(old_registry)
assert registry["physical_mapping_readiness_review"] is None
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
            assert "does not enlarge this prerequisite approval" in value
            assert "physical mapping remains unauthorized" in value
        else:
            assert value == previous
    for item in review["evidence"]:
        verify(item)

prerequisites = read("prerequisite-validation.json")
assert prerequisites["passed"] is True
assert [check["exit_code"] for check in prerequisites["checks"]] == [0, 1]
assert prerequisites["physical_mapping_authorized"] is False
assert prerequisites["physical_mapping_readiness_review"] is None
verify(prerequisites["registry"])
for record in (prerequisites, docs):
    for check in record["checks"]:
        verify(check["log"])
for item in docs["validated_sources"]:
    verify(item)
assert "4/4 resolved; physical mapping blocked" in (OUT / "foundations.log").read_text(encoding="utf-8")
assert "missing final readiness review" in (OUT / "physical-mapping-block.log").read_text(encoding="utf-8")

required_final = ALLOWED | {
    (OUT / name).relative_to(ROOT).as_posix()
    for name in (
        "applied-decision.json", "mathematical-review.json", "independent-review.json",
        "field-and-contract-review.json", "prerequisite-validation.json",
        "documentation-validation.json", "rendered-links.json", "finalize.py",
        "operator_causality-review.json", "state_observation-review.json",
        "organizational_identity-review.json", "research_selection-review.json",
    )
}
assert len(required_final) == 15
payload = {
    "schema": "tnfr.native-finite-observation-final-maintenance-review.v1",
    "author": "/root",
    "reviewer": "/root/joint_review_controls",
    "independent": True,
    "verdict": "PASS",
    "reviewed_files": [receipt(ROOT / path) for path in sorted(required_final)],
    "scope": "Independent final maintenance, preservation, scope and evidence-association review. This receipt is not a new independent derivation or numerical evaluation of FO1-FO5.",
    "contribution_disclosure": "The reviewer previously audited the preceding joint scope decision and proposed observational safeguards. The reviewer did not derive, author or edit the FO proof or its new program fragments, and did not edit the maintained files during this final review. Mathematical proof assessment remains in independent-review.json and the separately delimited contributing reviews.",
    "assessments": {
        "preservation": "The baseline and prior final-state source inventory agree. All 4626 maintained paths remain present, with exactly three authorized changes and 4623 unchanged sources. HEAD and the dirty-path inventory remain unchanged. The previous artifact receipts and archived pre-edit sources verify.",
        "canonical_application": "The original 159069 chapter bytes are preserved exactly and followed by the reviewed FO fragment. PROGRAM differs only by the two exact reviewed fragments. All three proof reviews bind the same draft bytes, and their historical canonical consulted sources match the pre-edit archive.",
        "review_independence": "The recovery and field-bound contributors disclose their own contributions and delimit independent claims. The separate reviewer finite_independent_review declares no derivation or drafting contribution and records a complete FO1-FO5 review. This maintenance reviewer does not relabel that evidence as an additional reproof.",
        "prerequisite_scope": "The registry changes only evidence hashes and review associations. All four statuses, evidence paths and anchors and primitive/derived assessments remain unchanged. The exact approved prepared-scope assessment prefix is retained. FO is a separate conditional extension; no universal or physical readiness is added.",
        "observation_limits": "The proof and program retain a paired HH/DH consumer, actual prepared held capacities, full preparation errors, clock/readout requirements and conservative sufficient bounds. The scalar is not an autonomous state or arbitrary-law classifier, and the tiny budget is not practical precision or a necessity claim.",
        "next_task": "The sole next dependency is causal access to the same paired consumer, including the independent preparation, held reference and clock/readout assumptions. No mapping, external fit, unit assignment or automatic precision/executor campaign is authorized.",
        "validation": "The prerequisite check exits zero, while strict physical admission exits one because final readiness is absent. Five documentation checks exit zero, their source and log receipts verify, and rendered links report no failures. The finalizer's required evidence set and preservation checks were inspected; the finalizer was not run by this reviewer."
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
    "physical_mapping_authorized": False,
    "physical_mapping_readiness_review": None,
    "performed": {
        "metadata_and_source_audit": True,
        "new_scientific_evaluations": 0,
        "scientific_tests": 0,
        "trajectory_runs": 0,
        "historical_producer_replays": 0,
        "documentation_checks_rerun": 0,
        "canonical_edits": 0,
        "finalizer_runs": 0
    },
    "limitations": [
        "Hash and metadata checks establish source association and preservation, not scientific truth or empirical validation.",
        "The analytical proof is reviewed in the separate evidence-bound independent-review.json; this receipt does not claim a fresh proof verification.",
        "The prepared prerequisite scope remains unchanged and physical mapping remains blocked."
    ],
    "receipt_writer": receipt(Path(__file__).resolve())
}
write_json_once(OUT / "final-admission-review.json", payload)
print(receipt(OUT / "final-admission-review.json"))
