"""Record an independent read-only mathematical scope review; no evaluation."""

from pathlib import Path

from tnfr.research.artifact_io import file_receipt, write_json_once


ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
drafts = (
    "joint-review-section.md",
    "program-introduction.md",
    "program-decision.md",
)
sources = (
    "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md",
    "theory/RESEARCH_PROGRAM.md",
    "theory/PHYSICAL_REGIME_CORRESPONDENCES.md",
    "theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md",
    "src/tnfr/research/artifact_io.py",
)


def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())


record = {
    "schema": "tnfr-native-joint-scope-mathematical-review-v1",
    "decision": "PASS",
    "author": "/root",
    "reviewer": "/root/joint_review_math",
    "review_date": "2026-10-29",
    "scope": "Mathematical and scope review of three proposed documentation fragments before canonical insertion.",
    "independence": {
        "distinct_author_and_reviewer": True,
        "reviewer_authored_reviewed_drafts": False,
        "reviewer_modified_reviewed_drafts": False,
        "reviewer_read_final_draft_bytes": True,
        "prior_scoping_feedback_disclosed": "Reviewer supplied scope analysis and suggested finite-error closure criteria before the author prepared these drafts.",
        "limits": "Independent source and scope review, not formal machine verification or an independent replay of the earlier scientific evidence.",
    },
    "reviewed_drafts": [receipt(DEST / name) for name in drafts],
    "consulted_source_receipts": [receipt(ROOT / name) for name in sources],
    "consulted_sections": {
        "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md": [
            "CT/JO complete law and JO1-JO12",
            "native-foundation-scope-decision",
            "DA1-DA3 and acquisition limits",
            "CS1-CS6 and selection controls",
            "W fixed witness scope",
            "RB1-RB7 and formal radius limits",
        ],
        "theory/RESEARCH_PROGRAM.md": [
            "Joint review active task and prohibition on automatic extensions",
            "Physical comparison separate admission",
        ],
        "theory/PHYSICAL_REGIME_CORRESPONDENCES.md": [
            "First card and corrections governing later cards",
            "Realizability and calibration P1-P3",
        ],
        "theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md": [
            "Coefficient and capacity identification and whole-law limits",
            "prepared-coefficient-identification",
            "coefficient-jet-uncertainty",
        ],
        "src/tnfr/research/artifact_io.py": [
            "file_receipt and write_json_once serialization contracts",
        ],
    },
    "findings": [
        "CT/JO prepared response retains one bridge (0,5), fixed kappa>0 and mean capacity b>0; its response jets are not transferred to two-bridge results.",
        "DA and CS retain the actual two-bridge graph and mean-one capacity; their existence claims hold for each fixed kappa>0 without uniformity at kappa=0.",
        "DA is winding-zero acquisition; CS starts with winding one and establishes matched-law retention versus loss, not another acquisition theorem.",
        "W and RB are restricted to kappa=1 and the exact represented mean-one preparation. Capture times 123/4 and 63/2 are not equilibrium times.",
        "RB has a positive symbolic sup-norm radius in all thirty coordinates on the twenty-nine-dimensional mean-one leaf; no practical tolerance or actual basin size is inferred.",
        "The scalar JO output equals chi_x. Its two initial receiver accelerations agree, and dynamic-minus-held cubic coefficient is kappa*eta*a*(eta-b)*K/6<0 under JO6.",
        "The selected next task requires an independently bounded regular horizon, full-law remainder and preparation/readout/clock errors. No finite separation margin or admissible numerical preparation is claimed already proved.",
        "Positivity eta<b is correctly distinguished from JO recovery-basin membership. The held law must retain its actual prepared capacity profile including errors.",
        "Shared state uncertainty and trial mismatch are distinguished; common quadratic-response mismatch may dominate the cubic discriminator.",
        "Full predictive state, constitutive alternatives, primitive chart and supplied support remain explicit. The scalar observation is not promoted to autonomous state.",
        "Prepared prerequisite approvals are not broadened, and the independent physical bridge and task-specific physical readiness remain absent.",
    ],
    "scientific_evaluations": 0,
    "trajectory_runs": 0,
    "producer_replays": 0,
    "tests_run": 0,
    "canonical_edits_by_reviewer": 0,
    "blocking_findings": [],
    "limitations": [
        "This receipt associates the inspected draft and source bytes; it does not authenticate historical acquisitions or rerun their producers.",
        "The proposed finite observation contract is an uncompleted next dependency, not a newly demonstrated margin or physical mapping.",
    ],
    "receipt_writer": receipt(Path(__file__).resolve()),
}
write_json_once(DEST / "mathematical-review.json", record)
print("PASS: mathematical-review.json written; scientific evaluations = 0")
