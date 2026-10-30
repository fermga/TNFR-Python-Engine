"""Serialize the field author's transcription check and independent contract review.

Only file receipts and review metadata are written. No scientific calculation,
model evaluation, trajectory, test or historical producer is executed.
"""
from pathlib import Path
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DRAFTS = (
    "finite-observation-section.md",
    "program-introduction.md",
    "program-decision.md",
)

write_json_once(OUT / "field-and-contract-review.json", {
    "schema": "tnfr.native-finite-observation-field-contract-review.v1",
    "verdict": "PASS",
    "independent": True,
    "author": "/root",
    "reviewer": "/root/finite_jet_bounds",
    "reviewed_files": [
        file_receipt(OUT / name, label=(OUT / name).relative_to(ROOT).as_posix())
        for name in DRAFTS
    ],
    "independence_scope": {
        "contributed_derivation": "The reviewer supplied the holomorphic field bound, parameter/time analyticity, factorized Cauchy fourth-derivative bound and resulting nominal margin used in sections FO2-FO3.",
        "own_work_check": "The check of FO2-FO3 is a transcription and consistency check by their contributing analyst; it is not presented as independent mathematical verification.",
        "independent_review": "The reviewer independently inspected the root's concrete recovery admission in FO1, complete-state/clock/readout propagation and HH/DH classifier in FO4, abstention/access limits in FO5, and the two program fragments.",
        "no_draft_authorship": "The reviewer did not write or edit the three submitted drafts. Two wording clarifications were requested before association with their final bytes.",
        "whole_proof_review": "A separate reviewer must independently assess the complete proof, including FO2-FO3.",
    },
    "assessments": {
        "analytic_transcription": "The complex cube, native logarithm/integral extensions, complete vector-field bound, Picard interior estimate, three-variable Cauchy coefficient bound and a*eta factorization are transcribed consistently. The fourth-derivative bound applies to the nominal JO family; the full-state preparation errors are separately transported.",
        "recovery": "The one-bridge diameter-five path inequality yields the stated spectral lower bound. The actual target edge gaps, acute cosine bound and native energy guard are retained on the current support. The stated full-coordinate envelope stays inside both the recovery radius and energy barrier with positive mean-one capacities. This does not transfer two-bridge captures.",
        "sensitivity": "The full Jacobian bounds include capacity columns and phase-metric derivatives. The real continuation cube, common horizon, Gronwall factor and scalar observation norm support the declared preparation error radius. Every held trial retains its own actual initial capacity profile.",
        "timing_and_readout": "Elapsed-time errors are separated into common and mismatched components, with both evaluated times inside the admitted horizon. Scalar readout error is distinguished from individual node errors, and the clock-conversion example preserves scale, origin and jitter obligations. No probabilistic independence or automatic shared-error cancellation is assumed.",
        "paired_decision": "The individual output radii and their sum remain strictly below one eighth of the nominal separation. This gives disjoint HH and DH contrast intervals with the stated threshold and conservative margin. A held reference, rather than an unknown absolute nominal output, supplies the paired comparison. Other laws are not excluded.",
        "limits": "Blind controls and arbitrary full-state deviations are not assigned the nominal cubic factorization. The declared error budget is sufficient, conservative and below ordinary binary64 spacing for near-unit capacity; it is neither a practical precision claim nor a necessity bound. No physical bridge or autonomous scalar state is inferred.",
        "next_dependency": "The next task appropriately audits independently justified causal access to preparation, held reference, clock and readout. It preserves the proven conditional observation theorem and does not automatically start a control implementation, numerical precision campaign or physical mapping.",
    },
    "consulted_sources": [
        file_receipt(ROOT / name, label=name)
        for name in (
            "AGENTS.md",
            "theory/RESEARCH_PROGRAM.md",
            "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md",
            "src/tnfr/research/artifact_io.py",
        )
    ],
    "blocking_findings": [],
    "performed": {
        "source_reading": True,
        "analytical_derivation": True,
        "artifact_bookkeeping_only": True,
        "new_scientific_evaluations": 0,
        "new_trajectories": 0,
        "historical_producer_replays": 0,
        "tests_run": 0,
        "draft_edits": 0,
        "canonical_edits": 0,
    },
    "physical_mapping_authorized": False,
    "receipt_writer": file_receipt(
        Path(__file__), label=Path(__file__).relative_to(ROOT).as_posix()
    ),
})
print("field-and-contract-review.json written")
