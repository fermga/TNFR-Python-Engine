"""Serialize an independent analytical review; do not evaluate a model."""
from pathlib import Path
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DRAFTS = ("causal-access-section.md", "program-introduction.md", "program-decision.md")

def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())

payload = {
    "schema": "tnfr.native-causal-access-independent-review.v1",
    "author": "/root",
    "reviewer": "/root/access_independent_review",
    "independent": True,
    "verdict": "PASS",
    "reviewed_files": [receipt(OUT / name) for name in DRAFTS],
    "independence_scope": (
        "The reviewer did not derive the submitted control or observation results "
        "and did not author or modify the three drafts. This is an independent "
        "mathematical and scope review of their complete final formulation, not "
        "a formal machine proof, numerical replication or empirical validation."
    ),
    "assessments": {
        "sufficient_record_and_symmetries": (
            "The record is declared sufficient rather than minimal. Common form "
            "translation and common phase rotation are actual symmetries of the "
            "complete field and receiver observation. Only one of each is allowed "
            "per connected trial; relative inter-ring offsets are not removed. "
            "The 29-dimensional leaf and 27-dimensional quotient are not promoted "
            "to a minimal observation or calibration dimension."
        ),
        "uniform_preparation_and_exact_hold": (
            "On the retained connected support, ker(B)=span(1) makes nonuniform "
            "unforced holding impossible. The capacity row is independent of form "
            "and phase, so interventions confined to those coordinates cannot "
            "create contrast from a uniform capacity profile. This does not exclude "
            "registered actions that actually change capacity under their own laws."
        ),
        "finite_passive_extension": (
            "For a finite closed connected reciprocal linear diffusion, each "
            "capacity coordinate is analytic in time and all coordinates converge "
            "to a common limit. If original coordinates were individually constant "
            "on an open interval, analytic continuation would make them constant "
            "for all time, and their constants would have to coincide. This proves "
            "the stated exact-hold obstruction, not an inability of nonuniformly "
            "initialized auxiliaries to prepare transient capacity contrasts. "
            "Approximation, inputs, other laws, changing support and infinite "
            "environments remain outside the obstruction."
        ),
        "ideal_selector": (
            "The supplied fixed selector s with s_dot=0 specifies the complete "
            "mathematical family for held versus dynamic capacity without changing "
            "the other rows. Total and positivity are preserved and the variance "
            "derivative is -s*nu^T*B*nu/2. This supplies a constitutive access premise, "
            "not an autonomous selection law or a physically realized switch. "
            "Removing shared graph edges would generally alter other native rows."
        ),
        "compensating_input_and_reactions": (
            "The feedback u=B*nu/2 cancels the native row exactly. With B=G*G^T, "
            "opposite edge fluxes cancel pointwise and regionally. Their node sum "
            "vanishes and their variance contribution equals the native loss in "
            "magnitude. Neither that total balance nor the native storage, which "
            "has no explicit capacity term, establishes actuator cost or resources."
        ),
        "stored_reference": (
            "The stated solution rho+exp(-B*t/2)*(nu(0)-rho) solves the augmented "
            "constant-reference law. Exact holding is equivalent to "
            "B*(nu(0)-rho)=0, and equals actual-profile matching on a shared mean "
            "leaf. Markov sup-norm contraction gives the stated sufficient "
            "positivity and drift bounds. A reference mismatch changes the "
            "ongoing control law and is not automatically FO initial uncertainty."
        ),
        "phase_derivative": (
            "At the one-bridge prepared state, own-phase rotation gives "
            "partial_theta5 g5=-1/pi, while neighbors 6 and 9 each give 1/(2*pi). "
            "Their unit capacities and receiver weights therefore give "
            "partial_delta partial_t Y=-5/(8*pi). All initial form and capacity "
            "coordinates and Y(delta,0)=0 are preserved."
        ),
        "analytic_family_and_cauchy": (
            "For complex |delta|<=1/128, the fixed FO form/capacity plus the "
            "phase perturbation remain within the initial r/4 cube. The same "
            "full holomorphic field and Picard argument therefore give |Y|<=1/16 "
            "on |t|<=1/8. Cauchy coefficient bounds yield "
            "|b'(delta)-b'(0)|<=2^16|delta| and "
            "|partial_delta Y-T*b'(delta)|<=2^12*T^2. "
            "The parameter and time geometric-series constants are correct."
        ),
        "monotonicity_and_collision": (
            "At T=2^-36 and delta<=2^-110, the derivative error divided by T "
            "is at most 2^-94+2^-24<1/32. With 3<pi<4, "
            "-T/4<f'(delta)<-T/8 follows. FO and K<1/36 give "
            "Delta<=-2^-158 and |Delta|<9*2^-158<2^-154. "
            "The endpoint decrement at delta=2^-110 exceeds that magnitude, "
            "so continuity and monotonicity give a unique collision in this "
            "interval, with delta_* in [2^-120,2^-115). All constants, signs "
            "and strict/non-strict interval endpoints are consistent."
        ),
        "information_boundary": (
            "The mismatched held trial and nominal dynamic trial have equal "
            "initial form/capacity, initial receiver form and terminal receiver "
            "form, while the held reference is unchanged. The mismatch is "
            "inside recovery but outside the FO preparation budget. Thus it "
            "refutes authentication of hidden phase by those readings, not "
            "FO with authenticated premises or observability of a full trace. "
            "The initial rate changes; no whole-trajectory equivalence is claimed."
        ),
        "scope_and_next_task": (
            "The access audit retains FO as conditional and does not extend the "
            "original four prepared approvals to an autonomous actuator. The "
            "next task is an exact-scope admission review for a possible "
            "realization-feasibility question. No physical source search, "
            "dataset campaign, mapping or controller installation is authorized. "
            "The program explicitly requires a substantive exact-task readiness "
            "decision and strict guard before any admitted physical work."
        ),
    },
    "review_correction": (
        "The reviewer requested that CA5 distinguish the form/phase-only "
        "preparation obstruction from the passive-extension exact-hold "
        "obstruction, avoiding an unsupported claim that passive auxiliaries "
        "cannot prepare transient contrasts. The final wording was re-read "
        "before this receipt was serialized."
    ),
    "consulted_sources": [
        receipt(ROOT / path) for path in (
            "AGENTS.md", "theory/RESEARCH_PROGRAM.md",
            "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md",
            "theory/research_prerequisites.json",
            "src/tnfr/research/artifact_io.py",
        )
    ],
    "scope_record": receipt(OUT / "scope.json"),
    "clarification_record": receipt(OUT / "clarification-record.json"),
    "performed": {
        "read_only_analytical_review": True,
        "bookkeeping_only_execution": True,
        "new_scientific_evaluations": 0,
        "new_trajectories": 0,
        "tests_executed": 0,
        "historical_producer_replays": 0,
        "draft_edits": 0,
        "canonical_edits": 0,
    },
    "physical_mapping_authorized": False,
    "receipt_writer": receipt(Path(__file__).resolve()),
}
section = " ".join((OUT / DRAFTS[0]).read_text(encoding="utf-8").split())
assert "Form/phase-only initialization does not create the required capacity contrast" in section
assert "finite passive extension above does not provide exact nonuniform holding" in section
write_json_once(OUT / "independent-review.json", payload)
print(receipt(OUT / "independent-review.json"))
