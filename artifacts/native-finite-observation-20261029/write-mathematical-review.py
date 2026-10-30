"""Serialize the completed mathematical review; no model is evaluated."""
from pathlib import Path
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DRAFTS = (
    "finite-observation-section.md",
    "program-introduction.md",
    "program-decision.md",
)
SOURCES = (
    "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md",
    "theory/nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md",
    "theory/RESEARCH_PROGRAM.md",
    "src/tnfr/research/artifact_io.py",
)

def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())

write_json_once(OUT / "mathematical-review.json", {
    "schema": "tnfr.native-finite-observation-mathematical-review.v1",
    "verdict": "PASS",
    "independent": True,
    "author": "/root",
    "reviewer": "/root/finite_recovery",
    "reviewed_files": [receipt(OUT / name) for name in DRAFTS],
    "consulted_sources": [receipt(ROOT / name) for name in SOURCES],
    "contribution_disclosure": (
        "Before these root-authored drafts, this reviewer supplied the one-bridge "
        "recovery admission constants and a coarse lower bound for K. The reviewer "
        "did not author or edit the submitted drafts. Review of FO1 verifies the "
        "root's adaptation of that contributed argument and is not represented "
        "as an independent derivation of those constants. The substantive "
        "independent mathematical review covers the other agent's holomorphic "
        "field, complex Picard, parameter-factorized Cauchy remainder and the "
        "root's finite margin and uncertainty contract. This is source-level "
        "proof review, not formal proof verification or independent empirical evidence."
    ),
    "proof_coverage": {
        "FO1_recovery_adaptation": (
            "The actual one-bridge graph has diameter five and maximum degree "
            "three. The pathwise spectral lower bound, acute r=1/16 ball, "
            "energy barrier 1/368640 and nominal a=eta=2^-20 admission are "
            "consistent. The full perturbation envelope retains common origins "
            "and the mean-one constraint. Dynamic recovery uses the native "
            "energy sublevel and positive capacity hull; capacity variance "
            "identifies the invariant zero-loss set without becoming an "
            "unnecessary entry threshold. Held recovery retains its actual profile."
        ),
        "FO2_holomorphic_field": (
            "For each relative exponential the complex edge perturbation has "
            "modulus at most 1/16, yielding the 1/15 edge error and 1/5 "
            "resultant error. Because each resultant or interpolated denominator "
            "lies in the disk centered at real C_i^*>1/2 with radius 1/5, "
            "its real part is strictly greater than 3/10, not merely its "
            "modulus. This explicitly justifies the principal logarithm "
            "branches and reciprocals on a neighborhood. On real states the "
            "log difference is the native Arg source and the integral is "
            "the native inverse phase metric, including S=0. The log and "
            "inverse bounds give |g|<2/9 and |h|<10/9. All complete field "
            "rows, including dynamic capacity and held capacity as an initial "
            "coordinate, satisfy the stated M=5r bound."
        ),
        "FO2_picard_and_cauchy": (
            "The closed complex parameter family begins within r/4. Picard "
            "iterates stay within r/4+R*M=7r/8 for R=1/8. A finite complex "
            "Lipschitz bound on that compact convex inner cube supplies "
            "factorial successive-difference convergence; R*L<1 is not "
            "needed. The uniform convergence gives joint holomorphy in "
            "a, eta and time. Vanishing at a=0 and eta=0 and the exact JO "
            "zero time orders through two justify i,j>=1 and k>=3. The "
            "Cauchy coefficient bound and the parameter geometric sums "
            "contribute 4|a*eta|/p^2; the fourth-derivative time sum is "
            "24*R^-4/(1-t/R)^5. With p^-2*R^-4=2^26 and "
            "(8/7)^5<2, the asserted fourth-derivative bound and the "
            "2^26|a*eta|T^4 Taylor remainder follow."
        ),
        "FO3_nominal_margin": (
            "The K lower bound is correct. Since eta<=1/256 in particular "
            "gives 1-eta>3/4, the cubic magnitude coefficient exceeds "
            "2^-9*a*eta. At T=2^-36 the remainder coefficient is 2^-10 "
            "times a*eta*T^3, so the negative nominal contrast is bounded "
            "away from zero by S=2^-158 at the selected preparation. "
            "This is an analytical lower bound and does not claim a "
            "sampled signal or practical precision."
        ),
        "FO4_full_state_error_transport": (
            "Substitution in the complete Jacobian row bounds with N=2, "
            "d<=3, X=1/32 and m=1/4 gives L_x<19 and L_theta<21; L=32 "
            "therefore covers both full fields. The field bound gives "
            "|y'|<1. All compared trajectories and straight state "
            "segments remain in the real cube through H=2T, and "
            "exp(LH)<2. Preparation error, elapsed-time error and scalar "
            "readout error consequently give less than 11*epsilon per "
            "trial. The shared errors are conservatively retained rather "
            "than cancelled; the held profile is the actually prepared one."
        ),
        "FO4_paired_threshold": (
            "At epsilon=2^-166 the paired error 22*epsilon equals "
            "(11/128)*S and is strictly below S/8. Held/held and "
            "dynamic/held output sets are therefore disjoint with "
            "threshold -S/2 and margin greater than 3S/8. The statement "
            "requires its admitted held reference and does not classify "
            "an isolated scalar sample. The clock and node-readout "
            "qualifications preserve the correct units and do not "
            "assume statistical independence of shared errors."
        ),
        "program_and_limits": (
            "The program correctly reports a conditional nonempty finite "
            "contract and selects causal access as the next dependency. "
            "The theorem does not authenticate preparation, capacity "
            "switching, a reference trial or instrument accuracy from "
            "the response. It does not transfer two-bridge selection, "
            "close a scalar state, claim attainable tolerance or permit "
            "physical mapping. The original prepared prerequisite "
            "scope and separate physical readiness requirement remain visible."
        ),
    },
    "blocking_findings": [],
    "limitations": [
        "FO1 recovery constants were contributed by this reviewer before drafting; independence is delimited above.",
        "No numerical trajectory, derivative evaluation, symbolic calculation program or historical producer was executed.",
        "The tiny sufficient uncertainty budget is a mathematical existence certificate, not an implementation or experimental accuracy claim.",
    ],
    "performed": {
        "static_source_reading": True,
        "analytical_review": True,
        "metadata_serialization_only": True,
        "scientific_evaluations": 0,
        "trajectory_runs": 0,
        "tests_executed": 0,
        "producer_replays": 0,
        "draft_edits": 0,
        "canonical_edits": 0,
    },
    "receipt_writer": receipt(OUT / "write-mathematical-review.py"),
})
print(receipt(OUT / "mathematical-review.json"))
