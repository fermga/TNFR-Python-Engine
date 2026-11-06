"""A conditional phase-image restriction for the two-call Mutation study.

This instrument checks coefficients of an analytic theorem. It does not
evaluate phase preparations or prove the theorem by testing constants.
For five angles on one real lift of width less than pi, write their phasor
sum as S. A selected pair has sum a*exp(i*m), with a=2*cos(half_gap)>0.
If |S|>=3, projection of that pair onto the direction of S is at least
|S|-3>=0, because the three remaining unit phasors contribute at most 3.
Pair averaging increases a and consequently cannot decrease |S|. The
resultant threshold R>=3/5 and the common hull are preserved by each step.

Every doubly stochastic averaging of a real vector can be obtained, up to
permutation, by at most four such pair transfers. For completeness, sort
source x and target y, where y is majorized by x. If y[0]!=x[0], choose k
with x[k]>=y[0]>=x[k+1], replace that pair by y[0] and
x[k]+x[k+1]-y[0], then permute y[0] first. The residual lies between the
original pair. For r<k+1 its largest r entries each dominate y[0]; for
r>=k+1 their sum is the original largest r+1 sum minus y[0]. The residual
therefore majorizes the remaining target. Induction completes the transfer
construction. Doubly stochasticity gives the starting majorization because
the sum of any k output entries has input weights in [0,1] summing to k.

The ideal C5 coordinator is (1-G)*B*theta plus a common origin, with B
doubly stochastic. Its final contraction also increases R squared, since
each absolute pair gap is below pi and decreases. Thus the actual first
stable branch gives R_after>=R_hi whenever R_hi>=3/5. A later one-block
Mutation of magnitude s changes the normalized resultant by at most
2*abs(sin(s/2))/5<=s/5. If R_hi-s/5>R_lo and the actual disruptor load is
low, a second dissonant branch is impossible.

These statements concern ideal coherent lifts and exact represented law
coefficients. They assert neither global low-resultant monotonicity nor
bit-exact wrapped runtime equivariance. All pressure, grammar, history,
capacity and source-existence obligations remain conditional.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..config.constants import DISRUPTORS
from ._native_mutation_feasibility import (
    ExecutionPremises,
    NativeMutationLaw,
    _gains,
    _owned_law,
)

Q = Fraction


@dataclass(frozen=True, slots=True)
class MutationSourceImageRestriction:
    """Owned coefficient evidence for the stated conditional phase theorem."""

    law: NativeMutationLaw
    execution: ExecutionPremises
    block_count: int
    source_lift_width_ceiling: Fraction
    averaging_resultant_threshold: Fraction
    first_stable_gains: tuple[Fraction, Fraction]
    normalized_phase_matrix: tuple[tuple[Fraction, ...], ...]
    contraction: Fraction
    source_resultant_lower: Fraction
    mutation_resultant_loss_bound: Fraction
    successor_resultant_lower: Fraction
    dissonant_threshold: Fraction
    strict_gap: Fraction
    disruptor_loads: tuple[Fraction, Fraction]
    excluded_branch_pair: tuple[str, str]
    status: str
    obligations: tuple[str, ...]


def derive_native_mutation_source_image() -> MutationSourceImageRestriction:
    """Rebuild default coefficients without accepting sources or reports.

    A result associates its fresh law and execution premises with one
    conditional exclusion. Unsupported coefficient domains raise instead
    of returning an unchanged or cached verdict. No old cosine certificate
    is evaluated or consumed here.
    """
    law, execution = _owned_law()
    count = 5
    threshold = Q(count - 2, count)
    policy = law.phase_policy
    gains = _gains(law, law.phase_gains, "stable")
    global_gain, local_gain = gains
    contraction = 1 - global_gain
    diagonal = contraction - local_gain
    if not (0 < contraction < 1 and 0 <= local_gain <= contraction):
        raise ValueError("the first coordinator leaves the positive averaging domain")
    matrix = tuple(
        tuple(
            (
                diagonal
                if i == j
                else local_gain / 2 if (i - j) % count in (1, count - 1) else Q(0)
            )
            / contraction
            for j in range(count)
        )
        for i in range(count)
    )
    if policy.R_hi < threshold:
        raise ValueError("the stable threshold leaves the high-resultant theorem")
    loss = law.mutation_shift / count
    successor_lower = policy.R_hi - loss
    gap = successor_lower - policy.R_lo
    if gap <= 0:
        raise ValueError("the owned Mutation budget does not exclude dissonance")
    disruptive = {getattr(code, "value", code) for code in DISRUPTORS}
    loads = tuple(
        Q(
            sum(code in disruptive for trace in traces for code in trace),
            sum(map(len, traces)),
        )
        for traces in execution.traces[1:]
    )
    if loads != (Q(1, 20), Q(2, 25)) or max(loads) > policy.disr_lo:
        raise ValueError("the declared low-load controller context changed")
    return MutationSourceImageRestriction(
        law,
        execution,
        count,
        Q(3),
        threshold,
        gains,
        matrix,
        contraction,
        policy.R_hi,
        loss,
        successor_lower,
        policy.R_lo,
        gap,
        loads,
        ("stable", "dissonant"),
        "conditionally_excluded",
        (
            "the same source actually selects the first stable controller branch",
            "paired primitive phases share a source lift of width at most 3 < pi",
            "first OZ/IL events preserve phase; actual C5 phase averaging uses these gains",
            "only the target block changes primitive phase before the second controller",
            "both actual IL masks, source pressure transport, rails and acquired history remain joint",
            "conditional branch exclusion neither admits a source nor establishes complete renewal",
            "ideal common-lift theorem; absolute telemetry and binary64 wrapping remain separate",
        ),
    )
