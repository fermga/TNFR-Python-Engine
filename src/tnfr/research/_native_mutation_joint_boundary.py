"""Conditional joint IL-floor restriction for the two-call Mutation model.

Write h=L*theta and g=L*theta_first=C1*h on the admitted common lift, with
C1=(1-G)I+K*L. The shared capacities and their actual maximum V are unchanged
over both calls. Set m=(Si_high-beta-gamma)/alpha. At a surrounding block
with v_i=mV, high-Si Coherence (IL) requires

    (beta/pi)*abs(h_i) + gamma*rho_i <= 0.

Thus h_i=p_i=0, including the actual zero-peak convention. At the second
boundary the same reasoning gives g_i=q_i=0. Since K>0, also (L*h)_i=0.
The exact matrices below reconstruct any zero-sum C5 vector h from these
four observations at two distinct indices. Both possible cyclic distances
are injective. Two saturated surrounding blocks therefore force h=0 and
g=0, contradicting the required strictly outward target phase gap.

For the admitted positive multiplicative OZ branch p0>source_tiny, the
same-source transport is q=p+dt*k*L*diag(v)*F1*p+d*(g-h). The literal tiny-
pressure fallback is excluded from this current identity. At a single
saturated surrounding block i, p_i=q_i=h_i=g_i=0 forces
(L*diag(v)*F1*p)_i=0. If i=1 or4, its other neighbor j=2 or3 obeys

    f*u*p0 + ell*v_j*p_j = 0,

where f is the Dissonance (OZ) pressure factor and ell the Coherence (IL)
factor. Hence p_j<0 when the actual acquired source has u>0,p0>0.

This no-argument coefficient instrument supplies a conditional theorem,
not a source verdict, selector, solver or trajectory. It does not assert
that a remaining capacity/phase configuration exists. Primitive phase is
circular; the identities here retain the declared ideal common-lift scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._native_mutation_feasibility import (
    BRANCHES,
    ExecutionPremises,
    NativeMutationLaw,
    _gains,
    _owned_law,
)

Q = Fraction
ExactMatrix = tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True, slots=True)
class GapReconstruction:
    """Map (h0, h_distance, (Lh)0, (Lh)_distance) to zero-sum h."""

    distance: int
    observation_nodes: tuple[int, int]
    matrix: ExactMatrix


@dataclass(frozen=True, slots=True)
class JointBoundaryGainRow:
    first_branch: str
    gains: tuple[Fraction, Fraction]
    phase_matrix: ExactMatrix
    # (Lh)_i = recovery[0]*h_i + recovery[1]*g_i.
    laplacian_recovery: tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class JointBoundaryRestriction:
    law: NativeMutationLaw
    execution: ExecutionPremises
    capacity_floor: Fraction
    laplacian: ExactMatrix
    reconstructions: tuple[GapReconstruction, GapReconstruction]
    first_gain_rows: tuple[JointBoundaryGainRow, ...]
    # (saturated surrounding index, its non-target neighbor).
    adjacent_current_balances: tuple[tuple[int, int], ...]
    obligations: tuple[str, ...]
    scope: str = "conditional two-boundary IL-floor theorem; no source verdict"


def derive_native_mutation_joint_boundary() -> JointBoundaryRestriction:
    """Rebuild the owned law and its finite exact observation identities.

    No primitive source or cached report is accepted. The two distance
    classes and three first-gain rows are coefficient proofs, not an
    assessment of any physical preparation or controller realization.
    """
    law, execution = _owned_law()
    alpha, beta, gamma = law.stability
    floor = (law.si_high - beta - gamma) / alpha
    if not (0 < floor < 1 and beta > 0 and gamma > 0):
        raise ValueError("the declared positive IL capacity floor changed")
    # Rebuild the block row from the owned ten-constituent support. Each
    # adjacent block contributes its two nodes to the four-neighbor mean.
    laplacian = tuple(
        tuple(
            Q(sum(node // 2 == j for node in execution.support[2 * i]), 4) - Q(i == j)
            for j in range(5)
        )
        for i in range(5)
    )
    expected = tuple(
        tuple(
            -Q(1) if i == j else Q(1, 2) if (j - i) % 5 in (1, 4) else Q(0)
            for j in range(5)
        )
        for i in range(5)
    )
    if laplacian != expected or any(
        execution.support[2 * i] != execution.support[2 * i + 1] for i in range(5)
    ):
        raise ValueError("the paired unit C5 pressure support changed")
    reconstructions = (
        GapReconstruction(
            1,
            (0, 1),
            tuple(
                tuple(map(Q, row))
                for row in (
                    (1, 0, 0, 0),
                    (0, 1, 0, 0),
                    (-1, 2, 0, 2),
                    (-2, -2, -2, -2),
                    (2, -1, 2, 0),
                )
            ),
        ),
        GapReconstruction(
            2,
            (0, 2),
            tuple(
                tuple(map(Q, row))
                for row in (
                    (1, 0, 0, 0),
                    (3, 3, 2, 2),
                    (0, 1, 0, 0),
                    (-3, -1, -2, 0),
                    (-1, -3, 0, -2),
                )
            ),
        ),
    )
    gain_rows = []
    for branch in BRANCHES:
        global_gain, local_gain = _gains(law, law.phase_gains, branch)
        if not (0 < global_gain < 1 and 0 < local_gain < 1 - global_gain):
            raise ValueError("the positive first phase row changed")
        phase = tuple(
            tuple(
                (1 - global_gain) * Q(i == j) + local_gain * laplacian[i][j]
                for j in range(5)
            )
            for i in range(5)
        )
        gain_rows.append(
            JointBoundaryGainRow(
                branch,
                (global_gain, local_gain),
                phase,
                (-(1 - global_gain) / local_gain, 1 / local_gain),
            )
        )
    return JointBoundaryRestriction(
        law=law,
        execution=execution,
        capacity_floor=floor,
        laplacian=laplacian,
        reconstructions=reconstructions,
        first_gain_rows=tuple(gain_rows),
        adjacent_current_balances=((1, 2), (4, 3)),
        obligations=(
            "unchanged paired unit C5 source OZ/IL then acquired negative-outward ZHIR/IL model",
            "same positive capacities and actual maximum at both fresh boundaries",
            "each saturated surrounding block has v_i/V exactly equal to the reported floor",
            "actual high-Si IL at both boundaries, including the zero-pressure-peak convention",
            "source and first coordinated primitive phases share the admitted ideal common lift",
            "actual first branch remains to be admitted; all three positive gain rows are retained",
            "two saturated blocks exclude the required strictly outward gap, not all Mutation",
            "the adjacent current identity requires the positive multiplicative OZ branch p0>source_tiny",
            "a single saturated adjacent block obeys the same-source held-pressure current balance",
            "full form rails, acquired history, remaining masks and joint source existence remain unresolved",
        ),
    )
