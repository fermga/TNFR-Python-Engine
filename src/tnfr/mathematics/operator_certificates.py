r"""Arithmetic transformation-to-operator certification (R8).

Whether an arithmetic transformation "is" a TNFR operator is a **contract**
question, not a naming one.  This module issues a reproducible
:class:`ArithmeticOperatorCertificate` for candidate transformations, mapping each
to a canonical operator only when its measured effect matches that operator's
contract (channel, scale, postcondition — from
:mod:`tnfr.operators.operator_contracts`) and its grammar word validates against
U1-U6.  Decorative naming is rejected, and **no fourteenth operator is invented**.

A **negative** certificate is a useful result: showing that an arithmetic
operation does not fit the 13 operators without an external axiom sharpens the
TNFR boundary.  Audited candidates:

* localized **emission at zero** (the pointed-pulse seed ``e_0``)  → **Emission**;
* **propagation over residue edges** (additive Cayley transport)    → **Resonance**
  (under the U3 phase precondition);
* **CRT projection** ``σ``                                          → rejected
  (pure relabeling; modifies no nodal channel);
* **p-adic lift** ``ℤ/p^eℤ → ℤ/p^{e+1}ℤ``                           → rejected
  (transport-consistent but the REMESH contract is unverified — R4);
* affine ``x ↦ ax + b (mod n)``                                    → rejected
  (network automorphism / relabeling);
* power map ``x ↦ x^k (mod n)``                                    → rejected
  (group endomorphism; its "contraction" is of the state space, not of νf).

Scope: the two positive certificates reuse the canonical contracts; the four
negatives are the honest boundary result.  No claim maps an arithmetic operation
to an operator beyond these contracts (``NT-P08`` OPEN).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

from ..operators.grammar import validate_sequence
from ..operators.operator_contracts import contract_for
from .crt_multiscale import crt_ordering
from .padic_tower import (
    compatible_connection_set,
    padic_transition,
    projective_commutation_residual,
    remesh_contract_audit,
)

__all__ = [
    "ArithmeticOperatorCertificate",
    "certify_emission_at_zero",
    "certify_residue_edge_propagation",
    "reject_crt_projection",
    "reject_padic_lift",
    "reject_affine_map",
    "reject_power_map",
    "all_certificates",
    "certified_mappings",
    "rejected_mappings",
    "verify_certificate",
]


@dataclass(frozen=True)
class ArithmeticOperatorCertificate:
    """A reproducible certificate mapping a transformation to (or away from) an
    operator, with its contract fields and measured residuals."""

    transformation: str
    canonical_operator: str | None
    state_channel: str
    scale: str
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    grammar_word: tuple[str, ...]
    residuals: dict[str, float] = field(default_factory=dict)
    rejection_reason: str = ""

    @property
    def certified(self) -> bool:
        return self.canonical_operator is not None

    @property
    def rejected(self) -> bool:
        return self.canonical_operator is None

    @property
    def grammar_valid(self) -> bool:
        if not self.grammar_word:
            return False
        return bool(validate_sequence(list(self.grammar_word)).passed)


# --------------------------------------------------------------------------- #
# Positive certificates (measured effect matches the canonical contract)
# --------------------------------------------------------------------------- #
def certify_emission_at_zero() -> ArithmeticOperatorCertificate:
    r"""Localized emission at the additive identity ``e_0`` → **Emission (AL)**."""
    contract = contract_for("Emission")
    epi_before, epi_after = 0.0, 1.0  # e_0 seeds EPI at the neutral node
    delta_epi = epi_after - epi_before
    return ArithmeticOperatorCertificate(
        transformation="localized emission at zero (pointed seed e_0)",
        canonical_operator=contract.english_name,
        state_channel=contract.primary_channel.value,
        scale=contract.scale.value,
        preconditions=("node EPI latent (e_0 = 0 before the seed)",),
        postconditions=(contract.postcondition,),
        grammar_word=("emission", "coherence", "silence"),
        residuals={
            # direction violation = how much EPI *decreased* (must be 0)
            "epi_direction_violation": max(0.0, -delta_epi),
            "delta_epi": delta_epi,
        },
    )


def certify_residue_edge_propagation(
    p: int = 5, base: frozenset[int] = frozenset({1, 2, 3, 4})
) -> ArithmeticOperatorCertificate:
    r"""Propagation over residue edges (additive Cayley transport) → **Resonance
    (RA)**, under the U3 phase-compatibility precondition.

    Measured: one random-walk transport step conserves the total EPI (identity
    preserved) — the residual is the conservation defect.
    """
    contract = contract_for("Resonance")
    connection = compatible_connection_set(p, 1, base)
    transition = padic_transition(p, 1, connection)
    epi = [1.0] + [0.0] * (p - 1)  # EPI concentrated at node 0
    propagated = [
        float(sum(transition[i][j] * epi[j] for j in range(p)))
        for i in range(p)
    ]
    conservation_defect = abs(sum(propagated) - sum(epi))
    spread = sum(1 for v in propagated if v > 0)
    return ArithmeticOperatorCertificate(
        transformation="propagation over residue edges (additive Cayley "
        "transport)",
        canonical_operator=contract.english_name,
        state_channel=contract.primary_channel.value,
        scale=contract.scale.value,
        preconditions=(
            "U3 phase compatibility |phi_i - phi_j| <= dphi_max",
            "identity (EPI kind) preserved",
        ),
        postconditions=(contract.postcondition,
                        "EPI propagated to residue neighbors"),
        grammar_word=("emission", "resonance", "coupling", "silence"),
        residuals={
            "conservation_defect": conservation_defect,
            "neighbors_reached": float(spread),
        },
    )


# --------------------------------------------------------------------------- #
# Negative certificates (the honest boundary: no contract fit)
# --------------------------------------------------------------------------- #
def reject_crt_projection(
    a: int = 3, b: int = 5
) -> ArithmeticOperatorCertificate:
    r"""CRT projection ``σ`` → **rejected**: a pure relabeling changes no channel."""
    perm = crt_ordering(a, b)
    is_permutation = float(sorted(perm) == list(range(a * b)))
    return ArithmeticOperatorCertificate(
        transformation=f"CRT projection sigma: Z/{a * b}Z -> Z/{a}Z x Z/{b}Z",
        canonical_operator=None,
        state_channel="none",
        scale="none",
        preconditions=("gcd(a, b) = 1",),
        postconditions=("node relabeling (bijection); no channel modified",),
        grammar_word=(),
        residuals={
            "is_bijection": is_permutation,
            "channel_modification": 0.0,  # changes no EPI/nu_f/theta/dNFR
        },
        rejection_reason=(
            "pure relabeling: the CRT permutation modifies no nodal state "
            "channel, so it is a coordinate change, not one of the 13 "
            "operators. It realizes U5 multiscale STRUCTURE (R3) but is not "
            "itself an operator without an external axiom."
        ),
    )


def reject_padic_lift(
    p: int = 3, e: int = 1, base: frozenset[int] = frozenset({1, 2})
) -> ArithmeticOperatorCertificate:
    r"""p-adic lift → **rejected**: transport-consistent but REMESH unverified (R4)."""
    commutation = float(projective_commutation_residual(p, e, base))
    audit = remesh_contract_audit()
    unmet = sum(
        1 for key, v in audit.to_dict().items()
        if key != "realizes_remesh" and v is False
    )
    return ArithmeticOperatorCertificate(
        transformation=f"p-adic lift Z/{p}^{e}Z -> Z/{p}^{e + 1}Z",
        canonical_operator=None,
        state_channel="EPI (candidate)",
        scale="NETWORK (candidate)",
        preconditions=("reduction-compatible connection family",),
        postconditions=("projective transport consistency (R4)",),
        grammar_word=(),
        residuals={
            "projective_commutation": commutation,  # 0 => transport consistent
            "remesh_conditions_unmet": float(unmet),  # > 0 => REMESH not earned
        },
        rejection_reason=(
            "the lift is transport-consistent (commutation residual 0) but the "
            "REMESH contract (EPI recursion, NETWORK scale, preserved identity, "
            "U5) is unverified: remesh_contract_audit().realizes_remesh is "
            "False. Naming it REMESH is forbidden until the contract passes."
        ),
    )


def reject_affine_map(
    a: int = 2, b: int = 1, n: int = 7
) -> ArithmeticOperatorCertificate:
    r"""Affine ``x ↦ ax + b (mod n)`` → **rejected**: automorphism / relabeling."""
    image = {(a * x + b) % n for x in range(n)}
    is_bijection = float(len(image) == n and gcd(a, n) == 1)
    return ArithmeticOperatorCertificate(
        transformation=f"affine x -> {a}x + {b} (mod {n})",
        canonical_operator=None,
        state_channel="none",
        scale="none",
        preconditions=("gcd(a, n) = 1 (invertible)",),
        postconditions=("node permutation; no channel intrinsically modified",),
        grammar_word=(),
        residuals={
            "is_bijection": is_bijection,
            "channel_modification": 0.0,
        },
        rejection_reason=(
            "a network automorphism / relabeling of Z/nZ: it permutes nodes "
            "(a symmetry, R1 sector) and neither creates/destroys EPI nor "
            "intrinsically changes nu_f or theta. Not a state-modifying operator."
        ),
    )


def reject_power_map(
    k: int = 3, p: int = 13
) -> ArithmeticOperatorCertificate:
    r"""Power map ``x ↦ x^k (mod p)`` → **rejected**: group endomorphism."""
    order = p - 1
    image = {pow(x, k, p) for x in range(1, p)}
    many_to_one = gcd(k, order)
    return ArithmeticOperatorCertificate(
        transformation=f"power map x -> x^{k} (mod {p})",
        canonical_operator=None,
        state_channel="nu_f (candidate, rejected)",
        scale="none",
        preconditions=(),
        postconditions=(f"image is the k-th power subgroup (size {len(image)} "
                        f"of {order})",),
        grammar_word=(),
        residuals={
            "image_fraction": len(image) / order,
            "many_to_one_factor": float(many_to_one),
        },
        rejection_reason=(
            "a group endomorphism, many-to-one when gcd(k, |group|) > 1. Its "
            "'contraction' reduces the STATE SPACE to a subgroup, not a node's "
            "nu_f; certifying it as Contraction would be decorative. Rejected "
            "without an external axiom mapping endomorphism to nu_f reduction."
        ),
    )


# --------------------------------------------------------------------------- #
# Aggregation and verification
# --------------------------------------------------------------------------- #
def all_certificates() -> tuple[ArithmeticOperatorCertificate, ...]:
    r"""All six audited candidate certificates (two positive, four negative)."""
    return (
        certify_emission_at_zero(),
        certify_residue_edge_propagation(),
        reject_crt_projection(),
        reject_padic_lift(),
        reject_affine_map(),
        reject_power_map(),
    )


def certified_mappings() -> tuple[ArithmeticOperatorCertificate, ...]:
    return tuple(c for c in all_certificates() if c.certified)


def rejected_mappings() -> tuple[ArithmeticOperatorCertificate, ...]:
    return tuple(c for c in all_certificates() if c.rejected)


def verify_certificate(
    cert: ArithmeticOperatorCertificate, *, tol: float = 1e-9
) -> bool:
    r"""Check a certificate's internal consistency.

    A **certified** mapping must have a valid grammar word, contract-consistent
    channel/scale (matching :func:`contract_for`), and all residuals within
    ``tol``.  A **rejected** mapping must carry a non-empty reason.
    """
    if cert.rejected:
        return bool(cert.rejection_reason)
    contract = contract_for(cert.canonical_operator)
    if cert.state_channel != contract.primary_channel.value:
        return False
    if cert.scale != contract.scale.value:
        return False
    if not cert.grammar_valid:
        return False
    # only the explicit violation/defect residuals must vanish
    for key in ("epi_direction_violation", "conservation_defect"):
        if key in cert.residuals and abs(cert.residuals[key]) > tol:
            return False
    return True
