"""TNFR Canonical Operator-Contract Specification — the single source of truth.

This module is the authoritative, physics-grounded, TNFR.pdf-anchored
specification of the **contracts** of the 13 structural operators: what each
operator does to the node state while remaining compatible with the nodal
equation ``∂EPI/∂t = νf · ΔNFR``, expressed as a verifiable postcondition.

It is the contract-layer companion of :mod:`grammar_canon` (which owns the U1-U6
*grammatical role* layer). Together they fully specify each operator:

    grammar_canon      → U1-U6 roles    (generator / stabilizer / destabilizer …)
    operator_contracts → state effects  (which nodal channel, which direction)

Ground truth (doctrinal)
------------------------
The executable ground truth of an operator contract is **the direct effect the
glyph has on the node state** — the deterministic mutation applied by the
``_op_*`` handlers in :mod:`tnfr.operators`. That mutation must remain
compatible with ``∂EPI/∂t = νf · ΔNFR``; the equation alone does not determine
all four state-channel updates. The three sources checked together are: (1) the
nodal equation, (2) the ``_op_*`` direct effects, and (3) TNFR.pdf §2.2.1
"Matriz operativa de los símbolos nodales" (the per-glyph formal expressions,
e.g. ``A'L ⇒ ∂EPI/∂t > 0, νf ≈ ν₀⁺``; ``I'L ⇒ ∂Wᵢ/∂t → 0, νf = const``).

The unifying structure (synergies)
-----------------------------------
The 13 operators distribute across the **four nodal-equation state channels**
(the structural triad EPI/νf/θ plus the pressure ΔNFR). This partition exposes:

  * the **dual-lever** (examples 37/130): the νf channel = capacity lever,
    the ΔNFR channel = pressure lever;
  * the **tetrad read-out relation** (example 39): at fixed topology Φ_s is a
    linear aggregation of ΔNFR, while θ determines |∇φ| and K_φ. ξ_C is a
    nonlinear, state- and topology-dependent correlation estimate and has no
    one-to-one primary operator channel;
  * the **number-theory grading** (example 147): the ΔNFR/pressure channel is the
    count-Ω arm, the νf/capacity channel is the size-log arm.

Channel partition (primary channel per operator)
------------------------------------------------
    EPI      (the form)                  : Emission, Reception, Resonance,
                                            Recursivity
    νf       (frequency / mobility)       : Silence, Expansion, Contraction
    θ        (phase → |∇φ|, K_φ)          : Coupling, Mutation
    ΔNFR     (pressure → Φ_s)             : Coherence, Dissonance,
                                            SelfOrganization, Transition

Scale partition (U5 operational fractality)
-------------------------------------------
A second, orthogonal axis: the **scale** at which the operator acts. Exactly one
operator implements operational fractality (grammar rule U5) and therefore acts
at NETWORK scale; the other twelve act at NODE scale:

    NODE     : the twelve operators whose ``_op_*`` handler mutates one node's
               state channel (they act on the *fiber* — the per-node substrate).
    NETWORK  : Recursivity (REMESH) — the multi-scale echo. Its node-level call
               is advisory; its canonical effect is the network-scale temporal
               EPI recurrence ``EPI_new = (1-α)²·EPI(t) + α(1-α)·EPI(t-τ_l) +
               α·EPI(t-τ_g)`` (``apply_network_remesh``) plus topological
               base-from-fiber regeneration (``apply_topological_remesh``). Its
               fixed-parameter history surrogate has a conditional Cesàro
               fixed-point projection (N15,
               theory/REMESH_INFINITY_DERIVATION.md); this is not the literal
               runtime τ_g→∞ limit of ``apply_network_remesh``. REMESH is an EPI
               operator (it echoes the form across time/scale), distinguished
               from Emission/Reception/Resonance only by its NETWORK scale —
               which *is* U5 fractality. This connects the contract layer to the
               base/fiber optic (examples 126-131): NODE operators act on the
               fiber, the NETWORK operator regenerates the base.

Public naming
-------------
The lowercase English token is the canonical public executable identifier
(``emission``, ``reception``, ``coherence``, …). ``english_name`` stores the
title-case public display/class name (Emission, Reception, Coherence, …), while
``glyph`` stores the internal symbolic code (AL, EN, IL, …). Contract lookup
accepts all three layers without conflating their roles.

Self-consistency
----------------
:func:`verify_contract_consistency` asserts that the materialised contracts agree
with the canonical glyph↔name mapping in :mod:`grammar_types` and that the
primary-channel partition matches the dual-lever / nodal-channel structure.
The agreement is pinned by ``tests/operators/test_operator_contracts.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from .grammar_types import FUNCTION_TO_GLYPH

__all__ = [
    "StateChannel",
    "OperatorScale",
    "EffectDirection",
    "ContractContext",
    "OperatorContract",
    "ContractIdentifiabilityCertificate",
    "OPERATOR_CONTRACTS",
    "contract_for",
    "iter_contracts",
    "operators_in_channel",
    "operators_at_scale",
    "english_name",
    "contract_identifiability_certificate",
    "verify_contract_consistency",
]


class StateChannel(Enum):
    """A state variable of the nodal equation ``∂EPI/∂t = νf · ΔNFR``.

    The structural triad (EPI form, νf frequency, θ phase) plus the structural
    pressure ΔNFR. Every canonical operator's primary effect lands on exactly
    one of these four channels.
    """

    EPI = "EPI"  # the form itself
    NU_F = "nu_f"  # structural frequency / mobility (capacity lever)
    THETA = "theta"  # phase → |∇φ| (1st order), K_φ (2nd order)
    DELTA_NFR = "delta_nfr"  # structural pressure → Φ_s (0th order)


class OperatorScale(Enum):
    """The scale at which an operator acts (grammar rule U5 fractality axis).

    Orthogonal to :class:`StateChannel`. Exactly one operator (REMESH) is
    NETWORK-scale — it is the operational-fractality (U5) operator whose effect
    is multi-scale rather than node-local.
    """

    NODE = "node"  # mutates one node's state channel (acts on the fiber)
    NETWORK = "network"  # multi-scale echo (REMESH): temporal + topological


class EffectDirection(Enum):
    """The canonical direction of an operator's primary channel effect."""

    INCREASE = "increase"  # channel magnitude rises (∂ > 0)
    DECREASE = "decrease"  # channel magnitude falls (∂ < 0)
    PRESERVE = "preserve"  # channel held (∂ ≈ 0, e.g. freeze)
    REORGANIZE = "reorganize"  # channel reshaped toward a target (mix/sync)
    TRANSFORM = "transform"  # channel crosses a threshold (θ → θ')


class ContractContext(Enum):
    """The context where an operator's contract canonically manifests.

    Because ΔNFR and C(t) are EMERGENT network fields, a contract is measured at
    the level where it physically appears (established by the operator-contract
    fidelity audit, example 115).
    """

    NETWORK = "network"  # emergent field (recompute ΔNFR / C(t) after)
    NODE = "node"  # node-local channel (e.g. local OZ pressure)
    IDENTITY = "identity"  # structural identity (EPI sign/kind preserved)
    PHASE = "phase"  # phase channel (θ transformed)
    STATE = "state"  # any state variable changed (regime shift)
    ADVISORY = "advisory"  # verified at network scale elsewhere


# Diagnostic relation between each channel and the structural-field tetrad.
# ξ_C is deliberately absent from any one-to-one assignment: it depends on the
# complete pressure field, topology and the estimator branch.
_CHANNEL_TETRAD = {
    StateChannel.EPI: "no unique tetrad component; effects require state recomputation",
    StateChannel.NU_F: "no direct tetrad component; νf sets evolution rate",
    StateChannel.THETA: "|∇φ| (1st), K_φ (2nd) — phase gradient/curvature",
    StateChannel.DELTA_NFR: "Φ_s — linear source aggregation at fixed topology",
}

# Dual-lever arm each channel corresponds to (synergy with examples 37/130).
_CHANNEL_LEVER = {
    StateChannel.EPI: "form",
    StateChannel.NU_F: "capacity (νf lever)",
    StateChannel.THETA: "phase",
    StateChannel.DELTA_NFR: "pressure (ΔNFR lever)",
}


@dataclass(frozen=True, slots=True)
class OperatorContract:
    """Canonical contract of one structural operator.

    Attributes
    ----------
    name : str
        Canonical lowercase public executable identifier (e.g. ``"emission"``).
    english_name : str
        Title-case public display/class name (e.g. ``"Emission"``).
    glyph : str
        Internal symbolic glyph code (e.g. ``"AL"``).
    purpose : str
        One-line canonical purpose of the operator.
    primary_channel : StateChannel
        The nodal-equation state channel the operator primarily acts on.
    primary_direction : EffectDirection
        The canonical direction of that primary effect.
    scale : OperatorScale
        The scale at which the operator acts (NODE for twelve operators,
        NETWORK for the U5 fractality operator REMESH).
    postcondition : str
        The verifiable postcondition (the invariant form of the direct effect).
    context : ContractContext
        Where the contract canonically manifests (example 115).
    nodal_expression : str
        The TNFR.pdf §2.2.1 formal expression of the operator's effect.
    pdf_reference : str
        The TNFR.pdf section anchoring the operator.
    """

    name: str
    english_name: str
    glyph: str
    purpose: str
    primary_channel: StateChannel
    primary_direction: EffectDirection
    scale: OperatorScale
    postcondition: str
    context: ContractContext
    nodal_expression: str
    pdf_reference: str

    @property
    def tetrad_field(self) -> str:
        """Describe the channel's scoped relation to tetrad diagnostics.

        This compatibility property is descriptive. It does not assert that
        the tetrad reconstructs an operator or the complete graph state.
        """
        return _CHANNEL_TETRAD[self.primary_channel]

    @property
    def lever(self) -> str:
        """The dual-lever arm (examples 37/130) of the primary channel."""
        return _CHANNEL_LEVER[self.primary_channel]


@dataclass(frozen=True, slots=True)
class ContractIdentifiabilityCertificate:
    """Equivalence classes induced by observable contract features.

    This is a structural identifiability result for the declared feature set,
    not an empirical classifier of engine trajectories.
    """

    features: tuple[str, ...]
    signature_groups: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    uniquely_identifiable: tuple[str, ...]
    ambiguous_groups: tuple[tuple[str, ...], ...]
    all_operators_identifiable: bool
    claim_status: str


# ════════════════════════════════════════════════════════════════════════════
# THE CANONICAL CONTRACTS — single source of truth (13 operators)
# Ground truth for executable effects: the direct ``_op_*`` state mutation,
# checked for compatibility with the nodal equation and anchored to TNFR.pdf
# §2.2.1 formal expressions. The handler is an operator realization, not the
# complete nodal dynamics. Contracts are ordered by their primary channel.
# ════════════════════════════════════════════════════════════════════════════

OPERATOR_CONTRACTS: dict[str, OperatorContract] = {
    # ── EPI channel (the form): Emission, Reception, Resonance ──────────────
    EMISSION: OperatorContract(
        name=EMISSION,
        english_name="Emission",
        glyph="AL",
        purpose="Activates an EPI from a latent state (founding emission).",
        primary_channel=StateChannel.EPI,
        primary_direction=EffectDirection.INCREASE,
        scale=OperatorScale.NODE,
        postcondition="EPI not decreased; νf, phase and ΔNFR unchanged",
        context=ContractContext.NETWORK,
        nodal_expression="A'L ⇒ ∂EPI/∂t > 0, νf ≈ ν₀⁺",
        pdf_reference="TNFR.pdf §2.2.1 (1) A'L — Emisión fundacional",
    ),
    RECEPTION: OperatorContract(
        name=RECEPTION,
        english_name="Reception",
        glyph="EN",
        purpose="Integrates an external emission, reorganizing EPI coherently.",
        primary_channel=StateChannel.EPI,
        primary_direction=EffectDirection.REORGANIZE,
        scale=OperatorScale.NODE,
        postcondition="C(t) not decreased (coherent integration)",
        context=ContractContext.NETWORK,
        nodal_expression="E'N ⇒ coherent input → modulation of Wᵢ(t)",
        pdf_reference="TNFR.pdf §2.2.1 (2) E'N — Recepción estructural",
    ),
    RESONANCE: OperatorContract(
        name=RESONANCE,
        english_name="Resonance",
        glyph="RA",
        purpose="Propagates an EPI across couplings without altering identity.",
        primary_channel=StateChannel.EPI,
        primary_direction=EffectDirection.REORGANIZE,
        scale=OperatorScale.NODE,
        postcondition="EPI structural identity (sign/kind) preserved",
        context=ContractContext.IDENTITY,
        nodal_expression="R'A ⇒ EPI propagation with amplified νf",
        pdf_reference="TNFR.pdf §2.2.1 R'A — Resonancia",
    ),
    # ── νf channel (frequency/mobility, capacity arm): SHA, VAL, NUL ────────
    SILENCE: OperatorContract(
        name=SILENCE,
        english_name="Silence",
        glyph="SHA",
        purpose="Freezes evolution by driving νf → 0 (structural latency).",
        primary_channel=StateChannel.NU_F,
        primary_direction=EffectDirection.DECREASE,
        scale=OperatorScale.NODE,
        postcondition="νf not increased (freeze)",
        context=ContractContext.NETWORK,
        nodal_expression="SH'A ⇒ νf → 0 ⇒ ∂EPI/∂t → 0",
        pdf_reference="TNFR.pdf §2.2.1 SH'A — Silencio",
    ),
    EXPANSION: OperatorContract(
        name=EXPANSION,
        english_name="Expansion",
        glyph="VAL",
        purpose="Adds reorganization capacity (νf), raising structural complexity.",
        primary_channel=StateChannel.NU_F,
        primary_direction=EffectDirection.INCREASE,
        scale=OperatorScale.NODE,
        postcondition="νf not decreased (capacity added)",
        context=ContractContext.NETWORK,
        nodal_expression="VA'L ⇒ νf ↑ (structural complexity)",
        pdf_reference="TNFR.pdf §2.2.1 VA'L — Expansión",
    ),
    CONTRACTION: OperatorContract(
        name=CONTRACTION,
        english_name="Contraction",
        glyph="NUL",
        purpose="Removes capacity (νf ↓) and concentrates pressure (ΔNFR ↑).",
        primary_channel=StateChannel.NU_F,
        primary_direction=EffectDirection.DECREASE,
        scale=OperatorScale.NODE,
        postcondition="νf not increased (capacity removed)",
        context=ContractContext.NETWORK,
        nodal_expression="NU'L ⇒ νf ↓, ΔNFR densified",
        pdf_reference="TNFR.pdf §2.2.1 NU'L — Contracción",
    ),
    # ── θ channel (phase → |∇φ|, K_φ): Coupling, Mutation ───────────────────
    COUPLING: OperatorContract(
        name=COUPLING,
        english_name="Coupling",
        glyph="UM",
        purpose="Synchronizes phase with neighbours (φᵢ ≈ φⱼ), reducing pressure.",
        primary_channel=StateChannel.THETA,
        primary_direction=EffectDirection.REORGANIZE,
        scale=OperatorScale.NODE,
        postcondition="|ΔNFR| not increased (mutual stabilization)",
        context=ContractContext.NETWORK,
        nodal_expression="U'M ⇒ φᵢ(t) → φⱼ(t) (phase synchronization)",
        pdf_reference="TNFR.pdf §2.2.1 U'M — Acoplamiento",
    ),
    MUTATION: OperatorContract(
        name=MUTATION,
        english_name="Mutation",
        glyph="ZHIR",
        purpose="Transforms the phase regime θ → θ' at a structural threshold.",
        primary_channel=StateChannel.THETA,
        primary_direction=EffectDirection.TRANSFORM,
        scale=OperatorScale.NODE,
        postcondition="θ transformed (θ → θ')",
        context=ContractContext.PHASE,
        nodal_expression="Z'HIR ⇒ θ → θ' when ΔEPI/Δt > ξ",
        pdf_reference="TNFR.pdf §2.2.1 Z'HIR — Mutación",
    ),
    # ── ΔNFR channel (pressure → Φ_s, count arm): IL, OZ, THOL, NAV ─────────
    COHERENCE: OperatorContract(
        name=COHERENCE,
        english_name="Coherence",
        glyph="IL",
        purpose="Stabilizes form by reducing |ΔNFR| (negative feedback).",
        primary_channel=StateChannel.DELTA_NFR,
        primary_direction=EffectDirection.DECREASE,
        scale=OperatorScale.NODE,
        postcondition="|ΔNFR| not increased and C(t) not decreased",
        context=ContractContext.NETWORK,
        nodal_expression="I'L ⇒ ∂Wᵢ/∂t → 0, νf = const",
        pdf_reference="TNFR.pdf §2.2.1 (3) I'L — Coherencia estructural",
    ),
    DISSONANCE: OperatorContract(
        name=DISSONANCE,
        english_name="Dissonance",
        glyph="OZ",
        purpose="Injects controlled instability, raising |ΔNFR| (may bifurcate).",
        primary_channel=StateChannel.DELTA_NFR,
        primary_direction=EffectDirection.INCREASE,
        scale=OperatorScale.NODE,
        postcondition="|ΔNFR| not decreased",
        context=ContractContext.NODE,
        nodal_expression="O'Z ⇒ |ΔNFR| ↑ (may trigger ∂²EPI/∂t² > τ)",
        pdf_reference="TNFR.pdf §2.2.1 O'Z — Disonancia",
    ),
    SELF_ORGANIZATION: OperatorContract(
        name=SELF_ORGANIZATION,
        english_name="SelfOrganization",
        glyph="THOL",
        purpose="Autopoietic reorganization: forms sub-EPIs while preserving global form.",
        primary_channel=StateChannel.DELTA_NFR,
        primary_direction=EffectDirection.REORGANIZE,
        scale=OperatorScale.NODE,
        postcondition=(
            "parent EPI, nu_f and phase fixed; DeltaNFR follows signed "
            "acceleration; nested child creation preserves parent identity"
        ),
        context=ContractContext.NETWORK,
        nodal_expression="T'HOL ⇒ ΔNFR += κ·∂²EPI/∂t² (sub-EPIs)",
        pdf_reference="TNFR.pdf §2.2.1 T'HOL — Autoorganización",
    ),
    TRANSITION: OperatorContract(
        name=TRANSITION,
        english_name="Transition",
        glyph="NAV",
        purpose="Controlled regime shift, retargeting ΔNFR toward a νf-aligned state.",
        primary_channel=StateChannel.DELTA_NFR,
        primary_direction=EffectDirection.REORGANIZE,
        scale=OperatorScale.NODE,
        postcondition="state changed (νf, θ, or ΔNFR)",
        context=ContractContext.STATE,
        nodal_expression="NA'V ⇒ ΔNFR → νf-aligned (regime transition)",
        pdf_reference="TNFR.pdf §2.2.1 NA'V — Transición",
    ),
    # ── EPI channel at NETWORK scale (U5 fractality): Recursivity ──────────────
    RECURSIVITY: OperatorContract(
        name=RECURSIVITY,
        english_name="Recursivity",
        glyph="REMESH",
        purpose="Echoes the form (EPI) across time and scale (operational "
        "fractality, U5).",
        primary_channel=StateChannel.EPI,
        primary_direction=EffectDirection.REORGANIZE,
        scale=OperatorScale.NETWORK,
        postcondition="node-level advisory; network effect = EPI mixed toward "
        "temporal/multi-scale history",
        context=ContractContext.ADVISORY,
        nodal_expression="RE'MESH ⇒ EPI_new = (1-α)²·EPI(t) + α(1-α)·EPI(t-τ_l) "
        "+ α·EPI(t-τ_g)",
        pdf_reference="TNFR.pdf §2.2.1 RE'MESH — Recursividad",
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# Contract accessors and display labels
# ════════════════════════════════════════════════════════════════════════════


def contract_for(identifier: str) -> OperatorContract:
    """Return the contract for an executable name, display name, or glyph.

    Resolution order: lowercase public executable identifier → title-case
    display/class name → internal glyph code.
    Raises :class:`KeyError` if not found.
    """
    c = OPERATOR_CONTRACTS.get(identifier)
    if c is not None:
        return c
    for contract in OPERATOR_CONTRACTS.values():
        if identifier in (contract.english_name, contract.glyph):
            return contract
    raise KeyError(identifier)


def iter_contracts() -> tuple[OperatorContract, ...]:
    """Iterate the 13 canonical contracts (channel-ordered)."""
    return tuple(OPERATOR_CONTRACTS.values())


def operators_in_channel(channel: StateChannel) -> tuple[str, ...]:
    """Title-case display names for operators in ``channel``."""
    return tuple(
        c.english_name
        for c in OPERATOR_CONTRACTS.values()
        if c.primary_channel is channel
    )


def operators_at_scale(scale: OperatorScale) -> tuple[str, ...]:
    """Title-case display names for operators at ``scale`` (the U5 axis)."""
    return tuple(
        c.english_name for c in OPERATOR_CONTRACTS.values() if c.scale is scale
    )


def english_name(identifier: str) -> str:
    """Return the title-case public display/class name for an identifier."""
    return contract_for(identifier).english_name


def contract_identifiability_certificate(
    features: tuple[str, ...] = (
        "primary_channel",
        "primary_direction",
        "scale",
        "context",
    ),
) -> ContractIdentifiabilityCertificate:
    """Partition operators by a declared instantaneous contract signature.

    Supported features are the four executable contract classifications.  Two
    operators in one returned group cannot be distinguished from these features
    alone.  Text descriptions, glyphs and names are excluded because including
    an identifier would make the inverse problem tautological.
    """
    allowed = {"primary_channel", "primary_direction", "scale", "context"}
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be a nonempty tuple without duplicates")
    unknown = set(features) - allowed
    if unknown:
        raise ValueError(f"unsupported contract features: {sorted(unknown)}")

    grouped: dict[tuple[str, ...], list[str]] = {}
    for contract in iter_contracts():
        signature = tuple(getattr(contract, feature).value for feature in features)
        grouped.setdefault(signature, []).append(contract.english_name)
    groups = tuple(
        (signature, tuple(names)) for signature, names in sorted(grouped.items())
    )
    unique = tuple(
        names[0] for _, names in groups if len(names) == 1
    )
    ambiguous = tuple(names for _, names in groups if len(names) > 1)
    return ContractIdentifiabilityCertificate(
        features=tuple(features),
        signature_groups=groups,
        uniquely_identifiable=unique,
        ambiguous_groups=ambiguous,
        all_operators_identifiable=not ambiguous,
        claim_status=(
            "EXACT for declared contract features; trajectory and word "
            "identification require temporal telemetry"
        ),
    )


# ════════════════════════════════════════════════════════════════════════════
# Self-consistency
# ════════════════════════════════════════════════════════════════════════════


def verify_contract_consistency() -> None:
    """Assert the contracts agree with the canonical glyph↔name mapping and the
    nodal-channel / dual-lever partition. Raises ``AssertionError`` on drift.
    """
    # 1. All 13 canonical operators are covered exactly once.
    names = set(OPERATOR_CONTRACTS)
    expected = set(FUNCTION_TO_GLYPH)
    assert names == expected, (
        f"contract coverage drift: missing {expected - names}, "
        f"extra {names - expected}"
    )
    # 2. glyph code matches the canonical grammar_types mapping.
    for name, contract in OPERATOR_CONTRACTS.items():
        canonical_glyph = FUNCTION_TO_GLYPH[name].value
        assert (
            contract.glyph == canonical_glyph
        ), f"{name}: glyph drift {contract.glyph} != {canonical_glyph}"
    # 3. The primary-channel partition is exactly the canonical four-channel
    #    grouping. νf and ΔNFR form the dual lever; the tetrad remains a lossy
    #    diagnostic read-out rather than an operator-identification basis.
    by_channel = {ch: set(operators_in_channel(ch)) for ch in StateChannel}
    assert by_channel[StateChannel.EPI] == {
        "Emission",
        "Reception",
        "Resonance",
        "Recursivity",
    }
    assert by_channel[StateChannel.NU_F] == {"Silence", "Expansion", "Contraction"}
    assert by_channel[StateChannel.THETA] == {"Coupling", "Mutation"}
    assert by_channel[StateChannel.DELTA_NFR] == {
        "Coherence",
        "Dissonance",
        "SelfOrganization",
        "Transition",
    }
    # 4. The scale partition (U5 fractality): exactly one NETWORK-scale operator
    #    (REMESH), and NETWORK scale ⟺ advisory node-level context.
    assert set(operators_at_scale(OperatorScale.NETWORK)) == {"Recursivity"}
    assert len(operators_at_scale(OperatorScale.NODE)) == 12
    for contract in OPERATOR_CONTRACTS.values():
        is_network = contract.scale is OperatorScale.NETWORK
        assert is_network == (
            contract.context is ContractContext.ADVISORY
        ), f"{contract.name}: network-scale ⟺ advisory mismatch"
