"""TNFR Grammar: Core Grammar Validator

GrammarValidator class - central validation engine for all grammar rules U1-U6.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph, Glyph
    from .definitions import Operator
else:
    NodeId = Any
    TNFRGraph = Any
    from ..types import Glyph
    from .definitions import Operator

from .grammar_types import (
    StructuralGrammarError,
    RepeatWindowError,
    MutationPreconditionError,
    TholClosureError,
    TransitionCompatibilityError,
    StructuralPotentialConfinementError,
    record_grammar_violation,
    GENERATORS,
    CLOSURES,
    STABILIZERS,
    DESTABILIZERS,
    COUPLING_RESONANCE,
    BIFURCATION_TRIGGERS,
    BIFURCATION_HANDLERS,
    TRANSFORMERS,
    RECURSIVE_GENERATORS,
    SCALE_STABILIZERS,
)
from .grammar_context import GrammarContext
from ..config.operator_names import (
    BIFURCATION_WINDOWS,
    CANONICAL_OPERATOR_NAMES,
    DESTABILIZERS_MODERATE,
    DESTABILIZERS_STRONG,
    DESTABILIZERS_WEAK,
    INTERMEDIATE_OPERATORS,
    SELF_ORGANIZATION,
    SELF_ORGANIZATION_CLOSURES,
    VALID_END_OPERATORS,
    VALID_START_OPERATORS,
)
from ..validation.compatibility import (
    CompatibilityLevel,
    get_compatibility_level,
)

class GrammarValidator:
    """Validates sequences using canonical TNFR grammar constraints.

    Implements U1-U5 rules that emerge inevitably from TNFR physics.
    This is the single source of truth for grammar validation.

    All rules derive from:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - Canonical invariants (AGENTS.md §3)
    - Formal contracts (AGENTS.md §4)

    No organizational conventions are enforced.

    Parameters
    ----------
    experimental_u6 : bool, optional
        Enable experimental U6: Temporal Ordering validation (default: False).
        U6 is under research and not yet canonical. When enabled, sequences
        are checked for temporal spacing violations after destabilizers.
        Violations log warnings but do not fail validation.
    """

    def __init__(self, experimental_u6: bool = False):
        """Initialize validator with optional experimental features.

        Parameters
        ----------
        experimental_u6 : bool, optional
            Enable U6 temporal ordering checks (default: False)
        """
        self.experimental_u6 = experimental_u6

    @staticmethod
    def validate_initiation(
        sequence: List[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, str]:
        """Validate U1a: Structural initiation.

        Physical basis: If EPI=0, then ∂EPI/∂t is undefined or zero.
        Cannot evolve structure that doesn't exist.

        Generators create structure from:
        - AL (Emission): vacuum via emission
        - NAV (Transition): latent EPI via regime shift
        - REMESH (Recursivity): dormant structure across scales

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate
        epi_initial : float, optional
            Initial EPI value (default: 0.0)

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        if epi_initial > 0.0:
            # Already initialized, no generator required
            return True, "U1a: EPI>0, initiation not required"

        if not sequence:
            return False, "U1a violated: Empty sequence with EPI=0"

        first_op = getattr(
            sequence[0],
            "canonical_name",
            sequence[0].name.lower(),
        )

        if first_op not in GENERATORS:
            return (
                False,
                (
                    "U1a violated: EPI=0 requires generator "
                    f"(got '{first_op}'). Valid: {sorted(GENERATORS)}"
                ),
            )

        return True, f"U1a satisfied: starts with generator '{first_op}'"

    @staticmethod
    def validate_closure(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U1b: Structural closure.

        Physical basis: Sequences are bounded action potentials in structural
        space. Like physical waves, they must have termination that leaves
        system in coherent attractor states.

        Closures stabilize via:
        - SHA (Silence): Terminal closure - freezes evolution (νf → 0)
        - NAV (Transition): Handoff closure - transfers to next regime
        - REMESH (Recursivity): Recursive closure - distributes across scales
        - OZ (Dissonance): Intentional closure - preserves activation/tension

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        if not sequence:
            return False, "U1b violated: Empty sequence has no closure"

        last_op = getattr(
            sequence[-1],
            "canonical_name",
            sequence[-1].name.lower(),
        )

        if last_op not in CLOSURES:
            return (
                False,
                (
                    "U1b violated: Sequence must end with closure "
                    f"(got '{last_op}'). Valid: {sorted(CLOSURES)}"
                ),
            )

        return True, f"U1b satisfied: ends with closure '{last_op}'"

    @staticmethod
    def validate_convergence(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U2: Convergence and boundedness.

        Physical basis: Without stabilizers, ∫νf·ΔNFR dt → ∞ (diverges).
        Stabilizers provide negative feedback ensuring integral convergence.

        From integrated nodal equation:
            EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf·ΔNFR dτ

        Without stabilizers:
            d(ΔNFR)/dt > 0 always → ΔNFR ~ e^(λt) → integral diverges

        With stabilizers (IL or THOL):
            d(ΔNFR)/dt can be < 0 → ΔNFR bounded → integral converges

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        # Check if sequence contains destabilizers
        destabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in DESTABILIZERS
        ]

        if not destabilizers_present:
            # No destabilizers = no divergence risk
            return True, "U2: not applicable (no destabilizers present)"

        # Check for stabilizers
        stabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in STABILIZERS
        ]

        if not stabilizers_present:
            return (
                False,
                f"U2 violated: destabilizers {destabilizers_present} present "
                f"without stabilizer. Integral ∫νf·ΔNFR dt may diverge. "
                f"Add: {sorted(STABILIZERS)}",
            )

        return (
            True,
            f"U2 satisfied: stabilizers {stabilizers_present} "
            f"bound destabilizers {destabilizers_present}",
        )

    @staticmethod
    def validate_resonant_coupling(
        sequence: List[Operator],
    ) -> tuple[bool, str]:
        """Validate U3: Resonant coupling.

        Physical basis: AGENTS.md Invariant #5 states "no coupling is valid
        without explicit phase verification (synchrony)".

        Resonance physics requires phase compatibility:
            |φᵢ - φⱼ| ≤ Δφ_max

        Without phase verification:
            Nodes with incompatible phases (antiphase) could attempt coupling
            → Destructive interference → Violates resonance physics

        With phase verification:
            Only synchronous nodes couple → Constructive interference

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)

        Notes
        -----
    U3 is a META-rule: it requires that when UM (Coupling) or
    RA (Resonance)
        operators are used, the implementation MUST verify phase compatibility.
        The actual phase check happens in operator preconditions.

        This grammar rule documents the requirement and ensures awareness
        that phase checks are MANDATORY (Invariant #5), not optional.
        """
        # Check if sequence contains coupling/resonance operators
        coupling_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in (
                COUPLING_RESONANCE
            )
        ]

        if not coupling_ops:
            # No coupling/resonance = U3 not applicable
            return True, "U3: not applicable (no coupling/resonance operators)"

        # U3 satisfied: Sequence contains coupling/resonance
        # Phase verification is MANDATORY per Invariant #5
        # Actual check happens in operator preconditions
        return (
            True,
            (
                "U3 awareness: operators "
                f"{coupling_ops} require phase verification "
                "(MANDATORY per Invariant #5). Enforced in preconditions."
            ),
        )

    @staticmethod
    def validate_bifurcation_triggers(
        sequence: List[Operator],
    ) -> tuple[bool, str]:
        """Validate U4a: Bifurcation triggers need handlers.

        Physical basis: AGENTS.md Contract OZ states dissonance may trigger
        bifurcation if ∂²EPI/∂t² > τ. When bifurcation is triggered, handlers
        are required to manage structural reorganization.

        Bifurcation physics:
            If ∂²EPI/∂t² > τ → multiple reorganization paths viable
            → System enters bifurcation regime
            → Requires handlers (THOL or IL) for stable transition

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        # Check if sequence contains bifurcation triggers
        trigger_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in (
                BIFURCATION_TRIGGERS
            )
        ]

        if not trigger_ops:
            # No triggers = U4a not applicable
            return True, "U4a: not applicable (no bifurcation triggers)"

        # Check for handlers
        handler_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in (
                BIFURCATION_HANDLERS
            )
        ]

        if not handler_ops:
            return (
                False,
                (
                    "U4a violated: bifurcation triggers "
                    f"{trigger_ops} present without handler. "
                    "If ∂²EPI/∂t² > τ, bifurcation may occur unmanaged. "
                    f"Add: {sorted(BIFURCATION_HANDLERS)}"
                ),
            )

        return (
            True,
            (
                f"U4a satisfied: bifurcation triggers {trigger_ops} have "
                f"handlers {handler_ops}"
            ),
        )

    @staticmethod
    def validate_transformer_context(
        sequence: List[Operator],
    ) -> tuple[bool, str]:
        """Validate U4b: Transformers need context.

        Physical basis: Bifurcations require threshold energy to cross
        critical points. Transformers (ZHIR, THOL) need recent destabilizers
        to provide sufficient |ΔNFR| for phase transitions.

        ZHIR (Mutation) requirements:
            1. Prior IL: Stable base prevents transformation from chaos
            2. Recent destabilizer: Threshold energy for bifurcation

        THOL (Self-organization) requirements:
            1. Recent destabilizer: Disorder to self-organize

        "Recent" = within ~3 operators (ΔNFR decays via structural relaxation)

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)

        Notes
        -----
        This implements "graduated destabilization" - transformers need
        sufficient ΔNFR context. The ~3 operator window captures when
        |ΔNFR| remains above bifurcation threshold.
        """
        # Check if sequence contains transformers
        transformer_ops = []
        for i, op in enumerate(sequence):
            op_name = getattr(op, "canonical_name", op.name.lower())
            if op_name in TRANSFORMERS:
                transformer_ops.append((i, op_name))

        if not transformer_ops:
            return True, "U4b: not applicable (no transformers)"

        # For each transformer, check context
        violations = []
        for idx, transformer_name in transformer_ops:
            # Check for recent destabilizer (within 3 operators before)
            window_start = max(0, idx - 3)
            recent_destabilizers = []
            prior_il = False

            for j in range(window_start, idx):
                op_name = getattr(
                    sequence[j],
                    "canonical_name",
                    sequence[j].name.lower(),
                )
                if op_name in DESTABILIZERS:
                    recent_destabilizers.append((j, op_name))
                if op_name == "coherence":
                    prior_il = True

            # Check requirements
            if not recent_destabilizers:
                violations.append(
                    (
                        f"{transformer_name} at position {idx} lacks recent "
                        "destabilizer (none in window "
                        f"[{window_start}:{idx}]). Need: {sorted(DESTABILIZERS)}"
                    )
                )

            # Additional requirement for ZHIR: prior IL
            if transformer_name == "mutation" and not prior_il:
                violations.append(
                    f"mutation at position {idx} lacks prior IL (coherence) "
                    f"for stable transformation base"
                )

        if violations:
            return (False, f"U4b violated: {'; '.join(violations)}")

        return (True, "U4b satisfied: transformers have proper context")

    @staticmethod
    def validate_remesh_amplification(
        sequence: List[Operator],
    ) -> tuple[bool, str]:
        """Validate U2-REMESH: Recursive amplification control.

        Physical basis: REMESH implements temporal coupling EPI(t) ↔ EPI(t-τ)
        which creates feedback that amplifies structural changes. When combined
        with destabilizers, this can cause unbounded growth.

        From integrated nodal equation:
            EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf·ΔNFR dτ

        REMESH temporal mixing:
            EPI_mixed = (1-α)·EPI_now + α·EPI_past

        Without stabilizers:
            REMESH + destabilizers → recursive amplification
            → ∫ νf·ΔNFR dt → ∞ (feedback loop)
            → System fragments

        With stabilizers:
            IL or THOL provides negative feedback
            → Bounded recursive evolution
            → ∫ νf·ΔNFR dt < ∞

        Specific combinations requiring stabilizers:
            - REMESH + VAL: Recursive expansion needs coherence stabilization
                        - REMESH + OZ: Recursive bifurcation needs self-organization
                            handlers
            - REMESH + ZHIR: Replicative mutation needs coherence consolidation

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)

        Notes
        -----
        This rule is DISTINCT from general U2 (convergence). While U2 checks
        for destabilizers needing stabilizers, U2-REMESH specifically addresses
    REMESH's amplification property: it multiplies the effect of
    destabilizers
        through recursive feedback across temporal/spatial scales.

        Physical derivation: See src/tnfr/operators/remesh.py module docstring,
    section "Grammar Implications from Physical Analysis" →
    U2: CONVERGENCE.
        """
        # Check if sequence contains REMESH
        has_remesh = any(
            (
                getattr(op, "canonical_name", op.name.lower())
                == "recursivity"
                for op in sequence
            )
        )

        if not has_remesh:
            return True, "U2-REMESH: not applicable (no recursivity present)"

        # Check for destabilizers
        destabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in DESTABILIZERS
        ]

        if not destabilizers_present:
            return True, "U2-REMESH: satisfied (no destabilizers to amplify)"

        # Check for stabilizers
        stabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in STABILIZERS
        ]

        if not stabilizers_present:
            return (
                False,
                f"U2-REMESH violated: recursivity amplifies destabilizers "
                f"{destabilizers_present} via recursive feedback. "
                f"Integral ∫νf·ΔNFR dt may diverge (unbounded growth). "
                f"Required: {sorted(STABILIZERS)} to bound recursive amplification",
            )

        return (
            True,
            f"U2-REMESH satisfied: stabilizers {stabilizers_present} "
            f"bound recursive amplification of {destabilizers_present}",
        )

    @staticmethod
    def validate_multiscale_coherence(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U5: Multi-scale coherence preservation.

        Physical basis: Multi-scale hierarchical structures created by REMESH
        with depth>1 require coherence conservation across scales. This emerges
        inevitably from the nodal equation applied to hierarchical systems.

        From the nodal equation at each hierarchical level:
            ∂EPI_parent/∂t = νf_parent · ΔNFR_parent(t)
            ∂EPI_child_i/∂t = νf_child_i · ΔNFR_child_i(t)  for each child i

        For hierarchical systems with N children:
            EPI_parent = f(EPI_child_1, ..., EPI_child_N)  (structural coupling)

        Taking time derivative and applying chain rule:
            ∂EPI_parent/∂t = Σ (∂f/∂EPI_child_i) · ∂EPI_child_i/∂t
                           = Σ w_i · νf_child_i · ΔNFR_child_i(t)

        where w_i = ∂f/∂EPI_child_i are coupling weights.

        Equating with nodal equation for parent:
            νf_parent · ΔNFR_parent = Σ w_i · νf_child_i · ΔNFR_child_i

        For coherence C(t) = measure of structural stability:
            C_parent ~ 1/|ΔNFR_parent|  (lower pressure = higher coherence)
            C_child_i ~ 1/|ΔNFR_child_i|

        This gives the conservation inequality:
            C_parent ≥ α · Σ C_child_i

        Where α = (1/√N) · η_phase(N) · η_coupling(N) captures:
        - 1/√N: Scale factor from coupling weight distribution
        - η_phase: Phase synchronization efficiency (U3 requirement)
        - η_coupling: Structural coupling efficiency losses
        - Typical range: α ∈ [0.1, 0.4]

        Without stabilizers:
            Deep REMESH (depth>1) creates nested EPIs
            → ΔNFR_parent grows from uncoupled child fluctuations
            → C_parent decreases below α·ΣC_child
            → Violation of conservation → System fragments

        With stabilizers (IL or THOL):
            IL/THOL reduce |ΔNFR| at each level (direct from operator contracts)
            → Maintains C_parent ≥ α·ΣC_child at all hierarchical levels
            → Conservation preserved → Bounded multi-scale evolution

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)

        Notes
        -----
        U5 is INDEPENDENT of U2+U4b:
        - U2/U4b: TEMPORAL dimension (operator sequences in time)
        - U5: SPATIAL dimension (hierarchical nesting in structure)

        Decision test case that passes U2+U4b but fails U5:
            [AL, REMESH(depth=3), SHA]
            - U2: ✓ No destabilizers (trivially convergent)
            - U4b: ✓ REMESH not a transformer (U4b doesn't apply)
            - U5: ✗ Deep recursivity without stabilization → fragmentation

        Physical derivation: See UNIFIED_GRAMMAR_RULES.md § U5
        Canonicity: STRONG (derived from nodal equation + structural coupling)

        References
        ----------
        - TNFR.pdf § 2.1: Nodal equation ∂EPI/∂t = νf · ΔNFR(t)
    - Problem statement: "The Pulse That Traverses Us.pdf"
        - AGENTS.md: Invariant #7 (Operational Fractality)
        - Contract IL: Reduces |ΔNFR| at all scales
        - Contract THOL: Autopoietic closure across hierarchical levels
        """
        # Check for deep REMESH (depth > 1)
        # Note: Currently Recursivity doesn't expose depth parameter in operator
        # This is a forward-looking validation for when depth is added
        deep_remesh_indices = []

        for i, op in enumerate(sequence):
            op_name = getattr(op, "canonical_name", op.name.lower())
            if op_name == "recursivity":
                # Check if operator has depth attribute
                depth = getattr(op, "depth", 1)  # Default depth=1 if not present
                if depth > 1:
                    deep_remesh_indices.append((i, depth))

        if not deep_remesh_indices:
            # No deep REMESH present, U5 not applicable
            return True, "U5: not applicable (no deep recursivity depth>1 present)"

        # For each deep REMESH, check for stabilizers in window
        violations = []
        for idx, depth in deep_remesh_indices:
            # Check window of ±3 operators for scale stabilizers
            window_start = max(0, idx - 3)
            window_end = min(len(sequence), idx + 4)

            has_stabilizer = False
            stabilizers_in_window = []

            for j in range(window_start, window_end):
                op_name = getattr(sequence[j], "canonical_name", sequence[j].name.lower())
                if op_name in SCALE_STABILIZERS:
                    has_stabilizer = True
                    stabilizers_in_window.append((j, op_name))

            if not has_stabilizer:
                violations.append(
                    f"recursivity at position {idx} (depth={depth}) lacks scale "
                    f"stabilizer in window [{window_start}:{window_end}]. "
                    f"Deep hierarchical nesting requires {sorted(SCALE_STABILIZERS)} "
                    f"for multi-scale coherence preservation (C_parent ≥ α·ΣC_child)"
                )

        if violations:
            return (False, f"U5 violated: {'; '.join(violations)}")

        return (
            True,
            "U5 satisfied: deep recursivity has scale stabilizers "
            "for multi-scale coherence preservation",
        )

    @staticmethod
    def validate_temporal_ordering(
        sequence: List[Operator],
        vf: float = 1.0,
        k_top: float = 1.0,
    ) -> tuple[bool, str]:
        """Validate U6: Temporal ordering (EXPERIMENTAL).

        **Status:** RESEARCH PHASE - Not Canonical
        **Canonicity:** MODERATE (40-55% confidence)

        Physical basis: After destabilizers inject structural pressure (increase
        |ΔNFR| and/or |∂²EPI/∂t²|), the network requires relaxation time for
        stabilizers to restore boundedness. Applying a second destabilizer
        too early causes nonlinear accumulation α(Δt) > 1 and risks coherence
        fragmentation via bifurcation cascades.

        From post-bifurcation relaxation dynamics:
            ΔNFR(t) = ΔNFR_0 · exp(-t/τ_damp) + ΔNFR_eq

        Relaxation time:
            τ_relax = τ_damp · ln(1/ε)
            τ_damp = (k_top / νf) · k_op

        Where:
        - k_top: topological factor (spectral gap dependent)
        - k_op: operator depth factor (OZ≈1.0, ZHIR≈1.5)
        - ε: recovery threshold (default 0.05 for 95% recovery)

        Sequence-based approximation: When physical time unavailable, require
        minimum operator spacing between destabilizers (~3 operators for νf=1.0).

        Parameters
        ----------
        sequence : List[Operator]
            Sequence to validate
        vf : float, optional
            Structural frequency (Hz_str) for time estimation (default: 1.0)
        k_top : float, optional
            Topological factor (default: 1.0, radial/star topology)

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
            Note: Violations generate warnings, not hard failures (experimental)

        Notes
        -----
        **Limitations preventing canonical status:**
        - Not formally derived from nodal equation (modeled, not proven)
        - Parameters k_top, k_op not yet computed from first principles
        - Empirical validation pending (correlation with C(t) fragmentation)
        - Conflates logical ordering with temporal spacing

        **Validation criteria for STRONG canonicity:**
        - >80% of violations cause coherence loss exceeding δC threshold
        - Derivation showing ∫νf·ΔNFR diverges without spacing
        - Parameters endogenized (k_top from spectral analysis, etc.)

        See docs/grammar/U6_TEMPORAL_ORDERING.md for complete derivation,
        experiments, and elevation roadmap.
        """
        # Check for destabilizers that trigger relaxation requirement
        destabilizer_positions = []
        for i, op in enumerate(sequence):
            op_name = getattr(op, "canonical_name", op.name.lower())
            if op_name in {"dissonance", "mutation", "expansion"}:
                destabilizer_positions.append((i, op_name))

        if len(destabilizer_positions) < 2:
            return True, "U6: not applicable (fewer than 2 destabilizers)"

        # Estimate minimum operator spacing from τ_relax
        # Assumption: each operator ≈ 1 structural time unit
        # τ_relax ≈ (k_top / νf) · ln(1/ε) · k_op
        # For k_op≈1.0 (OZ baseline), ε=0.05: ln(1/0.05)≈3.0
        k_op_baseline = 1.0
        tau_relax = (k_top / vf) * k_op_baseline * (3.0)  # ln(20) ≈ 3.0

        # Convert to operator positions (coarse: 1 op ≈ 1 time unit)
        min_spacing = max(2, int(tau_relax))  # At least 2 operators

        # Check spacing between consecutive destabilizers
        violations = []
        for j in range(1, len(destabilizer_positions)):
            prev_idx, prev_op = destabilizer_positions[j - 1]
            curr_idx, curr_op = destabilizer_positions[j]
            spacing = curr_idx - prev_idx

            if spacing <= min_spacing:
                # Calculate estimated τ_relax for this pair
                k_op_prev = 1.5 if prev_op == "mutation" else 1.0
                tau_est = (k_top / vf) * k_op_prev * 3.0

                violations.append(
                    f"{curr_op} at position {curr_idx} follows {prev_op} "
                    f"at position {prev_idx} (spacing={spacing} operators). "
                    f"Estimated τ_relax≈{tau_est:.2f} time units "
                    f"(≈{int(tau_est)} operators). Risk: nonlinear ΔNFR "
                    f"accumulation α(Δt)>1, bifurcation cascade, C(t) fragmentation"
                )

        if violations:
            return (
                False,
                f"U6 WARNING (experimental): {'; '.join(violations)}. "
                f"See docs/grammar/U6_TEMPORAL_ORDERING.md",
            )

        return (
            True,
            f"U6 satisfied: destabilizers properly spaced (min {min_spacing} operators)",
        )

    def validate(
        self,
        sequence: List[Operator],
        epi_initial: float = 0.0,
        vf: float = 1.0,
        k_top: float = 1.0,
        stop_on_first_error: bool = False,
    ) -> tuple[bool, List[str]]:
        """Validate sequence using all unified canonical constraints.

        This validates pure TNFR physics:
        - U1: Structural initiation & closure
        - U2: Convergence & boundedness
        - U3: Resonant coupling
        - U4: Bifurcation dynamics
        - U5: Multi-scale coherence
        - U6: Temporal ordering (if experimental_u6=True)

        Parameters
        ----------
        sequence : List[Operator]
            Sequence to validate
        epi_initial : float, optional
            Initial EPI value (default: 0.0)
        vf : float, optional
            Structural frequency for U6 timing (default: 1.0)
        k_top : float, optional
            Topological factor for U6 timing (default: 1.0)
        stop_on_first_error : bool, optional
            If True, return immediately on first constraint violation
            (early exit optimization). If False, collect all violations.
            Default: False (comprehensive reporting)

        Returns
        -------
        tuple[bool, List[str]]
            (is_valid, messages)
            is_valid: True if all constraints satisfied
            messages: List of validation messages

        Performance
        -----------
        Early exit (stop_on_first_error=True) can provide 10-30% speedup
        when sequences have errors, at cost of incomplete diagnostics.
        """
        messages = []
        all_valid = True

        # U1a: Initiation
        valid_init, msg_init = self.validate_initiation(sequence, epi_initial)
        messages.append(f"U1a: {msg_init}")
        all_valid = all_valid and valid_init
        if stop_on_first_error and not valid_init:
            return False, messages

        # U1b: Closure
        valid_closure, msg_closure = self.validate_closure(sequence)
        messages.append(f"U1b: {msg_closure}")
        all_valid = all_valid and valid_closure
        if stop_on_first_error and not valid_closure:
            return False, messages

        # U2: Convergence
        valid_conv, msg_conv = self.validate_convergence(sequence)
        messages.append(f"U2: {msg_conv}")
        all_valid = all_valid and valid_conv
        if stop_on_first_error and not valid_conv:
            return False, messages

        # U3: Resonant coupling
        valid_coupling, msg_coupling = self.validate_resonant_coupling(sequence)
        messages.append(f"U3: {msg_coupling}")
        all_valid = all_valid and valid_coupling
        if stop_on_first_error and not valid_coupling:
            return False, messages

        # U4a: Bifurcation triggers
        valid_triggers, msg_triggers = self.validate_bifurcation_triggers(sequence)
        messages.append(f"U4a: {msg_triggers}")
        all_valid = all_valid and valid_triggers
        if stop_on_first_error and not valid_triggers:
            return False, messages

        # U4b: Transformer context
        valid_context, msg_context = self.validate_transformer_context(sequence)
        messages.append(f"U4b: {msg_context}")
        all_valid = all_valid and valid_context
        if stop_on_first_error and not valid_context:
            return False, messages

        # U2-REMESH: Recursive amplification control
        valid_remesh, msg_remesh = self.validate_remesh_amplification(sequence)
        messages.append(f"U2-REMESH: {msg_remesh}")
        all_valid = all_valid and valid_remesh
        if stop_on_first_error and not valid_remesh:
            return False, messages

        # U5: Multi-scale coherence
        valid_multiscale, msg_multiscale = self.validate_multiscale_coherence(sequence)
        messages.append(f"U5: {msg_multiscale}")
        all_valid = all_valid and valid_multiscale
        if stop_on_first_error and not valid_multiscale:
            return False, messages

        # U6: Temporal ordering (experimental)
        if self.experimental_u6:
            valid_temporal, msg_temporal = self.validate_temporal_ordering(
                sequence, vf=vf, k_top=k_top
            )
            messages.append(f"U6 (experimental): {msg_temporal}")
            # Note: U6 violations generate warnings, not hard failures
            # all_valid intentionally not updated for experimental rule

        return all_valid, messages

    # --- U6 Telemetry Warning Aggregator (non-blocking) ---
    def telemetry_warnings(
        self,
        G: Any,
        *,
        phi_grad_threshold: float = 0.38,
        kphi_abs_threshold: float = 3.0,
        kphi_multiscale: bool = True,
        kphi_alpha_hint: float | None = 2.76,
        xi_regime_multipliers: tuple[float, float] = (1.0, 3.0),
    ) -> list[str]:
        """Compute U6 telemetry warnings for |∇φ|, K_φ, and ξ_C (non-blocking).

        Returns a list of human-readable messages. Intended as a convenience
        aggregator; does not affect structural validation outcome (U1–U5).
        """
        messages: list[str] = []

        try:
            safe_g, stats_g, msg_g, _ = warn_phase_gradient_telemetry(
                G, threshold=phi_grad_threshold
            )
            messages.append(msg_g)
        except Exception as e:  # pragma: no cover
            messages.append(f"U6 (|∇φ|): telemetry error: {e}")

        try:
            safe_k, stats_k, msg_k, _ = warn_phase_curvature_telemetry(
                G,
                abs_threshold=kphi_abs_threshold,
                multiscale_check=kphi_multiscale,
                alpha_hint=kphi_alpha_hint,
            )
            messages.append(msg_k)
        except Exception as e:  # pragma: no cover
            messages.append(f"U6 (K_φ): telemetry error: {e}")

        try:
            safe_x, stats_x, msg_x = warn_coherence_length_telemetry(
                G, regime_multipliers=xi_regime_multipliers
            )
            messages.append(msg_x)
        except Exception as e:  # pragma: no cover
            messages.append(f"U6 (ξ_C): telemetry error: {e}")

        return messages


