"""TNFR Grammar: Core Grammar Validator

``GrammarValidator`` is the central validator for the operator-word rules
U1--U5 and the U2-REMESH sub-rule. Canonical U6 is a before/after structural-
potential observation implemented in :mod:`tnfr.operators.grammar_u6`; it
cannot be decided from an operator sequence alone.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph
    from .definitions import Operator
else:
    NodeId = Any
    TNFRGraph = Any
    from .definitions import Operator

from ..constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
)
from .grammar_telemetry import (
    warn_coherence_length_telemetry,
    warn_phase_curvature_telemetry,
    warn_phase_gradient_telemetry,
)
from ..config.operator_names import BIFURCATION_WINDOW, U2_DEBT_CAPACITY
from .grammar_debt import advance_debt
from .grammar_types import (
    BIFURCATION_HANDLERS,
    BIFURCATION_TRIGGERS,
    CLOSURES,
    COUPLING_RESONANCE,
    DESTABILIZERS,
    GENERATORS,
    SCALE_STABILIZERS,
    STABILIZERS,
    TRANSFORMERS,
)


class GrammarValidator:
    """Validates sequences using canonical TNFR grammar constraints.

    Implements the sequence-level U1--U5 engine policies. This is the central
    sequence validator; canonical U6 requires field snapshots and is evaluated
    by :mod:`tnfr.operators.grammar_u6`.

    The policies are motivated and constrained by:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - Canonical invariants (AGENTS.md §3)
    - Formal contracts (AGENTS.md §4)

    No organizational conventions are enforced.

    Parameters
    ----------
    experimental_u6 : bool, optional
        Enable experimental temporal-ordering validation (default: False).
        This check is labelled **U6-EXP** internally to distinguish it from
        canonical U6 = Φ_s Structural Potential Confinement (grammar_u6.py).
        When enabled, sequences are checked for temporal spacing violations
        after destabilizers.  Violations log warnings but do not fail
        validation (all_valid is NOT updated by this rule).
    """

    def __init__(self, experimental_u6: bool = False):
        """Initialize validator with optional experimental features.

        Parameters
        ----------
        experimental_u6 : bool, optional
            Enable U6-EXP temporal ordering checks (default: False).
            Does NOT correspond to canonical U6 (Φ_s confinement).
        """
        self.experimental_u6 = experimental_u6

    @staticmethod
    def validate_initiation(
        sequence: list[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, str]:
        """Validate U1a: Structural initiation.

        Contract basis: the derivative ``νf·ΔNFR`` is defined at ``EPI=0``
        whenever its factors are finite. U1a is an operator-history policy:
        a standalone word starting from the null state must declare how form
        is generated or latent form is activated.

        Generators create structure from:
        - AL (Emission): vacuum via emission
        - NAV (Transition): latent EPI via regime shift
        - REMESH (Recursivity): dormant structure across scales

        Parameters
        ----------
        sequence : list[Operator]
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
    def validate_closure(sequence: list[Operator]) -> tuple[bool, str]:
        """Validate U1b: registered structural closure.

        This syntactic rule checks whether the last operator belongs to the
        registered closure set. Membership records an operational boundary; it
        does not prove that the resulting trajectory is at a coherent
        attractor.

        Closures stabilize via:
        - SHA (Silence): Terminal closure - freezes evolution (νf → 0)
        - NAV (Transition): Handoff closure - transfers to next regime
        - REMESH (Recursivity): Recursive closure - distributes across scales
        - OZ (Dissonance): Intentional closure - preserves activation/tension

        Parameters
        ----------
        sequence : list[Operator]
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
    def validate_convergence(sequence: list[Operator]) -> tuple[bool, str]:
        """Validate the finite-word U2 debt and coverage policy.

        The public method name is retained for compatibility. The check counts
        declared destabilizer debt, rejects any prefix above
        ``U2_DEBT_CAPACITY``, and requires at least one declared stabilizer when
        destabilizers occur. It does not integrate ``νf·ΔNFR`` or prove
        convergence, boundedness, or a Lyapunov inequality for the executed
        trajectory.

        Parameters
        ----------
        sequence : list[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        names = [getattr(op, "canonical_name", op.name.lower()) for op in sequence]
        debt = 0
        for index, name in enumerate(names):
            debt = advance_debt(debt, name)
            if debt > U2_DEBT_CAPACITY:
                return False, (
                    f"U2 violated: uncompensated destabilizer debt={debt} "
                    f"exceeds capacity {U2_DEBT_CAPACITY} at position {index}. "
                    "A later stabilizer cannot repair an over-capacity prefix."
                )
        destabilizers_present = [
            name for name in names if name in DESTABILIZERS
        ]

        if not destabilizers_present:
            # No declared destabilizer means that U2 debt is not opened. Other
            # dynamics can still be unbounded and require trajectory analysis.
            return True, "U2: not applicable (no destabilizers present)"

        # Check for stabilizers
        stabilizers_present = [
            name for name in names if name in STABILIZERS
        ]

        if not stabilizers_present:
            return (
                False,
                f"U2 violated: destabilizers {destabilizers_present} present "
                f"without declared stabilizer coverage. "
                f"Add: {sorted(STABILIZERS)}",
            )

        return (
            True,
            f"U2 coverage satisfied: stabilizers {stabilizers_present} "
            f"cover destabilizers {destabilizers_present}; trajectory "
            "boundedness remains a telemetry question",
        )

    @staticmethod
    def validate_resonant_coupling(
        sequence: list[Operator],
    ) -> tuple[bool, str]:
        """Validate U3: Resonant coupling.

            Physical basis: AGENTS.md Invariant #2 states "no coupling is valid
            without explicit phase verification (synchrony)".

            Resonance physics requires phase compatibility:
                |wrap(φᵢ - φⱼ)| ≤ Δφ_max

            Without phase verification:
                Nodes with incompatible phases (antiphase) could attempt coupling
                → Destructive interference → Violates resonance physics

            With phase verification:
                Only synchronous nodes couple → Constructive interference

            Parameters
            ----------
            sequence : list[Operator]
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
            that phase checks are MANDATORY (Invariant #2), not optional.
        """
        # Check if sequence contains coupling/resonance operators
        coupling_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in (COUPLING_RESONANCE)
        ]

        if not coupling_ops:
            # No coupling/resonance = U3 not applicable
            return True, "U3: not applicable (no coupling/resonance operators)"

        # U3 satisfied: Sequence contains coupling/resonance
        # Phase verification is MANDATORY per Invariant #2
        # Actual check happens in operator preconditions
        return (
            True,
            (
                "U3 awareness: operators "
                f"{coupling_ops} require phase verification "
                "(MANDATORY per Invariant #2). Enforced in preconditions."
            ),
        )

    @staticmethod
    def validate_bifurcation_triggers(
        sequence: list[Operator],
    ) -> tuple[bool, str]:
        """Validate U4a: Bifurcation triggers need handlers.

        Physical basis: AGENTS.md Contract OZ states dissonance may trigger
        bifurcation if ∂²EPI/∂t² > τ. When bifurcation is triggered, handlers
        are required to manage structural reorganization.

        Bifurcation physics:
            If ∂²EPI/∂t² > τ → multiple reorganization paths viable
            → System enters bifurcation regime
            → Requires declared handling coverage (THOL or IL)

        Parameters
        ----------
        sequence : list[Operator]
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
            if getattr(op, "canonical_name", op.name.lower()) in (BIFURCATION_TRIGGERS)
        ]

        if not trigger_ops:
            # No triggers = U4a not applicable
            return True, "U4a: not applicable (no bifurcation triggers)"

        # Check for handlers
        handler_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in (BIFURCATION_HANDLERS)
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
        sequence: list[Operator],
    ) -> tuple[bool, str]:
        """Validate U4b: Transformers need context.

        Policy basis: transformers (ZHIR, THOL) need a recent declared
        perturbation before a structural change. Because the destabilizer set
        spans pressure (OZ), phase (ZHIR), and capacity (VAL), this label-only
        check does not assert a common energy or |ΔNFR| threshold.

        ZHIR (Mutation) requirements:
            1. Prior IL: Stable base prevents transformation from chaos
            2. Recent destabilizer: Declared perturbation context

        THOL (Self-organization) requirements:
            1. Recent destabilizer: Declared perturbation context

        "Recent" = within ``BIFURCATION_WINDOW`` operator positions. The
        window is calibrated from a scalar mean-rate surrogate and is a policy
        parameter, not a topology-independent relaxation time.

        Parameters
        ----------
        sequence : list[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)

        Notes
        -----
        The single ``BIFURCATION_WINDOW`` supplies one deterministic recency
        rule for every destabilizer. It does not observe |ΔNFR| or establish
        that a graph mode remains above a physical threshold.
        """
        # Check if sequence contains transformers
        transformer_ops = []
        has_prior_il = False
        for i, op in enumerate(sequence):
            op_name = getattr(op, "canonical_name", op.name.lower())
            if op_name in TRANSFORMERS:
                transformer_ops.append((i, op_name, has_prior_il))
            if op_name == "coherence":
                has_prior_il = True

        if not transformer_ops:
            return True, "U4b: not applicable (no transformers)"

        # For each transformer, check context
        violations = []
        for idx, transformer_name, prior_il in transformer_ops:
            # "Recent" is the configured scalar-surrogate policy window. It is
            # deterministic and shared by all destabilizers but does not certify
            # modal relaxation on the current topology. Single source:
            # config.operator_names.BIFURCATION_WINDOW.
            window_start = max(0, idx - BIFURCATION_WINDOW)
            recent_destabilizers = []
            # The window constrains declared perturbation context across the
            # pressure, phase, and capacity roles. U4b requires a prior stable
            # base, captured by the linear scan above without expiring earlier
            # IL or rescanning each sequence prefix.

            for j in range(window_start, idx):
                op_name = getattr(
                    sequence[j],
                    "canonical_name",
                    sequence[j].name.lower(),
                )
                if op_name in DESTABILIZERS:
                    recent_destabilizers.append((j, op_name))

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
        sequence: list[Operator],
    ) -> tuple[bool, str]:
        """Validate the finite-word U2-REMESH coverage sub-rule.

            REMESH mixes present and delayed EPI snapshots. When a word also
            contains a declared destabilizer, policy requires IL or THOL. This
            presence check records coverage only: it neither evaluates the
            runtime delayed recurrence nor proves amplification, boundedness,
            convergence, or fragmentation.

            Specific combinations requiring stabilizers:
                - REMESH + VAL: Recursive expansion needs coherence stabilization
                            - REMESH + OZ: Recursive bifurcation needs self-organization
                                handlers
                - REMESH + ZHIR: Replicative mutation needs coherence consolidation

            Parameters
            ----------
            sequence : list[Operator]
                Sequence of operators to validate

            Returns
            -------
            tuple[bool, str]
                (is_valid, message)

            Notes
            -----
            This rule is distinct from the general U2 debt check. It records
            the extra stabilizer obligation attached to a word containing both
            REMESH and a destabilizer.

            Physical derivation: See src/tnfr/operators/remesh.py module docstring,
        section "Grammar implications" → U2 debt coverage.
        """
        # Check if sequence contains REMESH
        has_remesh = any(
            (
                getattr(op, "canonical_name", op.name.lower()) == "recursivity"
                for op in sequence
            )
        )

        if not has_remesh:
            return True, "U2-REMESH: not applicable (no recursivity present)"

        # DESIGN NOTE (B4): Same presence-only limitation as U2 — ordering of
        # stabilizers relative to destabilizers is not verified here.  See B4.
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
                f"U2-REMESH violated: recursivity appears with destabilizers "
                f"{destabilizers_present} but has no declared stabilizer "
                f"coverage. Required: {sorted(STABILIZERS)}",
            )

        return (
            True,
            f"U2-REMESH coverage satisfied: stabilizers {stabilizers_present} "
            f"cover recursivity with {destabilizers_present}; delayed-state "
            "stability is not inferred",
        )

    @staticmethod
    def validate_multiscale_coherence(sequence: list[Operator]) -> tuple[bool, str]:
        """Validate U5: Multi-scale coherence preservation.

            The sequence layer can inspect declared Recursivity depth and require
            a nearby scale stabilizer. It has no parent/child field snapshots, so
            it cannot evaluate ``C_parent >= alpha*sum(C_child)`` or infer
            multiscale conservation, boundedness, or fragmentation. Those are
            trajectory-level observations with an explicit hierarchy and alpha.

            Parameters
            ----------
            sequence : list[Operator]
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

            Sequence-policy example that passes U2+U4b but fails this U5 check:
                [AL, REMESH(depth=3), SHA]
                - U2: ✓ No declared destabilizer debt
                - U4b: ✓ REMESH not a transformer (U4b doesn't apply)
                - U5: ✗ Deep recursivity lacks declared scale-stabilizer coverage

            Scope analysis: see ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md``.

            References
            ----------
            - TNFR.pdf § 2.1: Nodal equation ∂EPI/∂t = νf · ΔNFR(t)
        - Problem statement: "The Pulse That Traverses Us.pdf"
            - AGENTS.md: Invariant #3 (Multi-Scale Fractality)
            - Contract IL: pressure-reduction role
            - Contract THOL: autopoietic reorganization and hierarchy handling
        """
        from .recursivity import validate_recursivity_depth

        # Recursivity exposes depth. This check validates that declared scale
        # and nearby stabilizers; it does not measure parent/child coherence.
        deep_remesh_indices = []

        for i, op in enumerate(sequence):
            op_name = getattr(op, "canonical_name", op.name.lower())
            if op_name == "recursivity":
                try:
                    depth = validate_recursivity_depth(getattr(op, "depth", 1))
                except ValueError as exc:
                    return False, f"U5 violated: recursivity at position {i}: {exc}"
                if depth > 1:
                    deep_remesh_indices.append((i, depth))

        if not deep_remesh_indices:
            return True, "U5: not applicable — sequence has no recursivity depth > 1"

        # For each deep REMESH, check for stabilizers in window
        violations = []
        for idx, depth in deep_remesh_indices:
            # Scale stabilizers must fall within the configured policy window
            # on either side. This reuses BIFURCATION_WINDOW for deterministic
            # bookkeeping; it is not a topology-independent physical reach.
            window_start = max(0, idx - BIFURCATION_WINDOW)
            window_end = min(len(sequence), idx + BIFURCATION_WINDOW + 1)

            has_stabilizer = False
            stabilizers_in_window = []

            for j in range(window_start, window_end):
                op_name = getattr(
                    sequence[j], "canonical_name", sequence[j].name.lower()
                )
                if op_name in SCALE_STABILIZERS:
                    has_stabilizer = True
                    stabilizers_in_window.append((j, op_name))

            if not has_stabilizer:
                violations.append(
                    f"recursivity at position {idx} (depth={depth}) lacks scale "
                    f"stabilizer in window [{window_start}:{window_end}]. "
                    f"Deep hierarchical nesting requires {sorted(SCALE_STABILIZERS)} "
                    "for declared multi-scale coverage; evaluate parent/child "
                    "coherence separately"
                )

        if violations:
            return (False, f"U5 violated: {'; '.join(violations)}")

        return (
            True,
            "U5 sequence coverage satisfied: deep recursivity has nearby scale "
            "stabilizers; parent/child coherence was not measured",
        )

    @staticmethod
    def validate_temporal_ordering(
        sequence: list[Operator],
        vf: float = 1.0,
        k_top: float = 1.0,
    ) -> tuple[bool, str]:
        """Validate U6-EXP: Temporal ordering (EXPERIMENTAL — NOT canonical U6).

        .. warning::
            This rule is labelled **U6-EXP** (temporal ordering) to avoid
            confusion with canonical U6 = Φ_s Structural Potential Confinement
            implemented in grammar_u6.py.  Both share the number "U6" in
            earlier drafts; the canonical definition wins.

        **Status:** RESEARCH PHASE - Not Canonical
        **Canonicity:** MODERATE (40-55% confidence)

        Model premise: after declared destabilizers, a spacing surrogate may
        flag possible accumulation before a stabilizer has acted. It is an
        experimental warning and does not establish a relaxation time,
        boundedness, or fragmentation for the executed graph.

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
        sequence : list[Operator]
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
            return True, "U6-EXP: not applicable (fewer than 2 destabilizers)"

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
                f"U6-EXP WARNING (experimental): {'; '.join(violations)}. "
                f"See docs/grammar/U6_TEMPORAL_ORDERING.md",
            )

        return (
            True,
            f"U6-EXP surrogate satisfied: destabilizers spaced by the "
            f"configured estimate (min {min_spacing} operators)",
        )

    def validate(
        self,
        sequence: list[Operator],
        epi_initial: float = 0.0,
        vf: float = 1.0,
        k_top: float = 1.0,
        stop_on_first_error: bool = False,
    ) -> tuple[bool, list[str]]:
        """Validate the sequence-level grammar policies available here.

        This validates:
        - U1: Structural initiation & closure
        - U2: finite debt/coverage policy (+ U2-REMESH sub-rule)
        - U3: presence awareness; phase values are checked by operator preconditions
        - U4: Bifurcation dynamics
        - U5: Declared Recursivity depth and nearby scale stabilizers
        - U6-EXP: Temporal ordering (experimental; DISTINCT from canonical U6 = Φ_s
          confinement in grammar_u6.py — enabled only when experimental_u6=True)

        Canonical U6 is absent because this method receives no before/after
        structural-potential snapshots. A ``True`` result therefore means that
        the available sequence policies passed; it is not full U1--U6 runtime
        certification.

        Parameters
        ----------
        sequence : list[Operator]
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
        tuple[bool, list[str]]
            (is_valid, messages)
            is_valid: True if all constraints satisfied
            messages: list of validation messages

        Performance
        -----------
        Early exit (stop_on_first_error=True) avoids later validation steps after
        an error, at the cost of incomplete diagnostics. Runtime depends on the
        sequence and which constraint fails.
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

        # U2: finite debt and stabilizer coverage (legacy method name retained)
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

        # U6-EXP: Temporal ordering (experimental).
        # DISTINCT from canonical U6 = Φ_s Structural Potential Confinement
        # (grammar_u6.py). all_valid is intentionally NOT updated here.
        if self.experimental_u6:
            valid_temporal, msg_temporal = self.validate_temporal_ordering(
                sequence, vf=vf, k_top=k_top
            )
            messages.append(f"U6-EXP (temporal ordering, experimental): {msg_temporal}")

        return all_valid, messages

    # --- U6 Telemetry Warning Aggregator (non-blocking) ---
    def telemetry_warnings(
        self,
        G: Any,
        *,
        phi_grad_threshold: float = GRAD_PHI_CANONICAL_THRESHOLD,  # canonical (≈ 0.196, π/16)
        kphi_abs_threshold: float = K_PHI_CANONICAL_THRESHOLD,  # 0.9×π canonical hotspot flag
        kphi_multiscale: bool = True,
        kphi_alpha_hint: float | None = 2.76,
        xi_regime_multipliers: tuple[float, float] = (1.0, 3.0),
    ) -> list[str]:
        """Compute structural field tetrad telemetry warnings (non-blocking).

        Monitors |∇φ|, K_φ, and ξ_C safety thresholds. These are auxiliary
        structural health diagnostics from the tetrad, distinct from U6
        grammar rule (Φ_s structural potential confinement in grammar_u6.py).

        Returns a list of human-readable messages. Does not affect
        structural validation outcome (U1–U5).
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
