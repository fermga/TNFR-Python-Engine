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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph
    from .definitions import Operator
else:
    NodeId = Any
    TNFRGraph = Any
    from .definitions import Operator

from .._exact_time import finite_represented_real
from ..config.operator_names import (
    BIFURCATION_WINDOW,
    CANONICAL_OPERATOR_NAMES,
    U2_DEBT_CAPACITY,
)
from ..config.parsing import parse_bool
from ..constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
)
from ..types import require_finite_real_scalar_epi
from .grammar_debt import advance_debt
from .grammar_telemetry import (
    warn_coherence_length_telemetry,
    warn_phase_curvature_telemetry,
    warn_phase_gradient_telemetry,
)
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
    _operator_name,
)


@dataclass(frozen=True, slots=True)
class GrammarCheckResult:
    """One shared word check; advisory failures do not reject the word."""

    rule: str
    passed: bool
    message: str
    blocking: bool = True

    def legacy_message(self) -> str:
        """Preserve the existing human-readable validation message."""
        label = (
            "U6-EXP (temporal ordering, experimental)"
            if self.rule == "U6-EXP"
            else self.rule
        )
        return f"{label}: {self.message}"


class GrammarValidator:
    """Validates sequences using canonical TNFR grammar constraints.

    Implements the sequence-level U1--U5 engine policies. This is the central
    sequence validator; canonical U6 requires field snapshots and is evaluated
    by :mod:`tnfr.operators.grammar_u6`.

    The policies are motivated and constrained by:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - Canonical invariants (AGENTS.md §3)
    - Formal contracts (AGENTS.md §4)

    Calibrated word policies do not replace live operator gates or trajectory
    theorems. Their derivation limits are documented in the grammar scope note.

    Parameters
    ----------
    experimental_u6 : bool, optional
        Enable experimental temporal-ordering validation (default: False).
        This check is labelled **U6-EXP** internally to distinguish it from
        canonical U6 = Φ_s Structural Potential Confinement (grammar_u6.py).
        When enabled, sequences are checked for temporal spacing violations
        after destabilizers. Violations append diagnostic messages but do not
        fail validation (all_valid is NOT updated by this rule).
    """

    def __init__(self, experimental_u6: bool = False):
        """Initialize validator with optional experimental features.

        Parameters
        ----------
        experimental_u6 : bool, optional
            Enable U6-EXP temporal ordering checks (default: False).
            Does NOT correspond to canonical U6 (Φ_s confinement).
        """
        self.experimental_u6 = parse_bool(experimental_u6)

    @staticmethod
    def validate_initiation(
        sequence: list[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, str]:
        """Validate U1a: Structural initiation.

        Contract basis: the derivative ``νf·ΔNFR`` is defined at ``EPI=0``
        whenever its factors are finite. U1a is an operator-history policy:
        a standalone word starting from zero form must declare how form
        is generated or latent form is activated. Zero EPI does not remove
        the existing node, phase, capacity or support.

        Generators create structure from:
        - AL (Emission): form change on an existing node
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
        epi = require_finite_real_scalar_epi(epi_initial, "initial EPI")
        if epi != 0.0:
            # Already initialized, no generator required
            return True, "U1a: EPI!=0, initiation not required"

        if not sequence:
            return False, "U1a violated: Empty sequence with EPI=0"

        first_op = _operator_name(sequence[0])

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

        Registered closure modes:
        - SHA (Silence): attenuates capacity; it need not make it zero
        - NAV (Transition): declared regime handoff
        - REMESH (Recursivity): declared recursive endpoint
        - OZ (Dissonance): permits intentional activation/tension at the endpoint

        Their separate live contracts determine actual effects. Endpoint-role
        membership does not guarantee future immobility or stabilization.

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

        last_op = _operator_name(sequence[-1])

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
        names = [_operator_name(op) for op in sequence]
        debt = 0
        for index, name in enumerate(names):
            debt = advance_debt(debt, name)
            if debt > U2_DEBT_CAPACITY:
                return False, (
                    f"U2 violated: uncompensated destabilizer debt={debt} "
                    f"exceeds capacity {U2_DEBT_CAPACITY} at position {index}. "
                    "A later stabilizer cannot repair an over-capacity prefix."
                )
        destabilizers_present = [name for name in names if name in DESTABILIZERS]

        if not destabilizers_present:
            # No declared destabilizer means that U2 debt is not opened. Other
            # dynamics can still be unbounded and require trajectory analysis.
            return True, "U2: not applicable (no destabilizers present)"

        # Check for stabilizers
        stabilizers_present = [name for name in names if name in STABILIZERS]

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

            The U3 contract requires an explicit wrapped phase comparison:
                |wrap(φᵢ - φⱼ)| ≤ Δφ_max.

            This word reader records which operators require the live gate.
            It receives no node phases and cannot certify compatibility,
            synchronization or preservation of the gate during later motion.

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
            _operator_name(op)
            for op in sequence
            if _operator_name(op) in (COUPLING_RESONANCE)
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

        This word policy requires a declared handler (THOL or IL) whenever
        a trigger-role operator occurs. A name neither establishes an observed
        threshold crossing nor proves a bifurcation or successful stabilization.
        Temporal evidence and live operator admission remain separate.

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
            _operator_name(op)
            for op in sequence
            if _operator_name(op) in (BIFURCATION_TRIGGERS)
        ]

        if not trigger_ops:
            # No triggers = U4a not applicable
            return True, "U4a: not applicable (no bifurcation triggers)"

        # Check for handlers
        handler_ops = [
            _operator_name(op)
            for op in sequence
            if _operator_name(op) in (BIFURCATION_HANDLERS)
        ]

        if not handler_ops:
            return (
                False,
                (
                    "U4a violated: bifurcation triggers "
                    f"{trigger_ops} present without handler. "
                    "No temporal threshold or trajectory outcome was evaluated. "
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
            1. Prior IL: Registered stable-base history
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
            op_name = _operator_name(op)
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
                op_name = _operator_name(sequence[j])
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

        Policy scope: see ``theory/UNIFIED_GRAMMAR_RULES.md`` and the
        REMESH delayed-state contract in ``docs/contracts/OPERATOR_EVENTS.md``.
        """
        # Check if sequence contains REMESH
        has_remesh = any((_operator_name(op) == "recursivity" for op in sequence))

        if not has_remesh:
            return True, "U2-REMESH: not applicable (no recursivity present)"

        # DESIGN NOTE (B4): Same presence-only limitation as U2 — ordering of
        # stabilizers relative to destabilizers is not verified here.  See B4.
        destabilizers_present = [
            _operator_name(op) for op in sequence if _operator_name(op) in DESTABILIZERS
        ]

        if not destabilizers_present:
            return True, "U2-REMESH: satisfied (no destabilizers to amplify)"

        # Check for stabilizers
        stabilizers_present = [
            _operator_name(op) for op in sequence if _operator_name(op) in STABILIZERS
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
        - theory/FUNDAMENTAL_THEORY.md: Nodal identity and complete laws
        - theory/UNIFIED_GRAMMAR_RULES.md: U5 declared hierarchy policy
        - theory/STRUCTURAL_OPERATORS.md: IL and THOL event contracts
        """
        from .recursivity import validate_recursivity_depth

        # Recursivity exposes depth. This check validates that declared scale
        # and nearby stabilizers; it does not measure parent/child coherence.
        deep_remesh_indices = []

        for i, op in enumerate(sequence):
            op_name = _operator_name(op)
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
                op_name = _operator_name(sequence[j])
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
        """Evaluate the optional U6-EXP operator-spacing heuristic.

        For consecutive Dissonance, Mutation or Expansion positions, this
        supplied policy flags gaps at or below ``max(2, int((k_top/vf)*3))``.
        The message uses a factor of 1.5 after Mutation; that factor does not
        change the admission threshold. Operator positions supply the spacing
        coordinate: this method consumes no elapsed times, pressure evolution,
        graph spectrum or measured recovery.

        U6-EXP is distinct from canonical U6, whose structural-potential
        observations belong to ``grammar_u6.py``. A spacing result proves no
        relaxation time, bifurcation, boundedness or fragmentation.

        Parameters
        ----------
        sequence : list[Operator]
            Ordered operators to inspect.
        vf : float, optional
            Finite positive represented capacity scale (default: 1.0).
        k_top : float, optional
            Finite nonnegative represented multiplier (default: 1.0);
            no topology is read. Zero retains the minimum two-position scale.

        Returns
        -------
        tuple[bool, str]
            Whether the spacing policy passed, and its diagnostic message.
            ``GrammarValidator.validate`` appends the message only when
            ``experimental_u6=True``; it does not change the word verdict.

        Notes
        -----
        The numerical constants are policy choices. See
        ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md`` for the separate assumptions
        needed to establish trajectory decay under a declared dynamics law.
        """
        vf, _ = finite_represented_real(vf, "vf")
        k_top, _ = finite_represented_real(k_top, "k_top")
        if vf <= 0.0:
            raise ValueError("vf must be strictly positive for temporal spacing")
        if k_top < 0.0:
            raise ValueError("k_top must be nonnegative for temporal spacing")

        def spacing_scale(factor):
            value, _ = finite_represented_real(
                (k_top / vf) * factor * 3.0, "configured temporal spacing"
            )
            if k_top != 0.0 and value == 0.0:
                raise ValueError("configured temporal spacing underflows to zero")
            return value

        # Check for destabilizers that trigger relaxation requirement
        destabilizer_positions = []
        for i, op in enumerate(sequence):
            op_name = _operator_name(op)
            if op_name in DESTABILIZERS:
                destabilizer_positions.append((i, op_name))

        if len(destabilizer_positions) < 2:
            return True, "U6-EXP: not applicable (fewer than 2 destabilizers)"

        # Supplied spacing scale in operator positions; no clock is measured.
        tau_relax = spacing_scale(1.0)  # Retained multiplier 3.0 ≈ ln(20).

        # Retained integer spacing policy.
        min_spacing = max(2, int(tau_relax))  # At least 2 operators

        # Check spacing between consecutive destabilizers
        violations = []
        for j in range(1, len(destabilizer_positions)):
            prev_idx, prev_op = destabilizer_positions[j - 1]
            curr_idx, curr_op = destabilizer_positions[j]
            spacing = curr_idx - prev_idx

            if spacing <= min_spacing:
                # The pair-specific scale is diagnostic, not the threshold.
                k_op_prev = 1.5 if prev_op == "mutation" else 1.0
                tau_est = spacing_scale(k_op_prev)

                violations.append(
                    f"{curr_op} at position {curr_idx} follows {prev_op} "
                    f"at position {prev_idx} (spacing={spacing} operators). "
                    f"Configured pair scale={tau_est:.2f} operator positions "
                    f"(integer estimate={int(tau_est)}); "
                    f"required spacing is greater than {min_spacing}. "
                    "No trajectory recovery or instability was evaluated"
                )

        if violations:
            return (
                False,
                f"U6-EXP WARNING (experimental): {'; '.join(violations)}. "
                "See theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md",
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
            Capacity scale for the optional operator-spacing policy (default: 1.0).
        k_top : float, optional
            Multiplier for the optional operator-spacing policy (default: 1.0).
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
        checks = self.validate_checks(
            sequence,
            epi_initial,
            vf=vf,
            k_top=k_top,
            stop_on_first_error=stop_on_first_error,
        )
        return (
            all(check.passed or not check.blocking for check in checks),
            [check.legacy_message() for check in checks],
        )

    def validate_checks(
        self,
        sequence: list[Operator],
        epi_initial: float = 0.0,
        vf: float = 1.0,
        k_top: float = 1.0,
        stop_on_first_error: bool = False,
    ) -> tuple[GrammarCheckResult, ...]:
        """Return immutable outcomes without inferring validity from prose.

        This is the shared execution behind ``validate`` and structured error
        readers. Original operator instances retain their declared metadata.
        Syntax rejection precedes all word checks; optional U6-EXP remains
        advisory and has no live-state or trajectory evidence.
        """
        for index, operator in enumerate(sequence):
            name = _operator_name(operator)
            if not isinstance(name, str) or name not in CANONICAL_OPERATOR_NAMES:
                return (
                    GrammarCheckResult(
                        "SYNTAX",
                        False,
                        f"Unknown operator at position {index}: {name!r}",
                    ),
                )

        checks = []
        # One order and one verdict source for the legacy and structured APIs.
        validators = (
            ("U1a", self.validate_initiation, (sequence, epi_initial)),
            ("U1b", self.validate_closure, (sequence,)),
            ("U2", self.validate_convergence, (sequence,)),
            ("U3", self.validate_resonant_coupling, (sequence,)),
            ("U4a", self.validate_bifurcation_triggers, (sequence,)),
            ("U4b", self.validate_transformer_context, (sequence,)),
            ("U2-REMESH", self.validate_remesh_amplification, (sequence,)),
            ("U5", self.validate_multiscale_coherence, (sequence,)),
        )
        for rule, validate, args in validators:
            passed, message = validate(*args)
            checks.append(GrammarCheckResult(rule, passed, message))
            if stop_on_first_error and not passed:
                return tuple(checks)

        if self.experimental_u6:
            passed, message = self.validate_temporal_ordering(
                sequence, vf=vf, k_top=k_top
            )
            checks.append(GrammarCheckResult("U6-EXP", passed, message, blocking=False))
        return tuple(checks)

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
