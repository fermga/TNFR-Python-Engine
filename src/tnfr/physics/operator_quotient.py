r"""Read-only closure certificates for operators under coarse-graining.

The pure-EPI diffusion quotient in :mod:`tnfr.physics.structural_morphism`
certifies one linear channel of the nodal equation.  This module supplies the
next, deliberately more limited, question: if a micro-scale operator is
represented by a fixed-dimensional map ``F`` and a macro-scale map ``F_bar``
is declared, do they commute with a projection ``R`` and lift ``P``?

For linear maps the two global identities are

``R F = F_bar R`` (projected closure for every micro state), and
``F P = P F_bar`` (invariance of the lifted macro subspace).

They are independent.  The first says unresolved micro coordinates cannot
    change the macro update.  The second says a block-constant state does not leak
    out of the lifted subspace.  Both identities define global strong closure.
    Because this implementation evaluates floating-point matrices, its Boolean
    fields report agreement within the declared tolerance; the residuals remain
    available for an exact symbolic or analytic assessment.

For callable maps (including nonlinear ones) no finite probe set proves a
global identity.  The
certificate therefore labels every result as *sampled*, keeps lifted-state and
arbitrary-state defects separate, and also compares ``R F(x)`` with
``R F(P R x)`` to expose sampled dependence on unresolved fiber coordinates.

Canonical engine operators can mutate graph attributes, topology, histories
and nested EPIs.  Such actions are not fixed-dimensional state maps.  Passing
``operation_kind="graph_mutation"`` returns an explicit unsupported boundary
without invoking either callable.  The certificate is a structural morphism
diagnostic, never a fourteenth TNFR operator and never a proof that all 13
canonical operators close under U5 coarse-graining.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

__all__ = [
    "OperatorQuotientCertificate",
    "certify_operator_quotient",
]


_SUPPORTED_KINDS = frozenset({"vector_field", "state_map"})
_ALL_KINDS = _SUPPORTED_KINDS | {"graph_mutation"}


@dataclass(frozen=True, slots=True)
class OperatorQuotientCertificate:
    """Closure diagnostics for one declared micro/macro operator pair.

    ``global_*`` fields are populated only when both dynamics are matrices.
    Matrix residuals cover all states of the declared finite vector spaces;
    their Boolean values are explicitly conditioned on ``tolerance``.
    ``sampled_*`` fields are populated
    only for callable dynamics and describe exactly the supplied probes.
    Sampled strong closure remains ``None`` until arbitrary micro probes also
    test the unresolved fibers.
    ``None`` means that a property was outside the selected protocol, not that
    it failed.
    """

    operator_name: str | None
    operation_kind: str
    support_status: str
    supported: bool
    unsupported_reason: str | None
    micro_dimension: int
    macro_dimension: int
    projection_rank: int
    lift_rank: int
    right_inverse_residual: float
    information_loss_dimension: int
    has_information_loss: bool
    tolerance: float
    residual_scale: float | None
    macro_probe_count: int
    micro_probe_count: int
    sampled_repeatability_residual: float | None
    sampled_callables_repeatable: bool | None
    global_projection_residual: float | None
    global_lift_residual: float | None
    global_projected_closure_within_tolerance: bool | None
    global_lift_invariance_within_tolerance: bool | None
    global_strong_closure_within_tolerance: bool | None
    sampled_lifted_projection_residual: float | None
    sampled_lift_invariance_residual: float | None
    sampled_arbitrary_projection_residual: float | None
    sampled_fiber_dependence_residual: float | None
    sampled_lifted_projected_closure: bool | None
    sampled_lift_invariance: bool | None
    sampled_arbitrary_projected_closure: bool | None
    sampled_fiber_independence: bool | None
    sampled_strong_closure: bool | None
    claim_status: str


def _contains_boolean(value: Any) -> bool:
    """Detect logical payloads before NumPy coerces them to zero or one."""
    try:
        values = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in values.flat)


def _contains_nonreal(value: Any) -> bool:
    """Detect text, complex values, and other coercible non-real payloads."""
    try:
        values = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError):
        return True
    return any(not isinstance(item, Real) for item in values.flat)


def _real_matrix(value: Any, name: str) -> np.ndarray:
    """Return a finite real rank-two array without changing caller storage."""
    if _contains_boolean(value):
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    if _contains_nonreal(value):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, dtype=float, copy=True)


def _finite_product(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Multiply finite arrays or reject an unrepresentable result explicitly."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = left @ right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_difference(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Subtract finite arrays without allowing an infinite residual."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = left - right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_norm(value: np.ndarray, *, matrix: bool, name: str) -> float:
    """Return a scale-safe 2-norm, rejecting an unrepresentable norm."""
    array = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(array), initial=0.0))
    if scale == 0.0:
        return 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = array / scale
            norm = float(np.linalg.norm(normalized, 2 if matrix else None))
            result = scale * norm
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _relative_rank(matrix: np.ndarray, name: str) -> int:
    """Numerical rank under a scale-invariant precision-derived cutoff."""
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        return 0
    try:
        singular_values = np.linalg.svd(matrix / scale, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} rank computation did not converge") from exc
    if not np.all(np.isfinite(singular_values)):
        raise ValueError(f"{name} rank computation is non-finite")
    relative_cutoff = max(matrix.shape) * np.finfo(float).eps
    return int(np.sum(singular_values > relative_cutoff * singular_values[0]))


def _within_relative_tolerance(residual: float, scale: float, tolerance: float) -> bool:
    """Compare a finite residual and scale without overflowing ``tol*scale``."""
    return bool(residual / scale <= tolerance)


def _probe_matrix(value: Any, dimension: int, name: str) -> np.ndarray:
    """Normalize one vector or a row-wise collection of finite probe vectors."""
    if _contains_boolean(value):
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    if _contains_nonreal(value):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        probes = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if probes.ndim == 1:
        probes = probes.reshape(1, -1)
    if probes.ndim != 2 or probes.shape[1] != dimension:
        raise ValueError(
            f"{name} must have shape (probe_count, {dimension})"
        )
    if probes.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one probe")
    if not np.all(np.isfinite(probes)):
        raise ValueError(f"{name} must be finite")
    return np.array(probes, dtype=float, copy=True)


def _evaluate_map(
    function: Callable[[np.ndarray], Any],
    state: np.ndarray,
    dimension: int,
    name: str,
) -> np.ndarray:
    """Evaluate a declared fixed-dimensional map on an isolated input copy."""
    isolated_state = np.array(state, dtype=float, copy=True)
    if isolated_state.shape != (dimension,) or not np.all(np.isfinite(isolated_state)):
        raise ValueError(f"{name} received a non-finite or incompatible state")
    output = function(isolated_state)
    if _contains_boolean(output):
        raise ValueError(f"{name} returned booleans instead of a real state")
    if _contains_nonreal(output):
        raise ValueError(f"{name} returned a non-real state")
    try:
        result = np.asarray(output, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} returned a non-real state") from exc
    if result.shape != (dimension,):
        raise ValueError(
            f"{name} must return shape ({dimension},); "
            "dimension-changing or graph-mutating actions are unsupported"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} returned a non-finite state")
    # A callable may return a view of a mutable internal work buffer.  Retain
    # the value observed on this invocation before evaluating it again.
    return np.array(result, dtype=float, copy=True)


def _max_norm(vectors: list[np.ndarray]) -> float:
    """Maximum Euclidean norm, with an exact zero for an empty collection."""
    return max(
        (
            _finite_norm(vector, matrix=False, name="sampled quotient residual")
            for vector in vectors
        ),
        default=0.0,
    )


def _empty_certificate(
    *,
    operator_name: str | None,
    operation_kind: str,
    micro_dimension: int,
    macro_dimension: int,
    projection_rank: int,
    lift_rank: int,
    right_inverse_residual: float,
    information_loss_dimension: int,
    tolerance: float,
    reason: str,
) -> OperatorQuotientCertificate:
    """Build the explicit boundary certificate for graph mutation."""
    return OperatorQuotientCertificate(
        operator_name=operator_name,
        operation_kind=operation_kind,
        support_status="unsupported_graph_mutation",
        supported=False,
        unsupported_reason=reason,
        micro_dimension=micro_dimension,
        macro_dimension=macro_dimension,
        projection_rank=projection_rank,
        lift_rank=lift_rank,
        right_inverse_residual=right_inverse_residual,
        information_loss_dimension=information_loss_dimension,
        has_information_loss=information_loss_dimension > 0,
        tolerance=tolerance,
        residual_scale=None,
        macro_probe_count=0,
        micro_probe_count=0,
        sampled_repeatability_residual=None,
        sampled_callables_repeatable=None,
        global_projection_residual=None,
        global_lift_residual=None,
        global_projected_closure_within_tolerance=None,
        global_lift_invariance_within_tolerance=None,
        global_strong_closure_within_tolerance=None,
        sampled_lifted_projection_residual=None,
        sampled_lift_invariance_residual=None,
        sampled_arbitrary_projection_residual=None,
        sampled_fiber_dependence_residual=None,
        sampled_lifted_projected_closure=None,
        sampled_lift_invariance=None,
        sampled_arbitrary_projected_closure=None,
        sampled_fiber_independence=None,
        sampled_strong_closure=None,
        claim_status=(
            "OUTSIDE SCOPE: topology/history/nesting changes are not maps on "
            "one fixed-dimensional quotient; no closure claim"
        ),
    )


def certify_operator_quotient(
    projection: Any,
    lift: Any,
    micro_dynamics: Any = None,
    macro_dynamics: Any = None,
    *,
    macro_probes: Any = None,
    micro_probes: Any = None,
    operation_kind: str = "vector_field",
    operator_name: str | None = None,
    tolerance: float = 1e-10,
) -> OperatorQuotientCertificate:
    r"""Certify a declared operator quotient without evolving or mutating a graph.

    Parameters
    ----------
    projection, lift:
        Matrices ``R`` (macro by micro) and ``P`` (micro by macro).  A valid
        coordinate quotient requires ``R P = I`` within ``tolerance``.  The
        tolerance is dimensionless: every identity residual is divided by the
        maximum of one and the norms of its two sides.  Projection/lift ranks
        use a separate precision-derived relative cutoff, so reciprocal
        coordinate rescaling cannot change the quotient dimension.
    micro_dynamics, macro_dynamics:
        Either two finite square matrices or two callables accepting and
        returning one-dimensional NumPy-compatible states.  Matrices obtain a
        global finite-dimensional certificate; callables, including nonlinear
        maps, obtain only a sampled certificate.
    macro_probes:
        Required for callables.  Rows are macro states ``z`` used to test
        ``R F(Pz) = F_bar(z)`` and ``F(Pz) = P F_bar(z)``.
    micro_probes:
        Optional rows of arbitrary micro states.  When supplied, they test
        ``R F(x) = F_bar(Rx)`` and fiber independence by comparing ``x`` with
        its lifted representative ``P R x``.
    operation_kind:
        ``"vector_field"`` for continuous generators, ``"state_map"`` for
        discrete updates, or ``"graph_mutation"`` for an explicit unsupported
        boundary.  Graph mutation returns a certificate without invoking the
        dynamics.

    Notes
    -----
    A zero residual from finitely many nonlinear probes is reproducible
    evidence on those probes, not a global identity.  Exact nonlinear closure
    requires an analytic proof external to this numerical certificate.
    """
    if not isinstance(operation_kind, str) or operation_kind not in _ALL_KINDS:
        choices = ", ".join(sorted(_ALL_KINDS))
        raise ValueError(f"operation_kind must be one of: {choices}")
    if operator_name is not None and not isinstance(operator_name, str):
        raise TypeError("operator_name must be a string or None")
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, Real)
    ):
        raise ValueError("tolerance must be finite and positive")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    r = _real_matrix(projection, "projection")
    p = _real_matrix(lift, "lift")
    macro_dimension, micro_dimension = r.shape
    if macro_dimension == 0 or micro_dimension == 0:
        raise ValueError("projection and lift dimensions must be positive")
    if p.shape != (micro_dimension, macro_dimension):
        raise ValueError(
            "lift shape must reverse the projection dimensions "
            f"({micro_dimension}, {macro_dimension})"
        )
    if macro_dimension > micro_dimension:
        raise ValueError("a coarse quotient cannot increase dimension")

    # Rank is structural metadata and therefore uses a precision-derived
    # relative cutoff, independently of the caller's closure tolerance.  In
    # particular, reciprocal rescaling R -> aR, P -> P/a leaves it unchanged.
    projection_rank = _relative_rank(r, "projection")
    lift_rank = _relative_rank(p, "lift")
    right_inverse = _finite_product(r, p, "projection @ lift")
    right_inverse_defect = _finite_difference(
        right_inverse,
        np.eye(macro_dimension),
        "projection @ lift residual",
    )
    right_inverse_residual = _finite_norm(
        right_inverse_defect,
        matrix=True,
        name="projection @ lift residual",
    )
    quotient_scale = max(
        1.0,
        _finite_norm(
            right_inverse,
            matrix=True,
            name="projection @ lift scale",
        ),
    )
    if not _within_relative_tolerance(
        right_inverse_residual, quotient_scale, tolerance_value
    ):
        raise ValueError(
            "projection and lift must satisfy projection @ lift = identity"
        )
    if projection_rank != macro_dimension or lift_rank != macro_dimension:
        raise ValueError("projection must be surjective and lift injective")
    information_loss_dimension = micro_dimension - projection_rank

    if operation_kind == "graph_mutation":
        return _empty_certificate(
            operator_name=operator_name,
            operation_kind=operation_kind,
            micro_dimension=micro_dimension,
            macro_dimension=macro_dimension,
            projection_rank=projection_rank,
            lift_rank=lift_rank,
            right_inverse_residual=right_inverse_residual,
            information_loss_dimension=information_loss_dimension,
            tolerance=tolerance_value,
            reason=(
                "graph mutation, operator history, or nested-EPI changes cannot "
                "be compared by a fixed projection/lift vector identity"
            ),
        )

    micro_callable = callable(micro_dynamics)
    macro_callable = callable(macro_dynamics)
    if micro_callable != macro_callable:
        raise ValueError(
            "micro_dynamics and macro_dynamics must use the same representation"
        )

    if not micro_callable:
        if micro_dynamics is None or macro_dynamics is None:
            raise ValueError("supported operation kinds require both dynamics")
        micro = _real_matrix(micro_dynamics, "micro_dynamics")
        macro = _real_matrix(macro_dynamics, "macro_dynamics")
        if micro.shape != (micro_dimension, micro_dimension):
            raise ValueError("micro_dynamics must be square on the micro state space")
        if macro.shape != (macro_dimension, macro_dimension):
            raise ValueError("macro_dynamics must be square on the macro state space")

        projected_micro = _finite_product(r, micro, "projection @ micro_dynamics")
        macro_projection = _finite_product(macro, r, "macro_dynamics @ projection")
        micro_lift = _finite_product(micro, p, "micro_dynamics @ lift")
        lifted_macro = _finite_product(p, macro, "lift @ macro_dynamics")
        projection_defect = _finite_difference(
            projected_micro,
            macro_projection,
            "global projected-closure residual",
        )
        lift_defect = _finite_difference(
            micro_lift,
            lifted_macro,
            "global lift-invariance residual",
        )
        projection_residual = _finite_norm(
            projection_defect,
            matrix=True,
            name="global projected-closure residual",
        )
        lift_residual = _finite_norm(
            lift_defect,
            matrix=True,
            name="global lift-invariance residual",
        )
        projection_scale = max(
            1.0,
            _finite_norm(
                projected_micro,
                matrix=True,
                name="global projected-closure scale",
            ),
            _finite_norm(
                macro_projection,
                matrix=True,
                name="global projected-closure scale",
            ),
        )
        lift_scale = max(
            1.0,
            _finite_norm(
                micro_lift,
                matrix=True,
                name="global lift-invariance scale",
            ),
            _finite_norm(
                lifted_macro,
                matrix=True,
                name="global lift-invariance scale",
            ),
        )
        residual_scale = max(projection_scale, lift_scale)
        projected = _within_relative_tolerance(
            projection_residual, projection_scale, tolerance_value
        )
        invariant = _within_relative_tolerance(
            lift_residual, lift_scale, tolerance_value
        )
        if projected and invariant:
            claim_status = (
                "PASSED within the declared tolerance: both global "
                "finite-dimensional linear identities hold numerically; "
                "residual zero is the exact algebraic condition; not "
                "catalog-wide closure"
            )
        elif projected:
            claim_status = (
                "COUNTEREXAMPLE to strong quotient closure: projected autonomy "
                "holds within the declared tolerance but lift invariance fails"
            )
        elif invariant:
            claim_status = (
                "FAILED strong quotient closure: lift invariance holds within "
                "the declared tolerance but projected autonomy fails"
            )
        else:
            claim_status = (
                "FAILED strong quotient closure: neither global linear identity "
                "holds within the declared tolerance"
            )
        return OperatorQuotientCertificate(
            operator_name=operator_name,
            operation_kind=operation_kind,
            support_status="global_linear",
            supported=True,
            unsupported_reason=None,
            micro_dimension=micro_dimension,
            macro_dimension=macro_dimension,
            projection_rank=projection_rank,
            lift_rank=lift_rank,
            right_inverse_residual=right_inverse_residual,
            information_loss_dimension=information_loss_dimension,
            has_information_loss=information_loss_dimension > 0,
            tolerance=tolerance_value,
            residual_scale=residual_scale,
            macro_probe_count=0,
            micro_probe_count=0,
            sampled_repeatability_residual=None,
            sampled_callables_repeatable=None,
            global_projection_residual=projection_residual,
            global_lift_residual=lift_residual,
            global_projected_closure_within_tolerance=projected,
            global_lift_invariance_within_tolerance=invariant,
            global_strong_closure_within_tolerance=projected and invariant,
            sampled_lifted_projection_residual=None,
            sampled_lift_invariance_residual=None,
            sampled_arbitrary_projection_residual=None,
            sampled_fiber_dependence_residual=None,
            sampled_lifted_projected_closure=None,
            sampled_lift_invariance=None,
            sampled_arbitrary_projected_closure=None,
            sampled_fiber_independence=None,
            sampled_strong_closure=None,
            claim_status=claim_status,
        )

    if macro_probes is None:
        raise ValueError("macro_probes are required for callable dynamics")
    macro_probe_array = _probe_matrix(
        macro_probes, macro_dimension, "macro_probes"
    )
    micro_probe_array = (
        None
        if micro_probes is None
        else _probe_matrix(micro_probes, micro_dimension, "micro_probes")
    )

    lifted_projection_defects: list[np.ndarray] = []
    lift_defects: list[np.ndarray] = []
    arbitrary_projection_defects: list[np.ndarray] = []
    fiber_defects: list[np.ndarray] = []
    observed_outputs: list[np.ndarray] = []
    lifted_projection_outputs: list[np.ndarray] = []
    lift_outputs: list[np.ndarray] = []
    arbitrary_projection_outputs: list[np.ndarray] = []
    fiber_outputs: list[np.ndarray] = []
    repeatability_defects: list[np.ndarray] = []
    repeatability_outputs: list[np.ndarray] = []

    def evaluate_repeatable(
        function: Callable[[np.ndarray], Any],
        state: np.ndarray,
        dimension: int,
        name: str,
    ) -> np.ndarray:
        first = _evaluate_map(function, state, dimension, name)
        second = _evaluate_map(function, state, dimension, name)
        repeatability_defects.append(
            _finite_difference(first, second, f"{name} repeatability residual")
        )
        repeatability_outputs.extend((first, second))
        return first

    for macro_state in macro_probe_array:
        lifted_state = _finite_product(p, macro_state, "lifted macro probe")
        micro_output = evaluate_repeatable(
            micro_dynamics, lifted_state, micro_dimension, "micro_dynamics"
        )
        macro_output = evaluate_repeatable(
            macro_dynamics, macro_state, macro_dimension, "macro_dynamics"
        )
        projected_micro_output = _finite_product(
            r, micro_output, "projected micro output"
        )
        lifted_macro_output = _finite_product(
            p, macro_output, "lifted macro output"
        )
        lifted_projection_defects.append(
            _finite_difference(
                projected_micro_output,
                macro_output,
                "sampled lifted projected-closure residual",
            )
        )
        lift_defects.append(
            _finite_difference(
                micro_output,
                lifted_macro_output,
                "sampled lift-invariance residual",
            )
        )
        lifted_projection_outputs.extend((projected_micro_output, macro_output))
        lift_outputs.extend((micro_output, lifted_macro_output))
        observed_outputs.extend(
            (
                projected_micro_output,
                macro_output,
                micro_output,
                lifted_macro_output,
            )
        )

    if micro_probe_array is not None:
        for micro_state in micro_probe_array:
            macro_state = _finite_product(r, micro_state, "projected micro probe")
            lifted_representative = _finite_product(
                p, macro_state, "lifted projected micro probe"
            )
            micro_output = evaluate_repeatable(
                micro_dynamics, micro_state, micro_dimension, "micro_dynamics"
            )
            representative_output = evaluate_repeatable(
                micro_dynamics,
                lifted_representative,
                micro_dimension,
                "micro_dynamics",
            )
            macro_output = evaluate_repeatable(
                macro_dynamics, macro_state, macro_dimension, "macro_dynamics"
            )
            projected_micro_output = _finite_product(
                r, micro_output, "projected arbitrary micro output"
            )
            projected_representative_output = _finite_product(
                r,
                representative_output,
                "projected representative output",
            )
            arbitrary_projection_defects.append(
                _finite_difference(
                    projected_micro_output,
                    macro_output,
                    "sampled arbitrary projected-closure residual",
                )
            )
            fiber_defects.append(
                _finite_difference(
                    projected_micro_output,
                    projected_representative_output,
                    "sampled fiber-dependence residual",
                )
            )
            arbitrary_projection_outputs.extend(
                (projected_micro_output, macro_output)
            )
            fiber_outputs.extend(
                (projected_micro_output, projected_representative_output)
            )
            observed_outputs.extend(
                (
                    projected_micro_output,
                    macro_output,
                    projected_representative_output,
                )
            )

    residual_scale = max(1.0, _max_norm(observed_outputs))
    lifted_projection_residual = _max_norm(lifted_projection_defects)
    lift_residual = _max_norm(lift_defects)
    arbitrary_residual = (
        None
        if micro_probe_array is None
        else _max_norm(arbitrary_projection_defects)
    )
    fiber_residual = (
        None if micro_probe_array is None else _max_norm(fiber_defects)
    )
    lifted_projection_scale = max(1.0, _max_norm(lifted_projection_outputs))
    lift_scale = max(1.0, _max_norm(lift_outputs))
    arbitrary_projection_scale = max(
        1.0, _max_norm(arbitrary_projection_outputs)
    )
    fiber_scale = max(1.0, _max_norm(fiber_outputs))
    repeatability_residual = _max_norm(repeatability_defects)
    repeatability_scale = max(1.0, _max_norm(repeatability_outputs))
    callables_repeatable = (
        _within_relative_tolerance(
            repeatability_residual, repeatability_scale, tolerance_value
        )
    )
    lifted_projected = (
        _within_relative_tolerance(
            lifted_projection_residual,
            lifted_projection_scale,
            tolerance_value,
        )
    )
    lift_invariant = _within_relative_tolerance(
        lift_residual, lift_scale, tolerance_value
    )
    arbitrary_projected = (
        None
        if arbitrary_residual is None
        else _within_relative_tolerance(
            arbitrary_residual,
            arbitrary_projection_scale,
            tolerance_value,
        )
    )
    fiber_independent = (
        None
        if fiber_residual is None
        else _within_relative_tolerance(
            fiber_residual, fiber_scale, tolerance_value
        )
    )
    sampled_strong = None
    if arbitrary_projected is not None and fiber_independent is not None:
        sampled_strong = (
            lifted_projected
            and lift_invariant
            and arbitrary_projected
            and fiber_independent
            and callables_repeatable
        )

    return OperatorQuotientCertificate(
        operator_name=operator_name,
        operation_kind=operation_kind,
        support_status="sampled_callable",
        supported=True,
        unsupported_reason=None,
        micro_dimension=micro_dimension,
        macro_dimension=macro_dimension,
        projection_rank=projection_rank,
        lift_rank=lift_rank,
        right_inverse_residual=right_inverse_residual,
        information_loss_dimension=information_loss_dimension,
        has_information_loss=information_loss_dimension > 0,
        tolerance=tolerance_value,
        residual_scale=residual_scale,
        macro_probe_count=len(macro_probe_array),
        micro_probe_count=0 if micro_probe_array is None else len(micro_probe_array),
        sampled_repeatability_residual=repeatability_residual,
        sampled_callables_repeatable=callables_repeatable,
        global_projection_residual=None,
        global_lift_residual=None,
        global_projected_closure_within_tolerance=None,
        global_lift_invariance_within_tolerance=None,
        global_strong_closure_within_tolerance=None,
        sampled_lifted_projection_residual=lifted_projection_residual,
        sampled_lift_invariance_residual=lift_residual,
        sampled_arbitrary_projection_residual=arbitrary_residual,
        sampled_fiber_dependence_residual=fiber_residual,
        sampled_lifted_projected_closure=lifted_projected,
        sampled_lift_invariance=lift_invariant,
        sampled_arbitrary_projected_closure=arbitrary_projected,
        sampled_fiber_independence=fiber_independent,
        sampled_strong_closure=sampled_strong,
        claim_status=(
            "MEASURED on declared probes, including repeated evaluations; zero "
            "sampled residual and repeatability are not a global callable-map "
            "purity or catalog-wide closure proof"
        ),
    )
