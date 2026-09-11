"""Fixed-branch coarse-graining of the circular phase contribution.

The canonical phase-only pressure used by :mod:`tnfr.dynamics.dnfr` is

``g_i(theta) = -wrap(theta_i - Arg sum_{j in N(i)} exp(i theta_j)) / pi``.

Its contribution to the nodal EPI rate is ``nu_f_i * g_i(theta)``.  This map
is circular and nonlinear: selecting one wrap branch removes the discontinuity
of pairwise differences, but it does not turn a mean of phasors into an
arithmetic mean.

This module therefore keeps two statements separate.  In a selected open
semicircle chart, the pairwise wrapped-pressure realization is exactly

``r_pair(q) = -diag(nu_f) L_rw q / pi``.

It inherits the reversible EPI quotient identities and gives a global matrix
certificate for every state that remains in that fixed-wrap chamber.  For the
actual mean-of-phasors channel, an exact lifted-subspace result needs stronger
support hypotheses: equitable neighbor profiles, no edges inside a fiber,
equal active macro-neighbor multiplicities, and block-constant capacity.
Projected autonomy for arbitrary micro phases is not implied.  Comparing one
state with its lifted representative gives an executable counterexample when
they have the same macro chart coordinate but different projected nodal rates.

The certificate is read-only.  It holds support and capacities fixed, advances
neither EPI nor phase, and does not promote phase coarse-graining to a canonical
operator, a finite-time theorem, or a changing-topology U5 law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA, ALIAS_VF
from ..utils import angle_diff
from .operator_quotient import (
    OperatorQuotientCertificate,
    certify_operator_quotient,
)
from .structural_morphism import _build_reversible_partition_geometry

__all__ = [
    "PhaseNodalCoarseGrainingCertificate",
    "certify_phase_nodal_coarse_graining",
]


@dataclass(frozen=True, slots=True)
class PhaseNodalCoarseGrainingCertificate:
    """Closure evidence for one fixed-support circular phase quotient.

    ``pairwise_*`` fields concern the exact edge-difference realization in the
    selected wrap chamber.  ``canonical_*`` fields concern the engine's
    mean-of-phasors phase channel.  A ``None`` global decision means that the
    supplied state did not prove or refute autonomy for every micro state.
    ``macro_conductance`` belongs to the weighted pairwise quotient;
    ``macro_neighbor_support`` lists the block indices used by the canonical
    unweighted circular quotient.
    """

    nodes: tuple[Any, ...]
    blocks: tuple[tuple[Any, ...], ...]
    projection: np.ndarray
    lift: np.ndarray
    micro_pairwise_generator: np.ndarray
    macro_pairwise_generator: np.ndarray
    macro_conductance: np.ndarray
    macro_frequency: np.ndarray
    reversible_partition_closure_within_tolerance: bool
    pairwise_operator_quotient: OperatorQuotientCertificate
    canonical_operator_quotient: OperatorQuotientCertificate
    phase_chart: np.ndarray
    macro_phase_chart: np.ndarray
    neighbor_multiplicity: np.ndarray
    macro_neighbor_support: tuple[tuple[int, ...], ...]
    canonical_neighbor_averaging: str
    projection_metric: str
    micro_canonical_nodal_rate: np.ndarray
    macro_canonical_nodal_rate: np.ndarray
    lifted_micro_canonical_nodal_rate: np.ndarray
    projected_micro_canonical_nodal_rate: np.ndarray
    chart_reference: float
    chart_span: float
    wrap_branch_margin: float
    fixed_wrap_branch: bool
    minimum_circular_resultant: float
    circular_means_defined: bool
    pairwise_projection_residual: float
    pairwise_lift_residual: float
    relative_pairwise_projection_residual: float
    relative_pairwise_lift_residual: float
    pairwise_strong_closure_certified: bool
    equitable_neighbor_profiles: bool
    no_internal_fiber_edges: bool
    uniform_active_macro_multiplicity: bool
    block_constant_capacity: bool
    canonical_lift_hypotheses_satisfied: bool
    canonical_lift_residual: float
    relative_canonical_lift_residual: float
    canonical_lift_closure_certified: bool
    sampled_projection_residual: float
    relative_sampled_projection_residual: float
    sampled_projected_closure_within_tolerance: bool | None
    sampled_fiber_dependence_residual: float
    relative_sampled_fiber_dependence_residual: float
    same_macro_state_residual: float
    current_state_is_counterexample: bool
    global_canonical_projected_closure: bool | None
    tolerance: float
    support_status: str
    claim_status: str
    scope: str


def _finite_scalar(value: Any, name: str) -> float:
    """Read one real binary64 scalar without accepting logical payloads."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _readonly(value: Any) -> np.ndarray:
    """Return an isolated read-only binary64 array."""
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _finite_product(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Multiply finite arrays or reject an unrepresentable result."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = left @ right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_difference(
    left: np.ndarray, right: np.ndarray, name: str
) -> np.ndarray:
    """Subtract finite arrays or reject an unrepresentable result."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = left - right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_norm(value: np.ndarray, *, matrix: bool, name: str) -> float:
    """Return a scale-safe Euclidean or spectral norm."""
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
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _relative_diagnostic(
    left: np.ndarray,
    right: np.ndarray,
    *,
    matrix: bool,
    name: str,
) -> tuple[float, float]:
    """Return an absolute defect and its dimensionless relative value."""
    defect = _finite_difference(left, right, name)
    residual = _finite_norm(defect, matrix=matrix, name=name)
    scale = max(
        1.0,
        _finite_norm(left, matrix=matrix, name=f"{name} scale"),
        _finite_norm(right, matrix=matrix, name=f"{name} scale"),
    )
    return residual, residual / scale


def _canonical_phase_rate(
    phases: np.ndarray,
    frequencies: np.ndarray,
    neighbors: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, float]:
    """Evaluate the canonical unweighted-support phasor contribution."""
    rates = np.zeros(len(phases), dtype=float)
    minimum_resultant = 1.0
    for index, adjacent in enumerate(neighbors):
        if not adjacent:
            mean_phase = float(phases[index])
            resultant = 1.0
        else:
            cosine = math.fsum(math.cos(float(phases[item])) for item in adjacent)
            sine = math.fsum(math.sin(float(phases[item])) for item in adjacent)
            resultant = math.hypot(cosine, sine) / len(adjacent)
            mean_phase = math.atan2(sine, cosine)
        minimum_resultant = min(minimum_resultant, resultant)
        pressure = -angle_diff(float(phases[index]), mean_phase) / math.pi
        rate = float(frequencies[index]) * pressure
        if not math.isfinite(rate):
            raise ValueError("phase-driven nodal rate exceeds finite range")
        rates[index] = rate
    return rates, minimum_resultant


def _support_diagnostics(
    neighbors: tuple[tuple[int, ...], ...],
    block_indices: tuple[tuple[int, ...], ...],
    block_of: tuple[int, ...],
) -> tuple[np.ndarray, bool, bool, bool]:
    """Describe the exact neighbor-count profiles of the fixed support."""
    block_count = len(block_indices)
    profiles = np.zeros((len(block_of), block_count), dtype=float)
    for node, adjacent in enumerate(neighbors):
        for target in adjacent:
            profiles[node, block_of[target]] += 1.0

    multiplicity = np.zeros((block_count, block_count), dtype=float)
    equitable = True
    no_internal = True
    for source, indices in enumerate(block_indices):
        rows = profiles[np.asarray(indices, dtype=int)]
        multiplicity[source] = np.mean(rows, axis=0)
        equitable = equitable and bool(np.all(rows == rows[0]))
        no_internal = no_internal and bool(np.all(rows[:, source] == 0.0))

    uniform_active = True
    for row in multiplicity:
        positive = row[row > 0.0]
        if positive.size > 1 and not np.all(positive == positive[0]):
            uniform_active = False
            break
    return multiplicity, equitable, no_internal, uniform_active


def certify_phase_nodal_coarse_graining(
    graph: Any,
    partition: Any,
    *,
    chart_reference: float | None = None,
    tolerance: float = 1e-10,
) -> PhaseNodalCoarseGrainingCertificate:
    r"""Certify the fixed-branch phase contribution under one partition.

    The selected chart lifts every phase relative to ``chart_reference``.  Its
    diameter must be strictly below ``pi`` (up to ``tolerance``) to certify one
    open chamber in which every pairwise wrapped edge difference is affine.
    The default reference is the first node's normalized phase.

    The pairwise result is a global matrix identity on that chamber.  The
    canonical circular result is weaker: structural hypotheses can certify the
    lifted block-constant subspace, while the supplied nonconstant state can
    refute global projected autonomy.  A zero sampled residual never promotes
    the nonlinear canonical map to a global theorem.

    The graph must have fixed symmetric nonnegative conductance, positive
    capacity, and a strict reducing partition.  The reversible projection uses
    the conductance metric ``d_i / nu_f_i``.  The canonical phasor mean instead
    uses the unique unweighted neighbors returned by ``graph.neighbors``,
    exactly as ``dnfr_phase_only`` does: zero-weight edges remain in this
    support, parallel edges count once, and directed graphs use successor
    support.  Its macro support therefore comes from neighbor multiplicities,
    independently of the pairwise macro conductance.  EPI is neither read nor
    constrained.
    """
    tolerance_value = _finite_scalar(tolerance, "tolerance")
    if tolerance_value <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    quotient = _build_reversible_partition_geometry(
        graph,
        partition,
        tolerance=tolerance_value,
        label="phase coarse-graining",
    )
    nodes = quotient.nodes
    node_index = {node: index for index, node in enumerate(nodes)}

    def read_node_scalar(node: Any, aliases: tuple[str, ...], name: str) -> float:
        label = f"{name} at node {node!r}"
        try:
            raw = get_attr(
                graph.nodes[node],
                aliases,
                None,
                strict=True,
                conv=lambda value: value,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a finite real scalar") from exc
        return _finite_scalar(raw, label)

    raw_phases = np.array(
        [read_node_scalar(node, ALIAS_THETA, "phase") for node in nodes],
        dtype=float,
    )
    frequencies = np.array(
        [read_node_scalar(node, ALIAS_VF, "capacity") for node in nodes],
        dtype=float,
    )
    if np.any(frequencies <= 0.0):
        raise ValueError("phase coarse-graining requires positive capacity")

    if chart_reference is None:
        reference = math.remainder(float(raw_phases[0]), math.tau)
    else:
        reference = math.remainder(
            _finite_scalar(chart_reference, "chart_reference"),
            math.tau,
        )
    normalized = np.remainder(raw_phases, math.tau)
    phase_chart = np.array(
        [reference + angle_diff(float(value), reference) for value in normalized],
        dtype=float,
    )
    chart_span = float(np.ptp(phase_chart))
    branch_margin = math.pi - chart_span
    fixed_branch = branch_margin > tolerance_value

    projection = np.asarray(quotient.projection, dtype=float)
    lift = np.asarray(quotient.lift, dtype=float)
    macro_phase = _finite_product(
        projection,
        phase_chart,
        "phase chart projection",
    )
    lifted_phase = _finite_product(lift, macro_phase, "phase chart lift")

    neighbors = tuple(
        tuple(node_index[neighbor] for neighbor in graph.neighbors(node))
        for node in nodes
    )
    blocks = quotient.blocks
    block_indices = tuple(
        tuple(node_index[node] for node in block) for block in blocks
    )
    block_of_list = [0] * len(nodes)
    for block_index, indices in enumerate(block_indices):
        for index in indices:
            block_of_list[index] = block_index
    block_of = tuple(block_of_list)
    (
        multiplicity,
        equitable,
        no_internal,
        uniform_active,
    ) = _support_diagnostics(neighbors, block_indices, block_of)

    # The circular channel ignores edge weights and reads NetworkX neighbor
    # support directly.  In particular, a zero-weight edge is absent from the
    # pairwise conductance but remains a canonical phase neighbor.  Parallel
    # edges contribute one neighbor, matching ``graph.neighbors``.
    macro_support = tuple(
        tuple(
            int(target)
            for target in np.flatnonzero(multiplicity[source] > 0.0)
        )
        for source in range(len(blocks))
    )

    micro_rate, current_resultant = _canonical_phase_rate(
        phase_chart,
        frequencies,
        neighbors,
    )
    lifted_micro_rate, lifted_resultant = _canonical_phase_rate(
        lifted_phase,
        frequencies,
        neighbors,
    )
    macro_rate, macro_resultant = _canonical_phase_rate(
        macro_phase,
        np.asarray(quotient.macro_frequency, dtype=float),
        macro_support,
    )
    minimum_resultant = min(
        current_resultant,
        lifted_resultant,
        macro_resultant,
    )
    circular_means_defined = minimum_resultant > tolerance_value

    micro_pairwise = -np.asarray(quotient.micro_generator) / math.pi
    macro_pairwise = -np.asarray(quotient.macro_generator) / math.pi
    projected_pairwise = _finite_product(
        projection,
        micro_pairwise,
        "pairwise phase projected generator",
    )
    macro_pairwise_projection = _finite_product(
        macro_pairwise,
        projection,
        "pairwise phase macro generator",
    )
    pair_projection_residual, relative_pair_projection = _relative_diagnostic(
        projected_pairwise,
        macro_pairwise_projection,
        matrix=True,
        name="pairwise phase projection residual",
    )
    micro_pairwise_lift = _finite_product(
        micro_pairwise,
        lift,
        "pairwise phase lifted generator",
    )
    lifted_macro_pairwise = _finite_product(
        lift,
        macro_pairwise,
        "pairwise phase macro lift",
    )
    pair_lift_residual, relative_pair_lift = _relative_diagnostic(
        micro_pairwise_lift,
        lifted_macro_pairwise,
        matrix=True,
        name="pairwise phase lift residual",
    )
    pairwise_operator_quotient = certify_operator_quotient(
        projection,
        lift,
        micro_pairwise,
        macro_pairwise,
        operation_kind="vector_field",
        operator_name="fixed-branch pairwise phase contribution",
        tolerance=tolerance_value,
    )
    pairwise_certified = bool(
        fixed_branch
        and pairwise_operator_quotient.global_strong_closure_within_tolerance
    )

    canonical_operator_quotient = certify_operator_quotient(
        projection,
        lift,
        lambda state: _canonical_phase_rate(
            np.asarray(state, dtype=float), frequencies, neighbors
        )[0],
        lambda state: _canonical_phase_rate(
            np.asarray(state, dtype=float),
            np.asarray(quotient.macro_frequency, dtype=float),
            macro_support,
        )[0],
        macro_probes=macro_phase,
        micro_probes=phase_chart,
        operation_kind="vector_field",
        operator_name="canonical circular phase contribution",
        tolerance=tolerance_value,
    )

    projected_micro_rate = _finite_product(
        projection,
        micro_rate,
        "canonical phase rate projection",
    )
    sampled_projection_residual, relative_sampled_projection = (
        _relative_diagnostic(
            projected_micro_rate,
            macro_rate,
            matrix=False,
            name="canonical sampled projection residual",
        )
    )
    lifted_macro_rate = _finite_product(
        lift,
        macro_rate,
        "canonical phase lifted macro rate",
    )
    lift_residual, relative_lift_residual = _relative_diagnostic(
        lifted_micro_rate,
        lifted_macro_rate,
        matrix=False,
        name="canonical phase lift residual",
    )
    projected_lifted_rate = _finite_product(
        projection,
        lifted_micro_rate,
        "canonical lifted phase rate projection",
    )
    fiber_residual, relative_fiber_residual = _relative_diagnostic(
        projected_micro_rate,
        projected_lifted_rate,
        matrix=False,
        name="canonical phase fiber-dependence residual",
    )
    lifted_macro_state = _finite_product(
        projection,
        lifted_phase,
        "lifted macro phase state",
    )
    same_macro_residual, relative_same_macro = _relative_diagnostic(
        macro_phase,
        lifted_macro_state,
        matrix=False,
        name="same macro phase state residual",
    )

    block_constant_capacity = True
    macro_frequency = np.asarray(quotient.macro_frequency, dtype=float)
    for block_index, indices in enumerate(block_indices):
        block_frequency = frequencies[np.asarray(indices, dtype=int)]
        scale = max(
            1.0,
            abs(float(macro_frequency[block_index])),
            float(np.max(np.abs(block_frequency))),
        )
        if (
            float(
                np.max(
                    np.abs(block_frequency - macro_frequency[block_index])
                )
            )
            / scale
            > tolerance_value
        ):
            block_constant_capacity = False
            break

    lift_hypotheses = bool(
        fixed_branch
        and circular_means_defined
        and quotient.nodal_closure_within_tolerance
        and equitable
        and no_internal
        and uniform_active
        and block_constant_capacity
    )
    lift_certified = bool(
        lift_hypotheses and relative_lift_residual <= tolerance_value
    )
    sampled_projected = (
        relative_sampled_projection <= tolerance_value
        if circular_means_defined
        else None
    )
    counterexample = bool(
        fixed_branch
        and circular_means_defined
        and relative_same_macro <= tolerance_value
        and relative_fiber_residual > tolerance_value
    )
    global_canonical: bool | None = False if counterexample else None

    if not fixed_branch:
        support_status = "abstained_wrap_branch"
        claim_status = (
            "ABSTAINED: the supplied phases do not occupy one declared open "
            "semicircle, so a fixed-wrap quotient is not certified"
        )
    elif not circular_means_defined:
        support_status = "abstained_circular_singularity"
        claim_status = (
            "ABSTAINED: at least one neighbor phasor resultant is numerically "
            "zero, where the circular mean has no differentiable direction"
        )
    elif lift_certified and counterexample:
        support_status = "lift_closed_with_global_counterexample"
        claim_status = (
            "CERTIFIED on the lifted block-constant subspace; COUNTEREXAMPLE "
            "to global canonical projected autonomy from unresolved phase fibers"
        )
    elif counterexample:
        support_status = "global_counterexample"
        claim_status = (
            "COUNTEREXAMPLE to global canonical projected autonomy; the "
            "lifted-subspace sufficient hypotheses are not all satisfied"
        )
    elif lift_certified:
        support_status = "lift_subspace_closed"
        claim_status = (
            "CERTIFIED on the lifted block-constant subspace; global canonical "
            "projected autonomy remains unproved"
        )
    elif lift_hypotheses:
        support_status = "failed_lift_residual"
        claim_status = (
            "FAILED: the measured lift residual contradicts the declared "
            "sufficient fixed-support hypotheses"
        )
    else:
        support_status = "sampled_only"
        claim_status = (
            "ABSTAINED from canonical lift promotion: support or capacity "
            "hypotheses fail; current-state residuals are sampled evidence only"
        )

    return PhaseNodalCoarseGrainingCertificate(
        nodes=nodes,
        blocks=blocks,
        projection=_readonly(projection),
        lift=_readonly(lift),
        micro_pairwise_generator=_readonly(micro_pairwise),
        macro_pairwise_generator=_readonly(macro_pairwise),
        macro_conductance=_readonly(quotient.macro_conductance),
        macro_frequency=_readonly(quotient.macro_frequency),
        reversible_partition_closure_within_tolerance=(
            quotient.nodal_closure_within_tolerance
        ),
        pairwise_operator_quotient=pairwise_operator_quotient,
        canonical_operator_quotient=canonical_operator_quotient,
        phase_chart=_readonly(phase_chart),
        macro_phase_chart=_readonly(macro_phase),
        neighbor_multiplicity=_readonly(multiplicity),
        macro_neighbor_support=macro_support,
        canonical_neighbor_averaging="unweighted_support",
        projection_metric="conductance_degree_over_capacity",
        micro_canonical_nodal_rate=_readonly(micro_rate),
        macro_canonical_nodal_rate=_readonly(macro_rate),
        lifted_micro_canonical_nodal_rate=_readonly(lifted_micro_rate),
        projected_micro_canonical_nodal_rate=_readonly(projected_micro_rate),
        chart_reference=reference,
        chart_span=chart_span,
        wrap_branch_margin=branch_margin,
        fixed_wrap_branch=fixed_branch,
        minimum_circular_resultant=minimum_resultant,
        circular_means_defined=circular_means_defined,
        pairwise_projection_residual=pair_projection_residual,
        pairwise_lift_residual=pair_lift_residual,
        relative_pairwise_projection_residual=relative_pair_projection,
        relative_pairwise_lift_residual=relative_pair_lift,
        pairwise_strong_closure_certified=pairwise_certified,
        equitable_neighbor_profiles=equitable,
        no_internal_fiber_edges=no_internal,
        uniform_active_macro_multiplicity=uniform_active,
        block_constant_capacity=block_constant_capacity,
        canonical_lift_hypotheses_satisfied=lift_hypotheses,
        canonical_lift_residual=lift_residual,
        relative_canonical_lift_residual=relative_lift_residual,
        canonical_lift_closure_certified=lift_certified,
        sampled_projection_residual=sampled_projection_residual,
        relative_sampled_projection_residual=relative_sampled_projection,
        sampled_projected_closure_within_tolerance=sampled_projected,
        sampled_fiber_dependence_residual=fiber_residual,
        relative_sampled_fiber_dependence_residual=relative_fiber_residual,
        same_macro_state_residual=same_macro_residual,
        current_state_is_counterexample=counterexample,
        global_canonical_projected_closure=global_canonical,
        tolerance=tolerance_value,
        support_status=support_status,
        claim_status=claim_status,
        scope=(
            "fixed symmetric support, positive capacity, selected open-"
            "semicircle chart, phase-only contribution to the nodal EPI rate"
        ),
    )