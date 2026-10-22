"""Prepared winding acquisition and its actual full-law receiver response.

The fixed positive-loss doubled-C5 model has two exact phase-flat source
families. Analytic semigroup bounds retain their complete nonlinear transit,
then assess the reached state and recorded form without a reset or solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._sine_preparation import (
    _prepared_duhamel_bounds,
    _sine_domain,
    _sine_preparation_from_rows,
)
from .phase_cycle_geometry import _derive
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

__all__ = ("SineFormationResponse", "assess_sine_formation_response")

_NODES = tuple(range(10))
_PAIRS = tuple((2 * a, 2 * a + 1) for a in range(5))
_EDGES = tuple(
    sorted(
        (min(2 * a + i, 2 * b + j), max(2 * a + i, 2 * b + j))
        for a, b in ((a, (a + 1) % 5) for a in range(5))
        for i in (0, 1)
        for j in (0, 1)
    )
)
_NEIGHBORS = tuple(
    tuple(j if i == a else i for i, j in _EDGES if a in (i, j)) for a in _NODES
)
_CAPACITY = (Q(1),) * 10
_PHASE = (Q(0),) * 10
_TARGET_TURNS = tuple(Q(a - 2, 5) for a in range(5) for _ in (0, 1))
_GAP = Q(2, 3)
_RATE = Q(2)


def _poisson_geometry():
    """Verify the fixed mean-free inverse and semigroup spectral premises."""
    size = len(_NODES)
    laplacian = tuple(
        tuple(Q(4 if i == j else -int(j in _NEIGHBORS[i])) for j in _NODES)
        for i in _NODES
    )
    projector = tuple(
        tuple(Q(int(i == j)) - Q(1, size) for j in _NODES) for i in _NODES
    )
    inverse = exact_matrix_inverse(
        tuple(tuple(value + Q(1, size) for value in row) for row in laplacian)
    )
    poisson = tuple(tuple(value - Q(1, size) for value in row) for row in inverse)
    if (
        exact_matrix_product(laplacian, poisson) != projector
        or exact_matrix_product(poisson, laplacian) != projector
        or any(sum(row, Q(0)) for row in poisson)
    ):
        raise ArithmeticError("the exact Poisson inverse lost its mean-free identities")
    for lower, upper in ((_GAP, None), (None, _RATE)):
        shifted = tuple(
            tuple(
                (
                    laplacian[i][j] / 4 - lower * projector[i][j]
                    if lower is not None
                    else upper * projector[i][j] - laplacian[i][j] / 4
                )
                for j in _NODES
            )
            for i in _NODES
        )
        if not exact_symmetric_semidefinite(shifted):
            raise ArithmeticError("the fixed mean-free semigroup bound failed")
    return poisson


def _proxy_fields(preparation, poisson):
    """Evaluate nominal sine currents and Poisson profiles from shared edges."""
    gamma = preparation.alpha
    gaps = tuple(gamma * (preparation.form[j] - preparation.form[i]) for i, j in _EDGES)
    edge_currents = tuple(sin(gap) for gap in gaps)
    currents = [I(0) for _ in _NODES]
    for (i, j), current in zip(_EDGES, edge_currents):
        currents[i] += current
        currents[j] -= current
    # Exact inverse-column differences retain each edge current's zero-sum
    # contribution instead of treating nodal interval currents independently.
    profile = tuple(
        sum(
            (
                (row[i] - row[j]) * current
                for (i, j), current in zip(_EDGES, edge_currents)
                if row[i] != row[j]
            ),
            I(0),
        )
        for row in poisson
    )
    profile_norm = sqrt(4 * sum((value**2 for value in profile), I(0))).hi
    gradient_norm = sqrt(sum((value**2 for value in currents), I(0))).hi
    potential = sum((1 - cos(gap) for gap in gaps), I(0))
    return tuple(currents), profile, profile_norm, gradient_norm, potential


@dataclass(frozen=True)
class SineFormationResponse:
    """Conditional acquisition, forward retention and actual receiver form.

    Initial rows are absolute nominal coordinates plus the declared per-node
    errors, with no additional free common origin. Geometry uses each actual
    member's conserved means; absolute receiver bounds retain its uncertain
    form mean separately. ``poisson_profile_bounds_by_preparation`` encloses
    L^+ S(v), not an observed or substituted endpoint. The transient and
    nonlinear history remain in ``scaled_form_remainder_upper_bounds``.

    All preparation-indexed tuples use ``preparation_order``. Endpoint norm
    and storage bounds concern the actual reached families under the same
    positive-loss law, not an independent product of observed endpoint boxes.
    The joint status requires initial zero winding, both strict forward
    trapping admissions and strict positive recorded A-minus-B contrast.
    No autonomous preparation, support selection or physical identification
    follows. Unavailable bounds do not prove a failed physical trajectory.
    """

    scaled_time: Q
    form_error_bound: Q
    phase_error_bound: Q
    readout_error_bound: Q
    radius: Q
    original_time: Q
    reference_model: RelationalExchangeModel
    full_dimensional_preparation: bool
    initial_forms_by_preparation: tuple[tuple[Q, ...], ...]
    initial_form_storage_by_preparation: tuple[Q, Q]
    gamma_bounds: I
    inverse_gamma_bounds: I
    eta_bounds: I
    forcing_norm_bounds: I
    scaled_initial_form_bounds: tuple[tuple[I, ...], ...]
    scaled_initial_norm_upper_bounds: tuple[Q, Q]
    initial_scaled_form_error_norm_upper_bounds: tuple[Q, Q]
    initial_phase_error_norm_upper_bound: Q
    exponential_decay_bounds: I
    poisson_inverse: tuple[tuple[Q, ...], ...]
    phase_current_bounds_by_preparation: tuple[tuple[I, ...], ...]
    poisson_profile_bounds_by_preparation: tuple[tuple[I, ...], ...]
    poisson_profile_norm_upper_bounds: tuple[Q, Q]
    scaled_form_remainder_upper_bounds: tuple[Q, Q]
    endpoint_form_norm_upper_bounds: tuple[Q, Q]
    endpoint_phase_error_norm_upper_bounds: tuple[Q, Q]
    proxy_target_distance_upper_bounds: tuple[Q, Q]
    endpoint_relative_norm_squared_upper_bounds: tuple[Q, Q]
    proxy_phase_storage_bounds: tuple[I, I]
    target_phase_storage_bounds: I
    proxy_excess_storage_bounds: tuple[I, I]
    proxy_phase_gradient_norm_upper_bounds: tuple[Q, Q]
    endpoint_excess_storage_upper_bounds: tuple[Q, Q]
    receiver_dual_norm_bounds: I
    receiver_center_bounds: tuple[I, I]
    receiver_error_upper_bounds: tuple[Q, Q]
    recorded_readout_bounds: tuple[I, I]
    recorded_difference_bounds: I
    initial_zero_winding_margin_bounds: I
    initial_zero_winding_certified: bool
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    acute_radius_certified: bool
    coercivity_lower_bound: Q | None
    barrier_lower_bound: Q | None
    endpoint_radius_margin_bounds: tuple[I, I]
    endpoint_radius_certified_by_preparation: tuple[bool, bool]
    storage_barrier_margin_bounds: tuple[I, I] | None
    storage_barrier_certified_by_preparation: tuple[bool, bool]
    formation_certified_by_preparation: tuple[bool, bool]
    response_separation_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    preparation_order: tuple[str, str] = ("pair0_form_split", "pair3_form_split")
    nodes: tuple[int, ...] = _NODES
    pairs: tuple[tuple[int, int], ...] = _PAIRS
    edges: tuple[tuple[int, int], ...] = _EDGES
    initial_phases: tuple[Q, ...] = _PHASE
    target_phase_turns: tuple[Q, ...] = _TARGET_TURNS
    held_capacities: tuple[Q, ...] = _CAPACITY
    metric_weights: tuple[Q, ...] = (Q(4),) * 10
    lambda_lower_bound: Q = _GAP
    lambda_upper_bound: Q = _RATE
    receiver_nodes: tuple[int, int] = (2, 3)
    form_loss: Q = Q(1023, 1024)
    exchange_weight: Q = Q(1, 1024)
    phase_exchange_beta: Q = Q(1)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "same_complete_positive_loss_law_all_twenty_coordinates_evolve",
        "absolute_phase_flat_integer_sources_with_independent_all_node_errors",
        "no_free_common_origins_beyond_the_declared_coordinate_error_budgets",
        "nominal_pair_means_and_storage_agree_perturbed_members_need_not",
        "initial_zero_winding_and_reached_nonzero_winding_are_separate_obligations",
        "semigroup_transient_and_nonlinear_history_retained_in_actual_form_readout",
        "Poisson_profile_is_an_analytic_reference_not_a_substituted_endpoint",
        "geometry_uses_each_members_conserved_form_and_lifted_phase_means",
        "positive_loss_storage_first_exit_bound_gives_forward_retention",
        "no_reset_to_target_or_transfer_of_conservative_response_coefficients",
        "strict_outward_initial_endpoint_barrier_and_recorded_response_margins",
        "unavailable_does_not_prove_failed_acquisition_or_equal_responses",
        "no_inputs_events_resets_numerical_trajectory_or_parameter_search",
        "no_autonomous_preparation_support_origin_law_selection_or_physical_identity",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-formation-response.v1", "report": _project(self)}


def assess_sine_formation_response(
    *, scaled_time, form_error_bound, phase_error_bound, readout_error_bound, radius
) -> SineFormationResponse:
    """Assess fixed phase-flat preparations and their actual reached receiver.

    All arguments are required. Admit nonnegative scaled time with
    ``(2/3)*scaled_time <= 4096``, nonnegative componentwise preparation and
    readout errors, and positive radius. Exact rational inputs remain exact;
    other reals follow shared represented-real admission. Invalid primitives
    reject before source construction or interval arithmetic. Nonpositive
    geometric/response margins return an unavailable certificate.

    The fixed original law has e=1023/1024, w=1/1024 and beta=capacity=1.
    Source means are 4039*(a-2) with (+96,-96) in pair zero or pair three.
    The phase-flat nominal state and every perturbed coordinate are evolved
    by the same law in the proof. This detached assessor performs no solver
    run, graph mutation, endpoint reset or cached-report consumption.
    """
    time = exact_or_represented_real(scaled_time, "scaled_time")
    errors = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_error_bound, "form_error_bound"),
            (phase_error_bound, "phase_error_bound"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    r = exact_or_represented_real(radius, "radius")
    if time < 0 or _GAP * time > _MAX_RATIONAL_EXPONENT:
        raise ValueError("scaled_time requires 0 <= (2/3)*scaled_time <= 4096")
    if any(error < 0 for error in errors):
        raise ValueError("preparation and readout error bounds must be nonnegative")
    if r <= 0:
        raise ValueError("radius must be strictly positive")
    rx, rt, readout_error = errors

    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    initial_forms = tuple(
        tuple(
            Q(4039 * (i // 2 - 2) + (96 * (1 - 2 * (i % 2)) if i // 2 == pair else 0))
            for i in _NODES
        )
        for pair in (0, 3)
    )
    # Every row is constructed here from fixed exact coordinates and the
    # already admitted budgets. Share a freshly admitted domain only within
    # this call; no temporary relative-rate/storage report is needed.
    domain = _sine_domain(_derive(_NODES, _EDGES), model, _CAPACITY)
    preparations = tuple(
        _sine_preparation_from_rows(
            domain,
            form=forms,
            phase=_PHASE,
            form_errors=(rx,) * 10,
            phase_errors=(rt,) * 10,
        )
        for forms in initial_forms
    )
    poisson = _poisson_geometry()
    first = preparations[0]
    gamma, inverse_gamma, eta, forcing = (
        first.alpha,
        first.inverse_alpha,
        first.eta,
        first.forcing,
    )
    decay = I(*_negative_exp_bounds(_GAP * time))
    phase_error_initial = forcing.hi * rt
    dual = 1 / sqrt(I(10))
    target_angles = tuple(2 * pi_interval() * turn for turn in _TARGET_TURNS)
    alpha = 2 * pi_interval() / 5
    target_storage = 20 * (1 - cos(alpha))
    fields = tuple(_proxy_fields(preparation, poisson) for preparation in preparations)
    remainders, phase_errors, form_norms, distances = [], [], [], []
    norm_squared, energy_upper, center_bounds, receiver_errors = [], [], [], []
    for preparation, (_, profile, profile_norm, gradient_norm, potential) in zip(
        preparations, fields
    ):
        remainder, phase_error_weighted = _prepared_duhamel_bounds(
            time=time,
            decay_upper=decay.hi,
            gap_lower=_GAP,
            rate_upper=_RATE,
            forcing_upper=forcing.hi,
            initial_norm_upper=preparation.nominal_initial_norm.hi,
            scaled_form_error_upper=preparation.initial_error_norm,
            phase_error_upper=phase_error_initial,
            feedback_upper=eta.hi,
        )
        phase_error = phase_error_weighted / 2
        form_norm = inverse_gamma.hi * (eta.hi * profile_norm + remainder) / 2
        distance = sqrt(
            sum(
                (
                    (value - target) ** 2
                    for value, target in zip(preparation.scaled_nominal, target_angles)
                ),
                I(0),
            )
        ).hi
        remainders.append(remainder)
        phase_errors.append(phase_error)
        form_norms.append(form_norm)
        distances.append(distance)
        norm_squared.append(form_norm**2 + (distance + phase_error) ** 2)
        energy_upper.append(
            (potential - target_storage).hi
            + gradient_norm * phase_error
            + 4 * phase_error**2
            + 4 * form_norm**2
        )
        center_bounds.append(gamma * (profile[2] + profile[3]) / 2)
        receiver_errors.append(
            rx + inverse_gamma.hi * remainder * dual.hi + readout_error
        )

    recorded = tuple(
        I(center.lo - error, center.hi + error)
        for center, error in zip(center_bounds, receiver_errors)
    )
    difference = I(
        center_bounds[0].lo - center_bounds[1].hi - sum(receiver_errors),
        center_bounds[0].hi - center_bounds[1].lo + sum(receiver_errors),
    )
    initial_margin = pi_interval() / 2 - 2 * rt
    initial_zero = initial_margin.lo > 0
    radius_angle = alpha + sqrt(I(2)) * r
    acute_margin = pi_interval() / 2 - radius_angle
    cosine = cos(radius_angle).lo if acute_margin.lo > 0 else None
    acute = cosine is not None and cosine > 0
    coercivity = ((5 - sqrt(I(5))) * cosine / 2).lo if acute else None
    barrier = (I(coercivity) * r**2).lo if coercivity is not None else None
    radius_margins = tuple(I(r**2 - norm) for norm in norm_squared)
    radius_flags = tuple(margin.lo > 0 for margin in radius_margins)
    energy_margins = (
        tuple(I(barrier - value) for value in energy_upper)
        if barrier is not None
        else None
    )
    energy_flags = tuple(
        energy_margins is not None and energy_margins[i].lo > 0 for i in (0, 1)
    )
    formation = tuple(
        initial_zero and acute and radius_flags[i] and energy_flags[i] for i in (0, 1)
    )
    separation = difference.lo > 0
    reasons = (
        tuple(
            reason
            for condition, reason in (
                (initial_zero, "whole_initial_set_zero_winding_not_certified"),
                (acute, "strict_acute_radius_not_certified"),
            )
            if not condition
        )
        + tuple(
            f"{name}:{reason}"
            for i, name in enumerate(("pair0_form_split", "pair3_form_split"))
            for condition, reason in (
                (radius_flags[i], "strict_endpoint_radius_not_certified"),
                (energy_flags[i], "strict_excess_storage_barrier_not_certified"),
            )
            if not condition
        )
        + (
            ()
            if separation
            else ("recorded_response_difference_not_strictly_positive",)
        )
    )
    return SineFormationResponse(
        scaled_time=time,
        form_error_bound=rx,
        phase_error_bound=rt,
        readout_error_bound=readout_error,
        radius=r,
        original_time=time / first.e,
        reference_model=model,
        full_dimensional_preparation=rx > 0 and rt > 0,
        initial_forms_by_preparation=initial_forms,
        initial_form_storage_by_preparation=tuple(
            sum(((forms[i] - forms[j]) ** 2 / 2 for i, j in _EDGES), Q(0))
            for forms in initial_forms
        ),
        gamma_bounds=gamma,
        inverse_gamma_bounds=inverse_gamma,
        eta_bounds=eta,
        forcing_norm_bounds=forcing,
        scaled_initial_form_bounds=tuple(p.scaled_nominal for p in preparations),
        scaled_initial_norm_upper_bounds=tuple(
            p.nominal_initial_norm.hi for p in preparations
        ),
        initial_scaled_form_error_norm_upper_bounds=tuple(
            p.initial_error_norm for p in preparations
        ),
        initial_phase_error_norm_upper_bound=phase_error_initial,
        exponential_decay_bounds=decay,
        poisson_inverse=poisson,
        phase_current_bounds_by_preparation=tuple(row[0] for row in fields),
        poisson_profile_bounds_by_preparation=tuple(row[1] for row in fields),
        poisson_profile_norm_upper_bounds=tuple(row[2] for row in fields),
        scaled_form_remainder_upper_bounds=tuple(remainders),
        endpoint_form_norm_upper_bounds=tuple(form_norms),
        endpoint_phase_error_norm_upper_bounds=tuple(phase_errors),
        proxy_target_distance_upper_bounds=tuple(distances),
        endpoint_relative_norm_squared_upper_bounds=tuple(norm_squared),
        proxy_phase_storage_bounds=tuple(row[4] for row in fields),
        target_phase_storage_bounds=target_storage,
        proxy_excess_storage_bounds=tuple(row[4] - target_storage for row in fields),
        proxy_phase_gradient_norm_upper_bounds=tuple(row[3] for row in fields),
        endpoint_excess_storage_upper_bounds=tuple(energy_upper),
        receiver_dual_norm_bounds=dual,
        receiver_center_bounds=tuple(center_bounds),
        receiver_error_upper_bounds=tuple(receiver_errors),
        recorded_readout_bounds=recorded,
        recorded_difference_bounds=difference,
        initial_zero_winding_margin_bounds=initial_margin,
        initial_zero_winding_certified=initial_zero,
        radius_angle_bounds=radius_angle,
        acute_radius_margin_bounds=acute_margin,
        acute_radius_certified=acute,
        coercivity_lower_bound=coercivity,
        barrier_lower_bound=barrier,
        endpoint_radius_margin_bounds=radius_margins,
        endpoint_radius_certified_by_preparation=radius_flags,
        storage_barrier_margin_bounds=energy_margins,
        storage_barrier_certified_by_preparation=energy_flags,
        formation_certified_by_preparation=formation,
        response_separation_certified=separation,
        status="certified_formation_response" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
