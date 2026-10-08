"""Degree-aware reduced C9 networks with complete storage and source bounds.

Every component retains five forms and five real target/origin phase deviations.
Internal sine currents are linearized, bridge currents remain exact sine, and
the contact graph determines all mobilities. No live graph or solver is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._sine_formed_contact import _unprobed_handoff, _UnprobedHandoff
from ._sine_port_geometry import _central_port_geometry, _PortGeometry
from .relational_observations import _ordered

__all__ = (
    "SinePortCompositionState",
    "SinePortComposition",
    "evaluate_sine_port_composition",
    "assess_sine_port_composition",
)

_MAX_COMPONENTS = 16


def _scalars(values, label, size):
    raw = _ordered(values, label, limit=size + 1)
    if len(raw) != size:
        raise ValueError(f"{label} must contain exactly {size} values")
    return tuple(
        exact_or_represented_real(x, f"{label}[{i}]") for i, x in enumerate(raw)
    )


def _support(classes, contacts, phase_origins):
    kinds = _ordered(classes, "classes", limit=_MAX_COMPONENTS + 1)
    if not 1 <= len(kinds) <= _MAX_COMPONENTS:
        raise ValueError("classes must contain between one and sixteen components")
    if any(type(k) is not int or k not in (1, 2) for k in kinds):
        raise ValueError("classes must be ordinary integers one or two")
    count = len(kinds)
    raw_edges = _ordered(contacts, "contacts", limit=count * (count - 1) // 2 + 1)
    edges = []
    for raw in raw_edges:
        pair = _ordered(raw, "contact", limit=3)
        if len(pair) != 2 or any(
            type(i) is not int or not 0 <= i < count for i in pair
        ):
            raise ValueError("each contact must contain two ordinary component indices")
        if pair[0] == pair[1]:
            raise ValueError("self contacts are not admitted")
        edges.append(tuple(sorted(pair)))
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate contacts are not admitted")
    origins = _scalars(phase_origins, "phase_origins", count)
    return tuple(kinds), tuple(sorted(edges)), origins


def _action(matrix, vector):
    return tuple(
        row[0] for row in exact_matrix_product(matrix, tuple((x,) for x in vector))
    )


def _sum(values):
    return sum(values, I(0))


def _parameters(classes):
    pi = pi_interval()
    return 1 / (1023 * pi), tuple(cos(2 * k * pi / 9) for k in classes)


def _bridge_data(geometry, origins, y):
    differences = tuple(
        origins[j] - origins[i] + y[5 * j] - y[5 * i] for i, j in geometry.contacts
    )
    currents = tuple(sin(I(value)) for value in differences)
    inputs = [I(0) for _ in origins]
    for (i, j), value in zip(geometry.contacts, currents):
        inputs[i] = inputs[i] + value
        inputs[j] = inputs[j] - value
    return differences, currents, tuple(inputs)


@dataclass(frozen=True)
class SinePortCompositionState:
    """Instantaneous reduced rows and balances for a supplied contact network.

    Coordinates are component-major radial layers, five entries per channel.
    Phase deviations subtract the winding target AND the supplied common origin.
    Port inputs are neighbor-minus-own form and exact sine currents. Conjugate
    efforts use internal gradients, not the bare form/phase port coordinates.
    storage_rate is the algebraic closed-network identity; the independently
    evaluated rowwise interval retains elementary-function rounding uncertainty.
    """

    classes: tuple[int, ...]
    contacts: tuple[tuple[int, int], ...]
    phase_origins: tuple[Q, ...]
    forms: tuple[Q, ...]
    phase_deviations: tuple[Q, ...]
    geometry: _PortGeometry
    gamma_bounds: I
    class_cosine_bounds: tuple[I, ...]
    form_gradients: tuple[Q, ...]
    phase_gradient_bounds: tuple[I, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    bridge_form_differences: tuple[Q, ...]
    bridge_phase_differences: tuple[Q, ...]
    bridge_sine_bounds: tuple[I, ...]
    internal_form_storage: tuple[Q, ...]
    internal_phase_storage_bounds: tuple[I, ...]
    bridge_form_storage: tuple[Q, ...]
    bridge_phase_storage_bounds: tuple[I, ...]
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    dissipation: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I
    weighted_form_charges: tuple[Q, ...]
    weighted_phase_charges: tuple[Q, ...]
    component_form_charge_rate_bounds: tuple[I, ...]
    component_phase_charge_rate_bounds: tuple[I, ...]
    network_form_charge_rate: Q
    network_phase_charge_rate: Q
    port_form_inputs: tuple[Q, ...]
    port_phase_input_bounds: tuple[I, ...]
    port_form_effort_bounds: tuple[I, ...]
    port_phase_effort_bounds: tuple[I, ...]
    internal_dissipation: tuple[Q, ...]
    internal_supply_rate_bounds: tuple[I, ...]
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    implementation_component_cap: int = _MAX_COMPONENTS
    scope: tuple[str, ...] = (
        "simple_unit_undirected_central_contacts_with_actual_degree_dependent_mobility",
        "ten_coordinates_per_component_linear_interior_and_exact_sine_bridge",
        "all_origins_retained_bridge_differences_derived_from_component_origins",
        "reduced_storage_omits_constant_internal_target_potentials",
        "gradient_conjugate_port_supply_and_closed_network_dissipation",
        "instantaneous_surrogate_not_a_full_nonlinear_quotient_or_formation_certificate",
        "no_live_graph_solver_event_selector_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-port-composition-state.v1",
            "report": _project(self),
        }


def evaluate_sine_port_composition(
    *,
    classes,
    contacts,
    phase_origins,
    forms,
    phase_deviations,
) -> SinePortCompositionState:
    """Evaluate a fresh degree-aware composition after original-value admission.

    Allow one to sixteen components, including disconnected contacts. The cap
    bounds dense exact assembly, not a mathematical limit. Contacts contain
    unique distinct integer component endpoints and have unit weight. Forms and
    target/origin-subtracted real phase deviations have five entries/component.
    Every origin is held in the declared structural clock; no edge offset is
    independently supplied and no phase deviation is wrapped.
    """
    kinds, edges, origins = _support(classes, contacts, phase_origins)
    n, size = len(kinds), 5 * len(kinds)
    x = _scalars(forms, "forms", size)
    y = _scalars(phase_deviations, "phase_deviations", size)
    geometry = _central_port_geometry(n, edges)
    masses = geometry.layer_masses
    gamma, coefficients = _parameters(kinds)
    gx = _action(geometry.joined_laplacian, x)
    ix, iy = _action(geometry.internal_laplacian, x), _action(
        geometry.internal_laplacian, y
    )
    gaps, currents, theta_inputs = _bridge_data(geometry, origins, y)
    form_gaps = tuple(x[5 * j] - x[5 * i] for i, j in edges)
    form_inputs = [Q(0) for _ in kinds]
    gradient = [coefficients[i // 5] * iy[i] for i in range(size)]
    for (i, j), gap, current in zip(edges, form_gaps, currents):
        form_inputs[i] += gap
        form_inputs[j] -= gap
        gradient[5 * i] = gradient[5 * i] - current
        gradient[5 * j] = gradient[5 * j] + current
    rates_x = tuple((-gx[i] - gamma * gradient[i]) / masses[i] for i in range(size))
    rates_y = tuple(gamma * (gx[i] / masses[i]) for i in range(size))
    internal_form = tuple(
        sum((x[j] * ix[j] for j in range(5 * i, 5 * i + 5)), Q(0)) / 2 for i in range(n)
    )
    internal_phase = tuple(
        coefficients[i]
        * (sum((y[j] * iy[j] for j in range(5 * i, 5 * i + 5)), Q(0)) / 2)
        for i in range(n)
    )
    bridge_form = tuple(value**2 / 2 for value in form_gaps)
    bridge_phase = tuple(2 * sin(I(value / 2)) ** 2 for value in gaps)
    form_storage = sum(internal_form + bridge_form, Q(0))
    phase_storage = _sum(internal_phase + bridge_phase)
    loss = sum((gx[j] ** 2 / masses[j] for j in range(size)), Q(0))
    rowwise = _sum(gx[j] * rates_x[j] + gradient[j] * rates_y[j] for j in range(size))
    form_charges = tuple(
        sum((masses[j] * x[j] for j in range(5 * i, 5 * i + 5)), Q(0)) for i in range(n)
    )
    phase_charges = tuple(
        sum((masses[j] * (y[j] + origins[i]) for j in range(5 * i, 5 * i + 5)), Q(0))
        for i in range(n)
    )
    effort_x = tuple(
        ix[5 * i] / masses[5 * i]
        - gamma * coefficients[i] * (iy[5 * i] / masses[5 * i])
        for i in range(n)
    )
    effort_y = tuple(gamma * (ix[5 * i] / masses[5 * i]) for i in range(n))
    internal_loss = tuple(
        sum((ix[j] ** 2 / masses[j] for j in range(5 * i, 5 * i + 5)), Q(0))
        for i in range(n)
    )
    supply = tuple(
        effort_x[i] * form_inputs[i] + effort_y[i] * theta_inputs[i] for i in range(n)
    )
    # These are exact coefficient checks, not cancellation of independent
    # rounded intervals for the same sine or cosine quantity.
    for matrix in (geometry.internal_laplacian, geometry.joined_laplacian):
        if any(sum(row[j] for row in matrix) != 0 for j in range(size)):
            raise ArithmeticError("assembled Laplacian does not conserve total charge")
        if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(size)):
            raise ArithmeticError("assembled Laplacian is not symmetric")
    if sum(form_inputs) != 0:
        raise ArithmeticError("contact form incidence failed cancellation")
    return SinePortCompositionState(
        classes=kinds,
        contacts=edges,
        phase_origins=origins,
        forms=x,
        phase_deviations=y,
        geometry=geometry,
        gamma_bounds=gamma,
        class_cosine_bounds=coefficients,
        form_gradients=gx,
        phase_gradient_bounds=tuple(gradient),
        form_rate_bounds=rates_x,
        phase_rate_bounds=rates_y,
        bridge_form_differences=form_gaps,
        bridge_phase_differences=gaps,
        bridge_sine_bounds=currents,
        internal_form_storage=internal_form,
        internal_phase_storage_bounds=internal_phase,
        bridge_form_storage=bridge_form,
        bridge_phase_storage_bounds=bridge_phase,
        form_storage=form_storage,
        phase_storage_bounds=phase_storage,
        storage_bounds=form_storage + phase_storage,
        dissipation=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=rowwise,
        weighted_form_charges=form_charges,
        weighted_phase_charges=phase_charges,
        component_form_charge_rate_bounds=tuple(
            form_inputs[i] + gamma * theta_inputs[i] for i in range(n)
        ),
        component_phase_charge_rate_bounds=tuple(
            -gamma * value for value in form_inputs
        ),
        network_form_charge_rate=Q(0),
        network_phase_charge_rate=Q(0),
        port_form_inputs=tuple(form_inputs),
        port_phase_input_bounds=theta_inputs,
        port_form_effort_bounds=effort_x,
        port_phase_effort_bounds=effort_y,
        internal_dissipation=internal_loss,
        internal_supply_rate_bounds=supply,
    )


@dataclass(frozen=True)
class SinePortComposition:
    """Actual-family accuracy and whole-network identity under supplied contact.

    The common source library is freshly certified for both classes. Accuracy
    is a uniform maximum-norm bound on all fine form and phase coordinates over
    the whole contact window; it is not a response contrast or exact closure.
    The single global recovery certificate additionally requires at least two
    components and connected contact support. Missing premises retain None.
    """

    classes: tuple[int, ...]
    contacts: tuple[tuple[int, int], ...]
    phase_origins: tuple[Q, ...]
    formation_time: Q
    relaxation_duration: Q
    contact_duration: Q
    form_error_bound: Q
    phase_error_bound: Q
    endpoint_radius: Q
    radius: Q
    decay_power: int
    work_allowance: Q
    approximation_allowance: Q
    geometry: _PortGeometry
    unprobed_handoff: _UnprobedHandoff
    ideal_port_forcing_bounds: tuple[I, ...]
    ideal_forcing_norm_upper_bound: Q
    bootstrap_denominator_lower_bound: Q
    ideal_surrogate_discrepancy_upper_bound: Q
    preparation_error_upper_bound: Q | None
    total_approximation_error_upper_bound: Q | None
    approximation_margin_bounds: I | None
    joined_gap_lower_bound: Q | None
    joined_cosine_bounds: I
    joined_barrier_lower_bound: Q | None
    joined_radius_squared_upper_bound: Q | None
    joined_excess_storage_upper_bound: Q | None
    joined_radius_margin_bounds: I | None
    joined_storage_margin_bounds: I | None
    contact_work_upper_bound: Q | None
    work_margin_bounds: I | None
    joined_form_mean_bounds: I | None
    joined_phase_mean_bounds: I | None
    approximation_certified: bool
    identity_certified: bool
    work_within_allowance: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    implementation_component_cap: int = _MAX_COMPONENTS
    scope: tuple[str, ...] = (
        "fresh_two_class_source_library_with_independent_original_zero_sum_errors",
        "unprobed_actual_source_images_no_incoming_report_or_ideal_reset",
        "phase_origins_supplied_per_component_and_all_edge_offsets_derived",
        "uniform_all_fine_coordinate_error_over_the_whole_window",
        "initial_and_generated_odd_modes_retained_in_full_space_error",
        "connected_multi_component_full_sine_barrier_not_surrogate_storage",
        "supplied_contact_work_separate_from_continuous_dissipation",
        "no_contrast_prediction_solver_event_selector_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-port-composition.v1", "report": _project(self)}


def assess_sine_port_composition(
    *,
    classes,
    contacts,
    phase_origins,
    formation_time,
    relaxation_duration,
    contact_duration,
    form_error_bound,
    phase_error_bound,
    endpoint_radius,
    radius,
    decay_power,
    work_allowance,
    approximation_allowance,
) -> SinePortComposition:
    """Rebuild generic degree-aware composition bounds from original primitives.

    Support admission matches the row evaluator. All scalar budgets/times are
    finite exact admitted nonnegative reals, with positive endpoint_radius and
    0<radius<=1/12, contact_duration<=1/4. decay_power is an ordinary integer
    zero through4096. Shared formation/relaxation exponential work caps apply.
    A missing connected multi-component theorem or insufficient bound returns
    unavailable rather than inventing a recovered common mean.
    """
    kinds, edges, origins = _support(classes, contacts, phase_origins)
    raw = dict(
        formation_time=formation_time,
        relaxation_duration=relaxation_duration,
        contact_duration=contact_duration,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        endpoint_radius=endpoint_radius,
        radius=radius,
        work_allowance=work_allowance,
        approximation_allowance=approximation_allowance,
    )
    v = {name: exact_or_represented_real(value, name) for name, value in raw.items()}
    if any(value < 0 for value in v.values()):
        raise ValueError("composition times and budgets must be nonnegative")
    if not 0 < v["radius"] <= Q(1, 12) or v["endpoint_radius"] <= 0:
        raise ValueError("require 0 < radius <= 1/12 and positive endpoint_radius")
    if v["contact_duration"] > Q(1, 4):
        raise ValueError("contact_duration must not exceed1/4")
    if type(decay_power) is not int or not 0 <= decay_power <= 4096:
        raise ValueError("decay_power must be an ordinary integer zero through4096")
    n, h, eps, r = len(kinds), v["contact_duration"], v["endpoint_radius"], v["radius"]
    geometry = _central_port_geometry(n, edges)
    handoff = _unprobed_handoff(
        formation_time=v["formation_time"],
        relaxation_duration=v["relaxation_duration"],
        form_error_bound=v["form_error_bound"],
        phase_error_bound=v["phase_error_bound"],
        radius=r,
        endpoint_radius=eps,
        decay_power=decay_power,
    )
    gamma, _ = _parameters(kinds)
    _, _, sine_inputs = _bridge_data(geometry, origins, (Q(0),) * (5 * n))
    forcing = tuple(
        value / (2 + degree)
        for value, degree in zip(sine_inputs, geometry.contact_degrees)
    )
    sigma = max(value.abs_max for value in forcing)
    denominator = 1 - Q(2, 3) * gamma.hi**2 * h**2
    if denominator <= 0:
        raise ArithmeticError("fixed composition bootstrap denominator is nonpositive")
    discrepancy = 2 * gamma.hi**5 * sigma**2 * h**5 / (5 * denominator**2 * (1 - 3 * h))
    cosine = cos(4 * pi_interval() / 9 + sqrt(I(2)) * r)
    supported = n >= 2 and geometry.connected
    gap = Q(4, 9 * n * (geometry.contact_diameter + 8)) if supported else None
    barrier = gap * cosine.lo * r**2 / 2 if gap is not None and cosine.lo > 0 else None
    prep = total = approx_margin = z2 = energy = radius_margin = storage_margin = None
    work = work_margin = mean_x = mean_phase = None
    accuracy = identity = allowed = False
    if all(handoff.handoff_certified_by_class):
        prep = eps / (1 - 3 * h)
        total = discrepancy + prep
        approx_margin = I(v["approximation_allowance"] - total)
        accuracy = approx_margin.lo > 0
        origin_mean = sum(origins, Q(0)) / n
        spread = sum(((value - origin_mean) ** 2 for value in origins), Q(0))
        z2 = 2 * n * eps**2 + 9 * spread
        phase_cost = sum(
            ((abs(origins[j] - origins[i]) + 2 * eps) ** 2 / 2 for i, j in edges), Q(0)
        )
        energy = (4 + max(geometry.contact_degrees)) * n * eps**2 + phase_cost
        radius_margin = I(r**2 - z2)
        storage_margin = I(barrier - energy) if barrier is not None else None
        identity = bool(
            storage_margin is not None
            and radius_margin.lo > 0
            and storage_margin.lo > 0
        )
        work = 2 * len(edges) * eps**2 + phase_cost
        work_margin = I(v["work_allowance"] - work)
        allowed = work <= v["work_allowance"]
        mass = 18 * n + 2 * len(edges)
        mean_error = Q(2 * len(edges), mass) * eps
        mean_x = I(-mean_error, mean_error)
        center = (
            sum(
                (
                    (18 + degree) * origin
                    for degree, origin in zip(geometry.contact_degrees, origins)
                ),
                Q(0),
            )
            / mass
        )
        mean_phase = I(center - mean_error, center + mean_error)
    reasons = tuple(
        reason
        for condition, reason in (
            (
                handoff.formation_certificate.status == "certified_two_formed_classes",
                "formation_unavailable",
            ),
            (
                all(handoff.handoff_certified_by_class),
                "unprobed_endpoint_budget_not_certified",
            ),
            (supported, "connected_multi_component_identity_not_supported"),
            (identity, "whole_network_identity_not_certified"),
            (allowed, "supplied_contact_work_allowance_not_certified"),
            (accuracy, "uniform_approximation_allowance_not_certified"),
        )
        if not condition
    )
    return SinePortComposition(
        **v,
        classes=kinds,
        contacts=edges,
        phase_origins=origins,
        decay_power=decay_power,
        geometry=geometry,
        unprobed_handoff=handoff,
        ideal_port_forcing_bounds=forcing,
        ideal_forcing_norm_upper_bound=sigma,
        bootstrap_denominator_lower_bound=denominator,
        ideal_surrogate_discrepancy_upper_bound=discrepancy,
        preparation_error_upper_bound=prep,
        total_approximation_error_upper_bound=total,
        approximation_margin_bounds=approx_margin,
        joined_gap_lower_bound=gap,
        joined_cosine_bounds=cosine,
        joined_barrier_lower_bound=barrier,
        joined_radius_squared_upper_bound=z2,
        joined_excess_storage_upper_bound=energy,
        joined_radius_margin_bounds=radius_margin,
        joined_storage_margin_bounds=storage_margin,
        contact_work_upper_bound=work,
        work_margin_bounds=work_margin,
        joined_form_mean_bounds=mean_x,
        joined_phase_mean_bounds=mean_phase,
        approximation_certified=accuracy,
        identity_certified=identity,
        work_within_allowance=allowed,
        status="certified_sine_port_composition" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
