"""Detached actual-state and conditional-minimum sine mediation reports.

Retained environmental pressure preserves the captured hidden state and
derives its causal-memory coefficients and interface work. The separate
stationary comparison replaces it with a conditional storage minimum.
Neither report evolves a graph or establishes an invariant reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._phase_resultant_chamber import (
    certified_cosine_bounds,
    certified_sine_bounds,
    relative_resultant_bounds,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from .relational_sine_comparison import (
    SineExchangeComparison,
    _capture_sine_state,
    _comparison_neighbors,
    _sine_work,
    _validate_comparison_labels,
)

__all__ = (
    "SineMediationComparison",
    "SineMediatedPressure",
    "bound_relational_sine_mediation",
)


@dataclass(frozen=True)
class SineMediatedPressure:
    """Exact captured hidden state and ideal environmental-pressure bounds.

    All port arrays follow ``ports`` and retain each original incidence
    degree. ``port_phase_currents`` uses sin(theta_port-theta_hidden); the
    corresponding environmental current into a port has the opposite sign.
    The pressure and phase-rate splits are additive; their separate squared
    losses are not asserted to add. ``incident_phase_cost`` excludes beta.
    ``port_boundary_form_work``, ``port_boundary_phase_work`` and
    ``port_boundary_work`` give signed instantaneous input to the hidden
    incident star, in the same port order. Their sums are the corresponding
    boundary work fields. If one port-to-hidden edge is the entire boundary
    of an external region, that region's input power is the negative port
    entry. These snapshots do not evaluate accumulated regional work.

    The coefficients lambda=mu*e, kappa=mu*w/(beta*pi), and
    gamma=mu**2*w**2/(beta*pi**2*k) belong to the derived hidden equation
    theta_h''+lambda*theta_h'=gamma*sum_port sin(theta_p-theta_h)-kappa*X'.
    They specify its causal integral representation without evaluating a
    memory convolution, eliminating hidden initial state or selecting a law.
    """

    comparison: SineExchangeComparison
    mediator: Any
    ports: tuple[Any, ...]
    hidden_capacity: Q
    port_count: int
    hidden_form: Q
    port_mean_form: Q
    hidden_form_contrast: Q
    hidden_form_rate: I
    hidden_phase_rate: I
    port_mean_form_rate: I
    hidden_form_contrast_rate: I
    hidden_phase_acceleration: I
    memory_decay: Q
    phase_gain: I
    memory_phase_gain: I
    port_form_contrasts: tuple[Q, ...]
    port_phase_currents: tuple[I, ...]
    port_degrees: tuple[int, ...]
    environmental_pressures: tuple[I, ...]
    internal_pressures: tuple[I, ...]
    environmental_phase_rates: tuple[I, ...]
    internal_phase_rates: tuple[I, ...]
    pressure_reconstruction_residuals: tuple[I, ...]
    phase_rate_reconstruction_residuals: tuple[I, ...]
    incident_form_storage: Q
    incident_phase_cost: I
    incident_storage: I
    boundary_form_work: I
    boundary_phase_work: I
    boundary_work: I
    hidden_loss: Q
    incident_storage_rate: I
    balance_residual: I
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_supplied_sine_law_represented_by_causal_environmental_pressure",
        "captured_hidden_form_phase_capacity_retained_without_initial_time_authentication",
        "at_least_two_supplied_ports_with_original_incidence_normalization",
        "zero_hidden_capacity_zero_form_loss_and_cancelled_resultants_are_admitted",
        "ideal_instantaneous_rates_acceleration_storage_and_boundary_work_bounds",
        "volterra_coefficients_do_not_evaluate_a_memory_trajectory",
        "pressure_and_phase_rate_splits_do_not_imply_additive_squared_dissipations",
        "computed_reconstruction_and_balance_residuals_are_not_forced_to_zero",
        "no_graph_reread_solver_stationary_replacement_support_event_or_new_law",
        "supplied_comparison_is_retained_not_authenticated_by_dataclass_projection",
    )
    port_boundary_form_work: tuple[I, ...] | None = None
    port_boundary_phase_work: tuple[I, ...] | None = None
    port_boundary_work: tuple[I, ...] | None = None

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        _validate_label(self.mediator)
        for port in self.ports:
            _validate_label(port)
        return {
            "schema": "tnfr.relational-sine-mediated-pressure.v1",
            "report": _project(self),
        }


def _mediated_pressure(comparison, *, mediator):
    """Derive an incident-star decomposition from one existing comparison."""
    if not isinstance(comparison, SineExchangeComparison):
        raise TypeError("mediated pressure requires a captured sine comparison")
    positions = {node: i for i, node in enumerate(comparison.nodes)}
    if mediator not in positions:
        raise ValueError("mediator must be a node of the captured support")
    hidden = positions[mediator]
    neighbors = _comparison_neighbors(comparison)
    port_indices = neighbors[hidden]
    k = len(port_indices)
    if k < 2:
        raise ValueError("mediated pressure requires at least two distinct ports")

    epi, phase, capacity = comparison.epi, comparison.phase, comparison.capacity
    y, mu = epi[hidden], capacity[hidden]
    mean_form = sum((epi[i] for i in port_indices), Q(0)) / k
    contrast = y - mean_form
    e, w = map(Q, comparison.reference_model.effective_weights)
    beta = Q(comparison.reference_model.storage_scale)
    pi = pi_interval()
    phase_gain = mu * w / (beta * pi)
    memory_phase_gain = mu**2 * w**2 / (beta * pi**2 * k)
    mean_form_rate = sum((comparison.form_rates[i] for i in port_indices), I(0)) / k
    contrast_rate = comparison.form_rates[hidden] - mean_form_rate

    port_contrasts = tuple(epi[i] - y for i in port_indices)
    port_currents = tuple(
        I(*certified_sine_bounds(phase[i] - phase[hidden])) for i in port_indices
    )
    degrees = tuple(comparison.degrees[i] for i in port_indices)
    port_capacity = tuple(capacity[i] for i in port_indices)
    internal_gradients = tuple(
        sum((epi[i] - epi[j] for j in neighbors[i] if j != hidden), Q(0))
        for i in port_indices
    )
    internal_currents = tuple(
        sum(
            (
                I(*certified_sine_bounds(phase[j] - phase[i]))
                for j in neighbors[i]
                if j != hidden
            ),
            I(0),
        )
        for i in port_indices
    )
    environmental = _sine_work(
        comparison.reference_model,
        degrees,
        port_contrasts,
        port_capacity,
        tuple(-current for current in port_currents),
    )
    internal = _sine_work(
        comparison.reference_model,
        degrees,
        internal_gradients,
        port_capacity,
        internal_currents,
    )
    pressure_residuals = tuple(
        env + direct - comparison.pressure[i]
        for i, env, direct in zip(
            port_indices, environmental["pressure"], internal["pressure"]
        )
    )
    phase_residuals = tuple(
        env + direct - comparison.phase_rates[i]
        for i, env, direct in zip(
            port_indices, environmental["phase_rates"], internal["phase_rates"]
        )
    )

    form_storage = sum((value**2 / 2 for value in port_contrasts), Q(0))
    phase_cost = sum(
        (
            1 - I(*certified_cosine_bounds(phase[i] - phase[hidden]))
            for i in port_indices
        ),
        I(0),
    )
    port_form_work = tuple(
        value * comparison.form_rates[i]
        for i, value in zip(port_indices, port_contrasts)
    )
    port_phase_work = tuple(
        beta * current * comparison.phase_rates[i]
        for i, current in zip(port_indices, port_currents)
    )
    port_work = tuple(
        form + phase for form, phase in zip(port_form_work, port_phase_work)
    )
    boundary_form = sum(port_form_work, I(0))
    boundary_phase = sum(port_phase_work, I(0))
    boundary_work = boundary_form + boundary_phase
    storage_rate = sum(
        (
            value * (comparison.form_rates[i] - comparison.form_rates[hidden])
            + beta
            * current
            * (comparison.phase_rates[i] - comparison.phase_rates[hidden])
            for i, value, current in zip(port_indices, port_contrasts, port_currents)
        ),
        I(0),
    )
    hidden_loss = mu * e * k * contrast**2
    return SineMediatedPressure(
        comparison=comparison,
        mediator=mediator,
        ports=tuple(comparison.nodes[i] for i in port_indices),
        hidden_capacity=mu,
        port_count=k,
        hidden_form=y,
        port_mean_form=mean_form,
        hidden_form_contrast=contrast,
        hidden_form_rate=comparison.form_rates[hidden],
        hidden_phase_rate=comparison.phase_rates[hidden],
        port_mean_form_rate=mean_form_rate,
        hidden_form_contrast_rate=contrast_rate,
        hidden_phase_acceleration=phase_gain * contrast_rate,
        memory_decay=mu * e,
        phase_gain=phase_gain,
        memory_phase_gain=memory_phase_gain,
        port_form_contrasts=port_contrasts,
        port_phase_currents=port_currents,
        port_degrees=degrees,
        environmental_pressures=environmental["pressure"],
        internal_pressures=internal["pressure"],
        environmental_phase_rates=environmental["phase_rates"],
        internal_phase_rates=internal["phase_rates"],
        pressure_reconstruction_residuals=pressure_residuals,
        phase_rate_reconstruction_residuals=phase_residuals,
        incident_form_storage=form_storage,
        incident_phase_cost=phase_cost,
        incident_storage=form_storage + beta * phase_cost,
        boundary_form_work=boundary_form,
        boundary_phase_work=boundary_phase,
        boundary_work=boundary_work,
        hidden_loss=hidden_loss,
        incident_storage_rate=storage_rate,
        balance_residual=storage_rate - boundary_work + hidden_loss,
        port_boundary_form_work=port_form_work,
        port_boundary_phase_work=port_phase_work,
        port_boundary_work=port_work,
    )


@dataclass(frozen=True)
class SineMediationComparison:
    """Ideal visible field after a declared stationary hidden replacement.

    Source coordinates retain the actual capture, including the supplied
    hidden state. The replacement uses an exact form mean and a unit complex
    direction relative to the first port, never a rounded angle. Effective
    degrees retain the original incidence, not degrees of an invented graph.
    """

    reference_model: RelationalExchangeModel
    source_nodes: tuple[Any, ...]
    source_edges: tuple[tuple[Any, Any], ...]
    source_epi: tuple[Q, ...]
    source_phase: tuple[Q, ...]
    source_capacity: tuple[Q, ...]
    mediator: Any
    ports: tuple[Any, ...]
    hidden_capacity: Q
    hidden_form: Q
    resultant_magnitude: I
    hidden_phase_curvature: I
    minimum_phase_relative_to_first_port: tuple[I, I]
    port_relative_resultants: tuple[tuple[I, I], ...]
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    capacity: tuple[Q, ...]
    degrees: tuple[int, ...]
    form_gradient: tuple[Q, ...]
    relative_resultant: tuple[tuple[I, I], ...]
    inverse_phase_metric: tuple[I, ...]
    phase_sources: tuple[I, ...]
    pressure: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    form_storage: Q
    phase_storage: I
    storage: I
    dissipation: tuple[Q, ...]
    continuous_loss: Q
    form_work: tuple[I, ...]
    phase_work: tuple[I, ...]
    node_balance_residual: tuple[I, ...]
    storage_rate: I
    balance_residual: I
    hidden_form_tracking_defect: I
    hidden_phase_tracking_defect: I
    law: str = "stationary_minimum_reduction_of_normalized_sine_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "one_supplied_mediator_with_at_least_two_ports_and_positive_hidden_capacity",
        "fixed_simple_connected_unit_support_held_nonnegative_capacity_no_input_or_events",
        "nonzero_port_resultant_certifies_unique_hidden_phase_minimum_modulo_a_turn",
        "stationary_hidden_minimum_is_not_the_unique_stationary_branch_or_whole_equilibrium",
        "source_coordinates_captured_before_hypothetical_stationary_replacement",
        "conditional_visible_gradient_field_retains_original_degrees",
        "no_hidden_angle_materialization_or_branch_choice",
        "tracking_defects_test_instantaneous_tangency_not_finite_time_error",
        "no_graph_write_solver_invariant_manifold_fast_limit_or_memoryless_closure_claim",
        "no_native_arg_pressure_derivation_occurrence_law_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        for node in (*self.source_nodes, *self.nodes, *self.ports, self.mediator):
            _validate_label(node)
        for edge in (*self.source_edges, *self.edges):
            for node in edge:
                _validate_label(node)

        return {
            "schema": "tnfr.relational-sine-mediation.v1",
            "report": _project(self),
        }


def _unit_component(numerator, magnitude):
    """Intersect an enclosed normalized phasor coordinate with its unit range.

    Each numerator is a real or imaginary coordinate of the same resultant
    whose strictly positive modulus is enclosed by magnitude. Division loses
    that dependence; its exact coordinate still lies in [-1, 1]. The
    intersection retains every possible exact value without asserting zero.
    """
    quotient = numerator / magnitude
    return I(max(Q(-1), quotient.lo), min(Q(1), quotient.hi))


def bound_relational_sine_mediation(graph, *, mediator, reference_model):
    """Enclose the visible stationary-minimum reduction without evolution.

    For k ports, the declared hidden minimum is x_h=mean(x_port) and
    exp(i*theta_h)=Z/|Z| with Z=sum exp(i*theta_port). A strictly positive
    certified resultant is required. Source hidden coordinates remain in
    provenance but are replaced only inside this detached comparison.

    The visible form gradient and sine current are derivatives of the exact
    reduced storage. Shared sine-law rate/work rows preserve its conditional
    loss identity. At finite hidden capacity, the hidden row vanishes at this
    minimum while its required position generally moves; the two returned
    tracking defects make this failure of invariance explicit.
    """
    state = _capture_sine_state(graph, reference_model)
    indices = {node: i for i, node in enumerate(state.nodes)}
    if mediator not in indices:
        raise ValueError("mediator must be a node of the supplied support")
    hidden = indices[mediator]
    port_indices = state.neighbors[hidden]
    k = len(port_indices)
    if k < 2:
        raise ValueError("mediator requires at least two distinct ports")
    if state.capacity[hidden] <= 0:
        raise ValueError("stationary mediation requires positive hidden capacity")

    port_phases = tuple(state.phase[i] for i in port_indices)
    port_rows = tuple(tuple(j for j in range(k) if j != i) for i in range(k))
    port_resultants = tuple(
        (1 + I(*real), I(*imag))
        for real, imag in relative_resultant_bounds(port_phases, port_rows)
    )
    real, imag = port_resultants[0]
    magnitude = sqrt(real**2 + imag**2)
    # Triangle inequality: a sum of k unit phasors has modulus at most k.
    magnitude = I(magnitude.lo, min(Q(k), magnitude.hi))
    if magnitude.lo <= 0:
        raise ValueError("port resultant is zero or not certified away from zero")
    port_directions = tuple(
        (_unit_component(real, magnitude), _unit_component(imag, magnitude))
        for real, imag in port_resultants
    )

    visible = tuple(i for i in range(len(state.nodes)) if i != hidden)
    visible_positions = {source: i for i, source in enumerate(visible)}
    port_positions = {source: i for i, source in enumerate(port_indices)}
    nodes = tuple(state.nodes[i] for i in visible)
    epi = tuple(state.epi[i] for i in visible)
    phase = tuple(state.phase[i] for i in visible)
    capacity = tuple(state.capacity[i] for i in visible)
    degrees = tuple(state.degrees[i] for i in visible)
    internal_rows = tuple(
        tuple(visible_positions[j] for j in state.neighbors[i] if j != hidden)
        for i in visible
    )
    internal_resultants = relative_resultant_bounds(phase, internal_rows)
    hidden_form = sum((state.epi[i] for i in port_indices), Q(0)) / k
    gradient, resultants = [], []
    for position, source in enumerate(visible):
        row = internal_rows[position]
        q = sum((epi[position] - epi[j] for j in row), Q(0))
        real_bounds, imag_bounds = internal_resultants[position]
        real, imag = I(*real_bounds), I(*imag_bounds)
        if source in port_positions:
            q += epi[position] - hidden_form
            port_real, port_imag = port_directions[port_positions[source]]
            real += port_real
            imag += port_imag
        gradient.append(q)
        resultants.append((real, imag))
    gradient, resultants = tuple(gradient), tuple(resultants)

    edges = tuple(edge for edge in state.edges if mediator not in edge)
    form_storage = sum(
        ((state.epi[i] - hidden_form) ** 2 / 2 for i in port_indices), Q(0)
    )
    phase_storage = k - magnitude
    for left, right in edges:
        i, j = indices[left], indices[right]
        form_storage += (state.epi[i] - state.epi[j]) ** 2 / 2
        phase_storage += 1 - I(
            *certified_cosine_bounds(state.phase[j] - state.phase[i])
        )
    beta = Q(reference_model.storage_scale)
    field = _sine_work(
        reference_model,
        degrees,
        gradient,
        capacity,
        tuple(imag for _, imag in resultants),
    )
    form_defect = (
        -sum((field["form_rates"][visible_positions[i]] for i in port_indices), I(0))
        / k
    )
    phase_defect = (
        -sum(
            (
                real * field["phase_rates"][visible_positions[i]]
                for i, (real, _) in zip(port_indices, port_directions)
            ),
            I(0),
        )
        / magnitude
    )
    return SineMediationComparison(
        reference_model=reference_model,
        source_nodes=state.nodes,
        source_edges=state.edges,
        source_epi=state.epi,
        source_phase=state.phase,
        source_capacity=state.capacity,
        mediator=mediator,
        ports=tuple(state.nodes[i] for i in port_indices),
        hidden_capacity=state.capacity[hidden],
        hidden_form=hidden_form,
        resultant_magnitude=magnitude,
        hidden_phase_curvature=beta * magnitude,
        minimum_phase_relative_to_first_port=port_directions[0],
        port_relative_resultants=port_resultants,
        nodes=nodes,
        edges=edges,
        epi=epi,
        phase=phase,
        capacity=capacity,
        degrees=degrees,
        form_gradient=gradient,
        relative_resultant=resultants,
        form_storage=form_storage,
        phase_storage=phase_storage,
        storage=form_storage + beta * phase_storage,
        hidden_form_tracking_defect=form_defect,
        hidden_phase_tracking_defect=phase_defect,
        **field,
    )
