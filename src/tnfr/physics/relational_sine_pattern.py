"""Full-node relative pattern observations under the supplied smooth sine law.

The observation model retains one residual error radius per node and an
arbitrary shared offset in each coordinate. Referencing one supplied node
removes those common offsets, but no node, support incidence or dynamical
rate is removed. A rectangular relative-state box conservatively forgets
correlations between the residual errors; it does not authenticate data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from typing import TYPE_CHECKING, Any

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, sin
from .relational_observations import _ordered
from .relational_sine_comparison import (
    _capture_sine_state,
    _sine_work,
    _validate_comparison_labels,
)

if TYPE_CHECKING:
    from .relational_sine_forecast import SineForecast

__all__ = (
    "SineRelativePattern",
    "SineRelativeForecast",
    "bound_relational_sine_pattern",
)


@dataclass(frozen=True)
class SineRelativePattern:
    """Conditional full-pattern enclosures after removing shared offsets.

    Nominal form and phase values are the engine's captured represented
    coordinates. Actual values may differ by an arbitrary common offset and
    separately bounded residual errors. Phase observations require mutually
    consistent real lifts. Capacities and support are supplied exactly.

    The static edge bounds use each endpoint's original residual radius once.
    The relative coordinate box also contains the reference residual, losing
    its correlation between rows. These two uncertainty descriptions have
    different widths and must not be substituted for each other.
    """

    reference_model: RelationalExchangeModel
    reference_node: Any
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    neighbors: tuple[tuple[int, ...], ...]
    degrees: tuple[int, ...]
    capacity: tuple[Q, ...]
    nominal_form: tuple[Q, ...]
    nominal_phase: tuple[Q, ...]
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    relative_form_bounds: tuple[I, ...]
    relative_phase_bounds: tuple[I, ...]
    edge_form_gap_bounds: tuple[I, ...]
    edge_phase_gap_bounds: tuple[I, ...]
    form_gradient_bounds: tuple[I, ...]
    phase_current_bounds: tuple[I, ...]
    pressure_bounds: tuple[I, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    relative_form_rate_bounds: tuple[I, ...]
    relative_phase_rate_bounds: tuple[I, ...]
    reference_form_rate_bounds: I
    reference_phase_rate_bounds: I
    form_storage_bounds: I
    phase_storage_bounds: I
    storage_bounds: I
    continuous_loss_bounds: I
    storage_rate_bounds: I
    balance_residual_bounds: I
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_complete_supplied_node_set_and_simple_connected_unit_support",
        "held_exact_nonnegative_capacities_no_forcing_events_or_clipping",
        "synchronous_common_form_phase_offsets_cancel_with_per_node_residual_bounds",
        "phase_centers_and_errors_refer_to_supplied_consistent_real_lifts",
        "relative_coordinates_preserve_all_nodes_and_subtract_moving_reference_rates",
        "reference_relative_coordinates_and_rates_are_identically_zero",
        "static_edge_bounds_use_original_residuals_without_duplicating_reference_error",
        "relative_rectangular_box_is_a_conservative_outer_bound_losing_correlations",
        "static_storage_and_rates_do_not_bound_every_point_of_the_larger_relative_box",
        "no_hidden_node_elimination_reconstruction_winding_or_phase_unwrapping",
        "no_measurement_authentication_constitutive_selection_or_physical_identity",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self)
        return {
            "schema": "tnfr.relational-sine-relative-pattern.v1",
            "report": _project(self),
        }

    def certify_cycle_recovery(self, *, cycle, winding, radius, phase_turns=None):
        """Assess the original residual-error set against an exact cycle target."""
        from .relational_sine_recovery import certify_sine_cycle_recovery

        return certify_sine_cycle_recovery(
            self, cycle=cycle, winding=winding, radius=radius, phase_turns=phase_turns
        )

    def certify_pattern_recovery(self, *, target_phase_turns, radius, phase_turns=None):
        """Assess the complete residual-error set about an exact critical target."""
        from .relational_sine_recovery import certify_sine_pattern_recovery

        return certify_sine_pattern_recovery(
            self,
            target_phase_turns=target_phase_turns,
            radius=radius,
            phase_turns=phase_turns,
        )

    def certify_sector_capture(self, *, edge_turn_offsets):
        """Assess target-free capture of the original complete observation set."""
        from .relational_sine_recovery import certify_sine_sector_capture

        return certify_sine_sector_capture(self, edge_turn_offsets=edge_turn_offsets)

    def certify_prepared_entry(self, *, scaled_time, edge_turn_offsets):
        """Assess same-law acquisition for the complete residual preparation set."""
        from .relational_sine_entry import certify_sine_prepared_entry

        return certify_sine_prepared_entry(
            self, scaled_time=scaled_time, edge_turn_offsets=edge_turn_offsets
        )

    def bound_slow_phase(self, *, slow_time):
        """Bound each member against its own initially shifted phase reference."""
        from .relational_sine_reduction import bound_sine_slow_phase

        return bound_sine_slow_phase(self, slow_time=slow_time)

    def certify_slow_capture(self, *, slow_time, target_phase_turns):
        """Certify full-state capture through an exact critical phase reference."""
        from .relational_sine_reduction import certify_sine_slow_capture

        return certify_sine_slow_capture(
            self, slow_time=slow_time, target_phase_turns=target_phase_turns
        )

    def forecast(
        self, *, observation_time, end_time, time_step, order=6
    ) -> SineRelativeForecast:
        """Propagate one full gauge representative with the existing solver.

        The initial reference coordinates are zero. Its physical rates are
        not frozen: the entire supplied network evolves, and the reference
        is subtracted only after propagation. The shared solver stores its
        final node's held capacity as an additional coordinate; this layout
        neither makes that node hidden nor eliminates any node or edge.

        Primitive source admission is repeated and the initial relative box
        is rebuilt from nominal coordinates and residual radii. Cached boxes
        cannot replace that declared preparation or authenticate its origin.

        The rectangular initial box encloses all compatible relative states
        and some extra states from lost residual correlations. The source's
        tighter static storage bounds are not a premise for that larger box.
        """
        from ._sine_admission import _admit_sine_source
        from .relational_sine_forecast import bound_sine_flow

        admitted, _ = _admit_sine_source(self)
        anchor = admitted.nodes.index(admitted.reference_node)
        pattern = replace(
            admitted,
            relative_form_bounds=_relative_initial_rows(
                admitted.nominal_form, admitted.form_error_bounds, anchor
            ),
            relative_phase_bounds=_relative_initial_rows(
                admitted.nominal_phase, admitted.phase_error_bounds, anchor
            ),
        )

        full = bound_sine_flow(
            pattern.relative_form_bounds
            + pattern.relative_phase_bounds
            + (I(pattern.capacity[-1]),),
            neighbors=pattern.neighbors,
            visible_capacity=pattern.capacity[:-1],
            model=pattern.reference_model,
            observation_time=observation_time,
            end_time=end_time,
            time_step=time_step,
            order=order,
        )
        size = len(pattern.nodes)
        form = full.endpoint[:size]
        phase = full.endpoint[size : 2 * size]
        return SineRelativeForecast(
            pattern=pattern,
            full_forecast=full,
            relative_form_bounds=_relative_rows(form, anchor),
            relative_phase_bounds=_relative_rows(phase, anchor),
            reference_form_displacement_bounds=form[anchor],
            reference_phase_displacement_bounds=phase[anchor],
        )


@dataclass(frozen=True)
class SineRelativeForecast:
    """Relative endpoint bounds with the complete full-state solver evidence.

    Endpoint coordinates refer to full_forecast.validated_end_time. An
    unavailable full forecast is not silently promoted to its requested
    horizon. Common initial offsets remain unobserved; reference displacement
    is relative to the chosen zero initial gauge representative.
    """

    pattern: SineRelativePattern
    full_forecast: SineForecast
    relative_form_bounds: tuple[I, ...]
    relative_phase_bounds: tuple[I, ...]
    reference_form_displacement_bounds: I
    reference_phase_displacement_bounds: I
    scope: tuple[str, ...] = (
        "all_supplied_nodes_support_and_exact_held_capacities_retained",
        "initial_zero_reference_selects_one_common_shift_representative",
        "reference_evolves_under_the_same_full_sine_law_not_a_clamped_node",
        "shared_full_coordinate_validated_solver_then_relative_endpoint_projection",
        "final_node_capacity_coordinate_is_solver_layout_not_a_hidden_node_premise",
        "full_rectangular_outer_box_preserved_not_replaced_by_nominal_centers",
        "source_static_edge_storage_bounds_are_not_bounds_for_the_larger_solver_box",
        "relative_endpoints_refer_only_to_the_reported_validated_end_time",
        "no_absolute_initial_offsets_raw_sample_authentication_or_physical_prediction",
    )

    @property
    def admitted(self):
        return self.full_forecast.admitted

    @property
    def status(self):
        return self.full_forecast.status

    @property
    def reasons(self):
        return self.full_forecast.reasons

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.pattern)
        return {
            "schema": "tnfr.relational-sine-relative-forecast.v1",
            "report": _project(self),
        }

    def certify_cycle_recovery(self, *, cycle, winding, radius, phase_turns=None):
        """Assess the full endpoint box at its actual validated time only."""
        from .relational_sine_recovery import certify_sine_cycle_recovery

        return certify_sine_cycle_recovery(
            self, cycle=cycle, winding=winding, radius=radius, phase_turns=phase_turns
        )

    def certify_pattern_recovery(self, *, target_phase_turns, radius, phase_turns=None):
        """Assess the full endpoint box against a supplied complete critical shape."""
        from .relational_sine_recovery import certify_sine_pattern_recovery

        return certify_sine_pattern_recovery(
            self,
            target_phase_turns=target_phase_turns,
            radius=radius,
            phase_turns=phase_turns,
        )

    def certify_sector_capture(self, *, edge_turn_offsets):
        """Assess the whole endpoint box at its actual reported validated time."""
        from .relational_sine_recovery import certify_sine_sector_capture

        return certify_sine_sector_capture(self, edge_turn_offsets=edge_turn_offsets)


def _error_radii(values, size, label):
    raw = _ordered(values, label, limit=size + 1)
    if len(raw) != size:
        raise ValueError(f"{label} must contain one radius per node in captured order")
    radii = tuple(exact_or_represented_real(value, label) for value in raw)
    if any(value < 0 for value in radii):
        raise ValueError(f"{label} must be nonnegative")
    return radii


def _difference_bounds(values, radii, left, right):
    center = values[right] - values[left]
    radius = radii[right] + radii[left]
    return I(center - radius, center + radius)


def _relative_rows(values, anchor):
    """Project an enclosure while preserving the exact reference identity."""
    return tuple(
        I(0) if i == anchor else value - values[anchor]
        for i, value in enumerate(values)
    )


def _relative_initial_rows(values, radii, anchor):
    """Rebuild the relative outer box from admitted primitive uncertainty."""
    return tuple(
        I(0) if i == anchor else _difference_bounds(values, radii, anchor, i)
        for i in range(len(values))
    )


def bound_relational_sine_pattern(
    graph, *, reference_node, reference_model, form_error_bounds, phase_error_bounds
) -> SineRelativePattern:
    """Enclose a complete supplied relative pattern with residual uncertainty.

    Error inputs are ordered nonnegative radius sequences matching graph node
    order, not confidence scores or fitted noise. For every node i the actual
    coordinate has the form nominal_i + common_offset + residual_i with the
    declared residual bound. Shared offsets may be unknown; capacities and
    support are exact premises. Every supplied node, including any intermediary,
    must remain in the graph. This is not partial hidden-state inference.

    Supplied real phase lifts are retained. No winding, wrap branch, missing
    node or offset is reconstructed. Existing sine capture enforces signed
    form, aliases, unit support, capacity admission and absent Gamma without
    invoking the native Arg field or changing the graph.
    """
    state = _capture_sine_state(graph, reference_model)
    positions = {node: index for index, node in enumerate(state.nodes)}
    if reference_node not in positions:
        raise ValueError("reference_node must belong to the complete supplied support")
    anchor, size = positions[reference_node], len(state.nodes)
    form_errors = _error_radii(form_error_bounds, size, "form_error_bounds")
    phase_errors = _error_radii(phase_error_bounds, size, "phase_error_bounds")
    relative_form = _relative_initial_rows(state.epi, form_errors, anchor)
    relative_phase = _relative_initial_rows(state.phase, phase_errors, anchor)
    edge_form, edge_phase = [], []
    gradient, currents = [I(0) for _ in state.nodes], [I(0) for _ in state.nodes]
    form_storage = phase_storage = I(0)
    for left, right in state.edges:
        i, j = positions[left], positions[right]
        form_gap = _difference_bounds(state.epi, form_errors, i, j)
        phase_gap = _difference_bounds(state.phase, phase_errors, i, j)
        edge_form.append(form_gap)
        edge_phase.append(phase_gap)
        current = sin(phase_gap)
        gradient[i], gradient[j] = gradient[i] - form_gap, gradient[j] + form_gap
        currents[i], currents[j] = currents[i] + current, currents[j] - current
        form_storage += form_gap**2 / 2
        phase_storage += 1 - cos(phase_gap)
    gradient, currents = tuple(gradient), tuple(currents)
    work = _sine_work(
        reference_model, state.degrees, gradient, state.capacity, currents
    )
    forms, phases = work["form_rates"], work["phase_rates"]
    return SineRelativePattern(
        reference_model=reference_model,
        reference_node=reference_node,
        nodes=state.nodes,
        edges=state.edges,
        neighbors=state.neighbors,
        degrees=state.degrees,
        capacity=state.capacity,
        nominal_form=state.epi,
        nominal_phase=state.phase,
        form_error_bounds=form_errors,
        phase_error_bounds=phase_errors,
        relative_form_bounds=relative_form,
        relative_phase_bounds=relative_phase,
        edge_form_gap_bounds=tuple(edge_form),
        edge_phase_gap_bounds=tuple(edge_phase),
        form_gradient_bounds=gradient,
        phase_current_bounds=currents,
        pressure_bounds=work["pressure"],
        form_rate_bounds=forms,
        phase_rate_bounds=phases,
        relative_form_rate_bounds=_relative_rows(forms, anchor),
        relative_phase_rate_bounds=_relative_rows(phases, anchor),
        reference_form_rate_bounds=forms[anchor],
        reference_phase_rate_bounds=phases[anchor],
        form_storage_bounds=form_storage,
        phase_storage_bounds=phase_storage,
        storage_bounds=form_storage + Q(reference_model.storage_scale) * phase_storage,
        continuous_loss_bounds=I.coerce(work["continuous_loss"]),
        storage_rate_bounds=work["storage_rate"],
        balance_residual_bounds=work["balance_residual"],
    )
