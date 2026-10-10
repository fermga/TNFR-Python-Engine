"""Admit actual acquired hosts to the finite-input observation prior.

The no-argument assessment is reserved for the prospectively declared source
and protocol. It rebuilds five independent child histories from one uniform
recovery theorem and preserves both acquired winding families. No donor,
earlier observation word, copied actual state or response verdict is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._organized_electrical_actuation import _source_boxes
from ._organized_electrical_source import _integer_labels, _same_undirected_edges
from ._organized_finite_observation import (
    _OrganizedFiniteObservationProtocol,
    _bound_organized_finite_observation,
)
from ._organized_outer_winding import (
    _OrganizedOuterResourceBounds,
    _derive_organized_outer_resource_bounds,
)
from ._sine_recovery_resources import (
    _SineRecoveryResources,
    _assess_sine_recovery_resources,
)


def _read_interval(value, label):
    """Normalize primitive interval endpoints before source association."""
    if type(value) is not I:
        raise TypeError(f"{label} requires a shared rational interval")
    return I(_exact(value.lo, label + ".lo"), _exact(value.hi, label + ".hi"))


def _read_capacities(values, label):
    return tuple(
        _exact(value, f"{label}[{index}]") for index, value in enumerate(values)
    )


@dataclass(frozen=True)
class _OrganizedFiniteObservationSource:
    recovery: _SineRecoveryResources
    outer: _OrganizedOuterResourceBounds
    protocol: _OrganizedFiniteObservationProtocol
    nominal_form_upper: Q
    initial_form_upper: Q
    initial_hidden_form_upper: Q
    initial_hidden_phase_upper: Q
    initial_mean_gap_upper: Q
    prerequisite_flags: tuple[tuple[str, bool], ...]
    prerequisite_margins: tuple[tuple[str, Q], ...]
    admitted: bool
    unresolved_conditions: tuple[str, ...]
    independent_child_count: int = 5
    source_families: tuple[str, ...] = ("outer_zero", "outer_one")
    source_stage: str = "actual_end_of_outer_formation_before_first_plateau"
    raw_readout_error_upper_bound: Q = Q(1, 10**14)
    scope: tuple[str, ...] = (
        "one_fresh_uniform_recovery_theorem_for_five_independent_actual_children",
        "actual_acquired_outer_zero_and_outer_one_hosts_with_all_fine_coordinates",
        "orthogonal_child_projection_retains_full_euclidean_hidden_error_bound",
        "first_unforced_plateau_is_part_of_the_new_protocol_not_an_extra_prefix",
        "finite_protocol_is_uniform_over_the_broader_unknown_origin_prior",
        "no_donor_previous_observation_word_or_contact_response_prerequisite",
        "no_hidden_state_or_source_label_consumed_by_the_reading_inverse",
        "one_analytic_assessment_no_trajectory_or_physical_measurement",
    )


def _assess_organized_finite_observation() -> _OrganizedFiniteObservationSource:
    """Assess the committed source inclusion and finite protocol once."""
    radius, noise = Q(1, 10**8), Q(1, 10**14)
    recovery = _assess_sine_recovery_resources(
        form_error_bound=Q(1, 10**6),
        phase_error_bound=Q(1, 10**6),
        quarter_dwell=Q(8 * 10**10),
        endpoint_radius=radius,
        decay_power=80,
    )
    outer = _derive_organized_outer_resource_bounds(recovered_radius=radius)
    protocol = _bound_organized_finite_observation(raw_readout_error=noise)
    forms = _source_boxes(outer.primitive_entry.endpoint_form_bounds, "nominal_form")
    gaps = _source_boxes(
        outer.primitive_entry.capture.phase_edge_gap_bounds, "nominal_gap"
    )
    nominal = max(box.abs_max for box in forms)
    form_error = _exact(
        outer.full_form_error_upper_bound, "full_form_error_upper_bound"
    )
    phase_error = _exact(
        outer.full_phase_error_upper_bound, "full_phase_error_upper_bound"
    )
    if min(form_error, phase_error) < 0:
        raise ValueError("acquired source norm bounds must be nonnegative")
    mean_gap = max(box.abs_max for box in gaps) + 2 * phase_error
    initial_form = nominal + form_error
    geometry = outer.geometry
    gamma, pi = Q(1, 1023) / pi_interval(), pi_interval()
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
    )
    external = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for c in range(5) for j in range(9)
    )
    expected_scalars = (
        ("form_upper_bound", Q(1, 100)),
        ("fine_phase_upper_bound", Q(1, 10**6)),
        ("child_origin_gap_upper_bound", Q(7, 5)),
        ("probe_amplitude", Q(1, 10)),
        ("plateau_duration", Q(1, 10**5)),
        ("pulse_duration", Q(1)),
        ("total_elapsed_scaled_time", 15 * (1 + Q(1, 10**5))),
        ("total_input_scaled_time", Q(15)),
        ("raw_readout_error_upper_bound", noise),
    )
    scalar_protocol_matches = all(
        _exact(getattr(protocol, name), "protocol." + name) == expected
        for name, expected in expected_scalars
    )
    flags = (
        ("five_independent_child_recoveries", recovery.recovery_certified is True),
        (
            "recovered_radius_matches",
            _exact(recovery.endpoint_radius, "recovery_radius") == radius
            and _exact(outer.recovered_child_norm_bound, "outer_radius") == radius,
        ),
        ("primitive_law_and_clock", outer.primitive_law_and_clock_certified is True),
        ("full_fine_comparison", outer.fine_error_certified is True),
        ("child_transit_identity", outer.child_winding_retained_during_transit is True),
        ("outer_one_cell", outer.outer_cell_certified is True),
        ("outer_one_capture", outer.conditional_capture_certified is True),
        ("outer_zero_capture", outer.control_conditional_capture_certified is True),
        ("outer_attachment_budget", outer.attachment_budget_certified is True),
        (
            "precontact_product_support",
            _same_undirected_edges(
                geometry.weighted_edges, internal + external, "host_edges"
            )
            and _integer_labels(geometry.node_order, tuple(range(45)))
            and _integer_labels(geometry.support_degrees, (4,) * 45),
        ),
        (
            "complete_quarter_law",
            _read_capacities(geometry.held_capacities, "capacity") == (Q(1),) * 45
            and _exact(geometry.clock_factor, "clock_factor") == Q(1, 4)
            and geometry.phase_domain == "regular"
            and _read_interval(geometry.gamma_bounds, "gamma_bounds") == gamma
            and _read_interval(geometry.alpha_bounds, "alpha_bounds") == 2 * pi / 9,
        ),
        (
            "fixed_finite_observation_protocol",
            scalar_protocol_matches
            and _integer_labels(protocol.node_order, tuple(range(45)))
            and _integer_labels(protocol.marked_child_order, tuple(range(5)))
            and _integer_labels(protocol.receiver_child_order, (1, 2, 3, 4, 0))
            and _integer_labels(protocol.plateau_command_multipliers, (0, 1, -1))
            and _integer_labels(protocol.pulse_multipliers, (1, -2, 1))
            and _read_capacities(protocol.held_capacities, "protocol.capacity")
            == (Q(1),) * 45
            and _exact(protocol.clock_factor, "protocol.clock_factor") == Q(1, 4)
            and protocol.phase_domain == "regular"
            and _read_interval(protocol.gamma_bounds, "protocol.gamma_bounds") == gamma,
        ),
        (
            "finite_protocol_admitted",
            protocol.admitted is True and protocol.unresolved_conditions == (),
        ),
    )
    history = tuple(
        (label, _exact(value, label))
        for label, value in (
            ("actual_recovery_form", recovery.endpoint_form_margin),
            ("actual_recovery_phase", recovery.endpoint_phase_margin),
            ("outer_one_capture", outer.capture_margin),
            ("outer_zero_capture", outer.control_capture_margin),
            ("outer_attachment_work", outer.attachment_work_margin),
            ("outer_initial_storage", outer.initial_storage_margin),
        )
        if value is not None
    )
    flags += (("complete_numeric_source_margins", len(history) == 6),)
    inclusion = (
        ("outer_one_form_in_prior", Q(1, 100) - initial_form),
        ("outer_zero_form_in_prior", Q(1, 100) - form_error),
        ("orthogonal_hidden_phase_in_prior", Q(1, 10**6) - phase_error),
        ("outer_one_mean_gap_in_prior", Q(7, 5) - mean_gap),
        ("outer_zero_mean_gap_in_prior", Q(7, 5) - 2 * phase_error),
    )
    margins = history + inclusion
    unresolved = tuple(name for name, accepted in flags if not accepted) + tuple(
        name for name, margin in margins if margin <= 0
    )
    return _OrganizedFiniteObservationSource(
        recovery=recovery,
        outer=outer,
        protocol=protocol,
        nominal_form_upper=nominal,
        initial_form_upper=initial_form,
        initial_hidden_form_upper=form_error,
        initial_hidden_phase_upper=phase_error,
        initial_mean_gap_upper=mean_gap,
        prerequisite_flags=flags,
        prerequisite_margins=margins,
        admitted=not unresolved,
        unresolved_conditions=unresolved,
    )
