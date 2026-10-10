"""Source-inclusive finite collective electrical comparison.

The prospective declaration and proof are ORGANIZED_ELECTRICAL_SOURCE.md.
This owner transports hidden deviations through the entire acquired history;
it installs no electrical law, sensor, switching event or external controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import pi_interval
from ._organized_joint_resources import (
    _OrganizedJointResources,
    _assess_organized_joint_resources,
)
from ._organized_slot_registration import _derive_organized_slot_registration


_HISTORY_MARGINS = frozenset(
    (
        "outer_one_form",
        "outer_zero_form",
        "reprojected_fine_phase",
        "outer_one_origin_gap",
        "outer_zero_origin_gap",
        "internal_positive_gap",
        "internal_acute_gap",
        "outer_acute_gap",
        "absolute_event_work",
        "cumulative_absolute_event_work",
        "finite_nonlinear_comparison_domain",
        "positive_cosine_chart",
        "complete_cycle_winding_resolution",
        "host_internal_positive",
        "host_internal_acute",
        "donor_internal_positive",
        "donor_internal_acute",
        "outer_acute",
        "outer_capture",
        "outer_control_capture",
        "outer_attachment_work",
        "outer_initial_storage",
        "contact_attachment_work",
        "outer_one_joined_storage",
        "outer_zero_joined_storage",
    )
)
_OLD_RESPONSE_MARGINS = frozenset(
    (
        "outer_one_all_origin_response",
        "outer_zero_witness_response",
        "fixed_origin_response_contrast",
    )
)


def _integer_labels(values, expected):
    """Compare discrete labels without identifying Booleans with integers."""
    return tuple(values) == expected and all(type(value) is int for value in values)


def _same_undirected_edges(edges, expected, name):
    """Admit conductances before comparing the fixed undirected incidence."""
    normalized = []
    for index, (a, b, weight) in enumerate(edges):
        if type(a) is not int or type(b) is not int:
            raise ValueError(f"{name}[{index}] requires ordinary integer nodes")
        normalized.append(
            (min(a, b), max(a, b), _exact(weight, f"{name}[{index}].weight"))
        )
    canonical = frozenset((min(a, b), max(a, b), w) for a, b, w in expected)
    return len(normalized) == len(expected) and frozenset(normalized) == canonical


@dataclass(frozen=True)
class _HiddenDeviationBounds:
    initial_form: Q
    initial_phase: Q
    gamma_upper: Q
    laplacian_upper: Q
    duration: Q
    coupling: Q
    domain_margin: Q
    form_upper: Q
    phase_upper: Q


def _transport_hidden_deviations(
    *, initial_form, initial_phase, gamma_upper, laplacian_upper, duration
):
    """Bound Euclidean hidden norms, conditional on the projected model.

    This exact rational majorant uses heat contraction and two integral
    inequalities. It is not a trajectory solver or a generic phase theorem.
    An unsupported comparison domain raises rather than extending the bound.
    """
    names = (
        "initial_form",
        "initial_phase",
        "gamma_upper",
        "laplacian_upper",
        "duration",
    )
    values = tuple(
        _exact(value, name)
        for name, value in zip(
            names, (initial_form, initial_phase, gamma_upper, laplacian_upper, duration)
        )
    )
    if min(values) < 0:
        raise ValueError("hidden-deviation bounds require nonnegative primitives")
    a, z, g, lam, time = values
    coupling = g * lam * time / 4
    margin = 1 - coupling**2
    if margin <= 0:
        raise ValueError("hidden-deviation comparison requires gamma*lambda*time/4 < 1")
    return _HiddenDeviationBounds(
        *values,
        coupling,
        margin,
        (a + coupling * z) / margin,
        (z + coupling * a) / margin,
    )


@dataclass(frozen=True)
class _OrganizedElectricalSource:
    source: _OrganizedJointResources
    observation_hidden: _HiddenDeviationBounds
    contact_hidden: _HiddenDeviationBounds
    prerequisite_flags: tuple[tuple[str, bool], ...]
    prerequisite_margins: tuple[tuple[str, Q], ...]
    collar_margin: Q
    closure_l1_upper: Q
    regional_closure_upper: Q
    comparison_domain_margin: Q
    mean_form_error_upper: Q
    mean_phase_error_upper: Q
    form_resolution_margin: Q
    storage_remainder_lower: Q
    storage_remainder_upper: Q
    hidden_loss_rate_upper: Q
    integrated_hidden_loss_upper: Q
    collective_hidden_work_rate_upper: Q
    integrated_collective_hidden_work_upper: Q
    admitted: bool
    unresolved_conditions: tuple[str, ...]
    contact_horizon: Q = Q(1, 1000)
    source_families: tuple[str, ...] = ("outer_zero", "outer_one")
    slot_offsets: tuple[int, ...] = (0,) * 5
    node_order: tuple[int, ...] = tuple(range(54))
    phase_template: str = "2*pi*(j-4)/9"
    clock: str = "tau=(1023/1024)*t; physical tau=tau0+b*t_lab remains uncalibrated"
    scope: tuple[str, ...] = (
        "same_actual_sources_after_the_entire_sequential_observation_word",
        "circular_collar_mean_available_not_inferred_from_old_mean_form_records",
        "ideal_wheel_initialized_at_actual_observed_means_not_nominal_carrier",
        "uniform_finite_L2_state_error_and_endogenous_zero_sum_current_defect",
        "signed_hidden_storage_and_separate_hidden_loss_and_work",
        "fixed_postcontact_support_only_no_electrical_switching_certificate",
        "supplied_Josephson_reference_and_free_scales_not_derived_physical_constants",
        "one_analytic_assessment_no_trajectory_search_or_frozen_producer_replay",
    )


def _assess_organized_electrical_source() -> _OrganizedElectricalSource:
    """Assess the one declared source box and original same-slot contact."""
    source = _assess_organized_joint_resources(
        form_error_bounds=(Q(1, 10**6),) * 6,
        phase_error_bounds=(Q(1, 10**6),) * 6,
        quarter_dwell=Q(8 * 10**10),
        recovered_radius=Q(1, 10**8),
        decay_power=80,
        sequential_readout_error=Q(1, 10**14),
        contact_readout_error=Q(1, 10**9),
    )
    outer, observation = source.outer_resources, source.observation
    geometry = _derive_organized_slot_registration(slot_offsets=(0,) * 5)
    host = outer.geometry
    pi = pi_interval()
    gamma, alpha = Q(1, 1023) / pi, 2 * pi / 9
    g = _exact(geometry.gamma_bounds.hi, "gamma_upper")
    h = _exact(source.contact_horizon, "contact_horizon")
    q = _exact(source.recovered_radius, "recovered_radius")
    noise = _exact(source.contact_readout_error, "contact_readout_error")
    form_bound = _exact(source.contact_form_upper_bound, "contact_form_upper_bound")
    observation_duration = _exact(
        observation.total_elapsed_scaled_time, "total_elapsed_scaled_time"
    )
    if min(g, h, q, noise, form_bound, observation_duration) < 0:
        raise ValueError("source comparison bounds require nonnegative scalars")
    contact_capacities = tuple(
        _exact(value, f"contact_capacity[{index}]")
        for index, value in enumerate(geometry.held_capacities)
    )
    host_capacities = tuple(
        _exact(value, f"host_capacity[{index}]")
        for index, value in enumerate(host.held_capacities)
    )
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(6) for j in range(9)
    )
    outer_edges = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for c in range(5) for j in range(9)
    )
    contact_edges = tuple((9 * c + j, 45 + j, Q(1)) for c in range(5) for j in range(9))
    internal_cycles = tuple(tuple(9 * c + j for j in range(9)) for c in range(6))
    outer_cycles = tuple(tuple(9 * c + j for c in range(5)) for j in range(9))

    def marked_cycles(actual, expected):
        return len(actual) == len(expected) and all(
            _integer_labels(row, wanted) for row, wanted in zip(actual, expected)
        )

    flags = (
        (
            "six_actual_recoveries",
            len(source.recoveries) == 6
            and all(s.recovery_certified is True for s in source.recoveries),
        ),
        ("outer_one_capture", outer.conditional_capture_certified is True),
        ("outer_zero_capture", outer.control_conditional_capture_certified is True),
        ("outer_attachment_budget", outer.attachment_budget_certified is True),
        ("sequential_observation", observation.admitted is True),
        (
            "same_slot_product_support",
            _same_undirected_edges(
                geometry.weighted_edges,
                internal + outer_edges + contact_edges,
                "contact_edges",
            )
            and _integer_labels(geometry.node_order, tuple(range(54)))
            and _integer_labels(geometry.slot_offsets, (0,) * 5)
            and _integer_labels(geometry.support_degrees, (5,) * 45 + (7,) * 9)
            and marked_cycles(geometry.internal_cycles, internal_cycles)
            and marked_cycles(geometry.outer_cycles, outer_cycles),
        ),
        (
            "complete_observation_support",
            _same_undirected_edges(
                host.weighted_edges, internal[:45] + outer_edges, "host_edges"
            )
            and _integer_labels(host.node_order, tuple(range(45)))
            and _integer_labels(host.support_degrees, (4,) * 45)
            and marked_cycles(host.internal_cycles, internal_cycles[:5])
            and marked_cycles(host.outer_cycles, outer_cycles),
        ),
        (
            "complete_contact_law",
            contact_capacities == (Q(1),) * 54
            and _exact(geometry.clock_factor, "contact_clock_factor") == Q(1, 4)
            and geometry.phase_domain == "regular"
            and geometry.gamma_bounds == gamma
            and geometry.alpha_bounds == alpha
            and geometry.external_phase_origin_bounds == pi / 2
            and h == Q(1, 1000),
        ),
        (
            "complete_observation_law",
            host_capacities == (Q(1),) * 45
            and _exact(host.clock_factor, "host_clock_factor") == Q(1, 4)
            and host.phase_domain == "regular"
            and host.gamma_bounds == gamma
            and host.alpha_bounds == alpha
            and observation.gamma_bounds == gamma,
        ),
        (
            "complete_observation_word",
            _integer_labels(observation.marked_child_order, tuple(range(5)))
            and _integer_labels(observation.receiver_child_order, (1, 2, 3, 4, 0))
            and _integer_labels(observation.event_multipliers, (1, -2, 1))
            and _integer_labels(observation.window_command_multipliers, (0, 1, -1, 0))
            and _exact(observation.probe_amplitude, "probe_amplitude") == Q(1, 10)
            and _exact(
                observation.elapsed_window_scaled_time, "elapsed_window_scaled_time"
            )
            == Q(1, 10**5)
            and _exact(
                observation.elapsed_block_scaled_time, "elapsed_block_scaled_time"
            )
            == Q(1, 25000)
            and observation_duration == Q(1, 5000),
        ),
    )
    margins = tuple(
        (name, _exact(value, name))
        for name, value in source.sufficient_margins
        if name not in _OLD_RESPONSE_MARGINS
    )
    names = tuple(name for name, _ in margins)
    flags += (
        (
            "complete_history_margin_inventory",
            len(names) == len(set(names)) and _HISTORY_MARGINS.issubset(names),
        ),
    )
    observed = _transport_hidden_deviations(
        initial_form=outer.full_form_error_upper_bound,
        initial_phase=outer.full_phase_error_upper_bound,
        gamma_upper=g,
        laplacian_upper=8,
        duration=observation_duration,
    )
    contact = _transport_hidden_deviations(
        initial_form=observed.form_upper + q,
        initial_phase=observed.phase_upper + q,
        gamma_upper=g,
        laplacian_upper=10,
        duration=h,
    )
    a, z = contact.form_upper, contact.phase_upper
    collar = pi.lo / 2 - 2 * z
    residual = g * z**2 / 6
    coupling = Q(3, 2) * g * h
    comparison_margin = 1 - coupling**2
    if comparison_margin <= 0:
        raise ValueError("mean comparison requires 3*gamma*h/2 < 1")
    form_error = h * residual / comparison_margin
    phase_error = coupling * form_error
    resolution = noise - form_error
    work = 10 * form_bound * residual
    decisions = (
        ("circular_collar", collar),
        ("mean_comparison_domain", comparison_margin),
        ("form_error_below_original_raw_readout_budget", resolution),
    )
    unresolved = tuple(name for name, ok in flags if not ok) + tuple(
        name for name, value in margins + decisions if value <= 0
    )
    return _OrganizedElectricalSource(
        source=source,
        observation_hidden=observed,
        contact_hidden=contact,
        prerequisite_flags=flags,
        prerequisite_margins=margins,
        collar_margin=collar,
        closure_l1_upper=residual,
        regional_closure_upper=residual / 2,
        comparison_domain_margin=comparison_margin,
        mean_form_error_upper=form_error,
        mean_phase_error_upper=phase_error,
        form_resolution_margin=resolution,
        storage_remainder_lower=-5 * z**2,
        storage_remainder_upper=5 * (a**2 + z**2),
        hidden_loss_rate_upper=25 * a**2,
        integrated_hidden_loss_upper=25 * h * a**2,
        collective_hidden_work_rate_upper=work,
        integrated_collective_hidden_work_upper=h * work,
        admitted=not unresolved,
        unresolved_conditions=unresolved,
    )
