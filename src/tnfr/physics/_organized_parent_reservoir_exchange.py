"""One autonomous finite-storage extension of the acquired-parent model.

ORGANIZED_PARENT_RESERVOIR_EXCHANGE.md owns the fixed joint law, preparation
and finite response criterion. The auxiliary signed reservoir is supplied
constitutive state, not an identified physical bath or an acquired NFR mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._organized_parent_acquisition import (
    _assess_organized_parent_acquisition,
    _OrganizedParentAcquisition,
)
from ._repeated_organized_composition import (
    _observe_repeated_composition,
    _RepeatedCompositionObservation,
    _vector,
)
from .reversible_eigenmode_reference import _negative_exp_bounds

_CLOCK_FACTOR = Q(1023, 1024)
_RESERVOIR_CENTER = Q(12, 5)
_STORAGE_BUDGET = Q(3)


def _port(value):
    if type(value) is not str or value not in ("connected", "disconnected"):
        raise ValueError("port must be 'connected' or 'disconnected'")
    return value


@dataclass(frozen=True)
class _ParentReservoirObservation:
    """Complete joint rows, signed exchange and storage in all three clocks."""

    network: _RepeatedCompositionObservation
    reservoir_state: Q
    port: str
    coupling_bounds: tuple[I, ...]
    reservoir_decay_bounds: I
    port_output_tau_bounds: I
    network_input_tau_bounds: tuple[I, ...]
    network_input_slow_bounds: tuple[I, ...]
    network_input_original_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    reservoir_tau_rate_bounds: I
    reservoir_slow_rate_bounds: I
    reservoir_original_rate_bounds: I
    reservoir_pressure_tau_bounds: I
    reservoir_storage: Q
    joint_storage_bounds: I
    network_input_power_tau_bounds: I
    network_input_power_slow_bounds: I
    network_input_power_original_bounds: I
    network_storage_tau_rate_bounds: I
    reservoir_storage_tau_rate_bounds: I
    joint_storage_tau_rate_bounds: I
    joint_storage_slow_rate_bounds: I
    joint_storage_original_rate_bounds: I
    network_dissipation_tau: Q
    reservoir_dissipation_tau_bounds: I
    joint_dissipation_tau_bounds: I
    joint_storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    reservoir_capacity: Q = Q(1)
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2"
    pressure_law: str = "p_r_tau=-b^T*L*x-lambda*r; unit capacity in tau"
    scope: tuple[str, ...] = (
        "signed_reservoir_and_all_network_primitives_admitted_before_observation",
        "fixed_matching_network_with_an_added_constitutive_exchange_channel",
        "reservoir_pressure_supplied_from_joint_state_not_reconstructed_from_response",
        "all_three_rows_inputs_and_power_retain_their_clock_transformations",
        "signed_network_work_is_paid_by_reservoir_storage_change_and_reservoir_loss",
        "zero_initial_reservoir_does_not_disconnect_the_coupled_dynamics",
        "joint_balance_does_not_certify_registration_response_or_source_history",
        "no_phase_coordinate_for_auxiliary_state_no_graph_mutation_or_trajectory",
    )


def _observe_parent_reservoir_exchange(
    *, form, phase_offsets, reservoir_state, port
) -> _ParentReservoirObservation:
    """Observe the fixed autonomous joint law or its disconnected control."""
    selected = _port(port)
    values = _vector(form, "form", 36)
    offsets = _vector(phase_offsets, "phase_offsets", 36)
    reservoir = _exact(reservoir_state, "reservoir_state")
    return _observe_admitted_parent_reservoir_exchange(
        form=values,
        phase_offsets=offsets,
        reservoir_state=reservoir,
        port=selected,
        leakage="original",
    )


def _observe_admitted_parent_reservoir_exchange(
    *, form, phase_offsets, reservoir_state, port, leakage
):
    """Joint algebra for caller-admitted primitives and one declared leakage.

    The public-to-this-module adapters own scalar, vector and law admission.
    This shared kernel installs no law and certifies no trapping or source.
    """
    values, offsets, reservoir, selected = form, phase_offsets, reservoir_state, port
    network = _observe_repeated_composition(
        interface="matching", form=values, phase_offsets=offsets
    )
    gamma = network.geometry.gamma_bounds
    decay = gamma**2 / 2 if leakage == "original" else I(0)
    coupling = tuple(
        gamma * Q(2 * (node // 9 % 2) - 1, 12) if selected == "connected" else I(0)
        for node in range(36)
    )
    output = sum((b * q for b, q in zip(coupling, network.form_gradient)), I(0))
    forcing = tuple(b * reservoir for b in coupling)
    form_rate = tuple(
        rate + added for rate, added in zip(network.form_rate_bounds, forcing)
    )
    phase_rate = network.phase_rate_bounds
    reservoir_rate = -output - decay * reservoir
    power = output * reservoir
    reservoir_storage = reservoir**2 / 2
    network_loss = network.dissipation
    reservoir_loss = decay * reservoir**2
    joint_loss = network_loss + reservoir_loss
    joint_rate = -joint_loss
    # Independent contraction of the complete rows retains both exchange signs.
    direct_rate = (
        sum(
            (
                q * dx - current * dt
                for q, current, dx, dt in zip(
                    network.form_gradient,
                    network.phase_current_bounds,
                    form_rate,
                    phase_rate,
                )
            ),
            I(0),
        )
        + reservoir * reservoir_rate
    )
    slow_factor = 2 / gamma**2
    return _ParentReservoirObservation(
        network=network,
        reservoir_state=reservoir,
        port=selected,
        coupling_bounds=coupling,
        reservoir_decay_bounds=decay,
        port_output_tau_bounds=output,
        network_input_tau_bounds=forcing,
        network_input_slow_bounds=tuple(slow_factor * value for value in forcing),
        network_input_original_bounds=tuple(_CLOCK_FACTOR * value for value in forcing),
        form_tau_rate_bounds=form_rate,
        form_slow_rate_bounds=tuple(slow_factor * value for value in form_rate),
        form_original_rate_bounds=tuple(_CLOCK_FACTOR * value for value in form_rate),
        phase_tau_rate_bounds=phase_rate,
        phase_slow_rate_bounds=tuple(slow_factor * value for value in phase_rate),
        phase_original_rate_bounds=tuple(_CLOCK_FACTOR * value for value in phase_rate),
        reservoir_tau_rate_bounds=reservoir_rate,
        reservoir_slow_rate_bounds=slow_factor * reservoir_rate,
        reservoir_original_rate_bounds=_CLOCK_FACTOR * reservoir_rate,
        reservoir_pressure_tau_bounds=reservoir_rate,
        reservoir_storage=reservoir_storage,
        joint_storage_bounds=network.storage_bounds + reservoir_storage,
        network_input_power_tau_bounds=power,
        network_input_power_slow_bounds=slow_factor * power,
        network_input_power_original_bounds=_CLOCK_FACTOR * power,
        network_storage_tau_rate_bounds=-network_loss + power,
        reservoir_storage_tau_rate_bounds=-power - reservoir_loss,
        joint_storage_tau_rate_bounds=joint_rate,
        joint_storage_slow_rate_bounds=slow_factor * joint_rate,
        joint_storage_original_rate_bounds=_CLOCK_FACTOR * joint_rate,
        network_dissipation_tau=network_loss,
        reservoir_dissipation_tau_bounds=reservoir_loss,
        joint_dissipation_tau_bounds=joint_loss,
        joint_storage_rate_from_rows_bounds=direct_rate,
        balance_residual_bounds=direct_rate - joint_rate,
    )


@dataclass(frozen=True)
class _OrganizedParentReservoirExchange:
    """Finite response bounds and a separately admitted fixed-criterion result.

    Parent gap bounds are absolute lifted same-slot gaps. The nominal envelope
    fields enclose analytic supersolutions, not a sampled nominal trajectory.
    Registration at the deadline does not assert earlier loss or restoration.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    reservoir_error_bound: Q
    acquisition: _OrganizedParentAcquisition
    reservoir_initial_bounds: I
    reservoir_initial_storage_upper_bound: Q
    network_input_work_upper_bound: Q
    initial_joint_error_energy_upper_bound: Q
    readout_joint_form_error_upper_bound: Q
    readout_joint_phase_error_upper_bound: Q
    readout_joint_reservoir_error_upper_bound: Q
    whole_joint_form_error_upper_bound: Q
    whole_joint_phase_error_upper_bound: Q
    whole_joint_reservoir_error_upper_bound: Q
    nominal_readout_envelope_bounds: I
    nominal_deadline_envelope_bounds: I
    connected_parent_gap_at_readout_upper_bound: Q
    connected_parent_gap_at_deadline_upper_bound: Q
    disconnected_parent_gap_upper_bound: Q
    child_positive_gap_margin: Q
    child_acute_gap_margin: Q
    registration_retention_margin: Q
    deadline_registration_margin: Q
    disconnected_registration_margin: Q
    storage_budget_margin: Q
    elapsed_scaled_readout_time_bounds: I
    elapsed_scaled_deadline_time_bounds: I
    elapsed_original_readout_time_bounds: I
    elapsed_original_deadline_time_bounds: I
    sufficient_margins: tuple[tuple[str, Q], ...]
    network_source_certified: bool
    reservoir_preparation_certified: bool
    source_certified: bool
    response_domain_certified: bool
    finite_work_certified: bool
    child_identities_retained: bool
    disconnected_registration_retained: bool
    registered_at_deadline: bool
    disruption_ruled_out: bool
    conditional_obstruction_certified: bool
    obstruction_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    disruption_certified: bool = False
    protocol_satisfied: bool = False
    source_slow_time: Q = Q(4)
    readout_elapsed_slow_time: Q = Q(1, 2)
    deadline_elapsed_slow_time: Q = Q(8)
    readout_slow_time: Q = Q(9, 2)
    deadline_slow_time: Q = Q(12)
    reservoir_initial_center: Q = _RESERVOIR_CENTER
    reservoir_storage_budget: Q = _STORAGE_BUDGET
    reservoir_capacity: Q = Q(1)
    parent_identity_gap: Q = Q(1, 4)
    protocol_revision: str = "9a0071de"
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2; h=s-4"
    scope: tuple[str, ...] = (
        "all_nine_primitive_allowances_admitted_before_fresh_original_acquisition",
        "auxiliary_signed_form_pressure_storage_and_coupling_are_supplied_premises",
        "each_actual_reservoir_initial_value_indexes_its_own_nominal_reference",
        "reservoir_uncertainty_retained_in_the_entire_response_domain",
        "joint_error_energy_includes_reservoir_feedback_not_an_identical_input_assumption",
        "mean_zero_form_error_and_joint_storage_exchange_cancellation_are_required",
        "complete_joint_loss_pays_network_net_work_from_finite_initial_reservoir_storage",
        "network_work_upper_bound_is_not_positive_only_supply_or_actuator_bandwidth",
        "disconnected_control_changes_b_to_zero_not_initial_reservoir_to_zero",
        "child_winding_retained_in_both_branches_through_the_fixed_deadline",
        "certified_negative_result_concerns_only_the_fixed_half_unit_disruption_criterion",
        "registration_at_deadline_does_not_establish_a_preceding_disruption_or_restoration",
        "intermediate_parent_registration_and_other_reservoir_laws_are_unclassified",
        "no_trajectory_source_refinement_coefficient_scan_or_physical_reservoir_identification",
    )


def _assess_organized_parent_reservoir_exchange(
    *, form_error_bounds, phase_error_bounds, reservoir_error_bound
) -> _OrganizedParentReservoirExchange:
    """Assess the frozen reservoir mechanism without moving its response gate.

    Network acquisition is rebuilt from primitive allowances. The independent
    reservoir interval remains a parameter of the nominal proof, while the
    full network/reservoir separation uses a new joint-energy inequality.
    No preceding actuation or restoring-response verdict is consumed.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 4)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 4)
    radius = _exact(reservoir_error_bound, "reservoir_error_bound")
    if min(*rx, *rt, radius) < 0:
        raise ValueError("all preparation error allowances must be nonnegative")
    acquisition = _assess_organized_parent_acquisition(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    geometry = acquisition.geometry
    gamma, eta_upper = geometry.gamma_bounds, geometry.gamma_bounds.hi**2
    initial_form = acquisition.full_form_error_upper_bound
    initial_phase = acquisition.full_phase_error_upper_bound
    # W=xi.L.xi/2+rho^2/2+4||zeta||^2; W_h<=8W. Reservoir differences
    # start at zero for each actual r0, but subsequent feedback is retained.
    initial_energy = 4 * (initial_form**2 + initial_phase**2)
    readout_phase = (initial_form + initial_phase) * 3**2
    whole_phase = (initial_form + initial_phase) * 3**32
    reservoir_lower = _RESERVOIR_CENTER - radius
    reservoir_upper = _RESERVOIR_CENTER + radius
    storage = reservoir_upper**2 / 2
    storage_margin = _STORAGE_BUDGET - storage
    initial_cap, sector_cap, restoring_slope = Q(1, 64), Q(17, 64), Q(9, 10)
    readout_envelope = initial_cap + (1 - I(*_negative_exp_bounds(Q(2)))) / 4
    deadline_envelope = I(*_negative_exp_bounds(Q(8, 5))) / 3
    readout_gap = readout_envelope.hi + 2 * readout_phase
    deadline_gap = deadline_envelope.hi + 2 * whole_phase
    control_gap = initial_cap + 2 * whole_phase
    child_positive = geometry.alpha_bounds.lo - 2 * whole_phase
    child_acute = pi_interval().lo / 2 - geometry.alpha_bounds.hi - 2 * whole_phase
    registration_margin = acquisition.parent_identity_gap - readout_gap
    deadline_margin = acquisition.parent_identity_gap - deadline_gap
    control_margin = acquisition.parent_identity_gap - control_gap
    response_margins = (
        (
            "acquired_nominal_inner_positive",
            acquisition.nominal_inner_gap_at_entry_bounds.lo,
        ),
        (
            "acquired_nominal_inner_gap",
            initial_cap - acquisition.nominal_inner_gap_at_entry_bounds.hi,
        ),
        ("cooperative_joint_rows", 1 - 16 * eta_upper),
        ("retained_initial_velocity_cone", 3 - 2 * eta_upper),
        ("restoring_sine_sector", 1 - sector_cap**2 / 6 - restoring_slope),
        (
            "bounded_supersolution_curvature",
            Q(1, 3) - restoring_slope / 4 - 4 * eta_upper,
        ),
        (
            "bounded_supersolution_initial_reservoir",
            restoring_slope * initial_cap
            + restoring_slope / 4
            - (reservoir_upper - 2) / 3,
        ),
        ("decaying_supersolution_initial_phase", Q(1, 3) - initial_cap),
        (
            "decaying_supersolution_initial_reservoir",
            Q(5, 2) - reservoir_upper - 6 * initial_cap,
        ),
        ("decaying_supersolution_residual", Q(1, 5)),
    )
    network_conditions = (
        ("original_children_not_certified", acquisition.sources_certified),
        ("original_parent_entry_not_certified", acquisition.entry_certified),
        (
            "original_child_identities_not_certified",
            acquisition.child_identities_retained,
        ),
    )
    network_ok = all(passed for _, passed in network_conditions)
    reservoir_ok = reservoir_lower >= 0 and storage_margin >= 0
    source_ok = network_ok and reservoir_ok
    response_ok = all(value > 0 for _, value in response_margins)
    children_ok = min(child_positive, child_acute) > 0
    non_disruption = response_ok and registration_margin > 0
    control_ok = control_margin > 0
    conditional = reservoir_ok and non_disruption and children_ok and control_ok
    obstruction = network_ok and conditional
    margins = response_margins + (
        ("reservoir_nonnegative_preparation", reservoir_lower),
        ("finite_reservoir_storage_budget", storage_margin),
        ("retained_child_positive_gap", child_positive),
        ("retained_child_acute_gap", child_acute),
        ("connected_registration_retained_at_readout", registration_margin),
        ("connected_registration_at_deadline", deadline_margin),
        ("disconnected_registration_retained", control_margin),
    )
    decisions = network_conditions + (
        ("reservoir_preparation_not_certified", reservoir_ok),
        ("connected_response_domain_not_certified", response_ok),
        ("child_identities_not_certified", children_ok),
        ("connected_non_disruption_not_certified", non_disruption),
        (
            "connected_deadline_registration_not_certified",
            response_ok and deadline_margin > 0,
        ),
        ("disconnected_registration_not_certified", control_ok),
    )
    tau_readout, tau_deadline = 1 / gamma**2, 16 / gamma**2
    return _OrganizedParentReservoirExchange(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        reservoir_error_bound=radius,
        acquisition=acquisition,
        reservoir_initial_bounds=I(reservoir_lower, reservoir_upper),
        reservoir_initial_storage_upper_bound=storage,
        network_input_work_upper_bound=storage,
        initial_joint_error_energy_upper_bound=initial_energy,
        readout_joint_form_error_upper_bound=6 * readout_phase,
        readout_joint_phase_error_upper_bound=readout_phase,
        readout_joint_reservoir_error_upper_bound=4 * readout_phase,
        whole_joint_form_error_upper_bound=6 * whole_phase,
        whole_joint_phase_error_upper_bound=whole_phase,
        whole_joint_reservoir_error_upper_bound=4 * whole_phase,
        nominal_readout_envelope_bounds=readout_envelope,
        nominal_deadline_envelope_bounds=deadline_envelope,
        connected_parent_gap_at_readout_upper_bound=readout_gap,
        connected_parent_gap_at_deadline_upper_bound=deadline_gap,
        disconnected_parent_gap_upper_bound=control_gap,
        child_positive_gap_margin=child_positive,
        child_acute_gap_margin=child_acute,
        registration_retention_margin=registration_margin,
        deadline_registration_margin=deadline_margin,
        disconnected_registration_margin=control_margin,
        storage_budget_margin=storage_margin,
        elapsed_scaled_readout_time_bounds=tau_readout,
        elapsed_scaled_deadline_time_bounds=tau_deadline,
        elapsed_original_readout_time_bounds=tau_readout / _CLOCK_FACTOR,
        elapsed_original_deadline_time_bounds=tau_deadline / _CLOCK_FACTOR,
        sufficient_margins=margins,
        network_source_certified=network_ok,
        reservoir_preparation_certified=reservoir_ok,
        source_certified=source_ok,
        response_domain_certified=response_ok,
        finite_work_certified=source_ok,
        child_identities_retained=source_ok and children_ok,
        disconnected_registration_retained=source_ok and control_ok,
        registered_at_deadline=source_ok
        and response_ok
        and children_ok
        and deadline_margin > 0,
        disruption_ruled_out=source_ok and non_disruption,
        conditional_obstruction_certified=conditional,
        obstruction_certified=obstruction,
        status=(
            "certified_reservoir_disruption_obstruction"
            if obstruction
            else "unavailable"
        ),
        unresolved_conditions=tuple(name for name, passed in decisions if not passed),
    )
