"""Finite-duration contact evidence for the fixed recovered C5 preparations.

The supplied bridge is held for a declared structural-clock duration. A
continuous growth comparison bounds accumulated form response against the
counterfactual recovery without contact. A separate fixed-protocol reader
bounds the receiver's persistent form mean after a supplied cut and isolated
recovery. These readers neither execute events nor replace the ODE by Euler steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real, exp_unit_bounds
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin
from .relational_cycle_memory import (
    RelationalMemoryReadoutCertificate,
    certify_relational_cycle_memory_readout,
)

__all__ = (
    "RelationalMemoryContactCase",
    "RelationalMemoryContactCertificate",
    "certify_relational_memory_contact",
    "RelationalMemoryRetentionCase",
    "RelationalMemoryRetentionCertificate",
    "certify_relational_memory_retention",
)


@dataclass(frozen=True)
class RelationalMemoryContactCase:
    """One preparation's accumulated responses and separate contact work.

    Form-change bounds are differences between the final and initial port
    forms. Contact-induced changes subtract the separate no-contact solution
    from the contacted solution at the same horizon. They do not subtract a
    constant-rate extrapolation of the original state. The disturbance bound
    applies to every form and lifted phase coordinate throughout the contact.
    Winding preservation is conditional on the supplied bridge and native law.
    """

    form_sign: int
    initial_state_radius: Q
    whole_time_state_radius_upper_bound: Q
    state_change_upper_bound: Q
    accumulated_form_remainder_bound: Q
    left_contact_form_change_bounds: tuple[Q, Q]
    right_contact_form_change_bounds: tuple[Q, Q]
    left_no_contact_form_change_bounds: tuple[Q, Q]
    right_no_contact_form_change_bounds: tuple[Q, Q]
    left_contact_induced_form_change_bounds: tuple[Q, Q]
    right_contact_induced_form_change_bounds: tuple[Q, Q]
    internal_acute_margin_lower_bound: Q
    bridge_acute_margin_lower_bound: Q
    contact_storage_bounds: tuple[Q, Q]
    continuous_loss_upper_bound: Q
    winding_preserved: bool
    contact_sign_certified: bool
    contact_induced_sign_certified: bool
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RelationalMemoryContactCertificate:
    """Conditional finite contact, with an independently bounded control.

    The initial state is the fixed readout at structural time 165120, using
    the unchanged amplitude-2**-20 memory preparation. One matching-port
    bridge is supplied and held for ``duration``; the alternative has no
    contact. Both use unit held capacities, beta=1 and e=w=1/2 without forcing.

    ``required_event_work_upper_bound`` bounds the positive storage required
    by insertion. It neither supplies a reservoir nor claims that previous
    continuous loss pays this cost. ``continuous_loss_upper_bound`` is a
    separate per-case bound during contact. No removal event is assumed.

    ``admitted`` requires signed accumulated responses, signed differences
    from the no-contact control, and preservation of both ring windings.
    An inconclusive sign check retains the nonempty continuous enclosures.
    """

    readout: RelationalMemoryReadoutCertificate
    duration: Q
    duration_limit: Q
    end_time: Q
    neighborhood_radius: Q
    field_growth_constant: Q
    form_rate_lipschitz_constant: Q
    exponential_bounds: tuple[Q, Q]
    proof_checks: tuple[tuple[str, bool], ...]
    required_event_work_upper_bound: Q
    cases: tuple[RelationalMemoryContactCase, RelationalMemoryContactCase]
    contact_signs_separated: bool
    contact_induced_signs_separated: bool
    windings_preserved: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = INTERVAL_METHOD + ";exact_rational_exponential_series"
    scope: tuple[str, ...] = (
        "fixed_readout_at_structural_time_165120_for_amplitude_two_to_minus_twenty",
        "supplied_single_matching_port_bridge_held_during_contact_without_forcing",
        "native_unit_capacity_beta_one_equal_half_weights_on_two_simple_unit_C5_rings",
        "whole_time_continuous_growth_and_integrated_form_rate_remainder",
        "no_contact_control_continues_its_own_unforced_recovery",
        "all_node_form_and_lifted_phase_disturbance_and_both_ring_windings_bounded",
        "event_insertion_storage_is_separate_from_continuous_dissipation",
        "no_live_graph_event_execution_ODE_simulation_or_Euler_trajectory_claim",
        "no_event_selector_reservoir_physical_clock_or_indefinite_postcontact_claim",
    )

    @property
    def admitted(self) -> bool:
        """Whether accumulated contact and control differences separate."""
        return self.status == "admitted"


def certify_relational_memory_contact(
    *, duration=Q(1, 4096)
) -> RelationalMemoryContactCertificate:
    """Enclose a finite supplied contact without executing a trajectory.

    Require exact/represented ``0 < duration <= 1/3``. Around the aligned
    twist pair, an admitted radius-1/100 neighborhood has edge cosines above
    7/25. The native field satisfies ``||F(z)||_inf <= 3*||z||_inf`` there;
    its form rows are 3-Lipschitz in that norm. The smooth phase mobility is
    ``atan_ratio(Y/X)/(2*pi*X)``, including its continuous value at Y=0.

    For initial radius rho and duration h, Gronwall gives the whole-time
    bound ``rho*exp(3*h)`` and displacement ``rho*(exp(3*h)-1)``. Integrating
    the form-rate deviation gives ``rho*(exp(3*h)-1-3*h)``. Thus scaling the
    initial contact rate by h is accompanied by an explicit continuous
    remainder; it is not an Euler-step interpretation. The no-contact left
    rate remains bounded by ``4*R/3`` throughout its separate recovery, with
    untouched right rate zero. All scaling and remainders use exact rational
    arithmetic after the shared transcendental enclosures.
    """
    h = exact_or_represented_real(duration, "duration")
    limit, neighborhood, growth = Q(1, 3), Q(1, 100), Q(3)
    if not 0 < h <= limit:
        raise ValueError("duration must satisfy 0 < duration <= 1/3")
    readout = certify_relational_cycle_memory_readout(decay_blocks=40)
    pi = pi_interval()
    edge_cosine_lower = cos(2 * pi / 5 + 2 * neighborhood).lo
    rate_factor = 1 + 1 / (pi.lo * edge_cosine_lower)
    checks = (
        ("fixed_finite_readout_admitted", readout.admitted),
        ("pi_above_three", pi.lo > 3),
        ("neighborhood_edge_cosine_above_7_over25", edge_cosine_lower > Q(7, 25)),
        ("field_growth_and_form_row_lipschitz_below_three", rate_factor < growth),
        ("neighborhood_internal_edges_acute", pi.lo / 10 > 2 * neighborhood),
    )
    if not all(passed for _, passed in checks):
        raise ArithmeticError("fixed finite-contact proof constants were not certified")
    exponential = exp_unit_bounds(growth * h)
    cases = []
    for initial in readout.cases:
        rho = max(
            readout.state_norm_upper_bound,
            max(map(abs, initial.bridge_phase_difference_bounds)),
        )
        whole_time_radius = rho * exponential[1]
        displacement = rho * (exponential[1] - 1)
        remainder = rho * (exponential[1] - 1 - growth * h)
        if whole_time_radius >= neighborhood:
            raise ArithmeticError(
                "finite-contact growth bound exceeds its neighborhood"
            )
        left = (
            h * initial.left_contact_form_rate_bounds[0] - remainder,
            h * initial.left_contact_form_rate_bounds[1] + remainder,
        )
        right = (
            h * initial.right_contact_form_rate_bounds[0] - remainder,
            h * initial.right_contact_form_rate_bounds[1] + remainder,
        )
        control_radius = h * 4 * readout.state_norm_upper_bound / 3
        control_left = -control_radius, control_radius
        control_right = Q(0), Q(0)
        induced_left = left[0] - control_radius, left[1] + control_radius
        induced_right = right
        internal_margin = pi.lo / 10 - 2 * whole_time_radius
        bridge_margin = pi.lo / 2 - 2 * whole_time_radius
        winding = internal_margin > 0 and bridge_margin > 0
        sign = initial.form_sign
        contact_sign = (
            left[1] < 0 and right[0] > 0 if sign == 1 else left[0] > 0 and right[1] < 0
        )
        induced_sign = (
            induced_left[1] < 0 and induced_right[0] > 0
            if sign == 1
            else induced_left[0] > 0 and induced_right[1] < 0
        )
        reasons = tuple(
            reason
            for passed, reason in (
                (winding, "whole_contact_acute_winding_preservation_not_certified"),
                (contact_sign, "accumulated_contact_form_sign_not_certified"),
                (induced_sign, "contact_induced_form_sign_not_certified"),
            )
            if not passed
        )
        cases.append(
            RelationalMemoryContactCase(
                form_sign=sign,
                initial_state_radius=rho,
                whole_time_state_radius_upper_bound=whole_time_radius,
                state_change_upper_bound=displacement,
                accumulated_form_remainder_bound=remainder,
                left_contact_form_change_bounds=left,
                right_contact_form_change_bounds=right,
                left_no_contact_form_change_bounds=control_left,
                right_no_contact_form_change_bounds=control_right,
                left_contact_induced_form_change_bounds=induced_left,
                right_contact_induced_form_change_bounds=induced_right,
                internal_acute_margin_lower_bound=internal_margin,
                bridge_acute_margin_lower_bound=bridge_margin,
                contact_storage_bounds=initial.contact_storage_bounds,
                continuous_loss_upper_bound=44 * whole_time_radius**2 * h,
                winding_preserved=winding,
                contact_sign_certified=contact_sign,
                contact_induced_sign_certified=induced_sign,
                unavailable_reasons=reasons,
            )
        )
    contact_separated = all(case.contact_sign_certified for case in cases)
    induced_separated = all(case.contact_induced_sign_certified for case in cases)
    windings = all(case.winding_preserved for case in cases)
    reasons = tuple(
        reason
        for passed, reason in (
            (windings, "whole_contact_winding_preservation_not_certified"),
            (contact_separated, "accumulated_contact_sign_separation_not_certified"),
            (induced_separated, "contact_induced_sign_separation_not_certified"),
        )
        if not passed
    )
    return RelationalMemoryContactCertificate(
        readout=readout,
        duration=h,
        duration_limit=limit,
        end_time=readout.horizon + h,
        neighborhood_radius=neighborhood,
        field_growth_constant=growth,
        form_rate_lipschitz_constant=growth,
        exponential_bounds=exponential,
        proof_checks=checks,
        required_event_work_upper_bound=max(
            case.contact_storage_bounds[1] for case in cases
        ),
        cases=tuple(cases),
        contact_signs_separated=contact_separated,
        contact_induced_signs_separated=induced_separated,
        windings_preserved=windings,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalMemoryRetentionCase:
    """Receiver mean and two-component capture after a supplied bridge cut.

    ``receiver_persistent_mean_bounds`` enclose the arithmetic form mean
    at removal and at every subsequent isolated time. Each receiver node
    converges to that same mean under the admitted native law; no finite
    recovery time or final common phase is specified. The cycle energy bound
    applies separately to each ring. Event-storage change is the negative
    endpoint bridge cost, not subtraction of the initial insertion cost.
    """

    form_sign: int
    receiver_persistent_mean_bounds: tuple[Q, Q]
    no_contact_receiver_mean_bounds: tuple[Q, Q]
    isolated_cycle_excess_storage_upper_bound: Q
    isolated_cycle_storage_upper_bound: Q
    isolated_cycle_capture_margin_lower_bound: Q
    removal_bridge_form_difference_bounds: tuple[Q, Q]
    removal_bridge_phase_difference_bounds: tuple[Q, Q]
    removal_bridge_storage_bounds: tuple[Q, Q]
    removal_storage_change_bounds: tuple[Q, Q]
    receiver_mean_sign_certified: bool
    both_rings_captured: bool
    removal_storage_nonincrease_certified: bool
    removal_storage_strict_decrease_certified: bool
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RelationalMemoryRetentionCertificate:
    """Fixed contact, supplied state-preserving removal and retained form mean.

    The bridge is removed at the end of the default duration-1/4096 contact.
    Both resulting C5 components then evolve independently under their native
    unforced common-unit-capacity law, with support held fixed. Their separate
    acute energy bounds admit convergence to a winding-one twist and uniform
    form. The receiver's ordinary form mean is conserved throughout recovery.

    The untouched no-contact receiver has mean exactly zero. Opposite signed
    retained means therefore distinguish the preparations after contact ends.
    This is conditional transfer of a supplied preparation, not autonomous
    support selection, a physical-memory identification or substrate creation.
    """

    contact: RelationalMemoryContactCertificate
    cycle_reference_storage_bounds: tuple[Q, Q]
    cycle_capture_barrier_bounds: tuple[Q, Q]
    cases: tuple[RelationalMemoryRetentionCase, RelationalMemoryRetentionCase]
    receiver_mean_signs_separated: bool
    both_rings_captured: bool
    removal_storage_nonincrease_certified: bool
    removal_storage_strict_decrease_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = INTERVAL_METHOD + ";exact_rational_exponential_series"
    scope: tuple[str, ...] = (
        "fixed_default_finite_contact_followed_by_supplied_state_preserving_single_bridge_cut",
        "each_resulting_C5_has_native_unforced_held_common_unit_capacity_dynamics",
        "independent_acute_winding_one_energy_capture_of_both_components",
        "receiver_arithmetic_form_mean_conserved_after_removal",
        "receiver_nodes_converge_to_the_retained_mean_without_a_finite_recovery_time_claim",
        "no_contact_receiver_mean_is_exactly_zero_under_its_separate_native_law",
        "insertion_work_continuous_loss_and_endpoint_removal_storage_are_separate",
        "capture_mean_sign_and_removal_work_availability_are_evaluated_independently",
        "no_graph_projection_event_execution_ODE_simulation_or_autonomous_support_law",
        "no_physical_measurement_bridge_or_indefinite_memory_under_later_interventions",
    )

    @property
    def admitted(self) -> bool:
        """Whether the retained means separate with capture and an admitted cut."""
        return self.status == "admitted"


def certify_relational_memory_retention() -> RelationalMemoryRetentionCertificate:
    """Certify persistent receiver form means after the fixed contact and cut.

    Only the receiver port has a nonzero initial contact rate. If every
    coordinate's accumulated remainder is bounded by B, its five-node mean
    lies in ``h*initial_right_port_rate/5 +/- B``. The remainder is B, not B/5:
    the other four receiver coordinates evolve during contact as well.

    At removal, each ring's storage above the exact twist is at most
    ``20*rho_tube**2``. The form contribution is bounded by ``10*rho_tube**2``;
    the phase Taylor linear term telescopes and its second derivative is at
    most one, giving the same bound for phase. The shared memory geometry
    retains the reference energy and the strict C5 face barrier. No rounded
    endpoint graph or new capture criterion is substituted for these bounds.

    After the cut, constant degree two and common unit capacity make the
    Laplacian form term and native phase-pressure sums vanish. Hence each
    arithmetic form mean is conserved. Capture gives uniform limiting form equal to that
    mean, while the phase twist recovers its own unprescribed common offset.
    """
    contact = certify_relational_memory_contact(duration=Q(1, 4096))
    geometry = contact.readout.memory.cases[0].asymptotic
    reference = geometry.reference_storage_bounds
    barrier = geometry.capture_barrier_bounds
    cases = []
    for episode, initial in zip(contact.cases, contact.readout.cases):
        h = contact.duration
        remainder = episode.accumulated_form_remainder_bound
        receiver = (
            h * initial.right_contact_form_rate_bounds[0] / 5 - remainder,
            h * initial.right_contact_form_rate_bounds[1] / 5 + remainder,
        )
        excess = 20 * episode.whole_time_state_radius_upper_bound**2
        storage_upper = reference[1] + excess
        capture_margin = barrier[0] - storage_upper
        captured = episode.winding_preserved and capture_margin > 0
        form_radius = (
            contact.readout.state_norm_upper_bound
            + 2 * episode.state_change_upper_bound
        )
        form_difference = -form_radius, form_radius
        phase_difference = (
            initial.bridge_phase_difference_bounds[0]
            - 2 * episode.state_change_upper_bound,
            initial.bridge_phase_difference_bounds[1]
            + 2 * episode.state_change_upper_bound,
        )
        bridge_storage = (
            I(*form_difference) ** 2 / 2 + 2 * sin(I(*phase_difference) / 2) ** 2
        )
        removal = -bridge_storage.hi, -bridge_storage.lo
        nonincrease = removal[1] <= 0
        strict_decrease = removal[1] < 0
        sign = episode.form_sign
        mean_sign = receiver[0] > 0 if sign == 1 else receiver[1] < 0
        reasons = tuple(
            reason
            for passed, reason in (
                (mean_sign, "receiver_persistent_mean_sign_not_certified"),
                (captured, "both_isolated_cycle_captures_not_certified"),
                (nonincrease, "removal_storage_nonincrease_not_certified"),
            )
            if not passed
        )
        cases.append(
            RelationalMemoryRetentionCase(
                form_sign=sign,
                receiver_persistent_mean_bounds=receiver,
                no_contact_receiver_mean_bounds=(Q(0), Q(0)),
                isolated_cycle_excess_storage_upper_bound=excess,
                isolated_cycle_storage_upper_bound=storage_upper,
                isolated_cycle_capture_margin_lower_bound=capture_margin,
                removal_bridge_form_difference_bounds=form_difference,
                removal_bridge_phase_difference_bounds=phase_difference,
                removal_bridge_storage_bounds=(bridge_storage.lo, bridge_storage.hi),
                removal_storage_change_bounds=removal,
                receiver_mean_sign_certified=mean_sign,
                both_rings_captured=captured,
                removal_storage_nonincrease_certified=nonincrease,
                removal_storage_strict_decrease_certified=strict_decrease,
                unavailable_reasons=reasons,
            )
        )
    mean_separated = all(case.receiver_mean_sign_certified for case in cases)
    captured = all(case.both_rings_captured for case in cases)
    nonincrease = all(case.removal_storage_nonincrease_certified for case in cases)
    strict_decrease = all(
        case.removal_storage_strict_decrease_certified for case in cases
    )
    reasons = tuple(
        reason
        for passed, reason in (
            (mean_separated, "persistent_receiver_mean_sign_separation_not_certified"),
            (captured, "both_isolated_cycle_captures_not_certified"),
            (nonincrease, "removal_storage_nonincrease_not_certified"),
        )
        if not passed
    )
    return RelationalMemoryRetentionCertificate(
        contact=contact,
        cycle_reference_storage_bounds=reference,
        cycle_capture_barrier_bounds=barrier,
        cases=tuple(cases),
        receiver_mean_signs_separated=mean_separated,
        both_rings_captured=captured,
        removal_storage_nonincrease_certified=nonincrease,
        removal_storage_strict_decrease_certified=strict_decrease,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )
