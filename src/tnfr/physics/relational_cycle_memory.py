"""Conditional second-order phase memory for an ideal isolated C5 family.

The native law can recover its twist while retaining a changed common phase.
The general family reader bounds its quadratic coefficient and a separate
hypothetical contact readout, without a finite-amplitude remainder. A narrower
fixed-preparation certificate also bounds the limiting phase shift and its
nonlinear contact response through integrated continuous inequalities.
A fixed-clock readout certificate also bounds the remaining recovery error,
the separate background/contact rates and the proposed event cost.
None of these readers executes a trajectory or selects a support event.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from numbers import Integral

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    atan,
    cos,
    pi_interval,
    sin,
)
from ._cycle_algebra import dirichlet_energy, dot, ordered_vector
from .relational_capture import _require_sector
from .relational_observations import _ordered

__all__ = (
    "RelationalCycleMemoryBounds",
    "bound_relational_cycle_memory",
    "RelationalFiniteMemoryCase",
    "RelationalFiniteMemoryCertificate",
    "certify_relational_cycle_memory",
    "RelationalMemoryReadoutCase",
    "RelationalMemoryReadoutCertificate",
    "certify_relational_cycle_memory_readout",
)


@dataclass(frozen=True)
class RelationalCycleMemoryBounds:
    """Asymptotic coefficients and a separate sufficient C5 family basin.

    Directions describe ``x=epsilon*u`` and ``theta=theta_*+epsilon*v`` in
    the supplied ordered cycle, where ``theta_*`` is the exact mathematical
    winding-one twist of the declared sign. Their exact sums are zero.
    Rational capacity, directions and radius need not be binary64 states.

    The limiting lifted common-phase shift is ``C*epsilon**2+O(epsilon**3)``.
    ``quadratic_phase_shift_coefficient_bounds`` enclose C, not that shift at
    a nonzero epsilon. ``remainder_bound`` is therefore explicitly unavailable.
    The other coefficient is the left-port form-rate readout after a supplied
    bridge to an untouched equal-capacity, equal-form, aligned reference C5.
    This readout is hypothetical and does not select or execute a contact.
    """

    model: RelationalExchangeModel
    form_direction: tuple[Q, ...]
    phase_direction: tuple[Q, ...]
    capacity: Q
    amplitude_radius: Q
    target_sector: int
    cyclic_pairing: Q
    quadratic_phase_shift_coefficient_bounds: tuple[Q, Q]
    quadratic_left_port_form_rate_coefficient_bounds: tuple[Q, Q]
    pi_bounds: tuple[Q, Q]
    twist_cosine_bounds: tuple[Q, Q]
    twist_sine_bounds: tuple[Q, Q]
    reference_storage_bounds: tuple[Q, Q]
    acute_margin_lower_bound: Q
    storage_upper_bound: Q
    capture_barrier_bounds: tuple[Q, Q]
    storage_margin_lower_bound: Q
    basin_admitted: bool
    unavailable_reasons: tuple[str, ...]
    remainder_order: int = 3
    remainder_bound: None = None
    enclosure_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "ideal_simple_unit_C5_native_unforced_law_with_positive_homogeneous_held_capacity",
        "exact_mathematical_twist_with_supplied_ordered_zero_mean_perturbation_directions",
        "quadratic_limiting_common_phase_coefficient_not_a_finite_amplitude_response_bound",
        "basin_admitted_gates_sufficient_capture_for_all_absolute_amplitudes_within_radius",
        "coefficient_availability_independent_of_sufficient_whole_family_basin_admission",
        "hypothetical_left_port_form_rate_after_one_supplied_aligned_reference_contact",
        "relative_phase_readout_requires_a_retained_reference_not_intrinsic_shape_change",
        "no_graph_admission_field_evaluation_trajectory_event_selector_or_physical_clock",
    )


def _direction(values, label):
    direction = ordered_vector(_ordered(values, label, limit=6), label)
    if len(direction) != 5:
        raise ValueError(f"{label} must contain exactly five ordered coordinates")
    if sum(direction, Q(0)):
        raise ValueError(f"{label} must have exactly zero sum")
    return direction


def _scaled_bounds(interval, factor):
    """Scale enclosed transcendental constants without rounding exact inputs."""
    endpoints = factor * interval.lo, factor * interval.hi
    return min(endpoints), max(endpoints)


def bound_relational_cycle_memory(
    *,
    model: RelationalExchangeModel,
    form_direction,
    phase_direction,
    capacity,
    amplitude_radius,
    target_sector: int = 1,
) -> RelationalCycleMemoryBounds:
    """Bound a declared C5 family's quadratic phase memory without evolution.

    With ``D*v = v_(i+1)-v_(i-1)``, the coefficient is
    ``w*sin(kappa)*(u.T*D*v)/(10*e*beta*pi*cos(kappa)**2)`` for
    ``kappa=target_sector*2*pi/5``. It is independent of the positive common
    capacity, which changes the clock of the isolated relaxation. The separate
    contact coefficient multiplies it by ``-capacity*w/(pi*(1+2*cos(kappa)))``.

    Model coefficients retain their admitted represented values. Directions,
    capacity and radius preserve exact rational inputs; other real scalars
    use the shared finite binary64 boundary. Invalid inputs raise. Failing a
    sufficient whole-family acute/storage test returns its reasons while
    preserving the local asymptotic coefficient. No finite-amplitude error,
    valid binary64 realization, graph support or recovery time is certified.
    """
    if not isinstance(model, RelationalExchangeModel):
        raise TypeError("model must be a RelationalExchangeModel")
    _require_sector(target_sector)
    form = _direction(form_direction, "form_direction")
    phase = _direction(phase_direction, "phase_direction")
    nu = exact_or_represented_real(capacity, "capacity")
    radius = exact_or_represented_real(amplitude_radius, "amplitude_radius")
    if nu <= 0 or radius <= 0:
        raise ValueError("capacity and amplitude_radius must be strictly positive")
    e, w, beta = map(Q, (model.epi_weight, model.phase_weight, model.storage_scale))
    if e <= 0:
        raise ValueError("cycle memory requires positive epi_weight")

    difference = tuple(phase[(i + 1) % 5] - phase[i - 1] for i in range(5))
    pairing = dot(form, difference)
    pi = pi_interval()
    kappa = 2 * pi / 5
    cosine, sine = cos(kappa), sin(kappa)
    if cosine.lo <= 0 or sine.lo <= 0:
        raise ArithmeticError("twist trigonometry requires positive certified bounds")
    geometry = sine / (pi * cosine**2)
    # Exact rational grouping avoids underflow in beta/e or loss of a very
    # small nonzero direction before interval division and multiplication.
    factor = target_sector * w * pairing / (10 * e * beta)
    coefficient = _scaled_bounds(geometry, factor)
    readout_geometry = geometry / (pi * (1 + 2 * cosine))
    readout = _scaled_bounds(readout_geometry, -nu * w * factor)

    reference = _scaled_bounds(5 * (1 - cosine), beta)
    phase_edge_size = max(abs(phase[(i + 1) % 5] - phase[i]) for i in range(5))
    acute_margin = pi.lo / 10 - radius * phase_edge_size
    quadratic_storage = dirichlet_energy(form) + beta * dirichlet_energy(phase)
    storage_upper = reference[1] + radius**2 * quadratic_storage
    barrier = _scaled_bounds(5 - 4 * cos(3 * pi / 8), beta)
    storage_margin = barrier[0] - storage_upper
    reasons = tuple(
        reason
        for passed, reason in (
            (acute_margin > 0, "strict_family_acute_margin_not_certified"),
            (storage_margin > 0, "strict_family_cycle_energy_barrier_not_certified"),
        )
        if not passed
    )
    return RelationalCycleMemoryBounds(
        model=model,
        form_direction=form,
        phase_direction=phase,
        capacity=nu,
        amplitude_radius=radius,
        target_sector=target_sector,
        cyclic_pairing=pairing,
        quadratic_phase_shift_coefficient_bounds=coefficient,
        quadratic_left_port_form_rate_coefficient_bounds=readout,
        pi_bounds=(pi.lo, pi.hi),
        twist_cosine_bounds=(cosine.lo, cosine.hi),
        twist_sine_bounds=_scaled_bounds(sine, Q(target_sector)),
        reference_storage_bounds=reference,
        acute_margin_lower_bound=acute_margin,
        storage_upper_bound=storage_upper,
        capture_barrier_bounds=barrier,
        storage_margin_lower_bound=storage_margin,
        basin_admitted=not reasons,
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalFiniteMemoryCase:
    """One signed ideal preparation's limiting offset and proposed contact.

    Contact joins the recovered left ring to an untouched aligned reference
    ring at matching ports, preserving their primitive states. Left/right
    intervals enclose the exact nonlinear form rates at that hypothetical
    contact, not the end of a finite recovery time. Positive bridge storage
    is event work; it is not supplied by the earlier continuous loss.
    """

    form_sign: int
    asymptotic: RelationalCycleMemoryBounds
    leading_phase_shift_bounds: tuple[Q, Q]
    limiting_phase_shift_bounds: tuple[Q, Q]
    readout_acute_margin_lower_bound: Q
    left_contact_form_rate_bounds: tuple[Q, Q]
    right_contact_form_rate_bounds: tuple[Q, Q]
    contact_storage_bounds: tuple[Q, Q]
    phase_sign_certified: bool
    contact_sign_certified: bool
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RelationalFiniteMemoryCertificate:
    """Finite-amplitude proof for two fixed common-capacity C5 preparations.

    The two initial forms are +/-epsilon*(1,-1,0,0,0), with the same phase
    perturbation epsilon*(0,1,-1,0,0) about exact winding +1. Both initial
    means agree. Coefficients are e=w=1/2 and beta=capacity=1, with fixed
    support and no forcing. They are proof premises, not fitted parameters.

    Integrated bounds control the cubic remainder of the limiting phase.
    They give no recovery time or finite-time numerical response certificate.
    ``admitted`` additionally requires opposite phase and contact-rate signs
    for the two cases. An inconclusive sign bound remains unavailable, even
    if caused only by fixed interval precision at a very small amplitude.
    """

    amplitude: Q
    amplitude_limit: Q
    neighborhood_radius: Q
    initial_excess_storage_upper_bound: Q
    first_exit_margin_lower_bound: Q
    trajectory_norm_upper_bound: Q
    form_norm_squared_integral_upper_bound: Q
    phase_norm_squared_integral_upper_bound: Q
    mixed_norm_integral_upper_bound: Q
    remainder_factor: Q
    phase_remainder_bound: Q
    proof_checks: tuple[tuple[str, bool], ...]
    cases: tuple[RelationalFiniteMemoryCase, RelationalFiniteMemoryCase]
    phase_signs_separated: bool
    contact_signs_separated: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_ideal_C5_winding_plus_one_unit_capacity_beta_one_equal_half_weights",
        "opposite_zero_mean_form_directions_and_identical_zero_mean_phase_direction",
        "first_exit_storage_and_integrated_cross_balance_without_trajectory_execution",
        "finite_amplitude_limiting_phase_remainder_not_a_finite_recovery_time_bound",
        "nonlinear_hypothetical_single_contact_to_an_untouched_aligned_reference",
        "positive_contact_storage_is_event_work_without_a_supplied_occurrence_or_reservoir",
        "fixed_interval_precision_can_leave_readout_signs_explicitly_unavailable",
        "no_graph_admission_binary64_flow_external_measurement_or_physical_identity_claim",
    )

    @property
    def admitted(self) -> bool:
        """Whether both finite offsets and hypothetical readouts separate."""
        return self.status == "admitted"


def certify_relational_cycle_memory(
    *, amplitude=Q(1, 2**20)
) -> RelationalFiniteMemoryCertificate:
    """Certify the fixed +/-form memory comparison without integrating it.

    Require exact/represented ``0 < amplitude <= 1/1024``. A first-exit
    argument traps the centered state inside radius 1/100. Shared capture
    then permits integrated form/phase estimates, giving
    ``|alpha_infinity-C*epsilon**2| <= 2**15*epsilon**3``. Larger amplitudes
    or different laws/directions require another proof and are not accepted.

    The nonlinear contact map is
    ``atan(sin(delta)/(2*cos(2*pi/5)+cos(delta)))/(2*pi)`` at the right
    reference port, and its negative at the recovered left port. Its storage
    cost is ``2*sin(delta/2)**2``. No contact is performed or claimed passive.
    Exact rational scaling precedes interval materialization. At unresolved
    interval precision the report retains the bounds and abstains on signs;
    it does not assign an exact zero response.
    """
    epsilon = exact_or_represented_real(amplitude, "amplitude")
    limit, radius = Q(1, 1024), Q(1, 100)
    if not 0 < epsilon <= limit:
        raise ValueError("amplitude must satisfy 0 < amplitude <= 1/1024")
    square = epsilon**2
    initial_storage = 6 * square
    first_exit_margin = radius**2 / 8 - initial_storage
    remainder_factor = Q(2**15)
    remainder = remainder_factor * epsilon**3

    pi = pi_interval()
    kappa = 2 * pi / 5
    cosine, sine = cos(kappa), sin(kappa)
    metric_cosine_floor = cos(kappa + radius).lo
    sinc_floor = 1 - (2 * radius) ** 2 / 6
    # For h=c*sec(kappa+t)/sinc(m), |t|<=||v|| and |m|<=2||v||.
    # These bounds imply |h-1|<=4||v|| and the centered linear remainder
    # <=17||v||**2 throughout the radius-1/100 neighborhood.
    metric_scale = cosine.hi / metric_cosine_floor
    derivative_one = cosine.hi / metric_cosine_floor**2
    half_derivative_two = cosine.hi / metric_cosine_floor**3
    sinc_remainder = 1 / (6 * sinc_floor)
    linear_phase_gain = 1 / (4 * pi.lo * cosine.lo)
    correction_scale = sine.hi / (10 * pi.lo * cosine.lo**2)
    checks = (
        ("pi_between_three_and_four", 3 < pi.lo and pi.hi < 4),
        ("cycle_spectral_gap_above_one", 2 - 2 * cosine.hi > 1),
        ("phase_hessian_cosine_above_quarter", cos(kappa + 2 * radius).lo > Q(1, 4)),
        ("metric_cosine_above_299_over1000", metric_cosine_floor > Q(299, 1000)),
        ("metric_scale_below_11_over10", metric_scale < Q(11, 10)),
        ("metric_first_derivative_below_7_over2", derivative_one < Q(7, 2)),
        ("metric_half_second_derivative_below_twelve", half_derivative_two < 12),
        ("inverse_sinc_remainder_below_fifth", sinc_remainder < Q(1, 5)),
        ("phase_metric_above_one", 2 * pi.lo * metric_cosine_floor * sinc_floor > 1),
        ("linear_phase_gain_below_third", linear_phase_gain < Q(1, 3)),
        ("quadratic_correction_scale_below_half", correction_scale < Q(1, 2)),
        ("initial_point_inside_neighborhood", 4 * square < radius**2),
        ("strict_first_exit_storage_margin", first_exit_margin > 0),
    )
    if not all(passed for _, passed in checks):
        raise ArithmeticError("fixed cycle-memory proof constants were not certified")

    model = RelationalExchangeModel(storage_scale=1, epi_weight=0.5, phase_weight=0.5)
    cases = []
    for sign in (1, -1):
        asymptotic = bound_relational_cycle_memory(
            model=model,
            form_direction=(sign, -sign, 0, 0, 0),
            phase_direction=(0, 1, -1, 0, 0),
            capacity=1,
            amplitude_radius=epsilon,
        )
        if not asymptotic.basin_admitted:
            raise ArithmeticError("fixed cycle-memory family failed capture admission")
        leading = tuple(
            square * value
            for value in asymptotic.quadratic_phase_shift_coefficient_bounds
        )
        phase_bounds = leading[0] - remainder, leading[1] + remainder
        acute_margin = pi.lo / 2 - max(map(abs, phase_bounds))
        if acute_margin <= 0:
            raise ArithmeticError(
                "limiting phase bounds do not admit the proposed contact"
            )
        delta = I(*phase_bounds)
        denominator = 2 * cosine + cos(delta)
        if denominator.lo <= 0:
            raise ArithmeticError("contact resultant denominator is unresolved")
        rate = atan(sin(delta) / denominator) / (2 * pi)
        right_rate = rate.lo, rate.hi
        left_rate = -rate.hi, -rate.lo
        cost = 2 * sin(delta / 2) ** 2
        phase_sign = phase_bounds[0] > 0 if sign == 1 else phase_bounds[1] < 0
        contact_sign = (
            (right_rate[0] > 0 and left_rate[1] < 0)
            if sign == 1
            else (right_rate[1] < 0 and left_rate[0] > 0)
        )
        reasons = tuple(
            reason
            for passed, reason in (
                (phase_sign, "limiting_phase_sign_not_certified"),
                (contact_sign, "nonlinear_contact_rate_sign_not_certified"),
            )
            if not passed
        )
        cases.append(
            RelationalFiniteMemoryCase(
                form_sign=sign,
                asymptotic=asymptotic,
                leading_phase_shift_bounds=leading,
                limiting_phase_shift_bounds=phase_bounds,
                readout_acute_margin_lower_bound=acute_margin,
                left_contact_form_rate_bounds=left_rate,
                right_contact_form_rate_bounds=right_rate,
                contact_storage_bounds=(cost.lo, cost.hi),
                phase_sign_certified=phase_sign,
                contact_sign_certified=contact_sign,
                unavailable_reasons=reasons,
            )
        )
    phase_separated = all(case.phase_sign_certified for case in cases)
    contact_separated = all(case.contact_sign_certified for case in cases)
    reasons = tuple(
        reason
        for passed, reason in (
            (phase_separated, "limiting_phase_sign_separation_not_certified"),
            (contact_separated, "nonlinear_contact_rate_sign_separation_not_certified"),
        )
        if not passed
    )
    return RelationalFiniteMemoryCertificate(
        amplitude=epsilon,
        amplitude_limit=limit,
        neighborhood_radius=radius,
        initial_excess_storage_upper_bound=initial_storage,
        first_exit_margin_lower_bound=first_exit_margin,
        trajectory_norm_upper_bound=7 * epsilon,
        form_norm_squared_integral_upper_bound=24 * square,
        phase_norm_squared_integral_upper_bound=2048 * square,
        mixed_norm_integral_upper_bound=224 * square,
        remainder_factor=remainder_factor,
        phase_remainder_bound=remainder,
        proof_checks=checks,
        cases=tuple(cases),
        phase_signs_separated=phase_separated,
        contact_signs_separated=contact_separated,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalMemoryReadoutCase:
    """One finite-clock recovery enclosure and its hypothetical contact.

    Background rates refer to the two still separate rings at the declared
    horizon. Contact rates refer to adding a single matching-port bridge at
    that same state; the rate changes subtract the background enclosures.
    Residual form and phase coordinates need not vanish. The two contact
    rates consequently need not be exact negatives. Contact storage is the
    event cost, without a claimed reservoir or occurrence law.
    """

    form_sign: int
    mean_phase_bounds: tuple[Q, Q]
    bridge_phase_difference_bounds: tuple[Q, Q]
    left_resultant_real_bounds: tuple[Q, Q]
    right_resultant_real_bounds: tuple[Q, Q]
    internal_acute_margin_lower_bound: Q
    bridge_acute_margin_lower_bound: Q
    left_no_contact_form_rate_bounds: tuple[Q, Q]
    right_no_contact_form_rate_bounds: tuple[Q, Q]
    left_contact_form_rate_bounds: tuple[Q, Q]
    right_contact_form_rate_bounds: tuple[Q, Q]
    left_contact_form_rate_change_bounds: tuple[Q, Q]
    right_contact_form_rate_change_bounds: tuple[Q, Q]
    contact_storage_bounds: tuple[Q, Q]
    phase_sign_certified: bool
    contact_sign_certified: bool
    rate_change_sign_certified: bool
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RelationalMemoryReadoutCertificate:
    """Finite-time readout for the fixed amplitude-2**-20 preparations.

    The independent variable is the model's declared structural clock.
    ``horizon=4128*decay_blocks`` is a sufficient observation time, not a
    fitted recovery time or an optimal earliest readout. The exponential
    comparison supplies a centered state ball and a mean-phase tail bound.
    Contact intervals then use the full nonlinear phasor pressure, retaining
    the remaining form and phase deviations and separate background rates.

    All enclosures remain available if signs overlap. ``admitted`` requires
    the expected opposite signs for the actual means/bridge gaps, both contact
    ports and both contact-induced rate changes. No simulation, graph mutation or live
    support event is performed.
    """

    memory: RelationalFiniteMemoryCertificate
    decay_blocks: int
    block_duration: Q
    horizon: Q
    state_norm_upper_bound: Q
    mean_phase_tail_upper_bound: Q
    modified_storage_equivalence_bounds: tuple[Q, Q]
    initial_modified_storage_upper_bound: Q
    contact_rate_error_upper_bound: Q
    rate_change_error_upper_bound: Q
    cases: tuple[RelationalMemoryReadoutCase, RelationalMemoryReadoutCase]
    phase_signs_separated: bool
    contact_signs_separated: bool
    rate_change_signs_separated: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_ideal_C5_memory_preparations_at_amplitude_two_to_minus_twenty",
        "modified_storage_decay_for_native_unforced_held_unit_capacity_law",
        "finite_structural_clock_horizon_not_laboratory_time_or_earliest_recovery",
        "centered_state_and_remaining_mean_drift_bounds_without_trajectory_execution",
        "hypothetical_single_contact_to_an_untouched_aligned_reference",
        "background_contact_and_contact_induced_rate_change_are_separate_observations",
        "positive_contact_storage_is_event_work_not_reused_continuous_dissipation",
        "unresolved_signs_remain_unavailable_with_nonempty_response_enclosures",
        "no_live_graph_event_selector_physical_bridge_or_postcontact_future_claim",
    )

    @property
    def admitted(self) -> bool:
        """Whether finite-clock means, contact rates and increments separate."""
        return self.status == "admitted"


def certify_relational_cycle_memory_readout(
    *, decay_blocks: int = 40
) -> RelationalMemoryReadoutCertificate:
    """Bound the fixed preparation's contact response at a finite horizon.

    A nonnegative integer ``n`` specifies ``t=4128*n``. The modified storage
    ``H=E_excess+(u.T*B_pseudoinverse*v)/32`` satisfies
    ``7*||z||**2/64 <= H <= 129*||z||**2/64`` and ``H' <= -H/2064``.
    Thus ``||z(t)|| <= R=8*epsilon/2**n`` and the remaining mean drift is
    bounded by ``Q=4096*R**2``. These are exact conditional continuous
    estimates, not assertions about an Euler step or a numerical trajectory.

    For bridge phase ``delta=m+v[0]``, the contact resultants are
    ``exp(i*(kappa+v[1]-v[0])) + exp(i*(-kappa+v[4]-v[0])) + exp(-i*delta)``
    on the left and ``2*cos(kappa)+exp(i*delta)`` on the right. Their real
    parts and all internal/bridge acute margins are checked before evaluating
    their argument through the shared interval arctangent. Exact event cost
    is ``u[0]**2/2 + 2*sin(delta/2)**2``. Independent coordinate enclosures
    deliberately discard correlations to provide conservative response bounds.
    """
    if isinstance(decay_blocks, bool) or not isinstance(decay_blocks, Integral):
        raise TypeError("decay_blocks must be a nonnegative integer")
    n = int(decay_blocks)
    if n < 0:
        raise ValueError("decay_blocks must be a nonnegative integer")
    memory = certify_relational_cycle_memory(amplitude=Q(1, 2**20))
    radius = 8 * memory.amplitude / 2**n
    tail = 4096 * radius**2
    block_duration = Q(4128)
    pi = pi_interval()
    kappa = 2 * pi / 5
    cosine = cos(kappa)
    internal_margin = pi.lo / 10 - 2 * radius
    before_left = (-4 * radius / 3, 4 * radius / 3)
    before_right = (Q(0), Q(0))
    neighbor_plus = kappa + I(-2 * radius, 2 * radius)
    neighbor_minus = -kappa + I(-2 * radius, 2 * radius)
    cases = []
    for limiting in memory.cases:
        mean = (
            limiting.limiting_phase_shift_bounds[0] - tail,
            limiting.limiting_phase_shift_bounds[1] + tail,
        )
        phase = mean[0] - radius, mean[1] + radius
        delta = I(*phase)
        bridge_margin = pi.lo / 2 - max(map(abs, phase))
        left_real = cos(neighbor_plus) + cos(neighbor_minus) + cos(delta)
        left_imag = sin(neighbor_plus) + sin(neighbor_minus) - sin(delta)
        right_real = 2 * cosine + cos(delta)
        if (
            min(internal_margin, bridge_margin) <= 0
            or min(left_real.lo, right_real.lo) <= 1
        ):
            raise ArithmeticError("finite readout box failed acute/resultant admission")
        left_rate = I(-5 * radius / 6, 5 * radius / 6) + atan(left_imag / left_real) / (
            2 * pi
        )
        right_rate = I(-radius / 6, radius / 6) + atan(sin(delta) / right_real) / (
            2 * pi
        )
        left_change = left_rate - I(*before_left)
        right_change = right_rate
        cost = I(0, radius**2 / 2) + 2 * sin(delta / 2) ** 2
        sign = limiting.form_sign
        phase_sign = phase[0] > 0 if sign == 1 else phase[1] < 0
        contact_sign = (
            left_rate.hi < 0 and right_rate.lo > 0
            if sign == 1
            else left_rate.lo > 0 and right_rate.hi < 0
        )
        change_sign = (
            left_change.hi < 0 and right_change.lo > 0
            if sign == 1
            else left_change.lo > 0 and right_change.hi < 0
        )
        reasons = tuple(
            reason
            for passed, reason in (
                (phase_sign, "finite_time_mean_or_bridge_phase_sign_not_certified"),
                (contact_sign, "finite_time_contact_rate_sign_not_certified"),
                (change_sign, "finite_time_contact_rate_change_sign_not_certified"),
            )
            if not passed
        )
        cases.append(
            RelationalMemoryReadoutCase(
                form_sign=sign,
                mean_phase_bounds=mean,
                bridge_phase_difference_bounds=phase,
                left_resultant_real_bounds=(left_real.lo, left_real.hi),
                right_resultant_real_bounds=(right_real.lo, right_real.hi),
                internal_acute_margin_lower_bound=internal_margin,
                bridge_acute_margin_lower_bound=bridge_margin,
                left_no_contact_form_rate_bounds=before_left,
                right_no_contact_form_rate_bounds=before_right,
                left_contact_form_rate_bounds=(left_rate.lo, left_rate.hi),
                right_contact_form_rate_bounds=(right_rate.lo, right_rate.hi),
                left_contact_form_rate_change_bounds=(left_change.lo, left_change.hi),
                right_contact_form_rate_change_bounds=(
                    right_change.lo,
                    right_change.hi,
                ),
                contact_storage_bounds=(cost.lo, cost.hi),
                phase_sign_certified=phase_sign,
                contact_sign_certified=contact_sign,
                rate_change_sign_certified=change_sign,
                unavailable_reasons=reasons,
            )
        )
    phase_separated = all(case.phase_sign_certified for case in cases)
    contact_separated = all(case.contact_sign_certified for case in cases)
    change_separated = all(case.rate_change_sign_certified for case in cases)
    reasons = tuple(
        reason
        for passed, reason in (
            (
                phase_separated,
                "finite_time_mean_or_bridge_phase_sign_separation_not_certified",
            ),
            (
                contact_separated,
                "finite_time_contact_rate_sign_separation_not_certified",
            ),
            (
                change_separated,
                "finite_time_contact_rate_change_sign_separation_not_certified",
            ),
        )
        if not passed
    )
    return RelationalMemoryReadoutCertificate(
        memory=memory,
        decay_blocks=n,
        block_duration=block_duration,
        horizon=block_duration * n,
        state_norm_upper_bound=radius,
        mean_phase_tail_upper_bound=tail,
        modified_storage_equivalence_bounds=(Q(7, 64), Q(129, 64)),
        initial_modified_storage_upper_bound=97 * memory.amplitude**2 / 16,
        contact_rate_error_upper_bound=2 * radius + tail,
        rate_change_error_upper_bound=4 * radius + tail,
        cases=tuple(cases),
        phase_signs_separated=phase_separated,
        contact_signs_separated=contact_separated,
        rate_change_signs_separated=change_separated,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )
