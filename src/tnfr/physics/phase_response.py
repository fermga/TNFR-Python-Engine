"""Exact conditional phase geometry and joint nodal response.

Cosine Gram data describe unit planar phasors without approximating their
angles. These detached coefficients do not identify live phase gates, a
binary64 derivative, a fixed point or a complete operator trajectory.
Joint pressure/acceleration identities reuse the transport owner and retain
the still-supplied capacity and phase velocities as explicit inputs.
Finite held-source compatibility determines admissible capacity profiles
without selecting an evolution law or authenticating an oriented phase source.
Local numerical phase/form predictions reuse the consensus tangent and shared
structural modes, with separate ideal nonlinear-error envelopes.
Current-state lock/source observations retain represented sine residuals and
production phasor pressure separately, without a tolerance-based certificate.
"""

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..mathematics._rational_interval import I, pi_interval
from ..mathematics.krylov import exact_rank
from ._cycle_algebra import Matrix, Vector, dot, ordered_vector
from ._exact_linear_algebra import exact_matrix_inverse
from .forcing_realization import (
    NonEpiForcingObservation,
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from .support_transport import (
    SupportTransportDerivative,
    SupportTransportSnapshot,
    _rebuild,
    _support_gradient,
    observe_support_transport_derivative,
)

__all__ = [
    "PhaseResponseReference",
    "derive_phase_response",
    "PhaseSourceGeometry",
    "observe_phase_source_geometry",
    "JointNodalResponse",
    "derive_joint_nodal_response",
    "PhaseCapacityBalance",
    "derive_phase_capacity_balance",
    "PhaseLockSourceObservation",
    "observe_phase_lock_source",
    "JointLinearSample",
    "SynchronizedJointPrediction",
    "predict_synchronized_joint_euler",
    "PhaseMomentInformationAssessment",
    "assess_phase_moment_information",
    "PhaseInformationResponse",
    "assess_phase_information_response",
    "PhaseMomentMotion",
    "derive_phase_moment_motion",
    "SineStarMomentClosure",
    "derive_sine_star_moment_closure",
]


def _admit_relative_phasors(values, label):
    from .relational_observations import _ordered

    rows = _ordered(values, label, limit=129)
    if not rows or len(rows) > 128:
        raise ValueError("phasors require 1..128 ordered neighbor pairs")
    admitted = []
    for index, row in enumerate(rows):
        pair = _ordered(row, f"{label}[{index}]", limit=3)
        if len(pair) != 2:
            raise ValueError("each phasor must be one cosine/sine pair")
        c, s = (exact_or_represented_real(value, f"{label}[{index}]") for value in pair)
        if c**2 + s**2 != 1:
            raise ValueError("phasor coefficients must have exact unit norm")
        admitted.append((c, s))
    return tuple(admitted)


def _phase_moments(phasors, multipliers=None):
    """Exact first/third moments, retaining each phasor's own multiplier."""
    if multipliers is None:
        multipliers = (Fraction(1),) * len(phasors)
    first = [Fraction(0), Fraction(0)]
    third = [Fraction(0), Fraction(0)]
    for (c, s), factor in zip(phasors, multipliers):
        first[0] += factor * c
        first[1] += factor * s
        third[0] += factor * (c**3 - 3 * c * s**2)
        third[1] += factor * (3 * c**2 * s - s**3)
    return tuple(first), tuple(third)


def _phase_source_storage(first, third, degree, coefficient):
    scale = 1 + 3 * coefficient / 4
    return (
        first[1] / degree,
        (scale * first[1] - coefficient * third[1] / 4) / degree,
        degree - first[0],
        (1 + 2 * coefficient / 3) * degree
        - scale * first[0]
        + coefficient * third[0] / 12,
    )


@dataclass(frozen=True)
class PhaseMomentInformationAssessment:
    """Exact first/third moment information for two declared rooted multisets.

    Phasor pairs are exact admitted rational cosine/sine coordinates of
    relative circular gaps, including repeated gaps at distinct neighbors.
    Floating inputs denote their represented coefficients and must satisfy
    the unit-circle identity exactly; no approximate normalization is made.
    Source numerators are multiplied by mathematical pi. Storage values are
    sums over the root's incidences, not a whole graph's energy or its rate.
    """

    left_phasors: tuple[tuple[Fraction, Fraction], ...]
    right_phasors: tuple[tuple[Fraction, Fraction], ...]
    degree: int
    epsilon: Fraction
    first_resultants: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    third_resultants: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    sine_source_pi_numerators: tuple[Fraction, Fraction]
    cubic_source_pi_numerators: tuple[Fraction, Fraction]
    cosine_storage_sums: tuple[Fraction, Fraction]
    cubic_storage_sums: tuple[Fraction, Fraction]
    first_resultants_equal: bool
    nonzero_first_resultants: tuple[bool, bool]
    strictly_acute: tuple[bool, bool]
    cubic_source_difference_pi_numerator: Fraction
    cubic_source_difference_bounds: I
    cosine_storage_difference: Fraction
    cubic_storage_difference: Fraction
    sufficiency_obstruction: bool
    scope: tuple[str, ...] = (
        "ordered_equal_degree_rooted_relative_unit_phasor_multisets",
        "exact_unit_circle_admission_without_angle_reconstruction_or_renormalization",
        "represented_real_inputs_retain_only_their_exact_represented_coefficients",
        "first_and_third_circular_moments_computed_by_exact_complex_powers",
        "epsilon_declares_the_existing_sine_cubic_static_countermodel",
        "right_minus_left_signed_source_and_incident_potential_differences",
        "first_moment_match_does_not_assert_full_state_or_future_equivalence",
        "obstruction_requires_exact_equal_first_moment_and_nonzero_source_difference",
        "no_obstruction_for_this_pair_is_not_a_universal_sufficiency_certificate",
        "128_neighbor_budget_is_computational_not_a_physical_degree_bound",
        "no_graph_capture_pressure_law_installation_flow_or_physical_identification",
    )

    def to_dict(self):
        """Project exact declared coordinates and their detached static evidence."""
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.phase-moment-information.v1",
            "report": _project(self),
        }


def assess_phase_moment_information(
    left_phasors, right_phasors, *, epsilon=1
) -> PhaseMomentInformationAssessment:
    """Test first-resultant information against the declared cubic sine source.

    Both nonempty ordered collections contain at most 128 exact unit phasors
    (c,s), with equal degree d. They describe relative gaps, not absolute phase
    measurements. The existing countermodel has pi*d*P=sum(s+epsilon*s**3)
    and incident potential sum(1-c+epsilon*(2/3-c+c**3/3)), epsilon>=0.
    The sine source and cosine potential are retained as controls. This
    static readout neither supplies companion evolution rows nor selects a
    unique pressure law. A zero epsilon is an admitted equal-law control.

    Exact agreement of degree and Z1, together with unequal cubic sources,
    disproves first-resultant sufficiency for that declared source. Overlapping
    numerical enclosures never replace the exact equality test. A comparison
    without such a collision establishes no universal information theorem.
    """
    coefficient = exact_or_represented_real(epsilon, "epsilon")
    if coefficient < 0:
        raise ValueError("epsilon must be nonnegative")

    left = _admit_relative_phasors(left_phasors, "left_phasors")
    right = _admit_relative_phasors(right_phasors, "right_phasors")
    if len(left) != len(right):
        raise ValueError("the two phase multisets must have equal degree")
    degree = len(left)
    first, third, sine, cubic, cosine_storage, cubic_storage = [], [], [], [], [], []
    for phasors in (left, right):
        z1, z3 = _phase_moments(phasors)
        first.append(z1)
        third.append(z3)
        values = _phase_source_storage(z1, z3, degree, coefficient)
        for column, value in zip((sine, cubic, cosine_storage, cubic_storage), values):
            column.append(value)
    same = first[0] == first[1]
    difference = cubic[1] - cubic[0]
    return PhaseMomentInformationAssessment(
        left_phasors=left,
        right_phasors=right,
        degree=degree,
        epsilon=coefficient,
        first_resultants=tuple(first),
        third_resultants=tuple(third),
        sine_source_pi_numerators=tuple(sine),
        cubic_source_pi_numerators=tuple(cubic),
        cosine_storage_sums=tuple(cosine_storage),
        cubic_storage_sums=tuple(cubic_storage),
        first_resultants_equal=same,
        nonzero_first_resultants=tuple(any(value != 0 for value in z) for z in first),
        strictly_acute=tuple(all(c > 0 for c, _ in row) for row in (left, right)),
        cubic_source_difference_pi_numerator=difference,
        cubic_source_difference_bounds=I(difference) / pi_interval(),
        cosine_storage_difference=cosine_storage[1] - cosine_storage[0],
        cubic_storage_difference=cubic_storage[1] - cubic_storage[0],
        sufficiency_obstruction=same and difference != 0,
    )


@dataclass(frozen=True)
class PhaseInformationResponse:
    """Finite root-form predictions for two equal-information preparations.

    Model-indexed tuples put sine first and the supplied cubic member second.
    Within each model, initial root rates put the left preparation first.
    Preparation error bounds every initial form and lifted radian coordinate;
    observation error bounds each endpoint root reading independently.
    """

    information: PhaseMomentInformationAssessment
    duration: Fraction
    preparation_error: Fraction
    observation_error: Fraction
    model_epsilon_values: tuple[Fraction, Fraction]
    root_initial_rate_values: tuple[tuple[Fraction, Fraction], ...]
    current_abs_upper_bounds: tuple[Fraction, Fraction]
    current_lipschitz_upper_bounds: tuple[Fraction, Fraction]
    form_contrast_centers: tuple[Fraction, Fraction]
    finite_time_error_upper_bounds: tuple[Fraction, Fraction]
    preparation_error_upper_bounds: tuple[Fraction, Fraction]
    observation_error_upper_bound: Fraction
    form_contrast_radii: tuple[Fraction, Fraction]
    form_contrast_prediction_bounds: tuple[I, I]
    separation_margin_lower_bound: Fraction
    discrimination_certified: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    observable: str = "x_root_right(duration)-x_root_left(duration)"
    complete_rows: tuple[str, ...] = (
        "dx_i/dtau=sum_j j(theta_j-theta_i)/d_i",
        "dtheta_i/dtau=x_i-sum_j x_j/d_i",
        "j(delta)=sin(delta)+eta*sin(delta)^3; eta=0 or epsilon",
        "fixed simple unit star; held unit capacities; e=0,w=beta=1; no inputs or events",
    )
    scope: tuple[str, ...] = (
        "independently_declared_equal_degree_and_exact_equal_first_moment_preparations",
        "root_first_node_order_followed_by_the_ordered_incident_unit_phasors",
        "ideal_forms_zero_root_phase_zero_and_leaf_phases_given_on_the_circle",
        "all_full_star_form_and_phase_coordinates_evolve_under_each_complete_law",
        "independent_initial_form_and_lifted_radian_error_at_every_node_in_each_trial",
        "each_endpoint_root_reading_has_its_own_absolute_observation_error",
        "one_fixed_common_model_clock_and_duration_for_both_preparations_and_laws",
        "global_current_and_derivative_bounds_control_the_whole_continuous_window",
        "equal_initial_sine_sources_do_not_assert_equal_finite_sine_responses",
        "cubic_current_uses_its_own_supplied_phase_potential_and_conserved_full_storage",
        "overlapping_prediction_intervals_are_unavailable_not_equivalent_models",
        "no_sampled_pressure_reconstruction_trajectory_replay_or_graph_mutation",
        "no_clock_calibration_physical_source_admission_or_fundamental_law_selection",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.phase-information-response.v1",
            "report": _project(self),
        }


def assess_phase_information_response(
    left_phasors,
    right_phasors,
    *,
    duration,
    epsilon=1,
    preparation_error=0,
    observation_error=0,
) -> PhaseInformationResponse:
    """Bound a finite root-form contrast from independently supplied preparation.

    Rebuild exact static information from primitive unit phasors. Equal degree
    and Z1 are required; positive epsilon declares the existing cubic current
    and its own potential. Both complete conservative star laws use held unit
    capacities and tau=t/pi, with zero ideal forms and no forcing or events.
    Phases are not held fixed. No observed response enters the prediction.

    For eta=0 or epsilon put J=1+eta, L=1+3*eta, h=duration and
    rho=preparation_error. Globally |j|<=J and |j'|<=L. Every actual
    coordinate satisfies |x(tau)|<=rho+J*tau, hence each phase moves by at
    most 2*rho*tau+J*tau**2. Relative departure from its ideal initial gap
    is at most 2*rho+4*rho*tau+2*J*tau**2. Integrating the root row gives
    the per-root error rho*(1+2*L*h+2*L*h**2)+(2/3)*L*J*h**3 about its
    independently predicted h*mean(j). Two preparations double this bound;
    two endpoint readings add 2*observation_error. Errors refer to initial
    lifted phase angles, not coordinate errors in the supplied phasor pairs.

    Outward intervals certify separation in either signed direction. Failed
    separation returns unavailable, never physical equivalence or rejection.
    The clock and all remaining complete-law premises are supplied, not
    selected by the first-moment classification or by this finite bound.
    """
    h, coefficient, rho, sigma = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("duration", duration),
            ("epsilon", epsilon),
            ("preparation_error", preparation_error),
            ("observation_error", observation_error),
        )
    )
    if h <= 0 or coefficient <= 0 or rho < 0 or sigma < 0:
        raise ValueError(
            "positive duration/epsilon and nonnegative preparation/observation errors required"
        )
    information = assess_phase_moment_information(
        left_phasors, right_phasors, epsilon=coefficient
    )
    if not information.first_resultants_equal:
        raise ValueError("the two preparations must have equal first resultants")
    coefficients = (Fraction(0), coefficient)
    currents = tuple(1 + eta for eta in coefficients)
    lipschitz = tuple(1 + 3 * eta for eta in coefficients)
    rates = (
        information.sine_source_pi_numerators,
        information.cubic_source_pi_numerators,
    )
    centers = tuple(h * (right - left) for left, right in rates)
    finite_errors = tuple(
        Fraction(4, 3) * bound * current * h**3
        for bound, current in zip(lipschitz, currents)
    )
    preparation_errors = tuple(
        2 * rho * (1 + 2 * bound * h + 2 * bound * h**2) for bound in lipschitz
    )
    reading_error = 2 * sigma
    radii = tuple(
        finite + preparation + reading_error
        for finite, preparation in zip(finite_errors, preparation_errors)
    )
    predictions = tuple(
        I(center - radius, center + radius) for center, radius in zip(centers, radii)
    )
    separation = max(
        predictions[0].lo - predictions[1].hi,
        predictions[1].lo - predictions[0].hi,
    )
    certified = separation > 0
    return PhaseInformationResponse(
        information=information,
        duration=h,
        preparation_error=rho,
        observation_error=sigma,
        model_epsilon_values=coefficients,
        root_initial_rate_values=rates,
        current_abs_upper_bounds=currents,
        current_lipschitz_upper_bounds=lipschitz,
        form_contrast_centers=centers,
        finite_time_error_upper_bounds=finite_errors,
        preparation_error_upper_bounds=preparation_errors,
        observation_error_upper_bound=reading_error,
        form_contrast_radii=radii,
        form_contrast_prediction_bounds=predictions,
        separation_margin_lower_bound=separation,
        discrimination_certified=certified,
        status="certified" if certified else "unavailable",
        reasons=() if certified else ("finite_response_intervals_not_separated",),
    )


@dataclass(frozen=True)
class PhaseMomentMotion:
    """Derived moment motion for exact gaps paired with supplied angular rates."""

    phasors: tuple[tuple[Fraction, Fraction], ...]
    gap_rates: tuple[Fraction, ...]
    epsilon: Fraction
    degree: int
    first_resultant: tuple[Fraction, Fraction]
    third_resultant: tuple[Fraction, Fraction]
    first_rate_weighted_resultant: tuple[Fraction, Fraction]
    third_rate_weighted_resultant: tuple[Fraction, Fraction]
    first_resultant_rate: tuple[Fraction, Fraction]
    third_resultant_rate: tuple[Fraction, Fraction]
    sine_source_pi_numerator: Fraction
    cubic_source_pi_numerator: Fraction
    cosine_storage_sum: Fraction
    cubic_storage_sum: Fraction
    sine_source_rate_pi_numerator: Fraction
    cubic_source_rate_pi_numerator: Fraction
    cosine_storage_rate: Fraction
    cubic_storage_rate: Fraction
    scope: tuple[str, ...] = (
        "exact_relative_unit_phasors_with_paired_supplied_radian_rates_in_one_declared_clock",
        "same_unweighted_equal_gain_incident_source_and_potential_as_phase_moment_information",
        "fixed_incidence_and_epsilon_on_the_declared_differentiable_segment",
        "source_and_source_rate_numerators_include_the_common_mathematical_pi_factor",
        "storage_and_work_are_incident_sums_not_whole_network_energy_or_balance",
        "rate_weighted_moments_are_derived_observables_not_independent_state_or_new_laws",
        "rates_require_their_own_complete_law_or_independent_kinematic_provenance",
        "current_and_rate_equality_do_not_establish_future_closure_or_finite_equivalence",
        "no_angle_reconstruction_no_measured_phasor_renormalization_no_graph_or_state_mutation",
        "no_trajectory_capacity_law_clock_identification_or_physical_validation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.phase-moment-motion.v1", "report": _project(self)}


def derive_phase_moment_motion(phasors, gap_rates, *, epsilon=1) -> PhaseMomentMotion:
    """Differentiate the admitted sine/cubic current and incident potential.

    Each rate is delta_dot in radians per the caller's declared time, paired
    with its own exact relative unit phasor. For Z_m=sum exp(i*m*delta) and
    M_m=sum delta_dot*exp(i*m*delta), Z_m_dot=i*m*M_m. The existing pressure
    convention is pi*d*p=sum(sin(delta)+epsilon*sin(delta)^3).

    Return exact numerators pi*p and pi*p_dot, and the incident potential and
    its rate. Supplying a rate does not derive its law. These instantaneous
    observations do not generally determine the mixed moments and accelerations
    needed to evolve. A separately admitted closure, such as the regular star
    chart below, needs its complete law and domain; none is inferred here.
    """
    from .relational_observations import _ordered

    phase = _admit_relative_phasors(phasors, "phasors")
    raw_rates = _ordered(gap_rates, "gap_rates", limit=len(phase) + 1)
    if len(raw_rates) != len(phase):
        raise ValueError("one ordered gap rate is required for each phasor")
    rates = tuple(exact_or_represented_real(value, "gap_rates") for value in raw_rates)
    coefficient = exact_or_represented_real(epsilon, "epsilon")
    if coefficient < 0:
        raise ValueError("epsilon must be nonnegative")
    first, third = _phase_moments(phase)
    motion1, motion3 = _phase_moments(phase, rates)
    sine, cubic, storage, cubic_storage = _phase_source_storage(
        first, third, len(phase), coefficient
    )
    scale = 1 + 3 * coefficient / 4
    return PhaseMomentMotion(
        phasors=phase,
        gap_rates=rates,
        epsilon=coefficient,
        degree=len(phase),
        first_resultant=first,
        third_resultant=third,
        first_rate_weighted_resultant=motion1,
        third_rate_weighted_resultant=motion3,
        first_resultant_rate=(-motion1[1], motion1[0]),
        third_resultant_rate=(-3 * motion3[1], 3 * motion3[0]),
        sine_source_pi_numerator=sine,
        cubic_source_pi_numerator=cubic,
        cosine_storage_sum=storage,
        cubic_storage_sum=cubic_storage,
        sine_source_rate_pi_numerator=motion1[0] / len(phase),
        cubic_source_rate_pi_numerator=(
            scale * motion1[0] - 3 * coefficient * motion3[0] / 4
        )
        / len(phase),
        cosine_storage_rate=motion1[1],
        cubic_storage_rate=scale * motion1[1] - coefficient * motion3[1] / 4,
    )


@dataclass(frozen=True)
class SineStarMomentClosure:
    """Exact regular observation chart of the conservative three-node star.

    Complex quantities are (real, imaginary) Fraction pairs. Rates use only
    tau=t/pi, unit held capacity, e=0 and w=beta=1. This is the full relative
    state modulo simultaneous leaf exchange, not a lower-dimensional model.
    """

    first_resultant: tuple[Fraction, Fraction]
    first_rate_weighted_resultant: tuple[Fraction, Fraction]
    resultant_norm_squared: Fraction
    relative_mean_form: Fraction
    internal_form_squared: Fraction
    phase_product: tuple[Fraction, Fraction]
    first_resultant_rate: tuple[Fraction, Fraction]
    first_rate_weighted_resultant_rate: tuple[Fraction, Fraction]
    full_storage: Fraction
    scope: tuple[str, ...] = (
        "complete_three_node_star_unit_support_and_held_unit_capacities",
        "conservative_sine_e_zero_w_beta_one_no_inputs_or_events",
        "gap_rates_and_all_reported_derivatives_use_tau_equals_t_over_pi",
        "strict_regular_chart_zero_less_than_resultant_norm_squared_less_than_four",
        "faithful_relative_state_modulo_common_origins_and_simultaneous_leaf_swap",
        "four_real_coordinates_no_continuous_degree_of_freedom_removed",
        "exact_observable_inputs_may_lift_to_irrational_unit_phasors",
        "local_chart_not_an_all_time_domain_or_a_numerical_trajectory_certificate",
        "singular_observations_require_retained_state_not_tolerance_clipping",
        "no_fundamental_law_selection_graph_mutation_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-star-moment-closure.v1", "report": _project(self)}


def derive_sine_star_moment_closure(
    first_resultant, first_rate_weighted_resultant
) -> SineStarMomentClosure:
    """Push forward the complete sine star law in its regular moment chart.

    Z=sum exp(i*delta_j), M=sum delta_j'*exp(i*delta_j), where delta_j is
    relative to the root and prime means d/d(t/pi). Each input is an ordered
    (real, imaginary) pair of finite represented reals. Require 0<|Z|^2<4;
    coincident/antipodal phases lose state information and are rejected here,
    though the fine nodal law remains smooth. No uncertainty is inferred.

    With a+i*b=M/Z, recover X=a/2 and U=b^2*|Z|^2/(4-|Z|^2). The unordered
    phase product is P=Z/conj(Z), and Z2=Z^2-2P. The inherited exact rows are
    Z'=i*M and M'=i*(Z2-2)/2-3*Im(Z)*Z/2+i*((U-a^2)*Z+2*a*M).
    Full storage is X^2+U+2-Re(Z), not just incident phase storage.
    This specialization reuses the retained pair-state proof; its coefficients
    do not transfer to other attachments, capacities, clocks or laws.
    """
    from .relational_observations import _ordered

    pairs = []
    for values, label in (
        (first_resultant, "first_resultant"),
        (first_rate_weighted_resultant, "first_rate_weighted_resultant"),
    ):
        row = _ordered(values, label, limit=3)
        if len(row) != 2:
            raise ValueError(f"{label} requires one real/imaginary pair")
        pairs.append(tuple(exact_or_represented_real(v, label) for v in row))
    z, m = pairs
    c, s = z
    mr, mi = m
    norm = c * c + s * s
    if not 0 < norm < 4:
        raise ValueError("regular star moment chart requires 0 < |Z|^2 < 4")
    a, b = (mr * c + mi * s) / norm, (mi * c - mr * s) / norm
    x = a / 2
    u2 = b * b * norm / (4 - norm)
    product = ((c * c - s * s) / norm, 2 * c * s / norm)
    z2 = (c * c - s * s - 2 * product[0], 2 * c * s - 2 * product[1])
    rate2 = ((u2 - a * a) * c + 2 * a * mr, (u2 - a * a) * s + 2 * a * mi)
    mdot = (
        -z2[1] / 2 - 3 * s * c / 2 - rate2[1],
        (z2[0] - 2) / 2 - 3 * s * s / 2 + rate2[0],
    )
    return SineStarMomentClosure(
        first_resultant=z,
        first_rate_weighted_resultant=m,
        resultant_norm_squared=norm,
        relative_mean_form=x,
        internal_form_squared=u2,
        phase_product=product,
        first_resultant_rate=(-mi, mr),
        first_rate_weighted_resultant_rate=mdot,
        full_storage=x * x + u2 + 2 - c,
    )


@dataclass(frozen=True)
class PhaseLockSourceObservation:
    """Detached sine-lock residuals and separately observed phasor pressure.

    ``full_support_sine`` and ``relative_real`` are exact sums of materialized
    ``math.sin`` and ``math.cos`` coefficients, not exact transcendental
    evaluations. Phase rates use their unweighted support mean. ``common_rate``
    is the degree-weighted capacity candidate for a fully admitted reciprocal
    lock, not a measured common rate or proof that a lock exists.

    ``sine_reduction_defect`` is in angular-rate units: K/admitted_degree
    times the difference between the producer's represented ``math.fsum``
    and the exact sum of its represented terms. It excludes later product,
    addition, integration and modulo rounding. Both rate-residual fields are
    exact rational references to coefficients, not binary64 step derivatives.

    Acuity and source estimates are ordinary numerical observations. The
    relative phasor uses raw differences; production pressure separately uses
    its absolute phasor kernel and certified two-neighbor override. Their
    difference remains visible even at a nearly locked state. No tolerance
    certifies an exact phase lock, positive resultant or future invariant.
    """

    capture: NonEpiForcingObservation
    coupling_strength: Fraction
    phase_degrees: Vector
    common_rate: Fraction
    full_support_sine: Vector
    relative_real: Vector
    full_support_rate_residual: Vector
    actual_admitted_rate_residual: Vector
    sine_reduction_defect: Vector
    full_u3_admission: bool
    strict_acute_edges_estimate: bool
    relative_phase_source_estimate: tuple[float | None, ...]
    production_source_discrepancy: tuple[Fraction | None, ...]
    lock_source_estimate: tuple[float | None, ...]
    other_forcing: Vector
    homogeneous_capacity: bool
    positive_capacities: bool
    positive_transport_connected: bool
    scope: tuple[str, ...] = (
        "connected_reciprocal_unique_phase_support_and_canonical_raw_phase_chart",
        "exact_rational_residuals_of_materialized_sine_coefficients",
        "configured_omega_equals_capacity_phase_law_with_supplied_positive_K",
        "partial_U3_admission_keeps_its_own_mean_and_residual",
        "numerical_acuity_and_phasor_estimates_are_not_transcendental_enclosures",
        "production_phase_source_and_nonphase_forcing_remain_separate",
        "no_tolerance_lock_test_capacity_solver_target_solver_or_graph_write",
        "no_admission_of_Gamma_events_clipping_controllers_or_future_evolution",
    )


def observe_phase_lock_source(
    graph, *, coupling_strength
) -> PhaseLockSourceObservation:
    """Compare the configured sine-lock equation with actual phase pressure.

    On reciprocal connected unique support with degrees d, full U3 admission
    gives the candidate Omega=sum(d*nu)/sum(d). Return the exact represented
    residual ``nu + K*sum(sin(theta_j-theta_i))/d - Omega``. The admitted-row
    residual uses the actual U3 subset and its own denominator; an empty
    admitted row retains free angular rate nu. Neither residual is thresholded.

    Reuse the shared forcing capture and U3 gate without executing callbacks or
    changing graph-owned state. Canonical raw phases must lie in [0, 2*pi),
    matching the selected represented chart. Sine coefficients use raw phase
    subtraction exactly as the configured phase producer does, not wrapped
    displacement. Zero-weight support links remain in both phase equations.

    For relative phasor R+i*S, report atan2(S,R)/pi where its represented sums
    are not jointly zero. Under full admission and numerically strict acute
    edges, also report the conditional lock-source estimate
    ``atan((Omega-nu)/(K*(R/d)))/pi`` when R>0. This formula presumes an exact
    ideal lock; a small observed residual does not establish that premise.
    ``production_source_discrepancy`` is actual captured phase pressure minus
    the relative-phasor estimate. Undefined estimates remain None.

    Capacity and topology forcing are retained together as ``other_forcing``;
    a nonzero source therefore cannot silently be attributed only to phase.
    EPI transport uses conductance strengths, not these support degrees.
    Positive transport connectivity is reported separately for consumers of
    the existing forced-support theorem. No stationary profile is solved here.
    """
    from ..operators._phase_gate import resolve_u3_phase_neighbors
    from ..utils import angle_diff

    capture = capture_non_epi_forcing(graph)
    source = capture.snapshot
    support = source.support_neighbors
    size = len(source.nodes)
    if not size or any(not row for row in support):
        raise ValueError(
            "phase lock observation requires nonempty support at every node"
        )
    if any(i not in support[j] for i, row in enumerate(support) for j in row):
        raise ValueError("phase lock observation requires reciprocal support")

    def connected(rows):
        reached, pending = {0}, [0]
        while pending:
            for j in rows[pending.pop()]:
                if j not in reached:
                    reached.add(j)
                    pending.append(j)
        return len(reached) == size

    if not connected(support):
        raise ValueError("phase lock observation requires connected phase support")
    if any(not 0 <= value < Fraction.from_float(math.tau) for value in capture.phase):
        raise ValueError(
            "phase lock observation requires canonical raw phases in [0, 2*pi)"
        )
    _, strength = finite_represented_real(coupling_strength, "coupling_strength")
    if strength <= 0:
        raise ValueError("coupling_strength must be positive")

    degrees = tuple(Fraction(len(row)) for row in support)
    omega = dot(degrees, source.capacity) / sum(degrees)
    phase = tuple(map(float, capture.phase))
    index = {node: i for i, node in enumerate(source.nodes)}
    phase_by_node = dict(zip(source.nodes, phase, strict=True))
    sine, real, full_residual, admitted_residual, reduction_defect = [], [], [], [], []
    full_admission, acute = True, True
    for i, (node, row) in enumerate(zip(source.nodes, support, strict=True)):
        raw_terms = {j: math.sin(phase[j] - phase[i]) for j in row}
        imag = sum(
            (Fraction.from_float(value) for value in raw_terms.values()), Fraction(0)
        )
        real_part = sum(
            (Fraction.from_float(math.cos(phase[j] - phase[i])) for j in row),
            Fraction(0),
        )
        sine.append(imag)
        real.append(real_part)
        full_residual.append(source.capacity[i] + strength * imag / degrees[i] - omega)
        acute = acute and all(
            abs(angle_diff(phase[j], phase[i])) < math.pi / 2 for j in row
        )
        neighbors = tuple(graph.neighbors(node))
        gate = resolve_u3_phase_neighbors(
            graph.graph,
            phase[i],
            neighbors,
            phase_getter=phase_by_node.__getitem__,
            operator_code="UM",
            require_compatible=False,
        )
        full_admission = full_admission and gate.neighbors == neighbors
        admitted = tuple(raw_terms[index[neighbor]] for neighbor in gate.neighbors)
        admitted_sum = sum(map(Fraction.from_float, admitted), Fraction(0))
        divisor = len(admitted)
        admitted_residual.append(
            source.capacity[i]
            + (strength * admitted_sum / divisor if divisor else 0)
            - omega
        )
        reduction_defect.append(
            strength
            * (Fraction.from_float(math.fsum(admitted)) - admitted_sum)
            / divisor
            if divisor
            else Fraction(0)
        )

    relative_estimate = tuple(
        (
            math.atan2(float(imag), float(real_part)) / math.pi
            if imag or real_part
            else None
        )
        for imag, real_part in zip(sine, real, strict=True)
    )
    discrepancy = tuple(
        actual - Fraction.from_float(estimate) if estimate is not None else None
        for actual, estimate in zip(
            capture.phase_gradient, relative_estimate, strict=True
        )
    )
    lock_estimate = []
    for nu, real_part, degree in zip(source.capacity, real, degrees, strict=True):
        if not (full_admission and acute and real_part > 0):
            lock_estimate.append(None)
            continue
        # Form the exact coefficient ratio before conversion: separately
        # materializing numerator and denominator can erase an O(1) ratio
        # when both are subnormal. The reciprocal branch avoids overflow.
        # The final arctangent remains a numerical estimate, not an enclosure.
        ratio = (omega - nu) * degree / (strength * real_part)
        if abs(ratio) <= 1:
            angle = math.atan(float(ratio))
        else:
            angle = math.pi / 2 - math.atan(float(1 / abs(ratio)))
            if ratio < 0:
                angle = -angle
        lock_estimate.append(angle / math.pi)
    components = dict(decompose_non_epi_forcing(capture))
    other = tuple(
        a + b for a, b in zip(components["vf"], components["topo"], strict=True)
    )
    transport = [set() for _ in range(size)]
    for i, j, _ in source.conductance:
        transport[i].add(j)
    return PhaseLockSourceObservation(
        capture=capture,
        coupling_strength=strength,
        phase_degrees=degrees,
        common_rate=omega,
        full_support_sine=tuple(sine),
        relative_real=tuple(real),
        full_support_rate_residual=tuple(full_residual),
        actual_admitted_rate_residual=tuple(admitted_residual),
        sine_reduction_defect=tuple(reduction_defect),
        full_u3_admission=full_admission,
        strict_acute_edges_estimate=acute,
        relative_phase_source_estimate=relative_estimate,
        production_source_discrepancy=discrepancy,
        lock_source_estimate=tuple(lock_estimate),
        other_forcing=other,
        homogeneous_capacity=len(set(source.capacity)) == 1,
        positive_capacities=all(value > 0 for value in source.capacity),
        positive_transport_connected=connected(transport),
    )


def _ordered(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered sequence")
    try:
        return tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an ordered sequence") from error


def _unit_planar_gram(values):
    gram = tuple(
        ordered_vector(row, "cosine Gram row")
        for row in _ordered(values, "cosine_gram")
    )
    n = len(gram)
    if not n or any(len(row) != n for row in gram):
        raise ValueError("cosine Gram must be a nonempty square matrix")
    if any(gram[i][i] != 1 for i in range(n)):
        raise ValueError("unit cosine Gram requires diagonal one")
    if any(gram[i][j] != gram[j][i] for i in range(n) for j in range(n)):
        raise ValueError("cosine Gram must be symmetric")
    # Unit first vector fixes one axis. The Schur complement must be a
    # positive semidefinite rank-at-most-one Gram on the remaining axis.
    residual = tuple(
        tuple(gram[i][j] - gram[i][0] * gram[0][j] for j in range(n)) for i in range(n)
    )
    if any(residual[i][i] < 0 for i in range(n)):
        raise ValueError("cosine Gram must be positive semidefinite and planar")
    pivot = next((i for i in range(n) if residual[i][i] > 0), None)
    if pivot is None:
        valid = not any(value for row in residual for value in row)
    else:
        valid = all(
            residual[i][j] * residual[pivot][pivot]
            == residual[i][pivot] * residual[pivot][j]
            for i in range(n)
            for j in range(n)
        )
    if not valid:
        raise ValueError(
            "cosine Gram must be positive semidefinite with rank at most two"
        )
    return gram


def _index_rows(values, n, label):
    rows = tuple(_ordered(row, label) for row in _ordered(values, label))
    if len(rows) != n:
        raise ValueError(f"{label} rows must match the Gram dimension")
    for row in rows:
        if (
            not row
            or any(type(i) is not int or not 0 <= i < n for i in row)
            or len(set(row)) != len(row)
        ):
            raise ValueError(
                f"{label} requires nonempty rows of distinct valid indices"
            )
    return rows


@dataclass(frozen=True)
class PhaseResponseReference:
    """One exact phase Jacobian from declared mean and receiving-source rows."""

    cosine_gram: Matrix
    mean_neighbors: tuple
    receiver_sources: tuple
    phase_factor: Fraction
    mean_response: Matrix
    mean_resultant_squared: Vector
    receiver_response: Matrix
    jacobian: Matrix
    row_sum_residuals: Vector
    is_nonnegative: bool


def derive_phase_response(
    *,
    cosine_gram,
    mean_neighbors,
    receiver_sources,
    phase_factor,
) -> PhaseResponseReference:
    """Differentiate unweighted phasor means and average receiver proposals.

    For a nonzero S_i=sum_{j in N_i} exp(i*theta_j), differentiation gives
    d Arg(S_i)/d theta_j = Re(exp(i*theta_j)/S_i). A supplied unit planar
    cosine Gram G_jk=cos(theta_j-theta_k) realizes this coefficient exactly as
    1[j in N_i]*sum_{k in N_i}G_jk / sum_{l,k in N_i}G_lk. Its rows sum to
    one but may contain negative entries. Zero resultants are rejected.

    Each receiver i averages the phase proposals from its declared sources
    T_i. In any regular fixed shortest-arc chart the Jacobian is
    (1-factor)*I + factor*mean_{s in T_i} R_s. Pointwise IL uses T_i=(i,);
    all-target bidirectional UM must supply its actual contributing sources,
    with the target INCLUDED in each source mean. Nonzero baseline proposal
    displacements need not vanish individually for this derivative to hold.

    Input Gram data must be exactly symmetric, unit, positive semidefinite
    and rank <=2. A rounded cosine table may fail those identities and is
    never repaired into an apparent theorem. Exact rational inputs and other
    supported represented reals use the shared scalar reader. Factors in
    [0,1] are detached algebraic inputs, not live operator admission.

    No phase angles, orientation, fixed point, U3 support or local chart are
    inferred. This is not the derivative of binary64 arithmetic. Nodal EPI
    pressure uses its own factor and support, even when sharing the mean
    derivative. A nonnegative Jacobian alone is not strict contraction.
    """
    gram = _unit_planar_gram(cosine_gram)
    n = len(gram)
    neighbors = _index_rows(mean_neighbors, n, "mean_neighbors")
    receivers = _index_rows(receiver_sources, n, "receiver_sources")
    factor = exact_or_represented_real(phase_factor, "phase_factor")
    if not 0 <= factor <= 1:
        raise ValueError("phase_factor must lie in [0,1]")
    squared = tuple(
        sum((gram[j][k] for j in row for k in row), Fraction(0)) for row in neighbors
    )
    if any(value <= 0 for value in squared):
        raise ValueError("every declared phasor mean requires a nonzero resultant")
    mean = tuple(
        tuple(
            (
                sum((gram[j][k] for k in row), Fraction(0)) / squared[i]
                if j in row
                else Fraction(0)
            )
            for j in range(n)
        )
        for i, row in enumerate(neighbors)
    )
    received = tuple(
        tuple(
            sum((mean[s][j] for s in sources), Fraction(0)) / len(sources)
            for j in range(n)
        )
        for sources in receivers
    )
    jacobian = tuple(
        tuple((1 - factor) * int(i == j) + factor * received[i][j] for j in range(n))
        for i in range(n)
    )
    residuals = tuple(sum(row, Fraction(0)) - 1 for row in jacobian)
    if any(residuals) or any(sum(row, Fraction(0)) != 1 for row in mean):
        raise RuntimeError("exact phase derivative lost rotation covariance")
    return PhaseResponseReference(
        gram,
        neighbors,
        receivers,
        factor,
        mean,
        squared,
        received,
        jacobian,
        residuals,
        all(value >= 0 for row in jacobian for value in row),
    )


@dataclass(frozen=True)
class PhaseSourceGeometry:
    """Conditional exact tangent geometry of the canonical phase source.

    On a regular shortest-arc chart, Dg=(R-I)/pi. The stored matrix is R-I,
    whose kernel and rank equal those of Dg. A one-dimensional kernel allows
    only common rotation; a larger tangent space does not prove a finite
    source-preserving path. These detached data do not verify an actual phase
    chart, U3 gates, a dynamical law, stability or runtime provenance.
    """

    reference: PhaseResponseReference
    scaled_source_jacobian: Matrix
    rank: int
    tangent_dimension: int
    mean_is_nonnegative: bool

    @property
    def only_common_rotation(self) -> bool:
        """Whether the conditional tangent kernel is exactly the rotation line."""
        return self.tangent_dimension == 1


def observe_phase_source_geometry(reference) -> PhaseSourceGeometry:
    """Read exact fixed-source tangent freedom from the shared mean derivative.

    Rebuild the supplied reference from its primitive Gram and incidence data
    before using any cached coefficient. This observes mean_response R, not
    the final merged operator-stage Jacobian: a zero stage factor must not
    turn a rigid source into an apparent identity map. Signed mean responses
    are supported; nonnegativity alone is neither necessary nor sufficient
    for one-dimensional tangent freedom. See FORCED_SUPPORT_BALANCE section 24
    for the nonnegative irreducible theorem and its regular-chart hypotheses.
    """
    if type(reference) is not PhaseResponseReference:
        raise TypeError("phase source geometry requires a PhaseResponseReference")
    rebuilt = derive_phase_response(
        cosine_gram=reference.cosine_gram,
        mean_neighbors=reference.mean_neighbors,
        receiver_sources=reference.receiver_sources,
        phase_factor=reference.phase_factor,
    )
    if rebuilt != reference:
        raise ValueError(
            "phase response reference differs from its rebuilt coefficients"
        )
    mean = rebuilt.mean_response
    size = len(mean)
    jacobian = tuple(
        tuple(value - int(i == j) for j, value in enumerate(row))
        for i, row in enumerate(mean)
    )
    rank = exact_rank(jacobian)
    if rank >= size:
        raise RuntimeError("phase source derivative lost common-rotation invariance")
    return PhaseSourceGeometry(
        rebuilt,
        jacobian,
        rank,
        size - rank,
        all(value >= 0 for row in mean for value in row),
    )


@dataclass(frozen=True)
class JointLinearSample:
    """One numerical tangent prediction with separate ideal nonlinear bounds.

    Nodal and modal float tuples follow the prediction's source order and
    spectral column order, respectively. Phase is the lifted offset from the
    initial degree-weighted origin and common rotation, not wrapped phase.
    The rational sup-norm bounds exclude spectral and runtime arithmetic error.
    """

    step: int
    epi: tuple[float, ...]
    phase_offset: tuple[float, ...]
    epi_modes: tuple[float, ...]
    phase_modes: tuple[float, ...]
    phase_error_upper: Fraction
    epi_error_upper: Fraction


@dataclass(frozen=True)
class SynchronizedJointPrediction:
    """Frozen local joint prediction from supplied fixed support and laws.

    ``right_modes`` is row-major D^(-1/2)V, where V contains the existing
    symmetric-normalized Laplacian eigenvectors. The modal projection is
    V^T D^(1/2), so degree weights are not discarded on irregular graphs.
    Public fields do not authenticate a trajectory or a future observation.
    """

    capture: NonEpiForcingObservation
    dt: Fraction
    coupling_strength: Fraction
    phase_origin: Fraction
    phase_width: Fraction
    degree_weights: Vector
    right_modes: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    initial_epi_modes: tuple[float, ...]
    initial_phase_modes: tuple[float, ...]
    phase_reference: PhaseResponseReference
    samples: tuple[JointLinearSample, ...]
    scope: str


def predict_synchronized_joint_euler(
    graph, *, dt, coupling_strength, steps
) -> SynchronizedJointPrediction:
    r"""Freeze the consensus-tangent phase/form response before observation.

    On connected unit support with common capacity kappa, effective weights
    e>0 and w>=0, and zero topology coefficient, the ideal local model is

        z'=-K*L_rw*z,  x'=-kappa*e*L_rw*x-kappa*w/pi*L_rw*z.

    Here z subtracts the initial degree-weighted phase origin and kappa*t.
    The exact all-ones phasor Gram derives the pressure tangent from the
    existing phase-response owner. Actual degree-three or higher phasor means
    remain nonlinear; the derivative is not substituted into their runtime.
    Closed modal Euler powers, including the finite convolution at equal
    clocks, evaluate this prediction without running a graph solver.

    Require canonical raw phases in a common lift of width rho<=1, admitted
    by U3, and h*K<=1/2, h*kappa*e<=1/2. For the unwrapped ideal nonlinear
    averaged-sine/fresh-phasor Euler composition these conditions preserve
    the width. With r=K*rho^3/6, the returned exact sup-norm comparison bounds
    are n*h*r for phase and

        kappa*w*(n*h*rho^3/(18*(1-rho^2/2))
                 +h^2*n*(n-1)*r/3)

    for form. They compare the ideal nonlinear and linear Euler models with
    identical initial data. Binary64 eigensystem, power, source, phase-wrap,
    pressure assembly and integration errors are separate; the numerical
    predictions are not enclosed by these bounds. Small represented spectral
    endpoint errors are retained rather than repaired into exact eigenvalues.

    This read-out excludes Gamma, events, clipping, changing capacity/support
    and controllers; their runtime configuration is not admitted. Limits of
    12 nodes and 256 steps, plus the capture owner's 100 directed support
    entries, are evaluation budgets, not physical thresholds. The input graph
    and its spectral caches remain untouched.
    """
    import networkx as nx
    import numpy as np

    from ..operators._phase_gate import resolve_u3_phase_limits
    from .structural_diffusion import structural_eigenmodes

    if (
        isinstance(steps, bool)
        or not isinstance(steps, Integral)
        or not 0 <= steps <= 256
    ):
        raise ValueError("steps must be an integer in [0,256]")
    _, h = finite_represented_real(dt, "dt")
    _, k = finite_represented_real(coupling_strength, "coupling_strength")
    if h <= 0 or k <= 0:
        raise ValueError("dt and coupling_strength must be positive")
    capture = capture_non_epi_forcing(graph)
    source = capture.snapshot
    size = len(source.nodes)
    if not 2 <= size <= 12:
        raise ValueError("joint prediction requires between 2 and 12 nodes")
    transport = {(i, j): weight for i, j, weight in source.conductance}
    support = {(i, j) for i, row in enumerate(source.support_neighbors) for j in row}
    if (
        set(transport) != support
        or any(i == j or value != 1 for (i, j), value in transport.items())
        or any((j, i) not in support for i, j in support)
        or any(len(row) != len(set(row)) for row in source.support_neighbors)
    ):
        raise ValueError(
            "joint prediction requires matching unit symmetric nonloop unique support"
        )
    reached, pending = {0}, [0]
    while pending:
        added = set(source.support_neighbors[pending.pop()]) - reached
        reached.update(added)
        pending.extend(added)
    if len(reached) != size:
        raise ValueError("joint prediction requires connected support")
    capacity = source.capacity[0]
    if capacity <= 0 or any(value != capacity for value in source.capacity):
        raise ValueError("joint prediction requires common positive capacity")
    weights = dict(capture.normalized_weights)
    e, w = capture.epi_weight, weights["phase"]
    if e <= 0:
        raise ValueError("joint prediction requires positive EPI weight")
    if weights["topo"] != 0:
        raise ValueError("joint prediction requires zero topology weight")
    phase = capture.phase
    if any(not 0 <= value < Fraction.from_float(math.tau) for value in phase):
        raise ValueError("joint phases must be canonical [0,2*pi) coordinates")
    width = max(phase) - min(phase)
    if width > 1:
        raise ValueError("joint raw phase lift must have width <= 1 radian")
    _, gate = resolve_u3_phase_limits(graph.graph, operator_code="UM")
    if width > Fraction.from_float(gate):
        raise ValueError("joint phase lift must be fully U3-admitted")
    if h * k > Fraction(1, 2) or h * capacity * e > Fraction(1, 2):
        raise ValueError("dt exceeds the joint monotone modal Euler ceiling")

    degree = tuple(Fraction(len(row)) for row in source.support_neighbors)
    origin = dot(degree, phase) / sum(degree, Fraction(0))
    reference = derive_phase_response(
        cosine_gram=tuple(tuple(Fraction(1) for _ in phase) for _ in phase),
        mean_neighbors=source.support_neighbors,
        receiver_sources=tuple((i,) for i in range(size)),
        phase_factor=1,
    )
    # Never inherit a live spectral cache, attributes, callbacks or state.
    detached = nx.Graph()
    detached.add_nodes_from(source.nodes)
    detached.add_edges_from(
        (source.nodes[i], source.nodes[j], {"weight": 1.0})
        for i, j in sorted(support)
        if i < j
    )

    def float_tuple(values, label):
        return tuple(
            finite_represented_real(value, f"{label}[{i}]")[0]
            for i, value in enumerate(values)
        )

    phase_step = finite_represented_real(h * k, "phase Euler coefficient")[0]
    form_step = finite_represented_real(h * capacity * e, "form Euler coefficient")[0]
    source_step = finite_represented_real(
        h * capacity * w / Fraction.from_float(math.pi),
        "phase-to-form Euler coefficient",
    )[0]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            eigenvalues, vectors = structural_eigenmodes(detached)
            lambdas = float_tuple(eigenvalues, "eigenvalue")
            root_degree = np.sqrt(np.asarray(tuple(map(float, degree))))
            right = vectors / root_degree[:, None]
            right_modes = tuple(float_tuple(row, "right mode") for row in right)
            x0 = np.asarray(float_tuple(source.epi, "initial EPI"))
            z0 = np.asarray(
                float_tuple((value - origin for value in phase), "phase offset")
            )
            epi_modes = float_tuple(vectors.T @ (root_degree * x0), "initial EPI mode")
            phase_modes = float_tuple(
                vectors.T @ (root_degree * z0), "initial phase mode"
            )
            form_factors = tuple(1 - form_step * value for value in lambdas)
            phase_factors = tuple(1 - phase_step * value for value in lambdas)
            samples = []
            residual = k * width**3 / 6
            for n in range(int(steps) + 1):
                z_modes = float_tuple(
                    (a**n * value for a, value in zip(phase_factors, phase_modes)),
                    "predicted phase mode",
                )
                x_modes = float_tuple(
                    (
                        af**n * xi
                        - source_step
                        * value
                        * math.fsum(af ** (n - 1 - j) * at**j for j in range(n))
                        * zi
                        for af, at, value, xi, zi in zip(
                            form_factors, phase_factors, lambdas, epi_modes, phase_modes
                        )
                    ),
                    "predicted EPI mode",
                )
                phase_error = n * h * residual
                epi_error = (
                    capacity
                    * w
                    * (
                        n * h * width**3 / (18 * (1 - width**2 / 2))
                        + h**2 * n * (n - 1) * residual / 3
                    )
                )
                samples.append(
                    JointLinearSample(
                        step=n,
                        epi=float_tuple(right @ np.asarray(x_modes), "predicted EPI"),
                        phase_offset=float_tuple(
                            right @ np.asarray(z_modes), "predicted phase offset"
                        ),
                        epi_modes=x_modes,
                        phase_modes=z_modes,
                        phase_error_upper=phase_error,
                        epi_error_upper=epi_error,
                    )
                )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            "joint modal prediction exceeds finite numerical range"
        ) from exc
    return SynchronizedJointPrediction(
        capture=capture,
        dt=h,
        coupling_strength=k,
        phase_origin=origin,
        phase_width=width,
        degree_weights=degree,
        right_modes=right_modes,
        eigenvalues=lambdas,
        initial_epi_modes=epi_modes,
        initial_phase_modes=phase_modes,
        phase_reference=reference,
        samples=tuple(samples),
        scope=(
            "Numerical closed modal prediction of the consensus-tangent ideal Euler "
            "model on supplied fixed unit support and common capacity; exact rational "
            "bounds concern ideal nonlinear-versus-linear truncation only. Spectral "
            "and binary64 runtime errors are excluded. No Gamma/events/clipping/"
            "controllers or changing support/capacity; their runtime configuration "
            "is not admitted. No autonomous law, asymptotic execution or physical "
            "identification is certified."
        ),
    )


@dataclass(frozen=True)
class JointNodalResponse:
    """Conditional exact joint pressure response and nodal acceleration.

    Stored pressure is the declared p in x'=nu*p. Equality to the canonical
    pressure law, the phase/Gram relationship and the regular wrap chart are
    hypotheses, not certified by these detached data. Supplied phase/capacity
    rates are independent inputs; this observer does not derive their laws.
    """

    source: SupportTransportSnapshot
    phase_geometry: PhaseSourceGeometry
    epi_weight: Fraction
    phase_weight: Fraction
    capacity_weight: Fraction
    phase_rate_over_pi: Vector
    capacity_rate: Vector
    epi_pressure_rate: Vector
    phase_pressure_rate: Vector
    capacity_pressure_rate: Vector
    pressure_rate: Vector
    capacity_acceleration: Vector
    pressure_acceleration: Vector
    epi_acceleration: Vector
    transport: SupportTransportDerivative
    scope: tuple[str, ...] = (
        "conditional_exact_real_smooth_response",
        "fixed_active_conductance_edges_and_unique_support",
        "supplied_symmetric_conductance_rates_without_edge_birth_or_removal",
        "fixed_channel_coefficients_without_renormalization",
        "nonempty_support_at_every_node",
        "declared_stored_pressure_p_and_nodal_rate_nu_times_p",
        "requires_p_equal_canonical_pressure_and_regular_phase_chart",
        "phase_gram_state_and_pressure_compatibility_not_certified",
        "phase_rate_over_pi_and_capacity_rate_are_supplied_not_derived",
        "fixed_topology_channel_has_zero_derivative",
        "no_binary64_derivative_runtime_or_future_admissibility_claim",
    )

    @property
    def conductance_rates(self) -> Vector:
        """Supplied effective-edge rates in the transport owner's order."""
        return self.transport.conductance_rates

    @property
    def epi_flow_pressure_rate(self) -> Vector:
        """EPI-channel change due to the declared nodal form rate."""
        return tuple(
            self.epi_weight * value for value in self.transport.flow_gradient_rate
        )

    @property
    def epi_geometry_pressure_rate(self) -> Vector:
        """EPI-channel change due to changing normalized conductance."""
        return tuple(
            self.epi_weight * value for value in self.transport.geometry_gradient_rate
        )


def derive_joint_nodal_response(
    snapshot,
    phase_reference,
    *,
    epi_weight,
    phase_weight,
    capacity_weight,
    phase_rate_over_pi,
    capacity_rate,
    conductance_rates=None,
) -> JointNodalResponse:
    """Differentiate the joint canonical pressure law on declared fixed support.

    Let B_W be the weighted neighbor-difference operator, B_U the unweighted
    unique-support operator, and R the circular-mean response. With fixed
    support and channel coefficients, the conditional identity is

        p' = w_E*(B_W*(nu*p) + B_W'*x)
             + w_phi*(R-I)*(theta'/pi) + w_nu*B_U*nu',
        x'' = nu'*p + nu*p'.

    Optional ``conductance_rates`` align with every effective conductance
    entry, including both directions. The shared transport derivative owns
    their symmetry and row-normalization terms; omission means zero rates.
    The active edge set stays fixed: a zero-conductance support edge cannot
    acquire positive weight through this derivative. Births/removals require
    reset accounting. The rates are supplied, never inferred or selected.

    The topology channel has zero derivative on this fixed support. Loops
    count once, parallel edges aggregate only for EPI, and zero-conductance
    support edges still enter phase/capacity means. ``phase_rate_over_pi`` is
    an exact declared coordinate: no rational approximation to pi is inserted.
    Weights are nonnegative exact/represented scalars and are not normalized.

    Rebuild all support caches and validate the phase reference through the
    existing source-geometry owner. Its mean neighborhoods must match the
    snapshot's unique support, independent of iteration order. This first
    domain excludes empty graphs and isolates even when a weight is zero.

    The source's stored pressure is DECLARED p. The identity requires p to
    equal the canonical pressure law along a differentiable path with a
    nonzero-resultant regular shortest-arc chart. This call cannot establish
    that hypothesis, associate the exact Gram with a live phase state, or
    identify a derivative of floating arithmetic. No missing constitutive
    law or offset is supplied, and neither a graph nor a trajectory is changed.
    """
    transport = observe_support_transport_derivative(
        snapshot, conductance_rates=conductance_rates
    )
    source = transport.source
    size = len(source.nodes)
    if not size or any(not row for row in source.support_neighbors):
        raise ValueError("joint nodal response requires nonempty support at every node")
    geometry = observe_phase_source_geometry(phase_reference)
    reference = geometry.reference
    if len(reference.mean_neighbors) != size or any(
        set(mean) != set(support)
        for mean, support in zip(
            reference.mean_neighbors,
            source.support_neighbors,
            strict=True,
        )
    ):
        raise ValueError("phase mean neighborhoods must match the snapshot support")
    weights = tuple(
        exact_or_represented_real(value, name)
        for name, value in (
            ("epi_weight", epi_weight),
            ("phase_weight", phase_weight),
            ("capacity_weight", capacity_weight),
        )
    )
    if any(value < 0 for value in weights):
        raise ValueError("joint response weights must be nonnegative")
    epi_weight, phase_weight, capacity_weight = weights
    phase_rate = ordered_vector(phase_rate_over_pi, "phase_rate_over_pi")
    capacity_rate = ordered_vector(capacity_rate, "capacity_rate")
    if len(phase_rate) != size or len(capacity_rate) != size:
        raise ValueError(
            "joint response rate vectors must match the snapshot node order"
        )

    epi_response = tuple(epi_weight * value for value in transport.epi_gradient_rate)
    phase_response = tuple(
        phase_weight * dot(row, phase_rate) for row in geometry.scaled_source_jacobian
    )
    capacity_response = tuple(
        capacity_weight * value
        for value in _support_gradient(source.support_neighbors, capacity_rate)
    )
    pressure_rate = tuple(
        a + b + c
        for a, b, c in zip(
            epi_response,
            phase_response,
            capacity_response,
            strict=True,
        )
    )
    capacity_acceleration = tuple(
        a * b
        for a, b in zip(
            capacity_rate,
            source.stored_pressure,
            strict=True,
        )
    )
    pressure_acceleration = tuple(
        a * b
        for a, b in zip(
            source.capacity,
            pressure_rate,
            strict=True,
        )
    )
    acceleration = tuple(
        a + b
        for a, b in zip(
            capacity_acceleration,
            pressure_acceleration,
            strict=True,
        )
    )
    return JointNodalResponse(
        source,
        geometry,
        epi_weight,
        phase_weight,
        capacity_weight,
        phase_rate,
        capacity_rate,
        epi_response,
        phase_response,
        capacity_response,
        pressure_rate,
        capacity_acceleration,
        pressure_acceleration,
        acceleration,
        transport,
    )


@dataclass(frozen=True)
class PhaseCapacityBalance:
    """Finite capacity solutions for a declared phase source and held forcing.

    A compatible solution is ``centered_capacity + c*1``. Its admissible
    uniform shifts have the reported lower endpoint (possibly open) and a
    closed upper endpoint, or no upper bound. An incompatible source has no
    profile or shift endpoints. An empty band intersection may have a profile
    but ``has_admissible_capacity=False``. No shift or evolution law is chosen.
    """

    source: SupportTransportSnapshot
    phase_gradient: Vector
    forcing: Vector
    phase_weight: Fraction
    capacity_weight: Fraction
    topology_weight: Fraction
    support_degrees: Vector
    source_difference: Vector
    compatibility_residual: Fraction
    centered_capacity: Vector | None
    equation_residual: Vector | None
    center_residual: Fraction | None
    capacity_lower: Vector
    capacity_upper: Vector | None
    uniform_shift_lower: Fraction | None
    uniform_shift_lower_inclusive: bool
    uniform_shift_upper: Fraction | None
    has_admissible_capacity: bool
    scope: tuple[str, ...] = (
        "exact_declared_source_on_connected_reciprocal_unique_support",
        "positive_capacity_weight_and_fixed_nonnegative_channel_weights",
        "phase_gradient_is_supplied_not_certified_from_angles_or_gram",
        "uniform_capacity_freedom_is_not_equivalence_of_nodal_dynamics",
        "closed_declared_bands_intersect_strict_capacity_positivity",
        "no_selected_capacity_phase_law_graph_write_or_runtime_certificate",
        "no_epi_equilibrium_stability_or_future_invariance_claim",
    )

    @property
    def source_compatible(self) -> bool:
        """Whether unrestricted real capacities solve the declared source."""
        return self.compatibility_residual == 0


def derive_phase_capacity_balance(
    snapshot,
    *,
    phase_gradient,
    forcing,
    phase_weight,
    capacity_weight,
    topology_weight=0,
    capacity_lower=None,
    capacity_upper=None,
) -> PhaseCapacityBalance:
    """Solve ``f = w_phi*g - v*L_U*nu - w_topo*L_U*d`` exactly.

    ``U`` is the unweighted unique-support neighbor mean, ``L_U=I-U`` and
    ``d`` its row sizes. Reciprocal connected nonempty support gives
    ``range(L_U)={r: d.r=0}``. Thus ``r=w_phi*g-w_topo*L_U*d-f`` admits
    capacities precisely when ``d.r=0``. The positive-weight EPI graph may
    be disconnected; it is not substituted for this support equation.

    On compatibility, reuse the exact inverse to solve
    ``(v*B+d*d.T)*z=D*r``, where ``B=D*L_U`` and ``D=diag(d)``. The rank-one
    term only fixes ``d.z=0``; it is not a physical source. All solutions are
    ``z+c*1``. Require strict positive capacity and optional closed nodewise
    nonnegative bounds. A missing lower bound means zero; a missing upper
    bound means unbounded. No coefficients are normalized or fitted.

    Inputs are exact rationals or finite represented reals. In particular,
    ``g`` is DECLARED oriented phase-pressure data, not reconstructed from a
    cosine Gram. Identifying it with canonical phase pressure requires its
    own regular circular-mean/wrap chart evidence. Rebuild snapshot caches;
    count loops once and retain zero-weight support edges. No graph changes,
    missing velocities, automatic clip policy or trajectory are supplied.
    """
    if type(snapshot) is not SupportTransportSnapshot:
        raise TypeError("state must be a SupportTransportSnapshot")
    # Materialize once, rejecting unordered containers before the shared rebuild.
    nodes = _ordered(snapshot.nodes, "snapshot nodes")
    support = tuple(
        _ordered(row, "support row")
        for row in _ordered(snapshot.support_neighbors, "support rows")
    )
    source = _rebuild(replace(snapshot, nodes=nodes, support_neighbors=support))
    size = len(nodes)
    if not size or any(not row for row in support):
        raise ValueError("capacity balance requires nonempty support at every node")
    if any(i not in support[j] for i, row in enumerate(support) for j in row):
        raise ValueError("capacity balance requires reciprocal support")
    reached, pending = {0}, [0]
    while pending:
        for j in support[pending.pop()]:
            if j not in reached:
                reached.add(j)
                pending.append(j)
    if len(reached) != size:
        raise ValueError("capacity balance requires connected support")

    w, v, t = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("phase_weight", phase_weight),
            ("capacity_weight", capacity_weight),
            ("topology_weight", topology_weight),
        )
    )
    if w < 0 or t < 0 or v <= 0:
        raise ValueError(
            "phase/topology weights must be nonnegative and capacity weight positive"
        )
    g = ordered_vector(phase_gradient, "phase_gradient")
    f = ordered_vector(forcing, "forcing")
    lower = (
        (Fraction(0),) * size
        if capacity_lower is None
        else ordered_vector(capacity_lower, "capacity_lower")
    )
    upper = (
        None
        if capacity_upper is None
        else ordered_vector(capacity_upper, "capacity_upper")
    )
    if any(len(values) != size for values in (g, f, lower)) or (
        upper is not None and len(upper) != size
    ):
        raise ValueError("capacity balance vectors must match the snapshot node order")
    if any(value < 0 for value in lower) or (
        upper is not None and any(a > b for a, b in zip(lower, upper, strict=True))
    ):
        raise ValueError("capacity bands must satisfy 0 <= lower <= upper")

    degrees = tuple(Fraction(len(row)) for row in support)
    difference = tuple(
        w * g_i + t * h_i - f_i
        for g_i, h_i, f_i in zip(g, source.topology_gradient, f, strict=True)
    )
    compatibility = dot(degrees, difference)
    profile = residual = center = shift_lower = shift_upper = None
    inclusive = admissible = False
    if compatibility == 0:
        matrix = tuple(
            tuple(
                v * (degrees[i] * int(i == j) - int(j in row)) + degrees[i] * degrees[j]
                for j in range(size)
            )
            for i, row in enumerate(support)
        )
        rhs = tuple(d * r for d, r in zip(degrees, difference, strict=True))
        profile = tuple(dot(row, rhs) for row in exact_matrix_inverse(matrix))
        residual = tuple(
            -v * value - r
            for value, r in zip(
                _support_gradient(support, profile), difference, strict=True
            )
        )
        center = dot(degrees, profile)
        if any(residual) or center:
            raise ArithmeticError(
                "exact capacity equation or centering identity failed"
            )
        shift_lower = max(a - z for a, z in zip(lower, profile, strict=True))
        inclusive = shift_lower > -min(profile)
        if upper is not None:
            shift_upper = min(b - z for b, z in zip(upper, profile, strict=True))
        admissible = (
            shift_upper is None
            or shift_lower < shift_upper
            or (shift_lower == shift_upper and inclusive)
        )
    return PhaseCapacityBalance(
        source,
        g,
        f,
        w,
        v,
        t,
        degrees,
        difference,
        compatibility,
        profile,
        residual,
        center,
        lower,
        upper,
        shift_lower,
        inclusive,
        shift_upper,
        admissible,
    )
