"""Global unordered pair state and finite responses on the declared doubled C5.

Exact phasors retain pair state through cancellation. The finite exchange and
receiver assessments keep their separately declared complete laws, preparations
and error budgets; they install no runtime law or numerical trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from .relational_observations import _ordered

__all__ = (
    "SineGlobalPairState",
    "SinePairCancellationObservation",
    "SinePairFiniteExchange",
    "SinePairReceiverReadout",
    "SinePairReceiverConfounding",
    "SinePairReceiverTwoTime",
    "SinePairReceiverTwoLaw",
    "SinePairReceiverDefect",
    "SinePairPersistentResponse",
    "derive_sine_global_pair_state",
    "evaluate_sine_global_pair_state",
    "observe_sine_pair_cancellation",
    "assess_sine_pair_finite_exchange",
    "assess_sine_pair_receiver_readout",
    "assess_sine_pair_receiver_confounding",
    "assess_sine_pair_receiver_two_time",
    "assess_sine_pair_receiver_two_law",
    "assess_sine_pair_receiver_defect",
    "assess_sine_pair_persistent_response",
)


_GLOBAL_PAIR_BASE_EDGES = tuple((i, (i + 1) % 5) for i in range(5))


def _pair_complex_product(left, right):
    """Multiply exact Cartesian circular-state coefficients, never EPI."""
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c


def _pair_complex_conjugate(value):
    return value[0], -value[1]


def _pair_complex_linear(*terms):
    return tuple(
        sum((factor * value[k] for factor, value in terms), Q(0)) for k in (0, 1)
    )


def _pair_complex_i(value):
    return -value[1], value[0]


def _global_pair_scalars(values, label, size):
    raw = _ordered(values, label, limit=size + 1)
    if len(raw) != size:
        raise ValueError(f"{label} must contain exactly {size} values")
    return tuple(
        exact_or_represented_real(value, f"{label}[{i}]") for i, value in enumerate(raw)
    )


def _global_pair_complex_rows(values, label):
    rows = _ordered(values, label, limit=6)
    if len(rows) != 5:
        raise ValueError(f"{label} must contain exactly five Cartesian pairs")
    return tuple(
        _global_pair_scalars(row, f"{label}[{i}]", 2) for i, row in enumerate(rows)
    )


def _global_pair_interfaces(form_means, resultants):
    """Derived h/B interfaces for the fixed unit doubled-C5 law."""
    return tuple(
        (
            form_means[a] - (form_means[(a - 1) % 5] + form_means[(a + 1) % 5]) / 2,
            _pair_complex_linear(
                (Q(1, 2), resultants[(a - 1) % 5]),
                (Q(1, 2), resultants[(a + 1) % 5]),
            ),
        )
        for a in range(5)
    )


@dataclass(frozen=True)
class SinePairCancellationObservation:
    """Conditional inversion of two exact interface derivatives at Z=0.

    The same fixed doubled-C5 law as SineGlobalPairState is assumed. The
    supplied derivatives use tau=t/pi: first and second tau derivatives are
    pi and pi squared times the corresponding structural-t derivatives.
    They are evidence supplied independently of the hidden P/U/W coordinates,
    not estimated from samples or filled from a forward-state report.

    An unavailable phase product means that these two derivatives do not
    identify its value. It does not classify the full interface history.
    """

    form_means: tuple[Q, ...]
    resultants: tuple[tuple[Q, Q], ...]
    pair_index: int
    resultant_first_tau_derivative: tuple[Q, Q]
    resultant_second_tau_derivative: tuple[Q, Q]
    form_contrast: Q
    neighbor_resultant: tuple[Q, Q]
    internal_form_squared: Q
    form_phase_moment: tuple[Q, Q]
    phase_product: tuple[Q, Q] | None
    phase_product_reconstruction_order: int | None
    unavailable_reason: str | None
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=t/pi"
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "no_supplied_forcing_or_events",
        "selected_resultant_exactly_zero_all_five_mean_phasors_in_unit_disk",
        "exact_supplied_first_and_second_tau_derivatives_not_forward_predictions",
        "h_and_B_derived_from_supplied_form_means_and_resultants",
        "U_and_W_recovered_from_first_derivative_P_available_conditionally",
        "exact_second_derivative_compatibility_without_tolerances",
        "two_derivatives_do_not_classify_persistent_invisibility",
        "no_neighbor_hidden_state_or_complete_trajectory_reconstruction",
        "no_derivative_estimation_noise_admission_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-pair-cancellation-observation.v1",
            "report": _project(self),
        }


def observe_sine_pair_cancellation(
    *,
    form_means,
    resultants,
    pair_index,
    resultant_first_tau_derivative,
    resultant_second_tau_derivative,
) -> SinePairCancellationObservation:
    """Recover a pair's hidden invariants conditionally from exact Z derivatives.

    Supply all five X/Z means in base-cycle order and both selected-pair
    derivatives in tau=t/pi. The selected resultant must vanish exactly. The
    fixed complete law gives W=-i*Z', U=|Z'|**2 and, when Z' is nonzero,
    P=Z'**2/U. Otherwise nonzero B yields P=(2*Z''-B)/conj(B). Any recovered
    phase product must have unit norm and reproduce the supplied second
    derivative, including the 2*i*h*Z' contribution for moving pairs.

    When Z'=B=0, compatibility requires Z''=0; P remains unavailable.
    Higher derivatives or persistent environmental evidence are separate
    obligations. No forward report, hidden state or measured sample series
    is accepted as a substitute for these declared derivative inputs.
    """
    from .phase_response import _admit_relative_phasors

    x = _global_pair_scalars(form_means, "form_means", 5)
    z = _global_pair_complex_rows(resultants, "resultants")
    if type(pair_index) is not int or not 0 <= pair_index < 5:
        raise ValueError("pair_index must be a nonboolean pair index in 0..4")
    for a, value in enumerate(z):
        if sum((part * part for part in value), Q(0)) > 1:
            raise ValueError(f"pair {a} has a mean phasor outside the unit disk")
    zero = (Q(0), Q(0))
    if z[pair_index] != zero:
        raise ValueError("the selected pair must have an exactly zero resultant")
    first = _global_pair_scalars(
        resultant_first_tau_derivative, "resultant_first_tau_derivative", 2
    )
    second = _global_pair_scalars(
        resultant_second_tau_derivative, "resultant_second_tau_derivative", 2
    )
    h, neighbor = _global_pair_interfaces(x, z)[pair_index]
    moment = (first[1], -first[0])
    squared = sum((part * part for part in first), Q(0))
    product, order, reason = None, None, None
    if squared:
        product = tuple(part / squared for part in _pair_complex_product(first, first))
        order = 1
    elif neighbor != zero:
        numerator = _pair_complex_linear((2, second), (-1, neighbor))
        norm = sum((part * part for part in neighbor), Q(0))
        product = tuple(
            part / norm for part in _pair_complex_product(numerator, neighbor)
        )
        order = 2
    else:
        if second != zero:
            raise ValueError(
                "zero first derivative and neighbor resultant require zero second derivative"
            )
        reason = "higher_order_or_persistent_environment_evidence_required"
    if product is not None:
        product = _admit_relative_phasors((product,), "phase_product")[0]
        expected_second = _pair_complex_linear(
            (2 * h, _pair_complex_i(first)),
            (
                Q(1, 2),
                _pair_complex_product(product, _pair_complex_conjugate(neighbor)),
            ),
            (Q(1, 2), neighbor),
        )
        if second != expected_second:
            raise ValueError(
                "supplied second derivative is incompatible with the fixed pair law"
            )
    return SinePairCancellationObservation(
        form_means=x,
        resultants=z,
        pair_index=pair_index,
        resultant_first_tau_derivative=first,
        resultant_second_tau_derivative=second,
        form_contrast=h,
        neighbor_resultant=neighbor,
        internal_form_squared=squared,
        form_phase_moment=moment,
        phase_product=product,
        phase_product_reconstruction_order=order,
        unavailable_reason=reason,
    )


@dataclass(frozen=True)
class SineGlobalPairState:
    """Realizable global unordered-pair state on the fixed doubled C5.

    Each Cartesian pair is an exact (real, imaginary) phase coefficient. Form
    remains scalar and signed. Ordered base positions are 0..4 and fine pairs
    are (0,1),...,(8,9); every base edge has all four unit fine cross-edges.
    The complete law is conservative normalized sine exchange with unit held
    capacity, exchange coefficient and beta. No input or event is supplied.

    Rate numerators multiply derivatives by mathematical pi in the original
    structural t clock. Current rows are (receiver, source); each current is
    a contribution to the receiver's mean-form rate, not its weighted regional
    total. Its derivative numerator multiplies by pi squared. Coordinates
    describe swap orbits, including coincident and antipodal phases, without
    removing continuous degrees of freedom or reconstructing an angle.
    """

    form_means: tuple[Q, ...]
    resultants: tuple[tuple[Q, Q], ...]
    phase_products: tuple[tuple[Q, Q], ...]
    internal_form_squared: tuple[Q, ...]
    form_phase_moments: tuple[tuple[Q, Q], ...]
    pair_strata: tuple[str, ...]
    form_mean_rate_pi_numerators: tuple[Q, ...]
    resultant_rate_pi_numerators: tuple[tuple[Q, Q], ...]
    phase_product_rate_pi_numerators: tuple[tuple[Q, Q], ...]
    internal_form_squared_rate_pi_numerators: tuple[Q, ...]
    form_phase_moment_rate_pi_numerators: tuple[tuple[Q, Q], ...]
    directed_block_edges: tuple[tuple[int, int], ...]
    block_current_pi_numerators: tuple[Q, ...]
    block_current_rate_pi_squared_numerators: tuple[Q, ...]
    form_storage: Q
    phase_storage: Q
    storage: Q
    form_storage_rate_pi_numerator: Q
    phase_storage_rate_pi_numerator: Q
    full_storage_rate_pi_numerator: Q
    pairs: tuple[tuple[int, int], ...] = tuple((2 * i, 2 * i + 1) for i in range(5))
    base_edges: tuple[tuple[int, int], ...] = _GLOBAL_PAIR_BASE_EDGES
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "structural_t"
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "signed_scalar_form_and_exact_unit_circular_phasors_not_complex_EPI",
        "X_mean_form_Z_mean_phasor_P_phase_product_U_internal_form_square_W_form_phase_correlation",
        "exact_realizability_admission_without_tolerances_or_phasor_renormalization",
        "global_swap_orbits_retain_coincident_and_antipodal_phase_information",
        "polynomial_full_field_pushforward_without_resultant_or_internal_form_division",
        "phase_product_retains_state_information_not_an_added_harmonic_current_law",
        "directed_currents_are_block_mean_form_rates_weighted_cut_currents_are_eight_times_larger",
        "first_rates_have_pi_numerators_current_derivatives_have_pi_squared_numerators",
        "full_storage_and_its_rate_are_summed_from_all_declared_base_edges",
        "no_removed_continuous_state_dimensions_formation_or_maintenance_certificate",
        "no_graph_capture_runtime_law_installation_trajectory_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-global-pair-state.v1", "report": _project(self)}


def derive_sine_global_pair_state(forms, phasors) -> SineGlobalPairState:
    """Project ten exact fine coordinates to their global unordered-pair state.

    Inputs follow consecutive pairs around the fixed five-cycle. Phasors are
    Cartesian unit-circle pairs, not radian angles or Python complex values.
    Real inputs use shared exact/represented admission; represented Cartesian
    coordinates must have exactly unit norm. No approximate normalization is
    performed. The returned state retains the orbit, not the member labeling.
    """
    from .phase_response import _admit_relative_phasors

    x = _global_pair_scalars(forms, "forms", 10)
    raw = _ordered(phasors, "phasors", limit=11)
    if len(raw) != 10:
        raise ValueError("phasors must contain exactly ten Cartesian unit pairs")
    z = _admit_relative_phasors(raw, "phasors")
    means, resultants, products, squared, moments = [], [], [], [], []
    for i in range(0, 10, 2):
        u = (x[i] - x[i + 1]) / 2
        means.append((x[i] + x[i + 1]) / 2)
        resultants.append(_pair_complex_linear((Q(1, 2), z[i]), (Q(1, 2), z[i + 1])))
        products.append(_pair_complex_product(z[i], z[i + 1]))
        squared.append(u * u)
        moments.append(_pair_complex_linear((u / 2, z[i]), (-u / 2, z[i + 1])))
    return evaluate_sine_global_pair_state(
        form_means=means,
        resultants=resultants,
        phase_products=products,
        internal_form_squared=squared,
        form_phase_moments=moments,
    )


def evaluate_sine_global_pair_state(
    *, form_means, resultants, phase_products, internal_form_squared, form_phase_moments
) -> SineGlobalPairState:
    """Admit realizable global pairs and evaluate their exact complete-law rates.

    For z_plus/minus on the circle and x_plus/minus real, the coordinates are
    X=mean(x), Z=mean(z), P=z_plus*z_minus, U=u**2, W=u*(z_plus-z_minus)/2,
    with u=half_difference(x). Five supplied coordinate collections must satisfy
    |P|=1, P*conj(Z)=Z, |Z|<=1, U>=0 and W**2=U*(Z**2-P). The equivalent
    norm identity for W is also checked. These exact conditions are sufficient
    for a fine representative; its components need not themselves be rational.

    No derivative, current or cached report verdict is accepted as an input.
    The invariant state is sufficient only for the fixed law and support of
    SineGlobalPairState, not for arbitrary attachments or capacity assignments.
    """
    from .phase_response import _admit_relative_phasors

    x = _global_pair_scalars(form_means, "form_means", 5)
    z = _global_pair_complex_rows(resultants, "resultants")
    product = _admit_relative_phasors(
        _global_pair_complex_rows(phase_products, "phase_products"), "phase_products"
    )
    u2 = _global_pair_scalars(internal_form_squared, "internal_form_squared", 5)
    w = _global_pair_complex_rows(form_phase_moments, "form_phase_moments")
    norms, differences, strata = [], [], []
    for a in range(5):
        norm = sum((part * part for part in z[a]), Q(0))
        d2 = _pair_complex_linear(
            (1, _pair_complex_product(z[a], z[a])), (-1, product[a])
        )
        if (
            norm > 1
            or _pair_complex_product(product[a], _pair_complex_conjugate(z[a])) != z[a]
        ):
            raise ValueError(f"pair {a} has unrealizable mean phasor and phase product")
        if u2[a] < 0:
            raise ValueError("internal_form_squared must be nonnegative")
        if _pair_complex_product(w[a], w[a]) != tuple(u2[a] * part for part in d2):
            raise ValueError(f"pair {a} violates the form-phase square constraint")
        if sum((part * part for part in w[a]), Q(0)) != u2[a] * (1 - norm):
            raise ValueError(f"pair {a} violates the form-phase norm constraint")
        norms.append(norm)
        differences.append(d2)
        strata.append(
            "coincident_phase"
            if norm == 1
            else "antipodal_phase" if norm == 0 else "split_phase"
        )

    mean_rates, resultant_rates, product_rates, squared_rates, moment_rates = (
        [],
        [],
        [],
        [],
        [],
    )
    for a, (q, neighbor) in enumerate(_global_pair_interfaces(x, z)):
        mean_rates.append(
            _pair_complex_product(_pair_complex_conjugate(z[a]), neighbor)[1]
        )
        resultant_rates.append(
            _pair_complex_i(_pair_complex_linear((q, z[a]), (1, w[a])))
        )
        product_rates.append(
            _pair_complex_i(tuple(2 * q * part for part in product[a]))
        )
        squared_rates.append(
            2 * _pair_complex_product(_pair_complex_conjugate(w[a]), neighbor)[1]
        )
        moment_rates.append(
            _pair_complex_i(
                _pair_complex_linear(
                    (q, w[a]),
                    (u2[a], z[a]),
                    (
                        Q(1, 2),
                        _pair_complex_product(
                            differences[a], _pair_complex_conjugate(neighbor)
                        ),
                    ),
                    (-(1 - norms[a]) / 2, neighbor),
                )
            )
        )
    directed = tuple((a, b) for a in range(5) for b in ((a - 1) % 5, (a + 1) % 5))
    currents, current_rates = [], []
    for a, b in directed:
        currents.append(
            _pair_complex_product(_pair_complex_conjugate(z[a]), z[b])[1] / 2
        )
        current_rates.append(
            (
                _pair_complex_product(
                    _pair_complex_conjugate(resultant_rates[a]), z[b]
                )[1]
                + _pair_complex_product(
                    _pair_complex_conjugate(z[a]), resultant_rates[b]
                )[1]
            )
            / 2
        )
    form_storage, phase_storage, form_work, phase_work = Q(0), Q(0), Q(0), Q(0)
    for a, b in _GLOBAL_PAIR_BASE_EDGES:
        form_storage += 2 * ((x[a] - x[b]) ** 2 + u2[a] + u2[b])
        phase_storage += 4 * (
            1 - _pair_complex_product(_pair_complex_conjugate(z[a]), z[b])[0]
        )
        form_work += 4 * (x[a] - x[b]) * (mean_rates[a] - mean_rates[b]) + 2 * (
            squared_rates[a] + squared_rates[b]
        )
        phase_work -= 4 * (
            _pair_complex_product(_pair_complex_conjugate(resultant_rates[a]), z[b])[0]
            + _pair_complex_product(_pair_complex_conjugate(z[a]), resultant_rates[b])[
                0
            ]
        )
    return SineGlobalPairState(
        form_means=x,
        resultants=z,
        phase_products=product,
        internal_form_squared=u2,
        form_phase_moments=w,
        pair_strata=tuple(strata),
        form_mean_rate_pi_numerators=tuple(mean_rates),
        resultant_rate_pi_numerators=tuple(resultant_rates),
        phase_product_rate_pi_numerators=tuple(product_rates),
        internal_form_squared_rate_pi_numerators=tuple(squared_rates),
        form_phase_moment_rate_pi_numerators=tuple(moment_rates),
        directed_block_edges=directed,
        block_current_pi_numerators=tuple(currents),
        block_current_rate_pi_squared_numerators=tuple(current_rates),
        form_storage=form_storage,
        phase_storage=phase_storage,
        storage=form_storage + phase_storage,
        form_storage_rate_pi_numerator=form_work,
        phase_storage_rate_pi_numerator=phase_work,
        full_storage_rate_pi_numerator=form_work + phase_work,
    )


@dataclass(frozen=True)
class SinePairFiniteExchange:
    """Analytic finite exchange near a fixed invisible doubled-C5 family.

    The two preparations differ only in selected pair 0's antipodal orientation.
    The reported difference is real-antipodal minus imaginary-antipodal of the
    time integral of pair 1's contribution to pair 0's mean-form rate. Integration
    uses tau=t/pi and its correspondingly scaled current; the value equals the
    original-clock integral. A weighted cut integral is eight times this value.

    The analytic remainder bounds the full autonomous sine flow. No numerical
    trajectory or fitted derivative supplies the result. A zero-containing
    interval is unavailable, including when the leading coefficient vanishes.
    The separate control is a proved stationary family with exactly zero current,
    not a conclusion inferred from a finite zero jet. Export preserves the
    report and its premises without authenticating an edited dataclass.
    """

    phase_rotation: tuple[Q, Q]
    horizon_tau: Q
    comparison_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    control_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    comparison_initial_states: tuple[SineGlobalPairState, ...]
    control_initial_states: tuple[SineGlobalPairState, ...]
    integrated_current_difference_leading_term: Q
    integrated_current_difference_remainder_upper_bound: Q
    integrated_current_difference_bounds: I
    status: str
    unavailable_reason: str | None
    control_integrated_currents: tuple[Q, Q] = (Q(0), Q(0))
    control_integrated_current_difference: Q = Q(0)
    orientation_order: tuple[str, str] = (
        "real_antipodal",
        "imaginary_antipodal",
    )
    directed_block_edge: tuple[int, int] = (0, 1)
    nodes: tuple[int, ...] = tuple(range(10))
    initial_forms: tuple[Q, ...] = (Q(0),) * 10
    held_capacities: tuple[Q, ...] = (Q(1),) * 10
    form_loss: Q = Q(0)
    exchange_weight: Q = Q(1)
    beta: Q = Q(1)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=t/pi"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "zero_initial_forms_exact_unit_phasors_and_full_retained_environment",
        "comparison_environment_pair_phasors_q_one_minus_one_minus_one",
        "stationary_control_environment_pair_phasors_q_q_minus_q_minus_q",
        "orientations_compared_under_the_same_prepared_autonomous_environment",
        "directed_edge_is_receiver_source_and_difference_follows_orientation_order",
        "actual_integrated_mean_form_current_weighted_cut_integral_is_eight_times_larger",
        "global_analytic_remainder_for_the_complete_nonlinear_sine_flow",
        "control_invariance_is_proved_separately_not_inferred_from_initial_zero_rates",
        "no_forcing_events_fitted_pressure_or_numerical_trajectory_evaluation",
        "unavailable_bound_does_not_prove_zero_exchange_or_orientation_independence",
        "no_formation_robustness_unique_law_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-pair-finite-exchange.v1", "report": _project(self)}


def _sine_pair_exchange_phasors(q):
    """Build realizable comparison/control phasors from an admitted unit q."""
    one, imaginary = (Q(1), Q(0)), (Q(0), Q(1))
    negative_one = (-one[0], -one[1])
    negative_q = (-q[0], -q[1])
    comparison_environment = (q, one, negative_one, negative_one)
    control_environment = (q, q, negative_q, negative_q)

    def preparations(environment):
        return tuple(
            (orientation, (-orientation[0], -orientation[1]))
            + tuple(phasor for phasor in environment for _ in range(2))
            for orientation in (one, imaginary)
        )

    comparison_phasors = preparations(comparison_environment)
    control_phasors = preparations(control_environment)
    return comparison_phasors, control_phasors


def _sine_pair_exchange_preparations(q):
    """Admit the fixed broken-cancellation and stationary sine controls."""
    comparison_phasors, control_phasors = _sine_pair_exchange_phasors(q)
    forms = (Q(0),) * 10
    comparison_states = tuple(
        derive_sine_global_pair_state(forms, phasors) for phasors in comparison_phasors
    )
    control_states = tuple(
        derive_sine_global_pair_state(forms, phasors) for phasors in control_phasors
    )
    return comparison_phasors, control_phasors, comparison_states, control_states


def _sine_pair_finite_remainder(horizon):
    """Sum two integrated block-current Taylor remainder bounds."""
    return _sine_pair_cubic_receiver_remainder(horizon, Q(0))


def _sine_pair_cubic_current_bounds(epsilon):
    """Bounds for j_epsilon and its first three angular derivatives."""
    return tuple(1 + multiplier * epsilon for multiplier in (1, 3, 9, 27))


def _sine_pair_cubic_receiver_remainder(horizon, epsilon):
    """Bound the complete cubic-law receiver remainder for zero initial forms."""
    value, first, second, third = _sine_pair_cubic_current_bounds(epsilon)
    fifth = (48 * second * value**2 + 16 * first**2 * value) / 120
    seventh = 64 * third * value**3 / 840
    return fifth * horizon**5 + seventh * horizon**7


def _sine_pair_cubic_receiver_a_coefficients(q, epsilon):
    """Absolute receiver linear/cubic coefficients for the real-antipodal A."""
    a, b = q
    current = b + epsilon * b**3
    return -current / 2, a * (1 + 3 * epsilon * b**2) * current / 6


def _sine_pair_receiver_center_coefficients(q):
    """Shared absolute receiver coefficients for the two baseline sine states."""
    a, b = q
    linear, real_cubic = _sine_pair_cubic_receiver_a_coefficients(q, Q(0))
    return linear, (real_cubic, b * (2 * a + 1) / 24)


def assess_sine_pair_finite_exchange(
    *, phase_rotation, horizon_tau
) -> SinePairFiniteExchange:
    """Bound a finite integrated current difference from declared preparations.

    Supply one exact Cartesian unit phasor q and a strictly positive duration h
    in tau=t/pi. Shared represented-real admission retains exact input values;
    no phasor normalization or angular reconstruction occurs. Exact rational
    inputs are never materialized as floats.
    Pair 0 is prepared as (1,-1) or (i,-i), with all initial forms zero. The
    comparison's coincident surrounding pairs are (q,1,-1,-1), while the
    stationary symmetry control uses (q,q,-q,-q).

    For the directed pair 1 -> pair 0 mean-form current, the integrated
    real-minus-imaginary difference has leading term -Im(q*q-q)*h**3/24 and
    absolute remainder at most 8*h**5/15+8*h**7/105. The bound is valid for
    every positive horizon; a large or unresolved bound remains unavailable.
    Sign admission uses the outward interval, not its unrounded center.
    Every initial state is freshly admitted through the global-pair owner.
    """
    from .phase_response import _admit_relative_phasors

    q = _admit_relative_phasors((phase_rotation,), "phase_rotation")[0]
    horizon = exact_or_represented_real(horizon_tau, "horizon_tau")
    if horizon <= 0:
        raise ValueError("horizon_tau must be strictly positive")
    (
        comparison_phasors,
        control_phasors,
        comparison_states,
        control_states,
    ) = _sine_pair_exchange_preparations(q)
    leading = -(_pair_complex_product(q, q)[1] - q[1]) * horizon**3 / 24
    remainder = _sine_pair_finite_remainder(horizon)
    bounds = I(leading - remainder, leading + remainder)
    status = (
        "certified_negative"
        if bounds.hi < 0
        else "certified_positive" if bounds.lo > 0 else "unavailable"
    )
    return SinePairFiniteExchange(
        phase_rotation=q,
        horizon_tau=horizon,
        comparison_initial_phasors=comparison_phasors,
        control_initial_phasors=control_phasors,
        comparison_initial_states=comparison_states,
        control_initial_states=control_states,
        integrated_current_difference_leading_term=leading,
        integrated_current_difference_remainder_upper_bound=remainder,
        integrated_current_difference_bounds=bounds,
        status=status,
        unavailable_reason=(
            "finite_exchange_interval_contains_zero"
            if status == "unavailable"
            else None
        ),
    )


@dataclass(frozen=True)
class SinePairReceiverReadout:
    """Absolute receiver mean-form enclosures for two uncertain preparations.

    Only pair 1's mean form is read at t=pi*horizon_tau. Nominal preparations
    are the same complete fine states as SinePairFiniteExchange. Each branch's
    center includes both incident currents and its common linear response.
    Its nominal Taylor remainder, all-node preparation errors and absolute
    readout error remain separately available.

    Phase preparation errors are real lift offsets about the nominal unit
    phasors, measured in radians. The two law-specific amplification coefficients
    map the separate form and phase budgets to a form error. Bounds concern
    every admitted initial perturbation under the same fixed complete law.
    They are outer enclosures: membership is only a necessary compatibility
    condition, not a reconstructed state or an inverse classification.

    Only the nominal symmetry controls are stationary. Their uncertainty
    enclosures allow evolving perturbed controls. Export projects the declared
    evidence without authenticating altered reports or evaluating trajectories.
    """

    phase_rotation: tuple[Q, Q]
    horizon_tau: Q
    form_error_bound: Q
    phase_error_bound: Q
    readout_error_bound: Q
    comparison_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    control_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    comparison_initial_states: tuple[SineGlobalPairState, ...]
    control_initial_states: tuple[SineGlobalPairState, ...]
    nominal_linear_coefficient: Q
    nominal_cubic_coefficients: tuple[Q, Q]
    nominal_readout_centers: tuple[Q, Q]
    nominal_remainder_upper_bound: Q
    nominal_readout_bounds: tuple[I, I]
    form_error_amplification_upper_bound: Q
    phase_error_amplification_upper_bound: Q
    propagated_preparation_error_upper_bound: Q
    readout_uncertainty_radius: Q
    expanded_readout_bounds: tuple[I, I]
    readout_gap_lower_bound: Q
    status: str
    unavailable_reason: str | None
    higher_readout_orientation: str | None
    symmetry_control_uncertainty_radius: Q
    symmetry_control_readout_bounds: tuple[I, I]
    symmetry_control_readout_centers: tuple[Q, Q] = (Q(0), Q(0))
    orientation_order: tuple[str, str] = (
        "real_antipodal",
        "imaginary_antipodal",
    )
    receiver_pair_index: int = 1
    receiver_nodes: tuple[int, int] = (2, 3)
    nodes: tuple[int, ...] = tuple(range(10))
    initial_forms: tuple[Q, ...] = (Q(0),) * 10
    held_capacities: tuple[Q, ...] = (Q(1),) * 10
    form_loss: Q = Q(0)
    exchange_weight: Q = Q(1)
    beta: Q = Q(1)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=t/pi"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "same_nominal_comparison_and_control_preparations_as_finite_pair_exchange",
        "receiver_pair_one_absolute_mean_form_includes_both_incident_currents",
        "all_node_initial_form_and_phase_lift_errors_have_separate_uniform_budgets",
        "phase_lift_errors_are_radians_in_the_declared_normalized_complete_law",
        "coupled_form_phase_comparison_retains_both_error_channels",
        "rational_amplification_majorant_requires_zero_less_than_horizon_less_than_one_half",
        "nominal_nonlinear_remainder_and_preparation_and_readout_errors_are_separate",
        "strict_gap_uses_outward_expanded_intervals_not_unrounded_centers",
        "interval_overlap_does_not_establish_actual_response_overlap_or_equality",
        "interval_membership_is_necessary_compatibility_not_a_sufficient_inverse",
        "only_nominal_symmetry_controls_are_stationary_uncertain_controls_may_evolve",
        "no_forcing_events_law_uncertainty_sensor_calibration_or_trajectory_evaluation",
        "no_formation_universal_orientation_recovery_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-pair-receiver-readout.v1",
            "report": _project(self),
        }


def assess_sine_pair_receiver_readout(
    *,
    phase_rotation,
    horizon_tau,
    form_error_bound,
    phase_error_bound,
    readout_error_bound,
) -> SinePairReceiverReadout:
    """Bound pair 1's absolute mean-form readout for both selected orientations.

    q=a+ib is an exact Cartesian unit phasor. Require 0<horizon_tau<1/2
    and nonnegative uniform all-node form/lift errors and scalar readout error.
    The fixed-law cubic centers are -b*h/2+a*b*h**3/6 and
    -b*h/2+b*(2*a+1)*h**3/24. Each nominal remainder is at most
    8*h**5/15+8*h**7/105, retaining both incident block currents.

    Coupled full-field comparison bounds the receiver preparation error by
    (form_error_bound+2*h*phase_error_bound)/(1-4*h**2). Add the readout error
    and nominal remainder before outward interval construction. A strict gap
    gives disjoint enclosures; overlap supplies no assertion of equal responses.
    No measured response, cached report or initial-state inverse is consumed.
    """
    from .phase_response import _admit_relative_phasors

    q = _admit_relative_phasors((phase_rotation,), "phase_rotation")[0]
    horizon = exact_or_represented_real(horizon_tau, "horizon_tau")
    if not 0 < horizon < Q(1, 2):
        raise ValueError("horizon_tau must lie strictly between zero and one half")
    budgets = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_error_bound, "form_error_bound"),
            (phase_error_bound, "phase_error_bound"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    if any(value < 0 for value in budgets):
        raise ValueError("preparation and readout error bounds must be nonnegative")
    form_error, phase_error, readout_error = budgets
    (
        comparison_phasors,
        control_phasors,
        comparison_states,
        control_states,
    ) = _sine_pair_exchange_preparations(q)

    linear, cubic = _sine_pair_receiver_center_coefficients(q)
    centers = tuple(
        linear * horizon + coefficient * horizon**3 for coefficient in cubic
    )
    remainder = _sine_pair_finite_remainder(horizon)
    nominal = tuple(I(center - remainder, center + remainder) for center in centers)
    form_amplification = 1 / (1 - 4 * horizon**2)
    phase_amplification = 2 * horizon * form_amplification
    preparation_error = (
        form_amplification * form_error + phase_amplification * phase_error
    )
    uncertainty = preparation_error + readout_error
    expanded = tuple(
        I(center - remainder - uncertainty, center + remainder + uncertainty)
        for center in centers
    )
    gap = max(expanded[1].lo - expanded[0].hi, expanded[0].lo - expanded[1].hi)
    higher = (
        "real_antipodal"
        if expanded[0].lo > expanded[1].hi
        else "imaginary_antipodal" if expanded[1].lo > expanded[0].hi else None
    )
    control_bounds = I(-uncertainty, uncertainty)
    return SinePairReceiverReadout(
        phase_rotation=q,
        horizon_tau=horizon,
        form_error_bound=form_error,
        phase_error_bound=phase_error,
        readout_error_bound=readout_error,
        comparison_initial_phasors=comparison_phasors,
        control_initial_phasors=control_phasors,
        comparison_initial_states=comparison_states,
        control_initial_states=control_states,
        nominal_linear_coefficient=linear,
        nominal_cubic_coefficients=cubic,
        nominal_readout_centers=centers,
        nominal_remainder_upper_bound=remainder,
        nominal_readout_bounds=nominal,
        form_error_amplification_upper_bound=form_amplification,
        phase_error_amplification_upper_bound=phase_amplification,
        propagated_preparation_error_upper_bound=preparation_error,
        readout_uncertainty_radius=uncertainty,
        expanded_readout_bounds=expanded,
        readout_gap_lower_bound=gap,
        status="certified_disjoint" if gap > 0 else "unavailable",
        unavailable_reason=(
            "receiver_readout_intervals_overlap_or_touch" if gap <= 0 else None
        ),
        higher_readout_orientation=higher,
        symmetry_control_uncertainty_radius=uncertainty,
        symmetry_control_readout_bounds=(control_bounds, control_bounds),
    )


@dataclass(frozen=True)
class SinePairReceiverConfounding:
    """Existence of a law/orientation collision in one absolute receiver readout.

    The real-antipodal A preparation varies over the complete conservative
    current family j_epsilon(delta)=sin(delta)+epsilon*sin(delta)**3. The
    imaginary-antipodal B reference retains epsilon=0. Their full initial
    coordinates, support, capacities, phase row and elapsed clock are fixed.
    Every family member has its own phase potential and conserved storage.

    Endpoint intervals enclose X1_A(epsilon,T)-X1_B(0,T) at epsilon=0 and
    epsilon_upper. Strict opposite signs and continuous dependence on epsilon
    prove at least one exact equality at an interior coefficient. Overlapping
    intervals, a zero-containing endpoint, or equal endpoint signs establish
    no such result. The certificate supplies neither a root estimate nor its
    uniqueness, nor an observed response or a numerical trajectory.
    """

    phase_rotation: tuple[Q, Q]
    horizon_tau: Q
    epsilon_upper: Q
    coefficient_endpoints: tuple[Q, Q]
    comparison_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    orientation_a_linear_coefficients: tuple[Q, Q]
    orientation_a_cubic_coefficients: tuple[Q, Q]
    orientation_a_centers: tuple[Q, Q]
    orientation_a_remainder_bounds: tuple[Q, Q]
    orientation_b_reference_linear_coefficient: Q
    orientation_b_reference_cubic_coefficient: Q
    orientation_b_reference_center: Q
    orientation_b_reference_remainder: Q
    current_value_upper_bounds: tuple[Q, Q]
    current_first_derivative_upper_bounds: tuple[Q, Q]
    current_second_derivative_upper_bounds: tuple[Q, Q]
    current_third_derivative_upper_bounds: tuple[Q, Q]
    endpoint_difference_centers: tuple[Q, Q]
    endpoint_difference_bounds: tuple[I, I]
    endpoint_difference_signs: tuple[int, int]
    collision_parameter_open_interval: tuple[Q, Q] | None
    status: str
    unavailable_reason: str | None
    orientation_order: tuple[str, str] = (
        "real_antipodal",
        "imaginary_antipodal",
    )
    receiver_pair_index: int = 1
    receiver_nodes: tuple[int, int] = (2, 3)
    nodes: tuple[int, ...] = tuple(range(10))
    initial_forms: tuple[Q, ...] = (Q(0),) * 10
    held_capacities: tuple[Q, ...] = (Q(1),) * 10
    form_loss: Q = Q(0)
    exchange_weight: Q = Q(1)
    beta: Q = Q(1)
    law: str = "normalized_sine_cubic_reciprocal_exchange"
    reference_law: str = "normalized_sine_reciprocal_exchange"
    phase_potential: str = "1-cos(delta)+epsilon*(2/3-cos(delta)+cos(delta)**3/3)"
    clock: str = "tau=t/pi"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "same_exact_zero_forms_and_full_initial_phasors_for_every_coefficient",
        "A_is_real_antipodal_with_epsilon_varying_B_is_imaginary_antipodal_sine_reference",
        "receiver_pair_one_absolute_mean_form_retains_both_incoming_currents",
        "coefficient_interval_is_declared_before_evaluating_its_endpoint_contrasts",
        "complete_cubic_current_family_retains_the_linear_form_to_phase_row",
        "each_coefficient_conserves_its_own_full_storage_not_the_sine_storage",
        "global_nonlinear_remainders_include_the_full_evolving_environment",
        "strict_opposite_reported_interval_signs_and_continuity_prove_interior_equality",
        "collision_parameter_open_interval_excludes_both_endpoints_and_estimates_no_root",
        "unavailable_endpoint_signs_do_not_prove_absence_or_presence_of_a_collision",
        "single_scalar_equality_does_not_identify_full_states_or_complete_histories",
        "no_preparation_noise_readout_noise_clock_uncertainty_or_fitted_coefficient",
        "no_runtime_installation_trajectory_evaluation_or_physical_law_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-pair-receiver-confounding.v1",
            "report": _project(self),
        }


def assess_sine_pair_receiver_confounding(
    *, phase_rotation, horizon_tau, epsilon_upper
) -> SinePairReceiverConfounding:
    """Certify an interior coefficient with the same scalar receiver response.

    Admit an exact Cartesian unit q=a+ib, positive finite horizon h in tau=t/pi
    and positive finite epsilon_upper. These are declared experiment inputs,
    never fitted to a response. At each coefficient endpoint epsilon, let
    g=b+epsilon*b**3. The A center is -g*h/2+a*(1+3*epsilon*b**2)*g*h**3/6;
    the B reference is the existing epsilon-zero imaginary-antipodal center.

    Full-field derivative bounds give a separate remainder for each center.
    Strict opposite signs of the two outward A-minus-B intervals certify an
    exact collision for some epsilon strictly between zero and epsilon_upper.
    A zero-containing interval is unresolved, not an equality certificate.
    No root solver, trajectory evaluation, cached report or response fitting
    enters this calculation. Positive horizons have no artificial upper cap.
    """
    from .phase_response import _admit_relative_phasors

    q = _admit_relative_phasors((phase_rotation,), "phase_rotation")[0]
    horizon = exact_or_represented_real(horizon_tau, "horizon_tau")
    upper = exact_or_represented_real(epsilon_upper, "epsilon_upper")
    if horizon <= 0 or upper <= 0:
        raise ValueError("horizon_tau and epsilon_upper must be strictly positive")
    endpoints = (Q(0), upper)
    comparison_phasors, _ = _sine_pair_exchange_phasors(q)
    coefficients = tuple(
        _sine_pair_cubic_receiver_a_coefficients(q, epsilon) for epsilon in endpoints
    )
    linear = tuple(pair[0] for pair in coefficients)
    cubic = tuple(pair[1] for pair in coefficients)
    centers = tuple(
        first * horizon + third * horizon**3 for first, third in coefficients
    )
    remainders = tuple(
        _sine_pair_cubic_receiver_remainder(horizon, epsilon) for epsilon in endpoints
    )
    reference_linear, reference_cubic = _sine_pair_receiver_center_coefficients(q)
    reference_center = reference_linear * horizon + reference_cubic[1] * horizon**3
    reference_remainder = _sine_pair_finite_remainder(horizon)
    differences = tuple(center - reference_center for center in centers)
    bounds = tuple(
        I(center - error - reference_remainder, center + error + reference_remainder)
        for center, error in zip(differences, remainders)
    )
    signs = tuple(1 if bound.lo > 0 else -1 if bound.hi < 0 else 0 for bound in bounds)
    certified = signs[0] * signs[1] == -1
    derivative_bounds = tuple(
        _sine_pair_cubic_current_bounds(epsilon) for epsilon in endpoints
    )
    return SinePairReceiverConfounding(
        phase_rotation=q,
        horizon_tau=horizon,
        epsilon_upper=upper,
        coefficient_endpoints=endpoints,
        comparison_initial_phasors=comparison_phasors,
        orientation_a_linear_coefficients=linear,
        orientation_a_cubic_coefficients=cubic,
        orientation_a_centers=centers,
        orientation_a_remainder_bounds=remainders,
        orientation_b_reference_linear_coefficient=reference_linear,
        orientation_b_reference_cubic_coefficient=reference_cubic[1],
        orientation_b_reference_center=reference_center,
        orientation_b_reference_remainder=reference_remainder,
        current_value_upper_bounds=tuple(row[0] for row in derivative_bounds),
        current_first_derivative_upper_bounds=tuple(
            row[1] for row in derivative_bounds
        ),
        current_second_derivative_upper_bounds=tuple(
            row[2] for row in derivative_bounds
        ),
        current_third_derivative_upper_bounds=tuple(
            row[3] for row in derivative_bounds
        ),
        endpoint_difference_centers=differences,
        endpoint_difference_bounds=bounds,
        endpoint_difference_signs=signs,
        collision_parameter_open_interval=endpoints if certified else None,
        status="certified_collision_exists" if certified else "unavailable",
        unavailable_reason=(
            None
            if certified
            else "endpoint_contrasts_do_not_have_strict_opposite_signs"
        ),
    )


def _sine_pair_cubic_preparation_error(horizon, epsilon, form_error, phase_error):
    """Rational full-field form-error majorant in the declared structural clock."""
    first = _sine_pair_cubic_current_bounds(epsilon)[1]
    return (form_error + 2 * first * horizon * phase_error) / (
        1 - 4 * first * horizon**2
    )


@dataclass(frozen=True)
class SinePairReceiverTwoTime:
    """Joint necessary-coefficient test for two readings of the same receiver.

    One constant cubic-current coefficient and one uncertain initial state
    evolve through both readings in the A family. B retains the sine law and
    its own single uncertain initial state. The nominal full preparations are
    the earlier finite-confounding preparations; every coordinate may vary
    within the separately supplied form and circular-phase preparation bounds.

    Matching an admitted record at each time requires the same coefficient in
    both ``necessary_coefficient_bounds`` and in ``coefficient_endpoints``'s
    closed domain. These outward intervals are deliberately not clipped to the
    domain. Strict disjointness is sufficient to exclude a joint match; overlap
    neither supplies a coefficient estimate nor proves a realizable collision.

    Affine bounds concern A-minus-B recorded form differences, including both
    branches' errors: N-A*epsilon <= difference <= U-B*epsilon. Positive slopes
    yield the necessary interval [N/A,U/B]. If these intervals separate, the
    reported weights cancel the common coefficient between two affine bounds.
    Their outward gap lower bound is in the same form units as the records.
    Status follows the coefficient intervals; outward rounding can make an
    extremely small optional readout gap zero without invalidating that test.

    No observed response, trajectory solver, search or fitted coefficient is
    consumed. The certificate distinguishes only the two declared families.
    """

    phase_rotation: tuple[Q, Q]
    horizons_tau: tuple[Q, Q]
    epsilon_upper: Q
    form_error_bound: Q
    phase_error_bound: Q
    readout_error_bound: Q
    coefficient_endpoints: tuple[Q, Q]
    comparison_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    control_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    contrast_linear_epsilon_coefficient: Q
    contrast_cubic_coefficients: tuple[Q, Q, Q]
    remainder_upper_bounds_by_time: tuple[tuple[Q, Q], ...]
    preparation_error_bounds_by_time: tuple[tuple[Q, Q], ...]
    reference_contrast_radii: tuple[Q, Q]
    uncertainty_chord_slopes: tuple[Q, Q]
    lower_affine_intercepts: tuple[Q, Q]
    lower_affine_epsilon_slopes: tuple[Q, Q]
    upper_affine_intercepts: tuple[Q, Q]
    upper_affine_epsilon_slopes: tuple[Q, Q]
    necessary_coefficient_bounds: tuple[I, I] | None
    coefficient_gap_lower_bound: Q | None
    joint_readout_weights: tuple[Q, Q] | None
    joint_readout_gap_lower_bound: Q | None
    status: str
    unavailable_reason: str | None
    orientation_order: tuple[str, str] = (
        "real_antipodal",
        "imaginary_antipodal",
    )
    receiver_pair_index: int = 1
    receiver_nodes: tuple[int, int] = (2, 3)
    nodes: tuple[int, ...] = tuple(range(10))
    initial_forms: tuple[Q, ...] = (Q(0),) * 10
    held_capacities: tuple[Q, ...] = (Q(1),) * 10
    nominal_control_readout_centers: tuple[Q, Q] = (Q(0), Q(0))
    form_loss: Q = Q(0)
    exchange_weight: Q = Q(1)
    beta: Q = Q(1)
    law: str = "normalized_sine_cubic_reciprocal_exchange"
    reference_law: str = "normalized_sine_reciprocal_exchange"
    phase_potential: str = "1-cos(delta)+epsilon*(2/3-cos(delta)+cos(delta)**3/3)"
    clock: str = "tau=t/pi"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "same_nominal_full_preparations_as_the_single_time_constitutive_collision",
        "A_real_antipodal_cubic_family_against_B_imaginary_antipodal_sine_reference",
        "one_constant_coefficient_and_one_initial_state_per_branch_for_both_times",
        "same_receiver_absolute_mean_form_at_two_strictly_ordered_structural_times",
        "all_ten_form_and_circular_phase_preparation_errors_admitted_independently",
        "initial_absolute_form_origin_is_bounded_with_the_other_fine_coordinates",
        "readout_errors_may_be_correlated_within_the_componentwise_absolute_bound",
        "complete_law_derivative_remainders_retain_the_evolving_environment",
        "convex_coefficient_chords_bound_remainders_and_preparation_errors",
        "rational_preparation_majorant_domain_is_not_a_physical_stability_limit",
        "necessary_coefficient_intervals_are_unclipped_outward_dyadic_enclosures",
        "strict_interval_separation_excludes_a_shared_coefficient_for_equal_records",
        "optional_affine_readout_weights_have_unit_sum_of_absolute_values",
        "interval_overlap_does_not_prove_a_realizable_collision_or_fit_a_coefficient",
        "each_actual_trajectory_has_its_own_law_specific_conserved_storage",
        "only_nominal_symmetry_controls_are_stationary_uncertain_controls_may_evolve",
        "no_unknown_clock_forcing_events_or_time_varying_coefficient",
        "no_response_input_trajectory_evaluation_parameter_search_or_sensor_calibration",
        "no_general_state_reconstruction_unique_physical_law_or_constituent_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-pair-receiver-two-time.v1",
            "report": _project(self),
        }


def assess_sine_pair_receiver_two_time(
    *,
    phase_rotation,
    horizons_tau,
    epsilon_upper,
    form_error_bound,
    phase_error_bound,
    readout_error_bound,
) -> SinePairReceiverTwoTime:
    """Test two receiver records for the same constitutive/orientation ambiguity.

    Supply exact Cartesian unit q=a+ib with a>1/2 and b>0, two strictly
    increasing positive horizons, positive epsilon_upper, and nonnegative
    all-node form/phase preparation and per-scalar readout error bounds.
    Exact rational inputs remain exact; other admitted reals follow the shared
    represented-real boundary. The comparison requires
    4*(1+3*epsilon_upper)*max(horizons_tau)**2 < 1 for its rational error
    majorant. This is a sufficient analytic domain, not a stability assertion.

    The nominal difference is -c*epsilon*h+(D0+D1*epsilon+D2*epsilon**2)*h**3.
    Full-law Taylor remainders and coupled form/phase preparation errors are
    convex in epsilon. Their endpoint chord gives R_epsilon<=R0+Q*epsilon.
    Keeping the positive cubic linear term in the lower bound and bounding
    epsilon**2 by epsilon_upper*epsilon in the upper bound yields N-A*epsilon
    <= recorded_difference <= U-B*epsilon. Positive A,B give [N/A,U/B].

    Report outward intervals without clipping to [0,epsilon_upper]. Strict
    separation excludes any coefficient matching both admitted records.
    Nonpositive slopes or overlapping/touching intervals remain unavailable.
    The optional weighted readout margin uses the same affine inequalities,
    not a new trajectory approximation or measured response.
    """
    from .phase_response import _admit_relative_phasors

    q = _admit_relative_phasors((phase_rotation,), "phase_rotation")[0]
    a, b = q
    if a <= Q(1, 2) or b <= 0:
        raise ValueError(
            "phase_rotation must satisfy real part > 1/2 and imaginary part > 0"
        )
    horizons = _global_pair_scalars(horizons_tau, "horizons_tau", 2)
    if not 0 < horizons[0] < horizons[1]:
        raise ValueError(
            "horizons_tau must contain two strictly increasing positive values"
        )
    upper = exact_or_represented_real(epsilon_upper, "epsilon_upper")
    if upper <= 0:
        raise ValueError("epsilon_upper must be strictly positive")
    budgets = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_error_bound, "form_error_bound"),
            (phase_error_bound, "phase_error_bound"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    if any(value < 0 for value in budgets):
        raise ValueError("preparation and readout error bounds must be nonnegative")
    form_error, phase_error, readout_error = budgets
    if 4 * (1 + 3 * upper) * horizons[1] ** 2 >= 1:
        raise ValueError(
            "preparation majorant requires 4*(1+3*epsilon_upper)*max(horizons_tau)**2 < 1"
        )

    endpoints = (Q(0), upper)
    comparison_phasors, control_phasors = _sine_pair_exchange_phasors(q)
    remainders = tuple(
        tuple(_sine_pair_cubic_receiver_remainder(h, epsilon) for epsilon in endpoints)
        for h in horizons
    )
    preparation = tuple(
        tuple(
            _sine_pair_cubic_preparation_error(h, epsilon, form_error, phase_error)
            for epsilon in endpoints
        )
        for h in horizons
    )
    radii = tuple(
        2 * (errors[0] + initial[0] + readout_error)
        for errors, initial in zip(remainders, preparation)
    )
    chord_slopes = tuple(
        (errors[1] - errors[0] + initial[1] - initial[0]) / upper
        for errors, initial in zip(remainders, preparation)
    )
    linear = b**3 / 2
    cubic = (b * (2 * a - 1) / 24, 2 * a * b**3 / 3, a * b**5 / 2)
    d0, d1, d2 = cubic
    lower_intercepts = tuple(d0 * h**3 - radius for h, radius in zip(horizons, radii))
    upper_intercepts = tuple(d0 * h**3 + radius for h, radius in zip(horizons, radii))
    lower_slopes = tuple(
        linear * h - d1 * h**3 + slope for h, slope in zip(horizons, chord_slopes)
    )
    upper_slopes = tuple(
        linear * h - (d1 + d2 * upper) * h**3 - slope
        for h, slope in zip(horizons, chord_slopes)
    )
    bounds = None
    gap = None
    weights = None
    readout_gap = None
    status = "unavailable"
    reason = "coefficient_interval_slope_not_positive"
    if all(slope > 0 for slope in (*lower_slopes, *upper_slopes)):
        bounds = tuple(
            I(lower / lower_slope, higher / upper_slope)
            for lower, higher, lower_slope, upper_slope in zip(
                lower_intercepts, upper_intercepts, lower_slopes, upper_slopes
            )
        )
        directed_gaps = (bounds[0].lo - bounds[1].hi, bounds[1].lo - bounds[0].hi)
        gap = max(directed_gaps)
        reason = "necessary_coefficient_intervals_overlap_or_touch"
        if gap > 0:
            higher = 0 if directed_gaps[0] > 0 else 1
            lower = 1 - higher
            denominator = lower_slopes[higher] + upper_slopes[lower]
            weights = tuple(
                (
                    upper_slopes[lower] / denominator
                    if index == higher
                    else -lower_slopes[higher] / denominator
                )
                for index in range(2)
            )
            margin = (
                upper_slopes[lower] * lower_intercepts[higher]
                - lower_slopes[higher] * upper_intercepts[lower]
            ) / denominator
            readout_gap = I(margin).lo
            status = "certified_disjoint"
            reason = None

    return SinePairReceiverTwoTime(
        phase_rotation=q,
        horizons_tau=horizons,
        epsilon_upper=upper,
        form_error_bound=form_error,
        phase_error_bound=phase_error,
        readout_error_bound=readout_error,
        coefficient_endpoints=endpoints,
        comparison_initial_phasors=comparison_phasors,
        control_initial_phasors=control_phasors,
        contrast_linear_epsilon_coefficient=linear,
        contrast_cubic_coefficients=cubic,
        remainder_upper_bounds_by_time=remainders,
        preparation_error_bounds_by_time=preparation,
        reference_contrast_radii=radii,
        uncertainty_chord_slopes=chord_slopes,
        lower_affine_intercepts=lower_intercepts,
        lower_affine_epsilon_slopes=lower_slopes,
        upper_affine_intercepts=upper_intercepts,
        upper_affine_epsilon_slopes=upper_slopes,
        necessary_coefficient_bounds=bounds,
        coefficient_gap_lower_bound=gap,
        joint_readout_weights=weights,
        joint_readout_gap_lower_bound=readout_gap,
        status=status,
        unavailable_reason=reason,
    )


def _sine_pair_cubic_initial_rate_remainder(horizon, epsilon, initial_rate):
    """Bound nominal form growth, rate and cubic remainder from x(0)=0."""
    _, first, second, third = _sine_pair_cubic_current_bounds(epsilon)
    growth = initial_rate / (1 - Q(2, 3) * first * horizon**2)
    rate = initial_rate + 2 * first * growth * horizon**2
    remainder = (
        48 * second * growth * rate + 16 * first**2 * growth
    ) * horizon**5 / 120 + 64 * third * growth**3 * horizon**7 / 840
    return growth, rate, remainder


@dataclass(frozen=True)
class SinePairReceiverTwoLaw:
    """Uniform two-reading separation with independent constant cubic laws.

    The A and B branches each choose one coefficient in the same closed
    ``coefficient_endpoints`` domain and one perturbed complete initial state.
    Each choice persists through both times. The declared readout weights
    cancel the linear term separately in each branch. Exact extrema of each
    nominal cubic coefficient and full-law error radii then bound every
    A-minus-B joint recorded value in ``joint_difference_bounds``.

    Polynomial coefficients, extrema, initial-rate bounds and error radii use
    ``orientation_order``. Rows named ``by_time`` follow ``horizons_tau``;
    two-component rows within them follow ``orientation_order``. Preparation
    bounds are common to both orientations and have one entry per time.
    A form-growth entry X bounds max_i |x_i(t)| by X*t through that horizon;
    a form-rate entry V bounds max_i |x_i'(t)| on the same nominal interval.
    The current coefficient upper endpoint bounds the nominal remainders and
    preparation amplification uniformly. Their rational domain is sufficient
    for this proof, not a physical stability limit.

    Status requires the direct outward difference lower bound to be strictly
    positive. Failure of that sufficient bound does not establish a collision.
    No response input, fitted coefficient or numerical trajectory is consumed.
    """

    phase_rotation: tuple[Q, Q]
    horizons_tau: tuple[Q, Q]
    epsilon_upper: Q
    form_error_bound: Q
    phase_error_bound: Q
    readout_error_bound: Q
    coefficient_endpoints: tuple[Q, Q]
    comparison_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    control_initial_phasors: tuple[tuple[tuple[Q, Q], ...], ...]
    readout_weights: tuple[Q, Q]
    cubic_time_factor: Q
    nominal_cubic_coefficients: tuple[tuple[Q, Q, Q], ...]
    nominal_cubic_coefficient_ranges: tuple[tuple[Q, Q], ...]
    nominal_initial_rate_bounds: tuple[Q, Q]
    form_growth_bounds_by_time: tuple[tuple[Q, Q], ...]
    form_rate_bounds_by_time: tuple[tuple[Q, Q], ...]
    remainder_upper_bounds_by_time: tuple[tuple[Q, Q], ...]
    preparation_error_bounds_by_time: tuple[Q, Q]
    joint_error_radii: tuple[Q, Q]
    joint_difference_bounds: I
    status: str
    unavailable_reason: str | None
    orientation_order: tuple[str, str] = (
        "real_antipodal",
        "imaginary_antipodal",
    )
    receiver_pair_index: int = 1
    receiver_nodes: tuple[int, int] = (2, 3)
    nodes: tuple[int, ...] = tuple(range(10))
    initial_forms: tuple[Q, ...] = (Q(0),) * 10
    held_capacities: tuple[Q, ...] = (Q(1),) * 10
    nominal_control_readout_centers: tuple[Q, Q] = (Q(0), Q(0))
    form_loss: Q = Q(0)
    exchange_weight: Q = Q(1)
    phase_exchange_beta: Q = Q(1)
    law: str = "normalized_sine_cubic_reciprocal_exchange"
    phase_potential: str = "1-cos(delta)+epsilon*(2/3-cos(delta)+cos(delta)**3/3)"
    clock: str = "tau=t/pi"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "same_nominal_full_preparations_as_the_single_time_constitutive_collision",
        "each_orientation_has_its_own_independent_constant_cubic_current_coefficient",
        "one_coefficient_and_one_initial_state_per_branch_persist_through_both_times",
        "same_receiver_absolute_mean_form_at_two_strictly_ordered_structural_times",
        "fixed_time_weights_cancel_each_branch_linear_term_and_have_absolute_sum_one",
        "all_ten_form_and_circular_phase_preparation_errors_admitted_independently",
        "initial_absolute_form_origin_is_bounded_with_the_other_fine_coordinates",
        "readout_errors_may_be_correlated_within_the_componentwise_absolute_bound",
        "exact_quadratic_extrema_retain_each_law_coefficient_across_both_times",
        "nominal_initial_rate_remainders_retain_the_complete_evolving_environment",
        "coupled_preparation_majorants_compare_each_perturbed_path_with_its_own_law",
        "rational_majorant_domain_is_not_a_physical_stability_limit",
        "strict_positive_outward_joint_difference_certifies_disjoint_record_sets",
        "unavailable_does_not_prove_a_realizable_collision_or_fit_a_coefficient",
        "each_actual_trajectory_has_its_own_law_specific_conserved_storage",
        "only_nominal_symmetry_controls_are_stationary_uncertain_controls_may_evolve",
        "no_unknown_clock_forcing_events_or_time_varying_coefficient",
        "no_response_input_trajectory_evaluation_parameter_search_or_sensor_calibration",
        "no_general_state_reconstruction_unique_physical_law_or_constituent_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-pair-receiver-two-law.v1",
            "report": _project(self),
        }


def assess_sine_pair_receiver_two_law(
    *,
    phase_rotation,
    horizons_tau,
    epsilon_upper,
    form_error_bound,
    phase_error_bound,
    readout_error_bound,
) -> SinePairReceiverTwoLaw:
    """Bound the fixed weighted readout for two independently unknown laws.

    Supply exact Cartesian unit q=a+ib with a>1/2 and b>0, two strictly
    increasing positive horizons, positive epsilon_upper, and nonnegative
    all-node form/phase preparation and per-scalar readout error bounds.
    Exact rational inputs remain exact; other admitted reals follow shared
    represented-real admission. The sufficient majorant domain is
    4*(1+3*epsilon_upper)*max(horizons_tau)**2 < 1.

    The weights (-h1,h0)/(h0+h1) cancel the linear time coefficient for each
    law separately. The remaining cubic factor is h0*h1*(h1-h0)>0. Both
    nominal cubic coefficient polynomials increase on [0,epsilon_upper],
    giving exact endpoint extrema without parameter search. Uniform nominal
    Taylor remainders use the largest actual initial fine-node form rate for
    each family, with complete evolving form and phase rows. Separate-channel
    preparation majorants compare each perturbed trajectory to its own nominal
    law; circular phase errors admit continuous initial lifts for this bound.

    Each recorded scalar has its supplied error allowance. Correlation across
    times is unrestricted. The joint A-minus-B interval includes every error
    before outward rounding. A strictly positive lower endpoint certifies
    separation; otherwise the result is explicitly unavailable.
    """
    from .phase_response import _admit_relative_phasors

    q = _admit_relative_phasors((phase_rotation,), "phase_rotation")[0]
    a, b = q
    if a <= Q(1, 2) or b <= 0:
        raise ValueError(
            "phase_rotation must satisfy real part > 1/2 and imaginary part > 0"
        )
    horizons = _global_pair_scalars(horizons_tau, "horizons_tau", 2)
    if not 0 < horizons[0] < horizons[1]:
        raise ValueError(
            "horizons_tau must contain two strictly increasing positive values"
        )
    upper = exact_or_represented_real(epsilon_upper, "epsilon_upper")
    if upper <= 0:
        raise ValueError("epsilon_upper must be strictly positive")
    budgets = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_error_bound, "form_error_bound"),
            (phase_error_bound, "phase_error_bound"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    if any(value < 0 for value in budgets):
        raise ValueError("preparation and readout error bounds must be nonnegative")
    form_error, phase_error, readout_error = budgets
    if 4 * (1 + 3 * upper) * horizons[1] ** 2 >= 1:
        raise ValueError(
            "preparation majorant requires 4*(1+3*epsilon_upper)*max(horizons_tau)**2 < 1"
        )

    endpoints = (Q(0), upper)
    comparison_phasors, control_phasors = _sine_pair_exchange_phasors(q)
    h0, h1 = horizons
    weights = (-h1 / (h0 + h1), h0 / (h0 + h1))
    factor = h0 * h1 * (h1 - h0)
    cubic = (
        (a * b / 6, 2 * a * b**3 / 3, a * b**5 / 2),
        (
            b * (2 * a + 1) / 24,
            b * (1 - a**3 + 3 * a**2 * (1 - a) + 12 * a * b**2) / 24,
            b * (3 * a**2 * (1 - a**3) + 9 * a * b**4) / 24,
        ),
    )
    cubic_ranges = tuple((c0, c0 + c1 * upper + c2 * upper**2) for c0, c1, c2 in cubic)
    g = b + upper * b**3
    d = 1 - a + upper * (1 - a**3)
    initial_rates = (g / 2, max(g, d) / 2)
    growth_rate_remainders = tuple(
        tuple(
            _sine_pair_cubic_initial_rate_remainder(h, upper, rate)
            for rate in initial_rates
        )
        for h in horizons
    )
    growth = tuple(tuple(row[0] for row in rows) for rows in growth_rate_remainders)
    rates = tuple(tuple(row[1] for row in rows) for rows in growth_rate_remainders)
    remainders = tuple(tuple(row[2] for row in rows) for rows in growth_rate_remainders)
    preparation = tuple(
        _sine_pair_cubic_preparation_error(h, upper, form_error, phase_error)
        for h in horizons
    )
    radii = tuple(
        sum(
            (
                abs(weight) * (errors[orientation] + initial)
                for weight, errors, initial in zip(weights, remainders, preparation)
            ),
            Q(0),
        )
        + readout_error
        for orientation in range(2)
    )
    bounds = I(
        factor * (cubic_ranges[0][0] - cubic_ranges[1][1]) - sum(radii),
        factor * (cubic_ranges[0][1] - cubic_ranges[1][0]) + sum(radii),
    )
    certified = bounds.lo > 0
    return SinePairReceiverTwoLaw(
        phase_rotation=q,
        horizons_tau=horizons,
        epsilon_upper=upper,
        form_error_bound=form_error,
        phase_error_bound=phase_error,
        readout_error_bound=readout_error,
        coefficient_endpoints=endpoints,
        comparison_initial_phasors=comparison_phasors,
        control_initial_phasors=control_phasors,
        readout_weights=weights,
        cubic_time_factor=factor,
        nominal_cubic_coefficients=cubic,
        nominal_cubic_coefficient_ranges=cubic_ranges,
        nominal_initial_rate_bounds=initial_rates,
        form_growth_bounds_by_time=growth,
        form_rate_bounds_by_time=rates,
        remainder_upper_bounds_by_time=remainders,
        preparation_error_bounds_by_time=preparation,
        joint_error_radii=radii,
        joint_difference_bounds=bounds,
        status="certified_disjoint" if certified else "unavailable",
        unavailable_reason=(
            None if certified else "joint_readout_difference_not_strictly_positive"
        ),
    )


@dataclass(frozen=True)
class SinePairReceiverDefect:
    """Two-reading separation under bounded complete-row evolution defects.

    ``reference_certificate`` is rebuilt internally from primitive inputs. It
    retains the exact zero-defect law family's nominal preparation, cubic
    coefficients, remainders, preparation/readout budgets and observation
    weights. Its conservation and stationary-control premises describe that
    reference family only. The present certificate admits additive form and
    phase residuals on every node throughout the complete time window.

    Response factors multiply their respective rate-defect bounds. Their sums
    bound the additional receiver error at each reference horizon. The
    ``joint_defect_radius`` is the weighted extra error for either branch;
    ``joint_error_radii`` add it to both exact reference radii in orientation
    order. ``joint_difference_bounds`` encloses the resulting A-minus-B
    recorded contrast directly, without using the reference rounded interval
    or verdict as a premise.

    Residual histories are measurable and essentially bounded, may differ
    between branches, and need not preserve total form or own-law storage.
    Each branch retains one coefficient, initial state and residual history
    across both readings. No residual source, trajectory or measured response
    is inferred or installed by this detached assessment.
    """

    reference_certificate: SinePairReceiverTwoLaw
    form_rate_defect_bound: Q
    phase_rate_defect_bound: Q
    form_defect_response_factors_by_time: tuple[Q, Q]
    phase_defect_response_factors_by_time: tuple[Q, Q]
    defect_error_bounds_by_time: tuple[Q, Q]
    joint_defect_radius: Q
    joint_error_radii: tuple[Q, Q]
    joint_difference_bounds: I
    status: str
    unavailable_reason: str | None
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "reference_certificate_rebuilt_from_primitives_not_an_incoming_report",
        "reference_conservation_and_stationary_control_apply_only_without_defects",
        "same_fixed_doubled_C5_support_unit_capacities_and_tau_clock_as_reference",
        "independent_constant_cubic_current_coefficients_for_the_two_orientations",
        "one_initial_state_coefficient_and_residual_history_per_branch_for_both_times",
        "additive_defects_in_every_fine_form_and_phase_evolution_row",
        "form_rate_defect_in_form_units_per_tau_phase_rate_defect_in_radians_per_tau",
        "measurable_residual_bounds_hold_almost_everywhere_on_the_entire_time_window",
        "absolutely_continuous_full_state_paths_and_continuous_phase_lifts",
        "residuals_may_be_arbitrarily_correlated_across_nodes_times_and_branches",
        "same_law_comparison_uses_no_residual_derivatives_or_defective_path_Taylor_jet",
        "preparation_readout_and_continuous_defect_budgets_remain_distinct",
        "total_form_and_own_law_storage_include_residual_source_and_work_terms",
        "no_stationary_control_claim_for_nonzero_defects_or_perturbed_preparations",
        "rational_majorant_domain_is_not_a_physical_stability_limit",
        "strict_positive_outward_joint_difference_certifies_disjoint_record_sets",
        "unavailable_does_not_prove_a_realizable_collision_or_fit_a_residual",
        "no_forcing_installation_autonomous_residual_selector_or_trajectory_execution",
        "fixed_reference_clock_and_coefficients_no_support_events_or_state_resets",
        "no_response_derived_tolerance_source_calibration_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-pair-receiver-defect.v1",
            "report": _project(self),
        }


def assess_sine_pair_receiver_defect(
    *,
    phase_rotation,
    horizons_tau,
    epsilon_upper,
    form_error_bound,
    phase_error_bound,
    readout_error_bound,
    form_rate_defect_bound,
    phase_rate_defect_bound,
) -> SinePairReceiverDefect:
    """Expand the two-law certificate by independently supplied row defects.

    Retain every primitive domain of ``assess_sine_pair_receiver_two_law`` and
    supply nonnegative all-node form/phase rate-defect bounds in its tau clock.
    Exact rationals remain exact; other admitted reals use shared represented
    admission. Normalize both defect bounds before rebuilding the reference
    from the other primitive inputs. No incoming certificate is accepted.

    For L=1+3*epsilon_upper, comparison with the inhomogeneous two-channel
    error system gives the additional form bound
    D(h)=(h*delta_x+L*h**2*delta_theta)/(1-4*L*h**2).
    It follows by integrating the matrix-exponential series and bounding its
    factorial coefficients by geometric-series coefficients. The reference
    admission ensures the denominator is positive. The comparison covers
    measurable residuals on the full interval and needs no derivative of them.

    Add sum(abs(weight)*D(h)) separately to each reference error radius, then
    reconstruct the exact contrast extrema before outward materialization.
    Only a strictly positive outward lower bound certifies separation.
    Storage and total form need not be conserved by these admitted paths;
    bounded residuals are premises and do not select a forcing mechanism.
    """
    defects = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_rate_defect_bound, "form_rate_defect_bound"),
            (phase_rate_defect_bound, "phase_rate_defect_bound"),
        )
    )
    if any(value < 0 for value in defects):
        raise ValueError("form and phase rate defect bounds must be nonnegative")
    form_defect, phase_defect = defects
    reference = assess_sine_pair_receiver_two_law(
        phase_rotation=phase_rotation,
        horizons_tau=horizons_tau,
        epsilon_upper=epsilon_upper,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        readout_error_bound=readout_error_bound,
    )
    first = _sine_pair_cubic_current_bounds(reference.epsilon_upper)[1]
    denominators = tuple(1 - 4 * first * h**2 for h in reference.horizons_tau)
    form_factors = tuple(
        h / denominator for h, denominator in zip(reference.horizons_tau, denominators)
    )
    phase_factors = tuple(
        first * h**2 / denominator
        for h, denominator in zip(reference.horizons_tau, denominators)
    )
    errors = tuple(
        form_factor * form_defect + phase_factor * phase_defect
        for form_factor, phase_factor in zip(form_factors, phase_factors)
    )
    joint_defect = sum(
        (
            abs(weight) * error
            for weight, error in zip(reference.readout_weights, errors)
        ),
        Q(0),
    )
    radii = tuple(radius + joint_defect for radius in reference.joint_error_radii)
    cubic_ranges = reference.nominal_cubic_coefficient_ranges
    factor = reference.cubic_time_factor
    bounds = I(
        factor * (cubic_ranges[0][0] - cubic_ranges[1][1]) - sum(radii),
        factor * (cubic_ranges[0][1] - cubic_ranges[1][0]) + sum(radii),
    )
    certified = bounds.lo > 0
    return SinePairReceiverDefect(
        reference_certificate=reference,
        form_rate_defect_bound=form_defect,
        phase_rate_defect_bound=phase_defect,
        form_defect_response_factors_by_time=form_factors,
        phase_defect_response_factors_by_time=phase_factors,
        defect_error_bounds_by_time=errors,
        joint_defect_radius=joint_defect,
        joint_error_radii=radii,
        joint_difference_bounds=bounds,
        status="certified_disjoint" if certified else "unavailable",
        unavailable_reason=(
            None if certified else "joint_readout_difference_not_strictly_positive"
        ),
    )


@dataclass(frozen=True)
class SinePairPersistentResponse:
    """Joint full-state trapping and finite response of two storage allocations.

    The nominal winding-one centers are symbolic: target phases are
    ``2*pi*target_phase_turns``, and only pair zero changes. The phase-split
    preparation has phases ``(+delta,-delta)`` and zero forms. The form-split
    preparation has forms ``(+u,-u)``, where
    ``u**2=2*cos(2*pi/5)*(1-cos(delta))``, and target phases. Their nominal
    collective means and excess storage agree for the same delta. Each
    perturbed branch may choose its own delta and independent fine errors.

    Initial norm and excess-storage bounds cover every fine coordinate.
    Persistence means retention of the acute winding-one chart around the
    exact target, not all-pair activity, attraction or formation. The nominal
    forcing/growth bounds are (D_A,D_B) and (K_A,K_B): A's mean forcing and
    mean form are bounded by D_A and K_A*tau; B's by D_B*tau**2 and
    K_B*tau**3. These reductions are used only for the nominal centers.
    The preparation comparison covers the complete perturbed flow.

    All margins and recorded intervals are outward enclosures. The joint
    status requires both persistence flags and a strictly positive recorded
    A-minus-B contrast. Failure is an unavailable sufficient certificate,
    never a proof of escape or equality. No source graph, trajectory or
    previously computed report is consumed or installed.
    """

    delta_bounds: tuple[Q, Q]
    horizon_tau: Q
    form_error_bound: Q
    phase_error_bound: Q
    readout_error_bound: Q
    radius: Q
    full_dimensional_preparation: bool
    target_angle_bounds: I
    target_cosine_bounds: I
    target_sine_bounds: I
    one_minus_cosine_endpoint_bounds: tuple[I, I]
    nominal_internal_form_squared_bounds: I
    nominal_internal_form_amplitude_bounds: I
    nominal_excess_storage_bounds: I
    nominal_receiver_rate_bounds: I
    nominal_mean_forcing_upper_bounds: tuple[Q, Q]
    nominal_mean_growth_upper_bounds: tuple[Q, Q]
    nominal_response_remainder_upper_bounds: tuple[Q, Q]
    preparation_error_bound: Q
    nominal_readout_bounds: tuple[I, I]
    recorded_readout_bounds: tuple[I, I]
    recorded_difference_bounds: I
    spectral_gap_bounds: I
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    cosine_lower_bound: Q | None
    coercivity_lower_bound: Q | None
    barrier_lower_bound: Q | None
    initial_norm_squared_upper_bounds: tuple[Q, Q]
    initial_excess_storage_upper_bounds: tuple[Q, Q]
    initial_radius_margin_bounds: tuple[I, I]
    storage_barrier_margin_bounds: tuple[I, I] | None
    persistence_certified_by_preparation: tuple[bool, bool]
    persistence_unavailable_reasons_by_preparation: tuple[tuple[str, ...], ...]
    response_separation_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    preparation_order: tuple[str, str] = ("phase_split", "form_split")
    nodes: tuple[int, ...] = tuple(range(10))
    pairs: tuple[tuple[int, int], ...] = tuple((2 * a, 2 * a + 1) for a in range(5))
    base_edges: tuple[tuple[int, int], ...] = _GLOBAL_PAIR_BASE_EDGES
    edges: tuple[tuple[int, int, Q], ...] = tuple(
        sorted(
            (min(2 * a + i, 2 * b + j), max(2 * a + i, 2 * b + j), Q(1))
            for a, b in _GLOBAL_PAIR_BASE_EDGES
            for i in (0, 1)
            for j in (0, 1)
        )
    )
    target_phase_turns: tuple[Q, ...] = tuple(Q(a, 5) for a in range(5) for _ in (0, 1))
    target_forms: tuple[Q, ...] = (Q(0),) * 10
    held_capacities: tuple[Q, ...] = (Q(1),) * 10
    selected_pair_index: int = 0
    receiver_pair_index: int = 1
    receiver_nodes: tuple[int, int] = (2, 3)
    nominal_preparation_formulas: tuple[str, str] = (
        "target_with_pair0_phases_plus_minus_delta_and_zero_forms",
        "target_with_pair0_forms_plus_minus_sqrt(2*cos(2*pi/5)*(1-cos(delta)))",
    )
    form_loss: Q = Q(0)
    exchange_weight: Q = Q(1)
    phase_exchange_beta: Q = Q(1)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=t/pi"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_doubled_C5_all_four_unit_cross_edges_no_internal_pair_edges",
        "all_twenty_coordinates_evolve_under_the_same_unforced_conservative_law",
        "zero_form_loss_unit_exchange_beta_and_positive_held_unit_capacities",
        "symbolic_winding_one_target_no_materialized_phase_center",
        "nominal_means_and_storage_match_only_for_the_same_delta_subcomparison",
        "independent_deltas_and_all_node_form_and_phase_errors_between_branches",
        "initial_circular_phase_errors_admit_continuous_lifts_about_the_target",
        "absolute_form_origin_is_bounded_with_the_other_fine_coordinates",
        "full_dimensional_open_preparation_interiors_only_when_both_budgets_positive",
        "full_state_conserved_storage_first_exit_bound_retains_geometric_identity",
        "no_all_pair_activity_attraction_formation_or_individual_recurrence_claim",
        "nominal_symmetries_do_not_restrict_the_perturbed_preparation_family",
        "complete_flow_error_comparison_is_separate_from_nominal_reduced_dynamics",
        "receiver_absolute_mean_form_at_one_declared_structural_time",
        "strict_outward_persistence_and_response_margins_required_jointly",
        "unavailable_does_not_prove_escape_a_realizable_collision_or_equal_responses",
        "no_forcing_defects_events_resets_or_fitted_pressure",
        "no_trajectory_execution_parameter_search_or_sensor_calibration",
        "no_autonomous_law_selection_physical_binding_or_constituent_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-pair-persistent-response.v1",
            "report": _project(self),
        }


def assess_sine_pair_persistent_response(
    *,
    delta_bounds,
    horizon_tau,
    form_error_bound,
    phase_error_bound,
    readout_error_bound,
    radius,
) -> SinePairPersistentResponse:
    """Bound persistent geometric identity and an internal-allocation response.

    Admit exact/represented real ``0 < delta_lo <= delta_hi <= 1``, positive
    horizon ``h`` with ``2*h**2 < 1``, nonnegative preparation/readout budgets
    and positive relative radius. Rational primitives remain exact. All-node
    preparation errors are independent and cover the closed boxes about the
    symbolic winding-one centers in ``SinePairPersistentResponse``.

    Use the full doubled-C5 gap ``5-sqrt(5)`` and the conserved-storage
    first-exit bound around the target. Both boxes need strict initial
    radius, acute radius and energy-barrier margins. The mean-field bounds
    for the two nominal centers retain their evolving environment; the
    separate complete-flow comparison adds
    ``rho_x/(1-2*h**2) + 2*h*rho_theta/(1-2*h**2/3)`` to either readout.

    Range-safe ``1-cos(delta)=2*sin(delta/2)**2`` retains nonnegative form
    amplitudes even when tiny valid inputs are unresolved on the interval
    grid. Such resolution limits can make the certificate unavailable.
    A positive contrast alone is not the joint certificate. No numerical
    trajectory or materialized approximation of the target is required.
    """
    delta = _global_pair_scalars(delta_bounds, "delta_bounds", 2)
    if not 0 < delta[0] <= delta[1] <= 1:
        raise ValueError("delta_bounds must satisfy 0 < lower <= upper <= 1")
    h = exact_or_represented_real(horizon_tau, "horizon_tau")
    if h <= 0 or 2 * h**2 >= 1:
        raise ValueError("horizon_tau must satisfy h > 0 and 2*h**2 < 1")
    budgets = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_error_bound, "form_error_bound"),
            (phase_error_bound, "phase_error_bound"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    if any(value < 0 for value in budgets):
        raise ValueError("preparation and readout error bounds must be nonnegative")
    rx, rt, eta = budgets
    r = exact_or_represented_real(radius, "radius")
    if r <= 0:
        raise ValueError("radius must be strictly positive")

    pi = pi_interval()
    alpha = 2 * pi / 5
    c, s = cos(alpha), sin(alpha)
    # Monotonicity on [0,1] gives endpoint extrema. The squared half-angle
    # expression proves nonnegativity without clipping a negative enclosure.
    one_minus_cosine = tuple(2 * sin(I(value / 2)) ** 2 for value in delta)
    phase_energy = I(one_minus_cosine[0].lo, one_minus_cosine[1].hi)
    p_squared = 2 * c * phase_energy
    p = sqrt(p_squared)
    energy = 8 * c * phase_energy
    receiver_rate = s * phase_energy / 2
    q = delta[1]

    forcing_a = q**2 / (4 * (1 - h**2 / 2) ** 2)
    growth_a = forcing_a / (1 - 2 * h**2 / 3)
    remainder_a = (2 * growth_a + forcing_a) * h**3 / 3
    forcing_b = p_squared.hi / (4 * (1 - h**2 / 2) ** 2)
    growth_b = forcing_b / (3 * (1 - h**2 / 5))
    remainder_b = growth_b * h**3
    preparation = rx / (1 - 2 * h**2) + 2 * h * rt / (1 - 2 * h**2 / 3)
    error = preparation + eta
    nominal = (
        I(h * receiver_rate.lo - remainder_a, h * receiver_rate.hi + remainder_a),
        I(-remainder_b, remainder_b),
    )
    recorded = (
        I(
            h * receiver_rate.lo - remainder_a - error,
            h * receiver_rate.hi + remainder_a + error,
        ),
        I(-remainder_b - error, remainder_b + error),
    )
    # Rebuild the contrast from unrounded rational endpoint arithmetic rather
    # than subtracting the separately rounded marginal readout intervals.
    contrast_radius = remainder_a + remainder_b + 2 * error
    difference = I(
        h * receiver_rate.lo - contrast_radius,
        h * receiver_rate.hi + contrast_radius,
    )

    root_two, root_ten = sqrt(I(2)), sqrt(I(10))
    norm_bounds = (
        (10 * rx**2 + (root_two * q + root_ten * rt) ** 2).hi,
        ((root_two * p + root_ten * rx) ** 2 + 10 * rt**2).hi,
    )
    common_energy_error = 40 * (rx**2 + rt**2)
    energy_bounds = (
        (
            8 * c * one_minus_cosine[1]
            + 8 * rt * (c * sin(I(q)) + s * one_minus_cosine[1])
            + common_energy_error
        ).hi,
        (8 * c * one_minus_cosine[1] + 8 * p * rx + common_energy_error).hi,
    )
    gap = 5 - sqrt(I(5))
    radius_angle = alpha + root_two * r
    acute_margin = pi / 2 - radius_angle
    cosine = cos(radius_angle).lo if acute_margin.lo > 0 else None
    coercivity = (gap * cosine / 2).lo if cosine is not None and cosine > 0 else None
    barrier = (I(coercivity) * r**2).lo if coercivity is not None else None
    radius_margins = tuple(I(r**2 - bound) for bound in norm_bounds)
    energy_margins = (
        tuple(I(barrier - bound) for bound in energy_bounds)
        if barrier is not None
        else None
    )
    reasons = tuple(
        tuple(
            reason
            for admitted, reason in (
                (
                    acute_margin.lo > 0 and cosine is not None and cosine > 0,
                    "strict_acute_radius_not_certified",
                ),
                (
                    radius_margins[index].lo > 0,
                    "strict_initial_radius_not_certified",
                ),
                (
                    energy_margins is not None and energy_margins[index].lo > 0,
                    "strict_excess_storage_barrier_not_certified",
                ),
            )
            if not admitted
        )
        for index in (0, 1)
    )
    persistence = tuple(not failures for failures in reasons)
    separation = difference.lo > 0
    unavailable = tuple(
        f"{name}:{reason}"
        for name, failures in zip(("phase_split", "form_split"), reasons)
        for reason in failures
    ) + (() if separation else ("recorded_response_difference_not_strictly_positive",))
    return SinePairPersistentResponse(
        delta_bounds=delta,
        horizon_tau=h,
        form_error_bound=rx,
        phase_error_bound=rt,
        readout_error_bound=eta,
        radius=r,
        full_dimensional_preparation=rx > 0 and rt > 0,
        target_angle_bounds=alpha,
        target_cosine_bounds=c,
        target_sine_bounds=s,
        one_minus_cosine_endpoint_bounds=one_minus_cosine,
        nominal_internal_form_squared_bounds=p_squared,
        nominal_internal_form_amplitude_bounds=p,
        nominal_excess_storage_bounds=energy,
        nominal_receiver_rate_bounds=receiver_rate,
        nominal_mean_forcing_upper_bounds=(forcing_a, forcing_b),
        nominal_mean_growth_upper_bounds=(growth_a, growth_b),
        nominal_response_remainder_upper_bounds=(remainder_a, remainder_b),
        preparation_error_bound=preparation,
        nominal_readout_bounds=nominal,
        recorded_readout_bounds=recorded,
        recorded_difference_bounds=difference,
        spectral_gap_bounds=gap,
        radius_angle_bounds=radius_angle,
        acute_radius_margin_bounds=acute_margin,
        cosine_lower_bound=cosine,
        coercivity_lower_bound=coercivity,
        barrier_lower_bound=barrier,
        initial_norm_squared_upper_bounds=norm_bounds,
        initial_excess_storage_upper_bounds=energy_bounds,
        initial_radius_margin_bounds=radius_margins,
        storage_barrier_margin_bounds=energy_margins,
        persistence_certified_by_preparation=persistence,
        persistence_unavailable_reasons_by_preparation=reasons,
        response_separation_certified=separation,
        status="certified_persistent_response" if not unavailable else "unavailable",
        unavailable_reasons=unavailable,
    )
