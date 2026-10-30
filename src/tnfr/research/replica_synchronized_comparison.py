"""Source-based error allowances for the synchronized full-sine restriction.

The comparator has the same initial collective coordinates and zero internal
coordinates. It is a restriction of the unchanged fine law, with different
internal storage. A positive comparison system bounds errors; it is neither
the actual trajectory nor the comparator trajectory. No retained response is
read, and a large allowance does not establish a large actual effect.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._comparison_flow import (
    NonnegativeExponentialUpper,
    nonnegative_exponential_upper,
)
from ..mathematics._rational_interval import (
    INTERVAL_BITS,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from .replica_collective_preparation import (
    ERROR_RADIUS,
    HORIZON,
    INTERNAL_FORMS,
    INTERNAL_PHASES,
    INTERNAL_SCALE,
    MARGIN,
    PAIRS,
    ReplicaPreparationSource,
    _box,
    _energy_upper,
    _exact,
    _source_center,
    complete_field,
    declared_source,
)
from .replica_modal_criterion import C5ModalGeometry, _endpoints, _rows, c5_modes

CLOCK = "original structural t; both full rows retain 1/pi"
TIMES = (Q(2), Q(6))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _means(values):
    return tuple((values[a] + values[b]) / 2 for a, b in PAIRS)


@dataclass(frozen=True)
class InternalModulationObservation:
    state: tuple[I, ...]
    synchronized_state: tuple[I, ...]
    forms: tuple[I, ...]
    phases: tuple[I, ...]
    internal_forms: tuple[I, ...]
    internal_phases: tuple[I, ...]
    fine_rates: tuple[I, ...]
    synchronized_fine_rates: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    synchronized_form_rates: tuple[I, ...]
    synchronized_phase_rates: tuple[I, ...]
    oriented_gaps: tuple[I, ...]
    resultant_products: tuple[I, ...]
    edge_modulation: tuple[I, ...]
    form_discrepancy: tuple[I, ...]
    phase_discrepancy: tuple[I, ...]
    discrepancy_from_edges: tuple[I, ...]
    clock: str = CLOCK
    continuous_lifts_retained: bool = True
    same_state_comparison: bool = True
    evolved_prediction_error_evaluated: bool = False
    source_membership_inferred: bool = False
    equal_internal_storage_claimed: bool = False
    comparator_internal_activity_claimed: bool = False
    actual_acquisition_claimed: bool = False


def observe_internal_modulation(state):
    """Compare actual and synchronized rates at identical supplied means.

    All twenty coordinates are admitted before field compilation or sine work.
    Both rate vectors come from the existing complete field. The signed edge
    expression independently exposes what is omitted by synchronization.
    Interval projection can lose correlations but only enlarges enclosures.
    """
    values = _box(state)
    x, theta = _means(values[:10]), _means(values[10:])
    u = tuple((values[a] - values[b]) / 2 for a, b in PAIRS)
    delta = tuple((values[10 + a] - values[10 + b]) / 2 for a, b in PAIRS)
    synchronized = tuple(value for row in (x, theta) for value in row for _ in range(2))
    flow = complete_field()
    rates, synchronized_rates = flow(values), flow(synchronized)
    dx, dt = _means(rates[:10]), _means(rates[10:])
    sx, st = _means(synchronized_rates[:10]), _means(synchronized_rates[10:])
    gaps = tuple(theta[(i + 1) % 5] - theta[i] for i in range(5))
    resultants = tuple(cos(value) for value in delta)
    products = tuple(resultants[i] * resultants[(i + 1) % 5] for i in range(5))
    edge = tuple((1 - product) * sin(gap) for product, gap in zip(products, gaps))
    pi = pi_interval()
    divergence = tuple((edge[(i - 1) % 5] - edge[i]) / (2 * pi) for i in range(5))
    return InternalModulationObservation(
        values,
        synchronized,
        x,
        theta,
        u,
        delta,
        rates,
        synchronized_rates,
        dx,
        dt,
        sx,
        st,
        gaps,
        products,
        edge,
        tuple(a - b for a, b in zip(dx, sx)),
        tuple(a - b for a, b in zip(dt, st)),
        divergence,
    )


@dataclass(frozen=True)
class TargetErosionAssessment:
    error_bounds: tuple[Q, Q]
    margin_threshold: Q
    joint_threshold: I
    within_margin_certified: bool
    exceeds_margin_certified: bool
    margin_contact_unresolved: bool
    jointly_nonempty_certified: bool
    jointly_empty_certified: bool
    joint_contact_unresolved: bool
    comparator_state_evaluated: bool = False
    arrival_claimed: bool = False
    actual_error_lower_bound_claimed: bool = False
    dilation_uses_integer_assignments: bool = True


def target_erosion(error):
    """Assess only the certifying power of a uniform five-gap allowance.

    The eroded target is nonempty exactly when E<=pi/10-mu: its five
    gaps must still sum to2pi. Equality is feasible, and unresolved interval
    contact remains unavailable. A dilated target would require an integer
    wrap assignment summing to one, not a principal-winding test when E>mu.
    No comparator state or target arrival is evaluated by this helper.
    """
    lower, upper = _endpoints(error, "gap allowance")
    _require(lower >= 0, "gap allowance must be nonnegative")
    threshold = pi_interval() / 10 - MARGIN
    within, exceeds = upper <= MARGIN, lower > MARGIN
    nonempty, empty = upper <= threshold.lo, lower > threshold.hi
    return TargetErosionAssessment(
        (lower, upper),
        MARGIN,
        threshold,
        within,
        exceeds,
        not within and not exceeds,
        nonempty,
        empty,
        not nonempty and not empty,
    )


@dataclass(frozen=True)
class SynchronizedComparisonBound:
    time: Q
    kernel: NonnegativeExponentialUpper
    envelope_upper: tuple[Q, ...]
    collective_form_error_upper: Q
    collective_phase_error_upper: Q
    collective_gap_error_upper: Q
    target_erosion: TargetErosionAssessment
    all_prior_times_covered: bool = True
    actual_error_evaluated: bool = False
    comparator_trajectory_evaluated: bool = False


@dataclass(frozen=True)
class SynchronizedSourceCertificate:
    source: ReplicaPreparationSource
    geometry: C5ModalGeometry
    alpha_upper: Q
    kappa_upper: Q
    forcing_coefficient_enclosure: I
    forcing_coefficient_upper: Q
    internal_phase_initial_upper: Q
    internal_form_initial_upper: Q
    internal_scale_factors: tuple[Q, ...]
    edge_quadratic_factors: tuple[Q, ...]
    comparison_matrix: tuple[tuple[Q, ...], ...]
    initial_envelope: tuple[Q, ...]
    bounds: tuple[SynchronizedComparisonBound, ...]
    certified: bool
    clock: str = CLOCK
    physical_chart_horizon: Q = HORIZON
    same_memberwise_collective_initial_state: bool = True
    continuous_lifts_retained: bool = True
    full_sine_retained: bool = True
    comparator_is_actual_invariant_restriction: bool = True
    equal_internal_storage_claimed: bool = False
    comparator_internal_activity_claimed: bool = False
    actual_acquisition_claimed: bool = False
    actual_error_evaluated: bool = False
    same_state_discrepancy_is_prediction_error: bool = False
    global_contraction_assumed: bool = False
    retained_response_read: bool = False
    new_integrations: int = 0
    empirical_validation: bool = False


def _admitted_source():
    """Rebuild only the fixed source primitives consumed by this comparison."""
    source = declared_source()
    nominal = tuple(
        _exact(value, "nominal coordinate")
        for value in _rows(source.nominal, 20, "nominal source")
    )
    expected = _source_center(redistributed=True)
    _require(
        nominal == expected and source.clock == CLOCK,
        "source preparation or clock differs",
    )
    epsilon = _exact(source.intended_coordinate_error, "source error")
    slack = _exact(source.numerical_enclosure_allowance, "numerical allowance")
    cover = _exact(source.proof_coordinate_error, "proof cover")
    _require(
        epsilon == ERROR_RADIUS
        and slack == Q(1, 2**INTERVAL_BITS)
        and cover == epsilon + slack,
        "source uncertainty contract differs",
    )
    initial = _box(source.initial_box)
    _require(
        all(
            bound.lo <= value - epsilon <= value + epsilon <= bound.hi
            and value - cover <= bound.lo <= bound.hi <= value + cover
            for value, bound in zip(nominal, initial)
        ),
        "source enclosure differs",
    )
    energy = _energy_upper(nominal, cover)
    _require(
        _exact(source.storage_upper_bound, "source storage") == energy and energy < 48,
        "source storage premise unavailable",
    )
    d, v = Q(1, 50) + cover, 2 * INTERNAL_SCALE + cover
    pi = pi_interval()
    _require(
        pi.lo > 3
        and Q(65, 24) + Q(1, 100) < Q(11, 4)
        and Q(11, 4) ** 2 < 8
        and 64 * (v * v + d * d) < 1
        and (INTERNAL_SCALE - cover) / 8 > Q(1, 160),
        "actual source chart/nontip premise unavailable",
    )
    return source, cover, d, v


def source_certificate():
    """Bound propagated errors at two fixed times without either trajectory.

    The componentwise positive envelope evolves (P,Q,D²,DU,U²), starting
    with zero collective mismatch. Pair-specific scale factors retain the
    heterogeneous source's internal sizes. Global sine secants may be signed;
    the comparison uses their operator-norm bound, not convexity or damping.
    """
    source, cover, d, v = _admitted_source()
    geometry = c5_modes()
    pi = pi_interval()
    eigenvalue = (5 + sqrt(I(5))) / 2
    eigen_bounds = _endpoints(geometry.eigenvalues[2], "C5 eigenvalue")
    _require(
        eigen_bounds == (eigenvalue.lo, eigenvalue.hi),
        "C5 eigenvalue association differs",
    )
    alpha, kappa = (1 / pi).hi, (eigenvalue / (2 * pi)).hi
    scales = tuple(
        max((INTERNAL_SCALE * u + cover) / v, (delta + cover) / d)
        for u, delta in zip(INTERNAL_FORMS, INTERNAL_PHASES)
    )
    _require(
        all(0 < value <= 1 for value in scales), "source profile scaling unavailable"
    )
    edge_factors = tuple(
        (scales[i] ** 2 + scales[(i + 1) % 5] ** 2) / 2 for i in range(5)
    )
    forcing = sqrt(eigenvalue * sum((value**2 for value in edge_factors), Q(0))) / (
        2 * pi
    )
    coefficient = forcing.hi
    zero = Q(0)
    matrix = (
        (zero, kappa, coefficient, zero, zero),
        (kappa, zero, zero, zero, zero),
        (zero, zero, zero, 2 * alpha, zero),
        (zero, zero, alpha, zero, alpha),
        (zero, zero, zero, 2 * alpha, zero),
    )
    initial = (zero, zero, d * d, d * v, v * v)
    root2_upper = sqrt(I(2)).hi
    bounds = []
    for time in TIMES:
        kernel = nonnegative_exponential_upper(matrix, time)
        envelope = tuple(
            sum((entry * value for entry, value in zip(row, initial)), Q(0))
            for row in kernel.upper
        )
        gap = root2_upper * envelope[1]
        bounds.append(
            SynchronizedComparisonBound(
                time,
                kernel,
                envelope,
                envelope[0],
                envelope[1],
                gap,
                target_erosion(gap),
            )
        )
    return SynchronizedSourceCertificate(
        source,
        geometry,
        alpha,
        kappa,
        forcing,
        coefficient,
        d,
        v,
        scales,
        edge_factors,
        matrix,
        initial,
        tuple(bounds),
        True,
    )
