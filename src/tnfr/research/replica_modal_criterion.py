"""Reference-dependent modal criteria for the conservative doubled C5.

The harmonic reference retains the same initial collective state and its
constant mode. Actual static rates come only from the existing complete fine
field. Neither the reference nor its residual is an autonomous reduction of
the nonlinear twenty-coordinate law. No trajectory or retained response is
evaluated here; z=X+iTheta is mathematical packaging in fixed normalized
units, not a physical complex amplitude.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._exact_linear_algebra import exact_square_matrix_product
from ..mathematics._rational_interval import (
    INTERVAL_BITS,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ..physics.relational_observations import _ordered
from .replica_collective_preparation import (
    COLLECTIVE_SCALE,
    ERROR_RADIUS,
    HORIZON,
    INTERNAL_SCALE,
    MARGIN,
    PAIRS,
    PROFILE,
    ReplicaPreparationSource,
    _box,
    _energy_upper,
    _exact,
    _source_center,
    complete_field,
    declared_source,
)

CLOCK = "original structural t; both full rows retain 1/pi"
REFERENCE = "aligned harmonic reference in the declared continuous phase lifts"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(values, size, label):
    rows = _ordered(values, label, limit=size + 1)
    _require(len(rows) == size, f"{label} dimension differs")
    return rows


def _endpoints(value, label):
    if isinstance(value, I):
        lower, upper = _exact(value.lo, label), _exact(value.hi, label)
    else:
        lower = upper = _exact(value, label)
    _require(lower <= upper, f"{label} has reversed bounds")
    return lower, upper


def _apply(matrix, values):
    return tuple(
        sum((coefficient * value for coefficient, value in zip(row, values)), I(0))
        for row in matrix
    )


def _laplacian(values):
    return tuple(
        2 * values[i] - values[(i - 1) % 5] - values[(i + 1) % 5] for i in range(5)
    )


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), I(0))


@dataclass(frozen=True)
class C5ModalGeometry:
    laplacian: tuple[tuple[Q, ...], ...]
    constant_projector: tuple[tuple[Q, ...], ...]
    centered_projector: tuple[tuple[Q, ...], ...]
    projector_rational_parts: tuple[tuple[tuple[tuple[Q, ...], ...], ...], ...]
    projectors: tuple[tuple[tuple[I, ...], ...], ...]
    eigenvalues: tuple[I, ...]
    multiplicities: tuple[int, ...]
    polynomial_certified: bool
    coefficient_basis: str = "A_plus_B_sqrt5"


def c5_modes():
    """Build the two real rank-two eigenspaces without an eigensolver.

    L^2-5L+5Pc=0 on the complete space, with L*P0=0. The rational
    parts retain the exact projectors even though their evaluated entries
    are outward enclosures. No privileged basis within either plane is used.
    """
    identity = tuple(tuple(Q(i == j) for j in range(5)) for i in range(5))
    zero = tuple((Q(0),) * 5 for _ in range(5))
    p0 = tuple((Q(1, 5),) * 5 for _ in range(5))
    pc = tuple(tuple(identity[i][j] - p0[i][j] for j in range(5)) for i in range(5))
    laplacian = tuple(
        tuple(Q(2 if i == j else -1 if (i - j) % 5 in (1, 4) else 0) for j in range(5))
        for i in range(5)
    )
    squared = exact_square_matrix_product(laplacian, laplacian)
    polynomial = all(
        squared[i][j] - 5 * laplacian[i][j] + 5 * pc[i][j] == 0
        for i in range(5)
        for j in range(5)
    )
    polynomial = (
        polynomial
        and exact_square_matrix_product(laplacian, p0) == zero
        and exact_square_matrix_product(p0, p0) == p0
    )
    _require(polynomial, "fixed C5 polynomial identities unavailable")
    a = tuple(tuple(value / 2 for value in row) for row in pc)
    b = tuple(
        tuple(pc[i][j] / 2 - laplacian[i][j] / 5 for j in range(5)) for i in range(5)
    )
    minus_b = tuple(tuple(-value for value in row) for row in b)
    parts = ((p0, zero), (a, b), (a, minus_b))
    root = sqrt(I(5))
    projectors = tuple(
        tuple(
            tuple(left[i][j] + right[i][j] * root for j in range(5)) for i in range(5)
        )
        for left, right in parts
    )
    return C5ModalGeometry(
        laplacian,
        p0,
        pc,
        parts,
        projectors,
        (I(0), (5 - root) / 2, (5 + root) / 2),
        (1, 2, 2),
        polynomial,
    )


@dataclass(frozen=True)
class HarmonicReference:
    forms: tuple[I, ...]
    phases: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    time: I
    geometry: C5ModalGeometry
    constant_form: I
    constant_phase: I
    clock: str = CLOCK
    reference: str = REFERENCE
    continuous_lifts_retained: bool = True
    actual_nonlinear_trajectory: bool = False
    acquisition_claimed: bool = False
    physical_complex_amplitude: bool = False


def harmonic_reference(forms, phases, *, time):
    """Enclose the exact unforced reference with the same supplied initial state.

    All ten initial coordinates and the nonnegative time interval are admitted
    before geometry or trigonometry. Continuous lifts are retained, including
    their constant mode. An individual full-turn change alters this reference.
    """
    raw_forms, raw_phases = _rows(forms, 5, "collective forms"), _rows(
        phases, 5, "collective phases"
    )
    endpoints = tuple(
        _endpoints(value, "initial coordinate") for value in raw_forms + raw_phases
    )
    time_endpoints = _endpoints(time, "reference time")
    _require(time_endpoints[0] >= 0, "reference time must be nonnegative")
    initial = tuple(I(*pair) for pair in endpoints)
    clock = I(*time_endpoints)
    x0, theta0 = initial[:5], initial[5:]
    geometry = c5_modes()
    pi = pi_interval()
    constant_form, constant_phase = sum(x0, I(0)) / 5, sum(theta0, I(0)) / 5
    if clock.lo == clock.hi == 0:
        x, theta = x0, theta0
    else:
        x, theta = (constant_form,) * 5, (constant_phase,) * 5
        for eigenvalue, projector in zip(
            geometry.eigenvalues[1:], geometry.projectors[1:]
        ):
            mode_x, mode_theta = _apply(projector, x0), _apply(projector, theta0)
            angle = eigenvalue * clock / (2 * pi)
            cosine, sine = cos(angle), sin(angle)
            x = tuple(
                value + cosine * a - sine * b
                for value, a, b in zip(x, mode_x, mode_theta)
            )
            theta = tuple(
                value + sine * a + cosine * b
                for value, a, b in zip(theta, mode_x, mode_theta)
            )
    return HarmonicReference(
        x,
        theta,
        tuple(-value / (2 * pi) for value in _laplacian(theta)),
        tuple(value / (2 * pi) for value in _laplacian(x)),
        clock,
        geometry,
        constant_form,
        constant_phase,
    )


@dataclass(frozen=True)
class ModalStateObservation:
    state: tuple[I, ...]
    forms: tuple[I, ...]
    phases: tuple[I, ...]
    internal_forms: tuple[I, ...]
    internal_phases: tuple[I, ...]
    fine_rates: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    oriented_gaps: tuple[I, ...]
    resultant_products: tuple[I, ...]
    eta: tuple[I, ...]
    harmonic_storage: I
    harmonic_storage_rate: I
    harmonic_storage_rate_from_defect: I
    clock: str = CLOCK
    reference: str = REFERENCE
    continuous_lifts_retained: bool = True
    source_membership_inferred: bool = False
    acquisition_claimed: bool = False
    physical_complex_amplitude: bool = False
    harmonic_storage_is_complete_conserved_storage: bool = False


def observe_modal_state(state):
    """Project the unchanged actual field at one completely admitted fine state.

    eta is the difference from the fixed harmonic row, not an independent
    forcing law or a predictor obtained by replacing hidden coordinates.
    Qharm and its derivative are lift/reference-dependent diagnostics.
    """
    values = _box(state)
    rates = complete_field()(values)
    x = tuple((values[a] + values[b]) / 2 for a, b in PAIRS)
    theta = tuple((values[10 + a] + values[10 + b]) / 2 for a, b in PAIRS)
    internal_x = tuple((values[a] - values[b]) / 2 for a, b in PAIRS)
    internal_theta = tuple((values[10 + a] - values[10 + b]) / 2 for a, b in PAIRS)
    dx = tuple((rates[a] + rates[b]) / 2 for a, b in PAIRS)
    dtheta = tuple((rates[10 + a] + rates[10 + b]) / 2 for a, b in PAIRS)
    gaps = tuple(theta[(i + 1) % 5] - theta[i] for i in range(5))
    resultants = tuple(cos(value) for value in internal_theta)
    products = tuple(resultants[i] * resultants[(i + 1) % 5] for i in range(5))
    pi = pi_interval()
    lx, ltheta = _laplacian(x), _laplacian(theta)
    eta = tuple(value + linear / (2 * pi) for value, linear in zip(dx, ltheta))
    storage = 2 * (_dot(x, lx) + _dot(theta, ltheta))
    storage_rate = 4 * (_dot(dx, lx) + _dot(dtheta, ltheta))
    defect_rate = 4 * _dot(lx, eta)
    return ModalStateObservation(
        values,
        x,
        theta,
        internal_x,
        internal_theta,
        rates,
        dx,
        dtheta,
        gaps,
        products,
        eta,
        storage,
        storage_rate,
        defect_rate,
    )


@dataclass(frozen=True)
class ReplicaModalSourceCertificate:
    source: ReplicaPreparationSource
    geometry: C5ModalGeometry
    reference_gap_upper_bound: Q
    transport_threshold_enclosure: I
    necessary_deviation_lower_bound: Q
    necessary_absolute_transport_lower_bound: Q
    target_remainder_width_bound: I
    pointwise_lower_bound_enclosure: I
    pointwise_defect_lower_bound: Q
    source_harmonic_storage_upper_bound: Q
    target_harmonic_storage_lower_bound: Q
    necessary_auxiliary_work_lower_bound: Q
    harmonic_target_excluded: bool
    necessary_transport_certified: bool
    pointwise_target_defect_certified: bool
    auxiliary_work_certified: bool
    certified: bool
    physical_chart_horizon: Q = HORIZON
    clock: str = CLOCK
    reference: str = REFERENCE
    continuous_lifts_retained: bool = True
    physical_complex_amplitude: bool = False
    actual_acquisition_claimed: bool = False
    actual_transport_budget_evaluated: bool = False
    instantaneous_defect_is_accumulated_error: bool = False
    harmonic_storage_is_complete_conserved_storage: bool = False
    arbitrary_winding_change_excluded: bool = False
    new_integrations: int = 0
    empirical_validation: bool = False


def source_certificate():
    """Rebuild the fixed source and source-specific necessary target budgets.

    The all-time harmonic gap envelope is not an exclusion of every winding
    change. It excludes the specified all-acute-margin target. Its comparison
    to physical pair phases uses the actual source chart through horizon six.
    No absolute residual-history integral or selected-state future is read.
    """
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
    slack = _exact(
        source.numerical_enclosure_allowance, "numerical enclosure allowance"
    )
    cover = _exact(source.proof_coordinate_error, "source proof cover")
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
        "initial source enclosure differs",
    )
    energy = _energy_upper(nominal, cover)
    _require(
        _exact(source.storage_upper_bound, "source storage bound") == energy
        and energy < 48,
        "source storage premise unavailable",
    )
    radius_squared = 64 * ((2 * INTERNAL_SCALE + cover) ** 2 + (Q(1, 50) + cover) ** 2)
    radius_lower = (INTERNAL_SCALE - cover) / 8
    pi = pi_interval()
    # The same elementary majorant proves exp(6/pi)<8. These are
    # independently rebuilt source premises, not a cached identity verdict.
    chart = (
        pi.lo > 3
        and Q(65, 24) + Q(1, 100) < Q(11, 4)
        and Q(11, 4) ** 2 < 8
        and radius_squared < 1
        and radius_lower > Q(1, 160)
    )
    _require(chart, "source pair chart/nontip premise unavailable")
    geometry = c5_modes()
    gap_upper = 2 * COLLECTIVE_SCALE + 5 * cover
    target_gap = 3 * pi / 2 + MARGIN
    threshold = (target_gap - gap_upper) / sqrt(I(2))
    lower_q = MARGIN - sin(I(MARGIN))
    upper_q = pi / 2 - MARGIN - cos(I(MARGIN)) / 4
    width = upper_q - lower_q
    pointwise = sqrt(I(2)) * (1 - width / (2 * pi))
    collective = tuple(COLLECTIVE_SCALE * value for value in PROFILE)
    source_q = sum(
        (
            2
            * (
                (abs(collective[(i + 1) % 5] - collective[i]) + 2 * cover) ** 2
                + (2 * cover) ** 2
            )
            for i in range(5)
        ),
        Q(0),
    )
    target_q = (Q(5, 2) * target_gap**2).lo
    auxiliary_work = target_q - source_q
    harmonic_excluded = gap_upper < target_gap.lo
    transport = harmonic_excluded and threshold.lo > Q(1, 2)
    instantaneous = (
        width.hi < pi.lo / 2
        and pointwise.lo > (3 * sqrt(I(2)) / 4).hi
        and pointwise.lo > 1
    )
    auxiliary = source_q < 48 and target_q > Q(405, 8) and auxiliary_work > 0
    return ReplicaModalSourceCertificate(
        source,
        geometry,
        gap_upper,
        threshold,
        threshold.lo,
        threshold.lo,
        width,
        pointwise,
        pointwise.lo,
        source_q,
        target_q,
        auxiliary_work,
        harmonic_excluded,
        transport,
        instantaneous,
        auxiliary,
        geometry.polynomial_certified and transport and instantaneous and auxiliary,
    )
