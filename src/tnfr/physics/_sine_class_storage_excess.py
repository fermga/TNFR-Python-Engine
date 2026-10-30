"""Conditional storage excess for simultaneous distinct-port interventions.

This is static edge algebra and analytic error accounting, not an evolution or
an observation of a supplied trajectory. Full and Hessian-tangent histories
share their complete initial source and the two time-zero form events. Their
individual storage differences need not vanish initially; the four-history
difference cancels that common initial term and the matched event work.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I
from ._sine_class_collective_interface import (
    _GAMMA_UPPER,
    _MAX_INPUT_VARIATION,
    _bound_collective_interface,
)
from ._sine_class_neighbor_nonadditivity import (
    _derive_neighbor_onset,
    _neighbor_static_cone,
    _NeighborOnset,
)
from .relational_sine_class_mediation import _EDGES
from .relational_sine_class_memory import _normalized_laplacian


@dataclass(frozen=True)
class _StorageExcessOnset:
    """Exact edge sums before the common factor -gamma**4 * H**5 / 60.

    The three columns multiply a**3*b, a**2*b**2 and a*b**3. The target cosine
    channels remain outward enclosures of the supplied sine law. These onset
    coefficients alone do not certify a finite-time sign.
    """

    geometry: _NeighborOnset
    mixed_channel_factors: tuple[tuple[Q, Q, Q], ...]
    common_rational_factor: Q = Q(-1, 60)
    time_degree: int = 5
    amplitude_degree: int = 4


def _derive_storage_excess_onset(mediator_class) -> _StorageExcessOnset:
    """Rebuild quartic mixed edge factors from A*e and A**2*e columns."""
    geometry = _derive_neighbor_onset(mediator_class)
    channels = [[Q(0)] * 3 for _ in range(4)]
    for index, (i, j) in enumerate(_EDGES):
        u, v, r, s = (
            column[j] - column[i]
            for column in (
                geometry.donor_phase_velocity_column,
                geometry.receiver_phase_velocity_column,
                geometry.donor_phase_curvature_column,
                geometry.receiver_phase_curvature_column,
            )
        )
        row = channels[min(index // 9, 3)]
        row[0] += 3 * r * u**2 * v + s * u**3
        row[1] += 3 * r * u * v**2 + 3 * s * u**2 * v
        row[2] += r * v**3 + 3 * s * u * v**2
    return _StorageExcessOnset(geometry, tuple(map(tuple, channels)))


@dataclass(frozen=True)
class _DirectStorageExcess:
    """Heat-transported direct quartic term, without full-law corrections.

    Selfadjointness in the actual degree metric gives adjoint time v=2*t-s,
    which lies in [0,2H]. No time coefficients or responses are evaluated.
    """

    donor_spatial_columns: tuple[tuple[Q, ...], ...]
    receiver_spatial_columns: tuple[tuple[Q, ...], ...]
    donor_phase_cones: tuple[I, ...]
    receiver_phase_cones: tuple[I, ...]
    donor_adjoint_remainder_bound: Q
    receiver_adjoint_remainder_bound: Q
    donor_adjoint_edge_bounds: tuple[I, ...]
    receiver_adjoint_edge_bounds: tuple[I, ...]
    mixed_edge_bounds: tuple[I, ...]
    mixed_edge_sum_bounds: I
    direct_quartic_storage_bounds: tuple[Q, Q]


def _direct_storage_excess(onset, amplitude, horizon):
    """Consume locally derived geometry and already admitted positive budgets."""
    geometry = onset.geometry
    phase = _neighbor_static_cone(geometry, amplitude, horizon)
    matrix = _normalized_laplacian(_EDGES, geometry.degrees)

    def powers(first, second):
        columns = [first, second]
        for _ in range(3):
            columns.append(
                tuple(
                    sum((a * b for a, b in zip(row, columns[-1])), Q(0))
                    for row in matrix
                )
            )
        return tuple(columns)

    donor = powers(
        geometry.donor_phase_velocity_column,
        geometry.donor_phase_curvature_column,
    )
    receiver = powers(
        geometry.receiver_phase_velocity_column,
        geometry.receiver_phase_curvature_column,
    )

    def adjoint(columns):
        error = (2 * horizon) ** 3 * max(map(abs, columns[4])) / 3
        bounds = tuple(
            I(columns[1][j] - columns[1][i])
            - (columns[2][j] - columns[2][i]) * I(0, 2 * horizon)
            + (columns[3][j] - columns[3][i]) / 2 * I(0, (2 * horizon) ** 2)
            + I(-error, error)
            for i, j in _EDGES
        )
        return error, bounds

    donor_error, donor_adjoint = adjoint(donor)
    receiver_error, receiver_adjoint = adjoint(receiver)
    mixed = tuple(
        (I(Q(1, 6), 1) if index < 27 else I(1))
        * (
            rd * (3 * u**2 * v + 3 * u * v**2 + v**3)
            + rr * (u**3 + 3 * u**2 * v + 3 * u * v**2)
        )
        for index, (u, v, rd, rr) in enumerate(
            zip(
                phase.donor_edge_phase_cones,
                phase.receiver_edge_phase_cones,
                donor_adjoint,
                receiver_adjoint,
            )
        )
    )
    total = sum(mixed, I(0))
    # Multiply exact endpoints below the absolute dyadic grid; materializing
    # gamma**4*m**4*H**5 first could erase this storage scale.
    corners = tuple(
        -(gamma**4) * amplitude**4 * horizon**5 * value / 60
        for gamma in (geometry.gamma_bounds.lo, geometry.gamma_bounds.hi)
        for value in (total.lo, total.hi)
    )
    return _DirectStorageExcess(
        donor,
        receiver,
        phase.donor_edge_phase_cones,
        phase.receiver_edge_phase_cones,
        donor_error,
        receiver_error,
        donor_adjoint,
        receiver_adjoint,
        mixed,
        total,
        (min(corners), max(corners)),
    )


@dataclass(frozen=True)
class _StorageHistoryError:
    total_input_variation: Q
    higher_amplitude_storage_error_upper_bound: Q
    nominal_form_nonlinearity_upper_bound: Q
    nonlinear_initialization_error_upper_bound: Q
    storage_source_error_upper_bound: Q


@dataclass(frozen=True)
class _StorageExcessBound:
    """Conditional full-minus-tangent mixed storage and opposite loss integral.

    All four full histories and their four tangent partners have the same complete
    source and matched simultaneous events. The scalar loss identity uses the
    structural tau clock and the whole graph, not a local port-only energy. This
    report supplies no acquisition, work, identity or sensor verdict.
    """

    mediator_class: int
    amplitude: Q
    horizon: Q
    endpoint_radius: Q
    onset: _StorageExcessOnset
    direct_term: _DirectStorageExcess | None
    per_history_errors: tuple[_StorageHistoryError, ...]
    quartic_correction_upper_bound: Q
    higher_amplitude_error_upper_bound: Q
    source_error_upper_bound: Q
    nominal_excess_storage_bounds: tuple[Q, Q]
    full_excess_storage_bounds: tuple[Q, Q]
    integrated_excess_loss_bounds: tuple[Q, Q]
    negative_storage_margin: Q
    negative_storage_certified: bool
    exact_mixed_zero: bool
    history_order: tuple[str, ...] = ("neither", "donor_only", "receiver_only", "both")
    intervention_nodes: tuple[int, int] = (4, 22)
    clock: str = "tau=e*t"
    scope: tuple[str, ...] = (
        "same_complete_source_and_simultaneous_distinct_port_events_in_all_histories",
        "full_and_tangent_storage_use_their_own_states_and_the_same_target",
        "tangent_potential_uses_the_target_hessian_not_the_evolving_full_cosines",
        "common_initial_storage_gap_and_matched_event_work_cancel_only_in_the_mixed_balance",
        "full_storage_excess_is_the_negative_integrated_full_minus_tangent_loss_contrast",
        "ordinary_tangent_cross_energy_is_not_evidence_of_nonlinearity",
        "no_sensor_error_is_inherited_from_a_form_readout_budget",
        "no_source_acquisition_trajectory_or_time_coefficient_is_evaluated",
    )


def _bound_storage_excess(
    *, mediator_class, amplitude, horizon, endpoint_radius
) -> _StorageExcessBound:
    """Bound equal nonnegative impulses with 2m<=7/5000 and 0<=H<=1/8.

    The source premise is a complete residual max bound about the declared common
    origins, not independently chosen sources in different histories. Common
    means are retained; neither physical preparation nor trapping is inferred.
    """
    if type(mediator_class) is not int or mediator_class not in (1, 2):
        raise ValueError("mediator class must be ordinary integer one or two")
    m, h, eps = (
        _exact(value, label)
        for value, label in (
            (amplitude, "amplitude"),
            (horizon, "horizon"),
            (endpoint_radius, "endpoint_radius"),
        )
    )
    if not 0 <= 2 * m <= _MAX_INPUT_VARIATION:
        raise ValueError("equal amplitude must lie in [0,7/10000]")
    if not 0 <= h <= Q(1, 8) or eps < 0:
        raise ValueError("horizon must lie in [0,1/8] and source radius be nonnegative")
    onset = _derive_storage_excess_onset(mediator_class)
    g = _GAMMA_UPPER
    d, ell = 1 - 2 * g**2 * h**2, 1 - 2 * g * h
    sigma = eps / ell
    cauchy_bound = 232 * (4 * h**2 / d**2 + Q(64, 3) * g**2 * h**3 / d)
    errors = []
    for variation in (Q(0), m, m, 2 * m):
        tail = cauchy_bound * (2 * g * variation) ** 6 / (1 - (2 * g * variation) ** 2)
        nonlinear = 2 * g**3 * variation**2 * h / d**3
        initialization = _bound_collective_interface(
            total_input_variation=variation, horizon=h, endpoint_radius=eps
        ).nonlinear_initialization_error_upper_bound
        source = (
            232
            * h
            * (
                2 * sigma * nonlinear
                + (2 * variation / d + 2 * sigma + 2 * nonlinear + initialization)
                * initialization
            )
        )
        errors.append(
            _StorageHistoryError(variation, tail, nonlinear, initialization, source)
        )
    zero = m == 0 or h == 0
    direct = None if zero else _direct_storage_excess(onset, m, h)
    correction = (
        3
        * 232
        * m**4
        * (
            g**6 * h**7 * (Q(2, 21) / d**4 + Q(8, 105) / d**5 + Q(4, 63) / d**6)
            + 2 * g**8 * h**9 / (567 * d**6)
        )
    )
    tail = sum((row.higher_amplitude_storage_error_upper_bound for row in errors), Q(0))
    source = sum((row.storage_source_error_upper_bound for row in errors), Q(0))
    if zero:
        correction = tail = source = Q(0)
        nominal = full = (Q(0), Q(0))
    else:
        lower, upper = direct.direct_quartic_storage_bounds
        nominal = lower - correction - tail, upper + correction + tail
        full = nominal[0] - source, nominal[1] + source
    return _StorageExcessBound(
        mediator_class,
        m,
        h,
        eps,
        onset,
        direct,
        tuple(errors),
        correction,
        tail,
        source,
        nominal,
        full,
        (-full[1], -full[0]),
        -full[1],
        full[1] < 0,
        zero,
    )


def _bound_storage_observation_error(
    *, form_bound, phase_bound, form_error_bound, phase_error_bound
) -> Q:
    """Eight-storage allowance from explicitly supplied coordinate errors.

    X,Y bound each true state relative to its declared target/origins; dx,dy bound
    each coordinate measurement error, for both models and all four histories.
    This is a Lipschitz/Taylor storage bound, not a form-sensor calibration. The
    target phase gradient is zero; each potential Hessian has sum of absolute
    matrix entries at most 116 on this support.
    """
    x, y, dx, dy = (
        _exact(value, label)
        for value, label in (
            (form_bound, "form_bound"),
            (phase_bound, "phase_bound"),
            (form_error_bound, "form_error_bound"),
            (phase_error_bound, "phase_error_bound"),
        )
    )
    if min(x, y, dx, dy) < 0:
        raise ValueError("coordinate state and observation bounds must be nonnegative")
    return 8 * (116 * (x * dx + y * dy) + 58 * (dx**2 + dy**2))
