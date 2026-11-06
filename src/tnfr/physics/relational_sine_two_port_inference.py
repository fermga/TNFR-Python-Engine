"""Necessary local geometry bounds from a calibrated finite sine response.

The unknown is a coordinate of a realizable, generally noncritical affine
phase family. No equilibrium root, acquired-source report or forward verdict
is consumed. An outer compatibility interval does not assert existence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleGeometry, _cycle_row, _derive
from .relational_observations import _ordered
from .relational_sine_two_port_compatibility import (
    _EDGES,
    _NODES,
    _affine_geometry,
    _current_factorization,
)

__all__ = ("SineTwoPortInference", "infer_sine_two_port_geometry")

_BULK_DOMAIN = (Q(11, 8), Q(3, 2))
_RECEIVER_DOMAIN = (Q(2, 3), Q(1))
_GAMMA_UPPER = Q(1, 3069)
_FORCING_UPPER = Q(4)


def _primitive_bounds(raw, label):
    """Retain admitted exact endpoints before any outward rounding."""
    values = _ordered(raw, label, limit=3)
    if len(values) != 2:
        raise ValueError(f"{label} must contain two ordered endpoints")
    lower, upper = (exact_or_represented_real(value, label) for value in values)
    if lower > upper:
        raise ValueError(f"{label} lower endpoint exceeds upper endpoint")
    return lower, upper


def _inference_geometry():
    """Rebuild the affine chart and the three port-forcing cancellations."""
    geometry = _derive(_NODES, _EDGES)
    degrees, _, nodes, edges, offsets = _affine_geometry((2, 1), geometry)

    def radians(row):
        # A_turn=2-4*b/pi and C_turn=c/(2*pi): coefficients of (pi,b,c).
        return 2 * row[0] + 4 * row[1], -8 * row[1], row[2]

    nodal_rows = tuple(map(radians, nodes))
    edge_rows = tuple(map(radians, edges))
    laplacian = tuple(
        tuple(
            Q(degrees[i] if i == j else -int(tuple(sorted((i, j))) in _EDGES))
            for j in _NODES
        )
        for i in _NODES
    )
    normalized = tuple(
        tuple(value / degrees[i] for value in row) for i, row in enumerate(laplacian)
    )
    upper = tuple(
        tuple(2 * degrees[i] * int(i == j) - laplacian[i][j] for j in _NODES)
        for i in _NODES
    )
    if not exact_symmetric_semidefinite(upper):
        raise ArithmeticError("fixed normalized spectral upper bound failed")
    _, nodal_symbols, _, _ = _current_factorization(geometry)
    forcing = tuple(
        tuple(value / degrees[i] for value in row)
        for i, row in enumerate(nodal_symbols)
    )
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in _NODES)
    moments, row = [], (q,)
    for _ in range(3):
        moments.append(exact_matrix_product(row, forcing)[0])
        row = exact_matrix_product(row, normalized)
    if any(any(moment) for moment in moments):
        raise ArithmeticError("local readout does not cancel the first port moments")
    if (
        any(any(forcing[i]) for i in _NODES if i not in (0, 1, 9, 10))
        or any(sum(abs(v) for v in forcing[i]) > 1 for i in _NODES)
        or sum(degrees[i] for i in (0, 1, 9, 10)) > _FORCING_UPPER**2
        or sum(d * v**2 for d, v in zip(degrees, q)) != 4
        or sum(v**2 / d for d, v in zip(degrees, q)) != 1
        or sum(d * v for d, v in zip(degrees, q)) != 0
    ):
        raise ArithmeticError("fixed forcing or dipole norm bound failed")
    cycles = (tuple(range(9)), tuple(range(9, 18)), (0, 9, 10, 1))
    indices = {edge: i for i, edge in enumerate(_EDGES)}
    periods = []
    for cycle in cycles:
        cycle_row = _cycle_row(cycle, indices)
        coefficients = tuple(
            sum((s * edge[j] for s, edge in zip(cycle_row, edge_rows)), Q(0))
            for j in range(3)
        )
        if coefficients[1:] != (0, 0):
            raise ArithmeticError("affine family has a variable cycle period")
        periods.append(coefficients[0] / 2)
    if tuple(periods) != (2, 1, 0):
        raise ArithmeticError("affine family cycle periods failed")
    return dict(
        geometry=geometry,
        degrees=degrees,
        nodal_angle_affine_coefficients=nodal_rows,
        edge_angle_affine_coefficients=edge_rows,
        edge_integer_offsets=offsets,
        named_cycle_periods=tuple(periods),
        laplacian=laplacian,
        normalized_laplacian=normalized,
        normalized_upper_slack_matrix=upper,
        normalized_sine_symbol_coefficients=forcing,
        readout_forcing_moments=tuple(moments),
        dipole=q,
    )


def _decreasing_cosine_outer(prior, shift, observed, refinements):
    """Exclude only strict incompatible signs, retaining unresolved midpoints.

    On the caller's decreasing branch, the returned interval encloses every
    preimage of ``observed``. No root existence or successful sign resolution
    is required. Each boundary consumes at most ``refinements`` midpoint calls.
    """
    left, right = prior
    image = I(cos(I(right - shift)).lo, cos(I(left - shift)).hi)
    lower, upper = max(image.lo, observed.lo), min(image.hi, observed.hi)
    if lower > upper:
        return None, None, (0, 0), False
    band = I(lower, upper)
    endpoints, counts, unresolved = [], [], False
    for is_lower in (True, False):
        low, high = left, right
        count = 0
        for _ in range(refinements):
            if low == high:
                break
            midpoint = (low + high) / 2
            value = cos(I(midpoint - shift))
            count += 1
            if is_lower:
                if value.lo > band.hi:
                    low = midpoint
                elif value.hi <= band.hi:
                    high = midpoint
                else:
                    unresolved = True
                    break
            elif value.hi < band.lo:
                high = midpoint
            elif value.lo >= band.lo:
                low = midpoint
            else:
                unresolved = True
                break
        endpoints.append(low if is_lower else high)
        counts.append(count)
    if endpoints[0] > endpoints[1]:
        return None, band, tuple(counts), unresolved
    return I(*endpoints), band, tuple(counts), unresolved


@dataclass(frozen=True)
class SineTwoPortInference:
    """Outer geometry compatibility under an independently calibrated readout.

    Primitive angle bounds describe a full affine source family, not a family
    of equilibria. Form and phase radii use the full degree norm after removing
    their respective common means; phase coordinates and increments use radians.
    The sensor has one held positive gain shared by its before/after readings,
    each with absolute error at most ``readout_error_bound`` in sensor units.

    ``nominal_bulk_angle_outer_bounds`` is only a necessary compatibility
    enclosure. ``actual_long_arc_mean_outer_bounds`` additionally retains Y/8
    of source residual uncertainty, without clipping to the nominal prior.
    No incoming source history, formation, root or forward report is admitted.
    """

    bulk_angle_bounds: tuple[Q, Q]
    receiver_short_angle_bounds: tuple[Q, Q]
    form_radius: Q
    phase_radius: Q
    phase_increment: Q
    probe_duration: Q
    recorded_increment_bounds: tuple[Q, Q]
    readout_error_bound: Q
    readout_gain_bounds: tuple[Q, Q]
    refinements: int
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    nodal_angle_affine_coefficients: tuple[tuple[Q, Q, Q], ...]
    edge_angle_affine_coefficients: tuple[tuple[Q, Q, Q], ...]
    edge_integer_offsets: tuple[int, ...]
    named_cycle_periods: tuple[Q, ...]
    laplacian: tuple[tuple[Q, ...], ...]
    normalized_laplacian: tuple[tuple[Q, ...], ...]
    normalized_upper_slack_matrix: tuple[tuple[Q, ...], ...]
    normalized_sine_symbol_coefficients: tuple[tuple[Q, ...], ...]
    readout_forcing_moments: tuple[tuple[Q, ...], ...]
    dipole: tuple[Q, ...]
    nominal_edge_angle_bounds: tuple[I, ...]
    nominal_acute_margin: Q
    source_acute_margin: Q
    whole_window_acute_margin_candidate: Q
    whole_window_form_norm_candidate: Q
    whole_window_phase_norm_candidate: Q
    background_forcing_norm_upper_bound: Q
    finite_remainder_candidate: Q
    finite_remainder_upper_bound: Q | None
    gamma_bounds: I
    response_scale_bounds: I
    true_increment_bounds: I | None
    calibrated_increment_bounds: I | None
    compatible_cosine_bounds: I | None
    nominal_bulk_angle_outer_bounds: I | None
    actual_long_arc_mean_outer_bounds: I | None
    boundary_refinement_counts: tuple[int, int]
    refinement_precision_limited: bool
    source_admitted: bool
    finite_response_certified: bool
    whole_window_acute_certified: bool
    inverse_enclosure_available: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t"
    capacity: tuple[Q, ...] = (Q(1),) * 18
    scope: tuple[str, ...] = (
        "realizable_noncritical_affine_phase_family_on_fixed_two_port_C9_support",
        "radian_affine_coefficients_multiply_pi_donor_bulk_receiver_short",
        "source_acute_admission_precedes_the_phase_dipole",
        "complete_unforced_two_row_sine_flow_after_supplied_phase_event",
        "finite_remainder_retains_nonzero_four_port_background_forcing",
        "single_held_positive_sensor_gain_and_two_scalar_reading_errors",
        "supplied_calibration_and_observation_are_not_authenticated_or_inferred",
        "bounded_candidate_is_necessary_outer_compatibility_not_existence",
        "receiver_short_angle_remains_a_declared_nuisance_coordinate",
        "actual_eight_long_arc_mean_expands_nominal_enclosure_by_phase_radius_over_eight",
        "no_target_root_acquisition_warmup_recovery_or_physical_identification",
        "whole_window_acuity_is_optional_and_not_an_inverse_admission_gate",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-two-port-inference.v1", "report": _project(self)}


def infer_sine_two_port_geometry(
    *,
    bulk_angle_bounds,
    receiver_short_angle_bounds,
    form_radius,
    phase_radius,
    phase_increment,
    probe_duration,
    recorded_increment_bounds,
    readout_error_bound,
    readout_gain_bounds,
    refinements,
) -> SineTwoPortInference:
    """Enclose compatible donor bulk angles from primitive finite observations.

    The independent nominal rectangle lies within b=[11/8,3/2], c=[2/3,1]
    radians. Radii and per-reading errors are nonnegative, increment and fast
    duration lie in (0,1], gain endpoints are positive, and the ordinary integer
    refinement budget lies in 1..64. Exact rationals remain exact primitives;
    other finite reals use shared represented-real admission before arithmetic.

    ``unavailable`` means source or numerical division premises are unresolved.
    ``incompatible`` excludes the combined supplied premises by a necessary
    condition. ``bounded_candidate`` retains an outer interval, not an exact
    inverse or evidence that any compatible full trajectory exists.
    """
    b = _primitive_bounds(bulk_angle_bounds, "bulk_angle_bounds")
    receiver = _primitive_bounds(
        receiver_short_angle_bounds, "receiver_short_angle_bounds"
    )
    measured = _primitive_bounds(recorded_increment_bounds, "recorded_increment_bounds")
    gain = _primitive_bounds(readout_gain_bounds, "readout_gain_bounds")
    values = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_radius, "form_radius"),
            (phase_radius, "phase_radius"),
            (phase_increment, "phase_increment"),
            (probe_duration, "probe_duration"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    x, y, amplitude, horizon, noise = values
    if not (_BULK_DOMAIN[0] <= b[0] <= b[1] <= _BULK_DOMAIN[1]):
        raise ValueError("bulk_angle_bounds must lie within [11/8,3/2] radians")
    if not (_RECEIVER_DOMAIN[0] <= receiver[0] <= receiver[1] <= _RECEIVER_DOMAIN[1]):
        raise ValueError("receiver_short_angle_bounds must lie within [2/3,1] radians")
    if min(x, y, noise) < 0:
        raise ValueError(
            "form_radius, phase_radius and readout_error_bound must be nonnegative"
        )
    if not 0 < amplitude <= 1 or not 0 < horizon <= 1:
        raise ValueError("phase_increment and probe_duration must lie in (0,1]")
    if gain[0] <= 0:
        raise ValueError("readout_gain_bounds must be strictly positive")
    if type(refinements) is not int or not 1 <= refinements <= 64:
        raise ValueError("refinements must be an ordinary integer in 1..64")
    geometry = _inference_geometry()
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    _sine_model_coefficients(model, positive_loss=True)
    pi = pi_interval()
    coordinates = (pi, I(*b), I(*receiver))
    edge_bounds = tuple(
        sum((value * coordinate for value, coordinate in zip(row, coordinates)), I(0))
        for row in geometry["edge_angle_affine_coefficients"]
    )
    nominal_margin = min((pi / 2).lo - edge.abs_max for edge in edge_bounds)
    source_margin = nominal_margin - y
    source_ok = source_margin > 0
    coupling = 2 * _GAMMA_UPPER * horizon
    qmax = (x + _GAMMA_UPPER * horizon * (_FORCING_UPPER + 4 * amplitude + 2 * y)) / (
        1 - coupling**2
    )
    pmax = y + 2 * amplitude + coupling * qmax
    error = (
        2 * horizon * x
        + _GAMMA_UPPER * _FORCING_UPPER * horizon**4 / 3
        + 4 * _GAMMA_UPPER * amplitude * horizon**2
        + 2 * _GAMMA_UPPER * horizon * y
        + 4 * _GAMMA_UPPER**2 * horizon**2 * qmax
    )
    gamma = 1 / (1023 * pi)
    scale = 2 * gamma * horizon * sin(I(Q(3, 2) * amplitude))
    cosine = I(cos(I(b[1] - amplitude / 2)).lo, cos(I(b[0] - amplitude / 2)).hi)
    true = -scale * cosine + I(-error, error) if source_ok else None
    reasons = []
    if not source_ok:
        reasons.append(
            "source_phase_ball_is_not_certified_inside_the_declared_acute_chart"
        )
    if scale.lo <= 0:
        reasons.append("positive_response_scale_is_unresolved_at_interval_precision")
    if I(*gain).lo <= 0:
        reasons.append("positive_readout_gain_is_unresolved_at_interval_precision")
    calibrated = band = outer = actual = None
    counts, limited = (0, 0), False
    status = "unavailable"
    if not reasons:
        calibrated = (I(*measured) + I(-2 * noise, 2 * noise)) / I(*gain)
        needed = (-calibrated + I(-error, error)) / scale
        outer, band, counts, limited = _decreasing_cosine_outer(
            b, amplitude / 2, needed, refinements
        )
        status = "incompatible" if outer is None else "bounded_candidate"
        if outer is not None:
            actual = outer + I(-y / 8, y / 8)
    return SineTwoPortInference(
        bulk_angle_bounds=b,
        receiver_short_angle_bounds=receiver,
        form_radius=x,
        phase_radius=y,
        phase_increment=amplitude,
        probe_duration=horizon,
        recorded_increment_bounds=measured,
        readout_error_bound=noise,
        readout_gain_bounds=gain,
        refinements=refinements,
        reference_model=model,
        **geometry,
        nominal_edge_angle_bounds=edge_bounds,
        nominal_acute_margin=nominal_margin,
        source_acute_margin=source_margin,
        whole_window_acute_margin_candidate=nominal_margin - pmax,
        whole_window_form_norm_candidate=qmax,
        whole_window_phase_norm_candidate=pmax,
        background_forcing_norm_upper_bound=_FORCING_UPPER,
        finite_remainder_candidate=error,
        finite_remainder_upper_bound=error if source_ok else None,
        gamma_bounds=gamma,
        response_scale_bounds=scale,
        true_increment_bounds=true,
        calibrated_increment_bounds=calibrated,
        compatible_cosine_bounds=band,
        nominal_bulk_angle_outer_bounds=outer,
        actual_long_arc_mean_outer_bounds=actual,
        boundary_refinement_counts=counts,
        refinement_precision_limited=limited,
        source_admitted=source_ok,
        finite_response_certified=source_ok,
        whole_window_acute_certified=source_ok and nominal_margin - pmax > 0,
        inverse_enclosure_available=outer is not None,
        status=status,
        unavailable_reasons=tuple(reasons),
    )
