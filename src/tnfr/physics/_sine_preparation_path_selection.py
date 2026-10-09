"""Uniform source-to-sector tests along one energy-isometric C9 path.

Proof: theory/nodal/SINE_PREPARATION_PATH_SELECTION.md. The parameter is a
preparation coordinate, not time. For every common latent source (xi,zeta),
the family (Q_s*(m*a+xi),zeta) preserves its initial means and storage.
Complete-law transit gives correlated phase corridors and total storage
bounds. Acute-sector capture and eventual exit from winding two are distinct
conclusions; an exit certificate supplies neither its time nor the final mode.
No trajectory, parameter sweep, equilibrium fit or old verdict is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._cycle_algebra import dot
from ._sine_preparation import _prepared_transit_radii
from ._sine_preparation_mode_selection import (
    _LIFT_SCALE,
    _STORAGE_BUDGET,
    _paired_source_construction,
    _PairedPreparation,
)
from ._sine_sector_boundary import _sector_boundary_bounds
from .phase_cycle_geometry import _derive
from .relational_observations import _ordered
from .relational_sine_formed_classes import (
    _CAPACITY,
    _EDGES,
    _GAP,
    _NODES,
    _SOURCE_SCALE,
    _WINDINGS,
    _verify_cycle_gap,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

_ERROR_TRANSPORT = Q(1025, 1023)


@dataclass(frozen=True)
class _CayleyPath:
    source: _PairedPreparation
    generator: tuple[tuple[Q, ...], ...]
    generator_squared: tuple[tuple[Q, ...], ...]
    generator_norm: Q
    kappa: Q
    p_zero: Q
    p_linear: Q
    isometry_certified: bool


def _path_construction() -> _CayleyPath:
    """Verify the rank-two rational Cayley identities, not a sampled path."""
    source = _paired_source_construction()
    first, second = source.normalized_initial_forms_by_class
    laplacian = source.source_laplacian
    u = tuple(a + b for a, b in zip(first, second))
    v = tuple(b - a for a, b in zip(first, second))
    lu, lv = (tuple(dot(row, vector) for row in laplacian) for vector in (u, v))
    uu, vv = dot(u, lu), dot(v, lv)
    if uu <= 0 or vv <= 0 or dot(u, lv) != 0:
        raise ArithmeticError("the fixed Cayley plane failed exact L-orthogonality")
    kappa = vv / uu
    generator = tuple(
        tuple((v[i] * lu[j] - u[i] * lv[j]) / uu for j in _NODES) for i in _NODES
    )
    squared = exact_matrix_product(generator, generator)
    cubed = exact_matrix_product(squared, generator)
    left = exact_matrix_product(tuple(zip(*generator)), laplacian)
    right = exact_matrix_product(laplacian, generator)
    norm = max(sum(map(abs, row), Q(0)) for row in generator)
    checks = (
        all(left[i][j] + right[i][j] == 0 for i in _NODES for j in _NODES),
        all(cubed[i][j] == -kappa * generator[i][j] for i in _NODES for j in _NODES),
        tuple(dot(row, u) for row in generator) == v,
        tuple(dot(row, v) for row in generator) == tuple(-kappa * value for value in u),
        all(sum(row, Q(0)) == 0 for row in generator),
        all(sum(row, Q(0)) == 0 for row in zip(*generator)),
        norm < Q(1, 1024),
        0 < kappa < Q(1, 3),
    )
    p_zero = Q(9 * _LIFT_SCALE) + Q(1, _LIFT_SCALE)
    p_linear = 18 * _LIFT_SCALE * kappa - Q(2, _LIFT_SCALE)
    if not all(checks) or p_linear >= 0:
        raise ArithmeticError("the fixed Cayley path failed its algebraic bounds")
    # K^3=-kappa*K gives Q_s=I+2s*K/(1+kappa*s^2)
    # +2s^2*K^2/(1+kappa*s^2). L-skewness proves its exact L-isometry;
    # the row/column sums preserve both the common vector and the form mean.
    return _CayleyPath(source, generator, squared, norm, kappa, p_zero, p_linear, True)


def _path_coefficients(path, parameter):
    """Exact p,q,f,g at an already admitted rational parameter in [0,1]."""
    denominator = 1 + path.kappa * parameter**2
    p = (
        path.p_zero
        + path.p_linear * parameter
        - path.kappa * path.p_zero * parameter**2
    ) / denominator
    q = (1 + parameter + path.kappa * (3 * parameter - parameter**2)) / denominator
    return p, q, 2 * parameter / denominator, 2 * parameter**2 / denominator


def _correlated_ramp_storage_bounds(angle):
    """Enclose U0(t)=9-8cos(t)-cos(8t) from its finite analytic extrema.

    On [0,pi/2], U0'=16sin(9t/2)cos(7t/2). Endpoints and the listed
    stationary angles therefore suffice. Intersecting critical-angle boxes
    may include an extra candidate conservatively; no parameter grid is used.
    """
    pi = pi_interval()
    if angle.lo < 0 or angle.hi > pi.lo / 2:
        raise ValueError("the ramp angle enclosure must lie within [0,pi/2]")
    candidates = [I(angle.lo), I(angle.hi)]
    for multiple in (Q(1, 7), Q(2, 9), Q(3, 7), Q(4, 9)):
        critical = multiple * pi
        if critical.hi >= angle.lo and critical.lo <= angle.hi:
            candidates.append(critical)
    values = tuple(9 - 8 * cos(point) - cos(8 * point) for point in candidates)
    return (
        I(min(value.lo for value in values), max(value.hi for value in values)),
        tuple(candidates),
        values,
    )


@dataclass(frozen=True)
class _PreparationPathSelection:
    """A uniform conditional statement for the entire admitted path interval.

    Residuals are Q_s*xi with one latent zero-sum xi and a common zero-sum
    phase row zeta. Enclosing coordinate boxes are not independent sources.
    The classifications concern the same complete law at the declared time.
    Capture is of an acute sector, not entry into a previously used small
    target ball. A winding-two exit does not specify its later winding.
    """

    parameter_interval: tuple[Q, Q]
    scaled_time: Q
    form_error_bound: Q
    phase_error_bound: Q
    original_time: Q
    source: _PairedPreparation
    path_generator: tuple[tuple[Q, ...], ...]
    path_generator_squared: tuple[tuple[Q, ...], ...]
    path_generator_norm: Q
    path_kappa: Q
    path_isometry_certified: bool
    path_operator_bounds: tuple[tuple[I, ...], ...]
    p_bounds: I
    q_bounds: I
    nominal_initial_form_bounds: tuple[I, ...]
    uniform_form_error_transport_upper_bound: Q
    transported_form_error_bound: Q
    preparation_dimension: int
    full_relative_preparation: bool
    common_initial_storage_upper_bound: Q
    source_budget_margin_bounds: I
    source_budget_certified: bool
    gamma_bounds: I
    eta_bounds: I
    forcing_norm_bounds: I
    nominal_scaled_initial_norm_upper_bound: Q
    scaled_initial_error_norm_upper_bound: Q
    scaled_initial_norm_upper_bound: Q
    initial_phase_error_norm_upper_bound: Q
    exponential_decay_bounds: I
    scaled_form_radius_upper_bound: Q
    endpoint_form_norm_upper_bound: Q
    endpoint_phase_error_norm_upper_bound: Q
    endpoint_phase_edge_error_upper_bound: Q
    ramp_angle_bounds: I
    lift_correction_bounds: I
    endpoint_unwrapped_phase_edge_bounds: tuple[I, ...]
    ramp_extremum_angle_bounds: tuple[I, ...]
    ramp_extremum_storage_bounds: tuple[I, ...]
    ramp_storage_bounds: I
    proxy_phase_storage_upper_bound: Q
    endpoint_phase_storage_upper_bound: Q
    endpoint_form_storage_upper_bound: Q
    endpoint_total_storage_upper_bound: Q
    initial_zero_winding_margin_bounds: I
    initial_zero_winding_certified: bool
    edge_turn_offsets_by_winding: tuple[tuple[int, ...], ...]
    phase_edge_bounds_by_winding: tuple[tuple[I, ...], ...]
    phase_cut_margin_bounds_by_winding: tuple[tuple[I, ...], ...]
    acute_margin_bounds_by_winding: tuple[tuple[I, ...], ...]
    sector_cycle_periods_by_winding: tuple[tuple[int, ...], ...]
    sector_boundary_face_lower_bounds: tuple[tuple[Q, ...], ...]
    sector_barrier_lower_bounds: tuple[Q, Q]
    capture_margin_bounds: tuple[I, I]
    acute_capture_certified_by_winding: tuple[bool, bool]
    winding_at_time: int | None
    acute_capture_certified: bool
    captured_winding: int | None
    winding_two_critical_storage_lower_bound: Q
    forced_winding_two_exit_margin_bounds: I
    forced_winding_two_exit_certified: bool
    preparation_policy_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    held_capacities: tuple[Q, ...] = _CAPACITY
    initial_phases: tuple[Q, ...] = (Q(0),) * 9
    metric_weights: tuple[Q, ...] = (Q(2),) * 9
    candidate_windings: tuple[int, int] = _WINDINGS
    conserved_form_mean: Q = Q(0)
    conserved_lifted_phase_mean: Q = Q(0)
    common_source_storage_budget: Q = _STORAGE_BUDGET
    form_loss: Q = Q(1023, 1024)
    exchange_weight: Q = Q(1, 1024)
    phase_exchange_beta: Q = Q(1)
    deformation: Q = Q(0)
    latent_residual_constraints: tuple[str, ...] = (
        "xi_and_zeta_have_separate_exact_zero_sums",
        "form_residual_is_Q_s_times_the_same_latent_xi_at_every_parameter",
        "the_same_zeta_is_the_lifted_phase_residual_at_every_parameter",
    )
    clock: str = "tau=e*t; e=1023/1024"
    law: str = "normalized_sine_reciprocal_exchange"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "one_fixed_energy_isometric_preparation_path_not_a_time_dependent_input",
        "each_latent_source_keeps_exact_initial_phase_means_and_storage_along_the_path",
        "complete_Q_s_times_xi_error_family_not_the_earlier_reflection_endpoint_family",
        "uniform_bounds_on_the_closed_parameter_interval_without_point_sampling",
        "full_continuous_phase_lifts_retained_before_exact_integer_edge_offset_cancellation",
        "global_complete_law_transit_through_nonacute_states",
        "correlated_ramp_potential_extrema_and_all_face_sector_storage_barriers",
        "winding_at_time_is_not_by_itself_capture_or_a_crossing_history",
        "acute_capture_is_not_old_small_ball_entry_or_a_uniform_dwell_certificate",
        "forced_winding_two_exit_has_no_claimed_exit_time_final_mode_or_single_threshold",
        "source_budget_and_initial_zero_winding_are_separate_preparation_policy_flags",
        "unresolved_sufficient_tests_do_not_prove_failed_formation_or_a_basin_boundary",
        "no_trajectory_coefficient_producer_source_sweep_or_frozen_evidence_consumption",
    )


def _assess_preparation_path_selection(
    *, parameter_interval, scaled_time, form_error_bound, phase_error_bound
) -> _PreparationPathSelection:
    """Admit a closed preparation interval and prove only sufficient selectors.

    The two parameter endpoints are exact/represented finite reals in [0,1],
    ordered before outward materialization. Time and residual budgets are
    nonnegative, with scaled_time/5 <= 4096. All four primitives are required.
    The fixed source, law and storage policy are rebuilt before their use.
    """
    raw = _ordered(parameter_interval, "parameter_interval", limit=3)
    if len(raw) != 2:
        raise ValueError("parameter_interval must contain exactly two endpoints")
    lower, upper = tuple(_exact(value, "parameter_interval") for value in raw)
    time = _exact(scaled_time, "scaled_time")
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if not 0 <= lower <= upper <= 1:
        raise ValueError("parameter_interval requires 0 <= lower <= upper <= 1")
    if time < 0 or _GAP * time > _MAX_RATIONAL_EXPONENT:
        raise ValueError("scaled_time requires 0 <= scaled_time/5 <= 4096")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")

    path = _path_construction()
    source = path.source
    _verify_cycle_gap()
    low_coefficients, high_coefficients = (
        _path_coefficients(path, value) for value in (lower, upper)
    )
    # p decreases; q,f,g increase on [0,1], as proved by the verified
    # kappa<1/3 and p_linear<0. Exact endpoint fractions precede I rounding.
    p = I(high_coefficients[0], low_coefficients[0])
    q = I(low_coefficients[1], high_coefficients[1])
    f = I(low_coefficients[2], high_coefficients[2])
    g = I(low_coefficients[3], high_coefficients[3])
    operator = tuple(
        tuple(
            I(int(i == j)) + f * path.generator[i][j] + g * path.generator_squared[i][j]
            for j in _NODES
        )
        for i in _NODES
    )
    ramp = tuple(Q(j - 4) for j in _NODES)
    dipole = tuple(Q(int(j == 3) - int(j == 4)) for j in _NODES)
    forms = tuple(_SOURCE_SCALE * (p * h + q * r) for h, r in zip(dipole, ramp))
    pi = pi_interval()
    gamma, inverse_gamma = Q(1, 1023) / pi, 1023 * pi
    eta, forcing = Q(1, 1023**2) / pi**2, sqrt(I(18))
    transported_error = _ERROR_TRANSPORT * rx
    # ||p*h+q*r||^2=2p^2-2pq+60q^2; p,q are positive. Drop the
    # negative cross term after algebra, rather than box the large lift.
    nominal_norm = (
        gamma * _SOURCE_SCALE * sqrt(I(2 * (2 * p.hi**2 + 60 * q.hi**2)))
    ).hi
    form_error = (gamma * forcing * transported_error).hi
    phase_error = (forcing * rt).hi
    initial_norm = nominal_norm + form_error
    decay = I(*_negative_exp_bounds(_GAP * time))
    scaled_radius, phase_radius = _prepared_transit_radii(
        time=time,
        decay_upper=decay.hi,
        gap_lower=_GAP,
        initial_norm_upper=initial_norm,
        feedback_upper=eta.hi,
        forcing_upper=forcing.hi,
    )
    inverse_root_two = (1 / sqrt(I(2))).hi
    form_norm = inverse_gamma.hi * scaled_radius * inverse_root_two
    phase_remainder = (phase_radius + form_error + phase_error) * inverse_root_two
    edge_error = sqrt(I(2)).hi * phase_remainder
    t, delta = (
        gamma * _SOURCE_SCALE * q,
        gamma * _SOURCE_SCALE * p - 2 * pi * _LIFT_SCALE,
    )
    error_interval = I(-edge_error, edge_error)
    unwrapped = tuple(
        2 * pi * (_LIFT_SCALE * (dipole[j] - dipole[i]))
        + t * (ramp[j] - ramp[i])
        + delta * (dipole[j] - dipole[i])
        + error_interval
        for i, j in _EDGES
    )
    ramp_storage, critical_angles, critical_values = _correlated_ramp_storage_bounds(t)
    proxy_storage = ramp_storage.hi + 3 * delta.abs_max**2
    phase_storage = proxy_storage + 6 * phase_remainder
    form_storage = 2 * form_norm**2
    total_storage = form_storage + phase_storage
    first = source.initial_forms_by_class[0]
    source_storage = (
        sum(((abs(first[j] - first[i]) + 2 * rx) ** 2 / 2 for i, j in _EDGES), Q(0))
        + 18 * rt**2
    )
    source_margin = I(_STORAGE_BUDGET - source_storage)
    initial_margin = pi / 2 - 2 * rt
    initial_zero, source_ok = initial_margin.lo > 0, source_margin.lo > 0
    geometry = _derive(_NODES, _EDGES)
    offsets, phase_edges, cut_margins, acute_margins, periods, faces, barriers = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for target, principal in zip(
        source.target_phase_turns_by_class, source.target_principal_edge_turns_by_class
    ):
        raw_offsets = tuple(
            turn - target[j] + target[i] for (i, j), turn in zip(_EDGES, principal)
        )
        if any(value.denominator != 1 for value in raw_offsets):
            raise ArithmeticError("the fixed sector offsets must be integral")
        integers = tuple(int(value) for value in raw_offsets)
        # Combine the known integer lift and cell offset before multiplying pi.
        edges = tuple(
            t * (ramp[j] - ramp[i])
            + delta * (dipole[j] - dipole[i])
            + 2 * pi * (_LIFT_SCALE * (dipole[j] - dipole[i]) + offset)
            + error_interval
            for (i, j), offset in zip(_EDGES, integers)
        )
        cycles = tuple(
            sum(sign * offset for sign, offset in zip(row, integers))
            for row in geometry.cycle_rows
        )
        boundary = _sector_boundary_bounds(geometry, cycles)
        offsets.append(integers)
        phase_edges.append(edges)
        cut_margins.append(tuple(pi - abs(value) for value in edges))
        acute_margins.append(tuple(pi / 2 - abs(value) for value in edges))
        periods.append(cycles)
        faces.append(boundary)
        barriers.append(min(boundary))
    windings = tuple(
        k
        for k, row in zip(_WINDINGS, cut_margins)
        if all(value.lo > 0 for value in row)
    )
    if len(windings) > 1:
        raise ArithmeticError(
            "disjoint strict winding charts both admitted one source family"
        )
    winding = windings[0] if windings else None
    capture_margins = tuple(I(barrier - total_storage) for barrier in barriers)
    capture_flags = tuple(
        all(value.lo > 0 for value in row) and margin.lo > 0
        for row, margin in zip(acute_margins, capture_margins)
    )
    captures = tuple(k for k, ok in zip(_WINDINGS, capture_flags) if ok)
    captured = captures[0] if captures else None
    critical_minimum = (9 * (1 - cos(4 * pi / 9))).lo
    exit_margin = I(critical_minimum - total_storage)
    exits = winding == 2 and exit_margin.lo > 0
    status = (
        "certified_acute_capture"
        if captured is not None
        else "certified_winding_two_exit" if exits else "unresolved"
    )
    reasons = tuple(
        reason
        for ok, reason in (
            (source_ok, "strict_common_source_storage_budget_not_certified"),
            (initial_zero, "whole_initial_set_zero_winding_not_certified"),
            (winding is not None, "uniform_winding_at_time_not_certified"),
            (
                captured is not None or exits,
                "capture_or_winding_two_exit_not_certified",
            ),
        )
        if not ok
    )
    return _PreparationPathSelection(
        parameter_interval=(lower, upper),
        scaled_time=time,
        form_error_bound=rx,
        phase_error_bound=rt,
        original_time=time / Q(1023, 1024),
        source=source,
        path_generator=path.generator,
        path_generator_squared=path.generator_squared,
        path_generator_norm=path.generator_norm,
        path_kappa=path.kappa,
        path_isometry_certified=path.isometry_certified,
        path_operator_bounds=operator,
        p_bounds=p,
        q_bounds=q,
        nominal_initial_form_bounds=forms,
        uniform_form_error_transport_upper_bound=_ERROR_TRANSPORT,
        transported_form_error_bound=transported_error,
        preparation_dimension=8 * int(rx > 0) + 8 * int(rt > 0),
        full_relative_preparation=rx > 0 and rt > 0,
        common_initial_storage_upper_bound=source_storage,
        common_source_storage_budget=_STORAGE_BUDGET,
        source_budget_margin_bounds=source_margin,
        source_budget_certified=source_ok,
        gamma_bounds=gamma,
        eta_bounds=eta,
        forcing_norm_bounds=forcing,
        nominal_scaled_initial_norm_upper_bound=nominal_norm,
        scaled_initial_error_norm_upper_bound=form_error,
        scaled_initial_norm_upper_bound=initial_norm,
        initial_phase_error_norm_upper_bound=phase_error,
        exponential_decay_bounds=decay,
        scaled_form_radius_upper_bound=scaled_radius,
        endpoint_form_norm_upper_bound=form_norm,
        endpoint_phase_error_norm_upper_bound=phase_remainder,
        endpoint_phase_edge_error_upper_bound=edge_error,
        ramp_angle_bounds=t,
        lift_correction_bounds=delta,
        endpoint_unwrapped_phase_edge_bounds=unwrapped,
        ramp_extremum_angle_bounds=critical_angles,
        ramp_extremum_storage_bounds=critical_values,
        ramp_storage_bounds=ramp_storage,
        proxy_phase_storage_upper_bound=proxy_storage,
        endpoint_phase_storage_upper_bound=phase_storage,
        endpoint_form_storage_upper_bound=form_storage,
        endpoint_total_storage_upper_bound=total_storage,
        initial_zero_winding_margin_bounds=initial_margin,
        initial_zero_winding_certified=initial_zero,
        edge_turn_offsets_by_winding=tuple(offsets),
        phase_edge_bounds_by_winding=tuple(phase_edges),
        phase_cut_margin_bounds_by_winding=tuple(cut_margins),
        acute_margin_bounds_by_winding=tuple(acute_margins),
        sector_cycle_periods_by_winding=tuple(periods),
        sector_boundary_face_lower_bounds=tuple(faces),
        sector_barrier_lower_bounds=tuple(barriers),
        capture_margin_bounds=capture_margins,
        acute_capture_certified_by_winding=capture_flags,
        winding_at_time=winding,
        acute_capture_certified=captured is not None,
        captured_winding=captured,
        winding_two_critical_storage_lower_bound=critical_minimum,
        forced_winding_two_exit_margin_bounds=exit_margin,
        forced_winding_two_exit_certified=exits,
        preparation_policy_certified=source_ok and initial_zero,
        status=status,
        unresolved_conditions=reasons,
    )
