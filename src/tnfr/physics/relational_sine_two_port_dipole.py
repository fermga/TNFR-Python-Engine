"""Finite interior phase-dipole response after conditional acute warmup.

The source premise is a trapped family in the stated metric neighborhood.
Shared modified-energy kernels prove a finite warmup; an exact target and
full-law remainder then bound the local readout. No source history is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin
from ._sine_admission import _sine_model_coefficients
from ._sine_lyapunov import (
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)
from .phase_cycle_geometry import PhaseCycleGeometry, _derive
from .relational_sine_two_port_compatibility import (
    _EDGES,
    _NODES,
    SineTwoPortCompatibility,
    assess_sine_two_port_compatibility,
)

__all__ = ("SineTwoPortDipole", "assess_sine_two_port_dipole")

_GAP = Q(1, 90)
_RATE = Q(2)
_COSINE = Q(1, 25)
_RADIUS = Q(1, 12)
_BARRIER = Q(1, 648000)
_ETA_LOWER = Q(1, 11000000)
_ETA_UPPER = Q(1, 9000000)
_GAMMA_UPPER = Q(1, 3069)


def _dipole_geometry():
    """Verify both actual supports on their respective mean-free spaces."""
    geometry = _derive(_NODES, _EDGES)
    control = tuple(edge for edge in _EDGES if edge not in ((0, 9), (1, 10)))
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in _NODES)
    degree_rows, laplacians, lower_rows, upper_rows = [], [], [], []
    for edges, pieces in (
        (_EDGES, (_NODES,)),
        (control, (tuple(range(9)), tuple(range(9, 18)))),
    ):
        degrees = tuple(sum(i in edge for edge in edges) for i in _NODES)
        laplacian = tuple(
            tuple(
                Q(degrees[i] if i == j else -int(tuple(sorted((i, j))) in edges))
                for j in _NODES
            )
            for i in _NODES
        )
        centered = tuple(
            tuple(
                Q(degrees[i] * int(i == j))
                - sum(
                    (
                        Q(degrees[i] * degrees[j], sum(degrees[k] for k in piece))
                        for piece in pieces
                        if i in piece and j in piece
                    ),
                    Q(0),
                )
                for j in _NODES
            )
            for i in _NODES
        )
        lower = tuple(
            tuple(laplacian[i][j] - _GAP * centered[i][j] for j in _NODES)
            for i in _NODES
        )
        upper = tuple(
            tuple(_RATE * degrees[i] * int(i == j) - laplacian[i][j] for j in _NODES)
            for i in _NODES
        )
        if not all(map(exact_symmetric_semidefinite, (lower, upper))):
            raise ArithmeticError("fixed dipole normalized spectral bounds failed")
        if (
            any(sum(degrees[i] * q[i] for i in piece) != 0 for piece in pieces)
            or sum(d * v**2 for d, v in zip(degrees, q)) != 4
            or sum(v**2 / d for d, v in zip(degrees, q)) != 1
            or tuple(
                (edge, q[edge[1]] - q[edge[0]])
                for edge in edges
                if q[edge[1]] != q[edge[0]]
            )
            != (((3, 4), Q(1)), ((4, 5), Q(-2)), ((5, 6), Q(1)))
        ):
            raise ArithmeticError("fixed interior dipole geometry failed")
        degree_rows.append(degrees)
        laplacians.append(laplacian)
        lower_rows.append(lower)
        upper_rows.append(upper)
    return dict(
        geometry=geometry,
        disconnected_edges=control,
        degrees_by_model=tuple(degree_rows),
        laplacians=tuple(laplacians),
        normalized_gap_slack_matrices=tuple(lower_rows),
        normalized_upper_slack_matrices=tuple(upper_rows),
        dipole=q,
    )


def _warmup_envelope(duration):
    """Pure rational bounds in Euclidean coordinates after the degree isometry."""
    mu, stiffness, lower, upper, rate = _sine_lyapunov_coefficients(
        eta_lower=_ETA_LOWER,
        eta_upper=_ETA_UPPER,
        cosine_lower=_COSINE,
        gap_lower=_GAP,
        rate_upper=_RATE,
    )
    exponent = rate * duration
    power = exponent.numerator // exponent.denominator
    if not 0 <= power <= 4096:
        raise ValueError(
            "floor(lyapunov_decay_rate*warmup_duration) must lie in 0..4096"
        )
    decay = Q(1, 2**power)
    initial = _sine_lyapunov_initial_upper(
        eta_upper=_ETA_UPPER,
        rate_upper=_RATE,
        gap_lower=_GAP,
        position_upper=upper,
        form_norm_upper=_RADIUS,
        phase_norm_upper=_RADIUS,
    )
    returned, form_squared, phase_squared = _sine_lyapunov_return_squared(
        initial_upper=initial,
        decay_upper=decay,
        eta_lower=_ETA_LOWER,
        gap_lower=_GAP,
        rate_upper=_RATE,
        position_lower=lower,
    )
    return dict(
        phase_stiffness_lower_bound=mu,
        phase_stiffness_upper_bound=stiffness,
        lyapunov_position_lower_coefficient=lower,
        lyapunov_position_upper_coefficient=upper,
        lyapunov_decay_rate=rate,
        warmup_decay_exponent=exponent,
        warmup_decay_power=power,
        warmup_decay_upper_bound=decay,
        initial_lyapunov_upper_bound=initial,
        returned_lyapunov_candidate=returned,
        warmup_form_norm_squared_candidate=form_squared,
        warmup_phase_norm_squared_candidate=phase_squared,
    )


@dataclass(frozen=True)
class SineTwoPortDipole:
    """Conditional finite warmup, local response, supplied work and recovery.

    Both source families are already trapped in the radius-1/12 acute target
    chart, with source form and phase norms at most1/12. The disconnected norm
    is the combined eighteen-node degree norm after removing each component's
    own form and phase means. Source acquisition is a separate premise.

    ``form_radius`` and ``phase_radius`` are desired post-warmup norm bounds.
    Delivered response, work and recovery require their strict verification;
    endpoint candidates alone do not supply this handoff. Phase uses radians.
    Each readout error bounds one scalar q-transpose-x observation, not a node.
    All four observation errors remain in the joined-minus-control increment.

    The explicit phase-blind alternative retains x'=-A*x, theta'=gamma*A*x,
    the same support, held capacities, clocks and original preparation family.
    Its finite heat warmup is independently bounded; it is not a modified
    native relational model or a claim about every constitutive alternative.
    """

    warmup_duration: Q
    form_radius: Q
    phase_radius: Q
    phase_increment: Q
    probe_duration: Q
    readout_error_bound: Q
    contrast_threshold: Q
    work_allowance: Q
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    disconnected_edges: tuple[tuple[int, int], ...]
    degrees_by_model: tuple[tuple[int, ...], tuple[int, ...]]
    laplacians: tuple[tuple[tuple[Q, ...], ...], ...]
    normalized_gap_slack_matrices: tuple[tuple[tuple[Q, ...], ...], ...]
    normalized_upper_slack_matrices: tuple[tuple[tuple[Q, ...], ...], ...]
    dipole: tuple[Q, ...]
    target: SineTwoPortCompatibility
    target_acute_margin_lower_bound: Q | None
    target_admitted: bool
    phase_stiffness_lower_bound: Q
    phase_stiffness_upper_bound: Q
    lyapunov_position_lower_coefficient: Q
    lyapunov_position_upper_coefficient: Q
    lyapunov_decay_rate: Q
    warmup_decay_exponent: Q
    warmup_decay_power: int
    warmup_decay_upper_bound: Q
    initial_lyapunov_upper_bound: Q
    returned_lyapunov_candidate: Q
    warmup_form_norm_squared_candidate: Q
    warmup_phase_norm_squared_candidate: Q
    warmup_form_norm_squared_upper_bound: Q | None
    warmup_phase_norm_squared_upper_bound: Q | None
    warmup_form_margin: Q | None
    warmup_phase_margin: Q | None
    warmup_certified: bool
    gamma_bounds: I
    bulk_angle_bounds_by_model: tuple[I, I] | None
    cosine_gap_bounds: I | None
    geometry_certified: bool
    ideal_initial_slope_bounds_by_model: tuple[I, I] | None
    ideal_increment_bounds_by_model: tuple[I, I] | None
    ideal_correlated_contrast_bounds: I | None
    whole_window_form_norm_candidate: Q
    whole_window_phase_norm_candidate: Q
    finite_remainder_candidate: Q
    finite_remainder_upper_bound: Q | None
    true_increment_bounds_by_model: tuple[I, I] | None
    recorded_increment_bounds_by_model: tuple[I, I] | None
    recorded_contrast_bounds: I | None
    response_margin: Q | None
    response_certified: bool
    heat_warmup_exponent: Q
    heat_warmup_form_norm_upper_bound: Q | None
    heat_warmup_certified: bool
    phase_blind_recorded_contrast_upper_bound: Q | None
    phase_blind_exclusion_margin: Q | None
    heat_control_excluded: bool
    nominal_work_bounds_by_model: tuple[I, I] | None
    work_error_candidate: Q
    work_bounds_candidates: tuple[I, I] | None
    work_bounds_by_model: tuple[I, I] | None
    work_allowance_margins: tuple[Q, Q] | None
    positive_work_certified_by_model: tuple[bool, bool]
    work_certified_by_model: tuple[bool, bool]
    work_certified: bool
    post_probe_relative_norm_squared_candidate: Q
    post_probe_excess_storage_candidates: tuple[Q, Q] | None
    post_probe_excess_storage_upper_bounds: tuple[Q, Q] | None
    post_probe_radius_margin: Q | None
    capture_storage_margins: tuple[Q, Q] | None
    identity_certified_by_model: tuple[bool, bool]
    identity_certified: bool
    recovery_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    model_order: tuple[str, str] = ("joined_two_port", "unjoined_two_cycles")
    root_outer_refinements: int = 32
    root_inner_refinements: int = 64
    source_form_norm_upper_bound: Q = _RADIUS
    source_phase_norm_upper_bound: Q = _RADIUS
    eta_bounds: tuple[Q, Q] = (_ETA_LOWER, _ETA_UPPER)
    normalized_gap_lower_bound: Q = _GAP
    normalized_rate_upper_bound: Q = _RATE
    local_radius: Q = _RADIUS
    local_cosine_lower_bound: Q = _COSINE
    capture_barrier_lower_bound: Q = _BARRIER
    cosine_gap_threshold: Q = Q(1, 32)
    heat_source_form_norm_upper_bound: Q = Q(7, 65536)
    heat_decay_power: int = 128
    capacity: tuple[Q, ...] = (Q(1),) * 18
    probe_law: str = "theta_plus=theta_minus+phase_increment*(e4-e5); x_plus=x_minus"
    readout: str = "(e4-e5)^T*(x_after-x_before); one bounded error per scalar reading"
    phase_blind_law: str = "x'=-K*L*x; theta'=gamma*K*L*x; no phase-to-form feedback"
    clock: str = "tau=e*t; e=1023/1024; gamma=1/(1023*pi)"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "conditional_already_trapped_source_families_not_an_acquisition_certificate",
        "same_positive_loss_complete_sine_law_and_held_unit_capacities",
        "joined_degree_mean_and_separate_control_component_means_retained",
        "control_norm_is_combined_full_eighteen_node_degree_norm",
        "degree_isometry_reuses_the_shared_nonlinear_modified_energy_kernels",
        "finite_declared_warmup_exact_dyadic_decay_no_asymptotic_state_reset",
        "every_consumed_post_warmup_radius_requires_its_own_strict_squared_norm_margin",
        "same_interior_phase_dipole_and_local_scalar_observation_on_both_supports",
        "correlated_target_cosine_difference_retained_before_error_expansion",
        "full_nonlinear_remainder_not_an_initial_slope_only_prediction",
        "four_scalar_observation_errors_not_per_node_noise_or_probabilistic_independence",
        "phase_blind_control_has_its_own_complete_law_and_original_source_heat_warmup",
        "signed_supplied_phase_jump_work_separate_from_continuous_dissipation",
        "strict_post_jump_barriers_retain_each_target_chart_and_prove_eventual_recovery",
        "no_capture_producer_trajectory_autonomous_probe_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-two-port-dipole.v1", "report": _project(self)}


def assess_sine_two_port_dipole(
    *,
    warmup_duration,
    form_radius,
    phase_radius,
    phase_increment,
    probe_duration,
    readout_error_bound,
    contrast_threshold,
    work_allowance,
) -> SineTwoPortDipole:
    """Assess a supplied dipole after a finite conditional complete-law warmup.

    Eight exact/represented scalars are mandatory. Duration, desired radii,
    readout error, threshold and work allowance are nonnegative; phase
    increment and probe duration lie in (0,1]. The floor of kappa times warmup
    duration must lie in0..4096. All admission precedes the fresh target.

    The source is already trapped with both norm caps1/12. A separate source
    theorem or evidence audit must establish that premise for actual prepared
    states. No incoming report supplies it here. Failed warmup or target gates
    leave delivered response, work, identity and recovery unavailable.
    """
    labels = (
        "warmup_duration",
        "form_radius",
        "phase_radius",
        "phase_increment",
        "probe_duration",
        "readout_error_bound",
        "contrast_threshold",
        "work_allowance",
    )
    values = tuple(
        exact_or_represented_real(value, name)
        for value, name in zip(
            (
                warmup_duration,
                form_radius,
                phase_radius,
                phase_increment,
                probe_duration,
                readout_error_bound,
                contrast_threshold,
                work_allowance,
            ),
            labels,
        )
    )
    duration, x, y, amplitude, horizon, noise, threshold, allowance = values
    if min(duration, x, y, noise, threshold, allowance) < 0:
        raise ValueError(
            "warmup, radii, readout error, threshold and allowance must be nonnegative"
        )
    if not 0 < amplitude <= 1 or not 0 < horizon <= 1:
        raise ValueError("phase_increment and probe_duration must lie in (0,1]")
    warmup = _warmup_envelope(duration)
    geometry = _dipole_geometry()
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    _sine_model_coefficients(model, positive_loss=True)
    target = assess_sine_two_port_compatibility(
        classes=(2, 1), outer_refinements=32, inner_refinements=64
    )
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    target_margin = (
        (2 * pi * target.acute_margin_turns_bounds).lo
        if target.acute_margin_turns_bounds is not None
        else None
    )
    target_ok = (
        target.local_attraction_certified
        and target_margin is not None
        and target_margin > Q(1, 8)
        and target.bulk_arc_turn_bounds is not None
    )
    xmargin = x**2 - warmup["warmup_form_norm_squared_candidate"]
    ymargin = y**2 - warmup["warmup_phase_norm_squared_candidate"]
    warmed = target_ok and xmargin > 0 and ymargin > 0
    c = 2 * _GAMMA_UPPER * horizon
    qmax = (x + c * (y + 2 * amplitude)) / (1 - c**2)
    pmax = y + 2 * amplitude + c * qmax
    error = (
        2 * horizon * x
        + 4 * _GAMMA_UPPER * amplitude * horizon**2
        + 2 * _GAMMA_UPPER * horizon * y
        + 4 * _GAMMA_UPPER**2 * horizon**2 * qmax
    )
    angles = gap = slopes = ideal = contrast_ideal = nominal_work = work_candidates = (
        energies
    ) = None
    geometry_ok = False
    work_error = 4 * amplitude * y
    if target_ok:
        angles = (2 * pi * target.bulk_arc_turn_bounds[0], 4 * pi / 9)
        gap = cos(angles[1] - amplitude / 2) - cos(angles[0] - amplitude / 2)
        geometry_ok = gap.lo > Q(1, 32)
        slopes = tuple(
            -gamma * (sin(b + amplitude) - sin(b - 2 * amplitude)) for b in angles
        )
        ideal = tuple(value * horizon for value in slopes)
        contrast_ideal = 2 * gamma * horizon * sin(I(Q(3, 2) * amplitude)) * gap
        nominal_work = tuple(
            3 * cos(b) - 2 * cos(b + amplitude) - cos(b - 2 * amplitude) for b in angles
        )
        work_candidates = tuple(
            value + I(-work_error, work_error) for value in nominal_work
        )
        energies = tuple(x**2 + y**2 + value.hi for value in work_candidates)
    true = tuple(value + I(-error, error) for value in ideal) if warmed else None
    recorded = (
        tuple(value + I(-2 * noise, 2 * noise) for value in true) if warmed else None
    )
    contrast = (
        contrast_ideal + I(-2 * error - 4 * noise, 2 * error + 4 * noise)
        if warmed
        else None
    )
    response_margin = contrast.lo - threshold if contrast is not None else None
    response = warmed and geometry_ok and response_margin > 0
    heat_exponent = _GAP * duration
    heat_form = Q(7, 65536 * 2**128) if heat_exponent >= 128 else None
    heat_ok = heat_form is not None and heat_form <= x
    heat_bound = 4 * horizon * x + 4 * noise if heat_ok else None
    heat_margin = (
        contrast.lo - heat_bound
        if contrast is not None and heat_bound is not None
        else None
    )
    heat_excluded = (
        warmed and geometry_ok and heat_margin is not None and heat_margin > 0
    )
    works = work_candidates if warmed else None
    work_margins = tuple(allowance - value.hi for value in works) if warmed else None
    work_flags = (
        tuple(value >= 0 for value in work_margins) if warmed else (False, False)
    )
    positive_work = tuple(value.lo > 0 for value in works) if warmed else (False, False)
    norm = x**2 + (y + 2 * amplitude) ** 2
    radius_margin = _RADIUS**2 - norm if warmed else None
    storage_margins = tuple(_BARRIER - value for value in energies) if warmed else None
    identity_flags = (
        tuple(radius_margin > 0 and margin > 0 for margin in storage_margins)
        if warmed
        else (False, False)
    )
    reasons = tuple(
        reason
        for flag, reason in (
            (target_ok, "required_implicit_target_geometry_not_certified"),
            (warmed, "strict_finite_warmup_residual_budgets_not_certified"),
            (geometry_ok, "strict_target_cosine_gap_not_certified"),
            (response, "recorded_contrast_not_strictly_above_threshold"),
            (heat_ok, "phase_blind_original_source_heat_warmup_not_certified"),
            (
                heat_excluded,
                "recorded_contrast_does_not_strictly_exclude_phase_blind_control",
            ),
            (all(work_flags), "supplied_work_allowance_not_certified_for_both_models"),
            (
                all(identity_flags),
                "strict_post_probe_capture_not_certified_for_both_models",
            ),
        )
        if not flag
    )
    return SineTwoPortDipole(
        **dict(zip(labels, values)),
        **geometry,
        **warmup,
        reference_model=model,
        target=target,
        target_acute_margin_lower_bound=target_margin,
        target_admitted=target_ok,
        warmup_form_norm_squared_upper_bound=(
            warmup["warmup_form_norm_squared_candidate"] if target_ok else None
        ),
        warmup_phase_norm_squared_upper_bound=(
            warmup["warmup_phase_norm_squared_candidate"] if target_ok else None
        ),
        warmup_form_margin=xmargin if target_ok else None,
        warmup_phase_margin=ymargin if target_ok else None,
        warmup_certified=warmed,
        gamma_bounds=gamma,
        bulk_angle_bounds_by_model=angles,
        cosine_gap_bounds=gap,
        geometry_certified=geometry_ok,
        ideal_initial_slope_bounds_by_model=slopes,
        ideal_increment_bounds_by_model=ideal,
        ideal_correlated_contrast_bounds=contrast_ideal,
        whole_window_form_norm_candidate=qmax,
        whole_window_phase_norm_candidate=pmax,
        finite_remainder_candidate=error,
        finite_remainder_upper_bound=error if warmed else None,
        true_increment_bounds_by_model=true,
        recorded_increment_bounds_by_model=recorded,
        recorded_contrast_bounds=contrast,
        response_margin=response_margin,
        response_certified=response,
        heat_warmup_exponent=heat_exponent,
        heat_warmup_form_norm_upper_bound=heat_form,
        heat_warmup_certified=heat_ok,
        phase_blind_recorded_contrast_upper_bound=heat_bound,
        phase_blind_exclusion_margin=heat_margin,
        heat_control_excluded=heat_excluded,
        nominal_work_bounds_by_model=nominal_work,
        work_error_candidate=work_error,
        work_bounds_candidates=work_candidates,
        work_bounds_by_model=works,
        work_allowance_margins=work_margins,
        positive_work_certified_by_model=positive_work,
        work_certified_by_model=work_flags,
        work_certified=all(work_flags),
        post_probe_relative_norm_squared_candidate=norm,
        post_probe_excess_storage_candidates=energies,
        post_probe_excess_storage_upper_bounds=energies if warmed else None,
        post_probe_radius_margin=radius_margin,
        capture_storage_margins=storage_margins,
        identity_certified_by_model=identity_flags,
        identity_certified=all(identity_flags),
        recovery_certified=all(identity_flags),
        status="certified_dipole" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
