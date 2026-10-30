"""Own-law inheritance through two prescribed interfaces of organized C9s.

The proof owner is ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md. Exact
restriction to registered rigid twists differs from actual-source admission:
the latter rebuilds isolated recovery, new support work and full-state
identity. These detached calculations execute no event or trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sqrt
from ._sine_class_constitutive_comparison import _deformation, _edge_law
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
    _target_winding,
)
from .phase_response import _admit_relative_phasors
from .relational_observations import _ordered

_OFFSETS = {"matching": (0,), "symmetric": (-1, 0, 1)}
_INTERNAL = tuple(
    (9 * component + j, 9 * component + (j + 1) % 9)
    for component in (0, 1)
    for j in range(9)
)
_HORIZON = Q(1, 10)


def _interface(value):
    if type(value) is not str or value not in _OFFSETS:
        raise ValueError("interface must be 'matching' or 'symmetric'")
    return value


@dataclass(frozen=True)
class _ConstitutiveInheritance:
    """Exact support and enclosed coefficients of the organized restriction.

    The fine reciprocal coefficient is fixed. The symmetric interface needs
    an enlarged comparison family with both deformation and reciprocal
    coefficient variable; its regular form chart is not a passive sensor.
    Membership of the mapped deformation in [0,1] transfers no source proof.
    """

    interface: str
    deformation: Q
    offsets: tuple[int, ...]
    cross_edges: tuple[tuple[int, int], ...]
    edges: tuple[tuple[int, int], ...]
    degree: int
    cross_degree: int
    collective_clock_factor: Q
    alpha_cosine_bounds: I
    first_harmonic_multiplier_bounds: I
    form_chart_scale_bounds: I
    gamma_bounds: I
    effective_gamma_bounds: I
    effective_deformation_bounds: I
    effective_deformation_within_unit_interval: bool | None
    retains_fixed_exchange_parameter: bool
    same_current_profile_up_to_scale: bool
    nominal_attachment_work_bounds: I
    aligned_target_storage_bounds: I
    normalized_pair_storage_factor_bounds: I
    form_input_pushforward_bounds: I
    phase_input_pushforward: Q
    nodes: tuple[int, ...] = tuple(range(18))
    internal_edges: tuple[tuple[int, int], ...] = _INTERNAL
    target_phase_turns: tuple[Q, ...] = tuple(Q(j % 9 - 4, 9) for j in range(18))
    target_winding: int = 1
    fine_clock: str = "tau=(1023/1024)*t"
    collective_clock: str = "sigma=(cross_degree/degree)*tau"
    form_chart: str = "Z=X/sqrt(r); phase origins retain their registered circles"
    scope: tuple[str, ...] = (
        "supplied_registered_two_C9_partition_orientation_and_interface",
        "exact_invariant_family_not_a_generic_mean_only_quotient",
        "both_collective_rows_and_storage_transform_with_clock_and_form",
        "fine_capacity_held_at_one_not_inferred_from_degree_normalization",
        "matching_preserves_original_fixed_gamma_family",
        "symmetric_requires_mapped_gamma_as_well_as_deformation",
        "larger_algebraic_family_does_not_inherit_preparation_certificates",
        "global_family_invariance_does_not_imply_transverse_stability",
    )


def _derive_constitutive_inheritance(*, interface, deformation, target_winding=1):
    """Admit one of the two fixed interfaces and its nonnegative fine law."""
    winding = _target_winding(target_winding)
    name, coefficient = _interface(interface), _deformation(deformation)
    offsets = _OFFSETS[name]
    q, d = len(offsets), 2 + len(offsets)
    cross = tuple((j, 9 + (j + offset) % 9) for j in range(9) for offset in offsets)
    pi = pi_interval()
    cosine = cos(2 * winding * pi / 9)
    r = I(1) if name == "matching" else (1 + 2 * cosine) / 3
    scale = sqrt(r)
    gamma = Q(1, 1023) / pi
    mapped = I(coefficient) / r
    inside = True if coefficient <= r.lo else False if coefficient > r.hi else None
    work = 9 * q * (1 - r)
    return _ConstitutiveInheritance(
        interface=name,
        deformation=coefficient,
        offsets=offsets,
        cross_edges=cross,
        edges=_INTERNAL + cross,
        degree=d,
        cross_degree=q,
        collective_clock_factor=Q(q, d),
        alpha_cosine_bounds=cosine,
        first_harmonic_multiplier_bounds=r,
        form_chart_scale_bounds=scale,
        gamma_bounds=gamma,
        effective_gamma_bounds=gamma * scale,
        effective_deformation_bounds=mapped,
        effective_deformation_within_unit_interval=inside,
        retains_fixed_exchange_parameter=name == "matching",
        same_current_profile_up_to_scale=name == "matching" or coefficient == 0,
        nominal_attachment_work_bounds=work,
        aligned_target_storage_bounds=18 * (1 - cosine) + work,
        normalized_pair_storage_factor_bounds=9 * q * r,
        form_input_pushforward_bounds=1 / (Q(q, d) * scale),
        phase_input_pushforward=Q(d, q),
        target_phase_turns=tuple(Q(winding * (j % 9 - 4), 9) for j in range(18)),
        target_winding=winding,
    )


@dataclass(frozen=True)
class _InheritedPairObservation:
    """Static complete collective rows for supplied forms and relative phase.

    The phase phasor retains relative registration; no lifted mean or actual
    prepared state is inferred. Intervals enclose the fixed irrational
    geometric coefficients, not a numerical integration or a physical sensor.
    """

    law: _ConstitutiveInheritance
    form: tuple[Q, Q]
    relative_phase_phasor: tuple[Q, Q]
    averaged_current_bounds: I
    relative_potential_bounds: I
    form_rate_bounds: tuple[I, I]
    phase_rate_bounds: tuple[I, I]
    chart_form_bounds: tuple[I, I]
    chart_form_rate_bounds: tuple[I, I]
    chart_phase_rate_bounds: tuple[I, I]
    form_storage: Q
    full_storage_bounds: I
    loss: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I


def _observe_inherited_constitutive_pair(
    *, interface, deformation, form, relative_phase_phasor, target_winding=1
):
    """Compute both collective rows after all primitive coordinates admit."""
    winding = _target_winding(target_winding)
    name, coefficient = _interface(interface), _deformation(deformation)
    raw = _ordered(form, "form", limit=3)
    if len(raw) != 2:
        raise ValueError("form requires two registered component forms")
    values = tuple(_exact(value, f"form[{i}]") for i, value in enumerate(raw))
    phase = _admit_relative_phasors((relative_phase_phasor,), "relative_phase_phasor")[
        0
    ]
    law = _derive_constitutive_inheritance(
        interface=name, deformation=coefficient, target_winding=winding
    )
    r, gamma = law.first_harmonic_multiplier_bounds, law.gamma_bounds
    potential, jet = _edge_law(*phase, coefficient)
    current = jet[0] + (r - 1) * phase[1]
    relative_potential = potential + (r - 1) * (1 - phase[0])
    difference = values[1] - values[0]
    k, q, scale = (
        law.collective_clock_factor,
        law.cross_degree,
        law.form_chart_scale_bounds,
    )
    dx = k * (difference + gamma * current)
    dy = -k * gamma * difference
    form_storage = Q(9 * q, 2) * difference**2
    energy = (
        law.aligned_target_storage_bounds + form_storage + 9 * q * relative_potential
    )
    loss = 18 * q * k * difference**2
    rate_from_rows = -18 * q * difference * dx - 18 * q * current * dy
    return _InheritedPairObservation(
        law=law,
        form=values,
        relative_phase_phasor=phase,
        averaged_current_bounds=current,
        relative_potential_bounds=relative_potential,
        form_rate_bounds=(dx, -dx),
        phase_rate_bounds=(dy, -dy),
        chart_form_bounds=tuple(I(value) / scale for value in values),
        chart_form_rate_bounds=(dx / (k * scale), -dx / (k * scale)),
        chart_phase_rate_bounds=(dy / k, -dy / k),
        form_storage=form_storage,
        full_storage_bounds=energy,
        loss=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=rate_from_rows,
    )


@dataclass(frozen=True)
class _AcquiredConstitutiveInheritance:
    """Own-law actual-source attachment, identity and finite projection bounds.

    Tiny residual work is stored separately from the irrational nominal work.
    Failed premises leave conditional candidate arithmetic, never certified
    reached states. The interface is supplied with external work, not funded
    by a silently introduced reservoir or selected by an occurrence law.
    The entire-future radius uses the joint Euclidean form/phase norm; the
    finite raw-state radius uses the coordinate maximum norm. The chart-form
    error bounds one coordinate after the declared form-unit transformation.
    """

    law: _ConstitutiveInheritance
    recovery: _ConstitutiveRecovery
    attachment_work_excess_bounds: tuple[Q, Q]
    joined_excess_storage_upper_bound: Q
    initial_relative_norm_squared_upper_bound: Q
    phase_hessian_lower_bound: Q
    full_graph_gap_lower_bound: Q
    storage_coercivity_lower_bound: Q
    storage_barrier: Q
    radius_margin: Q
    storage_margin: Q
    entire_future_norm_upper_bound: Q
    finite_full_state_error_upper_bound: Q
    finite_chart_form_error_upper_bound: Q
    projected_form_row_defect_upper_bound: Q
    projected_phase_row_defect: Q
    identity_certified: bool
    finite_projection_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    finite_duration: Q = _HORIZON
    original_finite_duration: Q = _HORIZON * Q(1024, 1023)
    identity_radius: Q = Q(1, 12)
    conserved_form_mean: Q = Q(0)
    conserved_lifted_phase_mean: Q = Q(0)
    scope: tuple[str, ...] = (
        "two_original_class_one_sources_with_independent_zero_sum_errors",
        "fresh_isolated_recovery_not_old_three_component_contact",
        "state_unchanged_on_new_support_event_with_external_work",
        "nominal_work_plus_exact_nonnegative_subgrid_excess",
        "own_law_full_state_trapping_near_aligned_registered_target",
        "actual_source_not_reset_to_invariant_family",
        "finite_error_against_aligned_zero_mean_collective_reference",
        "actual_regional_means_have_exact_diffusion_and_phase_rows",
        "nonlinear_mean_form_row_has_bounded_retained_internal_defect",
        "no_trajectory_hierarchy_autonomous_contact_or_physical_binding_claim",
    )


def _assess_acquired_constitutive_inheritance(
    *, interface, deformation, form_error_bound, phase_error_bound
):
    """Rebuild the actual source before admitting its prescribed composition."""
    name, coefficient = _interface(interface), _deformation(deformation)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if rx < 0 or rt < 0:
        raise ValueError("source error bounds must be nonnegative")
    law = _derive_constitutive_inheritance(interface=name, deformation=coefficient)
    recovery = _assess_constitutive_recovery(
        deformation=coefficient, form_error_bound=rx, phase_error_bound=rt
    )
    eps = recovery.endpoint_radius
    u = recovery.source.phase_hessian_upper_bound
    q, d, k = law.cross_degree, law.degree, law.collective_clock_factor
    excess = 2 * d * (1 + u) * eps**2
    initial_norm = 4 * eps**2
    barrier, radius = Q(1, 1440), Q(1, 12)
    radius_margin, storage_margin = radius**2 - initial_norm, barrier - excess
    # These elementary strict bounds concern the new full 18-node support.
    curvature = (
        cos(2 * pi_interval() / 9 + sqrt(I(2)) * radius).lo - 2 * coefficient / 243
    )
    geometry_ok = (
        coefficient <= 1
        and curvature > Q(1, 2)
        and 2 - 2 * law.alpha_cosine_bounds.hi > Q(2, 5)
    )
    identity = (
        recovery.recovery_certified
        and geometry_ok
        and radius_margin > 0
        and storage_margin > 0
    )
    finite_error = eps / (1 - 3 * _HORIZON)
    row_defect = 4 * law.gamma_bounds.hi * k * u * finite_error
    future_radius = 15 * eps
    finite_ok = (
        identity
        and 2 + 2 * law.gamma_bounds.hi * u < 3
        and 3 * _HORIZON < 1
        and 10 * excess < future_radius**2
    )
    reasons = []
    if not recovery.recovery_certified:
        reasons.append("isolated_source_recovery_not_certified")
    if not geometry_ok:
        reasons.append("new_support_curvature_or_gap_not_certified")
    if radius_margin <= 0 or storage_margin <= 0:
        reasons.append("new_support_identity_margin_not_certified")
    if not finite_ok:
        reasons.append("finite_actual_state_projection_not_certified")
    return _AcquiredConstitutiveInheritance(
        law=law,
        recovery=recovery,
        attachment_work_excess_bounds=(Q(0), 2 * q * (1 + u) * eps**2),
        joined_excess_storage_upper_bound=excess,
        initial_relative_norm_squared_upper_bound=initial_norm,
        phase_hessian_lower_bound=Q(1, 2),
        full_graph_gap_lower_bound=Q(2, 5),
        storage_coercivity_lower_bound=Q(1, 10),
        storage_barrier=barrier,
        radius_margin=radius_margin,
        storage_margin=storage_margin,
        entire_future_norm_upper_bound=future_radius,
        finite_full_state_error_upper_bound=finite_error,
        finite_chart_form_error_upper_bound=finite_error
        / law.form_chart_scale_bounds.lo,
        projected_form_row_defect_upper_bound=row_defect,
        projected_phase_row_defect=Q(0),
        identity_certified=identity,
        finite_projection_certified=finite_ok,
        status=(
            "certified_acquired_constitutive_inheritance"
            if finite_ok
            else "unavailable"
        ),
        unresolved_conditions=tuple(reasons),
    )
