"""Acquired mixed identity and response under two explicit loss normalizations.

Both detached laws retain the same bare sine exchange, reciprocal phase row,
unit physical capacities and structural clock. Only the form-loss denominator
changes. The strength choice uses the native EPI channel's normalization; it
does not import the native resultant-phase law. No trajectory is integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I, sin
from ._sine_mixed_capture import _derive_mixed_constitutive_pair
from ._sine_mixed_conductance import (
    _assess_sine_mixed_conductance,
    _contact_conductances,
    _mixed_conductance_field,
    _SineMixedConductance,
    _weighted_edges,
    _within_contact_box,
)
from .relational_observations import _ordered
from .support_transport import _laplacian

_NORMALIZATIONS = ("bare_degree", "transport_strength")
_SOURCE_LIMIT = Q(1, 10**10)
_REGISTRATION_LIMIT = Q(1, 10**6)
_FORM_READOUT_LIMIT = Q(1, 10**13)
_ETA = Q(20, 10**70)
_DELTA = Q(1, 1024)
_GAMMA_LOWER = Q(1, 3300)
_GAMMA_UPPER = Q(1, 3000)
_DEADLINE = Q(2 * 10**10)
_EARLY_PROFILE = (Q(1),) + (Q(5, 4),) * 8
_EARLY_TIME = Q(1, 32)
_EARLY_TARGET = Q(5, 10**13)


def _mixed_normalization_field(*, contact_conductances, loss_normalization):
    """Compile either complete law without changing capacity or phase transport.

    The callable consumes all 18 forms and then all 18 phase lifts, as shared
    exact/interval scalars or common-order Taylor jets. It detaches each input
    iterator once. The original field admits the complete state before the
    strength-only correction is computed from the same forms and weighted
    edges. No gradient is reconstructed from an evaluated phase velocity.
    """
    if not isinstance(loss_normalization, str):
        raise TypeError("loss_normalization must be a string")
    if loss_normalization not in _NORMALIZATIONS:
        raise ValueError("loss_normalization must be bare_degree or transport_strength")
    weights = _contact_conductances(contact_conductances)
    if not _within_contact_box(weights):
        raise ValueError("field requires every contact conductance in [3/4, 5/4]")
    degree_field = _mixed_conductance_field(contact_conductances=weights)
    if loss_normalization == "bare_degree":
        return degree_field

    geometry = _derive_mixed_constitutive_pair()
    edges = _weighted_edges(geometry, weights)
    # The shared Laplacian consumes directed incidence: retain both endpoints.
    directed = tuple(row for a, b, w in edges for row in ((a, b, w), (b, a, w)))
    size = len(geometry.nodes)
    strengths = tuple(
        sum((w for a, _, w in directed if a == i), Q(0)) for i in geometry.nodes
    )
    correction = tuple(Q(1, 3) - 1 / strength for strength in strengths)

    def flow(values):
        state = _ordered(values, "state", limit=2 * size + 1)
        rates = degree_field(state)
        forms = state[:size]
        if not isinstance(forms[0], Jet):
            forms = tuple(I.coerce(value) for value in forms)
        gradient = _laplacian(directed, forms)
        return (
            tuple(
                value + q * factor
                for value, q, factor in zip(rates[:size], gradient, correction)
            )
            + rates[size:]
        )

    return flow


@dataclass(frozen=True)
class _SineMixedNormalization:
    """Conditional same-source identity and a separately scoped early reading.

    All scalar bounds are candidates until their corresponding flags pass.
    The strength return concerns ``(P*x, theta-Theta)``, not zero full form.
    Its limiting common form follows from the separately conserved mixed
    inventory. The fresh degree-law report retains only its own conclusions.

    The nonzero early discriminator is supported only at the declared marked
    profile. Unit weights instead certify exact equality of the two complete
    laws on the same actual state; no independent source error is added there.
    Other box members can certify both identities while the early protocol is
    unsupported. Public dataclass construction authenticates no provenance.
    """

    contact_conductances: tuple[Q, ...]
    form_error_bound: Q
    phase_error_bound: Q
    registration_error_bound: Q
    form_readout_error_bound: Q
    bare_degree_reference: _SineMixedConductance
    strength_loss_mobilities: tuple[Q, ...]
    source_certified: bool
    isolated_waiting_certified: bool
    initial_identity_absent: bool
    initial_quotient_norm_squared_upper_bound: Q
    strength_post_attachment_excess_upper_bound: Q
    strength_radius_margin: Q
    strength_storage_margin: Q
    strength_metric_lower_bound: Q
    strength_initial_metric_upper_bound: Q
    strength_decay_rate: Q
    strength_recovery_exponent_lower_bound: Q
    strength_decay_upper_bound: Q
    strength_returned_quotient_norm_squared_upper_bound: Q
    strength_returned_registered_gap_squared_upper_bound: Q
    strength_joint_return_margin: Q
    strength_registration_margin: Q
    strength_common_form_limit_bounds: tuple[Q, Q]
    strength_constituent_identity_certified: bool
    strength_joint_identity_certified: bool
    both_joint_identities_certified: bool
    matched_attachment_work_certified: bool
    registration_actual_contrast_bounds: tuple[Q, Q]
    registration_recorded_contrast_bounds: tuple[Q, Q]
    registration_contrast_margin: Q
    registration_response_certified: bool
    early_profile_supported: bool
    exact_law_equivalence: bool
    early_forcing_bounds: I | None
    early_quadratic_coefficient: Q | None
    early_constant_force_remainder_coefficient: Q | None
    early_phase_feedback_remainder_coefficient: Q | None
    early_nominal_contrast_bounds: tuple[Q, Q] | None
    early_source_error_upper_bound: Q | None
    early_actual_contrast_bounds: tuple[Q, Q] | None
    early_recorded_contrast_bounds: tuple[Q, Q] | None
    early_contrast_margin: Q | None
    early_response_certified: bool
    early_null_certified: bool
    early_status: str
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_allowance_limit: Q = _SOURCE_LIMIT
    registration_allowance_limit: Q = _REGISTRATION_LIMIT
    form_readout_allowance_limit: Q = _FORM_READOUT_LIMIT
    waiting_radius: Q = _ETA
    relative_phase_gap: Q = _DELTA
    recovery_duration: Q = _DEADLINE
    registration_target: Q = _DELTA / 2
    early_contact_profile: tuple[Q, ...] = _EARLY_PROFILE
    early_time: Q = _EARLY_TIME
    early_response_target: Q = _EARLY_TARGET
    strength_metric_relative_cross_bound: Q = Q(1, 2000)
    strength_phase_storage_comparison: Q = Q(210, 11)
    strength_form_dissipation_lower_bound: Q = Q(9, 100)
    strength_decay_power: int = 14
    gamma_bounds: tuple[Q, Q] = (_GAMMA_LOWER, _GAMMA_UPPER)
    strength_inventory: str = "sum(x)+(3/gamma)*sum(B*(theta-Theta))"
    clock: str = "tau=(1023/1024)*t; unchanged in every form and phase row"
    scope: tuple[str, ...] = (
        "same_actual_acquired_source_and_contact_work_for_both_complete_laws",
        "only_form_loss_denominator_changes_from_bare_degree_to_transport_strength",
        "bare_sine_current_reciprocal_phase_row_and_unit_capacities_unchanged",
        "strength_EPI_normalization_does_not_import_native_resultant_phase_law",
        "strength_modified_energy_proved_on_quotient_not_transferred_common_metric",
        "nonzero_common_form_limit_retained_through_mixed_conserved_inventory",
        "both_late_registration_bounds_hold_from_same_fixed_deadline_onward",
        "early_form_discriminator_precedes_guaranteed_joint_registration",
        "fixed_profile_early_bound_has_live_phase_feedback_and_actual_source_errors",
        "unit_same_source_null_is_exact_without_artificial_independent_uncertainty",
        "no_trajectory_source_replay_event_execution_or_physical_sensor_claim",
    )


def _assess_sine_mixed_normalization(
    *,
    contact_conductances,
    form_error_bound,
    phase_error_bound,
    registration_error_bound,
    form_readout_error_bound,
):
    """Admit primitives before rebuilding either law's conditional conclusion.

    Nonnegative out-of-protocol inputs yield unavailable conclusions; malformed
    inputs raise before geometry, sources or interval functions. Registration
    and early form errors are independent supplied per-reading allowances.
    Their correlation is unrestricted; each two-reading contrast uses twice
    its own allowance. No observed response is an input to this certificate.
    """
    weights = _contact_conductances(contact_conductances)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    registration_noise = _exact(registration_error_bound, "registration_error_bound")
    form_noise = _exact(form_readout_error_bound, "form_readout_error_bound")
    if min(rx, rt, registration_noise, form_noise) < 0:
        raise ValueError("source and readout error bounds must be nonnegative")

    in_box = _within_contact_box(weights)
    source_limits = max(rx, rt) <= _SOURCE_LIMIT
    registration_limit = registration_noise <= _REGISTRATION_LIMIT
    form_limit = form_noise <= _FORM_READOUT_LIMIT
    reference = _assess_sine_mixed_conductance(
        contact_conductances=weights,
        form_error_bound=rx,
        phase_error_bound=rt,
        observation_error_bound=registration_noise,
    )
    geometry = _derive_mixed_constitutive_pair()
    geometry_ok = (
        reference.precontact_reference.geometry == geometry
        and geometry.support_certified
        and geometry.phase_alignment_certified
        and _GAMMA_LOWER <= geometry.gamma_bounds.lo
        and geometry.gamma_bounds.hi <= _GAMMA_UPPER
    )
    source_ok = source_limits and reference.source_certified
    waiting_ok = (
        source_ok
        and reference.isolated_waiting_certified
        and reference.waiting_radius == _ETA
    )
    edges = _weighted_edges(geometry, weights)
    strengths = tuple(
        sum((w for a, b, w in edges if i in (a, b)), Q(0)) for i in geometry.nodes
    )
    mobilities = tuple(1 / strength for strength in strengths)
    initial_absence = waiting_ok and _DELTA - 2 * _ETA > _DELTA / 2
    initial_norm = Q(9, 2) * _DELTA**2 + 4 * _ETA**2
    excess = Q(13, 4) * initial_norm
    radius_margin = Q(1, 12) ** 2 - initial_norm
    storage_margin = Q(1, 14400) - excess
    retained = (
        in_box
        and geometry_ok
        and waiting_ok
        and radius_margin > 0
        and storage_margin > 0
    )

    # W=E+gamma*v.y/3+y.B.y/2 on the actual quotient v=P*x. Its derivative
    # cancels the mixed damping exactly; B is not a fictitious capacity.
    lower = Q(5997, 40000)
    initial_metric = Q(7, 2) * initial_norm
    rate = 11 * _GAMMA_LOWER**2 / (1890 * (1 + Q(1, 2000)))
    exponent = rate * _DEADLINE
    # e>27/10 and (27/10)**10>2**14; no exponential evaluation or settling
    # trace is needed. The candidate requires the strict exponent gate.
    decay = Q(1, 2**14)
    returned = initial_metric * decay / lower
    return_margin = _DELTA**2 / 128 - returned
    registration_margin = (_DELTA / 8) ** 2 - 2 * returned
    strength_joint = (
        retained
        and initial_absence
        and exponent > 10
        and return_margin > 0
        and registration_margin > 0
    )
    both_joint = strength_joint and reference.joint_identity_certified
    # Nominal opposite phase origins cancel because each contact has equal
    # strength at its two endpoints. Fine source phases are not reset.
    common_form_limit = _ETA / (33 * _GAMMA_LOWER)
    registration_actual = (Q(7, 8) * _DELTA, Q(9, 8) * _DELTA)
    registration_recorded = (
        registration_actual[0] - 2 * registration_noise,
        registration_actual[1] + 2 * registration_noise,
    )
    registration_contrast_margin = registration_recorded[0] - _DELTA / 2
    registration_ok = (
        both_joint and registration_limit and registration_contrast_margin > 0
    )

    selected = weights == _EARLY_PROFILE
    equivalent = weights == (Q(1),) * 9
    forcing = quadratic = force_remainder = phase_remainder = None
    nominal = source_error = actual = recorded = early_margin = None
    if selected:
        forcing = geometry.gamma_bounds * sin(I(_DELTA)) / 3
        quadratic = Q(10, 351)
        force_remainder = Q(1565, 2808)
        phase_remainder = Q(26, 9) * _GAMMA_UPPER**2
        remainder = force_remainder + phase_remainder
        leading = quadratic * _EARLY_TIME**2
        allowance = remainder * _EARLY_TIME**3
        # The signed interval product retains validity even if a hypothetical
        # wider trig enclosure failed to resolve the positive leading term.
        nominal = (
            (forcing * (leading - allowance)).lo,
            (forcing * (leading + allowance)).hi,
        )
        source_error = 4 * _ETA
        actual = (nominal[0] - source_error, nominal[1] + source_error)
    elif equivalent:
        nominal = actual = (Q(0), Q(0))
        source_error = Q(0)
    if actual is not None:
        recorded = (actual[0] - 2 * form_noise, actual[1] + 2 * form_noise)
        early_margin = (
            recorded[0] - _EARLY_TARGET
            if selected
            else _EARLY_TARGET - max(abs(value) for value in recorded)
        )
    early_premises = in_box and geometry_ok and waiting_ok and form_limit
    early_ok = selected and early_premises and early_margin > 0
    null_ok = equivalent and early_premises and early_margin > 0
    early_status = (
        "certified_nonzero_response"
        if early_ok
        else (
            "certified_exact_same_source_null"
            if null_ok
            else (
                "unsupported_contact_profile"
                if not (selected or equivalent)
                else "unavailable"
            )
        )
    )
    certified = registration_ok and (early_ok or null_ok)
    reasons = tuple(
        label
        for passed, label in (
            (in_box, "contact_conductance_outside_fixed_positive_box"),
            (source_limits, "source_allowance_exceeds_fixed_protocol"),
            (source_ok, "original_mixed_acquired_sources_not_certified"),
            (waiting_ok, "actual_isolated_waiting_not_certified"),
            (geometry_ok, "mixed_geometry_not_certified"),
            (initial_absence, "initial_joint_registration_absence_not_certified"),
            (retained, "strength_quotient_trapping_not_certified"),
            (strength_joint, "strength_fixed_deadline_return_not_certified"),
            (reference.joint_identity_certified, "degree_law_return_not_certified"),
            (registration_limit, "registration_readout_allowance_exceeds_protocol"),
            (registration_contrast_margin > 0, "registration_contrast_not_resolved"),
            (form_limit, "form_readout_allowance_exceeds_protocol"),
            (selected or equivalent, "early_contact_profile_not_supported"),
            (
                not (selected or equivalent) or early_ok or null_ok,
                "early_form_observation_not_certified",
            ),
        )
        if not passed
    )
    status = (
        "certified_identity_and_nonzero_response"
        if certified and early_ok
        else (
            "certified_identity_and_exact_null"
            if certified
            else (
                "identity_certified_discriminator_unsupported"
                if registration_ok and not (selected or equivalent)
                else "unavailable"
            )
        )
    )
    return _SineMixedNormalization(
        contact_conductances=weights,
        form_error_bound=rx,
        phase_error_bound=rt,
        registration_error_bound=registration_noise,
        form_readout_error_bound=form_noise,
        bare_degree_reference=reference,
        strength_loss_mobilities=mobilities,
        source_certified=source_ok,
        isolated_waiting_certified=waiting_ok,
        initial_identity_absent=initial_absence,
        initial_quotient_norm_squared_upper_bound=initial_norm,
        strength_post_attachment_excess_upper_bound=excess,
        strength_radius_margin=radius_margin,
        strength_storage_margin=storage_margin,
        strength_metric_lower_bound=lower,
        strength_initial_metric_upper_bound=initial_metric,
        strength_decay_rate=rate,
        strength_recovery_exponent_lower_bound=exponent,
        strength_decay_upper_bound=decay,
        strength_returned_quotient_norm_squared_upper_bound=returned,
        strength_returned_registered_gap_squared_upper_bound=2 * returned,
        strength_joint_return_margin=return_margin,
        strength_registration_margin=registration_margin,
        strength_common_form_limit_bounds=(-common_form_limit, common_form_limit),
        strength_constituent_identity_certified=retained,
        strength_joint_identity_certified=strength_joint,
        both_joint_identities_certified=both_joint,
        matched_attachment_work_certified=in_box and geometry_ok and waiting_ok,
        registration_actual_contrast_bounds=registration_actual,
        registration_recorded_contrast_bounds=registration_recorded,
        registration_contrast_margin=registration_contrast_margin,
        registration_response_certified=registration_ok,
        early_profile_supported=selected or equivalent,
        exact_law_equivalence=equivalent,
        early_forcing_bounds=forcing,
        early_quadratic_coefficient=quadratic,
        early_constant_force_remainder_coefficient=force_remainder,
        early_phase_feedback_remainder_coefficient=phase_remainder,
        early_nominal_contrast_bounds=nominal,
        early_source_error_upper_bound=source_error,
        early_actual_contrast_bounds=actual,
        early_recorded_contrast_bounds=recorded,
        early_contrast_margin=early_margin,
        early_response_certified=early_ok,
        early_null_certified=null_ok,
        early_status=early_status,
        certified=certified,
        status=status,
        unresolved_conditions=reasons,
    )
