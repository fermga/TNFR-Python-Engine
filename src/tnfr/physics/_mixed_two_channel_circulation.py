"""Conditional finite phase discrimination through two supplied form channels.

The sources are actual conditional UM images with an additional, independently
associated maturity premise. A balanced regional phase jump is supplied; no
phase clamp, source endpoint, preparation deadline or runtime event is installed.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I
from ._mixed_phase_feedback import _mixed_without_form_feedback_field
from ._mixed_reciprocal_partner import (
    _mixed_reciprocal_partner_field,
    _nonnegative,
    _nonnegative_pair,
)
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._sine_coupling_successor import (
    _assess_sine_coupling_successor,
    _CouplingSuccessorGeometry,
    _SineCouplingSuccessor,
)
from .relational_observations import _ordered

_SOURCE_CAP = Q(1, 10**10)
_MATURE_CAP = _JUMP_ERROR_CAP = Q(1, 10**8)
_READOUT_CAP = Q(1, 10**6)
_DELTA = Q(1, 1024)
_REGISTRATION = _DELTA / 4
_RADIUS = Q(1, 32)
_GAMMA_LOWER, _GAMMA_UPPER = Q(1, 3300), Q(1, 3000)
_WEIGHT_DEVIATION = Q(20 * 10**9, 10**70)
_PREFIX_TIME = Q(1, 32)
_TIME = Q(512)
_AMPLITUDE = Q(1, 8192)
_NORM = Q(1, 1024)
_COSINE = Q(1, 10)
_FORM_GAP = Q(1, 3)
_FORCING_TARGET = Q(1, 250000000)
_THRESHOLD = Q(1, 400000)
_CONTACT_PAIRS = tuple((j, 18 + j) for j in range(18))


def _mixed_two_channel_field(
    *, first_new_edge_conductances, second_new_edge_conductances, law
):
    """Add the B channel to the shared complete actual or comparator field.

    The 72 coordinates remain 36 forms followed by 36 phase lifts. Both
    channels are external form inputs: all internal degrees and phase rows
    remain unchanged. The phase jump is a separate supplied preparation,
    not performed by this evaluator. Legal weights do not authenticate UM
    association. Every state entry is normalized before any field arithmetic.
    """
    if not isinstance(law, str):
        raise TypeError("law must be a string")
    if law not in ("full", "without_form_feedback"):
        raise ValueError("law must be 'full' or 'without_form_feedback'")
    builder = (
        _mixed_reciprocal_partner_field
        if law == "full"
        else _mixed_without_form_feedback_field
    )
    field = builder(
        first_new_edge_conductances=first_new_edge_conductances,
        second_new_edge_conductances=second_new_edge_conductances,
    )

    def flow(values):
        state = _ordered(values, "state", limit=73)
        if len(state) != 72:
            raise ValueError("state requires all 36 forms followed by all 36 phases")
        jets = tuple(isinstance(value, Jet) for value in state)
        if any(jets):
            if not all(jets) or any(value.order != state[0].order for value in state):
                raise ValueError("state Taylor jets must all have the same order")
        else:
            state = tuple(I.coerce(value) for value in state)
        rates = field(state)
        exchange = tuple(state[18 + j] - state[j] for j in range(9, 18))
        first = rates[:9] + tuple(rates[9 + j] + exchange[j] for j in range(9))
        second = rates[18:27] + tuple(rates[27 + j] - exchange[j] for j in range(9))
        return first + second + rates[36:]

    return flow


@dataclass(frozen=True)
class _MixedTwoChannelCirculation:
    """A finite theorem conditional on mature sources and a supplied actuator.

    Positive mature bounds have finite-entry existence under the original
    degree-loss source theorem. Association of those bounds with the sources
    actually used remains independent: no age or state is computed here.
    The leak's possible common form is retained. Both hypothesis record sets
    allow independently selected preparations and jump errors; the classifier
    margins do not merely compare identical nominal preparations.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    mature_norm_upper_bounds: tuple[Q, Q]
    phase_jump_error_bound: Q
    readout_error_bound: Q
    sources: tuple[_SineCouplingSuccessor, ...]
    source_geometry: _CouplingSuccessorGeometry
    cross_incidence: tuple[tuple[int, ...], ...]
    source_certified: tuple[bool, bool]
    finite_mature_ages_exist: tuple[bool, bool]
    mature_norm_domain_admitted: tuple[bool, bool]
    conditional_mature_sources: tuple[bool, bool]
    source_association_certified: bool
    prefix_norm_upper_bound: Q
    prefix_growth_margin: Q
    norm_growth_exponent: Q
    norm_growth_upper_bound: Q
    calculated_joint_norm_upper_bound: Q
    joint_norm_upper_bound: Q
    norm_margin: Q
    form_generator_gap_lower_bound: Q
    marked_gap_uncertainty_upper_bound: Q
    marked_gap_lower_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    signed_gap_precision_margin: Q
    finite_identity_certified: bool
    phase_intervention_certified: bool
    paired_form_difference_upper_bound: Q
    signed_cross_current_lower_bound: Q
    regional_weight_residual_upper_bound: Q
    differential_forcing_lower_bound: Q
    forcing_margin: Q
    paired_transfer_gap_lower_bound: Q
    comparator_transfer_absolute_upper_bound: Q
    total_transfer_absolute_upper_bound: Q
    returning_channel_margin: Q
    circulation_certified: bool
    comparator_record_bounds: tuple[Q, Q]
    full_record_lower_bound: Q
    record_separation_margin: Q
    full_threshold_margin: Q
    comparator_threshold_margin: Q
    independent_hypotheses_separated: bool
    prefix_storage_work_upper_bound: Q
    phase_jump_work_upper_bound: Q
    contact_storage_work_upper_bound: Q
    jump_work_margin: Q
    work_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_error_cap: Q = _SOURCE_CAP
    mature_norm_limit: Q = _MATURE_CAP
    phase_jump_error_cap: Q = _JUMP_ERROR_CAP
    readout_error_cap: Q = _READOUT_CAP
    charge_duration: Q = _PREFIX_TIME
    charge_amplitude: Q = Q(0)
    prefix_leak_duration: Q = _PREFIX_TIME
    contact_duration: Q = _TIME
    phase_jump_amplitude: Q = _AMPLITUDE
    phase_radius: Q = _RADIUS
    registration_gap_bound: Q = _REGISTRATION
    gamma_bounds: tuple[Q, Q] = (_GAMMA_LOWER, _GAMMA_UPPER)
    cosine_lower_bound: Q = _COSINE
    edge_weight_deviation_upper_bound: Q = _WEIGHT_DEVIATION
    forcing_lower_target: Q = _FORCING_TARGET
    classifier_threshold: Q = _THRESHOLD
    full_state_dimension: int = 72
    contact_pairs: tuple[tuple[int, int], ...] = _CONTACT_PAIRS
    independently_observed_channels: int = 1
    maturity_is_independent_condition: bool = True
    maturity_authenticated: bool = False
    preparation_age_evaluated: bool = False
    comparator_own_law_source_acquisition_claimed: bool = False
    phase_actuator_spatial_uniformity_assumed: bool = True
    paired_total_transfer_difference: Q = Q(0)
    output_orientation: str = "A_channel_outgoing_from_first_copy_to_second"
    clock: str = "tau=(1023/1024)*t; both complete rows and supplied inputs use tau"
    phase_jump_work: str = "U(theta+zeta)-U(theta), bare potential, forms unchanged"
    contact_storage_work: str = "-(q1-q2) dot (x1-x2), not assumed nonpositive"
    scope: tuple[str, ...] = (
        "new_positive_full_norm_maturity_not_old_gap_readiness_or_a_reset",
        "actual_sources_zero_charge_and_leak_carry_common_form_and_phase_gauges",
        "balanced_regional_jump_with_independent_amplitude_error_no_phase_clamp",
        "two_external_form_channels_keep_internal_support_degree_and_phase_rows",
        "conditional_finite_identity_under_both_complete_post_handoff_laws",
        "A_channel_integral_is_not_inferred_from_total_form_or_model_pressure",
        "uniform_separation_for_independently_allowed_hypothesis_preparations",
        "no_runtime_authentication_operator_occurrence_outer_NFR_or_physical_claim",
    )


def _assess_mixed_two_channel_circulation(
    *,
    form_error_bounds,
    phase_error_bounds,
    mature_norm_upper_bounds,
    phase_jump_error_bound,
    readout_error_bound,
):
    """Rebuild the source and admit the frozen finite circulation protocol."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    mature = _nonnegative_pair(mature_norm_upper_bounds, "mature_norm_upper_bounds")
    jump_error = _nonnegative(phase_jump_error_bound, "phase_jump_error_bound")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    sources = tuple(
        _assess_sine_coupling_successor(form_error_bound=rx, phase_error_bound=rt)
        for rx, rt in zip(forms, phases, strict=True)
    )
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    geometry_ok = (
        registered.cross_symmetry_certified
        and geometry.completion == "common_sense"
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.components == (tuple(range(18)),)
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.weighted_gap_lower_bound >= 3
    )
    source_ok = tuple(
        max(rx, rt) <= _SOURCE_CAP
        and source.form_error_bound == rx
        and source.phase_error_bound == rt
        and source.source_certified
        and source.isolated_waiting_certified
        and source.geometry_certified
        and source.conditional_image_certified
        and source.phase_radius == _RADIUS
        and source.relative_phase_gap == _DELTA
        and source.phase_cosine_lower_bound >= _COSINE
        and source.phase_curvature_bounds.lo >= _COSINE
        and source.phase_curvature_margin > 0
        and _GAMMA_LOWER
        <= source.gamma_range[0]
        <= source.gamma_range[1]
        <= _GAMMA_UPPER
        and source.new_weight_range == (Q(3, 4), Q(1))
        and 0 <= source.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
        and source.precontact_reference.initial_form_means == (Q(0), Q(0))
        and source.completion_reports[0].geometry == geometry
        and source.completion_reports[0].constituent_identity_certified
        and source.completion_reports[0].joint_registration_certified
        and source.completion_reports[0].degree_form_limit_bounds == ((Q(0), Q(0)),)
        and geometry_ok
        for rx, rt, source in zip(forms, phases, sources, strict=True)
    )
    exists = tuple(
        good
        and radius > 0
        and source.completion_reports[0].asymptotic_convergence_certified
        for good, radius, source in zip(source_ok, mature, sources, strict=True)
    )
    maturity_ok = tuple(0 < radius <= _MATURE_CAP for radius in mature)
    ready = tuple(good and admitted for good, admitted in zip(exists, maturity_ok))
    source_association = all(ready)

    # sqrt(2)*exp(2*g*(L+h)) < 3/2: square a rational geometric majorant.
    prefix_exponent = 4 * _GAMMA_UPPER * _PREFIX_TIME
    prefix_growth_margin = Q(9, 4) - 2 / (1 - prefix_exponent) ** 2
    prefix = Q(3, 2) * max(mature)
    exponent = 2 * _GAMMA_UPPER * _TIME
    growth = 1 / (1 - exponent)
    norm = (prefix + 3 * (_AMPLITUDE + jump_error)) * growth
    norm_margin = _NORM - norm
    gap = min(geometry.weighted_gap_lower_bound / 9, Q(2))
    uncertainty = Q(3, 2) * (
        (1 + 6 * _GAMMA_UPPER) * prefix + 12 * _GAMMA_UPPER**2 * _NORM * _TIME
    )
    lower_gap = _AMPLITUDE - jump_error - uncertainty
    upper_gap = _AMPLITUDE + jump_error + uncertainty
    registration_margin = _REGISTRATION - upper_gap
    signed_precision = _AMPLITUDE / 100 - jump_error - uncertainty
    jump_ok = jump_error <= _JUMP_ERROR_CAP
    identity = (
        source_association
        and jump_ok
        and prefix_growth_margin > 0
        and 0 <= exponent < 1
        and norm_margin > 0
        and _NORM < _RADIUS
        and gap >= _FORM_GAP
        and lower_gap > 0
        and registration_margin > 0
        and signed_precision > 0
    )
    paired_form = 6 * _GAMMA_UPPER * _NORM
    current = 45 * _COSINE * lower_gap
    weight_residual = Q(10, 9) * _WEIGHT_DEVIATION * paired_form
    forcing = 2 * _GAMMA_LOWER * current / 81 - 2 * weight_residual
    forcing_margin = forcing - _FORCING_TARGET
    signal = (3 * _TIME - 1) * _FORCING_TARGET
    comparator = Q(51, 4) * prefix
    total = 3 * prefix
    returning_margin = signal - comparator - total
    circulation = identity and forcing_margin > 0 and returning_margin > 0
    record_bounds = (-comparator - readout, comparator + readout)
    full_lower = signal - comparator - readout
    separation_margin = full_lower - record_bounds[1]
    full_threshold = full_lower - _THRESHOLD
    comparator_threshold = _THRESHOLD - record_bounds[1]
    separated = (
        circulation
        and readout <= _READOUT_CAP
        and separation_margin > 0
        and min(full_threshold, comparator_threshold) > 0
    )
    prefix_work = 18 * _PREFIX_TIME * prefix**2
    jump_work = (
        54 * prefix * (_AMPLITUDE + jump_error) + 81 * (_AMPLITUDE + jump_error) ** 2
    )
    contact_work = 36 * _TIME * _NORM**2
    jump_work_margin = Q(2, 10**6) - jump_work
    work_ok = identity and jump_work_margin > 0
    unresolved = tuple(
        label
        for good, label in (
            (
                all(source_ok),
                "actual_source_or_original_weight_association_unavailable",
            ),
            (all(exists), "positive_mature_entry_existence_unavailable"),
            (all(maturity_ok), "independently_associated_maturity_outside_protocol"),
            (jump_ok, "phase_jump_error_outside_protocol"),
            (identity, "conditional_finite_identity_unavailable"),
            (circulation, "signed_two_channel_circulation_unavailable"),
            (readout <= _READOUT_CAP, "channel_record_allowance_outside_protocol"),
            (separated, "independent_hypothesis_record_sets_not_separated"),
            (work_ok, "supplied_work_bound_unavailable"),
        )
        if not good
    )
    return _MixedTwoChannelCirculation(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        mature_norm_upper_bounds=mature,
        phase_jump_error_bound=jump_error,
        readout_error_bound=readout,
        sources=sources,
        source_geometry=geometry,
        cross_incidence=registered.cross_incidence,
        source_certified=source_ok,
        finite_mature_ages_exist=exists,
        mature_norm_domain_admitted=maturity_ok,
        conditional_mature_sources=ready,
        source_association_certified=source_association,
        prefix_norm_upper_bound=prefix,
        prefix_growth_margin=prefix_growth_margin,
        norm_growth_exponent=exponent,
        norm_growth_upper_bound=growth,
        calculated_joint_norm_upper_bound=norm,
        joint_norm_upper_bound=_NORM,
        norm_margin=norm_margin,
        form_generator_gap_lower_bound=gap,
        marked_gap_uncertainty_upper_bound=uncertainty,
        marked_gap_lower_bound=lower_gap,
        marked_gap_upper_bound=upper_gap,
        registration_margin=registration_margin,
        signed_gap_precision_margin=signed_precision,
        finite_identity_certified=identity,
        phase_intervention_certified=identity,
        paired_form_difference_upper_bound=paired_form,
        signed_cross_current_lower_bound=current,
        regional_weight_residual_upper_bound=weight_residual,
        differential_forcing_lower_bound=forcing,
        forcing_margin=forcing_margin,
        paired_transfer_gap_lower_bound=signal,
        comparator_transfer_absolute_upper_bound=comparator,
        total_transfer_absolute_upper_bound=total,
        returning_channel_margin=returning_margin,
        circulation_certified=circulation,
        comparator_record_bounds=record_bounds,
        full_record_lower_bound=full_lower,
        record_separation_margin=separation_margin,
        full_threshold_margin=full_threshold,
        comparator_threshold_margin=comparator_threshold,
        independent_hypotheses_separated=separated,
        prefix_storage_work_upper_bound=prefix_work,
        phase_jump_work_upper_bound=jump_work,
        contact_storage_work_upper_bound=contact_work,
        jump_work_margin=jump_work_margin,
        work_certified=work_ok,
        certified=not unresolved,
        status=(
            "conditional_circulation_discriminator" if not unresolved else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
