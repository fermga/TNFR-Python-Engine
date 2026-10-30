"""Carry regional information through a finite reciprocal partner interaction.

Two actual conditional UM images retain their internal complete laws. The
supplied form-only exchange is a joint input, not new graph edges. Original
pre-leak readings initialize the reference; no contact-entry state is observed
or reset. Source and observation association are independent premises.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._comparison_flow import (
    MetzlerExponentialBounds,
    metzler_exponential_bounds,
)
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I
from ._mixed_regional_boundary import (
    _assess_mixed_regional_boundary,
    _MixedRegionalBoundary,
    _regional_reference,
    _RegionalBoundaryReference,
)
from ._sine_coupling_successor import _coupling_successor_field
from ._sine_preparation import _prepared_coordinate_vector as _vector
from .relational_observations import _ordered

_TIME = Q(1, 32)
_AMPLITUDE = Q(1, 16)
_DELTA = Q(1, 1024)
_RADIUS = Q(1, 32)
_GAMMA_UPPER = Q(1, 3000)
_PREFIX_NORM = Q(9, 512)
_JOINT_NORM = Q(1, 40)
_MEAN_RADIUS = Q(3, 512)
_WEIGHT_DEVIATION = Q(20 * 10**9, 10**70)
_RESIDUAL = Q(1, 10**5)
_SOURCE_CAP = Q(1, 10**10)
_MEAN_ERROR_CAP = _READOUT_ERROR_CAP = Q(1, 10**6)
_ARITHMETIC_BUDGET = Q(1, 10**35)
_PREDICTION_TARGET = Q(2, 10**6)
_SEPARATION_TARGET = Q(9, 10**4)


def _nonnegative(value, label):
    value = _exact(value, label)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _nonnegative_pair(values, label):
    values = _vector(values, label, 2)
    if min(values) < 0:
        raise ValueError(f"{label} must be nonnegative")
    return values


def _scale(bounds, value):
    endpoints = tuple(value * endpoint for endpoint in bounds)
    return min(endpoints), max(endpoints)


def _product(left, right):
    endpoints = tuple(a * b for a in left for b in right)
    return min(endpoints), max(endpoints)


def _sum(bounds):
    bounds = tuple(bounds)
    return sum((row[0] for row in bounds), Q(0)), sum((row[1] for row in bounds), Q(0))


def _midpoint(bounds):
    return (bounds[0] + bounds[1]) / 2


def _radius(bounds):
    return (bounds[1] - bounds[0]) / 2


@dataclass(frozen=True)
class _ReciprocalPartnerReference:
    """Exact rational enclosures for the carried linear reference only.

    The signed outgoing row uses first-copy form before minus after contact.
    Its composition with both prefix columns retains common form created by
    leakage. All fine dynamics remain in the separately admitted residual.
    """

    prefix: _RegionalBoundaryReference
    generator: tuple[tuple[Q, ...], ...]
    exponential: MetzlerExponentialBounds
    prefix_difference_bounds: tuple[tuple[Q, Q], ...]
    outgoing_row_bounds: tuple[tuple[Q, Q], ...]
    response_coefficient_bounds: tuple[tuple[Q, Q], ...]
    contact_residual_calculated_upper_bound: Q
    residual_upper_bound: Q
    reference_certified: bool


def _reciprocal_reference():
    prefix = _regional_reference()
    k = prefix.reference_coupling
    matrix = (
        (-k - 1, k, Q(1), Q(0)),
        (k, -k, Q(0), Q(0)),
        (Q(1), Q(0), -k - 1, k),
        (Q(0), Q(0), k, -k),
    )
    exponential = metzler_exponential_bounds(matrix, _TIME)
    old = prefix.exponential
    columns = tuple(
        (old.lower[i][0] - old.upper[i][1], old.upper[i][0] - old.lower[i][1])
        for i in range(2)
    )
    outgoing = tuple(
        (
            9 * (int(j < 2) - exponential.upper[0][j] - exponential.upper[1][j]),
            9 * (int(j < 2) - exponential.lower[0][j] - exponential.lower[1][j]),
        )
        for j in range(4)
    )
    coefficients = tuple(
        _sum(_product(outgoing[2 * copy + i], columns[i]) for i in range(2))
        for copy in range(2)
    )
    residual = Q(10, 9) * _JOINT_NORM * (_GAMMA_UPPER + _WEIGHT_DEVIATION)
    residual += Q(2, 3) * _JOINT_NORM * prefix.coupling_radius
    return _ReciprocalPartnerReference(
        prefix=prefix,
        generator=matrix,
        exponential=exponential,
        prefix_difference_bounds=columns,
        outgoing_row_bounds=outgoing,
        response_coefficient_bounds=coefficients,
        contact_residual_calculated_upper_bound=residual,
        residual_upper_bound=_RESIDUAL,
        reference_certified=(
            prefix.coefficient_domain_certified
            and prefix.residual_certified
            and prefix.gain_certified
            and 0 < k < Q(1, 2)
            and residual < _RESIDUAL
        ),
    )


def _mixed_reciprocal_partner_field(
    *, first_new_edge_conductances, second_new_edge_conductances
):
    """Compile all 72 rows with nine equal/opposite matched form inputs.

    State order is 36 forms (first copy then second), followed by their 36
    phase lifts. The internal support, bare degree9, capacity and both phase
    rows are unchanged. Legal weight vectors are not authenticated UM images.
    All primitive weights and all state entries are admitted before field work.
    """
    weights = tuple(
        _vector(values, label, 63)
        for values, label in (
            (first_new_edge_conductances, "first_new_edge_conductances"),
            (second_new_edge_conductances, "second_new_edge_conductances"),
        )
    )
    if any(not Q(3, 4) <= value <= 1 for row in weights for value in row):
        raise ValueError("new edge conductances must lie in [3/4,1]")
    fields = tuple(
        _coupling_successor_field(
            completion="common_sense",
            new_edge_conductances=row,
            loss_normalization="bare_degree",
        )
        for row in weights
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
        first = fields[0](state[:18] + state[36:54])
        second = fields[1](state[18:36] + state[54:])
        exchange = tuple(state[18 + j] - state[j] for j in range(9))
        first_form = tuple(first[j] + exchange[j] for j in range(9)) + first[9:18]
        second_form = tuple(second[j] - exchange[j] for j in range(9)) + second[9:18]
        return first_form + second_form + first[18:] + second[18:]

    return flow


@dataclass(frozen=True)
class _ReciprocalTransferPrediction:
    """A forecast from original pre-leak readings, without contact-entry data.

    Four carried intervals are genuine prefix predictions, not new readings.
    Numerical prefix/response widths are enclosed with rational endpoints;
    the final arithmetic allowance prices their complete composition once.
    """

    regional_mean_readings: tuple[Q, Q]
    regional_mean_error_bounds: tuple[Q, Q]
    readout_error_bound: Q
    regional_mean_reading_bounds: tuple[tuple[Q, Q], ...]
    mean_observation_consistent: tuple[bool, bool]
    reference: _ReciprocalPartnerReference
    carried_reference_bounds: tuple[tuple[Q, Q], ...]
    carried_mean_centers: tuple[Q, ...]
    carried_state_error_bounds: tuple[Q, ...]
    carried_mean_bounds: tuple[tuple[Q, Q], ...]
    contact_reference_bounds: tuple[tuple[Q, Q], ...]
    contact_mean_bounds: tuple[tuple[Q, Q], ...]
    reference_transfer_bounds: tuple[Q, Q]
    forecast: Q
    inherited_error_upper_bound: Q
    contact_error_upper_bound: Q
    arithmetic_error_upper_bound: Q
    record_prediction_error_upper_bound: Q
    true_transfer_bounds: tuple[Q, Q]
    recorded_transfer_bounds: tuple[Q, Q]
    allowances_admitted: bool
    arithmetic_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    prefix_leak_duration: Q = _TIME
    contact_duration: Q = _TIME
    regional_mean_error_cap: Q = _MEAN_ERROR_CAP
    readout_error_cap: Q = _READOUT_ERROR_CAP
    arithmetic_budget: Q = _ARITHMETIC_BUDGET
    prediction_target: Q = _PREDICTION_TARGET
    preparation_mean_bounds: tuple[Q, Q] = (-_MEAN_RADIUS, _MEAN_RADIUS)
    regional_order: tuple[str, ...] = ("A1", "B1", "A2", "B2")
    output_orientation: str = "outgoing_from_first_copy_to_second"
    decoder_inputs: tuple[str, ...] = (
        "two_associated_original_pre_leak_regional_mean_readings_and_errors",
        "independent_future_contact_transfer_readout_error_bound",
        "fixed_complete_laws_registered_sources_and_prefix_contact_metadata",
    )


def _predict_mixed_reciprocal_transfer(
    *, regional_mean_readings, regional_mean_error_bounds, readout_error_bound
):
    """Carry both initialized states and errors through leak and mutual contact."""
    means = _vector(regional_mean_readings, "regional_mean_readings", 2)
    errors = _nonnegative_pair(regional_mean_error_bounds, "regional_mean_error_bounds")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    reading_bounds = tuple((m - e, m + e) for m, e in zip(means, errors, strict=True))
    consistent = tuple(
        lo <= _MEAN_RADIUS and hi >= -_MEAN_RADIUS for lo, hi in reading_bounds
    )
    allowances = max(errors) <= _MEAN_ERROR_CAP and readout <= _READOUT_ERROR_CAP
    reference = _reciprocal_reference()
    carried = tuple(
        _scale(coefficient, mean)
        for mean in means
        for coefficient in reference.prefix_difference_bounds
    )
    centers = tuple(_midpoint(bounds) for bounds in carried)
    carried_errors = tuple(
        errors[i // 2] + _TIME * _RESIDUAL + _radius(bounds)
        for i, bounds in enumerate(carried)
    )
    carried_actual = tuple(
        (center - error, center + error)
        for center, error in zip(centers, carried_errors, strict=True)
    )
    exponential = reference.exponential
    contact = tuple(
        _sum(
            _product((exponential.lower[i][j], exponential.upper[i][j]), carried[j])
            for j in range(4)
        )
        for i in range(4)
    )
    state_error = max(errors) + 2 * _TIME * _RESIDUAL
    contact_actual = tuple((lo - state_error, hi + state_error) for lo, hi in contact)
    transfer = _sum(
        _scale(coefficient, mean)
        for coefficient, mean in zip(
            reference.response_coefficient_bounds, means, strict=True
        )
    )
    forecast = _midpoint(transfer)
    arithmetic = _radius(transfer)
    inherited = 18 * _TIME * (max(errors) + _TIME * _RESIDUAL)
    contact_error = 9 * _TIME**2 * _RESIDUAL
    nonnumerical = inherited + contact_error
    true_bounds = (transfer[0] - nonnumerical, transfer[1] + nonnumerical)
    record_bounds = (true_bounds[0] - readout, true_bounds[1] + readout)
    error = nonnumerical + readout + arithmetic
    arithmetic_ok = arithmetic <= _ARITHMETIC_BUDGET
    unresolved = tuple(
        label
        for passed, label in (
            (allowances, "observation_allowance_outside_protocol"),
            (all(consistent), "original_regional_mean_inconsistent_with_preparation"),
            (reference.reference_certified, "reference_or_residual_unavailable"),
            (arithmetic_ok, "arithmetic_allowance_unavailable"),
            (error < _PREDICTION_TARGET, "record_prediction_target_unavailable"),
        )
        if not passed
    )
    return _ReciprocalTransferPrediction(
        regional_mean_readings=means,
        regional_mean_error_bounds=errors,
        readout_error_bound=readout,
        regional_mean_reading_bounds=reading_bounds,
        mean_observation_consistent=consistent,
        reference=reference,
        carried_reference_bounds=carried,
        carried_mean_centers=centers,
        carried_state_error_bounds=carried_errors,
        carried_mean_bounds=carried_actual,
        contact_reference_bounds=contact,
        contact_mean_bounds=contact_actual,
        reference_transfer_bounds=transfer,
        forecast=forecast,
        inherited_error_upper_bound=inherited,
        contact_error_upper_bound=contact_error,
        arithmetic_error_upper_bound=arithmetic,
        record_prediction_error_upper_bound=error,
        true_transfer_bounds=true_bounds,
        recorded_transfer_bounds=record_bounds,
        allowances_admitted=allowances,
        arithmetic_certified=arithmetic_ok,
        certified=not unresolved,
        status="certified" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _MixedReciprocalPartner:
    """Two source-inclusive finite continuations, not a newly acquired joint NFR.

    Source reports are freshly rebuilt from the original primitive allowances.
    Only their connected-source, prefix and readiness premises are consumed;
    earlier response verdicts and measured transfers supply no new premises.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    starting_gap_upper_bounds: tuple[Q, Q]
    readout_error_bound: Q
    prefix_reports: tuple[_MixedRegionalBoundary, ...]
    reference: _ReciprocalPartnerReference
    source_certified: tuple[bool, bool]
    finite_protocol_ready_ages_exist: tuple[bool, bool]
    conditional_ready_sources: tuple[bool, bool]
    joint_norm_upper_bound: Q
    joint_norm_margin: Q
    full_state_tube_certified: bool
    marked_gap_change_upper_bound: Q
    retained_gap_upper_bounds: tuple[Q, Q]
    registration_margins: tuple[Q, Q]
    constituent_identity_retained_certified: bool
    contact_storage_work_upper_bound: Q
    inherited_error_upper_bound: Q
    contact_error_upper_bound: Q
    uniform_arithmetic_error_upper_bound: Q
    record_prediction_error_upper_bound: Q
    prediction_margin: Q
    partner_history_leading_gap: Q
    partner_history_model_loss_upper_bound: Q
    actual_partner_history_gap_lower_bound: Q
    recorded_partner_history_gap_lower_bound: Q
    weak_prediction_error_lower_bound: Q
    separation_margin: Q
    positive_information_certified: bool
    partner_history_obstruction_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    charge_duration: Q = _TIME
    prefix_leak_duration: Q = _TIME
    contact_duration: Q = _TIME
    source_count: int = 2
    full_state_dimension: int = 72
    reference_state_dimension: int = 4
    original_regional_readings: int = 2
    new_contact_entry_readings: int = 0
    contact_rate: Q = Q(1)
    contact_pairs: tuple[tuple[int, int], ...] = tuple((j, 18 + j) for j in range(9))
    supplied_signed_charge_amount: Q = Q(0)
    supplied_absolute_charge_amount_upper_bound: Q = Q(9, 128)
    contact_form_channels: int = 18
    existence_gap_threshold: Q = _DELTA / 8
    retained_gap_threshold: Q = _DELTA / 4
    phase_radius: Q = _RADIUS
    regional_mean_error_cap: Q = _MEAN_ERROR_CAP
    readout_error_cap: Q = _READOUT_ERROR_CAP
    arithmetic_budget: Q = _ARITHMETIC_BUDGET
    prediction_target: Q = _PREDICTION_TARGET
    separation_target: Q = _SEPARATION_TARGET
    clock: str = "tau=(1023/1024)*t; internal rows and joint input use tau"
    output_orientation: str = "outgoing_from_first_copy_to_second"
    negative_orientation: str = (
        "Q_with_negative_partner_charge_minus_Q_with_positive_partner_charge"
    )
    scope: tuple[str, ...] = (
        "two_actual_conditional_UM_images_with_independent_source_associations",
        "known_degree_loss_internal_laws_and_unchanged_degree9_supports",
        "nine_matched_form_only_reciprocal_inputs_not_added_graph_edges",
        "two_original_readings_carried_through_prefix_without_new_handoff_data",
        "four_evolving_means_with_bounded_live_phase_and_weight_residuals",
        "joint_form_conservation_and_norm_dissipation_not_gradient_storage_passivity",
        "weak_information_discards_partner_command_regional_data_and_post_leak_records",
        "no_joint_identity_acquisition_runtime_authentication_or_empirical_claim",
    )


def _assess_mixed_reciprocal_partner(
    *,
    form_error_bounds,
    phase_error_bounds,
    starting_gap_upper_bounds,
    readout_error_bound,
):
    """Re-admit both original sources before the fixed carried-contact theorem."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    gaps = _nonnegative_pair(starting_gap_upper_bounds, "starting_gap_upper_bounds")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    prefixes = tuple(
        _assess_mixed_regional_boundary(
            form_error_bound=rx,
            phase_error_bound=rt,
            starting_gap_upper_bound=gap,
            # This rebuilds prefix state bounds, not a preceding measured Q.
            readout_error_bound=Q(0),
        )
        for rx, rt, gap in zip(forms, phases, gaps, strict=True)
    )
    source_ok = tuple(
        max(rx, rt) <= _SOURCE_CAP
        and prefix.form_error_bound == rx
        and prefix.phase_error_bound == rt
        and prefix.starting_gap_upper_bound == gap
        and prefix.source_certified
        and prefix.full_state_tube_certified
        and prefix.norm_tube_upper_bound <= _PREFIX_NORM
        and prefix.phase_radius == _RADIUS
        and prefix.gamma_upper_bound <= _GAMMA_UPPER
        and prefix.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
        and prefix.charge_duration == prefix.probe_duration == _TIME
        and prefix.charge_amplitude_bounds == (-_AMPLITUDE, _AMPLITUDE)
        and prefix.preparation_mean_bounds == (-_MEAN_RADIUS, _MEAN_RADIUS)
        and prefix.regions == (tuple(range(9)), tuple(range(9, 18)))
        for prefix, rx, rt, gap in zip(prefixes, forms, phases, gaps, strict=True)
    )
    exists = tuple(
        admitted
        and prefix.finite_protocol_ready_age_exists
        and prefix.existence_gap_threshold == _DELTA / 8
        for admitted, prefix in zip(source_ok, prefixes, strict=True)
    )
    ready = tuple(
        admitted
        and gap <= _DELTA / 8
        and prefix.conditional_ready_source
        and prefix.joint_registration_retained_certified
        for admitted, gap, prefix in zip(source_ok, gaps, prefixes, strict=True)
    )
    exponent = 2 * _GAMMA_UPPER * _TIME
    joint_bound = Q(17, 12) * _PREFIX_NORM / (1 - exponent)
    joint_margin = _JOINT_NORM - joint_bound
    tube_ok = all(ready) and Q(17, 12) ** 2 > 2 and exponent < 1 and joint_margin > 0
    motion = 4 * _GAMMA_UPPER * _RADIUS * 3 * _TIME
    retained = tuple(gap + motion for gap in gaps)
    margins = tuple(_DELTA / 4 - bound for bound in retained)
    identity = tube_ok and _JOINT_NORM < _RADIUS and min(margins) > 0
    reference = _reciprocal_reference()
    arithmetic = (_MEAN_RADIUS + _MEAN_ERROR_CAP) * sum(
        (_radius(bounds) for bounds in reference.response_coefficient_bounds), Q(0)
    )
    inherited = 18 * _TIME * (_MEAN_ERROR_CAP + _TIME * _RESIDUAL)
    contact_error = 9 * _TIME**2 * _RESIDUAL
    error = inherited + contact_error + readout + arithmetic
    prediction_margin = _PREDICTION_TARGET - error
    leading = (
        18 * _AMPLITUDE * _TIME**2 * (1 - _TIME / 2) * (1 - 2 * _TIME - 3 * _TIME / 2)
    )
    loss = 2 * (18 * _TIME * (2 * _TIME) + 9 * _TIME**2) * _RESIDUAL
    actual_gap = leading - loss
    recorded_gap = actual_gap - 2 * readout
    weak_error = recorded_gap / 2
    separation_margin = recorded_gap - _SEPARATION_TARGET
    observation_ok = readout <= _READOUT_ERROR_CAP
    arithmetic_ok = arithmetic <= _ARITHMETIC_BUDGET
    positive = (
        identity
        and reference.reference_certified
        and observation_ok
        and arithmetic_ok
        and prediction_margin > 0
    )
    obstruction = (
        identity
        and reference.reference_certified
        and observation_ok
        and separation_margin > 0
        and weak_error > _PREDICTION_TARGET
    )
    unresolved = tuple(
        label
        for passed, label in (
            (all(source_ok), "original_sources_or_prefix_associations_unavailable"),
            (all(exists), "finite_protocol_ready_age_existence_unavailable"),
            (all(ready), "conditional_source_readiness_unavailable"),
            (tube_ok, "full_joint_state_tube_unavailable"),
            (identity, "constituent_identity_retention_unavailable"),
            (reference.reference_certified, "reference_or_residual_unavailable"),
            (observation_ok, "contact_readout_allowance_outside_protocol"),
            (arithmetic_ok, "uniform_reference_arithmetic_unavailable"),
            (positive, "carried_regional_prediction_unavailable"),
            (obstruction, "partner_history_obstruction_unavailable"),
        )
        if not passed
    )
    return _MixedReciprocalPartner(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        starting_gap_upper_bounds=gaps,
        readout_error_bound=readout,
        prefix_reports=prefixes,
        reference=reference,
        source_certified=source_ok,
        finite_protocol_ready_ages_exist=exists,
        conditional_ready_sources=ready,
        joint_norm_upper_bound=joint_bound,
        joint_norm_margin=joint_margin,
        full_state_tube_certified=tube_ok,
        marked_gap_change_upper_bound=motion,
        retained_gap_upper_bounds=retained,
        registration_margins=margins,
        constituent_identity_retained_certified=identity,
        contact_storage_work_upper_bound=36 * _JOINT_NORM**2 * _TIME,
        inherited_error_upper_bound=inherited,
        contact_error_upper_bound=contact_error,
        uniform_arithmetic_error_upper_bound=arithmetic,
        record_prediction_error_upper_bound=error,
        prediction_margin=prediction_margin,
        partner_history_leading_gap=leading,
        partner_history_model_loss_upper_bound=loss,
        actual_partner_history_gap_lower_bound=actual_gap,
        recorded_partner_history_gap_lower_bound=recorded_gap,
        weak_prediction_error_lower_bound=weak_error,
        separation_margin=separation_margin,
        positive_information_certified=positive,
        partner_history_obstruction_certified=obstruction,
        certified=not unresolved,
        status="certified" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
