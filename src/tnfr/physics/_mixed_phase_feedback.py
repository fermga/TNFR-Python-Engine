"""Resolution of phase feedback in the existing reciprocal transfer task.

The comparator is switched on only after the shared actual preparation and
regional leaks. It retains the live phase row and reciprocal form inputs.
Its source association is a premise, not a claim of own-law acquisition.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I
from ._mixed_reciprocal_partner import (
    _assess_mixed_reciprocal_partner,
    _MixedReciprocalPartner,
    _nonnegative,
    _nonnegative_pair,
)
from ._sine_coupling_successor import _derive_coupling_successor_geometry
from ._sine_preparation import _prepared_coordinate_vector as _vector
from .relational_observations import _ordered
from .relational_sine_comparison import _sine_rate_evaluator
from .support_transport import _laplacian

_LOSS = Q(1023, 1024)
_TIME = Q(1, 32)
_DELTA = Q(1, 1024)
_RADIUS = Q(1, 32)
_GAMMA_UPPER = Q(1, 3000)
_PREFIX_NORM = Q(9, 512)
_JOINT_NORM = Q(1, 40)
_SOURCE_CAP = Q(1, 10**10)
_READOUT_CAP = Q(1, 10**6)
_CONTACT_PAIRS = tuple((j, 18 + j) for j in range(9))


def _mixed_without_form_feedback_field(
    *, first_new_edge_conductances, second_new_edge_conductances
):
    """Compile the complete matched comparator in structural time tau.

    Only the sine contribution to form evolution is removed. The shared
    rate owner receives the original nonzero phase coefficient and zero
    sine currents; its phase rates remain gamma*L_W*x/9. Nine reciprocal
    form inputs are added afterwards, without changing internal degrees.

    Both weight vectors precede geometry work. Each evaluation admits all
    36 forms and then all 36 phase lifts before any gradient arithmetic.
    Exact/interval scalars and uniformly ordered Taylor jets are supported.
    Legal weights are an enclosing family, not authenticated UM proposals.
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
    geometry = _derive_coupling_successor_geometry(completion="common_sense")
    if not (
        geometry.support_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degree == 9
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
    ):
        raise ValueError("the fixed degree-nine successor support is unavailable")
    edges = tuple(
        (a + 18 * copy, b + 18 * copy, weight)
        for copy, row in enumerate(weights)
        for a, b, weight in geometry.internal_edges
        + tuple(
            (a, b, weight)
            for (a, b), weight in zip(geometry.new_edge_pairs, row, strict=True)
        )
    )
    directed = tuple(row for a, b, w in edges for row in ((a, b, w), (b, a, w)))
    model = RelationalExchangeModel(
        1, epi_weight=_LOSS, phase_weight=Q(1, 1024), phase_domain="regular"
    )
    evaluate = _sine_rate_evaluator(model, (9,) * 36, (Q(1),) * 36)

    def flow(values):
        state = _ordered(values, "state", limit=73)
        if len(state) != 72:
            raise ValueError("state requires all 36 forms followed by all 36 phases")
        jets = tuple(isinstance(value, Jet) for value in state)
        if any(jets):
            if not all(jets) or any(value.order != state[0].order for value in state):
                raise ValueError("state Taylor jets must all have the same order")
            zero = Jet.constant(0, state[0].order)
        else:
            state = tuple(I.coerce(value) for value in state)
            zero = I(0)
        gradient = _laplacian(directed, state[:36])
        rates = evaluate(gradient, (zero,) * 36)
        forms = tuple(value / _LOSS for value in rates["form_rates"])
        phases = tuple(value / _LOSS for value in rates["phase_rates"])
        exchange = tuple(state[18 + j] - state[j] for j in range(9))
        first = tuple(forms[j] + exchange[j] for j in range(9)) + forms[9:18]
        second = tuple(forms[18 + j] - exchange[j] for j in range(9)) + forms[27:]
        return first + second + phases

    return flow


@dataclass(frozen=True)
class _MixedPhaseFeedback:
    """Conditional true-response bounds and a sufficient record overlap.

    The two laws start from the identical actual reached state at contact.
    Bounds concern their complete nonlinear solutions, not overlapping
    computed endpoint boxes. A certified overlap supplies a common possible
    record under the declared independent bounded errors. Failure to certify
    overlap never certifies separation. All candidate bounds require their
    source, retention and difference flags; no runtime association is checked.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    starting_gap_upper_bounds: tuple[Q, Q]
    readout_error_bound: Q
    reference: _MixedReciprocalPartner
    source_certified: tuple[bool, bool]
    finite_protocol_ready_ages_exist: tuple[bool, bool]
    conditional_ready_sources: tuple[bool, bool]
    actual_tube_certified: bool
    comparator_norm_upper_bound: Q
    comparator_norm_margin: Q
    comparator_tube_certified: bool
    marked_gap_change_upper_bound: Q
    retained_gap_upper_bounds: tuple[Q, Q]
    registration_margins: tuple[Q, Q]
    constituent_identity_retained_certified: bool
    form_difference_upper_bound: Q
    phase_difference_upper_bound: Q
    transfer_difference_upper_bound: Q
    difference_bound_certified: bool
    readout_allowance_admitted: bool
    overlap_margin: Q
    shared_record_overlap_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    charge_duration: Q = _TIME
    prefix_leak_duration: Q = _TIME
    contact_duration: Q = _TIME
    gamma_upper_bound: Q = _GAMMA_UPPER
    full_norm_bound: Q = _JOINT_NORM
    phase_radius: Q = _RADIUS
    source_error_cap: Q = _SOURCE_CAP
    readout_error_cap: Q = _READOUT_CAP
    existence_gap_threshold: Q = _DELTA / 8
    retained_gap_threshold: Q = _DELTA / 4
    source_count: int = 2
    full_state_dimension: int = 72
    contact_pairs: tuple[tuple[int, int], ...] = _CONTACT_PAIRS
    contact_rate: Q = Q(1)
    matched_initial_state_assumed: bool = True
    comparator_own_law_source_acquisition_claimed: bool = False
    new_contact_entry_readings: int = 0
    clock: str = "tau=(1023/1024)*t; both complete rows and joint input use tau"
    output_orientation: str = "outgoing_from_first_copy_to_second"
    scope: tuple[str, ...] = (
        "shared_actual_preparation_and_both_leaks_before_supplied_law_switch",
        "remove_only_form_sine_feedback_keep_live_phase_and_mutual_reaction",
        "same_weights_degree9_capacities_clock_and_two_independent_phase_gauges",
        "true_response_gap_bound_not_overlap_of_coarse_endpoint_enclosures",
        "worst_case_bounded_record_errors_not_statistical_equivalence",
        "no_equality_of_responses_or_absence_of_phase_feedback_asserted",
        "no_comparator_formation_outer_NFR_runtime_authentication_or_physical_claim",
    )


def _assess_mixed_phase_feedback(
    *,
    form_error_bounds,
    phase_error_bounds,
    starting_gap_upper_bounds,
    readout_error_bound,
):
    """Admit the existing contact's phase-resolution limit without a new probe."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    gaps = _nonnegative_pair(starting_gap_upper_bounds, "starting_gap_upper_bounds")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    reference = _assess_mixed_reciprocal_partner(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        starting_gap_upper_bounds=gaps,
        # Rebuild the actual source/tube, not its previous observation verdict.
        readout_error_bound=Q(0),
    )
    association = (
        reference.form_error_bounds == forms
        and reference.phase_error_bounds == phases
        and reference.starting_gap_upper_bounds == gaps
        and reference.source_count == 2
        and reference.full_state_dimension == 72
        and reference.charge_duration
        == reference.prefix_leak_duration
        == reference.contact_duration
        == _TIME
        and reference.phase_radius == _RADIUS
        and reference.contact_pairs == _CONTACT_PAIRS
        and reference.contact_rate == 1
        and reference.existence_gap_threshold == _DELTA / 8
        and reference.retained_gap_threshold == _DELTA / 4
    )
    source_ok = tuple(
        association and max(rx, rt) <= _SOURCE_CAP and admitted
        for rx, rt, admitted in zip(
            forms, phases, reference.source_certified, strict=True
        )
    )
    exists = tuple(
        source and present
        for source, present in zip(
            source_ok, reference.finite_protocol_ready_ages_exist, strict=True
        )
    )
    ready = tuple(
        source and present and gap <= _DELTA / 8
        for source, present, gap in zip(
            source_ok, reference.conditional_ready_sources, gaps, strict=True
        )
    )
    actual_tube = (
        all(exists)
        and all(ready)
        and reference.full_state_tube_certified
        and 0 <= reference.joint_norm_upper_bound < _JOINT_NORM
        and reference.joint_norm_margin > 0
    )
    exponent = _GAMMA_UPPER * _TIME
    comparator_bound = Q(17, 12) * _PREFIX_NORM / (1 - exponent)
    comparator_margin = _JOINT_NORM - comparator_bound
    comparator_tube = (
        actual_tube and Q(17, 12) ** 2 > 2 and exponent < 1 and comparator_margin > 0
    )
    motion = 4 * _GAMMA_UPPER * _RADIUS * 3 * _TIME
    retained = tuple(gap + motion for gap in gaps)
    margins = tuple(_DELTA / 4 - bound for bound in retained)
    identity = (
        comparator_tube
        and reference.constituent_identity_retained_certified
        and _JOINT_NORM < _RADIUS
        and min(margins) > 0
    )
    form_error = 2 * _GAMMA_UPPER * _JOINT_NORM * _TIME
    phase_error = 2 * _GAMMA_UPPER**2 * _JOINT_NORM * _TIME**2
    transfer_error = Q(17, 4) * _GAMMA_UPPER * _JOINT_NORM * _TIME**2
    difference_ok = identity and Q(17, 4) ** 2 > 18
    observation_ok = readout <= _READOUT_CAP
    overlap_margin = 2 * readout - transfer_error
    overlap = difference_ok and observation_ok and overlap_margin >= 0
    unresolved = tuple(
        label
        for passed, label in (
            (all(source_ok), "original_sources_or_shared_prefix_unavailable"),
            (all(exists), "finite_protocol_ready_age_existence_unavailable"),
            (all(ready), "conditional_source_readiness_unavailable"),
            (actual_tube, "actual_joint_state_tube_unavailable"),
            (comparator_tube, "comparator_joint_state_tube_unavailable"),
            (identity, "both_law_constituent_identity_retention_unavailable"),
            (difference_ok, "true_response_difference_bound_unavailable"),
            (observation_ok, "readout_allowance_outside_protocol"),
            (overlap_margin >= 0, "reading_allowance_does_not_certify_overlap"),
        )
        if not passed
    )
    return _MixedPhaseFeedback(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        starting_gap_upper_bounds=gaps,
        readout_error_bound=readout,
        reference=reference,
        source_certified=source_ok,
        finite_protocol_ready_ages_exist=exists,
        conditional_ready_sources=ready,
        actual_tube_certified=actual_tube,
        comparator_norm_upper_bound=comparator_bound,
        comparator_norm_margin=comparator_margin,
        comparator_tube_certified=comparator_tube,
        marked_gap_change_upper_bound=motion,
        retained_gap_upper_bounds=retained,
        registration_margins=margins,
        constituent_identity_retained_certified=identity,
        form_difference_upper_bound=form_error,
        phase_difference_upper_bound=phase_error,
        transfer_difference_upper_bound=transfer_error,
        difference_bound_certified=difference_ok,
        readout_allowance_admitted=observation_ok,
        overlap_margin=overlap_margin,
        shared_record_overlap_certified=overlap,
        certified=not unresolved,
        status="shared_record_overlap_certified" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
