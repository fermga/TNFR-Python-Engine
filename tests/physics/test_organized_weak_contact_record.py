"""Static relational-record inversion with the actual source uncertainty.

The two truth families define constructed controls, never decoder arguments.
Universal bounds retain both readout enlargements and the separately derived
rounding allowance; representative readings test composition, not universality.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import product

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics import _rational_interval as arithmetic
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_contact_maintenance as owner

ERROR = Q(1, 10**10)
EPSILON = Q(1, 10000)
Q_DONOR = Q(1, 10**70)
ARGUMENT_ALLOWANCE = Q(1, 2**100)
TOLERANCE = Q(1, 1000)


@pytest.fixture(scope="module", autouse=True)
def no_execution_or_retained_response():
    def forbidden(*args, **kwargs):
        pytest.fail("record controls cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _assess(**changes):
    return owner._assess_organized_weak_contact_record(
        **(
            dict(
                form_error_bounds=(ERROR,) * 6,
                phase_error_bounds=(ERROR,) * 6,
                quadratures=(Q(1), Q(0)),
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def actual():
    calls = []
    rebuild = owner._assess_organized_external_interaction

    def counted(**kwargs):
        calls.append(kwargs)
        return rebuild(**kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_assess_organized_external_interaction", counted)
        result = _assess()
    assert calls == [
        dict(form_error_bounds=(ERROR,) * 6, phase_error_bounds=(ERROR,) * 6)
    ]
    return result


@pytest.fixture
def source_wiring(monkeypatch, actual):
    # Later interface controls reuse this freshly rebuilt source through an
    # explicit test stub. Neither public-facing reader accepts a report input.
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: actual.source
    )
    return actual.source


def _truth(label):
    return I(Q(3, 5)) if label == 0 else 2 * pi_interval() / 5 - Q(3, 5)


def _source_offsets(source):
    host = source.host_acquisition
    radius = host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    h18 = host.primitive_entry.endpoint_phase_bounds[2] + I(-radius, radius)
    return h18, I(-Q_DONOR, Q_DONOR)


def _uniform_inverse(source, label):
    h18, donor = _source_offsets(source)
    truth = _truth(label)
    gap = truth - h18 + donor
    true_quadratures = cos(gap), sin(gap)
    # First epsilon covers every raw sample. The second covers the rectangle
    # decoded from that sample. Their sum is necessary even for correlated errors.
    super_rectangle = tuple(
        box + I(-2 * EPSILON, 2 * EPSILON) for box in true_quadratures
    )
    real, imaginary = super_rectangle
    ratio = imaginary / real
    assert real.lo > Q(3, 4) and imaginary.lo > 0
    assert 0 <= ratio.lo <= ratio.hi <= 1
    # Numerical atan endpoints need not be monotone. The frozen endpoint-error
    # lemma in argument-budget.json dominates all returned per-record intervals.
    uniform_angle = arg(*super_rectangle) + I(-ARGUMENT_ALLOWANCE, ARGUMENT_ALLOWANCE)
    inverse = uniform_angle + h18 - donor
    return truth, gap, true_quadratures, super_rectangle, inverse


def test_raw_record_uses_one_fresh_full_source_and_actual_port_association(actual):
    assert actual.status == "certified_observed_registration"
    assert actual.registration_label == 0 and actual.limiting_relative_origin_turns == 0
    assert actual.marked_nodes == (45, 18) and actual.readout_error == EPSILON
    assert actual.source is actual.basin.source
    assert actual.source.relative_preparation_dimension == 96
    assert (
        actual.source.form_error_bounds
        == actual.source.phase_error_bounds
        == (ERROR,) * 6
    )
    h18, donor = _source_offsets(actual.source)
    assert actual.host_port_offset_bounds == h18
    assert h18 == actual.source.outer_one_host_phase_offset_bounds[2]
    assert h18.abs_max < Q(1, 40000)
    assert actual.donor_port_residual_bounds == donor
    assert not donor.contains(actual.source.donor_phase_offset_bounds)
    assert actual.donor_origin_bounds == actual.argument_bounds + h18 - donor
    assert actual.basin.donor_origin_bounds == (
        actual.donor_origin_bounds.lo,
        actual.donor_origin_bounds.hi,
    )
    assert actual.argument_certified and actual.origin_lift_certified
    with pytest.raises(FrozenInstanceError):
        actual.registration_label = 4


@pytest.mark.parametrize("label", (0, 1))
def test_frozen_whole_source_and_all_readout_errors_have_one_destination(actual, label):
    truth, gap, _, _, inverse = _uniform_inverse(actual.source, label)
    assert 0 < gap.lo <= gap.hi < 1
    assert (inverse - truth).abs_max < TOLERANCE
    center = 2 * label * pi_interval() / 5
    assert (inverse - center).abs_max < Q(5, 8)
    assert Q(3, 5) + TOLERANCE < Q(5, 8)
    basin = owner._decide_admitted_weak_contact_basin(
        (ERROR,) * 6, (ERROR,) * 6, (inverse.lo, inverse.hi), actual.source
    )
    assert basin.status == "certified_unique_registration"
    assert basin.registration_label == label
    assert basin.limiting_relative_origin_turns == Q(-label, 5)


@pytest.mark.parametrize("label", (0, 1))
def test_representative_raw_records_compose_without_truth_arguments(
    source_wiring, label
):
    truth, _, true_boxes, _, uniform = _uniform_inverse(source_wiring, label)
    h18, _ = _source_offsets(source_wiring)
    point_gap = truth - h18.midpoint
    central = tuple(f(point_gap).midpoint for f in (cos, sin))
    assert all(
        box.lo - EPSILON <= value <= box.hi + EPSILON
        for box, value in zip(true_boxes, central)
    )
    # These finite controls exercise the consumer. The preceding whole-family
    # allowance, not these four readings, proves the arbitrary-error claim.
    for signs in product((-1, 1), repeat=2):
        reading = tuple(
            value + sign * EPSILON / 2 for value, sign in zip(central, signs)
        )
        result = _assess(quadratures=reading)
        assert result.status == "certified_observed_registration"
        assert result.registration_label == label
        assert result.limiting_relative_origin_turns == Q(-label, 5)
        assert result.donor_origin_bounds.subset_of(uniform)
        assert (result.donor_origin_bounds - truth).abs_max < TOLERANCE
        assert result.quadratures == reading


def test_uniform_argument_rounding_allowance_has_an_independent_analytic_budget():
    assert arithmetic.INTERVAL_BITS == 128 and arithmetic._ATAN_TERMS == 64
    grid = Q(1, 2**128)
    # The induction is stated in the frozen argument-budget receipt. These
    # exact inequalities check its constants, including the true series tail.
    assert 3**64 > 2**101
    assert Q(1, 3**64 * 129) < Q(1, 2**108)
    assert 388 * grid + Q(1, 2**108) < Q(1, 2**107)
    assert 2 * Q(1, 2**105) < ARGUMENT_ALLOWANCE
    # Independent geometric budget, before the much smaller arithmetic slack:
    # L1 distance <=4eps, radial segment length >=1-4eps, then two source offsets.
    angle_error = 4 * EPSILON / (1 - 4 * EPSILON)
    assert angle_error == Q(1, 2499) < Q(1, 2000)
    assert angle_error + Q(1, 20000) + 2 * Q_DONOR < TOLERANCE


def test_local_only_record_keeps_independent_donor_phase_symmetry(actual):
    source = actual.source
    rigid_donor_shift = (Q(0),) * 45 + (Q(1),) * 9
    before_contact = source.geometry.internal_edges + source.geometry.outer_edges
    assert len(before_contact) == 99
    assert all(
        rigid_donor_shift[b] - rigid_donor_shift[a] == 0 for a, b, _ in before_contact
    )
    assert all(rigid_donor_shift[node] == 0 for node in range(45))
    assert rigid_donor_shift[45] - rigid_donor_shift[18] == 1
    # Each local host coordinate and each intrinsic donor edge difference is
    # unchanged by the same translation; only the marked relational channel sees it.
    assert _truth(1).lo > _truth(0).hi
    destinations = []
    for label in (0, 1):
        inverse = _uniform_inverse(source, label)[-1]
        result = owner._decide_admitted_weak_contact_basin(
            (ERROR,) * 6, (ERROR,) * 6, (inverse.lo, inverse.hi), source
        )
        destinations.append(result.registration_label)
    assert destinations == [0, 1]


def _forbidden_source(**kwargs):
    pytest.fail("unadmitted or unavailable record reached source rebuilding")


@pytest.mark.parametrize("position", range(14))
@pytest.mark.parametrize("bad", (True, "0", 0j, float("inf"), float("nan")))
def test_all_fourteen_scalars_reject_invalid_values_before_source(
    monkeypatch, position, bad
):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    forms, phases, reading = [ERROR] * 6, [ERROR] * 6, [Q(1), Q(0)]
    if position < 6:
        forms[position] = bad
    elif position < 12:
        phases[position - 6] = bad
    else:
        reading[position - 12] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(form_error_bounds=forms, phase_error_bounds=phases, quadratures=reading)


@pytest.mark.parametrize("position", range(12))
@pytest.mark.parametrize("bad", (Q(-1, 2**600), ERROR + Q(1, 2**600)))
def test_every_source_bound_retains_its_exact_physical_domain(
    monkeypatch, position, bad
):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    forms, phases = [ERROR] * 6, [ERROR] * 6
    (forms if position < 6 else phases)[position % 6] = bad
    with pytest.raises(ValueError, match="frozen"):
        _assess(form_error_bounds=forms, phase_error_bounds=phases)


@pytest.mark.parametrize(
    "field", ("form_error_bounds", "phase_error_bounds", "quadratures")
)
@pytest.mark.parametrize("bad", (None, "00", {0, 1}, {0: 1}, ()))
def test_order_and_vector_shape_are_admitted_before_rebuilding(monkeypatch, field, bad):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


def test_binary64_source_ceiling_is_not_decimal_rounded(monkeypatch):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    assert Q.from_float(1e-10) > ERROR
    with pytest.raises(ValueError, match="frozen"):
        _assess(form_error_bounds=(1e-10,) * 6)


@pytest.mark.parametrize("reading", ((0, 0), (2, 0), (1 + EPSILON + Q(1, 2**600), 0)))
def test_finite_impossible_observations_remain_distinct_from_argument_cuts(
    monkeypatch, reading
):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    result = _assess(quadratures=reading)
    assert result.status == "incompatible_observation"
    assert not result.unit_circle_compatible
    assert result.source is None and result.registration_label is None
    assert result.argument_bounds is None


def test_tangent_circle_feasibility_is_exact_before_interval_rounding(
    source_wiring, monkeypatch
):
    tangent = Q(3, 5) + EPSILON, Q(4, 5) + EPSILON
    compatible = _assess(quadratures=tangent)
    assert compatible.unit_circle_compatible
    assert compatible.unit_circle_norm_squared_bounds[0] == 1
    shifted = tangent[0] + Q(1, 2**600), tangent[1]
    # The two real intervals become identical on the dyadic grid. Exact
    # feasibility must still distinguish the actual rational raw samples.
    assert I(tangent[0] - EPSILON, tangent[0] + EPSILON) == I(
        shifted[0] - EPSILON, shifted[0] + EPSILON
    )
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    impossible = _assess(quadratures=shifted)
    assert impossible.unit_circle_norm_squared_bounds[0] > 1
    assert impossible.status == "incompatible_observation"


@pytest.mark.parametrize("imaginary", (Q(0), EPSILON, EPSILON + Q(1, 2**600)))
def test_compatible_cut_or_unresolved_argument_does_not_build_source(
    monkeypatch, imaginary
):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    result = _assess(quadratures=(Q(-1), imaginary))
    assert result.unit_circle_compatible
    assert result.status == "unavailable_argument"
    assert result.source is None and result.registration_label is None


def test_regular_argument_can_have_an_unresolved_transported_origin_lift(source_wiring):
    result = _assess(quadratures=(Q(-1), EPSILON + Q(1, 2**100)))
    assert result.unit_circle_compatible and result.argument_certified
    assert result.status == "unavailable_lift"
    assert not result.origin_lift_certified and result.registration_label is None


@pytest.mark.parametrize(
    "real,imaginary,label",
    (
        (Q(3, 5), Q(4, 5), 1),
        (Q(3, 5), Q(-4, 5), 4),
        (Q(-3, 5), Q(4, 5), 2),
        (Q(-3, 5), Q(-4, 5), 3),
    ),
)
def test_regular_quadrants_preserve_the_destination_minus_sign(
    source_wiring, real, imaginary, label
):
    result = _assess(quadratures=(real, imaginary))
    assert result.status == "certified_observed_registration"
    assert result.registration_label == label
    assert result.limiting_relative_origin_turns % 1 == Q(-label, 5) % 1


def test_nonunique_outer_interval_does_not_claim_mixed_raw_record_outcomes(
    source_wiring, monkeypatch
):
    mixed = owner._decide_admitted_weak_contact_basin(
        (ERROR,) * 6, (ERROR,) * 6, (Q(5, 8), Q(79, 125)), source_wiring
    )
    assert mixed.status == "certified_mixed_registrations"
    # Interface control: even a proven mixed OUTER interval is insufficient
    # to assert that its witnesses both realize the same fixed raw record.
    monkeypatch.setattr(
        owner, "_decide_admitted_weak_contact_basin", lambda *args: mixed
    )
    result = _assess()
    assert result.status == "unavailable_registration"
    assert result.registration_label is None
    assert result.basin is mixed


def test_unresolved_boundary_record_is_not_promoted_to_a_destination(source_wiring):
    phase = pi_interval() / 5
    reading = cos(phase).midpoint, sin(phase).midpoint
    result = _assess(quadratures=reading)
    assert result.status == "unavailable_registration"
    assert result.registration_label is None
    assert result.basin.status == "unavailable_insufficient_certificate"


@pytest.mark.parametrize("failure", ("source_flag", "host_energy", "clock"))
def test_source_failure_prevents_a_label_despite_supported_raw_argument(
    actual, monkeypatch, failure
):
    source = actual.source
    if failure == "source_flag":
        changed = replace(
            source,
            source_certified=False,
            source_admission_flags=(False,) + source.source_admission_flags[1:],
        )
    elif failure == "host_energy":
        changed = replace(
            source,
            host_acquisition=replace(
                source.host_acquisition, endpoint_excess_upper_bound=Q(1)
            ),
        )
    else:
        changed = replace(source, geometry=replace(source.geometry, clock_factor=Q(1)))
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: changed
    )
    result = _assess()
    assert result.status == "unavailable_source"
    assert result.argument_certified and not result.source_certified
    assert result.registration_label is None


def test_legacy_donor_origin_and_unit_contact_verdicts_are_not_consumed(
    actual, monkeypatch
):
    changed = replace(
        actual.source,
        donor_phase_offset_bounds=I(100),
        admitted=False,
        comparison_certified=False,
        identity_certified=False,
        work_certified=False,
    )
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: changed
    )
    result = _assess()
    assert result.status == actual.status
    assert result.donor_port_residual_bounds == actual.donor_port_residual_bounds
    assert result.donor_origin_bounds == actual.donor_origin_bounds
    assert result.registration_label == actual.registration_label


@pytest.mark.parametrize(
    "keyword",
    (
        "beta",
        "donor_origin_bounds",
        "readout_error",
        "epsilon",
        "source",
        "source_certified",
        "shape_initial_upper",
        "host_port_offset_bounds",
        "argument_bounds",
    ),
)
def test_record_reader_has_no_truth_noise_policy_or_cached_evidence_inputs(
    monkeypatch, keyword
):
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", _forbidden_source
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        _assess(**{keyword: Q(0)})
