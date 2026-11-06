"""Optional read-only audit of retained seeded-response records, without replay.

Digests bind these checks to observed local bytes, not authenticated chronology
or independent physical evidence. Current equivalent field arithmetic checks
the saved tubes; no producer, trajectory step or reserved response is executed.
"""

import hashlib
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval
from tnfr.physics._relational_reflected_flow import evaluate_reflected_regular_flow
from tnfr.physics.relational_reflected_transit import (
    certify_relational_reflected_barrier,
)
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
DIGESTS = {
    "v1": "d65eb3c9fc88979d71d6abed3178a1f02307b5b37b4cb6bd0d077cdfd4c0e867",
    "v2": "183c762fa4853a1f42c566a3683eb948263e0327f60c50819684cf4bb51ece75",
}
BUDGET_DIGESTS = {
    "record": "f142e75070b141644541e91ac549e7fa9d6c5bfbce8700735130b870a6a0c483",
    "protocol": "5c748341025197b1c6219c2b553b37f29bee8199ca1088edf157e33154189bf4",
    "archive": "37e2150bed71757de7399e03bf13158919c090330af1b5909784f7cfa66f059d",
}
EXTENDED_BUDGET_DIGESTS = {
    "record": "d3174948f69f600114a3dce2e4915d79cef75230c7a7c7100b6da7de0c6ef332",
    "protocol": "a278574a78e65cd3d55376256f4e230eebb631ac645d34884907eb51b1c8cf1d",
    "archive": "b76ce10679b49c180b21bd5491e2fd10a24922727e0ba342302df0097990f92a",
}
COEFFICIENTS = dict(epi_weight=Q(1, 2), phase_weight=Q(1, 2), storage_scale=Q(1))


def _rational(value):
    assert set(value) == {"numerator", "denominator"}
    assert type(value["numerator"]) is type(value["denominator"]) is int
    assert value["denominator"] > 0
    return Q(value["numerator"], value["denominator"])


def _interval(value):
    assert set(value) == {"lo", "hi"}
    return I(_rational(value["lo"]), _rational(value["hi"]))


def _box(values):
    assert len(values) == 8
    return tuple(map(_interval, values))


def _intersection(left, right):
    return I(max(left.lo, right.lo), min(left.hi, right.hi))


def _winding(box, *, receiver):
    a, b = box[6:8] if receiver else box[4:6]
    pi = pi_interval()
    internal = (a - b).abs_max < pi.lo and b.abs_max < pi.lo
    return internal and (
        (2 * a).abs_max < pi.lo if receiver else a.lo > pi.hi / 2 and a.hi < pi.lo
    )


def _load_bundle(path, digest):
    required = (
        path,
        path.with_suffix(".protocol.json"),
        path.with_suffix(".source.zip"),
    )
    if any(not path.is_file() for path in required):
        pytest.skip(
            "optional local seeded-response records and source archives are absent"
        )
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    record = json_loads(data)
    certificate = record["response"]["certificate"]["report"]
    steps = []
    for step in certificate["steps"]:
        tube, endpoint = _box(step["tube"]), _box(step["endpoint"])
        steps.append(
            (
                step,
                tube,
                endpoint,
                evaluate_reflected_regular_flow(tube, **COEFFICIENTS),
                evaluate_reflected_regular_flow(endpoint, **COEFFICIENTS),
            )
        )
    return path, record, certificate, tuple(steps)


@pytest.fixture(scope="module")
def retained():
    directory = ROOT / "artifacts/research/relational_seeded_response"
    return {
        version: _load_bundle(directory / f"response-{version}.json", digest)
        for version, digest in DIGESTS.items()
    }


@pytest.fixture(scope="module")
def retained_budget():
    # Separate availability prevents a missing continuation bundle from
    # skipping the earlier short-response audits.
    path = ROOT / "artifacts/research/relational_seeded_response/budget-t1-v1.json"
    return _load_bundle(path, BUDGET_DIGESTS["record"])


@pytest.fixture(scope="module")
def retained_extended_budget():
    path = ROOT / "artifacts/research/relational_seeded_response/budget-t9over8-v1.json"
    return _load_bundle(path, EXTENDED_BUDGET_DIGESTS["record"])


def _assert_provenance(bundle):
    path, record, _, _ = bundle
    protocol_bytes = path.with_suffix(".protocol.json").read_bytes()
    archive_path = path.with_suffix(".source.zip")
    protocol = json_loads(protocol_bytes)
    assert record["schema"] == "tnfr.relational-seeded-response-record.v1"
    assert record["evaluation_error"] is None
    assert record["protocol"] == protocol
    assert record["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert (
        record["source_archive_sha256"]
        == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    _verify_archive(archive_path, protocol["source_sha256"])


def test_retained_protocols_archives_and_correction_link_bind_exact_observed_bytes(
    retained,
):
    for bundle in retained.values():
        _assert_provenance(bundle)
    first, corrected = (retained[version][1]["protocol"] for version in ("v1", "v2"))
    assert first["protocol"] == corrected["protocol"]
    assert "correction" not in first
    correction = corrected["correction"]
    assert correction["prior_record_sha256"] == DIGESTS["v1"]
    assert correction["prior_record"] == retained["v1"][0].relative_to(ROOT).as_posix()
    assert correction["prior_passed"] is False
    assert correction["kind"] == "separately_identified_numerical_correction"
    assert (
        correction["scope"] == "numerical_correction_not_independent_blind_replication"
    )


def _assert_picard_and_chains(bundle, *, horizon):
    _, record, certificate, steps = bundle
    pi = pi_interval()
    prior = (I(0),) * 4 + (4 * pi / 5, 2 * pi / 5, I(0), I(0))
    assert _box(certificate["initial_box"]) == prior
    declaration = record["protocol"]["protocol"]
    assert _rational(declaration["horizon"]) == horizon
    assert _rational(declaration["time_step"]) == Q(1, 64)
    assert _rational(certificate["horizon"]) == horizon
    assert _rational(certificate["time_step"]) == Q(1, 64)
    assert declaration["order"] == certificate["order"] == 6
    for name, value in COEFFICIENTS.items():
        assert _rational(declaration["model"][name]) == value
        assert Q(certificate["model"][name]) == value
    assert (
        declaration["model"]["phase_domain"]
        == certificate["model"]["phase_domain"]
        == "regular"
    )
    assert declaration["model"]["forcing"] == 0
    assert declaration["model"]["events"] == []
    assert (
        declaration["initial_interpretation"]
        == "mathematical_pi_IVP_enclosed_outward_not_projected_float_graph"
    )
    time = Q(0)
    for step, tube, endpoint, field, _ in steps:
        duration = _rational(step["duration"])
        assert _rational(step["time"]) == time
        assert duration == Q(1, 64)
        assert all(value.subset_of(bound) for value, bound in zip(prior, tube))
        assert all(value.subset_of(bound) for value, bound in zip(endpoint, tube))
        image = tuple(
            value + rate * I(0, duration) for value, rate in zip(prior, field.rates)
        )
        margin = min(
            min(value.lo - bound.lo, bound.hi - value.hi)
            for value, bound in zip(image, tube)
        )
        assert margin > 0
        assert _rational(step["picard_interior_margin"]) > 0
        assert all(_rational(value) > 0 for value in step["domain_lower_bounds"])
        current_margins = field.resultant_margin_lower_bounds + (
            pi.lo - tube[4].abs_max,
            pi.lo / 2 - tube[5].abs_max,
            pi.lo - tube[6].abs_max,
            pi.lo / 2 - tube[7].abs_max,
        )
        assert min(current_margins) > 0
        assert (
            len(step["local_remainder_bounds"])
            == len(step["propagated_initial_radii"])
            == 8
        )
        assert all(_rational(value) >= 0 for value in step["propagated_initial_radii"])
        prior, time = endpoint, time + duration
    assert prior == _box(certificate["endpoint"])
    assert time == _rational(certificate["validated_horizon"])
    # The equivalent corrected chart can tighten v1 arithmetic. The audit
    # does not demand identical margins or regenerate Taylor endpoints.


def _assert_storage_and_observations(bundle):
    _, _, certificate, steps = bundle
    initial = evaluate_reflected_regular_flow(
        _box(certificate["initial_box"]), **COEFFICIENTS
    )
    target = 10 * (1 - cos(2 * pi_interval() / 5))
    assert initial.storage == _interval(certificate["initial_storage"])
    assert arg(*initial.resultants[0]) == _interval(
        certificate["initial_source_argument"]
    )
    assert target == _interval(certificate["target_storage"])
    cumulative = I(0)
    assert len(certificate["observations"]) == len(steps)
    for (step, tube, _, field, endpoint_field), observation in zip(
        steps, certificate["observations"]
    ):
        duration = _rational(step["duration"])
        increment = field.continuous_loss * duration
        cumulative_candidate = cumulative + increment
        storage = _intersection(
            endpoint_field.storage, initial.storage - cumulative_candidate
        )
        cumulative = _intersection(cumulative_candidate, initial.storage - storage)
        assert increment.lo >= 0
        assert _interval(observation["integrated_loss"]) == increment
        assert _interval(observation["cumulative_loss"]) == cumulative
        assert _interval(observation["storage"]) == storage
        assert _interval(observation["target_budget"]) == storage - target
        assert _interval(observation["source_argument"]) == arg(
            *endpoint_field.resultants[0]
        )
        assert _rational(observation["time"]) == _rational(step["time"]) + duration
        assert observation["source_winding_one_on_tube"] is _winding(
            tube, receiver=False
        )
        assert observation["receiver_winding_zero_on_tube"] is _winding(
            tube, receiver=True
        )
    final = certificate["observations"][-1]
    for summary, observed in (
        ("endpoint_storage", "storage"),
        ("cumulative_loss", "cumulative_loss"),
        ("target_budget", "target_budget"),
        ("endpoint_source_argument", "source_argument"),
    ):
        assert certificate[summary] == final[observed]


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_retained_whole_time_tubes_have_strict_picard_inclusion_and_endpoint_chains(
    retained, version
):
    _assert_picard_and_chains(retained[version], horizon=Q(1, 8))


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_retained_storage_loss_and_observations_follow_exact_interval_accounting(
    retained, version
):
    _assert_storage_and_observations(retained[version])


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_retained_gates_distinguish_the_original_partial_result_from_the_correction(
    retained, version
):
    _, record, certificate, steps = retained[version]
    response, declaration = record["response"], record["protocol"]["protocol"]
    endpoint = _box(certificate["endpoint"])
    width = max(value.width for value in endpoint)
    change = _interval(certificate["endpoint_source_argument"]) - _interval(
        certificate["initial_source_argument"]
    )
    complete = _rational(certificate["validated_horizon"]) == _rational(
        certificate["horizon"]
    )
    expected = {
        "horizon_admitted": complete and not certificate["unavailable_reasons"],
        "endpoint_width_budget": width <= _rational(declaration["max_endpoint_width"]),
        "receiver_phase_advanced": endpoint[6].lo > 0,
        "source_argument_approached_branch": change.hi < 0,
        "positive_accumulated_loss": _interval(certificate["cumulative_loss"]).lo > 0,
        "remaining_target_budget_positive": _interval(certificate["target_budget"]).lo
        > 0,
        "source_winding_one_throughout": all(
            _winding(tube, receiver=False) for _, tube, _, _, _ in steps
        ),
        "receiver_winding_zero_throughout": all(
            _winding(tube, receiver=True) for _, tube, _, _, _ in steps
        ),
    }
    assert response["gates"] == expected
    for label in ("source_winding_one_throughout", "receiver_winding_zero_throughout"):
        assert certificate[label] is expected[label]
    assert response["scope"] == declaration["scope"]
    assert _rational(response["max_endpoint_width"]) == width
    assert _interval(response["source_argument_change"]) == change
    assert record["passed"] is response["passed"] is all(expected.values())
    if version == "v1":
        assert len(steps) == 2
        assert _rational(certificate["validated_horizon"]) == Q(1, 32)
        assert certificate["status"] == "unavailable"
        assert certificate["failed_tube"] is not None
        assert "requested_horizon_not_validated" in certificate["unavailable_reasons"]
        assert any(
            "comparison flow" in reason for reason in certificate["unavailable_reasons"]
        )
        assert not record["passed"]
    else:
        assert len(steps) == 8
        assert _rational(certificate["validated_horizon"]) == Q(1, 8)
        assert certificate["status"] == "admitted"
        assert certificate["failed_tube"] is None
        assert record["passed"]


def test_retained_budget_bundle_has_its_own_frozen_protocol_and_source_archive(
    retained_budget,
):
    _assert_provenance(retained_budget)
    _, record, _, _ = retained_budget
    assert record["protocol_sha256"] == BUDGET_DIGESTS["protocol"]
    assert record["source_archive_sha256"] == BUDGET_DIGESTS["archive"]
    declaration = record["protocol"]["protocol"]
    assert declaration["schema"] == "tnfr.relational-seeded-budget-protocol.v1"
    assert (
        declaration["continuation"]
        == "same_exact_initial_IVP_restarted_at_zero_not_a_new_preparation"
    )
    assert "correction" not in record["protocol"]


def test_retained_budget_continuation_reuses_the_exact_short_window_prefix(
    retained_budget, retained
):
    _, budget_record, budget, _ = retained_budget
    _, short_record, short, _ = retained["v2"]
    assert budget["initial_box"] == short["initial_box"]
    assert budget["steps"][:8] == short["steps"]
    assert budget["observations"][:8] == short["observations"]
    assert budget["steps"][7]["endpoint"] == short["endpoint"]
    budget_declaration = budget_record["protocol"]["protocol"]
    short_declaration = short_record["protocol"]["protocol"]
    for key in (
        "nodes",
        "edges",
        "original_phase_pi",
        "form",
        "capacity",
        "coordinate_order",
        "initial_form_coordinates",
        "initial_phase_coordinates_pi",
        "initial_interpretation",
        "phase_frame",
        "model",
        "clock",
        "time_step",
        "order",
        "interval_bits",
        "max_endpoint_width",
        "picard_policy",
        "comparison_policy",
        "failure_policy",
    ):
        assert budget_declaration[key] == short_declaration[key]


def test_retained_budget_all_sixty_four_tubes_preserve_picard_inclusion_and_chains(
    retained_budget,
):
    _assert_picard_and_chains(retained_budget, horizon=Q(1))
    _, _, certificate, steps = retained_budget
    assert len(steps) == 64
    assert _rational(certificate["validated_horizon"]) == 1
    assert certificate["status"] == "admitted"
    assert certificate["unavailable_reasons"] == []
    assert certificate["failed_tube"] is None


def test_retained_budget_storage_and_loss_keep_the_same_exact_accounting(
    retained_budget,
):
    _assert_storage_and_observations(retained_budget)


def _assert_budget_classification(
    bundle, *, horizon, expected_sign, first_negative_time
):
    _, record, certificate, steps = bundle
    response, declaration = record["response"], record["protocol"]["protocol"]
    budget = _interval(certificate["target_budget"])
    sign = "negative" if budget.hi < 0 else "positive" if budget.lo > 0 else "undecided"
    negative_times = [
        _rational(observation["time"])
        for observation in certificate["observations"]
        if _interval(observation["target_budget"]).hi < 0
    ]
    first_negative = min(negative_times, default=None)
    assert sign == expected_sign
    assert first_negative == first_negative_time
    assert response["budget_sign_at_validated_horizon"] == sign
    recorded_first = response["first_certified_target_exclusion_time"]
    assert (
        None if recorded_first is None else _rational(recorded_first)
    ) == first_negative
    assert response["future_target_excluded_from_validated_horizon"] is (
        sign == "negative"
    )
    width = max(value.width for value in _box(certificate["endpoint"]))
    expected_gates = {
        "horizon_admitted": _rational(certificate["validated_horizon"]) == horizon
        and not certificate["unavailable_reasons"],
        "endpoint_width_budget": width <= _rational(declaration["max_endpoint_width"]),
        "strict_budget_sign": sign != "undecided",
    }
    assert response["gates"] == expected_gates
    assert record["passed"] is response["passed"] is all(expected_gates.values())
    assert _rational(response["max_endpoint_width"]) == width
    assert response["target_budget_verdict"] == (
        "target_excluded" if sign == "negative" else "target_not_excluded_by_storage"
    )
    assert (
        response["passed_means"]
        == declaration["passed_means"]
        == "full_horizon_and_width_admitted_with_strict_budget_sign_not_formation"
    )
    assert response["scope"] == declaration["scope"]
    argument_change = _interval(certificate["endpoint_source_argument"]) - _interval(
        certificate["initial_source_argument"]
    )
    assert _interval(response["source_argument_change"]) == argument_change
    assert certificate["source_winding_one_throughout"] is all(
        _winding(tube, receiver=False) for _, tube, _, _, _ in steps
    )
    assert certificate["receiver_winding_zero_throughout"] is all(
        _winding(tube, receiver=True) for _, tube, _, _, _ in steps
    )


def test_retained_budget_classification_is_not_a_positive_formation_verdict(
    retained_budget,
):
    _assert_budget_classification(
        retained_budget,
        horizon=Q(1),
        expected_sign="positive",
        first_negative_time=None,
    )


def test_retained_endpoint_source_is_nonacute_despite_its_preserved_winding(
    retained_budget,
):
    _, _, certificate, steps = retained_budget
    endpoint = _box(certificate["endpoint"])
    pi = pi_interval()
    source_port_gap = 2 * pi - 2 * endpoint[4]
    assert source_port_gap.lo > pi.hi / 2
    assert source_port_gap.hi < pi.lo
    assert _winding(endpoint, receiver=False)
    assert _winding(endpoint, receiver=True)
    assert steps[-1][4].continuous_loss.lo > 0
    # This is the first retained endpoint that certifies nonacuteness, not
    # the exact continuous boundary-crossing time between the saved states.
    certified_times = [
        _rational(step["time"]) + _rational(step["duration"])
        for step, _, point, _, _ in steps
        if (2 * pi - 2 * point[4]).lo > pi.hi / 2
    ]
    assert min(certified_times) == Q(49, 64)


def test_extended_budget_has_a_separate_frozen_bundle_and_unchanged_first_64_steps(
    retained_extended_budget, retained_budget
):
    _assert_provenance(retained_extended_budget)
    _, record, certificate, _ = retained_extended_budget
    _, earlier_record, earlier, _ = retained_budget
    assert record["protocol_sha256"] == EXTENDED_BUDGET_DIGESTS["protocol"]
    assert record["source_archive_sha256"] == EXTENDED_BUDGET_DIGESTS["archive"]
    assert "correction" not in record["protocol"]
    declaration = record["protocol"]["protocol"]
    previous = earlier_record["protocol"]["protocol"]
    assert {key: value for key, value in declaration.items() if key != "horizon"} == {
        key: value for key, value in previous.items() if key != "horizon"
    }
    assert _rational(declaration["horizon"]) == Q(9, 8)
    assert _rational(previous["horizon"]) == 1
    assert certificate["initial_box"] == earlier["initial_box"]
    assert certificate["steps"][:64] == earlier["steps"]
    assert certificate["observations"][:64] == earlier["observations"]
    assert certificate["steps"][63]["endpoint"] == earlier["endpoint"]


def test_extended_budget_all_72_tubes_have_strict_picard_inclusion_and_chains(
    retained_extended_budget,
):
    _assert_picard_and_chains(retained_extended_budget, horizon=Q(9, 8))
    _, _, certificate, steps = retained_extended_budget
    assert len(steps) == 72
    assert _rational(certificate["validated_horizon"]) == Q(9, 8)
    assert certificate["status"] == "admitted"
    assert certificate["unavailable_reasons"] == []
    assert certificate["failed_tube"] is None


def test_extended_budget_retains_exact_storage_and_integrated_loss_accounting(
    retained_extended_budget,
):
    _assert_storage_and_observations(retained_extended_budget)


def test_extended_budget_success_certifies_exclusion_not_pattern_formation(
    retained_extended_budget,
):
    _assert_budget_classification(
        retained_extended_budget,
        horizon=Q(9, 8),
        expected_sign="negative",
        first_negative_time=Q(71, 64),
    )
    _, record, certificate, _ = retained_extended_budget
    budget = _interval(certificate["target_budget"])
    assert Q("-0.010524533169") < budget.lo <= budget.hi < Q("-0.010524532905")
    assert certificate["source_winding_one_throughout"] is True
    assert certificate["receiver_winding_zero_throughout"] is True
    assert record["passed"] is True
    assert record["response"]["future_target_excluded_from_validated_horizon"] is True
    # A pass means successful conditional discrimination, here a negative
    # formation result. Whole-time receiver winding zero excludes earlier
    # entry; the negative remaining storage excludes later acute two-twist entry.


def test_retrospective_separator_theorem_is_separate_from_frozen_target_budget(
    retained_budget,
):
    path, record, _, steps = retained_budget
    paths = (path, path.with_suffix(".protocol.json"), path.with_suffix(".source.zip"))
    original_digests = tuple(
        hashlib.sha256(item.read_bytes()).hexdigest() for item in paths
    )
    model = RelationalExchangeModel(phase_domain="regular", **COEFFICIENTS)
    before = certify_relational_reflected_barrier(steps[49][2], model=model)
    after = certify_relational_reflected_barrier(steps[50][2], model=model)
    assert _rational(steps[49][0]["time"]) + Q(1, 64) == Q(50, 64)
    assert _rational(steps[50][0]["time"]) + Q(1, 64) == Q(51, 64)
    assert before.obstructed is False
    assert before.energy_deficit.hi < 0
    assert "storage_not_strictly_below_separating_barrier" in before.unavailable_reasons
    assert after.obstructed is True
    assert after.unavailable_reasons == ()
    assert after.storage == steps[50][4].storage
    assert after.barrier_storage == I(7)
    assert (
        Q("0.00063") < after.energy_deficit.lo <= after.energy_deficit.hi < Q("0.00064")
    )
    assert after.separator_margin.lo > 0
    assert min(after.domain_lower_bounds) > 0
    # These already reconstructed endpoint fields establish that no earlier
    # saved endpoint meets the strict energy premise. No trajectory is replayed.
    assert all(field.storage.lo > 7 for _, _, _, _, field in steps[:50])
    assert (
        "snapshot_theorem_not_new_trajectory_or_physical_identification" in after.scope
    )
    assert (
        "no_exclusion_of_single_receiver_formation_nonacute_patterns_or_other_preparations"
        in after.scope
    )
    # The geometric theorem was added after this response was frozen. Its
    # earlier exclusion is retrospective; the original energy-only result
    # at T=1 remains positive and is not rewritten into a different prediction.
    assert record["passed"] is True
    assert record["response"]["budget_sign_at_validated_horizon"] == "positive"
    assert (
        record["response"]["target_budget_verdict"] == "target_not_excluded_by_storage"
    )
    assert record["response"]["first_certified_target_exclusion_time"] is None
    assert (
        tuple(hashlib.sha256(item.read_bytes()).hexdigest() for item in paths)
        == original_digests
    )
