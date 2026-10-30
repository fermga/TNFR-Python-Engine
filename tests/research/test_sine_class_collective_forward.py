"""Forward policy and retained-prediction controls without scientific execution.

The prior artifact is decoded once. Its coefficient generation remains an
execution premise; these tests inspect admission and rebuilt endpoint arithmetic.
Synthetic bands exercise prospective decisions, never a reserved forward flow.
"""

from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.research import sine_class_collective_forward as owner

ROOT = Path(__file__).resolve().parents[2]
EPS, G, H, PULSE = Q(1, 10**32), Q(1, 3000), Q(1), Q(7, 10000)
SOURCE = EPS / (1 - 2 * G * H)
DELTA, RADIUS, RESOLUTION = Q(1, 10**8), Q(1, 10**12), Q(1, 10**10)
SMALL = Q(1, 2**600)


@pytest.fixture(scope="module", autouse=True)
def no_coefficient_flow_or_worker_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def prior(no_coefficient_flow_or_worker_execution):
    reports, reads = [], []
    reconstruct = owner._reconstruct_prediction_report
    read = owner.read_bytes_bounded

    def capture_report(report, mediator_class):
        reports.append(report)
        return reconstruct(report, mediator_class)

    def capture_bytes(path, **kwargs):
        data = read(path, **kwargs)
        reads.append(data)
        return data

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_reconstruct_prediction_report", capture_report)
        patch.setattr(owner, "read_bytes_bounded", capture_bytes)
        bands = owner.read_collective_prediction(ROOT)
    assert len(reads) == 1 and len(reports) == 2
    return SimpleNamespace(bands=bands, reports=reports, archive=reads[0])


def _change(value, path, replacement):
    """Copy only the changed branch; leave the module's retained record intact."""
    if not path:
        return replacement
    key, *rest = path
    if isinstance(value, dict):
        result = dict(value)
        result[key] = _change(value[key], rest, replacement)
        return result
    result = list(value)
    result[key] = _change(value[key], rest, replacement)
    return tuple(result) if isinstance(value, tuple) else result


def _prediction(center=Q(2), mediator_class=1, **changes):
    pair = (center, center)
    return replace(
        owner.CollectivePredictionBands(
            mediator_class,
            pair,
            pair,
            pair,
            (Q(0), Q(0)),
            (Q(0), Q(0)),
            Q(0),
            Q(0),
        ),
        **changes,
    )


def test_retained_reader_rebuilds_fidelity_and_independent_comparator(prior):
    d, ell = 1 - 2 * G**2 * H**2, 1 - 2 * G * H
    fifth = 256 * G**6 * PULSE**5 * H / (d * (1 - 4 * G**2 * PULSE**2))
    initial = 4 * G * H * EPS / (ell * d) * (G * PULSE / d + EPS / ell)
    for k, band in enumerate(prior.bands, 1):
        assert band.mediator_class == k
        lo, hi = band.nominal_cubic_bounds
        assert band.nominal_full_bounds == (lo - fifth, hi + fifth)
        assert band.actual_full_bounds == (
            lo - fifth - initial - SOURCE,
            hi + fifth + initial + SOURCE,
        )
        lower, upper = band.nominal_comparator_bounds
        assert band.actual_comparator_bounds == (lower - SOURCE, upper + SOURCE)
        assert band.numerical_radius < Q(6654, 10**27)
        assert band.comparator_numerical_radius < Q(8677, 10**34)
        assert band.actual_full_bounds[0] - band.actual_comparator_bounds[
            1
        ] - 2 * DELTA > Q(1, 40000)


@pytest.mark.parametrize(
    "path,value",
    [
        (("mediator_class",), True),
        (("order",), 32.0),
        (("horizon",), True),
        (("horizon",), float("nan")),
        (("initial_form_bounds", 0, 0), False),
        (("initial_phase_bounds", 0, 0), Q(0)),
        (("comparator_initial_bounds", 0, 1), float("inf")),
        (("port_impulse", 0), False),
        (("descriptor", "port_indices", 0), Q(4)),
        (("descriptor", "parameter_bounds", "degrees", 0), 2.0),
        (("descriptor", "parameter_bounds", "classes", 0), True),
        (("source_coordinates",), "absolute theta"),
    ],
)
def test_prior_primitives_are_readmitted_before_derived_values(prior, path, value):
    altered = _change(prior.reports[0], path, value)
    with pytest.raises(ValueError):
        owner._reconstruct_prediction_report(altered, 1)


@pytest.mark.parametrize(
    "field", ["nominal_endpoint_bounds", "comparator_nominal_endpoint_bounds"]
)
def test_cached_endpoint_cannot_replace_coefficient_reconstruction(prior, field):
    altered = _change(prior.reports[0], (field, 1), {"lo": Q(0), "hi": Q(0)})
    with pytest.raises(ValueError, match="cached"):
        owner._reconstruct_prediction_report(altered, 1)


def test_cached_error_and_verdict_metadata_are_not_consumed(prior):
    altered = dict(prior.reports[0])
    altered.update(
        nominal_time_tail_upper_bound=Q(0),
        linear_source_uniform_bound=Q(0),
        comparator_source_uniform_bound=Q(0),
        status="fabricated_pass",
    )
    assert owner._reconstruct_prediction_report(altered, 1) == prior.bands[0]


def test_prior_transport_tamper_rejects_before_record_reconstruction(
    prior, monkeypatch
):
    altered = bytearray(prior.archive)
    altered[-1] ^= 1
    monkeypatch.setattr(owner, "read_bytes_bounded", lambda *a, **k: bytes(altered))

    def forbidden(*args, **kwargs):
        pytest.fail("a changed transport reached coefficient reconstruction")

    monkeypatch.setattr(owner, "_reconstruct_prediction_report", forbidden)
    with pytest.raises(ValueError, match="artifact differs"):
        owner.read_collective_prediction(ROOT)


def test_reference_absolute_phase_and_actual_residual_covers_are_distinct():
    sources = owner.collective_forward_sources()
    pi = sources["pi_bounds"]
    assert sources["classes"] == ((1, 1, 1), (1, 2, 1))
    for index, classes in enumerate(sources["classes"]):
        for node in range(27):
            winding = classes[node // 9]
            factor = Q(2 * winding * (node % 9 - 4), 9)
            products = [factor * bound for bound in pi]
            target = (min(products), max(products))
            assert sources["reference_phase_bounds"][index][node] == target
            assert sources["actual_phase_bounds"][index][node] == (
                target[0] - EPS,
                target[1] + EPS,
            )
            assert sources["reference_form_bounds"][index][node] == (0, 0)
            assert sources["actual_form_bounds"][index][node] == (-EPS, EPS)
    assert sources["reference_phase_bounds"][1][9][1] < -1
    assert sources["reference_phase_bounds"][1][13] == (0, 0)


def test_fixed_numerical_policy_does_not_consume_a_prediction():
    inputs = owner.collective_forward_inputs()
    assert set(inputs) == {
        "initial_form_bounds",
        "initial_phase_bounds",
        "port_impulse",
        "horizon",
        "time_step",
        "order",
        "max_steps",
    }
    assert inputs["port_impulse"] == (0, PULSE, 0)
    assert inputs["horizon"] == 1 and inputs["time_step"] == Q(1, 16)
    assert type(inputs["order"]) is int and inputs["order"] == 12
    assert type(inputs["max_steps"]) is int and inputs["max_steps"] == 32
    assert 2 * inputs["horizon"] / inputs["time_step"] == inputs["max_steps"]
    policy = owner.collective_forward_policy()
    assert policy["linear_source_allowance"] == SOURCE
    assert policy["reference_radius_ceiling"] == RADIUS
    assert policy["actual_prediction_resolution_allowance"] == RESOLUTION
    assert policy["reading_error_per_model"] == DELTA


def test_forward_source_is_transported_once_without_amplitude_remainder(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("forward source transport must not add a cubic remainder")

    monkeypatch.setattr(owner, "_bound_collective_interface", forbidden)
    reference = (Q(2) - SMALL, Q(2) + SMALL)
    row = owner.assess_collective_forward_bounds(reference, _prediction())
    assert row["reference_bounds"] == reference
    assert row["actual_full_bounds"] == (reference[0] - SOURCE, reference[1] + SOURCE)
    assert row["recorded_full_bounds"] == (
        reference[0] - SOURCE - DELTA,
        reference[1] + SOURCE + DELTA,
    )
    assert row["recorded_comparator_bounds"] == (-DELTA, DELTA)


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_nominal_overlap_is_closed_and_does_not_clip_forward_band(side):
    reference = (Q(2), Q(2))
    prediction = _prediction(nominal_full_bounds=(Q(1), Q(2) + side * SMALL))
    row = owner.assess_collective_forward_bounds(reference, prediction)
    assert row["nominal_prediction_overlap"] is (side >= 0)
    assert row["reference_bounds"] == reference
    assert (row["status"] == "consistency_conflict") is (side < 0)


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_recorded_separation_requires_strict_margin_after_two_errors(side):
    center = SOURCE + 2 * DELTA + side * SMALL
    row = owner.assess_collective_forward_bounds((center, center), _prediction(center))
    assert row["recorded_separation_margin"] == side * SMALL
    assert row["comparator_separated"] is (side > 0)
    assert row["all_conditions_met"] is (side > 0)


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_reference_radius_is_a_separate_closed_acceptance_gate(side):
    radius = RADIUS + side * SMALL
    prediction = _prediction(
        nominal_full_bounds=(Q(1), Q(3)), actual_full_bounds=(Q(1), Q(3))
    )
    row = owner.assess_collective_forward_bounds((2 - radius, 2 + radius), prediction)
    assert row["comparator_separated"]
    assert row["reference_radius"] == radius
    assert row["reference_radius_within_budget"] is (side <= 0)
    assert (row["status"] == "numerically_unresolved") is (side > 0)


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_resolution_window_contains_whole_actual_band_or_abstains(side):
    prediction = _prediction(
        actual_full_bounds=(Q(1), Q(2) + SOURCE - RESOLUTION + side * SMALL)
    )
    row = owner.assess_collective_forward_bounds((Q(2), Q(2)), prediction)
    assert row["comparator_separated"] and row["nominal_prediction_overlap"]
    assert row["actual_prediction_resolution_met"] is (side >= 0)
    assert (row["status"] == "resolution_not_met") is (side < 0)
    assert row["actual_full_bounds"] == (2 - SOURCE, 2 + SOURCE)


@pytest.mark.parametrize(
    "bad", [(True, 2), (0.0, 2), (Q(2), Q(1)), (0, float("nan")), (0,), None]
)
def test_supplied_forward_band_has_exact_primitive_admission(bad):
    with pytest.raises(ValueError):
        owner.assess_collective_forward_bounds(bad, _prediction())


@pytest.mark.parametrize(
    "field", ["nominal_full_bounds", "actual_full_bounds", "actual_comparator_bounds"]
)
def test_supplied_prediction_bands_are_readmitted(field):
    with pytest.raises(ValueError):
        owner.assess_collective_forward_bounds(
            (Q(2), Q(2)), _prediction(**{field: (False, Q(3))})
        )


@pytest.mark.parametrize("complete", [False, True])
def test_report_wiring_consumes_reconstructed_evidence_not_status(
    monkeypatch, complete
):
    from tnfr.physics import _sine_class_port_readout_evidence as evidence_owner
    from tnfr.physics import relational_sine_class_port_readout as producer

    sentinel, report, admitted = object(), SimpleNamespace(status="fabricated_pass"), []
    evidence = SimpleNamespace(complete=complete, endpoint_bounds=(I(2), I(3)))

    def admit(**inputs):
        admitted.append(inputs)
        return sentinel

    def reconstruct(value, primitive):
        assert value is report and primitive is sentinel
        return evidence

    monkeypatch.setattr(producer, "_admit_port_readout_inputs", admit)
    monkeypatch.setattr(evidence_owner, "_reconstruct_port_readout", reconstruct)
    result = owner.assess_collective_forward_report(
        report, (_prediction(), _prediction(Q(3), 2))
    )
    assert admitted == [owner.collective_forward_inputs()]
    assert result["evidence"] is evidence
    if complete:
        assert result["status"] == "all_conditions_met"
        assert len(result["comparisons"]) == 2
        assert result["all_conditions_met"]
    else:
        assert result["status"] == "unavailable"
        assert result["comparisons"] is None
        assert not result["all_conditions_met"]


@pytest.mark.parametrize("classes", [(1, 1), (2, 1), (True, 2)])
def test_report_prediction_order_is_not_a_cached_label(classes):
    prediction = tuple(_prediction(mediator_class=k) for k in classes)
    with pytest.raises(ValueError, match="prediction order|ordinary integer"):
        owner.assess_collective_forward_report(object(), prediction)
