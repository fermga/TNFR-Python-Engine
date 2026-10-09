"""Read-only reconstruction of the first complete-cubic analytic certificate.

Stored time coefficients are retained execution premises. These controls rebuild
source/event association, polynomial endpoints, analytic tails and exact decisions;
they do not regenerate coefficient derivatives, a nonlinear flow or acquisition.
Hashes establish content association, not provenance authentication.
"""

import io
import subprocess
import zipfile
from datetime import datetime
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

import pytest

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    sha256_bytes,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/assets/sine_formed_classes/class-cubic-response-v1.evidence.zip"
INVENTORY = {
    "policy-v1.json": "f958646ea96ca8d850563e12b18e72ce0d51b8605967de208f29ffd9e1281245",
    "attempt-v1.json": "0c3c2a17e5ce673721e40adb9c572e60568dfec442ca3056f020032808ab8afb",
    "outcome-v1.json": "a56348c90862d0acecbedb610c6599b2de956dba11b7524f005989eb655dbe12",
    "evaluated-source-v1-complete.zip": "6f559fdc6d1fb69739915723e78ac9690701236328a34c23542ef35d935d5209",
    "evaluated-source-v1.zip": "f9e232b5b30d2df3dd5030ed6f9454cf3917cd0eba2eaeed63a2678923d440b5",
    "preparation-error-v1.txt": "4e994ca95be80f6818eaafaf3b9b34ee49f2e7bf110a3f93f9bc9fcc7f8114ad",
    "evaluate_once.py": "f651506d7471bc66e12da36f05118312edb4efc92bc5a5a857d399cddcd39423",
}
TRANSPORT_BYTES = 1048901
TRANSPORT_SHA256 = "991d771464d84f65a25c0a82c9b08f7e5a579e450885141b32fef1a1ed8239cb"


@pytest.fixture(scope="module", autouse=True)
def no_coefficient_response_or_worker_execution():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
    )
    from tnfr.physics import relational_sine_class_cubic_response as cubic
    from tnfr.physics import relational_sine_class_readout as readout

    def forbidden(*args, **kwargs):
        pytest.fail("retained cubic evidence must not regenerate coefficients or flows")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                cubic,
                (
                    "bound_sine_class_cubic_response",
                    "_class_cubic_coefficients",
                    "_coefficient_segment",
                    "_time_coefficients",
                    "_linear_variation",
                ),
            ),
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (_sine_flow, ("_full_sine_field", "_sine_rate_evaluator")),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (readout, ("bound_sine_class_four_history_readout", "_full_sine_field")),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


def _integer(value):
    assert type(value) is int and value >= 0
    return value


def _interval(value):
    assert isinstance(value, dict) and set(value) == {"lo", "hi"}
    lo, hi = map(exact_record, (value["lo"], value["hi"]))
    assert lo <= hi
    interval = I(lo, hi)
    assert (interval.lo, interval.hi) == (lo, hi)
    return interval


def _pair(value):
    assert len(value) == 2
    lo, hi = map(exact_record, value)
    assert lo <= hi
    return lo, hi


def _levels(value):
    assert len(value) == 3 and all(len(row) == 54 for row in value)
    return tuple(tuple(map(_interval, row)) for row in value)


def _tail(z, m):
    assert 0 <= z < m + 1
    return z**m / factorial(m) / (1 - z / (m + 1))


def _event(levels, amplitude):
    return (
        tuple(
            v + amplitude if i == 4 and amplitude else v
            for i, v in enumerate(levels[0])
        ),
        *levels[1:],
    )


def _audit_segment(segment, before, *, label, parent, start, end, jump, eta):
    assert segment["label"] == label
    stored_parent = segment["parent_segment_index"]
    assert (
        stored_parent is None if parent is None else _integer(stored_parent) == parent
    )
    assert tuple(
        map(
            exact_record,
            (segment["start_time"], segment["end_time"], segment["form_jump"]),
        )
    ) == (start, end, jump)
    initial = _levels(segment["initial_levels"])
    assert initial == _event(before, jump)
    norms = tuple(max(v.abs_max for v in row) for row in initial)
    assert tuple(map(exact_record, segment["initial_level_norm_upper_bounds"])) == norms
    v1, v2, v3 = norms
    h, rate = end - start, Q(201, 100)
    tails = (
        v1 * _tail(rate * h, 65),
        v2 * _tail(2 * rate * h, 65) + 2 * v1**2 * h * _tail(2 * rate * h, 64),
        v3 * _tail(3 * rate * h, 65)
        + (4 * eta * v1 * v2 + Q(4, 3) * v1**3) * h * _tail(3 * rate * h, 64)
        + 4 * eta * v1**3 * h**2 * _tail(3 * rate * h, 63),
    )
    assert tuple(map(exact_record, segment["time_tail_upper_bounds"])) == tails
    assert len(segment["time_coefficients"]) == 3
    endpoints = []
    for index, records in enumerate(segment["time_coefficients"]):
        assert len(records) == 65 and all(len(row) == 54 for row in records)
        rows = tuple(tuple(map(_interval, row)) for row in records)
        assert rows[0] == initial[index]
        value = rows[-1]
        for row in reversed(rows[:-1]):
            value = tuple(v * h + c for v, c in zip(value, row))
        endpoints.append(tuple(v + I(-tails[index], tails[index]) for v in value))
    result = tuple(endpoints)
    assert result == _levels(segment["endpoint_levels"])
    return result


@pytest.fixture(scope="module")
def retained():
    receipt = file_receipt(ARCHIVE, max_bytes=2**27)
    assert receipt["bytes"] == TRANSPORT_BYTES and receipt["sha256"] == TRANSPORT_SHA256
    verify_archive_members(ARCHIVE, INVENTORY, max_bytes=2**27)
    with zipfile.ZipFile(ARCHIVE) as archive:
        raw = {name: archive.read(name) for name in INVENTORY}
    decoded = {
        name: decode_exact_tree(json_loads(data))
        for name, data in raw.items()
        if name.endswith(".json")
    }
    return raw, decoded


def test_exact_transport_policy_attempt_and_evaluated_source_association(retained):
    raw, decoded = retained
    policy, attempt, outcome = (
        decoded[name]
        for name in ("policy-v1.json", "attempt-v1.json", "outcome-v1.json")
    )
    assert policy["schema"] == "tnfr.sine-class-cubic-response-policy.v1"
    assert policy["status_at_declaration"] == "not_evaluated"
    assert (
        policy["base_revision_before_implementation"]
        == "276e2108a2980a503099c873f046ae5801c27fab"
    )
    assert policy["adaptive_retries"] is False
    assert _integer(policy["time_polynomial_order"]) == 64
    assert _integer(policy["interval_bits"]) == 128
    assert _integer(policy["segment_count"]) == 8
    assert tuple(map(_integer, policy["tail_first_omitted_indices"])) == (65, 64, 63)
    assert exact_record(policy["linear_norm_bound"]) == Q(201, 100)
    assert exact_record(policy["cauchy_gamma_upper_bound"]) == Q(1, 3000)
    assert "For each history separately" in policy["cauchy_radius_rule"]
    assert attempt["schema"] == "tnfr.sine-class-cubic-response-attempt.v1"
    assert datetime.fromisoformat(attempt["started_utc"]).tzinfo is not None
    assert attempt["platform"] and attempt["python"]
    assert outcome["schema"] == "tnfr.sine-class-cubic-response-outcome.v1"
    assert outcome["error"] is None
    for receipt, name in (
        (attempt["policy"], "policy-v1.json"),
        (attempt["evaluated_source_archive"], "evaluated-source-v1-complete.zip"),
        (outcome["attempt"], "attempt-v1.json"),
    ):
        assert _integer(receipt["bytes"]) == len(raw[name])
        assert receipt["sha256"] == sha256_bytes(raw[name])
    expected = {name: r["sha256"] for name, r in attempt["evaluated_runtime"].items()}
    expected.update(
        {
            name: sha256_bytes(raw[name])
            for name in ("policy-v1.json", "evaluate_once.py")
        }
    )
    verify_archive_members(
        io.BytesIO(raw["evaluated-source-v1-complete.zip"]), expected, max_bytes=2**20
    )
    with zipfile.ZipFile(
        io.BytesIO(raw["evaluated-source-v1-complete.zip"])
    ) as archive:
        for name, receipt in attempt["evaluated_runtime"].items():
            assert _integer(receipt["bytes"]) == len(archive.read(name))
        assert b"lru_cache" not in archive.read(
            "src/tnfr/physics/relational_sine_class_cubic_response.py"
        )
    assert b"before attempt-ledger creation" in raw["preparation-error-v1.txt"]
    with zipfile.ZipFile(io.BytesIO(raw["evaluated-source-v1.zip"])) as partial:
        assert "src/tnfr/_exact_time.py" not in partial.namelist()
        assert "attempt-v1.json" not in partial.namelist()


@pytest.fixture(scope="module")
def evidence(retained):
    _, decoded = retained
    policy = decoded["policy-v1.json"]
    output = decoded["outcome-v1.json"]["output"]
    assert output["schema"] == "tnfr.sine-class-cubic-response.v1"
    report = output["report"]
    inputs = {key: exact_record(value) for key, value in policy["inputs"].items()}
    assert inputs == dict(
        first_probe_amplitude=Q(1, 2000),
        second_probe_amplitude=Q(1, 2000),
        delay=Q(1),
        total_duration=Q(2),
        endpoint_radius=Q(1, 10**32),
        readout_error_bound=Q(1, 10**30),
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        first_probe_work_allowance=Q(1, 500000),
        second_probe_work_allowance=Q(1, 500000),
    )
    assert {key: exact_record(report[key]) for key in inputs} == inputs
    assert _integer(report["time_polynomial_order"]) == 64
    assert exact_record(report["linear_majorant_rate"]) == Q(201, 100)
    assert (
        report["method"]
        == "complete_amplitude_cubic_time_polynomial64_dyadic128_majorant_v1"
    )
    assert report["clock"] == "tau=e*t; e=1023/1024"
    nodes = tuple(map(_integer, report["nodes"]))
    edges = tuple(tuple(map(_integer, edge)) for edge in report["edges"])
    expected_edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(3) for j in range(9)
    ) + ((4, 13), (13, 22))
    assert nodes == tuple(range(27)) and edges == expected_edges
    degrees = tuple(sum(i in edge for edge in edges) for i in nodes)
    gamma = 1 / (1023 * pi_interval())
    assert _interval(report["gamma_bounds"]) == gamma
    assert len(report["class_parameters"]) == len(report["class_segments"]) == 2
    class_mixed = []
    for k, parameters, segments in zip(
        (1, 2), report["class_parameters"], report["class_segments"]
    ):
        classes = (1, k, 1)
        assert tuple(map(_integer, parameters["classes"])) == classes
        assert tuple(map(_integer, parameters["degrees"])) == degrees
        assert _interval(parameters["gamma"]) == gamma
        eta = _interval(parameters["eta"])
        assert eta == gamma**2
        assert (
            len(parameters["edge_sines"])
            == len(parameters["edge_cosines"])
            == len(edges)
        )
        target = tuple(Q(classes[c] * (j - 4), 9) for c in range(3) for j in range(9))
        for index, (left, right) in enumerate(edges):
            gap = target[right] - target[left]
            gap -= (gap + Q(1, 2)).__floor__()
            expected_sine = sin(2 * gap * pi_interval()) if gap else I(0)
            expected_cosine = cos(2 * gap * pi_interval()) if gap else I(1)
            assert _interval(parameters["edge_sines"][index]) == expected_sine
            assert _interval(parameters["edge_cosines"][index]) == expected_cosine
        assert len(segments) == 4
        zero = ((I(0),) * 54,) * 3
        prefix = _audit_segment(
            segments[0],
            zero,
            label="first_prefix",
            parent=None,
            start=Q(0),
            end=Q(1),
            jump=Q(1, 2000),
            eta=eta.hi,
        )
        first = _audit_segment(
            segments[1],
            prefix,
            label="first_suffix",
            parent=0,
            start=Q(1),
            end=Q(2),
            jump=Q(0),
            eta=eta.hi,
        )
        both = _audit_segment(
            segments[2],
            prefix,
            label="both_suffix",
            parent=0,
            start=Q(1),
            end=Q(2),
            jump=Q(1, 2000),
            eta=eta.hi,
        )
        second = _audit_segment(
            segments[3],
            zero,
            label="second_only_suffix",
            parent=None,
            start=Q(1),
            end=Q(2),
            jump=Q(1, 2000),
            eta=eta.hi,
        )
        class_mixed.append(both[2][22] - first[2][22] - second[2][22])
    return inputs, report, gamma, tuple(class_mixed)


@pytest.mark.parametrize("tamper", ("parent", "source", "tail", "endpoint"))
def test_retained_arithmetic_rejects_changed_ancestry_source_or_cached_bounds(
    retained, tamper
):
    _, decoded = retained
    report = decoded["outcome-v1.json"]["output"]["report"]
    prefix, first, _, _ = report["class_segments"][0]
    segment = dict(first)
    before = _levels(prefix["endpoint_levels"])
    if tamper == "parent":
        segment["parent_segment_index"] = False
    elif tamper == "source":
        levels = list(segment["initial_levels"])
        first_level = list(levels[0])
        first_level[4] = {"lo": Q(1), "hi": Q(1)}
        levels[0] = tuple(first_level)
        segment["initial_levels"] = tuple(levels)
    elif tamper == "tail":
        segment["time_tail_upper_bounds"] = (Q(0),) * 3
    else:
        levels = list(segment["endpoint_levels"])
        third_level = list(levels[2])
        third_level[22] = {"lo": Q(1), "hi": Q(1)}
        levels[2] = tuple(third_level)
        segment["endpoint_levels"] = tuple(levels)
    with pytest.raises(AssertionError):
        _audit_segment(
            segment,
            before,
            label="first_suffix",
            parent=0,
            start=Q(1),
            end=Q(2),
            jump=Q(0),
            eta=_interval(report["class_parameters"][0]["eta"]).hi,
        )


def test_complete_time_evidence_and_exact_class_subtraction(evidence):
    _, report, gamma, mixed = evidence

    def scale(value):
        values = [g**4 * v for g in (gamma.lo, gamma.hi) for v in (value.lo, value.hi)]
        return min(values), max(values)

    assert tuple(map(_pair, report["scaled_class_cubic_bounds"])) == tuple(
        map(scale, mixed)
    )
    assert _pair(report["complete_cubic_contrast_bounds"]) == scale(mixed[0] - mixed[1])
    assert (
        report["coefficient_evaluated"] is True
        and report["exact_contrast_zero"] is False
    )


def test_amplitude_tail_source_error_and_every_observation_decision(evidence):
    inputs, report, gamma, mixed = evidence
    a, b, t = (
        inputs["first_probe_amplitude"],
        inputs["second_probe_amplitude"],
        inputs["total_duration"],
    )
    eps, delta, g = inputs["endpoint_radius"], inputs["readout_error_bound"], Q(1, 3000)
    bootstrap, radius = 1 - 16 * g * g * t * t, 1 - 2 * g * (abs(a) + abs(b))
    assert exact_record(report["cauchy_bootstrap_margin"]) == bootstrap > 0
    assert exact_record(report["cauchy_radius_margin"]) == radius > 0
    assert report["cauchy_admitted"] is True
    remainders = tuple(
        256 * g**6 * A**5 * t / ((1 - 2 * g * g * t * t) * (1 - 4 * g * g * A * A))
        for A in (Q(0), abs(a), abs(b), abs(a) + abs(b))
    )
    higher = 2 * sum(remainders)
    source = 8 * eps / (1 - 2 * gamma.hi * t)
    assert (
        tuple(
            map(
                exact_record,
                report["per_history_higher_amplitude_remainder_upper_bounds"],
            )
        )
        == remainders
    )
    assert exact_record(report["higher_amplitude_contrast_error_upper_bound"]) == higher
    assert exact_record(report["source_contrast_error_upper_bound"]) == source
    values = [
        g0**4 * v
        for g0 in (gamma.lo, gamma.hi)
        for v in ((mixed[0] - mixed[1]).lo, (mixed[0] - mixed[1]).hi)
    ]
    true = min(values) - higher - source, max(values) + higher + source
    recorded = true[0] - 8 * delta, true[1] + 8 * delta
    decision = report["decision"]
    assert _pair(decision["true_bounds"]) == true
    assert _pair(decision["recorded_bounds"]) == recorded
    assert type(decision["orientation"]) is int and decision["orientation"] == -1
    lower = -true[1]
    for key, value in (
        ("oriented_lower", lower),
        ("recorded_sign_margin", lower - 8 * delta),
        ("null_separation_margin", lower - 16 * delta),
        ("noise_ceiling", lower / 8),
        ("cancellation_margin", 8 * delta - max(map(abs, true))),
    ):
        assert exact_record(decision[key]) == value
    for key, value in (
        ("true_sign", lower > 0),
        ("recorded_sign", lower > 8 * delta),
        ("null_excluded", lower > 16 * delta),
        ("scalar_cancellation", max(map(abs, true)) <= 8 * delta),
    ):
        assert decision[key] is value
    assert report["status"] == decision["status"] == "true_sign_certified"
    assert (
        report["response_bound_available"] is True
        and report["unavailable_reasons"] == ()
    )
    assert Q(-7096, 10**33) < true[0] < true[1] < Q(-6932, 10**33)
    assert recorded[0] < 0 < recorded[1]
    # This is an allowed error for the scalar statistic at every true value,
    # not a common observation vector for the original eight readings.
    assert max(map(abs, true)) / 8 <= delta


def test_work_and_identity_are_rebuilt_separately(evidence):
    values, report, gamma, _ = evidence
    eps, r, g, s, t = (
        values["endpoint_radius"],
        values["radius"],
        gamma.hi,
        values["delay"],
        values["total_duration"],
    )
    initial_storage, barrier = 22 * eps**2, r**2 / 2700
    initial = 6 * eps**2 < r * r and initial_storage < barrier
    contact = 8 * eps**2 <= values["contact_work_allowance"]
    assert report["joined_initial_identity_certified"] is initial
    assert report["contact_work_within_allowance"] is contact

    def envelope(A, h):
        x = (A + eps + 2 * g * h * eps) / (1 - 2 * g * g * h * h)
        return x, eps + 2 * g * h * x

    identities, works = [], []
    assert len(report["uniform_history_bounds"]) == 4
    for history, (label, a, b) in zip(
        report["uniform_history_bounds"],
        (
            ("neither", Q(0), Q(0)),
            ("first_only", Q(1, 2000), Q(0)),
            ("second_only", Q(0), Q(1, 2000)),
            ("both", Q(1, 2000), Q(1, 2000)),
        ),
    ):
        assert history["label"] == label
        assert exact_record(history["first_probe_amplitude"]) == a
        assert exact_record(history["second_probe_amplitude"]) == b
        x, y = envelope(abs(a) + abs(b), t)
        x1, y1 = envelope(abs(a), t)
        pre, pre_y = envelope(abs(a), s)
        work1 = Q(3, 2) * a * a + 6 * abs(a) * eps
        work2 = Q(3, 2) * b * b + 6 * abs(b) * pre
        assert exact_record(history["pre_second_form_coordinate_upper_bound"]) == pre
        assert exact_record(history["pre_second_phase_deviation_upper_bound"]) == pre_y
        assert exact_record(history["maximum_form_coordinate_upper_bound"]) == x
        assert exact_record(history["maximum_phase_deviation_upper_bound"]) == y
        assert exact_record(history["first_probe_work_upper_bound"]) == work1
        assert exact_record(history["second_probe_work_upper_bound"]) == work2
        identity1 = (
            initial
            and 27 * (x1 * x1 + y1 * y1) < r * r
            and initial_storage + work1 < barrier
        )
        identity2 = (
            identity1
            and 27 * (x * x + y * y) < r * r
            and initial_storage + work1 + work2 < barrier
        )
        work = (
            contact
            and work1 <= values["first_probe_work_allowance"]
            and work2 <= values["second_probe_work_allowance"]
        )
        assert history["first_probe_identity_certified"] is identity1
        assert history["second_probe_identity_certified"] is identity2
        assert history["identity_certified"] is identity2
        assert history["work_within_allowances"] is work
        assert history["tangent_endpoint_discrepancy_upper_bound"] is None
        identities.append(identity2)
        works.append(work)
    assert report["all_identities_certified"] is all(identities)
    assert report["all_work_within_allowances"] is all(works)
