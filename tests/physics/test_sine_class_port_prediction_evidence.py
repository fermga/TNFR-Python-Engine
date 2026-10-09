"""Read-only audit of the first changed-input causal port prediction.

Retained causal coefficient generation remains an execution premise. These
controls re-admit the fixed model/source, reconstruct all endpoint arithmetic
and error budgets, and check the declared observation. No coefficient, memory
kernel, full-law response or archived worker is regenerated. Hashes associate
bytes; they do not authenticate acquisition or execution.
"""

import ast
import io
import zipfile
from datetime import datetime
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
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
ARCHIVE = (
    ROOT / "docs/assets/sine_formed_classes/class-collective-prediction-v1.evidence.zip"
)
MEMBERS = {
    "design.json",
    "protocol.json",
    "source.zip",
    "freeze.json",
    "attempt.json",
    "outcome.json",
}
TRANSPORT_BYTES = 1016399
TRANSPORT_SHA256 = "01b443211665c027bcd5b86ace8a22a4e7ae051f34d172d5520644e6225ba16a"
INVENTORY = {
    "design.json": "0c8fc1b178f4b86b173fa45b597cbf6896120a0b6689b9441fc199ae0bfa4652",
    "protocol.json": "f195980d12a99510f3a38166976f015ebe20c9952b985f808f6b0090a2ab2ac4",
    "source.zip": "fa3369a2be0ee776253ed52f82f5e1462a70b636db4b741614eb886f03eceea5",
    "freeze.json": "f98b958d716bf9cef83c729f4d277fba1ce95ffdbdf0bb972b5fba56dfa94222",
    "attempt.json": "e484509a6d47f658ea54382fddf2bbff9c1e487599fc611be881642568ce3843",
    "outcome.json": "c2417e2c9a4faf2ceda0df9bf75da795e80ed963abe4ac1927d4b3159c6fd336",
}
BASE = "5b9e721dc6240ea2d21d0690e1cdaa9b4d5b318d"
SOURCE_MEMBERS = {
    "src/tnfr/physics/_sine_class_port_prediction.py",
    "src/tnfr/research/sine_class_collective_protocol.py",
    "theory/nodal/SINE_CLASS_CHANGED_INPUT_PREDICTION.md",
    "evaluate_prediction.py",
}
PORTS = (4, 13, 22, 31, 40, 49)
EDGES = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(3) for j in range(9)) + (
    (4, 13),
    (13, 22),
)
EPS, PULSE, HORIZON = Q(1, 10**32), Q(7, 10000), Q(1)
GAMMA_UPPER, RATE = Q(1, 3000), Q(201, 100)
DELTA, NUMERICAL_RADIUS = Q(1, 10**8), Q(1, 10**12)


@pytest.fixture(scope="module", autouse=True)
def no_prediction_response_or_worker_execution():
    with forbid_sine_regeneration():
        yield


def _integer(value):
    assert type(value) is int
    return value


def _interval(value):
    assert isinstance(value, dict) and set(value) == {"lo", "hi"}
    lo, hi = exact_record(value["lo"]), exact_record(value["hi"])
    assert lo <= hi
    result = I(lo, hi)
    assert (result.lo, result.hi) == (lo, hi)
    return result


def _vector(values):
    return tuple(map(_interval, values))


def _pairs(values):
    result = tuple(tuple(map(exact_record, row)) for row in values)
    assert all(len(row) == 2 and row[0] <= row[1] for row in result)
    return result


def _input(record):
    assert set(record) == {
        "mediator_class",
        "initial_form_bounds",
        "initial_phase_bounds",
        "comparator_initial_bounds",
        "port_impulse",
        "horizon",
    }
    return {
        "mediator_class": _integer(record["mediator_class"]),
        "initial_form_bounds": _pairs(record["initial_form_bounds"]),
        "initial_phase_bounds": _pairs(record["initial_phase_bounds"]),
        "comparator_initial_bounds": _pairs(record["comparator_initial_bounds"]),
        "port_impulse": tuple(map(exact_record, record["port_impulse"])),
        "horizon": exact_record(record["horizon"]),
    }


def _series(values, size):
    assert len(values) == 33 and all(len(row) == size for row in values)
    return tuple(map(_vector, values))


def _endpoint(series, indices):
    result = tuple(series[-1][i] for i in indices)
    for row in reversed(series[:-1]):
        result = tuple(v * HORIZON + row[i] for v, i in zip(result, indices))
    return result


def _tail(argument, first):
    assert 0 <= argument < first + 1
    return argument**first / factorial(first) / (1 - argument / (first + 1))


@pytest.fixture(scope="module")
def retained():
    before = file_receipt(ARCHIVE, max_bytes=32 * 1024**2)
    assert before["bytes"] == TRANSPORT_BYTES
    assert before["sha256"] == TRANSPORT_SHA256
    verify_archive_members(ARCHIVE, INVENTORY, max_bytes=64 * 1024**2)
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert set(archive.namelist()) == MEMBERS
        raw = {name: archive.read(name) for name in MEMBERS}
    assert len(raw["outcome.json"]) == 36871908
    decoded = {
        name: decode_exact_tree(json_loads(data))
        for name, data in raw.items()
        if name.endswith(".json")
    }
    yield raw, decoded
    assert file_receipt(ARCHIVE, max_bytes=32 * 1024**2) == before


def test_exclusive_attempt_and_prediction_only_scope(retained):
    raw, decoded = retained
    attempt, outcome = decoded["attempt.json"], decoded["outcome.json"]
    assert attempt["schema"] == "tnfr.sine-class-collective-prediction-attempt.v1"
    assert datetime.fromisoformat(attempt["started_utc"]).tzinfo is not None
    assert attempt["reserved_full_law_evaluation"] is False
    assert attempt["freeze_sha256"] == sha256_bytes(raw["freeze.json"])
    assert outcome["schema"] == "tnfr.sine-class-collective-prediction-outcome.v1"
    assert outcome["full_law_response"] == "not_evaluated"
    assert outcome["status"] == "complete"
    assert outcome["attempt"]["bytes"] == len(raw["attempt.json"])
    assert outcome["attempt"]["sha256"] == sha256_bytes(raw["attempt.json"])
    assert len(outcome["reports"]) == 2


def test_full_base_overlays_receipts_and_prospective_proof(retained):
    raw, decoded = retained
    freeze, protocol = decoded["freeze.json"], decoded["protocol.json"]
    assert freeze["schema"] == "tnfr.sine-class-collective-prediction-freeze.v1"
    assert protocol["schema"] == "tnfr.sine-class-collective-prediction-protocol.v1"
    for record in (freeze, protocol):
        assert record["source_base_commit"] == BASE
        assert record["status_at_freeze"] == "causal_coefficients_not_evaluated"
    assert {record["path"] for record in freeze["artifacts"]} == {
        "design.json",
        "protocol.json",
        "source.zip",
    }
    for receipt in (
        *freeze["artifacts"],
        protocol["design"],
        protocol["source_archive"],
    ):
        data = raw[receipt["path"]]
        assert _integer(receipt["bytes"]) == len(data)
        assert receipt["sha256"] == sha256_bytes(data)
    assert "not absolute phase lifts" in protocol["source_coordinate_declaration"]
    assert "not independently preparable" in protocol["source_cover_declaration"]
    assert protocol["gap_orientation"] == (
        "full-model x13 minus grounded-comparator x13, separately in each class"
    )
    assert protocol["source_owner"] == (
        "theory/nodal/SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source"
    )
    assert "formation time100" in protocol["preparation_premise"]
    assert "unprobed dwell1e13" in protocol["preparation_premise"]
    assert "guarded exponent512" in protocol["preparation_premise"]
    with zipfile.ZipFile(io.BytesIO(raw["source.zip"])) as archive:
        assert set(archive.namelist()) == SOURCE_MEMBERS | {"manifest.json"}
        source = {name: archive.read(name) for name in archive.namelist()}
    manifest = json_loads(source["manifest.json"])
    assert manifest["schema"] == "tnfr.sine-class-collective-prediction-source.v1"
    assert manifest["source_base_commit"] == BASE
    assert set(manifest["runtime_overlays"]) == {
        "src/tnfr/physics/_sine_class_port_prediction.py",
        "src/tnfr/research/sine_class_collective_protocol.py",
    }
    assert tuple(manifest["runtime_overlays"]) == protocol["runtime_overlays"]
    assert {row["path"] for row in manifest["files"]} == SOURCE_MEMBERS
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    expected["manifest.json"] = sha256_bytes(source["manifest.json"])
    verify_archive_members(
        io.BytesIO(raw["source.zip"]), expected, max_bytes=32 * 1024**2
    )
    for receipt in manifest["files"]:
        assert _integer(receipt["bytes"]) == len(source[receipt["path"]])
    proof_path = "theory/nodal/SINE_CLASS_CHANGED_INPUT_PREDICTION.md"
    archived = source[proof_path].decode("utf-8").replace("\r\n", "\n")
    current = (ROOT / proof_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert current.startswith(archived)
    assert 'id="sine-changed-input-causal-numerics"' in archived
    assert "No reduction of the reading error" in archived
    driver = ast.parse(source["evaluate_prediction.py"])
    calls = [node for node in ast.walk(driver) if isinstance(node, ast.Call)]
    predictor = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_predict_collective_port_response"
    ]
    assert len(predictor) == 1
    assert not any(isinstance(node, ast.While) for node in ast.walk(driver))
    assert (
        "export_failed_no_retry_authorized" in source["evaluate_prediction.py"].decode()
    )
    assert (
        'encode_exact_tree(protocol["producer_inputs"])'
        in source["evaluate_prediction.py"].decode()
    )


def test_original_source_and_changed_observation_are_explicit(retained):
    _, decoded = retained
    design = decoded["design.json"]
    assert design["status_at_design"] == (
        "no_changed_input_coefficients_or_complete_response_evaluated"
    )
    assert design["source"]["component_zero_sums"] is True
    assert design["source"]["state_reset"] is False
    assert (
        exact_record(design["source"]["component_form_and_phase_euclidean_radius"])
        == EPS
    )
    assert design["source"]["pre_input_state"] == "z(0-)"
    assert exact_record(design["source"]["common_form_origin"]) == 0
    assert exact_record(design["source"]["common_phase_origin"]) == 0
    assert design["source"]["kind"] == "original_acquired_class_families"
    assert tuple(tuple(map(_integer, row)) for row in design["classes"]) == (
        (1, 1, 1),
        (1, 2, 1),
    )
    support = design["support"]
    assert tuple(map(_integer, support["nodes"])) == tuple(range(27))
    assert tuple(tuple(map(_integer, row)) for row in support["cycles"]) == tuple(
        tuple(range(9 * c, 9 * (c + 1))) for c in range(3)
    )
    assert tuple(tuple(map(_integer, row)) for row in support["contacts"]) == (
        (4, 13),
        (13, 22),
    )
    assert tuple(map(_integer, support["ports"])) == (4, 13, 22)
    assert exact_record(support["capacity"]) == 1
    assert support["support_events"] == ()
    assert design["law"] == {
        "form": "x_prime=-D_inverse_L_x+gamma_D_inverse_S(theta)",
        "phase": "theta_prime=gamma_D_inverse_L_x",
        "gamma": "1/(1023*pi)",
        "structural_clock": "tau=(1023/1024)*t",
        "pressure_refresh": "continuous_state_dependent",
        "phase_target": "Theta[c,j]=2*pi*k[c]*(j-4)/9",
        "forcing": "one_exact_hybrid_form_jump_only",
    }
    assert design["observation"]["coordinate"] == "x_13"
    assert design["observation"]["baseline_subtraction"] is False
    assert exact_record(design["observation"]["reading_error_per_model"]) == DELTA
    assert exact_record(design["observation"]["time"]) == HORIZON
    assert design["observation"]["evaluation"] == "each_class_separately"
    assert design["observation"]["endpoint"] == "right_continuous"
    assert len(design["input"]["events"]) == 1
    event = design["input"]["events"][0]
    assert exact_record(event["time"]) == 0
    assert tuple(map(exact_record, event["port_form_jump"])) == (0, PULSE, 0)
    assert exact_record(design["input"]["horizon"]) == HORIZON
    assert exact_record(design["input"]["total_variation"]) == PULSE
    assert design["predictor"]["adaptation"] is False
    assert _integer(design["predictor"]["time_degree"]) == 32
    assert design["predictor"]["hidden_quadratic_feedback"] is True
    assert design["comparator"]["omitted_terms"] == (
        "hidden_convolution",
        "hidden_initial_source",
        "nonlinear_forcing",
    )
    assert exact_record(design["comparator"]["initial_visible_max_radius"]) == EPS
    assert exact_record(design["comparator"]["reading_error"]) == DELTA
    assert design["comparator"]["law"] == "v_prime=E_k*v, E_k=J_k[V,V]"
    for channel in ("predictor", "comparator"):
        assert _integer(design[channel]["time_degree"]) == 32
        assert (
            exact_record(design[channel]["numerical_radius_ceiling"])
            == NUMERICAL_RADIUS
        )
    for records in (
        decoded["protocol.json"]["producer_inputs"],
        decoded["outcome.json"]["producer_inputs"],
    ):
        assert len(records) == 2
        for k, record in zip((1, 2), records):
            assert _input(record) == dict(
                mediator_class=k,
                initial_form_bounds=((-EPS, EPS),) * 27,
                initial_phase_bounds=((-EPS, EPS),) * 27,
                comparator_initial_bounds=((-EPS, EPS),) * 6,
                port_impulse=(Q(0), PULSE, Q(0)),
                horizon=HORIZON,
            )


@pytest.fixture(scope="module")
def reconstructed(retained):
    _, decoded = retained
    reports = decoded["outcome.json"]["reports"]
    rebuilt = []
    g, h, a = GAMMA_UPPER, HORIZON, PULSE
    ell, denominator = 1 - 2 * g * h, 1 - 2 * g**2 * h**2
    tails = (
        a * _tail(RATE * h, 33),
        2 * g * a**2 * h * _tail(2 * RATE * h, 32),
        Q(4, 3) * g * a**3 * h * _tail(3 * RATE * h, 32)
        + 4 * g**2 * a**3 * h**2 * _tail(3 * RATE * h, 31),
    )
    fifth = 256 * g**6 * a**5 * h / (denominator * (1 - 4 * g**2 * a**2))
    initial_error = (
        4 * g * h * EPS / (ell * denominator) * (g * a / denominator + EPS / ell)
    )
    source_tail = EPS * _tail(RATE * h, 33)
    for k, report in zip((1, 2), reports):
        assert _integer(report["mediator_class"]) == k
        assert _pairs(report["initial_form_bounds"]) == ((-EPS, EPS),) * 27
        assert _pairs(report["initial_phase_bounds"]) == ((-EPS, EPS),) * 27
        assert _pairs(report["comparator_initial_bounds"]) == ((-EPS, EPS),) * 6
        assert tuple(map(exact_record, report["port_impulse"])) == (0, a, 0)
        assert exact_record(report["horizon"]) == h
        assert report["source_radius"] == report["comparator_source_radius"] == EPS
        assert _integer(report["order"]) == 32
        assert "theta-Theta_k at 0-" in report["source_coordinates"]
        coefficients = report["coefficients"]
        levels = tuple(
            _series(rows, 54) for rows in coefficients["nominal_level_coefficients"]
        )
        assert len(levels) == 3
        for degree, rows in enumerate(levels):
            expected = tuple(I(a if degree == 0 and i == 13 else 0) for i in range(54))
            assert rows[0] == expected
        assert all(row[i] == I(0) for row in levels[1] for i in PORTS)
        endpoint_levels = tuple(_endpoint(rows, PORTS) for rows in levels)
        assert (
            tuple(map(_vector, report["nominal_level_endpoint_polynomials"]))
            == endpoint_levels
        )
        assert report["nominal_level_time_tail_upper_bounds"] == tails
        nominal = tuple(a + b for a, b in zip(endpoint_levels[0], endpoint_levels[2]))
        assert _vector(report["nominal_endpoint_polynomials"]) == nominal
        nominal_tail = tails[0] + tails[2]
        assert report["nominal_time_tail_upper_bound"] == nominal_tail
        nominal_bounds = tuple(
            value + I(-nominal_tail, nominal_tail) for value in nominal
        )
        assert _vector(report["nominal_endpoint_bounds"]) == nominal_bounds
        assert report["nominal_numerical_radius_upper_bounds"] == tuple(
            v.radius for v in nominal_bounds
        )
        source = _series(coefficients["linear_source_coefficients"], 54)
        assert source[0] == (I(-EPS, EPS),) * 54
        source_poly = _endpoint(source, PORTS)
        assert _vector(report["linear_source_endpoint_polynomials"]) == source_poly
        assert report["linear_source_time_tail_upper_bound"] == source_tail
        assert _vector(report["linear_source_endpoint_bounds"]) == tuple(
            v + I(-source_tail, source_tail) for v in source_poly
        )
        comparator = _series(coefficients["comparator_nominal_coefficients"], 6)
        assert comparator[0] == tuple(I(a if i == 1 else 0) for i in range(6))
        comparator_poly = _endpoint(comparator, range(6))
        assert (
            _vector(report["comparator_nominal_endpoint_polynomials"])
            == comparator_poly
        )
        assert report["comparator_nominal_time_tail_upper_bound"] == tails[0]
        comparator_bounds = tuple(v + I(-tails[0], tails[0]) for v in comparator_poly)
        assert (
            _vector(report["comparator_nominal_endpoint_bounds"]) == comparator_bounds
        )
        assert report["comparator_numerical_radius_upper_bounds"] == tuple(
            v.radius for v in comparator_bounds
        )
        comparator_source = _series(coefficients["comparator_source_coefficients"], 6)
        assert comparator_source[0] == (I(-EPS, EPS),) * 6
        comparator_source_poly = _endpoint(comparator_source, range(6))
        assert (
            _vector(report["comparator_source_endpoint_polynomials"])
            == comparator_source_poly
        )
        assert report["comparator_source_time_tail_upper_bound"] == source_tail
        assert _vector(report["comparator_source_endpoint_bounds"]) == tuple(
            v + I(-source_tail, source_tail) for v in comparator_source_poly
        )
        assert report["linear_source_uniform_bound"] == EPS / ell
        assert report["comparator_source_uniform_bound"] == EPS / ell
        fidelity = report["fidelity"]
        assert fidelity["total_input_variation"] == a
        assert fidelity["horizon"] == h and fidelity["endpoint_radius"] == EPS
        assert fidelity["flow_comparison_margin"] == ell
        assert fidelity["phase_bootstrap_margin"] == denominator
        assert fidelity["nominal_fifth_order_error_upper_bound"] == fifth
        assert fidelity["nonlinear_initialization_error_upper_bound"] == initial_error
        assert fidelity["form_error_upper_bound"] == fifth + initial_error
        assert fidelity["phase_error_upper_bound"] == 2 * g * h * (
            fifth + initial_error
        )
        assert fidelity["port_pressure_error_upper_bounds"] == tuple(
            2 * d * (fifth + initial_error) for d in (3, 4, 3)
        )
        rebuilt.append(
            (
                report,
                nominal_bounds,
                comparator_bounds,
                fifth + initial_error + EPS / ell,
                EPS / ell,
            )
        )
    return tuple(rebuilt)


def test_all_endpoint_horner_time_tail_and_source_arithmetic(reconstructed):
    assert len(reconstructed) == 2


def test_original_target_coefficients_and_port_partition(reconstructed):
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    degrees = tuple(sum(i in edge for edge in EDGES) for i in range(27))
    for k, (report, *_rest) in zip((1, 2), reconstructed):
        descriptor = report["descriptor"]
        parameters = descriptor["parameter_bounds"]
        assert parameters["classes"] == (1, k, 1)
        assert tuple(map(_integer, parameters["degrees"])) == degrees
        assert _interval(parameters["gamma"]) == gamma
        assert _interval(parameters["eta"]) == gamma**2
        turns = tuple(Q((1, k, 1)[c] * (j - 4), 9) for c in range(3) for j in range(9))
        sine, cosine = [], []
        for left, right in EDGES:
            gap = turns[right] - turns[left]
            gap -= (gap + Q(1, 2)).__floor__()
            angle = 2 * gap * pi
            sine.append(sin(angle) if gap else I(0))
            cosine.append(cos(angle) if gap else I(1))
        assert _vector(parameters["edge_sines"]) == tuple(sine)
        assert _vector(parameters["edge_cosines"]) == tuple(cosine)
        assert descriptor["port_indices"] == PORTS
        assert descriptor["hidden_indices"] == tuple(
            i for i in range(54) if i not in PORTS
        )
        assert descriptor["port_jump_quadratic_matrix"] == (
            (3, -1, 0),
            (-1, 4, -1),
            (0, -1, 3),
        )
        kernels = report["coefficients"]["kernels"]
        assert kernels["port_indices"] == PORTS and _integer(kernels["order"]) == 32
        assert len(kernels["grounded_kernel_coefficients"]) == 3
        for group, rows in zip(
            kernels["hidden_component_indices"], kernels["grounded_kernel_coefficients"]
        ):
            assert len(group) == 16 and len(rows) == 33
            assert all(
                len(matrix) == 16 and all(len(row) == 16 for row in matrix)
                for matrix in rows
            )
            assert tuple(map(_vector, rows[0])) == tuple(
                tuple(I(int(i == j)) for j in range(16)) for i in range(16)
            )


def test_separate_model_source_and_two_reading_discrimination(reconstructed):
    documented_gaps = (Q("3.5084390938e-5"), Q("3.5084391373e-5"))
    for documented_gap, (
        report,
        nominal,
        comparator,
        full_error,
        comparator_error,
    ) in zip(documented_gaps, reconstructed):
        full = (nominal[1].lo - full_error, nominal[1].hi + full_error)
        alternative = (
            comparator[1].lo - comparator_error,
            comparator[1].hi + comparator_error,
        )
        recorded = full[0] - DELTA, full[1] + DELTA
        alternative_recorded = alternative[0] - DELTA, alternative[1] + DELTA
        assert nominal[1].radius <= NUMERICAL_RADIUS
        assert comparator[1].radius <= NUMERICAL_RADIUS
        assert nominal[1].radius < Q("6.654e-24")
        assert comparator[1].radius < Q("8.677e-31")
        assert recorded[0] > alternative_recorded[1]
        assert recorded[0] - alternative_recorded[1] > documented_gap
        assert (
            recorded[0] - alternative_recorded[1]
            == full[0] - alternative[1] - 2 * DELTA
        )


def test_retained_first_rates_memory_and_hidden_source_have_original_factors(
    reconstructed,
):
    gamma = 1 / (1023 * pi_interval())
    for k, (report, *_rest) in zip((1, 2), reconstructed):
        coefficients = report["coefficients"]
        rows = _series(coefficients["nominal_level_coefficients"][0], 54)
        degrees = tuple(sum(i in edge for edge in EDGES) for i in range(27))
        laplacian_column = tuple(
            Q(1) if i == 13 else Q(-1, degrees[i]) if i in (4, 12, 14, 22) else Q(0)
            for i in range(27)
        )
        first_rate = tuple(-I(PULSE) * value for value in laplacian_column) + tuple(
            gamma * I(PULSE) * value for value in laplacian_column
        )
        # Causal arithmetic has a different outward operation order. This
        # low-order association is not a regeneration of retained higher jets.
        for actual, expected in zip(rows[1], first_rate):
            assert max(actual.lo, expected.lo) <= min(actual.hi, expected.hi)
        cosine = cos(2 * k * pi_interval() / 9)
        memory0 = _interval(
            coefficients["kernels"]["memory_kernel_coefficients"][0][1][1]
        )
        expected_memory = (1 - gamma**2 * cosine) / 4
        assert memory0.lo > 0
        assert max(memory0.lo, expected_memory.lo) <= min(
            memory0.hi, expected_memory.hi
        )
        source_force = _series(
            coefficients["hidden_initialization_port_forcing_coefficients"], 6
        )
        extent = (1 + gamma * cosine) * EPS / 2
        expected_force = I(-extent.hi, extent.hi)
        assert source_force[0][1].lo < 0 < source_force[0][1].hi
        assert Q(49, 100) * EPS < source_force[0][1].radius < Q(51, 100) * EPS
        assert max(source_force[0][1].lo, expected_force.lo) <= min(
            source_force[0][1].hi, expected_force.hi
        )


def test_response_free_policy_is_rebuilt_without_cached_admission(retained):
    _, decoded = retained
    admission = decoded["protocol.json"]["response_free_admission"]
    g, eps, q = GAMMA_UPPER, EPS, PULSE
    ell, denominator = 1 - 2 * g, 1 - 2 * g**2
    work = (2 * q**2 - 8 * q * eps, 2 * q**2 + 8 * q * eps)
    fifth = 256 * g**6 * q**5 / (denominator * (1 - 4 * g**2 * q**2))
    initialization = (
        4 * g * eps / (ell * denominator) * (g * q / denominator + eps / ell)
    )
    nominal_gap = q * (Q(1, 24) - 4 * g**2 / denominator)
    gap = (
        nominal_gap
        - 2 * eps / ell
        - 2 * DELTA
        - 4 * NUMERICAL_RADIUS
        - 2 * fifth
        - initialization
    )
    exact = {
        "event_work_bounds": work,
        "event_work_ceiling": Q(1, 500000),
        "post_event_excess_storage_upper_bound": 22 * eps**2 + work[1],
        "storage_barrier_lower_bound": Q(1, 388800),
        "post_event_euclidean_quotient_radius_squared_upper_bound": 27
        * ((q + eps) ** 2 + eps**2),
        "identity_radius_squared": Q(1, 144),
        "form_mean_increment": 2 * q / 29,
        "phase_mean_increment": Q(0),
        "linear_source_allowance_per_model": eps / ell,
        "reading_error_per_model": DELTA,
        "numerical_radius_ceiling_per_model": NUMERICAL_RADIUS,
        "nominal_gap_strict_lower_bound": nominal_gap,
        "conservative_recorded_gap_strict_lower_bound": gap,
    }
    for key, value in exact.items():
        observed = (
            tuple(map(exact_record, admission[key]))
            if isinstance(value, tuple)
            else exact_record(admission[key])
        )
        assert observed == value
    expected_flags = {
        "work_admitted": work[1] < Q(1, 500000),
        "identity_admitted": exact["post_event_excess_storage_upper_bound"]
        < Q(1, 388800)
        and exact["post_event_euclidean_quotient_radius_squared_upper_bound"]
        < Q(1, 144),
        "prospective_separation_sufficient": gap > 0,
    }
    for key, value in expected_flags.items():
        assert admission[key] is value
    assert nominal_gap > Q(1, 40000)
    assert gap > Q(29, 10**6)


def test_full_model_work_and_source_mean_are_not_comparator_assumptions():
    pressure = 2 * 4 * EPS
    work_upper = 2 * PULSE**2 + PULSE * pressure
    assert work_upper < Q(1, 10**6) < Q(1, 500000)
    assert 22 * EPS**2 + work_upper < Q(1, 388800)
    assert 27 * ((PULSE + EPS) ** 2 + EPS**2) < Q(1, 144)
    assert (
        4 * PULSE / sum(sum(i in edge for edge in EDGES) for i in range(27))
        == 2 * PULSE / 29
    )
