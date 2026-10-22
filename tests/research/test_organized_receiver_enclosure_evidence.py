"""Read the first frozen receiver prefix without regenerating any trajectory.

Rebuild retained arithmetic and static observation decisions from primitive
records. Picard admission and Taylor derivative generation remain numerical
premises: this audit neither repeats them nor authenticates chronology.
"""

import io
import zipfile
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr._exact_time import exp_unit_bounds
from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics import _organized_receiver_enclosure_geometry as geometry
from tnfr.research import organized_receiver_enclosure as runner
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from tnfr.research.frozen_source import inspect_frozen_source
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
STEM = "docs/assets/sine_formed_classes/organized-receiver-enclosure-v1"
BASE = "3e5f552bc5b8f7e8a73e2469f71bf9305032c7ef"
PINS = {
    ".protocol.json": (
        144353,
        "513381a90728baa063c2314c495bc5a8a62a82d28ec8f95e4325e5c86ff6b4c9",
    ),
    ".source.zip": (
        134454,
        "25df9e88511ca086d19f2caae25af77d35cfac4b4e052165df629ea020d2fe48",
    ),
    ".freeze.json": (
        908,
        "1a08d0818f4cf89f7193cc0f44f32fdb8e2daba9f6620bf13aa93dded269707a",
    ),
    ".attempt.json": (
        766,
        "6294afbd56400f1f415f27fe6a6eb9e389612e7db47af0fb642b07fb50c1e909",
    ),
}
# Installed only after the first fixed attempt finishes, never used by it.
OUTCOME_PIN = (
    189956,
    "2dbb6e1b738f33f1189ba0de1436b5682c8109000d0ea400250d9a7ae763b79c",
)
TRANSPORT_PIN = (
    741711,
    "77b38c4785a57a355aa0d86a318b7acd22702e69de317687e93222ab5758d044",
)
H, RHO = Q(1, 8), Q(11, 10**6)
REPRESENTATIVES = tuple(range(9)) + tuple(range(18, 22))
REFLECTION = tuple(9 * ((1 - node // 9) % 3) + 8 - node % 9 for node in range(27))
EDGES = tuple(
    (9 * region + j, 9 * region + (j + 1) % 9, Q(1))
    for region in range(3)
    for j in range(9)
) + tuple(
    (9 * a + j, 9 * b + (j + offset) % 9, Q(1, 2))
    for a, b in ((0, 1), (1, 2), (2, 0))
    for j in range(9)
    for offset in (0, 1)
)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    original_open = Path.open

    def forbidden(*args, **kwargs):
        pytest.fail("retained receiver audit regenerated a field or observation report")

    def without_unpacked_response(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved == ROOT / (STEM + ".json") or resolved.is_relative_to(
            ROOT / (STEM + ".steps")
        ):
            pytest.fail("response audit requires untracked unpacked evidence")
        return original_open(path, *args, **kwargs)

    with (
        forbid_sine_regeneration(git_root=ROOT, git_base=BASE),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setattr(Path, "open", without_unpacked_response)
        for module in (geometry, runner):
            for name in (
                "_receiver_reference_flow",
                "_receiver_local_osl",
                "_observe_receiver_tube",
            ):
                patch.setattr(module, name, forbidden)
        yield


def _same(actual, expected):
    assert json_dumps(encode_exact_tree(actual), sort_keys=True) == json_dumps(
        encode_exact_tree(expected), sort_keys=True
    )


def _integer(value, *, minimum=0):
    assert type(value) is int and value >= minimum
    return value


def _boolean(value):
    assert type(value) is bool
    return value


def _interval(record):
    assert isinstance(record, dict) and set(record) == {"lo", "hi"}
    lower, upper = exact_record(record["lo"]), exact_record(record["hi"])
    assert lower <= upper
    value = I(lower, upper)
    assert value.lo == lower and value.hi == upper
    return value


def _box(records, size):
    assert isinstance(records, tuple) and len(records) == size
    return tuple(_interval(value) for value in records)


def _lift(values):
    assert len(values) == 13
    result = [I(0)] * 27
    for node, value in zip(REPRESENTATIVES, values):
        result[node], result[REFLECTION[node]] = I.coerce(value), -I.coerce(value)
    return tuple(result)


def _metric_radius(errors):
    scale = max(errors)
    return (
        Q(0)
        if scale == 0
        else scale * sqrt(I(sum(2 * (e / scale) ** 2 for e in errors))).hi
    )


def _discrepancy(radius, rate):
    if rate == 0:
        return I(radius + RHO * H).hi
    exponential = I(*exp_unit_bounds(abs(rate * H)))
    if rate < 0:
        exponential = 1 / exponential
    return (exponential * radius + ((exponential - 1) / rate) * RHO).hi


def _pinned(path, pin):
    size, digest = pin
    data = read_bytes_bounded(path, max_bytes=size)
    assert len(data) == size and sha256_bytes(data) == digest
    return data


@pytest.fixture(scope="module")
def retained(no_scientific_regeneration):
    assert TRANSPORT_PIN is not None, "lossless response transport pin is not installed"
    pins = PINS | {".response.zip": TRANSPORT_PIN}
    raw = {suffix: _pinned(ROOT / (STEM + suffix), pin) for suffix, pin in pins.items()}
    records = {
        suffix: decode_exact_tree(json_loads(data))
        for suffix, data in raw.items()
        if suffix.endswith(".json")
    }
    inspection = inspect_frozen_source(ROOT, STEM + ".freeze.json")
    with zipfile.ZipFile(io.BytesIO(raw[".source.zip"])) as archive:
        with archive.open("source-manifest.json") as stream:
            manifest_data = stream.read(1024**2 + 1)
        assert len(manifest_data) <= 1024**2
        manifest = decode_exact_tree(json_loads(manifest_data))
        expected = {item["path"]: item["sha256"] for item in manifest["files"]}
        assert len(expected) == len(manifest["files"])
        expected["source-manifest.json"] = sha256_bytes(manifest_data)
        verify_archive_members(
            io.BytesIO(raw[".source.zip"]), expected, max_bytes=8 * 1024**2
        )
        assert archive.read(STEM + ".protocol.json") == raw[".protocol.json"]
        assert not any(name.startswith(STEM + ".steps/") for name in archive.namelist())
    transport = raw[".response.zip"]
    with zipfile.ZipFile(io.BytesIO(transport)) as archive:
        with archive.open(STEM + ".json") as stream:
            outcome_data = stream.read(OUTCOME_PIN[0] + 1)
        assert len(outcome_data) == OUTCOME_PIN[0]
        assert sha256_bytes(outcome_data) == OUTCOME_PIN[1]
        outcome = decode_exact_tree(json_loads(outcome_data))
        expected = {STEM + ".json": OUTCOME_PIN[1]}
        steps = []
        for index, receipt in enumerate(outcome["step_receipts"], 1):
            assert set(receipt) == {"path", "bytes", "sha256"}
            assert receipt["path"] == f"{STEM}.steps/step-{index:04d}.json"
            _integer(receipt["bytes"], minimum=1)
            expected[receipt["path"]] = receipt["sha256"]
        verify_archive_members(io.BytesIO(transport), expected, max_bytes=16 * 1024**2)
        for receipt in outcome["step_receipts"]:
            with archive.open(receipt["path"]) as stream:
                data = stream.read(receipt["bytes"] + 1)
            assert len(data) == receipt["bytes"]
            assert sha256_bytes(data) == receipt["sha256"]
            steps.append(decode_exact_tree(json_loads(data)))
    yield SimpleNamespace(
        inspection=inspection,
        manifest=manifest,
        protocol=records[".protocol.json"],
        freeze=records[".freeze.json"],
        attempt=records[".attempt.json"],
        outcome=outcome,
        report=outcome["report"],
        steps=tuple(steps),
    )
    for suffix, pin in pins.items():
        _pinned(ROOT / (STEM + suffix), pin)


def test_complete_frozen_source_attempt_and_first_outcome_association(retained):
    for record in (
        retained.freeze,
        retained.protocol,
        retained.manifest,
        retained.attempt,
        retained.outcome,
    ):
        assert record["source_base_commit"] == BASE
    assert retained.inspection.source_base_commit == BASE
    assert retained.inspection.prior_artifact_count == 0
    assert retained.protocol["source_specification"]["prior_artifact_receipts"] == ()
    assert retained.freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert retained.protocol["evaluation_status_at_freeze"] == "not_evaluated"
    assert (
        retained.freeze["runtime_overlays"]
        == retained.protocol["runtime_overlays"]
        == ()
    )
    assert (
        retained.attempt["freeze_sha256"]
        == retained.outcome["freeze_sha256"]
        == PINS[".freeze.json"][1]
    )
    assert (
        retained.attempt["runtime_environment"]
        == retained.protocol["runtime_environment"]
    )
    assert retained.attempt["seed"] is None
    assert (
        datetime.fromisoformat(retained.freeze["created_utc"])
        <= datetime.fromisoformat(retained.attempt["started_utc"])
        <= datetime.fromisoformat(retained.outcome["finished_utc"])
    )
    assert retained.outcome["attempt"] == file_receipt(
        ROOT / (STEM + ".attempt.json"), label=STEM + ".attempt.json"
    )
    assert retained.outcome["incomplete_step_paths"] == ()
    existing = set(retained.inspection.existing_outcome_files)
    assert STEM + ".attempt.json" in existing
    assert existing <= {STEM + ".attempt.json", STEM + ".json", STEM + ".steps"}


def test_frozen_policy_and_original_source_admission_rebuild_without_flow(retained):
    _same(retained.protocol["numerical_policy"], runner.receiver_enclosure_policy())
    _same(retained.outcome["numerical_policy"], retained.protocol["numerical_policy"])
    source = asdict(runner.receiver_enclosure_source())
    _same(retained.protocol["source_admission"], source)
    _same(retained.report["source"], source)
    assert source["release_time"] == 400
    assert source["donor_release_bridge_certified"] is True
    assert source["release_windings"] == (1, 1, 0)
    assert source["uniform_velocity_defect_upper_bound"] < RHO
    assert retained.protocol["kernel_policy"]["picard_iterations"] == 16
    assert retained.protocol["kernel_policy"]["picard_additive_padding"] == Q(1, 2**90)
    assert retained.protocol["reserved_attempt_policy"]["maximum_step_records"] == 256
    assert retained.protocol["reserved_attempt_policy"]["adaptation_or_retry"] is False


@pytest.fixture(scope="module")
def chain(retained):
    alpha = 2 * pi_interval() / 9
    prepared = tuple(alpha * (j - 4) for j in range(9)) + (I(0),) * 4
    center = tuple(value.midpoint for value in prepared)
    rounding = _metric_radius(tuple(value.radius for value in prepared))
    assert retained.report["initial_rounding_error_bound"] == rounding
    source = retained.report["source"]
    radius = I(
        exact_record(source["release_phase_error_norm_upper_bound"]) + rounding
    ).hi
    assert retained.report["initial_observation"]["phase_error_bound"] == radius
    _same(
        retained.report["initial_observation"]["phase_box"],
        tuple(asdict(value) for value in _lift(center)),
    )
    accepted = []
    for index, record in enumerate(retained.steps, 1):
        assert _integer(record["step_index"], minimum=1) == index
        if not _boolean(record["accepted"]):
            assert index == len(retained.steps)
            break
        step = record["reference_step"]
        assert step["time"] == (index - 1) * H and step["duration"] == H
        assert _integer(step["order"], minimum=1) == 8
        _same(step["initial_center"], center)
        assert exact_record(step["initial_radius"]) == 0
        assert record["initial_phase_error"] == radius
        _same(
            step["metric"],
            tuple(tuple(Q(2 * int(i == j)) for j in range(13)) for i in range(13)),
        )
        assert step["picard_interior_margin"] > 0
        assert step["domain_lower_bounds"] == (Q(1),)
        tube = _box(step["tube"], 13)
        polynomial = _box(step["center_polynomial_bounds"], 13)
        remainder = _box(step["local_remainder_bounds"], 13)
        endpoints = tuple(value + error for value, error in zip(polynomial, remainder))
        _same(
            step["center_solution_endpoint"],
            tuple(asdict(value) for value in endpoints),
        )
        assert all(
            max(value.lo, bound.lo) <= min(value.hi, bound.hi)
            for value, bound in zip(endpoints, tube)
        )
        center = tuple(value.midpoint for value in endpoints)
        coordinate_errors = tuple(value.radius for value in endpoints)
        _same(step["endpoint_center"], center)
        _same(step["local_coordinate_error_bounds"], coordinate_errors)
        local_error = _metric_radius(coordinate_errors)
        assert step["local_metric_error_upper_bound"] == local_error
        assert step["propagated_initial_radius"] == 0
        assert step["endpoint_radius"] == I(local_error).hi
        assert (
            step["growth_rate_upper_bound"] == 2 and step["growth_certificate"] is None
        )
        projection = sqrt(I(Q(1, 2))).hi
        _same(step["coordinate_projection_factors"], (projection,) * 13)
        _same(
            step["endpoint"],
            tuple(
                asdict(
                    I(
                        value - projection * step["endpoint_radius"],
                        value + projection * step["endpoint_radius"],
                    )
                )
                for value in center
            ),
        )
        global_envelope = _discrepancy(radius, Q(2))
        prospective = I(Q(5, 4) * (global_envelope + Q(1, 2**80))).hi
        assert record["global_error_envelope"] == global_envelope
        assert record["prospective_radius"] == prospective
        growth = exact_record(record["osl"]["growth_rate_upper_bound"])
        assert -1 <= growth <= 2
        continuous = _discrepancy(radius, growth)
        whole = max(radius, continuous)
        assert record["continuous_endpoint_error"] == continuous
        assert record["whole_step_phase_error"] == whole
        assert record["strict_error_margin"] == prospective - whole > 0
        radius = I(continuous + step["endpoint_radius"]).hi
        assert record["endpoint_phase_error"] == radius
        _same(record["osl"]["phase_box"], tuple(asdict(value) for value in _lift(tube)))
        assert record["osl"]["phase_radius"] == prospective
        assert record["donor_corridor"]["phase_error_bound"] == whole
        assert record["endpoint"]["phase_error_bound"] == radius
        _same(
            record["donor_corridor"]["phase_box"],
            tuple(asdict(value) for value in _lift(tube)),
        )
        _same(
            record["endpoint"]["phase_box"],
            tuple(asdict(value) for value in _lift(center)),
        )
        accepted.append(record)
    assert retained.report["accepted_steps"] == len(accepted)
    assert retained.report["elapsed_slow_time"] == len(accepted) * H
    assert retained.report["final_phase_error"] == radius
    return tuple(accepted)


def test_source_error_and_endpoint_recurrences_keep_every_accepted_step(chain):
    assert chain


def test_retained_local_growth_bounds_cover_all_mean_zero_directions(chain):
    for record in chain:
        osl = record["osl"]
        assert _boolean(osl["certified"])
        assert _integer(osl["phase_mean_leaf_dimension"]) == 26
        assert _integer(osl["bisections_performed"]) == 8
        assert _integer(osl["cosine_rounding_bits"]) == 20
        assert osl["search_rate_bounds"] == (Q(-1), Q(2))
        phases = _box(osl["phase_box"], 27)
        allowance = sqrt(I(2)).hi * exact_record(osl["phase_radius"])
        lower_by_edge = {}
        for a, b, _ in EDGES:
            value = cos(phases[b] - phases[a] + I(-allowance, allowance)).lo
            lower_by_edge[frozenset((a, b))] = Q(
                (value.numerator * 2**20) // value.denominator, 2**20
            )
        jacobian = [[Q(0) for _ in range(27)] for _ in range(27)]
        assert len(osl["edge_cosine_lower_bounds"]) == len(EDGES)
        for (a, b, weight), raw in zip(EDGES, osl["edge_cosine_lower_bounds"]):
            lower = exact_record(raw)
            assert lower == min(
                lower_by_edge[frozenset((a, b))],
                lower_by_edge[frozenset((REFLECTION[a], REFLECTION[b]))],
            )
            coefficient = weight * lower / 4
            jacobian[a][a] -= coefficient
            jacobian[b][b] -= coefficient
            jacobian[a][b] += coefficient
            jacobian[b][a] += coefficient
        # Independent full basis e_i-e_26; no nominal-symmetry restriction.
        growth = exact_record(osl["growth_rate_upper_bound"])
        slack = tuple(
            tuple(
                growth * (1 + int(i == j))
                - (
                    jacobian[i][j]
                    - jacobian[i][26]
                    - jacobian[26][j]
                    + jacobian[26][26]
                )
                for j in range(26)
            )
            for i in range(26)
        )
        assert exact_symmetric_semidefinite(slack)


def _observation(record, form_bound):
    phases = _box(record["phase_box"], 27)
    radius = exact_record(record["phase_error_bound"])
    assert radius >= 0 and record["form_norm_upper_bound"] == form_bound
    pi = pi_interval()
    error = sqrt(I(2)).hi * radius
    margins, winding = [], []
    for region in range(3):
        differences = tuple(
            phases[9 * region + (j + 1) % 9] - phases[9 * region + j] for j in range(9)
        )
        if region < 2:
            margins.append(
                pi.lo
                - max(
                    (d + (2 * pi if j == 8 else 0)).abs_max
                    for j, d in enumerate(differences)
                )
                - error
            )
        integers = []
        for difference in differences:
            ratio = difference.midpoint / (2 * pi.midpoint) + Q(1, 2)
            branch = ratio.numerator // ratio.denominator
            if (difference - 2 * branch * pi).abs_max + error >= pi.lo:
                break
            integers.append(branch)
        winding.append(-sum(integers) if len(integers) == 9 else None)
    _same(record["donor_registered_branch_margins"], tuple(margins))
    _same(record["windings"], tuple(winding))
    potential, gradient = I(0), [I(0)] * 27
    maximum_gap = Q(0)
    for a, b, weight in EDGES:
        difference = phases[b] - phases[a]
        potential += weight * (1 - cos(difference))
        current = weight * sin(difference)
        gradient[a] -= current
        gradient[b] += current
        branch = ((b % 9) - (a % 9)) // 9
        maximum_gap = max(maximum_gap, (difference - 2 * branch * pi).abs_max)
    _same(record["complete_reference_potential_bounds"], asdict(potential))
    norm = sqrt(I(sum(value.abs_max**2 for value in gradient))).hi
    assert record["reference_gradient_norm_upper_bound"] == norm
    target = 27 * (1 - cos(2 * pi / 9)) + Q(27, 2) * (1 - cos(2 * pi / 9))
    _same(record["target_potential_bounds"], asdict(target))
    assert record["target_registered_gap_upper_bound"] == maximum_gap
    cell_margin = 2 - maximum_gap - error
    excess = (
        potential.hi - target.lo + norm * radius + 4 * radius**2 + 4 * form_bound**2
    )
    energy_margin = Q(196, 625) - excess
    assert record["capture_cell_margin"] == cell_margin
    assert record["complete_capture_excess_upper_bound"] == excess
    assert record["capture_energy_margin"] == energy_margin
    assert _boolean(record["capture_inequalities_pass"]) == (
        cell_margin > 0 and energy_margin > 0
    )
    return tuple(margins), tuple(winding), cell_margin > 0 and energy_margin > 0


def test_observations_and_stop_priority_rebuild_the_original_verdict(retained, chain):
    source = retained.report["source"]
    form = exact_record(source["release_form_norm_upper_bound"])
    _, initial_winding, _ = _observation(retained.report["initial_observation"], form)
    assert initial_winding == (1, 1, 0)
    history = _boolean(source["donor_release_bridge_certified"])
    first_lost = None
    last_stop = None
    for index, record in enumerate(chain, 1):
        corridor, corridor_winding, _ = _observation(record["donor_corridor"], form)
        assert corridor_winding[2] == 0
        _, winding, capture = _observation(record["endpoint"], form)
        history = history and all(margin > 0 for margin in corridor)
        assert _boolean(record["donor_history_certified"]) == history
        if not history and first_lost is None:
            first_lost = ((index - 1) * H, index * H)
        stop = (
            "certified_donor_loss"
            if any(w is not None and w != 1 for w in winding[:2])
            else (
                "certified_joined_capture_with_donor_retention"
                if history and capture
                else (
                    "phase_error_budget_exhausted"
                    if record["endpoint_phase_error"] > Q(1, 2)
                    else None
                )
            )
        )
        if index < len(chain):
            assert stop is None
        last_stop = stop
    assert retained.report["donor_history_certified"] is history
    _same(retained.report["first_unavailable_donor_interval"], first_lost)
    _same(retained.report["final_observation"], chain[-1]["endpoint"])
    if len(retained.steps) > len(chain):
        assert last_stop is None
        assert retained.report["status"] == "unavailable"
        assert retained.report["reason"] == retained.steps[-1]["reason"]
    elif last_stop in (
        "certified_donor_loss",
        "certified_joined_capture_with_donor_retention",
    ):
        assert retained.report["status"] == retained.report["reason"] == last_stop
    else:
        assert retained.report["status"] == "unavailable"
        assert retained.report["reason"] == (last_stop or "horizon_exhausted")
        if last_stop is None:
            assert len(chain) == 256
    assert retained.outcome["status"] == retained.report["status"]
    assert retained.report["physical_identification"] == "not_supplied_or_evaluated"
    assert len(chain) == 76
    assert retained.report["elapsed_slow_time"] == Q(19, 2)
    assert retained.report["status"] == "certified_donor_loss"
    assert retained.report["final_observation"]["windings"] == (0, 0, 0)
    assert first_lost == (Q(37, 4), Q(75, 8))
