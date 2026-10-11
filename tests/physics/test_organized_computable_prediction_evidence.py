"""Read-only arithmetic audit of the frozen post-observation forecast.

No source acquisition, target integration or response assessor is invoked.
Stored compact witnesses retain their conditional smooth-field/Taylor proof;
this audit independently rebuilds their enclosure and decision arithmetic.
Receipts establish content associations, not execution authentication.
"""

from fractions import Fraction as Q
from functools import lru_cache
import hashlib
from math import factorial
from pathlib import Path
import subprocess
import zipfile

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.utils.io import json_loads


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_computable_prediction"
REVISION = "8f5a28c58f23ab1586cc91ac36d49e6ddd9bbce0"
DECLARATION = "theory/nodal/ORGANIZED_COMPUTABLE_PREDICTION.md"
EVALUATOR = "benchmarks/organized_computable_prediction.py"
CONSUMER = "src/tnfr/physics/_organized_computable_prediction.py"
MARKER = (
    b"<!-- FROZEN DECLARATION AND PROOF END: append evaluated results only below. -->"
)
EXPANSION_LIMIT = 128 * 1024**2


def _decode(raw):
    return decode_exact_tree(json_loads(raw))


def _interval(value):
    assert set(value) == {"lo", "hi"}
    return I(exact_record(value["lo"]), exact_record(value["hi"]))


def _coordinates(values):
    assert len(values) == 10
    return tuple(map(exact_record, values))


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


@pytest.fixture(scope="module")
def source_snapshot():
    # Git inspection occurs before the opt-in guard blocks all subprocesses.
    try:
        _git("cat-file", "-e", REVISION + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        return None
    scope = ("src/tnfr", EVALUATOR, DECLARATION, "pyproject.toml")
    return {
        "listing": _git("ls-tree", "-r", "-z", "--full-tree", REVISION, "--", *scope),
        "src": _git("rev-parse", REVISION + ":src").decode("ascii").strip(),
        "declaration": _git("show", REVISION + ":" + DECLARATION),
    }


@pytest.fixture(scope="module")
def retained():
    receipt = _decode(read_bytes_bounded(EVIDENCE / "receipt.json", max_bytes=2**20))
    archive = EVIDENCE / "response.zip"
    assert receipt["response_archive"] == file_receipt(
        archive, label="response.zip", max_bytes=EXPANSION_LIMIT
    )
    member = receipt["result"]
    assert member["path"] == "result.json"
    verify_archive_members(
        archive, {"result.json": member["sha256"]}, max_bytes=EXPANSION_LIMIT
    )
    with zipfile.ZipFile(archive) as stream:
        raw_result = stream.read("result.json")
    assert len(raw_result) == member["bytes"]
    assert hashlib.sha256(raw_result).hexdigest() == member["sha256"]
    raw_attempt = read_bytes_bounded(EVIDENCE / "attempt.json", max_bytes=2**22)
    return _decode(raw_attempt), _decode(raw_result), raw_attempt


@pytest.fixture(scope="module", autouse=True)
def preserve_evidence_and_exclude_regeneration(retained, source_snapshot):
    paths = tuple(
        EVIDENCE / name for name in ("attempt.json", "receipt.json", "response.zip")
    )
    paths += tuple(
        ROOT / name
        for name in (
            EVALUATOR,
            CONSUMER,
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "src/tnfr/physics/_organized_finite_observation_source.py",
            "docs/assets/organized_robust_current/attempt.json",
            "docs/assets/organized_robust_current/result.json",
            "src/tnfr/physics/_organized_robust_current.py",
        )
    )
    before = tuple(file_receipt(path) for path in paths)
    with forbid_organized_response_evaluation():
        yield
    assert tuple(file_receipt(path) for path in paths) == before


def test_frozen_attempt_archive_and_complete_source_association(
    retained, source_snapshot
):
    attempt, result, raw_attempt = retained
    assert attempt["source_revision"] == result["source_revision"] == REVISION
    assert attempt["schema"] == "organized_computable_prediction_attempt_v1"
    assert result["schema"] == "organized_computable_prediction_result_v1"
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["parameters"] == {} and attempt["seed"] is None
    assert attempt["interval_bits"] == 128
    assert (
        attempt["arithmetic_method"]
        == "rational_outward_dyadic128_machin_trigonometry_v1"
    )
    assert attempt["declaration"] == DECLARATION and attempt["evaluator"] == EVALUATOR
    assert attempt["source_scope"] == (
        "src/tnfr",
        EVALUATOR,
        DECLARATION,
        "pyproject.toml",
    )
    assert result["attempt"] == {
        "path": "attempt.json",
        "bytes": len(raw_attempt),
        "sha256": hashlib.sha256(raw_attempt).hexdigest(),
    }
    assert {path.name for path in EVIDENCE.iterdir()} == {
        "attempt.json",
        "receipt.json",
        "response.zip",
    }
    if source_snapshot is None:
        pytest.skip("retained source revision unavailable in this checkout")
    listing = source_snapshot["listing"]
    inventory = {}
    for entry in filter(None, listing.split(b"\0")):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        assert kind == "blob"
        inventory[path.decode("utf-8")] = {"mode": mode, "blob": blob}
    assert attempt["source_inventory"] == inventory
    assert attempt["source_inventory_sha256"] == hashlib.sha256(listing).hexdigest()
    assert attempt["source_tree"] == source_snapshot["src"]
    assert {
        DECLARATION,
        EVALUATOR,
        CONSUMER,
        "src/tnfr/mathematics/_validated_metric.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "src/tnfr/physics/_sine_flow.py",
    } <= set(inventory)
    for record in (attempt, result):
        assert set(record["imported_sources"].values()) <= set(inventory)
    assert (
        result["imported_sources"]["tnfr.physics._organized_computable_prediction"]
        == CONSUMER
    )
    frozen = source_snapshot["declaration"].replace(b"\r\n", b"\n")
    current = (ROOT / DECLARATION).read_bytes().replace(b"\r\n", b"\n")
    assert frozen.count(MARKER) == current.count(MARKER) == 1
    assert frozen.split(MARKER)[0] == current.split(MARKER)[0]


def _expected_stages():
    stages = [(Q(0), Q(200), "formation", None, Q(0), Q(0))]
    elapsed = Q(200)
    for child in range(5):
        for base, increment in (
            (Q(0), Q(1, 10)),
            (Q(1, 10), Q(-1, 5)),
            (Q(-1, 10), Q(1, 10)),
        ):
            stages.append((elapsed, Q(1, 100000), "plateau", child, base, Q(0)))
            elapsed += Q(1, 100000)
            stages.append((elapsed, Q(1), "ramp", child, base, increment))
            elapsed += 1
    stages.append((elapsed, 216 - elapsed, "released", None, Q(0), Q(0)))
    return tuple(stages)


def test_fixed_complete_history_and_unchanged_common_prefix(retained):
    report = retained[1]["report"]
    expected = _expected_stages()
    assert len(report["stages"]) == 32
    for stage, row in zip(report["stages"], expected):
        child = stage["child"]
        assert child is None or (type(child) is int and 0 <= child < 5)
        assert (
            exact_record(stage["start"]),
            exact_record(stage["duration"]),
            stage["name"],
            child,
            exact_record(stage["base"]),
            exact_record(stage["increment"]),
        ) == row
    formation, observed, omitted = (
        report[key] for key in ("formation", "observed", "omitted")
    )
    assert (
        formation["initial_center"]
        == tuple(Q(4092 * (c - 2)) for c in range(5)) + (Q(0),) * 5
    )
    assert formation["initial_radius"] == 0
    assert formation["name"] == "common_formation"
    assert (formation["start"], formation["end"], len(formation["steps"])) == (
        0,
        200,
        1600,
    )
    assert (observed["name"], omitted["name"]) == (
        "finite_observation",
        "omitted_observation",
    )
    for branch in (observed, omitted):
        assert branch["initial_center"] == formation["endpoint_center"]
        assert branch["initial_radius"] == formation["endpoint_radius"] > 0
        assert (branch["start"], branch["end"], len(branch["steps"])) == (200, 216, 143)
        assert len(formation["steps"]) + len(branch["steps"]) == 1743
    assert sum(len(leg["steps"]) for leg in (formation, observed, omitted)) == 1886
    assert report["branch"] == "outer_one"
    assert report["clock"] == "tau=(1023/1024)*t"
    assert report["maximum_step"] == Q(1, 8)
    assert report["maximum_steps_per_branch"] == 1800
    assert report["growth_upper"] == Q(1, 3000)
    assert (
        report["numerical_method"]
        == "prescribed_time_identity_metric_Taylor16_dyadic128"
    )
    assert report["coordinate_order"] == "Y[0:5],Psi[0:5]; exact command removed"
    assert report["phase_domain"] == "full_circle_with_declared_continuous_lifts"
    assert (
        report["source_scope"]
        == "conditional_on_original_independent_preparation_and_zero_sums"
    )


def _prescribed_command(time, stage, omit_word):
    start, duration, _name, child, base, increment = stage
    if omit_word or child is None:
        return (I(0),) * 5
    lower, upper = (time.lo - start) / duration, (time.hi - start) / duration
    # Rebuild the cubic range from its two endpoints and all critical points,
    # rather than copying the producer's interval evaluation order.
    candidates = [lower, upper] + [
        value for value in (Q(0), Q(1)) if lower <= value <= upper
    ]
    values = [base + increment * (3 * value**2 - 2 * value**3) for value in candidates]
    amplitude = I(min(values), max(values))
    return tuple(amplitude * (Q(int(c == child)) - Q(1, 5)) for c in range(5))


def _edge_built_rates(time, tube, stage, omit_word):
    gamma = Q(1, 1023) / pi_interval()
    y = tube[:5]
    phases = tuple(
        value + offset
        for value, offset in zip(tube[5:], _prescribed_command(time, stage, omit_word))
    )
    laplacian, sine = [I(0) for _ in range(5)], [I(0) for _ in range(5)]
    for c in range(5):
        d = (c + 1) % 5
        difference, current = y[c] - y[d], sin(phases[d] - phases[c])
        laplacian[c] += difference
        laplacian[d] -= difference
        sine[c] += current
        sine[d] -= current
    return tuple(
        (-value + gamma * current) / 4 for value, current in zip(laplacian, sine)
    ) + tuple(gamma * value / 4 for value in laplacian)


@lru_cache(maxsize=8)
def _independent_growth_upper(duration):
    exponent = duration / 3000
    assert 0 < exponent < Q(1, 10000)
    # The omitted positive exponential terms have successive ratio at most
    # exponent/14. This independent rational series bounds exp(exponent).
    partial = sum((exponent**k / factorial(k) for k in range(13)), Q(0))
    remainder = exponent**13 / factorial(13) / (1 - exponent / 14)
    return partial + remainder


@pytest.mark.parametrize(
    "name,omit_word", (("formation", True), ("observed", False), ("omitted", True))
)
def test_every_compact_step_retains_time_picard_and_full_metric_error(
    retained, name, omit_word
):
    report = retained[1]["report"]
    leg = report[name]
    assert leg["admitted"] is True and leg["reason"] is leg["failed_tube"] is None
    center, radius = _coordinates(leg["initial_center"]), exact_record(
        leg["initial_radius"]
    )
    elapsed = exact_record(leg["start"])
    stages = _expected_stages()[:1] if name == "formation" else _expected_stages()[1:]
    stage_index = 0
    for step in leg["steps"]:
        while elapsed == stages[stage_index][0] + stages[stage_index][1]:
            stage_index += 1
        stage = stages[stage_index]
        duration = exact_record(step["duration"])
        assert exact_record(step["time"]) == elapsed
        assert duration == min(Q(1, 8), stage[0] + stage[1] - elapsed)
        assert elapsed + duration <= stage[0] + stage[1]
        tube, polynomial, remainder = (
            tuple(map(_interval, step[key]))
            for key in ("tube", "polynomial", "remainder")
        )
        assert len(tube) == len(polynomial) == len(remainder) == 10
        initial = tuple(I(value - radius, value + radius) for value in center)
        rates = _edge_built_rates(
            I(elapsed, elapsed + duration), tube, stage, omit_word
        )
        image = tuple(
            value + rate * I(0, duration) for value, rate in zip(initial, rates)
        )
        assert all(
            bound.lo < value.lo <= value.hi < bound.hi
            for value, bound in zip(image, tube)
        )
        assert exact_record(step["picard_margin"]) > 0
        local = tuple(value + error for value, error in zip(polynomial, remainder))
        assert _coordinates(step["endpoint_center"]) == tuple(
            value.midpoint for value in local
        )
        error = exact_record(step["local_error"])
        assert error >= 0 and error**2 >= sum(
            (value.radius**2 for value in local), Q(0)
        )
        growth = exact_record(step["growth"])
        assert growth >= _independent_growth_upper(duration) > 1
        new_radius = exact_record(step["endpoint_radius"])
        assert new_radius == I(I(growth * radius).hi + error).hi
        assert new_radius >= growth * radius + error
        assert new_radius <= report["numerical_budget"] == Q(1, 10**15)
        center, radius = _coordinates(step["endpoint_center"]), new_radius
        elapsed += duration
    assert elapsed == leg["end"]
    assert center == leg["endpoint_center"] and radius == leg["endpoint_radius"]


def test_source_identity_and_suffix_work_are_rebuilt_from_declared_primitives(retained):
    report = retained[1]["report"]
    growth, horizon, duration = Q(1, 3000), Q(216), Q(16)
    hidden_phase = 2 * (3 * Q(2, 10**12) + growth * 3 * Q(1, 10**9))
    source = horizon * growth * hidden_phase**2 / (9 * (1 - (growth * horizon) ** 2))
    assert source == Q(49, 31088000000000000000000000)
    assert exact_record(report["source_error"]) == source < Q(16, 10**25)
    form, drift, collar = (
        Q(1, 100) + duration * growth,
        2 * growth / 100 + 2 * duration * growth**2,
        Q(1, 10**6),
    )
    pi = pi_interval()
    margins = (
        2 * pi.lo / 9 - 2 * collar - 2 * drift,
        pi.lo / 2 - 2 * pi.hi / 9 - 2 * collar - 2 * drift,
        pi.lo / 2 - Q(7, 5) - Q(1, 10) - 2 * collar - 2 * drift,
        pi.lo / 2 - 2 * collar - 4 * drift,
    )
    assert tuple(map(exact_record, report["identity_margins"])) == margins
    assert min(margins) > Q(314899, 4500000)
    suffix = horizon - 200 - 15 * (1 + Q(1, 100000))
    loss = Q(45, 4) * (8 * form) ** 2 * suffix
    assert report["unforced_tail_loss_upper"] == loss == Q(10578413, 62500000)
    assert exact_record(report["unforced_tail_supplied_work"]) == 0
    assert (
        exact_record(report["input_error"]) == exact_record(report["clock_error"]) == 0
    )


def test_record_separation_counts_each_error_source_once(retained):
    report = retained[1]["report"]
    source, reading = exact_record(report["source_error"]), exact_record(
        report["reading_error"]
    )
    assert reading == Q(1, 10**14)
    numerical, records = [], []
    for branch, forms_key, records_key in (
        ("observed", "predicted_form_bounds", "predicted_record_bounds"),
        ("omitted", "omitted_form_bounds", "omitted_record_bounds"),
    ):
        leg = report[branch]
        center, radius = _coordinates(leg["endpoint_center"]), exact_record(
            leg["endpoint_radius"]
        )
        expected = tuple(I(value - radius, value + radius) for value in center[:5])
        expanded = tuple(
            I(value - radius - source - reading, value + radius + source + reading)
            for value in center[:5]
        )
        assert tuple(map(_interval, report[forms_key])) == expected
        assert tuple(map(_interval, report[records_key])) == expanded
        numerical.append(expected)
        records.append(expanded)
    separation = max(
        max(Q(0), left.lo - right.hi, right.lo - left.hi)
        for left, right in zip(*numerical)
    )
    required = 2 * (source + reading)
    assert report["numerical_interval_separation"] == separation
    assert report["required_separation"] == required
    assert separation > required
    assert any(
        left.hi < right.lo or right.hi < left.lo for left, right in zip(*records)
    )
    assert report["numerical_certified"] is report["admitted"] is True
    assert report["reason"] is None
    assert retained[1]["verdict"] == "conditional_post_observation_forecast_separated"
