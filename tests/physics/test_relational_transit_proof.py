"""Retained validated-transit witnesses without repeating the time integration."""

import hashlib
import json
import zipfile
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from benchmarks import relational_transit_proof as audit
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_transit as owner

HASHES = {
    "continuous-transit.audit.json": "ac3104c42cf9054968eac770fee5c1b1e174547a408875edac0b635fe04ca781",
    "continuous-transit.audit.protocol.json": "d8913d6af0d40fd1c903eaf24be29fb8ebeb6d3be28d6415d4e8dafb9974db99",
    "continuous-transit.audit.source.zip": "381212374968f32fe59751c752b4dc812844a3309e00f771704a72b31a154c93",
}


def interval(record):
    return I(record["lo"], record["hi"])


@pytest.fixture(scope="module")
def retained():
    directory = audit.RECORDS
    for name, digest in HASHES.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    protocol = audit._read(directory / "continuous-transit.audit.protocol.json")
    report = audit._read(directory / "continuous-transit.audit.json")
    assert report["protocol"] == protocol
    return protocol, report, report["certificate"]["report"]


def test_source_archive_preserves_every_executed_source(retained):
    protocol, _, _ = retained
    with zipfile.ZipFile(
        audit.RECORDS / "continuous-transit.audit.source.zip"
    ) as source:
        assert set(source.namelist()) == set(protocol["source_sha256"])
        for path, digest in protocol["source_sha256"].items():
            assert hashlib.sha256(source.read(path)).hexdigest() == digest


def test_original_represented_seed_and_old_verdict_remain_distinct(retained):
    protocol, report, certificate = retained
    original, response = audit._retained()
    for name in (
        "initial_form",
        "initial_phase",
        "capacity",
        "cycles",
        "edges",
        "model",
    ):
        assert protocol[name] == original[name]
    assert not response["finite_prediction_passed"]
    assert not report["original_finite_prediction_passed"]
    assert report["continuous_capture_admitted"]
    assert certificate["initial_winding_zero"]
    box = tuple(map(interval, certificate["initial_box"]))
    assert box[0] == I(1 + Q(1, 1 << 54))
    assert box[1] == I(0)
    assert box[2] == box[3] == I(Q(original["initial_phase"][0]))


def test_every_retained_whole_time_tube_has_strict_self_inclusion(retained):
    _, _, report = retained
    box = tuple(map(interval, report["initial_box"]))
    time = Q(0)
    for step in report["steps"]:
        assert step["time"] == time
        assert step["duration"] == Q(1, 8)
        tube = tuple(map(interval, step["tube"]))
        assert all(value.subset_of(bound) for value, bound in zip(box, tube))
        lower = owner._regular_bounds(tube)
        assert tuple(step["resultant_real_lower_bounds"]) == lower
        assert min(lower) > 0
        rate = owner._flow(tube, Q(1, 2), Q(1, 2), Q(1))
        image = tuple(x + f * I(0, step["duration"]) for x, f in zip(box, rate))
        margin = min(min(y.lo - b.lo, b.hi - y.hi) for y, b in zip(image, tube))
        assert margin == step["picard_interior_margin"] > 0
        box = tuple(map(interval, step["endpoint"]))
        assert all(value.subset_of(bound) for value, bound in zip(box, tube))
        time += step["duration"]
    assert time == report["horizon"] == report["validated_horizon"] == 32
    assert len(report["steps"]) == 256
    assert box == tuple(map(interval, report["endpoint"]))


def test_entire_endpoint_box_admits_existing_protected_capture(retained):
    _, _, report = retained
    box = tuple(map(interval, report["endpoint"]))
    storage = owner._storage(box, Q(1))
    margins = owner._positive_margins(box)
    assert storage == interval(report["endpoint_storage"])
    assert all(value.lo > 0 for value in margins)
    assert tuple(map(interval, report["positive_rectangle_margins"])) == margins
    assert storage.hi < Q(6971, 1000) < 7
    assert max(value.width for value in box) < Q(7, 10**15)
    assert report["target_sector"] == 1 and report["status"] == "admitted"
    assert not report["unavailable_reasons"] and report["failed_tube"] is None


def test_changed_frozen_policy_rejects_before_any_calculation(retained, monkeypatch):
    protocol, _, _ = retained
    monkeypatch.setattr(audit, "prepare_protocol", lambda: deepcopy(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("changed protocol executed")

    monkeypatch.setattr(owner, "certify_relational_transit_capture", forbidden)
    changed = deepcopy(protocol)
    changed["numerical_policy"]["horizon"] = Q(64)
    with pytest.raises(ValueError, match="frozen proof protocol"):
        audit.evaluate_protocol(changed)


def test_cli_preserves_freeze_failure_and_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    monkeypatch.setattr(audit, "prepare_protocol", lambda: {"source_sha256": {}})

    def fail(protocol):
        raise RuntimeError("synthetic proof failure")

    monkeypatch.setattr(audit, "evaluate_protocol", fail)
    assert audit.main(["--prepare", "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        audit.main(["--prepare", "--output", str(output)])
    with pytest.raises(RuntimeError):
        audit.main(["--output", str(output)])
    assert "synthetic proof failure" in json.loads(output.read_text())["error"]
    with pytest.raises(FileExistsError):
        audit.main(["--output", str(output)])
