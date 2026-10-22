"""Analytic source geometry control, declaration binding and write-once transport."""

import hashlib
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from benchmarks import conservative_regional_winding as owner
from tnfr.utils.io import json_loads

DECLARATION = (
    Path(__file__).resolve().parents[2]
    / "docs/assets/conservative_source_geometry/declaration.json"
)


@pytest.fixture(scope="module")
def declaration():
    return json_loads(DECLARATION.read_bytes())


def _q(value):
    return Q(value["numerator"], value["denominator"])


def test_analytic_control_freezes_one_source_and_retains_equal_budget_obstruction(
    declaration, monkeypatch, tmp_path
):
    import tnfr.physics.relational_sine_forecast as forecast

    def forbidden(*args, **kwargs):
        pytest.fail(
            "the analytic control must not run a forecast or a different campaign"
        )

    monkeypatch.setattr(forecast, "bound_sine_flow", forbidden)
    monkeypatch.setattr(owner, "assess_control", forbidden)
    output = tmp_path / "certificate.json"
    assert owner.main(["--declaration", str(DECLARATION), "--output", str(output)]) == 0
    record = json_loads(output.read_bytes())
    assert record["schema"] == "tnfr.conservative-source-geometry-control.v1"
    assert record["passed"] and all(record["gates"].values())
    assert (
        record["declaration_sha256"]
        == hashlib.sha256(DECLARATION.read_bytes()).hexdigest()
    )
    archive = output.with_suffix(".source.zip")
    assert (
        record["source_archive_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    owner._verify_archive(archive, record["source_sha256"])
    positive = record["certificate"]["report"]
    assert positive["acute_acquisition_certified"]
    assert _q(positive["acute_margin_lower_bound"]) > Q(
        declaration["minimum_acute_margin"]
    )
    assert positive["certified_winding"] == 1
    assert record["reflection_control"]["report"]["zero_winding_when_nonantipodal"]
    assert record["source_geometry"]["report"]["relative_source_rank"] == 4
    # Independently reconstruct storage and the conserved full-degree mean
    # from both primitive sources, rather than trusting the matching flags.
    for key in ("initial_storage_balance", "control_storage_balance"):
        source = record[key]["report"]["comparison"]
        forms = tuple(map(_q, source["epi"]))
        degree = source["degrees"]
        energy = sum((forms[j] - forms[i]) ** 2 / 2 for i, j in source["edges"])
        assert energy == 4500
        assert sum(d * x for d, x in zip(degree, forms)) == 0
    retained = output.read_bytes()
    with pytest.raises(FileExistsError):
        owner.main(["--declaration", str(DECLARATION), "--output", str(output)])
    assert output.read_bytes() == retained


@pytest.mark.parametrize(
    "mutation",
    (
        "capacity_boolean",
        "missing_form",
        "wrong_law",
        "winding_boolean",
        "window_boolean",
        "margin_boolean",
        "edge_boolean",
        "edge_float",
        "duplicate_edge",
        "undeclared_node",
        "cycle_boolean",
    ),
)
def test_invalid_declaration_is_not_coerced(declaration, mutation):
    changed = deepcopy(declaration)
    if mutation == "capacity_boolean":
        changed["capacity"][0] = True
    elif mutation == "missing_form":
        changed["form"].pop()
    elif mutation == "wrong_law":
        changed["law"] = "native"
    elif mutation == "winding_boolean":
        changed["declared_window_winding"] = True
    elif mutation == "window_boolean":
        changed["scaled_window"][1] = True
    elif mutation == "margin_boolean":
        changed["minimum_acute_margin"] = True
    elif mutation == "edge_boolean":
        changed["ring_edges"][0][1] = True
    elif mutation == "edge_float":
        changed["ring_edges"][0][1] = 1.0
    elif mutation == "duplicate_edge":
        changed["connecting_edges"].append([1, 0])
    elif mutation == "undeclared_node":
        changed["connecting_edges"].append([0, len(changed["nodes"])])
    else:
        changed["receiver_cycle"][1] = True
    with pytest.raises((TypeError, ValueError)):
        owner.assess_source_geometry_control(changed)


def test_failed_acute_bound_remains_a_failed_sufficient_control(declaration):
    changed = deepcopy(declaration)
    changed["minimum_acute_margin"] = "1"
    report = owner.assess_source_geometry_control(changed)
    assert not report["passed"]
    assert not report["gates"]["declared_acute_margin"]
    assert report["gates"]["finite_acute_acquisition"]
    assert report["gates"]["control_acute_entry_excluded"]


def test_unavailable_margin_cannot_pass_the_declared_margin_gate(
    declaration, monkeypatch
):
    original = owner.certify_sine_conservative_winding_entry

    def unavailable(*args, **kwargs):
        return replace(original(*args, **kwargs), acute_margin_lower_bound=None)

    monkeypatch.setattr(owner, "certify_sine_conservative_winding_entry", unavailable)
    report = owner.assess_source_geometry_control(declaration)
    assert not report["passed"]
    assert not report["gates"]["declared_acute_margin"]


def test_unknown_schema_does_not_fall_back_to_an_old_control(tmp_path, monkeypatch):
    declaration = tmp_path / "declaration.json"
    declaration.write_text('{"schema":"unsupported"}', encoding="utf-8")
    monkeypatch.setattr(owner, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("an unsupported declaration must not select another experiment")

    monkeypatch.setattr(owner, "assess_control", forbidden)
    monkeypatch.setattr(owner, "assess_source_geometry_control", forbidden)
    output = tmp_path / "certificate.json"
    with pytest.raises(ValueError, match="unsupported analytic declaration schema"):
        owner.main(["--declaration", str(declaration), "--output", str(output)])
    assert not output.exists() and not output.with_suffix(".source.zip").exists()


@pytest.mark.parametrize("change", ("declaration_before_archive", "archive", "runtime"))
def test_certificate_requires_stable_evaluated_bytes_and_archive(
    declaration, tmp_path, monkeypatch, change
):
    # Small files test evidence transport only, without another scientific run.
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text(owner.evidence._encoded(declaration), encoding="utf-8")
    producer = tmp_path / "producer.py"
    producer.write_bytes(b"# synthetic analytic producer\n")
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    monkeypatch.setattr(owner, "__file__", str(producer))
    original_runtime = owner.evidence._runtime()
    runtime = dict(original_runtime)
    monkeypatch.setattr(owner.evidence, "_runtime", lambda: dict(runtime))

    def files():
        if change == "declaration_before_archive":
            declaration_path.write_text('{"changed":true}', encoding="utf-8")
        return {}

    monkeypatch.setattr(owner.evidence, "_source_files", files)
    output = tmp_path / "certificate.json"

    def assess(received):
        assert received == declaration
        if change == "archive":
            with output.with_suffix(".source.zip").open("ab") as stream:
                stream.write(b"modified after verification")
        elif change == "runtime":
            runtime["fixture_change"] = "changed during evaluation"
        return {"passed": True}

    monkeypatch.setattr(owner, "assess_source_geometry_control", assess)
    with pytest.raises(ValueError, match="changed during evaluation"):
        owner.main(["--declaration", str(declaration_path), "--output", str(output)])
    assert not output.exists()
    assert output.with_suffix(".source.zip").exists()
