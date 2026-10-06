"""Frozen reverse-preparation transport; never evaluate its research trajectory."""

import copy
from fractions import Fraction as Q

import pytest

from benchmarks import conservative_regional_organization as owner
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_regional as regional
from tnfr.utils.io import json_loads


@pytest.fixture
def declaration():
    form = tuple(Q(value, 32) for value in (-4, -4, 0, 4, 4, -6, -5, 0, 5, 6))
    phase = tuple((i - 2) * Q(1256637, 10**6) for i in range(5)) * 2
    return {
        "schema": "tnfr.conservative-regional-organization-declaration.v1",
        "evaluation_kind": "reversible_preparation",
        "nodes": list(range(10)),
        "edges": [[i, (i + 1) % 5] for i in range(5)] + [[i, 5 + i] for i in range(5)],
        "form": [str(-value) for value in form],
        "phase": list(map(str, phase)),
        "capacity": [1] * 10,
        "checkpoint_form": list(map(str, form)),
        "checkpoint_phase": list(map(str, phase)),
        "law": "normalized_sine_reciprocal_e0_w1_beta1",
        "support": "fixed_simple_connected_unit_undirected",
        "forcing": "none",
        "events": "none",
        "clock": "original_structural_t; tau=t/pi",
        "observation_time": "0",
        "end_time": "128/5",
        "time_step": "1/10",
        "order": 16,
        "maximum_steps": 256,
        "receiver_cycle_indices": list(range(5)),
        "target_error_bound": "1/4096",
        "source_error_bound": str(Q(1, 1 << 100)),
        "scaled_retention_duration": "1",
        "selection_rule": "first_certified_declared_endpoint_after_backward_retention_window",
    }


def test_reverse_preflight_uses_exact_checkpoint_and_retention_without_a_flow(
    declaration, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("preparation cannot consume the reserved trajectory")

    monkeypatch.setattr(owner, "bound_sine_flow", forbidden)
    prepared = owner._preparation(declaration)
    checkpoint, options = owner._reversible_checkpoint(declaration, prepared)
    assert prepared["end_time"] / prepared["time_step"] == 256
    assert prepared["order"] == 16 and len(prepared["initial"]) == 21
    assert prepared["model"].effective_weights == (0, 1)
    assert checkpoint.epi == tuple(-Q(value) for value in declaration["form"])
    assert checkpoint.phase == tuple(map(Q, declaration["phase"]))
    assert (
        sum(degree * form for degree, form in zip(checkpoint.degrees, checkpoint.epi))
        == 0
    )
    assert checkpoint.form_storage == Q(53, 1024)
    assert checkpoint.storage.lo > Q(7, 2)
    target = regional.assess_sine_cycle_retention(
        checkpoint,
        cycle=options["cycle"],
        scaled_duration=options["scaled_retention_duration"],
        source_error_bound=options["target_error_bound"],
    )
    assert target.whole_window_retention_certified
    assert options["source_error_bound"] == Q(1, 1 << 100)
    assert "minimum_duration" not in declaration and "acute_margin" not in declaration


def test_evaluation_calls_one_shared_forecast_and_reversible_consumer(
    declaration, monkeypatch
):
    forecast, calls = object(), []
    expected = owner._preparation(declaration)

    def run(**kwargs):
        assert kwargs == expected
        calls.append("flow")
        return forecast

    response = {
        "schema": "tnfr.sine-reversible-preparation.v1",
        "report": {"outcome": "unavailable"},
    }

    class Report:
        def to_dict(self):
            return response

    def assess(actual, checkpoint, **kwargs):
        assert actual is forecast
        assert checkpoint.epi == tuple(map(Q, declaration["checkpoint_form"]))
        assert checkpoint.phase == tuple(map(Q, declaration["checkpoint_phase"]))
        assert kwargs == {
            "cycle": tuple(range(5)),
            "target_error_bound": Q(1, 4096),
            "source_error_bound": Q(1, 1 << 100),
            "scaled_retention_duration": Q(1),
        }
        calls.append("observer")
        return Report()

    monkeypatch.setattr(owner, "bound_sine_flow", run)
    monkeypatch.setattr(
        regional, "assess_sine_reversible_preparation", assess, raising=False
    )
    assert owner.evaluate(declaration) is response
    assert calls == ["flow", "observer"]


@pytest.mark.parametrize(
    "field,value",
    (
        ("evaluation_kind", "unknown"),
        ("evaluation_kind", True),
        ("selection_rule", "largest_success_after_observation"),
        ("target_error_bound", True),
        ("target_error_bound", 0),
        ("target_error_bound", 2),
        ("source_error_bound", False),
        ("source_error_bound", 0),
        ("source_error_bound", float("inf")),
        ("scaled_retention_duration", True),
        ("scaled_retention_duration", -1),
        ("forcing", "supplied"),
        ("events", "allowed"),
        ("clock", "scaled_tau"),
        ("observation_time", "1/10"),
        ("maximum_steps", 255),
        ("order", 17),
        ("receiver_cycle_indices", [False, 1, 2, 3, 4]),
    ),
)
def test_invalid_reverse_declarations_reject_before_flow(
    declaration, field, value, monkeypatch
):
    declaration[field] = value

    def forbidden(**kwargs):
        pytest.fail("invalid declaration must not evaluate a trajectory")

    monkeypatch.setattr(owner, "bound_sine_flow", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner.evaluate(declaration)


@pytest.mark.parametrize("field", ("checkpoint_form", "checkpoint_phase"))
def test_checkpoint_requires_every_admitted_exact_coordinate(declaration, field):
    missing = copy.deepcopy(declaration)
    missing[field].pop()
    with pytest.raises(ValueError, match="every exact checkpoint coordinate"):
        owner._preparation(missing)
    declaration[field][0] = False
    with pytest.raises((TypeError, ValueError)):
        owner._preparation(declaration)


def test_exact_reverse_association_cannot_be_replaced_by_equal_materialized_boxes(
    declaration,
):
    original = Q(declaration["phase"][0])
    altered = original + Q(1, 1 << 250)
    assert original != altered and I(original) == I(altered)
    declaration["phase"][0] = str(altered)
    with pytest.raises(ValueError, match="exactly R"):
        owner._preparation(declaration)


def _transport_fixture(declaration, monkeypatch, tmp_path):
    path = tmp_path / "declaration.json"
    path.write_text(owner.evidence._encoded(declaration), encoding="utf-8")
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    monkeypatch.setattr(owner, "DECLARATION", path)
    monkeypatch.setattr(
        owner,
        "_source_files",
        lambda selected: {
            "declaration.json": selected.read_bytes(),
            "instrument.py": b"transport test only; no numerical evaluation",
        },
    )
    output = tmp_path / "response.json"
    return path, output, ["--declaration", str(path), "--output", str(output)]


@pytest.mark.parametrize(
    "outcome", ("certified", "unavailable", "no_certificate_on_declared_grid")
)
def test_frozen_mode_retains_each_outcome_once_without_rerunning(
    declaration, monkeypatch, tmp_path, outcome
):
    _, output, args = _transport_fixture(declaration, monkeypatch, tmp_path)
    calls = []

    def evaluate(actual):
        calls.append(actual)
        return {
            "schema": "tnfr.sine-reversible-preparation.v1",
            "report": {"outcome": outcome},
        }

    monkeypatch.setattr(owner, "evaluate", evaluate)
    assert owner.main(["--prepare", *args]) == 0
    assert calls == []
    assert owner.main(args) == (0 if outcome == "certified" else 1)
    record = json_loads(output.read_bytes())
    assert record["passed"] is (outcome == "certified")
    assert record["evaluation_error"] is None
    assert record["protocol"]["declaration"] == declaration
    assert record["response"]["report"]["outcome"] == outcome
    assert len(calls) == 1
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        owner.main(args)
    assert output.read_bytes() == original and len(calls) == 1
    with pytest.raises(ValueError, match="completed response evaluation"):
        owner._load_channel_source(output)


def test_changed_checkpoint_cannot_reuse_a_frozen_declaration(
    declaration, monkeypatch, tmp_path
):
    path, output, args = _transport_fixture(declaration, monkeypatch, tmp_path)
    assert owner.main(["--prepare", *args]) == 0
    changed = copy.deepcopy(declaration)
    changed["source_error_bound"] = str(Q(1, 1 << 99))
    path.write_text(owner.evidence._encoded(changed), encoding="utf-8")

    def forbidden(actual):
        pytest.fail("a changed declaration cannot evaluate a frozen trial")

    monkeypatch.setattr(owner, "evaluate", forbidden)
    with pytest.raises(
        ValueError, match="frozen source, runtime or declaration changed"
    ):
        owner.main(args)
    assert not output.exists()


def test_reverse_mode_cannot_consume_a_legacy_success_schema(
    declaration, monkeypatch, tmp_path
):
    _, output, args = _transport_fixture(declaration, monkeypatch, tmp_path)
    assert owner.main(["--prepare", *args]) == 0
    monkeypatch.setattr(
        owner,
        "evaluate",
        lambda actual: {
            "schema": "tnfr.sine-regional-organization.v1",
            "report": {"outcome": "certified"},
        },
    )
    assert owner.main(args) == 1
    record = json_loads(output.read_bytes())
    assert record["passed"] is False
    assert record["evaluation_error"]["error_type"] == "ValueError"
