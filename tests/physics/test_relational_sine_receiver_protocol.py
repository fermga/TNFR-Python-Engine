"""Frozen H2 protocol and verdict wiring, without running its reserved IVP.

The fabricated step records below test the consumer's logic only. They are
deliberately not numerical certificates of a physical solution or its source.
"""

import hashlib
import zipfile
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from mpmath import mp

from benchmarks import relational_receiver_barrier as producer
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import ValidatedTaylorStep
from tnfr.physics.relational_sine_forecast import SineForecast
from tnfr.research import relational_receiver_barrier as owner
from tnfr.utils.io import json_loads


def _forbidden(*args, **kwargs):
    pytest.fail("a routine protocol control must not evaluate the frozen IVP")


@pytest.fixture(autouse=True)
def no_reserved_trajectory(monkeypatch):
    monkeypatch.setattr(owner, "bound_sine_flow", _forbidden)
    monkeypatch.setattr(producer, "evaluate_receiver_barrier", _forbidden)


@pytest.fixture(scope="module")
def preparation():
    return owner._preparation()


def _synthetic_forecast(preparation, *, endpoint=None):
    """Structurally consistent records only; no field or step is evaluated."""
    source, initial = preparation
    if endpoint is None:
        # Keep the donor twist, but fabricate removal of the initial form.
        endpoint = initial[:10] + (I(0),) + initial[11:]
    first_tube = tuple(a.hull(b) for a, b in zip(initial, endpoint))
    steps = tuple(
        ValidatedTaylorStep(
            time=Q(index, 8),
            duration=Q(1, 8),
            tube=first_tube if index == 0 else endpoint,
            endpoint=endpoint,
            picard_interior_margin=Q(1),
            domain_lower_bounds=(Q(1),),
            propagated_initial_radii=(Q(0),) * 23,
            local_remainder_bounds=(I(0),) * 23,
        )
        for index in range(256)
    )
    return SineForecast(
        model=source.model,
        neighbors=source.neighbors,
        visible_capacity=source.capacity[:-1],
        initial_box=initial,
        observation_time=Q(0),
        end_time=Q(32),
        time_step=Q(1, 8),
        order=16,
        steps=steps,
        validated_end_time=Q(32),
        endpoint=endpoint,
        failed_tube=None,
        status="admitted",
        reasons=(),
    )


@pytest.fixture(scope="module")
def synthetic(preparation):
    return _synthetic_forecast(preparation)


def _number(context, rational):
    return context.mpf(rational.numerator) / rational.denominator


def _assert_encloses(bounds, expected, context):
    assert _number(context, bounds.lo) <= expected <= _number(context, bounds.hi)


def test_frozen_source_is_full_exact_turn_preparation_without_execution(preparation):
    source, initial = preparation
    protocol = owner.prepare_receiver_barrier()
    context = mp.clone()
    context.dps = 90
    edges = {
        tuple(sorted((offset + i, offset + (i + 1) % 5)))
        for offset in (0, 5)
        for i in range(5)
    } | {(0, 10), (5, 10)}
    assert set(source.edges) == edges
    assert len(initial) == 23 and len(source.nodes) == 11
    assert source.initial_epi == (Q(0),) * 10 + (Q(2),)
    assert source.capacity == (
        Q(1),
        Q(1),
        Q(1),
        Q(1),
        Q(1),
        Q(1),
        Q(3, 2),
        Q(1),
        Q(1),
        Q(1, 2),
        Q(1),
    )
    assert source.model.effective_weights == (0.5, 0.5)
    assert source.model.storage_scale == 1
    for i in range(11):
        expected = 2 * context.pi * i / 5 if i < 5 else context.mpf(0)
        _assert_encloses(initial[11 + i], expected, context)
        assert (initial[11 + i].width > 0) == (0 < i < 5)
    assert initial[-1] == I(1)
    assert source.initial_form_storage == 4
    expected_energy = 4 + (25 - 5 * context.sqrt(5)) / 4
    _assert_encloses(owner._storage(initial, tuple(edges)), expected_energy, context)
    assert protocol["state_layout"] == "x0..x10,theta0..theta10,held_nu10"
    assert protocol["end_time"] == {"numerator": 32, "denominator": 1}
    assert protocol["time_step"] == {"numerator": 1, "denominator": 8}
    assert protocol["order"] == 16 and protocol["maximum_steps"] == 256
    assert "128" in protocol["arithmetic"]
    assert "no_early_tail_closure" in protocol["failure_policy"]


def test_complete_tubes_and_full_endpoint_are_both_needed(preparation, synthetic):
    before = deepcopy(synthetic)
    verdict = owner.assess_receiver_barrier_forecast(synthetic)
    assert verdict["status"] == "certified_receiver_barrier_exclusion"
    assert verdict["validated_steps"] == 256
    assert verdict["full_horizon_completed"]
    assert verdict["all_future_receiver_potential_barrier_excluded"]
    assert verdict["maintained_receiver_twists_excluded"]
    assert verdict["asymptotic_receiver_consensus_certified"]
    assert verdict["receiver_potential_upper_by_tube"] == (Q(0),) * 256
    assert 0 < verdict["endpoint_tail_margin"] < Q(1, 20)
    assert synthetic == before

    # A tube can contain a receiver-barrier state even when both endpoints
    # have flat receiver phases. A wide enclosure is not a crossing verdict.
    middle = synthetic.steps[137]
    tube = list(middle.tube)
    tube[11 + 6] = I(0, 4)
    steps = list(synthetic.steps)
    steps[137] = replace(middle, tube=tuple(tube))
    wide = owner.assess_receiver_barrier_forecast(
        replace(synthetic, steps=tuple(steps))
    )
    assert wide["status"] == "unresolved_receiver_prefix"
    assert wide["receiver_prefix_margin"] <= 0
    assert wide["full_horizon_tail_passed"]
    assert not wide["all_future_receiver_potential_barrier_excluded"]
    assert not wide["asymptotic_receiver_consensus_certified"]

    # Both bridges contribute form storage although the receiver ring itself
    # is flat. Keeping H=1/4 at this synthetic endpoint defeats the tail bound.
    endpoint = list(synthetic.endpoint)
    endpoint[10] = I(Q(1, 4))
    tail = owner.assess_receiver_barrier_forecast(
        _synthetic_forecast(preparation, endpoint=tuple(endpoint))
    )
    context = mp.clone()
    context.dps = 90
    expected = (25 - 5 * context.sqrt(5)) / 4 + context.mpf(1) / 16
    _assert_encloses(tail["endpoint_total_storage_bounds"], expected, context)
    assert tail["full_prefix_passed"]
    assert tail["status"] == "unresolved_tail"
    assert not tail["all_future_receiver_potential_barrier_excluded"]


def test_partial_low_energy_endpoint_cannot_shorten_the_frozen_horizon(synthetic):
    partial = replace(
        synthetic,
        steps=synthetic.steps[:1],
        validated_end_time=Q(1, 8),
        failed_tube=synthetic.steps[1].tube,
        status="unavailable",
        reasons=("synthetic_comparison_bound_unresolved",),
    )
    verdict = owner.assess_receiver_barrier_forecast(partial)
    assert verdict["endpoint_tail_margin"] > 0
    assert verdict["receiver_prefix_margin"] > 0
    assert verdict["status"] == "unavailable_prefix"
    assert verdict["validated_end_time"] == Q(1, 8)
    assert not verdict["full_horizon_completed"]
    assert not verdict["full_prefix_passed"]
    assert not verdict["full_horizon_tail_passed"]
    assert not verdict["all_future_receiver_potential_barrier_excluded"]
    assert verdict["numerical_reasons"] == partial.reasons
    with pytest.raises(ValueError):
        owner.assess_receiver_barrier_forecast(replace(partial, status="admitted"))


@pytest.mark.parametrize(
    "change",
    (
        {"model": RelationalExchangeModel(2, phase_domain="regular")},
        {"model": RelationalExchangeModel(1, 3, 2, phase_domain="regular")},
        {"neighbors": ((1,), (0,))},
        {"visible_capacity": (Q(1),) * 10},
        {"visible_capacity": (True,) + (Q(1),) * 5 + (Q(3, 2), Q(1), Q(1), Q(1, 2))},
        {"observation_time": True},
        {"end_time": Q(31)},
        {"time_step": Q(1, 16)},
        {"order": 8},
        {"freeze_hidden": True},
        {"forecast_start": Q(1)},
        {"initial_box": (I(0),) * 23},
    ),
)
def test_other_source_law_clock_or_budget_is_rejected(synthetic, change):
    with pytest.raises((ValueError, TypeError)):
        owner.assess_receiver_barrier_forecast(replace(synthetic, **change))


def test_coverage_and_enclosure_structure_are_checked(synthetic):
    first = synthetic.steps[0]
    malformed = (
        replace(first, time=Q(1, 8)),
        replace(first, duration=Q(1, 16)),
        replace(first, picard_interior_margin=Q(0)),
        replace(first, domain_lower_bounds=(Q(0),)),
        replace(first, tube=first.tube[:-1]),
        replace(first, tube=synthetic.endpoint),
        replace(first, endpoint=(I(99),) + first.endpoint[1:]),
    )
    for step in malformed:
        candidate = replace(synthetic, steps=(step,) + synthetic.steps[1:])
        with pytest.raises((TypeError, ValueError)):
            owner.assess_receiver_barrier_forecast(candidate)
    for change in (
        {"validated_end_time": Q(31)},
        {"endpoint": synthetic.initial_box},
        {"steps": synthetic.steps + (synthetic.steps[-1],)},
    ):
        with pytest.raises((TypeError, ValueError)):
            owner.assess_receiver_barrier_forecast(replace(synthetic, **change))

    # The final endpoint has no subsequent tube to expose the inconsistency.
    last = synthetic.steps[-1]
    escaped = (I(99),) + last.endpoint[1:]
    outside = replace(
        synthetic,
        steps=synthetic.steps[:-1] + (replace(last, endpoint=escaped),),
        endpoint=escaped,
    )
    with pytest.raises(ValueError, match="initial and endpoint"):
        owner.assess_receiver_barrier_forecast(outside)

    # A larger tube does not authorize changing the exactly held capacity.
    changed_capacity = last.endpoint[:-1] + (I(2),)
    changed_tube = last.tube[:-1] + (I(1, 2),)
    changed = replace(
        synthetic,
        steps=synthetic.steps[:-1]
        + (replace(last, tube=changed_tube, endpoint=changed_capacity),),
        endpoint=changed_capacity,
    )
    with pytest.raises(ValueError, match="held intermediary capacity"):
        owner.assess_receiver_barrier_forecast(changed)


def test_evaluator_delegates_one_full_frozen_call_without_reinterpretation(
    preparation, synthetic, monkeypatch
):
    source, initial = preparation
    calls = []

    def flow(received, **options):
        calls.append((received, options))
        return synthetic

    monkeypatch.setattr(owner, "bound_sine_flow", flow)
    result = owner.evaluate_receiver_barrier()
    assert calls == [
        (
            initial,
            {
                "neighbors": source.neighbors,
                "visible_capacity": source.capacity[:-1],
                "model": source.model,
                "observation_time": 0,
                "end_time": Q(32),
                "time_step": Q(1, 8),
                "order": 16,
            },
        )
    ]
    assert result["schema"] == "tnfr.sine-receiver-barrier-response.v1"
    assert result["declaration"] == owner.prepare_receiver_barrier()
    assert result["forecast"] == synthetic.to_dict()
    assert result["assessment"]["all_future_receiver_potential_barrier_excluded"]


@pytest.fixture
def plumbing(monkeypatch):
    files = {"fixture.py": b"# Protocol wiring only; not scientific source.\n"}
    declaration = {"schema": "fixture.receiver.protocol", "horizon": 32}
    monkeypatch.setattr(producer, "_source_files", lambda: dict(files))
    monkeypatch.setattr(
        producer, "prepare_receiver_barrier", lambda: deepcopy(declaration)
    )
    return files, declaration


def _prepare(tmp_path):
    output = tmp_path / "response.json"
    args = ["--output", str(output)]
    assert producer.main(["--prepare", *args]) == 0
    return output, args


@pytest.mark.parametrize("certified", (False, True))
def test_benchmark_prepares_then_retains_one_immutable_response(
    plumbing, monkeypatch, tmp_path, certified
):
    files, declaration = plumbing
    output, args = _prepare(tmp_path)
    frozen, archive = output.with_suffix(".protocol.json"), output.with_suffix(
        ".source.zip"
    )
    original = frozen.read_bytes(), archive.read_bytes()
    assert not output.exists()
    protocol = json_loads(original[0])
    assert protocol["protocol"] == declaration
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(files)
        assert bundle.read("fixture.py") == files["fixture.py"]
    calls = []

    def evaluate():
        calls.append(True)
        return {
            "assessment": {
                "status": (
                    "certified_receiver_barrier_exclusion"
                    if certified
                    else "unavailable_prefix"
                ),
                "all_future_receiver_potential_barrier_excluded": certified,
            },
            "fixture": "synthetic response plumbing only",
        }

    monkeypatch.setattr(producer, "evaluate_receiver_barrier", evaluate)
    producer.main(args)
    saved = output.read_bytes()
    record = json_loads(saved)
    assert record["response"]["fixture"] == "synthetic response plumbing only"
    assert record["protocol_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert record["source_archive_sha256"] == hashlib.sha256(original[1]).hexdigest()
    assert calls == [True]
    assert (frozen.read_bytes(), archive.read_bytes()) == original
    for flags in (args, ["--prepare", *args]):
        with pytest.raises(FileExistsError):
            producer.main(flags)
    assert output.read_bytes() == saved and calls == [True]


@pytest.mark.parametrize("change", ("source", "protocol", "archive"))
def test_benchmark_rejects_changed_preparation_before_reserved_response(
    plumbing, tmp_path, change
):
    files, _ = plumbing
    output, args = _prepare(tmp_path)
    if change == "source":
        files["fixture.py"] = b"changed before evaluation\n"
    elif change == "protocol":
        frozen = output.with_suffix(".protocol.json")
        declaration = json_loads(frozen.read_bytes())
        declaration["protocol"]["horizon"] = 31
        frozen.write_text(producer.evidence._encoded(declaration), encoding="utf-8")
    else:
        with zipfile.ZipFile(output.with_suffix(".source.zip"), "w") as archive:
            archive.writestr("fixture.py", b"changed archived source\n")
    with pytest.raises(ValueError):
        producer.main(args)
    assert not output.exists()


@pytest.mark.parametrize("failure", ("exception", "source_change"))
def test_benchmark_retains_execution_failure_instead_of_allowing_rerun(
    plumbing, monkeypatch, tmp_path, failure
):
    files, _ = plumbing
    output, args = _prepare(tmp_path)

    def evaluate():
        if failure == "exception":
            raise ArithmeticError("injected failed enclosure")
        files["fixture.py"] = b"changed during evaluation\n"
        return {
            "assessment": {
                "status": "certified_receiver_barrier_exclusion",
                "all_future_receiver_potential_barrier_excluded": True,
            }
        }

    monkeypatch.setattr(producer, "evaluate_receiver_barrier", evaluate)
    assert producer.main(args) == 1
    saved = output.read_bytes()
    record = json_loads(saved)
    assert record["evaluation_error"] is not None
    assert record["passed"] is False
    with pytest.raises(FileExistsError):
        producer.main(args)
    assert output.read_bytes() == saved
