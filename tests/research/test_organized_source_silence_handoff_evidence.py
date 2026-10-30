"""Read-only reconstruction of the two retained Silence suffixes.

No driver import, event, source replay or trajectory belongs to this owner.
Independent incidence sums check complete rows and the changed weighted leaf.
Latency diagnostics are recorded telemetry, not temporal maintenance evidence.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions import Coherence, Reception, Silence
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as owner
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/assets/organized_source_silence_handoff"
UPSTREAM = ROOT / "docs/assets/organized_source_coherence_handoff"
TARGETS = (0, 1)


@pytest.fixture(scope="module", autouse=True)
def no_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Silence evidence audit may not execute an event or trajectory")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_pointwise_stage"),
            (Reception, "_execute"),
            (Coherence, "_execute"),
            (Silence, "_execute"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _read(base, name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(base / name, max_bytes=2**27))
    )


def _field(snapshot, name):
    return tuple(row[name] for row in snapshot["nodes"])


def _observe(snapshot):
    return owner._observe_organized_source_raw_capacity(
        form=_field(snapshot, "form"),
        phases=_field(snapshot, "phase"),
        capacities=_field(snapshot, "capacity"),
        phase_mobility="shared_capacity",
    )


@pytest.fixture(scope="module")
def evidence():
    inventories = {
        base: tuple(sorted(path for path in base.iterdir() if path.is_file()))
        for base in (BASE, UPSTREAM)
    }
    receipts = {
        base: tuple(file_receipt(path) for path in paths)
        for base, paths in inventories.items()
    }
    records = {
        name: _read(BASE, name)
        for name in ("protocol.json", "freeze.json", "attempt.json", "response.json")
    }
    records["upstream"] = _read(UPSTREAM, "response.json")
    records["upstream_protocol"] = _read(UPSTREAM, "protocol.json")
    yield records
    for base, paths in inventories.items():
        assert paths == tuple(sorted(path for path in base.iterdir() if path.is_file()))
        assert receipts[base] == tuple(file_receipt(path) for path in paths)


@pytest.fixture(scope="module")
def reports(evidence):
    turns = evidence["protocol.json"]["phase_representation"]["turns"]
    return tuple(
        (
            _observe(stage["before"]),
            _observe(stage["after"]),
            capture._assess_organized_source_raw_capacity_retention(
                form=_field(stage["after"], "form"),
                phases=_field(stage["after"], "phase"),
                capacities=_field(stage["after"], "capacity"),
                phase_turns=turns,
            ),
        )
        for stage in evidence["response.json"]["stages"]
    )


def test_source_archive_upstream_association_and_exact_counts(evidence):
    protocol, freeze = evidence["protocol.json"], evidence["freeze.json"]
    attempt, response = evidence["attempt.json"], evidence["response.json"]
    assert protocol["status_at_freeze"] == attempt["status"] == "not_evaluated"
    assert protocol["execution_counts"] == dict(
        atomic_silence_responses=2, reception=0, coherence=0, flow_steps=0, attempts=1
    )
    assert attempt["execution_counts"] == protocol["execution_counts"]
    assert attempt["reconstruction_supplement"]["declared_absent"] == (
        "epi_history_during_silence",
    )
    for name in ("protocol", "freeze"):
        assert attempt[name] == file_receipt(
            BASE / f"{name}.json", label=f"{name}.json"
        )
    archive = freeze["archive"]
    assert archive == file_receipt(BASE / archive["path"], label=archive["path"])
    verify_archive_members(BASE / archive["path"], freeze["members"])
    assert attempt["driver"]["path"] == "benchmarks/organized_source_silence_handoff.py"
    assert attempt["driver"]["sha256"] == freeze["members"][attempt["driver"]["path"]]
    assert freeze["protocol"] == attempt["protocol"]
    assert (
        attempt["protocol"]["sha256"]
        == freeze["members"][
            "docs/assets/organized_source_silence_handoff/protocol.json"
        ]
    )
    assert attempt["upstream"] == protocol["upstream"]
    for name, receipt in protocol["upstream"]["receipts"].items():
        assert receipt == file_receipt(UPSTREAM / name, label=name)
    upstream_freeze = _read(UPSTREAM, "freeze.json")
    verify_archive_members(UPSTREAM / "source.zip", upstream_freeze["members"])
    assert response["status"] == "evaluated"
    assert tuple(stage["glyph"] for stage in response["stages"]) == ("SHA", "SHA")
    assert tuple(stage["preparation"] for stage in response["stages"]) == (
        "reception_handoff",
        "moving_control",
    )
    for stage in response["stages"]:
        assert stage["stage"] == dict(
            operator="silence",
            glyph="SHA",
            schedule="two_phase_jacobi",
            nodes_processed=2,
        )
        assert len(stage["callback"]) == 1


@pytest.mark.parametrize("index", (0, 1))
def test_consumed_state_is_the_retained_endpoint_without_event_replay(evidence, index):
    protocol = evidence["protocol.json"]
    stage = evidence["response.json"]["stages"][index]
    upstream_index = protocol["preparations"][stage["preparation"]][
        "upstream_stage_index"
    ]
    previous = evidence["upstream"]["stages"][upstream_index]
    assert previous["glyph"] == "IL" and previous["preparation"] == stage["preparation"]
    before = stage["before"]
    consumed = {
        **before,
        "nodes": tuple(
            {key: value for key, value in row.items() if key != "lifecycle"}
            for row in before["nodes"]
        ),
    }
    assert consumed == previous["after"]
    assert not any(_field(before, "lifecycle"))
    assert not any(_field(before, "temporal_history_present"))
    for key, value in evidence["upstream_protocol"]["configuration"].items():
        assert protocol["configuration"][key] == value
    assert protocol["configuration"]["sha_word_index"] == 2
    assert protocol["configuration"]["word"] == ("EN", "IL", "SHA")
    assert protocol["configuration"]["grammar_rejection_mode"] == "raise"
    assert protocol["configuration"]["phase_mobility"] == "shared_capacity"


@pytest.mark.parametrize("index", (0, 1))
def test_atomic_capacity_event_and_latency_do_not_advance_time(evidence, index):
    stage, config = (
        evidence["response.json"]["stages"][index],
        evidence["protocol.json"]["configuration"],
    )
    before, after = stage["before"], stage["after"]
    factor = config["silence_capacity_factor"]
    predicted = evidence["attempt.json"]["predictions"][stage["preparation"]]
    assert _field(after, "capacity") == predicted["capacities"]
    for key in ("weighted_edges", "neighbors", "time"):
        assert before[key] == after[key]
    assert before["time"] == 7
    for field in (
        "form",
        "phase",
        "phase_hex",
        "kind",
        "stored_dEPI",
        "reception_sources",
        "temporal_history_present",
        "prior_coherence",
    ):
        assert _field(before, field) == _field(after, field)
    stamps = []
    for n, (old, new) in enumerate(zip(before["nodes"], after["nodes"])):
        if n not in TARGETS:
            assert {key: value for key, value in new.items() if key != "pressure"} == {
                key: value for key, value in old.items() if key != "pressure"
            }
            continue
        assert old["capacity"] >= config["silence_min_capacity"]
        assert new["capacity"] == Q(float(old["capacity"]) * float(factor))
        assert 0 < new["capacity"] < old["capacity"]
        assert old["history"] == ("AL", "IL", "EN", "IL")
        assert new["history"] == (*old["history"], "SHA")
        assert (old["ordinal"], new["ordinal"]) == (4, 5)
        assert old["debt"] == new["debt"] == 0
        assert new["source_glyph"] == "SHA"
        lifecycle = new["lifecycle"]
        assert set(lifecycle) == {
            "latent",
            "latency_start_time",
            "preserved_epi",
            "silence_duration",
            "was_initial_on_silence",
        }
        assert lifecycle["latent"] is True
        assert Q(lifecycle["preserved_epi"]) == new["form"]
        assert lifecycle["silence_duration"] == 0
        assert lifecycle["was_initial_on_silence"] is (abs(float(new["form"])) < 1e-6)
        timestamp = datetime.fromisoformat(lifecycle["latency_start_time"])
        assert timestamp.utcoffset() == timezone.utc.utcoffset(timestamp)
        stamps.append(timestamp)
    assert stamps[0] == stamps[1]


def _oracle(snapshot):
    """Independent 80-digit graph incidence and capacity law, without evolution."""
    mp = mpmath.mp.clone()
    mp.dps = 80

    def value(q):
        return mp.mpf(q.numerator) / q.denominator

    x = tuple(map(value, _field(snapshot, "form")))
    theta = tuple(map(value, _field(snapshot, "phase")))
    capacity = tuple(map(value, _field(snapshot, "capacity")))
    gradient, current = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    storage = mp.mpf(0)
    for a, b, weight in snapshot["weighted_edges"]:
        w = value(weight)
        difference, gap = x[a] - x[b], theta[b] - theta[a]
        gradient[a] += w * difference
        gradient[b] -= w * difference
        flux = w * mp.sin(gap)
        current[a] += flux
        current[b] -= flux
        storage += w * (difference**2 / 2 + 1 - mp.cos(gap))
    gamma = 1 / (1023 * mp.pi)
    pressure = tuple((-q + gamma * s) / 4 for q, s in zip(gradient, current))
    dx = tuple(nu * p for nu, p in zip(capacity, pressure))
    dtheta = tuple(nu * gamma * q / 4 for nu, q in zip(capacity, gradient))
    loss = sum(nu * q**2 / 4 for nu, q in zip(capacity, gradient))
    return mp, value, gradient, current, pressure, dx, dtheta, storage, loss, gamma


def _contains(value, bounds, expected):
    assert value(bounds.lo) <= expected <= value(bounds.hi)


@pytest.mark.parametrize("index", (0, 1))
def test_complete_rows_all_clocks_storage_and_callback_pressure(
    evidence, reports, index
):
    stage = evidence["response.json"]["stages"][index]
    before, after, retention = reports[index]
    callback = stage["callback"][0]
    mp, value, gradient, current, pressure, dx, dt, storage, loss, gamma = _oracle(
        stage["after"]
    )
    assert asdict(after) == callback["observation"] == stage["analysis"]["observation"]
    assert asdict(retention) == stage["analysis"]["retention"]
    assert callback["after"] == stage["after"]
    assert _field(callback["before"], "pressure") == _field(stage["before"], "pressure")
    assert before.unit_capacity_reference == after.unit_capacity_reference
    assert before.pressure_tau_bounds == after.pressure_tau_bounds
    reference = after.unit_capacity_reference
    assert tuple(map(value, reference.form_gradient)) == tuple(gradient)
    for bounds, exact in zip(reference.phase_current_bounds, current):
        _contains(value, bounds, exact)
    for bounds, exact in zip(after.pressure_tau_bounds, pressure):
        _contains(value, bounds, exact)
    for clock, scale in (
        ("tau", mp.mpf(1)),
        ("slow", 2 / gamma**2),
        ("original", mp.mpf(1023) / 1024),
    ):
        for field, exact in (("form", dx), ("phase", dt)):
            for bounds, expected in zip(
                getattr(after, f"{field}_{clock}_rate_bounds"), exact
            ):
                _contains(value, bounds, scale * expected)
        _contains(value, getattr(after, f"storage_{clock}_rate_bounds"), -scale * loss)
    _contains(value, reference.joint_storage_bounds, storage)
    assert abs(value(after.dissipation_tau) - loss) < mp.mpf("1e-75")
    _contains(value, after.balance_residual_bounds, mp.mpf(0))
    assert stage["analysis"]["event_storage_work"] == 0
    assert loss > 0
    _mp0, _value0, _q0, _s0, _p0, old_dx, old_dt, _h0, _loss0, _gamma0 = _oracle(
        stage["before"]
    )
    for n in TARGETS:
        # A positive attenuated capacity and latency flag do not freeze either row.
        assert abs(dx[n]) > 0 and abs(dt[n]) > 0
        # Outward interval multiplication is not associative at its last bit.
        # Both evaluations must enclose the same independently computed rate.
        for channel, previous in (("form", old_dx), ("phase", old_dt)):
            scaled = value(after.capacities[n]) * previous[n]
            _contains(value, getattr(after, f"{channel}_tau_rate_bounds")[n], scaled)
            _contains(
                value,
                after.capacities[n] * getattr(before, f"{channel}_tau_rate_bounds")[n],
                scaled,
            )
        assert after.form_tau_rate_bounds[n] != after.pressure_tau_bounds[n]
    budget = evidence["protocol.json"]["budgets"]["pressure_materialization_tau"]
    for stored, bounds in zip(
        _field(stage["after"], "pressure"), after.pressure_tau_bounds
    ):
        assert (I(stored) - bounds).abs_max <= budget
    metrics = callback["metrics_before_refresh"]
    assert len(metrics) == 2 and all(metric["glyph"] == "SHA" for metric in metrics)
    for n, metric in zip(TARGETS, metrics):
        assert metric["latent"] is True
        assert Q(metric["vf_final"]) == after.capacities[n]
        assert metric["epi_preservation"] == 0 and metric["silence_duration"] == 0
        assert metric["time_to_collapse"] == dict(
            available=False,
            reason="nonfinite_runtime_diagnostic",
            representation="+inf",
        )
    assert not any(_field(stage["after"], "temporal_history_present"))


@pytest.mark.parametrize("index", (0, 1))
def test_changed_weighted_leaf_lift_and_continuous_retention(evidence, reports, index):
    stage = evidence["response.json"]["stages"][index]
    _before, after, retention = reports[index]
    snapshot = stage["after"]
    turns = evidence["protocol.json"]["phase_representation"]["turns"]
    weights = tuple(1 / v for v in _field(snapshot, "capacity"))
    weight_sum = sum(weights, Q(0))
    raw_sum = sum((w * p for w, p in zip(weights, _field(snapshot, "phase"))), Q(0))
    turn_sum = sum((w * t for w, t in zip(weights, turns)), Q(0))
    slot_sum = sum((w * (n % 9 - 4) for n, w in enumerate(weights)), Q(0))
    assert sum((w * x for w, x in zip(weights, _field(snapshot, "form"))), Q(0)) == 0
    assert after.inverse_capacity_weighted_form_total == 0
    assert after.inverse_capacity_weighted_raw_phase_total == raw_sum
    assert retention.inverse_capacity_weights == weights
    assert retention.inverse_capacity_weight_sum == weight_sum
    assert retention.weighted_raw_phase_sum == raw_sum
    assert retention.weighted_phase_turn_sum == turn_sum
    assert retention.weighted_carrier_slot_sum == slot_sum != 0
    mp, value, _q, _s, _p, dx, dt, _h, _loss, _gamma = _oracle(snapshot)
    origin = (
        value(raw_sum) + 2 * mp.pi * (value(turn_sum) - value(slot_sum) / 9)
    ) / value(weight_sum)
    _contains(value, retention.weighted_phase_origin_bounds, origin)
    assert abs(sum(value(w) * rate for w, rate in zip(weights, dx))) < mp.mpf("1e-75")
    assert abs(sum(value(w) * rate for w, rate in zip(weights, dt))) < mp.mpf("1e-75")
    lifted = tuple(
        value(angle) + 2 * mp.pi * turn
        for angle, turn in zip(_field(snapshot, "phase"), turns)
    )
    for a, b, _ in snapshot["weighted_edges"]:
        gap = lifted[b] - lifted[a]
        if a // 9 == b // 9 and a % 9 == 8 and b % 9 == 0:
            gap += 2 * mp.pi
        assert abs(gap) < mp.pi / 2
    assert retention.admitted and retention.geometric_retention_certified
    assert retention.form_rails_certified and retention.converges_to_target is True
    assert retention.retained_windings == (1,) * 5
    assert retention.excess_bounds.hi < Q(1, 100)
    assert retention.spectral_gap_lower_bound == Q(4, 15)
    assert retention.form_squared_upper_bound == (
        retention.capacity_ratio * 2 * retention.energy_ceiling / Q(4, 15)
    )
    assert (
        retention.form_squared_upper_bound
        < evidence["protocol.json"]["acceptance"]["squared_form_bound_ceiling"]
    )
    assert retention.form_absolute_upper_bound**2 >= retention.form_squared_upper_bound
    for n, bounds in enumerate(retention.asymptotic_phase_lift_bounds):
        _contains(value, bounds, (2 * mp.pi / 9) * (n % 9 - 4) + origin)
    assert retention.domain_boundary_margin > 0
    assert not retention.unresolved_conditions
    assert not stage["analysis"]["retention"]["scope"] == ()
    # The retained continuous lift remains an explicit input, not inferred wrapping.
    assert turns == (-1, -1) + (0,) * 43
    _contains(value, pi_interval(), mp.pi)
