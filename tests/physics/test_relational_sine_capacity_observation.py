"""Hidden-capacity inference from strictly prior visible accelerations.

The independent oracle differentiates every fine edge of a supplied sine law
before withholding the hidden state/capacity. No inverse formula from the
inference owner generates the data, and no trajectory or producer is run.
"""

import json
import pickle
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_comparison, relational_sine_mediation
from tnfr.physics.relational_sine_observation import infer_relational_sine_hidden_state

MODEL = RelationalExchangeModel(2, phase_domain="regular")
PORTS = (1, 2, 3)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(interval, value):
    assert _mp(interval.lo) <= value <= _mp(interval.hi)


def _bounds(value):
    center = Q(mp.nstr(value, 85))
    error = Q(1, 10**60)
    return center - error, center + error


def _visible_graph():
    graph = nx.Graph()
    graph.add_nodes_from((1, 2, 3, 4))
    graph.add_edges_from(((1, 2), (2, 4)))
    for node, x, phase, capacity in zip(
        graph, (1, -0.5, 0.75, -1), (0, 0.5, -0.75, 1), (1, 2, 0.5, 1.5)
    ):
        graph.nodes[node].update(EPI=x, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.graph.update(_t=99, retained={"history": ["unused"]})
    return graph


def _oracle(
    visible,
    ports,
    *,
    hidden_capacity,
    hidden_form=Q(1, 4),
    hidden_phase=Q(-1, 4),
    model=MODEL,
):
    with mp.workdps(90):
        hidden = "withheld-node"
        fine = visible.copy()
        fine.add_node(hidden, EPI=hidden_form, theta=hidden_phase, nu_f=hidden_capacity)
        fine.add_edges_from((hidden, port) for port in ports)
        x = {i: _mp(fine.nodes[i]["EPI"]) for i in fine}
        theta = {i: _mp(fine.nodes[i]["theta"]) for i in fine}
        nu = {i: _mp(fine.nodes[i]["nu_f"]) for i in fine}
        e, w = map(_mp, model.effective_weights)
        beta = _mp(model.storage_scale)
        form_rate, phase_rate = {}, {}
        for i in fine:
            gradient = sum(x[i] - x[j] for j in fine[i])
            current = sum(mp.sin(theta[j] - theta[i]) for j in fine[i])
            form_rate[i] = (
                nu[i] * (-e * gradient + w * current / mp.pi) / fine.degree[i]
            )
            phase_rate[i] = nu[i] * w * gradient / (beta * mp.pi * fine.degree[i])
        form_acceleration, phase_acceleration = {}, {}
        for i in fine:
            gradient_rate = sum(form_rate[i] - form_rate[j] for j in fine[i])
            current_rate = sum(
                mp.cos(theta[j] - theta[i]) * (phase_rate[j] - phase_rate[i])
                for j in fine[i]
            )
            form_acceleration[i] = (
                nu[i] * (-e * gradient_rate + w * current_rate / mp.pi) / fine.degree[i]
            )
            phase_acceleration[i] = (
                nu[i] * w * gradient_rate / (beta * mp.pi * fine.degree[i])
            )
        return {
            "form_rates": {i: _bounds(form_rate[i]) for i in ports},
            "phase_rates": {i: _bounds(phase_rate[i]) for i in ports},
            "form_accelerations": {i: _bounds(form_acceleration[i]) for i in ports},
            "phase_accelerations": {i: _bounds(phase_acceleration[i]) for i in ports},
            "exact_form_rate": form_rate,
            "exact_phase_rate": phase_rate,
            "exact_form_acceleration": form_acceleration,
            "exact_phase_acceleration": phase_acceleration,
            "hidden": hidden,
        }


def _source(graph, oracle, *, ports=PORTS, model=MODEL):
    return infer_relational_sine_hidden_state(
        graph,
        ports=ports,
        form_rate_bounds=oracle["form_rates"],
        phase_rate_bounds=oracle["phase_rates"],
        reference_model=model,
        source_id="independent-prior-first-rates",
        clock_id="declared-structural-clock",
        observation_time=Q(2),
        evidence_window=(Q(1), Q(3)),
        forecast_start=Q(5),
    )


def _infer(source, oracle, **changes):
    arguments = dict(
        form_acceleration_bounds=oracle["form_accelerations"],
        phase_acceleration_bounds=oracle["phase_accelerations"],
        source_id="independent-prior-accelerations",
        clock_id="declared-structural-clock",
        observation_time=Q(2),
        evidence_window=(Q(1), Q(4)),
    )
    arguments.update(changes)
    return source.infer_capacity(**arguments)


def _channel_truth(oracle, port, kind, model):
    form = oracle["exact_form_acceleration"][port]
    phase = oracle["exact_phase_acceleration"][port]
    if kind == "form_acceleration":
        return form
    if kind == "phase_acceleration":
        return phase
    assert kind == "exchange_acceleration"
    factor = (
        _mp(model.epi_weight)
        * _mp(model.storage_scale)
        * mp.pi
        / _mp(model.phase_weight)
    )
    return form + factor * phase


@pytest.fixture(scope="module")
def sample():
    graph = _visible_graph()
    oracle = _oracle(graph, PORTS, hidden_capacity=Q(3))
    source = _source(graph, oracle)
    return graph, oracle, source, _infer(source, oracle)


def test_capacity_rebuilds_hidden_state_from_evidence_not_cached_inverse(sample):
    _, oracle, source, expected = sample
    stale = replace(
        source,
        port_degrees=(99,) * len(PORTS),
        active_ports=(),
        hidden_form_bounds=I(100),
        hidden_projection_bounds=(I(0),) * len(PORTS),
        hidden_unit_phase_relative_to_anchor_bounds=None,
        phase_anchor=None,
        status="inconsistent",
    )
    result = _infer(stale, oracle)
    # The independently differentiated fine graph has hidden capacity 3.
    assert result.capacity_bounds.contains(3)
    assert result == expected
    assert result.state_inference == source
    assert result.state_inference is not stale
    assert stale.hidden_form_bounds == I(100)


@pytest.mark.parametrize(
    "changes",
    (
        {"visible_capacity": (1, -2, Q(1, 2), Q(3, 2))},
        {"visible_epi": (True, Q(-1, 2), Q(3, 4), -1)},
        {"visible_phase": (0, float("inf"), Q(-3, 4), 1)},
        {"visible_nodes": (1, 1, 3, 4)},
        {"visible_edges": ((1, 2), (2, 1), (2, 4))},
        {"visible_edges": ((1, 1), (2, 4))},
        {"visible_edges": ((1, 99),)},
        {"visible_edges": ((1, 2),)},  # Node 4 is isolated from all ports.
        {"ports": (1, 1, 3)},
        {"form_rate_bounds": (I(0),)},
        {"evidence_window": (Q(1), Q(6))},
        {"observation_time": True},
    ),
)
def test_capacity_rejects_invalid_retained_primitive_evidence(sample, changes):
    _, oracle, source, _ = sample
    with pytest.raises((TypeError, ValueError)):
        _infer(replace(source, **changes), oracle)


def test_capacity_rebuild_retains_exact_values_without_live_graph_admission(
    sample, monkeypatch
):
    from tnfr.physics import relational_sine_observation as module

    _, oracle, source, _ = sample
    tiny = Q(1, 10**400)
    changed = replace(source, visible_epi=(tiny,) + source.visible_epi[1:])

    def forbidden(*args, **kwargs):
        pytest.fail("detached inverse recaptured a live graph")

    monkeypatch.setattr(module, "_admit_graph", forbidden)
    result = _infer(changed, oracle)
    assert result.state_inference.visible_epi[0] == tiny
    assert result.state_inference.visible_epi[0] != 0


@pytest.mark.parametrize("mu", (Q(0), Q(1, 2), Q(3)))
def test_independent_fine_edge_accelerations_enclose_hidden_capacity(mu):
    graph = _visible_graph()
    oracle = _oracle(graph, PORTS, hidden_capacity=mu)
    source = _source(graph, oracle)
    result = _infer(source, oracle)
    assert source.hidden_capacity_identified is False
    assert result.state_inference is source
    assert result.status == "bounded_candidate"
    assert result.capacity_bounds.contains(mu)
    assert result.capacity_bounds.width < Q(1, 10**12)
    assert result.capacity_bounds.lo >= 0
    if mu == 0:
        assert result.capacity_bounds.lo == 0
    zero = _oracle(graph, PORTS, hidden_capacity=Q(0))
    one = _oracle(graph, PORTS, hidden_capacity=Q(1))
    with mp.workdps(90):
        _inside(
            result.hidden_form_response_per_capacity_bounds,
            one["exact_form_rate"][one["hidden"]],
        )
        _inside(
            result.hidden_phase_response_per_capacity_bounds,
            one["exact_phase_rate"][one["hidden"]],
        )
        kinds = set()
        for channel in result.channels:
            kinds.add(channel.kind)
            observed = _channel_truth(oracle, channel.port, channel.kind, MODEL)
            baseline = _channel_truth(zero, channel.port, channel.kind, MODEL)
            sensitivity = (
                _channel_truth(one, channel.port, channel.kind, MODEL) - baseline
            )
            _inside(channel.observed_bounds, observed)
            _inside(channel.baseline_bounds, baseline)
            _inside(channel.sensitivity_bounds, sensitivity)
            _inside(channel.baseline_bounds + mu * channel.sensitivity_bounds, observed)
            assert (
                channel.residual_bounds is not None
                and channel.residual_bounds.contains(0)
            )
        assert kinds == {
            "form_acceleration",
            "phase_acceleration",
            "exchange_acceleration",
        }


def test_zero_diffusion_does_not_remove_capacity_observability():
    model = RelationalExchangeModel(
        2, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    graph = _visible_graph()
    oracle = _oracle(graph, PORTS, hidden_capacity=Q(2), model=model)
    source = _source(graph, oracle, model=model)
    result = _infer(source, oracle)
    assert result.status == "bounded_candidate"
    assert result.capacity_bounds.contains(2)
    assert result.capacity_bounds.width < Q(1, 10**12)
    assert any(
        not channel.sensitivity_bounds.contains(0) for channel in result.channels
    )


def test_nonport_rates_are_derived_from_visible_support_while_port_rates_stay_observed(
    sample,
):
    graph, oracle, source, result = sample
    with mp.workdps(90):
        for position, node in enumerate(source.visible_nodes):
            _inside(
                result.visible_form_rate_bounds[position],
                oracle["exact_form_rate"][node],
            )
            _inside(
                result.visible_phase_rate_bounds[position],
                oracle["exact_phase_rate"][node],
            )
            if node in source.ports:
                port_position = source.ports.index(node)
                assert (
                    result.visible_form_rate_bounds[position]
                    == source.form_rate_bounds[port_position]
                )
                assert (
                    result.visible_phase_rate_bounds[position]
                    == source.phase_rate_bounds[port_position]
                )
        assert (
            result.visible_rate_provenance[source.visible_nodes.index(4)]
            != result.visible_rate_provenance[source.visible_nodes.index(1)]
        )
        assert 4 not in source.ports


def test_stationary_hidden_snapshot_remains_capacity_blind_despite_moving_ports():
    graph = nx.empty_graph((1, 2))
    for node, x, phase, capacity in ((1, 1, -0.5, 1), (2, -1, 0.5, 2)):
        graph.nodes[node].update(EPI=x, theta=phase, nu_f=capacity)
    oracles = [
        _oracle(graph, (1, 2), hidden_capacity=mu, hidden_form=Q(0), hidden_phase=Q(0))
        for mu in (Q(0), Q(3))
    ]
    assert oracles[0]["form_rates"] == oracles[1]["form_rates"]
    assert oracles[0]["phase_rates"] == oracles[1]["phase_rates"]
    assert oracles[0]["form_accelerations"] == oracles[1]["form_accelerations"]
    assert oracles[0]["phase_accelerations"] == oracles[1]["phase_accelerations"]
    for oracle in oracles:
        assert oracle["exact_form_rate"][oracle["hidden"]] == 0
        assert oracle["exact_phase_rate"][oracle["hidden"]] == 0
    with mp.workdps(90):
        assert abs(
            sum(oracles[0]["exact_form_rate"][node] for node in (1, 2)) / 2
        ) > mp.mpf("0.01")
    source = _source(graph, oracles[0], ports=(1, 2))
    assert source.status == "bounded_candidate"
    result = _infer(source, oracles[0])
    assert result.status == "unavailable"
    assert result.capacity_bounds is None
    assert all(channel.sensitivity_bounds.contains(0) for channel in result.channels)
    # This is blindness at the declared observation order, not a statement
    # that the unequal-capacity moving environment stays blind at all times.


def test_phase_ambiguity_still_allows_capacity_from_an_identified_form_pressure():
    graph = nx.empty_graph((1, 2))
    for node, x, capacity in ((1, 1, 1), (2, 0, 2)):
        graph.nodes[node].update(EPI=x, theta=0, nu_f=capacity)
    oracle = _oracle(graph, (1, 2), hidden_capacity=Q(2), hidden_phase=Q(0))
    source = _source(graph, oracle, ports=(1, 2))
    assert source.status == "unavailable" and source.phase_rank == 1
    assert source.hidden_form_bounds.contains(Q(1, 4))
    assert source.hidden_unit_phase_relative_to_anchor_bounds is None
    result = _infer(source, oracle, form_acceleration_bounds={})
    assert result.status == "bounded_candidate"
    assert result.capacity_bounds.contains(2)
    assert result.capacity_bounds.width < Q(1, 10**12)
    assert result.hidden_phase_available is False
    assert result.state_inference.hidden_unit_phase_relative_to_anchor_bounds is None


def test_negative_capacity_required_by_accelerations_is_inconsistent():
    graph = _visible_graph()
    oracle = _oracle(graph, PORTS, hidden_capacity=Q(-1))
    source = _source(graph, oracle)
    assert source.status == "bounded_candidate"
    result = _infer(source, oracle)
    assert result.status == "inconsistent"
    assert result.capacity_bounds is None


def test_different_capacities_in_the_two_acceleration_channels_are_inconsistent():
    graph = _visible_graph()
    first = _oracle(graph, PORTS, hidden_capacity=Q(1))
    second = _oracle(graph, PORTS, hidden_capacity=Q(3))
    assert first["form_rates"] == second["form_rates"]
    assert first["phase_rates"] == second["phase_rates"]
    source = _source(graph, first)
    result = _infer(
        source, first, phase_acceleration_bounds=second["phase_accelerations"]
    )
    assert result.status == "inconsistent"
    assert result.capacity_bounds is None


@pytest.mark.parametrize("channel", ("form", "phase"))
def test_one_informative_partial_acceleration_channel_is_sufficient(sample, channel):
    _, oracle, source, _ = sample
    arguments = {"form_acceleration_bounds": {}, "phase_acceleration_bounds": {}}
    arguments[f"{channel}_acceleration_bounds"] = {
        1: oracle[f"{channel}_accelerations"][1]
    }
    result = _infer(source, oracle, **arguments)
    assert result.status == "bounded_candidate"
    assert result.capacity_bounds.contains(3)
    assert result.form_acceleration_bounds[1:] == (None, None)
    assert result.phase_acceleration_bounds[1:] == (None, None)
    assert {row.kind for row in result.channels} == {f"{channel}_acceleration"}


def test_zero_capacity_port_cannot_have_a_nonzero_acceleration():
    graph = _visible_graph()
    graph.nodes[3]["nu_f"] = 0
    oracle = _oracle(graph, PORTS, hidden_capacity=Q(3))
    source = _source(graph, oracle)
    changed = dict(oracle["form_accelerations"])
    changed[3] = (Q(1), Q(1))
    result = _infer(source, oracle, form_acceleration_bounds=changed)
    assert result.status == "inconsistent"
    assert result.capacity_bounds is None


def test_frozen_ports_reject_nonzero_acceleration_before_missing_state_exit():
    graph = _visible_graph()
    for node in PORTS:
        graph.nodes[node]["nu_f"] = 0
    oracle = _oracle(graph, PORTS, hidden_capacity=Q(3))
    source = _source(graph, oracle)
    assert source.status == "unavailable"
    assert source.hidden_form_bounds is None
    for field in ("form_rates", "phase_rates"):
        assert all(lo <= 0 <= hi for lo, hi in oracle[field].values())

    valid = _infer(source, oracle)
    assert valid.status == "unavailable"
    assert valid.capacity_bounds is None
    for channel in ("form", "phase"):
        changed = dict(oracle[f"{channel}_accelerations"])
        changed[PORTS[0]] = (Q(1), Q(2))
        result = _infer(source, oracle, **{f"{channel}_acceleration_bounds": changed})
        assert result.status == "inconsistent"
        assert result.capacity_bounds is None
        assert result.channels == ()


@pytest.mark.parametrize("kind", ("inconsistent", "unavailable"))
def test_missing_or_inconsistent_hidden_form_prevents_capacity_inference(kind):
    graph = _visible_graph()
    if kind == "unavailable":
        for node in PORTS:
            graph.nodes[node]["nu_f"] = 0
    oracle = _oracle(graph, PORTS, hidden_capacity=Q(3))
    if kind == "inconsistent":
        oracle["phase_rates"][2] = tuple(
            value + 1 for value in oracle["phase_rates"][2]
        )
    source = _source(graph, oracle)
    assert source.status == kind
    result = _infer(source, oracle)
    assert result.status == kind
    assert result.capacity_bounds is None
    assert result.channels == ()
    assert any(bound is not None for bound in result.form_acceleration_bounds)


@pytest.mark.parametrize(
    "changes",
    (
        {"clock_id": "other-clock"},
        {"observation_time": Q(3)},
        {"evidence_window": (Q(1), Q(5))},
        {"form_acceleration_bounds": {}, "phase_acceleration_bounds": {}},
        {"form_acceleration_bounds": {4: (Q(0), Q(0))}},
    ),
)
def test_mismatched_time_clock_or_missing_acceleration_evidence_is_rejected(
    sample, changes
):
    _, oracle, source, _ = sample
    with pytest.raises((ValueError, TypeError)):
        _infer(source, oracle, **changes)


def test_detached_inference_uses_no_full_hidden_state_evaluation_or_solver(
    sample, monkeypatch
):
    graph, oracle, source, _ = sample
    before = pickle.dumps(source, protocol=5)

    def forbidden(*args, **kwargs):
        pytest.fail("capacity inference must use only prior visible evidence")

    for owner, names in (
        (
            relational,
            (
                "_stage",
                "_field",
                "evaluate_relational_exchange",
                "step_relational_exchange",
            ),
        ),
        (
            relational_sine_comparison,
            ("_capture_sine_state", "bound_relational_sine_exchange"),
        ),
        (
            relational_sine_mediation,
            ("_capture_sine_state", "bound_relational_sine_mediation"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    result = _infer(source, oracle)
    assert result.capacity_bounds.contains(3)
    assert pickle.dumps(source, protocol=5) == before


def test_export_retains_both_prior_sources_and_the_joint_window(sample):
    _, _, source, result = sample
    assert result.state_inference is source
    assert result.combined_evidence_window == (Q(1), Q(4))
    assert result.combined_evidence_window[1] < result.forecast_start
    encoded = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    body = encoded["report"]
    assert body["source_id"] == "independent-prior-accelerations"
    assert body["state_inference"]["source_id"] == "independent-prior-first-rates"
    assert body["status"] == "bounded_candidate"
