"""Synthetic child acquisition wiring; no retained response is evaluated.

The graph, polynomial samples, stages and successful error associations below
are manufactured controls. Source preparation, real numerical integration and
all actual operator proposals are forbidden. The tests establish routing and
failure behavior, not accuracy or acquisition of the retained TNFR trajectory.
"""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.glyph_history import push_glyph
from tnfr.operators import _dissonance_stage_kernel, network_stage
from tnfr.operators.definitions import Dissonance, SelfOrganization
from tnfr.operators.stage_contracts import stage_schedule_metadata
from tnfr.research import maintained_child_history_acquisition as owner
from tnfr.research.maintained_child_sample_certificate import ChildSampleCertificate
from tnfr.research.retained_graph_state import restore_recorded_graph_state


@dataclass(frozen=True)
class SyntheticTiming:
    source_time: Q = Q(3)
    threshold: Q = Q(1, 10)


@dataclass(frozen=True)
class SyntheticStage:
    operator: str = "dissonance"
    glyph: str = "OZ"
    schedule: str = "two_phase_jacobi"
    nodes_processed: int = 1


@dataclass(frozen=True)
class SyntheticBoundary:
    index: int
    sample_time: Q
    state: tuple
    pressures: tuple
    passed: bool = True


@dataclass(frozen=True)
class SyntheticAssessment:
    passed: bool = True


def _synthetic_graph():
    """Different coordinates, capacities and support from retained evidence."""
    graph = nx.Graph(
        RANDOM_SEED=23,
        _t=3.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        hierarchy={2: [owner.CHILD]},
    )
    for index, node in enumerate(tuple(range(20)) + (owner.CHILD,)):
        graph.add_node(
            node,
            EPI=3 / 8 if node == owner.CHILD else 1 / 2,
            theta=0.0 if node in (0, owner.CHILD) else 1 / 8,
            nu_f=3.0 if node == owner.CHILD else 1.0,
            delta_nfr=0.05 if node == owner.CHILD else 0.02,
            glyph_history=[],
            epi_history=[3 / 8 if node == owner.CHILD else 1 / 2],
            synthetic_marker=index,
        )
    # Deliberately retain a non-sorted neighbor order in the recorded clone.
    for offset in (2, 1):
        graph.add_edges_from(
            ((index, (index + offset) % 20) for index in reversed(range(20))),
            weight=1 / 2,
        )
    graph.add_edges_from(((index, index + 3) for index in range(12)), weight=1 / 4)
    graph.add_edge(2, owner.CHILD, weight=3 / 4)
    assert graph.number_of_edges() == 53
    graph.nodes[2].update(
        sub_nodes=[owner.CHILD],
        epi_time_history=((2.75, 0.4), (3.0, 0.5)),
    )
    graph.nodes[owner.CHILD].update(parent_node=2, hierarchy_level=1)
    graph._last_operator_applied = "synthetic_previous_stage"
    return graph


@pytest.fixture(autouse=True)
def forbid_retained_source_and_real_execution(monkeypatch):
    from tnfr.dynamics import integrators
    from tnfr.research import maintained_child_sample_certificate as samples
    from tnfr.research import maintained_sha_motion_source

    def forbidden(*args, **kwargs):
        pytest.fail("wiring controls must not evaluate retained source or response")

    monkeypatch.setattr(owner, "prepare_child_sample_certificate", forbidden)
    monkeypatch.setattr(samples, "prepare_child_sample_certificate", forbidden)
    monkeypatch.setattr(samples, "load_child_joint_window", forbidden)
    monkeypatch.setattr(maintained_sha_motion_source, "load_sha_successor", forbidden)
    monkeypatch.setattr(integrators, "rk4_coupled_step", forbidden)
    monkeypatch.setattr(owner, "rk4_coupled_step", forbidden)
    monkeypatch.setattr(owner, "execute_network_operator_stage", forbidden)
    monkeypatch.setattr(network_stage, "execute_dissonance_stage", forbidden)
    monkeypatch.setattr(_dissonance_stage_kernel, "propose_dissonance_stage", forbidden)
    monkeypatch.setattr(Dissonance, "_execute", forbidden)
    monkeypatch.setattr(SelfOrganization, "_prepare_execution", forbidden)


@pytest.fixture
def synthetic(monkeypatch):
    original = _synthetic_graph()
    initial = owner._state(original)
    times = tuple(Q(3) + elapsed for elapsed in (Q(3, 128), Q(25, 1024), Q(13, 512)))
    source = SimpleNamespace(
        retention=SimpleNamespace(admitted=SimpleNamespace(graph=original)),
        receipts=(),
    )
    # Boxes are deliberately unavailable. A wiring test must never initialize
    # state or manufacture acquired samples from an exact-flow enclosure.
    certificate = ChildSampleCertificate(source, (), times)
    controls = SimpleNamespace(
        original=original,
        certificate=certificate,
        live=None,
        calls=[],
        records=[],
        boundary_calls=0,
        fail_boundary_call=None,
        handoffs=1,
        stage_returned=False,
        refresh_calls=0,
        final_passed=True,
        step_mutation=None,
        raw_mutation=None,
    )

    def expected_state(time):
        elapsed = Q(time) - Q(3)
        values = list(initial)
        values[owner.CHILD_INDEX] = float(Q(3, 8) + 64 * elapsed**2)
        values[21] = -float(elapsed)  # A continuous phase lift, without wrapping.
        return tuple(values)

    controls.expected_state = expected_state

    def capture(graph):
        controls.live = graph
        return SimpleNamespace(
            nodes=tuple(graph),
            epi=tuple(Q(get_attr(graph.nodes[node], ALIAS_EPI)) for node in graph),
            phase=tuple(Q(get_attr(graph.nodes[node], ALIAS_THETA)) for node in graph),
            capacity=tuple(Q(get_attr(graph.nodes[node], ALIAS_VF)) for node in graph),
        )

    def refresh(graph):
        controls.refresh_calls += 1
        controls.calls.append(("refresh", Q(graph.graph["_t"])))
        for node in graph:
            set_dnfr(graph, node, 0.05 if node == owner.CHILD else 0.02)

    def field_factory(source):
        controls.calls.append(("field", source.nodes))

        def field(time, values):
            rates = [0.0] * 42
            rates[owner.CHILD_INDEX] = 128 * (time - 3.0)
            rates[21] = -1.0
            return tuple(rates)

        return field

    def manufactured_step(state, field, *, t, dt):
        assert dt == 2**-14
        assert state == expected_state(t)
        controls.calls.append(("step", Q(t), Q(dt)))
        for moment in (t, t + dt / 2, t + dt / 2, t + dt):
            assert len(field(moment, state)) == 42
        if controls.step_mutation is not None:
            controls.step_mutation(controls.live)
        return expected_state(t + dt)

    def boundary(prepared, *, index, state, pressures, sample_time):
        controls.boundary_calls += 1
        controls.calls.append(("boundary", index, Q(sample_time)))
        assert prepared is certificate
        assert Q(sample_time) == times[index]
        assert state == expected_state(sample_time)
        assert state == owner._state(controls.live)
        assert len(pressures) == 21
        return SyntheticBoundary(
            index,
            Q(sample_time),
            state,
            pressures,
            controls.boundary_calls != controls.fail_boundary_call,
        )

    def assessment(prepared, *, states, pressures, sample_times):
        controls.calls.append(("assessment", sample_times))
        assert prepared is certificate
        assert sample_times == times
        assert states == tuple(expected_state(time) for time in times)
        assert len(pressures) == 3
        return SyntheticAssessment(controls.final_passed)

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        controls.calls.append(("stage", Q(graph.graph["_t"])))
        assert operator.glyph.value == "OZ" and targets == (owner.CHILD,)
        assert sequence_context.index == 0
        assert sequence_context.sequence.names == (
            "dissonance",
            "self_organization",
            "silence",
        )
        assert "epi_time_history" not in graph.nodes[owner.CHILD]
        assert graph.graph["_t"] == float(times[0])
        child = graph.nodes[owner.CHILD]
        push_glyph(child, "OZ", 7)
        child["source_glyph"] = "OZ"
        graph._last_operator_applied = "dissonance"
        set_dnfr(graph, owner.CHILD, 0.125)
        graph.graph["_oz_propagation_events"] = [
            {
                "source_node": owner.CHILD,
                "magnitude": 0.075,
                "affected_nodes": [],
                "affected_count": 0,
            }
        ]
        if controls.raw_mutation is not None:
            controls.raw_mutation(graph)
        for _ in range(controls.handoffs):
            compute_delta_nfr(graph)
        graph.graph[network_stage.STAGE_SCHEDULE_KEY] = owner.encode_result(
            SyntheticStage()
        )
        graph.graph[network_stage.STAGE_CONTRACT_KEY] = stage_schedule_metadata(
            "dissonance", observed_schedule="two_phase_jacobi"
        )
        controls.stage_returned = True
        return SyntheticStage()

    def sink(kind, record):
        controls.records.append((kind, deepcopy(record)))
        if kind.startswith("sample_"):
            assert controls.stage_returned

    controls.sink = sink
    monkeypatch.setattr(
        owner, "_admit_child_oz_timing_source", lambda _: SyntheticTiming()
    )
    monkeypatch.setattr(owner, "capture", capture)
    monkeypatch.setattr(owner, "numerical_field", field_factory)
    monkeypatch.setattr(owner, "refresh_pressure", refresh)
    monkeypatch.setattr(owner, "rk4_coupled_step", manufactured_step)
    monkeypatch.setattr(owner, "_assess_child_boundary", boundary)
    monkeypatch.setattr(owner, "_assess_child_samples", assessment)
    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    return controls


def _run(controls, *, sink=None):
    return owner._acquire_child_history(
        controls.certificate, on_record=controls.sink if sink is None else sink
    )


def _kinds(controls):
    return [kind for kind, _ in controls.records]


def test_public_entry_prepares_once_before_delegating(monkeypatch):
    prepared, outcome = object(), object()
    calls = []

    def sink(kind, record):
        return None

    def prepare(path, *, repository_root):
        calls.append(("prepare", path, repository_root))
        return prepared

    def acquire(certificate, *, on_record):
        calls.append(("acquire", certificate, on_record))
        return outcome

    monkeypatch.setattr(owner, "prepare_child_sample_certificate", prepare)
    monkeypatch.setattr(owner, "_acquire_child_history", acquire)
    assert (
        owner.evaluate_child_history(
            "synthetic", repository_root="unused", on_record=sink
        )
        is outcome
    )
    assert calls == [("prepare", "synthetic", "unused"), ("acquire", prepared, sink)]


def test_synthetic_fixed_schedule_records_actual_routed_boundaries_and_own_history(
    synthetic,
):
    before = owner._signature(synthetic.original)
    result = _run(synthetic)
    assert owner._signature(synthetic.original) == before
    assert result["status"] == "acquired"
    assert result["numerical_steps"] == 416 and result["field_evaluations"] == 1664
    assert result["executed_prefix"] == ("OZ",)
    assert result["outstanding_u2_debt"] == 1
    assert result["complete_word_executed"] is False
    assert result["second_birth_executed"] is False
    assert result["physical_observation"] is False
    assert result["observation"]["value"] == 128.0
    assert sum(call[0] == "stage" for call in synthetic.calls) == 1
    assert sum(call[0] == "step" for call in synthetic.calls) == 416
    assert sum(call[0] == "field" for call in synthetic.calls) == 1
    assert [call[1] for call in synthetic.calls if call[0] == "boundary"] == [
        0,
        0,
        1,
        2,
    ]

    kinds = _kinds(synthetic)
    assert kinds[0] == "source"
    assert kinds[1:385] == ["step"] * 384
    assert kinds[385:389] == ["event_before", "event_raw", "event_after", "sample_0"]
    assert kinds[389:405] == ["step"] * 16
    assert kinds[405] == "sample_1"
    assert kinds[406:422] == ["step"] * 16
    assert kinds[422] == "sample_2"
    records = dict(
        (kind, record) for kind, record in synthetic.records if kind != "step"
    )
    source = restore_recorded_graph_state(records["source"]["endpoint"]).graph
    assert source._last_operator_applied == "synthetic_previous_stage"
    assert tuple((node, tuple(source[node])) for node in source) == tuple(
        (node, tuple(synthetic.original[node])) for node in synthetic.original
    )
    raw = restore_recorded_graph_state(records["event_raw"]).graph
    after = restore_recorded_graph_state(records["event_after"]["endpoint"]).graph
    assert "epi_time_history" not in raw.nodes[owner.CHILD]
    assert "epi_time_history" not in after.nodes[owner.CHILD]
    assert get_attr(raw.nodes[owner.CHILD], ALIAS_DNFR) == 0.125
    assert get_attr(after.nodes[owner.CHILD], ALIAS_DNFR) == 0.05
    endpoint = restore_recorded_graph_state(result["endpoint"]).graph
    history = tuple(endpoint.nodes[owner.CHILD]["epi_time_history"])
    assert history == tuple(
        (float(time), synthetic.expected_state(time)[owner.CHILD_INDEX])
        for time in synthetic.certificate.sample_times
    )
    assert history[-1][0] == endpoint.graph["_t"]
    assert history[-1][1] == get_attr(endpoint.nodes[owner.CHILD], ALIAS_EPI)
    assert get_attr(endpoint.nodes[0], ALIAS_THETA) == -13 / 512
    assert tuple(endpoint.nodes[2]["epi_time_history"])[-3:] == tuple(
        (float(time), 0.5) for time in synthetic.certificate.sample_times
    )


@pytest.mark.parametrize("call,index,steps,samples", ((1, 0, 384, 0), (3, 1, 400, 1)))
def test_failed_boundary_stops_without_later_event_or_samples(
    synthetic, call, index, steps, samples
):
    synthetic.fail_boundary_call = call
    with pytest.raises(ValueError, match=f"boundary {index} failed error association"):
        _run(synthetic)
    assert sum(item[0] == "step" for item in synthetic.calls) == steps
    assert sum(kind.startswith("sample_") for kind in _kinds(synthetic)) == samples
    assert _kinds(synthetic).count("event_raw") == int(call > 1)
    assert sum(item[0] == "stage" for item in synthetic.calls) == int(call > 1)


@pytest.mark.parametrize("handoffs", (0, 2))
def test_missing_or_repeated_raw_handoff_rejects_before_samples(synthetic, handoffs):
    synthetic.handoffs = handoffs
    with pytest.raises(ValueError, match="handoff"):
        _run(synthetic)
    assert _kinds(synthetic).count("event_raw") == min(handoffs, 1)
    assert not any(kind.startswith("sample_") for kind in _kinds(synthetic))


def test_forbidden_refresh_write_retains_raw_before_rejection(synthetic, monkeypatch):
    refresh = owner.refresh_pressure

    def malicious_refresh(graph):
        refresh(graph)
        if synthetic.refresh_calls == 2:
            graph.nodes[0]["nu_f"] = 2.0

    monkeypatch.setattr(owner, "refresh_pressure", malicious_refresh)
    with pytest.raises(ValueError, match="non-pressure"):
        _run(synthetic)
    assert _kinds(synthetic).count("event_raw") == 1
    assert "event_after" not in _kinds(synthetic)
    assert not any(kind.startswith("sample_") for kind in _kinds(synthetic))
    raw = restore_recorded_graph_state(dict(synthetic.records)["event_raw"]).graph
    assert get_attr(raw.nodes[0], ALIAS_VF) == 1.0


def test_sink_cannot_change_live_clock_and_raw_record_survives(synthetic):
    def sink(kind, record):
        synthetic.sink(kind, record)
        if kind == "event_raw":
            synthetic.live.graph["_t"] += 1.0

    with pytest.raises(ValueError, match="record sink changed live graph"):
        _run(synthetic, sink=sink)
    assert _kinds(synthetic).count("event_raw") == 1
    raw = restore_recorded_graph_state(dict(synthetic.records)["event_raw"]).graph
    assert Q(raw.graph["_t"]) == synthetic.certificate.sample_times[0]


def test_detached_sink_payload_cannot_rewrite_shared_source_receipt(synthetic):
    receipt = {"path": "synthetic-only", "sha256": "not-retained-evidence"}
    synthetic.certificate.source.receipts = (receipt,)

    def sink(kind, record):
        synthetic.sink(kind, record)
        if kind == "source":
            record["source_receipts"][0]["sha256"] = "sink-local-change"

    result = _run(synthetic, sink=sink)
    assert receipt["sha256"] == "not-retained-evidence"
    assert result["source_receipts"] == (receipt,)


def test_numerical_callback_cannot_change_held_capacity_before_oz(synthetic):
    synthetic.step_mutation = lambda graph: graph.nodes[0].update(nu_f=2.0)
    with pytest.raises(
        ValueError,
        match="(flow|advance|numerical).*changed|changed.*(flow|advance|numerical)",
    ):
        _run(synthetic)
    assert not any(item[0] == "stage" for item in synthetic.calls)
    assert "event_raw" not in _kinds(synthetic)


def test_stage_cannot_change_primary_coordinate_before_pressure_handoff(synthetic):
    synthetic.raw_mutation = lambda graph: set_attr(graph.nodes[0], ALIAS_EPI, 0.75)
    with pytest.raises(ValueError, match="continuous coordinates"):
        _run(synthetic)
    assert "sample_0" not in _kinds(synthetic)


def test_passing_reader_cannot_override_failed_independent_assessment(synthetic):
    synthetic.final_passed = False
    with pytest.raises(
        ValueError, match="complete acquired history failed certification"
    ):
        _run(synthetic)
    assert _kinds(synthetic).count("sample_2") == 1
    assert synthetic.live.nodes[owner.CHILD]["glyph_history"][-1] == "OZ"
    assert sum(call[0] == "stage" for call in synthetic.calls) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        lambda graph: set_attr(graph.nodes[0], ALIAS_EPI, 0.75),
        lambda graph: set_attr(graph.nodes[0], ALIAS_THETA, -0.5),
        lambda graph: set_attr(graph.nodes[0], ALIAS_VF, 2.0),
        lambda graph: graph.graph.update(_t=4.0),
        lambda graph: graph.nodes[owner.CHILD].update(epi_time_history=((3.0, 0.375),)),
        lambda graph: graph.nodes[owner.CHILD]["glyph_history"].append("IL"),
        lambda graph: graph.edges[2, owner.CHILD].update(weight=1.0),
        lambda graph: graph.graph.update(hierarchy={}),
    ),
)
def test_guarded_refresh_rejects_non_pressure_footprint(monkeypatch, mutation):
    graph = _synthetic_graph()
    monkeypatch.setattr(owner, "refresh_pressure", mutation)
    with pytest.raises(ValueError, match="non-pressure"):
        owner._guarded_refresh(graph)


def test_invalid_sink_rejects_before_any_numerical_work(synthetic):
    with pytest.raises(TypeError, match="record sink"):
        owner._acquire_child_history(synthetic.certificate, on_record=True)
    assert synthetic.calls == []


@pytest.mark.parametrize(
    "end,first_step,reason",
    ((Q(3) + owner.INTEGRATION_STEP / 2, 0, "dyadic"), (Q(3), 417, "budget")),
)
def test_invalid_fixed_step_schedule_rejects_before_numerical_execution(
    end, first_step, reason
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid schedule reached numerical work or emission")

    with pytest.raises(ValueError, match=reason):
        owner._advance(
            (0.5,) * 42,
            forbidden,
            Q(3),
            end,
            first_step=first_step,
            emit=forbidden,
            check_budget=forbidden,
        )
