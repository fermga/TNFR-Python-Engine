"""Manufactured controls for the complete host cut-response certificate.

Small shared-kernel steps concern constructed graphs only. Synthetic loop
ledgers test orchestration and availability, not a retained or actual flow.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics import _validated_taylor as taylor
from tnfr.mathematics._rational_interval import I, pi_interval, sin, sqrt
from tnfr.research import maintained_sha_cut_response as owner
from tnfr.research import maintained_sha_direct_response as direct

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
STEP = Q(1, 512)


@pytest.fixture(autouse=True)
def no_retained_source(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructed cut-response controls cannot read a retained source")

    monkeypatch.setattr(owner, "validate_live_sha_successor", forbidden)
    monkeypatch.setattr(direct, "load_sha_retention", forbidden)


def _overlap(actual, expected):
    assert (actual - expected).contains(0)


def _persistent(graph):
    return deepcopy(
        (
            dict(graph.graph),
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
        )
    )


def _graph(
    *,
    forms=(Q(-1, 8), Q(1, 8), Q(0), Q(0)),
    phases=(Q(-1, 16), Q(-1, 16), Q(-1, 16), Q(3, 16)),
    capacities=(2, 3, 2, 1),
):
    graph = nx.Graph()
    for node, form, phase, capacity in zip(
        ("parent", "child", "rest", "remote"), forms, phases, capacities, strict=True
    ):
        graph.add_node(node, EPI=form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.add_weighted_edges_from(
        (("parent", "child", 2), ("parent", "rest", 3), ("rest", "remote", 5))
    )
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return graph


def _prepare(graph):
    return owner._prepare_cut_response(
        graph, parent="parent", child="child", reference_model=MODEL
    )


@pytest.fixture(scope="module")
def unequal_host_case():
    graph = _graph()
    before = _persistent(graph)
    prepared, flow = _prepare(graph)
    assert _persistent(graph) == before
    return prepared, flow


def test_exact_host_embedding_retains_the_child_and_own_weighted_means(
    unequal_host_case,
):
    prepared, flow = unequal_host_case
    source, observation = prepared.source, prepared.observation_source
    assert source.host_indices == (0, 2, 3)
    assert len(prepared.initial) == 6
    assert observation.degrees == (1, 0, 2, 1)
    assert observation.beta == (Q(2), Q(0), Q(1), Q(1))
    assert observation.mean_weights == (Q(1, 2), Q(0), Q(1), Q(1))
    assert observation.form_mean == Q(-1, 40)
    assert observation.phase_lift_mean == Q(3, 80)
    assert observation.reference_conductance == 2
    assert observation.edges == source.cut_edges
    assert observation.conductances == source.cut_conductances
    embedded = owner._embed_host_state(prepared, prepared.initial)
    assert embedded == tuple(map(I, source.epi + source.phase))
    assert owner._project_full_state(prepared, embedded) == prepared.initial
    host_rates = flow(prepared.initial)
    assert len(host_rates) == 6
    # Both host weighted rates cancel, with no isolated-child contribution.
    for row in (host_rates[:3], host_rates[3:]):
        assert sum(
            (
                weight * rate
                for weight, rate in zip(source.host_mean_weights, row, strict=True)
            ),
            I(0),
        ).contains(0)
    assert prepared.persistent_source_unchanged is True


def test_cut_original_observation_retains_full_parent_row_and_capacity_factor(
    unequal_host_case,
):
    prepared, flow = unequal_host_case
    full = owner._embed_host_state(prepared, prepared.initial)
    cell = direct._bound_cell(prepared.observation_source, full)
    pi, root_w = pi_interval(), sqrt(I(2))
    assert cell.r.contains(Q(1, 4))
    assert cell.delta.contains(0)
    assert cell.r_dot.contains(0)
    _overlap(cell.delta_dot, 3 / (4 * pi))
    _overlap(cell.angular_rate, -3 * root_w / (2 * pi))
    assert cell.orientation == "clockwise"
    assert all(cell.means_consistent)
    rates = flow(prepared.initial)
    _overlap(cell.r_dot, -rates[0])
    _overlap(cell.delta_dot, -rates[3])


def test_phase_axis_observation_uses_the_external_host_current():
    graph = _graph(forms=(0, 0, 0, 0), phases=(0, Q(1, 4), Q(1, 4), Q(1, 4)))
    prepared, flow = _prepare(graph)
    cell = direct._bound_cell(
        prepared.observation_source, owner._embed_host_state(prepared, prepared.initial)
    )
    expected = -2 * sin(I(Q(1, 4))) / pi_interval()
    _overlap(cell.r_dot, expected)
    assert cell.delta_dot.contains(0)
    _overlap(cell.angular_rate, -8 * sqrt(I(2)) * sin(I(Q(1, 4))) / pi_interval())
    _overlap(cell.r_dot, -flow(prepared.initial)[0])


def test_child_constants_do_not_enter_host_flow_but_remain_in_the_observation(
    unequal_host_case,
):
    original, original_flow = unequal_host_case
    graph = _graph()
    graph.nodes["child"].update(EPI=2, theta=Q(1, 2), nu_f=30)
    changed, changed_flow = _prepare(graph)
    assert changed.initial == original.initial
    assert changed_flow(changed.initial) == original_flow(original.initial)
    assert changed.observation_source.form_mean == original.observation_source.form_mean
    assert (
        changed.observation_source.phase_lift_mean
        == original.observation_source.phase_lift_mean
    )
    full = owner._embed_host_state(changed, changed.initial)
    assert full[1] == I(2)
    assert full[5] == I(Q(1, 2))
    cell = direct._bound_cell(changed.observation_source, full)
    assert cell.r.contains(Q(17, 8))
    assert cell.delta.contains(Q(9, 16))


def test_embedding_checks_host_dimension_and_keeps_constants_on_whole_boxes(
    unequal_host_case,
):
    prepared, _ = unequal_host_case
    box = tuple(value + I(Q(-1, 16), Q(1, 16)) for value in prepared.initial)
    full = owner._embed_host_state(prepared, box)
    assert full[1] == I(prepared.source.epi[1])
    assert full[5] == I(prepared.source.phase[1])
    assert full[0] == box[0] and full[2] == box[1] and full[3] == box[2]
    assert full[4] == box[3] and full[6] == box[4] and full[7] == box[5]
    assert owner._project_full_state(prepared, full) == box
    with pytest.raises(ValueError):
        owner._embed_host_state(prepared, box[:-1])
    changed_child = full[:1] + (full[1] + 1,) + full[2:]
    with pytest.raises(ValueError):
        owner._project_full_state(prepared, changed_child)


@pytest.fixture(scope="module")
def small_verified_step():
    graph = nx.Graph()
    for node, form, capacity in (
        ("parent", Q(-1, 8), 1),
        ("child", Q(1, 8), 3),
        ("rest", 0, 1),
    ):
        graph.add_node(node, EPI=form, theta=0, nu_f=capacity, delta_nfr=99)
    graph.add_weighted_edges_from((("parent", "child", 2), ("parent", "rest", 1)))
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    prepared, flow = _prepare(graph)
    step, failed, reason = taylor.validated_box_taylor_step(
        prepared.initial, STEP, flow, lambda box: (Q(1),), order=12, time=Q(0)
    )
    assert step is not None and failed is None and reason is None
    return prepared, flow, step


def test_one_constructed_shared_step_has_independently_revalidated_evidence(
    small_verified_step,
):
    prepared, flow, step = small_verified_step
    verification = owner._verify_step(step, prepared.initial, flow, index=0)
    assert verification is not None
    assert step.picard_interior_margin > 0
    assert step.initial_box == prepared.initial
    assert len(step.series) == len(prepared.initial)
    # The two-node unit host starts with x=(-1/8,0), theta=(0,0).
    # Thus x_a''=1/(4*pi^2), x_j''=-x_a'', and the normalized
    # second Taylor coefficients divide these derivatives by2!.
    _overlap(step.series[0][2], 1 / (8 * pi_interval() ** 2))
    _overlap(step.series[1][2], -1 / (8 * pi_interval() ** 2))
    _overlap(step.series[2][1], -1 / (8 * pi_interval()))
    _overlap(step.series[3][1], 1 / (8 * pi_interval()))


@pytest.mark.parametrize(
    "changed",
    (
        "series",
        "remainder",
        "picard",
        "clock",
        "duration",
        "carry",
        "endpoint",
        "method",
    ),
)
def test_tampered_step_evidence_is_rejected_without_trusting_cached_fields(
    small_verified_step, changed
):
    prepared, flow, step = small_verified_step
    if changed == "series":
        row = list(step.series[0])
        row[1] += I(1)
        invalid = replace(step, series=(tuple(row),) + step.series[1:])
    elif changed == "remainder":
        invalid = replace(
            step,
            local_remainder_bounds=(step.local_remainder_bounds[0] + 1,)
            + step.local_remainder_bounds[1:],
        )
    elif changed == "picard":
        invalid = replace(step, picard_interior_margin=step.picard_interior_margin + 1)
    elif changed == "clock":
        invalid = replace(step, time=STEP)
    elif changed == "duration":
        invalid = replace(step, duration=2 * STEP)
    elif changed == "carry":
        invalid = replace(
            step, initial_box=(step.initial_box[0] + 1,) + step.initial_box[1:]
        )
    elif changed == "endpoint":
        invalid = replace(step, endpoint=(step.endpoint[0] + 1,) + step.endpoint[1:])
    else:
        invalid = replace(step, method="unadmitted")
    with pytest.raises((ValueError, TypeError)):
        owner._verify_step(invalid, prepared.initial, flow, index=0)


@pytest.fixture
def synthetic_loop(monkeypatch, unequal_host_case):
    """A deliberately mocked certificate ledger; no trajectory is claimed."""
    prepared, _ = unequal_host_case
    full = owner._embed_host_state(prepared, prepared.initial)
    base = direct._bound_cell(prepared.observation_source, full)
    stationary = replace(
        base,
        angular_rate=I(0),
        angular_increment=I(0),
        orientation="stationary",
    )
    calls, records = [], []

    def no_flow(_):
        pytest.fail("synthetic orchestration must not evaluate a trajectory field")

    def stepper(box, duration, flow, domain, *, order, time):
        assert flow is no_flow
        assert duration == STEP and order == 12
        assert time == len(calls) * STEP
        assert box == prepared.initial
        assert domain(box) == (Q(1),)
        calls.append(time)
        return SimpleNamespace(initial_box=box, tube=box, endpoint=box), None, None

    def verify(step, initial, flow, *, index):
        assert flow is no_flow and initial == prepared.initial
        assert index == len(calls) - 1
        return owner.CutStepVerification(
            tube_rates=(I(0),) * len(initial), picard_interior_margin=Q(1)
        )

    def observe(source, tube, *, index, time, duration):
        assert source is prepared.observation_source
        assert tube == full
        assert duration == STEP and time == index * STEP
        return replace(stationary, index=index, time=time, duration=duration)

    monkeypatch.setattr(owner, "_verify_step", verify)
    monkeypatch.setattr(owner, "_bound_cell", observe)
    inputs = dict(
        preparation=prepared,
        flow=no_flow,
        record_step=records.append,
        monotonic_clock=lambda: 0,
        started=0,
        stepper=stepper,
    )
    return inputs, calls, records, stationary


def test_full_signed_response_is_retained_even_when_its_net_turn_gate_does_not_pass(
    synthetic_loop,
):
    inputs, calls, records, _ = synthetic_loop
    result = owner._certify_cut_response(**inputs)
    assert result.status == "completed"
    assert result.attempted_steps == result.accepted_steps == len(calls) == 256
    assert len(records) == len(result.cells) == 256
    assert result.certified_time == Q(1, 2)
    assert result.aggregate.full_window is True
    assert result.aggregate.radius_gate.status == "pass"
    assert result.aggregate.clockwise_travel == I(0)
    assert result.aggregate.net_turn_gate.status == "excluded"
    assert all(record["accepted"] is True for record in records)


def test_resource_boundary_stops_before_a_new_step_and_does_not_shorten_the_target(
    synthetic_loop,
):
    inputs, calls, records, _ = synthetic_loop
    inputs["monotonic_clock"] = lambda: 2401
    result = owner._certify_cut_response(**inputs)
    assert result.status == "resource_budget_exhausted"
    assert result.attempted_steps == result.accepted_steps == 0
    assert calls == records == []
    assert result.aggregate.full_window is False
    assert result.aggregate.clockwise_travel is None


def test_failed_step_retains_only_the_accepted_prefix_without_retry(synthetic_loop):
    inputs, calls, records, _ = synthetic_loop
    successful = inputs["stepper"]

    def one_step_then_fail(box, *args, **kwargs):
        if calls:
            return None, box, "constructed_unavailable"
        return successful(box, *args, **kwargs)

    inputs["stepper"] = one_step_then_fail
    result = owner._certify_cut_response(**inputs)
    assert result.status == "enclosure_unavailable"
    assert result.attempted_steps == 2 and result.accepted_steps == 1
    assert result.certified_time == STEP
    assert len(result.cells) == len(calls) == 1 and len(records) == 2
    assert records[-1]["accepted"] is False
    assert records[-1]["step"] is None
    assert result.aggregate.full_window is False
    assert result.aggregate.clockwise_travel is None


def test_verification_failure_is_recorded_before_any_observation(
    monkeypatch, synthetic_loop
):
    inputs, calls, records, _ = synthetic_loop

    def reject(*args, **kwargs):
        raise ValueError("constructed_bad_coefficient")

    def forbidden(*args, **kwargs):
        pytest.fail("an unverified cell must not reach the observation")

    monkeypatch.setattr(owner, "_verify_step", reject)
    monkeypatch.setattr(owner, "_bound_cell", forbidden)
    result = owner._certify_cut_response(**inputs)
    assert result.status == "verification_failed"
    assert result.attempted_steps == len(calls) == len(records) == 1
    assert result.accepted_steps == len(result.cells) == 0
    assert records[0]["accepted"] is False
    assert records[0]["observation"] is None


@pytest.mark.parametrize("missing", ("radius", "mean"))
def test_failed_observation_never_enters_the_certified_cell_sequence(
    monkeypatch, synthetic_loop, missing
):
    inputs, calls, records, base = synthetic_loop
    if missing == "radius":
        observation = replace(
            base,
            radius_squared=I(0),
            angular_rate=None,
            angular_increment=None,
            orientation="unavailable",
            unavailable_reasons=("constructed_zero_containing_radius",),
        )
    else:
        observation = replace(
            base,
            means_consistent=(False, True),
            unavailable_reasons=("cell_does_not_enclose_both_conserved_means",),
        )
    monkeypatch.setattr(owner, "_bound_cell", lambda *args, **kwargs: observation)
    result = owner._certify_cut_response(**inputs)
    assert result.status == "observation_unavailable"
    assert result.attempted_steps == len(calls) == len(records) == 1
    assert result.accepted_steps == len(result.cells) == 0
    assert records[0]["observation"] is observation
    assert records[0]["accepted"] is False
    assert result.aggregate.clockwise_travel is None


def test_recording_failure_propagates_instead_of_accepting_an_unrecorded_step(
    synthetic_loop,
):
    inputs, _, _, _ = synthetic_loop

    def fail_record(_):
        raise RuntimeError("constructed_recording_failure")

    inputs["record_step"] = fail_record
    with pytest.raises(RuntimeError, match="constructed_recording_failure"):
        owner._certify_cut_response(**inputs)


def test_budget_crossed_by_recording_keeps_the_prefix_without_a_full_window_claim(
    synthetic_loop,
):
    inputs, calls, records, _ = synthetic_loop
    inputs["monotonic_clock"] = lambda: 2400 if records else 0
    result = owner._certify_cut_response(**inputs)
    assert result.status == "resource_budget_exhausted"
    assert result.reason == "wall_budget_after_durable_record"
    assert result.attempted_steps == result.accepted_steps == len(calls) == 1
    assert records[0]["accepted"] is True
    assert result.certified_time == STEP
    assert result.aggregate.full_window is False
    assert result.aggregate.clockwise_travel is None


@pytest.mark.parametrize("stage", ("producer", "verification", "observation"))
def test_instrumentation_assertions_propagate_instead_of_becoming_method_outcomes(
    monkeypatch, synthetic_loop, stage
):
    inputs, _, records, _ = synthetic_loop

    def guard(*args, **kwargs):
        raise AssertionError("constructed_execution_guard")

    if stage == "producer":
        inputs["stepper"] = guard
    else:
        monkeypatch.setattr(
            owner, "_verify_step" if stage == "verification" else "_bound_cell", guard
        )
    with pytest.raises(AssertionError, match="constructed_execution_guard"):
        owner._certify_cut_response(**inputs)
    assert records == []


def _constructed_wrapper_graph():
    # Shape and clock exercise only the mocked public boundary. This is not
    # a reconstruction of the retained source or a numerical research fixture.
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    graph.add_node("2_sub_0")
    pairs = tuple((i, j) for i in range(20) for j in range(i + 1, 20))
    graph.add_edges_from(pairs[:52], weight=1)
    graph.add_edge(2, "2_sub_0", weight=1)
    graph.graph.update(_t=owner.TIMES[-1], GAMMA={"type": "none"})
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1, delta_nfr=99)
    return graph


def test_public_preparation_re_admits_and_copies_only_fresh_receipts(
    monkeypatch, unequal_host_case
):
    preparation, flow = unequal_host_case
    graph = _constructed_wrapper_graph()
    before = _persistent(graph)
    receipts = ({"path": "constructed", "nested": {"version": 1}},)
    fresh = SimpleNamespace(graph=graph, receipts=receipts)
    stale = SimpleNamespace(graph=object(), receipts=({"cached": True},))
    calls = []

    def readmit(value):
        assert value is stale
        calls.append("fresh_admission")
        return fresh

    def prepare(value, *, parent, child, reference_model):
        assert value is graph
        assert (parent, child) == (2, "2_sub_0")
        assert reference_model is owner.MODEL
        calls.append("prepare_fresh_primitives")
        return preparation, flow

    monkeypatch.setattr(owner, "validate_live_sha_successor", readmit)
    monkeypatch.setattr(owner, "_prepare_cut_response", prepare)
    result, result_flow = owner.prepare_sha_cut_response(stale)
    assert calls == ["fresh_admission", "prepare_fresh_primitives"]
    assert result_flow is flow and result.source is preparation.source
    assert result.source_receipts == receipts and result.source_receipts is not receipts
    receipts[0]["nested"]["version"] = 2
    assert result.source_receipts[0]["nested"]["version"] == 1
    assert _persistent(graph) == before


@pytest.mark.parametrize("invalid", ("shape", "clock"))
def test_public_preparation_checks_fresh_shape_and_clock_before_field_construction(
    monkeypatch, invalid
):
    graph = _constructed_wrapper_graph()
    if invalid == "shape":
        graph.remove_edge(0, 1)
    else:
        graph.graph["_t"] = 0
    fresh = SimpleNamespace(graph=graph, receipts=())
    monkeypatch.setattr(owner, "validate_live_sha_successor", lambda _: fresh)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid fresh source must not reach field construction")

    monkeypatch.setattr(owner, "_prepare_cut_response", forbidden)
    with pytest.raises(ValueError):
        owner.prepare_sha_cut_response(object())


def test_public_certificate_starts_its_budget_before_fresh_preparation(
    monkeypatch, unequal_host_case
):
    preparation, flow = unequal_host_case
    supplied, answer = object(), object()
    calls = []

    def clock():
        calls.append("clock")
        return Q(7, 2)

    def prepare(value):
        assert value is supplied
        calls.append("prepare")
        return preparation, flow

    def record(_):
        pytest.fail("the mocked public wiring must not emit a numerical record")

    def certify(prepared, actual_flow, *, record_step, monotonic_clock, started):
        assert prepared is preparation and actual_flow is flow
        assert record_step is record and monotonic_clock is clock
        assert started == Q(7, 2)
        calls.append("certify")
        return answer

    monkeypatch.setattr(owner, "prepare_sha_cut_response", prepare)
    monkeypatch.setattr(owner, "_certify_cut_response", certify)
    assert (
        owner.certify_sha_cut_response(
            supplied, record_step=record, monotonic_clock=clock
        )
        is answer
    )
    assert calls == ["clock", "prepare", "certify"]
