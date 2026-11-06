"""A frozen six-call native composition attempt exits at its second batch.

Keep the nominal neighbor-gain source, seed 17, t=1, dt=1/16, strict grammar,
default laws and consecutive twin order. The admitted deterministic sector is
AL/EN/IL/RA. A test-local observer stops the entire batch before any primitive
when a completed native preparation contains a mark outside that sector.
This protocol stop is neither a grammar rejection nor native rollback.

Before any native call, an independent exact certificate applies fresh
pressure/Si to the existing first-call forecast. Its next phase contrasts
cancel the common global angle, leaving rational quantities and the request
RA/NAV/IL/IL/IL. Thus a supported first block must also remain unexecuted at
the second call. The six-call maximum is retained without forcing EN or AL.

Prepare all four graphs first. Evaluate both five-node predictor trajectories,
check their first calls against the unchanged independent forecast (1e-14),
and freeze their records before either reserved doubled-C5 trajectory. The
cross-support budget is 2e-14 per ordinate, with exact decisions and discrete
memory. The maximum is 24 native invocations; the predicted first exit needs
eight. Unexpected defects propagate; no source search or error-budget tuning
is allowed. This proves no invariant sector, autonomous organization or
physical identification, and creates no production prediction API.
"""

from copy import deepcopy
from fractions import Fraction as Q
from functools import wraps

import pytest

from tests.physics import test_native_collective_step as base
from tests.physics import test_native_ra_il_block_descent as descent
from tests.physics import test_native_resonance_neighbor_gain as gain
from tests.physics import test_native_resonance_u3_boundary as boundary
from tnfr.operators.grammar_debt import canonical_operator_code, read_grammar_state

HORIZON = 6
SECTOR = frozenset(("AL", "EN", "IL", "RA"))
FIRST_MARKS = ("RA", "RA", "IL", "IL", "IL")
SECOND_MARKS = ("RA", "NAV", "IL", "IL", "IL")
FIRST_ERROR = Q(1, 10**14)
CROSS_ERROR = 2 * FIRST_ERROR


class PreparedSectorExit(RuntimeError):
    """A research-protocol boundary after preparation, before every event."""

    def __init__(self, requests):
        self.requests = tuple(requests)
        self.outside = tuple(
            (node, code) for node, code in requests if code not in SECTOR
        )
        super().__init__(f"prepared requests leave AL/EN/IL/RA: {self.outside!r}")


def _check_complete_requests(nodes, forced, selected):
    """Validate complete coverage before testing membership, never substitute."""
    assert not forced.keys() & selected.keys()
    requests = {**forced, **selected}
    assert set(requests) == set(nodes) and len(requests) == len(nodes)
    ordered = tuple((node, canonical_operator_code(requests[node])) for node in nodes)
    if any(code not in SECTOR for _, code in ordered):
        raise PreparedSectorExit(ordered)
    return ordered


def _second_call_certificate(graph, first):
    """Exact next diagnostics: the common coordinated global angle cancels."""
    phase = first["event_phase"]
    kg, kl = first["gains"]
    assert 0 < min(phase) <= max(phase) < gain.PI
    assert kg > 0 and kl > 0 and kg + kl < 1

    def neighbor(values, block):
        return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2

    local = tuple(neighbor(phase, i) for i in range(5))
    z = tuple((1 - kg - kl) * value + kl * local[i] for i, value in enumerate(phase))
    assert max(z) - min(z) < Q(1083, 500) < gain.PI
    # The defined global resultant lies in the same strict event-phase arc;
    # convex coordination retains that arc. The approximate absolute-angle
    # forecast below is only a consistency check, not this domain proof.
    assert all(0 < value < gain.PI for value in first["phase"])
    offsets = tuple(neighbor(z, i) - value for i, value in enumerate(z))
    # Independently verify cancellation against the complete first forecast.
    assert offsets == tuple(
        neighbor(first["phase"], i) - first["phase"][i] for i in range(5)
    )
    k, b, w = (
        base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf", "phase")
    )
    x, v = first["form"], first["capacity"]
    pressure = tuple(
        k * (neighbor(x, i) - x[i])
        + b * (neighbor(v, i) - v[i])
        + w * offsets[i] / gain.PI
        for i in range(5)
    )
    peak = max(map(abs, pressure))
    assert peak == -pressure[3]
    ratios = tuple(abs(p) / peak for p in pressure)
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * v[i] / max(v)
        + beta * (1 - abs(offsets[i]) / gain.PI)
        + gamma * (1 - ratios[i])
        for i in range(5)
    )
    limits = graph.graph["SELECTOR_THRESHOLDS"]
    assert Q(38, 100) < si[0] < Q(381, 1000)
    assert Q(376, 1000) < si[1] < Q(377, 1000)
    assert Q(limits["si_lo"]) < min(si[:2]) <= max(si[:2]) < Q(limits["si_hi"])
    assert ratios[0] < Q(1, 200) < Q(limits["dnfr_hi"])
    assert Q(235, 1000) < ratios[1] < Q(237, 1000)
    assert ratios[1] - Q(limits["dnfr_hi"]) > Q(1, 10)
    assert min(si[2:]) > Q(84, 100) > Q(limits["si_hi"])
    assert graph.graph["AL_MAX_LAG"] == 5 and graph.graph["EN_MAX_LAG"] == 3
    return {"pressure": pressure, "si": si, "ratios": ratios, "marks": SECOND_MARKS}


def _memory(graph):
    history = graph.graph["history"]
    return {
        "grammar": tuple(read_grammar_state(data) for data in graph.nodes.values()),
        "kinds": tuple(data["epi_kind"] for data in graph.nodes.values()),
        "since_AL": tuple(history["since_AL"][node] for node in graph),
        "since_EN": tuple(history["since_EN"][node] for node in graph),
        "gains": (graph.graph["PHASE_K_GLOBAL"], graph.graph["PHASE_K_LOCAL"]),
        "phase_history": {
            key: tuple(history.get(key, ()))
            for key in ("phase_state", "phase_R", "phase_disr", "phase_kG", "phase_kL")
        },
        "remesh_history": tuple(
            tuple(snapshot.items()) for snapshot in graph.graph.get("_epi_hist", ())
        ),
        "stable_frac": tuple(history.get("stable_frac", ())),
        "remesh_fired": "_REMESH_META" in graph.graph,
    }


def _observe_call(graph, *, scalar):
    """Observe actual preparation; stop before returning its last proposal.

    The inner resolver delegates recursively. Only its outer invocation is
    one actual target request. Forced AL/EN rows come from the exact admission
    object consumed by this batch, with its incremented lag counts and priority.
    No target is executed, reselected or substituted by this observer.
    """
    selectors = base.selectors
    actual_preflight = selectors._preflight_glyph_batch
    actual_batch = selectors._apply_glyph_batch
    actual_choice = selectors._resolve_preselected_glyph
    actual_gate = selectors.enforce_canonical_grammar
    actual_primitive = selectors.apply_glyph
    actual_step = base.runtime.step
    state = {
        "active": False,
        "depth": 0,
        "admission": None,
        "forced": {},
        "selected": {},
        "requests": None,
        "gate_count": 0,
        "primitive_count": 0,
        "exit": None,
    }
    before = _memory(graph)

    @wraps(actual_batch)
    def observe_batch(*args, **kwargs):
        assert state["active"] is False
        state["active"] = True
        try:
            return actual_batch(*args, **kwargs)
        finally:
            state["active"] = False

    @wraps(actual_preflight)
    def observe_admission(*args, **kwargs):
        admission = actual_preflight(*args, **kwargs)
        if state["active"]:
            assert state["admission"] is None
            state["admission"] = admission
            state["forced"] = {
                node: "AL" if admission.next_al[node] > admission.al_max else "EN"
                for node in admission.nodes
                if admission.next_al[node] > admission.al_max
                or admission.next_en[node] > admission.en_max
            }
            if len(state["forced"]) == len(admission.nodes):
                # This path contains only forced AL/EN, both in the sector.
                state["requests"] = _check_complete_requests(
                    admission.nodes, state["forced"], {}
                )
        return admission

    @wraps(actual_choice)
    def observe_choice(current, node, *args, **kwargs):
        outer = state["depth"] == 0
        if outer:
            assert current is graph and state["active"] is True
            assert args[0] is selectors.default_glyph_selector
        state["depth"] += 1
        try:
            result = actual_choice(current, node, *args, **kwargs)
        finally:
            state["depth"] -= 1
        if outer:
            admission = state["admission"]
            assert node in admission.nodes and node not in state["forced"]
            assert node not in state["selected"]
            state["selected"][node] = canonical_operator_code(result)
            if len(state["selected"]) + len(state["forced"]) == len(admission.nodes):
                assert state["gate_count"] == state["primitive_count"] == 0
                try:
                    state["requests"] = _check_complete_requests(
                        admission.nodes, state["forced"], state["selected"]
                    )
                except PreparedSectorExit as exc:
                    state["requests"] = exc.requests
                    raise
        return result

    @wraps(actual_gate)
    def observe_gate(current, node, glyph, **kwargs):
        assert state["requests"] is not None
        assert dict(state["requests"])[node] == canonical_operator_code(glyph)
        state["gate_count"] += 1
        return actual_gate(current, node, glyph, **kwargs)

    @wraps(actual_primitive)
    def observe_primitive(current, node, glyph, **kwargs):
        assert canonical_operator_code(glyph) in SECTOR
        state["primitive_count"] += 1
        return actual_primitive(current, node, glyph, **kwargs)

    @wraps(actual_step)
    def observe_step(*args, **kwargs):
        try:
            return actual_step(*args, **kwargs)
        except PreparedSectorExit as exc:
            assert state["gate_count"] == state["primitive_count"] == 0
            state["exit"] = exc
            return None

    with pytest.MonkeyPatch.context() as patch:
        for name, observer in (
            ("_apply_glyph_batch", observe_batch),
            ("_preflight_glyph_batch", observe_admission),
            ("_resolve_preselected_glyph", observe_choice),
            ("enforce_canonical_grammar", observe_gate),
            ("apply_glyph", observe_primitive),
        ):
            patch.setattr(selectors, name, observer)
        patch.setattr(base.runtime, "step", observe_step)
        case = boundary._execute(graph, descent._configuration(graph), scalar=scalar)
    case.update(
        requests=state["requests"],
        protocol_exit=state["exit"],
        status=(
            "outside_sector"
            if state["exit"] is not None
            else "grammar_rejection" if case["error"] is not None else "complete"
        ),
        memory_before=before,
        memory_after=_memory(graph),
    )
    return case


def _record(case):
    """Freeze declared observations, excluding live graph/cache references."""
    return deepcopy(
        {
            "status": case["status"],
            "requests": case["requests"],
            "initial": case["initial"],
            "final": case["final"],
            "memory_before": case["memory_before"],
            "memory_after": case["memory_after"],
            "event_names": tuple(event["name"] for event in case["events"]),
            "primitive_requests": tuple(
                (node, code) for node, code, _, _ in case["primitives"]
            ),
            "gate_requests": tuple(
                (entry["node"], entry["glyph"]) for entry in case["gates"]
            ),
            "fresh_pressure": base._event(case, "pressure")["after"]["pressure"],
            "fresh_si": base._event(case, "si")["after"]["si"],
        }
    )


def _first_check(case, expected, multiplicity):
    assert case["status"] == "complete"
    assert case["requests"] == tuple(
        enumerate(descent._lift(FIRST_MARKS, multiplicity))
    )
    for name in base.CHANNELS:
        descent._close(
            case["final"][name],
            descent._lift(expected[name], multiplicity),
            FIRST_ERROR,
        )


def _trajectory(graph, scalar, first, multiplicity):
    cases = []
    for invocation in range(HORIZON):
        case = _observe_call(graph, scalar=scalar)
        cases.append(case)
        if invocation == 0:
            _first_check(case, first, multiplicity)
        if case["status"] != "complete":
            break
    return cases


@pytest.fixture(scope="module")
def evidence():
    frozen = {}
    for scalar in (False, True):
        fine = gain._source()
        gain._family_certificate(fine)
        first = gain._forecast(fine)
        second = _second_call_certificate(fine, first)
        coarse = descent._coarse_source(fine)
        # Reuse the support projector, retaining this gate's original stored
        # fields rather than the previous gate's unrelated discard controls.
        for block in coarse:
            for name in ("delta_nfr", "Si", "d2EPI"):
                if name in fine.nodes[2 * block]:
                    coarse.nodes[block][name] = deepcopy(fine.nodes[2 * block][name])
                else:
                    coarse.nodes[block].pop(name, None)
        if scalar:
            coarse.graph["vectorized_dnfr"] = fine.graph["vectorized_dnfr"] = False
        frozen[scalar] = dict(
            fine=fine,
            coarse=coarse,
            first=first,
            second=second,
            fine_source=base._snapshot(fine),
            configurations=(
                descent._configuration(fine),
                descent._configuration(coarse),
            ),
        )
    # Both sources, all initial laws and both independent forecasts precede
    # the first native coarse response. No fine evaluation occurs in this loop.
    for scalar, item in frozen.items():
        item["coarse_cases"] = _trajectory(item["coarse"], scalar, item["first"], 1)
        item["prediction"] = tuple(_record(case) for case in item["coarse_cases"])
    assert all(
        base._snapshot(item["fine"]) == item["fine_source"] for item in frozen.values()
    )
    # Every coarse trajectory is now frozen before either reserved fine run.
    for scalar, item in frozen.items():
        item["fine_cases"] = _trajectory(item["fine"], scalar, item["first"], 2)
    return frozen


def test_complete_batch_guard_rejects_a_later_unsupported_request():
    for unsupported in ("NAV", "ZHIR", "OZ"):
        with pytest.raises(PreparedSectorExit) as caught:
            _check_complete_requests((0, 1, 2), {2: "EN"}, {0: "RA", 1: unsupported})
        assert caught.value.requests == ((0, "RA"), (1, unsupported), (2, "EN"))
        assert caught.value.outside == ((1, unsupported),)
    assert _check_complete_requests((0, 1), {0: "AL", 1: "EN"}, {}) == (
        (0, "AL"),
        (1, "EN"),
    )


@pytest.mark.parametrize("mode", ("first_zhir", "mixed_forcing", "all_forced"))
def test_observer_wiring_with_mocked_step_and_no_native_dynamics(mode, monkeypatch):
    """Exercise the real observer around a synthetic prepared-batch driver.

    Only shared read-only batch admission is real here. Selection outputs and
    event bodies are mocks; no pressure, operator action, integration or phase
    response is evaluated. This is a wiring control, not another preparation
    in the scientific trajectory campaign.
    """
    graph = gain._source()
    history = graph.graph["history"]
    requested = {node: "IL" for node in graph}
    forced = {}
    if mode == "first_zhir":
        requested[0] = "ZHIR"
    else:
        targets = (0, 1) if mode == "mixed_forcing" else tuple(graph)
        for node in targets:
            # Both lags exceed their limits at even nodes; AL must win.
            history["since_AL"][node] = 5 if node % 2 == 0 else 0
            history["since_EN"][node] = 3
            forced[node] = "AL" if node % 2 == 0 else "EN"
        requested.update(forced)
    prelude = []

    def mocked_choice(current, node, selector, preselection):
        assert current is graph
        return requested[node]

    def mocked_batch(current, selector, hist):
        admission = base.selectors._preflight_glyph_batch(current, hist)
        selector._prepared_graph = current
        selector._preselection = object()
        hist["since_AL"].update(admission.next_al)
        hist["since_EN"].update(admission.next_en)
        proposals = dict(forced)
        for node in admission.nodes:
            if node not in forced:
                proposals[node] = base.selectors._resolve_preselected_glyph(
                    current, node, selector, None
                )
        for node in admission.nodes:
            glyph = proposals[node]
            if glyph == "ZHIR":
                prelude.append(node)
            admitted = base.selectors.enforce_canonical_grammar(current, node, glyph)
            base.selectors.apply_glyph(current, node, admitted, window=admission.window)

    def mocked_step(current, **kwargs):
        try:
            return base.selectors._apply_glyph_batch(
                current, base.selectors.default_glyph_selector, current.graph["history"]
            )
        finally:
            base.selectors.default_glyph_selector.clear()

    monkeypatch.setattr(base.runtime, "step", mocked_step)
    monkeypatch.setattr(base.selectors, "_apply_glyph_batch", mocked_batch)
    monkeypatch.setattr(base.selectors, "_resolve_preselected_glyph", mocked_choice)
    monkeypatch.setattr(
        base.selectors, "enforce_canonical_grammar", lambda graph, node, glyph: glyph
    )
    monkeypatch.setattr(base.selectors, "apply_glyph", lambda *args, **kwargs: None)

    observed = _observe_call(graph, scalar=False)

    assert observed["requests"] == tuple(requested.items())
    assert observed["selector_cleared"] is True
    assert prelude == []
    if mode == "first_zhir":
        assert observed["status"] == "outside_sector"
        assert observed["protocol_exit"].outside == ((0, "ZHIR"),)
        assert observed["primitives"] == observed["gates"] == ()
    else:
        assert observed["status"] == "complete"
        assert tuple(
            (node, code) for node, code, _, _ in observed["primitives"]
        ) == tuple(requested.items())
        if mode == "all_forced":
            assert observed["choices"] == observed["prepared"] == ()


def test_independent_rational_certificate_predicts_the_first_sector_exit():
    graph = gain._source()
    gain._family_certificate(graph)
    predicted = _second_call_certificate(graph, gain._forecast(graph))
    assert predicted["marks"] == SECOND_MARKS
    assert predicted["ratios"][1] - Q(
        graph.graph["SELECTOR_THRESHOLDS"]["dnfr_hi"]
    ) > Q(1, 10)


@pytest.mark.parametrize("scalar", (False, True))
def test_coarse_predictions_are_frozen_before_matching_reserved_fine_calls(
    evidence, scalar
):
    item = evidence[scalar]
    assert (
        len(item["prediction"])
        == len(item["coarse_cases"])
        == len(item["fine_cases"])
        == 2
        < HORIZON
    )
    assert tuple(record["status"] for record in item["prediction"]) == (
        "complete",
        "outside_sector",
    )
    for coarse, fine in zip(item["prediction"], item["fine_cases"], strict=True):
        assert fine["status"] == coarse["status"]
        assert fine["requests"] == tuple(
            enumerate(descent._lift(tuple(code for _, code in coarse["requests"]), 2))
        )
        assert tuple(event["name"] for event in fine["events"]) == coarse["event_names"]
        assert tuple((node, code) for node, code, _, _ in fine["primitives"]) == tuple(
            enumerate(
                descent._lift(
                    tuple(code for _, code in coarse["primitive_requests"]), 2
                )
            )
        )
        for name in base.CHANNELS:
            descent._close(
                fine["final"][name],
                descent._lift(coarse["final"][name], 2),
                CROSS_ERROR,
            )
        for event, channel, name in (
            ("pressure", "pressure", "fresh_pressure"),
            ("si", "si", "fresh_si"),
        ):
            descent._close(
                base._event(fine, event)["after"][channel],
                descent._lift(coarse[name], 2),
                CROSS_ERROR,
            )
        assert fine["final"]["time"] == coarse["final"]["time"]
        assert fine["final"]["ordinal"] == coarse["final"]["ordinal"]
        assert fine["final"]["counters"] == descent._lift(
            coarse["final"]["counters"], 2
        )
        for actual, predicted in zip(
            fine["final"]["temporal"],
            descent._lift(coarse["final"]["temporal"], 2),
            strict=True,
        ):
            assert tuple(t for t, _ in actual) == tuple(t for t, _ in predicted)
            descent._close(
                tuple(x for _, x in actual), tuple(x for _, x in predicted), CROSS_ERROR
            )
        assert fine["memory_after"]["grammar"] == descent._lift(
            coarse["memory_after"]["grammar"], 2
        )
        assert fine["memory_after"]["kinds"] == descent._lift(
            coarse["memory_after"]["kinds"], 2
        )
        for name in ("since_AL", "since_EN"):
            assert fine["memory_after"][name] == descent._lift(
                coarse["memory_after"][name], 2
            )
        descent._close(
            fine["memory_after"]["gains"], coarse["memory_after"]["gains"], CROSS_ERROR
        )
        for name, predicted in coarse["memory_after"]["phase_history"].items():
            actual = fine["memory_after"]["phase_history"][name]
            if name == "phase_state":
                assert actual == predicted
            else:
                descent._close(actual, predicted, CROSS_ERROR)
        for actual, predicted in zip(
            fine["memory_after"]["remesh_history"],
            coarse["memory_after"]["remesh_history"],
            strict=True,
        ):
            lifted = tuple(
                (2 * node + offset, value)
                for node, value in predicted
                for offset in (0, 1)
            )
            assert tuple(node for node, _ in actual) == tuple(
                node for node, _ in lifted
            )
            descent._close(
                tuple(x for _, x in actual), tuple(x for _, x in lifted), CROSS_ERROR
            )


@pytest.mark.parametrize("scalar", (False, True))
def test_second_refresh_matches_the_exact_angle_cancelling_forecast(evidence, scalar):
    item = evidence[scalar]
    for multiplicity, cases in ((1, item["coarse_cases"]), (2, item["fine_cases"])):
        case = cases[1]
        for name, event in (("pressure", "pressure"), ("si", "si")):
            descent._close(
                base._event(case, event)["after"][name],
                descent._lift(item["second"][name], multiplicity),
                FIRST_ERROR,
            )
        assert case["requests"] == tuple(
            enumerate(descent._lift(SECOND_MARKS, multiplicity))
        )
        assert case["protocol_exit"].outside == tuple(
            (node, "NAV") for node in range(multiplicity, 2 * multiplicity)
        )


@pytest.mark.parametrize("scalar", (False, True))
def test_protocol_exit_precedes_even_the_supported_first_event(evidence, scalar):
    item = evidence[scalar]
    for cases in (item["coarse_cases"], item["fine_cases"]):
        first, stopped = cases
        assert stopped["requests"][0] == (0, "RA")
        assert stopped["error"] is None and isinstance(
            stopped["protocol_exit"], PreparedSectorExit
        )
        assert stopped["gates"] == stopped["primitives"] == stopped["u3"] == ()
        assert tuple(event["name"] for event in stopped["events"]) == (
            "flow_history",
            "pressure",
            "si",
            "glyphs",
        )
        assert "after" not in base._event(stopped, "glyphs")
        for name in (
            "form",
            "phase",
            "capacity",
            "rate",
            "acceleration",
            "counters",
            "edges",
        ):
            assert (
                stopped["final"][name]
                == stopped["initial"][name]
                == first["final"][name]
            )
        assert stopped["final"]["time"] == 1 + base.DT
        assert stopped["final"]["ordinal"] == 2
        assert stopped["memory_after"]["grammar"] == first["memory_after"]["grammar"]
        assert stopped["memory_after"]["gains"] == first["memory_after"]["gains"]
        assert (
            stopped["memory_after"]["phase_history"]
            == first["memory_after"]["phase_history"]
        )
        assert (
            stopped["memory_after"]["remesh_history"]
            == first["memory_after"]["remesh_history"]
        )
        assert stopped["final"]["temporal"] == first["final"]["temporal"]
        for name in ("since_AL", "since_EN"):
            assert set(first["memory_after"][name]) == {1}
            assert set(stopped["memory_after"][name]) == {2}
        assert stopped["selector_cleared"] is True


@pytest.mark.parametrize("scalar", (False, True))
def test_every_call_retains_its_predecessor_and_default_law(evidence, scalar):
    item = evidence[scalar]
    for multiplicity, cases in ((1, item["coarse_cases"]), (2, item["fine_cases"])):
        first, second = cases
        assert second["initial"] == first["final"]
        assert second["memory_before"] == first["memory_after"]
        for case in cases:
            cfg = case["configuration"]
            assert cfg["GRAMMAR_REJECTION_MODE"] == "raise" and cfg["RANDOM_SEED"] == 17
            assert cfg["compute_delta_nfr"] is base.default_compute_delta_nfr
            assert cfg["GAMMA"]["type"] == "none"
            assert tuple(case["graph"]) == tuple(range(5 * multiplicity))
            assert len(case["final"]["edges"]) == 5 * multiplicity**2
            assert (
                not case["memory_after"]["stable_frac"]
                and not case["memory_after"]["remesh_fired"]
            )
            assert case["selector_cleared"] is True
            assert "mutation_abstentions" not in case["graph"].graph["history"]
        assert (
            second["configuration"]["PHASE_K_GLOBAL"]
            == first["memory_after"]["gains"][0]
        )
        assert (
            second["configuration"]["PHASE_K_LOCAL"]
            == first["memory_after"]["gains"][1]
        )
