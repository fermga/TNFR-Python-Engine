"""Contextual NAV controls on an independent EN/IL/SHA contact successor.

Only named hybrid events are executed. No retained research preparation,
producer, continuous trajectory or physical inference is evaluated here.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from fractions import Fraction as Q
from math import cos, pi, tau

import pytest

from tests.operators.test_collective_reception import (
    PERMUTATION,
    PORTS,
    _contact,
    _fields,
    _refresh,
)
from tnfr.constants.canonical import COUPLING_FINE
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators import network_stage
from tnfr.operators.definitions import Coherence, Reception, Silence, Transition
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import (
    validate_candidate,
    validate_sequence_incremental,
)
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.relational_sine_symmetry import assess_sine_support_involution
from tnfr.types import require_finite_real_scalar_epi

PREPARATION_TIME = datetime(2030, 1, 2, 3, 4, 7, tzinfo=timezone.utc)
TRANSITION_TIME = PREPARATION_TIME + timedelta(seconds=2)


class _SuppliedClock:
    observed = PREPARATION_TIME

    @classmethod
    def now(cls, tz=None):
        return cls.observed if tz is not None else cls.observed.replace(tzinfo=None)


def _physical(graph):
    return tuple(
        (
            require_finite_real_scalar_epi(graph.nodes[node]["EPI"]),
            graph.nodes[node]["theta"],
            graph.nodes[node]["nu_f"],
            graph.nodes[node]["delta_nfr"],
        )
        for node in graph
    )


def _preparation(*, broad_counterexample=False):
    graph = _contact()
    graph.graph.update(NAV_RANDOM=False, NAV_STRICT=False, RANDOM_SEED=37)
    for node in graph:
        reference = 2 * pi * (node % 10 // 2) / 5
        if broad_counterexample:
            offset = (
                3 / 25
                if node in (0, 1, 2, 3, 10, 11)
                else -3 / 25 if node in (8, 9) else 0.0
            )
        else:
            offset = 1 / 128 if node == 1 else 1 / 32 if node in (10, 11) else 0.0
        graph.nodes[node]["theta"] = reference + offset
    _refresh(graph)
    operators = (Reception(), Coherence(), Silence())
    sequence = ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_SuppliedClock, "observed", PREPARATION_TIME)
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        for index, operator in enumerate(operators):
            if index == 0:
                with pytest.warns(UserWarning, match="no emission sources detected"):
                    execute_network_operator_stage(
                        graph,
                        operator,
                        PORTS,
                        sequence_context=sequence.step(index),
                        compute_delta_nfr=_refresh,
                    )
            else:
                execute_network_operator_stage(
                    graph,
                    operator,
                    PORTS,
                    sequence_context=sequence.step(index),
                    compute_delta_nfr=_refresh,
                )
    return graph


@pytest.fixture(scope="module")
def successor():
    return _preparation()


def _transition(graph, *, targets=PORTS):
    sequence = ValidatedSequence(
        (Transition(),),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    raw = []

    def refresh(subject):
        raw.append(tuple(subject.nodes[node]["delta_nfr"] for node in subject))
        _refresh(subject)

    baseline = compute_structural_potential(graph)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_SuppliedClock, "observed", TRANSITION_TIME)
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        result = execute_network_operator_stage(
            graph,
            Transition(),
            targets,
            sequence_context=sequence.step(0),
            compute_delta_nfr=refresh,
        )
    observation = observe_structural_potential_confinement(
        graph,
        baseline,
        compute_structural_potential(graph),
        reference="synthetic_before_nav",
    )
    return result, raw[0], observation


@pytest.fixture(scope="module")
def completed_transition(successor):
    graph = deepcopy(successor)
    result, raw, observation = _transition(graph)
    return graph, result, raw, observation


def test_singleton_nav_is_live_admissible_and_consumes_latent_successor_context(
    successor, completed_transition
):
    after, result, _, observation = completed_transition
    assert all(validate_candidate(successor, node, "NAV").allowed for node in PORTS)
    assert all(
        item.allowed
        for node in PORTS
        for item in validate_sequence_incremental(successor, node, ("NAV",))
    )
    assert result.glyph == "NAV"
    assert result.schedule == "two_phase_jacobi"
    assert result.nodes_processed == 2
    assert observation.confined
    assert after.graph["_t"] == successor.graph["_t"] == 7
    assert tuple(after.edges(data=True)) == tuple(successor.edges(data=True))
    before_physical, after_physical = _physical(successor), _physical(after)
    for node in after:
        if node not in PORTS:
            assert before_physical[node][:3] == after_physical[node][:3]
            continue
        assert successor.nodes[node]["latent"] is True
        assert successor.nodes[node]["nu_f"] > 0.05
        assert after.nodes[node]["_regime_before"] == "latent"
        assert after.nodes[node]["nu_f"] == successor.nodes[node]["nu_f"] * 1.2
        assert (
            after.nodes[node]["theta"]
            == (successor.nodes[node]["theta"] % tau + 0.1) % tau
        )
        assert after_physical[node][0] == before_physical[node][0]
        assert node_debt(successor.nodes[node]) == node_debt(after.nodes[node]) == 0
        assert tuple(after.nodes[node]["glyph_history"])[-4:] == (
            "EN",
            "IL",
            "SHA",
            "NAV",
        )
        assert after.nodes[node]["silence_duration"] == 2.0
        assert not {"latent", "latency_start_time", "preserved_epi"}.intersection(
            after.nodes[node]
        )
    reduction = assess_sine_support_involution(
        _fields(after), permutation_indices=PERMUTATION
    )
    assert reduction.source_trajectory_reduction_certified
    assert reduction.multiplicities == (1, 1) + (2,) * 9
    for left, right in after.edges:
        gap = (after.nodes[right]["theta"] - after.nodes[left]["theta"] + pi) % tau - pi
        assert abs(gap) < pi / 2


def test_equal_physical_states_do_not_select_the_same_nav_regime(
    successor, completed_transition
):
    active = deepcopy(successor)
    for node in PORTS:
        active.nodes[node]["latent"] = False
    assert _physical(active) == _physical(successor)
    _transition(active)
    latent, *_ = completed_transition
    for node in PORTS:
        assert active.nodes[node]["_regime_before"] == "active"
        assert active.nodes[node]["nu_f"] == successor.nodes[node]["nu_f"]
        assert (
            active.nodes[node]["theta"]
            == (successor.nodes[node]["theta"] % tau + 0.2) % tau
        )
        assert active.nodes[node]["theta"] != latent.nodes[node]["theta"]
        assert active.nodes[node]["nu_f"] != latent.nodes[node]["nu_f"]


def test_fresh_rows_and_storage_use_lasting_phase_and_capacity_changes(
    successor, completed_transition
):
    graph, _, _, _ = completed_transition
    before, after = _fields(successor), _fields(graph)
    assert before.epi == after.epi
    assert before.form_gradient == after.form_gradient
    assert before.form_storage == after.form_storage
    phase_work = sum(
        cos(float(before.phase[j] - before.phase[i]))
        - cos(float(after.phase[j] - after.phase[i]))
        for i, j in before.edges
    )
    interval_work = after.phase_storage - before.phase_storage
    assert float(interval_work.midpoint) == pytest.approx(phase_work, abs=1e-14)
    assert float((after.storage - before.storage).midpoint) == pytest.approx(
        phase_work, abs=1e-14
    )
    assert phase_work > 0
    assert before.pressure != after.pressure
    assert before.form_rates != after.form_rates
    before_rates, after_rates = (
        before.phase_rate_numerators(),
        after.phase_rate_numerators(),
    )
    for node in graph:
        ratio = after.capacity[node] / before.capacity[node]
        assert after_rates[node] == ratio * before_rates[node]
        assert after.form_rates[node] == after.pressure[node] * after.capacity[node]
        if node not in (0, 1, 2, 3, 8, 9, 10, 11):
            assert before.pressure[node] == after.pressure[node]
    rho_before = tuple(
        Q(d) / v for d, v in zip(before.degrees, before.capacity, strict=True)
    )
    rho_after = tuple(
        Q(d) / v for d, v in zip(after.degrees, after.capacity, strict=True)
    )
    mean_before = sum(r * x for r, x in zip(rho_before, before.epi, strict=True)) / sum(
        rho_before
    )
    mean_after = sum(r * x for r, x in zip(rho_after, after.epi, strict=True)) / sum(
        rho_after
    )
    mean_shift = sum(
        (new - old) * (x - mean_before)
        for old, new, x in zip(rho_before, rho_after, before.epi, strict=True)
    ) / sum(rho_after)
    assert mean_after - mean_before == mean_shift
    assert mean_shift != 0
    phase_mean_before = sum(
        r * angle for r, angle in zip(rho_before, before.phase, strict=True)
    ) / sum(rho_before)
    phase_mean_after = sum(
        r * angle for r, angle in zip(rho_after, after.phase, strict=True)
    ) / sum(rho_after)
    phase_mean_shift = sum(
        new_r * (new_angle - old_angle)
        + (new_r - old_r) * (old_angle - phase_mean_before)
        for old_r, new_r, old_angle, new_angle in zip(
            rho_before, rho_after, before.phase, after.phase, strict=True
        )
    ) / sum(rho_after)
    assert phase_mean_after - phase_mean_before == phase_mean_shift
    assert phase_mean_shift != 0


def test_deterministic_mode_retains_signed_jitter_before_refresh(
    successor, completed_transition
):
    after, _, raw, _ = completed_transition
    no_jitter = deepcopy(successor)
    no_jitter.graph["GLYPH_FACTORS"]["NAV_jitter"] = 0.0
    _, raw_zero, _ = _transition(no_jitter)
    assert _physical(no_jitter) == _physical(after)
    for node in PORTS:
        pressure = successor.nodes[node]["delta_nfr"]
        capacity = successor.nodes[node]["nu_f"]
        target = capacity if pressure >= 0 else -capacity
        base = 0.5 * pressure + 0.5 * target
        expected = 0.7 * (base + (COUPLING_FINE if base >= 0 else -COUPLING_FINE))
        assert raw[node] == expected
        assert raw_zero[node] == 0.7 * base
        assert abs(raw[node] - raw_zero[node]) == pytest.approx(0.7 * COUPLING_FINE)
        assert after.graph["_nav_transitions"][node]["dnfr_after"] == raw[node]
        assert raw[node] != after.nodes[node]["delta_nfr"]
        assert "_rng_jitter_progress" not in after.nodes[node]
        assert "_rng_jitter_progress" not in no_jitter.nodes[node]


def test_random_raw_pressure_and_stream_context_survive_only_as_execution_evidence(
    successor,
):
    first, second, repeat = (deepcopy(successor) for _ in range(3))
    for graph, seed in ((first, 37), (second, 73), (repeat, 37)):
        graph.graph.update(NAV_RANDOM=True, RANDOM_SEED=seed)
    _, raw_first, _ = _transition(first)
    _, raw_second, _ = _transition(second)
    _, raw_repeat, _ = _transition(repeat)
    assert raw_first == raw_repeat
    assert raw_first != raw_second
    assert _physical(first) == _physical(second) == _physical(repeat)
    assert first.graph["_nav_transitions"] == repeat.graph["_nav_transitions"]
    assert first.graph["_nav_transitions"] != second.graph["_nav_transitions"]
    for node in PORTS:
        progress = first.nodes[node]["_rng_jitter_progress"]
        assert progress["seed"] == 37
        assert progress["draws"] == 1
        assert progress == repeat.nodes[node]["_rng_jitter_progress"]
        assert second.nodes[node]["_rng_jitter_progress"]["seed"] == 73


def test_strict_handler_changes_raw_pressure_but_not_refreshed_physical_state(
    successor, completed_transition
):
    graph = deepcopy(successor)
    graph.graph["NAV_STRICT"] = True
    _, raw, _ = _transition(graph)
    after, _, relaxed_raw, _ = completed_transition
    assert raw != relaxed_raw
    assert _physical(graph) == _physical(after)
    for node in PORTS:
        assert raw[node] == 0.7 * (successor.nodes[node]["nu_f"] + COUPLING_FINE)


def test_two_percent_latency_drift_is_advisory_and_does_not_restore_form(
    successor, completed_transition
):
    graph = deepcopy(successor)
    current_form = require_finite_real_scalar_epi(graph.nodes[1]["EPI"])
    graph.nodes[1]["preserved_epi"] = current_form / 1.02
    with pytest.warns(UserWarning, match="EPI drift"):
        _transition(graph)
    after, *_ = completed_transition
    assert _physical(graph) == _physical(after)
    assert require_finite_real_scalar_epi(graph.nodes[1]["EPI"]) == current_form
    assert "preserved_epi" not in graph.nodes[1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("preserved_epi", float("nan")),
        ("latency_start_time", (TRANSITION_TIME + timedelta(seconds=1)).isoformat()),
        ("latent", 1),
    ],
)
def test_invalid_lifecycle_of_one_target_rejects_atomically_without_rng_progress(
    successor, key, value
):
    graph = deepcopy(successor)
    graph.graph["NAV_RANDOM"] = True
    graph.nodes[1][key] = value
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_cache = dict(graph.graph.get("_node_cache", {}))
    before_graph = deepcopy(
        {key: value for key, value in graph.graph.items() if key != "_node_cache"}
    )
    with pytest.raises(OperatorPreconditionError):
        _transition(graph)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert {
        key: value for key, value in graph.graph.items() if key != "_node_cache"
    } == before_graph
    assert graph.graph.get("_node_cache", {}) == before_cache
    assert "_nav_transitions" not in graph.graph
    assert all("_rng_jitter_progress" not in graph.nodes[node] for node in graph)


def test_admitted_nav_can_leave_the_old_broad_acute_sector():
    graph = _preparation(broad_counterexample=True)
    for left, right in graph.edges:
        gap = (graph.nodes[right]["theta"] - graph.nodes[left]["theta"] + pi) % tau - pi
        assert abs(gap) < pi / 2
    assert all(validate_candidate(graph, node, "NAV").allowed for node in PORTS)
    result, _, observation = _transition(graph)
    assert result.glyph == "NAV"
    assert observation.confined
    gap = (graph.nodes[0]["theta"] - graph.nodes[8]["theta"] + pi) % tau - pi
    assert gap > pi / 2 + 0.01
    assert all(node_debt(graph.nodes[node]) == 0 for node in PORTS)


@pytest.mark.parametrize("path", ["direct", "atomic"])
@pytest.mark.parametrize(
    "invalid_epi",
    [
        True,
        "0.3",
        Q(1, 10**400),
        BEPIElement((0.5, 0.6), (0.5, 0.5), (0.0, 1.0)),
    ],
)
def test_public_nav_rejects_invalid_form_before_executing_any_target(
    successor, path, invalid_epi
):
    graph = deepcopy(successor)
    graph.nodes[1].update(EPI=invalid_epi, latent=False)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_cache = dict(graph.graph.get("_node_cache", {}))
    before_graph = deepcopy(
        {key: value for key, value in graph.graph.items() if key != "_node_cache"}
    )
    with pytest.raises(TNFRValueError, match="EPI"):
        if path == "direct":
            Transition()(graph, 1)
        else:
            _transition(graph)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert {
        key: value for key, value in graph.graph.items() if key != "_node_cache"
    } == before_graph
    assert graph.graph.get("_node_cache", {}) == before_cache
