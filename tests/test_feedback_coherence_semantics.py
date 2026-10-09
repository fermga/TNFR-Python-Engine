"""Feedback decisions must use the canonical local coherence channels."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from sys import float_info

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    SELF_ORGANIZATION,
)
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.dynamics import feedback
from tnfr.dynamics.feedback import StructuralFeedbackLoop
from tnfr.mathematics import backend
from tnfr.mathematics.epi import BEPIElement


def _loop(*, dnfr: float, depi: float) -> StructuralFeedbackLoop:
    graph = nx.Graph()
    graph.add_node(
        "n",
        EPI=1.0,
        nu_f=1.0,
        phase=0.0,
        delta_nfr=dnfr,
        dEPI_dt=depi,
    )
    return StructuralFeedbackLoop(graph, "n")


def test_feedback_local_coherence_uses_constitutive_kernel() -> None:
    assert _loop(dnfr=1.0, depi=0.0)._compute_local_coherence() == pytest.approx(0.5)
    assert _loop(dnfr=0.0, depi=0.5)._compute_local_coherence() == pytest.approx(
        2.0 / 3.0
    )


def test_recorded_rate_changes_feedback_operator_selection() -> None:
    static = _loop(dnfr=0.0, depi=0.0)
    driven = _loop(dnfr=0.0, depi=1.0)
    for loop in (static, driven):
        loop.target_coherence = 0.8
        loop.COHERENCE_TOL_LOW = 0.1
        loop.COHERENCE_TOL_HIGH = 0.1
        loop.DNFR_THRESHOLD = 10.0
        loop.EPI_THRESHOLD = 0.0

    assert static.regulate() == DISSONANCE
    assert driven.regulate() == COHERENCE


@pytest.mark.parametrize("pressure", [-0.2, 0.2])
def test_feedback_pressure_decision_is_sign_symmetric(pressure: float) -> None:
    loop = _loop(dnfr=pressure, depi=0.0)
    loop.target_coherence = 0.7
    loop.COHERENCE_TOL_LOW = 0.5
    loop.COHERENCE_TOL_HIGH = 0.5
    loop.DNFR_THRESHOLD = 0.1

    assert loop.regulate() == SELF_ORGANIZATION


def test_controller_does_not_initialize_an_unused_backend(monkeypatch):
    def unexpected_backend(*_args, **_kwargs):
        pytest.fail("feedback has no backend-dependent arithmetic")

    monkeypatch.setattr(backend, "get_backend", unexpected_backend)
    loop = _loop(dnfr=0.0, depi=0.0)
    assert loop.backend is None
    assert loop._use_optimizations is False


@pytest.mark.parametrize(
    "name, invalid",
    [
        ("target_coherence", True),
        ("target_coherence", Fraction(1) + Fraction(1, 2**60)),
        ("tau_adaptive", np.bool_(False)),
        ("learning_rate", "0.1"),
        ("coherence_tolerance_low", np.nan),
        ("coherence_tolerance_high", Fraction(1, 2**1075)),
        ("dnfr_threshold", np.inf),
        ("epi_threshold", "0.3"),
    ],
)
def test_constructor_rejects_invalid_raw_policy_values(name, invalid):
    with pytest.raises((TypeError, ValueError)):
        StructuralFeedbackLoop(nx.empty_graph(1), 0, **{name: invalid})


@pytest.mark.parametrize(
    "name",
    [
        "target_coherence",
        "tau_adaptive",
        "learning_rate",
        "coherence_tolerance_low",
        "coherence_tolerance_high",
        "dnfr_threshold",
    ],
)
def test_unsigned_controller_parameters_reject_negative_values(name):
    with pytest.raises(ValueError):
        StructuralFeedbackLoop(nx.empty_graph(1), 0, **{name: -0.25})


@pytest.mark.parametrize(
    "attribute",
    [
        "target_coherence",
        "tau_adaptive",
        "learning_rate",
        "COHERENCE_TOL_LOW",
        "COHERENCE_TOL_HIGH",
        "DNFR_THRESHOLD",
        "EPI_THRESHOLD",
    ],
)
def test_mutated_invalid_settings_fail_before_operator_selection_or_graph_write(
    monkeypatch, attribute
):
    loop = _loop(dnfr=0.0, depi=0.0)
    setattr(loop, attribute, "0.1")
    before = deepcopy((loop.G.graph, dict(loop.G.nodes(data=True))))

    def unexpected_operator(_name):
        pytest.fail("invalid policy must be admitted before operator lookup")

    monkeypatch.setattr(feedback, "get_operator_class", unexpected_operator)
    with pytest.raises((TypeError, ValueError)):
        loop.homeostatic_cycle(1)
    assert (loop.G.graph, dict(loop.G.nodes(data=True))) == before


@pytest.mark.parametrize(
    "raw_epi",
    [
        True,
        "0.0",
        Fraction(1, 2**1075),
        BEPIElement((1.0, 2.0), (1.0, 2.0), (0.0, 1.0)),
    ],
)
def test_feedback_never_scalarizes_invalid_or_rich_form_to_choose_an_operator(raw_epi):
    loop = _loop(dnfr=0.0, depi=0.0)
    loop.G.nodes["n"][ALIAS_EPI[0]] = raw_epi
    loop.G.nodes["n"][ALIAS_EPI[1]] = 0.0
    with pytest.raises((TypeError, ValueError)):
        loop.regulate()


def test_signed_uniform_embedding_and_signed_form_threshold_keep_emission_semantics():
    loop = _loop(dnfr=0.0, depi=0.0)
    loop.target_coherence = 1.0
    loop.EPI_THRESHOLD = -0.5
    loop.G.nodes["n"][ALIAS_EPI[0]] = BEPIElement(
        (-0.75, -0.75), (-0.75, -0.75), (0.0, 1.0)
    )
    assert loop.regulate() == EMISSION


@pytest.mark.parametrize("invalid", [np.nan, True, "0.5", Fraction(1, 2**1075)])
def test_invalid_adaptation_input_cannot_become_a_successful_clamped_threshold(invalid):
    loop = _loop(dnfr=0.0, depi=0.0)
    before = loop.tau_adaptive
    with pytest.raises((TypeError, ValueError)):
        loop.adapt_thresholds(invalid)
    assert loop.tau_adaptive == before


def test_unrepresentable_update_rejects_before_threshold_assignment():
    loop = _loop(dnfr=0.0, depi=0.0)
    loop.learning_rate = float_info.max
    before = loop.tau_adaptive
    with pytest.raises(ValueError, match="finite"):
        loop.adapt_thresholds(-float_info.max)
    assert loop.tau_adaptive == before


@pytest.mark.parametrize("performance, expected", [(-1.0, 0.234375), (2.0, 0.05)])
def test_generic_finite_performance_keeps_proportional_update_and_policy_clamp(
    performance, expected
):
    loop = _loop(dnfr=0.0, depi=0.0)
    loop.target_coherence, loop.learning_rate, loop.tau_adaptive = 0.75, 0.0625, 0.125
    loop.adapt_thresholds(performance)
    assert loop.tau_adaptive == expected


@pytest.mark.parametrize("count", [-1, True, np.bool_(False), 1.5, "1"])
def test_invalid_cycle_count_cannot_execute_an_operator(monkeypatch, count):
    loop = _loop(dnfr=0.0, depi=0.0)

    def unexpected_operator(_name):
        pytest.fail("invalid count must not execute an operator")

    monkeypatch.setattr(feedback, "get_operator_class", unexpected_operator)
    with pytest.raises((TypeError, ValueError)):
        loop.homeostatic_cycle(count)


def test_cycle_reads_each_endpoint_once_and_adapts_from_the_actual_new_state(
    monkeypatch,
):
    loop = _loop(dnfr=1.0, depi=0.0)
    loop.target_coherence, loop.learning_rate, loop.tau_adaptive = 0.75, 0.0625, 0.125
    reads, actions = [], []
    original = loop._compute_local_coherence

    def observe():
        result = original()
        reads.append(result)
        return result

    class SuppliedOperator:
        def __call__(self, graph, node, *, tau):
            actions.append((graph, node, tau))
            graph.nodes[node][ALIAS_DNFR[0]] = 0.5

    def lookup(name):
        assert name == COHERENCE
        return SuppliedOperator

    monkeypatch.setattr(loop, "_compute_local_coherence", observe)
    monkeypatch.setattr(feedback, "get_operator_class", lookup)
    loop.homeostatic_cycle(0)
    assert reads == [] and actions == []
    loop.homeostatic_cycle(np.int64(1))
    assert reads == [0.5, 2 / 3]
    assert actions == [(loop.G, "n", 0.125)]
    assert loop.tau_adaptive == 0.125 + 0.0625 * (0.75 - 2 / 3)


def test_real_feedback_cycle_uses_registered_coherence_and_its_observed_endpoint():
    loop = _loop(dnfr=1.0, depi=0.0)
    loop.target_coherence, loop.learning_rate, loop.tau_adaptive = 0.75, 0.0625, 0.125
    assert loop.regulate() == COHERENCE
    loop.homeostatic_cycle(1)
    pressure = get_attr(loop.G.nodes["n"], ALIAS_DNFR)
    assert 0.0 < pressure < 1.0
    assert get_attr(loop.G.nodes["n"], ALIAS_EPI) == 1.0
    endpoint_coherence = 1 / (1 + pressure)
    assert loop.tau_adaptive == pytest.approx(
        0.125 + 0.0625 * (0.75 - endpoint_coherence), abs=1e-15
    )
