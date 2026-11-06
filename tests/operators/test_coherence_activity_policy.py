"""Coherence policy admission is distinct from its event and grammar domains.

The fixed generic P2 controls use signed or zero form, zero capacity, stored
pressure one and phase gap pi/4. They are not the organized-source preparation.
No continuous response or archived research event is evaluated. Existing Jacobi
tests own phase scheduling, callback rollback and deferred warning emission.
"""

import math
import warnings
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.constants.canonical import COHERENCE_RETENTION
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.node import NodeNX
from tnfr.operators import _op_IL, apply_glyph
from tnfr.operators._coherence_stage_kernel import (
    DEFAULT_PHASE_LOCKING_COEFFICIENT,
    propose_coherence_stage,
)
from tnfr.operators.definitions import Coherence
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.network_stage import TWO_PHASE_JACOBI, execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.coherence import (
    coherence_precondition_warnings,
    diagnose_coherence_readiness,
    validate_coherence_strict,
)
from tnfr.types import ensure_bepi, require_finite_real_scalar_epi, serialize_bepi


def _graph(forms=(0.25, 0.25), capacity=1.0, pressure=0.25):
    graph = nx.path_graph(2)
    graph.graph.update(
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
    )
    for node, form in enumerate(forms):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: form,
                ALIAS_VF[0]: capacity,
                ALIAS_DNFR[0]: pressure,
                ALIAS_THETA[0]: node * math.pi / 4,
                "epi_kind": "supplied-p2",
                "glyph_history": ["AL"],
            }
        )
    return graph


def _state(graph):
    return (
        deepcopy(
            {key: value for key, value in graph.graph.items() if key != "_node_cache"}
        ),
        deepcopy(dict(graph.nodes(data=True))),
        deepcopy(list(graph.edges(data=True))),
        hasattr(graph, "_last_operator_applied"),
        getattr(graph, "_last_operator_applied", None),
        "_node_cache" in graph.graph,
        tuple(
            (node, id(adapter))
            for node, adapter in graph.graph.get("_node_cache", {}).items()
        ),
    )


def _primary(graph, node):
    data = graph.nodes[node]
    return (
        require_finite_real_scalar_epi(
            get_attr(data, ALIAS_EPI, strict=True, conv=lambda value: value)
        ),
        get_attr(data, ALIAS_VF),
        get_attr(data, ALIAS_DNFR),
        get_attr(data, ALIAS_THETA),
    )


def _strict(graph, node):
    return validate_coherence_strict(graph, node, emit_warnings=False)


READERS = (_strict, diagnose_coherence_readiness, coherence_precondition_warnings)


@pytest.mark.parametrize("aliases", (ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA))
@pytest.mark.parametrize(
    "invalid", (True, math.nan, math.inf, "0.25", Fraction(1, 10**400))
)
def test_all_readers_reject_invalid_authoritative_state_without_alias_fallback(
    aliases, invalid
):
    graph = _graph()
    graph.nodes[0][aliases[0]] = invalid
    graph.nodes[0][aliases[1]] = 0.25
    before = _state(graph)

    for reader in READERS:
        with pytest.raises((TNFRValueError, OperatorPreconditionError)):
            reader(graph, 0)

    assert _state(graph) == before


@pytest.mark.parametrize("stored", (False, True))
def test_nonuniform_form_is_not_projected_to_a_ready_magnitude(stored):
    rich = BEPIElement((0.25, 0.5), (0.25, 0.25), (0.0, 1.0))
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = serialize_bepi(rich) if stored else rich
    before = _state(graph)
    for reader in READERS:
        with pytest.raises((TypeError, ValueError), match="scalar EPI|uniform-real"):
            reader(graph, 0)
    assert _state(graph) == before


@pytest.mark.parametrize("key", ("min_epi", "min_vf", "dnfr_critical_threshold"))
@pytest.mark.parametrize("invalid", (True, math.nan, "0.1", Fraction(1, 10**400)))
def test_thresholds_cannot_fail_open_or_disagree_between_readers(key, invalid):
    graph = _graph()
    graph.graph["IL_PRECONDITIONS"] = {key: invalid}
    before = _state(graph)
    for reader in READERS:
        with pytest.raises(OperatorPreconditionError, match=key):
            reader(graph, 0)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "policy",
    (
        None,
        [],
        {"activity_policy": True},
        {"activity_policy": "EVENT_DOMAIN"},
        {"min_vf": -1.0},
        {"dnfr_critical_threshold": -1.0},
        {"warn_zero_dnfr": "false"},
        {"warn_isolated": 1},
    ),
)
def test_policy_shape_selector_domains_and_warning_flags_are_admitted(policy):
    graph = _graph()
    graph.graph["IL_PRECONDITIONS"] = policy
    before = _state(graph)
    for reader in READERS:
        with pytest.raises(OperatorPreconditionError):
            reader(graph, 0)
    assert _state(graph) == before


@pytest.mark.parametrize("key", ("min_epi", "min_vf"))
def test_event_domain_rejects_conflicting_thresholds_even_when_they_are_zero(key):
    graph = _graph()
    graph.graph["IL_PRECONDITIONS"] = {"activity_policy": "event_domain", key: 0.0}
    for reader in READERS:
        with pytest.raises(OperatorPreconditionError, match=key):
            reader(graph, 0)


@pytest.mark.parametrize("representation", (float, ensure_bepi, serialize_bepi))
@pytest.mark.parametrize("form", (-0.25, 0.0, 0.25))
def test_event_domain_preserves_signed_scalar_representations_and_zero_capacity(
    representation, form
):
    graph = _graph(forms=(representation(form), 0.25), capacity=0.0)
    graph.graph["IL_PRECONDITIONS"] = {"activity_policy": "event_domain"}
    before = _state(graph)
    _strict(graph, 0)
    report = diagnose_coherence_readiness(graph, 0)
    assert report["activity_policy"] == "event_domain"
    assert report["ready"] is True
    assert report["checks"]["epi_active"] and report["checks"]["vf_active"]
    assert report["values"]["epi"] == form
    assert report["values"]["vf"] == 0.0
    assert _state(graph) == before


@pytest.mark.parametrize("explicit", (False, True))
@pytest.mark.parametrize("form, ready", ((-0.25, False), (0.0, False), (0.25, True)))
def test_default_signed_threshold_policy_is_preserved(explicit, form, ready):
    graph = _graph(forms=(form, 0.25))
    if explicit:
        graph.graph["IL_PRECONDITIONS"] = {"activity_policy": "signed_threshold"}
    before = _state(graph)
    report = diagnose_coherence_readiness(graph, 0)
    assert report["activity_policy"] == "signed_threshold"
    assert report["ready"] is ready
    if ready:
        _strict(graph, 0)
    else:
        with pytest.raises(ValueError):
            _strict(graph, 0)
    assert _state(graph) == before


def test_configured_signed_bounds_and_pressure_warning_equality_are_unchanged():
    graph = _graph(forms=(-0.25, 0.25), capacity=0.5, pressure=-0.25)
    graph.graph["IL_PRECONDITIONS"] = {
        "min_epi": -0.5,
        "min_vf": 0.25,
        "dnfr_critical_threshold": 0.25,
        "max_epi": -1.0,  # Deprecated, not an EPI headroom requirement.
    }
    _strict(graph, 0)
    assert diagnose_coherence_readiness(graph, 0)["ready"] is True
    assert coherence_precondition_warnings(graph, 0) == ()
    for key, equality in (("min_epi", -0.25), ("min_vf", 0.5)):
        graph.graph["IL_PRECONDITIONS"][key] = equality
        assert diagnose_coherence_readiness(graph, 0)["ready"] is False
        with pytest.raises(ValueError):
            _strict(graph, 0)
        graph.graph["IL_PRECONDITIONS"][key] = -0.5 if key == "min_epi" else 0.25


@pytest.mark.parametrize("policy", ("signed_threshold", "event_domain"))
def test_nonnegative_capacity_domain_cannot_be_replaced_by_activity_policy(policy):
    graph = _graph(capacity=-0.25)
    graph.graph["IL_PRECONDITIONS"] = {"activity_policy": policy}
    for reader in READERS:
        with pytest.raises(OperatorPreconditionError, match="nu_f"):
            reader(graph, 0)


@pytest.mark.parametrize("staged", (False, True))
@pytest.mark.parametrize("forms", ((0.25, -0.25), (0.0, 0.0)))
def test_explicit_event_domain_reaches_the_shared_pressure_and_phase_maps(
    staged, forms
):
    graph = _graph(forms=forms, capacity=0.0, pressure=1.0)
    graph.graph["IL_PRECONDITIONS"] = {
        "activity_policy": "event_domain",
        "dnfr_critical_threshold": 2.0,
    }
    coefficient = DEFAULT_PHASE_LOCKING_COEFFICIENT
    expected_phase = (coefficient * math.pi / 4, (1 - coefficient) * math.pi / 4)
    before = {node: _primary(graph, node) for node in graph}

    if staged:
        result = execute_pointwise_stage(graph, Coherence(), (0, 1))
        assert result.schedule == TWO_PHASE_JACOBI
    else:
        Coherence()(graph, 0)

    targets = (0, 1) if staged else (0,)
    for node in targets:
        assert _primary(graph, node) == pytest.approx(
            (forms[node], 0.0, COHERENCE_RETENTION, expected_phase[node])
        )
        assert list(graph.nodes[node]["glyph_history"]) == ["AL", "IL"]
        assert graph.nodes[node]["epi_kind"] == "supplied-p2"
    if not staged:
        assert _primary(graph, 1) == before[1]
    assert len(graph.graph["IL_dnfr_reductions"]) == len(targets)
    assert len(graph.graph["IL_phase_locking"]) == len(targets)


def test_positive_legacy_readiness_does_not_predict_nonzero_primary_action():
    graph = _graph(pressure=0.0)
    graph.graph["IL_PRECONDITIONS"] = {"warn_zero_dnfr": False}
    for node in graph:
        graph.nodes[node][ALIAS_THETA[0]] = 0.25
    before = {node: _primary(graph, node) for node in graph}
    assert diagnose_coherence_readiness(graph, 0)["ready"] is True

    execute_pointwise_stage(graph, Coherence(), (0, 1))

    assert {node: _primary(graph, node) for node in graph} == before
    assert all(list(graph.nodes[node]["glyph_history"])[-1] == "IL" for node in graph)


@pytest.mark.parametrize("failure", ("state", "grammar"))
def test_later_target_failure_keeps_the_complete_stage_uncommitted(failure):
    graph = _graph(forms=(0.0, 0.0), capacity=0.0, pressure=0.0)
    graph.graph["IL_PRECONDITIONS"] = {"activity_policy": "event_domain"}
    if failure == "state":
        graph.nodes[1][ALIAS_DNFR[0]] = True
        expected_error = OperatorPreconditionError
    else:
        graph.nodes[1]["glyph_history"] = []
        expected_error = StructuralGrammarError
        assert diagnose_coherence_readiness(graph, 1)["ready"] is True
    before = _state(graph)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(expected_error):
            execute_pointwise_stage(graph, Coherence(), (0, 1))
    assert caught == []
    assert _state(graph) == before


@pytest.mark.parametrize(
    "graph_options, kwargs",
    (
        ({}, {"validate_preconditions": 1}),
        ({}, {"validate_preconditions": "yes"}),
        ({}, {"collect_metrics": "yes"}),
        ({"VALIDATE_OPERATOR_PRECONDITIONS": "yes"}, {}),
    ),
)
@pytest.mark.parametrize("staged", (False, True))
def test_common_flags_reject_before_any_warning_or_execution(
    graph_options, kwargs, staged
):
    graph = _graph(pressure=0.0)
    graph.graph.update(graph_options)
    before = _state(graph)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(OperatorPreconditionError, match="boolean"):
            if staged:
                execute_pointwise_stage(graph, Coherence(), (0, 1), **kwargs)
            else:
                Coherence()(graph, 0, **kwargs)
    assert caught == []
    assert _state(graph) == before


@pytest.mark.parametrize("emit_warnings", (1, "false", None))
def test_direct_warning_emission_toggle_is_strictly_boolean(emit_warnings):
    graph = _graph(pressure=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(OperatorPreconditionError, match="emit_warnings"):
            validate_coherence_strict(graph, 0, emit_warnings=emit_warnings)
    assert caught == []


@pytest.mark.parametrize(
    "graph_enabled, kwargs", ((False, {}), (True, {"validate_preconditions": False}))
)
def test_disabled_optional_gate_does_not_consume_invalid_policy(graph_enabled, kwargs):
    graph = _graph(forms=(-0.25, 0.25), capacity=0.0)
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=graph_enabled,
        IL_PRECONDITIONS={"activity_policy": "unrecognized"},
    )
    Coherence()(graph, 0, **kwargs)
    assert _primary(graph, 0)[2] == pytest.approx(0.25 * COHERENCE_RETENTION)


@pytest.mark.parametrize("route", ("glyph", "adapter", "protocol", "proposal"))
@pytest.mark.parametrize("invalid", (True, "0.5", Fraction(1, 10**400)))
def test_pressure_consumers_admit_raw_values_before_conversion_or_cache(route, invalid):
    graph = _graph()
    graph.nodes[0][ALIAS_DNFR[0]] = invalid
    graph.nodes[0][ALIAS_DNFR[1]] = 0.25
    adapter = NodeNX(graph, 0) if route == "adapter" else None
    protocol = SimpleNamespace(dnfr=invalid)
    before = _state(graph)
    error = (
        TNFRValueError
        if route in ("adapter", "protocol")
        else OperatorPreconditionError
    )
    with pytest.raises(error, match="DeltaNFR"):
        if route == "glyph":
            apply_glyph(graph, 0, "IL")
        elif route == "adapter":
            _op_IL(adapter, {})
        elif route == "protocol":
            _op_IL(protocol, {})
        else:
            propose_coherence_stage(graph, 0, COHERENCE_RETENTION)
    assert protocol.dnfr is invalid
    assert _state(graph) == before


def test_low_level_pressure_glyph_does_not_acquire_public_activity_or_phase_policy():
    graph = _graph(forms=(-0.25, 0.25), capacity=0.0, pressure=0.5)
    before = _primary(graph, 0)
    with pytest.raises(ValueError):
        _strict(graph, 0)

    apply_glyph(graph, 0, "IL")

    assert _primary(graph, 0) == (
        before[0],
        before[1],
        0.5 * COHERENCE_RETENTION,
        before[3],
    )
    assert "IL_phase_locking" not in graph.graph
