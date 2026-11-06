"""Detached sine consumers calculate with admitted coefficient representations."""

from dataclasses import fields, is_dataclass
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_mediation import bound_relational_sine_mediation
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import certify_sine_cycle_recovery
from tnfr.physics.relational_sine_resonance import (
    assess_sine_pair_pulse,
    assess_sine_path_memory,
)
from tnfr.physics.relational_sine_scale import assess_sine_replica_scale


def _model(**values):
    model = RelationalExchangeModel(1, phase_domain="regular")
    for name, value in values.items():
        object.__setattr__(model, name, value)
    return model


def _graph(size=3, *, cycle=False):
    graph = nx.cycle_graph(size) if cycle else nx.path_graph(size)
    for node in graph:
        graph.nodes[node].update(EPI=node / 128, theta=node / 100, nu_f=1)
    return graph


def _pattern(model):
    return bound_relational_sine_pattern(
        _graph(5, cycle=True),
        reference_model=model,
        reference_node=0,
        form_error_bounds=(0,) * 5,
        phase_error_bounds=(0,) * 5,
    )


def _replica(model):
    pairs = tuple((2 * i, 2 * i + 1) for i in range(5))
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for i, pair in enumerate(pairs):
        for sign, node in zip((-1, 1), pair):
            graph.nodes[node].update(
                EPI=i / 128 + sign / 1024,
                theta=i / 100 + sign / 1024,
                nu_f=1,
            )
        graph.add_edges_from(
            (left, right) for left in pair for right in pairs[(i + 1) % 5]
        )
    return assess_sine_replica_scale(graph, reference_model=model, pairs=pairs)


def _evaluate(kind, model):
    if kind == "exchange":
        result = bound_relational_sine_exchange(_graph(), reference_model=model)
        return (
            result,
            result.phase_rate_numerators(),
            result.regional_transfer(region=(0,)),
            result.regional_storage_balance(region=(0,)),
            result.mediated_pressure(mediator=1),
        )
    if kind == "mediation":
        return bound_relational_sine_mediation(
            _graph(), reference_model=model, mediator=1
        )
    if kind == "pattern":
        return _pattern(model)
    if kind == "recovery":
        return certify_sine_cycle_recovery(
            _pattern(model), cycle=tuple(range(5)), winding=0, radius=Q(1, 8)
        )
    if kind == "pulse":
        return assess_sine_pair_pulse(_graph(2), reference_model=model)
    if kind == "memory":
        return assess_sine_path_memory(_graph(), reference_model=model, mediator=1)
    if kind == "replica":
        return _replica(model)
    raise AssertionError(kind)


def _same_normalized_result(actual, expected, raw_model, canonical_model):
    """Compare computed fields while retaining the original model association."""
    if isinstance(actual, RelationalExchangeModel):
        assert actual is raw_model
        assert expected is canonical_model
    elif is_dataclass(actual):
        assert type(actual) is type(expected)
        for field in fields(actual):
            _same_normalized_result(
                getattr(actual, field.name),
                getattr(expected, field.name),
                raw_model,
                canonical_model,
            )
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _same_normalized_result(left, right, raw_model, canonical_model)
    else:
        if isinstance(actual, Q):
            assert type(actual.numerator) is type(actual.denominator) is int
        assert actual == expected


@pytest.mark.parametrize(
    "kind,field,payload",
    [
        (kind, field, payload)
        for kind in (
            "exchange",
            "mediation",
            "pattern",
            "recovery",
            "pulse",
            "memory",
            "replica",
        )
        for field, payload in (
            ("storage_scale", np.int64(2**62)),
            ("storage_scale", Q(np.int64(2**62), np.int64(3))),
            ("phase_weight", Q(np.int64(2**62), np.int64(3))),
            ("epi_weight", Q(np.int64(2**62), np.int64(3))),
        )
        if field != "epi_weight" or kind not in ("memory", "replica")
    ],
)
def test_public_consumers_match_canonical_coefficients(kind, field, payload):
    # Stored coefficients are authoritative, with no constructor rescaling.
    values = {"epi_weight": Q(0) if kind in ("memory", "replica") else Q(1, 2)}
    canonical = Q(int(payload.numerator), int(payload.denominator))
    raw_model = _model(**(values | {field: payload}))
    canonical_model = _model(**(values | {field: canonical}))
    actual = _evaluate(kind, raw_model)
    expected = _evaluate(kind, canonical_model)
    _same_normalized_result(actual, expected, raw_model, canonical_model)


def test_consensus_phase_rates_and_loss_follow_independent_exact_formula():
    beta = 2**62
    model = _model(storage_scale=np.int64(beta))
    graph = _graph()
    for node, form in enumerate((1, 0, -1)):
        graph.nodes[node].update(EPI=form, theta=0)
    result = bound_relational_sine_exchange(graph, reference_model=model)
    assert result.phase_rate_numerators() == (Q(1, 2 * beta), Q(0), -Q(1, 2 * beta))
    assert result.continuous_loss == 1
    assert result.form_storage == 1
    assert result.phase_storage.lo == result.phase_storage.hi == 0
    assert result.storage.lo == result.storage.hi == 1
    assert result.balance_residual.lo == result.balance_residual.hi == 0


@pytest.mark.parametrize(
    "kind",
    ("exchange", "mediation", "pattern", "recovery", "pulse", "memory", "replica"),
)
@pytest.mark.parametrize("field", ("storage_scale", "epi_weight", "phase_weight"))
@pytest.mark.parametrize("value", (True, np.bool_(True)))
def test_reconstructed_boolean_coefficients_reject_before_calculation(
    kind, field, value
):
    values = {"epi_weight": 0 if kind in ("memory", "replica") else Q(1, 2)}
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        _evaluate(kind, _model(**values))
