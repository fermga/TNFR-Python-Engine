"""Independent static controls for the declared conservative support law.

Freeze a four-node cycle plus diagonal, with weights (1/2, 2, 0, 3/4,
3/2), forms (1/4,-1/2,3/4,1/8), phases (0,1/4,-3/8,1/2), capacities
(1/2,1,3/2,2), beta=2 and zero loss before evaluating the new reader.
These are field/work/admission controls, not an evolution or formation run.
"""

import json
import pickle
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import (
    SineExchangeComparison,
    bound_relational_sine_exchange,
)
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.sdk import relational_report_to_dict

_MODEL = RelationalExchangeModel(2, epi_weight=0, phase_domain="regular")
_EDGES = ((0, 1, Q(1, 2)), (1, 2, Q(2)), (2, 3, Q(0)), (3, 0, Q(3, 4)), (0, 2, Q(3, 2)))
_ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _graph(*, unit=False):
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    for left, right, weight in _EDGES:
        graph.add_edge(left, right, weight=1 if unit else weight, length=17)
    for node, form, phase, capacity in zip(
        graph,
        (Q(1, 4), Q(-1, 2), Q(3, 4), Q(1, 8)),
        (Q(0), Q(1, 4), Q(-3, 8), Q(1, 2)),
        (Q(1, 2), Q(1), Q(3, 2), Q(2)),
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.graph.update(GAMMA={"type": "none"}, untouched={"list": [1, 2]})
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("a detached support-law reader must not advance any trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def sample():
    return bound_relational_sine_support_exchange(_graph(), reference_model=_MODEL)


def test_independent_weighted_form_unweighted_phase_field_and_balance(sample, mp):
    graph = _graph()
    before = _snapshot(graph)
    report = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    assert _snapshot(graph) == before
    assert report == sample
    assert report.degrees == (3, 2, 3, 2)
    gradient = [Q(0)] * 4
    current = [mp.mpf(0)] * 4
    form_store, phase_store = Q(0), mp.mpf(0)
    for left, right, weight in _EDGES:
        difference = sample.epi[left] - sample.epi[right]
        sine = mp.sin(_mp(mp, sample.phase[right] - sample.phase[left]))
        gradient[left] += weight * difference
        gradient[right] -= weight * difference
        current[left] += sine
        current[right] -= sine
        form_store += weight * difference**2 / 2
        phase_store += 1 - mp.cos(_mp(mp, sample.phase[right] - sample.phase[left]))
    assert sample.form_gradient == tuple(gradient)
    assert sample.form_storage == form_store
    _contains(mp, sample.phase_storage, phase_store)
    _contains(mp, sample.storage, _mp(mp, form_store) + 2 * phase_store)
    expected_numerators = tuple(
        nu * q / (2 * degree)
        for nu, q, degree in zip(sample.capacity, gradient, sample.degrees)
    )
    assert sample.phase_rate_numerators() == expected_numerators
    for index, (capacity, degree) in enumerate(zip(sample.capacity, sample.degrees)):
        pressure = current[index] / (mp.pi * degree)
        form_rate = _mp(mp, capacity) * pressure
        phase_rate = _mp(mp, expected_numerators[index]) / mp.pi
        form_work = _mp(mp, gradient[index]) * form_rate
        phase_work = -2 * current[index] * phase_rate
        for bounds, value in (
            (sample.pressure[index], pressure),
            (sample.form_rates[index], form_rate),
            (sample.phase_rates[index], phase_rate),
            (sample.form_work[index], form_work),
            (sample.phase_work[index], phase_work),
        ):
            _contains(mp, bounds, value)
        assert abs(form_work + phase_work) < mp.mpf("1e-95")
        assert sample.node_balance_residual[index].contains(0)
    assert sample.continuous_loss == 0
    assert sample.dissipation == (Q(0),) * 4
    assert sample.storage_rate.contains(0)
    assert sample.balance_residual.contains(0)


def test_unit_support_reduces_to_existing_complete_law_with_heterogeneous_capacity():
    graph = _graph(unit=True)
    actual = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    old = bound_relational_sine_exchange(graph, reference_model=_MODEL)
    assert actual.law != old.law
    for field in (
        "nodes",
        "edges",
        "epi",
        "phase",
        "capacity",
        "degrees",
        "form_gradient",
        "relative_resultant",
        "pressure",
        "form_rates",
        "phase_rates",
        "form_storage",
        "phase_storage",
        "storage",
        "continuous_loss",
        "form_work",
        "phase_work",
        "node_balance_residual",
        "balance_residual",
    ):
        assert getattr(actual, field) == getattr(old, field), field
    assert actual.conductances == (Q(1),) * len(actual.edges)


def test_zero_weight_link_retains_phase_incidence_and_bare_degree(sample, mp):
    removed = _graph()
    removed.remove_edge(2, 3)
    absent = bound_relational_sine_support_exchange(removed, reference_model=_MODEL)
    assert absent.form_gradient == sample.form_gradient
    assert absent.form_storage == sample.form_storage
    assert absent.degrees == (3, 2, 2, 1)
    lost_phase_cost = 1 - mp.cos(_mp(mp, sample.phase[3] - sample.phase[2]))
    _contains(mp, sample.phase_storage - absent.phase_storage, lost_phase_cost)
    assert absent.pressure[2] != sample.pressure[2]
    assert (
        absent.phase_rate_numerators()[2] == Q(3, 2) * sample.phase_rate_numerators()[2]
    )


def test_zero_conductance_component_still_has_bare_phase_support():
    graph = _graph()
    nx.set_edge_attributes(graph, 0, "weight")
    report = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    assert report.degrees == (3, 2, 3, 2)
    assert report.form_gradient == (Q(0),) * 4
    assert report.form_storage == 0
    assert report.phase_rate_numerators() == (Q(0),) * 4
    assert report.phase_rates == (I(0),) * 4
    assert report.phase_storage.lo > 0
    assert any(not rate.contains(0) for rate in report.form_rates)
    assert report.storage_rate == I(0)


@pytest.mark.parametrize(
    "invalid",
    (
        True,
        False,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        Q(1, 10**500),
        Q(-1, 10**500),
        Decimal("1e-500"),
    ),
)
def test_invalid_or_unrepresentable_edge_weights_reject_without_writes(invalid):
    graph = _graph()
    graph.edges[0, 1]["weight"] = invalid
    before = _snapshot(graph)
    with pytest.raises(_ADMISSION_ERRORS):
        bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    "invalid_model",
    (
        RelationalExchangeModel(2, epi_weight=Q(1, 8), phase_domain="regular"),
        RelationalExchangeModel(2, epi_weight=0, phase_domain="acute"),
    ),
)
def test_other_loss_or_phase_domain_is_not_silently_reinterpreted(invalid_model):
    with pytest.raises(_ADMISSION_ERRORS):
        bound_relational_sine_support_exchange(_graph(), reference_model=invalid_model)


def test_live_gamma_registry_is_checked_without_calling_an_external_source(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the detached unforced reader must not invoke supplied Gamma")

    graph = _graph()
    before = _snapshot(graph)
    monkeypatch.setitem(GAMMA_REGISTRY, "none", GammaEntry(forbidden, False))
    with pytest.raises(_ADMISSION_ERRORS):
        bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    assert _snapshot(graph) == before


def test_primitive_changes_rebuild_consumed_rows_and_export(sample):
    graph = _graph()
    graph.edges[0, 1]["weight"] = Q(3, 4)
    rebuilt = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    index = sample.edges.index((0, 1))
    weights = list(sample.conductances)
    weights[index] = Q(3, 4)
    changed = replace(
        sample,
        conductances=tuple(weights),
        degrees=(99,) * 4,
        form_gradient=(Q(999),) * 4,
        pressure=(I(999),) * 4,
        form_rates=(I(999),) * 4,
        phase_rates=(I(999),) * 4,
        storage=I(999),
        continuous_loss=Q(999),
        balance_residual=I(999),
    )
    assert changed.phase_rate_numerators() == rebuilt.phase_rate_numerators()
    assert changed.to_dict() == rebuilt.to_dict()
    payload = relational_report_to_dict(changed)
    assert payload["report"] == rebuilt.to_dict()["report"]
    assert payload["report_type"] == "SineSupportExchangeComparison"
    assert rebuilt.to_dict()["schema"] == "tnfr.relational-sine-support-comparison.v1"
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "change",
    (
        {"epi": (True, Q(0), Q(0), Q(0))},
        {"conductances": (True, Q(1), Q(1), Q(1), Q(1))},
        {"conductances": (Q(-1), Q(1), Q(1), Q(1), Q(1))},
        {"capacity": (Q(-1), Q(1), Q(1), Q(1))},
        {"edges": ((0, 0), (0, 1), (0, 2), (1, 2), (2, 3))},
        {"law": "normalized_sine_reciprocal_exchange"},
    ),
)
def test_invalid_primitive_reports_cannot_export_or_calculate(sample, change):
    changed = replace(sample, **change)
    for consume in (changed.phase_rate_numerators, changed.to_dict):
        with pytest.raises(_ADMISSION_ERRORS):
            consume()


def test_old_unit_comparison_consumers_reject_distinct_support_report(sample):
    for consume in (
        SineExchangeComparison.phase_rate_numerators,
        SineExchangeComparison.resultant_kinematics,
    ):
        with pytest.raises(TypeError):
            consume(sample)
    with pytest.raises(ValueError, match="unit conductances"):
        bound_relational_sine_exchange(_graph(), reference_model=_MODEL)
