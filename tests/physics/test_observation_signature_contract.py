"""Admission and comparison boundaries for task-selected observations."""

import math
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.observability import (
    minimal_distinguishing_channels,
    observation_signature,
)


def state():
    graph = nx.path_graph(4)
    for node, (form, pressure, phase) in enumerate(
        ((0.0, -1.0, 0.0), (1.0, 0.5, 0.2), (-1.0, 0.5, 0.4), (0.5, 0.0, 0.6))
    ):
        graph.nodes[node].update(EPI=form, nu_f=1.0, delta_nfr=pressure, phase=phase)
    return graph


def observe(graph, **kwargs):
    return observation_signature(graph, tetrad_representation="full", **kwargs)


def test_form_selection_resolves_information_discarded_by_tetrad():
    left = state()
    right = deepcopy(left)
    right.nodes[0]["EPI"] = -2.0
    assert observe(left).tetrad == observe(right).tetrad
    before = observe(left, include_form=True)
    after = observe(right, include_form=True)
    assert after.form[0] == -2.0
    assert minimal_distinguishing_channels(before, after) == ("form",)
    assert before.xi_c_provenance.value == before.tetrad[-1]


@pytest.mark.parametrize(
    "key,flag",
    (
        ("nu_f", "include_capacity"),
        ("delta_nfr", "include_pressure"),
        ("EPI", "include_form"),
    ),
)
def test_missing_requested_primitive_is_not_an_observed_zero(key, flag):
    graph = state()
    del graph.nodes[0][key]
    with pytest.raises(ValueError, match="missing"):
        observe(graph, **{flag: True})


@pytest.mark.parametrize("value", (-1.0, True, "1", math.inf, Fraction(1, 10**400)))
def test_capacity_is_admitted_as_nonnegative_before_fields(value):
    graph = state()
    graph.nodes[0]["nu_f"] = value
    with pytest.raises((TypeError, ValueError)):
        observe(graph, include_capacity=True)


def test_zero_capacity_is_present_and_preserved():
    graph = state()
    graph.nodes[0]["nu_f"] = 0.0
    assert observe(graph, include_capacity=True).capacity[0] == 0.0


@pytest.mark.parametrize("value", (True, "1", complex(1, 2), Fraction(1, 10**400)))
def test_form_requires_signed_uniform_real_admission(value):
    graph = state()
    graph.nodes[0]["EPI"] = value
    with pytest.raises((TypeError, ValueError)):
        observe(graph, include_form=True)


@pytest.mark.parametrize(
    "flag", ("include_form", "include_capacity", "include_pressure", "include_history")
)
@pytest.mark.parametrize("value", ("false", 1, None))
def test_selection_flags_do_not_use_truthiness(flag, value):
    with pytest.raises(ValueError, match="boolean"):
        observe(state(), **{flag: value})


def test_comparison_requires_matching_representation_and_selection():
    full = observe(state())
    global_view = observation_signature(state(), tetrad_representation="global")
    with pytest.raises(ValueError, match="representation"):
        minimal_distinguishing_channels(full, global_view)
    with pytest.raises(ValueError, match="same observation channels"):
        minimal_distinguishing_channels(full, observe(state(), include_capacity=True))


def test_unavailable_nan_is_not_a_discriminating_value():
    left = observation_signature(nx.Graph(), tetrad_representation="global")
    right = observation_signature(nx.Graph(), tetrad_representation="global")
    assert all(math.isnan(value) for value in left.tetrad)
    assert minimal_distinguishing_channels(left, right) == ()


def test_comparison_checks_value_binding_and_method_policies():
    original = observe(state())
    with pytest.raises(ValueError, match="requires coherence provenance"):
        minimal_distinguishing_channels(
            original, replace(original, xi_c_provenance=None)
        )
    stale = replace(original.xi_c_provenance, value=original.tetrad[-1] + 1)
    with pytest.raises(ValueError, match="not bound"):
        minimal_distinguishing_channels(
            original, replace(original, xi_c_provenance=stale)
        )
    other = replace(
        original.xi_c_provenance, distance_weighting="another declared metric"
    )
    with pytest.raises(ValueError, match="compatible coherence"):
        minimal_distinguishing_channels(
            original, replace(original, xi_c_provenance=other)
        )
    other = replace(
        original.xi_c_provenance, method="autocorrelation_fit", fit_available=True
    )
    with pytest.raises(ValueError, match="compatible coherence"):
        minimal_distinguishing_channels(
            original, replace(original, xi_c_provenance=other)
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity", (-1.0, 1.0, 1.0, 1.0)),
        ("pressure", (True, 0.0, 0.0, 0.0)),
        ("form", (math.inf, 0.0, 0.0, 0.0)),
    ),
)
def test_comparison_readmits_carried_numeric_values(field, value):
    original = observe(
        state(), include_capacity=True, include_pressure=True, include_form=True
    )
    with pytest.raises((TypeError, ValueError)):
        minimal_distinguishing_channels(original, replace(original, **{field: value}))


def test_numpy_boolean_selection_and_unrequested_invalid_form():
    graph = state()
    graph.nodes[0]["EPI"] = complex(1, 2)
    assert observe(graph).form is None
    assert observe(graph, include_capacity=np.bool_(True)).capacity == (
        1.0,
        1.0,
        1.0,
        1.0,
    )


@pytest.mark.parametrize("value", (0.0, -1.0))
def test_unavailable_provenance_cannot_supply_a_finite_discriminator(value):
    original = observe(state())
    estimate = replace(
        original.xi_c_provenance, method="unavailable", fit_available=False, value=value
    )
    carried = replace(
        original, tetrad=original.tetrad[:-1] + (value,), xi_c_provenance=estimate
    )
    with pytest.raises(ValueError, match="method disagrees"):
        minimal_distinguishing_channels(carried, carried)
