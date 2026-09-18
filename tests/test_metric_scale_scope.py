"""Current-state normalization and the declared numerical scale of metrics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR, ALIAS_VF
from tnfr.metrics import sense_index
from tnfr.metrics.common import (
    _get_vf_dnfr_max,
    compute_coherence,
    is_structural_equilibrium,
    structural_coherence,
)


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("channel", ["capacity", "pressure"])
def test_si_normalization_uses_live_aliases_despite_stale_maxima(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool, channel: str
) -> None:
    if not use_numpy:
        monkeypatch.setattr(sense_index, "np", None)
    graph = nx.path_graph(2)
    for node, magnitude in enumerate((0.5, 1.0)):
        graph.nodes[node].update(
            {ALIAS_VF[0]: magnitude, ALIAS_DNFR[0]: magnitude, "phase": 0.0}
        )
    graph.graph.update(_vfmax=2.0, _dnfrmax=2.0)
    graph.graph["SI_WEIGHTS"] = {
        "alpha": float(channel == "capacity"),
        "beta": 0.0,
        "gamma": float(channel == "pressure"),
    }
    expected = {0: 0.5, 1: 1.0} if channel == "capacity" else {0: 0.5, 1: 0.0}

    assert sense_index.compute_Si(graph, inplace=False) == pytest.approx(expected)
    assert graph.graph["_vfmax"] == graph.graph["_dnfrmax"] == 1.0

    # A direct write bypasses the setter's incremental max-cache maintenance.
    graph.nodes[1][ALIAS_VF[0]] = 2.0
    graph.nodes[1][ALIAS_DNFR[0]] = 2.0
    expected[0] = 0.25 if channel == "capacity" else 0.75
    assert sense_index.compute_Si(graph, inplace=False) == pytest.approx(expected)
    assert graph.graph["_vfmax"] == graph.graph["_dnfrmax"] == 2.0


@pytest.mark.parametrize("node_count", [0, 2])
def test_zero_maxima_are_cached_as_zero_but_have_safe_divisors(
    node_count: int,
) -> None:
    graph = nx.empty_graph(node_count)
    graph.graph.update(_vfmax=3.0, _dnfrmax=4.0)

    assert _get_vf_dnfr_max(graph) == (1.0, 1.0)
    assert graph.graph["_vfmax"] == graph.graph["_dnfrmax"] == 0.0


@pytest.mark.parametrize("use_numpy", [True, False])
def test_uniform_capacity_unit_change_preserves_relative_si(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool
) -> None:
    if not use_numpy:
        monkeypatch.setattr(sense_index, "np", None)
    graph = nx.path_graph(2)
    for node, capacity in enumerate((0.5, 1.0)):
        graph.nodes[node].update(
            {ALIAS_VF[0]: capacity, ALIAS_DNFR[0]: 0.25, "phase": 0.0}
        )
    before = sense_index.compute_Si(graph, inplace=False)

    # This holds pressure fixed. It does not certify covariance of a pressure
    # law that itself contains an untransformed capacity-gradient coefficient.
    for _, data in graph.nodes(data=True):
        data[ALIAS_VF[0]] /= 4.0

    assert sense_index.compute_Si(graph, inplace=False) == pytest.approx(before)


def test_coherence_scale_depends_on_time_coordinate_and_stored_rate() -> None:
    graph = nx.empty_graph(1)
    graph.nodes[0].update({ALIAS_VF[0]: 2.0, ALIAS_DNFR[0]: 0.5, ALIAS_DEPI[0]: 1.0})
    assert compute_coherence(graph) == pytest.approx(2.0 / 5.0)

    # The same nodal curve in t_new=2*t has nu_new=nu/2 and rate_new=rate/2.
    # The chosen reciprocal diagnostic is not invariant to that unit change.
    graph.nodes[0][ALIAS_VF[0]] = 1.0
    graph.nodes[0][ALIAS_DEPI[0]] = 0.5
    assert compute_coherence(graph) == pytest.approx(1.0 / 2.0)

    # Reading coherence does not silently replace a recorded rate with nu*p.
    graph.nodes[0][ALIAS_VF[0]] = 0.0
    assert compute_coherence(graph) == pytest.approx(1.0 / 2.0)
    assert not is_structural_equilibrium(0.5, 0.0)


def test_network_coherence_differs_from_nodal_mean_and_projected_parent() -> None:
    graph = nx.empty_graph(2)
    for node, pressure in enumerate((1.0, -3.0)):
        graph.nodes[node].update({ALIAS_DNFR[0]: pressure, ALIAS_DEPI[0]: 0.0})

    network = compute_coherence(graph)
    nodal_mean = (structural_coherence(1.0) + structural_coherence(-3.0)) / 2.0
    projected_parent = structural_coherence((1.0 - 3.0) / 2.0)

    assert network == pytest.approx(1.0 / 3.0)
    assert nodal_mean == pytest.approx(3.0 / 8.0)
    assert projected_parent == pytest.approx(1.0 / 2.0)
    assert network < nodal_mean < projected_parent


@pytest.mark.parametrize("aliases", [ALIAS_DNFR, ALIAS_DEPI])
@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        "1",
        None,
        complex(1.0, 0.0),
        np.complex64(1.0),
        np.complex128(1.0),
    ],
)
def test_primary_coherence_rejects_invalid_authoritative_alias(
    aliases: tuple[str, ...], value: object
) -> None:
    graph = nx.empty_graph(1)
    graph.nodes[0].update({aliases[0]: value, aliases[1]: 0.0})

    # Neither a valid later alias nor the zero default may hide bad telemetry.
    with pytest.raises((TypeError, ValueError)):
        compute_coherence(graph)


def test_primary_coherence_preserves_missing_and_legacy_alias_conventions() -> None:
    graph = nx.empty_graph(1)
    assert compute_coherence(graph, return_means=True) == (1.0, 0.0, 0.0)

    graph.nodes[0][ALIAS_DNFR[1]] = -1.0
    assert compute_coherence(graph, return_means=True) == (0.5, 1.0, 0.0)
    graph.nodes[0][ALIAS_DEPI[1]] = 1.0
    assert compute_coherence(graph) == pytest.approx(1.0 / 3.0)
    assert compute_coherence(nx.Graph(), return_means=True) == (0.0, 0.0, 0.0)
