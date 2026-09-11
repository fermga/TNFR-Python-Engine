"""Shared structural-acceleration history semantics."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_EPI
from tnfr.errors import TNFRValueError
from tnfr.operators.mutation import Mutation
from tnfr.operators.nodal_equation import compute_d2epi_dt2


def _graph(**attributes: object) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("n", **attributes)
    return graph


def test_nonuniform_physical_history_uses_adjacent_secant_acceleration():
    graph = _graph(
        **{
            ALIAS_EPI[0]: 9.0,
            "epi_time_history": [(0.0, 0.0), (1.0, 1.0), (3.0, 9.0)],
            "_epi_history": [0.0, 100.0, 0.0],
        }
    )

    before = deepcopy(dict(graph.nodes["n"]))
    assert compute_d2epi_dt2(graph, "n", store=False) == pytest.approx(2.0)
    assert dict(graph.nodes["n"]) == before


def test_canonical_legacy_history_precedes_private_legacy_history():
    graph = _graph(
        epi_history=[0.0, 1.0, 4.0],
        _epi_history=[0.0, 10.0, 0.0],
    )

    assert compute_d2epi_dt2(graph, "n") == pytest.approx(2.0)
    assert graph.nodes["n"][ALIAS_D2EPI[0]] == pytest.approx(2.0)


def test_empty_canonical_history_falls_back_to_private_legacy_history():
    graph = _graph(epi_history=[], _epi_history=[0.0, 1.0, 4.0])

    assert compute_d2epi_dt2(graph, "n", store=False) == pytest.approx(2.0)


def test_timestamped_history_with_fewer_than_three_samples_is_unavailable():
    graph = _graph(
        **{
            ALIAS_EPI[0]: 1.0,
            "epi_time_history": [(0.0, 0.0), (1.0, 1.0)],
            "_epi_history": [0.0, 1.0, 4.0],
        }
    )

    assert compute_d2epi_dt2(graph, "n", store=False) == 0.0
    assert ALIAS_D2EPI[0] not in graph.nodes["n"]


@pytest.mark.parametrize(
    "history",
    [
        [(0.0, 0.0), (0.0, 1.0), (1.0, 2.0)],
        [(0.0, 0.0), (1.0, 1.0), (2.0, float("nan"))],
        [(False, 0.0), (1.0, 1.0), (2.0, 2.0)],
    ],
)
def test_invalid_physical_history_rejects_without_telemetry_write(history):
    graph = _graph(**{ALIAS_EPI[0]: 2.0, "epi_time_history": history})
    before = deepcopy(dict(graph.nodes["n"]))

    with pytest.raises(TNFRValueError):
        compute_d2epi_dt2(graph, "n")

    assert dict(graph.nodes["n"]) == before


def test_stale_physical_endpoint_rejects_without_telemetry_write():
    graph = _graph(
        **{
            ALIAS_EPI[0]: 3.0,
            "epi_time_history": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        }
    )
    before = deepcopy(dict(graph.nodes["n"]))

    with pytest.raises(TNFRValueError, match="final EPI"):
        compute_d2epi_dt2(graph, "n")

    assert dict(graph.nodes["n"]) == before


def test_mutation_reads_the_same_shared_acceleration_magnitude():
    graph = _graph(epi_history=[0.0, 3.0, 4.0])

    signed = compute_d2epi_dt2(graph, "n", store=False)
    assert signed == pytest.approx(-2.0)
    assert Mutation()._compute_epi_acceleration(graph, "n") == pytest.approx(2.0)


@pytest.mark.parametrize("store", [0, 1, None, "false"])
def test_store_switch_requires_a_real_boolean(store):
    graph = _graph(_epi_history=[0.0, 1.0, 4.0])
    before = deepcopy(dict(graph.nodes["n"]))

    with pytest.raises(TNFRValueError, match="store must be a boolean"):
        compute_d2epi_dt2(graph, "n", store=store)

    assert dict(graph.nodes["n"]) == before
