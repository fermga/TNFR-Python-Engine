"""Shared structural-acceleration history semantics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators.mutation import Mutation
from tnfr.operators.nodal_equation import (
    compute_d2epi_dt2, observe_structural_acceleration,
)


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


@pytest.mark.parametrize("count", (0, 1, 2))
def test_observation_distinguishes_short_physical_history_from_zero(count):
    graph = _graph(**{
        ALIAS_EPI[0]: 2.0,
        ALIAS_D2EPI[0]: 99.0,
        "epi_time_history": [(float(i), float(i)) for i in range(count)],
        "epi_history": [0.0, 0.0, 0.0],
    })
    before = deepcopy(dict(graph.nodes["n"]))
    observation = observe_structural_acceleration(graph, "n")
    assert observation.source == "epi_time_history"
    assert observation.history_length == count
    assert observation.time_basis == "physical_time"
    assert observation.available is False
    assert observation.value is None
    assert observation.samples == ()
    assert observation.current_endpoint_matches_state is None
    assert observation.reason == "insufficient_history"
    # Numeric compatibility neither interprets nor clears older telemetry.
    assert compute_d2epi_dt2(graph, "n") == 0.0
    assert dict(graph.nodes["n"]) == before


def test_missing_history_has_no_source_or_time_basis():
    graph = _graph()
    observation = observe_structural_acceleration(graph, "n")
    assert observation.source is None
    assert observation.time_basis is None
    assert observation.history_length == 0
    assert observation.available is False
    assert observation.value is None
    assert observation.reason == "missing_history"
    assert observation.samples == ()
    assert dict(graph.nodes["n"]) == {}


def test_valid_zero_has_complete_evidence_and_is_stored_only_by_wrapper():
    graph = _graph(**{
        ALIAS_EPI[0]: 7.0,
        "epi_time_history": [(0.0, 1.0), (1.0, 3.0), (3.0, 7.0)],
    })
    before = deepcopy(dict(graph.nodes["n"]))
    observation = observe_structural_acceleration(graph, "n")
    assert observation.available is True
    assert observation.value == 0.0
    assert observation.current_endpoint_matches_state is True
    assert observation.reason is None
    assert dict(graph.nodes["n"]) == before
    assert compute_d2epi_dt2(graph, "n") == 0.0
    assert graph.nodes["n"][ALIAS_D2EPI[0]] == 0.0


def test_observation_detaches_samples_and_preserves_nonuniform_quadratic_value():
    history = [[0.0, 0.0], [1.0, 1.0], [3.0, 9.0]]
    graph = _graph(**{ALIAS_EPI[0]: 9.0, "epi_time_history": history})
    observation = observe_structural_acceleration(graph, "n")
    assert observation.value == 2.0
    history[-1][1] = 42.0
    history.append([4.0, 16.0])
    assert observation.samples == ((0.0, 0.0), (1.0, 1.0), (3.0, 9.0))
    assert observation.history_length == 3
    payload = observation.to_dict()
    payload["samples"][-1][1] = -7.0
    assert observation.samples[-1] == (3.0, 9.0)
    with pytest.raises(FrozenInstanceError):
        observation.value = 42.0


def test_legacy_observation_keeps_unit_basis_and_unspecified_endpoint_provenance():
    graph = _graph(**{
        ALIAS_EPI[0]: 999.0,
        "epi_history": ["unassessed older prefix", 0.0, 3.0, 4.0],
        "_epi_history": [0.0, 1.0, 4.0],
    })
    observation = observe_structural_acceleration(graph, "n")
    assert observation.source == "epi_history"
    assert observation.history_length == 4
    assert observation.value == -2.0
    assert observation.time_basis == "legacy_unit_operator_step"
    assert observation.samples == (0.0, 3.0, 4.0)
    assert observation.current_endpoint_matches_state is None


@pytest.mark.parametrize("history, message", (
    ([(-1e308, 0.0), (1e308, 1.0), (1.1e308, 2.0)], "intervals"),
    ([(-1.1e308, 0.0), (-1e308, 1.0), (1e308, 2.0)], "intervals"),
    # Exact represented-coordinate acceleration is nonzero (~1e-308).
    ([(-1e308, 0.0), (0.0, 0.0), (1e308, 1e308)], "total span"),
    ([(0.0, 0.0), (5e-324, 1.0), (1.0, 2.0)], "secant rates"),
    ([(0.0, -1e308), (1.0, 1e308), (2.0, 0.0)], "secant rates"),
))
def test_nonfinite_derived_time_or_rate_rejects_before_any_write(history, message):
    graph = _graph(**{
        ALIAS_EPI[0]: history[-1][1],
        ALIAS_D2EPI[0]: 99.0,
        "epi_time_history": history,
    })
    before = deepcopy(dict(graph.nodes["n"]))
    for read in (observe_structural_acceleration, compute_d2epi_dt2):
        with pytest.raises(TNFRValueError, match=message):
            read(graph, "n")
        assert dict(graph.nodes["n"]) == before


def test_two_sample_mutation_rate_can_be_valid_with_unavailable_acceleration():
    from tnfr.physics.mutation_trigger import certify_mutation_trigger

    history = [(0.0, 0.0), (1.0, 1.0)]
    trigger = certify_mutation_trigger(
        current_epi=1.0, nu_f=1.0, delta_nfr=1.0, epi_time_history=history,
    )
    graph = _graph(**{ALIAS_EPI[0]: 1.0, "epi_time_history": history})
    acceleration = observe_structural_acceleration(graph, "n")
    assert trigger.evidence_valid is True
    assert trigger.observed_depi_dt == 1.0
    assert acceleration.available is False
    assert acceleration.value is None


def test_thol_gate_uses_one_history_selection_for_value_and_length(monkeypatch):
    from tnfr.operators import nodal_equation
    from tnfr.operators.preconditions.self_organization import (
        validate_self_organization_strict,
    )

    graph = _graph(**{
        ALIAS_EPI[0]: 1.0, ALIAS_DNFR[0]: 0.2, ALIAS_VF[0]: 1.0,
        "epi_time_history": [(0.0, 0.0), (1.0, 0.1), (3.0, 1.0)],
        "epi_history": [0.0, 0.0, 0.0],
    })
    graph.add_edge("n", "neighbor")
    graph.graph["THOL_METABOLIC_ENABLED"] = False
    original = nodal_equation._select_acceleration_history
    calls = []

    def count_selection(data):
        calls.append(data)
        return original(data)

    monkeypatch.setattr(nodal_equation, "_select_acceleration_history", count_selection)
    before = deepcopy(dict(graph.nodes["n"]))
    validate_self_organization_strict(graph, "n", emit_warnings=False)
    assert len(calls) == 1
    assert dict(graph.nodes["n"]) == before


def test_integrator_rhs_acceleration_is_distinct_from_clipped_epi_observation():
    from tnfr.dynamics.integrators import update_epi_via_nodal_equation

    graph = _graph(**{
        ALIAS_EPI[0]: 1.0, ALIAS_DNFR[0]: 2.0, ALIAS_VF[0]: 1.0,
    })
    graph.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, CLIP_MODE="hard", DT_MIN=1.0)
    update_epi_via_nodal_equation(graph, dt=1.0, t=0.0, method="euler")
    assert graph.nodes["n"][ALIAS_EPI[0]] == 1.0
    assert graph.nodes["n"][ALIAS_D2EPI[0]] == 2.0
    # These supplied continuous-flow samples record the clipped EPI endpoints.
    graph.nodes["n"]["epi_time_history"] = [(-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
    observation = observe_structural_acceleration(graph, "n")
    assert observation.available is True
    assert observation.value == 0.0
    assert graph.nodes["n"][ALIAS_D2EPI[0]] == 2.0
