"""Portable archival controls with no retained producer or trajectory access."""

import json
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.mathematics import BEPIElement
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import NonEpiForcingObservation
from tnfr.physics.support_transport import observe_support_transport
from tnfr.research.recorded_support import (
    bind_recorded_nodal_state,
    graph_from_recorded_nodal_state,
    read_recorded_forced_reference,
    read_recorded_forcing,
    read_recorded_support_snapshot,
)
from tnfr.types import serialize_bepi

F = Fraction


def _json(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


@pytest.fixture
def evidence():
    # Node order differs from lexical order. The zero-weight edge contributes
    # support neighbors but no EPI conductance; its attributes are retained.
    nodes = ((2, 0), "center", (1, 0))
    graph = nx.Graph()
    for node, epi, capacity, phase, pressure in zip(
        nodes,
        (-0.5, 0.25, 1.5),
        (0.5, 1.0, 2.0),
        (0.0, 0.25, 0.5),
        (0.125, -0.25, 0.5),
        strict=True,
    ):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: epi,
                ALIAS_VF[0]: capacity,
                ALIAS_THETA[0]: phase,
                ALIAS_DNFR[0]: pressure,
            },
        )
    graph.add_edge(nodes[0], nodes[2], weight=0.0, length=2.0, tags=["support"])
    graph.add_edge(nodes[0], nodes[1], weight=0.5)
    graph.add_edge(nodes[1], nodes[2], weight=1.0)
    graph.graph["_t"] = 0.1
    state = {
        "nodes": nodes,
        "time": graph.graph["_t"],
        "edges": tuple((u, v, deepcopy(data)) for u, v, data in graph.edges(data=True)),
        **{
            key: tuple(graph.nodes[node][alias[0]] for node in nodes)
            for key, alias in (
                ("epi", ALIAS_EPI),
                ("capacity", ALIAS_VF),
                ("phase", ALIAS_THETA),
                ("pressure", ALIAS_DNFR),
            )
        },
    }
    snapshot = observe_support_transport(graph)
    # This is a declared coefficient, deliberately not a computed phase law.
    gradient = (F(1, 7), F(-2, 7), F(3, 7))
    components = (
        ("phase", tuple(value / 4 for value in gradient)),
        ("vf", tuple(value / 4 for value in snapshot.capacity_gradient)),
        ("topo", tuple(value / 4 for value in snapshot.topology_gradient)),
    )
    forcing = tuple(sum(values[i] for _, values in components) for i in range(3))
    defect = (F(1, 128), F(0), F(-1, 128))
    full = tuple(
        g / 4 + f + d
        for g, f, d in zip(snapshot.epi_gradient, forcing, defect, strict=True)
    )
    observation = NonEpiForcingObservation(
        snapshot=snapshot,
        phase=tuple(map(F, state["phase"])),
        epi_weight=F(1, 4),
        forcing=forcing,
        phase_gradient=gradient,
        normalized_weights=tuple(
            (name, F(1, 4)) for name in ("phase", "epi", "vf", "topo")
        ),
        full_kernel_pressure=full,
        kernel_pressure_defect=defect,
        stored_pressure_residual=tuple(
            p - k for p, k in zip(snapshot.stored_pressure, full)
        ),
    )
    reference = derive_forced_support_balance(
        snapshot, epi_weight=F(1, 4), forcing=forcing
    )
    return graph, state, snapshot, observation, components, reference


@pytest.mark.parametrize("serialized", (False, True))
def test_native_and_json_records_rebuild_all_exact_fields_without_kernel(
    evidence, monkeypatch, serialized
):
    graph, state, snapshot, observation, components, reference = evidence
    raw = {
        "snapshot": asdict(snapshot),
        "reference": asdict(reference),
        "capture": {"observation": asdict(observation), "components": components},
        "state": state,
    }
    raw = _json(raw) if serialized else raw
    before = deepcopy(raw)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Detached archival admission must not run a phase kernel or pressure refresh"
        )

    monkeypatch.setattr(
        "tnfr.dynamics.fused_dnfr.compute_fused_gradients_symmetric", forbidden
    )
    monkeypatch.setattr("tnfr.dynamics.dnfr.default_compute_delta_nfr", forbidden)
    assert read_recorded_support_snapshot(raw["snapshot"]) == snapshot
    assert read_recorded_forced_reference(raw["reference"]) == reference
    assert read_recorded_forcing(raw["capture"]) == (snapshot, observation, components)
    assert bind_recorded_nodal_state(raw["state"], raw["capture"], snapshot) is None
    detached = graph_from_recorded_nodal_state(raw["state"])
    assert tuple(detached) == tuple(graph)
    assert tuple(detached.edges(data=True)) == tuple(graph.edges(data=True))
    assert observe_support_transport(detached) == snapshot
    assert detached.graph == {"_t": 0.1}
    assert F(detached.graph["_t"]) == F.from_float(0.1)
    assert detached.nodes[(2, 0)][ALIAS_EPI[0]] == -0.5
    assert raw == before
    detached.edges[(2, 0), (1, 0)]["tags"].append("changed")
    assert raw == before


@pytest.mark.parametrize(
    "bad", (True, float("nan"), float("inf"), 0.5, "1/0", "0.5", "nan", [], 1j)
)
def test_exact_snapshot_rejects_nonexact_or_malformed_coordinates(evidence, bad):
    payload = _json(asdict(evidence[2]))
    payload["epi"][0] = bad
    with pytest.raises((TypeError, ValueError)):
        read_recorded_support_snapshot(payload)


@pytest.mark.parametrize("change", ("rate", "energy", "missing", "extra"))
def test_complete_snapshot_rejects_modified_or_incomplete_caches(evidence, change):
    payload = _json(asdict(evidence[2]))
    if change == "rate":
        payload["rate"][0] = "999"
    elif change == "energy":
        payload["dirichlet_energy"] = "999"
    elif change == "missing":
        del payload["energy_rate"]
    else:
        payload["provenance"] = "not authenticated"
    with pytest.raises(ValueError, match="reconstructed evidence"):
        read_recorded_support_snapshot(payload)


@pytest.mark.parametrize("change", ("profile", "step", "boolean"))
def test_forced_reference_recomputes_then_rejects_tampered_caches(evidence, change):
    payload = _json(asdict(evidence[5]))
    if change == "profile":
        payload["relative_profile"][0] = "999"
    elif change == "step":
        payload["max_convex_step"] = "999"
    else:
        payload["has_zero_pressure_equilibrium"] = int(
            payload["has_zero_pressure_equilibrium"]
        )
    with pytest.raises(ValueError, match="reconstructed evidence"):
        read_recorded_forced_reference(payload)


@pytest.mark.parametrize(
    "change",
    (
        "forcing",
        "component",
        "phase_length",
        "weight_order",
        "negative_weight",
        "kernel_defect",
        "stored_residual",
        "extra",
    ),
)
def test_forcing_rejects_inconsistent_channels_and_all_saved_residuals(
    evidence, change
):
    payload = _json({"observation": asdict(evidence[3]), "components": evidence[4]})
    row = payload["observation"]
    if change == "forcing":
        row["forcing"][0] = "999"
    elif change == "component":
        payload["components"][0][1][0] = "999"
    elif change == "phase_length":
        row["phase"].pop()
    elif change == "weight_order":
        row["normalized_weights"].reverse()
    elif change == "negative_weight":
        row["normalized_weights"][0][1] = "-1"
    elif change == "kernel_defect":
        row["kernel_pressure_defect"][0] = "999"
    elif change == "stored_residual":
        row["stored_pressure_residual"][0] = "999"
    else:
        payload["unverified"] = True
    with pytest.raises(ValueError):
        read_recorded_forcing(payload)


@pytest.mark.parametrize("field", ("epi", "capacity", "phase", "pressure"))
@pytest.mark.parametrize("bad", (False, "0.0", float("nan"), float("inf")))
def test_raw_graph_and_binding_reject_nonfinite_or_coerced_triad(evidence, field, bad):
    payload = _json(evidence[1])
    payload[field][0] = bad
    capture = {"observation": asdict(evidence[3]), "components": evidence[4]}
    for read in (
        lambda: graph_from_recorded_nodal_state(payload),
        lambda: bind_recorded_nodal_state(payload, capture, evidence[2]),
    ):
        with pytest.raises((TypeError, ValueError)):
            read()


@pytest.mark.parametrize("serialized", (False, True))
@pytest.mark.parametrize("kind", ("nonuniform", "complex"))
def test_raw_graph_cannot_silently_replace_rich_form_by_a_magnitude(
    evidence, serialized, kind
):
    payload = _json(evidence[1])
    form = (
        BEPIElement((1.0, -1.0), (1.0, -1.0), (0.0, 1.0))
        if kind == "nonuniform"
        else BEPIElement((1j, 1j), (1j, 1j), (0.0, 1.0))
    )
    payload["epi"][0] = serialize_bepi(form) if serialized else form
    with pytest.raises(ValueError, match="finite real JSON numbers"):
        graph_from_recorded_nodal_state(payload)


@pytest.mark.parametrize(
    "change",
    (
        "duplicate_node",
        "vector_length",
        "negative_capacity",
        "unknown_node",
        "duplicate_edge",
        "loop",
        "missing_time",
        "missing_pressure",
        "missing_edges",
        "boolean_time",
        "text_time",
        "infinite_time",
        "negative_weight",
        "text_weight",
    ),
)
def test_graph_view_rejects_malformed_topology_and_missing_raw_inputs(evidence, change):
    payload = _json(evidence[1])
    if change == "duplicate_node":
        payload["nodes"][1] = payload["nodes"][0]
    elif change == "vector_length":
        payload["epi"].pop()
    elif change == "negative_capacity":
        payload["capacity"][0] = -1.0
    elif change == "unknown_node":
        payload["edges"][0][0] = "absent"
    elif change == "duplicate_edge":
        u, v, data = payload["edges"][0]
        payload["edges"].append([v, u, data])
    elif change == "loop":
        payload["edges"][0][1] = payload["edges"][0][0]
    elif change.startswith("missing_"):
        del payload[change.removeprefix("missing_")]
    elif change.endswith("time"):
        payload["time"] = {
            "boolean_time": True,
            "text_time": "0.1",
            "infinite_time": float("inf"),
        }[change]
    else:
        payload["edges"][0][2]["weight"] = (
            -1.0 if change == "negative_weight" else "0.0"
        )
    with pytest.raises((KeyError, TypeError, ValueError)):
        graph_from_recorded_nodal_state(payload)


@pytest.mark.parametrize(
    "reader,index",
    ((read_recorded_support_snapshot, 2), (read_recorded_forced_reference, 5)),
)
def test_archival_readers_require_mappings_not_unchecked_dataclasses(
    evidence, reader, index
):
    with pytest.raises(TypeError, match="mapping"):
        reader(evidence[index])
