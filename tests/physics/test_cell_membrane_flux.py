"""Canonical membrane-pressure tests for the cellular physics bridge."""

from __future__ import annotations

from copy import deepcopy
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.operators.nodal_equation import validate_nodal_equation
from tnfr.dynamics.structural_cache import StructuralCoherenceCache
from tnfr.physics.cell import (
    apply_membrane_flux,
    compute_homeostatic_index,
    compute_membrane_integrity,
    detect_cell_formation,
)


def _node(epi: float, phase: float = 0.0, *, nu_f: float = 1.0, dnfr: float = 0.0):
    return {"EPI": epi, "theta": phase, "nu_f": nu_f, "delta_nfr": dnfr}


def test_membrane_pressure_closes_the_nodal_equation_and_records_evidence() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.1, nu_f=2.0, dnfr=0.2))
    graph.nodes[1].update(_node(0.5))

    result = apply_membrane_flux(
        graph, [1], [0], permeability=0.5, phase_threshold=0.2, dt=0.25
    )

    report = result.nodes[0]
    assert report.intrinsic_delta_nfr == pytest.approx(0.2)
    assert report.membrane_delta_nfr == pytest.approx(0.2)
    assert report.effective_delta_nfr == pytest.approx(0.4)
    assert report.predicted_depi_dt == pytest.approx(0.8)
    assert report.measured_depi_dt == pytest.approx(0.8)
    assert report.nodal_residual == pytest.approx(0.0, abs=1e-15)
    assert result.max_abs_nodal_residual <= 1e-15
    assert not report.boundary_projection_applied
    assert validate_nodal_equation(
        graph,
        0,
        epi_before=0.1,
        epi_after=graph.nodes[0]["EPI"],
        dt=0.25,
        operator_name="cell membrane pressure",
        clip_aware=False,
        strict=True,
    )
    assert graph.nodes[0]["epi_history"] == pytest.approx([0.1, 0.3])
    assert graph.nodes[0]["epi_time_history"] == pytest.approx(
        [(0.0, 0.1), (0.25, 0.3)]
    )
    event = graph.nodes[0]["membrane_pressure_history"][-1]
    assert event["model"] == "cell_membrane_delta_nfr_v1"
    assert event["source"] == "tnfr.physics.cell.apply_membrane_flux"
    assert "source_glyph" not in graph.nodes[0]
    assert graph.graph["_t"] == pytest.approx(0.25)


def test_zero_capacity_retains_epi_under_nonzero_membrane_pressure() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.2, nu_f=0.0, dnfr=0.3))
    graph.nodes[1].update(_node(0.8))

    result = apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.4)

    report = result.nodes[0]
    assert report.membrane_delta_nfr == pytest.approx(0.3)
    assert report.effective_delta_nfr == pytest.approx(0.6)
    assert report.predicted_depi_dt == 0.0
    assert report.measured_depi_dt == 0.0
    assert report.nodal_residual == 0.0
    assert graph.nodes[0]["EPI"] == 0.2
    assert graph.nodes[0]["delta_nfr"] == pytest.approx(0.6)


def _order_graph() -> nx.Graph:
    graph = nx.Graph([(0, 2), (1, 2), (0, 3), (1, 3)])
    graph.nodes[0].update(_node(-0.2, phase=0.03, nu_f=0.7, dnfr=0.1))
    graph.nodes[1].update(_node(0.3, phase=-0.02, nu_f=1.1, dnfr=-0.05))
    graph.nodes[2].update(_node(0.8, phase=0.01))
    graph.nodes[3].update(_node(-0.4, phase=0.04))
    return graph


def test_boundary_proposals_are_simultaneous_and_order_independent() -> None:
    forward = _order_graph()
    reverse = _order_graph()

    apply_membrane_flux(forward, [2, 3], [0, 1], permeability=0.4, dt=0.2)
    apply_membrane_flux(reverse, [3, 2], [1, 0], permeability=0.4, dt=0.2)

    for node in (0, 1):
        assert forward.nodes[node]["EPI"] == reverse.nodes[node]["EPI"]
        assert forward.nodes[node]["delta_nfr"] == reverse.nodes[node]["delta_nfr"]
        assert forward.nodes[node]["dEPI_dt"] == reverse.nodes[node]["dEPI_dt"]
    assert forward.nodes[2] == reverse.nodes[2]
    assert forward.nodes[3] == reverse.nodes[3]


def test_invalid_late_boundary_rolls_back_every_live_channel() -> None:
    graph = nx.Graph([(0, 1), (2, 1)])
    graph.nodes[0].update(_node(0.1))
    graph.nodes[1].update(_node(0.5))
    graph.nodes[2].update(_node(-0.1))
    graph.nodes[2]["epi_time_history"] = [(1.0, -0.1)]
    graph.graph["membrane_flux_events"] = [{"model": "earlier"}]
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(graph.graph)

    with pytest.raises(ValueError, match="extends beyond"):
        apply_membrane_flux(graph, [1], [0, 2], dt=0.2)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


def test_shortest_arc_gate_filters_transport_and_cannot_be_weakened() -> None:
    compatible = nx.Graph([(0, 1)])
    compatible.nodes[0].update(_node(0.1, phase=0.01))
    compatible.nodes[1].update(_node(0.8, phase=math.tau - 0.01))
    accepted = apply_membrane_flux(
        compatible, [1], [0], permeability=0.5, phase_threshold=0.1, dt=0.1
    )

    blocked = nx.Graph([(0, 1)])
    blocked.nodes[0].update(_node(0.1, phase=0.0))
    blocked.nodes[1].update(_node(0.8, phase=2.0))
    rejected = apply_membrane_flux(
        blocked, [1], [0], permeability=0.5, phase_threshold=math.pi, dt=0.1
    )

    assert compatible.nodes[0]["EPI"] > 0.1
    assert accepted.nodes[0].compatible_internal_nodes == (1,)
    assert blocked.nodes[0]["EPI"] == 0.1
    assert rejected.phase_threshold == pytest.approx(math.pi / 2.0)
    assert rejected.nodes[0].compatible_internal_nodes == ()
    assert rejected.nodes[0].blocked_internal_nodes == (1,)


@pytest.mark.parametrize("bad_dt", [True, 0.0, -0.1, math.inf, math.nan])
def test_invalid_dt_rejects_before_live_mutation(bad_dt: object) -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.1))
    graph.nodes[1].update(_node(0.5))
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(ValueError, match="dt"):
        apply_membrane_flux(graph, [1], [0], dt=bad_dt)  # type: ignore[arg-type]

    assert dict(graph.nodes(data=True)) == before
    assert "_t" not in graph.graph


def test_hashable_node_ids_and_nonnegative_capacity_contract() -> None:
    boundary = "membrane"
    interior = ("inside", 1)
    graph = nx.Graph([(boundary, interior)])
    graph.nodes[boundary].update(_node(0.1))
    graph.nodes[interior].update(_node(0.5))

    result = apply_membrane_flux(graph, [interior], [boundary], dt=0.1)

    assert result.nodes[0].node == boundary
    assert result.nodes[0].compatible_internal_nodes == (interior,)

    invalid = nx.Graph([(boundary, interior)])
    invalid.nodes[boundary].update(_node(0.1, nu_f=-0.01))
    invalid.nodes[interior].update(_node(0.5))
    before = deepcopy(dict(invalid.nodes(data=True)))
    with pytest.raises(ValueError, match="nu_f must be nonnegative"):
        apply_membrane_flux(invalid, [interior], [boundary], dt=0.1)
    assert dict(invalid.nodes(data=True)) == before


def _compartment_snapshot() -> nx.Graph:
    graph = nx.complete_graph(4)
    for node in graph:
        graph.nodes[node].update(_node(0.5, dnfr=0.1))
    return graph


def test_cell_detector_requires_measured_flux_for_integrity() -> None:
    graph = _compartment_snapshot()
    unknown = detect_cell_formation(
        [graph],
        [0.0],
        [0, 1],
        [2, 3],
        c_boundary_threshold=0.0,
        selectivity_threshold=-1.0,
        homeostasis_threshold=0.0,
        integrity_threshold=0.7,
    )
    observed = detect_cell_formation(
        [graph],
        [0.0],
        [0, 1],
        [2, 3],
        c_boundary_threshold=0.0,
        selectivity_threshold=-1.0,
        homeostasis_threshold=0.0,
        integrity_threshold=0.7,
        membrane_fluxes=[(1.0, 0.0)],
    )

    assert math.isnan(float(unknown.membrane_integrity[0]))
    assert unknown.cell_formation_time is None
    assert observed.membrane_integrity[0] == pytest.approx(1.0)
    assert observed.cell_formation_time == pytest.approx(0.0)


def test_membrane_integrity_requires_positive_transport_evidence() -> None:
    assert compute_membrane_integrity(0.0, 0.0) == 0.0
    assert compute_membrane_integrity(3.0, 1.0) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="finite"):
        compute_membrane_integrity(math.nan, 0.0)


def test_cell_detector_validates_aligned_monotonic_observations() -> None:
    graph = _compartment_snapshot()
    with pytest.raises(ValueError, match="same length"):
        detect_cell_formation([graph], [], [0], [1])
    with pytest.raises(ValueError, match="increase strictly"):
        detect_cell_formation([graph, graph], [1.0, 1.0], [0], [1])
    with pytest.raises(ValueError, match="membrane_fluxes"):
        detect_cell_formation([graph], [0.0], [0], [1], membrane_fluxes=[])


def test_membrane_pressure_change_cannot_reuse_a_stale_structural_field() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.2, nu_f=0.0, dnfr=0.2))
    graph.nodes[1].update(_node(0.8))
    cache = StructuralCoherenceCache(enable_interpolation=True)
    before = cache.get_structural_fields(graph)

    apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.4)
    after = cache.get_structural_fields(graph)
    oracle = cache.get_structural_fields(graph, force_recompute=True)

    assert after is not before
    assert after.phi_s == oracle.phi_s
    assert after.phi_s[1] == pytest.approx(0.5)
    assert cache.interpolations == 0

def test_structural_cache_separates_canonical_coherence_from_phase_sync() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.0, phase=0.0, dnfr=10.0))
    graph.nodes[1].update(_node(1.0, phase=0.0, dnfr=-10.0))

    entry = StructuralCoherenceCache().get_structural_fields(graph)

    assert entry.coherence == pytest.approx(1.0 / 11.0)
    assert entry.phase_sync == pytest.approx(1.0)


def test_structural_cache_rejects_and_does_not_cache_nan_pressure() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.0, dnfr=float("nan")))
    graph.nodes[1].update(_node(1.0))
    cache = StructuralCoherenceCache()

    with pytest.raises(ValueError, match="finite"):
        cache.get_structural_fields(graph)

    assert cache.get_cache_stats()["structural_entries"] == 0


def test_structural_cache_returns_defensive_field_snapshots() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.0, dnfr=0.2))
    graph.nodes[1].update(_node(1.0, dnfr=-0.2))
    cache = StructuralCoherenceCache()

    first = cache.get_structural_fields(graph)
    expected = dict(first.phi_s)
    first.phi_s[0] = 999.0
    second = cache.get_structural_fields(graph)

    assert second.phi_s == expected
    assert second is not first

@pytest.mark.parametrize(
    "bad_samples",
    [
        True,
        1.0,
        [[0.1, 0.2]],
        [True, 0.1],
        [float("nan"), 0.1],
        [float("inf"), 0.1],
        ["0.1", 0.1],
    ],
)
def test_homeostatic_index_requires_finite_numeric_one_dimensional_samples(
    bad_samples: object,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional|finite|boolean"):
        compute_homeostatic_index(bad_samples)  # type: ignore[arg-type]


def test_homeostatic_index_coerces_finite_numeric_sequences() -> None:
    assert compute_homeostatic_index([0.2, 0.2]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="epsilon"):
        compute_homeostatic_index([0.2, 0.2], epsilon=0.0)


def test_cell_detector_rejects_corrupt_internal_pressure_before_classification(
) -> None:
    graph = _compartment_snapshot()
    graph.nodes[1]["delta_nfr"] = float("nan")

    with pytest.raises(ValueError, match="(?is)finite.*delta_nfr"):
        detect_cell_formation(
            [graph],
            [0.0],
            [0, 1],
            [2, 3],
            c_boundary_threshold=0.0,
            selectivity_threshold=-1.0,
            homeostasis_threshold=0.0,
            integrity_threshold=0.0,
            membrane_fluxes=[(1.0, 0.0)],
        )


def test_cell_detector_validates_partition_in_every_snapshot() -> None:
    first = _compartment_snapshot()
    second = _compartment_snapshot()
    second.remove_node(3)

    with pytest.raises(ValueError, match="snapshot 1.*unknown node 3"):
        detect_cell_formation([first, second], [0.0, 1.0], [0, 1], [2, 3])
    with pytest.raises(ValueError, match="must be disjoint"):
        detect_cell_formation([first], [0.0], [0, 1], [1, 2])


def test_membrane_pressure_ownership_uses_explicit_token_and_channels() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.1, dnfr=0.2))
    graph.nodes[1].update(_node(0.5))

    first = apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)
    raw_effective = graph.nodes[0]["delta_nfr"]
    provenance = graph.nodes[0]["membrane_pressure_provenance"]

    assert graph.nodes[0]["delta_nfr_intrinsic"] == pytest.approx(0.2)
    assert graph.nodes[0]["delta_nfr_membrane"] == pytest.approx(0.2)
    assert provenance["token"] == raw_effective.membrane_provenance_token
    assert first.nodes[0].effective_delta_nfr == pytest.approx(float(raw_effective))

    second = apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)
    assert second.nodes[0].intrinsic_delta_nfr == pytest.approx(0.2)


def test_numerically_equal_external_pressure_write_becomes_intrinsic() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.1, dnfr=0.2))
    graph.nodes[1].update(_node(0.5))
    apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)

    externally_written = float(graph.nodes[0]["delta_nfr"])
    graph.nodes[0]["delta_nfr"] = externally_written
    assert not hasattr(graph.nodes[0]["delta_nfr"], "membrane_provenance_token")

    result = apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)

    assert result.nodes[0].intrinsic_delta_nfr == pytest.approx(externally_written)
    assert graph.nodes[0]["delta_nfr_intrinsic"] == pytest.approx(externally_written)


def test_invalid_owned_pressure_provenance_rejects_atomically() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.1, dnfr=0.2))
    graph.nodes[1].update(_node(0.5))
    apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)
    graph.nodes[0]["membrane_pressure_provenance"]["token"] = "tampered"
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(graph.graph)

    with pytest.raises(ValueError, match="provenance is invalid"):
        apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


def test_cell_detector_homeostasis_is_snapshot_local() -> None:
    first = _compartment_snapshot()
    second = _compartment_snapshot()
    first.nodes[0]["delta_nfr"] = 10.0
    first.nodes[1]["delta_nfr"] = -10.0
    second.nodes[0]["delta_nfr"] = 0.2
    second.nodes[1]["delta_nfr"] = 0.2

    telemetry = detect_cell_formation(
        [first, second],
        [0.0, 1.0],
        [0, 1],
        [2, 3],
        membrane_fluxes=[(1.0, 0.0), (1.0, 0.0)],
    )

    assert telemetry.homeostatic_index[0] == 0.0
    assert telemetry.homeostatic_index[1] == pytest.approx(1.0)


def test_zero_total_flux_cannot_supply_membrane_formation_evidence() -> None:
    graph = _compartment_snapshot()

    telemetry = detect_cell_formation(
        [graph],
        [0.0],
        [0, 1],
        [2, 3],
        c_boundary_threshold=0.0,
        selectivity_threshold=-1.0,
        homeostasis_threshold=0.0,
        integrity_threshold=0.0,
        membrane_fluxes=[(0.0, 0.0)],
    )

    assert telemetry.membrane_integrity[0] == 0.0
    assert telemetry.cell_formation_time is None


def test_cell_detector_requires_explicit_internal_pressure_evidence() -> None:
    graph = _compartment_snapshot()
    del graph.nodes[1]["delta_nfr"]

    with pytest.raises(ValueError, match="requires explicit delta_nfr evidence"):
        detect_cell_formation(
            [graph],
            [0.0],
            [0, 1],
            [2, 3],
            membrane_fluxes=[(1.0, 0.0)],
        )


@pytest.mark.parametrize(
    "bad_times",
    [np.array([[0.0]]), [True], ["0.0"]],
)
def test_cell_detector_requires_numeric_one_dimensional_times(bad_times) -> None:
    graph = _compartment_snapshot()

    with pytest.raises(ValueError, match="one-dimensional|finite real scalar|boolean"):
        detect_cell_formation([graph], bad_times, [0, 1], [2, 3])


def test_membrane_step_advances_every_node_on_the_shared_graph_clock() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.1, dnfr=0.0))
    graph.nodes[1].update(_node(0.5, dnfr=0.1))

    apply_membrane_flux(graph, [1], [0], permeability=0.0, dt=1.0)

    assert graph.graph["_t"] == pytest.approx(1.0)
    assert graph.nodes[1]["EPI"] == pytest.approx(0.6)
    history = graph.nodes[1]["epi_time_history"]
    assert [sample[0] for sample in history] == pytest.approx([0.0, 1.0])
    assert [sample[1] for sample in history] == pytest.approx([0.5, 0.6])

    update_epi_via_nodal_equation(graph, dt=1.0)
    assert graph.graph["_t"] == pytest.approx(2.0)
    assert graph.nodes[1]["EPI"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("field", "value"),
    [("node", "foreign"), ("source", "foreign-source")],
)
def test_membrane_pressure_ownership_binds_node_and_source(
    field: str, value: object
) -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(_node(0.1, dnfr=0.2))
    graph.nodes[1].update(_node(0.5))
    apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)
    graph.nodes[0]["membrane_pressure_provenance"][field] = value
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(graph.graph)

    with pytest.raises(ValueError, match="provenance is invalid"):
        apply_membrane_flux(graph, [1], [0], permeability=0.5, dt=0.1)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


def test_membrane_commit_rolls_back_on_base_exception() -> None:
    class InterruptOnce(dict):
        armed = False

        def update(self, *args, **kwargs) -> None:
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt("interrupt commit")
            super().update(*args, **kwargs)

    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(_node(0.2 + 0.1 * node, dnfr=0.05))
    interrupting = InterruptOnce(graph.nodes[1])
    graph._node[1] = interrupting
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(graph.graph)
    interrupting.armed = True

    with pytest.raises(KeyboardInterrupt, match="interrupt commit"):
        apply_membrane_flux(graph, [2], [0], permeability=0.2, dt=0.1)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph
