"""Pure-EPI closure need not preserve the complete canonical pressure law.

On unit K_{p,q}, grouping the two sides gives macro conductance p*q,
macro metric (p*q,p*q), and unit capacity. The EPI quotient closes exactly.
Fine unique-support degrees are q and p, so their neighbor differences are
p-q and q-p. The two-node quotient instead has degrees (1,1) and zero topology
pressure. Thus a nonzero topology coefficient is not inherited by the same
four-channel macro rule when p != q, even at uniform phase/capacity/form.
The default topology coefficient is zero; the all-positive mix below is an
explicitly configured model, not a failure of the default configuration.

These are fixed-support algebra and actual pressure-hook controls. They do
not supply phase/capacity/support evolution, select a partition, or disprove
richer inherited geometry. The exact affine source is kept explicitly.
"""

from fractions import Fraction as F

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.epi_memory import observe_forced_support_closure
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.physics.structural_morphism import certify_epi_coarse_graining

EQUAL_WEIGHTS = {"phase": 0.25, "epi": 0.25, "vf": 0.25, "topo": 0.25}


def _prepared(graph, epi, weights):
    for node, value in zip(graph, epi, strict=True):
        graph.nodes[node].update(EPI=float(value), nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph["DNFR_WEIGHTS"] = dict(weights)
    # Exercise the public pressure writer before reading the exact record.
    default_compute_delta_nfr(graph)
    capture = capture_non_epi_forcing(graph)
    assert capture.stored_pressure_residual == (F(0),) * len(graph)
    assert capture.kernel_pressure_defect == (F(0),) * len(graph)
    assert capture.full_kernel_pressure == capture.snapshot.stored_pressure
    return capture


def _models(p, q, *, epi=None, weights=EQUAL_WEIGHTS):
    graph = nx.complete_bipartite_graph(p, q)
    values = (0.0,) * (p + q) if epi is None else epi
    fine = _prepared(graph, values, weights)
    blocks = (tuple(range(p)), tuple(range(p, p + q)))
    reference = derive_forced_support_balance(
        fine.snapshot, epi_weight=fine.epi_weight, forcing=fine.forcing
    )
    exact = observe_forced_support_closure(reference, blocks)
    coarse = certify_epi_coarse_graining(graph, blocks)
    macro = nx.Graph()
    macro.add_nodes_from(range(2))
    macro.add_edge(0, 1, weight=float(coarse.macro_conductance[0, 1]))
    # Uniform fine capacity descends unchanged. This is computed by the
    # existing quotient, not selected to compensate for the pressure defect.
    np.testing.assert_array_equal(coarse.macro_frequency, (1.0, 1.0))
    macro_capture = _prepared(macro, exact.projected_epi, weights)
    return fine, macro_capture, exact, coarse


def _project(closure, values):
    return tuple(dot(row, values) for row in closure.projection)


def test_k23_exact_epi_closure_does_not_preserve_all_positive_pressure_channels():
    fine, macro, exact, coarse = _models(2, 3)
    zero_fine = (F(0),) * 5
    assert exact.all_state_affine_closed
    assert exact.lifted_affine_subspace_invariant
    assert exact.witness is None
    assert exact.macro_metric_weights == (F(6), F(6))
    assert exact.macro_generator == ((F(1, 4), F(-1, 4)), (F(-1, 4), F(1, 4)))
    assert all(value == 0 for row in exact.hidden_to_macro for value in row)
    assert all(value == 0 for row in exact.instantaneous_kernel for value in row)
    assert coarse.nodal_closure_within_tolerance
    np.testing.assert_array_equal(coarse.macro_conductance, ((0.0, 6.0), (6.0, 0.0)))
    assert (
        dict(fine.normalized_weights)
        == dict(macro.normalized_weights)
        == {name: F(1, 4) for name in EQUAL_WEIGHTS}
    )
    assert (
        fine.phase_gradient
        == fine.snapshot.capacity_gradient
        == fine.snapshot.epi_gradient
        == zero_fine
    )
    assert fine.snapshot.topology_gradient == (F(-1), F(-1), F(1), F(1), F(1))
    assert macro.snapshot.topology_gradient == (F(0), F(0))
    assert macro.snapshot.rate == (F(0), F(0))
    assert (
        _project(exact, fine.snapshot.rate)
        == exact.projected_source
        == (F(-1, 4), F(1, 4))
    )
    channels = dict(decompose_non_epi_forcing(fine))
    assert channels["phase"] == channels["vf"] == zero_fine
    assert channels["topo"] == fine.snapshot.stored_pressure == fine.forcing


def test_inherited_topology_source_is_distinct_from_unresolved_epi_memory():
    # Nonzero macro contrast and hidden form coexist. Exact EPI closure
    # removes the hidden form's influence, retaining diffusion plus the
    # source computed independently from fine support degrees.
    fine, macro, exact, _ = _models(2, 3, epi=(-1.0, 0.0, 0.0, 1.0, 2.0))
    assert exact.projected_epi == (F(-1, 2), F(1))
    assert any(exact.hidden_epi)
    assert exact.hidden_rate_contribution == (F(0), F(0))
    assert exact.all_state_affine_closed
    assert exact.projected_source == (F(-1, 4), F(1, 4))
    inherited_rate = tuple(
        force - dot(row, exact.projected_epi)
        for force, row in zip(
            exact.projected_source, exact.macro_generator, strict=True
        )
    )
    assert (
        inherited_rate
        == exact.projected_nodal_rate
        == _project(exact, fine.snapshot.rate)
    )
    assert inherited_rate == (F(1, 8), F(-1, 8))
    assert macro.snapshot.rate == (F(3, 8), F(-3, 8))
    assert (
        tuple(a - b for a, b in zip(inherited_rate, macro.snapshot.rate))
        == exact.projected_source
    )
    # Multiplying any zero macro topology gradient by a different scalar
    # coefficient still cannot reproduce the inherited nonzero source.
    assert macro.snapshot.topology_gradient == (F(0), F(0))


def test_topology_off_control_preserves_nonzero_projected_epi_response():
    weights = {"phase": 0.25, "epi": 0.5, "vf": 0.25, "topo": 0.0}
    fine, macro, exact, _ = _models(
        2, 3, epi=(-1.0, 0.0, 0.0, 1.0, 2.0), weights=weights
    )
    assert fine.snapshot.topology_gradient == (F(-1), F(-1), F(1), F(1), F(1))
    assert fine.forcing == (F(0),) * 5
    assert exact.projected_source == (F(0), F(0))
    assert exact.projected_epi == (F(-1, 2), F(1))
    assert (
        _project(exact, fine.snapshot.rate)
        == macro.snapshot.rate
        == (F(3, 4), F(-3, 4))
    )


@pytest.mark.parametrize("p,q", ((1, 2), (2, 3), (3, 2), (3, 4), (2, 2), (3, 3)))
def test_bipartite_degree_law_including_balanced_positive_control(p, q):
    fine, macro, exact, coarse = _models(p, q)
    expected_gradient = (F(p - q),) * p + (F(q - p),) * q
    expected_rate = (F(p - q, 4), F(q - p, 4))
    assert fine.snapshot.topology_gradient == expected_gradient
    assert fine.snapshot.stored_pressure == tuple(
        value / 4 for value in expected_gradient
    )
    assert exact.macro_metric_weights == (F(p * q), F(p * q))
    assert exact.all_state_affine_closed
    assert exact.lifted_affine_subspace_invariant
    assert coarse.macro_conductance[0, 1] == p * q
    assert _project(exact, fine.snapshot.rate) == expected_rate
    assert macro.snapshot.rate == (F(0), F(0))
    assert (_project(exact, fine.snapshot.rate) == macro.snapshot.rate) is (p == q)
