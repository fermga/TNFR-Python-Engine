"""Independent exact capacity balances and materialized U3 support controls."""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.config import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators import _phase_gate
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics.coupling_support import (
    derive_antipodal_region_phase_balance,
    derive_compatible_capacity_balance,
    observe_antipodal_region_phase_response,
    observe_coupling_support,
)

F = Fraction


def _phase_reference(**overrides):
    factors = {"coupling_phase_factor": F(1, 2), "coherence_phase_factor": F(1, 3)}
    factors.update(overrides)
    return derive_antipodal_region_phase_balance(**factors)


def test_antipodal_phase_hand_matrices_and_a_contracting_direction():
    reference = _phase_reference()
    assert reference.coupling_matrix == ((F(5, 6), F(1, 6)), (F(1, 3), F(2, 3)))
    assert reference.coherence_matrix == ((F(5, 6), F(1, 6)), (F(2, 3), 1))
    assert reference.product_matrix == ((F(3, 4), F(1, 4)), (F(8, 9), F(7, 9)))
    assert reference.trace == F(55, 36)
    assert reference.determinant == F(13, 36)
    assert reference.det_identity_minus_product == F(-1, 6)
    assert reference.strict_expansion_certificate
    response = observe_antipodal_region_phase_response(reference, interior=2, bridge=-1)
    assert response.before == (2, -1)
    assert response.after_coupling == (F(3, 2), 0)
    assert response.after_coherence == (F(5, 4), 1)
    assert response.embedded_before == (2, 2, -1, 1, -2, -2)
    assert response.embedded_after == (F(5, 4), F(5, 4), 1, -1, F(-5, 4), F(-5, 4))
    assert response.energy_before == 9
    assert response.energy_after == F(33, 8)
    assert response.energy_change == F(-39, 8)


def test_antipodal_reduction_matches_independent_full_support_and_merged_proposals():
    reference = _phase_reference()
    t, alpha = reference.coupling_phase_factor, reference.coherence_phase_factor
    full_rows = ((1, 2), (0, 2), (0, 1, 3), (2, 4, 5), (3, 5), (3, 4))
    compatible = ((1, 2), (0, 2), (0, 1), (4, 5), (3, 5), (3, 4))
    signs = (1, 1, 1, -1, -1, -1)
    for direction in ((1, 1, 0, 0, -1, -1), (0, 0, 1, -1, 0, 0)):
        incoming = [[] for _ in range(6)]
        for source, neighbors in enumerate(compatible):
            receivers = (source, *neighbors)
            mean = sum((F(direction[i]) for i in receivers), F(0)) / len(receivers)
            for receiver in receivers:
                incoming[receiver].append((1 - t) * direction[receiver] + t * mean)
        assert tuple(map(len, incoming)) == (3,) * 6
        coupled = tuple(sum(values, F(0)) / len(values) for values in incoming)
        after = tuple(
            (1 - alpha) * coupled[i]
            + alpha
            * sum((signs[j] * coupled[j] for j in row), F(0))
            / sum(signs[j] for j in row)
            for i, row in enumerate(full_rows)
        )
        response = observe_antipodal_region_phase_response(
            reference,
            interior=direction[0],
            bridge=direction[2],
        )
        assert coupled[:3] == (*response.after_coupling[:1], *response.after_coupling)
        assert response.embedded_after == after


@pytest.mark.parametrize("t", (F(0), F(1, 7), F(1, 2), F(1)))
@pytest.mark.parametrize("alpha", (F(0), F(1, 3), F(1)))
def test_phase_characteristic_identity_and_uniform_direction_across_factor_boundaries(
    t, alpha
):
    reference = _phase_reference(coupling_phase_factor=t, coherence_phase_factor=alpha)
    assert (
        reference.det_identity_minus_product
        == -(alpha**2) * (1 - t) - 2 * alpha * t / 3
    )
    assert (
        reference.det_identity_minus_product
        == 1 - reference.trace + reference.determinant
    )
    assert reference.determinant == (1 - alpha / 2 - alpha**2) * (1 - t)
    assert reference.strict_expansion_certificate == (alpha > 0)
    response = observe_antipodal_region_phase_response(reference, interior=1, bridge=1)
    assert response.after_coupling == (1, 1)
    assert response.after_coherence == (1, 1 + 2 * alpha)
    assert response.energy_before == 3
    assert response.energy_change == 4 * alpha * (1 + alpha)


@pytest.mark.parametrize("t", (F(0), F(1, 4), F(1)))
def test_zero_coherence_phase_has_neutral_constant_mode_and_no_quadratic_growth(t):
    reference = _phase_reference(coupling_phase_factor=t, coherence_phase_factor=0)
    response = observe_antipodal_region_phase_response(reference, interior=2, bridge=-1)
    assert not reference.strict_expansion_certificate
    assert reference.det_identity_minus_product == 0
    assert response.energy_change == -6 * t * (2 - t)
    assert response.energy_change <= 0


def test_full_coupling_boundary_has_rank_one_and_an_expanding_exact_eigenvector():
    reference = _phase_reference(coupling_phase_factor=1)
    assert reference.determinant == 0
    assert reference.trace == F(11, 9)
    response = observe_antipodal_region_phase_response(
        reference, interior=1, bridge=F(5, 3)
    )
    assert response.after_coherence == (F(11, 9), F(55, 27))
    assert response.energy_after == F(121, 81) * response.energy_before


def test_represented_default_factors_are_retained_without_rational_decimal_substitution():
    t = canonical_glyph_factor_defaults()["UM_theta_push"]
    reference = _phase_reference(coupling_phase_factor=t, coherence_phase_factor=0.3)
    assert reference.coupling_phase_factor == F.from_float(t)
    assert reference.coherence_phase_factor == F.from_float(0.3)
    assert reference.coherence_phase_factor != F(3, 10)
    assert reference.strict_expansion_certificate


def test_phase_observer_rebuilds_forged_public_matrix_and_boolean_caches():
    reference = _phase_reference()
    forged = replace(
        reference,
        coupling_matrix=((99,),),
        coherence_matrix=(),
        product_matrix=(),
        trace=99,
        determinant=99,
        det_identity_minus_product=99,
        strict_expansion_certificate=False,
    )
    response = observe_antipodal_region_phase_response(forged, interior=1, bridge=1)
    assert response.reference == reference
    assert response.reference is not reference
    assert response.after_coherence == (1, F(5, 3))
    assert response.energy_change == F(16, 9)


def test_phase_observer_revalidates_forged_source_factors():
    forged = replace(_phase_reference(), coherence_phase_factor=F(7, 6))
    with pytest.raises(ValueError, match="phase factors"):
        observe_antipodal_region_phase_response(forged, interior=1, bridge=1)


@pytest.mark.parametrize("field", ("coupling_phase_factor", "coherence_phase_factor"))
@pytest.mark.parametrize(
    "value", (True, "0.3", complex(1, 0), math.nan, math.inf, F(-1, 9), F(10, 9))
)
def test_phase_reference_rejects_invalid_factor(field, value):
    with pytest.raises((TypeError, ValueError)):
        _phase_reference(**{field: value})


@pytest.mark.parametrize("value", (True, "1", math.nan, math.inf, (1, 2)))
def test_phase_response_rejects_invalid_tangent_coordinate(value):
    with pytest.raises((TypeError, ValueError)):
        observe_antipodal_region_phase_response(
            _phase_reference(), interior=value, bridge=0
        )


def test_phase_response_requires_the_declared_reference_type_and_is_frozen():
    with pytest.raises(TypeError, match="AntipodalRegionPhaseBalance"):
        observe_antipodal_region_phase_response(None, interior=0, bridge=0)
    reference = _phase_reference()
    response = observe_antipodal_region_phase_response(reference, interior=0, bridge=0)
    assert (
        response.energy_before == response.energy_after == response.energy_change == 0
    )
    with pytest.raises(FrozenInstanceError):
        reference.trace = 0
    with pytest.raises(FrozenInstanceError):
        response.energy_after = 1


def test_phase_tangent_action_preserves_exact_rationals_beyond_binary64_range():
    response = observe_antipodal_region_phase_response(
        _phase_reference(),
        interior=F(10**400),
        bridge=F(10**400),
    )
    assert response.after_coherence == (F(10**400), F(5 * 10**400, 3))
    assert response.energy_change == F(16 * 10**800, 9)


def _p3(**overrides):
    values = {
        "nodes": ("left", "center", "right"),
        "neighbors": ((1,), (0, 2), (1,)),
        "capacity": (1, 2, 4),
        "coupling_factor": F(1, 2),
    }
    values.update(overrides)
    return derive_compatible_capacity_balance(**values)


def _prepare(graph=None, *, capacities=None, phases=None):
    graph = nx.path_graph(3) if graph is None else graph
    inject_defaults(graph)
    count = len(graph)
    capacities = (1.0,) * count if capacities is None else capacities
    phases = (0.0,) * count if phases is None else phases
    for node, nu, theta in zip(graph, capacities, phases, strict=True):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.5,
                ALIAS_VF[0]: nu,
                ALIAS_THETA[0]: theta,
                ALIAS_DNFR[0]: 0.0,
                "glyph_history": [],
            }
        )
    for edge in graph.edges:
        graph.edges[edge].setdefault("weight", 1.0)
    return graph


def test_p3_has_the_independent_capacity_means_and_signed_energy_budget():
    result = _p3()
    assert result.nodes == ("left", "center", "right")
    assert result.degrees == (1, 2, 1)
    assert result.components == ((0, 1, 2),)
    assert result.component_means_before == result.component_means_after == (F(9, 4),)
    assert result.component_mean_identity_residuals == (0,)
    assert result.capacity_gradient == (1, F(1, 2), -2)
    assert result.capacity_after == (F(3, 2), F(9, 4), 3)
    assert not result.is_fixed and not result.component_constant
    assert result.energy_before == F(5, 2)
    assert result.energy_after == F(9, 16)
    assert result.drift_term == F(-11, 4)
    assert result.quadratic_term == F(13, 16)
    assert result.energy_change == F(-31, 16)
    assert result.gradient_norm_squared == F(11, 2)
    assert result.strict_drop_upper_bound == F(-11, 8)
    assert result.energy_change < result.strict_drop_upper_bound
    assert result.identity_residual == 0


@pytest.mark.parametrize("gamma", (F(1, 4), F(1, 2), F(3, 4)))
def test_pair_attains_the_dirichlet_drop_bound(gamma):
    result = derive_compatible_capacity_balance(
        nodes=("node", 7),
        neighbors=((1,), (0,)),
        capacity=(0, 2),
        coupling_factor=gamma,
    )
    assert result.energy_before == 2
    assert result.capacity_after == (2 * gamma, 2 * (1 - gamma))
    assert (
        result.energy_change
        == result.strict_drop_upper_bound
        == -8 * gamma * (1 - gamma)
    )
    assert result.component_means_after == (1,)


def test_disconnected_components_preserve_distinct_fixed_capacities():
    result = derive_compatible_capacity_balance(
        nodes=(0, 1, 2, 3),
        neighbors=((1,), (0,), (3,), (2,)),
        capacity=(1, 1, 2, 2),
        coupling_factor=F(1, 2),
    )
    assert result.components == ((0, 1), (2, 3))
    assert result.component_means_before == result.component_means_after == (1, 2)
    assert result.capacity_after == result.capacity
    assert result.is_fixed and result.component_constant
    assert (
        result.energy_change
        == result.energy_before
        == result.strict_drop_upper_bound
        == 0
    )


def test_connected_support_does_not_fix_the_same_nonuniform_capacity_field():
    result = derive_compatible_capacity_balance(
        nodes=(0, 1, 2, 3),
        neighbors=((1,), (0, 2), (1, 3), (2,)),
        capacity=(1, 1, 2, 2),
        coupling_factor=F(1, 2),
    )
    assert result.components == ((0, 1, 2, 3),)
    assert result.component_means_before == result.component_means_after == (F(3, 2),)
    assert result.capacity_after == (1, F(5, 4), F(7, 4), 2)
    assert not result.is_fixed and result.energy_change < 0


def test_zero_constant_capacities_are_an_exact_model_without_an_admission_claim():
    result = _p3(capacity=(0, 0, 0))
    assert result.is_fixed
    assert result.capacity_after == (0, 0, 0)
    assert result.component_means_after == (0,)


def test_neighbor_row_order_is_preserved_without_changing_exact_arithmetic():
    result = _p3(neighbors=((1,), (2, 0), (1,)))
    assert result.neighbors == ((1,), (2, 0), (1,))
    assert result.capacity_after == _p3().capacity_after
    assert result.energy_change == _p3().energy_change


@pytest.mark.parametrize(
    "override",
    (
        {"nodes": ()},
        {"nodes": ("a", "a", "c")},
        {"nodes": {0, 1, 2}},
        {"neighbors": ((1,), (0,), (1,))},
        {"neighbors": ((0, 1), (0, 2), (1,))},
        {"neighbors": ((1, 1), (0, 2), (1,))},
        {"neighbors": ((True,), (0, 2), (1,))},
        {"neighbors": ((3,), (0, 2), (1,))},
        {"neighbors": ((1,), (0,), ())},
        {"neighbors": ((1,), (0, 2))},
        {"neighbors": ({1}, (0, 2), (1,))},
        {"capacity": (1, 2)},
        {"capacity": (1, -1, 2)},
        {"capacity": (1, float("nan"), 2)},
        {"coupling_factor": 0},
        {"coupling_factor": 1},
    ),
)
def test_invalid_exact_support_or_factor_is_rejected(override):
    with pytest.raises((TypeError, ValueError)):
        _p3(**override)


def test_antipodal_triangles_keep_compatible_components_distinct_from_pressure_support():
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from(((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)))
    _prepare(
        graph,
        capacities=(1, 1, 1, 2, 2, 2),
        phases=(0, 0, 0, math.pi, math.pi, math.pi),
    )
    result = observe_coupling_support(graph)
    assert nx.is_connected(graph)
    assert result.components == ((0, 1, 2), (3, 4, 5))
    assert result.excluded_edges == ((2, 3),)
    assert result.blocked_targets == ()
    assert result.compatible_neighbors == (
        (1, 2),
        (0, 2),
        (1, 0),
        (4, 5),
        (3, 5),
        (4, 3),
    )
    assert result.ordered_support_neighbors[2] == (1, 0, 3)
    assert result.ordered_support_neighbors[3] == (4, 5, 2)
    assert result.snapshot.support_neighbors[2] == (0, 1, 3)
    assert result.balance is not None and result.balance.is_fixed
    assert result.balance.component_means_before == (1, 2)
    assert result.balance.energy_before == 0
    assert result.snapshot.capacity_gradient == (0, 0, F(1, 3), F(-1, 3), 0, 0)
    assert result.coupling_factor == F(canonical_glyph_factor_defaults()["UM_vf_sync"])


def test_zero_weight_edges_remain_in_the_actual_compatible_capacity_graph():
    graph = _prepare(capacities=(1, 2, 4))
    graph.edges[1, 2]["weight"] = 0.0
    result = observe_coupling_support(graph)
    assert result.compatible_neighbors == ((1,), (0, 2), (1,))
    assert result.excluded_edges == result.blocked_targets == ()
    assert result.components == ((0, 1, 2),)
    assert result.snapshot.conductance == ((0, 1, F(1)), (1, 0, F(1)))
    assert result.balance.capacity_gradient == (1, F(1, 2), -2)


def test_supported_but_incompatible_neighbor_is_blocked_without_identity_balance():
    graph = _prepare(capacities=(1, 2, 4), phases=(0, 0, math.pi))
    graph.edges[1, 2]["weight"] = 0.0
    result = observe_coupling_support(graph)
    assert result.excluded_edges == ((1, 2),)
    assert result.compatible_neighbors == ((1,), (0,), ())
    assert result.components == ((0, 1), (2,))
    assert result.blocked_targets == (2,)
    assert result.balance is None


def test_a_graph_isolate_is_explicitly_blocked_and_retains_its_original_id():
    graph = nx.Graph()
    graph.add_nodes_from(("first", 17, "isolate"))
    graph.add_edge("first", 17)
    _prepare(graph)
    result = observe_coupling_support(graph)
    assert result.snapshot.nodes == ("first", 17, "isolate")
    assert result.blocked_targets == ("isolate",)
    assert result.components == ((0, 1), (2,))
    assert result.balance is None


def test_capture_requires_enabled_capacity_synchronization_and_an_open_factor():
    graph = _prepare()
    graph.graph["UM_SYNC_VF"] = False
    with pytest.raises(ValueError, match="UM_SYNC_VF"):
        observe_coupling_support(graph)
    graph.graph["UM_SYNC_VF"] = True
    for factor in (0.0, 1.0):
        graph.graph["GLYPH_FACTORS"]["UM_vf_sync"] = factor
        with pytest.raises(ValueError):
            observe_coupling_support(graph)


@pytest.mark.parametrize("kind", (nx.DiGraph, nx.MultiGraph))
def test_directed_or_parallel_support_cannot_enter_the_reciprocal_simple_theorem(kind):
    graph = kind()
    graph.add_edge(0, 1)
    with pytest.raises(ValueError, match="simple undirected"):
        observe_coupling_support(graph)


def test_self_loops_are_rejected_before_an_implicit_identity_neighbor_is_added():
    graph = _prepare()
    graph.add_edge(0, 0)
    with pytest.raises(ValueError, match="loop-free"):
        observe_coupling_support(graph)


def test_asymmetric_materialized_selections_are_rejected(monkeypatch):
    graph = _prepare(nx.path_graph(2))
    original = _phase_gate.resolve_u3_phase_neighbors
    calls = 0

    def asymmetric(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        calls += 1
        return replace(value, neighbors=(), phases=()) if calls == 1 else value

    monkeypatch.setattr(_phase_gate, "resolve_u3_phase_neighbors", asymmetric)
    with pytest.raises(ValueError, match="reciprocal"):
        observe_coupling_support(graph)


def test_capture_does_not_write_graph_or_depend_on_unrelated_pressure_caches():
    graph = _prepare(capacities=(1, 2, 4))
    graph.graph["research_samples"] = {"values": [1, 2]}
    graph.graph["_dnfr_weights"] = {"phase": 99.0, "epi": -7.0}
    before_metadata = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = deepcopy(dict(graph.edges))
    result = observe_coupling_support(graph)
    assert graph.graph == before_metadata
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.edges) == before_edges
    assert result.balance.capacity_gradient == (1, F(1, 2), -2)
    with pytest.raises(FrozenInstanceError):
        result.blocked_targets = ()
    with pytest.raises(FrozenInstanceError):
        result.balance.capacity_after = (0, 0, 0)
