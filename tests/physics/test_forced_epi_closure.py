"""Independent rational geometry controls for a fixed forced EPI projection."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
from unittest.mock import patch

import networkx as nx
import pytest

from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import observe_support_transport


F = Fraction


def _transpose(matrix):
    return tuple(zip(*matrix, strict=True))


def _product(left, right):
    return tuple(tuple(sum((a*b for a, b in zip(row, column, strict=True)), F(0))
                       for column in _transpose(right)) for row in left)


def _apply(matrix, vector):
    return tuple(sum((a*b for a, b in zip(row, vector, strict=True)), F(0))
                 for row in matrix)


def _subtract(left, right):
    return tuple(tuple(a-b for a, b in zip(row, other, strict=True))
                 for row, other in zip(left, right, strict=True))


def _identity(size):
    return tuple(tuple(F(i == j) for j in range(size)) for i in range(size))


def _diagonal(values):
    return tuple(tuple(value if i == j else F(0) for j in range(len(values)))
                 for i, value in enumerate(values))


def _rank(matrix):
    rows = [list(row) for row in matrix]
    pivot = 0
    for column in range(len(rows[0])):
        chosen = next((i for i in range(pivot, len(rows)) if rows[i][column]), None)
        if chosen is None:
            continue
        rows[pivot], rows[chosen] = rows[chosen], rows[pivot]
        scale = rows[pivot][column]
        rows[pivot] = [value / scale for value in rows[pivot]]
        for i in range(len(rows)):
            if i != pivot:
                multiplier = rows[i][column]
                rows[i] = [a-multiplier*b for a, b in zip(rows[i], rows[pivot], strict=True)]
        pivot += 1
        if pivot == len(rows):
            break
    return pivot


def _reference(kind="nonclosed"):
    if kind == "cycle":
        graph = nx.cycle_graph(4)
        capacities = (1, 1, 1, 1)
    elif kind == "weighted":
        graph = nx.cycle_graph(4)
        for edge, weight in zip(graph.edges, (1, 2, 4, 3), strict=True):
            graph.edges[edge]["weight"] = weight
        graph.add_edge(1, 1, weight=2)
        capacities = (F(1, 2), 2, F(3, 2), 3)
    else:
        graph = nx.path_graph(3)
        capacities = (1, 2, 1)
    for i, capacity in zip(graph, capacities, strict=True):
        graph.nodes[i].update(EPI=i-1, nu_f=capacity, delta_nfr=7, theta=0.0)
    forcing = tuple(F(i+1, i+2) for i in graph)
    reference = derive_forced_support_balance(
        observe_support_transport(graph), epi_weight=F(2, 5), forcing=forcing,
    )
    return graph, reference


def _geometry(reference, blocks):
    """Construct every matrix directly from the immutable declared coefficients."""
    source = reference.source
    n, m = len(source.nodes), len(blocks)
    order = {node: i for i, node in enumerate(source.nodes)}
    weights = [[F(0)] * n for _ in range(n)]
    for i, j, weight in source.conductance:
        weights[i][j] += weight
    strengths = tuple(sum(row, F(0)) for row in weights)
    h = tuple(d/nu for d, nu in zip(strengths, source.capacity, strict=True))
    a = tuple(tuple(reference.epi_weight * source.capacity[i] * (
        F(i == j) - weights[i][j]/strengths[i]
    ) for j in range(n)) for i in range(n))
    lift = tuple(tuple(F(node in block) for block in blocks) for node in source.nodes)
    macro_h = tuple(sum((h[order[node]] for node in block), F(0)) for block in blocks)
    r = tuple(tuple(h[i]/macro_h[b] if node in block else F(0)
                    for i, node in enumerate(source.nodes)) for b, block in enumerate(blocks))
    q = _subtract(_identity(n), _product(lift, r))
    raq = _product(_product(r, a), q)
    qap = _product(_product(q, a), lift)
    k0 = _product(raq, qap)
    gram = _product(_product(_transpose(qap), _diagonal(h)), qap)
    force = tuple(nu*f for nu, f in zip(source.capacity, reference.forcing, strict=True))
    return {
        "A": a, "H": h, "P": lift, "R": r, "Q": q, "macro_H": macro_h,
        "RAP": _product(_product(r, a), lift), "QAQ": _product(_product(q, a), q),
        "RAQ": raq, "QAP": qap, "K0": k0, "gram": gram,
        "b": force, "Rb": _apply(r, force), "Qb": _apply(q, force),
        "rank": _rank(raq), "n": n, "m": m,
    }


def _assert_geometry(observation, expected):
    names = {
        "micro_generator": "A", "metric_weights": "H", "lift": "P",
        "projection": "R", "hidden_projector": "Q", "macro_metric_weights": "macro_H",
        "macro_generator": "RAP", "hidden_to_macro": "RAQ", "macro_to_hidden": "QAP",
        "instantaneous_kernel": "K0", "coupling_gram": "gram", "affine_source": "b",
        "projected_source": "Rb", "hidden_affine_source": "Qb",
    }
    for field, oracle in names.items():
        assert getattr(observation, field) == expected[oracle]
    weighted_kernel = _product(_diagonal(expected["macro_H"]), expected["K0"])
    assert observation.weighted_instantaneous_kernel == weighted_kernel == expected["gram"]
    assert weighted_kernel == _transpose(weighted_kernel)
    assert _product(expected["R"], expected["P"]) == _identity(expected["m"])
    assert _product(expected["Q"], expected["Q"]) == expected["Q"]
    assert _product(_diagonal(expected["H"]), expected["Q"]) == _product(
        _transpose(expected["Q"]), _diagonal(expected["H"]),
    )
    assert _rank(expected["QAP"]) == expected["rank"]
    assert observation.all_state_affine_closed == (expected["rank"] == 0)


@pytest.mark.parametrize("kind,blocks", (
    ("nonclosed", ((0, 1), (2,))),
    ("weighted", ((0, 2), (1, 3))),
    ("weighted", ((3,), (0, 1), (2,))),
    ("cycle", ((0, 2), (1, 3))),
))
def test_every_matrix_and_affine_term_matches_independent_rational_construction(kind, blocks):
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference(kind)
    expected = _geometry(reference, blocks)
    with patch("tnfr.physics.epi_memory.matrix_exponential", side_effect=AssertionError("no exponential")):
        observation = observe_forced_support_closure(reference, blocks)
    assert observation.nodes == reference.source.nodes
    assert observation.blocks == blocks
    _assert_geometry(observation, expected)


def test_equitable_projection_is_closed_despite_nonzero_unresolved_affine_drive():
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference("cycle")
    blocks = ((0, 2), (1, 3))
    observation = observe_forced_support_closure(reference, blocks)
    expected = _geometry(reference, blocks)
    assert any(expected["Qb"])
    assert observation.all_state_affine_closed
    assert not observation.lifted_affine_subspace_invariant
    assert observation.witness is None
    assert not any(value for row in expected["RAQ"] for value in row)
    assert not any(value for row in expected["QAP"] for value in row)
    x = (F(2), F(-1), F(3), F(5))
    fine_rate = tuple(b-a for b, a in zip(expected["b"], _apply(expected["A"], x), strict=True))
    macro_rate = tuple(b-a for b, a in zip(
        expected["Rb"], _apply(expected["RAP"], _apply(expected["R"], x)), strict=True,
    ))
    assert _apply(expected["R"], fine_rate) == macro_rate
    # A lifted block-constant state need not stay lifted under this affine force.
    assert any(_apply(expected["Q"], expected["b"]))


@pytest.mark.parametrize("blocks", (
    (), ((0, 1, 2),), ((0,), (1,), (2,)), ((0,), (1,)),
    ((0, 1), (1, 2)), ((0, 0), (1, 2)), ((0, 9), (1, 2)), ((0, 1, 2), ()),
    {frozenset((0, 1)), frozenset((2,))}, {"left": (0, 1), "right": (2,)},
    ({0, 1}, (2,)), ((0, 1), "2"), "invalid",
))
def test_partition_requires_complete_ordered_nonempty_disjoint_proper_blocks(blocks):
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference()
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_closure(reference, blocks)


@pytest.mark.parametrize("epi", ((1, 2), (1, 2, 3, 4), (True, 0, 0),
                                 (float("nan"), 0, 0), (float("inf"), 0, 0),
                                 ("1", 0, 0), (1j, 0, 0), {1, 2, 3}))
def test_explicit_epi_requires_ordered_finite_real_coordinates(epi):
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference()
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_closure(reference, ((0, 1), (2,)), epi=epi)


def test_stale_public_reference_caches_are_rebuilt_from_the_declared_model():
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference()
    blocks = ((0, 1), (2,))
    expected = observe_forced_support_closure(reference, blocks)
    corrupted = replace(
        reference, strengths=(F(100),) * 3, metric_weights=(F(100),) * 3,
        relative_profile=(F(100),) * 3, mean_drift=F(100),
        profile_residual=(F(100),) * 3, has_zero_pressure_equilibrium=True,
        source=replace(reference.source, epi_gradient=(F(100),) * 3,
                       rate=(F(100),) * 3, dirichlet_energy=F(100)),
    )
    assert observe_forced_support_closure(corrupted, blocks) == expected
    with pytest.raises(TypeError):
        observe_forced_support_closure(asdict(reference), blocks)


def test_invalid_primitive_reference_coefficients_cannot_hide_behind_valid_caches():
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference()
    blocks = ((0, 1), (2,))
    invalid = (
        replace(reference, epi_weight=F(0)),
        replace(reference, forcing=(F(0),)),
        replace(reference, source=replace(reference.source, capacity=(F(1), F(0), F(1)))),
        replace(reference, source=replace(reference.source, conductance=reference.source.conductance[:-1])),
    )
    for value in invalid:
        with pytest.raises((TypeError, ValueError)):
            observe_forced_support_closure(value, blocks)


def test_detached_geometry_is_immutable_and_does_not_write_or_follow_graph_state():
    from tnfr.physics.epi_memory import observe_forced_support_closure

    graph, reference = _reference()
    blocks = [[0, 1], [2]]
    epi = [F(2), F(-1), F(3)]
    before = deepcopy((graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))))
    observation = observe_forced_support_closure(reference, blocks, epi=epi)
    retained = asdict(observation)
    assert before == (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    epi[0] = F(99)
    blocks[0].reverse()
    graph.nodes[0]["nu_f"] = 99
    assert asdict(observation) == retained
    with pytest.raises(FrozenInstanceError):
        observation.all_state_affine_closed = True


@pytest.mark.parametrize("kind,blocks", (
    ("nonclosed", ((0, 1), (2,))), ("weighted", ((0, 2), (1, 3))),
))
def test_hidden_witness_has_same_exact_observation_and_distinct_forced_model_rate(kind, blocks):
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference(kind)
    x = tuple(F(2*i-3, i+1) for i in range(len(reference.source.nodes)))
    result = observe_forced_support_closure(reference, blocks, epi=x)
    geometry = _geometry(reference, blocks)
    witness = result.witness
    assert not result.all_state_affine_closed and witness is not None
    before, after = witness.before_epi, witness.after_epi
    assert before == x
    delta = tuple(b-a for a, b in zip(before, after, strict=True))
    assert delta == witness.hidden_delta == _apply(geometry["Q"], delta)
    assert witness.common_macro == _apply(geometry["R"], before) == _apply(geometry["R"], after)

    def projected_rate(values):
        return _apply(geometry["R"], tuple(b-a for b, a in zip(
            geometry["b"], _apply(geometry["A"], values), strict=True,
        )))

    assert witness.before_projected_rate == projected_rate(before)
    assert witness.after_projected_rate == projected_rate(after)
    difference = tuple(b-a for a, b in zip(projected_rate(before), projected_rate(after), strict=True))
    assert difference == witness.projected_rate_difference
    assert any(difference)
    assert difference == tuple(-value for value in _apply(geometry["RAQ"], delta))
    assert "need not satisfy a graph chart" in result.scope


def test_current_model_centering_and_rates_keep_nonzero_forcing_explicit():
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference("weighted")
    x = (F(5, 3), F(-2), F(7, 5), F(0))
    blocks = ((0, 2), (1, 3))
    result = observe_forced_support_closure(reference, blocks, epi=x)
    g = _geometry(reference, blocks)
    mean = sum(h*value for h, value in zip(g["H"], x, strict=True))/sum(g["H"])
    error = tuple(value-mean-z for value, z in zip(x, reference.relative_profile, strict=True))
    fine_rate = tuple(b-a for b, a in zip(g["b"], _apply(g["A"], x), strict=True))
    mean_rate = sum(h*value for h, value in zip(g["H"], fine_rate, strict=True))/sum(g["H"])
    assert result.current_mean == mean
    assert result.current_mean_rate == mean_rate == reference.mean_drift != 0
    assert result.current_relative_error == error
    assert result.projected_relative_error == _apply(g["R"], error)
    assert result.projected_epi == _apply(g["R"], x)
    assert result.hidden_epi == _apply(g["Q"], x)
    assert result.projected_nodal_rate == _apply(g["R"], fine_rate)
    assert result.current_relative_rate == tuple(value-mean_rate for value in fine_rate)
    assert result.current_relative_rate == tuple(-value for value in _apply(g["A"], error))
    assert result.centered_rate_residual == (0,) * len(x)
    assert result.projected_nodal_rate == tuple(a+b for a, b in zip(
        result.affine_macro_rate, result.hidden_rate_contribution, strict=True,
    ))
    changed = replace(reference, source=replace(reference.source, stored_pressure=(F(999),)*4))
    second = observe_forced_support_closure(changed, blocks, epi=x)
    for field in ("affine_source", "projected_source", "hidden_affine_source",
                  "projected_nodal_rate", "current_relative_rate", "witness"):
        assert getattr(second, field) == getattr(result, field)


def test_a_nonclosed_partition_can_have_zero_hidden_contribution_at_one_state():
    from tnfr.physics.epi_memory import observe_forced_support_closure

    _, reference = _reference()
    result = observe_forced_support_closure(reference, ((0, 1), (2,)), epi=(F(3),)*3)
    assert result.hidden_epi == (0, 0, 0)
    assert result.hidden_rate_contribution == (0, 0)
    assert not result.all_state_affine_closed and result.witness is not None
