"""Exact local transport and independent global contrast-bound controls."""

from copy import copy
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import analyze_sine_conservative_phase_transport


def _source(*, order=(0, 1, 2), form=(1, 0, -1), phase=0):
    graph = nx.Graph()
    graph.add_nodes_from(order)
    graph.add_edges_from(((0, 1), (1, 2)))
    for i in graph:
        graph.nodes[i].update(EPI=form[i], theta=phase, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def report(source):
    return analyze_sine_conservative_phase_transport(
        source, contrasts=((-1, 0, 1), (1, -2, 1))
    )


def test_path_eigenmodes_and_graph_correlation_bounds_are_distinct(report):
    # On P3, (1,0,-1) is an eigenvector of K L of eigenvalue1. Its
    # acceleration and third phase derivative follow from the complete rows.
    assert report.clock == "tau=t/pi"
    assert report.initial_phase_velocity == (1, 0, -1)
    assert report.initial_phase_acceleration == (0, 0, 0)
    assert report.initial_form_acceleration == (-1, 0, 1)
    assert report.initial_phase_jerk == (-1, 0, 1)
    assert report.contrast_initial_velocity == (-2, 0)
    assert report.contrast_initial_jerk == (2, 0)
    assert report.contrast_current_weights == ((-1, 0, 1), (2, -2, 2))
    assert report.contrast_edge_current_coefficients == ((-1, -1), (4, -4))
    assert report.contrast_acceleration_bounds == (2, 8)
    assert report.contrast_quadratic_remainder_coefficients == (1, 4)
    # Zero initial derivatives do not collapse a bound over arbitrary phases.
    assert report.contrast_initial_jerk[1] == 0 < report.contrast_acceleration_bounds[1]


def test_full_matrices_give_all_initial_derivatives_without_cached_rates():
    graph = nx.Graph(((0, 1), (1, 2), (2, 3), (1, 3)))
    x = tuple(map(Q, (3, -2, 1, 4)))
    for node in graph:
        graph.nodes[node].update(EPI=x[node], theta=Q(3, 7), nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )
    contrasts = ((1, -1, 0, 0), (Q(1, 3), 0, Q(-2, 3), Q(1, 3)))
    report = analyze_sine_conservative_phase_transport(source, contrasts=contrasts)
    # Construct independent dense L and K, not the implementation's edge action.
    adjacency = ((1,), (0, 2, 3), (1, 3), (1, 2))
    degree = tuple(map(len, adjacency))
    matrix = tuple(
        tuple(
            Q(int(i == j) * degree[i] - int(j in adjacency[i]), degree[i])
            for j in range(4)
        )
        for i in range(4)
    )

    def multiply(vector):
        return tuple(
            sum((row[j] * vector[j] for j in range(4)), Q(0)) for row in matrix
        )

    ax = multiply(x)
    aa = multiply(ax)
    aaa = multiply(aa)
    assert report.initial_phase_velocity == ax
    assert report.initial_form_acceleration == tuple(-v for v in aa)
    assert report.initial_phase_jerk == tuple(-v for v in aaa)
    for index, c in enumerate(contrasts):
        weights = tuple(
            sum((c[i] * matrix[i][j] for i in range(4)), Q(0)) / degree[j]
            for j in range(4)
        )
        assert report.contrast_current_weights[index] == weights
        assert report.contrast_initial_jerk[index] == -sum(
            c[i] * aaa[i] for i in range(4)
        )
        assert report.contrast_acceleration_bounds[index] == sum(
            abs(weights[i] - weights[j]) for i, j in graph.edges
        )


def test_global_bound_agrees_with_full_currents_at_unprepared_phase_states(report):
    # These static probes leave the initial common-phase state. The theorem's
    # bound remains valid there, while its initial jerk is not extrapolated.
    phases = ((0, Q(1, 3), Q(-4, 3)), (Q(1, 2), -2, 3), (0, 4, -3))
    with mp.workdps(70):
        for theta in phases:
            theta = tuple(mp.mpf(Q(v).numerator) / Q(v).denominator for v in theta)
            currents = (
                mp.sin(theta[1] - theta[0]),
                mp.sin(theta[0] - theta[1]) + mp.sin(theta[2] - theta[1]),
                mp.sin(theta[1] - theta[2]),
            )
            form_rate = (currents[0], currents[1] / 2, currents[2])
            acceleration = (
                form_rate[0] - form_rate[1],
                form_rate[1] - (form_rate[0] + form_rate[2]) / 2,
                form_rate[2] - form_rate[1],
            )
            for c, coefficients, bound in zip(
                report.contrasts,
                report.contrast_edge_current_coefficients,
                report.contrast_acceleration_bounds,
            ):
                projected = sum(
                    mp.mpf(v.numerator) / v.denominator * a
                    for v, a in zip(c, acceleration)
                )
                edge_sum = sum(
                    mp.mpf(v.numerator) / v.denominator * mp.sin(theta[j] - theta[i])
                    for v, (i, j) in zip(coefficients, report.edge_indices)
                )
                assert abs(projected - edge_sum) < mp.mpf("1e-65")
                assert abs(projected) <= mp.mpf(bound.numerator) / bound.denominator


def test_represented_scalars_and_tiny_nonzero_rationals_remain_exact(source):
    tiny = Q(1, 2**1200)
    report = analyze_sine_conservative_phase_transport(
        source, contrasts=((0.5, 0, -0.5), (-tiny, 0, tiny))
    )
    assert report.contrasts == ((Q(1, 2), Q(0), Q(-1, 2)), (-tiny, 0, tiny))
    assert report.contrast_initial_velocity == (1, -2 * tiny)
    assert report.contrast_acceleration_bounds == (1, 2 * tiny)


def test_common_origins_and_node_order_preserve_declared_contrasts(source, report):
    shifted = replace(
        source, epi=tuple(v + Q(7, 3) for v in source.epi), phase=(Q(5, 3),) * 3
    )
    actual = analyze_sine_conservative_phase_transport(
        shifted, contrasts=report.contrasts
    )
    reversed_report = analyze_sine_conservative_phase_transport(
        _source(order=(2, 1, 0)),
        contrasts=tuple(tuple(reversed(row)) for row in report.contrasts),
    )
    for field in (
        "initial_phase_velocity",
        "initial_phase_acceleration",
        "initial_phase_jerk",
        "initial_form_acceleration",
        "contrast_initial_velocity",
        "contrast_initial_jerk",
        "contrast_acceleration_bounds",
    ):
        assert getattr(actual, field) == getattr(report, field)
    assert reversed_report.initial_phase_velocity == tuple(
        reversed(report.initial_phase_velocity)
    )
    assert reversed_report.contrast_initial_jerk == report.contrast_initial_jerk
    assert (
        reversed_report.contrast_acceleration_bounds
        == report.contrast_acceleration_bounds
    )


def test_poisoned_derived_fields_are_not_derivative_evidence(source, report):
    poisoned = replace(
        source,
        form_gradient=(999,) * 3,
        degrees=source.degrees,
        form_rates=(I(999),) * 3,
        phase_rates=(I(999),) * 3,
        pressure=(I(999),) * 3,
        storage=I(999),
    )
    actual = analyze_sine_conservative_phase_transport(
        poisoned, contrasts=report.contrasts
    )
    for field in fields(report):
        if field.name != "source":
            assert getattr(actual, field.name) == getattr(report, field.name)


@pytest.mark.parametrize(
    "contrasts",
    [
        (),
        ((0, 0, 0),),
        ((1, 0, 0),),
        ((1, -1),),
        ((1, -1, 0, 0),),
        ((True, -1, 0),),
        ((False, 0, 0),),
        ((float("nan"), -1, 0),),
        ((float("inf"), -1, 0),),
        (("1", -1, 0),),
        ((0.1, 0.2, -0.3),),
        {(-1, 0, 1)},
        "-101",
        ({-1, 0, 1},),
        (None,),
    ],
)
def test_invalid_contrasts_fail_before_graph_computation(source, contrasts):
    with pytest.raises((TypeError, ValueError)):
        analyze_sine_conservative_phase_transport(source, contrasts=contrasts)


@pytest.mark.parametrize(
    "changes",
    [
        {"epi": (True, 0, -1)},
        {"phase": (0, Q(1, 100), 0)},
        {"phase": (False, 0, 0)},
        {"capacity": (True, 1, 1)},
        {"capacity": (0, 1, 1)},
        {"capacity": (2, 1, 1)},
        {"degrees": (1, 1, 1)},
        {"law": "native"},
        {"edges": ((0, 1),)},
    ],
)
def test_invalid_or_unsupported_source_is_not_repaired(source, changes):
    with pytest.raises((TypeError, ValueError)):
        analyze_sine_conservative_phase_transport(
            replace(source, **changes), contrasts=((-1, 0, 1),)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("epi_weight", Q(1, 4)),
        ("epi_weight", False),
        ("phase_weight", Q(1, 2)),
        ("storage_scale", 2),
        ("phase_domain", "acute"),
    ],
)
def test_complete_model_is_admitted_before_identity_comparison(source, field, value):
    model = copy(source.reference_model)
    object.__setattr__(model, field, value)
    with pytest.raises((TypeError, ValueError)):
        analyze_sine_conservative_phase_transport(
            replace(source, reference_model=model), contrasts=((-1, 0, 1),)
        )


def test_unsupported_source_and_mutation_rejected(report):
    with pytest.raises(TypeError, match="exact SineExchangeComparison"):
        analyze_sine_conservative_phase_transport({}, contrasts=((-1, 0, 1),))
    with pytest.raises(FrozenInstanceError):
        report.clock = "seconds"


def test_direct_projection_preserves_exact_data_and_is_detached(report):
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.sine-conservative-phase-transport.v1"
    assert payload["report"]["clock"] == "tau=t/pi"
    payload["report"]["contrasts"][0].append(99)
    assert report.contrasts[0] == (-1, 0, 1)
