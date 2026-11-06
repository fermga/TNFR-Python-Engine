"""Independent regional energy/work checks on the complete sine-law state."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import (
    SineExchangeComparison,
    bound_relational_sine_exchange,
)


def _source(*, zero_loss=False, capacities=(0, 2, 1, 3, 1)):
    graph = nx.cycle_graph(5)
    graph.add_edge(0, 2)
    forms = (Q(-3, 2), 2, Q(1, 3), -1, 4)
    phases = (Q(3, 2), Q(-1, 4), Q(5, 3), -2, Q(1, 2))
    for i in graph:
        graph.nodes[i].update(EPI=forms[i], theta=phases[i], nu_f=capacities[i])
    graph.graph["GAMMA"] = {"type": "none"}
    model = RelationalExchangeModel(
        Q(5, 4),
        epi_weight=0 if zero_loss else Q(1, 4),
        phase_weight=1 if zero_loss else Q(3, 4),
        phase_domain="regular",
    )
    return bound_relational_sine_exchange(graph, reference_model=model)


@pytest.fixture(scope="module")
def source():
    return _source()


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    assert bound.contains(Q(str(value))), (bound, value)


@pytest.mark.parametrize("zero_loss", [False, True])
def test_disjoint_storage_and_signed_work_match_independent_complete_rows(zero_loss):
    source = _source(zero_loss=zero_loss)
    report = source.regional_storage_balance(region=(2, 0))
    assert report.region_indices == (2, 0)
    assert report.complement_indices == (1, 3, 4)
    assert report.clock == "structural_t"
    with mp.workdps(95):
        x, theta, nu = (
            tuple(map(_mp, row)) for row in (source.epi, source.phase, source.capacity)
        )
        e, w = map(_mp, source.reference_model.effective_weights)
        beta = _mp(source.reference_model.storage_scale)
        neighbors = [set() for _ in x]
        for i, j in source.edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
        q = [sum(x[i] - x[j] for j in row) for i, row in enumerate(neighbors)]
        currents = [
            sum(mp.sin(theta[j] - theta[i]) for j in row)
            for i, row in enumerate(neighbors)
        ]
        fx = [
            nu[i] / len(row) * (-e * q[i] + w * currents[i] / mp.pi)
            for i, row in enumerate(neighbors)
        ]
        ft = [
            w * nu[i] * q[i] / (beta * mp.pi * len(row))
            for i, row in enumerate(neighbors)
        ]
        loss = [e * nu[i] * q[i] ** 2 / len(row) for i, row in enumerate(neighbors)]
        for prefix, edges in (
            ("regional", report.regional_edge_indices),
            ("complement", report.complement_edge_indices),
            ("boundary", report.boundary_edge_indices),
        ):
            form = sum((x[j] - x[i]) ** 2 / 2 for i, j in edges)
            phase = sum(1 - mp.cos(theta[j] - theta[i]) for i, j in edges)
            rate = sum(
                (x[j] - x[i]) * (fx[j] - fx[i])
                + beta * mp.sin(theta[j] - theta[i]) * (ft[j] - ft[i])
                for i, j in edges
            )
            assert abs(_mp(getattr(report, prefix + "_form_storage")) - form) < mp.mpf(
                "1e-90"
            )
            _contains(getattr(report, prefix + "_phase_storage"), phase)
            _contains(getattr(report, prefix + "_storage"), form + beta * phase)
            _contains(getattr(report, "direct_" + prefix + "_storage_rate"), rate)
        for prefix, pairs, indices in (
            ("regional", report.boundary_edge_indices, report.region_indices),
            (
                "complement",
                tuple((j, i) for i, j in report.boundary_edge_indices),
                report.complement_indices,
            ),
        ):
            form_power = sum((x[j] - x[i]) * fx[i] for i, j in pairs)
            phase_power = sum(
                beta * mp.sin(theta[j] - theta[i]) * ft[i] for i, j in pairs
            )
            _contains(getattr(report, prefix + "_boundary_form_work"), form_power)
            _contains(getattr(report, prefix + "_boundary_phase_work"), phase_power)
            _contains(
                getattr(report, prefix + "_boundary_work"), form_power + phase_power
            )
            assert abs(
                _mp(getattr(report, prefix + "_loss")) - sum(loss[i] for i in indices)
            ) < mp.mpf("1e-90")
    for field in (
        "regional_balance_residual",
        "complement_balance_residual",
        "boundary_balance_residual",
        "storage_partition_residual",
        "global_balance_residual",
    ):
        assert getattr(report, field).contains(0)
        # Shared resultant admission can retain wider certified trigonometric
        # enclosures than the subsequent rational arithmetic grid.
        assert getattr(report, field).abs_max < Q(1, 10**12)
    # There is actual cross-edge storage change: inward powers are not an
    # invented pair of equal-and-opposite transfers between internal stores.
    assert not report.direct_boundary_storage_rate.contains(0)
    assert not (
        report.regional_boundary_work + report.complement_boundary_work
    ).contains(0)
    assert report.regional_balance_residual.width > 0
    assert report.comparison.form_rates[0] == I(0)
    assert report.comparison.phase_rates[0] == I(0)


def test_phase_flat_environment_preparation_retains_cross_edge_storage():
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    for cycle in (tuple(range(5)), tuple(range(5, 10))):
        graph.add_edges_from(zip(cycle, cycle[1:] + cycle[:1]))
    graph.add_edges_from(((0, 10), (10, 5), (1, 6)))
    for i in graph:
        graph.nodes[i].update(
            EPI=24 if i == 10 else -24 if i == 1 else 0, theta=0, nu_f=1
        )
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    report = source.regional_storage_balance(region=tuple(range(5, 10)))
    assert report.regional_storage == I(0)
    assert report.complement_storage == I(864)
    assert report.boundary_storage == I(576)
    assert report.comparison.storage == I(1440)
    assert report.regional_boundary_work == I(0)
    assert report.regional_loss == report.complement_loss == 0
    assert report.direct_regional_storage_rate == I(0)
    assert any(rate.abs_max > 0 for rate in report.comparison.phase_rates)


def test_complement_exchange_and_whole_support_do_not_double_count(source):
    original = source.regional_storage_balance(region=(2, 0))
    reverse = source.regional_storage_balance(region=(1, 3, 4))
    assert original.regional_storage == reverse.complement_storage
    assert original.complement_storage == reverse.regional_storage
    assert original.boundary_storage == reverse.boundary_storage
    assert original.regional_boundary_work == reverse.complement_boundary_work
    assert original.complement_boundary_work == reverse.regional_boundary_work
    assert original.regional_loss == reverse.complement_loss
    all_nodes = source.regional_storage_balance(region=source.nodes)
    assert all_nodes.boundary_edge_indices == all_nodes.complement_edge_indices == ()
    assert all_nodes.regional_form_storage == source.form_storage
    assert all_nodes.storage_partition_residual.contains(0)
    assert all_nodes.regional_loss == source.continuous_loss
    for field in (
        "boundary_storage",
        "complement_storage",
        "regional_boundary_work",
        "complement_boundary_work",
    ):
        assert getattr(all_nodes, field) == I(0)


def test_cached_fields_do_not_enter_regional_calculation_or_retained_projection(source):
    poisoned = replace(
        source,
        form_gradient=(999,) * 5,
        relative_resultant=((I(999), I(999)),) * 5,
        pressure=(I(999),) * 5,
        form_rates=(I(999),) * 5,
        phase_rates=(I(999),) * 5,
        form_storage=Q(999),
        phase_storage=I(999),
        storage=I(999),
        dissipation=(Q(999),) * 5,
        continuous_loss=Q(999),
    )
    assert poisoned.regional_storage_balance(
        region=(2, 0)
    ) == source.regional_storage_balance(region=(2, 0))


@pytest.mark.parametrize("region", [(), (0, 0), (12,), {0, 1}, "01"])
def test_invalid_region_rejected(source, region):
    with pytest.raises((TypeError, ValueError)):
        source.regional_storage_balance(region=region)


@pytest.mark.parametrize(
    "changes",
    [
        {"epi": (True, 2, 3, 4, 5)},
        {"phase": (float("nan"), 0, 0, 0, 0)},
        {"capacity": (-1, 1, 1, 1, 1)},
        {"capacity": (False, 1, 1, 1, 1)},
        {"degrees": (2,) * 5},
        {"edges": ((0, 1), (1, 2))},
        {"law": "native"},
    ],
)
def test_invalid_source_primitives_are_not_repaired_by_cached_fields(source, changes):
    with pytest.raises((TypeError, ValueError)):
        replace(source, **changes).regional_storage_balance(region=(0, 2))


def test_unsupported_source_and_invalid_authoritative_model_rejected(source):
    with pytest.raises(TypeError, match="exact SineExchangeComparison"):
        SineExchangeComparison.regional_storage_balance({}, region=(0,))
    model = replace(source.reference_model)
    object.__setattr__(model, "epi_weight", False)
    with pytest.raises((TypeError, ValueError)):
        replace(source, reference_model=model).regional_storage_balance(region=(0,))


def test_direct_projection_is_detached_and_preserves_clock_and_partition(source):
    report = source.regional_storage_balance(region=(2, 0))
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-regional-storage-balance.v1"
    assert payload["report"]["clock"] == "structural_t"
    payload["report"]["region"].append(99)
    assert report.region == (2, 0)
