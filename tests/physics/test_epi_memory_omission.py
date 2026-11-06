"""Independent exact controls for a reversible memory-omission budget.

The three-node family has a two-mode closed-form solution. No reference
trajectory, numerical integrator, matrix exponential or frozen producer runs.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F

import mpmath as mp
import networkx as nx
import numpy as np
import pytest

from tnfr.physics import epi_memory as owner
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import observe_support_transport

BLOCKS = ((0, 1), (2,))


@pytest.fixture(autouse=True)
def no_flow_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("memory budget must not evaluate a trajectory or exponential")

    monkeypatch.setattr(owner, "matrix_exponential", forbidden)
    monkeypatch.setattr(owner, "euler_update", forbidden)


def _reference(delta=F(1, 2), epi_weight=F(1), forcing=None):
    graph = nx.Graph()
    graph.add_weighted_edges_from(((0, 1, F(1)), (0, 2, 1 + delta), (1, 2, 1 - delta)))
    for node in graph:
        strength = sum((graph.edges[node, j]["weight"] for j in graph[node]), F(0))
        graph.nodes[node].update(EPI=0, nu_f=strength, theta=0, delta_nfr=0)
    return graph, derive_forced_support_balance(
        observe_support_transport(graph),
        epi_weight=epi_weight,
        forcing=(F(0),) * 3 if forcing is None else forcing,
    )


def _bound(reference, epi, horizon=F(1, 4), decay=F(3)):
    return owner.bound_epi_memory_omission(
        reference, BLOCKS, epi=epi, horizon=horizon, hidden_decay_lower=decay
    )


@pytest.mark.parametrize("delta", (F(0), F(1, 8), F(1, 4), F(1, 2)))
def test_exact_family_geometry_and_quadratic_feedback(delta):
    _, reference = _reference(delta)
    result = _bound(reference, (F(1), F(1), F(-2)))

    assert result.status == "bounded" and not result.unavailable_reasons
    assert result.closure.metric_weights == (F(1),) * 3
    assert result.closure.macro_metric_weights == (F(2), F(1))
    assert result.closure.macro_to_hidden == (
        (delta, -delta),
        (-delta, delta),
        (F(0), F(0)),
    )
    assert result.coupling_squared_upper == 3 * delta**2
    assert result.hidden_decay_slack == ((F(0),) * 3,) * 3
    assert result.initial_relative_norm_squared == 6
    assert result.initial_hidden_norm_squared == 0
    assert result.initial_source_error_squared_upper == 0
    assert result.feedback_error_squared_upper == F(27, 512) * delta**4
    assert result.error_squared_upper == result.feedback_error_squared_upper


@pytest.mark.parametrize("delta", (F(1, 8), F(1, 2)))
def test_hidden_initialization_is_visible_at_first_order(delta):
    _, reference = _reference(delta)
    hidden = _bound(reference, (F(1), F(-1), F(0)))
    zero = _bound(reference, (F(0),) * 3)

    assert hidden.closure.projected_epi == zero.closure.projected_epi == (F(0), F(0))
    assert hidden.closure.projected_nodal_rate == (-delta, 2 * delta)
    assert zero.closure.projected_nodal_rate == (F(0), F(0))
    assert hidden.initial_hidden_norm_squared == 2
    assert hidden.initial_source_error_squared_upper == F(3, 8) * delta**2
    assert hidden.error_squared_upper == 2 * (
        hidden.initial_source_error_squared_upper + hidden.feedback_error_squared_upper
    )
    assert zero.error_squared_upper == 0


@pytest.mark.parametrize("time", (F(0), F(1, 4), F(1), F(2)))
@pytest.mark.parametrize("hidden", (False, True))
def test_budget_encloses_independent_two_mode_solution(time, hidden):
    delta = F(1, 2)
    _, reference = _reference(delta)
    initial = (F(1), F(-1), F(0)) if hidden else (F(1), F(1), F(-2))
    result = _bound(reference, initial, horizon=time)
    with mp.workdps(70):
        t = mp.mpf(time.numerator) / time.denominator
        rate = mp.sqrt(3) / 2
        # The orthonormal relative block is [[3,sqrt(3)*delta],[sqrt(3)*delta,3]].
        if hidden:
            squared_error = 2 * mp.exp(-6 * t) * mp.sinh(rate * t) ** 2
        else:
            squared_error = 6 * mp.exp(-6 * t) * (mp.cosh(rate * t) - 1) ** 2
        upper = result.error_squared_upper
        assert squared_error <= mp.mpf(upper.numerator) / upper.denominator


def test_invalid_hidden_gap_has_no_certified_error_or_gap_search():
    _, reference = _reference()
    result = _bound(reference, (F(1), F(1), F(-2)), decay=F(301, 100))

    assert not result.hidden_decay_certified
    assert result.hidden_decay_lower == F(301, 100)
    assert result.status == "unavailable"
    assert result.unavailable_reasons == ("hidden_decay_lower_not_certified",)
    assert result.initial_source_error_squared_upper is None
    assert result.feedback_error_squared_upper is None
    assert result.error_squared_upper is None
    assert result.hidden_decay_slack[0][0] < 0


def test_uniform_origin_clock_and_node_order_do_not_change_the_error_contract():
    graph, reference = _reference()
    initial = (F(1), F(1), F(-2))
    base = _bound(reference, initial)
    shifted = _bound(reference, tuple(value + 1000 for value in initial))
    _, faster = _reference(epi_weight=F(2))
    rescaled = _bound(faster, initial, horizon=F(1, 8), decay=F(6))
    reordered = nx.relabel_nodes(graph, {0: "left", 1: "right", 2: "port"})
    reordered_reference = derive_forced_support_balance(
        observe_support_transport(reordered), epi_weight=1, forcing=(0, 0, 0)
    )
    reordered_bound = owner.bound_epi_memory_omission(
        reordered_reference,
        (("port",), ("right", "left")),
        epi=initial,
        horizon=F(1, 4),
        hidden_decay_lower=3,
    )

    assert shifted.initial_relative_norm_squared == base.initial_relative_norm_squared
    assert (
        shifted.error_squared_upper
        == rescaled.error_squared_upper
        == base.error_squared_upper
    )
    assert reordered_bound.error_squared_upper == base.error_squared_upper
    assert reordered_bound.coupling_squared_upper == base.coupling_squared_upper


def test_tiny_exact_preparation_and_fixed_width_payloads_are_not_materialized():
    _, reference = _reference()
    base = _bound(reference, (F(1), F(1), F(-2)))
    scale = F(1, 2**300)
    tiny = _bound(reference, (scale, scale, -2 * scale))
    integer_payload = F(np.int64(2**32), np.int64(3))
    large = _bound(reference, (integer_payload, integer_payload, -2 * integer_payload))

    assert tiny.error_squared_upper == base.error_squared_upper * scale**2 > 0
    assert large.error_squared_upper == base.error_squared_upper * F(2**32, 3) ** 2
    assert type(large.error_squared_upper.numerator) is int


def test_reference_is_readmitted_and_graph_state_is_not_modified():
    graph, reference = _reference()
    before = deepcopy(
        (graph.graph, dict(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )
    initial = (F(1), F(1), F(-2))
    expected = _bound(reference, initial)
    stale = replace(
        reference,
        metric_weights=(F(100),) * 3,
        mean_drift=F(100),
        relative_profile=(F(100),) * 3,
    )
    assert _bound(stale, initial) == expected
    assert (
        graph.graph,
        dict(graph.nodes(data=True)),
        tuple(graph.edges(data=True)),
    ) == before
    with pytest.raises((TypeError, ValueError)):
        _bound(
            replace(
                reference, source=replace(reference.source, capacity=(F(1), F(0), F(1)))
            ),
            initial,
        )


def test_nonzero_forcing_is_not_silently_dropped():
    _, reference = _reference(forcing=(F(1), F(-1), F(0)))
    with pytest.raises(ValueError, match="unforced"):
        _bound(reference, (F(0),) * 3)


def _anchored(reference, epi, horizon=F(1, 4), decay=F(3)):
    return owner.bound_anchored_epi_memory_omission(
        reference, BLOCKS, epi=epi, horizon=horizon, hidden_decay_lower=decay
    )


def test_anchored_forcing_preserves_a_drifting_profile_despite_nonclosure():
    _, reference = _reference(forcing=(F(9, 5), F(-1), F(0)))
    initial = (F(1), F(-1), F(0))
    result = _anchored(reference, initial)

    assert reference.relative_profile == initial
    assert reference.mean_drift == 1
    assert not result.closure.all_state_affine_closed
    assert result.comparison_kind == "profile_anchored_affine"
    assert result.closure.projected_epi == (F(0), F(0))
    assert result.closure.projected_source == (F(3, 2), F(0))
    assert result.forcing_correction == (F(1, 2), F(-1))
    assert result.macro_forcing == (F(1), F(1))
    assert result.closure.projected_nodal_rate == result.macro_forcing
    assert result.closure.affine_macro_rate != result.macro_forcing
    correction_rate = tuple(
        sum(a * b for a, b in zip(row, result.forcing_correction, strict=True))
        for row in result.closure.macro_generator
    )
    assert correction_rate == tuple(3 * value for value in result.forcing_correction)
    assert result.closure.hidden_epi == initial
    assert result.initial_hidden_norm_squared == 0
    assert result.initial_relative_norm_squared == 0
    assert result.error_squared_upper == 0


@pytest.mark.parametrize("time", (F(0), F(1, 4), F(1), F(2)))
@pytest.mark.parametrize("hidden", (False, True))
def test_anchored_transient_budget_matches_independent_forced_solution(time, hidden):
    _, reference = _reference(forcing=(F(9, 5), F(-1), F(0)))
    residual = (F(1), F(-1), F(0)) if hidden else (F(1), F(1), F(-2))
    initial = tuple(
        a + b for a, b in zip(reference.relative_profile, residual, strict=True)
    )
    result = _anchored(reference, initial, horizon=time)
    with mp.workdps(70):
        t = mp.mpf(time.numerator) / time.denominator
        if hidden:
            squared_error = 2 * mp.exp(-6 * t) * mp.sinh(mp.sqrt(3) * t / 2) ** 2
        else:
            squared_error = 6 * mp.exp(-6 * t) * (mp.cosh(mp.sqrt(3) * t / 2) - 1) ** 2
        upper = result.error_squared_upper
        assert squared_error <= mp.mpf(upper.numerator) / upper.denominator


def test_anchor_preserves_mean_drift_and_linearity_in_the_held_input():
    first_input = (F(9, 5), F(-1), F(0))
    second_input = (F(0), F(0), F(1))
    _, first = _reference(forcing=first_input)
    _, second = _reference(forcing=second_input)
    _, combined = _reference(
        forcing=tuple(a + b for a, b in zip(first_input, second_input, strict=True))
    )
    reports = [_anchored(ref, (F(0),) * 3) for ref in (first, second, combined)]
    assert reports[2].macro_forcing == tuple(
        a + b
        for a, b in zip(reports[0].macro_forcing, reports[1].macro_forcing, strict=True)
    )
    assert reports[2].forcing_correction == tuple(
        a + b
        for a, b in zip(
            reports[0].forcing_correction, reports[1].forcing_correction, strict=True
        )
    )
    for report in reports:
        masses = report.closure.macro_metric_weights
        total = sum(
            mass * force
            for mass, force in zip(masses, report.macro_forcing, strict=True)
        )
        assert total == sum(masses) * report.closure.reference.mean_drift


def test_anchor_retains_full_clock_scaling_and_uniform_rate_source():
    forcing = (F(9, 5), F(-1), F(0))
    _, reference = _reference(forcing=forcing)
    initial = (F(2), F(0), F(-2))
    base = _anchored(reference, initial)
    _, faster = _reference(epi_weight=F(2), forcing=tuple(2 * f for f in forcing))
    rescaled = _anchored(faster, initial, horizon=F(1, 8), decay=F(6))
    assert rescaled.error_squared_upper == base.error_squared_upper
    assert rescaled.macro_forcing == tuple(2 * f for f in base.macro_forcing)
    _, drifting = _reference(
        forcing=tuple(
            f + F(2) / nu
            for f, nu in zip(forcing, reference.source.capacity, strict=True)
        )
    )
    shifted = _anchored(drifting, initial)
    assert shifted.error_squared_upper == base.error_squared_upper
    assert shifted.forcing_correction == base.forcing_correction
    assert shifted.macro_forcing == tuple(f + 2 for f in base.macro_forcing)


def test_anchored_zero_input_keeps_the_unforced_budget_and_explicit_law():
    _, reference = _reference()
    initial = (F(1), F(1), F(-2))
    unforced = _bound(reference, initial)
    anchored = _anchored(reference, initial)
    assert unforced.comparison_kind == "unforced"
    assert anchored.comparison_kind == "profile_anchored_affine"
    assert anchored.error_squared_upper == unforced.error_squared_upper
    assert anchored.macro_forcing == unforced.macro_forcing == (F(0), F(0))
    assert anchored.forcing_correction == unforced.forcing_correction == (F(0), F(0))


def test_anchor_rebuilds_profile_and_preserves_unavailable_gap_fields():
    _, reference = _reference(forcing=(F(9, 5), F(-1), F(0)))
    initial = (F(1), F(-1), F(0))
    expected = _anchored(reference, initial)
    stale = replace(reference, relative_profile=(F(999),) * 3, mean_drift=F(-100))
    assert _anchored(stale, initial) == expected
    unavailable = _anchored(reference, initial, decay=F(4))
    assert unavailable.status == "unavailable"
    assert unavailable.unavailable_reasons == ("hidden_decay_lower_not_certified",)
    assert unavailable.error_squared_upper is None
    assert unavailable.initial_source_error_squared_upper is None
    assert unavailable.feedback_error_squared_upper is None
    assert unavailable.macro_forcing == expected.macro_forcing
    with pytest.raises((TypeError, ValueError)):
        _anchored(replace(reference, forcing=(True, F(0), F(0))), initial)


def test_anchored_closed_partition_keeps_hidden_forcing_without_macro_feedback():
    _, reference = _reference(delta=F(0), forcing=(F(1), F(-1), F(0)))
    result = _anchored(reference, (F(1), F(2), F(3)))
    assert result.closure.all_state_affine_closed
    assert not result.closure.lifted_affine_subspace_invariant
    assert result.closure.hidden_affine_source == (F(2), F(-2), F(0))
    assert result.forcing_correction == result.macro_forcing == (F(0), F(0))
    assert result.error_squared_upper == 0


def test_tiny_forced_profile_deviations_remain_exact():
    _, reference = _reference(forcing=(F(9, 5), F(-1), F(0)))
    residual = (F(1), F(1), F(-2))
    scale = F(1, 2**300)
    base = _anchored(
        reference,
        tuple(z + u for z, u in zip(reference.relative_profile, residual, strict=True)),
    )
    tiny = _anchored(
        reference,
        tuple(
            z + scale * u
            for z, u in zip(reference.relative_profile, residual, strict=True)
        ),
    )
    assert tiny.error_squared_upper == base.error_squared_upper * scale**2 > 0
    assert tiny.initial_hidden_norm_squared == 0


@pytest.mark.parametrize(
    "predictor",
    (owner.bound_epi_memory_omission, owner.bound_anchored_epi_memory_omission),
)
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("horizon", True),
        ("horizon", -1),
        ("horizon", float("inf")),
        ("hidden_decay_lower", False),
        ("hidden_decay_lower", 0),
        ("hidden_decay_lower", -1),
        ("hidden_decay_lower", float("nan")),
    ),
)
def test_scalar_admission_precedes_the_static_calculation(predictor, field, value):
    _, reference = _reference()
    arguments = {"horizon": F(1), "hidden_decay_lower": F(3)}
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        predictor(reference, BLOCKS, **arguments)
