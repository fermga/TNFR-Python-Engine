"""Frozen controls for preliminary box-work obstructions, with no solver.

The two scientific preparations are the previously registered three-node
paths. Other controls exercise symbolic proof, availability or API admission.
No retained source, candidate tube or trajectory is evaluated here.
"""

import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest
import sympy as s

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics import _validated_taylor as taylor
from tnfr.mathematics._rational_interval import I, pi_interval, sin, sqrt
from tnfr.research import maintained_sha_box_budget as owner
from tnfr.research import maintained_sha_motion as motion
from tnfr.research import maintained_sha_reference as reference

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
PHASES = ((Q(0), Q(0), Q(0)), (Q(0), Q(1, 4), Q(1, 2)))


def _graph(phases=PHASES[0]):
    graph = nx.Graph()
    for node, form, phase, capacity in zip(
        ("parent", "child", "rest"), (0, 1, 2), phases, (2, 3, 5), strict=True
    ):
        graph.add_node(node, EPI=form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.add_weighted_edges_from((("parent", "child", 2), ("parent", "rest", 3)))
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return graph


def _bound(graph):
    return owner._bound_box_budget(
        graph, parent="parent", child="child", reference_model=MODEL
    )


def _persistent(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _overlap(actual, expected):
    assert (actual - expected).contains(0)


@pytest.fixture(autouse=True)
def no_source_tube_or_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "preliminary controls cannot read sources, construct tubes or evolve"
        )

    monkeypatch.setattr(owner, "prepare_sha_reference", forbidden)
    monkeypatch.setattr(reference, "prepare_sha_pair_budget", forbidden)
    monkeypatch.setattr(taylor, "picard_tube", forbidden)
    for name in (
        "validate_live_sha_successor",
        "validated_box_taylor_step",
        "certify_retention",
        "prepare_sha_motion",
    ):
        monkeypatch.setattr(motion, name, forbidden)


@pytest.fixture(scope="module", params=PHASES)
def phase_case(request):
    graph = _graph(request.param)
    before = _persistent(graph)
    result = _bound(graph)
    assert _persistent(graph) == before
    return request.param, result


def test_full_rational_rows_keep_the_child_cancellation_before_pi(phase_case):
    _, result = phase_case
    rows = result.coefficients
    assert rows.beta == (Q(1), Q(3), Q(5))
    assert rows.center_mobility_numerator == Q(3, 4)
    assert rows.external_port == (Q(3), Q(0), Q(-3))
    assert rows.weighted_laplacian == ((5, -2, -3), (-2, 2, 0), (-3, 0, 3))
    assert rows.external_indices == (2,)
    assert rows.phase_rate_numerators == ((Q(-69, 4), Q(0), Q(69, 4)),)
    assert rows.child_column_cancels
    assert rows.parent_column_absolute_sum_numerator == Q(69, 4)
    _overlap(rows.parent_column_absolute_sum, 69 / (4 * pi_interval()))
    assert sum(rows.phase_rate_numerators[0], Q(0)) == 0
    with pytest.raises(FrozenInstanceError):
        rows.child_column_cancels = False


def test_initial_image_coefficient_has_independent_three_node_formula(phase_case):
    phases, result = phase_case
    expected = (
        69 / (4 * pi_interval() ** 2) * (sin(I(phases[1])) + 6 * sin(I(phases[2])))
    )
    _overlap(result.initial_form_rate_image_coefficient, expected)
    assert result.initial_form_rate_image_coefficient.lo >= 0
    if phases == PHASES[0]:
        assert result.initial_form_rate_image_coefficient.contains(0)
    else:
        assert result.initial_form_rate_image_coefficient.lo > 0
    # These copied-through source rows retain their existing trigonometric
    # enclosures, not a new accuracy claim from 128-bit outer arithmetic.
    source = result.reference.pair_budget.connected.source
    assert source.nodes == ("parent", "child", "rest")
    assert source.degrees == (2, 1, 1)
    assert result.reference.pair_budget.connected.form_mean == Q(11, 23)


def test_both_chosen_budget_floors_and_signal_ceiling_match_their_independent_formulas(
    phase_case,
):
    phases, result = phase_case
    initial = result.reference
    chart, state = initial.chart, initial.state
    duration = initial.initial_work_budget.minimum_turn_time
    assert result.minimum_turn_time == duration
    assert duration.lo > 0
    lower, upper = chart.relative_curvature_lower, Q(35, 32)
    Q0, B0 = state.radial_excitation, state.reference_radius
    ceiling = (Q0 - sqrt(I(upper)) ** 3 * B0 / lower) / (
        1 + sqrt(I(upper)) ** 3 / (lower * sqrt(I(lower)))
    )
    _overlap(result.signal_budget_ceiling, ceiling)
    expected_K = (
        69 / (4 * pi_interval() ** 2) * (sin(I(phases[1])) + 6 * sin(I(phases[2])))
    )
    expected_image = initial.rates.work_norm * duration + expected_K * duration**2 / (
        4 * sqrt(I(chart.curvature_lower))
    )
    _overlap(result.picard_image_budget_floor, expected_image)
    _overlap(result.picard_image_gate.margin, ceiling - expected_image)
    assert result.saturation.status == "established"
    assert result.saturation.pair_margin.lo >= 0
    assert result.saturation.parent_margin.lo >= 0
    expected_explicit = initial.rates.work_norm * duration + 69 * duration**2 / (
        8 * pi_interval() ** 2 * sqrt(I(chart.curvature_lower))
    )
    _overlap(result.explicit_row_budget_floor, expected_explicit)
    _overlap(result.explicit_row_gate.margin, ceiling - expected_explicit)
    assert result.picard_image_gate.status == "excluded"
    assert result.explicit_row_gate.status == "excluded"
    assert (
        "picard_image_floor_is_not_an_actual_displacement_or_work_lower_bound"
        in result.scope
    )


@pytest.mark.parametrize(
    "pair,parent,status",
    (
        (I(0), I(0), "established"),
        (I(1), I(2), "established"),
        (I(-1), I(2), "not_established"),
        (I(2), I(-1), "not_established"),
        (I(-1, 1), I(2), "unavailable"),
        (None, None, "unavailable"),
    ),
)
def test_saturation_requires_both_non_strict_certified_activations(
    pair, parent, status
):
    result = owner._saturation(pair, parent)
    assert result.status == status
    assert bool(result.reasons) == (status != "established")


def test_unadmitted_initial_reference_keeps_all_horizon_budget_gates_unavailable():
    result = _bound(_graph((Q(0), Q(2), Q(0))))
    assert result.reference.chart.status == "unavailable"
    assert result.coefficients.child_column_cancels
    assert result.minimum_turn_time is None
    assert result.signal_budget_ceiling is None
    assert result.picard_image_budget_floor is None
    assert result.picard_image_gate.status == "unavailable"
    assert result.explicit_row_budget_floor is None
    assert result.explicit_row_gate.status == "unavailable"
    assert result.saturation.status == "unavailable"


@pytest.mark.parametrize(
    "attribute,value", (("EPI", True), ("nu_f", 0), ("theta", float("nan")))
)
def test_consumed_primitive_admission_remains_shared(attribute, value):
    graph = _graph()
    graph.nodes["rest"][attribute] = value
    with pytest.raises((TypeError, ValueError, TNFRUserError)):
        _bound(graph)


def test_public_wrapper_rebuilds_reference_before_any_internal_budget(monkeypatch):
    admitted, fresh, expected = object(), object(), object()
    calls = []

    def prepare(value):
        assert value is admitted
        calls.append("readmit_reference")
        return fresh

    def consume(value):
        assert value is fresh
        calls.append("consume_fresh")
        return expected

    monkeypatch.setattr(owner, "prepare_sha_reference", prepare)
    monkeypatch.setattr(owner, "_box_budget_from_reference", consume)
    assert owner.prepare_sha_box_budget(admitted) is expected
    assert calls == ["readmit_reference", "consume_fresh"]


def test_exact_parent_column_sum_follows_from_the_signed_laplacian_rows():
    ma, mc, mj, weight = s.symbols("ma mc mj weight", positive=True)
    mu = ma * mc / (ma + mc)
    # One external row is u_j^T M L. Its parent coefficient is negative,
    # so its absolute value has this exact sum before any interval arithmetic.
    gamma = ma / (ma + mc)
    incidence = s.Matrix(((1, 1), (-1, 0), (0, -1)))
    bridge = s.symbols("bridge", positive=True)
    laplacian = incidence * s.diag(bridge, weight) * incidence.T
    u = s.Matrix((-(1 - gamma), -gamma, 1))
    row = u.T * s.diag(ma, mc, mj) * laplacian
    assert s.simplify(row[1]) == 0
    assert s.simplify(row[0] + mj * weight + mu * weight) == 0
    assert s.simplify(row[2] - mj * weight - mu * weight) == 0


def test_lower_enclosure_budget_increases_with_necessary_duration_and_image_coefficient():
    G0, K, extra_K = s.symbols("G0 K extra_K", nonnegative=True)
    gamma, root_m, T, extra_T = s.symbols("gamma root_m T extra_T", positive=True)
    floor = G0 * T + gamma * K * T**2 / root_m
    increment = s.expand(
        floor.subs({T: T + extra_T, K: K + extra_K}, simultaneous=True) - floor
    )
    expected = G0 * extra_T + gamma / root_m * (
        K * (2 * T * extra_T + extra_T**2) + extra_K * (T + extra_T) ** 2
    )
    assert s.simplify(increment - expected) == 0
    assert expected.is_nonnegative


def test_pair_saturation_uses_strict_image_slacks_before_parent_saturation():
    T, k, w, phase_pair, slack = s.symbols("T k w phase_pair slack", positive=True)
    # In the hypothetical branch 0<b_pair<=1, a_pair>T*k*b_pair.
    # The second necessary inequality b_pair>T*k*w*a_pair then has this
    # negative remainder as soon as z=(A*T)^2>=1.
    form_pair = T * k * phase_pair + slack
    margin = phase_pair - T * k * w * form_pair
    z = T**2 * k**2 * w
    assert s.expand(margin - (phase_pair * (1 - z) - T * k * w * slack)) == 0
    excess = s.symbols("excess", nonnegative=True)
    rejected_margin = -phase_pair * excess - T * k * w * slack
    assert rejected_margin.is_negative
    # Once b_pair>1, both row sums are at least one: a_pair>T*k.
    # The parent's phase inequality gives b_a>T*m_a*w*a_pair>gamma*z.
    gamma = s.symbols("gamma", positive=True)
    parent_floor = T * (gamma * k) * w * (T * k + slack)
    assert s.expand(parent_floor - gamma * z - T * gamma * k * w * slack) == 0
    assert (T * gamma * k * w * slack).is_positive
