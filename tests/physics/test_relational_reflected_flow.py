"""Independent static controls for the exact reflected source/receiver field."""

from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
)
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics._relational_reflected_flow import (
    _phase_coefficients,
    evaluate_reflected_regular_flow,
)

COEFFICIENTS = dict(epi_weight=Q(3, 5), phase_weight=Q(2, 5), storage_scale=Q(7, 4))
REPRESENTATIVES = (0, 4, 5, 9, 10, 14, 15, 19)


def _graph(state):
    p, r, capital_p, capital_r, a, b, capital_a, capital_b = state
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    forms = (p, -p, -r, 0, r, capital_p, -capital_p, -capital_r, 0, capital_r)
    phases = (a, -a, -b, 0, b, capital_a, -capital_a, -capital_b, 0, capital_b)
    for i in graph:
        graph.nodes[i].update(EPI=float(forms[i]), theta=float(phases[i]), nu_f=1.0)
    return graph


def _model():
    return RelationalExchangeModel(
        phase_domain="regular",
        **{key: float(value) for key, value in COEFFICIENTS.items()},
    )


def _midpoints(values):
    return np.array([float(value.midpoint) for value in values])


@pytest.fixture(scope="module")
def generic_state():
    return (Q(1, 8), Q(-1, 16), Q(1, 4), Q(1, 32), Q(5, 2), Q(5, 4), Q(1, 4), Q(-1, 8))


@pytest.mark.parametrize("phase_sign", (1, -1))
def test_reflected_field_matches_the_native_full_graph_without_projection(
    generic_state, phase_sign
):
    state = generic_state[:4] + tuple(phase_sign * value for value in generic_state[4:])
    report = evaluate_reflected_regular_flow(state, **COEFFICIENTS)
    field = evaluate_relational_exchange(_graph(state), model=_model())
    assert report.state == tuple(I(value) for value in state)
    assert all(margin > 0 for margin in report.resultant_margin_lower_bounds)
    assert report.row_order == (0, 4, 3, 5, 9, 8)
    assert any(real.hi < 0 for real, _ in report.resultants)
    native_rates = np.array(field.form_rate + field.phase_rate)[list(REPRESENTATIVES)]
    np.testing.assert_allclose(
        _midpoints(report.rates), native_rates, rtol=3e-14, atol=2e-15
    )
    np.testing.assert_allclose(
        _midpoints(report.phase_sources),
        [field.phase_source[i] for i in report.row_order],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        _midpoints(report.inverse_phase_metrics),
        [1 / field.phase_metric[i] for i in report.row_order],
        rtol=3e-14,
    )
    assert float(report.storage.midpoint) == pytest.approx(
        float(field.storage), rel=3e-14
    )
    assert float(report.continuous_loss.midpoint) == pytest.approx(
        float(field.continuous_loss), rel=3e-14
    )


def test_uniform_form_derivative_jets_match_the_existing_twenty_coordinate_tangent(
    generic_state,
):
    state = (Q(0),) * 4 + generic_state[4:]
    tangent = evaluate_relational_uniform_tangent(_graph(state), model=_model())
    embedding = np.zeros((20, 8))
    for coordinate, first, second in (
        (0, 0, 1),
        (1, 4, 2),
        (2, 5, 6),
        (3, 9, 7),
        (4, 10, 11),
        (5, 14, 12),
        (6, 15, 16),
        (7, 19, 17),
    ):
        embedding[first, coordinate] = 1
        embedding[second, coordinate] = -1
    expected = (np.array(tangent.generator) @ embedding)[list(REPRESENTATIVES)]
    observed = np.zeros((8, 8))
    for column in range(8):
        variables = tuple(
            Jet((value, int(index == column))) for index, value in enumerate(state)
        )
        report = evaluate_reflected_regular_flow(variables, **COEFFICIENTS)
        observed[:, column] = [float(row.coeffs[1].midpoint) for row in report.rates]
    np.testing.assert_allclose(observed, expected, rtol=3e-14, atol=2e-15)


def test_directional_storage_jet_and_loss_obey_the_same_native_work_identity(
    generic_state,
):
    field = evaluate_reflected_regular_flow(generic_state, **COEFFICIENTS)
    # The first derivative in the field direction is the Lie derivative,
    # not a trajectory construction or a remainder enclosure.
    variables = tuple(
        Jet((value, rate)) for value, rate in zip(generic_state, field.rates)
    )
    jet = evaluate_reflected_regular_flow(variables, **COEFFICIENTS)
    assert (jet.storage.coeffs[1] + field.continuous_loss).contains(0)
    assert (jet.storage.coeffs[1] + field.continuous_loss).width < Q(1, 10**30)


def test_consensus_removable_axis_is_admitted_for_scalar_and_second_order_jets():
    state = (Q(1, 8), Q(-1, 16), Q(1, 4), Q(1, 32), 0, 0, 0, 0)
    field = evaluate_reflected_regular_flow(state, **COEFFICIENTS)
    assert all(value.contains(0) for value in field.phase_sources)
    for inverse, degree in zip(field.inverse_phase_metrics, (3, 2, 2, 3, 2, 2)):
        assert (inverse * pi_interval() * degree).contains(1)
    variables = tuple(
        Jet((value, Q(index - 3, 32), Q(index, 64)))
        for index, value in enumerate(state)
    )
    jets = evaluate_reflected_regular_flow(variables, **COEFFICIENTS)
    assert all(row.order == 2 for row in jets.rates)
    assert all(row.coeffs[0].lo > 0 for row in jets.inverse_phase_metrics)


@pytest.mark.parametrize("sign", (-1, 1))
def test_positive_real_metric_keeps_bounded_derivatives_on_tiny_signed_imaginary_boxes(
    sign,
):
    delta = Q(1, 2**40)
    imaginary = sign * I(delta, 3 * delta)
    _, inverse = _phase_coefficients(
        Jet((I(3), I(0), I(0))),
        Jet((imaginary, I(1), I(0))),
        is_jet=True,
        pi=pi_interval(),
    )
    # At fixed C=3, H^-1=1/(3*pi)-S^2/(81*pi)+O(S^4).
    # The independent removable-axis coefficients keep their finite limits
    # even when a very small S box is strictly separated from zero. Direct
    # Arg/S interval differentiation previously inflated the first derivative
    # to magnitude >10^11 instead of the correctly enclosed magnitude <delta/40.
    assert (inverse.coeffs[0] - 1 / (3 * pi_interval())).abs_max < delta**2
    assert (sign * inverse.coeffs[1]).hi < 0
    assert inverse.coeffs[1].abs_max < delta / 40
    assert (inverse.coeffs[2] + 1 / (81 * pi_interval())).abs_max < Q(1, 10**20)


def test_regular_metric_charts_agree_where_both_are_well_conditioned():
    from tnfr.mathematics._rational_interval import arg

    real, imaginary = I(2), I(Q(1, 4))
    source, inverse = _phase_coefficients(
        real, imaginary, is_jet=False, pi=pi_interval()
    )
    direct = arg(real, imaginary) / imaginary / pi_interval()
    assert (inverse - direct).contains(0)
    assert (inverse - direct).width < Q(1, 10**30)
    assert (source - imaginary * inverse).contains(0)


def test_wide_axis_regular_box_requires_resolved_metric_derivatives():
    # The law is regular at consensus, but wide phase boxes can make the
    # interval resultant hit the excluded ray. No clipped or midpoint field
    # can replace admission of all the consumed rows.
    state = (I(0),) * 4 + (I(-2, 2), I(0), I(0), I(0))
    with pytest.raises(ValueError, match="all six full resultant"):
        evaluate_reflected_regular_flow(state, **COEFFICIENTS)


def test_central_row_is_admitted_even_though_reflection_makes_its_rate_zero():
    state = (0, 0, 0, 0, 0, 2, 0, 0)
    with pytest.raises(ValueError, match="all six full resultant"):
        evaluate_reflected_regular_flow(state, **COEFFICIENTS)


def test_effective_weights_are_explicit_and_not_renormalized(generic_state):
    original = evaluate_reflected_regular_flow(generic_state, **COEFFICIENTS)
    scaled = evaluate_reflected_regular_flow(
        generic_state,
        epi_weight=3,
        phase_weight=2,
        storage_scale=COEFFICIENTS["storage_scale"],
    )
    assert scaled.epi_weight == 3 and scaled.phase_weight == 2
    for first, second in zip(original.rates, scaled.rates):
        assert (5 * first - second).contains(0)
    assert (5 * original.continuous_loss - scaled.continuous_loss).contains(0)
    assert (original.storage - scaled.storage).contains(0)


def test_zero_epi_weight_keeps_the_declared_conservative_exchange(generic_state):
    field = evaluate_reflected_regular_flow(
        generic_state, epi_weight=0, phase_weight=1, storage_scale=1
    )
    assert field.continuous_loss == I(0)
    assert any(not value.contains(0) for value in field.rates)


@pytest.mark.parametrize(
    "name,value,error",
    (
        ("epi_weight", -1, ValueError),
        ("phase_weight", 0, ValueError),
        ("storage_scale", 0, ValueError),
        ("epi_weight", True, TypeError),
        ("phase_weight", 0.5, TypeError),
        ("storage_scale", I(1), TypeError),
    ),
)
def test_invalid_constitutive_coefficients_are_rejected(
    generic_state, name, value, error
):
    parameters = dict(COEFFICIENTS, **{name: value})
    with pytest.raises(error):
        evaluate_reflected_regular_flow(generic_state, **parameters)


@pytest.mark.parametrize(
    "state,error",
    (
        ((0,) * 7, ValueError),
        ((0,) * 7 + (False,), TypeError),
        ((0,) * 7 + (0.1,), TypeError),
        ((Jet((0, 0)),) + (I(0),) * 7, TypeError),
        ((Jet((0, 0)),) + (Jet((0,)),) * 7, ValueError),
    ),
)
def test_invalid_coordinate_domains_are_rejected(state, error):
    with pytest.raises(error):
        evaluate_reflected_regular_flow(state, **COEFFICIENTS)
