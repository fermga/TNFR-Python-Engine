"""Contact-admission controls for separated and supplied-bridge components.

A candidate pair and common phase reference are supplied. Native fields check
the ideal guard derivative, and three shared Euler steps check offset
covariance. Neither these finite controls nor static directional probes prove
sustained locking, event times or an autonomous connection law.
"""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.operators.definitions import Emission, Reception
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.physics.relational_observations import observe_relational_attachment

MODEL = RelationalExchangeModel(storage_scale=1.0)
PHASE_LIMIT = math.pi / 4
BASE_FORM = 0.5
AMPLITUDE = 0.125


def _components(amplitude, *, relative_phase=PHASE_LIMIT):
    components = []
    for nodes, form, phase in (
        ((0, 1), (BASE_FORM + amplitude, BASE_FORM), 0.0),
        ((2, 3), (BASE_FORM, BASE_FORM), relative_phase),
    ):
        graph = nx.Graph()
        graph.add_edge(*nodes, weight=1.0)
        graph.graph.update(_t=3.0, retained={"history": ["unchanged"]})
        for node, value in zip(nodes, form, strict=True):
            graph.nodes[node].update(EPI=value, theta=phase, nu_f=1.0, delta_nfr=99.0)
        components.append(graph)
    return tuple(components)


def _state(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _phase_acceleration_probe(graph, field):
    """Probe D F(z)[F(z)] statically without advancing the shared integrator."""
    probe = 2.0**-10
    plus, minus = graph.copy(), graph.copy()
    for node, form_rate, phase_rate in zip(
        field.nodes, field.form_rate, field.phase_rate, strict=True
    ):
        plus.nodes[node]["EPI"] += probe * form_rate
        minus.nodes[node]["EPI"] -= probe * form_rate
        plus.nodes[node]["theta"] += probe * phase_rate
        minus.nodes[node]["theta"] -= probe * phase_rate
    forward = evaluate_relational_exchange(plus, model=MODEL)
    backward = evaluate_relational_exchange(minus, model=MODEL)
    return tuple(
        (new - old) / (2 * probe)
        for new, old in zip(forward.phase_rate, backward.phase_rate, strict=True)
    )


@pytest.mark.parametrize("amplitude", (AMPLITUDE, 0.0, -AMPLITUDE))
def test_native_component_fields_give_signed_contact_admission_rate(amplitude):
    components = _components(amplitude)
    before = tuple(map(_state, components))
    fields = tuple(
        evaluate_relational_exchange(graph, model=MODEL) for graph in components
    )
    left, right = fields

    # On either P2 at phase consensus, d=1, H=pi and g=0. This independent
    # edge calculation gives both rows, rather than reusing a guard observer.
    assert left.work.form_gradient == (Q(amplitude), -Q(amplitude))
    assert right.work.form_gradient == (0, 0)
    assert left.phase_metric == right.phase_metric == (math.pi, math.pi)
    assert left.phase_source == right.phase_source == (0.0, 0.0)
    assert left.form_rate == pytest.approx(
        (-MODEL.epi_weight * amplitude, MODEL.epi_weight * amplitude),
        rel=0,
        abs=1e-15,
    )
    assert right.form_rate == pytest.approx((0.0, 0.0), rel=0, abs=1e-15)
    phase_speed = MODEL.phase_weight * amplitude / (MODEL.storage_scale * math.pi)
    assert left.phase_rate == pytest.approx(
        (phase_speed, -phase_speed), rel=0, abs=1e-15
    )
    assert right.phase_rate == (0.0, 0.0)

    delta = right.phase[0] - left.phase[0]
    delta_rate = right.phase_rate[0] - left.phase_rate[0]
    actual = -math.sin(delta) * delta_rate
    expected = phase_speed * math.sin(PHASE_LIMIT)
    assert actual == pytest.approx(expected, rel=0, abs=1e-15)
    assert (actual > 0) == (amplitude > 0)
    assert (actual < 0) == (amplitude < 0)

    # A static directional derivative of the guard, using captured native
    # phase velocities. Neither perturbed point is an evolved endpoint.
    probe = 2.0**-12

    def guard(displacement):
        phase_j = right.phase[0] + displacement * right.phase_rate[0]
        phase_i = left.phase[0] + displacement * left.phase_rate[0]
        return math.cos(phase_j - phase_i) - math.cos(PHASE_LIMIT)

    assert guard(0) == 0
    derivative = (guard(probe) - guard(-probe)) / (2 * probe)
    assert derivative == pytest.approx(expected, rel=0, abs=1e-11)
    assert tuple(map(_state, components)) == before
    assert all(graph.number_of_edges() == 1 for graph in components)


def test_incompatible_zero_contrast_control_has_no_native_phase_motion():
    components = _components(0.0, relative_phase=PHASE_LIMIT + 1 / 32)
    fields = tuple(
        evaluate_relational_exchange(graph, model=MODEL) for graph in components
    )
    for field in fields:
        assert field.work.form_gradient == (0, 0)
        assert field.phase_rate == (0.0, 0.0)
        assert field.form_rate == pytest.approx((0.0, 0.0), rel=0, abs=1e-15)
    delta = fields[1].phase[0] - fields[0].phase[0]
    assert math.cos(delta) - math.cos(PHASE_LIMIT) < 0
    # The ideal uniform-form/component-consensus state is an equilibrium.
    # This static represented check does not assert a simulated hitting time.
    assert delta > PHASE_LIMIT


@pytest.mark.parametrize("phase_offset", (0.0, 3 * math.pi / 8))
def test_matching_component_motion_preserves_offset_without_creating_contact(
    phase_offset,
):
    left, right = _components(AMPLITUDE, relative_phase=phase_offset)
    internal_phase = 1 / 8
    form_offset = 1 / 4
    for graph, shift, offset in (
        (left, 0.0, 0.0),
        (right, form_offset, phase_offset),
    ):
        for node, form, phase in zip(
            graph,
            (BASE_FORM + AMPLITUDE + shift, BASE_FORM + shift),
            (offset, offset + internal_phase),
            strict=True,
        ):
            graph.nodes[node].update(EPI=form, theta=phase)
    supports = tuple(tuple(graph.edges(data="weight")) for graph in (left, right))
    initial = tuple(
        evaluate_relational_exchange(graph, model=MODEL) for graph in (left, right)
    )
    assert initial[0].phase_rate[0] > 0
    assert initial[0].phase_rate[1] < 0

    # This is a short executor-covariance check, not evidence that a locked
    # relation is attractive or that an unattached pair selects an event.
    dt = 1 / 64
    for iteration in range(3):
        before = tuple(
            evaluate_relational_exchange(graph, model=MODEL) for graph in (left, right)
        )
        assert before[1].form_rate == pytest.approx(
            before[0].form_rate, rel=0, abs=2e-14
        )
        assert before[1].phase_rate == pytest.approx(
            before[0].phase_rate, rel=0, abs=2e-14
        )
        steps = tuple(
            step_relational_exchange(graph, model=MODEL, dt=dt)
            for graph in (left, right)
        )
        first, second = (step.after for step in steps)
        assert second.epi == pytest.approx(
            tuple(value + form_offset for value in first.epi), rel=0, abs=2e-14
        )
        assert second.phase == pytest.approx(
            tuple(value + phase_offset for value in first.phase), rel=0, abs=2e-14
        )
        assert first.capacity == second.capacity == (1.0, 1.0)
        gap = abs(math.remainder(second.phase[0] - first.phase[0], math.tau))
        assert gap == pytest.approx(phase_offset, rel=0, abs=2e-14)
        assert (gap <= PHASE_LIMIT) == (phase_offset == 0)
        assert tuple(tuple(graph.edges(data="weight")) for graph in (left, right)) == (
            supports
        )
        assert nx.number_connected_components(nx.compose(left, right)) == 2
        assert left.graph["_t"] == right.graph["_t"] == 3 + (iteration + 1) * dt


def test_equal_port_phase_and_velocity_do_not_determine_phase_acceleration():
    components = _components(0.0, relative_phase=0.0)
    angle = 1 / 8
    for graph, direction in zip(components, (1, -1), strict=True):
        neighbor = tuple(graph)[1]
        graph.nodes[neighbor]["theta"] = direction * angle
    before = tuple(map(_state, components))
    fields = tuple(
        evaluate_relational_exchange(graph, model=MODEL) for graph in components
    )
    assert fields[0].phase[0] == fields[1].phase[0] == 0
    assert fields[0].phase_rate == fields[1].phase_rate == (0.0, 0.0)
    assert fields[0].work.form_gradient == fields[1].work.form_gradient == (0, 0)

    # Independently, on P2: H=pi*sinc(angle), and uniform form gives
    # (x0-x1)'=2*w*nu*angle/pi. Differentiating theta0'=w*nu*q0/(beta*H)
    # therefore leaves only q0' here, since q0=0 at this snapshot.
    sinc = math.sin(angle) / angle
    expected = (
        2 * MODEL.phase_weight**2 * angle / (MODEL.storage_scale * math.pi**2 * sinc)
    )
    observed = []
    for graph, field, direction in zip(components, fields, (1, -1), strict=True):
        speed = MODEL.phase_weight * direction * angle / math.pi
        assert field.form_rate == pytest.approx((speed, -speed), rel=0, abs=1e-15)
        assert field.phase_metric == pytest.approx(
            (math.pi * sinc, math.pi * sinc), rel=0, abs=1e-15
        )
        observed.append(_phase_acceleration_probe(graph, field)[0])

    # These are static D F(z)[F(z)] probes of the native field, not simulated
    # accelerations or an exact-real trigonometric enclosure.
    assert observed == pytest.approx((expected, -expected), rel=0, abs=1e-12)
    assert observed[0] > 0 > observed[1]
    assert tuple(map(_state, components)) == before


def test_existing_bridge_aligns_ports_while_regional_mean_phases_separate():
    graph = nx.Graph()
    for offset in (0, 5):
        nx.add_cycle(graph, range(offset, offset + 5), weight=1.0)
    graph.add_edge(0, 5, weight=1.0)
    kappa, delta = 2 * math.pi / 5, 1 / 8
    for node in graph:
        graph.nodes[node].update(
            EPI=BASE_FORM,
            theta=kappa * (node % 5) + delta * (node >= 5),
            nu_f=1.0,
            delta_nfr=99.0,
        )
    components = tuple(
        graph.subgraph(range(offset, offset + 5)).copy() for offset in (0, 5)
    )
    before = tuple(map(_state, (graph, *components)))
    field = evaluate_relational_exchange(graph, model=MODEL)
    assert field.work.form_gradient == (0,) * 10
    assert field.phase_rate == (0.0,) * 10

    # In the ideal winding-one rings the two internal port phasors sum to
    # 2*cos(kappa). Only the supplied bridge contributes the relative angle.
    real = 2 * math.cos(kappa) + math.cos(delta)
    imaginary = math.sin(delta)
    alpha = math.atan2(imaginary, real)
    source = alpha / math.pi
    metric = math.pi * math.hypot(real, imaginary) * math.sin(alpha) / alpha
    expected = 4 * MODEL.phase_weight**2 * source / (MODEL.storage_scale * metric)
    assert (field.phase_source[0], field.phase_source[5]) == pytest.approx(
        (source, -source), rel=0, abs=1e-15
    )
    assert (field.phase_metric[0], field.phase_metric[5]) == pytest.approx(
        (metric, metric), rel=0, abs=2e-15
    )

    acceleration = _phase_acceleration_probe(graph, field)
    assert (acceleration[0], acceleration[5]) == pytest.approx(
        (expected, -expected), rel=0, abs=1e-12
    )
    assert acceleration[5] - acceleration[0] == pytest.approx(
        -2 * expected, rel=0, abs=1e-12
    )
    assert acceleration[5] - acceleration[0] < 0

    # Internal neighbors have H_n=2*pi*cos(kappa). Their response reverses
    # the sign seen at the ports: regional means initially separate. Local
    # contact alignment is therefore not whole-pattern phase alignment.
    neighbor_metric = 2 * math.pi * math.cos(kappa)
    expected_mean_difference = (
        -2
        * MODEL.phase_weight**2
        * source
        / (5 * MODEL.storage_scale)
        * (4 / metric - 2 / neighbor_metric)
    )
    actual_mean_difference = (
        math.fsum(acceleration[5:]) - math.fsum(acceleration[:5])
    ) / 5
    assert expected_mean_difference > 0
    assert actual_mean_difference == pytest.approx(
        expected_mean_difference, rel=0, abs=1e-12
    )
    assert actual_mean_difference > 0

    # Removing that bridge restores the independent equilibrium rings. Native
    # wrapped phases have tiny roundoff; this is not an ideal-real certificate.
    for component in components:
        separate = evaluate_relational_exchange(component, model=MODEL)
        assert separate.phase_source == pytest.approx((0.0,) * 5, rel=0, abs=1e-15)
        assert _phase_acceleration_probe(component, separate) == pytest.approx(
            (0.0,) * 5, rel=0, abs=1e-12
        )
    assert tuple(map(_state, (graph, *components))) == before
    # The causal initial response requires the existing fine-support bridge;
    # it establishes neither a new edge nor a maintained effective relation.
    assert nx.is_connected(graph)
    assert tuple(component.number_of_edges() for component in components) == (5, 5)


def test_phase_admission_does_not_pay_for_a_state_preserving_attachment():
    left, right = _components(AMPLITUDE)
    before = (_state(left), _state(right))
    report = observe_relational_attachment(left, right, model=MODEL, bridge=(0, 2))
    assert report.form_storage_change == Q(AMPLITUDE) ** 2 / 2
    # Ideal edge cost uses mathematical pi; the native report retains the
    # represented sine evaluation. Agreement is numerical, not an enclosure.
    expected_phase_cost = 1 - math.cos(PHASE_LIMIT)
    assert float(report.phase_storage_change) == pytest.approx(
        expected_phase_cost, rel=0, abs=1e-15
    )
    expected_cost = AMPLITUDE**2 / 2 + MODEL.storage_scale * expected_phase_cost
    assert float(report.storage_change) == pytest.approx(
        expected_cost, rel=0, abs=1e-15
    )
    assert report.storage_change > 0
    assert not report.represented_zero_supply_passive
    assert not report.assess_supply(0).represented_balance_satisfied
    assert (_state(left), _state(right)) == before


def test_actual_emission_stage_supplies_form_storage_and_later_phase_motion():
    graph = _components(0.0)[0]
    graph.graph.update(GLYPH_FACTORS={"AL_boost": AMPLITUDE}, RANDOM_SEED=17)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.125,
            delta_nfr=0.125,
            EPI_kind="wave",
            glyph_history=["AL", "IL"],
        )
    before = evaluate_relational_exchange(graph, model=MODEL)
    edges = tuple(graph.edges(data="weight"))
    clock = graph.graph["_t"]

    # Invoke the actual atomic operator owner with its normal admission. No
    # callback refresh or continuous step is supplied during this form jump.
    stage = execute_pointwise_stage(graph, Emission(), (0,))
    after = evaluate_relational_exchange(graph, model=MODEL)
    assert stage.schedule == TWO_PHASE_JACOBI
    assert tuple(graph.nodes[0]["glyph_history"])[-1] == "AL"
    assert after.epi == (0.125 + AMPLITUDE, 0.125)
    assert after.phase == before.phase
    assert after.capacity == before.capacity
    assert tuple(graph.nodes[node]["delta_nfr"] for node in graph) == (0.125, 0.125)
    assert tuple(graph.edges(data="weight")) == edges
    assert graph.graph["_t"] == clock

    # Uniform initial form has q=0. The supplied one-node jump therefore costs
    # degree(0)*a**2/2, independently of its subsequent phase response.
    assert before.storage == 0
    assert after.storage - before.storage == graph.degree(0) * Q(AMPLITUDE) ** 2 / 2
    assert before.phase_rate == (0.0, 0.0)
    speed = MODEL.phase_weight * AMPLITUDE / (MODEL.storage_scale * math.pi)
    assert after.phase_rate == pytest.approx((speed, -speed), rel=0, abs=1e-15)
    assert after.form_rate == pytest.approx(
        (-MODEL.epi_weight * AMPLITUDE, MODEL.epi_weight * AMPLITUDE),
        rel=0,
        abs=1e-15,
    )


def test_actual_reception_stage_is_passive_and_changes_the_next_phase_field():
    graph = _components(AMPLITUDE)[0]
    rho = Q(1, 4)
    graph.graph.update(GLYPH_FACTORS={"EN_mix": float(rho)}, RANDOM_SEED=17)
    for node in graph:
        graph.nodes[node].update(
            delta_nfr=0.125, EPI_kind="wave", glyph_history=["AL", "IL"]
        )
    before = evaluate_relational_exchange(graph, model=MODEL)
    edges = tuple(graph.edges(data="weight"))
    clock = graph.graph["_t"]

    # Source tracking is optional telemetry; disabling it changes neither the
    # actual neighbor mean nor normal operator precondition/grammar checks.
    stage = execute_neighbor_stage(graph, Reception(), (0,), track_sources=False)
    after = evaluate_relational_exchange(graph, model=MODEL)
    assert stage.schedule == TWO_PHASE_JACOBI
    assert tuple(graph.nodes[0]["glyph_history"])[-1] == "EN"
    assert after.phase == before.phase
    assert after.capacity == before.capacity
    assert tuple(graph.nodes[node]["delta_nfr"] for node in graph) == (0.125, 0.125)
    assert tuple(graph.edges(data="weight")) == edges
    assert graph.graph["_t"] == clock

    degree = graph.degree(0)
    q0 = Q(before.epi[0]) - Q(before.epi[1])
    expected_jump = -rho * q0 / degree
    jump = tuple(
        Q(new) - Q(old) for new, old in zip(after.epi, before.epi, strict=True)
    )
    assert jump == (expected_jump, 0)  # The configured clip is inactive.
    expected_storage_change = -rho * (1 - rho / 2) * q0**2 / degree
    assert after.storage - before.storage == expected_storage_change < 0

    # Apply the P2 Laplacian independently to the observed jump. Exact
    # represented-rate defects distinguish this identity from ideal pi.
    gradient_change = (jump[0] - jump[1], jump[1] - jump[0])
    coefficient = Q(MODEL.phase_weight) / Q(MODEL.storage_scale)
    for index, change in enumerate(gradient_change):
        expected_rate_change = (
            coefficient
            * Q(before.capacity[index])
            / Q(before.phase_metric[index])
            * change
        )
        observed_rate_change = Q(after.phase_rate[index]) - Q(before.phase_rate[index])
        defect_change = (
            after.phase_rate_rounding_defect[index]
            - before.phase_rate_rounding_defect[index]
        )
        assert observed_rate_change == expected_rate_change + defect_change
    assert after.phase_rate[0] < before.phase_rate[0]
    assert after.phase_rate[1] > before.phase_rate[1]
