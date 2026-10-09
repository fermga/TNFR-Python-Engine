"""Bounded production controls for hidden phase/form on the fixed K3 lift.

The envelope is an ideal-Euler theorem. These 32-step binary64 traces check
its proposed comparison on one prepared state, retaining signed execution
defects and explicit finite numerical headroom; they are not runtime proofs.
"""

import math
from copy import deepcopy
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from tests.joint_phase_helpers import triangle
from tnfr.alias import set_theta
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import DefaultIntegrator
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.physics.canonical import compute_structural_potential
from tnfr.physics.joint_quotient import (
    bound_k3_transverse_euler,
    observe_joint_nodal_quotient,
    observe_k3_transverse_state,
    propose_joint_nodal_phase_step,
)
from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_euler,
)

H = F(1, 8)
K = F(1, 2)
STEPS = 32
# An explicit finite comparison allowance, not an analytic runtime enclosure.
HEADROOM = F(1, 10**12)


def _prepared(*, eta=F(1, 4), u=F(1, 8)):
    return triangle(
        epi=(F(1, 4), F(1, 8) + u / 2, F(1, 8) - u / 2),
        phase=(F(1, 4), F(3, 4) + eta / 2, F(3, 4) - eta / 2),
    )


def _trace(graph, steps):
    """Invoke the existing simultaneous phase and nodal-Euler owners."""
    states, potentials, budgets, phase_defects = [], [], [], []
    for index in range(steps + 1):
        state = observe_k3_transverse_state(graph)
        states.append(state)
        potential = compute_structural_potential(graph)
        potentials.append(tuple(F(float(potential[node])) for node in graph))
        if index == steps:
            break
        source = state.capture
        proposal = propose_u3_gated_phase_step(
            graph,
            source.snapshot.nodes,
            source.phase,
            source.snapshot.capacity,
            dt=float(H),
            coupling_strength=float(K),
        )
        # Rational reference to represented sine evaluations. The signed
        # difference also includes summation/staging/modulo rounding, but
        # does not certify the transcendental sine values themselves.
        phase_rate = tuple(
            source.snapshot.capacity[i]
            + K
            * sum(
                (
                    F(math.sin(float(source.phase[j]) - float(source.phase[i])))
                    for j in source.snapshot.support_neighbors[i]
                ),
                F(0),
            )
            / 2
            for i in range(3)
        )
        phase_defects.append(
            tuple(
                F(float(proposal[i])) - source.phase[i] - H * phase_rate[i]
                for i in range(3)
            )
        )
        DefaultIntegrator().integrate(
            graph, dt=float(H), t=float(index * H), method="euler", n_jobs=1
        )
        endpoint = observe_support_transport(graph)
        budgets.append(observe_support_transport_euler(source.snapshot, endpoint, H))
        for node, theta in zip(graph, proposal, strict=True):
            set_theta(graph, node, float(theta))
        default_compute_delta_nfr(graph)
    return SimpleNamespace(
        states=tuple(states),
        potentials=tuple(potentials),
        budgets=tuple(budgets),
        phase_defects=tuple(phase_defects),
    )


@pytest.fixture(scope="module")
def paired():
    full, reduced = _prepared(), _prepared(eta=F(0), u=F(0))
    envelope = bound_k3_transverse_euler(full, dt=H, coupling_strength=K, steps=STEPS)
    return envelope, _trace(full, STEPS), _trace(reduced, STEPS)


def test_coordinates_reconstruct_actual_nonconstant_phase_and_form():
    state = observe_k3_transverse_state(_prepared())
    assert state.eta != 0 and state.u != 0
    assert state.capture.snapshot.epi == (
        state.mean_epi + 2 * state.q / 3,
        state.mean_epi - state.q / 3 + state.u / 2,
        state.mean_epi - state.q / 3 - state.u / 2,
    )
    assert state.capture.phase == (
        state.mean_phase - 2 * state.delta / 3,
        state.mean_phase + state.delta / 3 + state.eta / 2,
        state.mean_phase + state.delta / 3 - state.eta / 2,
    )
    assert state.capture.snapshot.capacity_gradient == (0, 0, 0)
    assert state.capture.snapshot.topology_gradient == (0, 0, 0)


def test_bounds_and_observation_do_not_mutate_graph():
    graph = _prepared()
    nodes = deepcopy(dict(graph.nodes(data=True)))
    edges = deepcopy(tuple(graph.edges(data=True)))
    attributes = dict(graph.graph)
    weights = deepcopy(graph.graph["DNFR_WEIGHTS"])
    bound_k3_transverse_euler(graph, dt=H, coupling_strength=K, steps=2)
    assert dict(graph.nodes(data=True)) == nodes
    assert tuple(graph.edges(data=True)) == edges
    assert graph.graph.keys() == attributes.keys()
    assert all(graph.graph[key] is value for key, value in attributes.items())
    assert graph.graph["DNFR_WEIGHTS"] == weights


def test_ideal_envelope_coefficients_come_from_the_observed_channels(paired):
    envelope, *_ = paired
    assert envelope.phase_rate_lower == F(3, 4) * (1 - F(5, 8) ** 2 / 2)
    assert envelope.form_decay_rate == F(3, 4)
    assert envelope.phase_to_form_gain_upper == F(1, 4)
    assert len(envelope.samples) == STEPS + 1
    assert all(sample.step == i for i, sample in enumerate(envelope.samples))
    assert envelope.samples[0].delta_omission_upper == 0
    assert envelope.samples[0].q_omission_upper == 0
    assert envelope.samples[1].q_omission_upper == 0
    assert envelope.samples[1].delta_omission_upper > 0
    assert envelope.samples[2].q_omission_upper > 0


def test_shared_execution_retains_refresh_and_signed_euler_defects(paired):
    _, *traces = paired
    all_defects = []
    for trace in traces:
        for index, budget in enumerate(trace.budgets):
            before = trace.states[index].capture
            after = trace.states[index + 1].capture
            assert before.stored_pressure_residual == (0, 0, 0)
            assert after.stored_pressure_residual == (0, 0, 0)
            assert budget.before.stored_pressure == before.full_kernel_pressure
            assert budget.after.epi == after.snapshot.epi
            assert budget.identity_residual == 0
            assert budget.state_defect == tuple(
                after.snapshot.epi[i]
                - before.snapshot.epi[i]
                - H * before.snapshot.capacity[i] * before.full_kernel_pressure[i]
                for i in range(3)
            )
            all_defects.extend(budget.state_defect)
            assert max(map(abs, budget.state_defect)) < F(1, 10**14)
            assert max(map(abs, trace.phase_defects[index])) < F(1, 10**14)
            assert all(abs(x) < 1 for x in after.snapshot.epi)  # no clipping
        assert trace.states[-1].capture.full_kernel_pressure != (
            trace.states[0].capture.full_kernel_pressure
        )
    assert any(value > 0 for value in all_defects)
    assert any(value < 0 for value in all_defects)


def test_finite_transverse_decay_and_macro_omission_fit_separate_envelopes(paired):
    envelope, full, reduced = paired
    start = full.states[0]
    reference = reduced.states[0]
    assert (start.delta, start.q, start.mean_epi, start.mean_phase) == (
        reference.delta,
        reference.q,
        reference.mean_epi,
        reference.mean_phase,
    )
    for bound, fine, coarse in zip(
        envelope.samples, full.states, reduced.states, strict=True
    ):
        assert abs(fine.delta) <= bound.delta_upper + HEADROOM
        assert abs(fine.eta) <= bound.eta_upper + HEADROOM
        assert abs(fine.u) <= bound.u_upper + HEADROOM
        assert abs(fine.delta - coarse.delta) <= bound.delta_omission_upper + HEADROOM
        assert abs(fine.q - coarse.q) <= bound.q_omission_upper + HEADROOM
        assert fine.phase_width <= start.phase_width + HEADROOM
        assert abs(fine.mean_epi - start.mean_epi) < HEADROOM
        assert abs(fine.mean_phase - start.mean_phase - bound.step * H) < HEADROOM
        assert abs(coarse.eta) < HEADROOM
        assert abs(coarse.u) < HEADROOM
    assert abs(full.states[-1].eta) < abs(start.eta) / 10
    assert abs(full.states[-1].u) < abs(start.u) / 10
    assert abs(full.states[1].delta - reduced.states[1].delta) > HEADROOM
    # Fresh-pressure EPI uses the initial shared macro phase at the first
    # simultaneous step. The hidden phase reaches macro form only later.
    assert abs(full.states[1].q - reduced.states[1].q) < HEADROOM
    assert abs(full.states[2].q - reduced.states[2].q) > HEADROOM


def test_fine_pressure_and_actual_potential_omit_first_order_information(paired):
    envelope, full, reduced = paired
    for index, (bound, fine, coarse) in enumerate(
        zip(envelope.samples, full.states, reduced.states, strict=True)
    ):
        for i in range(3):
            assert (
                abs(
                    fine.capture.full_kernel_pressure[i]
                    - coarse.capture.full_kernel_pressure[i]
                )
                <= bound.fine_pressure_potential_upper[i] + HEADROOM
            )
            assert (
                abs(full.potentials[index][i] - reduced.potentials[index][i])
                <= bound.fine_pressure_potential_upper[i] + HEADROOM
            )
        for state, potential in (
            (fine, full.potentials[index]),
            (coarse, reduced.potentials[index]),
        ):
            pressure = state.capture.full_kernel_pressure
            pressure_sum = sum(pressure, F(0))
            # Exact model Phi=-p; runtime pressure-sum and field-summation
            # residuals retain distinct meanings instead of being zeroed.
            assert abs(pressure_sum) < F(1, 10**14)
            for i in range(3):
                assert abs(potential[i] + pressure[i] - pressure_sum) < F(1, 10**14)
    initial_bound = envelope.samples[0]
    assert initial_bound.delta_omission_upper == initial_bound.q_omission_upper == 0
    assert abs(full.potentials[0][1] - reduced.potentials[0][1]) > F(1, 100)


def test_transverse_sign_changes_do_not_change_the_macro_phase_proposal():
    positive, negative = _trace(_prepared(), 1), _trace(_prepared(eta=-F(1, 4)), 1)
    a, b = positive.states[1], negative.states[1]
    assert a.delta == b.delta
    assert a.eta == -b.eta
    # Same initial macro EPI and instantaneous phase mean give equal q row.
    assert abs(a.q - b.q) < HEADROOM


def test_macro_omission_is_quadratic_but_initial_fine_pressure_is_linear():
    full = _prepared(u=F(0))
    half = _prepared(eta=F(1, 8), u=F(0))
    zero = _prepared(eta=F(0), u=F(0))
    bounds = tuple(
        bound_k3_transverse_euler(g, dt=H, coupling_strength=K, steps=2)
        for g in (full, half)
    )
    assert bounds[0].samples[1].delta_omission_upper == (
        4 * bounds[1].samples[1].delta_omission_upper
    )
    assert bounds[0].samples[2].q_omission_upper == (
        4 * bounds[1].samples[2].q_omission_upper
    )
    assert bounds[0].samples[0].fine_pressure_potential_upper[1] == (
        2 * bounds[1].samples[0].fine_pressure_potential_upper[1]
    )
    traces = tuple(_trace(g, 1) for g in (full, half, zero))
    error_full = abs(traces[0].states[1].delta - traces[2].states[1].delta)
    error_half = abs(traces[1].states[1].delta - traces[2].states[1].delta)
    assert float(error_full / error_half) == pytest.approx(4.0, rel=0.002)
    initial_pressure = tuple(t.states[0].capture.full_kernel_pressure for t in traces)
    full_hidden = initial_pressure[0][1] - initial_pressure[2][1]
    half_hidden = initial_pressure[1][1] - initial_pressure[2][1]
    assert abs(full_hidden - 2 * half_hidden) < HEADROOM


def test_initial_macro_same_state_is_not_autonomous_off_the_phase_fiber():
    graph = _prepared()
    with pytest.raises(ValueError, match="block-constant phase"):
        observe_joint_nodal_quotient(graph, ((0,), (1, 2)))
    with pytest.raises(ValueError, match="block-constant phase"):
        propose_joint_nodal_phase_step(graph, ((0,), (1, 2)), dt=H, coupling_strength=K)


@pytest.mark.parametrize("steps", (-1, 257, True, 1.5, "2", None))
def test_bound_rejects_invalid_horizon(steps):
    with pytest.raises(ValueError, match="integer"):
        bound_k3_transverse_euler(_prepared(), dt=H, coupling_strength=K, steps=steps)


@pytest.mark.parametrize("name", ("dt", "coupling_strength"))
@pytest.mark.parametrize("value", (0, -1, True, math.nan, math.inf, 1j, F(1, 2**1200)))
def test_bound_rejects_invalid_operational_scalars(name, value):
    kwargs = {"dt": H, "coupling_strength": K, "steps": 0, name: value}
    with pytest.raises((TypeError, ValueError)):
        bound_k3_transverse_euler(_prepared(), **kwargs)


@pytest.mark.parametrize("capacity,strength", ((1, 1), (2, F(1, 4))))
def test_bound_checks_both_monotone_step_ceilings(capacity, strength):
    graph = triangle(capacity=capacity)
    with pytest.raises(ValueError, match="monotone Euler ceiling"):
        bound_k3_transverse_euler(graph, dt=1, coupling_strength=strength, steps=0)


def test_interior_step_and_zero_horizon_are_admitted():
    result = bound_k3_transverse_euler(
        triangle(), dt=1, coupling_strength=F(1, 2), steps=0
    )
    assert len(result.samples) == 1


@pytest.mark.parametrize("phase", ((-0.25, 0.25, 0.5), (0, 0.25, math.tau)))
def test_observation_rejects_noncanonical_phases(phase):
    with pytest.raises(ValueError, match="canonical"):
        observe_k3_transverse_state(triangle(phase=phase))


@pytest.mark.parametrize("phase", ((0, 1.125, 0.5), (math.tau - 0.125, 0.125, 0.25)))
def test_observation_rejects_wide_or_wrap_straddling_raw_lift(phase):
    with pytest.raises(ValueError, match="raw phase lift"):
        observe_k3_transverse_state(triangle(phase=phase))


def test_one_radian_chart_boundary_is_admitted_but_configured_u3_still_applies():
    graph = triangle(phase=(0, 0.5, 1))
    assert observe_k3_transverse_state(graph).phase_width == 1
    graph.graph["UM_MAX_PHASE_DIFF"] = 0.5
    with pytest.raises(ValueError, match="fully U3-admitted"):
        observe_k3_transverse_state(graph)


@pytest.mark.parametrize("capacity", (0, -1))
def test_observation_rejects_nonpositive_capacity(capacity):
    with pytest.raises((TypeError, ValueError)):
        observe_k3_transverse_state(triangle(capacity=capacity))


def test_observation_rejects_heterogeneous_capacity():
    graph = triangle()
    graph.nodes[2]["nu_f"] = 1.25
    with pytest.raises(ValueError, match="common positive capacity"):
        observe_k3_transverse_state(graph)


def test_observation_rejects_inactive_form_channel():
    graph = triangle(weights={"phase": 1.0, "epi": 0.0, "vf": 0.0, "topo": 0.0})
    with pytest.raises(ValueError, match="positive EPI weight"):
        observe_k3_transverse_state(graph)


@pytest.mark.parametrize("attribute", ("weight", "length"))
def test_observation_rejects_nonunit_transport_or_geometry(attribute):
    graph = triangle()
    graph.edges[0, 1][attribute] = 2.0
    with pytest.raises(ValueError, match="unit K3"):
        observe_k3_transverse_state(graph)


def test_observation_rejects_missing_support_edge():
    graph = triangle()
    graph.remove_edge(1, 2)
    with pytest.raises(ValueError, match="unit K3"):
        observe_k3_transverse_state(graph)
