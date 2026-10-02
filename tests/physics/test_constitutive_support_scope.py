"""Scale freedom and conditional completions of declared conductance.

Rational identities and finite dyadic binary64 read-outs are distinct controls.
Support comparisons use detached preparations; no graph trajectory, operator
or pressure write is executed.
Candidate-contact jets posit a fully weighted phase law distinct from the
native unique-neighbor phase channel; no completion is installed in the engine.
"""

from copy import deepcopy
from fractions import Fraction
from math import cos, pi, ulp

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.relational_observations import observe_relational_reset
from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_derivative,
    observe_support_transport_reset,
)

F = Fraction


def _graph(scale=1, *, explicit_lengths=False):
    graph = nx.Graph()
    graph.add_nodes_from(("center", "left", "right", "leaf"))
    for neighbor, weight in (("left", 1), ("right", 2), ("leaf", 4)):
        attributes = {"weight": float(scale * weight)}
        if explicit_lengths:
            attributes["length"] = float(weight)
        graph.add_edge("center", neighbor, **attributes)
    for node, epi, capacity, phase, pressure in zip(
        graph,
        (1.0, -0.5, 0.25, 2.0),
        (1.0, 2.0, 0.5, 4.0),
        (0.0, 0.25, -0.5, 0.75),
        (0.5, 1.0, -0.25, 0.125),
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi, nu_f=capacity, theta=phase, delta_nfr=pressure
        )
    graph.graph.update(
        DNFR_WEIGHTS={name: 1.0 for name in ("phase", "epi", "vf", "topo")},
        compute_delta_nfr=default_compute_delta_nfr,
    )
    return graph


def _normalized_rows(entries, count):
    rows = [[F(0) for _ in range(count)] for _ in range(count)]
    for i, j, weight in entries:
        rows[i][j] = weight
    return tuple(
        tuple(value / sum(row) if sum(row) else F(0) for value in row) for row in rows
    )


@pytest.mark.parametrize("scale", (F(1, 7), F(3, 2), F(17)))
def test_exact_positive_scale_leaves_every_normalized_row_including_isolate(scale):
    # General rational coefficients, deliberately not a claim about float casts.
    entries = (
        (0, 0, F(1, 3)),
        (0, 1, F(2, 7)),
        (1, 0, F(2, 7)),
        (1, 2, F(5, 11)),
        (2, 1, F(5, 11)),
    )
    scaled = tuple((i, j, scale * weight) for i, j, weight in entries)
    assert _normalized_rows(scaled, 4) == _normalized_rows(entries, 4)
    assert _normalized_rows(entries, 4)[3] == (0, 0, 0, 0)


@pytest.mark.parametrize("scale", (F(1, 2), F(2), F(8)))
def test_fixed_support_full_pressure_is_invariant_in_finite_dyadic_fixture(scale):
    first, second = _graph(), _graph(scale)
    saved = [
        (deepcopy(dict(g.nodes(data=True))), deepcopy(dict(g.edges)), deepcopy(g.graph))
        for g in (first, second)
    ]
    baseline, scaled = (capture_non_epi_forcing(g) for g in (first, second))
    assert scaled.snapshot.epi_gradient == baseline.snapshot.epi_gradient
    assert scaled.snapshot.capacity_gradient == baseline.snapshot.capacity_gradient
    assert scaled.snapshot.topology_gradient == baseline.snapshot.topology_gradient
    assert scaled.phase_gradient == baseline.phase_gradient
    assert scaled.forcing == baseline.forcing
    assert scaled.normalized_weights == baseline.normalized_weights
    assert scaled.full_kernel_pressure == baseline.full_kernel_pressure
    assert scaled.kernel_pressure_defect == baseline.kernel_pressure_defect
    assert (
        scaled.snapshot.dirichlet_energy == scale * baseline.snapshot.dirichlet_energy
    )
    for graph, expected in zip((first, second), saved, strict=True):
        assert (
            dict(graph.nodes(data=True)),
            dict(graph.edges),
            graph.graph,
        ) == expected


def test_same_initial_support_has_distinct_declared_tangents_and_energy_work():
    source = observe_support_transport(_graph())
    # W_1(t)=W0 and W_2(t)=(1+3t)W0 agree at t=0. Both are positive
    # near zero. Their distinct supplied derivatives are not runtime laws.
    static = observe_support_transport_derivative(
        source,
        conductance_rates=(F(0),) * len(source.conductance),
    )
    growing = observe_support_transport_derivative(
        source,
        conductance_rates=tuple(3 * w for _, _, w in source.conductance),
    )
    assert static.source == growing.source == source
    assert static.conductance_rates != growing.conductance_rates
    assert growing.geometry_gradient_rate == static.geometry_gradient_rate == (0,) * 4
    assert growing.epi_gradient_rate == static.epi_gradient_rate
    assert growing.nodal_work == static.nodal_work
    assert static.conductance_work == 0
    assert growing.conductance_work == 3 * source.dirichlet_energy > 0
    assert growing.energy_rate - static.energy_rate == 3 * source.dirichlet_energy


@pytest.mark.parametrize("explicit_lengths", (False, True))
@pytest.mark.parametrize("triangle", ("L", "U"))
def test_potential_distinguishes_metric_length_from_conductance(
    explicit_lengths, triangle, monkeypatch
):
    first = _graph(explicit_lengths=explicit_lengths)
    second = _graph(2, explicit_lengths=explicit_lengths)
    before = compute_structural_potential(first)
    after = compute_structural_potential(second)
    factor = 1 if explicit_lengths else F(1, 4)
    # Dyadic scaling of the same positive paths gives an exact finite result.
    assert after == {node: factor * value for node, value in before.items()}
    assert any(before.values())
    assert compute_phase_gradient(first) == compute_phase_gradient(second)
    assert compute_phase_curvature(first) == compute_phase_curvature(second)
    # Four nodes cannot supply the correlation fit's ten distinct pairs.
    # This explicitly tests the normalized-spectrum fallback, not every xi fit.
    eigvalsh = np.linalg.eigvalsh
    monkeypatch.setattr(
        np.linalg, "eigvalsh", lambda matrix: eigvalsh(matrix, UPLO=triangle)
    )
    xi_first = estimate_coherence_length_with_provenance(first)
    xi_second = estimate_coherence_length_with_provenance(second)
    assert xi_first.method == xi_second.method == "spectral_gap"
    # A positive weighted three-leaf star has exact normalized spectrum
    # (0, 1, 1, 2), hence xi=1. Square-root normalization and either equivalent
    # LAPACK triangle route can differ in their last bits under scaling.
    # This is a numerical test tolerance, not a universal solver error bound.
    assert (xi_first.value, xi_second.value) == pytest.approx(
        (1.0, 1.0), rel=0.0, abs=16 * ulp(1.0)
    )


def test_zero_conductance_support_still_changes_non_epi_channels():
    connected = _graph()
    connected.edges["center", "leaf"]["weight"] = 0.0
    absent = _graph()
    absent.remove_edge("center", "leaf")
    with_support = capture_non_epi_forcing(connected)
    without_support = capture_non_epi_forcing(absent)
    assert with_support.snapshot.conductance == without_support.snapshot.conductance
    assert with_support.snapshot.epi_gradient == without_support.snapshot.epi_gradient
    assert (
        with_support.snapshot.capacity_gradient
        != without_support.snapshot.capacity_gradient
    )
    assert with_support.phase_gradient != without_support.phase_gradient
    assert with_support.full_kernel_pressure != without_support.full_kernel_pressure


@pytest.fixture(scope="module")
def contact_preparation():
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from(((0, 1), (2, 3)), weight=1.0)
    for node, form, pressure in zip(graph, (0.0, 1.0, 0.0, 0.0), (0.5, -0.5, 0.0, 0.0)):
        graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0, delta_nfr=pressure)
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0.0, "topo": 0.0}
    model = RelationalExchangeModel(storage_scale=1.0)
    fields = tuple(
        evaluate_relational_exchange(graph.subgraph(nodes).copy(), model=model)
        for nodes in ((0, 1), (2, 3))
    )
    source = observe_support_transport(graph)
    hypothetical = graph.copy()
    hypothetical.add_edge(1, 2, weight=1.0)
    reset = observe_support_transport_reset(
        source, observe_support_transport(hypothetical)
    )
    return graph, model, fields, source, reset


def test_contact_initial_budget_is_checked_by_native_component_fields(
    contact_preparation,
):
    graph, model, fields, source, reset = contact_preparation
    before = deepcopy(graph)
    assert (
        source.rate
        == fields[0].form_rate + fields[1].form_rate
        == (F(1, 2), -F(1, 2), 0, 0)
    )
    assert fields[0].phase_rate == pytest.approx((-0.5 / pi, 0.5 / pi), abs=2e-16)
    assert fields[1].phase_rate == (0.0, 0.0)
    loss = sum(field.continuous_loss for field in fields)
    assert loss == -source.energy_rate == 1
    local_budget = fields[0].work.dissipation[1] + fields[1].work.dissipation[0]
    assert (
        local_budget
        == F(model.epi_weight)
        * (source.dirichlet_gradient[1] ** 2 + source.dirichlet_gradient[2] ** 2)
        == F(1, 2)
    )
    assert 0 < local_budget < loss
    cost = (source.epi[1] - source.epi[2]) ** 2 / 2
    assert reset.energy_change == reset.edge_energy_change == cost == F(1, 2)
    assert reset.identity_residual == 0
    capture = capture_non_epi_forcing(graph)
    assert capture.full_kernel_pressure == source.stored_pressure
    assert nx.utils.graphs_equal(graph, before)
    # The hypothetical reset measures attachment cost; it neither creates a
    # live edge nor supplies the candidate's occurrence law or time.


def test_contact_positive_weight_pressure_reuses_native_transport_only(
    contact_preparation,
):
    graph, model, _, _, _ = contact_preparation
    for weight in (F(1, 4), F(1, 2)):
        sample = graph.copy()
        sample.add_edge(1, 2, weight=float(weight))
        before = deepcopy(sample)
        capture = capture_non_epi_forcing(sample)
        expected = F(model.epi_weight) * weight / (1 + weight)
        assert capture.snapshot.epi_gradient[2] == weight / (1 + weight)
        assert float(capture.full_kernel_pressure[2]) == pytest.approx(
            float(expected), abs=1e-16
        )
        assert capture.phase_gradient == (0,) * 4
        rates = tuple(F({i, j} == {1, 2}) for i, j, _ in capture.snapshot.conductance)
        derivative = observe_support_transport_derivative(
            capture.snapshot, conductance_rates=rates
        )
        assert derivative.geometry_gradient_rate[2] == 1 / (1 + weight) ** 2
        assert nx.utils.graphs_equal(sample, before)
    # This owner handles existing positive conductances. It is deliberately
    # not called with a=0 birth, nor used for weighted nonlinear phase motion.


def test_two_extra_contact_laws_have_distinct_passive_receiver_onset_jets(
    contact_preparation,
):
    s = pytest.importorskip("sympy")
    _, model, fields, source, reset = contact_preparation
    loss = sum(field.continuous_loss for field in fields)
    local_budget = fields[0].work.dissipation[1] + fields[1].work.dissipation[0]
    cost, beta = reset.energy_change, F(model.storage_scale)
    rates = (local_budget / (beta + cost), beta * local_budget / (beta + cost) ** 2)
    assert rates == (F(1, 3), F(2, 9))

    t, rate = s.symbols("t rate", real=True)
    x = s.Matrix(source.epi) + t * s.Matrix(source.rate)
    a = rate * t
    energy = ((x[1] - x[0]) ** 2 + (x[3] - x[2]) ** 2 + a * (x[2] - x[1]) ** 2) / 2
    budget = s.diff(energy, t).subs(t, 0)
    assert s.simplify(budget + loss - cost * rate) == 0

    # These are first jets of the explicitly added, fully weighted closure.
    # At initial phase consensus the receiver's phase source has zero first
    # jet: its old neighbor is quiescent, and the new a*phase contrast is O(t²).
    # The phase metric equals pi*(1+a) to first order. Pi-scaled phase rates
    # therefore keep this coefficient calculation exactly rational.
    contrast = x[3] - x[2] + a * (x[1] - x[2])
    form_rate_jet = F(model.epi_weight) * contrast / (1 + a)
    pi_phase_rate_jet = -F(model.phase_weight) * contrast / (beta * (1 + a))
    assert form_rate_jet.subs(t, 0) == pi_phase_rate_jet.subs(t, 0) == 0
    form_acceleration = s.diff(form_rate_jet, t).subs(t, 0)
    pi_phase_acceleration = s.diff(pi_phase_rate_jet, t).subs(t, 0)
    assert form_acceleration == F(model.epi_weight) * rate
    assert pi_phase_acceleration == -F(model.phase_weight) * rate / beta
    assert tuple(form_acceleration.subs(rate, value) for value in rates) == (
        s.Rational(1, 6),
        s.Rational(1, 9),
    )
    assert tuple(pi_phase_acceleration.subs(rate, value) for value in rates) == (
        -s.Rational(1, 6),
        -s.Rational(1, 9),
    )
    assert tuple(budget.subs(rate, value) for value in rates) == (
        -s.Rational(5, 6),
        -s.Rational(8, 9),
    )
    # Both postulates meet this initial dissipation budget while predicting
    # distinct receiver acceleration. Neither is selected by that inequality,
    # and a finite jet proves no trajectory, duration or global phase domain.


def test_added_contact_energy_preserves_nodal_gradients_but_changes_edge_drive(
    contact_preparation,
):
    s = pytest.importorskip("sympy")
    _, model, _, source, reset = contact_preparation
    forms = s.symbols("x0:4", real=True)
    a = s.Symbol("a", nonnegative=True)
    beta = F(model.storage_scale)
    base_energy = (
        sum((forms[i] - forms[j]) ** 2 / 2 for i, j in ((0, 1), (2, 3)))
        + a * (forms[1] - forms[2]) ** 2 / 2
    )
    extra_energy = beta * (a**2 / 2 - a)
    changed_energy = base_energy + extra_energy
    assert tuple(s.diff(changed_energy, x) for x in forms) == tuple(
        s.diff(base_energy, x) for x in forms
    )
    prepared = {**dict(zip(forms, source.epi, strict=True)), a: 0}
    old_drive = s.diff(base_energy, a).subs(prepared)
    new_drive = s.diff(changed_energy, a).subs(prepared)
    assert old_drive == reset.energy_change > 0
    assert new_drive == reset.energy_change - beta < 0
    # The optional edge energy changes the sign of the gradient drive at zero
    # contact without changing any fixed-contact nodal gradient. Its selection
    # and the edge gradient-flow law are additional premises, not consequences
    # of the existing nodal equation or native component observations.


SHIFTS = (F(-1), F(-1, 2), F(-1, 4))


def _uncoupled_contact(shift, base=1):
    """Native a=0 fields, loss, port budget and cost rate for (base+shift,base,0,0).

    Phases agree, so the bridge cost is c=base^2/2 and c_dot=base*(x1_dot-x2_dot).
    """
    model = RelationalExchangeModel(storage_scale=1.0)
    fields = []
    for nodes, forms in (((0, 1), (base + shift, base)), ((2, 3), (0, 0))):
        graph = nx.Graph()
        graph.add_edge(*nodes, weight=1.0)
        for node, form in zip(nodes, forms, strict=True):
            graph.nodes[node].update(
                EPI=float(form), theta=0.0, nu_f=1.0, delta_nfr=0.0
            )
        fields.append(evaluate_relational_exchange(graph, model=model))
    loss = sum(field.continuous_loss for field in fields)
    budget = fields[0].work.dissipation[1] + fields[1].work.dissipation[0]
    cost_rate = base * (F(fields[0].form_rate[1]) - F(fields[1].form_rate[0]))
    return fields, F(model.storage_scale), loss, budget, cost_rate


def test_loss_funded_onset_vanishes_to_second_order_at_uncoupled_equilibrium():
    cost, measured = F(1, 2), {}
    for shift in (*SHIFTS, F(0)):
        _, beta, loss, budget, cost_rate = _uncoupled_contact(shift)
        # Independent edge calculation: q=(shift,-shift) on the left P2 gives
        # D=e*q^2 at each node, and the port form moves at rate e*shift.
        assert (loss, budget, cost_rate) == (shift**2, shift**2 / 2, shift / 2)
        rows = (budget / (beta + cost), beta * budget / (beta + cost) ** 2)
        assert rows == (shift**2 / 3, 2 * shift**2 / 9)
        measured[shift] = (cost_rate, loss, rows[0])
    for outer, inner in zip(SHIFTS[:-1], SHIFTS[1:], strict=True):
        # Halving the departure halves the nodal rate; loss and loss-funded
        # onset scale quadratically, so they vanish faster at equilibrium.
        assert measured[inner][0] == measured[outer][0] / 2
        assert measured[inner][1] == measured[outer][1] / 4
        assert measured[inner][2] == measured[outer][2] / 4
    assert measured[F(0)] == (0, 0, 0)


def test_derived_contact_coordinate_has_chain_rule_rate_and_gate_budget():
    s = pytest.importorskip("sympy")
    t, c2 = s.symbols("t c2", real=True)
    r, kappa = s.symbols("r kappa", positive=True)
    # The second jet c2 of the bridge cost is left unspecified: the gate's
    # first-order rate and the quadratic gate's acceleration use only r=-c_dot.
    drop = r * t - c2 * t**2 / 2
    linear, quadratic = drop / kappa, (drop / kappa) ** 2
    rate = s.diff(linear, t).subs(t, 0)
    acceleration = s.diff(quadratic, t, 2).subs(t, 0)
    assert rate == r / kappa
    assert s.diff(quadratic, t).subs(t, 0) == 0
    assert acceleration == 2 * (r / kappa) ** 2

    cost, passive = F(1, 2), {}
    for shift in SHIFTS:
        _, _, loss, _, cost_rate = _uncoupled_contact(shift)
        # Threshold kappa=c(0) places the preparation exactly on the gate.
        values = {r: -cost_rate, kappa: cost}
        assert rate.subs(values) == abs(shift)
        assert acceleration.subs(values) == 2 * shift**2
        # With Psi=0 the support force is c, so the linear gate's work is c*a_dot.
        passive[shift] = cost * abs(shift) <= loss
    # Support work is first order in the nodal rate, the loss second order.
    assert passive == {F(-1): True, F(-1, 2): True, F(-1, 4): False}


def _contact_storage(s):
    """Fully weighted storage of the contact fixture for beta=1, Psi=a^2/2-a."""
    x, theta = s.symbols("x0:4", real=True), s.symbols("th0:4", real=True)
    a = s.Symbol("a", real=True)
    edges = {(0, 1): 1, (2, 3): 1, (1, 2): a}
    nodal = sum(
        w * ((x[i] - x[j]) ** 2 / 2 + 1 - s.cos(theta[j] - theta[i]))
        for (i, j), w in edges.items()
    )
    return x, theta, a, nodal + a**2 / 2 - a


def test_gradient_support_row_sets_uncoupled_stability_by_mismatch_cost():
    s = pytest.importorskip("sympy")
    x, theta, a, stored = _contact_storage(s)
    flat = {v: 0 for v in theta}
    force = s.diff(stored, a)

    def uncoupled(jump):
        return {**dict(zip(x, (jump, jump, 0, 0), strict=True)), **flat, a: 0}

    # k*=c*-kappa with c*=jump^2/2 and kappa=-Psi'(0)=1; the sign changes at
    # c*=kappa, where the uncoupled state stops being a local minimum.
    jump = s.Symbol("jump", positive=True)
    assert s.solve(force.subs(uncoupled(jump)), jump) == [s.sqrt(2)]
    assert [force.subs(uncoupled(value)) for value in (1, 2)] == [-s.Rational(1, 2), 1]
    # Along the support axis at fixed X*, S_tilde falls for c*<kappa, rises above.
    along = {v: stored.subs({**uncoupled(v), a: F(1, 10)}) for v in (1, 2)}
    assert along[1] < 0 < along[2]

    # At an equilibrium D=0 natively, so passivity (-D+k*a_dot<=0) bounds a_dot
    # by D/k=0 when k*>0 and leaves the gradient row a_dot=-k>0 when k*<0.
    for base, forced_zero in ((2, True), (1, False)):
        _, _, loss, _, _ = _uncoupled_contact(F(0), base=base)
        assert loss == 0
        k = force.subs(uncoupled(base))
        assert bool(k > 0) == forced_zero
        if not forced_zero:
            assert -loss + k * (-k) == -s.Rational(1, 4)

    # Storage balance of the gradient row at the first-contact preparation (a=0)
    # with the native nodal rates: S_dot=-D-m*k^2 for m=1.
    fields, _, loss, _, _ = _uncoupled_contact(F(-1))
    point = {**dict(zip(x, (0, 1, 0, 0), strict=True)), **flat, a: 0}
    k = force.subs(point)
    form_rates = fields[0].form_rate + fields[1].form_rate
    phase_rates = fields[0].phase_rate + fields[1].phase_rate
    budget = k * (-k) + sum(
        s.diff(stored, v).subs(point) * F(rate)
        for v, rate in zip((*x, *theta), (*form_rates, *phase_rates), strict=True)
    )
    assert s.simplify(budget + loss + k**2) == 0
    assert budget == -s.Rational(5, 4)


def test_coupled_uniform_state_is_a_lower_minimum_of_the_joint_storage():
    s = pytest.importorskip("sympy")
    x, theta, a, stored = _contact_storage(s)
    flat = {v: 0 for v in theta}
    uniform = {**{v: 0 for v in x}, **flat, a: 1}
    separated = {**dict(zip(x, (2, 2, 0, 0), strict=True)), **flat, a: 0}
    # Nodal terms are nonnegative, so S_tilde>=Psi(a)>=Psi(1)=-1/2 with equality
    # only at uniform form and consensus phase; the uncoupled state stores Psi(0)=0.
    assert stored.subs(uniform) == -s.Rational(1, 2)
    assert stored.subs(separated) == 0
    hessian = s.hessian(stored, (*x, *theta, a)).subs(uniform)
    eigenvalues = np.linalg.eigvalsh(np.array(hessian.tolist(), dtype=float))
    # The cross derivatives equal grad(c)=0 here, leaving two copies of the
    # weighted path Laplacian (form and phase) and Psi''=1 for the support.
    path = np.linalg.eigvalsh(nx.laplacian_matrix(nx.path_graph(4)).toarray())
    expected = np.sort(np.concatenate([path, path, [1.0]]))
    assert eigenvalues == pytest.approx(expected, rel=0, abs=1e-12)
    assert np.sum(np.abs(eigenvalues) < 1e-12) == 2


def _winding_ring(offset, n, *, phase_shift=0.0, epi=0.0):
    """Acute winding-one n-ring: equal gaps 2*pi/n<pi/2 at a stable phase min."""
    graph = nx.Graph()
    nodes = [offset + i for i in range(n)]
    for i in range(n):
        graph.add_edge(nodes[i], nodes[(i + 1) % n], weight=1.0)
    for i, node in enumerate(nodes):
        graph.nodes[node].update(
            EPI=epi, theta=2 * pi * i / n + phase_shift, nu_f=1.0, delta_nfr=0.0
        )
    return graph


def _disjoint(left, right):
    graph = nx.Graph()
    graph.add_nodes_from(left.nodes(data=True))
    graph.add_nodes_from(right.nodes(data=True))
    graph.add_edges_from(left.edges(data=True))
    graph.add_edges_from(right.edges(data=True))
    return graph


def test_composite_internal_winding_storage_is_decoupled_from_first_contact():
    changes = []
    for n in (5, 7):
        gap = 2 * pi / n
        # Acute equal-gap winding one is a strict local phase-storage minimum.
        assert gap < pi / 2 and cos(gap) > 0
        before = _disjoint(_winding_ring(0, n), _winding_ring(n, n))
        after = before.copy()
        after.add_edge(0, n, weight=1.0)
        reset = observe_relational_reset(before, after, storage_scale=1.0)
        internal = 2 * n * (1 - cos(gap))
        assert float(reset.storage_before) == pytest.approx(internal, rel=0, abs=1e-12)
        # Aligned ports (theta=0, EPI=0): the first-contact cost is exactly zero
        # although the retained internal winding storage is large and differs.
        assert reset.storage_change == 0
        changes.append((internal, reset.storage_change))
    internals = [internal for internal, _ in changes]
    assert internals[0] != pytest.approx(internals[1])
    assert {change for _, change in changes} == {Fraction(0)}


def test_composite_first_contact_cost_is_port_only_and_nonnegative():
    beta = Fraction(3, 2)
    for shift in (pi / 3, pi / 2, 2 * pi / 3):
        before = _disjoint(_winding_ring(0, 5), _winding_ring(5, 5, phase_shift=shift))
        after = before.copy()
        after.add_edge(0, 5, weight=1.0)
        reset = observe_relational_reset(before, after, storage_scale=beta)
        # Only the port gap enters: beta*(1-cos delta), never negative.
        assert float(reset.storage_change) == pytest.approx(
            float(beta) * (1 - cos(shift)), rel=0, abs=1e-12
        )
        assert reset.phase_storage_change > 0
        assert reset.form_storage_change == 0
    # An EPI contrast at the ports adds r^2/2 to the form channel only.
    before = _disjoint(_winding_ring(0, 5, epi=0.0), _winding_ring(5, 5, epi=0.5))
    after = before.copy()
    after.add_edge(0, 5, weight=1.0)
    reset = observe_relational_reset(before, after, storage_scale=beta)
    assert reset.form_storage_change == Fraction(1, 8)
    assert reset.phase_storage_change == 0
    assert reset.storage_change == Fraction(1, 8)


def test_winding_class_internal_equilibrium_retains_storage_without_contact_drive():
    s = pytest.importorskip("sympy")
    g0, g1 = s.symbols("g0 g1", real=True)
    # Winding-one gap class on a 5-ring: four free gaps, the fifth closes 2*pi.
    g = (g0, g1, g0, g1)
    g_last = 2 * s.pi - sum(g)
    storage = sum(1 - s.cos(angle) for angle in (*g, g_last))
    symmetric = {g0: 2 * s.pi / 5, g1: 2 * s.pi / 5}
    gradient = [s.diff(storage, v).subs(symmetric) for v in (g0, g1)]
    assert [s.simplify(entry) for entry in gradient] == [0, 0]
    hessian = s.Matrix(2, 2, lambda i, j: s.diff(storage, (g0, g1)[i], (g0, g1)[j]))
    eigenvalues = hessian.subs(symmetric).eigenvals()
    # Positive-definite on the gap class: the winding storage is a strict min,
    # so eliminating the internal coordinates retains it; cos(2*pi/5)>0.
    assert all(s.re(value) > 0 for value in eigenvalues)
    internal = s.simplify(storage.subs(symmetric))
    assert internal == s.nsimplify(5 * (1 - s.cos(2 * s.pi / 5)))

    # The envelope slope of a supplied port contact reads only the port gap.
    a, port_gap = s.symbols("a port_gap", real=True, nonnegative=True)
    contact = a * (1 - s.cos(port_gap))
    slope = s.diff(storage.subs(symmetric) + contact, a).subs(a, 0)
    assert slope == 1 - s.cos(port_gap)
    assert slope.subs(port_gap, 0) == 0
    assert slope.subs(port_gap, s.pi / 3) == s.Rational(1, 2)


def _internal_composite(contrast):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.graph.update(GAMMA={"type": "none"}, _t=0.0)
    for node, form in zip((0, 1), (contrast, 0.0), strict=True):
        graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0, delta_nfr=0.0)
    return graph


def test_nonequilibrium_internal_excess_is_nonnegative_dissipating_and_zero_at_eq():
    model = RelationalExchangeModel(storage_scale=1.0)
    for contrast in (1.0, 0.5, 0.25, 0.0):
        field = evaluate_relational_exchange(_internal_composite(contrast), model=model)
        value = Fraction(contrast)
        # Excess over the uniform equilibrium is nonnegative; dissipation and
        # storage rate vanish together exactly at the equilibrium contrast 0.
        assert field.storage == value**2 / 2 >= 0
        assert field.continuous_loss == value**2
        assert field.storage_rate == -(value**2)
    # The prepared reservoir relaxes: storage strictly decreases under evolution.
    graph = _internal_composite(1.0)
    previous = None
    for _ in range(4):
        storage = evaluate_relational_exchange(graph, model=model).storage
        assert previous is None or storage < previous
        previous = storage
        step_relational_exchange(graph, model=model, dt=0.25)
    assert previous > 0
