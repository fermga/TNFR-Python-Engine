"""Static complete-law comparison with independent work and admission controls."""

import json
import math
import pickle
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_comparison as owner

MODEL = RelationalExchangeModel(1, phase_domain="regular")
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)
PI = Q(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628"
)


def _state(
    graph=None, *, forms=(1, -0.5, 0.25), phases=(0, 0.5, -0.25), capacities=(1, 2, 3)
):
    graph = nx.path_graph(3) if graph is None else graph
    for node, x, theta, nu in zip(graph, forms, phases, capacities):
        graph.nodes[node].update(EPI=x, theta=theta, nu_f=nu, delta_nfr=999)
    graph.graph.update(GAMMA={"type": "none"}, untouched={"list": [1, 2]})
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _trig(angle, *, sine=False):
    """Independent high-precision power series, for bounded test angles only."""
    with localcontext() as context:
        context.prec = 100
        x = Decimal(angle.numerator) / Decimal(angle.denominator)
        term = total = x if sine else Decimal(1)
        for k in range(1, 100):
            denominator = 2 * k * (2 * k + 1) if sine else (2 * k - 1) * 2 * k
            term *= -x * x / denominator
            total += term
        return Q(total)


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("a static comparison must not advance a trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def sample():
    return owner.bound_relational_sine_exchange(_state(), reference_model=MODEL)


def test_exact_consensus_phase_rates_and_loss():
    report = owner.bound_relational_sine_exchange(
        _state(forms=(1, 0, -1), phases=(0, 0, 0)), reference_model=MODEL
    )
    assert report.form_gradient == (1, 0, -1)
    assert report.phase_sources == (I(0),) * 3
    assert report.form_rates == (I(Q(-1, 2)), I(0), I(Q(3, 2)))
    expected_phase = (1 / (2 * PI), Q(0), -3 / (2 * PI))
    assert all(
        bound.contains(value)
        for bound, value in zip(report.phase_rates, expected_phase)
    )
    assert report.form_storage == 1
    assert report.phase_storage == I(0)
    assert report.continuous_loss == 2
    assert report.storage_rate == I(-2)
    assert report.balance_residual == I(0)


def test_independent_complete_field_and_work(sample):
    x, nu = sample.epi, sample.capacity
    phase = sample.phase
    # Independent P3 edge incidences and exact Dirichlet gradient.
    currents = (
        _trig(phase[1] - phase[0], sine=True),
        _trig(phase[0] - phase[1], sine=True) + _trig(phase[2] - phase[1], sine=True),
        _trig(phase[1] - phase[2], sine=True),
    )
    q = (x[0] - x[1], 2 * x[1] - x[0] - x[2], x[2] - x[1])
    storage = ((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2) / 2
    storage += 2 - _trig(phase[1] - phase[0]) - _trig(phase[2] - phase[1])
    assert sample.storage.contains(storage)
    exact_loss = Q(0)
    for i, degree in enumerate((1, 2, 1)):
        source = currents[i] / (PI * degree)
        pressure = (-q[i] / degree + source) / 2
        form_rate = nu[i] * pressure
        phase_rate = nu[i] * q[i] / (2 * PI * degree)
        exact_loss += nu[i] * q[i] ** 2 / (2 * degree)
        assert sample.phase_sources[i].contains(source)
        assert sample.pressure[i].contains(pressure)
        assert sample.form_rates[i].contains(form_rate)
        assert sample.phase_rates[i].contains(phase_rate)
        assert sample.form_work[i].contains(q[i] * form_rate)
        assert sample.phase_work[i].contains(-currents[i] * phase_rate)
        assert sample.node_balance_residual[i].contains(0)
    assert sample.continuous_loss == exact_loss
    assert sample.storage_rate.contains(-exact_loss)
    assert sample.balance_residual.contains(0)
    assert sample.balance_residual.width > 0  # Computed, not an installed zero.


def test_weighted_form_and_lifted_phase_rates_conserve(sample):
    weights = tuple(Q(d) / nu for d, nu in zip(sample.degrees, sample.capacity))
    for rates in (sample.form_rates, sample.phase_rates):
        assert sum(
            (weight * rate for weight, rate in zip(weights, rates)), I(0)
        ).contains(0)


def test_current_squared_mobility_retains_both_full_rows_and_storage_balance():
    model = RelationalExchangeModel(Q(3, 2), epi_weight=0, phase_domain="regular")
    source = owner.bound_relational_sine_exchange(
        _state(capacities=(Q(1, 2), 0, 2)), reference_model=model
    )
    report = source.with_current_squared_mobility(epsilon=1)
    phase, x, nu = source.phase, source.epi, source.capacity
    currents = (
        _trig(phase[1] - phase[0], sine=True),
        _trig(phase[0] - phase[1], sine=True) + _trig(phase[2] - phase[1], sine=True),
        _trig(phase[1] - phase[2], sine=True),
    )
    gradient = (x[0] - x[1], 2 * x[1] - x[0] - x[2], x[2] - x[1])
    for i, degree in enumerate((1, 2, 1)):
        factor = 1 + (currents[i] / degree) ** 2
        mobility = factor / (PI * degree)
        pressure = mobility * currents[i]
        phase_rate = nu[i] * mobility * gradient[i] / Q(3, 2)
        assert report.mobility_factors[i].contains(factor)
        assert report.inverse_phase_metric[i].contains(mobility)
        assert report.pressure[i].contains(pressure)
        assert report.form_rates[i].contains(nu[i] * pressure)
        assert report.phase_rates[i].contains(phase_rate)
        assert report.form_work[i].contains(gradient[i] * nu[i] * pressure)
        assert report.phase_work[i].contains(-Q(3, 2) * currents[i] * phase_rate)
        assert report.node_balance_residual[i].contains(0)
    assert report.form_rates[1] == report.phase_rates[1] == I(0)
    assert report.continuous_loss == 0
    assert report.storage_rate.contains(0)
    assert report.balance_residual.width > 0
    assert report.comparison is source
    assert report.law != source.law


def test_zero_epsilon_recovers_source_and_form_reversal_keeps_the_declared_mobility():
    model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
    graph = _state()
    source = owner.bound_relational_sine_exchange(graph, reference_model=model)
    zero = source.with_current_squared_mobility(epsilon=0)
    for name in (
        "inverse_phase_metric",
        "phase_sources",
        "pressure",
        "form_rates",
        "phase_rates",
        "dissipation",
        "continuous_loss",
        "form_work",
        "phase_work",
        "node_balance_residual",
        "storage_rate",
        "balance_residual",
    ):
        assert getattr(zero, name) == getattr(source, name)
    assert zero.mobility_factors == (I(1),) * 3
    assert zero.mobility_corrections == (I(0),) * 3
    reversed_graph = graph.copy()
    for node in reversed_graph:
        reversed_graph.nodes[node]["EPI"] *= -1
    original = source.with_current_squared_mobility(epsilon=1)
    reversed_report = owner.bound_relational_sine_exchange(
        reversed_graph, reference_model=model
    ).with_current_squared_mobility(epsilon=1)
    assert original.mobility_factors == reversed_report.mobility_factors
    assert original.form_rates == reversed_report.form_rates
    assert original.phase_rates == tuple(
        -value for value in reversed_report.phase_rates
    )


def test_recursive_replication_inherits_both_laws_without_selecting_one():
    """Complete blow-ups preserve rows, not a newly inferred hierarchy."""
    model = RelationalExchangeModel(Q(3, 2), epi_weight=0, phase_domain="regular")
    base = _state(capacities=(Q(1, 2), Q(3, 2), 2))

    def replicate(graph, copies):
        fine = nx.Graph()
        fine.graph.update(graph.graph)
        for node, attributes in graph.nodes(data=True):
            for copy in range(copies):
                fine.add_node((node, copy), **attributes)
        for left, right in graph.edges:
            fine.add_edges_from(
                ((left, i), (right, j)) for i in range(copies) for j in range(copies)
            )
        return fine

    doubled = replicate(base, 2)
    nested = replicate(doubled, 3)
    direct = replicate(base, 6)
    # Iterating fibers changes labels only: no within-fiber links are added.
    flatten = {
        ((i, r), s): (i, 3 * r + s) for i in base for r in range(2) for s in range(3)
    }
    assert {
        frozenset((flatten[left], flatten[right])) for left, right in nested.edges
    } == {frozenset(edge) for edge in direct.edges}
    for node in nested:
        assert nested.nodes[node] == direct.nodes[flatten[node]]

    graphs = (base, doubled, nested, direct)
    before = tuple(_snapshot(graph) for graph in graphs)
    captures = tuple(
        owner.bound_relational_sine_exchange(graph, reference_model=model)
        for graph in graphs
    )
    source = captures[0]
    x, theta, nu = source.epi, source.phase, source.capacity
    # Independent three-node incidences and trigonometric series, not a
    # projection of the returned fine rates.
    currents = (
        _trig(theta[1] - theta[0], sine=True),
        _trig(theta[0] - theta[1], sine=True) + _trig(theta[2] - theta[1], sine=True),
        _trig(theta[1] - theta[2], sine=True),
    )
    gradient = (x[0] - x[1], 2 * x[1] - x[0] - x[2], x[2] - x[1])
    phase_storage = sum(1 - _trig(theta[j] - theta[i]) for i, j in ((0, 1), (1, 2)))
    energy = sum((x[j] - x[i]) ** 2 / 2 for i, j in ((0, 1), (1, 2)))
    energy += Q(3, 2) * phase_storage
    degrees = (1, 2, 1)
    parents = (
        lambda node: node,
        lambda node: node[0],
        lambda node: node[0][0],
        lambda node: node[0],
    )
    for capture, copies, parent in zip(captures, (1, 2, 6, 6), parents):
        assert len(capture.nodes) == 3 * copies
        assert len(capture.edges) == 2 * copies**2
        assert capture.form_storage == copies**2 * source.form_storage
        assert capture.phase_storage.contains(copies**2 * phase_storage)
        assert capture.storage.contains(copies**2 * energy)
        assert (capture.storage - copies**2 * source.storage).contains(0)
        for epsilon in (0, 1):
            report = capture.with_current_squared_mobility(epsilon=epsilon)
            assert report.comparison is capture
            assert report.continuous_loss == 0
            assert report.storage_rate.contains(0)
            for row, node in enumerate(capture.nodes):
                i = parent(node)
                assert (
                    capture.epi[row],
                    capture.phase[row],
                    capture.capacity[row],
                ) == (x[i], theta[i], nu[i])
                assert capture.degrees[row] == copies * degrees[i]
                assert capture.form_gradient[row] == copies * gradient[i]
                current_bound = capture.relative_resultant[row][1]
                assert current_bound.contains(copies * currents[i])
                assert (current_bound / capture.degrees[row]).contains(
                    currents[i] / degrees[i]
                )
                factor = 1 + epsilon * (currents[i] / degrees[i]) ** 2
                mobility = factor / (PI * degrees[i])
                assert report.mobility_factors[row].contains(factor)
                assert report.inverse_phase_metric[row].contains(mobility / copies)
                assert report.form_rates[row].contains(nu[i] * mobility * currents[i])
                assert report.phase_rates[row].contains(
                    nu[i] * mobility * gradient[i] / Q(3, 2)
                )
    # Inheritance at both levels did not force equality of the nonlinear laws.
    original = source.with_current_squared_mobility(epsilon=0)
    alternative = source.with_current_squared_mobility(epsilon=1)
    assert original.form_rates[0].hi < alternative.form_rates[0].lo
    assert original.phase_rates[0].hi < alternative.phase_rates[0].lo
    assert tuple(_snapshot(graph) for graph in graphs) == before


def test_mobility_alternative_admission_detachment_and_export(monkeypatch, tmp_path):
    from tnfr.sdk import export_to_json, relational_report_to_dict

    graph = _state()
    before = _snapshot(graph)
    source = owner.bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("a captured mobility alternative must not recapture its graph")

    # Limit the guard to the operation under test. SDK projection lazily imports
    # consumers; importing them with a patched public capture function would
    # retain that test double after monkeypatch restoration in later tests.
    with monkeypatch.context() as capture_guard:
        capture_guard.setattr(owner, "bound_relational_sine_exchange", forbidden)
        report = source.with_current_squared_mobility(epsilon=Q(1))
    assert _snapshot(graph) == before
    assert report.epsilon == 1
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-mobility-comparison.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "mobility-comparison.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload
    for value in (True, "1", -1, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            source.with_current_squared_mobility(epsilon=value)
    assert source.with_current_squared_mobility(epsilon=Q(1, 2**1100)).epsilon == Q(
        1, 2**1100
    )
    with pytest.raises(ValueError, match="zero form loss"):
        replace(source, reference_model=MODEL).with_current_squared_mobility(epsilon=1)


def test_mobility_relative_field_mean_drift_and_divergence_match_full_derivatives():
    graph = _state(capacities=(Q(1, 2), 1, 2))
    source = owner.bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )
    report = source.with_current_squared_mobility(epsilon=1).relative_balance(
        reference_node=1
    )
    assert report.relative_indices == (0, 2)
    assert report.relative_field_closure_certified
    assert not report.constant_mobility_mean_and_volume_identities_certified
    assert not report.divergence_bounds.contains(0)
    assert not report.weighted_form_mean_rate_bounds.contains(0)
    with mp.workdps(100):

        def number(value):
            value = Q(value)
            return mp.mpf(value.numerator) / value.denominator

        def contains(bound, value):
            assert number(bound.lo) <= value <= number(bound.hi)

        state = mp.matrix([number(value) for value in source.epi + source.phase])
        capacity = tuple(number(value) for value in source.capacity)

        def field(z):
            form, phase = [], []
            for i in graph:
                neighbors = tuple(graph[i])
                s = sum(mp.sin(z[3 + j] - z[3 + i]) for j in neighbors)
                q = sum(z[i] - z[j] for j in neighbors)
                m = (1 + (s / len(neighbors)) ** 2) / (mp.pi * len(neighbors))
                form.append(capacity[i] * m * s)
                phase.append(capacity[i] * m * q)
            return mp.matrix(form + phase)

        actual = field(state)
        for k, i in enumerate(report.relative_indices):
            assert report.relative_form[k] == source.epi[i] - source.epi[1]
            assert report.relative_phase[k] == source.phase[i] - source.phase[1]
            contains(report.relative_form_rates[k], actual[i] - actual[1])
            contains(report.relative_phase_rates[k], actual[3 + i] - actual[4])
        rho = [number(graph.degree[i]) / capacity[i] for i in graph]
        mean_form = sum(rho[i] * actual[i] for i in graph) / sum(rho)
        mean_phase = sum(rho[i] * actual[3 + i] for i in graph) / sum(rho)
        contains(report.weighted_form_mean_rate_bounds, mean_form)
        contains(report.weighted_lifted_phase_mean_rate_bounds, mean_phase)
        assert report.weighted_form_mean_rate_residual_bounds.contains(0)
        assert report.weighted_lifted_phase_mean_rate_residual_bounds.contains(0)
        divergence = mp.mpf(0)
        for k in range(6):
            direction = mp.zeros(6, 1)
            direction[k] = 1
            divergence += mp.diff(lambda t: field(state + t * direction)[k], 0)
        contains(report.divergence_bounds, divergence)
        relative = mp.matrix(
            [
                state[0] - state[1],
                state[2] - state[1],
                state[3] - state[4],
                state[5] - state[4],
            ]
        )

        def reduced(z):
            lifted = mp.matrix([z[0], 0, z[1], z[2], 0, z[3]])
            f = field(lifted)
            return mp.matrix([f[0] - f[1], f[2] - f[1], f[3] - f[4], f[5] - f[4]])

        relative_trace = mp.mpf(0)
        for k in range(4):
            direction = mp.zeros(4, 1)
            direction[k] = 1
            relative_trace += mp.diff(lambda t: reduced(relative + t * direction)[k], 0)
        assert abs(divergence - relative_trace) < mp.mpf("1e-95")
        contains(report.relative_divergence_bounds, relative_trace)


def test_mobility_relative_origins_are_gauges_not_static_reference_rows():
    model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
    graph = _state(capacities=(1, 1, 1))
    source = owner.bound_relational_sine_exchange(graph, reference_model=model)
    report = source.with_current_squared_mobility(epsilon=1).relative_balance(
        reference_node=0
    )
    shifted = graph.copy()
    for node in shifted:
        shifted.nodes[node]["EPI"] += 8
        shifted.nodes[node]["theta"] -= 4
    changed = (
        owner.bound_relational_sine_exchange(shifted, reference_model=model)
        .with_current_squared_mobility(epsilon=1)
        .relative_balance(reference_node=0)
    )
    assert changed.relative_form == report.relative_form
    assert changed.relative_phase == report.relative_phase
    assert changed.relative_form_rates == report.relative_form_rates
    assert changed.relative_phase_rates == report.relative_phase_rates
    assert changed.divergence_bounds == report.divergence_bounds
    assert changed.weighted_form_mean == report.weighted_form_mean + 8
    assert changed.weighted_lifted_phase_mean == report.weighted_lifted_phase_mean - 4
    assert not report.mobility.phase_rates[0].contains(0)
    assert not (
        report.relative_phase_rates[0] - report.mobility.phase_rates[1]
    ).contains(0)
    zero = source.with_current_squared_mobility(epsilon=0).relative_balance(
        reference_node=0
    )
    assert zero.weighted_form_mean_rate_bounds == I(0)
    assert zero.weighted_lifted_phase_mean_rate_bounds == I(0)
    assert zero.divergence_bounds == zero.relative_divergence_bounds == I(0)
    assert zero.constant_mobility_mean_and_volume_identities_certified
    flat = (
        owner.bound_relational_sine_exchange(
            _state(forms=(0, 0, 0), phases=(0, 0, 0)), reference_model=model
        )
        .with_current_squared_mobility(epsilon=1)
        .relative_balance(reference_node=0)
    )
    assert flat.divergence_bounds == I(0)
    assert not flat.constant_mobility_mean_and_volume_identities_certified


def test_mobility_relative_balance_detachment_admission_labels_and_export(
    monkeypatch, tmp_path
):
    from tnfr.sdk import export_to_json, relational_report_to_dict

    model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
    source = owner.bound_relational_sine_exchange(
        _state(), reference_model=model
    ).with_current_squared_mobility(epsilon=1)

    def forbidden(*args, **kwargs):
        pytest.fail("relative balance must reuse its detached capture")

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", forbidden)
    report = source.relative_balance(reference_node=2)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-mobility-relative-balance.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "relative-balance.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload
    with pytest.raises(ValueError, match="captured node"):
        source.relative_balance(reference_node="missing")
    with pytest.raises(ValueError, match="strictly positive"):
        replace(
            source, comparison=replace(source.comparison, capacity=(0, 1, 1))
        ).relative_balance(reference_node=0)

    @dataclass(frozen=True)
    class OpaqueReference:
        index: int

    with pytest.raises(TypeError):
        replace(report, reference_node=OpaqueReference(0)).to_dict()


def test_native_and_comparison_agree_at_consensus_but_not_on_nonlinear_mean():
    consensus = _state(forms=(1, 0, -1), phases=(0, 0, 0))
    comparison = owner.bound_relational_sine_exchange(consensus, reference_model=MODEL)
    native = relational.evaluate_relational_exchange(consensus, model=MODEL)
    for interval, represented in zip(comparison.form_rates, native.form_rate):
        assert interval.contains(Q(represented))
    # Native phase rates contain represented pi; this verifies wiring, not
    # enclosure of the floating-point field by a mathematical-pi interval.
    for interval, represented in zip(comparison.phase_rates, native.phase_rate):
        assert float(interval.midpoint) == pytest.approx(represented, rel=2e-15)

    graph = _state(
        nx.star_graph(3),
        forms=(0,) * 4,
        phases=(0, 0, 0, math.pi / 2),
        capacities=(1,) * 4,
    )
    comparison = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    native = relational.evaluate_relational_exchange(graph, model=MODEL)
    smooth_mean = (
        sum(
            (
                degree * rate
                for degree, rate in zip(comparison.degrees, comparison.form_rates)
            ),
            I(0),
        )
        / 6
    )
    reference_mean = (
        sum(
            Q(degree) * Q(rate)
            for degree, rate in zip(comparison.degrees, native.form_rate)
        )
        / 6
    )
    assert smooth_mean.contains(0)
    assert smooth_mean.width < Q(1, 10**18)
    assert reference_mean < Q(-1, 1000)
    assert float(reference_mean) == pytest.approx(
        (3 * math.atan(0.5) / math.pi - 0.5) / 12, abs=2e-16
    )
    assert comparison.phase_rates == (I(0),) * 4
    assert native.phase_rate == (0,) * 4
    assert comparison.continuous_loss == native.continuous_loss == 0


def test_native_admission_and_field_are_not_evaluated(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("comparison must not evaluate a native phase field")

    monkeypatch.setattr(relational, "_field", forbidden)
    monkeypatch.setattr(relational, "evaluate_relational_exchange", forbidden)
    graph = _state()
    before = _snapshot(graph)
    report = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    assert _snapshot(graph) == before
    assert report.law == "normalized_sine_reciprocal_exchange"
    assert report.reference_model is MODEL


@pytest.mark.parametrize("phases", [(0, math.pi), (0, 0, math.pi)])
def test_boundary_probes_with_unresolved_native_branch_have_finite_comparison(phases):
    # These binary64 preparations probe the exact antipodal/zero boundaries;
    # represented pi is not promoted to mathematical pi or an exact zero.
    n = len(phases)
    graph = _state(
        nx.path_graph(n), forms=tuple(range(n)), phases=phases, capacities=(1,) * n
    )
    with pytest.raises(ValueError, match="branch|ray|resultant"):
        relational.evaluate_relational_exchange(graph, model=MODEL)
    report = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    for bound in (*report.pressure, *report.form_rates, *report.phase_rates):
        assert bound.lo <= bound.hi
        assert bound.abs_max < 10
    assert report.balance_residual.contains(0)


def test_zero_capacity_freezes_both_declared_rows():
    report = owner.bound_relational_sine_exchange(
        _state(capacities=(0, 2, 0)), reference_model=MODEL
    )
    for i in (0, 2):
        assert not report.pressure[i].contains(0)
        assert report.form_rates[i] == report.phase_rates[i] == I(0)
        assert report.dissipation[i] == 0


def test_cancellation_probe_retains_nonzero_phase_response_to_form():
    graph = _state(
        nx.path_graph(3), forms=(0, 1, 0), phases=(0, 0, math.pi), capacities=(1,) * 3
    )
    report = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    assert report.form_gradient[1] == 2
    # Binary64 pi probes cancellation; it is not an exact zero resultant.
    assert report.relative_resultant[1][0].abs_max < Q(1, 10**12)
    assert report.relative_resultant[1][1].abs_max < Q(1, 10**12)
    assert report.phase_rates[1].lo > Q(1, 10)
    assert report.phase_rates[1].hi < Q(1, 5)
    assert report.balance_residual.contains(0)


def test_changed_phase_geometry_changes_pressure_but_not_comparison_phase_rate(sample):
    altered = owner.bound_relational_sine_exchange(
        _state(phases=(0, -0.5, 0.25)), reference_model=MODEL
    )
    assert altered.phase_rates == sample.phase_rates
    assert any(
        left.hi < right.lo or right.hi < left.lo
        for left, right in zip(sample.form_rates, altered.form_rates)
    )


def test_capacity_linearity_and_beta_only_changes_phase_row(sample):
    double = owner.bound_relational_sine_exchange(
        _state(capacities=(2, 4, 6)), reference_model=MODEL
    )
    beta = owner.bound_relational_sine_exchange(
        _state(), reference_model=RelationalExchangeModel(2, phase_domain="regular")
    )
    for field in ("form_rates", "phase_rates"):
        for first, second in zip(getattr(sample, field), getattr(double, field)):
            assert abs(second.midpoint - 2 * first.midpoint) < Q(1, 10**28)
    assert double.continuous_loss == 2 * sample.continuous_loss
    assert beta.form_rates == sample.form_rates
    for first, second in zip(sample.phase_rates, beta.phase_rates):
        assert abs(2 * second.midpoint - first.midpoint) < Q(1, 10**28)


def test_effective_pressure_coefficients_are_affine(sample):
    reports = tuple(
        owner.bound_relational_sine_exchange(
            _state(),
            reference_model=RelationalExchangeModel(
                1, epi_weight=e, phase_weight=1 - e, phase_domain="regular"
            ),
        )
        for e in (0.25, 0.75)
    )
    for field in ("pressure", "form_rates", "phase_rates"):
        for first, left, right in zip(
            getattr(sample, field),
            getattr(reports[0], field),
            getattr(reports[1], field),
        ):
            assert abs((left.midpoint + right.midpoint) / 2 - first.midpoint) < Q(
                1, 10**28
            )


def test_exact_offset_and_relabel_covariance(sample):
    graph = _state(forms=(9, 7.5, 8.25), phases=(1.25, 1.75, 1))
    graph = nx.relabel_nodes(graph, {0: "left", 1: ("middle", 1), 2: "right"})
    report = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    assert report.nodes == ("left", ("middle", 1), "right")
    for field in (
        "pressure",
        "form_rates",
        "phase_rates",
        "storage",
        "continuous_loss",
    ):
        assert getattr(report, field) == getattr(sample, field)


def test_conjugate_form_and_phase_reversal(sample):
    report = owner.bound_relational_sine_exchange(
        _state(forms=(-1, 0.5, -0.25), phases=(0, -0.5, 0.25)), reference_model=MODEL
    )
    for field in ("pressure", "form_rates", "phase_rates"):
        for first, reverse in zip(getattr(sample, field), getattr(report, field)):
            assert abs(first.midpoint + reverse.midpoint) < Q(1, 10**28)
    assert report.storage == sample.storage


@pytest.mark.parametrize("model", [None, object(), RelationalExchangeModel(1)])
def test_requires_explicit_regular_reference(model):
    with pytest.raises(ValueError, match="regular reference"):
        owner.bound_relational_sine_exchange(_state(), reference_model=model)


@pytest.mark.parametrize(
    "key,value",
    [
        ("EPI", True),
        ("EPI", float("nan")),
        ("theta", float("inf")),
        ("theta", False),
        ("nu_f", -1),
        ("nu_f", True),
    ],
)
def test_shared_authoritative_admission_rejects_invalid_state_without_mutation(
    key, value
):
    graph = _state()
    graph.nodes[0][key] = value
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    "change", ["directed", "nonunit", "loop", "disconnected", "gamma"]
)
def test_shared_support_and_input_admission(change):
    graph = _state()
    if change == "directed":
        graph = graph.to_directed()
    elif change == "nonunit":
        graph.edges[0, 1]["weight"] = 2
    elif change == "loop":
        graph.add_edge(0, 0)
    elif change == "disconnected":
        graph.remove_edge(1, 2)
    else:
        graph.graph["GAMMA"] = {"type": "constant", "value": 1}
    with pytest.raises(ADMISSION_ERRORS):
        owner.bound_relational_sine_exchange(graph, reference_model=MODEL)


def test_export_keeps_distinct_law_and_exact_evidence(sample):
    payload = sample.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["schema"] == "tnfr.relational-sine-comparison.v1"
    assert payload["report"]["law"] == "normalized_sine_reciprocal_exchange"
    assert payload["report"]["form_gradient"][0] == {"numerator": 3, "denominator": 2}
    assert set(payload["report"]["pressure"][0]) == {"lo", "hi"}
    scope = " ".join(sample.scope)
    assert "not_a_native_boundary_continuation" in scope
    assert "no_frozen_response_reinterpretation" in scope


@dataclass(frozen=True)
class _OpaqueLabel:
    name: str


@pytest.mark.parametrize(
    "label", [_OpaqueLabel("node"), Q(1, 3), float("inf"), float("nan")]
)
@pytest.mark.parametrize("field", ["nodes", "edges"])
def test_export_rejects_non_json_labels_before_generic_projection(sample, label, field):
    changed = (
        (label, *sample.nodes[1:])
        if field == "nodes"
        else ((label, 1), *sample.edges[1:])
    )
    report = replace(sample, **{field: changed})
    with pytest.raises(TypeError, match="JSON scalar node labels"):
        report.to_dict()


def test_export_preserves_json_scalar_tuple_labels():
    label = ("port", None, True, 1.25)
    graph = nx.relabel_nodes(_state(), {0: label})
    report = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    payload = report.to_dict()
    assert payload["report"]["nodes"][0] == list(label)
    json.dumps(payload, allow_nan=False)


def test_ideal_resultant_kinematics_enclose_independent_chain_rule(sample):
    report = sample.resultant_kinematics()
    assert sample.phase_rate_numerators() == report.phase_rate_numerators
    assert report.comparison is sample
    rows = ((1,), (0, 2), (1,))
    rates = tuple(
        nu * q / (2 * degree * PI)
        for nu, q, degree in zip(sample.capacity, sample.form_gradient, sample.degrees)
    )
    for i, neighbors in enumerate(rows):
        real = -sum(
            (
                _trig(sample.phase[j] - sample.phase[i], sine=True)
                * (rates[j] - rates[i])
                for j in neighbors
            ),
            Q(0),
        )
        imag = sum(
            (
                _trig(sample.phase[j] - sample.phase[i]) * (rates[j] - rates[i])
                for j in neighbors
            ),
            Q(0),
        )
        assert report.resultant_rate_bounds[i][0].contains(real)
        assert report.resultant_rate_bounds[i][1].contains(imag)
        assert report.speed_upper_bounds[i].contains(
            sum((abs(rates[j] - rates[i]) for j in neighbors), Q(0))
        )
        assert report.phase_rate_numerators[i] == rates[i] * PI


@pytest.mark.parametrize("angle", [1.5, 2.0])
def test_reflected_p3_drift_on_both_sides_of_cancellation(angle):
    comparison = owner.bound_relational_sine_exchange(
        _state(
            forms=(-1, 0, 1),
            phases=(-angle, 0, angle),
            capacities=(2, 2, 2),
        ),
        reference_model=MODEL,
    )
    report = comparison.resultant_kinematics()
    assert report.phase_rate_numerators == (-1, 0, 1)
    real, imaginary = report.resultant_rate_bounds[1]
    assert real.contains(-2 * _trig(Q(angle), sine=True) / PI)
    assert real.hi < 0
    assert imaginary.contains(0)
    assert imaginary.width < Q(1, 10**12)
    assert report.speed_upper_bounds[1].contains(2 / PI)
    center_real = comparison.relative_resultant[1][0]
    assert center_real.lo > 0 if angle == 1.5 else center_real.hi < 0


def test_zero_capacity_gives_exact_zero_resultant_drift_and_speed():
    report = owner.bound_relational_sine_exchange(
        _state(capacities=(0, 0, 0)), reference_model=MODEL
    ).resultant_kinematics()
    assert report.phase_rate_numerators == (0, 0, 0)
    assert report.resultant_rate_bounds == ((I(0), I(0)),) * 3
    assert report.speed_upper_bounds == (I(0),) * 3


def test_equal_nonzero_neighbor_rates_cancel_the_common_pi_factor_exactly():
    comparison = owner.bound_relational_sine_exchange(
        _state(forms=(1, 0, -3), capacities=(1, 1, 1)), reference_model=MODEL
    )
    report = comparison.resultant_kinematics()
    assert report.phase_rate_numerators == (Q(1, 2), Q(1, 2), Q(-3, 2))
    assert comparison.phase_rates[0].lo > 0
    # Nodes 0 and 1 rotate together. Retaining their common pi factor makes
    # the leaf's relative phasor derivative exactly zero, not a wide residual.
    assert report.resultant_rate_bounds[0] == (I(0), I(0))
    assert report.speed_upper_bounds[0] == I(0)
    assert report.speed_upper_bounds[1].lo > 0


def test_resultant_kinematics_use_only_the_detached_comparison(monkeypatch):
    graph = _state()
    comparison = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)
    before = pickle.dumps(comparison, protocol=5)
    expected = comparison.resultant_kinematics()

    def forbidden(*args, **kwargs):
        pytest.fail("resultant kinematics must not capture or evaluate another field")

    monkeypatch.setattr(owner, "_capture_sine_state", forbidden)
    monkeypatch.setattr(owner, "bound_relational_sine_exchange", forbidden)
    monkeypatch.setattr(relational, "evaluate_relational_exchange", forbidden)
    graph.clear()
    assert comparison.resultant_kinematics() == expected
    assert pickle.dumps(comparison, protocol=5) == before


def test_exact_phase_row_needs_no_trigonometric_drift_or_graph(monkeypatch):
    graph = _state(forms=(1, 0, -3), capacities=(1, 1, 1))
    comparison = owner.bound_relational_sine_exchange(graph, reference_model=MODEL)

    def forbidden(*args, **kwargs):
        pytest.fail("the rational phase row needs no capture or trigonometric drift")

    monkeypatch.setattr(owner, "_capture_sine_state", forbidden)
    monkeypatch.setattr(owner, "relative_resultant_rate_bounds", forbidden)
    graph.clear()
    # Independent P3 row: w=1/2, beta=1 and degrees=(1,2,1).
    assert comparison.phase_rate_numerators() == (Q(1, 2), Q(1, 2), Q(-3, 2))


def test_resultant_kinematics_export_preserves_ideal_provenance(sample):
    report = sample.resultant_kinematics()
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-resultant-kinematics.v1"
    assert payload["report"]["comparison"]["law"] == sample.law
    assert set(payload["report"]["resultant_rate_bounds"][0][0]) == {"lo", "hi"}
    json.dumps(payload, allow_nan=False)
    scope = " ".join(report.scope)
    assert "ideal_normalized_sine_exchange_rates" in scope
    assert "not_whole_time_rate_or_trajectory_bounds" in scope
    assert "no_graph_reread_native_field" in scope
    invalid_source = replace(sample, nodes=(_OpaqueLabel("node"), *sample.nodes[1:]))
    with pytest.raises(TypeError, match="JSON scalar node labels"):
        replace(report, comparison=invalid_source).to_dict()
