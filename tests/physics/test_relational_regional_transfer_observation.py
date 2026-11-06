"""Full-source boundary-current and supplied-increment observation controls."""

import json
import pickle
from copy import copy
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import relational_sine_comparison as owner
from tnfr.physics.relational_sine_scale import assess_sine_pair_emission
from tnfr.sdk import export_to_json, relational_report_to_dict

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _graph(graph=None, *, form=None, phase=None, capacity=None):
    graph = nx.complete_bipartite_graph(2, 2) if graph is None else graph
    count = len(graph)
    form = (0,) * count if form is None else form
    phase = (0, 0, Q(1, 4), Q(1, 4)) if phase is None else phase
    capacity = (1,) * count if capacity is None else capacity
    for node, x, theta, nu in zip(graph, form, phase, capacity):
        graph.nodes[node].update(EPI=x, theta=theta, nu_f=nu, delta_nfr="not used")
    graph.graph.update(GAMMA={"type": "none"}, EPI_MIN=-1, EPI_MAX=1)
    return graph


def _capture(graph=None, model=MODEL):
    return owner.bound_relational_sine_exchange(
        _graph() if graph is None else graph, reference_model=model
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    if bound.lo == bound.hi:
        assert abs(value - _mp(bound.lo)) < mp.mpf("1e-85")
    else:
        assert _mp(bound.lo) <= value <= _mp(bound.hi)


def _fine_rows(comparison):
    """Independent arbitrary-degree complete rows using the retained support."""
    neighbors = [[] for _ in comparison.nodes]
    indices = {node: i for i, node in enumerate(comparison.nodes)}
    for a, b in comparison.edges:
        i, j = indices[a], indices[b]
        neighbors[i].append(j)
        neighbors[j].append(i)
    e, w = map(_mp, comparison.reference_model.effective_weights)
    beta = _mp(comparison.reference_model.storage_scale)
    x, theta, nu = (
        list(map(_mp, values))
        for values in (comparison.epi, comparison.phase, comparison.capacity)
    )
    xdot, phasedot = [], []
    for i, row in enumerate(neighbors):
        q = sum(x[i] - x[j] for j in row)
        current = sum(mp.sin(theta[j] - theta[i]) for j in row)
        xdot.append(nu[i] * (-e * q + w * current / mp.pi) / len(row))
        phasedot.append(w * nu[i] * q / (beta * len(row) * mp.pi))
    return neighbors, xdot, phasedot


@pytest.mark.parametrize("delta", [Q(1, 4), -Q(1, 4), Q(0)])
def test_synchronized_regions_exchange_form_and_induce_phase_response(delta):
    graph = _graph(phase=(0, 0, delta, delta))
    before = _snapshot(graph)
    comparison = _capture(graph)
    left = comparison.regional_transfer(region=(0, 1))
    right = comparison.regional_transfer(region=(2, 3))
    assert left.boundary_edge_indices == ((0, 2), (0, 3), (1, 2), (1, 3))
    assert left.form_weights == (Q(2),) * 4
    assert left.regional_weighted_form == left.complement_weighted_form == 0
    assert left.global_weighted_form_conserved
    with mp.workdps(100):
        neighbors, xdot, phasedot = _fine_rows(comparison)
        expected = 4 * mp.sin(_mp(delta)) / mp.pi
        _contains(left.regional_form_rate_bounds, expected)
        _contains(left.complement_form_rate_bounds, -expected)
        _contains(right.regional_form_rate_bounds, -expected)
        for i in range(4):
            _contains(comparison.form_rates[i], xdot[i])
            _contains(comparison.phase_rates[i], phasedot[i])
        # The phase row is linear in all forms. Its derivative retains all
        # four live rows, rather than appending an independent oscillator.
        accelerations = tuple(
            sum(
                (
                    comparison.form_rates[i] - comparison.form_rates[j]
                    for j in neighbors[i]
                ),
                I(0),
            )
            / (len(neighbors[i]) * pi_interval())
            for i in range(4)
        )
        expected_acceleration = 2 * mp.sin(_mp(delta)) / mp.pi**2
        _contains(accelerations[0], expected_acceleration)
        _contains(accelerations[2], -expected_acceleration)
        assert phasedot == [mp.mpf(0)] * 4
    if delta:
        assert (
            left.regional_form_rate_bounds.lo > 0
            if delta > 0
            else left.regional_form_rate_bounds.hi < 0
        )
    else:
        assert all(bound.lo == bound.hi == 0 for bound in comparison.form_rates)
    assert _snapshot(graph) == before


def test_cut_balance_keeps_actual_degrees_capacities_and_positive_form_loss():
    model = RelationalExchangeModel(
        2, epi_weight=1, phase_weight=3, phase_domain="regular"
    )
    graph = _graph(
        nx.path_graph(3),
        form=(Q(1, 4), -Q(1, 2), Q(3, 4)),
        phase=(-Q(1, 4), Q(1, 2), -Q(1, 4)),
        capacity=(Q(1, 2), 2, 3),
    )
    comparison = _capture(graph, model)
    report = comparison.regional_transfer(region=(1, 0))
    assert report.region == (1, 0)
    assert report.region_indices == (1, 0)
    assert report.complement_indices == (2,)
    assert report.boundary_edge_indices == ((1, 2),)
    assert report.form_weights == (Q(2), Q(1), Q(1, 3))
    assert report.regional_weighted_form == 0
    assert report.complement_weighted_form == Q(1, 4)
    assert comparison.continuous_loss > 0
    assert report.global_weighted_form_conserved
    with mp.workdps(100):
        _, xdot, _ = _fine_rows(comparison)
        e, w = map(_mp, model.effective_weights)
        diffusive = e * mp.mpf(5) / 4
        sinusoidal = -w * mp.sin(mp.mpf(3) / 4) / mp.pi
        assert report.boundary_diffusive_currents[0] == Q(5, 16)
        _contains(report.boundary_sine_currents[0], sinusoidal)
        _contains(report.boundary_form_currents[0], diffusive + sinusoidal)
        _contains(report.direct_regional_form_rate_bounds, 2 * xdot[0] + xdot[1])
        _contains(report.direct_complement_form_rate_bounds, xdot[2] / 3)
        assert abs(sum(xdot)) > mp.mpf("0.01")
        assert abs(2 * xdot[0] + xdot[1] + xdot[2] / 3) < mp.mpf("1e-90")
    for residual in (
        report.regional_balance_residual_bounds,
        report.complement_balance_residual_bounds,
        report.global_weighted_form_rate_bounds,
    ):
        assert residual.lo <= 0 <= residual.hi
    whole = comparison.regional_transfer(region=comparison.nodes)
    assert whole.boundary_edge_indices == ()
    assert whole.complement_indices == ()
    assert whole.regional_form_rate_bounds.lo == whole.regional_form_rate_bounds.hi == 0


def test_readers_reuse_one_capture_after_graph_changes_without_reinterpreting_law(
    monkeypatch,
):
    graph = _graph()
    comparison = _capture(graph)
    graph.nodes[0]["EPI"] = 1000
    graph.remove_edge(0, 2)
    changed = _snapshot(graph)

    def forbidden(*args, **kwargs):
        pytest.fail("a captured-source reader must not recapture or run a trajectory")

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", forbidden)
    monkeypatch.setattr(owner, "_capture_sine_state", forbidden)
    report = comparison.regional_transfer(region=(0, 1))
    endpoint = comparison.assess_form_increment(increments=(Q(1, 8),) * 4)
    assert report.comparison is endpoint.comparison is comparison
    assert len(report.boundary_edge_indices) == 4
    assert report.regional_weighted_form == 0
    assert endpoint.weighted_form_before == 0
    assert _snapshot(graph) == changed


def test_regional_transfer_rebuilds_gradient_currents_and_rates_from_primitives(
    monkeypatch,
):
    model = RelationalExchangeModel(2, phase_domain="regular")
    source = _capture(_graph(nx.path_graph(3), phase=(0, 0, 0)), model)
    changed = replace(
        source,
        epi=(0.25, -0.5, 0.75),
        phase=(0, 0.5, -0.25),
        capacity=(Q(1, 2), 2, 3),
        form_gradient=(Q(99),) * 3,
        relative_resultant=((I(99), I(99)),) * 3,
        form_rates=(I(99),) * 3,
        phase_rates=(I(99),) * 3,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("regional accounting must rebuild from detached primitives")

    monkeypatch.setattr(owner, "_capture_sine_state", forbidden)
    monkeypatch.setattr(owner, "_stage", forbidden)
    report = changed.regional_transfer(region=(1, 0))
    assert report.comparison is not changed
    assert report.comparison.epi == (Q(1, 4), -Q(1, 2), Q(3, 4))
    assert report.comparison.form_rates != changed.form_rates
    assert report.form_weights == (Q(2), Q(1), Q(1, 3))
    assert report.regional_weighted_form == 0
    assert report.complement_weighted_form == Q(1, 4)
    with mp.workdps(100):
        _, rates, _ = _fine_rows(changed)
        expected = 2 * rates[0] + rates[1]
        _contains(report.direct_regional_form_rate_bounds, expected)
        _contains(report.regional_form_rate_bounds, expected)
        _contains(report.direct_complement_form_rate_bounds, rates[2] / 3)
    assert report.regional_balance_residual_bounds.contains(0)
    assert report.complement_balance_residual_bounds.contains(0)
    assert report.global_weighted_form_rate_bounds.contains(0)
    assert report.global_weighted_form_conserved


@pytest.mark.parametrize("reader", ("regional", "increment"))
@pytest.mark.parametrize(
    "changes",
    (
        {"degrees": (100, 2, 2, 2)},
        {"degrees": (True, 2, 2, 2)},
        {"edges": ((0, 2), (0, 2), (1, 2), (1, 3))},
        {"law": "current_squared_reciprocal_mobility"},
        {"epi": (True, 0, 0, 0)},
        {"phase": (0, float("inf"), 0, 0)},
        {"capacity": (True, 1, 1, 1)},
        {"capacity": (-1, 1, 1, 1)},
        {"capacity": (0, 1, 1, 1)},
    ),
)
def test_form_accounting_rejects_unsupported_primitive_source_before_field(
    reader, changes, monkeypatch
):
    changed = replace(_capture(), **changes)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported source must reject before field arithmetic")

    monkeypatch.setattr(owner, "_sine_state_from_rows", forbidden)
    if reader == "regional":
        with pytest.raises(ADMISSION_ERRORS):
            changed.regional_transfer(region=(0, 1))
    else:
        with pytest.raises(ADMISSION_ERRORS):
            changed.assess_form_increment(increments=(1, -1, 0, 0))


@pytest.mark.parametrize("field,value", (("phase_weight", True), ("epi_weight", -1)))
def test_form_accounting_readmits_authoritative_model(field, value):
    source = _capture()
    model = copy(source.reference_model)
    object.__setattr__(model, field, value)
    changed = replace(source, reference_model=model)
    with pytest.raises(ADMISSION_ERRORS):
        changed.regional_transfer(region=(0, 1))
    with pytest.raises(ADMISSION_ERRORS):
        changed.assess_form_increment(increments=(1, -1, 0, 0))


def test_supplied_signed_increment_rebuilds_source_without_deriving_the_increment(
    monkeypatch,
):
    source = _capture()
    changed = replace(
        source,
        epi=(-1.0, 0, 0, 0),
        capacity=(1, 2, 3, 4),
        form_rates=(I(99),) * 4,
        form_gradient=(Q(99),) * 4,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("a supplied endpoint increment must not recapture the source")

    monkeypatch.setattr(owner, "_capture_sine_state", forbidden)
    monkeypatch.setattr(owner, "_stage", forbidden)
    balanced = changed.assess_form_increment(increments=iter((1, -2, 0, 0)))
    assert balanced.comparison is not changed
    assert balanced.comparison.form_rates != changed.form_rates
    assert balanced.increments == (Q(1), -Q(2), Q(0), Q(0))
    assert balanced.weighted_form_before == balanced.weighted_form_after == -2
    assert balanced.weighted_form_change == 0
    assert not balanced.closed_flow_endpoint_obstructed
    assert not balanced.endpoint_reachability_certified
    negative = changed.assess_form_increment(increments=(-1, 0, 0, 0))
    assert negative.weighted_form_change == -2
    assert negative.closed_flow_endpoint_obstructed


def test_full_endpoint_invariant_obstruction_is_not_a_regional_or_energy_test():
    comparison = _capture()
    shifted = comparison.assess_form_increment(increments=(Q(1, 8),) * 4)
    assert shifted.increments == (Q(1, 8),) * 4
    assert shifted.weighted_form_before == 0
    assert shifted.weighted_form_after == shifted.weighted_form_change == 1
    assert shifted.closed_flow_endpoint_obstructed
    assert not shifted.endpoint_reachability_certified
    # A global form translation leaves every form gap and the phase storage
    # unchanged. Energy equality still cannot overcome the conserved total.
    assert all(
        shifted.increments[i] - shifted.increments[j] == 0
        for i, j in ((0, 2), (0, 3), (1, 2), (1, 3))
    )
    balanced = comparison.assess_form_increment(
        increments=(Q(1, 8), Q(1, 8), -Q(1, 8), -Q(1, 8))
    )
    assert balanced.weighted_form_change == 0
    assert not balanced.closed_flow_endpoint_obstructed
    assert not balanced.endpoint_reachability_certified
    # Regional gain alone is consistent with compensating environmental loss;
    # the tested full endpoint is still not claimed dynamically reachable.
    assert sum(balanced.form_weights[i] * balanced.increments[i] for i in (0, 1)) > 0
    decreasing = comparison.assess_form_increment(increments=(-Q(1, 8),) * 4)
    assert decreasing.weighted_form_change == -1
    assert decreasing.closed_flow_endpoint_obstructed


def test_unequal_and_tiny_positive_capacity_weights_are_exact():
    comparison = _capture(_graph(capacity=(Q(1, 2**200), 2, 3, 4)))
    report = comparison.regional_transfer(region=(0,))
    assert report.form_weights == (Q(2**201), Q(1), Q(2, 3), Q(1, 2))
    change = comparison.assess_form_increment(increments=(Q(1, 2**200), 0, 0, 0))
    assert change.weighted_form_change == 2
    assert change.closed_flow_endpoint_obstructed


def test_al_adapter_uses_actual_clipped_and_rounded_full_increments():
    graph = _graph(form=(Q(7, 8), Q(7, 8), 0, 0))
    emission = assess_sine_pair_emission(
        graph,
        reference_model=MODEL,
        pairs=((0, 1), (2, 3)),
        pair_index=0,
        boost=Q(1, 2),
    )
    whole = emission.form_increment(outcome="whole_pair")
    singleton = emission.form_increment(outcome="first_member")
    assert whole.comparison is emission.comparison
    assert whole.increments == (Q(1, 8), Q(1, 8), 0, 0)
    assert whole.weighted_form_change == Q(1, 2)
    assert singleton.weighted_form_change == Q(1, 4)
    assert whole.closed_flow_endpoint_obstructed
    assert not whole.endpoint_reachability_certified
    for forms, boost in (
        ((1, 1, 0, 0), Q(1, 8)),
        ((Q(1, 4), Q(1, 4), 0, 0), Q(1, 2**200)),
    ):
        noop = assess_sine_pair_emission(
            _graph(form=forms),
            reference_model=MODEL,
            pairs=((0, 1), (2, 3)),
            pair_index=0,
            boost=boost,
        ).form_increment(outcome="whole_pair")
        assert noop.increments == (0, 0, 0, 0)
        assert noop.weighted_form_change == 0
        assert not noop.closed_flow_endpoint_obstructed
        assert not noop.endpoint_reachability_certified
    with pytest.raises(ADMISSION_ERRORS):
        emission.form_increment(outcome="unrecognized")


@pytest.mark.parametrize("region", [(), (0, 0), (4,), (0, 1, 2, 3, 4), {0, 1}, "01"])
def test_invalid_region_is_not_repaired(region):
    with pytest.raises(ADMISSION_ERRORS):
        _capture().regional_transfer(region=region)


@pytest.mark.parametrize(
    "increments",
    [
        (),
        (1, 0),
        (0,) * 5,
        (True, 0, 0, 0),
        (float("nan"), 0, 0, 0),
        (float("inf"), 0, 0, 0),
        {0: 1},
    ],
)
def test_full_increment_vector_requires_all_independent_admitted_coordinates(
    increments,
):
    with pytest.raises(ADMISSION_ERRORS):
        _capture().assess_form_increment(increments=increments)


def test_zero_capacity_and_live_forcing_do_not_inherit_the_divided_balance():
    comparison = _capture(_graph(capacity=(0, 1, 1, 1)))
    with pytest.raises(ADMISSION_ERRORS):
        comparison.regional_transfer(region=(1, 2, 3))
    with pytest.raises(ADMISSION_ERRORS):
        comparison.assess_form_increment(increments=(0, 1, 0, 0))
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "constant", "value": 1}
    with pytest.raises(ADMISSION_ERRORS):
        _capture(graph)


def test_exact_increment_domain_and_export_preserve_provenance(tmp_path):
    comparison = _capture()
    exact = Q(1, 2**2000)
    endpoint = comparison.assess_form_increment(increments=(exact, 0, 0, 0))
    assert endpoint.increments[0] == exact
    assert endpoint.weighted_form_change == 2 * exact
    assert endpoint.closed_flow_endpoint_obstructed
    transfer = comparison.regional_transfer(region=(0, 1))
    for report, schema, name in (
        (transfer, "tnfr.relational-sine-regional-transfer.v1", "transfer"),
        (endpoint, "tnfr.relational-sine-form-increment.v1", "increment"),
    ):
        specific = report.to_dict()
        generic = relational_report_to_dict(report)
        assert specific["schema"] == schema
        assert generic["report"] == specific["report"]
        assert generic["report_type"] == type(report).__name__
        output = tmp_path / f"{name}.json"
        export_to_json(report, output)
        assert json.loads(output.read_text(encoding="utf-8")) == specific

    @dataclass(frozen=True)
    class Opaque:
        label: str

    with pytest.raises(ADMISSION_ERRORS):
        replace(transfer, region=(Opaque("bad"), 1)).to_dict()
