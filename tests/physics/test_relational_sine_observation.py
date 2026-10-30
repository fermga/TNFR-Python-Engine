"""Prior visible-rate inference without observing or supplying a hidden node.

Synthetic prior intervals are generated independently at high precision from
held-out form and phase. They are test observations, not laboratory data or a
reserved prediction. No trajectory, pressure fit or frozen producer is run.
"""

import json
import math
import pickle
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_VF
from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_comparison, relational_sine_mediation
from tnfr.physics.relational_sine_observation import infer_relational_sine_hidden_state

MODEL = RelationalExchangeModel(2, phase_domain="regular")
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(interval, value):
    assert _mp(interval.lo) <= value <= _mp(interval.hi)


def _rate_interval(value):
    # The 90-digit independent evaluation is rounded to exact rational text;
    # a much larger declared test uncertainty is retained on either side.
    center = Q(mp.nstr(value, 85))
    error = Q(1, 10**60)
    return center - error, center + error


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _visible_graph(*, phases=(0, 0.5, -0.75), capacities=(1, 2, 0.5), offset=0):
    graph = nx.Graph()
    graph.add_nodes_from((1, 2, 3, 4))
    graph.add_edges_from(((1, 2), (2, 4)))
    for node, x, theta, nu in zip(
        graph, (1, -0.5, 0.75, -1), (*phases, 1), (*capacities, 1.5)
    ):
        graph.nodes[node].update(EPI=offset + x, theta=theta, nu_f=nu, delta_nfr=99)
    graph.graph.update(_t=99, retained={"history": ["unused"]})
    return graph


def _prior(graph, ports, *, hidden_form=Q(1, 4), hidden_phase=Q(-1, 4), phasor=None):
    """Generate only port rates; no hidden capacity or full-field owner enters."""
    with mp.workdps(90):
        e, w = map(_mp, MODEL.effective_weights)
        beta, hidden = _mp(MODEL.storage_scale), _mp(hidden_form)
        c, s = (
            (mp.cos(_mp(hidden_phase)), mp.sin(_mp(hidden_phase)))
            if phasor is None
            else phasor
        )
        form_rates, phase_rates = {}, {}
        for node in ports:
            data = graph.nodes[node]
            x, theta, nu = map(_mp, (data["EPI"], data["theta"], data["nu_f"]))
            degree = graph.degree[node] + 1
            gradient = (
                degree * x
                - sum(_mp(graph.nodes[j]["EPI"]) for j in graph[node])
                - hidden
            )
            internal = sum(
                mp.sin(_mp(graph.nodes[j]["theta"]) - theta) for j in graph[node]
            )
            current = internal + s * mp.cos(theta) - c * mp.sin(theta)
            form_rates[node] = _rate_interval(
                nu * (-e * gradient + w * current / mp.pi) / degree
            )
            phase_rates[node] = _rate_interval(
                nu * w * gradient / (beta * mp.pi * degree)
            )
        return form_rates, phase_rates


def _infer(graph, ports, form_rates, phase_rates, **changes):
    arguments = dict(
        ports=ports,
        form_rate_bounds=form_rates,
        phase_rate_bounds=phase_rates,
        reference_model=MODEL,
        source_id="independent-synthetic-prior",
        clock_id="declared-structural-clock",
        observation_time=Q(2),
        evidence_window=(Q(2), Q(3)),
        forecast_start=Q(4),
    )
    arguments.update(changes)
    return infer_relational_sine_hidden_state(graph, **arguments)


@pytest.fixture(scope="module")
def sample():
    graph, ports = _visible_graph(), (1, 2, 3)
    rates = _prior(graph, ports)
    return graph, ports, rates, _infer(graph, ports, *rates)


def test_independent_prior_recovers_a_conditional_state_without_hidden_input(sample):
    graph, ports, rates, report = sample
    assert 0 not in graph and 0 not in report.visible_nodes
    assert report.status == "bounded_candidate"
    assert report.phase_rank == 2
    assert report.port_degrees == (2, 3, 1)
    assert report.active_ports == ports
    assert report.hidden_form_bounds.contains(Q(1, 4))
    assert report.hidden_capacity_identified is False
    with mp.workdps(90):
        anchor = _mp(graph.nodes[report.phase_anchor]["theta"])
        hidden_phase = -mp.mpf(1) / 4
        cos_bound, sin_bound = report.hidden_unit_phase_relative_to_anchor_bounds
        _inside(cos_bound, mp.cos(hidden_phase - anchor))
        _inside(sin_bound, mp.sin(hidden_phase - anchor))
        for position, node in enumerate(ports):
            theta = _mp(graph.nodes[node]["theta"])
            _inside(
                report.hidden_projection_raw_bounds[position],
                mp.sin(hidden_phase - theta),
            )
            assert report.hidden_form_by_port[position].contains(Q(1, 4))
            assert report.projection_residuals[position].contains(0)
        assert report.unit_norm_squared_bounds.contains(1)
        phases = [_mp(graph.nodes[node]["theta"]) for node in ports]
        pair_sum = sum(
            mp.sin(right - left) ** 2
            for i, left in enumerate(phases)
            for right in phases[i + 1 :]
        )
        resultant2 = sum(mp.exp(2j * phase) for phase in phases)
        _inside(report.phase_gram_determinant_bounds, pair_sum)
        _inside(
            report.phase_gram_determinant_bounds,
            (len(ports) ** 2 - abs(resultant2) ** 2) / 4,
        )
    # Candidate enclosure and necessary compatibility are not existence,
    # source authentication, hidden capacity identification or a future trace.


def test_phase_inference_cancels_large_common_form_offsets_before_interval_projection(
    sample,
):
    graph, ports, rates, baseline = sample
    shifted = _visible_graph(offset=2**40)
    shifted_rates = _prior(shifted, ports, hidden_form=Q(2**40) + Q(1, 4))
    assert shifted_rates == rates
    result = _infer(shifted, ports, *shifted_rates)
    assert result.status == "bounded_candidate"
    assert result.hidden_form_bounds.contains(Q(2**40) + Q(1, 4))
    assert result.hidden_projection_raw_bounds == baseline.hidden_projection_raw_bounds
    assert result.hidden_unit_phase_raw_bounds == baseline.hidden_unit_phase_raw_bounds


def test_nearly_balanced_ports_are_observable_while_aligned_ports_are_not():
    balanced = _visible_graph(
        phases=(0, float(2 * math.pi / 3), float(-2 * math.pi / 3))
    )
    aligned = _visible_graph(phases=(0, 0, 0))
    reports = [
        _infer(graph, (1, 2, 3), *_prior(graph, (1, 2, 3)))
        for graph in (balanced, aligned)
    ]
    assert reports[0].phase_rank == 2
    assert reports[0].phase_gram_determinant_bounds.lo > 2
    assert reports[0].status == "bounded_candidate"
    assert reports[1].phase_rank == 1
    assert reports[1].phase_gram_determinant_bounds == I(0)
    assert reports[1].status == "unavailable"
    with mp.workdps(90):
        first_resultant = sum(
            mp.exp(1j * _mp(balanced.nodes[node]["theta"])) for node in (1, 2, 3)
        )
        # Represented angles are near-balanced, never asserted exactly 2*pi/3.
        assert 0 < abs(first_resultant) < mp.mpf("1e-14")


def test_identical_phase_ports_have_two_circle_branches_at_zero_rates():
    graph = nx.empty_graph((1, 2))
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    zero = {node: (Q(0), Q(0)) for node in graph}
    report = _infer(graph, (1, 2), zero, zero)
    assert report.phase_rank == 1
    assert report.status == "unavailable"
    assert report.hidden_form_bounds.contains(0)
    # Both (cos,sin)=(1,0) and (-1,0) satisfy the actual rows and unit circle.
    assert report.hidden_projection_bounds == (I(0), I(0))
    assert report.hidden_unit_phase_relative_to_anchor_bounds is None


def test_distinct_but_unresolved_phase_directions_do_not_get_an_assumed_rank():
    graph = nx.empty_graph((1, 2))
    for node, theta in ((1, 0), (2, 1e-30)):
        graph.nodes[node].update(EPI=0, theta=theta, nu_f=1)
    report = _infer(graph, (1, 2), *_prior(graph, (1, 2)))
    assert report.phase_rank is None
    assert report.status == "unavailable"


def test_individually_admissible_projections_can_fail_the_joint_circle_constraint():
    graph = nx.empty_graph((1, 2))
    for node, theta in ((1, 0), (2, 0.5)):
        graph.nodes[node].update(EPI=0, theta=theta, nu_f=1)
    with mp.workdps(90):
        rates = _prior(graph, (1, 2), phasor=(mp.mpf("0.9"), mp.mpf("0.9")))
    report = _infer(graph, (1, 2), *rates)
    assert all(
        bound.subset_of(I(-1, 1)) for bound in report.hidden_projection_raw_bounds
    )
    assert report.status == "inconsistent"
    assert report.unit_norm_squared_bounds.lo > 1


def test_circle_interval_overlap_is_only_a_candidate_not_an_existence_verdict():
    graph = nx.empty_graph((1, 2))
    for node, theta in ((1, 0), (2, 0.5)):
        graph.nodes[node].update(EPI=0, theta=theta, nu_f=1)
    with mp.workdps(90):
        scale = 1 + mp.mpf("1e-25")
        phasor = (scale * mp.cos(mp.mpf("0.25")), scale * mp.sin(mp.mpf("0.25")))
        assert sum(value**2 for value in phasor) > 1 + mp.mpf("1e-25")
        rates = _prior(graph, (1, 2), phasor=phasor)
    report = _infer(graph, (1, 2), *rates)
    # The deliberately off-circle nominal data sit inside arithmetic
    # enclosures too broad to settle the discrepancy. Overlap cannot certify
    # an admissible hidden state or repair the source's exact observation.
    assert report.unit_norm_squared_bounds.contains(1)
    assert report.status == "bounded_candidate"


def test_all_port_form_estimates_must_intersect(sample):
    graph, ports, (form_rates, phase_rates), _ = sample
    changed = dict(phase_rates)
    changed[2] = tuple(value + 1 for value in changed[2])
    report = _infer(graph, ports, form_rates, changed)
    assert report.status == "inconsistent"
    assert report.hidden_form_bounds is None
    assert report.hidden_form_by_port[0].lo > report.hidden_form_by_port[1].hi


def test_a_redundant_port_can_refute_a_candidate_from_the_identifying_pair(sample):
    graph, ports, (form_rates, phase_rates), _ = sample
    changed = dict(form_rates)
    changed[3] = tuple(value + Q(1, 64) for value in changed[3])
    report = _infer(graph, ports, changed, phase_rates)
    assert report.phase_rank == 2
    assert report.status == "inconsistent"
    assert report.hidden_projection_raw_bounds[2].subset_of(I(-1, 1))
    assert any(
        residual is not None and not residual.contains(0)
        for residual in report.projection_residuals
    )


@pytest.mark.parametrize("capacities,expected_rank", (((1, 2, 0), 2), ((0, 0, 0), 0)))
def test_zero_capacity_ports_carry_no_inferential_information(
    capacities, expected_rank
):
    graph = _visible_graph(capacities=capacities)
    report = _infer(graph, (1, 2, 3), *_prior(graph, (1, 2, 3)))
    assert report.phase_rank == expected_rank
    for position, capacity in enumerate(capacities):
        if capacity == 0:
            assert report.hidden_form_by_port[position] is None
            assert report.hidden_projection_raw_bounds[position] is None
            assert report.projection_residuals[position] is None
    assert report.status == (
        "bounded_candidate" if expected_rank == 2 else "unavailable"
    )


def test_nonzero_observed_rate_at_zero_capacity_is_inconsistent():
    graph = _visible_graph(capacities=(1, 2, 0))
    form_rates, phase_rates = _prior(graph, (1, 2, 3))
    form_rates[3] = (Q(1), Q(1))
    assert _infer(graph, (1, 2, 3), form_rates, phase_rates).status == "inconsistent"


def test_visible_prior_does_not_identify_hidden_capacity(sample):
    graph, ports, rates, report = sample
    assert report.hidden_capacity_identified is False
    # Independent hidden laws with these same visible observations may use
    # different capacities and consequently have different subsequent rates.
    with mp.workdps(90):
        contrast = mp.mpf("0.25") - sum(
            _mp(graph.nodes[node]["EPI"]) for node in ports
        ) / len(ports)
        first_hidden_phase_rate = (
            _mp(MODEL.phase_weight) * contrast / (_mp(MODEL.storage_scale) * mp.pi)
        )
        assert first_hidden_phase_rate != 3 * first_hidden_phase_rate
    assert not hasattr(report, "hidden_capacity")


def test_inference_never_captures_a_full_hidden_state_or_uses_a_native_field(
    monkeypatch,
):
    graph, ports = _visible_graph(), (1, 2, 3)
    rates = _prior(graph, ports)
    before = _snapshot(graph)

    def forbidden(*args, **kwargs):
        pytest.fail("prior-visible inference must not obtain a hidden coordinate")

    for owner, names in (
        (
            relational,
            (
                "_stage",
                "_field",
                "evaluate_relational_exchange",
                "step_relational_exchange",
            ),
        ),
        (
            relational_sine_comparison,
            ("_capture_sine_state", "bound_relational_sine_exchange"),
        ),
        (
            relational_sine_mediation,
            ("_capture_sine_state", "bound_relational_sine_mediation"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    report = _infer(graph, ports, *rates)
    assert report.status == "bounded_candidate"
    assert _snapshot(graph) == before
    graph.nodes[1]["EPI"] = float("nan")
    assert report.visible_epi[0] == 1


@pytest.mark.parametrize(
    "window,observation", (((Q(2), Q(3)), Q(2)), ((Q(2), Q(2)), Q(2)))
)
def test_prior_window_keeps_the_observation_time_explicit(sample, window, observation):
    graph, ports, rates, _ = sample
    report = _infer(
        graph, ports, *rates, evidence_window=window, observation_time=observation
    )
    assert report.observation_time == observation
    assert report.evidence_window == window
    assert report.evidence_window[1] < report.forecast_start
    assert report.observation_time != report.forecast_start


@pytest.mark.parametrize(
    "changes",
    (
        {"evidence_window": (Q(-1), Q(3))},
        {"observation_time": Q(1)},
        {"forecast_start": Q(3)},
        {"evidence_window": (Q(3), Q(2))},
    ),
)
def test_invalid_or_overlapping_evidence_window_is_rejected(sample, changes):
    graph, ports, rates, _ = sample
    with pytest.raises(ADMISSION_ERRORS):
        _infer(graph, ports, *rates, **changes)


@pytest.mark.parametrize(
    "fault", ("negative_capacity", "missing_rate", "nonfinite_rate")
)
def test_invalid_authoritative_state_or_rate_evidence_is_rejected(fault):
    graph, ports = _visible_graph(), (1, 2, 3)
    form_rates, phase_rates = _prior(graph, ports)
    if fault == "negative_capacity":
        graph.nodes[1]["nu_f"] = -1
    elif fault == "missing_rate":
        del phase_rates[2]
    else:
        form_rates[1] = (0, float("nan"))
    with pytest.raises(ADMISSION_ERRORS):
        _infer(graph, ports, form_rates, phase_rates)


def test_export_keeps_prior_source_clock_and_candidate_scope(sample):
    report = sample[3]
    encoded = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    body = encoded["report"]
    assert body["source_id"] == "independent-synthetic-prior"
    assert body["clock_id"] == "declared-structural-clock"
    assert body["status"] == "bounded_candidate"
    assert body["hidden_capacity_identified"] is False
    assert body["visible_nodes"] == [1, 2, 3, 4]


def test_complete_graph_consumers_still_reject_disconnected_support():
    graph = _visible_graph()
    with pytest.raises(ValueError, match="connected support"):
        relational_sine_comparison.bound_relational_sine_exchange(
            graph, reference_model=MODEL
        )
    with pytest.raises(ValueError, match="connected support"):
        relational.evaluate_relational_exchange(graph, model=MODEL)
    with pytest.raises(ValueError, match="connected support"):
        relational.step_relational_exchange(graph, model=MODEL, dt=0.125)


@pytest.mark.parametrize(
    "fault",
    ("unattached_component", "nonunit_edge", "known_gamma", "authoritative_alias"),
)
def test_visible_only_admission_preserves_support_forcing_and_alias_contracts(fault):
    graph, ports = _visible_graph(), (1, 2, 3)
    rates = _prior(graph, ports)
    if fault == "unattached_component":
        graph.add_node(5, EPI=0, theta=0, nu_f=1)
    elif fault == "nonunit_edge":
        graph.edges[1, 2]["weight"] = 2
    elif fault == "known_gamma":
        graph.graph["GAMMA"] = {"type": "constant", "value": 1}
    else:
        graph.nodes[1][ALIAS_VF[0]] = float("nan")
        graph.nodes[1][ALIAS_VF[1]] = 1
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        _infer(graph, ports, *rates)
    assert _snapshot(graph) == before


def test_visible_only_admission_rejects_a_replaced_none_gamma_registry_entry(
    monkeypatch,
):
    graph, ports = _visible_graph(), (1, 2, 3)
    rates = _prior(graph, ports)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "visible inference must reject, not execute, an altered Gamma source"
        )

    monkeypatch.setitem(GAMMA_REGISTRY, "none", GammaEntry(forbidden, False))
    with pytest.raises(ValueError, match="Gamma"):
        _infer(graph, ports, *rates)
