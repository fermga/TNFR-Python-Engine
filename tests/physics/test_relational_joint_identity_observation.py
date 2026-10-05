"""Joint observation and finite full-state identity-window admission controls."""

import json
import pickle
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.physics import relational_sine_scale as owner
from tnfr.sdk import export_to_json, relational_report_to_dict

PAIRS = ((0, 1), (2, 3))
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
RHO = Q(1, 10**6)
HORIZON = Q(10)
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _graph(*, phases=(0, 0, Q(1, 4), Q(1, 4))):
    graph = nx.complete_bipartite_graph(2, 2)
    for node, phase in zip(graph, phases):
        graph.nodes[node].update(EPI=Q(1, 2), theta=phase, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _assess(graph=None, **changes):
    arguments = dict(
        reference_model=MODEL,
        pairs=PAIRS,
        form_error_bounds=(RHO,) * 4,
        phase_error_bounds=(RHO,) * 4,
        window_end=HORIZON,
    )
    arguments.update(changes)
    return owner.assess_sine_joint_pairing_window(
        _graph() if graph is None else graph, **arguments
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    if bound.lo == bound.hi:
        assert abs(value - _mp(bound.lo)) < mp.mpf("1e-85")
    else:
        assert _mp(bound.lo) <= value <= _mp(bound.hi)


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def test_joint_observer_distinguishes_static_phase_collision_without_orbit_claim():
    # This exact represented static state illustrates phase-only information
    # loss. It is not asserted to be a turning point of the frozen reference.
    forms = (Q(5, 8), Q(5, 8), Q(3, 8), Q(3, 8))
    phases = (0,) * 4
    phase_only = owner.observe_phase_pairs(nodes=range(4), phases=phases)
    joint = owner.observe_joint_pairs(
        nodes=range(4), forms=forms, phases=phases, storage_scale=1
    )
    assert phase_only.candidate_pairs is None
    assert joint.candidate_pairs == PAIRS
    assert joint.nearest_partner_indices == (1, 0, 3, 2)
    for (i, j), bound in zip(
        joint.distance_indices, joint.squared_joint_distance_bounds
    ):
        expected = (forms[i] - forms[j]) ** 2
        assert bound.lo == bound.hi == expected
    assert all(margin > 0 for margin in joint.nearest_separation_margins)


def test_joint_metric_is_circular_and_keeps_law_scale_without_phase_unwrapping():
    nodes = ("a", "b", "c", "d")
    forms = (Q(1, 4), -Q(1, 4), Q(3, 4), Q(1, 2))
    phases = (Q(3), -Q(3), Q(1, 8), -Q(1, 8))
    beta = Q(3, 2)
    report = owner.observe_joint_pairs(
        nodes=nodes, forms=forms, phases=phases, storage_scale=beta
    )
    with mp.workdps(100):
        for (i, j), bound in zip(
            report.distance_indices, report.squared_joint_distance_bounds
        ):
            expected = _mp(forms[i] - forms[j]) ** 2 + 2 * _mp(beta) * (
                1 - mp.cos(_mp(phases[i] - phases[j]))
            )
            _contains(bound, expected)
    shifted = owner.observe_joint_pairs(
        nodes=nodes,
        forms=tuple(x + 10**50 for x in forms),
        phases=tuple(theta + 10**50 for theta in phases),
        storage_scale=beta,
    )
    assert shifted.squared_joint_distance_bounds == report.squared_joint_distance_bounds
    assert shifted.nearest_partner_indices == report.nearest_partner_indices
    assert report.forms == forms and report.phases == phases


def test_frozen_full_box_preserves_joint_identity_through_a_reference_exchange():
    graph = _graph()
    before = _snapshot(graph)
    report = _assess(graph)
    assert report.window == (Q(0), HORIZON)
    assert report.form_error_bounds == report.phase_error_bounds == (RHO,) * 4
    assert report.initial_scaled_error_squared == 8 * RHO**2
    assert report.reference_identity_certified
    assert report.whole_window_identity_certified
    assert report.candidate_pairs == PAIRS
    assert report.nearest_partner_indices == (1, 0, 3, 2)
    assert report.identification_budget_margin > 0
    assert report.window_covers_reference_period
    assert report.reference_period_bounds.hi < HORIZON
    with mp.workdps(100):
        separation = 2 * (1 - mp.cos(mp.mpf(1) / 4))
        lipschitz = 2 / mp.pi
        growth = mp.exp(2 * lipschitz * 10)
        propagated = 8 * _mp(RHO) ** 2 * growth
        _contains(report.reference_joint_separation_squared_bounds, separation)
        assert _mp(report.lipschitz_upper) >= lipschitz
        assert _mp(report.growth_exponent_upper) >= 2 * lipschitz * 10
        assert _mp(report.squared_growth_factor_upper) >= growth
        assert _mp(report.propagated_scaled_error_squared_upper) >= propagated
        assert _mp(report.identification_budget_margin) <= separation - 8 * propagated
        period = 2 * mp.pi * mp.ellipk(mp.sin(mp.mpf(1) / 8) ** 2)
        _contains(report.reference_period_bounds, period)
    assert _snapshot(graph) == before


def test_wide_box_contains_an_actual_stationary_tie_and_does_not_certify_identity():
    wide = _assess(form_error_bounds=(Q(1, 8),) * 4, phase_error_bounds=(Q(1, 8),) * 4)
    assert wide.reference_identity_certified
    assert not wide.whole_window_identity_certified
    assert wide.candidate_pairs is None
    assert wide.identification_budget_margin < 0
    # The independent box genuinely includes this exact full equilibrium.
    # It is a counterexample to universal strict pairing, not a sampled path.
    tied_phases = (Q(1, 8),) * 4
    assert all(
        abs(theta - captured) <= Q(1, 8)
        for theta, captured in zip(tied_phases, wide.comparison.phase)
    )
    tied = owner.observe_joint_pairs(
        nodes=range(4), forms=(Q(1, 2),) * 4, phases=tied_phases, storage_scale=1
    )
    assert tied.candidate_pairs is None
    assert all(
        bound.lo == bound.hi == 0 for bound in tied.squared_joint_distance_bounds
    )


def test_zero_amplitude_and_numerically_unresolved_nonzero_reference_are_separate():
    zero = _assess(_graph(phases=(0, 0, 0, 0)))
    assert not zero.reference_identity_certified
    assert not zero.whole_window_identity_certified
    assert zero.candidate_pairs is None
    assert zero.reference_period_bounds is None
    tiny = _assess(
        _graph(phases=(0, 0, Q(1, 2**200), Q(1, 2**200))),
        form_error_bounds=(0,) * 4,
        phase_error_bounds=(0,) * 4,
    )
    assert tiny.reference_identity_certified
    assert tiny.reference_joint_separation_squared_bounds.lo <= 0
    assert not tiny.whole_window_identity_certified
    assert tiny.candidate_pairs is None


def test_zero_error_keeps_exact_reference_separate_from_overflowing_growth_budget():
    horizon = Q(10**6)
    exact = _assess(
        form_error_bounds=(0,) * 4, phase_error_bounds=(0,) * 4, window_end=horizon
    )
    assert exact.reference_identity_certified
    assert exact.whole_window_identity_certified
    assert exact.propagated_scaled_error_squared_upper == 0
    assert exact.candidate_pairs == PAIRS
    uncertain = _assess(window_end=horizon)
    assert uncertain.reference_identity_certified
    assert not uncertain.whole_window_identity_certified
    assert uncertain.squared_growth_factor_upper is None
    assert uncertain.candidate_pairs is None


def test_reference_admission_rejects_averaging_away_internal_or_form_differences():
    for attr in ("EPI", "theta", "nu_f"):
        graph = _graph()
        graph.nodes[0][attr] += Q(1, 8)
        with pytest.raises(ADMISSION_ERRORS):
            _assess(graph)
    unequal_means = _graph()
    unequal_means.nodes[0]["EPI"] = unequal_means.nodes[1]["EPI"] = Q(5, 8)
    with pytest.raises(ADMISSION_ERRORS):
        _assess(unequal_means)
    graph = _graph()
    graph.add_edge(0, 1)
    with pytest.raises(ADMISSION_ERRORS):
        _assess(graph)
    with pytest.raises(ADMISSION_ERRORS):
        _assess(reference_model=RelationalExchangeModel(1, phase_domain="regular"))


def test_supplied_common_lift_turns_relabeling_and_single_capture_are_explicit(
    monkeypatch,
):
    graph = _graph()
    original = owner.bound_relational_sine_exchange
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", counted)
    report = _assess(graph, phase_turns=(10**40,) * 4)
    assert calls == [graph]
    assert report.reference_phase_gap_turn_part == 0
    assert report.reference_identity_certified
    assert report.whole_window_identity_certified
    renamed = nx.relabel_nodes(graph, {0: "west1", 1: "west0", 2: "east1", 3: "east0"})
    pairs = (("east0", "east1"), ("west0", "west1"))
    other = _assess(renamed, pairs=pairs)
    assert {frozenset(pair) for pair in other.candidate_pairs} == {
        frozenset(pair) for pair in pairs
    }
    assert (
        other.reference_joint_separation_squared_bounds
        == report.reference_joint_separation_squared_bounds
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"form_error_bounds": (RHO,) * 3},
        {"phase_error_bounds": (RHO,) * 5},
        {"form_error_bounds": (True, 0, 0, 0)},
        {"phase_error_bounds": (-1, 0, 0, 0)},
        {"form_error_bounds": (float("nan"), 0, 0, 0)},
        {"window_end": -1},
        {"window_end": True},
        {"window_end": float("inf")},
        {"phase_turns": (False, 0, 0, 0)},
    ],
)
def test_invalid_full_uncertainty_or_clock_is_not_silently_repaired(changes):
    graph = _graph()
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        _assess(graph, **changes)
    assert _snapshot(graph) == before


def test_exports_and_observation_inputs_do_not_authenticate_independent_coordinates(
    tmp_path,
):
    report = _assess()
    observed = owner.observe_joint_pairs(
        nodes=report.comparison.nodes,
        forms=report.comparison.epi,
        phases=report.comparison.phase,
        storage_scale=1,
    )
    for value, schema, name in (
        (report, "tnfr.relational-sine-joint-pairing-window.v1", "window"),
        (observed, "tnfr.joint-pairs.v1", "joint"),
    ):
        specific = value.to_dict()
        assert specific["schema"] == schema
        generic = relational_report_to_dict(value)
        assert generic["report"] == specific["report"]
        output = tmp_path / f"{name}.json"
        export_to_json(value, output)
        assert json.loads(output.read_text(encoding="utf-8")) == specific
    for beta in (0, -1, True, float("nan")):
        with pytest.raises(ADMISSION_ERRORS):
            owner.observe_joint_pairs(
                nodes=(0, 1), forms=(0, 1), phases=(0, 0), storage_scale=beta
            )
    with pytest.raises(ADMISSION_ERRORS):
        owner.observe_joint_pairs(
            nodes=(0, 1), forms=(0,), phases=(0, 0), storage_scale=1
        )

    @dataclass(frozen=True)
    class Opaque:
        value: str

    with pytest.raises(ADMISSION_ERRORS):
        replace(observed, candidate_pairs=((Opaque("bad"), 1), (2, 3))).to_dict()
