"""Runtime endpoints of a scoped Coupling policy, with canonical controls."""

from fractions import Fraction
import math

import pytest

from benchmarks.canonical_winding_persistence import (
    build_ring, phase_gaps, run_coupling_case, run_transition_counterexample,
)
from tnfr.constants.aliases import ALIAS_THETA
from tnfr.alias import get_attr
from tnfr.operators.definitions import Coupling, Silence
from tnfr.physics.coupling_winding import observe_coupling_gap_step
from tnfr.physics.winding_certificates import (
    certify_phase_winding, observe_winding_word,
)


@pytest.mark.parametrize("count", [8, 16])
@pytest.mark.parametrize("winding", [0, 1])
def test_all_target_policy_preserves_finite_winding_and_reduces_gap_spread(
    count, winding
):
    result = run_coupling_case(count, winding)
    assert result["endpoint_winding_preserved"]
    assert result["observed_endpoint_lifetime_stages"] == 16
    assert all(
        history == ("UM", "SHA") * 8
        for history in result["actual_histories"]
    )
    rows = result["observations"]
    previous_capacity = 1.0
    for row in rows:
        certificate = row["certificate"]
        assert certificate["winding"] == winding
        assert certificate["minimum_branch_margin"] > 1.0
        assert certificate["minimum_u3_margin"] > 0.0
        assert row["edge_support"] == result["initial_edge_support"]
        assert row["epi"] == result["initial_epi"]
        if row["operator"] == "coupling":
            exact = row["exact_gap_model"]
            assert exact["sum_residual"] == 0
            assert exact["spread_drop"] == exact["expected_spread_drop"] > 0
            assert exact["invariant_interval_preserved"]
            # An observed discrepancy budget, not a uniform binary64 theorem.
            assert row["max_runtime_gap_residual"] < Fraction(1, 10**13)
            assert not row["phase_endpoint_unchanged"]
            assert row["capacity"] == (previous_capacity,) * count
        else:
            assert row["operator"] == "silence"
            assert row["phase_endpoint_unchanged"]
            assert 0.0 < row["capacity"][0] < previous_capacity
            assert row["capacity"] == (row["capacity"][0],) * count
        previous_capacity = row["capacity"][0]
    assert rows[-1]["exact_observed_gap_spread"] < result["initial_gap_spread"]
    if winding == 1:
        initial = result["initial_positive_circulation_concentration"]
        final = rows[-1]["positive_circulation_concentration"]
        assert Fraction(1, count) < final < initial
        gaps = tuple(Fraction.from_float(value) for value in rows[-1]["gaps"])
        total = sum(gaps, Fraction(0))
        spread = rows[-1]["exact_observed_gap_spread"]
        assert final == Fraction(1, count) + 2 * spread / total**2
    else:
        assert all(row["positive_circulation_concentration"] is None for row in rows)


@pytest.mark.parametrize("winding", [0, 1])
def test_single_target_actual_word_matches_the_distinct_local_gap_map(winding):
    graph = build_ring(8, winding)
    phases = tuple(get_attr(graph.nodes[node], ALIAS_THETA, None) for node in graph)
    reference = observe_coupling_gap_step(phase_gaps(phases), target=1)
    observed = observe_winding_word(
        graph, range(8), 1, [Coupling(), Silence()]
    )
    after = tuple(get_attr(graph.nodes[node], ALIAS_THETA, None) for node in graph)
    gaps = phase_gaps(after)
    assert observed.history_preserved
    assert observed.actual_history == ("UM", "SHA")
    assert observed.steps[0].phase_changes
    assert all(step.certificate.winding == winding for step in observed.steps)
    residual = tuple(
        Fraction.from_float(value) - expected
        for value, expected in zip(gaps, reference.output_gaps, strict=True)
    )
    assert max(map(abs, residual)) < Fraction(1, 10**13)
    assert all(after[node] == phases[node] for node in graph if node != 1)
    assert not observed.steps[1].phase_changes


def test_default_canonical_transition_can_change_winding_without_support_loss():
    result = run_transition_counterexample()
    assert result["initial"]["winding"] == 1
    assert result["first_observed_winding_loss"] == 14
    before, after = result["observations"][-2:]
    assert before["certificate"]["winding"] == 1
    assert after["certificate"]["winding"] == 0
    assert before["certificate"]["minimum_branch_margin"] < 0.007
    assert after["certificate"]["minimum_branch_margin"] > 0.19
    assert before["certificate"]["cycle_exists"]
    assert after["certificate"]["cycle_exists"]
    assert all(row["actual_history"] == ("NAV",) for row in result["observations"])
    assert before["theta_after"] == pytest.approx(2.35)
    assert after["theta_after"] == pytest.approx(2.55)
    # The endpoints are observed. No interpolation was executed by NAV.


def test_branch_and_absent_cycle_controls_remain_undefined():
    branch = build_ring(8, 1, perturb=False)
    branch.nodes[1]["theta"] = math.pi
    certificate = certify_phase_winding(branch, range(8))
    assert not certificate.is_defined
    assert certificate.cycle_exists
    assert certificate.winding is None
    assert certificate.minimum_branch_margin == 0.0
    missing = build_ring(8, 1, perturb=False)
    missing.remove_edge(3, 4)
    certificate = certify_phase_winding(missing, range(8))
    assert not certificate.is_defined
    assert not certificate.cycle_exists
    assert certificate.winding is None
