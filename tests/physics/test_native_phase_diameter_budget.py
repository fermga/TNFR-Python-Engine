"""Native coordination contracts a coherent lifted phase interval.

The convex-map theorem and finite bookkeeping identity are exact statements.
One detached invocation binds them to captured native gains and targets, with
its represented realization defect retained. No trajectory, phase-reset rule,
physical clock, or complete EPI/capacity convergence is inferred here.
"""

import math
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import _nonrepeated_phase_geometry
from tnfr.alias import get_theta_attr
from tnfr.constants import DEFAULTS, STATE_DISSONANT, STATE_STABLE, STATE_TRANSITION
from tnfr.dynamics.coordination import _smooth_adjust_k, coordinate_global_local_phase


def _diameter(values):
    return max(values) - min(values)


def test_lifted_convex_coordinator_has_a_uniform_diameter_bound():
    s = pytest.importorskip("sympy")
    lo, hi, target, theta, local = s.symbols("lo hi target theta local", real=True)
    k_global, k_local = s.symbols("k_global k_local", nonnegative=True)
    own = 1 - k_global - k_local
    proposal = own * theta + k_global * target + k_local * local
    lower = (1 - k_global) * lo + k_global * target
    upper = (1 - k_global) * hi + k_global * target

    # In a lift of width < pi, every nonempty neighbor/global phasor mean
    # lies in [lo, hi]: rotate the cone to its midpoint, where all phasors
    # have positive real parts. An isolated node's fallback lies there too.
    # With own >= 0, each term in these two slack identities is nonnegative.
    assert s.expand(proposal - lower - own * (theta - lo) - k_local * (local - lo)) == 0
    assert s.expand(upper - proposal - own * (hi - theta) - k_local * (hi - local)) == 0
    assert s.expand(upper - lower - (1 - k_global) * (hi - lo)) == 0

    # The common target cancels in pair differences. The interval bound is
    # sharp when two local targets remain at their respective endpoints;
    # such targets are allowed, for example, by singleton self-neighborhoods.
    pair = proposal.subs({theta: hi, local: hi}) - proposal.subs({theta: lo, local: lo})
    assert s.expand(pair - (1 - k_global) * (hi - lo)) == 0
    # No positive global gain means this theorem promises only nonincrease.
    assert s.simplify(pair.subs(k_global, 0)) == hi - lo


def test_default_adaptive_clamps_have_an_exact_represented_contraction_floor():
    cfg = DEFAULTS["PHASE_ADAPT"]
    floor = Q(cfg["kG_min"])
    ceiling_g, ceiling_l = Q(cfg["kG_max"]), Q(cfg["kL_max"])
    assert cfg["enabled"] is True
    assert 0 < floor <= Q(DEFAULTS["PHASE_K_GLOBAL"]) <= ceiling_g
    assert 0 <= Q(cfg["kL_min"]) <= Q(DEFAULTS["PHASE_K_LOCAL"]) <= ceiling_l
    assert ceiling_g + ceiling_l < 1
    assert 0 < 1 - floor < 1

    # Exhaust the three policy branches, not a gain or state search. Each
    # update ends with min/max clamps; the exact bounds below are the stored
    # binary64 numbers, not an asserted equality to irrational 1/(8*pi**2).
    for state in (STATE_STABLE, STATE_TRANSITION, STATE_DISSONANT):
        global_gain, local_gain = _smooth_adjust_k(
            DEFAULTS["PHASE_K_GLOBAL"], DEFAULTS["PHASE_K_LOCAL"], state, cfg
        )
        global_gain, local_gain = Q(global_gain), Q(local_gain)
        assert floor <= global_gain <= ceiling_g
        assert Q(cfg["kL_min"]) <= local_gain <= ceiling_l
        assert 0 <= global_gain + local_gain < 1
        assert 1 - global_gain <= 1 - floor


def test_finite_spread_ledger_retains_intervening_changes_and_realization_errors():
    # Detached rational rows prove the bookkeeping statement. These are not
    # an executed history or an admission/occurrence law for phase writers.
    q = Q(3, 4)
    initial = (Q(-1, 2), Q(0), Q(1, 2))
    before_rows = (
        (Q(-3, 4), Q(0), Q(3, 4)),
        (Q(-1, 4), Q(0), Q(1, 4)),
        initial,
    )
    defects = (
        (Q(-1, 64), Q(0), Q(1, 64)),
        (Q(1, 128),) * 3,
        (Q(1, 64), Q(0), Q(-1, 64)),
    )
    previous = _diameter(initial)
    records = []
    for before, defect in zip(before_rows, defects, strict=True):
        recharge = max(Q(0), _diameter(before) - previous)
        # A common target zero and local targets equal to theta attain the
        # abstract bound. No such target selection is installed in runtime.
        ideal = tuple(q * value for value in before)
        realized = tuple(a + b for a, b in zip(ideal, defect, strict=True))
        epsilon = _diameter(defect)
        after = _diameter(realized)
        slack = q * previous + q * recharge + epsilon - after
        assert slack >= 0
        records.append((recharge, epsilon, slack))
        previous = after

    assert tuple(row[0] for row in records) == (Q(1, 2), Q(0), Q(5, 8))
    assert tuple(row[1] for row in records) == (Q(1, 32), Q(0), Q(1, 32))
    # The first realization increases spread; a common error in the second
    # leaves it unchanged. Positive epsilon also safely covers inward errors.
    assert records[0][2] == 0
    assert records[1][2] > 0
    count = len(records)
    upper = q**count * _diameter(initial) + sum(
        q ** (count - index - 1) * (q * recharge + epsilon)
        for index, (recharge, epsilon, _) in enumerate(records)
    )
    lost = sum(
        q ** (count - index - 1) * slack for index, (_, _, slack) in enumerate(records)
    )
    assert previous == upper - lost <= upper


def test_one_native_coordination_obeys_the_captured_lifted_budget_with_residual():
    _, graph, _, phases = _nonrepeated_phase_geometry()
    nodes = tuple(graph)
    for node, phase in zip(nodes, phases, strict=True):
        # Same retained relative angles and EPI amplitudes; a common pi shift
        # keeps represented writes away from the 0/2*pi storage boundary.
        graph.nodes[node]["theta"] = math.pi + float(phase)
    held_form = tuple((graph.nodes[n]["EPI"], graph.nodes[n]["nu_f"]) for n in nodes)
    held_edges = tuple(graph.edges(data="weight"))
    evidence = coordinate_global_local_phase(
        graph, n_jobs=1, global_reduction="exact_components_v1"
    )
    assert evidence.version == "exact_components_v1"
    assert evidence.status == "applied"
    assert evidence.gain_mode == "adaptive"
    assert evidence.requested_global_force is evidence.requested_local_force is None

    def lifted(value):
        represented = Q(value)
        return represented + 2 * Q(math.pi) if represented < 0 else represented

    before = tuple(map(Q, evidence.primitive_phases))
    local = tuple(map(lifted, evidence.local_targets))
    target = lifted(evidence.global_target)
    lo, hi = min(before), max(before)
    assert 0 < hi - lo < Q(math.pi)
    assert lo <= target <= hi
    assert all(lo <= value <= hi for value in local)
    k_global, k_local = Q(evidence.effective_global_force), Q(
        evidence.effective_local_force
    )
    floor = Q(DEFAULTS["PHASE_ADAPT"]["kG_min"])
    assert floor <= k_global and 0 <= k_local and k_global + k_local <= 1
    ideal = tuple(
        (1 - k_global - k_local) * theta + k_global * target + k_local * neighbor
        for theta, neighbor in zip(before, local, strict=True)
    )
    actual = tuple(map(Q, evidence.realized_phases))
    defect = tuple(a - b for a, b in zip(actual, ideal, strict=True))
    epsilon = _diameter(defect)
    assert _diameter(ideal) <= (1 - k_global) * _diameter(before)
    assert _diameter(actual) <= (1 - k_global) * _diameter(before) + epsilon
    assert _diameter(actual) <= (1 - floor) * _diameter(before) + epsilon
    assert 0 < max(map(abs, defect)) < Q(1, 2**45)
    assert epsilon < Q(1, 2**44)
    # This finite readout is not a sealed trajectory certificate. Its exact
    # rational defect includes arithmetic, wrap and normalization relative
    # to captured represented means, not certified transcendental angles.
    assert (
        tuple(get_theta_attr(graph.nodes[n]) for n in nodes) == evidence.realized_phases
    )
    assert (
        tuple((graph.nodes[n]["EPI"], graph.nodes[n]["nu_f"]) for n in nodes)
        == held_form
    )
    assert tuple(graph.edges(data="weight")) == held_edges
