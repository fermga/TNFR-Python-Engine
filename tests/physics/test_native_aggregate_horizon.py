"""Exact static controls for the unclipped aggregate transition's horizon.

Enumerate the 3**5 ordered event profiles as affine matrices, not trajectories.
Resolve represented coefficients from shared defaults. The fixed source is
the preceding transition owner's declared constants; its fixture and native
responses are never replayed. Lag snapshots exercise the actual selector
helper. Recurrence, geometric forcing and rail bounds remain analytical,
conditional on the unchanged admitted model and its clipping-free prefix.
"""

import math
from fractions import Fraction as Q
from itertools import product

import networkx as nx
import pytest

from tests.physics import test_native_aggregate_transition as source
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.dynamics import integrators, runtime, selectors
from tnfr.dynamics.dnfr import _resolve_dnfr_weights
from tnfr.metrics.sense_index import get_Si_weights
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    canonical_operator_code,
    reset_grammar_state_from_history,
)


@pytest.fixture(scope="module", autouse=True)
def no_native_evaluations():
    def forbidden(*args, **kwargs):
        pytest.fail("Horizon controls are static; native responses are not replayed")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (runtime, "_prepare_dnfr"),
            (integrators.DefaultIntegrator, "integrate"),
            (source.native, "_execute"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def policy():
    graph = nx.Graph()
    inject_defaults(graph)
    weights = _resolve_dnfr_weights(graph)
    factors = graph.graph["GLYPH_FACTORS"]
    m, h, boost = (Q(factors[key]) for key in ("EN_mix", "IL_dnfr_factor", "AL_boost"))
    dt, nu, k = Q(source.DT), Q(source.CAPACITY), Q(weights["epi"])
    z = dt * nu * k
    eta = 1 - m - z
    global_min = Q(graph.graph["PHASE_ADAPT"]["kG_min"])
    q = 1 - global_min
    diameter = Q(max(source.PHASES) - min(source.PHASES))
    forcing = dt * nu * Q(weights["phase"]) * diameter / Q(math.pi)
    period = graph.graph["AL_MAX_LAG"] + 1
    assert period == 6 and 0 < z < 1 - m < 1 and 0 < h < 1
    assert 0 < eta < 1 and 0 < q < 1
    return dict(
        m=m,
        h=h,
        boost=boost,
        dt=dt,
        k=k,
        vf_max=Q(graph.graph["VF_MAX"]),
        z=z,
        eta=eta,
        q=q,
        forcing=forcing,
        budget=forcing / global_min,
        period=period,
        increment=boost * eta ** (period - 1),
        alpha=get_Si_weights(graph)[0],
    )


def test_default_capacity_rails_have_a_positive_uniform_diagonal_margin(policy):
    z_max = policy["dt"] * policy["vf_max"] * policy["k"]
    reception_diagonal = 1 - policy["m"]
    lambda_lower = reception_diagonal - z_max
    assert z_max < Q(71925, 1_000_000)
    assert reception_diagonal > Q(758546, 1_000_000)
    assert lambda_lower > Q(686622, 1_000_000) > 0
    # The represented IL retention is resolved separately; no identity
    # between its rounded literal and 1-m is needed for this bound.


def _identity(size):
    return tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))


def _multiply(left, right):
    return tuple(
        tuple(
            sum(a * right[k][j] for k, a in enumerate(row) if a)
            for j in range(len(right[0]))
        )
        for row in left
    )


def _apply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _event_matrix(word, policy):
    result = _identity(6)
    for i, mark in enumerate(word):
        local = [list(row) for row in _identity(6)]
        if mark == "AL":
            local[i][-1] = policy["boost"]
        elif mark == "EN":
            local[i] = [Q(0)] * 6
            local[i][i] = 1 - policy["m"]
            local[i][(i - 1) % 5] = local[i][(i + 1) % 5] = policy["m"] / 2
        result = _multiply(local, result)
    return result


def _held_matrix(event, word, z, h):
    matrix = [list(row[:5]) for row in event[:5]]
    for i, mark in enumerate(word):
        coefficient = z * (h if mark == "IL" else 1)
        matrix[i][i] -= coefficient
        matrix[i][(i - 1) % 5] += coefficient / 2
        matrix[i][(i + 1) % 5] += coefficient / 2
    return tuple(map(tuple, matrix))


def _scalar_events(word, policy, initial):
    values = list(initial)
    for i, mark in enumerate(word):
        if mark == "AL":
            values[i] += policy["boost"]
        elif mark == "EN":
            values[i] = (1 - policy["m"]) * values[i] + policy["m"] * (
                values[(i - 1) % 5] + values[(i + 1) % 5]
            ) / 2
    return tuple(values)


@pytest.fixture(scope="module")
def profiles(policy):
    return tuple(
        (word, _event_matrix(word, policy))
        for word in product(("AL", "EN", "IL"), repeat=5)
    )


def test_every_ordered_event_profile_has_a_stochastic_linear_part(profiles, policy):
    initial = tuple(map(Q, source.MEANS))
    assert len(profiles) == 3**5
    for word, matrix in profiles:
        assert matrix[-1] == (0, 0, 0, 0, 0, 1)
        for i, row in enumerate(matrix[:5]):
            assert min(row) >= 0 and sum(row[:5]) == 1
            assert row[i] >= 1 - policy["m"]
            assert 0 <= row[-1] <= policy["boost"]
            if word[i] == "AL":
                assert row[-1] == policy["boost"]
        assert _apply(matrix, (*initial, Q(1)))[:5] == _scalar_events(
            word, policy, initial
        )


@pytest.mark.parametrize("regime", ("zero", "source", "endpoint"))
def test_combined_held_row_is_nonnegative_on_the_declared_step_interval(
    profiles, policy, regime
):
    z = {"zero": Q(0), "source": policy["z"], "endpoint": 1 - policy["m"]}[regime]
    initial = tuple(map(Q, source.MEANS))
    for word, event in profiles:
        matrix = _held_matrix(event, word, z, policy["h"])
        for i, row in enumerate(matrix):
            assert min(row) >= 0 and sum(row) == 1
            assert row[i] >= 1 - policy["m"] - z
        expected = tuple(
            y
            + z
            * (policy["h"] if mark == "IL" else 1)
            * ((initial[(i - 1) % 5] + initial[(i + 1) % 5]) / 2 - initial[i])
            for i, (y, mark) in enumerate(
                zip(_scalar_events(word, policy, initial), word)
            )
        )
        observed = tuple(
            value + event[i][-1] for i, value in enumerate(_apply(matrix, initial))
        )
        assert observed == expected
    # Every entry is affine in z, so the two checked interval endpoints imply
    # the nonnegativity/row-sum statement throughout, not only at source z.


def test_monotonicity_boundary_is_stricter_than_the_contrast_bound(policy):
    word = ("EN", "IL", "IL", "IL", "IL")
    event = _event_matrix(word, policy)
    endpoint = 1 - policy["m"]
    at_boundary = _held_matrix(event, word, endpoint, policy["h"])
    assert at_boundary[0][0] == 0
    # eta=0 gives no positive diagonal-survival drift certificate.
    assert policy["boost"] * at_boundary[0][0] ** 5 == 0
    outside = endpoint + Q(1, 1024)
    assert outside < 2 - policy["m"]
    assert _held_matrix(event, word, outside, policy["h"])[0][0] == -Q(1, 1024)
    assert abs(1 - policy["m"] - outside) < 1


def test_actual_lag_helper_forces_emission_within_six_calls(policy):
    graph = nx.cycle_graph(5)
    inject_defaults(graph)
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    for n in graph:
        graph.nodes[n].update(
            EPI=1 / 4,
            nu_f=source.CAPACITY,
            theta=1.0,
            Si=policy["alpha"],
            delta_nfr=0.0,
            d2EPI=0.0,
            glyph_history=["AL", "IL"],
            **{OPERATOR_STEP_KEY: 2},
        )
        reset_grammar_state_from_history(graph.nodes[n])
    selector = selectors._apply_selector(graph)
    before = tuple(tuple(graph.nodes[n]["glyph_history"]) for n in graph)
    # Independent counter snapshots, never an iterated native state. Actual
    # reset writes are covered by test_native_aggregate_transition's lag test.
    for al in range(7):
        for en in (0, 3, 4):
            chosen = canonical_operator_code(
                selectors._choose_glyph(
                    graph,
                    0,
                    selector,
                    True,
                    {0: al + 1},
                    {0: en + 1},
                    graph.graph["AL_MAX_LAG"],
                    graph.graph["EN_MAX_LAG"],
                )
            )
            expected = "AL" if al >= 5 else "EN" if en >= 3 else "IL"
            assert chosen == expected
        first_emission = max(1, 6 - al)
        assert al + first_emission > 5
        assert first_emission == 1 or al + first_emission - 1 <= 5
    assert tuple(tuple(graph.nodes[n]["glyph_history"]) for n in graph) == before


def test_worst_alignment_retains_five_later_diagonal_factors(policy):
    # Comparison-class sharpness, not a claim that this matrix is a native
    # cycle profile: every other coordinate is absorbing and receives no tag.
    eta, boost = policy["eta"], policy["boost"]
    matrix = [list(row) for row in _identity(5)]
    matrix[0] = [eta, 1 - eta, Q(0), Q(0), Q(0)]
    power = _identity(5)
    for later_calls in range(policy["period"]):
        tagged = _apply(power, (boost, Q(0), Q(0), Q(0), Q(0)))
        assert tagged[0] == boost * eta**later_calls >= policy["increment"]
        assert tagged[1:] == (0, 0, 0, 0)
        power = _multiply(matrix, power)
    assert policy["increment"] == boost * eta**5  # First-call input, five later maps.


@pytest.mark.parametrize("calls", (0, 1, 6, 12))
def test_finite_geometric_phase_budget_is_below_the_total_reserve(policy, calls):
    terms = tuple(policy["forcing"] * policy["q"] ** j for j in range(calls))
    finite = sum(terms, Q(0))
    assert finite * (1 - policy["q"]) == policy["forcing"] * (1 - policy["q"] ** calls)
    assert 0 <= finite < policy["budget"]


def _ceil(value):
    return -(-value.numerator // value.denominator)


@pytest.mark.parametrize("ratio", (Q(4), Q(9, 2)))
def test_strict_margin_exit_and_closed_rail_exit_have_different_equality_rules(ratio):
    initial, increment, reserve = Q(1, 8), Q(1, 4), Q(1, 16)
    upper = initial + ratio * increment - reserve
    strict_margin_blocks = _ceil(ratio)
    closed_rail_blocks = ratio.numerator // ratio.denominator + 1
    assert initial + (strict_margin_blocks - 1) * increment - reserve < upper
    assert initial + strict_margin_blocks * increment - reserve >= upper
    assert initial + (closed_rail_blocks - 1) * increment - reserve <= upper
    assert initial + closed_rail_blocks * increment - reserve > upper
    assert (closed_rail_blocks == strict_margin_blocks + 1) is (ratio.denominator == 1)


def test_previous_frozen_source_has_a_finite_analytic_exit_bound(policy):
    minimum = min(map(Q, source.MEANS))
    radius, upper = Q(source.FORM_RADIUS), Q(DEFAULTS["EPI_MAX"])
    strict_ratio = (upper - radius - minimum + policy["budget"]) / policy["increment"]
    closed_ratio = (upper - minimum + policy["budget"]) / policy["increment"]
    strict_blocks = _ceil(strict_ratio)
    closed_blocks = closed_ratio.numerator // closed_ratio.denominator + 1
    assert strict_blocks == 46 and closed_blocks == 47
    assert (
        minimum + (strict_blocks - 1) * policy["increment"] - policy["budget"]
        < upper - radius
    )
    assert (
        minimum + strict_blocks * policy["increment"] - policy["budget"]
        >= upper - radius
    )
    assert (
        minimum + (closed_blocks - 1) * policy["increment"] - policy["budget"] <= upper
    )
    assert minimum + closed_blocks * policy["increment"] - policy["budget"] > upper
    assert policy["period"] * strict_blocks == 276
    assert policy["period"] * closed_blocks == 282
    assert Q(source.DT) * policy["period"] * strict_blocks == Q(69, 4)
    assert Q(source.DT) * policy["period"] * closed_blocks == Q(141, 8)
    # These are upper bounds on an unchanged unclipped prefix, not evaluated
    # crossing times, guaranteed earlier admission or engine failure claims.
