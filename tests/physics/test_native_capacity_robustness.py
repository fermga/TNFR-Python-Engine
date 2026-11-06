"""Static sufficient neighborhood for persistent capacity adaptation.

Resolve the actual represented defaults before exact Fraction calculations.
Snapshot proposal/gate checks do not execute a native trajectory or refresh
their explicitly supplied diagnostic observations. Matrix powers are algebra,
not replayed adaptation histories. Consensus claims concern the ideal map;
test_cycle_capacity_synergies already owns the binary64 stalling witness.
Admission, rollback and adjacent-float hull contracts remain owned by
test_structural_stability_adaptation. No frozen producer is imported.
"""

from fractions import Fraction as Q
from itertools import product

import networkx as nx
import pytest

from tnfr.config.selector_thresholds import resolve_selector_thresholds
from tnfr.constants import inject_defaults
from tnfr.dynamics import adaptation, integrators, runtime, selectors
from tnfr.dynamics.dnfr import _resolve_dnfr_weights
from tnfr.metrics.sense_index import get_Si_weights

OFFSETS = tuple(Q(value, 16) for value in (-8, 8, -6, 6, -4, 4, -2, 2, 0, 1))
TOL = 2e-15


@pytest.fixture(scope="module", autouse=True)
def no_native_evaluations():
    def forbidden(*args, **kwargs):
        pytest.fail("Capacity robustness controls cannot execute a native trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (runtime, "_prepare_dnfr"),
            (selectors, "_apply_glyphs"),
            (integrators.DefaultIntegrator, "integrate"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


def _support():
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(
        (2 * i + left, 2 * ((i + 1) % 5) + right)
        for i in range(5)
        for left, right in product((0, 1), repeat=2)
    )
    nx.set_edge_attributes(graph, 1.0, "weight")
    inject_defaults(graph)
    return graph


@pytest.fixture(scope="module")
def policy():
    graph = _support()
    weights = _resolve_dnfr_weights(graph)
    factors = graph.graph["GLYPH_FACTORS"]
    k, c = (Q.from_float(weights[key]) for key in ("epi", "vf"))
    m, h, boost = (
        Q.from_float(factors[key]) for key in ("EN_mix", "IL_dnfr_factor", "AL_boost")
    )
    alpha = Q.from_float(get_Si_weights(graph)[0])
    high = Q.from_float(resolve_selector_thresholds(graph)["si_hi"])
    threshold = Q.from_float(graph.graph["EPS_DNFR_STABLE"])
    dt, center, ceiling = Q(1, 16), Q(1, 2), Q(1)
    z_max = dt * ceiling * k
    lam = 1 - m - z_max
    a = max(1 - boost / 2, z_max)
    contraction = 1 - (1 - a) * lam**5
    pressure_factor = c * (1 + 6 * z_max / (1 - contraction))
    radius = min(
        center / 2,
        center * (alpha - high) / (2 * (alpha + high)),
        threshold / (4 * pressure_factor),
    )
    return dict(
        k=k,
        c=c,
        m=m,
        h=h,
        boost=boost,
        alpha=alpha,
        high=high,
        threshold=threshold,
        dt=dt,
        center=center,
        ceiling=ceiling,
        z_max=z_max,
        lam=lam,
        a=a,
        contraction=contraction,
        pressure_factor=pressure_factor,
        radius=radius,
        mu=Q.from_float(graph.graph["VF_ADAPT_MU"]),
        tau=graph.graph["VF_ADAPT_TAU"],
        rho=1 - Q.from_float(graph.graph["PHASE_ADAPT"]["kG_min"]),
        vf_max=Q.from_float(graph.graph["VF_MAX"]),
    )


@pytest.fixture(scope="module")
def capacities(policy):
    # Use half the admitted radius, then associate exact predictions with the
    # represented primitive inputs consumed by the actual proposal kernel.
    values = tuple(
        Q.from_float(float(policy["center"] + offset * policy["radius"]))
        for offset in OFFSETS
    )
    assert policy["center"] - policy["radius"] < min(values) < max(values)
    assert max(values) < policy["center"] + policy["radius"]
    return values


def _neighbor_indices(node):
    return tuple(j for j in range(10) if (j // 2 - node // 2) % 5 in (1, 4))


def _matrix(mu):
    return tuple(
        tuple(
            1 - mu if i == j else mu / 4 if j in _neighbor_indices(i) else Q(0)
            for j in range(10)
        )
        for i in range(10)
    )


def _apply(matrix, values):
    return tuple(sum(a * value for a, value in zip(row, values)) for row in matrix)


def _multiply(left, right):
    return tuple(
        tuple(sum(row[k] * right[k][j] for k in range(10)) for j in range(10))
        for row in left
    )


def test_constructive_radius_satisfies_both_strict_native_policy_margins(policy):
    center, radius = policy["center"], policy["radius"]
    lower, upper = center - radius, center + radius
    assert 0 < lower < center < upper < policy["ceiling"] < policy["vf_max"]
    assert 0 < policy["lam"] < 1 and 0 < policy["a"] < 1
    assert 0 < policy["contraction"] < 1
    assert policy["alpha"] * lower / upper > policy["high"]
    spread = upper - lower
    assert policy["pressure_factor"] * spread <= policy["threshold"] / 2
    assert policy["pressure_factor"] * spread < policy["threshold"]
    # Derive the pressure bound from the deficit forcing, not just a cached
    # criterion: global limsup D is bounded by 6 Fcap/(1-Q), without +5 Fcap.
    capacity_forcing = policy["dt"] * policy["ceiling"] * policy["c"] * spread
    deficit_bound = 6 * capacity_forcing / (1 - policy["contraction"])
    pressure_bound = policy["k"] * deficit_bound + policy["c"] * spread
    assert pressure_bound == policy["pressure_factor"] * spread


@pytest.mark.parametrize("mark", ("AL", "EN", "IL"))
def test_variable_capacity_rows_share_the_conservative_deficit_bound(policy, mark):
    # Affine dependence on z makes the endpoints sufficient on [0,Z]; the
    # interior control separately exercises a genuinely different capacity.
    width, boost = Q(2), policy["boost"]
    spread = 2 * policy["radius"]
    forcing = policy["dt"] * policy["ceiling"] * policy["c"] * spread
    for z, maximum in product(
        (Q(0), policy["z_max"] / 3, policy["z_max"]), (boost / 2, width)
    ):
        for own, live, incoming, sign in product(
            (Q(0), min(boost, maximum), maximum),
            (Q(0), maximum),
            (Q(0), maximum),
            (-1, 1),
        ):
            if mark == "AL":
                event, held = max(Q(0), own - boost), Q(1)
                bound = policy["a"] * maximum + forcing
            elif mark == "EN":
                event, held = (1 - policy["m"]) * own + policy["m"] * live, Q(1)
                bound = policy["lam"] * own + (1 - policy["lam"]) * maximum + forcing
            else:
                event, held = own, policy["h"]
                bound = policy["lam"] * own + (1 - policy["lam"]) * maximum + forcing
            raw = event - z * held * own + z * held * incoming - sign * forcing
            assert min(width, max(Q(0), raw)) <= bound
    # The z=0 algebra endpoint does not assert admission of a zero-capacity
    # native state in this strictly positive-capacity neighborhood.


def test_every_eligibility_mask_preserves_the_capacity_hull(capacities, policy):
    proposals = _apply(_matrix(policy["mu"]), capacities)
    lower, upper = min(capacities), max(capacities)
    for eligible in product((False, True), repeat=10):
        following = tuple(
            p if due else v for p, due, v in zip(proposals, eligible, capacities)
        )
        assert lower <= min(following) <= max(following) <= upper
    assert lower < upper  # In particular the empty mask leaves a nonconsensus state.


@pytest.mark.parametrize("eligible", ((), (0,), (0, 3, 6, 9), tuple(range(10))))
def test_shared_snapshot_kernel_matches_independent_masked_rows(
    capacities, policy, eligible
):
    graph = _support()
    assert all(tuple(sorted(graph.neighbors(i))) == _neighbor_indices(i) for i in graph)
    expected = tuple(
        (
            (1 - policy["mu"]) * value
            + policy["mu"] * sum(capacities[j] for j in _neighbor_indices(i)) / 4
            if i in eligible
            else value
        )
        for i, value in enumerate(capacities)
    )
    work = [(i, i, _neighbor_indices(i)) for i in eligible]
    updates = dict(
        adaptation._vf_adapt_chunk(
            (work, tuple(map(float, capacities)), float(policy["mu"]))
        )
    )
    actual = tuple(updates.get(i, float(value)) for i, value in enumerate(capacities))
    assert actual == pytest.approx(tuple(map(float, expected)), abs=TOL, rel=0)
    assert (
        min(map(float, capacities))
        <= min(actual)
        <= max(actual)
        <= max(map(float, capacities))
    )
    if not eligible:
        assert tuple(map(Q.from_float, actual)) == capacities
    elif eligible == (0,):
        assert sum(expected) > sum(capacities)
    elif len(eligible) == 10:
        assert sum(expected) == sum(capacities)
    # Partial updates can move the mean; full activation preserves the mean
    # at its own onset, not necessarily the original preparation's mean.


def _gate_snapshot(capacities, policy, count, pressure):
    graph = _support()
    for node, capacity in enumerate(capacities):
        graph.nodes[node].update(
            nu_f=float(capacity),
            Si=float(policy["high"]),
            delta_nfr=float(pressure),
            stable_count=count,
        )
    return graph


@pytest.mark.parametrize("offset", (-2, -1, 0, 7))
def test_independent_qualifying_snapshots_start_and_continue_writes(
    capacities, policy, offset
):
    old_count = policy["tau"] + offset
    graph = _gate_snapshot(capacities, policy, old_count, policy["threshold"])
    due = old_count + 1 >= policy["tau"]
    expected = _apply(_matrix(policy["mu"]), capacities) if due else capacities
    adaptation.adapt_vf_after_structural_stability(graph, n_jobs=1)
    assert tuple(graph.nodes[i]["stable_count"] for i in graph) == (old_count + 1,) * 10
    actual = tuple(graph.nodes[i]["nu_f"] for i in graph)
    assert actual == pytest.approx(tuple(map(float, expected)), abs=TOL, rel=0)
    assert (actual[0] != float(capacities[0])) is due
    assert all(graph.nodes[i]["Si"] == float(policy["high"]) for i in graph)
    assert all(graph.nodes[i]["delta_nfr"] == float(policy["threshold"]) for i in graph)
    # Each parameter starts a fresh graph: no repeated native history or
    # claim that these supplied diagnostics came from a pressure preparation.


def test_strict_limsup_margin_cannot_be_replaced_by_threshold_equality(
    capacities, policy
):
    threshold, rho = policy["threshold"], policy["rho"]
    assert 0 < rho < 1
    for index in (0, 1, policy["tau"], policy["tau"] + 7):
        # Abstract counterexample p_n=T+T*rho^n has limsup T, yet never meets
        # p_n<=T at finite n. It is not asserted to be a native pressure path.
        pressure = threshold * (1 + rho**index)
        next_pressure = threshold * (1 + rho ** (index + 1))
        assert threshold < next_pressure < pressure
        assert next_pressure - threshold == rho * (pressure - threshold)
        graph = _gate_snapshot(capacities, policy, policy["tau"] + 7, pressure)
        adaptation.adapt_vf_after_structural_stability(graph, n_jobs=1)
        assert tuple(graph.nodes[i]["stable_count"] for i in graph) == (0,) * 10
        assert tuple(Q.from_float(graph.nodes[i]["nu_f"]) for i in graph) == capacities


def test_doubled_cycle_lazy_square_has_the_exact_minorization(policy, capacities):
    mu = policy["mu"]
    matrix = _matrix(mu)
    square = _multiply(matrix, matrix)
    assert 0 < mu < 1
    for i in range(10):
        assert sum(matrix[i]) == sum(row[i] for row in matrix) == 1
        for j in range(10):
            if i == j:
                expected = (1 - mu) ** 2 + mu**2 / 4
            elif i // 2 == j // 2:
                expected = mu**2 / 4
            elif j in _neighbor_indices(i):
                expected = mu * (1 - mu) / 2
            else:
                expected = mu**2 / 8
            assert square[i][j] == expected
    minorization = mu**2 / 8
    factor = 1 - 10 * minorization
    assert min(min(row) for row in square) == minorization > 0
    assert 0 < factor < 1
    residual = tuple(
        tuple((entry - minorization) / factor for entry in row) for row in square
    )
    assert all(min(row) >= 0 and sum(row) == 1 for row in residual)
    image = _apply(square, capacities)
    residual_image = _apply(residual, capacities)
    assert image == tuple(
        minorization * sum(capacities) + factor * value for value in residual_image
    )
    assert max(image) - min(image) <= factor * (max(capacities) - min(capacities))
    assert sum(image) == sum(capacities)
