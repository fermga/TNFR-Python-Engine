"""Static hard-boundary continuation of the paired aggregate form chart.

The declared event profile is Emission (AL), Reception (EN), then Coherence
(IL) on blocks 0,...,4 of the doubled unit C5. Exact arithmetic resolves
represented default coefficients; this is conditional algebra, not a native trajectory,
an occurrence law or a binary64 permutation-invariance certificate. Only
pure shared clipping and Emission proposal contracts execute.

The native path, clipping/rate distinction and invalid scalar boundaries
remain owned by test_native_collective_step, test_integrator_numerics and
test_structural_clip_contract. No preceding evaluated source is replayed.
"""

import math
from fractions import Fraction as Q
from itertools import product

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import integrators, runtime, selectors
from tnfr.dynamics.dnfr import _resolve_dnfr_weights
from tnfr.dynamics.structural_clip import resolve_clip_policy, structural_clip
from tnfr.operators.al_sha_stage_proposals import emission_epi_proposal

MEANS = (Q(15, 16), Q(1, 4), Q(-1, 4), Q(1, 8), Q(1, 2))
RADII = (Q(1, 16), Q(1, 8), Q(1, 32), Q(1, 16), Q(1, 8))
PHASES = (Q(1), Q(17, 16), Q(1), Q(15, 16), Q(1))
ORIENTATIONS = tuple(product((-1, 1), repeat=5))
TOL = 2e-15


@pytest.fixture(scope="module", autouse=True)
def no_native_evaluations():
    def forbidden(*args, **kwargs):
        pytest.fail("Clipping controls are static; native/flow execution is forbidden")

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


@pytest.fixture(scope="module")
def policy():
    graph = nx.Graph()
    inject_defaults(graph)
    weights = _resolve_dnfr_weights(graph)
    factors = graph.graph["GLYPH_FACTORS"]
    lo, hi, mode, steepness = resolve_clip_policy(graph.graph)
    dt, nu = Q(1, 16), Q(1, 2)
    m, h, boost = (Q(factors[key]) for key in ("EN_mix", "IL_dnfr_factor", "AL_boost"))
    k = Q(weights["epi"])
    z = dt * nu * k
    assert (lo, hi, mode) == (-1, 1, "hard")
    assert 0 < z < min(Q(1, 2), 1 - m) and 0 < h < 1
    return dict(
        config=graph.graph,
        clip=dict(lo=lo, hi=hi, mode=mode, k=steepness, record_stats=False),
        m=m,
        h=h,
        boost=boost,
        k=k,
        phase=Q(weights["phase"]) / Q(math.pi),
        delta=dt * nu,
        z=z,
    )


def _clip(value):
    return min(Q(1), max(Q(-1), value))


def _pair_projection(center, radius):
    """Independent piecewise center/radius formula on the declared rails."""
    assert radius >= 0
    lower, upper = center - radius, center + radius
    if upper <= -1:
        return Q(-1), Q(0)
    if lower >= 1:
        return Q(1), Q(0)
    if lower < -1:
        if upper > 1:
            return Q(0), Q(1)
        return (upper - 1) / 2, (upper + 1) / 2
    if upper > 1:
        return (lower + 1) / 2, (1 - lower) / 2
    return center, radius


def _neighbor_mean(values, index):
    return (values[(index - 1) % 5] + values[(index + 1) % 5]) / 2


def _quotient_formula(means, radii, phases, policy):
    # Prepared pressure belongs to the incoming primitive state throughout.
    pressure = tuple(
        policy["k"] * (_neighbor_mean(means, i) - means[i])
        + policy["phase"] * (_neighbor_mean(phases, i) - phases[i])
        for i in range(5)
    )
    events = list(zip(means, radii))
    events[0] = _pair_projection(means[0] + policy["boost"], radii[0])
    reception = (1 - policy["m"]) * means[1] + policy["m"] * (
        events[0][0] + means[2]
    ) / 2
    events[1] = _pair_projection(reception, (1 - policy["m"]) * radii[1])
    held = (Q(1), Q(1), policy["h"], policy["h"], policy["h"])
    raw = tuple(
        (y + policy["delta"] * h * b, t - policy["z"] * h * s)
        for (y, t), h, b, s in zip(events, held, pressure, radii)
    )
    return dict(
        means=means,
        radii=radii,
        phases=phases,
        pressure=pressure,
        events=tuple(events),
        held=held,
        raw=raw,
        final=tuple(_pair_projection(u, abs(v)) for u, v in raw),
    )


def _fine_formula(source, orientation, policy):
    """Fixed-profile constituent algebra; no runtime or generic executor."""
    initial = tuple(
        (m - sign * s, m + sign * s)
        for m, s, sign in zip(source["means"], source["radii"], orientation)
    )
    pressure = []
    for i, pair in enumerate(initial):
        adjacent = initial[(i - 1) % 5] + initial[(i + 1) % 5]
        phase_mean = _neighbor_mean(source["phases"], i)
        pressure.append(
            tuple(
                policy["k"] * (sum(adjacent) / 4 - value)
                + policy["phase"] * (phase_mean - source["phases"][i])
                for value in pair
            )
        )
    event = list(initial)
    event[0] = tuple(_clip(value + policy["boost"]) for value in initial[0])
    adjacent = event[0] + event[2]
    event[1] = tuple(
        _clip((1 - policy["m"]) * value + policy["m"] * sum(adjacent) / 4)
        for value in initial[1]
    )
    raw = tuple(
        tuple(value + policy["delta"] * h * p for value, p in zip(pair, pressures))
        for pair, pressures, h in zip(event, pressure, source["held"])
    )
    return dict(
        initial=initial,
        pressure=tuple(pressure),
        events=tuple(event),
        raw=raw,
        final=tuple(tuple(_clip(value) for value in pair) for pair in raw),
    )


@pytest.fixture(scope="module")
def cases(policy):
    # All three sources and independent forecasts precede shared scalar calls.
    sources = {
        "partial": (MEANS, RADII),
        "saturated": ((Q(31, 32), *MEANS[1:]), (Q(1, 64), *RADII[1:])),
        "zero": (MEANS, (Q(0),) * 5),
    }
    result = {}
    for name, (means, radii) in sources.items():
        assert all(-1 <= m - s <= m + s <= 1 for m, s in zip(means, radii))
        result[name] = _quotient_formula(means, radii, PHASES, policy)
    # The common lift has width 1/8 < pi. Each block sees two equally weighted
    # paired-phase neighbor blocks, so its circular resultant mean is their
    # arithmetic midpoint in this lift. This is not a global phase-Laplacian claim.
    assert max(PHASES) - min(PHASES) == Q(1, 8) < Q(math.pi)
    return result


def test_pair_branch_table_matches_actual_scalar_projection(policy):
    table = (
        (Q(0), Q(1, 4), Q(0), Q(1, 4)),
        (Q(15, 16), Q(1, 8), Q(29, 32), Q(3, 32)),
        (Q(-15, 16), Q(1, 8), Q(-29, 32), Q(3, 32)),
        (Q(0), Q(2), Q(0), Q(1)),
        (Q(5, 4), Q(1, 8), Q(1), Q(0)),
        (Q(-5, 4), Q(1, 8), Q(-1), Q(0)),
        (Q(3, 4), Q(1, 4), Q(3, 4), Q(1, 4)),
    )
    for center, radius, expected_mean, expected_radius in table:
        assert _pair_projection(center, radius) == (expected_mean, expected_radius)
        lower, upper = (
            structural_clip(value, **policy["clip"])
            for value in (center - radius, center + radius)
        )
        assert ((lower + upper) / 2, (upper - lower) / 2) == (
            expected_mean,
            expected_radius,
        )


@pytest.mark.parametrize("name", ("partial", "saturated", "zero"))
def test_complete_event_profile_closes_for_all_independent_pair_swaps(
    cases, policy, name
):
    source = cases[name]
    for orientation in ORIENTATIONS:
        fine = _fine_formula(source, orientation, policy)
        for i, sign in enumerate(orientation):
            b, s = source["pressure"][i], source["radii"][i]
            assert fine["pressure"][i] == (
                b + sign * policy["k"] * s,
                b - sign * policy["k"] * s,
            )
            y, t = source["events"][i]
            assert fine["events"][i] == (y - sign * t, y + sign * t)
            u, v = source["raw"][i]
            assert fine["raw"][i] == (u - sign * v, u + sign * v)
            lower, upper = sorted(fine["final"][i])
            assert ((lower + upper) / 2, (upper - lower) / 2) == source["final"][i]
    assert len(ORIENTATIONS) == 32


@pytest.mark.parametrize("name", ("partial", "saturated", "zero"))
def test_shared_emission_and_scalar_clip_match_frozen_predictions(cases, policy, name):
    source = cases[name]
    fine = _fine_formula(source, (1,) * 5, policy)
    for initial, expected in zip(fine["initial"][0], fine["events"][0]):
        before, after = emission_epi_proposal(
            policy["config"], float(initial), float(policy["boost"])
        )
        assert before == initial
        assert after == pytest.approx(float(expected), abs=TOL, rel=0)
        assert after >= before
    for raw, expected in zip(fine["raw"], fine["final"]):
        actual = tuple(structural_clip(value, **policy["clip"]) for value in raw)
        assert actual == pytest.approx(tuple(map(float, expected)), abs=TOL, rel=0)


@pytest.mark.parametrize("name", ("partial", "saturated", "zero"))
def test_final_radius_obeys_the_default_operator_contraction(cases, policy, name):
    source = cases[name]
    fine = _fine_formula(source, (-1, 1, -1, 1, -1), policy)
    coefficients = (Q(1), 1 - policy["m"], Q(1), Q(1), Q(1))
    factors = (
        1 - policy["z"],
        1 - policy["m"] - policy["z"],
        *((1 - policy["z"] * policy["h"],) * 3),
    )
    uniform = 1 - policy["z"] * policy["h"]
    assert 0 < max(factors) == uniform < 1
    for i, (a, h, factor) in enumerate(zip(coefficients, source["held"], factors)):
        s, t = source["radii"][i], source["events"][i][1]
        radius = abs(fine["final"][i][1] - fine["final"][i][0]) / 2
        assert 0 <= t <= a * s
        assert radius <= abs(t - policy["z"] * h * s)
        assert (
            abs(t - policy["z"] * h * s)
            <= max(policy["z"] * h, abs(a - policy["z"] * h)) * s
        )
        assert radius <= factor * s <= uniform * s
        if i:
            # Convex Reception cannot hit a new rail; Coherence leaves form.
            assert t == a * s
            assert radius == factor * s  # These frozen endpoints are interior.


def test_saturation_preserves_incoming_pressure_and_can_reverse_the_pair(cases, policy):
    source = cases["saturated"]
    fine = _fine_formula(source, (1,) * 5, policy)
    s = source["radii"][0]
    assert fine["events"][0] == (1, 1)
    assert source["events"][0] == (1, 0)
    assert source["raw"][0][1] == -policy["z"] * s < 0
    assert -1 < fine["final"][0][1] < fine["final"][0][0] < 1
    assert source["final"][0][1] == policy["z"] * s > 0

    # Dropping the incoming contrast after event saturation predicts equal
    # endpoints. Refreshing pressure on the equal event pair also does so.
    discarded = _clip(source["raw"][0][0])
    neighbors = fine["events"][4] + fine["events"][1]
    refreshed = policy["k"] * (sum(neighbors) / 4 - 1) + policy["phase"] * (
        _neighbor_mean(PHASES, 0) - PHASES[0]
    )
    refreshed_endpoint = _clip(1 + policy["delta"] * refreshed)
    assert fine["final"][0] != (discarded, discarded)
    assert fine["final"][0] != (refreshed_endpoint, refreshed_endpoint)


@pytest.mark.parametrize("rail", (-1, 1), ids=("lower", "upper"))
def test_held_coherence_flow_clips_one_constituent_and_corrects_the_mean(policy, rail):
    # Separately frozen boundary cases: no change to the A/B/C sources.
    radius = Q(1, 1024)
    means = (rail * (1 - radius),) * 5
    radii = (Q(0), Q(0), radius, Q(0), Q(0))
    phases = (Q(1), Q(1), 1 - Q(rail, 16), Q(1), Q(1))
    source = _quotient_formula(means, radii, phases, policy)
    u, v = source["raw"][2]
    final_mean, final_radius = source["final"][2]
    assert source["pressure"][2] == rail * policy["phase"] / 16
    assert v == (1 - policy["z"] * policy["h"]) * radius > 0
    if rail == -1:
        assert u - v < -1 < u + v < 1
        assert final_mean == (-1 + u + v) / 2
    else:
        assert -1 < u - v < 1 < u + v
        assert final_mean == (u - v + 1) / 2
    assert rail * (final_mean - u) < 0
    assert 0 < final_radius < v < radius
    for orientation in ORIENTATIONS:
        fine = _fine_formula(source, orientation, policy)
        for pair, expected in zip(fine["final"], source["final"]):
            assert (sum(pair) / 2, abs(pair[1] - pair[0]) / 2) == expected
    actual = tuple(structural_clip(value, **policy["clip"]) for value in (u - v, u + v))
    assert actual == pytest.approx(
        (float(final_mean - final_radius), float(final_mean + final_radius)),
        abs=TOL,
        rel=0,
    )


def test_ordinary_coarse_state_misses_partial_clipping_but_zero_contrast_agrees(
    cases, policy
):
    source = cases["partial"]
    coarse = list(source["means"])
    coarse[0] = _clip(coarse[0] + policy["boost"])
    coarse[1] = _clip(
        (1 - policy["m"]) * coarse[1] + policy["m"] * (coarse[0] + coarse[2]) / 2
    )
    coarse_final = tuple(
        _clip(y + policy["delta"] * h * b)
        for y, h, b in zip(coarse, source["held"], source["pressure"])
    )
    assert source["events"][0][0] < coarse[0]
    event_gap = coarse[0] - source["events"][0][0]
    assert coarse[1] - source["events"][1][0] == policy["m"] * event_gap / 2
    assert source["final"][0][0] != coarse_final[0]
    assert tuple(mean for mean, _ in cases["zero"]["final"]) == coarse_final
    assert all(radius == 0 for _, radius in cases["zero"]["final"])


def test_same_mean_boundary_family_requires_radius_and_live_event_order(policy):
    # Independently declared conditional boundary family; no acquisition claim.
    mean = 1 - policy["boost"]
    epsilon = policy["boost"] / 4
    coarse = _quotient_formula((mean,) * 5, (Q(0),) * 5, (Q(1),) * 5, policy)
    spread = _quotient_formula(
        (mean,) * 5, (epsilon, Q(0), Q(0), Q(0), Q(0)), (Q(1),) * 5, policy
    )
    assert all(b == 0 for b in coarse["pressure"] + spread["pressure"])
    assert -1 < mean - epsilon < mean + epsilon < 1
    for orientation in ORIENTATIONS:
        fine = _fine_formula(spread, orientation, policy)
        endpoints = fine["final"][0]
        assert sum(endpoints) / 2 == 1 - epsilon / 2
        assert abs(endpoints[1] - endpoints[0]) / 2 == (Q(1, 2) - policy["z"]) * epsilon
        assert (
            sum(fine["final"][1]) / 2
            == coarse["final"][1][0] - policy["m"] * epsilon / 4
        )
    assert coarse["final"][0] == (1, 0)
    assert spread["final"][0][0] < coarse["final"][0][0]
