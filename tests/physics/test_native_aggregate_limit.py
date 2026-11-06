"""Exact deficit comparisons for the hard-clipped aggregate primary limit.

These are scalar comparison bounds, finite algebraic ledgers and geometric
sums, not iterated native responses. Individual deficits retain the fine form
coordinates; no mean-projection error or extra spread forcing is introduced.
The clipping, horizon and one-call transition owners retain their evidence.
No frozen producer is imported or replayed.
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


@pytest.fixture(scope="module", autouse=True)
def no_native_evaluations():
    def forbidden(*args, **kwargs):
        pytest.fail("Limit controls are static; native/flow execution is forbidden")

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
    lower, upper = Q(lo), Q(hi)
    width = upper - lower
    k = Q(weights["epi"])
    z = dt * nu * k
    lam = 1 - m - z
    a = max(1 - boost / width, z)
    contraction = 1 - (1 - a) * lam**5
    rho = 1 - Q(graph.graph["PHASE_ADAPT"]["kG_min"])
    forcing = dt * nu * Q(weights["phase"]) * Q(1, 8) / Q(math.pi)
    assert mode == "hard" and graph.graph["AL_MAX_LAG"] + 1 == 6
    assert 0 < boost < width and 0 < lam < 1 and 0 < h < 1
    assert 0 < a < 1 and 0 < contraction < 1 and 0 < rho < 1
    return dict(
        lower=lower,
        upper=upper,
        width=width,
        clip=dict(lo=lo, hi=hi, mode=mode, k=steepness, record_stats=False),
        m=m,
        h=h,
        boost=boost,
        k=k,
        phase=Q(weights["phase"]) / Q(math.pi),
        delta=dt * nu,
        z=z,
        z_max=dt * Q(graph.graph["VF_MAX"]) * k,
        lam=lam,
        a=a,
        contraction=contraction,
        rho=rho,
        forcing=forcing,
    )


def _deficit_clip(value, width):
    return min(width, max(Q(0), value))


def _grid(policy):
    boost = policy["boost"]
    return boost / 2, boost, 2 * boost, policy["width"]


@pytest.mark.parametrize("mark", ("AL", "EN", "IL"))
def test_fine_deficit_rows_obey_the_frozen_comparison_grid(policy, mark):
    b, m, h, z, forcing = (policy[key] for key in ("boost", "m", "h", "z", "forcing"))
    for maximum in _grid(policy):
        for own, live, incoming, f in product(
            sorted({Q(0), min(b, maximum), maximum}),
            (Q(0), maximum),
            (Q(0), maximum),
            (-forcing, Q(0), forcing),
        ):
            # Independent live/incoming neighbor bounds form a relaxation;
            # not every grid combination is a complete native preparation.
            if mark == "AL":
                event, held = max(Q(0), own - b), Q(1)
                bound = max(maximum - b, z * maximum) + forcing
                affine = policy["a"] * maximum + forcing
            elif mark == "EN":
                event, held = (1 - m) * own + m * live, Q(1)
                bound = policy["lam"] * own + (m + z) * maximum + forcing
                affine = maximum + forcing
            else:
                event, held = own, h
                bound = (1 - z * h) * own + z * h * maximum + forcing
                affine = maximum + forcing
            raw = event - z * held * own + z * held * incoming - f
            actual = _deficit_clip(raw, policy["width"])
            assert actual <= bound <= affine
            # The implemented form projection is the same boundary contract
            # under y=U-x, up to materialization of this exact rational input.
            shared = policy["upper"] - Q(
                structural_clip(policy["upper"] - raw, **policy["clip"])
            )
            assert float(shared) == pytest.approx(float(actual), abs=2e-15, rel=0)


def test_zero_forcing_emission_extremizers_make_both_deficit_branches_necessary(policy):
    z, boost = policy["z"], policy["boost"]
    for maximum in _grid(policy):
        saturated = _deficit_clip(z * maximum, policy["width"])
        unsaturated = _deficit_clip(max(Q(0), maximum - boost), policy["width"])
        # These are respectively y=0,Nold=D and y=D,Nold=D.
        sharp = max(saturated, unsaturated)
        assert sharp == max(z * maximum, maximum - boost)
        assert sharp <= policy["a"] * maximum
        if maximum == boost / 2:
            assert saturated > unsaturated
        if maximum == policy["width"]:
            assert unsaturated > saturated
            assert sharp == policy["a"] * maximum


def test_saturated_emission_followed_by_held_flow_is_not_order_preserving(policy):
    upper, boost, z = (policy[key] for key in ("upper", "boost", "z"))
    low, high = upper - boost / 2, upper - boost / 4
    # Neighbor forms are zero, phases common, spread zero. Both event
    # proposals saturate, but incoming pressure remains proportional to -x.
    assert low < high < upper and low + boost > upper
    low_final = structural_clip(upper - z * low, **policy["clip"])
    high_final = structural_clip(upper - z * high, **policy["clip"])
    assert high_final < low_final
    exact_low, exact_high = upper - z * low, upper - z * high
    assert exact_high - exact_low == -z * boost / 4 < 0
    assert low_final == pytest.approx(float(exact_low), abs=2e-15, rel=0)
    assert high_final == pytest.approx(float(exact_high), abs=2e-15, rel=0)


def test_default_capacity_interval_retains_a_strict_six_call_comparison_factor(policy):
    # Endpoint inequalities cover every 0<nu<=VF_MAX. They do not certify
    # native admission at nu=0 or give a capacity-independent phase clock.
    z_max, m, boost = policy["z_max"], policy["m"], policy["boost"]
    assert 0 < z_max < 1 - m < 1
    assert z_max < 1 - boost / policy["width"] < 1
    a = 1 - boost / policy["width"]
    endpoint_factors = tuple(1 - (1 - a) * (1 - m - z) ** 5 for z in (Q(0), z_max))
    assert 0 < endpoint_factors[0] <= policy["contraction"] <= endpoint_factors[1] < 1


def test_six_call_comparison_ledger_covers_every_emission_position_and_suffix(policy):
    forcing = tuple(policy["forcing"] * policy["rho"] ** n for n in range(6))
    for initial in _grid(policy):
        for emission_at in range(6):
            for suffix in product(("EN", "IL"), repeat=5 - emission_at):
                # This ledger propagates upper bounds, not states or events.
                # Before the chosen Emission only the common envelope is used.
                envelope = initial + sum(forcing[:emission_at], Q(0))
                target = policy["a"] * envelope + forcing[emission_at]
                envelope += forcing[emission_at]
                gap = envelope - target
                assert gap >= (1 - policy["a"]) * initial
                for n, mark in enumerate(suffix, start=emission_at + 1):
                    retention = (
                        policy["lam"] if mark == "EN" else 1 - policy["z"] * policy["h"]
                    )
                    previous_gap = envelope - target
                    target = (
                        retention * target + (1 - retention) * envelope + forcing[n]
                    )
                    envelope += forcing[n]
                    assert envelope - target == retention * previous_gap
                    assert envelope - target >= policy["lam"] * previous_gap
                assert envelope == initial + sum(forcing, Q(0))
                assert (
                    envelope - target
                    >= (1 - policy["a"]) * policy["lam"] ** 5 * initial
                )
                assert target <= policy["contraction"] * initial + sum(forcing, Q(0))


@pytest.mark.parametrize("blocks", (0, 1, 2, 6))
@pytest.mark.parametrize("equal_factors", (False, True))
def test_geometric_forcing_convolution_and_positive_tail(policy, blocks, equal_factors):
    contraction = policy["contraction"]
    decay = contraction if equal_factors else policy["rho"] ** 6
    # Equality is an abstract recurrence specialization, not a retuned policy.
    terms = tuple(contraction ** (blocks - 1 - j) * decay**j for j in range(blocks))
    convolution = sum(terms, Q(0))
    if equal_factors:
        expected = Q(0) if blocks == 0 else blocks * contraction ** (blocks - 1)
    else:
        assert contraction != decay
        expected = (contraction**blocks - decay**blocks) / (contraction - decay)
    assert convolution == expected
    assert 0 <= convolution <= blocks * max(contraction, decay) ** max(0, blocks - 1)

    finite = sum(
        (policy["forcing"] * policy["rho"] ** j for j in range(6 * blocks)), Q(0)
    )
    tail = policy["forcing"] * policy["rho"] ** (6 * blocks) / (1 - policy["rho"])
    assert tail > 0
    assert finite + tail == policy["forcing"] / (1 - policy["rho"])
    # A positive bound is not a nonzero observed force. Summability neither
    # asserts zero force after a finite index nor finite-time stationarity.


@pytest.mark.parametrize("mark", ("AL", "EN", "IL"))
def test_upper_rail_with_common_phase_is_primary_stationary_for_each_event(
    policy, mark
):
    upper, lower = policy["upper"], policy["lower"]
    neighbor_form, own_phase, neighbor_phase = upper, Q(1), Q(1)
    # Both incoming pressure channels vanish independently; capacity and
    # degree contrasts also vanish on the uniform-capacity regular support.
    assert neighbor_form - upper == neighbor_phase - own_phase == 0
    pressure = policy["k"] * (neighbor_form - upper) + policy["phase"] * (
        neighbor_phase - own_phase
    )
    held_pressure = (policy["h"] if mark == "IL" else 1) * pressure
    assert pressure == held_pressure == 0
    if mark == "AL":
        raw_event = upper + policy["boost"]
        assert raw_event > upper
    elif mark == "EN":
        raw_event = (1 - policy["m"]) * upper + policy["m"] * neighbor_form
    else:
        raw_event = upper
    event = structural_clip(raw_event, **policy["clip"])
    final = structural_clip(
        Q(event) + policy["delta"] * held_pressure, **policy["clip"]
    )
    assert lower < event == final == upper
    # Each event preserves this primary slice, hence so does an ordered
    # profile. Grammar ordinals and lag resets still advance: their actual
    # writes belong to test_native_aggregate_transition, not this algebra.
