"""An accepted adjacent Resonance prefix changes a later amplification gate.

Freeze doubled C5, order 0..9, seed 17, t=1, dt=1/16, default laws, strict
grammar, supplied AL/IL/EN context and zero adaptation counters. Block forms
are (q,q,-q,-1/2,-q), q=15/128; capacities are (1/4,1/4,1,1,1), phases zero.
The exact one-parameter family q in [7/64,1/8] admits the same fresh native
RA/RA/IL/IL/IL block requests with strict selector and identity margins.
The exact family retains cancellation premises. A separate analytical
corollary bounds independent form-only perturbations while all other state
and laws stay fixed; it adds no reserved native responses.

Both RA blocks initially have compatible-neighbor form mean zero. The first
block mixes its form without a capacity boost. Its accepted prefix makes the
later block's live mean -r*q/2, activating amplification that a pre-batch
snapshot would miss. All ideal-real forecasts and both scalar/NumPy source
configurations precede the only two reserved native calls. Fraction algebra
retains represented defaults; finite observations use an absolute 1e-14
budget, not a universal binary64 certificate. No order/source sweep occurs.
"""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.physics import test_native_collective_step as base
from tests.physics import test_native_resonance_u3_boundary as boundary
from tnfr.constants import DEFAULTS
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, U2_DEBT_KEY

SOURCE_Q = Q(15, 128)
FAMILY = Q(7, 64), Q(1, 8)
CAPACITY = (Q(1, 4), Q(1, 4), Q(1), Q(1), Q(1))
MARKS = tuple((node, "RA" if node < 4 else "IL") for node in range(10))
ABS_ERROR = Q(1, 10**14)


def _source():
    graph = base._source(
        tuple(map(float, (SOURCE_Q, SOURCE_Q, -SOURCE_Q, Q(-1, 2), -SOURCE_Q))),
        base.PROFILES[0],
    )
    for node in graph:
        graph.nodes[node]["nu_f"] = float(CAPACITY[node // 2])
    return graph


def _coefficients(graph):
    k, b = (base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    factors = graph.graph["GLYPH_FACTORS"]
    r, a, h = (
        Q(factors[name])
        for name in ("RA_epi_diff", "RA_vf_amplification", "IL_dnfr_factor")
    )
    return k, b, alpha, beta, gamma, r, a, h


def _family_certificate(graph):
    """Bound the exact cancellation family without evaluating native responses."""
    k, b, alpha, beta, gamma, r, a, h = _coefficients(graph)
    lower, upper = FAMILY
    z_lower, z_upper = 3 * b / 8 - k * upper, 3 * b / 8 - k * lower
    peak_lower, peak_upper = k * (Q(1, 2) - upper), k * (Q(1, 2) - lower)
    z_maximum = max(abs(z_lower), abs(z_upper))
    ratio_maximum = z_maximum / peak_lower
    assert 0 < peak_lower <= peak_upper and ratio_maximum < Q(27, 1000)
    # p2=p4=-peak/2-z stays negative and smaller in magnitude than peak.
    assert z_maximum < peak_lower / 2
    ra_si_lower = alpha / 4 + beta + gamma * (1 - ratio_maximum)
    ra_si_upper = alpha / 4 + beta + gamma
    ordinary_si_lower = alpha + beta
    limits = graph.graph["SELECTOR_THRESHOLDS"]
    assert Q(limits["si_lo"]) < ra_si_lower <= ra_si_upper < Q(limits["si_hi"])
    assert ratio_maximum < Q(limits["dnfr_hi"])
    assert ordinary_si_lower > Q(limits["si_hi"])
    assert Q(429, 1000) < ra_si_lower < ra_si_upper < Q(432, 1000)
    assert 0 < r < 1 and 0 < a and h > 0
    # Exactly zero activates no first-block boost; the later live mean is
    # uniformly separated from the shared runtime amplification trigger.
    assert r * lower / 2 > Q(8, 1000) > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
    assert 1 - r - r**2 / 2 > 0
    assert (
        h * (peak_lower / 2 - z_maximum) > Q(1, 50) > Q(graph.graph["EPS_DNFR_STABLE"])
    )
    # Upper-bound every ideal form displacement in this one-call family;
    # all event forms have magnitude <=1/2, and no scalar clipping is needed.
    capacity_maximum = max(Q(1), (1 + a) / 4)
    held_maximum = max(z_maximum, h * peak_upper)
    assert Q(1, 2) + Q(base.DT) * capacity_maximum * held_maximum < Q(3, 5)
    assert Q(graph.graph["VF_MIN"]) < Q(1, 4) < (1 + a) / 4 < Q(graph.graph["VF_MAX"])
    # Bound the nominal ordered-minus-stale form gap throughout the family.
    # Its rate component may vanish where z=0; the form obstruction persists.
    assert -lower * r**2 / 2 + Q(base.DT) * a * z_maximum / 4 < -Q(1, 1000)
    return {
        "z": (z_lower, z_upper),
        "peak": (peak_lower, peak_upper),
        "ratio_ceiling": ratio_maximum,
        "ra_si": (ra_si_lower, ra_si_upper),
        "ordinary_si_floor": ordinary_si_lower,
    }


def _perturbation_certificate(graph):
    """Certify a form-only neighborhood without new runtime responses.

    The pure form pressure row has l1 norm 2*k. Every ordered, admitted RA
    form row is convex, including its live neighbor rows. Thus a sup-norm
    input error epsilon remains <=epsilon through every event prefix, and
    changes either live neighbor mean by at most epsilon. The inequalities
    below admit those same operator/capacity branches before this affine
    bound is used for the successor; no circular branch assumption is made.
    """
    family = _family_certificate(graph)
    k, _, alpha, beta, gamma, r, a, h = _coefficients(graph)
    threshold = Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
    epsilon, dt = threshold / 2, Q(base.DT)
    eta = 2 * k * epsilon
    z_maximum = max(map(abs, family["z"]))
    peak_lower = family["peak"][0]
    assert 0 < eta < peak_lower
    ratio_maximum = (z_maximum + eta) / (peak_lower - eta)
    limits = graph.graph["SELECTOR_THRESHOLDS"]
    assert ratio_maximum < Q(27, 1000) < Q(limits["dnfr_hi"])
    assert Q(limits["si_lo"]) < alpha / 4 + beta + gamma * (1 - ratio_maximum)
    assert alpha / 4 + beta + gamma < Q(limits["si_hi"])
    assert alpha + beta > Q(limits["si_hi"])
    # Initial compatible means need not remain zero: their errors still
    # lie strictly below the first-block trigger. The later live means
    # remain outside it, even after all accepted prefix perturbations.
    assert epsilon < threshold < r * FAMILY[0] / 2 - epsilon
    assert (1 - r - r**2 / 2) * FAMILY[0] > epsilon
    assert Q(1, 2) + epsilon < 1
    assert h * (peak_lower / 2 - z_maximum - eta) > Q(1, 50)
    assert h <= 1 and (1 + a) / 4 <= 1
    form_radius = epsilon + dt * eta
    assert form_radius == (1 + 2 * dt * k) * epsilon < 41 * epsilon / 40
    assert Q(3, 5) + form_radius < 1
    return {
        "epsilon": epsilon,
        "pressure_radius": eta,
        "ratio_ceiling": ratio_maximum,
        "form_radius": form_radius,
    }


def test_form_only_perturbation_certificate_retains_live_capacity_branches():
    """Check exact row identities independently of the hand-derived recurrence."""
    graph = _source()
    certificate = _perturbation_certificate(graph)
    k, _, _, _, _, r, _, _ = _coefficients(graph)
    size = len(graph)
    identity = [[Q(int(i == j)) for j in graph] for i in graph]
    rows = [row[:] for row in identity]
    # These exact affine coordinates represent every q in the declared family,
    # not extra evaluated preparations or sampled perturbation directions.
    q_coefficients = (1, 1, 1, 1, -1, -1, 0, 0, -1, -1)
    constants = (Q(0),) * 6 + (Q(-1, 2),) * 2 + (Q(0),) * 2

    def evaluate(row, coordinate):
        return sum(
            weight * value for weight, value in zip(row, coordinate, strict=True)
        )

    for node in graph:
        neighbors = tuple(graph.neighbors(node))
        assert len(neighbors) == 4
        pressure_row = [k * (Q(int(j in neighbors), 4) - int(j == node)) for j in graph]
        assert sum(pressure_row) == 0 and sum(map(abs, pressure_row)) == 2 * k
    for node in range(4):
        neighbors = tuple(graph.neighbors(node))
        neighbor_row = [sum(rows[n][j] for n in neighbors) / 4 for j in range(size)]
        assert all(value >= 0 for value in neighbor_row) and sum(neighbor_row) == 1
        assert evaluate(neighbor_row, constants) == 0
        assert evaluate(neighbor_row, q_coefficients) == (Q(0) if node < 2 else -r / 2)
        rows[node] = [
            (1 - r) * own + r * neighbor
            for own, neighbor in zip(rows[node], neighbor_row, strict=True)
        ]
        assert all(value >= 0 for row in rows for value in row)
        assert all(sum(row) == 1 for row in rows)
        assert evaluate(rows[node], constants) == 0
        assert evaluate(rows[node], q_coefficients) == (
            1 - r if node < 2 else 1 - r - r**2 / 2
        )
    assert certificate["pressure_radius"] == 2 * k * certificate["epsilon"]
    assert certificate["form_radius"] > certificate["epsilon"]


def test_pair_constant_phase_interpolation_cannot_lose_admitted_neighbors():
    """Certify ideal affine extrema; this does not evaluate runtime trig.

    At a complete pair boundary, each admitted neighboring block contributes
    two equal phases. For two admitted blocks with defined resultant, their
    common lifted mean has offset (a+b)/2. Normalize the U3 radius to one:
    all remaining tests are scale-free exact rational identities. Opposite
    boundary corners also test the affine extension; if the actual circular
    mean is undefined, the declared phase-hold branch preserves admission
    directly instead of evaluating that extension.
    """
    default_coupling = Q(DEFAULTS["GLYPH_FACTORS"]["RA_phase_coupling"])
    for coupling in (Q(0), default_coupling, Q(1)):
        assert 0 <= coupling <= 1

        def residuals(a, b):
            shift = coupling * (a + b) / 2
            return a - shift, b - shift

        # Recover each affine row from independent coordinate-basis inputs.
        columns = residuals(Q(1), Q(0)), residuals(Q(0), Q(1))
        rows = tuple(zip(*columns, strict=True))
        assert residuals(Q(0), Q(0)) == (0, 0)
        assert all(sum(map(abs, row)) == 1 for row in rows)
        # Include both the complete square and asymmetric interior rectangles.
        # Their corner extrema certify every enclosed offset by affinity.
        for mu_a, mu_b in ((Q(0), Q(0)), (Q(1, 4), Q(3, 8))):
            radius_a, radius_b = 1 - mu_a, 1 - mu_b
            corners = tuple(
                (sign_a * radius_a, sign_b * radius_b)
                for sign_a in (-1, 1)
                for sign_b in (-1, 1)
            )
            outputs = tuple(residuals(a, b) for a, b in corners)
            for point, actual in zip(corners, outputs, strict=True):
                assert actual == tuple(
                    sum(
                        weight * value for weight, value in zip(row, point, strict=True)
                    )
                    for row in rows
                )
            clearances = tuple(
                1 - max(abs(value[i]) for value in outputs) for i in (0, 1)
            )
            assert clearances == (
                (1 - coupling / 2) * mu_a + coupling * mu_b / 2,
                coupling * mu_a / 2 + (1 - coupling / 2) * mu_b,
            )
            assert min(clearances) >= 0
        # One admitted block has its own phase as the mean. Its scalar row
        # contracts, including the zero/complete interpolation endpoints.
        for clearance in (Q(0), Q(1, 4)):
            radius = 1 - clearance
            endpoints = tuple(
                offset - coupling * offset for offset in (-radius, radius)
            )
            assert 1 - max(map(abs, endpoints)) == coupling + (1 - coupling) * clearance
            assert max(map(abs, endpoints)) <= radius


def _forecast(graph):
    """Predict the ordered prefix and full native row by rational identities."""
    k, b, alpha, beta, gamma, r, a, h = _coefficients(graph)
    q, dt = SOURCE_Q, Q(base.DT)
    z, peak = 3 * b / 8 - k * q, k * (Q(1, 2) - q)
    pressure = (z, z, -peak / 2 - z, peak, -peak / 2 - z)
    assert peak == max(map(abs, pressure)) and z > 0
    si = tuple(
        alpha * capacity + beta + gamma * (1 - abs(p) / peak)
        for capacity, p in zip(CAPACITY, pressure, strict=True)
    )
    event_form = ((1 - r) * q, (1 - r - r**2 / 2) * q, -q, Q(-1, 2), -q)
    capacity = (Q(1, 4), (1 + a) / 4, Q(1), Q(1), Q(1))
    held = pressure[:2] + tuple(h * p for p in pressure[2:])
    rate = tuple(v * p for v, p in zip(capacity, held, strict=True))
    form = tuple(
        x + dt * velocity for x, velocity in zip(event_form, rate, strict=True)
    )
    stale_form = (1 - r) * q + dt * z / 4
    stale_rate = z / 4
    assert event_form[0] > event_form[1] > 0
    assert all(Q(graph.graph["EPI_MIN"]) < x < Q(graph.graph["EPI_MAX"]) for x in form)
    thresholds = graph.graph["SELECTOR_THRESHOLDS"]
    counters = tuple(
        int(abs(p) <= Q(graph.graph["EPS_DNFR_STABLE"]) and s >= Q(thresholds["si_hi"]))
        for p, s in zip(held, si, strict=True)
    )
    assert counters == (0,) * 5
    # The nominal first two pressures are small enough; their independently
    # consumed fresh Si nevertheless prevents adaptation qualification.
    assert abs(z) < Q(graph.graph["EPS_DNFR_STABLE"])
    assert max(si[:2]) < Q(thresholds["si_hi"])
    policy = graph.graph["PHASE_ADAPT"]
    gains = tuple(
        Q(graph.graph[f"PHASE_K_{name}"])
        + Q(policy["down"])
        * (Q(policy[f"k{suffix}_min"]) - Q(graph.graph[f"PHASE_K_{name}"]))
        for name, suffix in (("GLOBAL", "G"), ("LOCAL", "L"))
    )
    assert 1 > Q(policy["R_hi"]) and 0 <= Q(policy["disr_lo"])
    return {
        "initial_pressure": pressure,
        "peak": peak,
        "si": si,
        "event_form": event_form,
        "capacity": capacity,
        "pressure": held,
        "rate": rate,
        "form": form,
        "phase": (Q(0),) * 5,
        "acceleration": tuple(
            (value - Q(old)) / dt for value, old in zip(rate, base.RATES, strict=True)
        ),
        "later_mean": -r * q / 2,
        "stale_form": stale_form,
        "stale_rate": stale_rate,
        "form_gap": -(r**2) * q / 2 + dt * a * z / 4,
        "rate_gap": a * z / 4,
        "gains": gains,
    }


def _within(actual, expected, *, allowance=ABS_ERROR):
    assert abs(Q(actual) - expected) <= allowance


def _pair_values(values):
    """Retain exact oracle fractions when lifting a block-constant prediction."""
    return tuple(value for value in values for _ in range(2))


@pytest.fixture(scope="module")
def cases():
    # Both backends and every forecast are frozen before the first response.
    frozen = []
    for scalar in (False, True):
        graph = _source()
        configuration = deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key
                in set(DEFAULTS) | {"GRAMMAR_REJECTION_MODE", "compute_delta_nfr"}
            }
        )
        frozen.append(
            (graph, configuration, scalar, _family_certificate(graph), _forecast(graph))
        )
    result = {}
    for graph, configuration, scalar, certificate, expected in frozen:
        case = boundary._execute(graph, configuration, scalar=scalar)
        case.update(certificate=certificate, expected=expected)
        result[scalar] = case
    return result


@pytest.mark.parametrize("scalar", (False, True))
def test_family_bounds_admit_the_actual_preselected_adjacent_resonance_batch(
    cases, scalar
):
    case, expected = cases[scalar], cases[scalar]["expected"]
    assert case["error"] is None and case["choices"] == MARKS
    assert case["prepared"] == (True,) * 10 and case["selector_cleared"] is True
    for event, channel, forecast in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
    ):
        for actual, predicted in zip(
            base._event(case, event)["after"][channel],
            _pair_values(expected[forecast]),
            strict=True,
        ):
            _within(actual, predicted)
    certificate = case["certificate"]
    assert certificate["z"][0] <= expected["initial_pressure"][0] <= certificate["z"][1]
    assert certificate["peak"][0] <= expected["peak"] <= certificate["peak"][1]
    assert (
        certificate["ra_si"][0]
        <= min(expected["si"][:2])
        <= max(expected["si"][:2])
        <= certificate["ra_si"][1]
    )
    assert min(expected["si"][2:]) >= certificate["ordinary_si_floor"]


@pytest.mark.parametrize("scalar", (False, True))
def test_later_resonance_uses_the_accepted_prefix_for_its_capacity_trigger(
    cases, scalar
):
    case, expected = cases[scalar], cases[scalar]["expected"]
    assert tuple(node for node, _, _, _ in case["u3"]) == (0, 1, 2, 3)
    assert tuple((node, glyph) for node, glyph, _, _ in case["primitives"]) == MARKS
    for node, code, selection, neighbors in case["u3"]:
        assert code == "RA" and neighbors == tuple(case["graph"].neighbors(node))
        assert selection.target_phase == 0 and selection.phases == (0.0,) * 4
        before, after = case["primitives"][node][2:]
        initial_mean = sum(Q(case["initial"]["form"][n]) for n in neighbors) / 4
        live_mean = sum(Q(before["form"][n]) for n in neighbors) / 4
        assert initial_mean == 0
        _within(live_mean, Q(0) if node < 2 else expected["later_mean"])
        assert before["form"][node] > 0 and after["form"][node] > 0
        _within(after["form"][node], expected["event_form"][node // 2])
        _within(after["capacity"][node], expected["capacity"][node // 2])
        assert (after["capacity"][node] > before["capacity"][node]) is (node >= 2)
        assert before["pressure"] == after["pressure"]
        assert before["phase"] == after["phase"]
    # The later target consumes changed constituent forms, not an edited
    # prepared request or a pressure refresh between operators.
    assert case["primitives"][2][2]["form"][:2] != case["initial"]["form"][:2]
    assert (
        case["primitives"][2][2]["pressure"]
        == base._event(case, "pressure")["after"]["pressure"]
    )


@pytest.mark.parametrize("scalar", (False, True))
def test_complete_native_successor_retains_the_ordered_amplification_effect(
    cases, scalar
):
    case, expected = cases[scalar], cases[scalar]["expected"]
    for name in ("form", "capacity", "pressure", "rate", "acceleration", "si", "phase"):
        for actual, predicted in zip(
            case["final"][name], _pair_values(expected[name]), strict=True
        ):
            _within(actual, predicted)
    for actual, predicted in zip(
        base._event(case, "glyphs")["after"]["form"],
        _pair_values(expected["event_form"]),
        strict=True,
    ):
        _within(actual, predicted)
    integration = base._event(case, "integration")
    assert integration["before"]["capacity"] == case["final"]["capacity"]
    assert integration["before"]["pressure"] == integration["after"]["pressure"]
    for node, _, before, after in case["primitives"]:
        assert before["edges"] == after["edges"]
        for name in ("form", "phase", "capacity", "pressure"):
            assert tuple(before[name][n] for n in case["graph"] if n != node) == tuple(
                after[name][n] for n in case["graph"] if n != node
            )


@pytest.mark.parametrize("scalar", (False, True))
def test_prebatch_capacity_prediction_fails_by_the_frozen_nonzero_amount(cases, scalar):
    case, expected = cases[scalar], cases[scalar]["expected"]
    assert expected["rate_gap"] > Q(1, 10**6)
    assert expected["form_gap"] < -Q(1, 1000)
    for node in (2, 3):
        _within(
            Q(case["final"]["rate"][node]) - expected["stale_rate"],
            expected["rate_gap"],
        )
        _within(
            Q(case["final"]["form"][node]) - expected["stale_form"],
            expected["form_gap"],
        )
        assert case["final"]["capacity"][node] > case["initial"]["capacity"][node]
    for node in (0, 1):
        assert case["final"]["capacity"][node] == case["initial"]["capacity"][node]
        _within(case["final"]["form"][node], expected["stale_form"])


@pytest.mark.parametrize("scalar", (False, True))
def test_phase_and_adaptation_keep_their_own_native_predicates(cases, scalar):
    case, expected = cases[scalar], cases[scalar]["expected"]
    history = case["graph"].graph["history"]
    assert tuple(history["phase_state"]) == ("stable",)
    assert tuple(history["phase_R"]) == (1.0,) and tuple(history["phase_disr"]) == (
        0.0,
    )
    for suffix, gain in zip(("G", "L"), expected["gains"], strict=True):
        _within(history[f"phase_k{suffix}"][0], gain)
    assert case["initial"]["phase"] == case["final"]["phase"] == (0.0,) * 10
    assert case["final"]["counters"] == (0,) * 10
    assert not any(
        event["name"] in {"capacity_write", "network_remesh"}
        for event in case["events"]
    )
    assert (
        base._event(case, "capacity")["before"]["si"]
        == base._event(case, "si")["after"]["si"]
    )
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    for node in (0, 1, 2, 3):
        assert abs(case["final"]["pressure"][node]) < DEFAULTS["EPS_DNFR_STABLE"]
        assert case["final"]["si"][node] < DEFAULTS["SELECTOR_THRESHOLDS"]["si_hi"]


@pytest.mark.parametrize("scalar", (False, True))
def test_supplied_context_defaults_and_order_are_preserved(cases, scalar):
    case, graph = cases[scalar], cases[scalar]["graph"]
    for name in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "SELECTOR_THRESHOLDS",
        "GLYPH_FACTORS",
        "PHASE_ADAPT",
        "VF_ADAPT_TAU",
        "DT",
        "DT_MIN",
        "INTEGRATOR_METHOD",
        "GAMMA",
    ):
        assert case["configuration"][name] == DEFAULTS[name]
    assert case["configuration"]["GRAMMAR_REJECTION_MODE"] == "raise"
    assert case["configuration"]["GRAMMAR_CANON"]["enabled"] is True
    assert case["configuration"]["compute_delta_nfr"] is base.default_compute_delta_nfr
    assert base._event(case, "pressure")["result"] is base.default_compute_delta_nfr
    assert "glyph_selector" not in graph.graph and "integrator" not in graph.graph
    assert tuple(graph) == tuple(range(10)) and graph.graph["RANDOM_SEED"] == 17
    assert (
        case["initial"]["temporal"] == ((),) * 10
        and case["initial"]["counters"] == (0,) * 10
    )
    assert case["initial"]["ordinal"] is None and case["final"]["ordinal"] == 1
    assert case["final"]["time"] == 1 + base.DT
    assert (
        case["final"]["edges"] == case["initial"]["edges"]
        and len(case["final"]["edges"]) == 20
    )
    history = graph.graph["history"]
    assert not history.get("stable_frac") and "_REMESH_META" not in graph.graph
    assert "ra_metrics" not in graph.graph and "_ra_c_tracking" not in graph.graph
    for node, data in graph.nodes(data=True):
        assert data[OPERATOR_STEP_KEY] == 4
        # RA is neutral for U2: the supplied debt at block one is retained.
        assert data[U2_DEBT_KEY] == int(node // 2 == 1)
        assert data["epi_kind"] == "seed" and tuple(data["glyph_history"]) == (
            "EN",
            MARKS[node][1],
        )
        assert history["since_AL"][node] == history["since_EN"][node] == 1
        assert case["final"]["temporal"][node] == (
            (1.0, base._event(case, "integration")["before"]["form"][node]),
            (1 + base.DT, case["final"]["form"][node]),
        )
