"""Mutation admission and renewal are separate, analytically guarded gates.

Freeze doubled C5/order0..9, unit support, represented defaults, Gamma0,
dt1/16, t1 and two attempted ideal calls. Initial block form is
(19/20,0,-9/10,9/10,0), with only M0 uncertain by +/-1/4096; capacities are
(27/50,27/50,6,6,27/50), phases (0,3/2,0,3,3/2), internal contrasts zero.
AL/IL/EN is supplied grammar context, not an executed preparation: prior IL,
ordinal3, debt/lags/counts0, empty physical history and old rate/acceleration0.
First Transition twin inputs independently fill the default amplitude box.

The interval forecast precedes eight exact fine controls: the three input
intervals' vertices. The unchanged source generates a positive ideal-flow
secant and admits Mutation, then stops at the following Resonance identity
rejection. No second flow, coordinator, adaptation or Transition is invented.
The no-Mutation policy contrast stops at the same live-state obstruction.

A separate static source admits a renewing conditional phase action but
lacks acquired threshold evidence. It is not joined to the first source.
Histories derived here are ideal execution records, not measured native data.
The shift is the exact represented-factor product f*pi_rep/4; its binary64
realization, signed-zero direction and absolute telemetry are separate.
There is no native/scientific kernel evaluation, source tuning or tolerance.
"""

import math
from fractions import Fraction as Q
from hashlib import sha256
from itertools import combinations, product

import pytest

from tests.physics import test_native_collective_enclosure as enclosure
from tests.physics import test_native_relative_phase_descent as relative
from tnfr.config.constants import DISRUPTORS
from tnfr.config.operator_names import BIFURCATION_WINDOW, U2_DEBT_CAPACITY
from tnfr.mathematics._rational_interval import INTERVAL_METHOD, I
from tnfr.operators import _mutation_gate as mutation_gate
from tnfr.operators import _mutation_stage_kernel as mutation_kernel
from tnfr.operators.grammar_types import (
    BIFURCATION_HANDLERS,
    DESTABILIZERS,
    GLYPH_TO_FUNCTION,
    STABILIZERS,
)
from tnfr.physics import mutation_trigger

domain, scalar = relative.domain, relative.scalar
no_execution = relative.no_execution
calculus_guard = relative.calculus_guard
configuration = relative.configuration
DT, PI = relative.DT, relative.PI
RADIUS = Q(1, 4096)
MEANS = (Q(19, 20), Q(0), Q(-9, 10), Q(9, 10), Q(0))
CAPACITIES = (Q(27, 50), Q(27, 50), Q(6), Q(6), Q(27, 50))
PHASES = (Q(0), Q(3, 2), Q(0), Q(3), Q(3, 2))
CONTEXT = ("AL", "IL", "EN")
VERTICES = tuple(product((-1, 1), repeat=3))
FUNCTIONS = {glyph.value: name for glyph, name in GLYPH_TO_FUNCTION.items()}


@pytest.fixture(scope="module", autouse=True)
def renewal_guard(calculus_guard):
    previous, forbidden = calculus_guard
    additions = (
        (domain.operators, "_op_OZ"),
        (domain.operators, "_op_ZHIR"),
        (mutation_kernel, "propose_mutation_stage"),
        (mutation_kernel, "propose_mutation_network_stage"),
        (mutation_gate, "mutation_trigger_certificate"),
        (mutation_gate, "mutation_threshold_sample"),
        (mutation_gate, "validate_mutation_runtime_gate"),
        (mutation_trigger, "certify_mutation_trigger"),
    )
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in additions:
            patch.setattr(owner, name, forbidden)
        yield (*previous, *additions), forbidden


def _neighbors(node):
    return tuple(
        2 * block + parity
        for block in ((node // 2 - 1) % 5, (node // 2 + 1) % 5)
        for parity in (0, 1)
    )


def _pressure(law, forms, capacities, phases):
    """Fine-node affine pressure, valid on this strict common lift."""
    k, b, w = law["pressure"]
    return tuple(
        k * (sum(forms[j] for j in _neighbors(n)) / 4 - forms[n])
        + b * (sum(capacities[j] for j in _neighbors(n)) / 4 - capacities[n])
        + w * (sum(phases[j] for j in _neighbors(n)) / 4 - phases[n]) / PI
        for n in range(10)
    )


def _prepared(law, forms, capacities, phases):
    pressure = _pressure(law, forms, capacities, phases)
    peak = I(max(abs(p).lo for p in pressure), max(abs(p).hi for p in pressure))
    assert peak.lo > 0
    ratios = tuple(
        I(abs(p).lo / peak.hi, min(Q(1), abs(p).hi / peak.lo)) for p in pressure
    )
    alpha, beta, gamma = law["stability"]
    si = tuple(
        alpha * capacities[n] / max(capacities)
        + beta * (1 - abs(sum(phases[j] for j in _neighbors(n)) / 4 - phases[n]) / PI)
        + gamma * (1 - ratios[n])
        for n in range(10)
    )
    cuts, marks = law["thresholds"], []
    for value, ratio in zip(si, ratios):
        if value.lo >= cuts["si_hi"]:
            mark = "IL"
        elif value.hi <= cuts["si_lo"]:
            mark = (
                "OZ"
                if ratio.lo > cuts["dnfr_hi"]
                else "ZHIR" if ratio.hi <= cuts["dnfr_hi"] else None
            )
        elif value.lo > cuts["si_lo"] and value.hi < cuts["si_hi"]:
            mark = (
                "NAV"
                if ratio.lo > cuts["dnfr_hi"]
                else "RA" if ratio.hi <= cuts["dnfr_hi"] else None
            )
        else:
            mark = None
        assert mark is not None, "unresolved request; retain source, do not tune"
        marks.append(mark)
    return dict(pressure=pressure, peak=peak, ratios=ratios, si=si, marks=tuple(marks))


def _phase_step(law, phases, traces):
    """Exact relative coordinator, using interval bounds for its R branch."""
    assert max(phases) - min(phases) <= 3 < PI
    lower = upper = Q(5)
    for a, b in combinations(phases, 2):
        terms = tuple(
            Q((-1) ** j, math.factorial(2 * j)) * (a - b) ** (2 * j) for j in range(8)
        )
        lower += 2 * sum(terms)
        upper += 2 * sum(terms[:-1])
    bounds = lower / 25, upper / 25
    load = Q(
        sum(mark in DISRUPTORS for trace in traces for mark in trace),
        sum(map(len, traces)),
    )
    cfg = law["phase_policy"]
    if bounds[0] >= cfg["R_hi"] ** 2 and load <= cfg["disr_lo"]:
        branch, targets = "stable", (cfg["kG_min"], cfg["kL_min"])
    elif bounds[1] <= cfg["R_lo"] ** 2 or load >= cfg["disr_hi"]:
        branch, targets = "dissonant", (
            cfg["kG_max"],
            (cfg["kL_min"] + cfg["kL_max"]) / 2,
        )
    elif bounds[0] > cfg["R_lo"] ** 2 and bounds[1] < cfg["R_hi"] ** 2:
        branch, targets = "transition", (
            (cfg["kG_min"] + cfg["kG_max"]) / 2,
            (cfg["kL_min"] + cfg["kL_max"]) / 2,
        )
    else:
        raise AssertionError("unresolved controller branch")
    kg, kl = tuple(
        old + cfg["up" if target > old else "down"] * (target - old)
        for old, target in zip(law["phase_gains"], targets)
    )
    assert 0 < kg and 0 < kl and kg + kl < 1
    coordinate = tuple(
        (1 - kg - kl) * value + kl * domain._neighbor(phases, i)
        for i, value in enumerate(phases)
    )
    return dict(
        relative=relative._relative(coordinate),
        gains=(kg, kl),
        branch=branch,
        r2=bounds,
        load=load,
    )


def _grammar(mark, trace, prior, debt):
    """Only the named trigger obligations used by this fixed protocol."""
    name = FUNCTIONS[mark]
    if name in DESTABILIZERS and debt + 1 > U2_DEBT_CAPACITY:
        return "U2"
    if mark in {"OZ", "ZHIR"} and not any(
        FUNCTIONS[item] in BIFURCATION_HANDLERS for item in trace[-6:]
    ):
        return "U4a"
    if mark == "ZHIR" and (
        not prior
        or not any(
            FUNCTIONS[item] in DESTABILIZERS for item in trace[-BIFURCATION_WINDOW:]
        )
    ):
        return "U4b"
    return None


def _temporal_reason(history, current, time, xi):
    """Exact associated ideal record; no endpoint is reconstructed from p."""
    if history is None or len(history) < 2:
        return "missing_history"
    (t0, x0), (t1, x1) = history[-2:]
    if t1 != time or x1 != current or t1 <= t0:
        return "stale_or_unassociated_history"
    return None if (x1 - x0) / (t1 - t0) > xi else "threshold_not_crossed"


def _prefix_bookkeeping(first_marks, committed):
    """Record the complete first batch and the accepted second-call prefix."""
    traces = [list(CONTEXT) for _ in first_marks]
    ordinals, debt, lag_al, lag_en, draws = ([0] * len(first_marks) for _ in range(5))
    for n in range(len(first_marks)):
        ordinals[n] = len(CONTEXT)
    batches = (tuple(enumerate(first_marks)), tuple((n, "ZHIR") for n in committed))
    for events in batches:
        for n in range(len(first_marks)):
            lag_al[n] += 1
            lag_en[n] += 1
        for n, mark in events:
            traces[n].append(mark)
            ordinals[n] += 1
            name = FUNCTIONS[mark]
            if name in DESTABILIZERS:
                debt[n] += 1
            elif name in STABILIZERS:
                debt[n] = max(0, debt[n] - 1)
            draws[n] += int(mark == "NAV")
    assert lag_al == lag_en
    return dict(
        traces=tuple(map(tuple, traces)),
        ordinals=tuple(ordinals),
        debt=tuple(debt),
        lags=tuple(lag_al),
        rng_progress=tuple(draws),
        runtime_next=len(batches),
    )


def _renewing_static(law, xi, shift):
    m = (Q(1, 8), Q(0), Q(3, 4), Q(-3, 4), Q(0))
    v = (Q(1, 64), Q(1), Q(1), Q(1), Q(1))
    theta = (Q(1, 10), Q(0), Q(0), Q(0), Q(0))
    assert all(law["form_rails"][0] < value < law["form_rails"][1] for value in m)
    assert all(
        law["capacity_rails"][0] < value < law["capacity_rails"][1] for value in v
    )
    assert 0 < theta[0] + shift < PI
    prepared = _prepared(
        law,
        tuple(I(x) for x in domain._paired(m)),
        domain._paired(v),
        domain._paired(theta),
    )
    assert prepared["marks"] == ("ZHIR",) * 2 + ("IL",) * 8
    assert prepared["pressure"][0].lo > 0
    contexts = (CONTEXT + ("OZ",),) * 2 + (CONTEXT,) * 8
    assert _grammar("ZHIR", contexts[0], True, 1) is None
    # This source has no observed/derived previous sample. The event below
    # is a conditional action, not an admitted executed Mutation.
    assert _temporal_reason(None, m[0], Q(1), xi) == "missing_history"
    traces = tuple((*trace, mark) for trace, mark in zip(contexts, prepared["marks"]))
    action = _phase_step(law, (theta[0] + shift, *theta[1:]), traces)
    reference = _phase_step(law, theta, tuple((*trace, "IL") for trace in contexts))
    assert action["branch"] == reference["branch"] == "stable"
    assert action["gains"] == reference["gains"]
    diameter = max(action["relative"]) - min(action["relative"])
    assert diameter > theta[0]
    k, b, w = law["pressure"]
    # Prior scope: this same held target capacity, default rails and at most
    # one default OZ pressure amplification, including its zero fallback.
    pressure_bound = 2 * k + b * law["capacity_rails"][1] + w
    held_bound = max(law["factors"]["OZ_dnfr_factor"] * pressure_bound, Q(0.1))
    rate_bound = v[0] * held_bound
    assert rate_bound < Q(31, 1000) < xi
    return dict(
        prepared=prepared,
        theta=theta,
        contexts=contexts,
        action=action,
        reference=reference,
        diameter=diameter,
        rate_bound=rate_bound,
        history_status="unacquired",
    )


@pytest.fixture(scope="module")
def frozen(configuration, renewal_guard):
    law = enclosure._law(configuration)
    cfg = configuration.graph
    assert cfg["OZ_NOISE_MODE"] is False
    xi = Q(cfg["ZHIR_THRESHOLD_XI"])
    shift = law["factors"]["ZHIR_theta_shift_factor"] * PI / 4
    inputs = (
        I(MEANS[0] - RADIUS, MEANS[0] + RADIUS),
        I(-law["factors"]["NAV_jitter"], law["factors"]["NAV_jitter"]),
    )
    source = tuple(inputs[0] if n < 2 else I(MEANS[n // 2]) for n in range(10))
    v, phase = domain._paired(CAPACITIES), domain._paired(PHASES)
    first = _prepared(law, source, v, phase)
    assert first["marks"] == domain._paired(("OZ", "OZ", "IL", "IL", "NAV"))
    assert all(_grammar(mark, CONTEXT, True, 0) is None for mark in first["marks"])
    assert 2 <= min(law["lag_limits"])  # neither attempted batch is forced
    f, held = law["factors"], []
    for n, mark in enumerate(first["marks"]):
        p = first["pressure"][n]
        if mark == "OZ":
            assert not p.contains(0)
            assert abs(p).lo > Q(1e-9)
            q = f["OZ_dnfr_factor"] * p
        elif mark == "IL":
            q = f["IL_dnfr_factor"] * p
        else:
            assert mark == "NAV" and p.lo > 0
            q = (1 - f["NAV_eta"]) * p + f["NAV_eta"] * v[n] + inputs[1]
            assert (q - p).lo > 0
        held.append(q)
    rates = tuple(capacity * p for capacity, p in zip(v, held))
    endpoint = tuple(value + DT * rate for value, rate in zip(source, rates))
    assert all(-1 < value.lo <= value.hi < 1 for value in endpoint)
    assert min(abs(value).lo for value in held) > Q(24, 100) > law["epsilon"]
    assert all(rates[n].lo > Q(13, 100) > xi for n in (0, 1))
    traces = tuple((*CONTEXT, mark) for mark in first["marks"])
    debt = tuple(int(mark == "OZ") for mark in first["marks"])
    coordinator = _phase_step(law, PHASES, traces)
    assert coordinator["branch"] == "dissonant" and coordinator["load"] == Q(3, 20)
    relative_phase = coordinator["relative"]
    second = _prepared(law, endpoint, v, domain._paired(relative_phase))
    assert second["marks"] == domain._paired(("ZHIR", "RA", "IL", "IL", "NAV"))
    assert all(_grammar("ZHIR", traces[n], True, debt[n]) is None for n in (0, 1))
    assert all(second["pressure"][n].lo > Q(8, 100) for n in (0, 1))
    prefix_phase = (relative_phase[0] + shift, *relative_phase[1:])
    selected = tuple(
        j for j in (0, 2) if abs(prefix_phase[j] - prefix_phase[1]) <= PI / 2
    )
    reference_selected = tuple(
        j for j in (0, 2) if abs(relative_phase[j] - relative_phase[1]) <= PI / 2
    )
    assert selected == reference_selected == (0, 2)
    signal = sum(endpoint[2 * block + s] for block in selected for s in (0, 1)) / 4
    proposal = (1 - f["RA_epi_diff"]) * endpoint[2] + f["RA_epi_diff"] * signal
    assert endpoint[2].hi < 0 < proposal.lo
    assert proposal.lo > Q(11, 1000)
    # The no-Mutation action (IL at0/1) leaves all forms and both RA inputs
    # unchanged. It therefore encounters this same identity rejection.
    committed = (0, 1)
    records = _prefix_bookkeeping(first["marks"], committed)
    stop = dict(
        status="live_state_rejection",
        node=2,
        operator="RA",
        committed=committed,
        selected=selected,
        signal=signal,
        proposal=proposal,
        phase=prefix_phase,
        time=Q(1) + DT,
        **records,
        counters=(0,) * 10,
        preserved_rates=rates,
        preserved_acceleration=tuple(rate / DT for rate in rates),
        preserved_gains=coordinator["gains"],
        terminal_snapshots=1,
        post_rejection_flow=None,
        post_rejection_coordinator=None,
        next_nav_inputs_consumed=0,
    )
    static = _renewing_static(law, xi, shift)
    forecast = dict(
        source=source,
        first=first,
        held=tuple(held),
        rates=rates,
        endpoint=endpoint,
        traces=traces,
        coordinator=coordinator,
        second=second,
        stop=stop,
        static=static,
    )
    digest = sha256(
        repr(
            (
                law,
                xi,
                shift,
                inputs,
                forecast,
                CONTEXT,
                VERTICES,
                DT,
                INTERVAL_METHOD,
                "exact containment; no tolerance",
            )
        ).encode()
    ).hexdigest()
    return dict(law=law, xi=xi, shift=shift, forecast=forecast, digest=digest)


def _fine_control(law, xi, shift, vertex):
    displacement, even, odd = vertex
    m = (MEANS[0] + displacement * RADIUS, *MEANS[1:])
    x, v, theta = domain._paired(m), domain._paired(CAPACITIES), domain._paired(PHASES)
    p = _pressure(law, x, v, theta)
    alpha, beta, gamma = law["stability"]

    def prepare(pressure, phases):
        peak = max(map(abs, pressure))
        si = tuple(
            alpha * v[n] / max(v)
            + beta
            * (1 - abs(sum(phases[j] for j in _neighbors(n)) / 4 - phases[n]) / PI)
            + gamma * (1 - abs(pressure[n]) / peak)
            for n in range(10)
        )
        return si, tuple(
            relative._choice(value, abs(value_p) / peak, law["thresholds"])
            for value, value_p in zip(si, pressure)
        )

    si, marks = prepare(p, theta)
    f, q = law["factors"], []
    for n, mark in enumerate(marks):
        if mark == "OZ":
            q.append(p[n] * f["OZ_dnfr_factor"])
        elif mark == "IL":
            q.append(p[n] * f["IL_dnfr_factor"])
        else:
            assert mark == "NAV"
            q.append(
                (1 - f["NAV_eta"]) * p[n]
                + f["NAV_eta"] * v[n]
                + (even if n % 2 == 0 else odd) * f["NAV_jitter"]
            )
    rates = tuple(capacity * pressure for capacity, pressure in zip(v, q))
    end = tuple(value + DT * rate for value, rate in zip(x, rates))
    history = tuple(((Q(1), left), (Q(17, 16), right)) for left, right in zip(x, end))
    traces = tuple((*CONTEXT, mark) for mark in marks)
    coordinated = _phase_step(law, PHASES, traces)
    phases = list(domain._paired(coordinated["relative"]))
    fresh = _pressure(law, end, v, phases)
    next_si, requested = prepare(fresh, phases)
    committed = []
    for n, mark in enumerate(requested):
        if mark == "ZHIR":
            assert fresh[n] > 0
            assert _temporal_reason(history[n], end[n], Q(17, 16), xi) is None
            assert _grammar(mark, traces[n], True, int(marks[n] == "OZ")) is None
            phases[n] += shift
            committed.append(n)
        else:
            assert mark == "RA"
            neighbors = tuple(
                j for j in _neighbors(n) if abs(phases[j] - phases[n]) <= PI / 2
            )
            signal = sum(end[j] for j in neighbors) / len(neighbors)
            proposed = (1 - f["RA_epi_diff"]) * end[n] + f["RA_epi_diff"] * signal
            assert end[n] < 0 < proposed
            return dict(
                pressure=p,
                si=si,
                marks=marks,
                held=tuple(q),
                rates=rates,
                endpoint=end,
                history=history,
                fresh=fresh,
                next_si=next_si,
                requested=requested,
                committed=tuple(committed),
                stop_node=n,
                phase=tuple(phases),
                signal=signal,
                proposal=proposed,
            )
    raise AssertionError("frozen protocol unexpectedly completed")


@pytest.fixture(scope="module")
def controls(frozen):
    return tuple(
        _fine_control(frozen["law"], frozen["xi"], frozen["shift"], vertex)
        for vertex in VERTICES
    )


def test_frozen_domain_and_execution_guards(frozen, renewal_guard):
    targets, forbidden = renewal_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert len(frozen["digest"]) == 64 and len(VERTICES) == 8
    assert CONTEXT == ("AL", "IL", "EN")
    assert RADIUS == Q(1, 4096)


def test_interval_certificate_acquires_trigger_then_stops_at_identity(frozen):
    f = frozen["forecast"]
    assert f["first"]["marks"] == domain._paired(("OZ", "OZ", "IL", "IL", "NAV"))
    assert f["second"]["marks"] == domain._paired(("ZHIR", "RA", "IL", "IL", "NAV"))
    assert min(f["rates"][n].lo for n in (0, 1)) > frozen["xi"]
    assert f["stop"]["committed"] == (0, 1) and f["stop"]["node"] == 2
    assert f["endpoint"][2].hi < 0 < f["stop"]["proposal"].lo
    before = f["coordinator"]["relative"]
    after = f["stop"]["phase"]
    assert max(after) - min(after) < max(before) - min(before)
    assert f["stop"]["post_rejection_flow"] is None
    assert f["stop"]["post_rejection_coordinator"] is None


def test_eight_independent_exact_controls_fit_the_frozen_intervals(frozen, controls):
    f = frozen["forecast"]
    assert len(controls) == 8
    for c in controls:
        for name, boxes in (
            ("pressure", f["first"]["pressure"]),
            ("si", f["first"]["si"]),
            ("held", f["held"]),
            ("rates", f["rates"]),
            ("endpoint", f["endpoint"]),
            ("fresh", f["second"]["pressure"]),
            ("next_si", f["second"]["si"]),
        ):
            assert all(box.contains(value) for box, value in zip(boxes, c[name])), name
        assert c["marks"] == f["first"]["marks"]
        assert c["requested"] == f["second"]["marks"]
        assert c["committed"] == (0, 1) and c["stop_node"] == 2
        assert c["phase"] == domain._paired(f["stop"]["phase"])
        assert f["stop"]["signal"].contains(c["signal"])
        assert f["stop"]["proposal"].contains(c["proposal"])
        for n, samples in enumerate(c["history"]):
            assert samples[-1][1] == c["endpoint"][n]
            assert (samples[1][1] - samples[0][1]) / DT == c["rates"][n]
        # Suppressing the Mutation prefix preserves both RA neighbors, form
        # inputs and the identical rejection; it is not a ZHIR-caused failure.
        theta = domain._paired(f["coordinator"]["relative"])
        selected = tuple(j for j in _neighbors(2) if abs(theta[j] - theta[2]) <= PI / 2)
        signal = sum(c["endpoint"][j] for j in selected) / len(selected)
        assert signal == c["signal"]


def test_accepted_prefix_keeps_failed_call_memory_and_unconsumed_inputs(frozen):
    f, stop = frozen["forecast"], frozen["forecast"]["stop"]
    assert stop["traces"][:2] == (CONTEXT + ("OZ", "ZHIR"),) * 2
    assert stop["traces"][2:] == f["traces"][2:]
    assert stop["ordinals"] == (5,) * 2 + (4,) * 8
    assert stop["debt"] == (2,) * 2 + (1,) * 2 + (0,) * 6
    assert stop["lags"] == (2,) * 10 and stop["time"] == Q(17, 16)
    assert stop["runtime_next"] == 2
    assert stop["rng_progress"] == (0,) * 8 + (1,) * 2
    assert stop["next_nav_inputs_consumed"] == 0
    assert stop["preserved_rates"] == f["rates"]
    assert stop["preserved_gains"] == f["coordinator"]["gains"]
    assert stop["preserved_acceleration"] == tuple(rate / DT for rate in f["rates"])
    assert stop["counters"] == (0,) * 10 and stop["terminal_snapshots"] == 1


def test_trigger_grammar_and_history_are_independent_obligations(frozen, controls):
    c = controls[0]
    history, current, xi = c["history"][0], c["endpoint"][0], frozen["xi"]
    assert _temporal_reason(history, current, Q(17, 16), xi) is None
    assert _temporal_reason(None, current, Q(17, 16), xi) == "missing_history"
    assert (
        _temporal_reason(history, current + Q(1, 100), Q(17, 16), xi)
        == "stale_or_unassociated_history"
    )
    assert (
        _temporal_reason(
            ((Q(1), current), (Q(17, 16), current)), current, Q(17, 16), xi
        )
        == "threshold_not_crossed"
    )
    assert _grammar("OZ", ("EN",), True, 0) == "U4a"
    assert _grammar("ZHIR", CONTEXT, True, 0) == "U4b"
    assert _grammar("ZHIR", ("EN", "OZ"), True, 1) == "U4a"
    assert _grammar("ZHIR", CONTEXT + ("OZ",), False, 1) == "U4b"
    assert _grammar("ZHIR", CONTEXT + ("OZ",), True, U2_DEBT_CAPACITY) == "U2"


def test_conditional_renewal_criterion_does_not_supply_missing_history(frozen):
    c = frozen["forecast"]["static"]
    kg, kl = c["action"]["gains"]
    a, diameter, shift = 1 - kg - kl, Q(1, 10), frozen["shift"]
    assert a > kl / 2
    assert c["diameter"] == a * (diameter + shift) > diameter
    reference = c["reference"]["relative"]
    assert max(reference) - min(reference) == a * diameter < diameter
    assert c["history_status"] == "unacquired"
    assert c["rate_bound"] < frozen["xi"]
    # For this sole-maximum geometry the conclusion is uniform over every
    # allowed default gain pair, not only the stable policy branch above.
    policy = frozen["law"]["phase_policy"]
    a_min = 1 - policy["kG_max"] - policy["kL_max"]
    assert a_min > policy["kL_max"] / 2
    assert a_min * (diameter + shift) > diameter
    # Exact oscillation norm for C=A*Id+KL*N on C5. The common global term
    # cancels; actual gain selection still belongs to each complete action.
    rows = tuple(
        tuple(
            a if j == i else kl / 2 if j in ((i - 1) % 5, (i + 1) % 5) else Q(0)
            for j in range(5)
        )
        for i in range(5)
    )
    contraction = max(
        sum(abs(x - y) for x, y in zip(left, right)) / 2
        for left, right in combinations(rows, 2)
    )
    assert contraction == 1 - kg - kl / 2
    # Linear row differences attain their extrema at binary cube vertices;
    # these controls certify the sharp oscillation norm, not extra sources.
    vertex_oscillations = []
    for vertex in product((Q(0), Q(1)), repeat=5):
        image = tuple(
            sum(coefficient * value for coefficient, value in zip(row, vertex))
            for row in rows
        )
        oscillation = max(image) - min(image)
        assert oscillation <= contraction * (max(vertex) - min(vertex))
        vertex_oscillations.append(oscillation)
    assert max(vertex_oscillations) == contraction
    shift_oscillation = max(row[0] for row in rows) - min(row[0] for row in rows)
    assert shift_oscillation == a
    assert a * shift > (kg + kl / 2) * diameter  # necessary, not sufficient in general


def test_shift_scalar_signed_zero_and_absolute_telemetry_are_separate(frozen):
    shift = frozen["shift"]
    factor = frozen["law"]["factors"]["ZHIR_theta_shift_factor"]
    assert shift == factor * PI / 4 and shift != Q(1, 4)
    assert float(factor) * (float(PI) / 4) == 0.25  # labeled binary64 realization
    assert 0.0 == -0.0 and math.copysign(1.0, 0.0) != math.copysign(1.0, -0.0)
    # Equal relative actions at different absolute origins have different
    # compulsory quadrant telemetry. The physical quotient omits that data.
    before = PI / 2 - shift / 2
    shifted = before + PI / 4

    def regime(theta):
        return int(theta // (PI / 2))

    assert regime(before) != regime(before + shift)
    assert regime(shifted) == regime(shifted + shift)
    # For the fixed initial origin0 source, the unevaluated global target
    # lies in(0,3). Preserve that common-angle association as a bound rather
    # than replacing the absolute phase by its zero relative coordinate.
    kg, kl = frozen["forecast"]["coordinator"]["gains"]
    anchor = (1 - kg - kl) * PHASES[0] + kl * domain._neighbor(PHASES, 0)
    absolute_before = I(anchor, anchor + 3 * kg)
    assert 0 < absolute_before.lo < absolute_before.hi < PI / 2
    assert absolute_before.hi + shift < PI / 2
