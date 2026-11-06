"""One-call relative-phase descent, with no native scientific evaluations.

Freeze two nonzero relative-phase sources: the existing joint supplied
Emission/Reception/Coherence/Transition/Resonance transcript and the unchanged
neighbor-gain source with actual default RA/RA/IL/IL/IL requests. Retain every
non-phase coordinate, input, grammar record, lag, counter and previous rate.
Prepare both origin-free block forecasts before six independent fine controls:
one complete call per source and lifted origin -7/4, 0 or 5/2. All comparisons
are exact; no numerical tolerance, source search or additional trajectory.

The common-lift proof uses ideal circular phases and represented constitutive
coefficients. A coherent real lift is not the engine's binary64 modulo chart.
Global resultant angles remain symbolic; their common coordinator term cancels
from relative output exactly. This proves neither bitwise equivariance of
represented wrap/trigonometry nor origin-independent absolute readouts.
"""

from fractions import Fraction as Q
from hashlib import sha256
from itertools import combinations
from math import factorial

import pytest

from tests.physics import test_native_collective_enclosure as enclosure
from tests.physics import test_native_joint_collective_calculus as joint
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER
from tnfr.operators.grammar_types import DESTABILIZERS, GLYPH_TO_FUNCTION, STABILIZERS

scalar, domain = joint.scalar, joint.domain
gain = domain.gain
no_execution = joint.no_execution
calculus_guard = joint.calculus_guard
configuration = joint.configuration
DT, PI = joint.DT, joint.PI
ORIGINS = (Q(-7, 4), Q(0), Q(5, 2))
FUNCTIONS = {glyph.value: name for glyph, name in GLYPH_TO_FUNCTION.items()}


@pytest.fixture(scope="module", autouse=True)
def descent_guard(calculus_guard):
    yield calculus_guard


def _relative(values):
    return tuple(value - values[0] for value in values)


def _choice(si, ratio, cuts):
    if si >= cuts["si_hi"]:
        return "IL"
    if si <= cuts["si_lo"]:
        return "OZ" if ratio > cuts["dnfr_hi"] else "ZHIR"
    return "NAV" if ratio > cuts["dnfr_hi"] else "RA"


def _context(graph, old_rates):
    cfg = graph.graph
    assert domain.base.RUNTIME_STEP_NEXT_KEY not in cfg and not cfg.get("_epi_hist")
    result = dict(
        rates=old_rates,
        acceleration=(Q(0),) * 10,
        traces=tuple(tuple(graph.nodes[n]["glyph_history"]) for n in graph),
        debt=tuple(graph.nodes[n][domain.U2_DEBT_KEY] for n in graph),
        prior=tuple(graph.nodes[n][domain.PRIOR_COHERENCE_KEY] for n in graph),
        ordinals=tuple(graph.nodes[n][domain.OPERATOR_STEP_KEY] for n in graph),
        kinds=tuple(graph.nodes[n]["epi_kind"] for n in graph),
        counts=tuple(graph.nodes[n]["stable_count"] for n in graph),
        lag_al=tuple(cfg["history"]["since_AL"][n] for n in graph),
        lag_en=tuple(cfg["history"]["since_EN"][n] for n in graph),
        time=Q(cfg["_t"]),
        temporal=tuple(
            tuple(graph.nodes[n].get("epi_time_history", ())) for n in graph
        ),
        stable_fraction_history=(),
        runtime_ordinal=0,
    )
    assert result["traces"] == (("EN",),) * 10
    assert result["prior"] == (True,) * 10 and result["kinds"] == ("seed",) * 10
    assert result["ordinals"] == (3,) * 10 and result["time"] == 1
    assert result["counts"] == result["lag_al"] == result["lag_en"] == (0,) * 10
    assert result["temporal"] == ((),) * 10
    assert all(0 <= value <= U2_DEBT_CAPACITY for value in result["debt"])
    return result


def _controller(law, phases, traces):
    """R^2 uses differences only; polynomial tails bound the true cosine."""
    assert max(phases) - min(phases) < 3  # strict common semicircle
    cos_lower, cos_upper = [], []
    for a, b in combinations(phases, 2):
        angle = a - b
        terms = tuple(
            Q((-1) ** j, factorial(2 * j)) * angle ** (2 * j) for j in range(8)
        )
        # The omitted alternating tail decreases for |angle| < 3.
        cos_lower.append(sum(terms))
        cos_upper.append(sum(terms[:-1]))
    lower = (5 + 2 * sum(cos_lower)) / 25
    upper = (5 + 2 * sum(cos_upper)) / 25
    assert 0 < lower <= upper <= 1
    load = Q(
        sum(mark == "NAV" for trace in traces for mark in trace), sum(map(len, traces))
    )
    policy = law["phase_policy"]
    if lower >= policy["R_hi"] ** 2 and load <= policy["disr_lo"]:
        branch = "stable"
        targets = policy["kG_min"], policy["kL_min"]
    elif upper <= policy["R_lo"] ** 2 or load >= policy["disr_hi"]:
        branch = "dissonant"
        targets = policy["kG_max"], (policy["kL_min"] + policy["kL_max"]) / 2
    else:
        raise AssertionError("unresolved frozen controller branch")
    gains = tuple(
        old + policy["up" if target > old else "down"] * (target - old)
        for old, target in zip(law["phase_gains"], targets)
    )
    assert min(gains) > 0 and sum(gains) < 1
    return branch, gains, (lower, upper), load


def _adapt(law, si, held, capacities, counts):
    flags = tuple(
        value >= law["thresholds"]["si_hi"] and abs(p) <= law["epsilon"]
        for value, p in zip(si, held)
    )
    next_counts = tuple(count + 1 if flag else 0 for count, flag in zip(counts, flags))
    due = tuple(count >= law["tau"] for count in next_counts)
    # Immutable snapshot: twins have no edge to one another.
    neighbor = tuple(
        sum(
            capacities[2 * i + s]
            for i in ((n // 2 - 1) % 5, (n // 2 + 1) % 5)
            for s in (0, 1)
        )
        / 4
        for n in range(10)
    )
    after = tuple(
        value + law["mu"] * (mean - value) if flag else value
        for value, mean, flag in zip(capacities, neighbor, due)
    )
    return after, flags, next_counts, due


def _memory(law, context, marks, event, endpoint, rates, resets):
    assert not {FUNCTIONS[mark] for mark in marks} & DESTABILIZERS
    traces = tuple((*trace, mark) for trace, mark in zip(context["traces"], marks))
    return dict(
        traces=traces,
        debt=tuple(
            max(0, debt - 1) if FUNCTIONS[mark] in STABILIZERS else debt
            for debt, mark in zip(context["debt"], marks)
        ),
        prior=context["prior"],
        kinds=context["kinds"],
        ordinals=tuple(value + 1 for value in context["ordinals"]),
        lag_al=tuple(
            0 if mark == "AL" else value + 1
            for value, mark in zip(context["lag_al"], marks)
        ),
        lag_en=tuple(
            (
                0
                if mark == "EN"
                else min(value + 1, law["lag_limits"][1]) if mark == "AL" else value + 1
            )
            for value, mark in zip(context["lag_en"], marks)
        ),
        resets=resets,
        temporal=tuple(
            ((context["time"], left), (context["time"] + DT, right))
            for left, right in zip(event, endpoint)
        ),
        rates=rates,
        acceleration=tuple(
            (rate - old) / DT for rate, old in zip(rates, context["rates"])
        ),
        time=context["time"] + DT,
        runtime_ordinal=context["runtime_ordinal"] + 1,
        remesh_snapshot=endpoint,
        stable_fraction_history=context["stable_fraction_history"],
    )


def _quotient(law, source):
    """Exact block preparation and live events with no absolute phase origin."""
    k, b, w = law["pressure"]
    alpha, beta, gamma = law["stability"]
    f = law["factors"]
    m, d, v, e, z = (
        source[name] for name in ("means", "d", "capacities", "e", "phases")
    )
    assert z[0] == 0 and max(z) - min(z) <= 3
    for values, differences, rails in (
        (m, d, law["form_rails"]),
        (v, e, law["capacity_rails"]),
    ):
        assert all(
            rails[0] < value - abs(delta) <= value + abs(delta) < rails[1]
            for value, delta in zip(values, differences)
        )
    assert min(value - abs(delta) for value, delta in zip(v, e)) > 0
    offsets = tuple(domain._neighbor(z, i) - z[i] for i in range(5))
    baseline = tuple(
        k * (domain._neighbor(m, i) - m[i])
        + b * (domain._neighbor(v, i) - v[i])
        + w * offsets[i] / PI
        for i in range(5)
    )
    contrast = tuple(-k * a - b * c for a, c in zip(d, e))
    pressure = scalar._fine(baseline, contrast)
    peak = max(map(abs, pressure))
    vmax = max(value + abs(delta) for value, delta in zip(v, e))
    vf = scalar._fine(v, e)
    si = tuple(
        alpha * vf[n] / vmax
        + beta * (1 - abs(offsets[n // 2]) / PI)
        + gamma * (1 - abs(pressure[n]) / peak)
        for n in range(10)
    )
    defaults = tuple(
        _choice(value, abs(p) / peak, law["thresholds"])
        for value, p in zip(si, pressure)
    )
    marks = domain._paired(source["marks"])
    origins = []
    for n, mark in enumerate(marks):
        ctx = source["context"]
        forced = (
            "AL"
            if ctx["lag_al"][n] + 1 > law["lag_limits"][0]
            else "EN" if ctx["lag_en"][n] + 1 > law["lag_limits"][1] else None
        )
        assert forced is None
        if source["origin"] == "default":
            assert mark == defaults[n]
        origins.append(source["origin"])
    y, a, u, c, phi, qbar, qdelta = map(list, (m, d, v, e, z, baseline, contrast))
    subsets, signals, boosts, prefixes = [], [], [], []
    for i, mark in enumerate(source["marks"]):
        selected = ((i - 1) % 5, (i + 1) % 5)
        if mark == "RA":
            selected = tuple(j for j in selected if abs(phi[j] - phi[i]) <= PI / 2)
            assert selected
        signal = sum(y[j] for j in selected) / len(selected)
        boost = Q(1)
        if mark == "AL":
            y[i] += f["AL_boost"]
        elif mark in {"EN", "RA"}:
            mix = f["EN_mix"] if mark == "EN" else f["RA_epi_diff"]
            y[i], a[i] = (1 - mix) * m[i] + mix * signal, (1 - mix) * d[i]
        if mark == "RA":
            assert all(
                before * after > 0
                for before, after in zip(
                    (m[i] - d[i], m[i] + d[i]), (y[i] - a[i], y[i] + a[i])
                )
            )
            phi[i] += f["RA_phase_coupling"] * (
                sum(phi[j] for j in selected) / len(selected) - phi[i]
            )
            if abs(signal) > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER):
                boost += f["RA_vf_amplification"]
            u[i], c[i] = boost * v[i], boost * e[i]
        if mark == "IL":
            qbar[i], qdelta[i] = (
                f["IL_dnfr_factor"] * baseline[i],
                f["IL_dnfr_factor"] * contrast[i],
            )
        elif mark == "NAV":
            assert min(pressure[2 * i : 2 * i + 2]) > 0
            even, odd = source["innovations"]
            eta = f["NAV_eta"]
            qbar[i] = (1 - eta) * baseline[i] + eta * v[i] + (even + odd) / 2
            qdelta[i] = (1 - eta) * contrast[i] + eta * e[i] + (odd - even) / 2
            assert all(
                new != old
                for new, old in zip(
                    (qbar[i] - qdelta[i], qbar[i] + qdelta[i]),
                    pressure[2 * i : 2 * i + 2],
                )
            )
        subsets.append(selected)
        signals.append(signal)
        boosts.append(boost)
        prefixes.append(
            (
                scalar._fine(y, a),
                scalar._fine(u, c),
                domain._paired(phi),
                scalar._fine(qbar, qdelta),
            )
        )
    rates_mean = tuple(
        cap * p + ec * delta for cap, ec, p, delta in zip(u, c, qbar, qdelta)
    )
    rates_delta = tuple(
        cap * delta + ec * p for cap, ec, p, delta in zip(u, c, qbar, qdelta)
    )
    endpoint = scalar._fine(
        tuple(value + DT * rate for value, rate in zip(y, rates_mean)),
        tuple(value + DT * rate for value, rate in zip(a, rates_delta)),
    )
    event, event_v, _, held = prefixes[-1]
    rates = scalar._fine(rates_mean, rates_delta)
    assert all(
        law["form_rails"][0] < value < law["form_rails"][1]
        for value in (*event, *endpoint)
    )
    assert all(
        law["capacity_rails"][0] < value < law["capacity_rails"][1] for value in event_v
    )
    adapted, flags, counts, due = _adapt(
        law, si, held, event_v, source["context"]["counts"]
    )
    assert not any(flags) and min(map(abs, held)) > law["epsilon"]
    initial = scalar._fine(m, d)
    memory = _memory(
        law,
        source["context"],
        marks,
        event,
        endpoint,
        rates,
        tuple(x != old for x, old in zip(event, initial)),
    )
    branch, gains, r2_bounds, load = _controller(law, phi, memory["traces"])
    kg, kl = gains
    local = tuple(domain._neighbor(phi, i) for i in range(5))
    # This is a quotient, not a stored absolute angle or a recovered origin.
    relative = tuple(
        (1 - kg - kl) * (value - phi[0]) + kl * (neighbor - local[0])
        for value, neighbor in zip(phi, local)
    )
    return dict(
        pressure=pressure,
        si=si,
        defaults=defaults,
        marks=marks,
        origins=tuple(origins),
        subsets=tuple(subsets),
        signals=tuple(signals),
        boosts=tuple(boosts),
        prefixes=tuple(prefixes),
        endpoint=endpoint,
        capacities=adapted,
        rates=rates,
        held=held,
        counts=counts,
        due=due,
        flags=flags,
        memory=memory,
        event_phases=tuple(phi),
        relative_phases=relative,
        branch=branch,
        gains=gains,
        r2_bounds=r2_bounds,
        load=load,
    )


@pytest.fixture(scope="module")
def frozen(configuration, descent_guard):
    law = enclosure._law(configuration)
    gain._family_certificate(configuration)  # pure existing admission only
    mixed = dict(
        name="supplied_five",
        means=scalar.MEANS,
        d=scalar.CONTRASTS,
        capacities=scalar.CAPACITIES,
        e=joint.CAPACITY_CONTRASTS,
        phases=joint.PHASES,
        marks=scalar.SCHEDULE,
        origin="supplied",
        innovations=(
            -law["factors"]["NAV_jitter"] / 3,
            law["factors"]["NAV_jitter"] / 4,
        ),
        context=_context(configuration, tuple(Q(n - 4, 64) for n in range(10))),
    )
    actual = dict(
        name="actual_gain",
        means=gain.FORMS,
        d=(Q(0),) * 5,
        capacities=gain.CAPACITY,
        e=(Q(0),) * 5,
        phases=(
            Q(0),
            gain.LIMIT + gain.EPSILON,
            2 * gain.EPSILON,
            Q(3),
            gain.LIMIT - gain.EPSILON,
        ),
        marks=("RA", "RA", "IL", "IL", "IL"),
        origin="default",
        innovations=(Q(0), Q(0)),
        context=_context(
            configuration, domain._paired(tuple(map(Q, domain.base.RATES)))
        ),
    )
    sources = mixed, actual
    forecasts = tuple(_quotient(law, source) for source in sources)
    assert tuple(result["branch"] for result in forecasts) == ("stable", "dissonant")
    assert forecasts[1]["subsets"][:2] == ((4,), (0, 2))
    assert all(
        result["relative_phases"][0] == 0
        and max(result["relative_phases"]) - min(result["relative_phases"]) > 0
        for result in forecasts
    )
    digest = sha256(
        repr((law, sources, forecasts, ORIGINS, DT, "exact; no tolerance")).encode()
    ).hexdigest()
    return dict(
        law=law,
        sources=sources,
        forecasts=forecasts,
        graph=configuration,
        digest=digest,
    )


def _fine_lift(graph, law, source, origin):
    """Independent fine ordered arithmetic on the declared coherent lift."""
    x = list(scalar._fine(source["means"], source["d"]))
    v = list(scalar._fine(source["capacities"], source["e"]))
    theta = list(domain._paired(tuple(origin + value for value in source["phases"])))
    initial = tuple(x)
    p = joint._fresh(graph, tuple(x), tuple(v), tuple(theta))
    alpha, beta, gamma = law["stability"]
    si = tuple(
        alpha * v[n] / max(v)
        + beta
        * (1 - abs(sum(theta[j] for j in graph.neighbors(n)) / 4 - theta[n]) / PI)
        + gamma * (1 - abs(p[n]) / max(map(abs, p)))
        for n in graph
    )
    defaults = tuple(
        _choice(value, abs(pressure) / max(map(abs, p)), law["thresholds"])
        for value, pressure in zip(si, p)
    )
    marks = (
        defaults if source["origin"] == "default" else domain._paired(source["marks"])
    )
    q, prefixes, subsets, signals, boosts = list(p), [], [], [], []
    f = law["factors"]
    for n, mark in enumerate(marks):
        neighbors = tuple(graph.neighbors(n))
        selected = (
            tuple(j for j in neighbors if abs(theta[j] - theta[n]) <= PI / 2)
            if mark == "RA"
            else neighbors
        )
        assert selected
        signal = sum(x[j] for j in selected) / len(selected)
        boost = Q(1)
        if mark == "AL":
            x[n] += f["AL_boost"]
        elif mark in {"EN", "RA"}:
            mix = f["EN_mix"] if mark == "EN" else f["RA_epi_diff"]
            proposal = (1 - mix) * x[n] + mix * signal
            if mark == "RA":
                assert x[n] * proposal > 0
                theta[n] += f["RA_phase_coupling"] * (
                    sum(theta[j] for j in selected) / len(selected) - theta[n]
                )
                if abs(signal) > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER):
                    boost += f["RA_vf_amplification"]
                v[n] *= boost
            x[n] = proposal
        elif mark == "IL":
            q[n] *= f["IL_dnfr_factor"]
        else:
            assert mark == "NAV" and q[n] > 0
            q[n] = (
                (1 - f["NAV_eta"]) * q[n]
                + f["NAV_eta"] * v[n]
                + source["innovations"][n % 2]
            )
            assert q[n] != p[n]
        subsets.append(tuple(sorted({j // 2 for j in selected})))
        signals.append(signal)
        boosts.append(boost)
        prefixes.append((tuple(x), tuple(v), tuple(theta), tuple(q)))
    rates = tuple(value * pressure for value, pressure in zip(v, q))
    endpoint = tuple(value + DT * rate for value, rate in zip(x, rates))
    flags = tuple(
        value >= law["thresholds"]["si_hi"] and abs(pressure) <= law["epsilon"]
        for value, pressure in zip(si, q)
    )
    counts = tuple(
        count + 1 if flag else 0
        for count, flag in zip(source["context"]["counts"], flags)
    )
    due = tuple(count >= law["tau"] for count in counts)
    snapshot = tuple(v)
    for n in graph:
        if due[n]:
            v[n] += law["mu"] * (
                sum(snapshot[j] for j in graph.neighbors(n)) / 4 - snapshot[n]
            )
    traces = tuple(
        (*old, mark) for old, mark in zip(source["context"]["traces"], marks)
    )
    block_phases = tuple(theta[2 * i] - origin for i in range(5))
    branch, gains, r2_bounds, load = _controller(law, block_phases, traces)
    kg, kl = gains
    # Let Psi_abs = origin + Psi_relative. Preserve the unknown common term
    # kg*Psi_relative symbolically, instead of evaluating an absolute atan2.
    offsets = tuple(
        (1 - kg - kl) * theta[n]
        + kl * sum(theta[j] for j in graph.neighbors(n)) / 4
        + kg * origin
        for n in graph
    )
    relative = _relative(offsets)
    memory = _memory(
        law,
        source["context"],
        marks,
        tuple(x),
        endpoint,
        rates,
        tuple(value != old for value, old in zip(x, initial)),
    )
    return dict(
        pressure=p,
        si=si,
        defaults=defaults,
        marks=marks,
        prefixes=tuple(prefixes),
        subsets=tuple(subsets),
        signals=tuple(signals),
        boosts=tuple(boosts),
        endpoint=endpoint,
        capacities=tuple(v),
        rates=rates,
        held=tuple(q),
        flags=flags,
        counts=counts,
        due=due,
        memory=memory,
        relative_phases=relative,
        phase_offsets=offsets,
        common_coefficient=kg,
        branch=branch,
        gains=gains,
        r2_bounds=r2_bounds,
        load=load,
    )


@pytest.fixture(scope="module")
def controls(frozen):
    return tuple(
        tuple(
            _fine_lift(frozen["graph"], frozen["law"], source, origin)
            for origin in ORIGINS
        )
        for source in frozen["sources"]
    )


def test_protocol_freezes_nonzero_relative_sources_and_disables_execution(
    frozen, descent_guard
):
    targets, forbidden = descent_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert len(frozen["digest"]) == 64
    assert all(
        max(source["phases"]) > min(source["phases"]) for source in frozen["sources"]
    )
    mixed, actual = frozen["forecasts"]
    assert mixed["origins"] == ("supplied",) * 10
    assert (
        actual["origins"] == ("default",) * 10 and actual["defaults"] == actual["marks"]
    )
    assert mixed["defaults"] != mixed["marks"]


def test_six_fine_lifts_match_every_origin_free_nonphase_channel(frozen, controls):
    for expected, group in zip(frozen["forecasts"], controls):
        for actual in group:
            for name in (
                "pressure",
                "si",
                "defaults",
                "marks",
                "endpoint",
                "capacities",
                "rates",
                "held",
                "counts",
                "due",
                "flags",
                "memory",
                "branch",
                "gains",
                "r2_bounds",
                "load",
            ):
                assert actual[name] == expected[name], name
            assert actual["relative_phases"] == domain._paired(
                expected["relative_phases"]
            )


def test_live_prefix_selection_identity_and_amplification_descend(frozen, controls):
    for expected, group in zip(frozen["forecasts"], controls):
        for origin, actual in zip(ORIGINS, group):
            for i, block_prefix in enumerate(expected["prefixes"]):
                fine = actual["prefixes"][2 * i + 1]
                assert fine[:2] == block_prefix[:2] and fine[3] == block_prefix[3]
                assert tuple(value - origin for value in fine[2]) == block_prefix[2]
                assert (
                    actual["subsets"][2 * i : 2 * i + 2]
                    == (tuple(sorted(expected["subsets"][i])),) * 2
                )
                assert (
                    actual["signals"][2 * i : 2 * i + 2]
                    == (expected["signals"][i],) * 2
                )
                assert (
                    actual["boosts"][2 * i : 2 * i + 2] == (expected["boosts"][i],) * 2
                )
    gain_case = frozen["sources"][1]
    assert gain_case["phases"][1] - gain_case["phases"][0] > PI / 2
    assert frozen["forecasts"][1]["subsets"][:2] == ((4,), (0, 2))
    r = frozen["law"]["factors"]["RA_epi_diff"]
    assert frozen["forecasts"][1]["signals"][1] == (1 - r) * gain.AMPLITUDE


def test_global_angle_cancels_but_absolute_readouts_need_an_origin(frozen, controls):
    for expected, group in zip(frozen["forecasts"], controls):
        assert expected["gains"][0] > 0  # cancellation is not disabled coupling
        reference = group[1]
        for origin, actual in zip(ORIGINS, group):
            assert actual["common_coefficient"] == reference["common_coefficient"]
            assert (
                tuple(
                    value - old
                    for value, old in zip(
                        actual["phase_offsets"], reference["phase_offsets"]
                    )
                )
                == (origin,) * 10
            )
            assert actual["relative_phases"] == reference["relative_phases"]
        # These origins differ by <2*pi: even a circular absolute angle
        # distinguishes them, although the quotient input is identical.
        assert 0 < ORIGINS[2] - ORIGINS[1] < 2 * PI
        assert group[2]["phase_offsets"][0] != reference["phase_offsets"][0]


def test_controller_and_associated_memory_are_retained(frozen):
    mixed, actual = frozen["forecasts"]
    assert mixed["branch"] == "stable" and actual["branch"] == "dissonant"
    assert mixed["load"] == Q(1, 10) and actual["load"] == 0
    assert mixed["gains"] != actual["gains"]
    for source, result in zip(frozen["sources"], frozen["forecasts"]):
        memory = result["memory"]
        assert memory["ordinals"] == (4,) * 10 and memory["runtime_ordinal"] == 1
        assert memory["prior"] == (True,) * 10 and memory["kinds"] == ("seed",) * 10
        assert result["counts"] == (0,) * 10 and not any(result["due"])
        assert memory["stable_fraction_history"] == ()
        for n, samples in enumerate(memory["temporal"]):
            assert tuple(time for time, _ in samples) == (Q(1), Q(17, 16))
            assert samples[-1][1] == result["endpoint"][n]
            assert samples[1][1] - samples[0][1] == DT * result["rates"][n]
            assert (
                memory["acceleration"][n]
                == (result["rates"][n] - source["context"]["rates"][n]) / DT
            )
    assert mixed["memory"]["lag_al"] == (0,) * 2 + (1,) * 8
    assert mixed["memory"]["lag_en"] == (1,) * 2 + (0,) * 2 + (1,) * 6
    assert actual["memory"]["lag_al"] == actual["memory"]["lag_en"] == (1,) * 10
    assert mixed["memory"]["debt"] == domain._paired((0, 1, 0, 1, 0))
    assert actual["memory"]["debt"] == domain._paired((0, 1, 0, 0, 0))
    assert mixed["memory"]["resets"] == (True,) * 4 + (False,) * 4 + (True,) * 2
    assert actual["memory"]["resets"] == (True,) * 4 + (False,) * 6


def test_prefix_hull_and_coordinator_diameter_contract_conditionally(frozen):
    coupling = frozen["law"]["factors"]["RA_phase_coupling"]
    assert 0 < coupling < 1
    minimum_gain = frozen["law"]["phase_policy"]["kG_min"]
    assert 0 < minimum_gain < 1
    for source, result in zip(frozen["sources"], frozen["forecasts"]):
        lower, upper = min(source["phases"]), max(source["phases"])
        previous_diameter = upper - lower
        for _, _, phases, _ in result["prefixes"]:
            assert lower <= min(phases) <= max(phases) <= upper
            diameter = max(phases) - min(phases)
            assert diameter <= previous_diameter
            lower, upper, previous_diameter = min(phases), max(phases), diameter
        relative = result["relative_phases"]
        endpoint_diameter = max(relative) - min(relative)
        kg = result["gains"][0]
        assert endpoint_diameter <= (1 - kg) * previous_diameter
        assert endpoint_diameter < previous_diameter
        assert endpoint_diameter <= (1 - minimum_gain) * previous_diameter
    # Repeating this inequality is a conditional induction over successful
    # calls in the same domain, not evidence those future calls are admitted.


def test_active_adaptation_depends_on_invariant_inputs_not_an_origin(frozen):
    law = frozen["law"]
    # A supplied immutable adaptation snapshot, not another selected batch.
    capacities = domain._paired((Q(1, 2), Q(3, 4), Q(1), Q(1), Q(1)))
    si, held = (Q(1),) * 10, (Q(0),) * 10
    counts = (law["tau"] - 1, law["tau"] - 2) + (0,) * 8
    after, flags, next_counts, due = _adapt(law, si, held, capacities, counts)
    assert all(flags) and due == (True,) + (False,) * 9
    assert next_counts[0] == law["tau"] and next_counts[1] == law["tau"] - 1
    assert after[0] == capacities[0] + law["mu"] * (Q(7, 8) - capacities[0])
    assert after[1:] == capacities[1:] and after[0] != capacities[0]


def test_losing_relative_phases_is_not_a_valid_phase_quotient(frozen):
    source = frozen["sources"][0]
    k, b, w = frozen["law"]["pressure"]
    i = 0
    phase_term = w * (domain._neighbor(source["phases"], i) - source["phases"][i]) / PI
    assert phase_term > 0
    pressure_without_phase = k * (
        domain._neighbor(source["means"], i) - source["means"][i]
    ) + b * (domain._neighbor(source["capacities"], i) - source["capacities"][i])
    means, _ = scalar._moments(frozen["forecasts"][0]["pressure"])
    assert means[i] - pressure_without_phase == phase_term


def test_ideal_lift_equivalence_is_not_a_binary64_materialization_contract():
    # A raw conversion can erase a nonzero relative phase before any wrap or
    # trigonometry. This is not claimed to be an admitted engine conversion.
    origin, difference = Q(2**54), Q(1, 8)
    assert (origin + difference) - origin == difference
    assert float(origin + difference) - float(origin) == 0
    assert difference != 0
