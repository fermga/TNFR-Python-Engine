"""Frozen lossy collective enclosures, with no native scientific evaluations.

Before any fine control is evaluated, freeze three observations: the existing
joint five-operator supplied word with mean radii 2**-16; the actual live-prefix
RA/RA/IL/IL/IL source with form/capacity contrast radii 1/4096; and the existing
ambiguous all-IL capacity-pulse source. The predictor never receives signed
fine contrasts, fine diagnostic values, a source graph or evaluated verdicts.

The independent exact controls follow all coarse certificates. Their fixed
finite list checks containment, not a sampled proof of a continuous domain.
Outward dyadic128 interval operations and analytical absolute bounds provide
the domain enclosure. Frozen acceptance budgets are form-mean width <1/200,
form radius <1/20, capacity-mean width <1/100 and strict source/output rails.
Exact containment has no added tolerance. An unresolved admission gate stops
the predictor; uncertain adaptation retains all associated count/due options.

This private instrument predicts lossy observations conditionally. It supplies
no generic production API, new native response, autonomous occurrence law,
universal floating-point bound or indefinitely closed collective dynamics.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product

import pytest

from tests.physics import test_native_joint_collective_calculus as exact
from tests.physics import test_native_resonance_live_prefix as prefix
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.mathematics._rational_interval import INTERVAL_BITS, INTERVAL_METHOD, I
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER

scalar, domain, joint = exact.scalar, exact.domain, exact.joint
no_execution = exact.no_execution
calculus_guard = exact.calculus_guard
configuration = exact.configuration
DT, PI = exact.DT, exact.PI
MEAN_RADIUS = Q(1, 2**16)
WIDTH_BUDGET, FORM_RADIUS_BUDGET, CAPACITY_WIDTH_BUDGET = Q(1, 200), Q(1, 20), Q(1, 100)


@dataclass(frozen=True)
class Observation:
    """No exact discarded d/e or fine fields exist in this interface."""

    means: tuple
    capacities: tuple
    dx: tuple
    dv: tuple
    phases: tuple
    counts: tuple
    old_rates: tuple
    traces: tuple
    debt: tuple
    prior: tuple
    ordinals: tuple
    lag_al: tuple
    lag_en: tuple
    kinds: tuple
    old_acceleration: tuple
    time: Q
    initial_history: str
    marks: tuple
    origin: str
    innovation_mean: I
    innovation_difference: Q


@pytest.fixture(scope="module", autouse=True)
def enclosure_guard(calculus_guard):
    yield calculus_guard


def _around(interval, radius):
    return interval + I(-radius, radius)


def _hull(values):
    result = values[0]
    for value in values[1:]:
        result = result.hull(value)
    return result


def _stop(gate, block):
    return dict(status="unresolved", gate=gate, block=block)


def _law(graph):
    """Admit only configuration/support; return no graph or fine state."""
    cfg = graph.graph
    assert tuple(graph) == tuple(range(10))
    for node in graph:
        assert set(graph.neighbors(node)) == {
            2 * i + parity
            for i in ((node // 2 - 1) % 5, (node // 2 + 1) % 5)
            for parity in (0, 1)
        }
        assert all(Q(edge["weight"]) == 1 for edge in graph[node].values())
    assert cfg["GAMMA"]["type"] == "none" and cfg["INTEGRATOR_METHOD"] == "euler"
    assert cfg["CLIP_MODE"] == "hard" and cfg.get("MATH_ENGINE") is None
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise" and cfg["GRAMMAR_CANON"]["enabled"]
    assert cfg["PHASE_ADAPT"]["enabled"] is True
    assert cfg["NAV_RANDOM"] is True and cfg["NAV_STRICT"] is False
    assert not any(cfg.get("callbacks", {}).values())
    assert not cfg["history"].get("stable_frac") and not cfg.get("_epi_hist")
    assert "_REMESH_META" not in cfg
    return dict(
        pressure=tuple(
            domain.base._weight(graph, "DNFR_WEIGHTS", name)
            for name in ("epi", "vf", "phase")
        ),
        stability=tuple(
            domain.base._weight(graph, "SI_WEIGHTS", name)
            for name in ("alpha", "beta", "gamma")
        ),
        factors={name: Q(value) for name, value in cfg["GLYPH_FACTORS"].items()},
        thresholds={
            name: Q(value) for name, value in cfg["SELECTOR_THRESHOLDS"].items()
        },
        epsilon=Q(cfg["EPS_DNFR_STABLE"]),
        tau=cfg["VF_ADAPT_TAU"],
        mu=Q(cfg["VF_ADAPT_MU"]),
        form_rails=(Q(cfg["EPI_MIN"]), Q(cfg["EPI_MAX"])),
        capacity_rails=(Q(cfg["VF_MIN"]), Q(cfg["VF_MAX"])),
        lag_limits=(cfg["AL_MAX_LAG"], cfg["EN_MAX_LAG"]),
        phase_policy={
            name: Q(value)
            for name, value in cfg["PHASE_ADAPT"].items()
            if type(value) in (float, int)
        },
        phase_gains=(Q(cfg["PHASE_K_GLOBAL"]), Q(cfg["PHASE_K_LOCAL"])),
    )


def _predict(law, obs):
    """Interval admission and successor; hidden fine coordinates are absent."""
    k, b, w = law["pressure"]
    alpha, beta, gamma = law["stability"]
    f, cuts = law["factors"], law["thresholds"]
    m, v, dx, dv = obs.means, obs.capacities, obs.dx, obs.dv
    assert len(m) == len(v) == len(dx) == len(dv) == len(obs.phases) == 5
    assert all(value >= 0 for value in dx + dv)
    assert len(obs.counts) == len(obs.old_rates) == 10
    assert all(
        values and all(type(n) is int and n >= 0 for n in values)
        for values in obs.counts
    )
    assert obs.origin in {"supplied", "default"}
    assert all(trace == ("EN",) for trace in obs.traces)
    assert all(obs.prior) and all(value == 3 for value in obs.ordinals)
    assert obs.kinds == ("seed",) * 10 and obs.old_acceleration == (Q(0),) * 10
    assert obs.time == 1 and obs.initial_history == "empty"
    assert all(0 <= value <= U2_DEBT_CAPACITY for value in obs.debt)
    assert all(mark in {"AL", "EN", "IL", "RA", "NAV"} for mark in obs.marks)
    assert obs.innovation_mean.abs_max + obs.innovation_difference <= f["NAV_jitter"]
    assert max(obs.phases) - min(obs.phases) < PI
    for means, radii, rails in (
        (m, dx, law["form_rails"]),
        (v, dv, law["capacity_rails"]),
    ):
        if not all(
            rails[0] < value.lo - radius and value.hi + radius < rails[1]
            for value, radius in zip(means, radii)
        ):
            return _stop("source rails", None)
    if not all(value.lo > radius for value, radius in zip(v, dv)):
        return _stop("positive capacity", None)
    phase_offsets = tuple(
        (domain._neighbor(obs.phases, i) - obs.phases[i]) / PI for i in range(5)
    )
    baseline = tuple(
        k * (domain._neighbor(m, i) - m[i])
        + b * (domain._neighbor(v, i) - v[i])
        + w * phase_offsets[i]
        for i in range(5)
    )
    pressure_radius = tuple(k * x + b * y for x, y in zip(dx, dv))
    pressure = tuple(
        _around(value, radius) for value, radius in zip(baseline, pressure_radius)
    )
    peak_lower = max(abs(value).lo for value in baseline)
    peak_upper = max(
        value.abs_max + radius for value, radius in zip(baseline, pressure_radius)
    )
    vmax = I(
        max(value.lo for value in v),
        max(value.hi + radius for value, radius in zip(v, dv)),
    )
    ratio, si = [], []
    for i in range(5):
        if peak_upper == 0:
            normalized = I(0)
        elif peak_lower == 0:
            normalized = I(0, 1)
        else:
            normalized = I(
                abs(pressure[i]).lo / peak_upper,
                min(Q(1), pressure[i].abs_max / peak_lower),
            )
        ratio.append(normalized)
        si.append(
            alpha * (_around(v[i], dv[i]) / vmax)
            + beta * (1 - abs(phase_offsets[i]))
            + gamma * (1 - normalized)
        )
    # Effective origin is retained. Supplied marks do not acquire a selector
    # proof; default-origin requests require an entire-box branch certificate.
    request_origins = []
    for i, mark in enumerate(obs.marks):
        for node in (2 * i, 2 * i + 1):
            forced = None
            if obs.lag_al[node] + 1 > law["lag_limits"][0]:
                forced = expected = "AL"
            elif obs.lag_en[node] + 1 > law["lag_limits"][1]:
                forced = expected = "EN"
            elif si[i].lo >= cuts["si_hi"]:
                expected = "IL"
            elif si[i].lo > cuts["si_lo"] and si[i].hi < cuts["si_hi"]:
                expected = (
                    "NAV"
                    if ratio[i].lo > cuts["dnfr_hi"]
                    else "RA" if ratio[i].hi <= cuts["dnfr_hi"] else None
                )
            else:
                expected = None
            if forced is not None and forced != mark:
                return _stop("forced request", i)
            if obs.origin == "default" and expected != mark:
                return _stop("prepared request", i)
            request_origins.append(f"forced_{forced}" if forced else obs.origin)

    y, u, phi = list(m), list(v), list(obs.phases)
    form_factors, boost_factors, holds, capacity_radii = [], [], [], list(dv)
    qbar, covariance, next_means, next_dx, rate_means, rate_radii, resets = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    held = []
    for i, mark in enumerate(obs.marks):
        form_factor = Q(1)
        boost, hold, sign = Q(1), Q(1), 0
        signal = domain._neighbor(y, i)
        if mark == "AL":
            y[i] += f["AL_boost"]
            jump = I(f["AL_boost"])
        elif mark in {"EN", "RA"}:
            if mark == "RA":
                neighbors = tuple(
                    j
                    for j in ((i - 1) % 5, (i + 1) % 5)
                    if abs(scalar._wrap(phi[j] - phi[i])) <= PI / 2
                )
                if not neighbors:
                    return _stop("empty U3", i)
                signal = sum((y[j] for j in neighbors), I(0)) / len(neighbors)
                selected_phase = sum(phi[j] for j in neighbors) / len(neighbors)
                phi[i] += f["RA_phase_coupling"] * (selected_phase - phi[i])
                threshold = Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
                if abs(signal).lo > threshold:
                    boost += f["RA_vf_amplification"]
                elif abs(signal).hi > threshold:
                    return _stop("RA amplification", i)
            mix = f["EN_mix"] if mark == "EN" else f["RA_epi_diff"]
            form_factor = 1 - mix
            y[i] = form_factor * m[i] + mix * signal
            jump = _around(mix * (signal - m[i]), mix * dx[i])
            if mark == "RA":
                before, after = _around(m[i], dx[i]), _around(y[i], form_factor * dx[i])
                if not (
                    before.lo > 0 and after.lo > 0 or before.hi < 0 and after.hi < 0
                ):
                    return _stop("RA sign identity", i)
        else:
            jump = I(0)
        if mark == "IL":
            hold = f["IL_dnfr_factor"]
        elif mark == "NAV":
            if pressure[i].lo > 0:
                sign = 1
            elif pressure[i].hi < 0:
                sign = -1
            else:
                return _stop("NAV pressure sign", i)
            hold = 1 - f["NAV_eta"]
            action = f["NAV_eta"] * (sign * v[i] - baseline[i]) + obs.innovation_mean
            action_radius = (
                f["NAV_eta"] * (k * dx[i] + (b + 1) * dv[i]) + obs.innovation_difference
            )
            if _around(action, action_radius).contains(0):
                return _stop("NAV nonidentity", i)
        u[i], capacity_radii[i] = boost * v[i], boost * dv[i]
        pressure_mean = hold * baseline[i]
        z = Q(0)
        capacity_coefficient = -hold * b
        if mark == "NAV":
            pressure_mean += f["NAV_eta"] * sign * v[i] + obs.innovation_mean
            capacity_coefficient += f["NAV_eta"] * sign
            z = obs.innovation_difference
        zradius = hold * k * dx[i] + abs(capacity_coefficient) * dv[i] + z
        cov = boost * (
            I(-hold * k * dx[i] * dv[i], hold * k * dx[i] * dv[i])
            + capacity_coefficient * I(0, dv[i] ** 2)
            + I(-dv[i] * z, dv[i] * z)
        )
        rate_mean = u[i] * pressure_mean + cov
        rate_radius = u[i].abs_max * zradius + capacity_radii[i] * pressure_mean.abs_max
        drift_e = hold * (baseline[i] - b * v[i])
        if mark == "NAV":
            drift_e += 2 * f["NAV_eta"] * sign * v[i] + obs.innovation_mean
        radius = (
            (form_factor - DT * boost * hold * k * v[i]).abs_max * dx[i]
            + DT * boost * drift_e.abs_max * dv[i]
            + DT * boost * v[i].abs_max * z
        )
        next_means.append(y[i] + DT * rate_mean)
        next_dx.append(radius)
        qbar.append(pressure_mean)
        covariance.append(cov)
        held.append(_around(pressure_mean, zradius))
        rate_means.append(rate_mean)
        rate_radii.append(rate_radius)
        resets.append(
            frozenset((False,))
            if jump.abs_max == 0
            else (
                frozenset((True,)) if not jump.contains(0) else frozenset((False, True))
            )
        )
        form_factors.append(form_factor)
        boost_factors.append(boost)
        holds.append(hold)
        if (
            not law["form_rails"][0]
            < y[i].lo - form_factor * dx[i]
            <= y[i].hi + form_factor * dx[i]
            < law["form_rails"][1]
        ):
            return _stop("event form rails", i)
        if (
            not law["capacity_rails"][0]
            < u[i].lo - capacity_radii[i]
            <= u[i].hi + capacity_radii[i]
            < law["capacity_rails"][1]
        ):
            return _stop("event capacity rails", i)

    counts, due, qualification = [], [], []
    for node, possible in enumerate(obs.counts):
        i = node // 2
        if si[i].lo >= cuts["si_hi"] and held[i].abs_max <= law["epsilon"]:
            flags = frozenset((True,))
        elif si[i].hi < cuts["si_hi"] or abs(held[i]).lo > law["epsilon"]:
            flags = frozenset((False,))
        else:
            flags = frozenset((False, True))
        successors = frozenset(
            old + 1 if qualifies else 0 for old in possible for qualifies in flags
        )
        qualification.append(flags)
        counts.append(successors)
        due.append(frozenset(value >= law["tau"] for value in successors))
    next_v, next_dv = [], []
    for i in range(5):
        options, radii = [], []
        difference = domain._neighbor(u, i) - u[i]
        for even, odd in product(due[2 * i], due[2 * i + 1]):
            average, delta = Q(int(even) + int(odd), 2), Q(int(odd) - int(even), 2)
            options.append(
                _around(
                    u[i] + law["mu"] * average * difference,
                    law["mu"] * abs(delta) * capacity_radii[i],
                )
            )
            radii.append(
                abs(1 - law["mu"] * average) * capacity_radii[i]
                + law["mu"] * abs(delta) * difference.abs_max
            )
        next_v.append(_hull(options))
        next_dv.append(max(radii))
    for means, radii, rails in (
        (next_means, next_dx, law["form_rails"]),
        (next_v, next_dv, law["capacity_rails"]),
    ):
        if not all(
            rails[0] < value.lo - radius and value.hi + radius < rails[1]
            for value, radius in zip(means, radii)
        ):
            return _stop("successor rails", None)
    rate = tuple(_around(rate_means[n // 2], rate_radii[n // 2]) for n in range(10))
    traces = tuple((*trace, obs.marks[n // 2]) for n, trace in enumerate(obs.traces))
    load = Q(
        sum(mark == "NAV" for trace in traces for mark in trace), sum(map(len, traces))
    )
    policy = law["phase_policy"]
    resultant_floor = 1 - (max(phi) - min(phi)) ** 2 / 8
    if not resultant_floor > policy["R_hi"] or load > policy["disr_lo"]:
        return _stop("phase policy", None)
    kg, kl = tuple(
        value + policy["down"] * (policy[f"k{suffix}_min"] - value)
        for value, suffix in zip(law["phase_gains"], ("G", "L"))
    )
    assert 0 < kg and 0 < kl and kg + kl < 1
    relative_phase = tuple(
        (1 - kg - kl) * value + kl * domain._neighbor(phi, i)
        for i, value in enumerate(phi)
    )
    # All block phases share this single resultant-origin term. It is not
    # replaced by unrelated per-block angles or asserted to be a point.
    shared_phase_origin = kg * I(min(phi), max(phi))
    # Empty admitted incoming histories acquire a current-time boundary.
    # Keep associated left/rate/right envelopes and the exact secant relation;
    # arbitrary combinations of their marginals are not asserted attainable.
    temporal = tuple(
        dict(
            times=(obs.time, obs.time + DT),
            left=_around(y[n // 2], form_factors[n // 2] * dx[n // 2]),
            rate=rate[n],
            right=_around(next_means[n // 2], next_dx[n // 2]),
            relation="right=left+DT*rate",
        )
        for n in range(10)
    )
    return dict(
        status="enclosed",
        means=tuple(next_means),
        capacities=tuple(next_v),
        dx=tuple(next_dx),
        dv=tuple(next_dv),
        pressure=pressure,
        peak=I(peak_lower, peak_upper),
        vmax=vmax,
        ratio=tuple(ratio),
        si=tuple(si),
        covariance=tuple(covariance),
        held=tuple(held),
        rate=rate,
        acceleration=tuple(
            (value - old) / DT for value, old in zip(rate, obs.old_rates)
        ),
        counts=tuple(counts),
        due=tuple(due),
        qualification=tuple(qualification),
        resets=tuple(resets),
        event_phases=tuple(phi),
        boost=tuple(boost_factors),
        form_factors=tuple(form_factors),
        request_origins=tuple(request_origins),
        phase_origin=shared_phase_origin,
        relative_phase=relative_phase,
        phase_load=load,
        phase_gains=(kg, kl),
        temporal=temporal,
        traces=traces,
        debt=tuple(
            0 if obs.marks[n // 2] == "IL" else value
            for n, value in enumerate(obs.debt)
        ),
        ordinals=tuple(value + 1 for value in obs.ordinals),
        lag_al=tuple(
            0 if obs.marks[n // 2] == "AL" else value + 1
            for n, value in enumerate(obs.lag_al)
        ),
        lag_en=tuple(
            (
                0
                if obs.marks[n // 2] == "EN"
                else (
                    min(value + 1, law["lag_limits"][1])
                    if obs.marks[n // 2] == "AL"
                    else value + 1
                )
            )
            for n, value in enumerate(obs.lag_en)
        ),
    )


@pytest.fixture(scope="module")
def frozen(configuration, enclosure_guard):
    law = _law(configuration)
    ctx = dict(
        counts=(frozenset((0,)),) * 10,
        old_rates=tuple(I(Q(n - 4, 64)) for n in range(10)),
        traces=(("EN",),) * 10,
        debt=tuple(n // 2 % 2 for n in range(10)),
        prior=(True,) * 10,
        ordinals=(3,) * 10,
        lag_al=(0,) * 10,
        lag_en=(0,) * 10,
        kinds=("seed",) * 10,
        old_acceleration=(Q(0),) * 10,
        time=Q(1),
        initial_history="empty",
    )
    jitter = law["factors"]["NAV_jitter"]
    mixed = Observation(
        means=tuple(
            I(value - MEAN_RADIUS, value + MEAN_RADIUS) for value in scalar.MEANS
        ),
        capacities=tuple(
            I(value - MEAN_RADIUS, value + MEAN_RADIUS) for value in scalar.CAPACITIES
        ),
        dx=tuple(map(abs, scalar.CONTRASTS)),
        dv=tuple(map(abs, exact.CAPACITY_CONTRASTS)),
        phases=exact.PHASES,
        marks=scalar.SCHEDULE,
        origin="supplied",
        innovation_mean=I(-jitter / 4, jitter / 4),
        innovation_difference=jitter / 2,
        **ctx,
    )
    q = prefix.SOURCE_Q
    actual = replace(
        mixed,
        means=tuple(map(I, (q, q, -q, Q(-1, 2), -q))),
        capacities=tuple(map(I, prefix.CAPACITY)),
        dx=(Q(1, 4096),) * 5,
        dv=(Q(1, 4096),) * 5,
        phases=(Q(0),) * 5,
        marks=("RA", "RA", "IL", "IL", "IL"),
        origin="default",
        innovation_mean=I(0),
        innovation_difference=Q(0),
        old_rates=tuple(I(Q(value)) for value in domain._paired(domain.base.RATES)),
    )
    ambiguous = replace(
        actual,
        means=tuple(map(I, joint.feedback.MEANS)),
        capacities=tuple(map(I, joint.feedback.CAPACITIES)),
        dx=joint.AMBIGUOUS_RADII,
        dv=(Q(0),) * 5,
        counts=tuple(frozenset((value,)) for value in joint.AMBIGUOUS_COUNTS),
        marks=("IL",) * 5,
    )
    observations = (mixed, actual, ambiguous)
    # All laws, domains, numerical budgets and all three coarse outputs are
    # fixed before any discarded coordinate is chosen or fine step evaluated.
    forecasts = tuple(_predict(law, obs) for obs in observations)
    assert all(value["status"] == "enclosed" for value in forecasts)
    for result in forecasts:
        assert max(value.width for value in result["means"]) < WIDTH_BUDGET
        assert max(result["dx"]) < FORM_RADIUS_BUDGET
        assert (
            max(value.width for value in result["capacities"]) < CAPACITY_WIDTH_BUDGET
        )
    assert forecasts[2]["due"][1] == frozenset((False, True))
    assert all(
        possible == frozenset((False,))
        for result in forecasts[:2]
        for possible in result["due"]
    )
    digest = sha256(
        repr(
            (
                law,
                observations,
                forecasts,
                INTERVAL_METHOD,
                WIDTH_BUDGET,
                FORM_RADIUS_BUDGET,
                CAPACITY_WIDTH_BUDGET,
            )
        ).encode()
    ).hexdigest()
    return dict(
        law=law,
        observations=observations,
        forecasts=forecasts,
        digest=digest,
        graph=configuration,
    )


def _fine_control(graph, obs, means, capacities, d, e, noise_mean, noise_difference):
    """Independent fine formula evaluation after coarse protocol freezing."""
    cfg, f = graph.graph, graph.graph["GLYPH_FACTORS"]
    forms, vf = scalar._fine(means, d), scalar._fine(capacities, e)
    phi = list(domain._paired(obs.phases))
    pressure = exact._fresh(graph, forms, vf, phi)
    peak = max(map(abs, pressure)) or Q(1)
    alpha, beta, gamma = (
        domain.base._weight(graph, "SI_WEIGHTS", name)
        for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * vf[n] / max(vf)
        + beta
        * (1 - abs(domain._neighbor(obs.phases, n // 2) - obs.phases[n // 2]) / PI)
        + gamma * (1 - abs(pressure[n]) / peak)
        for n in graph
    )
    x, v, held = list(forms), list(vf), list(pressure)
    for n in graph:
        mark = obs.marks[n // 2]
        signal = sum(x[j] for j in graph.neighbors(n)) / 4
        if mark == "AL":
            x[n] += Q(f["AL_boost"])
        elif mark in {"EN", "RA"}:
            mix = Q(f["EN_mix"] if mark == "EN" else f["RA_epi_diff"])
            before = x[n]
            x[n] = (1 - mix) * before + mix * signal
            if mark == "RA":
                assert before * x[n] > 0
                assert all(
                    abs(scalar._wrap(phi[j] - phi[n])) <= PI / 2
                    for j in graph.neighbors(n)
                )
                phi[n] += Q(f["RA_phase_coupling"]) * (
                    sum(phi[j] for j in graph.neighbors(n)) / 4 - phi[n]
                )
                if abs(signal) > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER):
                    v[n] *= 1 + Q(f["RA_vf_amplification"])
        elif mark == "IL":
            held[n] *= Q(f["IL_dnfr_factor"])
        else:
            eta = Q(f["NAV_eta"])
            innovation = noise_mean + (-1 if n % 2 == 0 else 1) * noise_difference
            assert abs(innovation) <= Q(f["NAV_jitter"])
            held[n] = (
                (1 - eta) * held[n]
                + eta * v[n] * (1 if held[n] >= 0 else -1)
                + innovation
            )
            assert held[n] != pressure[n]
    rate = tuple(capacity * p for capacity, p in zip(v, held))
    endpoint = tuple(value + DT * velocity for value, velocity in zip(x, rate))
    qualified = tuple(
        s >= Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
        and abs(p) <= Q(cfg["EPS_DNFR_STABLE"])
        for s, p in zip(si, held)
    )
    assert all(len(values) == 1 for values in obs.counts)
    counts = tuple(
        next(iter(values)) + 1 if passes else 0
        for values, passes in zip(obs.counts, qualified)
    )
    due = tuple(value >= cfg["VF_ADAPT_TAU"] for value in counts)
    snapshot = tuple(v)
    adapted = tuple(
        (
            capacity
            + Q(cfg["VF_ADAPT_MU"])
            * (sum(snapshot[j] for j in graph.neighbors(n)) / 4 - capacity)
            if due[n]
            else capacity
        )
        for n, capacity in enumerate(snapshot)
    )
    return dict(
        forms=endpoint,
        capacities=adapted,
        pressure=pressure,
        si=si,
        held=tuple(held),
        rate=rate,
        acceleration=tuple(
            (value - old.midpoint) / DT for value, old in zip(rate, obs.old_rates)
        ),
        counts=counts,
        due=due,
        qualification=qualified,
        resets=tuple(a != b for a, b in zip(forms, x)),
        event_phases=tuple(phi),
        event_forms=tuple(x),
    )


def _assert_contains(obs, forecast, result):
    for field, center, radius in (
        ("forms", "means", "dx"),
        ("capacities", "capacities", "dv"),
    ):
        means, differences = scalar._moments(result[field])
        for i, (mean, difference) in enumerate(zip(means, differences)):
            assert forecast[center][i].contains(mean)
            assert abs(difference) <= forecast[radius][i]
    for field in ("pressure", "si", "held"):
        for n, value in enumerate(result[field]):
            assert forecast[field][n // 2].contains(value)
    for field in ("rate", "acceleration"):
        for enclosure, value in zip(forecast[field], result[field]):
            assert enclosure.contains(value)
    for field in ("counts", "due", "qualification"):
        assert all(
            value in possibilities
            for value, possibilities in zip(result[field], forecast[field])
        )
    assert all(
        value in forecast["resets"][n // 2] for n, value in enumerate(result["resets"])
    )
    assert result["event_phases"] == domain._paired(forecast["event_phases"])
    for n, envelope in enumerate(forecast["temporal"]):
        assert envelope["times"] == (obs.time, obs.time + DT)
        assert envelope["left"].contains(result["event_forms"][n])
        assert envelope["right"].contains(result["forms"][n])
        assert envelope["rate"].contains(result["rate"][n])
        assert result["forms"][n] == result["event_forms"][n] + DT * result["rate"][n]


@pytest.fixture(scope="module")
def controls(frozen):
    results = [[], [], []]
    for index, obs in enumerate(frozen["observations"]):
        if index == 0:
            choices = product((-1, 1), repeat=5)
        elif index == 1:
            choices = ((a, b, 0, 0, 0) for a, b in product((-1, 1), repeat=2))
        else:
            choices = ((a, 0, 0, 0, 0) for a in (-1, 1))
        for sd, se, sm, sn, sz in choices:
            means = tuple(
                value.midpoint + sm * (-1) ** i * value.radius
                for i, value in enumerate(obs.means)
            )
            capacities = tuple(
                value.midpoint - sm * (-1) ** i * value.radius
                for i, value in enumerate(obs.capacities)
            )
            d = tuple(sd * (-1) ** i * radius for i, radius in enumerate(obs.dx))
            e = tuple(se * (-1) ** i * radius for i, radius in enumerate(obs.dv))
            noise_mean = obs.innovation_mean.midpoint + sn * obs.innovation_mean.radius
            noise_difference = sz * obs.innovation_difference
            result = _fine_control(
                frozen["graph"],
                obs,
                means,
                capacities,
                d,
                e,
                noise_mean,
                noise_difference,
            )
            _assert_contains(obs, frozen["forecasts"][index], result)
            results[index].append(result)
    return tuple(tuple(group) for group in results)


def test_protocol_and_lossy_interface_precede_every_fine_control(
    frozen, enclosure_guard
):
    targets, forbidden = enclosure_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert INTERVAL_BITS == 128 and len(frozen["digest"]) == 64
    assert not {
        "forms",
        "signed_d",
        "signed_e",
        "pressure",
        "si",
        "graph",
    }.intersection(Observation.__dataclass_fields__)
    assert all(value["status"] == "enclosed" for value in frozen["forecasts"])
    assert frozen["observations"][0].origin == "supplied"
    assert tuple(obs.origin for obs in frozen["observations"][1:]) == (
        "default",
        "default",
    )


def test_five_operator_enclosure_keeps_all_fixed_fine_controls(frozen, controls):
    assert tuple(map(len, controls)) == (32, 4, 2)
    result = frozen["forecasts"][0]
    assert max(value.width for value in result["means"]) < WIDTH_BUDGET
    assert max(result["dx"]) < FORM_RADIUS_BUDGET
    assert all(value == frozenset((False,)) for value in result["due"])
    assert len({scalar._moments(control["forms"])[0] for control in controls[0]}) > 1
    assert any(value.width > 0 for value in result["covariance"])


def test_default_live_prefix_is_enclosed_without_signed_form_or_capacity(
    frozen, controls
):
    result = frozen["forecasts"][1]
    assert result["boost"][0] == 1
    assert result["boost"][1] == 1 + frozen["law"]["factors"]["RA_vf_amplification"]
    assert max(result["dx"]) < Q(1, 4096)
    assert all(value.width > 0 for value in result["means"])
    assert all(flags == frozenset((False,)) for flags in result["qualification"])
    assert len({scalar._moments(control["forms"])[0] for control in controls[1]}) > 1


def test_ambiguous_individual_counts_preserve_both_adaptation_outcomes(
    frozen, controls
):
    obs, result = frozen["observations"][2], frozen["forecasts"][2]
    assert obs.counts[0] == frozenset((0,)) and obs.counts[1] == frozenset((4,))
    assert result["due"][0] == frozenset((False,))
    assert result["due"][1] == frozenset((False, True))
    assert {value["due"][1] for value in controls[2]} == {False, True}
    assert result["capacities"][0].width > 0 and result["dv"][0] > 0


def test_radius_does_not_supply_a_positive_normalization_peak(frozen):
    obs = frozen["observations"][2]
    uniform = replace(
        obs,
        means=(I(Q(1, 8)),) * 5,
        capacities=(I(1),) * 5,
        dx=(Q(1, 4096),) * 5,
        dv=(Q(0),) * 5,
        counts=(frozenset((0,)),) * 10,
    )
    result = _predict(frozen["law"], uniform)
    assert result["status"] == "enclosed" and result["peak"].lo == 0 < result["peak"].hi
    assert all(value == I(0, 1) for value in result["ratio"])
    assert (
        min(value.lo for value in result["si"]) > frozen["law"]["thresholds"]["si_hi"]
    )


def test_unresolved_transition_sign_stops_before_inventing_a_successor(frozen):
    original = frozen["observations"][0]
    # Enlarge only the declared hidden-form budget at the NAV pair. This is
    # a separate analytical boundary control, not retuning the frozen class.
    radii = (*original.dx[:3], Q(1, 8), original.dx[4])
    result = _predict(frozen["law"], replace(original, dx=radii))
    assert result == dict(status="unresolved", gate="NAV pressure sign", block=3)
    assert "means" not in result


def test_forced_priority_applies_to_supplied_requests_and_emission_caps_reception_lag(
    frozen,
):
    obs, law = frozen["observations"][0], frozen["law"]
    al_limit, en_limit = law["lag_limits"]
    reception_due = replace(obs, lag_en=(en_limit, en_limit, *(0,) * 8))
    assert _predict(law, reception_due) == dict(
        status="unresolved", gate="forced request", block=0
    )
    both_due = replace(reception_due, lag_al=(al_limit, al_limit, *(0,) * 8))
    result = _predict(law, both_due)
    assert result["status"] == "enclosed"
    assert result["lag_al"][:2] == (0, 0)
    assert result["lag_en"][:2] == (en_limit, en_limit)
    assert result["request_origins"] == ("forced_AL", "forced_AL", *("supplied",) * 8)


def test_phase_origin_and_empty_history_relations_remain_associated(frozen):
    for obs, result in zip(frozen["observations"], frozen["forecasts"]):
        assert obs.initial_history == "empty" and obs.time == 1
        assert len(result["relative_phase"]) == 5
        assert result["phase_origin"].lo >= 0
        assert sum(result["phase_gains"]) < 1
        assert all(
            value["relation"] == "right=left+DT*rate" for value in result["temporal"]
        )
        assert all(value["times"] == (Q(1), Q(17, 16)) for value in result["temporal"])
        assert result["ordinals"] == (4,) * 10
    assert frozen["forecasts"][0]["phase_load"] == Q(1, 10)
    assert frozen["forecasts"][0]["phase_origin"].width > 0
    assert all(result["phase_origin"] == I(0) for result in frozen["forecasts"][1:])
