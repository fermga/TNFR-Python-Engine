"""Two-call lossy composition with associated memory and no native execution.

Freeze the prior actual live-prefix observation, doubled C5, represented
defaults, dt=1/16 and four deterministic lifts d_i=s_d*(-1)**i/4096,
e_i=s_e*(-1)**i/4096. The second preparation receives only the first abstract
successor and its associated histories/context. No discarded signed contrast,
exact endpoint, fine diagnostic or fine-selected request enters that observer.

Before fine controls, require final form/capacity mean widths <1e-7, form
radii <1/4096, capacity radii <1/3500 and strict rails. Exact containment adds
no tolerance. The new continuation is scoped to a fully admitted RA/IL batch;
an unresolved consumed observation or another effective mark stops it.
The frozen first-call producer and its response protocols are not modified.

History samples carry variable identities and flow relations, not independent
marginal choices. The same current form variable terminates the incoming
history. Phase retains its common-origin interval, relative coordinates and
current gains. The declared horizon ends after this second complete call.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product

import pytest

from tests.physics import test_native_collective_enclosure as first
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.mathematics._rational_interval import INTERVAL_METHOD, I
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER

domain, scalar, exact = first.domain, first.scalar, first.exact
no_execution = first.no_execution
calculus_guard = first.calculus_guard
enclosure_guard = first.enclosure_guard
configuration = first.configuration
first_preparation = first.frozen
DT, PI = first.DT, first.PI
MEAN_WIDTH_LIMIT = Q(1, 10**7)
FORM_RADIUS_LIMIT, CAPACITY_RADIUS_LIMIT = Q(1, 4096), Q(1, 3500)
LIFTS = tuple(product((-1, 1), repeat=2))


@dataclass(frozen=True)
class Sample:
    time: Q
    enclosure: I
    variable: tuple


@dataclass(frozen=True)
class FlowRelation:
    """The retained constraint is right = left + duration * rate."""

    left: tuple
    right: tuple
    duration: Q
    rate: I
    rate_variable: tuple

    def contains(self, assignments):
        velocity = assignments[self.rate_variable]
        return (
            self.rate.contains(velocity)
            and assignments[self.right]
            == assignments[self.left] + self.duration * velocity
        )


@dataclass(frozen=True)
class History:
    samples: tuple
    relations: tuple


@dataclass(frozen=True)
class AbstractState:
    time: Q
    runtime_ordinal: int
    means: tuple
    capacities: tuple
    dx: tuple
    dv: tuple
    form_variables: tuple
    rate_variables: tuple
    rates: tuple
    accelerations: tuple
    counts: tuple
    traces: tuple
    debt: tuple
    prior: tuple
    kinds: tuple
    ordinals: tuple
    lag_al: tuple
    lag_en: tuple
    phase_origin: I
    relative_phases: tuple
    phase_gains: tuple
    histories: tuple
    remesh_snapshots: tuple
    stable_fraction_history: tuple


def _form_variables(time):
    return tuple(("form", time, n) for n in range(10))


def _rate_variables(time):
    return tuple(("rate", time, n) for n in range(10))


def _form_boxes(state):
    return tuple(
        first._around(state.means[n // 2], state.dx[n // 2]) for n in range(10)
    )


def _first_successor(preparation):
    """Lift already-proved abstract output, without choosing a fine member."""
    obs, result = preparation["observations"][1], preparation["forecasts"][1]
    time = obs.time + DT
    variables = _form_variables(time)
    rate_variables = _rate_variables(time)
    histories = []
    for node, envelope in enumerate(result["temporal"]):
        left = Sample(obs.time, envelope["left"], ("event", obs.time, node))
        right = Sample(time, envelope["right"], variables[node])
        relation = FlowRelation(
            left.variable, right.variable, DT, envelope["rate"], rate_variables[node]
        )
        histories.append(History((left, right), (relation,)))
    snapshots = (tuple(history.samples[-1] for history in histories),)
    return AbstractState(
        time,
        1,
        result["means"],
        result["capacities"],
        result["dx"],
        result["dv"],
        variables,
        rate_variables,
        result["rate"],
        result["acceleration"],
        result["counts"],
        result["traces"],
        result["debt"],
        obs.prior,
        obs.kinds,
        result["ordinals"],
        result["lag_al"],
        result["lag_en"],
        result["phase_origin"],
        result["relative_phase"],
        result["phase_gains"],
        tuple(histories),
        snapshots,
        (),
    )


def _admit_memory(law, state):
    for values, length in (
        (state.means, 5),
        (state.capacities, 5),
        (state.dx, 5),
        (state.dv, 5),
        (state.relative_phases, 5),
        (state.phase_gains, 2),
        *(
            (getattr(state, name), 10)
            for name in (
                "form_variables",
                "rate_variables",
                "rates",
                "accelerations",
                "counts",
                "traces",
                "debt",
                "prior",
                "kinds",
                "ordinals",
                "lag_al",
                "lag_en",
                "histories",
            )
        ),
    ):
        if len(values) != length:
            raise ValueError("unsupported retained state shape")
    if not all(isinstance(value, I) for value in (*state.means, *state.capacities)):
        raise ValueError("unadmitted mean envelope")
    if not all(type(value) is Q and value >= 0 for value in (*state.dx, *state.dv)):
        raise ValueError("unadmitted contrast radius")
    if not state.stable_fraction_history == ():
        raise ValueError("REMESH stability history is outside this scope")
    if (
        type(state.time) is not Q
        or state.time <= 0
        or type(state.runtime_ordinal) is not int
        or state.runtime_ordinal != 1
    ):
        raise ValueError("this gate admits only the second call")
    if not state.kinds == ("seed",) * 10 or state.prior != (True,) * 10:
        raise ValueError("common kind and prior coherence are required")
    if not all(
        type(value) is int and 0 <= value <= U2_DEBT_CAPACITY for value in state.debt
    ):
        raise ValueError("invalid grammar debt")
    if (
        not all(type(step) is int for step in state.ordinals)
        or state.ordinals != (4,) * 10
    ):
        raise ValueError("invalid retained grammar ordinal")
    expected_traces = tuple(("EN", "RA") if n < 4 else ("EN", "IL") for n in range(10))
    if state.traces != expected_traces or state.debt != tuple(
        int(n // 2 == 1) for n in range(10)
    ):
        raise ValueError("unsupported retained grammar context")
    if not all(
        type(value) is int and value >= 0 for value in (*state.lag_al, *state.lag_en)
    ):
        raise ValueError("invalid retained lag")
    if not isinstance(state.phase_origin, I) or not all(
        type(value) is Q for value in state.relative_phases
    ):
        raise ValueError("unadmitted phase frame")
    if state.phase_origin != I(0) or state.relative_phases != (Q(0),) * 5:
        raise ValueError("unsupported propagated phase sector")
    if (
        not all(
            type(value) is Q
            and 0 < value
            and law["phase_policy"][f"k{suffix}_min"]
            <= value
            <= law["phase_policy"][f"k{suffix}_max"]
            for value, suffix in zip(state.phase_gains, ("G", "L"))
        )
        or sum(state.phase_gains) >= 1
    ):
        raise ValueError("unadmitted retained phase gains")
    if not all(
        values and all(type(value) is int and value >= 0 for value in values)
        for values in state.counts
    ):
        raise ValueError("invalid associated counters")
    boxes = _form_boxes(state)
    for n, history in enumerate(state.histories):
        if len(history.samples) != 2:
            raise ValueError("missing temporal evidence")
        if history.samples[-1].time != state.time:
            raise ValueError("latest sample time is stale")
        if history.samples[-1].variable != state.form_variables[n]:
            raise ValueError("current-form variable association is missing")
        if history.samples[-1].enclosure != boxes[n]:
            raise ValueError("current-form envelope association is missing")
        if len(history.relations) != len(history.samples) - 1:
            raise ValueError("missing flow relation")
        for left, right, relation in zip(
            history.samples, history.samples[1:], history.relations
        ):
            if (
                not left.time < right.time
                or relation.duration != right.time - left.time
            ):
                raise ValueError("invalid temporal clock")
            if (relation.left, relation.right) != (left.variable, right.variable):
                raise ValueError("broken flow variable association")
            possible = left.enclosure + relation.duration * relation.rate
            if right.enclosure.hi < possible.lo or possible.hi < right.enclosure.lo:
                raise ValueError("inconsistent temporal relation")
        if history.relations[-1].rate != state.rates[n]:
            raise ValueError("retained rate does not match latest flow")
        if history.relations[-1].rate_variable != state.rate_variables[n]:
            raise ValueError("retained rate variable association is missing")
        # The default choice reads and normalizes old acceleration although
        # its base decision does not use it. Retain finite admitted intervals.
        assert isinstance(state.accelerations[n], I) and isinstance(state.rates[n], I)
    if state.remesh_snapshots[-1] != tuple(
        history.samples[-1] for history in state.histories
    ):
        raise ValueError("REMESH snapshot is not associated with current form")


def _second_step(law, state):
    """Scoped continuation with genuine propagated state, not reset defaults."""
    _admit_memory(law, state)
    k, b, w = law["pressure"]
    alpha, beta, gamma = law["stability"]
    f, cuts = law["factors"], law["thresholds"]
    m, v, dx, dv = state.means, state.capacities, state.dx, state.dv
    for values, radii_, rails in (
        (m, dx, law["form_rails"]),
        (v, dv, law["capacity_rails"]),
    ):
        if not all(
            rails[0] < value.lo - radius and value.hi + radius < rails[1]
            for value, radius in zip(values, radii_)
        ):
            return first._stop("source rails", None)
    if not all(value.lo > radius for value, radius in zip(v, dv)):
        return first._stop("positive capacity", None)
    phases = state.relative_phases
    assert max(phases) - min(phases) < PI
    # The common inherited origin cancels from all fresh phase offsets.
    offsets = tuple((domain._neighbor(phases, i) - phases[i]) / PI for i in range(5))
    baseline = tuple(
        k * (domain._neighbor(m, i) - m[i])
        + b * (domain._neighbor(v, i) - v[i])
        + w * offsets[i]
        for i in range(5)
    )
    radii = tuple(k * x + b * y for x, y in zip(dx, dv))
    pressure = tuple(
        first._around(value, radius) for value, radius in zip(baseline, radii)
    )
    peak = I(
        max(abs(value).lo for value in baseline),
        max(value.abs_max + radius for value, radius in zip(baseline, radii)),
    )
    if peak.lo <= 0:
        return first._stop("positive normalizer", None)
    vmax = I(
        max(value.lo for value in v),
        max(value.hi + radius for value, radius in zip(v, dv)),
    )
    ratios = tuple(
        I(abs(value).lo / peak.hi, min(Q(1), value.abs_max / peak.lo))
        for value in pressure
    )
    si = tuple(
        alpha * (first._around(v[i], dv[i]) / vmax)
        + beta * (1 - abs(offsets[i]))
        + gamma * (1 - ratios[i])
        for i in range(5)
    )
    marks, origins = [], []
    for i in range(5):
        pair = []
        for n in (2 * i, 2 * i + 1):
            if (
                state.lag_al[n] + 1 > law["lag_limits"][0]
                or state.lag_en[n] + 1 > law["lag_limits"][1]
            ):
                return dict(status="outside_sector", gate="forced request", block=i)
            if si[i].lo >= cuts["si_hi"]:
                mark = "IL"
            elif (
                si[i].lo > cuts["si_lo"]
                and si[i].hi < cuts["si_hi"]
                and ratios[i].hi <= cuts["dnfr_hi"]
            ):
                mark = "RA"
            else:
                return first._stop("prepared request", i)
            pair.append(mark)
            origins.append("default")
        assert pair[0] == pair[1]
        marks.append(pair[0])
    y, u, phi = list(m), list(v), list(phases)
    (
        boosts,
        form_factors,
        held,
        qbar,
        covariance,
        next_m,
        next_dx,
        rate_mean,
        rate_radius,
    ) = ([], [], [], [], [], [], [], [], [])
    resets, signals, event_boxes = [], [], []
    for i, mark in enumerate(marks):
        boost, a, h = Q(1), Q(1), Q(1)
        signal = I(0)
        if mark == "RA":
            neighbors = tuple(
                j
                for j in ((i - 1) % 5, (i + 1) % 5)
                if abs(scalar._wrap(phi[j] - phi[i])) <= PI / 2
            )
            if not neighbors:
                return first._stop("empty U3", i)
            signal = sum((y[j] for j in neighbors), I(0)) / len(neighbors)
            threshold = Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
            if abs(signal).lo > threshold:
                boost += f["RA_vf_amplification"]
            elif abs(signal).hi > threshold:
                return first._stop("RA amplification", i)
            a = 1 - f["RA_epi_diff"]
            y[i] = a * m[i] + f["RA_epi_diff"] * signal
            before, after = first._around(m[i], dx[i]), first._around(y[i], a * dx[i])
            if not (before.lo > 0 and after.lo > 0 or before.hi < 0 and after.hi < 0):
                return first._stop("RA sign identity", i)
            jump = first._around(
                f["RA_epi_diff"] * (signal - m[i]), f["RA_epi_diff"] * dx[i]
            )
            if jump.contains(0):
                return first._stop("temporal reset", i)
            resets.append(True)
            phi[i] += f["RA_phase_coupling"] * (
                sum(phi[j] for j in neighbors) / len(neighbors) - phi[i]
            )
        else:
            h = f["IL_dnfr_factor"]
            resets.append(False)
        u[i] = boost * v[i]
        pressure_mean = h * baseline[i]
        pressure_radius = h * (k * dx[i] + b * dv[i])
        cov = boost * h * I(-k * dx[i] * dv[i] - b * dv[i] ** 2, k * dx[i] * dv[i])
        rmean = u[i] * pressure_mean + cov
        rradius = u[i].abs_max * pressure_radius + boost * dv[i] * pressure_mean.abs_max
        next_m.append(y[i] + DT * rmean)
        next_dx.append(
            (a - DT * boost * h * k * v[i]).abs_max * dx[i]
            + DT * boost * h * (baseline[i] - b * v[i]).abs_max * dv[i]
        )
        event_boxes.append(first._around(y[i], a * dx[i]))
        qbar.append(pressure_mean)
        held.append(first._around(pressure_mean, pressure_radius))
        covariance.append(cov)
        rate_mean.append(rmean)
        rate_radius.append(rradius)
        boosts.append(boost)
        form_factors.append(a)
        signals.append(signal)
        if (
            not law["form_rails"][0]
            < event_boxes[-1].lo
            <= event_boxes[-1].hi
            < law["form_rails"][1]
        ):
            return first._stop("event form rails", i)
        if (
            not law["capacity_rails"][0]
            < u[i].lo - boost * dv[i]
            <= u[i].hi + boost * dv[i]
            < law["capacity_rails"][1]
        ):
            return first._stop("event capacity rails", i)
    # This scoped certificate continues only when every individual fails the
    # complete stability predicate, rather than guessing capacity adaptation.
    for i in range(5):
        if not (si[i].hi < cuts["si_hi"] or abs(held[i]).lo > law["epsilon"]):
            return first._stop("adaptation qualification", i)
    next_dv = tuple(boost * radius for boost, radius in zip(boosts, dv))
    for means, radii_, rails in (
        (next_m, next_dx, law["form_rails"]),
        (u, next_dv, law["capacity_rails"]),
    ):
        if not all(
            rails[0] < value.lo - radius and value.hi + radius < rails[1]
            for value, radius in zip(means, radii_)
        ):
            return first._stop("successor rails", None)
    rates = tuple(
        first._around(rate_mean[n // 2], rate_radius[n // 2]) for n in range(10)
    )
    accelerations = tuple((value - old) / DT for value, old in zip(rates, state.rates))
    traces = tuple((*trace, marks[n // 2]) for n, trace in enumerate(state.traces))
    policy = law["phase_policy"]
    load = Q(
        sum(mark == "NAV" for trace in traces for mark in trace), sum(map(len, traces))
    )
    if (
        not 1 - (max(phi) - min(phi)) ** 2 / 8 > policy["R_hi"]
        or load > policy["disr_lo"]
    ):
        return first._stop("phase policy", None)
    kg, kl = tuple(
        value + policy["down"] * (policy[f"k{suffix}_min"] - value)
        for value, suffix in zip(state.phase_gains, ("G", "L"))
    )
    assert 0 < kg and 0 < kl and kg + kl < 1
    origin = state.phase_origin + kg * I(min(phi), max(phi))
    relative = tuple(
        (1 - kg - kl) * value + kl * domain._neighbor(phi, i)
        for i, value in enumerate(phi)
    )
    time, histories = state.time + DT, []
    variables = _form_variables(time)
    rate_variables = _rate_variables(time)
    for n in range(10):
        right = Sample(
            time, first._around(next_m[n // 2], next_dx[n // 2]), variables[n]
        )
        if resets[n // 2]:
            left = Sample(state.time, event_boxes[n // 2], ("event", state.time, n))
            samples, relations = (left,), ()
        else:
            samples, relations = (
                state.histories[n].samples,
                state.histories[n].relations,
            )
        relation = FlowRelation(
            samples[-1].variable, right.variable, DT, rates[n], rate_variables[n]
        )
        histories.append(History((*samples, right), (*relations, relation)))
        assert len(histories[-1].samples) <= 4
    snapshots = (
        *state.remesh_snapshots,
        tuple(history.samples[-1] for history in histories),
    )
    successor = AbstractState(
        time,
        state.runtime_ordinal + 1,
        tuple(next_m),
        tuple(u),
        tuple(next_dx),
        next_dv,
        variables,
        rate_variables,
        rates,
        accelerations,
        (frozenset((0,)),) * 10,
        traces,
        tuple(
            max(0, value - 1) if marks[n // 2] == "IL" else value
            for n, value in enumerate(state.debt)
        ),
        state.prior,
        state.kinds,
        tuple(value + 1 for value in state.ordinals),
        tuple(value + 1 for value in state.lag_al),
        tuple(value + 1 for value in state.lag_en),
        origin,
        relative,
        (kg, kl),
        tuple(histories),
        snapshots,
        (),
    )
    return dict(
        status="enclosed",
        state=successor,
        marks=tuple(marks),
        origins=tuple(origins),
        pressure=pressure,
        ratios=ratios,
        peak=peak,
        si=si,
        signals=tuple(signals),
        boosts=tuple(boosts),
        held=tuple(held),
        covariance=tuple(covariance),
        resets=tuple(resets),
        event_boxes=tuple(event_boxes),
    )


@pytest.fixture(scope="module")
def frozen(first_preparation, enclosure_guard):
    source = _first_successor(first_preparation)
    law = first_preparation["law"]
    result = _second_step(law, source)
    assert result["status"] == "enclosed"
    state = result["state"]
    assert max(value.width for value in state.means) < MEAN_WIDTH_LIMIT
    assert max(value.width for value in state.capacities) < MEAN_WIDTH_LIMIT
    assert max(state.dx) < FORM_RADIUS_LIMIT
    assert max(state.dv) < CAPACITY_RADIUS_LIMIT
    assert result["marks"] == ("RA", "RA", "IL", "IL", "IL")
    assert result["boosts"][:2] == (1 + law["factors"]["RA_vf_amplification"],) * 2
    digest = sha256(
        repr(
            (
                first_preparation["digest"],
                source,
                law,
                result,
                LIFTS,
                MEAN_WIDTH_LIMIT,
                FORM_RADIUS_LIMIT,
                CAPACITY_RADIUS_LIMIT,
                INTERVAL_METHOD,
            )
        ).encode()
    ).hexdigest()
    return dict(
        first=first_preparation, source=source, result=result, law=law, digest=digest
    )


def _fine_second(graph, previous, source, law):
    """Exact independent fine arithmetic from the genuine previous control."""
    f = law["factors"]
    forms, capacities = previous["forms"], previous["capacities"]
    phases = previous["event_phases"]
    assert phases == (Q(0),) * 10  # previous complete coordinator also holds0
    pressure = exact._fresh(graph, forms, capacities, phases)
    peak = max(map(abs, pressure))
    alpha, beta, gamma = law["stability"]
    si = tuple(
        alpha * capacity / max(capacities) + beta + gamma * (1 - abs(p) / peak)
        for capacity, p in zip(capacities, pressure)
    )
    assert min(si) > law["thresholds"]["si_lo"]
    marks = tuple(
        (
            "IL"
            if value >= law["thresholds"]["si_hi"]
            else "NAV" if abs(p) / peak > law["thresholds"]["dnfr_hi"] else "RA"
        )
        for value, p in zip(si, pressure)
    )
    assert marks == tuple(
        mark for mark in ("RA", "RA", "IL", "IL", "IL") for _ in range(2)
    )
    x, v, q = list(forms), list(capacities), list(pressure)
    for node, mark in enumerate(marks):
        if mark == "RA":
            signal = sum(x[j] for j in graph.neighbors(node)) / 4
            proposal = (1 - f["RA_epi_diff"]) * x[node] + f["RA_epi_diff"] * signal
            assert proposal * x[node] > 0 and abs(signal) > Q(
                RA_RUNTIME_AMPLIFICATION_TRIGGER
            )
            x[node] = proposal
            v[node] *= 1 + f["RA_vf_amplification"]
        else:
            q[node] *= f["IL_dnfr_factor"]
    rates = tuple(capacity * p for capacity, p in zip(v, q))
    endpoint = tuple(value + DT * rate for value, rate in zip(x, rates))
    acceleration = tuple(
        (rate - old) / DT for rate, old in zip(rates, previous["rate"])
    )
    flags = tuple(
        value >= law["thresholds"]["si_hi"] and abs(p) <= law["epsilon"]
        for value, p in zip(si, q)
    )
    assert not any(flags)
    assignments = []
    for n in graph:
        values = {
            source.histories[n].samples[0].variable: previous["event_forms"][n],
            source.form_variables[n]: forms[n],
            ("event", source.time, n): x[n],
            ("form", source.time + DT, n): endpoint[n],
            source.rate_variables[n]: previous["rate"][n],
            ("rate", source.time + DT, n): rates[n],
        }
        assignments.append(values)
    return dict(
        forms=endpoint,
        capacities=tuple(v),
        pressure=pressure,
        si=si,
        held=tuple(q),
        rates=rates,
        acceleration=acceleration,
        assignments=tuple(assignments),
        previous_rates=previous["rate"],
        resets=tuple(a != b for a, b in zip(x, forms)),
    )


@pytest.fixture(scope="module")
def controls(frozen):
    preparation = frozen["first"]
    obs, graph = preparation["observations"][1], preparation["graph"]
    outputs = []
    for sd, se in LIFTS:
        means, capacities = tuple(value.midpoint for value in obs.means), tuple(
            value.midpoint for value in obs.capacities
        )
        d = tuple(sd * (-1) ** i * radius for i, radius in enumerate(obs.dx))
        e = tuple(se * (-1) ** i * radius for i, radius in enumerate(obs.dv))
        previous = first._fine_control(graph, obs, means, capacities, d, e, Q(0), Q(0))
        first._assert_contains(obs, preparation["forecasts"][1], previous)
        outputs.append(_fine_second(graph, previous, frozen["source"], frozen["law"]))
    return tuple(outputs)


def test_two_call_protocol_uses_only_the_abstract_successor(frozen, enclosure_guard):
    targets, forbidden = enclosure_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    source, result = frozen["source"], frozen["result"]
    assert len(frozen["digest"]) == 64 and source.time == Q(17, 16)
    assert source.means == frozen["first"]["forecasts"][1]["means"]
    assert source.rates == frozen["first"]["forecasts"][1]["rate"]
    assert source.accelerations == frozen["first"]["forecasts"][1]["acceleration"]
    assert any(not value.contains(0) for value in source.accelerations)
    assert result["origins"] == ("default",) * 10
    assert result["state"].time == Q(9, 8)
    assert result["state"].runtime_ordinal == 2


def test_live_first_resonance_boost_switch_is_uniform_without_fine_state(frozen):
    result, law = frozen["result"], frozen["law"]
    assert frozen["first"]["forecasts"][1]["boost"][0] == 1
    assert result["signals"][0].hi < -Q(1, 100)
    assert result["signals"][1].hi < -Q(18, 1000)
    assert result["boosts"][:2] == (1 + law["factors"]["RA_vf_amplification"],) * 2
    assert result["resets"] == (True, True, False, False, False)
    assert max(result["ratios"][i].hi for i in (0, 1)) < law["thresholds"]["dnfr_hi"]
    assert max(result["si"][i].hi for i in (0, 1)) < law["thresholds"]["si_hi"]
    assert min(value.lo for value in result["si"][2:]) > law["thresholds"]["si_hi"]
    assert min(abs(value).lo for value in result["held"]) > Q(15, 10000)


def test_four_frozen_fine_paths_are_contained_without_extra_tolerance(frozen, controls):
    state, result = frozen["result"]["state"], frozen["result"]
    assert len(controls) == 4
    for output in controls:
        for field, means, radii in (
            ("forms", state.means, state.dx),
            ("capacities", state.capacities, state.dv),
        ):
            actual_mean, actual_difference = scalar._moments(output[field])
            assert all(
                bound.contains(value) for bound, value in zip(means, actual_mean)
            )
            assert all(
                abs(value) <= radius for value, radius in zip(actual_difference, radii)
            )
        for name in ("pressure", "si", "held"):
            assert all(
                result[name][n // 2].contains(value)
                for n, value in enumerate(output[name])
            )
        assert all(
            bound.contains(value) for bound, value in zip(state.rates, output["rates"])
        )
        assert all(
            bound.contains(value)
            for bound, value in zip(state.accelerations, output["acceleration"])
        )
        assert output["resets"] == (True,) * 4 + (False,) * 6
    assert max(value.width for value in state.means) < MEAN_WIDTH_LIMIT
    assert max(value.width for value in state.capacities) < MEAN_WIDTH_LIMIT
    assert max(state.dx) < FORM_RADIUS_LIMIT and max(state.dv) < CAPACITY_RADIUS_LIMIT


def test_propagated_histories_keep_exact_variable_and_rate_associations(
    frozen, controls
):
    state = frozen["result"]["state"]
    assert (
        tuple(len(history.samples) for history in state.histories)
        == (2,) * 4 + (3,) * 6
    )
    for output in controls:
        for n, history in enumerate(state.histories):
            values = output["assignments"][n]
            assert history.samples[-1].variable == state.form_variables[n]
            assert history.relations[-1].rate_variable == state.rate_variables[n]
            assert all(
                sample.enclosure.contains(values[sample.variable])
                for sample in history.samples
            )
            velocities = (
                (output["rates"][n],)
                if n < 4
                else (output["previous_rates"][n], output["rates"][n])
            )
            assert all(
                values[relation.rate_variable] == rate
                for relation, rate in zip(history.relations, velocities)
            )
            assert all(relation.contains(values) for relation in history.relations)
    assert state.traces == tuple(
        ("EN", "RA", "RA") if n < 4 else ("EN", "IL", "IL") for n in range(10)
    )
    assert state.ordinals == (5,) * 10 and state.lag_al == state.lag_en == (2,) * 10
    assert state.debt == tuple(int(n // 2 == 1) for n in range(10))
    assert state.counts == (frozenset((0,)),) * 10 and all(state.prior)


def test_phase_gains_and_remesh_evidence_are_carried_forward(frozen):
    source, state, law = frozen["source"], frozen["result"]["state"], frozen["law"]
    assert state.phase_origin == source.phase_origin == I(0)
    assert state.relative_phases == source.relative_phases == (Q(0),) * 5
    expected = tuple(
        value
        + law["phase_policy"]["down"] * (law["phase_policy"][f"k{suffix}_min"] - value)
        for value, suffix in zip(source.phase_gains, ("G", "L"))
    )
    assert state.phase_gains == expected and state.phase_gains != source.phase_gains
    policy = law["phase_policy"]
    assert state.phase_gains == tuple(
        policy[f"k{suffix}_min"]
        + (1 - policy["down"]) ** 2 * (value - policy[f"k{suffix}_min"])
        for value, suffix in zip(law["phase_gains"], ("G", "L"))
    )
    initial_rates = frozen["first"]["observations"][1].old_rates
    reset_acceleration = tuple(
        (rate - old) / DT for rate, old in zip(state.rates, initial_rates)
    )
    assert any(
        wrong.lo > actual.hi or actual.lo > wrong.hi
        for wrong, actual in zip(reset_acceleration, state.accelerations)
    )
    assert len(source.remesh_snapshots) == 1 and len(state.remesh_snapshots) == 2
    assert state.remesh_snapshots[0] == source.remesh_snapshots[0]
    assert state.remesh_snapshots[-1] == tuple(
        history.samples[-1] for history in state.histories
    )
    assert source.stable_fraction_history == state.stable_fraction_history == ()


@pytest.mark.parametrize(
    "damage",
    (
        "stale",
        "variable",
        "rate",
        "relation",
        "rate_variable",
        "trace",
        "gain",
        "ordinal",
        "phase",
    ),
)
def test_invalid_or_unassociated_memory_cannot_be_reset_to_initial_defaults(
    frozen, damage
):
    source = frozen["source"]
    if damage == "stale":
        broken, message = replace(source, time=source.time + DT), "latest sample time"
    elif damage == "variable":
        broken = replace(
            source,
            form_variables=(("unrelated", source.time, 0), *source.form_variables[1:]),
        )
        message = "current-form variable"
    elif damage == "rate":
        broken, message = (
            replace(source, rates=(I(1), *source.rates[1:])),
            "retained rate",
        )
    elif damage == "relation":
        history = source.histories[0]
        broken_relation = replace(history.relations[0], left=("unrelated", Q(1), 0))
        broken = replace(
            source,
            histories=(
                replace(history, relations=(broken_relation,)),
                *source.histories[1:],
            ),
        )
        message = "broken flow variable"
    elif damage == "rate_variable":
        broken = replace(
            source,
            rate_variables=(("unrelated", source.time, 0), *source.rate_variables[1:]),
        )
        message = "retained rate variable"
    elif damage == "trace":
        broken = replace(source, traces=(("EN", "OZ"), *source.traces[1:]))
        message = "unsupported retained grammar"
    elif damage == "gain":
        broken = replace(source, phase_gains=(Q(0), source.phase_gains[1]))
        message = "retained phase gains"
    elif damage == "phase":
        broken = replace(source, phase_origin=I(Q(1, 16)))
        message = "propagated phase sector"
    else:
        broken = replace(source, runtime_ordinal=True)
        message = "second call"
    with pytest.raises(ValueError, match=message):
        _second_step(frozen["law"], broken)
