"""Conditional joint form/capacity calculus without new native evaluations.

The shared five-operator rows are rewritten in complete signed pair moments,
including form-capacity covariance and target-associated Transition inputs.
This is not a smaller closed collective state or a generic predictor API.
Requests, live grammar, phases, individual memory and both contrasts remain.

A detached supplied AL/EN/IL/NAV/RA transcript checks fine ordered proposals
against independently derived moment rows. It declares its own physical state;
the older source supplies only support, defaults and admitted grammar context.
Separately, the existing joint-admission positive source supplies an actual
default all-IL adaptation exit and its previously admitted next IL step. Only
its pure source/forecast formulas are reused, never its response fixtures.

All arithmetic is exact ideal real over represented constants. Execution,
production pressure/selection/operators/integration and RNG remain disabled.
No new trajectory, fixed-seed input claim or physical identification is made.
"""

from fractions import Fraction as Q
from itertools import product

import pytest

from tests.physics import test_native_collective_calculus as scalar
from tests.physics import test_native_joint_admission as joint
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER

domain = scalar.domain
no_execution = scalar.no_execution
calculus_guard = scalar.calculus_guard
configuration = scalar.configuration
DT, PI = scalar.DT, scalar.PI
SCHEDULE = scalar.SCHEDULE
CAPACITY_CONTRASTS = (Q(1, 128), Q(-1, 256), Q(1, 128), Q(1, 256), Q(-1, 128))
PHASES = (Q(0), Q(1, 8), Q(1, 4), Q(1, 8), Q(0))


@pytest.fixture(scope="module", autouse=True)
def joint_guard(calculus_guard):
    yield calculus_guard


def _fresh(graph, forms, capacities, phases):
    """Fine arithmetic on a common arc, independently of moment pressure."""
    k, b, w = (
        domain.base._weight(graph, "DNFR_WEIGHTS", name)
        for name in ("epi", "vf", "phase")
    )
    assert max(phases) - min(phases) < PI
    return tuple(
        k * (sum(forms[j] for j in graph.neighbors(n)) / 4 - forms[n])
        + b * (sum(capacities[j] for j in graph.neighbors(n)) / 4 - capacities[n])
        + w * (sum(phases[j] for j in graph.neighbors(n)) / 4 - phases[n]) / PI
        for n in graph
    )


@pytest.fixture(scope="module")
def transcript(configuration, joint_guard):
    graph = configuration.copy()
    cfg = graph.graph
    factors = {name: Q(value) for name, value in cfg["GLYPH_FACTORS"].items()}
    k, b, w = (
        domain.base._weight(graph, "DNFR_WEIGHTS", name)
        for name in ("epi", "vf", "phase")
    )
    means, differences = scalar.MEANS, scalar.CONTRASTS
    capacities, capacity_differences = scalar.CAPACITIES, CAPACITY_CONTRASTS
    forms = scalar._fine(means, differences)
    vf = scalar._fine(capacities, capacity_differences)
    phases = domain._paired(PHASES)
    old_rates = tuple(Q(node - 4, 64) for node in graph)
    for node in graph:
        graph.nodes[node].update(
            EPI=forms[node], nu_f=vf[node], theta=phases[node], dEPI=old_rates[node]
        )
    assert all(Q(cfg["VF_MIN"]) < value < Q(cfg["VF_MAX"]) for value in vf)
    assert all(-1 < value < 1 for value in forms)
    centers = tuple(
        k * (domain._neighbor(means, i) - means[i])
        + b * (domain._neighbor(capacities, i) - capacities[i])
        + w * (domain._neighbor(PHASES, i) - PHASES[i]) / PI
        for i in range(5)
    )
    pressure_contrasts = tuple(
        -k * d - b * e for d, e in zip(differences, capacity_differences)
    )
    prepared = _fresh(graph, forms, vf, phases)
    assert prepared == scalar._fine(centers, pressure_contrasts)
    jitter = factors["NAV_jitter"]
    innovations = (-jitter / 3, jitter / 4)
    live_x, live_v, live_theta, live_q = map(list, (forms, vf, phases, prepared))
    fine_prefixes, live_inputs = [], []
    for node in graph:
        mark = SCHEDULE[node // 2]
        neighbors = tuple(graph.neighbors(node))
        signal = sum(live_x[j] for j in neighbors) / 4
        if mark == "AL":
            live_x[node] += factors["AL_boost"]
        elif mark == "EN":
            live_x[node] = (1 - factors["EN_mix"]) * live_x[node] + factors[
                "EN_mix"
            ] * signal
        elif mark == "IL":
            live_q[node] *= factors["IL_dnfr_factor"]
        elif mark == "NAV":
            assert live_q[node] > 0
            live_q[node] = (
                (1 - factors["NAV_eta"]) * live_q[node]
                + factors["NAV_eta"] * live_v[node]
                + innovations[node % 2]
            )
            assert live_q[node] != prepared[node]
        else:
            assert mark == "RA"
            admitted = tuple(
                j
                for j in neighbors
                if abs(scalar._wrap(live_theta[j] - live_theta[node])) <= PI / 2
            )
            assert admitted == neighbors
            signal = sum(live_x[j] for j in admitted) / len(admitted)
            phase_mean = sum(live_theta[j] for j in admitted) / len(admitted)
            proposal = (1 - factors["RA_epi_diff"]) * live_x[node] + factors[
                "RA_epi_diff"
            ] * signal
            assert live_x[node] * proposal > 0
            live_x[node] = proposal
            live_theta[node] += factors["RA_phase_coupling"] * (
                phase_mean - live_theta[node]
            )
            assert signal > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
            live_v[node] *= 1 + factors["RA_vf_amplification"]
        live_inputs.append(signal)
        assert -1 < live_x[node] < 1
        assert Q(cfg["VF_MIN"]) < live_v[node] < Q(cfg["VF_MAX"])
        fine_prefixes.append(tuple(map(tuple, (live_x, live_v, live_theta, live_q))))

    # Independent block sweep: live neighbor form/phase, prepared pressure.
    y, d, u, e, phi = map(
        list, (means, differences, capacities, capacity_differences, PHASES)
    )
    qbar, qdelta = list(centers), list(pressure_contrasts)
    rows = []
    for i, mark in enumerate(SCHEDULE):
        signal = domain._neighbor(y, i)
        form_factor = Q(1)
        if mark == "AL":
            y[i] += factors["AL_boost"]
        elif mark in {"EN", "RA"}:
            mix = factors["EN_mix"] if mark == "EN" else factors["RA_epi_diff"]
            form_factor = 1 - mix
            y[i] = form_factor * y[i] + mix * signal
            d[i] *= form_factor
        if mark == "RA":
            boost = 1 + factors["RA_vf_amplification"]
            u[i], e[i] = boost * u[i], boost * e[i]
            phi[i] += factors["RA_phase_coupling"] * (domain._neighbor(phi, i) - phi[i])
        if mark == "IL":
            qbar[i] *= factors["IL_dnfr_factor"]
            qdelta[i] *= factors["IL_dnfr_factor"]
        elif mark == "NAV":
            h, eta = 1 - factors["NAV_eta"], factors["NAV_eta"]
            qbar[i] = h * centers[i] + eta * capacities[i] + sum(innovations) / 2
            qdelta[i] = (
                h * pressure_contrasts[i]
                + eta * capacity_differences[i]
                + (innovations[1] - innovations[0]) / 2
            )
        rows.append(
            dict(
                form_factor=form_factor,
                signal=signal,
                means=tuple(y),
                differences=tuple(d),
                capacities=tuple(u),
                capacity_differences=tuple(e),
                phases=tuple(phi),
                qbar=tuple(qbar),
                qdelta=tuple(qdelta),
            )
        )
    rates = tuple(v * q for v, q in zip(live_v, live_q))
    endpoint = tuple(x + DT * rate for x, rate in zip(live_x, rates))
    acceleration = tuple((rate - old) / DT for rate, old in zip(rates, old_rates))
    assert all(-1 < value < 1 for value in endpoint)
    assert min(map(abs, live_q)) > Q(cfg["EPS_DNFR_STABLE"])
    return dict(
        graph=graph,
        factors=factors,
        k=k,
        b=b,
        w=w,
        forms=forms,
        vf=vf,
        phases=phases,
        prepared=prepared,
        centers=centers,
        pressure_contrasts=pressure_contrasts,
        innovations=innovations,
        prefixes=tuple(fine_prefixes),
        live_inputs=tuple(live_inputs),
        rows=tuple(rows),
        rate=rates,
        endpoint=endpoint,
        acceleration=acceleration,
        old_rates=old_rates,
    )


def test_source_provenance_fresh_normalizers_and_disabled_execution(
    transcript, joint_guard
):
    c, (targets, forbidden) = transcript, joint_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    graph = c["graph"]
    peak = max(map(abs, c["prepared"]))
    assert peak == max(
        abs(b) + abs(d) for b, d in zip(c["centers"], c["pressure_contrasts"])
    )
    maximum = max(c["vf"])
    assert maximum == max(
        v + abs(e) for v, e in zip(scalar.CAPACITIES, CAPACITY_CONTRASTS)
    )
    assert maximum > max(scalar.CAPACITIES)
    alpha, beta, gamma = (
        domain.base._weight(graph, "SI_WEIGHTS", name)
        for name in ("alpha", "beta", "gamma")
    )
    for node, data in graph.nodes(data=True):
        assert data["EPI"] == c["forms"][node] and data["nu_f"] == c["vf"][node]
        assert data["theta"] == PHASES[node // 2]
        assert data["epi_kind"] == "seed" and tuple(data["glyph_history"]) == ("EN",)
        assert data[domain.PRIOR_COHERENCE_KEY] is True
        assert data[domain.OPERATOR_STEP_KEY] == 3
        assert data[domain.U2_DEBT_KEY] == node // 2 % 2
        phase_offset = abs(domain._neighbor(PHASES, node // 2) - PHASES[node // 2]) / PI
        actual_si = (
            alpha * c["vf"][node] / maximum
            + beta * (1 - phase_offset)
            + gamma * (1 - abs(c["prepared"][node]) / peak)
        )
        assert 0 < actual_si < 1
    # The declared five-mark word is conditional, not a default-selector
    # verdict. This diagnostic calculation does not manufacture its origin.


def test_live_five_operator_prefixes_agree_in_both_contrasts(transcript):
    c = transcript
    for block, row in enumerate(c["rows"]):
        fine_x, fine_v, fine_theta, fine_q = c["prefixes"][2 * block + 1]
        assert scalar._moments(fine_x) == (row["means"], row["differences"])
        assert scalar._moments(fine_v) == (
            row["capacities"],
            row["capacity_differences"],
        )
        assert scalar._moments(fine_q) == (row["qbar"], row["qdelta"])
        assert fine_theta == domain._paired(row["phases"])
        assert (
            c["live_inputs"][2 * block]
            == c["live_inputs"][2 * block + 1]
            == row["signal"]
        )
    boost = c["factors"]["AL_boost"]
    for block in (1, 4):
        assert (
            c["rows"][block]["signal"] - domain._neighbor(scalar.MEANS, block)
            == boost / 2
        )
    final = c["rows"][-1]
    factor = 1 + c["factors"]["RA_vf_amplification"]
    assert final["capacities"][4] == factor * scalar.CAPACITIES[4]
    assert (
        final["capacity_differences"][4]
        == factor * CAPACITY_CONTRASTS[4]
        != CAPACITY_CONTRASTS[4]
    )
    # Prepared pressure is not refreshed after either form or capacity jumps.
    refreshed = _fresh(
        c["graph"], c["prefixes"][-1][0], c["prefixes"][-1][1], c["prefixes"][-1][2]
    )
    assert refreshed != c["prefixes"][-1][3]


def test_held_flow_requires_covariance_and_associated_prior_rates(transcript):
    c, row = transcript, transcript["rows"][-1]
    u, e, q, z = (
        row[name] for name in ("capacities", "capacity_differences", "qbar", "qdelta")
    )
    rate_mean = tuple(
        v * p + contrast * delta for v, p, contrast, delta in zip(u, q, e, z)
    )
    rate_difference = tuple(
        v * delta + contrast * p for v, delta, contrast, p in zip(u, z, e, q)
    )
    assert scalar._moments(c["rate"]) == (rate_mean, rate_difference)
    assert scalar._moments(c["endpoint"]) == (
        tuple(y + DT * rate for y, rate in zip(row["means"], rate_mean)),
        tuple(d + DT * rate for d, rate in zip(row["differences"], rate_difference)),
    )
    old_mean, old_difference = scalar._moments(c["old_rates"])
    assert scalar._moments(c["acceleration"]) == (
        tuple((a - b) / DT for a, b in zip(rate_mean, old_mean)),
        tuple((a - b) / DT for a, b in zip(rate_difference, old_difference)),
    )
    assert any(rate != v * p for rate, v, p in zip(rate_mean, u, q))
    # A bounded lossy product-of-means observation retains an explicit error
    # instead of silently discarding covariance. Input signs stay associated.
    f = c["factors"]
    for i, mark in enumerate(SCHEDULE):
        dx, dv = abs(scalar.CONTRASTS[i]), abs(CAPACITY_CONTRASTS[i])
        h = (
            f["IL_dnfr_factor"]
            if mark == "IL"
            else 1 - f["NAV_eta"] if mark == "NAV" else Q(1)
        )
        boost = 1 + f["RA_vf_amplification"] if mark == "RA" else Q(1)
        extra = (
            f["NAV_eta"] * dv + abs((c["innovations"][1] - c["innovations"][0]) / 2)
            if mark == "NAV"
            else Q(0)
        )
        budget = DT * boost * dv * (h * (c["k"] * dx + c["b"] * dv) + extra)
        observed_error = DT * (rate_mean[i] - u[i] * q[i])
        assert observed_error == DT * e[i] * z[i]
        assert abs(observed_error) <= budget
    for node, value in enumerate(c["endpoint"]):
        event_form = c["prefixes"][-1][0][node]
        history = ((Q(1), event_form), (1 + DT, value))
        assert (history[-1][1] - history[-2][1]) / DT == c["rate"][node]


def test_transition_capacity_contrast_changes_input_association(transcript):
    c, f, block = transcript, transcript["factors"], 3
    original = c["prepared"][6:8]
    vf, x = c["vf"][6:8], c["forms"][6:8]
    assert min(original) > 0
    assert f["NAV_eta"] * (min(vf) - max(original)) > f["NAV_jitter"]
    z = f["NAV_jitter"] / 4
    endpoints = []
    for innovations in ((-z, z), (z, -z)):
        held = tuple(
            (1 - f["NAV_eta"]) * p + f["NAV_eta"] * v + noise
            for p, v, noise in zip(original, vf, innovations)
        )
        assert all(new != old for new, old in zip(held, original))
        endpoints.append(
            tuple(
                value + DT * capacity * pressure
                for value, capacity, pressure in zip(x, vf, held)
            )
        )
    means = tuple(sum(values) / 2 for values in endpoints)
    differences = tuple((values[1] - values[0]) / 2 for values in endpoints)
    assert means[0] - means[1] == 2 * DT * CAPACITY_CONTRASTS[block] * z != 0
    assert differences[0] - differences[1] == 2 * DT * scalar.CAPACITIES[block] * z
    # Same mean and unordered innovations do not determine mean response
    # once capacities differ. These are supplied inputs, not RNG resampling.


def test_equal_pressure_can_hide_capacity_and_transition_contrasts(configuration):
    # Supplied common-sign Transition row, not a selected NAV witness; this
    # capacity configuration can independently satisfy the IL selection cut.
    graph, cfg = configuration, configuration.graph
    k, b = (domain.base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf"))
    alpha = domain.base._weight(graph, "SI_WEIGHTS", "alpha")
    eta = Q(cfg["GLYPH_FACTORS"]["NAV_eta"])
    mean, capacity, e, baseline = Q(1, 4), Q(1, 4), Q(1, 64), Q(1, 32)
    d = -b * e / k
    neighbor = mean + baseline / k
    means = (mean, neighbor, mean, mean, neighbor)
    forms = scalar._fine(means, (d, *(Q(0),) * 4))
    capacities = scalar._fine((capacity,) * 5, (e, *(Q(0),) * 4))
    pressure = _fresh(graph, forms, capacities, (Q(0),) * 10)
    assert k * d + b * e == 0 and pressure[:2] == (baseline, baseline)
    assert all(-1 < value < 1 for value in forms)
    assert min(capacities) > 0
    si_gap = alpha * (capacities[1] - capacities[0]) / max(capacities)
    assert si_gap == 2 * alpha * e / (capacity + e) > 0
    held = tuple((1 - eta) * p + eta * v for p, v in zip(pressure[:2], capacities[:2]))
    assert (held[1] - held[0]) / 2 == eta * e != 0
    rate = tuple(v * p for v, p in zip(capacities[:2], held))
    assert sum(rate) / 2 - capacity * sum(held) / 2 == eta * e**2 > 0


def test_paired_phase_survives_unequal_capacity_under_actual_policy(transcript):
    c, cfg = transcript, transcript["graph"].graph
    fine = c["prefixes"][-1][2]
    phi = c["rows"][-1]["phases"]
    assert fine == domain._paired(phi) and 0 <= min(phi) <= max(phi) <= Q(1, 4)
    # Projection onto the center of the containing arc gives a rigorous
    # resultant floor; no trig or production coordinator is evaluated.
    resultant_floor = 1 - Q(1, 8) ** 2 / 2
    traces = tuple(("EN", SCHEDULE[node // 2]) for node in c["graph"])
    load = Q(
        sum(mark == "NAV" for trace in traces for mark in trace), sum(map(len, traces))
    )
    policy = cfg["PHASE_ADAPT"]
    assert resultant_floor > Q(policy["R_hi"]) and load == Q(1, 10) < Q(
        policy["disr_lo"]
    )
    kg, kl = tuple(
        Q(cfg[f"PHASE_K_{name}"])
        + Q(policy["down"]) * (Q(policy[f"k{suffix}_min"]) - Q(cfg[f"PHASE_K_{name}"]))
        for name, suffix in (("GLOBAL", "G"), ("LOCAL", "L"))
    )
    assert 0 < kg and 0 < kl and kg + kl < 1
    # Keep the shared global resultant angle as an exact symbolic column.
    fine_rows = tuple(
        (
            kg,
            (1 - kg - kl) * fine[n]
            + kl * sum(fine[j] for j in c["graph"].neighbors(n)) / 4,
        )
        for n in c["graph"]
    )
    block_rows = tuple(
        (kg, (1 - kg - kl) * phi[i] + kl * domain._neighbor(phi, i)) for i in range(5)
    )
    assert fine_rows == domain._paired(block_rows)
    assert c["prefixes"][-1][1][0] != c["prefixes"][-1][1][1]


@pytest.fixture(scope="module")
def adaptation_entry(joint_guard):
    graph, _ = joint._source("positive_plus")
    forms = tuple(Q(graph.nodes[n]["EPI"]) for n in graph)
    capacities = tuple(Q(graph.nodes[n]["nu_f"]) for n in graph)
    counts = tuple(graph.nodes[n]["stable_count"] for n in graph)
    rates = tuple(Q(graph.nodes[n]["dEPI"]) for n in graph)
    first = joint._fine_step(graph, forms, capacities, counts, rates)
    second = joint._fine_step(
        graph, first["form"], first["capacity"], first["counts"], first["rate"]
    )
    return dict(
        graph=graph,
        forms=forms,
        capacities=capacities,
        counts=counts,
        rates=rates,
        first=first,
        second=second,
    )


def test_existing_adaptation_exit_is_readmitted_by_actual_all_coherence_selection(
    adaptation_entry,
):
    c, graph = adaptation_entry, adaptation_entry["graph"]
    cfg = graph.graph
    alpha, beta = (
        domain.base._weight(graph, "SI_WEIGHTS", key) for key in ("alpha", "beta")
    )
    h = Q(cfg["GLYPH_FACTORS"]["IL_dnfr_factor"])
    mu, epsilon = Q(cfg["VF_ADAPT_MU"]), joint.feedback.EPSILON
    assert c["first"]["due"][2:4] == (False, True)
    assert scalar._moments(c["first"]["capacity"])[1][1] == mu * epsilon / 4 > 0
    assert (
        scalar._moments(c["first"]["form"])[1][1]
        == -(1 - DT * h * domain.base._weight(graph, "DNFR_WEIGHTS", "epi")) * joint.RHO
    )
    for step, capacities in ((1, c["capacities"]), (2, c["first"]["capacity"])):
        assert alpha * min(capacities) / max(capacities) + beta > Q(
            cfg["SELECTOR_THRESHOLDS"]["si_hi"]
        )
        for node in graph:
            assert cfg["history"]["since_AL"][node] + step <= cfg["AL_MAX_LAG"]
            assert cfg["history"]["since_EN"][node] + step <= cfg["EN_MAX_LAG"]
            assert graph.nodes[node][domain.PRIOR_COHERENCE_KEY] is True
            assert graph.nodes[node]["epi_kind"] == "seed"
    assert all(
        value > Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"]) for value in c["second"]["si"]
    )
    assert c["second"]["due"][2:4] == (True, True)
    assert all(c["second"]["qualified"])


def test_joint_rows_continue_after_unequal_due_with_nodewise_memory(adaptation_entry):
    c, graph = adaptation_entry, adaptation_entry["graph"]
    k, b = (domain.base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf"))
    h, mu = Q(graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"]), Q(
        graph.graph["VF_ADAPT_MU"]
    )
    old = dict(
        form=c["forms"], capacity=c["capacities"], counts=c["counts"], rate=c["rates"]
    )
    histories = tuple(((Q(1), x),) for x in c["forms"])
    for step, result in enumerate((c["first"], c["second"]), start=1):
        means, d = scalar._moments(old["form"])
        capacities, e = scalar._moments(old["capacity"])
        baseline = tuple(
            k * (domain._neighbor(means, i) - means[i])
            + b * (domain._neighbor(capacities, i) - capacities[i])
            for i in range(5)
        )
        contrast = tuple(-k * dx - b * dv for dx, dv in zip(d, e))
        assert result["initial_pressure"] == scalar._fine(baseline, contrast)
        rate_mean = tuple(
            h * (v * p + dv * q)
            for v, p, dv, q in zip(capacities, baseline, e, contrast)
        )
        rate_contrast = tuple(
            h * (v * q + dv * p)
            for v, q, dv, p in zip(capacities, contrast, e, baseline)
        )
        assert scalar._moments(result["rate"]) == (rate_mean, rate_contrast)
        assert scalar._moments(result["form"]) == (
            tuple(m + DT * rate for m, rate in zip(means, rate_mean)),
            tuple(dx + DT * rate for dx, rate in zip(d, rate_contrast)),
        )
        due_mean, due_difference = scalar._moments(
            tuple(Q(int(value)) for value in result["due"])
        )
        expected_v = tuple(
            v + mu * (abar * (domain._neighbor(capacities, i) - v) - adelta * dv)
            for i, (v, dv, abar, adelta) in enumerate(
                zip(capacities, e, due_mean, due_difference)
            )
        )
        expected_e = tuple(
            dv + mu * (adelta * (domain._neighbor(capacities, i) - v) - abar * dv)
            for i, (v, dv, abar, adelta) in enumerate(
                zip(capacities, e, due_mean, due_difference)
            )
        )
        assert scalar._moments(result["capacity"]) == (expected_v, expected_e)
        for n in graph:
            assert result["counts"][n] == (
                old["counts"][n] + 1 if result["qualified"][n] else 0
            )
            assert (
                result["acceleration"][n] == (result["rate"][n] - old["rate"][n]) / DT
            )
        histories = tuple(
            (*history, (Q(1) + step * DT, x))
            for history, x in zip(histories, result["form"])
        )
        for history, rate in zip(histories, result["rate"]):
            assert (history[-1][1] - history[-2][1]) / DT == rate
        old = result
    assert tuple(map(len, histories)) == (3,) * 10
    # IL appends without form jumps; supplied traces EN/IL/IL, operator step5,
    # lags2, resolved debt and no draws give the actual unchanged phase load0.
    assert c["second"]["counts"][2] != c["second"]["counts"][3]
    assert (
        scalar._moments(c["second"]["capacity"])[1][1]
        == (1 - mu) * scalar._moments(c["first"]["capacity"])[1][1]
        > 0
    )


def test_same_means_different_form_capacity_association_changes_mean_flow(
    configuration,
):
    # The reused pure IL oracle declares zero phases. This graph supplies
    # configuration/support only; its retained physical phases are not used.
    graph = configuration
    k, b = (domain.base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf"))
    h = Q(graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"])
    d, e = Q(1, 128), Q(1, 64)
    forms = scalar._fine((Q(1, 8),) * 5, (d, *(Q(0),) * 4))
    results = []
    for sign in (1, -1):
        capacities = scalar._fine((Q(1),) * 5, (sign * e, *(Q(0),) * 4))
        result = joint._fine_step(graph, forms, capacities, (0,) * 10, (Q(0),) * 10)
        assert not any(result["due"])
        assert scalar._moments(capacities)[0] == (Q(1),) * 5
        observed = scalar._moments(result["rate"])[0][0]
        assert observed == -h * (sign * k * d * e + b * e**2)
        results.append(result)
    plus, minus = (scalar._moments(result["form"])[0] for result in results)
    assert plus[0] - minus[0] == -2 * DT * h * k * d * e != 0
    assert plus[1:] == minus[1:]


def test_joint_adaptation_moments_include_existing_capacity_contrast(configuration):
    mu = Q(configuration.graph["VF_ADAPT_MU"])
    u, e, neighbor = Q(5, 4), Q(1, 16), Q(3, 4)
    for flags in product((False, True), repeat=2):
        proposed = tuple(
            v + mu * (neighbor - v) if due else v
            for v, due in zip((u - e, u + e), flags)
        )
        abar, adelta = Q(sum(flags), 2), Q(int(flags[1]) - int(flags[0]), 2)
        assert sum(proposed) / 2 == u + mu * (abar * (neighbor - u) - adelta * e)
        assert (proposed[1] - proposed[0]) / 2 == (1 - mu * abar) * e + mu * adelta * (
            neighbor - u
        )
    assert (1 - mu) * e != 0


def test_zero_capacity_and_clamped_amplification_are_outside_clean_domain(
    configuration,
):
    cfg = configuration.graph
    a = Q(cfg["GLYPH_FACTORS"]["RA_vf_amplification"])
    v, e = Q(1, 4), Q(1, 4)
    fine = (v - e, v + e)
    assert min(fine) == 0 and not v > abs(e)
    pressure = (Q(1, 8), Q(1, 4))
    rate = tuple(capacity * p for capacity, p in zip(fine, pressure))
    assert rate[0] == 0 and pressure[0] != 0
    # This algebraic frozen row is not a fully admitted positive-capacity
    # event sector, and says nothing about freezing the complete state.
    upper = Q(cfg["VF_MAX"])
    mean, contrast = upper / (1 + a), upper * a / (4 * (1 + a))
    original = (mean - contrast, mean + contrast)
    amplified = tuple((1 + a) * value for value in original)
    assert max(original) < upper and amplified[0] < upper < amplified[1]
    clamped = tuple(min(upper, value) for value in amplified)
    assert sum(clamped) / 2 != (1 + a) * mean
    assert (clamped[1] - clamped[0]) / 2 != (1 + a) * contrast
    # The native flow consumes amplification before the later clamp; this
    # theorem excludes that boundary instead of silently using clipped U.
