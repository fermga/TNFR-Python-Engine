"""Finite contrast bounds and request admission, with no native evaluations.

All arithmetic is ideal real over represented coefficients. A finite supplied
five-row word checks the conditional recurrence, retaining innovation means;
it is not a default-selected trajectory. Separately, the existing live-prefix
source admits an entire positive-width box of zero-mean twin form contrasts
under the actual default selection inequalities and original lag/context.

Uniform bounds, not sampled verdicts, admit this box. Independent fine-node
formula controls check those bounds and ordered live event inputs. Individual
rates, temporal histories and qualification predicates remain associated.
The zero-pressure boundary has a genuine ideal-real selector discontinuity;
no global robustness, infinite retention or binary64 enclosure is claimed.
"""

from fractions import Fraction as Q
from itertools import product

import pytest

from tests.physics import test_native_collective_calculus as calculus
from tests.physics import test_native_resonance_live_prefix as prefix
from tnfr.constants import DEFAULTS
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER

domain = calculus.domain
no_execution = calculus.no_execution
calculus_guard = calculus.calculus_guard
DT = calculus.DT
RADIUS = Q(1, 4096)
WORD = ("AL", "EN", "IL", "RA", "NAV")


@pytest.fixture(scope="module", autouse=True)
def retention_guard(calculus_guard):
    # Importing earlier owners does not invoke their response fixtures.
    yield calculus_guard


def _fresh(graph, forms, capacities):
    """Independent exact zero-phase fine pressure, with no production reader."""
    k, b = (domain.base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    return tuple(
        k * (sum(forms[j] for j in graph.neighbors(n)) / 4 - forms[n])
        + b * (sum(capacities[j] for j in graph.neighbors(n)) / 4 - capacities[n])
        for n in graph
    )


@pytest.fixture(scope="module")
def certificate(retention_guard):
    graph = prefix._source()
    reference = prefix._forecast(graph)
    cfg = graph.graph
    k, b, alpha, beta, gamma, r, a, h = prefix._coefficients(graph)
    means = tuple(Q(graph.nodes[2 * i]["EPI"]) for i in range(5))
    capacity = prefix.CAPACITY
    factors = {name: Q(value) for name, value in cfg["GLYPH_FACTORS"].items()}
    assert tuple(graph) == tuple(range(10)) and max(capacity) == 1
    assert cfg["GAMMA"]["type"] == "none" and cfg["INTEGRATOR_METHOD"] == "euler"
    assert cfg["CLIP_MODE"] == "hard" and cfg.get("MATH_ENGINE") is None
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise" and cfg["GRAMMAR_CANON"]["enabled"]
    for name in ("DNFR_WEIGHTS", "SI_WEIGHTS", "GLYPH_FACTORS", "SELECTOR_THRESHOLDS"):
        assert cfg[name] == DEFAULTS[name]
    for node in graph:
        adjacent = ((node // 2 - 1) % 5, (node // 2 + 1) % 5)
        assert set(graph.neighbors(node)) == {
            2 * block + parity for block in adjacent for parity in (0, 1)
        }
        assert all(Q(edge["weight"]) == 1 for edge in graph[node].values())
        assert graph.nodes[node]["theta"] == 0
    centers = reference["initial_pressure"]
    peak = max(map(abs, centers))
    error = k * RADIUS
    normalized_error = 2 * error / (peak + error)
    assert 0 < peak and 0 < normalized_error < Q(13, 10000)
    low, high, cut = (
        Q(cfg["SELECTOR_THRESHOLDS"][key]) for key in ("si_lo", "si_hi", "dnfr_hi")
    )
    ratio_upper = tuple(min(Q(1), (abs(value) + error) / peak) for value in centers)
    ratio_lower = tuple(
        max(Q(0), abs(value) - error) / (peak + error) for value in centers
    )
    # These endpoints enclose the entire box before any fine controls. The
    # scalar peak association gives P0<=P<=P0+E. For E<P0, each endpoint's
    # deviation from |B|/P0 is at most 2E/(P0+E), including the peak row.
    assert error < peak
    assert all(
        max(abs(value) / peak - lower, upper - abs(value) / peak) <= normalized_error
        for value, lower, upper in zip(centers, ratio_lower, ratio_upper)
    )
    si_lower = tuple(
        alpha * v + beta + gamma * (1 - upper)
        for v, upper in zip(capacity, ratio_upper)
    )
    si_upper = tuple(
        alpha * v + beta + gamma * (1 - lower)
        for v, lower in zip(capacity, ratio_lower)
    )
    assert max(ratio_upper[:2]) < Q(7, 1000) < cut
    assert low < min(si_lower[:2]) <= max(si_upper[:2]) < high
    assert min(si_lower[2:]) > high
    # Nonadjacent twins and whole-pair order cancel every neighbor contrast.
    signal = (Q(0), -r * prefix.SOURCE_Q / 2)
    assert signal[0] == 0 and abs(signal[1]) > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
    assert min(means[:2]) - RADIUS > Q(11, 100)
    assert min(reference["event_form"][:2]) - (1 - r) * RADIUS > Q(9, 100)
    assert all(Q(cfg["VF_MIN"]) < v < Q(cfg["VF_MAX"]) for v in reference["capacity"])
    assert all(abs(value) + RADIUS < Q(3, 5) for value in reference["event_form"])
    # Nonzero RA jumps remain negative at both twins; IL jumps remain zero.
    jump = tuple(reference["event_form"][i] - means[i] for i in range(5))
    assert all(value + r * RADIUS < -Q(18, 1000) for value in jump[:2])
    assert jump[2:] == (0, 0, 0)
    assert h * min(abs(value) - error for value in centers[2:]) > Q(26, 1000)
    assert max(abs(value) + error for value in centers[:2]) < Q(cfg["EPS_DNFR_STABLE"])
    multipliers = tuple(
        (1 - r if i < 2 else Q(1))
        - DT * reference["capacity"][i] * (1 if i < 2 else h) * k
        for i in range(5)
    )
    assert all(0 < value < 1 for value in multipliers)
    assert max(multipliers) < Q(992, 1000)
    assert all(abs(value) + RADIUS < Q(3, 5) for value in reference["form"])
    return dict(
        graph=graph,
        reference=reference,
        means=means,
        capacity=capacity,
        k=k,
        b=b,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        r=r,
        a=a,
        h=h,
        factors=factors,
        centers=centers,
        peak=peak,
        pressure_error=error,
        normalized_error=normalized_error,
        ratio_lower=ratio_lower,
        ratio_upper=ratio_upper,
        si_lower=si_lower,
        si_upper=si_upper,
        signals=signal,
        multipliers=multipliers,
    )


def _source_controls(c):
    """Corners and axes check formulas; the uniform enclosure admits the box."""
    corners = tuple(product((-RADIUS, RADIUS), repeat=5))
    axes = tuple(
        tuple(sign * RADIUS if i == axis else Q(0) for i in range(5))
        for axis in range(5)
        for sign in (-1, 1)
    )
    return ((Q(0),) * 5, *corners, *axes)


def test_positive_peak_normalization_bound_retains_pressure_association(certificate):
    c = certificate
    assert c["peak"] > 0
    for contrast in _source_controls(c):
        forms = calculus._fine(c["means"], contrast)
        pressure = _fresh(c["graph"], forms, domain._paired(c["capacity"]))
        peak = max(map(abs, pressure))
        assert peak == max(
            abs(b) + c["k"] * abs(d) for b, d in zip(c["centers"], contrast)
        )
        assert c["peak"] <= peak <= c["peak"] + c["pressure_error"]
        for node, p in enumerate(pressure):
            block = node // 2
            ratio = abs(p) / peak
            assert c["ratio_lower"][block] <= ratio <= c["ratio_upper"][block]
            assert (
                abs(ratio - abs(c["centers"][block]) / c["peak"])
                <= c["normalized_error"]
            )
            si = (
                c["alpha"] * c["capacity"][block] + c["beta"] + c["gamma"] * (1 - ratio)
            )
            assert c["si_lower"][block] <= si <= c["si_upper"][block]
            assert (
                abs(si - c["reference"]["si"][block])
                <= c["gamma"] * c["normalized_error"]
            )
    # The denominator change matters even where that pair has no contrast.
    contrast = (RADIUS, Q(0), Q(0), RADIUS, Q(0))
    peak = c["peak"] + c["k"] * RADIUS
    assert abs(c["centers"][1]) / peak != abs(c["centers"][1]) / c["peak"]


def test_actual_default_request_and_lag_margins_are_uniform(
    certificate, retention_guard
):
    c, (targets, forbidden) = certificate, retention_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    cfg, history = c["graph"].graph, c["graph"].graph["history"]
    thresholds = cfg["SELECTOR_THRESHOLDS"]
    for node, data in c["graph"].nodes(data=True):
        i = node // 2
        # Actual increment-before-priority means neither forced mark is due.
        assert history["since_AL"][node] + 1 <= cfg["AL_MAX_LAG"]
        assert history["since_EN"][node] + 1 <= cfg["EN_MAX_LAG"]
        assert tuple(data["glyph_history"]) == ("EN",)
        assert data[domain.PRIOR_COHERENCE_KEY] is True
        assert data[domain.OPERATOR_STEP_KEY] == 3 and data["epi_kind"] == "seed"
        assert data[domain.U2_DEBT_KEY] == i % 2
        assert data["stable_count"] == 0 and not data.get("epi_time_history")
        if i < 2:
            assert Q(thresholds["si_lo"]) < c["si_lower"][i]
            assert c["si_upper"][i] < Q(thresholds["si_hi"])
            assert c["ratio_upper"][i] < Q(thresholds["dnfr_hi"])
        else:
            assert c["si_lower"][i] > Q(thresholds["si_hi"])
    # These inequalities identify RA/RA/IL/IL/IL under the default branch
    # order, not a supplied mark table or a call to a parallel selector.
    assert calculus._compatible(Q(0), (Q(0), Q(0)), calculus.PI / 2) == (0, 1)


def test_every_singleton_prefix_keeps_identity_boost_and_held_flow(certificate):
    c, graph = certificate, certificate["graph"]
    for contrast in _source_controls(c):
        original = calculus._fine(c["means"], contrast)
        live, capacity = list(original), list(domain._paired(c["capacity"]))
        held = list(_fresh(graph, original, capacity))
        for node in graph:
            if node < 4:
                signal = sum(live[j] for j in graph.neighbors(node)) / 4
                assert signal == c["signals"][node // 2]
                proposal = (1 - c["r"]) * live[node] + c["r"] * signal
                assert live[node] > 0 and proposal > Q(9, 100)
                live[node] = proposal
                if abs(signal) > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER):
                    capacity[node] *= 1 + c["a"]
            else:
                held[node] *= c["h"]
            assert all(-1 < value < 1 for value in live)
        assert tuple(capacity) == domain._paired(c["reference"]["capacity"])
        assert calculus._moments(live)[0] == c["reference"]["event_form"]
        rate = tuple(v * p for v, p in zip(capacity, held))
        endpoint = tuple(x + DT * dx for x, dx in zip(live, rate))
        means, differences = calculus._moments(endpoint)
        assert means == c["reference"]["form"]
        assert differences == tuple(a * d for a, d in zip(c["multipliers"], contrast))
        assert max(map(abs, differences)) <= max(c["multipliers"]) * RADIUS < RADIUS
        assert all(abs(value) < Q(3, 5) for value in endpoint)
        for node in graph:
            block = node // 2
            assert (live[node] != original[node]) is (node < 4)
            # Originally empty histories acquire the live left boundary;
            # RA replaces it and IL keeps it. Neither requires invented past.
            temporal = ((Q(1), live[node]), (1 + DT, endpoint[node]))
            assert (temporal[1][1] - temporal[0][1]) / DT == rate[node]
            acceleration = (rate[node] - Q(graph.nodes[node]["dEPI"])) / DT
            assert acceleration * DT + Q(graph.nodes[node]["dEPI"]) == rate[node]
            high = Q(graph.graph["SELECTOR_THRESHOLDS"]["si_hi"])
            if block < 2:
                assert c["si_upper"][block] < high
            else:
                assert abs(held[node]) > Q(graph.graph["EPS_DNFR_STABLE"])


def test_phase_and_counter_memory_do_not_follow_from_contrast_alone(certificate):
    c, cfg = certificate, certificate["graph"].graph
    # Here the independently checked adaptation failures retain every zero
    # count. Paired capacities are not inferred solely from contraction.
    assert cfg["VF_ADAPT_TAU"] > 1
    for node in c["graph"]:
        assert c["graph"].nodes[node]["stable_count"] == 0
        # Neutral RA retains incoming debt; later IL resolves it, including
        # the source's nonzero debt at block three.
        old_debt = c["graph"].nodes[node][domain.U2_DEBT_KEY]
        next_debt = old_debt if node < 4 else 0
        assert next_debt == int(node // 2 == 1)
    traces = tuple(("EN", "RA" if node < 4 else "IL") for node in c["graph"])
    assert sum(mark in ("OZ", "ZHIR", "NAV") for trace in traces for mark in trace) == 0
    policy = cfg["PHASE_ADAPT"]
    assert Q(1) > Q(policy["R_hi"]) and 0 <= Q(policy["disr_lo"])
    gains = tuple(
        Q(cfg[f"PHASE_K_{name}"])
        + Q(policy["down"]) * (Q(policy[f"k{suffix}_min"]) - Q(cfg[f"PHASE_K_{name}"]))
        for name, suffix in (("GLOBAL", "G"), ("LOCAL", "L"))
    )
    assert gains == c["reference"]["gains"]
    assert sum(gains) < 1 and all(value > 0 for value in gains)
    assert not cfg["history"].get("stable_frac") and "_REMESH_META" not in cfg
    # The same trace/lag/debt source is retained; this one-call conclusion
    # does not promise the next prepared request or future count equality.


def _conditional_word(c, initial_difference, innovations):
    """Fine algebra with prescribed common neighbor input, not a trajectory.

    This local word composes the five admitted row identities; its prescribed
    neighbor mean is 1/2, phases are zero, and neighbor capacity matches the
    current target capacity. It does not assert that the surrounding graph
    autonomously maintains those inputs or selects these marks. Mean
    innovation is retained in the independent collective reference.
    """
    f, k = c["factors"], c["k"]
    forms = (Q(1, 4) - initial_difference, Q(1, 4) + initial_difference)
    mean, capacity, neighbor = Q(1, 4), Q(1, 4), Q(1, 2)
    radii, coefficients, actual, means = [RADIUS], [], [initial_difference], [mean]
    for mark in WORD:
        prepared = tuple(k * (neighbor - value) for value in forms)
        baseline = k * (neighbor - mean)
        event, held, a, h = forms, prepared, Q(1), Q(1)
        y = mean
        if mark == "AL":
            event = tuple(value + f["AL_boost"] for value in forms)
            y += f["AL_boost"]
        elif mark in {"EN", "RA"}:
            mix = f["EN_mix"] if mark == "EN" else f["RA_epi_diff"]
            event = tuple((1 - mix) * value + mix * neighbor for value in forms)
            a, y = 1 - mix, (1 - mix) * mean + mix * neighbor
            if mark == "RA":
                assert min(forms) > 0 and min(event) > 0
                assert neighbor > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
                capacity *= 1 + f["RA_vf_amplification"]
        elif mark == "IL":
            h = f["IL_dnfr_factor"]
            held = tuple(h * value for value in prepared)
        else:
            assert mark == "NAV" and min(prepared) > 0
            h = 1 - f["NAV_eta"]
            held = tuple(
                h * value + f["NAV_eta"] * capacity + noise
                for value, noise in zip(prepared, innovations)
            )
            assert min(abs(new - old) for new, old in zip(held, prepared)) > Q(7, 100)
        zeta = (innovations[1] - innovations[0]) / 2 if mark == "NAV" else Q(0)
        forcing = (
            f["NAV_eta"] * capacity + sum(innovations) / 2 if mark == "NAV" else Q(0)
        )
        coefficient = a - DT * capacity * h * k
        assert 0 < coefficient < 1
        radius = abs(coefficient) * radii[-1] + DT * capacity * (
            f["NAV_jitter"] / 4 if mark == "NAV" else 0
        )
        forms = tuple(value + DT * capacity * p for value, p in zip(event, held))
        mean = y + DT * capacity * (h * baseline + forcing)
        observed = (forms[1] - forms[0]) / 2
        assert sum(forms) / 2 == mean
        assert observed == coefficient * actual[-1] + DT * capacity * zeta
        assert abs(observed) <= radius
        assert min(forms) > 0 and max(forms) < Q(3, 4)
        coefficients.append((coefficient, DT * capacity, zeta))
        actual.append(observed)
        radii.append(radius)
        means.append(mean)
    return tuple(actual), tuple(radii), tuple(coefficients), tuple(means)


def test_finite_five_row_bound_retains_mean_innovation_and_contrast_inputs(certificate):
    c, outcomes = certificate, []
    amplitude = c["factors"]["NAV_jitter"]
    for sign, innovation_sign in product((-1, 1), repeat=2):
        noise_mean, noise_difference = amplitude / 4, innovation_sign * amplitude / 4
        innovations = noise_mean - noise_difference, noise_mean + noise_difference
        assert max(map(abs, innovations)) <= amplitude
        actual, radii, rows, means = _conditional_word(c, sign * RADIUS, innovations)
        product_gain = Q(1)
        for coefficient, _, _ in rows:
            product_gain *= coefficient
        # Independent finite variation-of-constants expansion.
        forcing_sum = Q(0)
        for j, (_, scale, noise) in enumerate(rows):
            tail = Q(1)
            for coefficient, _, _ in rows[j + 1 :]:
                tail *= coefficient
            forcing_sum += tail * scale * noise
        assert actual[-1] == product_gain * actual[0] + forcing_sum
        assert radii[-1] == product_gain * RADIUS + rows[-1][1] * amplitude / 4
        # Every homogeneous row contracts, yet bounded innovation can widen
        # the final enclosure. Retention is not unconditional contraction.
        assert RADIUS < radii[-1] < Q(1, 2500)
        outcomes.append((actual, radii, means))
    assert all(value[2] == outcomes[0][2] for value in outcomes)
    assert len({value[0][-1] for value in outcomes}) == 4
    # Suppressing a common innovation changes the mean but not contrast.
    zero_mean = _conditional_word(c, RADIUS, (-amplitude / 4, amplitude / 4))
    positive_mean = _conditional_word(c, RADIUS, (Q(0), amplitude / 2))
    assert zero_mean[0] == positive_mean[0]
    assert (
        positive_mean[3][-1] - zero_mean[3][-1]
        == positive_mean[2][-1][1] * amplitude / 4
    )


def test_first_failed_strict_margin_stops_the_conditional_sector(certificate):
    c, f = certificate, certificate["factors"]
    baseline, capacity = Q(1, 50), Q(1, 4)
    critical = baseline / c["k"]
    # Increase only contrast. The strict common sign is the first of these
    # row premises to fail; nonidentity and scalar rails still have slack.
    for radius, valid in (
        (critical / 2, True),
        (critical, False),
        (3 * critical / 2, False),
    ):
        pressure = (baseline + c["k"] * radius, baseline - c["k"] * radius)
        assert (min(pressure) > 0) is valid
        assert f["NAV_eta"] * (capacity - max(pressure)) - f["NAV_jitter"] > Q(5, 100)
        assert Q(1, 4) - radius > 0 and Q(1, 4) + radius < 1
    # At equality the actual positive-zero sign exists, but the strict-sign
    # theorem has exited. A negative second pressure needs its distinct sign.
    radius = 3 * critical / 2
    even, odd = baseline + c["k"] * radius, baseline - c["k"] * radius
    h, eta = 1 - f["NAV_eta"], f["NAV_eta"]
    real_mean = ((h * even + eta * capacity) + (h * odd - eta * capacity)) / 2
    invalid_continuation = h * baseline + eta * capacity
    assert invalid_continuation - real_mean == eta * capacity


def test_zero_pressure_has_no_uniform_default_selection_neighborhood(certificate):
    c, cfg = certificate, certificate["graph"].graph
    means = tuple(-c["b"] * v / c["k"] for v in c["capacity"])
    forms = calculus._fine(means, (Q(0),) * 5)
    capacities = domain._paired(c["capacity"])
    assert _fresh(c["graph"], forms, capacities) == (0,) * 10
    low, high = (Q(cfg["SELECTOR_THRESHOLDS"][name]) for name in ("si_lo", "si_hi"))
    cut = Q(cfg["SELECTOR_THRESHOLDS"]["dnfr_hi"])
    reference_si = c["alpha"] / 4 + c["beta"] + c["gamma"]
    perturbed_si = c["alpha"] / 4 + c["beta"]
    assert low < perturbed_si < reference_si < high and 0 < cut < 1
    # For every positive ideal t, the pressure row is exactly t times this
    # coefficient vector. Dividing by its peak cancels t: ratios are one at
    # the perturbed pair, whereas P=0 uses divisor one and ratio zero.
    unit_row = (c["k"], -c["k"], *(Q(0),) * 8)
    for exponent in (12, 40, 200, 1200):
        t = Q(1, 2**exponent)
        perturbed = calculus._fine(means, (t, *(Q(0),) * 4))
        pressure = _fresh(c["graph"], perturbed, capacities)
        assert pressure == tuple(t * value for value in unit_row)
        assert max(map(abs, pressure)) == c["k"] * t > 0
        assert abs(pressure[0]) / max(map(abs, pressure)) == 1 > cut
        assert all(-1 < value < 0 for value in perturbed)
        assert max(abs(a - b) for a, b in zip(perturbed, forms)) == t
    # Under the actual middle-Si branch this is RA at t=0 and NAV at every
    # sufficiently small t>0 that retains the rails, with unchanged means,
    # positive capacities and unforced source lags.
    # Tiny rational t is not asserted to survive native float materialization.


def test_zero_peak_can_still_have_robust_coherence_requests(certificate):
    c, cfg = certificate, certificate["graph"].graph
    means, capacities = (Q(1, 8),) * 5, (Q(1),) * 10
    assert _fresh(c["graph"], domain._paired(means), capacities) == (0,) * 10
    high = Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
    # This independent capacity/phase floor admits IL before the pressure
    # branch. A discontinuous diagnostic need not change the actual request.
    assert c["alpha"] + c["beta"] > high
    assert c["h"] * c["k"] * RADIUS < Q(cfg["EPS_DNFR_STABLE"])
    for contrast in _source_controls(c):
        forms = calculus._fine(means, contrast)
        pressure = _fresh(c["graph"], forms, capacities)
        peak = max(map(abs, pressure))
        denominator = peak if peak else Q(1)
        ratios = tuple(abs(value) / denominator for value in pressure)
        assert max(ratios) == (1 if peak else 0)
        for ratio, p in zip(ratios, pressure):
            si = c["alpha"] + c["beta"] + c["gamma"] * (1 - ratio)
            assert si >= c["alpha"] + c["beta"] > high
            assert abs(c["h"] * p) <= c["h"] * c["k"] * RADIUS
        assert all(0 < value < 1 for value in forms)
    # From supplied zero counters both twins qualify and become one, below
    # tau. This is a separate one-call condition, not an invariant family.
    assert 1 < cfg["VF_ADAPT_TAU"]
