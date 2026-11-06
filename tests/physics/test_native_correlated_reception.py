"""Lag-forced Reception acts on the complete correlated Transition image.

This is an analytical continuation of test_native_contrast_domain, not a
native fourth call. The same six input coordinates and 64 input vertices
remain associated with every form, incoming rate and temporal sample. Their
affine image is never replaced by independent marginal intervals. The old
ordered Reception theorem is reused, with live fine prefixes checked against
the block sweep. Default lags select Reception (EN) on all ten targets.

Fresh pressure, event form, held-pressure Euler form, rate and acceleration
are exact affine rows. The fresh pressure peak now belongs to either twin
of block one: P=B1+k*abs(d1). Its ownership can switch, so Si is retained as
an exact max-normalized function rather than incorrectly declared affine.
No new random draw is consumed. Absolute phase retains its inherited common
origin; the phase branch follows exact rational cosine bounds.

All runtime, selector, pressure, operator and integration boundaries are
disabled. This ideal-real represented-coefficient certificate supplies no
native binary64 response, fixed-seed reachability of the whole amplitude
domain, indefinitely closed collective state or physical identification.
"""

from fractions import Fraction as Q

import pytest

from tests.physics import test_native_contrast_domain as domain
from tests.physics import test_native_reception_envelope as reception
from tnfr.operators.grammar_types import (
    COUPLING_RESONANCE,
    DESTABILIZERS,
    STABILIZERS,
    TRANSFORMERS,
)

# Import only the analytical fixtures. Their explicit dependency installs the
# fail-fast guard before any earlier pure forecast is constructed.
no_execution = domain.no_execution
contrast_domain = domain.certificate
DT = domain.DT
A = domain._Affine
ZERO = (Q(0),) * domain.DIMENSION


def _stable_gains(policy, current):
    return tuple(
        value + Q(policy["down"]) * (Q(policy[f"k{suffix}_min"]) - value)
        for value, suffix in zip(current, ("G", "L"), strict=True)
    )


def _phase_row(phase, gains):
    kg, kl = gains
    return tuple(
        (1 - kg - kl) * value + kl * domain._neighbor(phase, i)
        for i, value in enumerate(phase)
    )


def _pressure_rows(source, forms, offsets):
    """Independent mixed-pressure formula on the retained full fine support."""
    graph, v = source["graph"], source["capacity"]
    k, b, w = source["weights"]
    return tuple(
        k * (sum((forms[j] for j in graph.neighbors(n)), A.constant(0)) / 4 - forms[n])
        + b * (domain._neighbor(v, n // 2) - v[n // 2])
        + w * offsets[n // 2] / domain.gain.PI
        for n in graph
    )


def _sense_at(certificate, point):
    """Exact observation with the live, contrast-dependent pressure maximum."""
    pressure = tuple(row.at(point) for row in certificate["pressure"])
    peak = max(map(abs, pressure))
    gamma = certificate["si_gamma"]
    return tuple(
        certificate["si_base"][n // 2] + gamma * (1 - abs(value) / peak)
        for n, value in enumerate(pressure)
    )


@pytest.fixture(scope="module", autouse=True)
def reception_no_execution(no_execution):
    """Also reject accidental use of the old response runner or EN primitive."""
    targets, forbidden = no_execution
    additions = (
        (domain.base, "_execute"),
        (domain.operators, "_op_EN"),
        (domain.operators, "_mix_epi_with_neighbors"),
    )
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in additions:
            patch.setattr(owner, name, forbidden)
        yield (*targets, *additions), forbidden


@pytest.fixture(scope="module")
def certificate(contrast_domain, reception_no_execution):
    c = contrast_domain
    graph, cfg = c["graph"], c["graph"].graph
    forms, rates_before = c["endpoint"], c["rates"]
    means, contrasts = domain._moments(forms)
    v, policy = c["capacity"], cfg["PHASE_ADAPT"]
    gains3 = _stable_gains(policy, c["second"]["phase_certificate"]["gains"])
    phase = _phase_row(c["phase"], gains3)
    offsets = tuple(domain._neighbor(phase, i) - phase[i] for i in range(5))
    pressure = _pressure_rows(c, forms, offsets)
    pressure_means, pressure_contrasts = domain._moments(pressure)
    k, b, w = c["weights"]
    centers = tuple(
        k * (domain._neighbor(means, i) - means[i])
        + b * (domain._neighbor(v, i) - v[i])
        + w * offsets[i] / domain.gain.PI
        for i in range(5)
    )
    assert pressure_means == centers
    assert pressure_contrasts == tuple(-k * d for d in contrasts)
    # All rows being compared are affine and signs are fixed. These strict
    # vertex inequalities prove block-one ownership throughout the input hull;
    # they do not force one of its two twins to remain the unique maximizer.
    for point in c["vertices"]:
        p = tuple(row.at(point) for row in pressure)
        assert all(
            domain.SIGNS[n // 2] * value > Q(72, 1000) for n, value in enumerate(p)
        )
        for peak_node in (2, 3):
            assert all(
                p[peak_node] - domain.SIGNS[n // 2] * value > Q(18, 100000)
                for n, value in enumerate(p)
                if n not in (2, 3)
            )
    alpha, beta, gamma = (
        domain.base._weight(graph, "SI_WEIGHTS", name)
        for name in ("alpha", "beta", "gamma")
    )
    si_base = tuple(
        alpha * v[i] + beta * (1 - abs(offsets[i]) / domain.gain.PI) for i in range(5)
    )
    mix = Q(cfg["GLYPH_FACTORS"]["EN_mix"])
    assert 0 < mix < 1
    current = list(forms)
    prefixes, jumps, neighbor_means = [], [], []
    for node in graph:
        local = sum((current[j] for j in graph.neighbors(node)), A.constant(0)) / 4
        before = current[node]
        current[node] = (1 - mix) * before + mix * local
        jumps.append(current[node] - before)
        neighbor_means.append(local)
        prefixes.append(tuple(current))
    event_form = tuple(current)
    event_means = reception._ordered_reception(means, mix)
    event_contrasts = tuple((1 - mix) * d for d in contrasts)
    assert domain._moments(event_form) == (event_means, event_contrasts)
    rates = tuple(v[n // 2] * p for n, p in enumerate(pressure))
    endpoint = tuple(x + DT * rate for x, rate in zip(event_form, rates, strict=True))
    acceleration = tuple(
        (rate - old) / DT for rate, old in zip(rates, rates_before, strict=True)
    )
    simultaneous = tuple(
        (1 - mix) * value + mix * domain._neighbor(means, i)
        for i, value in enumerate(means)
    )
    refreshed_pressure = _pressure_rows(c, event_form, offsets)
    refreshed_endpoint = tuple(
        x + DT * v[n // 2] * refreshed_pressure[n] for n, x in enumerate(event_form)
    )
    traces = tuple((*trace, "EN") for trace in c["next_traces"])
    source_temporal = tuple(
        (*samples, (Q(19, 16), forms[n])) for n, samples in enumerate(c["temporal"])
    )
    return dict(
        source=c,
        forms=forms,
        means=means,
        contrasts=contrasts,
        rates_before=rates_before,
        acceleration_before=c["acceleration"],
        phase=phase,
        gains_before=gains3,
        offsets=offsets,
        pressure=pressure,
        centers=centers,
        si_base=si_base,
        si_gamma=gamma,
        mix=mix,
        neighbor_means=tuple(neighbor_means),
        prefixes=tuple(prefixes),
        jumps=tuple(jumps),
        event_form=event_form,
        event_means=event_means,
        event_contrasts=event_contrasts,
        rates=rates,
        endpoint=endpoint,
        acceleration=acceleration,
        simultaneous=simultaneous,
        refreshed_pressure=refreshed_pressure,
        refreshed_endpoint=refreshed_endpoint,
        traces=traces,
        source_temporal=source_temporal,
    )


def test_unchanged_correlated_source_and_disabled_runtime(
    certificate, reception_no_execution
):
    c, source = certificate, certificate["source"]
    targets, forbidden = reception_no_execution
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert c["forms"] is source["endpoint"]
    assert c["rates_before"] is source["rates"]
    assert len(source["vertices"]) == 64
    assert all(len(point) == domain.DIMENSION for point in source["vertices"])
    assert c["forms"][2].coefficients[2] != 0
    assert c["forms"][0].coefficients[3] != 0
    assert c["forms"][2].coefficients[5] != 0
    for node, samples in enumerate(c["source_temporal"]):
        assert samples[-1] == (Q(19, 16), c["forms"][node])
        assert (samples[-1][1] - samples[-2][1]) / DT == c["rates_before"][node]
    for rate, old, acceleration in zip(
        c["rates_before"], source["old_rates"], c["acceleration_before"], strict=True
    ):
        assert acceleration == (rate - old) / DT


def test_forced_reception_uses_lag_memory_despite_nonaffine_normalizer(certificate):
    c, source = certificate, certificate["source"]
    cfg, k = source["graph"].graph, source["weights"][0]
    assert (
        cfg["GRAMMAR_REJECTION_MODE"] == "raise"
        and cfg["GRAMMAR_CANON"]["enabled"] is True
    )
    assert 3 + 1 <= cfg["AL_MAX_LAG"] and 3 + 1 > cfg["EN_MAX_LAG"]
    assert (
        "reception"
        not in STABILIZERS | DESTABILIZERS | TRANSFORMERS | COUPLING_RESONANCE
    )
    for base in c["si_base"]:
        assert 0 < base < base + c["si_gamma"] < 1
    assert max(source["capacity"]) == 1
    pressure_values = tuple(
        tuple(row.at(point) for row in c["pressure"]) for point in source["vertices"]
    )
    # Positive affine block-one rows bound the maximum on the entire hull.
    # This global envelope handles switching/tied peak ownership without
    # treating the max-normalized Si itself as an affine vertex quantity.
    peak_lower = min(p[n] for p in pressure_values for n in (2, 3))
    peak_upper = max(p[n] for p in pressure_values for n in (2, 3))
    assert Q(2014, 10000) < peak_lower <= peak_upper < Q(2021, 10000)
    si_windows = (
        (Q(395, 1000), Q(396, 1000)),
        (Q(364, 1000), Q(365, 1000)),
        (Q(932, 1000), Q(934, 1000)),
        (Q(889, 1000), Q(890, 1000)),
        (Q(949, 1000), Q(950, 1000)),
    )
    for block, base in enumerate(c["si_base"]):
        magnitudes = tuple(
            domain.SIGNS[block] * p[n]
            for p in pressure_values
            for n in (2 * block, 2 * block + 1)
        )
        ratio_lower = min(magnitudes) / peak_upper
        ratio_upper = min(Q(1), max(magnitudes) / peak_lower)
        lower = base + c["si_gamma"] * (1 - ratio_upper)
        upper = base + c["si_gamma"] * (1 - ratio_lower)
        assert si_windows[block][0] < lower <= upper < si_windows[block][1]
    for point in source["vertices"]:
        pressure = tuple(row.at(point) for row in c["pressure"])
        peak = c["centers"][1].at(point) + k * abs(c["contrasts"][1].at(point))
        assert max(map(abs, pressure)) == peak
        assert peak_lower <= peak <= peak_upper
        assert all(0 < value < 1 for value in _sense_at(c, point))
        # Preparation still admits finite prior acceleration even though the
        # all-forced batch never consumes a parametric acceleration policy.
        assert max(abs(row.at(point)) for row in c["acceleration_before"]) < 4
    # Same means but opposite contrasts switch the maximizing fine target.
    half = source["radius"] / 2
    plus, minus = (Q(0), half, Q(0), Q(0), Q(0), Q(0)), (
        Q(0),
        -half,
        Q(0),
        Q(0),
        Q(0),
        Q(0),
    )
    assert tuple(row.at(plus) for row in c["means"]) == tuple(
        row.at(minus) for row in c["means"]
    )
    assert c["pressure"][2].at(plus) > c["pressure"][3].at(plus)
    assert c["pressure"][2].at(minus) < c["pressure"][3].at(minus)
    assert c["pressure"][2].at(ZERO) == c["pressure"][3].at(ZERO)
    assert _sense_at(c, plus)[2:4] == tuple(reversed(_sense_at(c, minus)[2:4]))
    assert _sense_at(c, plus)[2] != _sense_at(c, plus)[3]


def test_live_fine_prefixes_descend_to_the_ordered_mean_sweep(certificate):
    c = certificate
    means, contrasts, mix = c["means"], c["contrasts"], c["mix"]
    current_means = list(means)
    for block in range(5):
        local = domain._neighbor(current_means, block)
        assert (
            c["neighbor_means"][2 * block]
            == c["neighbor_means"][2 * block + 1]
            == local
        )
        current_means[block] = (1 - mix) * current_means[block] + mix * local
        got_means, got_contrasts = domain._moments(c["prefixes"][2 * block + 1])
        assert got_means == tuple(current_means)
        assert got_contrasts == tuple(
            (1 - mix) * d if i <= block else d for i, d in enumerate(contrasts)
        )
    assert tuple(current_means) == c["event_means"]
    assert c["event_form"] == c["prefixes"][-1]
    assert all(
        data["epi_kind"] == "seed" for _, data in c["source"]["graph"].nodes(data=True)
    )
    # Equal nonempty neighbor kinds remain seed regardless of which live
    # amplitude is largest; no emission-provenance condition is introduced.
    assert all(len(tuple(c["source"]["graph"].neighbors(n))) == 4 for n in range(10))


def test_held_pressure_flow_and_mean_contrast_identity(certificate):
    c, source = certificate, certificate["source"]
    v, k, mix = source["capacity"], source["weights"][0], c["mix"]
    actual_means, actual_contrasts = domain._moments(c["endpoint"])
    assert actual_means == tuple(
        y + DT * v[i] * c["centers"][i] for i, y in enumerate(c["event_means"])
    )
    assert actual_contrasts == tuple(
        (1 - mix - DT * v[i] * k) * d for i, d in enumerate(c["contrasts"])
    )
    assert all(0 < 1 - mix - DT * value * k < 1 - mix for value in v)
    mean_rates, contrast_rates = domain._moments(c["rates"])
    assert mean_rates == tuple(v[i] * row for i, row in enumerate(c["centers"]))
    assert contrast_rates == tuple(
        -v[i] * k * row for i, row in enumerate(c["contrasts"])
    )
    old_means, old_contrasts = domain._moments(c["rates_before"])
    assert domain._moments(c["acceleration"]) == (
        tuple((new - old) / DT for new, old in zip(mean_rates, old_means, strict=True)),
        tuple(
            (new - old) / DT
            for new, old in zip(contrast_rates, old_contrasts, strict=True)
        ),
    )


def test_all_prefixes_and_endpoints_are_admitted_on_the_joint_domain(certificate):
    c, source = certificate, certificate["source"]
    cfg = source["graph"].graph
    final_windows = (
        (Q(119, 1000), Q(121, 1000)),
        (Q(-367, 1000), Q(-364, 1000)),
        (Q(667, 1000), Q(668, 1000)),
        (Q(-84, 1000), Q(-83, 1000)),
        (Q(-675, 1000), Q(-674, 1000)),
    )
    for point in source["vertices"]:
        for node in range(10):
            block = node // 2
            assert domain.SIGNS[block] * c["jumps"][node].at(point) > Q(15, 1000)
            pressure = c["pressure"][node].at(point)
            assert (
                domain.SIGNS[block] * pressure > Q(72, 1000) > Q(cfg["EPS_DNFR_STABLE"])
            )
            # All event prefixes are convex mixtures, but checking their
            # exact rows also detects a wrong live target/order association.
            for prefix in c["prefixes"]:
                assert Q(cfg["EPI_MIN"]) < prefix[node].at(point) < Q(cfg["EPI_MAX"])
            lower, upper = final_windows[block]
            assert (
                Q(cfg["EPI_MIN"])
                < lower
                < c["endpoint"][node].at(point)
                < upper
                < Q(cfg["EPI_MAX"])
            )
            assert abs(c["rates"][node].at(point)) < 1
            assert abs(c["acceleration"][node].at(point)) < 2
    # Held magnitudes cannot satisfy the stability test, independently of Si;
    # incoming zero counters stay zero and no capacity adaptation qualifies.
    assert cfg["VF_ADAPT_TAU"] > 1


def test_simultaneous_mixing_and_postevent_refresh_are_distinct_models(certificate):
    c, source = certificate, certificate["source"]
    refreshed_mean = domain._moments(c["refreshed_endpoint"])[0]
    held_mean = domain._moments(c["endpoint"])[0]
    differences = tuple(
        live - simultaneous
        for live, simultaneous in zip(c["event_means"], c["simultaneous"], strict=True)
    )
    assert differences[0] == A.constant(0)
    for point in source["vertices"]:
        assert differences[1].at(point) < Q(-34, 1000)
        assert differences[2].at(point) > Q(34, 1000)
        assert differences[3].at(point) < Q(-33, 1000)
        assert differences[4].at(point) < Q(-36, 1000)
        assert (refreshed_mean[0] - held_mean[0]).at(point) > Q(168, 100000)
        assert (c["refreshed_endpoint"][4] - c["endpoint"][4]).at(point) > Q(47, 10000)
    held_contrasts = domain._moments(c["endpoint"])[1]
    refreshed_contrasts = domain._moments(c["refreshed_endpoint"])[1]
    for i, d in enumerate(c["contrasts"]):
        v, k, mix = source["capacity"][i], source["weights"][0], c["mix"]
        assert refreshed_contrasts[i] == (1 - DT * v * k) * (1 - mix) * d
        assert held_contrasts[i] - refreshed_contrasts[i] == -DT * v * k * mix * d


def test_same_mean_and_same_form_witnesses_separate_full_retained_state(certificate):
    c, source = certificate, certificate["source"]
    t = source["radius"] / 2
    contrast_only = (Q(0), t, Q(0), Q(0), Q(0), Q(0))
    mean_output, contrast_output = domain._moments(c["endpoint"])
    assert tuple(row.at(ZERO) for row in mean_output) == tuple(
        row.at(contrast_only) for row in mean_output
    )
    assert contrast_output[1].at(ZERO) != contrast_output[1].at(contrast_only)

    # Compensate old d with the subsequent opposite innovation contrast.
    # Current forms then coincide exactly while incoming rates remember d.
    damping = (
        1 - DT * source["capacity"][1] * (1 - source["eta"]) * source["weights"][0]
    )
    compensate = damping * t / (DT * source["capacity"][1])
    same_form = (Q(0), t, Q(0), Q(0), compensate, -compensate)
    assert t < source["radius"] and compensate < source["amplitude"]
    for rows in (c["forms"], c["pressure"], c["event_form"], c["endpoint"], c["rates"]):
        assert tuple(row.at(ZERO) for row in rows) == tuple(
            row.at(same_form) for row in rows
        )
    assert c["rates_before"][2].at(same_form) - c["rates_before"][2].at(ZERO) == t / DT
    assert c["rates_before"][3].at(same_form) - c["rates_before"][3].at(ZERO) == -t / DT
    assert (
        c["acceleration"][2].at(same_form) - c["acceleration"][2].at(ZERO) == -t / DT**2
    )
    assert (
        c["acceleration"][3].at(same_form) - c["acceleration"][3].at(ZERO) == t / DT**2
    )
    assert c["source_temporal"][2][-2][1].at(same_form) != c["source_temporal"][2][-2][
        1
    ].at(ZERO)
    # These points belong to the declared amplitude domain. They are not two
    # different realizations of the same fixed seed and target RNG progress.


def test_fourth_phase_coordinator_retains_the_common_origin(certificate):
    c, cfg = certificate, certificate["source"]["graph"].graph
    phase, policy = c["phase"], cfg["PHASE_ADAPT"]
    assert 0 < min(phase) < max(phase) < domain.gain.PI
    assert max(phase) - min(phase) < Q(129, 100) < domain.gain.PI
    differences = tuple(phase[i] - phase[j] for i in range(5) for j in range(i + 1, 5))
    lower = (5 + 2 * sum(domain._cosine(d, 8) for d in differences)) / 25
    upper = (5 + 2 * sum(domain._cosine(d, 7) for d in differences)) / 25
    assert (
        Q(policy["R_hi"]) ** 2 < Q(888, 1000) ** 2 < lower < upper < Q(889, 1000) ** 2
    )
    load = Q(
        sum(glyph == "NAV" for trace in c["traces"] for glyph in trace),
        sum(map(len, c["traces"])),
    )
    assert load == Q(3, 25) < Q(policy["disr_lo"])
    gains = _stable_gains(policy, c["gains_before"])
    assert Q(47, 1000) < gains[0] < Q(48, 1000)
    assert Q(111, 1000) < gains[1] < Q(112, 1000)
    for old, new, suffix in zip(c["gains_before"], gains, ("G", "L"), strict=True):
        assert Q(policy[f"k{suffix}_min"]) < new < old
    assert sum(gains) < 1
    relative = _phase_row(phase, gains)
    assert (
        max(relative) - min(relative)
        < (1 - gains[0]) * (max(phase) - min(phase))
        < Q(1229, 1000)
    )
    origin = A.variable(0)
    raw = tuple(origin + z for z in phase)
    assert tuple(domain._neighbor(raw, i) - raw[i] for i in range(5)) == tuple(
        A.constant(value) for value in c["offsets"]
    )
    # The absolute endpoint is origin + kg4*Arg(sum exp(i*phase)) + relative.
    # Origin is inherited phase state, not a quotient coordinate set to zero.


def test_event_history_resets_and_nonstochastic_memory(certificate):
    c, source = certificate, certificate["source"]
    cfg = source["graph"].graph
    assert cfg["GLYPH_HYSTERESIS_WINDOW"] >= 5
    for node in range(10):
        assert c["traces"][node] == (*source["next_traces"][node], "EN")
        assert len(c["traces"][node]) == 5
        assert source["graph"].nodes[node][domain.OPERATOR_STEP_KEY] + 4 == 7
        assert source["graph"].nodes[node][domain.PRIOR_COHERENCE_KEY] is True
        debt = source["graph"].nodes[node][domain.U2_DEBT_KEY]
        al_lag = source["graph"].graph["history"]["since_AL"][node]
        en_lag = source["graph"].graph["history"]["since_EN"][node]
        for mark in (
            domain.gain.MARKS[node][1],
            domain.transition.MARKS[node][1],
            domain.MARKS[node // 2],
            "EN",
        ):
            debt = max(0, debt - 1) if mark == "IL" else debt
            al_lag = 0 if mark == "AL" else al_lag + 1
            en_lag = 0 if mark == "EN" else en_lag + 1
        assert debt == (1 if node // 2 == 1 else 0)
        assert (al_lag, en_lag) == (4, 0)
        # Uniformly nonzero same-time jumps discard the old temporal segment;
        # the subsequent flow supplies the only new secant on that segment.
        history = ((Q(19, 16), c["event_form"][node]), (Q(5, 4), c["endpoint"][node]))
        assert (history[1][1] - history[0][1]) / DT == c["rates"][node]
        assert history[0][1] != c["source_temporal"][node][-1][1]
        assert len(history) <= domain.base.runtime._MUTATION_TIME_HISTORY_MAXLEN
        assert Q(1) + 4 * DT == history[-1][0]
        before_draws = sum(glyph == "NAV" for glyph in source["next_traces"][node])
        after_draws = sum(glyph == "NAV" for glyph in c["traces"][node])
        assert before_draws == after_draws == (1 if node < 2 else 2 if node < 4 else 0)
    # Accepted EN resets only its own lag; all targets retain the AL lag four.
    assert 4 <= cfg["AL_MAX_LAG"] and 4 > cfg["EN_MAX_LAG"]
    assert source["graph"].graph["history"].get("stable_frac") is None
    snapshots = (
        domain._paired(source["first"]["form"]),
        source["forms"],
        c["forms"],
        c["endpoint"],
    )
    assert len(snapshots) == 4 and snapshots[-2] is source["endpoint"]
