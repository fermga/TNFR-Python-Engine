"""Conditional five-operator calculus with constituent-associated boundaries.

These are detached exact algebra controls, not a new default-selector batch,
native response, stochastic campaign or production compression API. A supplied
whole-pair transcript AL/EN/IL/NAV/RA checks the five event identities against
an independent fine-node sweep. Its origin and nonempty AL/IL/EN grammar
context are premises; no selection or occurrence law is inferred from it.
Previous frozen native examples retain their separate preparation/evidence.

The clean affine domain has doubled C5 support, contiguous twins, paired
primitive phase/capacity, signed unequal forms, hard rails with no active
projection, zero Gamma, default direct-glyph factors and no custom callbacks.
Pressure is prepared once and held through the event prefix and Euler flow.
Resonance requires its actual live U3/identity/amplification branches; Transition
requires a common strict pressure sign and actual target-associated inputs.

Boundary controls retain empty U3, undefined resultant, sign/kind identity,
clipping, mixed Transition signs, nonidentity, temporal reset masks and
adaptation qualification/counters. An equal no-write verdict does not make
memory paired. All runtime and production evaluations remain disabled.
"""

from fractions import Fraction as Q
from itertools import product

import pytest

from tests.physics import test_native_contrast_domain as domain
from tests.physics import test_native_joint_admission as joint
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.constants import DEFAULTS
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER
from tnfr.operators.grammar_types import (
    COUPLING_RESONANCE,
    DESTABILIZERS,
    STABILIZERS,
    TRANSFORMERS,
)

no_execution = domain.no_execution
DT, PI = domain.DT, domain.gain.PI
SCHEDULE = ("AL", "EN", "IL", "NAV", "RA")
MEANS = tuple(map(Q, (Q(1, 8), Q(1, 4), Q(3, 8), Q(1, 4), Q(1, 8))))
CONTRASTS = (Q(1, 64), Q(-1, 128), Q(1, 32), Q(1, 64), Q(-1, 128))
CAPACITIES = (Q(1, 4), Q(1, 2), Q(1, 2), Q(1, 4), Q(1, 4))


def _fine(means, contrasts):
    return tuple(
        value for m, d in zip(means, contrasts, strict=True) for value in (m - d, m + d)
    )


def _moments(values):
    return (
        tuple((values[2 * i] + values[2 * i + 1]) / 2 for i in range(5)),
        tuple((values[2 * i + 1] - values[2 * i]) / 2 for i in range(5)),
    )


def _wrap(angle):
    return (angle + PI) % (2 * PI) - PI


def _compatible(target, neighbor_phases, limit):
    return tuple(
        i
        for i, phase in enumerate(neighbor_phases)
        if abs(_wrap(phase - target)) <= limit
    )


def _phase_proposal(target, admitted, coupling):
    """Exact two-block chart, including the runtime's antipodal hold policy."""
    assert 1 <= len(admitted) <= 2
    offsets = tuple(_wrap(value - target) for value in admitted)
    assert max(map(abs, offsets)) <= PI / 2
    if len(offsets) == 2 and max(offsets) - min(offsets) == PI:
        return target, False
    # On a strict semicircle two equally repeated block phases have the
    # midpoint resultant; a single block supplies its own phase.
    return (target + coupling * sum(offsets) / len(offsets)) % (2 * PI), True


@pytest.fixture(scope="module", autouse=True)
def calculus_guard(no_execution):
    targets, forbidden = no_execution
    additions = tuple((domain.operators, f"_op_{mark}") for mark in SCHEDULE)
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in additions:
            patch.setattr(owner, name, forbidden)
        yield (*targets, *additions), forbidden


@pytest.fixture(scope="module")
def configuration(calculus_guard):
    graph = domain.gain._source()
    cfg = graph.graph
    assert tuple(graph) == tuple(range(10))
    assert cfg["GAMMA"]["type"] == "none" and cfg.get("MATH_ENGINE") is None
    assert cfg["CLIP_MODE"] == "hard" and cfg["INTEGRATOR_METHOD"] == "euler"
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
    assert cfg["GRAMMAR_CANON"]["enabled"] is True
    for key in ("GLYPH_FACTORS", "DNFR_WEIGHTS", "SI_WEIGHTS", "PHASE_ADAPT"):
        assert cfg[key] == DEFAULTS[key]
    for node in graph:
        blocks = ((node // 2 - 1) % 5, (node // 2 + 1) % 5)
        assert set(graph.neighbors(node)) == {
            2 * block + parity for block in blocks for parity in (0, 1)
        }
        assert all(Q(edge["weight"]) == 1 for edge in graph[node].values())
    return graph


@pytest.fixture(scope="module")
def transcript(configuration):
    """Two independent scalar derivations of one supplied conditional word."""
    # Reuse configuration, support and supplied grammar context, not the
    # preceding research source's physical fields. This detached transcript
    # declares its own state and makes that distinction explicit on the copy.
    graph = configuration.copy()
    cfg = graph.graph
    f = {name: Q(value) for name, value in cfg["GLYPH_FACTORS"].items()}
    k, b = (domain.base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    forms = _fine(MEANS, CONTRASTS)
    capacities = domain._paired(CAPACITIES)
    prior_rates = tuple(Q(n - 4, 64) for n in graph)
    for node in graph:
        graph.nodes[node].update(
            EPI=forms[node], nu_f=capacities[node], theta=Q(0), dEPI=prior_rates[node]
        )
    # Equal zero phase and regular degree make these channels exactly zero.
    prepared = tuple(
        k * (sum(forms[j] for j in graph.neighbors(n)) / 4 - forms[n])
        + b * (sum(capacities[j] for j in graph.neighbors(n)) / 4 - capacities[n])
        for n in graph
    )
    centers = tuple(
        k * (domain._neighbor(MEANS, i) - MEANS[i])
        + b * (domain._neighbor(CAPACITIES, i) - CAPACITIES[i])
        for i in range(5)
    )
    assert prepared == _fine(centers, tuple(-k * d for d in CONTRASTS))
    amplitude = f["NAV_jitter"]
    # Supplied admissible innovation values, not purported seed-17 draws.
    innovations = (-amplitude / 3, amplitude / 4)
    fine_x, fine_v, fine_q = list(forms), list(capacities), list(prepared)
    prefixes, fine_inputs = [], []
    for node in graph:
        block, mark = node // 2, SCHEDULE[node // 2]
        local = sum(fine_x[j] for j in graph.neighbors(node)) / 4
        fine_inputs.append(local)
        if mark == "AL":
            fine_x[node] += f["AL_boost"]
        elif mark == "EN":
            fine_x[node] = (1 - f["EN_mix"]) * fine_x[node] + f["EN_mix"] * local
        elif mark == "IL":
            fine_q[node] *= f["IL_dnfr_factor"]
        elif mark == "NAV":
            assert fine_q[node] > 0
            fine_q[node] = (
                (1 - f["NAV_eta"]) * fine_q[node]
                + f["NAV_eta"] * fine_v[node]
                + innovations[node % 2]
            )
        else:
            assert mark == "RA" and local > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
            proposal = (1 - f["RA_epi_diff"]) * fine_x[node] + f["RA_epi_diff"] * local
            assert fine_x[node] * proposal > 0
            fine_x[node] = proposal
            fine_v[node] *= 1 + f["RA_vf_amplification"]
        assert Q(cfg["EPI_MIN"]) < fine_x[node] < Q(cfg["EPI_MAX"])
        prefixes.append((tuple(fine_x), tuple(fine_v), tuple(fine_q)))

    # Collective rows use an ordered live mean sweep and keep the independent
    # pre-batch pressure. The incoming contrast in q_delta is not overwritten
    # by the form event's contraction.
    live_means, live_contrasts = list(MEANS), list(CONTRASTS)
    event_capacity = list(CAPACITIES)
    qmeans, qcontrasts = list(centers), [-k * d for d in CONTRASTS]
    rows, live_inputs = [], []
    for block, mark in enumerate(SCHEDULE):
        local = domain._neighbor(live_means, block)
        live_inputs.append(local)
        form_factor = (
            1 - f["EN_mix"]
            if mark == "EN"
            else 1 - f["RA_epi_diff"] if mark == "RA" else Q(1)
        )
        pressure_factor = (
            f["IL_dnfr_factor"]
            if mark == "IL"
            else 1 - f["NAV_eta"] if mark == "NAV" else Q(1)
        )
        if mark == "AL":
            live_means[block] += f["AL_boost"]
        elif mark in ("EN", "RA"):
            live_means[block] = (
                form_factor * live_means[block] + (1 - form_factor) * local
            )
        live_contrasts[block] *= form_factor
        if mark == "RA":
            event_capacity[block] *= 1 + f["RA_vf_amplification"]
        qmeans[block] = pressure_factor * centers[block]
        qcontrasts[block] = -pressure_factor * k * CONTRASTS[block]
        if mark == "NAV":
            qmeans[block] += f["NAV_eta"] * CAPACITIES[block] + sum(innovations) / 2
            qcontrasts[block] += (innovations[1] - innovations[0]) / 2
        rows.append(
            (
                form_factor,
                pressure_factor,
                tuple(live_means),
                tuple(live_contrasts),
                tuple(event_capacity),
                tuple(qmeans),
                tuple(qcontrasts),
            )
        )
    fine_rate = tuple(v * q for v, q in zip(fine_v, fine_q, strict=True))
    fine_endpoint = tuple(
        x + DT * rate for x, rate in zip(fine_x, fine_rate, strict=True)
    )
    fine_acceleration = tuple(
        (rate - old) / DT for rate, old in zip(fine_rate, prior_rates, strict=True)
    )
    return dict(
        origin="supplied conditional paired transcript; no selector reachability claim",
        graph=graph,
        factors=f,
        k=k,
        b=b,
        forms=forms,
        capacities=capacities,
        prepared=prepared,
        centers=centers,
        innovations=innovations,
        prefixes=tuple(prefixes),
        fine_inputs=tuple(fine_inputs),
        rows=tuple(rows),
        live_inputs=tuple(live_inputs),
        event_form=tuple(fine_x),
        event_capacity=tuple(fine_v),
        held=tuple(fine_q),
        prior_rates=prior_rates,
        rate=fine_rate,
        endpoint=fine_endpoint,
        acceleration=fine_acceleration,
    )


def test_supplied_request_origin_grammar_premises_and_no_runtime(
    transcript, calculus_guard
):
    c, (targets, forbidden) = transcript, calculus_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert c["origin"].startswith("supplied conditional")
    assert set(SCHEDULE) == {"AL", "EN", "IL", "RA", "NAV"}
    assert "coherence" in STABILIZERS
    assert not {"emission", "reception", "resonance", "transition"}.intersection(
        DESTABILIZERS | TRANSFORMERS
    )
    assert "resonance" in COUPLING_RESONANCE
    for _, data in c["graph"].nodes(data=True):
        assert tuple(data["glyph_history"]) == ("EN",)
        assert data[domain.PRIOR_COHERENCE_KEY] is True
        assert data[domain.OPERATOR_STEP_KEY] == 3
        assert 0 <= data[domain.U2_DEBT_KEY] <= U2_DEBT_CAPACITY
        assert data["epi_kind"] == "seed"
        assert data["theta"] == 0
    # AL/IL/EN lifetime context is supplied, although only its EN suffix is
    # retained. Nonempty history, bounded debt and a live compatible edge
    # admit these roles; a real runtime still validates its actual requests.
    assert _compatible(Q(0), (Q(0), Q(0)), PI / 2) == (0, 1)


def test_all_five_live_event_rows_match_the_independent_fine_prefix(transcript):
    c = transcript
    for block, row in enumerate(c["rows"]):
        _, _, means, contrasts, capacity, qmean, qcontrast = row
        fine_x, fine_v, fine_q = c["prefixes"][2 * block + 1]
        assert _moments(fine_x) == (means, contrasts)
        assert fine_v == domain._paired(capacity)
        assert _moments(fine_q) == (qmean, qcontrast)
        assert (
            c["fine_inputs"][2 * block]
            == c["fine_inputs"][2 * block + 1]
            == c["live_inputs"][block]
        )
    boost = c["factors"]["AL_boost"]
    assert c["live_inputs"][1] - domain._neighbor(MEANS, 1) == boost / 2
    assert c["live_inputs"][4] - domain._neighbor(MEANS, 4) == boost / 2
    assert c["event_capacity"][8] > c["capacities"][8]
    assert c["held"][:4] == c["prepared"][:4]
    assert c["held"][8:] == c["prepared"][8:]


def test_one_held_flow_formula_retains_incoming_contrast_and_prior_rate(transcript):
    c = transcript
    k, final_row = c["k"], c["rows"][-1]
    y, v, qmean, qcontrast = final_row[2], final_row[4], final_row[5], final_row[6]
    expected_mean = tuple(
        x + DT * capacity * pressure
        for x, capacity, pressure in zip(y, v, qmean, strict=True)
    )
    expected_contrast = []
    for i, (factor_x, factor_p, *_) in enumerate(c["rows"]):
        innovation = (
            (c["innovations"][1] - c["innovations"][0]) / 2
            if SCHEDULE[i] == "NAV"
            else Q(0)
        )
        expected_contrast.append(
            (factor_x - DT * v[i] * factor_p * k) * CONTRASTS[i]
            + DT * v[i] * innovation
        )
    assert _moments(c["endpoint"]) == (expected_mean, tuple(expected_contrast))
    assert _moments(c["rate"]) == (
        tuple(v[i] * q for i, q in enumerate(qmean)),
        tuple(v[i] * q for i, q in enumerate(qcontrast)),
    )
    rmean, rcontrast = _moments(c["rate"])
    oldmean, oldcontrast = _moments(c["prior_rates"])
    assert _moments(c["acceleration"]) == (
        tuple((a - b) / DT for a, b in zip(rmean, oldmean)),
        tuple((a - b) / DT for a, b in zip(rcontrast, oldcontrast)),
    )
    cfg = c["graph"].graph
    assert all(Q(cfg["EPI_MIN"]) < x < Q(cfg["EPI_MAX"]) for x in c["endpoint"])
    assert all(Q(cfg["VF_MIN"]) < x < Q(cfg["VF_MAX"]) for x in c["event_capacity"])
    assert min(map(abs, c["held"])) > Q(cfg["EPS_DNFR_STABLE"])
    # Reconstructing pressure from current event form would change the model.
    graph = c["graph"]
    refreshed = tuple(
        k
        * (sum(c["event_form"][j] for j in graph.neighbors(n)) / 4 - c["event_form"][n])
        + c["b"]
        * (
            sum(c["event_capacity"][j] for j in graph.neighbors(n)) / 4
            - c["event_capacity"][n]
        )
        for n in graph
    )
    assert refreshed != c["held"] and refreshed[0] != c["held"][0]


def test_resonance_uses_actual_subset_resultant_and_strict_boost(configuration):
    f, limit = configuration.graph["GLYPH_FACTORS"], PI / 2
    r, a, coupling = (
        Q(f[name])
        for name in ("RA_epi_diff", "RA_vf_amplification", "RA_phase_coupling")
    )
    phases = (Q(1, 8), Q(2))
    admitted = _compatible(Q(0), phases, limit)
    assert admitted == (0,)
    # Whole compatible blocks contribute their means even with unequal forms.
    neighbor_means, neighbor_contrasts = (Q(1, 4), Q(3, 4)), (Q(1, 16), Q(1, 32))
    values = _fine(neighbor_means, neighbor_contrasts)
    selected = tuple(
        values[2 * block + parity] for block in admitted for parity in (0, 1)
    )
    signal = sum(selected) / len(selected)
    assert signal == Q(1, 4) != sum(neighbor_means) / 2
    theta, defined = _phase_proposal(Q(0), tuple(phases[i] for i in admitted), coupling)
    assert defined and theta == coupling / 8
    threshold = Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
    capacity = Q(1, 4)
    for signal, active in (
        (Q(0), False),
        (threshold, False),
        (-threshold, False),
        (3 * threshold / 2, True),
        (Q(1, 4), True),
    ):
        assert (abs(signal) > threshold) is active
        pair = (Q(1, 4), Q(3, 8))
        proposal = tuple((1 - r) * x + r * signal for x in pair)
        assert all(before * after > 0 for before, after in zip(pair, proposal))
        proposed_v = capacity * (1 + a if active else 1)
        assert capacity <= proposed_v < Q(configuration.graph["VF_MAX"])
    # A nonempty U3 subset can have an operationally undefined resultant.
    # Its exact represented-antipode guard holds phase, not form or capacity.
    boundary_phases = (Q(0), PI)
    assert _compatible(limit, boundary_phases, limit) == (0, 1)
    held_theta, defined = _phase_proposal(limit, boundary_phases, coupling)
    assert not defined and held_theta == limit
    assert (1 - r) * Q(1, 8) + r * Q(1, 4) != Q(1, 8)
    assert capacity * (1 + a) > capacity
    assert not _compatible(Q(0), (Q(2), Q(3)), limit)
    # Empty U3 is outside admitted concrete Resonance, not an identity event.


def test_resonance_mean_identity_does_not_admit_both_constituents(configuration):
    r = Q(configuration.graph["GLYPH_FACTORS"]["RA_epi_diff"])
    mean, contrast, signal = Q(-31, 256), Q(33, 256), Q(-1, 2)
    before = (mean - contrast, mean + contrast)
    after = tuple((1 - r) * x + r * signal for x in before)
    mean_after = (1 - r) * mean + r * signal
    assert mean * mean_after > 0
    assert before[0] * after[0] > 0 and before[1] * after[1] < 0
    paired_before = (mean, mean)
    paired_after = tuple((1 - r) * x + r * signal for x in paired_before)
    assert all(x * y > 0 for x, y in zip(paired_before, paired_after))
    # A mean-only gate would incorrectly authorize the first lift. Actual
    # ordering can accept its even target before rejecting the odd target.
    assert after[0] != before[0]
    zero_signal = -(1 - r) * Q(1, 8) / r
    assert (1 - r) * Q(1, 8) + r * zero_signal == 0
    assert Q(0) * (r * Q(1, 4)) == 0  # zero departure is sign-neutral too
    # Kind is separate from scalar sign: a larger differently labeled neighbor
    # changes the proposed established kind under the runtime's strict rule.
    current, neighbor = Q(1, 8), Q(1, 2)
    proposed = (1 - r) * current + r * neighbor
    assert current * proposed > 0 and abs(neighbor) > abs(proposed)
    target_kind, dominant_kind = "seed", "other"
    proposed_kind = dominant_kind if abs(neighbor) > abs(proposed) else target_kind
    assert proposed_kind != target_kind


def test_transition_common_sign_nonidentity_and_mixed_sign_boundary(transcript):
    c, f = transcript, transcript["factors"]
    eta, jitter, v = f["NAV_eta"], f["NAV_jitter"], CAPACITIES[3]
    original = c["prepared"][6:8]
    assert min(original) > 0
    assert eta * (v - max(original)) > jitter
    for innovations in product((-jitter, jitter), repeat=2):
        values = tuple(
            (1 - eta) * p + eta * v + xi for p, xi in zip(original, innovations)
        )
        assert all(value != before for value, before in zip(values, original))
        mean, contrast = sum(original) / 2, (original[1] - original[0]) / 2
        assert sum(values) / 2 == (1 - eta) * mean + eta * v + sum(innovations) / 2
        assert (values[1] - values[0]) / 2 == (1 - eta) * contrast + (
            innovations[1] - innovations[0]
        ) / 2
        assert (
            abs(sum(innovations) / 2) + abs((innovations[1] - innovations[0]) / 2)
            == jitter
        )
    for original in ((Q(-1, 8), Q(1, 8)), (Q(0), Q(-1, 8))):
        signs = tuple(1 if p >= 0 else -1 for p in original)
        values = tuple(
            (1 - eta) * p + eta * v * sign for p, sign in zip(original, signs)
        )
        sign_mean, sign_contrast = Q(sum(signs), 2), Q(signs[1] - signs[0], 2)
        assert sum(values) / 2 == (1 - eta) * sum(original) / 2 + eta * v * sign_mean
        assert (values[1] - values[0]) / 2 == (1 - eta) * (
            original[1] - original[0]
        ) / 2 + eta * v * sign_contrast
        assert sign_contrast != 0
    assert (1 - eta) * v + eta * v == v
    # Zero innovation is within the amplitude interval; p=v would be a
    # rejected zero-action proposal, not automatically an admitted NAV event.


def test_active_clipping_invalidates_the_clean_emission_row(configuration):
    boost = Q(configuration.graph["GLYPH_FACTORS"]["AL_boost"])
    upper = Q(configuration.graph["EPI_MAX"])
    mean, contrast = upper - boost, boost / 2
    before = (mean - contrast, mean + contrast)
    unclipped = tuple(x + boost for x in before)
    clipped = tuple(min(upper, x) for x in unclipped)
    assert unclipped[0] < upper < unclipped[1]
    assert sum(clipped) / 2 != mean + boost
    assert (clipped[1] - clipped[0]) / 2 == contrast / 2 != contrast
    assert min(upper, upper + boost) == upper


@pytest.mark.parametrize("mark", ("EN", "RA"))
def test_equal_marks_and_phase_capacity_allow_different_temporal_reset_masks(
    configuration, mark
):
    f = configuration.graph["GLYPH_FACTORS"]
    mix = Q(f["EN_mix"] if mark == "EN" else f["RA_epi_diff"])
    mean, contrast, signal = Q(1, 4), Q(1, 16), Q(5, 16)
    before = (mean - contrast, mean + contrast)
    proposed = tuple((1 - mix) * x + mix * signal for x in before)
    assert proposed[0] - before[0] == mix / 8 > 0
    assert proposed[1] == before[1]
    assert all(x * y > 0 for x, y in zip(before, proposed))
    histories = tuple(((Q(1) - DT, x - DT / 16), (Q(1), x)) for x in before)
    changed = tuple(x != y for x, y in zip(before, proposed))
    after_event = tuple(
        ((Q(1), new),) if reset else history
        for reset, new, history in zip(changed, proposed, histories)
    )
    capacity = Q(1, 4) * (1 + Q(f["RA_vf_amplification"]) if mark == "RA" else 1)
    pressure = (Q(1, 32), Q(1, 64))
    rates = tuple(capacity * p for p in pressure)
    final_histories = tuple(
        (*history, (Q(1) + DT, x + DT * rate))
        for history, x, rate in zip(after_event, proposed, rates)
    )
    assert changed == (True, False)
    assert tuple(map(len, final_histories)) == (2, 3)
    for history, rate in zip(final_histories, rates):
        assert (history[-1][1] - history[-2][1]) / DT == rate


def test_qualification_and_counters_split_despite_a_shared_no_write_verdict(
    configuration,
):
    # Reuse the prior joint-admission form/capacity geometry, now with both
    # target counters zero. This is a supplied memory control, not a response.
    means = joint.feedback.MEANS
    contrasts = joint.AMBIGUOUS_RADII
    capacities = domain._paired(joint.feedback.CAPACITIES)
    forms = _fine(means, contrasts)
    result = joint._fine_step(configuration, forms, capacities, (0,) * 10, (Q(0),) * 10)
    cfg = configuration.graph
    epsilon = Q(cfg["EPS_DNFR_STABLE"])
    held = result["pressure"][:2]
    assert epsilon - abs(held[0]) > Q(1, 25000)
    assert abs(held[1]) - epsilon > Q(1, 25000)
    assert min(result["si"]) > Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
    assert result["qualified"][:2] == (True, False)
    assert result["counts"][:2] == (1, 0)
    assert not any(result["due"])
    assert result["capacity"] == capacities
    # Identical write flags do not authorize replacing two counters by one.


def test_paired_capacity_requires_the_due_contrast_product_not_equal_flags(
    configuration,
):
    mu = Q(configuration.graph["VF_ADAPT_MU"])
    capacity = Q(5, 4)
    for neighbor in (Q(3, 4), capacity):
        for even_due, odd_due in product((False, True), repeat=2):
            flags = (even_due, odd_due)
            proposed = tuple(
                capacity + mu * (neighbor - capacity) if due else capacity
                for due in flags
            )
            average, difference = Q(int(even_due) + int(odd_due), 2), Q(
                int(odd_due) - int(even_due), 2
            )
            assert sum(proposed) / 2 == capacity + mu * average * (neighbor - capacity)
            assert (proposed[1] - proposed[0]) / 2 == mu * difference * (
                neighbor - capacity
            )
            assert (proposed[0] == proposed[1]) is (
                mu * difference * (neighbor - capacity) == 0
            )
    # Even an actual single-twin write need not split capacity when its
    # neighbor-capacity mean equals the current value; flags remain evidence.
    assert mu > 0


def test_equal_qualification_does_not_remove_the_independent_counter_boundary(
    configuration,
):
    cfg = configuration.graph
    tau, mu = cfg["VF_ADAPT_TAU"], Q(cfg["VF_ADAPT_MU"])
    epsilon = Q(cfg["EPS_DNFR_STABLE"])
    high = Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
    # The same established capacity-pulse geometry with zero form contrast
    # supplies two genuinely qualifying fresh observations under the law.
    forms = domain._paired(joint.feedback.MEANS)
    capacities = domain._paired(joint.feedback.CAPACITIES)
    previous = (tau - 2, tau - 1, *(0,) * 8)
    result = joint._fine_step(configuration, forms, capacities, previous, (Q(0),) * 10)
    assert all(abs(p) < epsilon for p in result["pressure"][:2])
    assert min(result["si"][:2]) > high
    assert result["qualified"][:2] == (True, True)
    assert result["counts"][:2] == (tau - 1, tau)
    assert result["due"][:2] == (False, True)
    capacity = joint.feedback.CAPACITIES[0]
    neighbor_mean = domain._neighbor(joint.feedback.CAPACITIES, 0)
    proposed = result["capacity"][:2]
    assert (proposed[1] - proposed[0]) / 2 == mu * (neighbor_mean - capacity) / 2 < 0
