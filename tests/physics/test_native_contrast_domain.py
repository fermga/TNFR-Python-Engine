"""Exact domain certificate after target-associated Transition breaks pairing.

This owner performs no native call, production pressure/selector/operator
evaluation or numerical integration. Only the earlier pure forecasts supply
the unchanged neighbor-gain source and its two-call algebraic history. Every
runtime boundary is disabled before the module certificate is constructed.

The preceding two innovations generate |u|+|d|<=rho, with d=(odd-even)/2.
Forms, incoming rates, stored pressures and temporal endpoints vary together;
this amplitude-box enclosure does not assert fixed-seed reachability. Exact
represented-coefficient arithmetic proves fixed fresh-pressure signs, a unique
unchanged peak and NAV/NAV/IL/IL/IL throughout that domain. The four subsequent
innovations each lie in [-j,j]. All event/flow rows are affine on the product
of the diamond and this box, so its 64 vertices certify the whole domain.

The complete common phase angle is retained symbolically. Two previous global
angles cancel from the exact relative chart used for fresh pressure and the
third coordinator. An alternating cosine bound certifies its stable branch.
No finite response, universal binary64 bound, autonomous occurrence law,
invariant family or physical identification is claimed here.
"""

import math
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product

import pytest

import tnfr.operators as operators
from tests.physics import test_native_collective_step as base
from tests.physics import test_native_resonance_neighbor_gain as gain
from tests.physics import test_native_resonance_u3_boundary as boundary
from tests.physics import test_native_transition_contrast as transition
from tnfr.constants import DEFAULTS
from tnfr.dynamics import dnfr
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
)
from tnfr.operators.grammar_types import (
    COUPLING_RESONANCE,
    DESTABILIZERS,
    STABILIZERS,
    TRANSFORMERS,
)

DT = Q(1, 16)
MARKS = ("NAV", "NAV", "IL", "IL", "IL")
SIGNS = (-1, 1, -1, -1, 1)
DIMENSION = 6  # u, d, xi_0_even, xi_0_odd, xi_1_even, xi_1_odd


@dataclass(frozen=True)
class _Affine:
    """A test-local exact coefficient row, not a production predictor."""

    coefficients: tuple

    @classmethod
    def constant(cls, value):
        return cls((Q(value),) + (Q(0),) * DIMENSION)

    @classmethod
    def variable(cls, index):
        return cls((Q(0),) + tuple(Q(i == index) for i in range(DIMENSION)))

    def __add__(self, other):
        if not isinstance(other, _Affine):
            other = self.constant(other)
        return _Affine(
            tuple(a + b for a, b in zip(self.coefficients, other.coefficients))
        )

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, scalar):
        return _Affine(tuple(Q(scalar) * value for value in self.coefficients))

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1 / Q(scalar))

    def at(self, point):
        assert len(point) == DIMENSION
        return self.coefficients[0] + sum(
            a * Q(b) for a, b in zip(self.coefficients[1:], point, strict=True)
        )


def _neighbor(values, block):
    return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2


def _paired(values):
    return tuple(value for value in values for _ in range(2))


def _moments(rows):
    return (
        tuple((rows[2 * i] + rows[2 * i + 1]) / 2 for i in range(5)),
        tuple((rows[2 * i + 1] - rows[2 * i]) / 2 for i in range(5)),
    )


def _cosine(angle, terms):
    return sum(
        Q((-1) ** degree, math.factorial(2 * degree)) * angle ** (2 * degree)
        for degree in range(terms)
    )


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    """Fail before any dynamic response, including an accidentally reused fixture."""
    targets = (
        (boundary, "_execute"),
        (base.runtime, "step"),
        (base.runtime, "_refresh_delta_nfr"),
        (base.runtime, "default_compute_delta_nfr"),
        (base, "default_compute_delta_nfr"),
        (dnfr, "default_compute_delta_nfr"),
        (base.dynamics, "compute_Si"),
        (operators, "apply_glyph"),
        (operators, "apply_glyph_obj"),
        (operators, "random_jitter"),
        (base.selectors, "apply_glyph"),
        (base.selectors, "_apply_glyphs"),
        (base.selectors, "_apply_glyph_batch"),
        (base.selectors, "_resolve_preselected_glyph"),
        (base.selectors.DefaultGlyphSelector, "prepare"),
        (base.selectors.DefaultGlyphSelector, "select"),
        (base.integrators.DefaultIntegrator, "integrate"),
        (base.coordination, "coordinate_global_local_phase"),
        (base.adaptation, "adapt_vf_after_structural_stability"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("This exact-domain owner must not evaluate a native response")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in targets:
            patch.setattr(owner, name, forbidden)
        yield targets, forbidden


@pytest.fixture(scope="module")
def certificate(no_execution):
    graph = gain._source()
    assert tuple(graph) == tuple(range(10))
    for node in graph:
        adjacent = ((node // 2 - 1) % 5, (node // 2 + 1) % 5)
        assert set(graph.neighbors(node)) == {
            2 * block + parity for block in adjacent for parity in (0, 1)
        }
        assert all(Q(edge["weight"]) == 1 for edge in graph[node].values())
    cfg = graph.graph
    for key in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "GLYPH_FACTORS",
        "SELECTOR_THRESHOLDS",
        "PHASE_ADAPT",
        "GAMMA",
        "INTEGRATOR_METHOD",
        "DT_MIN",
    ):
        assert cfg[key] == DEFAULTS[key]
    assert cfg["GAMMA"]["type"] == "none" and cfg.get("MATH_ENGINE") is None
    assert "glyph_selector" not in cfg and "integrator" not in cfg
    assert cfg["INTEGRATOR_METHOD"] == "euler" and Q(cfg["DT_MIN"]) == DT
    first = gain._forecast(graph)
    second = transition._forecast(graph, first)
    k, b, w = (
        base._weight(graph, "DNFR_WEIGHTS", channel)
        for channel in ("epi", "vf", "phase")
    )
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", channel)
        for channel in ("alpha", "beta", "gamma")
    )
    eta, h = (Q(cfg["GLYPH_FACTORS"][key]) for key in ("NAV_eta", "IL_dnfr_factor"))
    capacity = second["event_capacity"]
    assert max(capacity) == 1 and eta == Q(1, 2)
    amplitude, radius = second["amplitude"], second["form_diamond_radius"]
    means = list(second["means"]["form"])
    means[1] = second["deterministic_form_mean"]
    means = tuple(means)
    u, d, *innovations = tuple(_Affine.variable(i) for i in range(DIMENSION))
    forms = list(_paired(tuple(_Affine.constant(value) for value in means)))
    forms[2], forms[3] = forms[2] + u - d, forms[3] + u + d
    forms = tuple(forms)

    # Both earlier common global angles are absent from this exact chart.
    psi = second["phase_certificate"]["relative_phase"]
    kg2, kl2 = second["phase_certificate"]["gains"]
    z = tuple((1 - kg2 - kl2) * x + kl2 * _neighbor(psi, i) for i, x in enumerate(psi))
    offsets = tuple(_neighbor(z, i) - z[i] for i in range(5))
    baseline_pressure = tuple(
        k * (_neighbor(means, i) - means[i])
        + b * (_neighbor(capacity, i) - capacity[i])
        + w * offsets[i] / gain.PI
        for i in range(5)
    )
    pressure = []
    for node in range(10):
        block = node // 2
        neighbors = tuple(graph.neighbors(node))
        form_mean = sum((forms[n] for n in neighbors), _Affine.constant(0)) / 4
        pressure.append(
            k * (form_mean - forms[node])
            + b * (_neighbor(capacity, block) - capacity[block])
            + w * offsets[block] / gain.PI
        )
    pressure = tuple(pressure)
    peak = -baseline_pressure[3]
    diamond = ((radius, Q(0)), (-radius, Q(0)), (Q(0), radius), (Q(0), -radius))
    source_vertices = tuple((*point, Q(0), Q(0), Q(0), Q(0)) for point in diamond)
    # Admit signs and the unique peak before constructing normalized rows.
    # Sign-adjusted pressures and their peak gaps are affine, so the strict
    # vertex inequalities apply everywhere in the convex source diamond.
    for point in source_vertices:
        values = tuple(row.at(point) for row in pressure)
        assert all(SIGNS[n // 2] * value > 0 for n, value in enumerate(values))
        assert values[6] == values[7] == -peak
        assert all(
            abs(value) < peak for n, value in enumerate(values) if n not in (6, 7)
        )
    ratios = tuple(SIGNS[n // 2] * p / peak for n, p in enumerate(pressure))
    si = tuple(
        _Affine.constant(
            alpha * capacity[n // 2] + beta * (1 - abs(offsets[n // 2]) / gain.PI)
        )
        + gamma * (_Affine.constant(1) - ratios[n])
        for n in range(10)
    )
    held = tuple(
        (
            (1 - eta) * p + eta * SIGNS[n // 2] * capacity[n // 2] + innovations[n]
            if n < 4
            else h * p
        )
        for n, p in enumerate(pressure)
    )
    rates = tuple(capacity[n // 2] * value for n, value in enumerate(held))
    endpoint = tuple(x + DT * rate for x, rate in zip(forms, rates, strict=True))

    old_pressure = list(map(_Affine.constant, second["pressure"]))
    old_pressure[2] = _Affine.constant(second["deterministic_pressure"]) + (u - d) / (
        DT * capacity[1]
    )
    old_pressure[3] = _Affine.constant(second["deterministic_pressure"]) + (u + d) / (
        DT * capacity[1]
    )
    old_pressure = tuple(old_pressure)
    old_rates = tuple(capacity[n // 2] * row for n, row in enumerate(old_pressure))
    old_acceleration = tuple(
        (rate - first["rate"][n // 2]) / DT for n, rate in enumerate(old_rates)
    )
    acceleration = tuple((new - old) / DT for new, old in zip(rates, old_rates))
    vertices = tuple(
        (u0, d0, *noise)
        for u0, d0 in diamond
        for noise in product((-amplitude, amplitude), repeat=4)
    )
    traces = second["phase_certificate"]["traces"]
    next_traces = tuple((*trace, MARKS[n // 2]) for n, trace in enumerate(traces))
    temporal = []
    for n, form in enumerate(forms):
        block = n // 2
        if block == 0:
            samples = ((Q(17, 16), _Affine.constant(second["event_form"][0])),)
        else:
            samples = (
                (Q(1), _Affine.constant(first["event_form"][block])),
                (Q(17, 16), _Affine.constant(first["form"][block])),
            )
        temporal.append((*samples, (Q(9, 8), form)))
    return dict(
        graph=graph,
        first=first,
        second=second,
        weights=(k, b, w),
        eta=eta,
        retention=h,
        capacity=capacity,
        amplitude=amplitude,
        radius=radius,
        means=means,
        variables=(u, d, *innovations),
        forms=forms,
        phase=z,
        offsets=offsets,
        baseline_pressure=baseline_pressure,
        peak=peak,
        pressure=pressure,
        ratios=ratios,
        si=si,
        held=held,
        rates=rates,
        endpoint=endpoint,
        old_rates=old_rates,
        old_acceleration=old_acceleration,
        old_pressure=old_pressure,
        acceleration=acceleration,
        source_vertices=source_vertices,
        vertices=vertices,
        traces=traces,
        next_traces=next_traces,
        temporal=tuple(temporal),
    )


def test_no_response_boundaries_remain_disabled(no_execution, certificate):
    targets, forbidden = no_execution
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert certificate["graph"].graph["compute_delta_nfr"] is forbidden


def test_diamond_retains_associated_source_and_temporal_evidence(certificate):
    c, s = certificate, certificate["second"]
    assert c["radius"] == DT * c["capacity"][1] * c["amplitude"]
    nominal = (
        s["means"]["form"][1] - s["deterministic_form_mean"],
        s["contrasts"]["form"][1],
        Q(0),
        Q(0),
        Q(0),
        Q(0),
    )
    assert abs(nominal[0]) + abs(nominal[1]) < c["radius"]
    for rows, key in (
        (c["forms"], "form"),
        (c["old_rates"], "rate"),
        (c["old_acceleration"], "acceleration"),
        (c["old_pressure"], "pressure"),
    ):
        assert tuple(row.at(nominal) for row in rows) == s[key]
    for point in c["source_vertices"]:
        u, d = point[:2]
        preceding = (
            (u - d) / (DT * c["capacity"][1]),
            (u + d) / (DT * c["capacity"][1]),
        )
        assert max(map(abs, preceding)) == c["amplitude"]
        assert max(abs(u - d), abs(u + d)) == abs(u) + abs(d) == c["radius"]
    for node in range(10):
        samples = c["temporal"][node]
        assert samples[-1] == (Q(9, 8), c["forms"][node])
        assert (samples[-1][1] - samples[-2][1]) / DT == c["old_rates"][node]
        assert all(a[0] < b[0] for a, b in zip(samples, samples[1:]))
    # The independently associated prior rate is essential to acceleration.
    assert c["old_rates"][2].coefficients[1:3] == (1 / DT, -1 / DT)
    assert c["old_rates"][3].coefficients[1:3] == (1 / DT, 1 / DT)


def test_exact_full_support_pressure_rows_and_contrast(certificate):
    c = certificate
    u, d = c["variables"][:2]
    k = c["weights"][0]
    baseline = tuple(map(_Affine.constant, c["baseline_pressure"]))
    expected = (
        baseline[0] + k * u / 2,
        baseline[0] + k * u / 2,
        baseline[1] - k * u + k * d,
        baseline[1] - k * u - k * d,
        baseline[2] + k * u / 2,
        baseline[2] + k * u / 2,
        baseline[3],
        baseline[3],
        baseline[4],
        baseline[4],
    )
    assert c["pressure"] == expected
    means, contrasts = _moments(c["pressure"])
    assert contrasts == (_Affine.constant(0), -k * d, *(_Affine.constant(0),) * 3)
    assert all(row.coefficients[2] == 0 for row in means)
    assert c["pressure"][2] != c["pressure"][3]
    assert Q(268, 1000) < c["peak"] < Q(269, 1000)


def test_diamond_signs_unique_peak_and_prepared_policy(certificate):
    c, cfg = certificate, certificate["graph"].graph
    threshold = {name: Q(value) for name, value in cfg["SELECTOR_THRESHOLDS"].items()}
    observed_marks = set()
    for point in c["source_vertices"]:
        p = tuple(row.at(point) for row in c["pressure"])
        assert all(SIGNS[n // 2] * value > 0 for n, value in enumerate(p))
        assert p[6] == p[7] == -c["peak"]
        assert all(
            abs(value) < c["peak"] for n, value in enumerate(p) if n not in (6, 7)
        )
        ratios = tuple(row.at(point) for row in c["ratios"])
        si = tuple(row.at(point) for row in c["si"])
        assert ratios == tuple(abs(value) / c["peak"] for value in p)
        assert all(0 < value < 1 for value in si)
        assert all(Q(13843, 100000) < value < Q(13888, 100000) for value in ratios[:2])
        assert all(Q(58473, 100000) < value < Q(58562, 100000) for value in ratios[2:4])
        assert min(ratios[:2]) - threshold["dnfr_hi"] > Q(54, 10000)
        assert all(Q(399, 1000) < value < Q(400, 1000) for value in si[:2])
        assert all(Q(377, 1000) < value < Q(379, 1000) for value in si[2:4])
        assert min(si[4:]) > Q(874, 1000) > threshold["si_hi"]
        assert all(threshold["si_lo"] < value < threshold["si_hi"] for value in si[:4])
        assert min(ratios[:4]) > threshold["dnfr_hi"]
        # This is the declared comparison formula, not a production selector.
        observed_marks.add(
            tuple("IL" if value >= threshold["si_hi"] else "NAV" for value in si)
        )
    assert observed_marks == {_paired(MARKS)}
    assert len(c["source_vertices"]) == 4
    # Each sign-adjusted pressure is affine; its positive vertex lower bound
    # proves fixed signs first. Peak gaps, ratios and Si are then affine too.
    assert all(
        len(row.coefficients) == DIMENSION + 1 for row in (*c["pressure"], *c["si"])
    )


def test_complete_mean_contrast_map_is_affine(certificate):
    c = certificate
    u, d, e0, o0, e1, o1 = c["variables"]
    k, eta, h = c["weights"][0], c["eta"], c["retention"]
    v, p = c["capacity"], c["baseline_pressure"]
    qmean = (
        (1 - eta) * (_Affine.constant(p[0]) + k * u / 2) - eta * v[0] + (e0 + o0) / 2,
        (1 - eta) * (_Affine.constant(p[1]) - k * u) + eta * v[1] + (e1 + o1) / 2,
        h * (_Affine.constant(p[2]) + k * u / 2),
        _Affine.constant(h * p[3]),
        _Affine.constant(h * p[4]),
    )
    qcontrast = (
        (o0 - e0) / 2,
        -(1 - eta) * k * d + (o1 - e1) / 2,
        *(_Affine.constant(0),) * 3,
    )
    assert _moments(c["held"]) == (qmean, qcontrast)
    mean_rate = tuple(v[i] * value for i, value in enumerate(qmean))
    contrast_rate = tuple(v[i] * value for i, value in enumerate(qcontrast))
    assert _moments(c["rates"]) == (mean_rate, contrast_rate)
    xmean, xcontrast = _moments(c["forms"])
    endpoint_mean = tuple(x + DT * rate for x, rate in zip(xmean, mean_rate))
    endpoint_contrast = tuple(
        x + DT * rate for x, rate in zip(xcontrast, contrast_rate)
    )
    assert _moments(c["endpoint"]) == (endpoint_mean, endpoint_contrast)
    damping = 1 - DT * v[1] * (1 - eta) * k
    assert 0 < damping < 1
    assert endpoint_contrast[0] == DT * v[0] * (o0 - e0) / 2
    assert endpoint_contrast[1] == damping * d + DT * v[1] * (o1 - e1) / 2
    previous_mean, previous_contrast = _moments(c["old_rates"])
    assert _moments(c["acceleration"]) == (
        tuple((new - old) / DT for new, old in zip(mean_rate, previous_mean)),
        tuple((new - old) / DT for new, old in zip(contrast_rate, previous_contrast)),
    )
    # Finite acceleration retains the preceding contrast as well as the new one.
    assert previous_contrast[1] == d / DT


def test_whole_product_domain_has_no_clipping_noop_or_capacity_write(certificate):
    c, cfg = certificate, certificate["graph"].graph
    form_bounds = (
        (Q(408, 1000), Q(410, 1000)),
        (Q(-659, 1000), Q(-655, 1000)),
        (Q(954, 1000), Q(955, 1000)),
        (Q(-56, 1000), Q(-55, 1000)),
        (Q(-911, 1000), Q(-910, 1000)),
    )
    assert len(c["vertices"]) == len(set(c["vertices"])) == 64
    assert cfg["NAV_RANDOM"] is True and cfg["NAV_STRICT"] is False
    assert cfg["VF_ADAPT_TAU"] > 1
    for v in c["capacity"]:
        assert Q(cfg["VF_MIN"]) < v < Q(cfg["VF_MAX"])
    for point in c["vertices"]:
        for node in range(10):
            block = node // 2
            p, q = c["pressure"][node].at(point), c["held"][node].at(point)
            # Fixed signs are essential: lower bounds on |affine| do not
            # follow from corners unless crossing zero is excluded first.
            assert SIGNS[block] * q > Q(11, 1000) > Q(cfg["EPS_DNFR_STABLE"])
            if node < 4:
                assert SIGNS[block] * (q - p) > Q(11, 1000)
            x = c["endpoint"][node].at(point)
            lower, upper = form_bounds[block]
            assert Q(cfg["EPI_MIN"]) < lower < x < upper < Q(cfg["EPI_MAX"])
            assert x == c["forms"][node].at(point) + DT * c["rates"][node].at(point)
            assert (
                c["acceleration"][node].at(point)
                == (c["rates"][node].at(point) - c["old_rates"][node].at(point)) / DT
            )
            assert abs(c["rates"][node].at(point)) < 1
            assert abs(c["acceleration"][node].at(point)) < 4
    # Every held pressure misses stability, so all incoming zero counters
    # remain zero; there is no eligible capacity write or new stable fraction.


def test_joint_output_enclosures_preserve_mean_contrast_correlations(certificate):
    c = certificate
    u = c["variables"][0]
    k, eta, v = c["weights"][0], c["eta"], c["capacity"]
    damping = 1 - DT * v[1] * (1 - eta) * k
    mean_rows, contrast_rows = _moments(c["endpoint"])
    base0 = c["means"][0] + DT * v[0] * (
        (1 - eta) * c["baseline_pressure"][0] - eta * v[0]
    )
    base1 = c["means"][1] + DT * v[1] * (
        (1 - eta) * c["baseline_pressure"][1] + eta * v[1]
    )
    drift0 = DT * v[0] * (1 - eta) * k * u / 2
    centered0 = mean_rows[0] - base0 - drift0
    centered1 = mean_rows[1] - base1
    radii = (
        DT * v[0] * c["amplitude"],
        damping * c["radius"] + DT * v[1] * c["amplitude"],
    )
    for centered, contrast, radius in zip(
        (centered0, centered1), contrast_rows[:2], radii
    ):
        for sign_mean, sign_contrast in product((-1, 1), repeat=2):
            linear = sign_mean * centered + sign_contrast * contrast
            values = tuple(linear.at(point) for point in c["vertices"])
            assert max(values) == radius
    # The diamond, unlike its marginal square, excludes this simultaneous
    # maximal mean and maximal contrast; no fixed-seed reachability follows.
    marginal_corner = (c["radius"], c["radius"])
    recovered_odd_input = sum(marginal_corner) / (DT * v[1])
    assert recovered_odd_input > c["amplitude"]


def test_phase_chart_cancellation_and_stable_coordinator(certificate):
    c, cfg = certificate, certificate["graph"].graph
    z, policy = c["phase"], cfg["PHASE_ADAPT"]
    assert 0 < min(z) < max(z) < gain.PI
    assert max(z) - min(z) < Q(321, 200) < gain.PI
    common = _Affine.variable(0)
    absolute = tuple(common + value for value in z)
    assert tuple(_neighbor(absolute, i) - absolute[i] for i in range(5)) == tuple(
        _Affine.constant(value) for value in c["offsets"]
    )
    differences = tuple(z[i] - z[j] for i in range(5) for j in range(i + 1, 5))
    # For |angle|<1.605, cosine's alternating terms decrease after degree2;
    # degree14/12 are rigorous lower/upper bounds for every pairwise cosine.
    lower = (5 + 2 * sum(_cosine(d, 8) for d in differences)) / 25
    upper = (5 + 2 * sum(_cosine(d, 7) for d in differences)) / 25
    assert Q(policy["R_hi"]) ** 2 < Q(830, 1000) ** 2 < lower
    assert lower < upper < Q(831, 1000) ** 2
    traces = c["next_traces"]
    load = Q(
        sum(glyph == "NAV" for trace in traces for glyph in trace),
        sum(map(len, traces)),
    )
    assert load == Q(3, 20) < Q(policy["disr_lo"])
    gains = tuple(
        current + Q(policy["down"]) * (Q(policy[f"k{suffix}_min"]) - current)
        for current, suffix in zip(
            c["second"]["phase_certificate"]["gains"], ("G", "L")
        )
    )
    assert Q(53, 1000) < gains[0] < Q(54, 1000)
    assert Q(125, 1000) < gains[1] < Q(126, 1000)
    for current, new, suffix in zip(
        c["second"]["phase_certificate"]["gains"], gains, ("G", "L")
    ):
        assert Q(policy[f"k{suffix}_min"]) < new < current
    assert sum(gains) < 1
    relative = tuple(
        (1 - sum(gains)) * x + gains[1] * _neighbor(z, i) for i, x in enumerate(z)
    )
    assert (
        max(relative) - min(relative)
        <= (1 - gains[0]) * (max(z) - min(z))
        < Q(1519, 1000)
    )
    # Absolute output is C + kg3*Arg(sum exp(i*z)) + relative. The unknown
    # common angle C is retained, not reconstructed from structural form.
    assert all(value.coefficients[1] == 1 for value in absolute)


def test_history_grammar_clock_and_rng_progress_remain_target_associated(certificate):
    c, cfg = certificate, certificate["graph"].graph
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
    assert cfg["GLYPH_HYSTERESIS_WINDOW"] >= 4
    assert (
        "transition"
        not in STABILIZERS | DESTABILIZERS | TRANSFORMERS | COUPLING_RESONANCE
    )
    assert "coherence" in STABILIZERS
    assert 2 + 1 <= cfg["AL_MAX_LAG"] and 2 + 1 <= cfg["EN_MAX_LAG"]
    assert tuple(c["next_traces"][2 * i] for i in range(5)) == (
        ("EN", "RA", "RA", "NAV"),
        ("EN", "RA", "NAV", "NAV"),
        ("EN", "IL", "IL", "IL"),
        ("EN", "IL", "IL", "IL"),
        ("EN", "IL", "IL", "IL"),
    )
    for node in range(10):
        block, source = node // 2, c["graph"].nodes[node]
        debt = source[U2_DEBT_KEY]
        # The original admitted source supplied AL/IL/EN and lifetime state;
        # the retained EN suffix alone must not erase prior coherence/debt.
        for mark in (gain.MARKS[node][1], transition.MARKS[node][1], MARKS[block]):
            debt = max(0, debt - 1) if mark == "IL" else debt
        assert debt == (1 if block == 1 else 0)
        assert source[PRIOR_COHERENCE_KEY] is True
        assert source[OPERATOR_STEP_KEY] + 3 == 6 >= len(c["next_traces"][node])
        samples = c["temporal"][node]
        extended = (*samples, (Q(19, 16), c["endpoint"][node]))
        assert len(extended) == (3 if block == 0 else 4)
        assert len(extended) <= base.runtime._MUTATION_TIME_HISTORY_MAXLEN
        assert extended[:-1] == samples
        assert (extended[-1][1] - extended[-2][1]) / DT == c["rates"][node]
        assert Q(1) + 3 * DT == extended[-1][0]
        previous_draws = sum(mark == "NAV" for mark in c["traces"][node])
        if node < 4:
            assert previous_draws == (0 if node < 2 else 1)
            # Stream offsets are the original fine-target positions, not a
            # newly enumerated selection list or a coarse block identifier.
            progress = dict(
                offset=node, draws=previous_draws + 1, seed=cfg["RANDOM_SEED"]
            )
            assert progress == dict(offset=node, draws=1 if node < 2 else 2, seed=17)
        else:
            assert previous_draws == 0
    # The two previous graph snapshots and the new one retain family forms.
    epi_history = (_paired(c["first"]["form"]), c["forms"], c["endpoint"])
    assert len(epi_history) == 3
    assert epi_history[-2][2].coefficients[1:3] == (1, -1)
