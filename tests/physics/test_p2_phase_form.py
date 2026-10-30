"""Independent finite controls for the opt-in P2 phase/form composition.

Targets and local equations are fixed from declared coefficients before any
step. Runtime residuals, ideal lock estimates and observed finite attraction
remain distinct; none supplies an autonomous substrate or an infinite-time
binary64 certificate.
"""

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.config import DEFAULTS
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_SI, ALIAS_VF
from tnfr.constants.operational import NODAL_OPT_COUPLING_CANONICAL
from tnfr.dynamics import adaptation, dnfr, integrators, phase_evolution
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.metrics.common import merge_and_normalize_weights
from tnfr.metrics.sense_index import compute_Si, get_Si_weights
from tnfr.physics.forcing_realization import decompose_non_epi_forcing
from tnfr.physics.p2_phase_form import (
    derive_p2_phase_form_model,
    propose_p2_phase_form_step,
)

CHANNELS = ("phase", "epi", "vf", "topo")
CAPACITIES = (0.95, 1.05)
MIX = {"phase": 1.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
PHASE_OFF = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 1.0}


def _represented(value):
    return F.from_float(float(value))


def _gap(phases):
    return math.remainder(float(phases[1]) - float(phases[0]), math.tau)


def _phases(gap, common=0.25):
    return ((common - gap / 2) % math.tau, (common + gap / 2) % math.tau)


def _epi_at_mean(contrast, capacities=CAPACITIES, mean=0.0):
    first, second = capacities
    return mean + first * contrast / (first + second), mean - second * contrast / (
        first + second
    )


def _p2_graph(model, epi, phase):
    graph = nx.path_graph(2)
    graph.edges[0, 1].update(weight=1.0, length=1.0)
    graph.graph.update(
        DNFR_WEIGHTS={key: float(value) for key, value in model.configured_weights},
        vectorized_dnfr=True,
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=float(epi[node]),
            nu_f=float(model.capacities[node]),
            theta=float(phase[node]),
        )
    return graph


def _fresh_pressure(model, epi, phase):
    graph = _p2_graph(model, epi, phase)
    default_compute_delta_nfr(graph)
    return tuple(
        _represented(get_attr(graph.nodes[node], ALIAS_DNFR, strict=True))
        for node in graph
    )


@pytest.mark.parametrize("raw", [None, {"phase": 2, "epi": 3, "vf": 1, "topo": 4}])
def test_model_preserves_the_recipe_and_actual_shared_normalization(raw):
    model = derive_p2_phase_form_model(CAPACITIES, pressure_weights=raw)
    configured = DEFAULTS["DNFR_WEIGHTS"] if raw is None else raw
    graph = nx.Graph()
    graph.graph["DNFR_WEIGHTS"] = dict(configured)
    normalized = merge_and_normalize_weights(
        graph, "DNFR_WEIGHTS", CHANNELS, default=0.0
    )
    assert model.capacities == tuple(map(_represented, CAPACITIES))
    assert model.configured_weights == tuple(
        (key, _represented(configured.get(key, 0))) for key in CHANNELS
    )
    assert model.normalized_weights == tuple(
        (key, _represented(normalized[key])) for key in CHANNELS
    )
    assert model.coupling_strength == _represented(NODAL_OPT_COUPLING_CANONICAL)
    weights = dict(model.normalized_weights)
    first, second = model.capacities
    ratio = (second - first) / (2 * model.coupling_strength)
    delta = math.asin(float(ratio))
    contrast = float(weights["phase"] / weights["epi"]) * delta / math.pi + float(
        weights["vf"] * (second - first) / weights["epi"]
    )
    assert model.lock_status == "strict_attracting"
    assert model.locking_ratio == ratio
    assert model.locked_phase_estimate == pytest.approx(delta, rel=2e-15)
    assert model.locked_contrast_estimate == pytest.approx(contrast, rel=2e-15)
    assert model.phase_decay_rate_estimate == pytest.approx(
        float(2 * model.coupling_strength) * math.cos(delta), rel=2e-15
    )
    assert model.form_decay_rate == weights["epi"] * (first + second)
    assert model.monotone_step_ceiling == min(
        1 / model.form_decay_rate, 1 / (2 * model.coupling_strength)
    )


@pytest.mark.parametrize(
    "capacities,coupling,status,ratio",
    [
        ((1, 1), 0, "neutral_phase_family", None),
        ((1, 2), 0, "no_strict_lock", None),
        ((1, 1), 0.25, "strict_attracting", F(0)),
        ((1, 1.5), 0.25, "no_strict_lock", F(1)),
        ((1.5, 1), 0.25, "no_strict_lock", F(-1)),
        ((1, 2), 0.25, "no_strict_lock", F(2)),
    ],
)
def test_lock_classification_is_strict_and_distinguishes_the_neutral_family(
    capacities, coupling, status, ratio
):
    model = derive_p2_phase_form_model(
        capacities, pressure_weights=MIX, coupling_strength=coupling
    )
    assert model.lock_status == status
    assert model.locking_ratio == ratio
    if status != "strict_attracting":
        assert model.locked_phase_estimate is None
        assert model.locked_contrast_estimate is None
        assert model.phase_decay_rate_estimate is None
    if not coupling:
        assert model.monotone_step_ceiling == 1 / model.form_decay_rate


def test_zero_locked_contrast_survives_a_subnormal_epi_coefficient():
    minimum = math.ulp(0.0)
    model = derive_p2_phase_form_model(
        (1, 1), pressure_weights={"phase": 1, "epi": minimum}, coupling_strength=0.5
    )
    assert model.lock_status == "strict_attracting"
    assert dict(model.normalized_weights)["epi"] == _represented(minimum)
    assert model.form_decay_rate == 2 * _represented(minimum)
    assert model.locked_phase_estimate == model.locked_contrast_estimate == 0.0
    assert model.phase_decay_rate_estimate == 1.0


@pytest.mark.parametrize(
    "capacities,coupling",
    [
        ((1e307, 1.6e308), 1e308),
        ((math.ulp(0.0), math.ldexp(1.0, 1023)), math.ldexp(1.0, 1022)),
    ],
)
def test_phase_rate_display_avoids_intermediate_overflow_and_near_lock_cancellation(
    capacities, coupling
):
    model = derive_p2_phase_form_model(
        capacities, pressure_weights=MIX, coupling_strength=coupling
    )
    difference = model.capacities[1] - model.capacities[0]
    twice_coupling = 2 * model.coupling_strength
    radicand = (twice_coupling - difference) * (twice_coupling + difference)
    rate = model.phase_decay_rate_estimate
    assert model.lock_status == "strict_attracting"
    assert 0 < model.locking_ratio < 1
    assert rate > 0 and math.isfinite(rate)
    # This field is the least binary64 upper display, not a lower decay
    # certificate. Compare squares exactly, including subnormal separation.
    assert _represented(rate) ** 2 >= radicand
    assert _represented(math.nextafter(rate, 0.0)) ** 2 < radicand
    if capacities[0] == math.ulp(0.0):
        assert float(model.locking_ratio) == 1.0
        assert rate == math.ldexp(1.0, -25)


@pytest.mark.parametrize("epi", [(0.2, -0.1), (0.25, 0.25)])
def test_one_step_uses_the_initial_snapshot_and_retains_exact_euler_accounting(
    epi, monkeypatch
):
    model = derive_p2_phase_form_model(
        CAPACITIES, pressure_weights={"phase": 2, "epi": 3, "vf": 1, "topo": 4}
    )
    phases, dt = (0.2, 0.5), 0.125
    calls = []
    phase_owner = phase_evolution.propose_u3_gated_phase_step
    pressure_owner = dnfr.default_compute_delta_nfr
    integrator_owner = integrators.DefaultIntegrator.integrate

    def phase_spy(graph, *args, **kwargs):
        calls.append(("phase", tuple(graph.nodes[i]["EPI"] for i in graph)))
        return phase_owner(graph, *args, **kwargs)

    def pressure_spy(graph, *args, **kwargs):
        calls.append(("pressure", tuple(graph.nodes[i]["theta"] for i in graph)))
        return pressure_owner(graph, *args, **kwargs)

    def integrator_spy(self, graph, *args, **kwargs):
        calls.append(("integrator", tuple(graph.nodes[i]["theta"] for i in graph)))
        assert graph.graph["DT_MIN"] == 0
        assert graph.graph["GAMMA"] == {"type": "none"}
        assert kwargs["method"] == "euler" and kwargs["dt"] == dt
        return integrator_owner(self, graph, *args, **kwargs)

    monkeypatch.setattr(phase_evolution, "propose_u3_gated_phase_step", phase_spy)
    monkeypatch.setattr(dnfr, "default_compute_delta_nfr", pressure_spy)
    monkeypatch.setattr(integrators.DefaultIntegrator, "integrate", integrator_spy)
    step = propose_p2_phase_form_step(model, epi, phases, dt=dt)
    assert [name for name, _ in calls] == [
        "phase",
        "pressure",
        "integrator",
        "pressure",
    ]
    assert calls[0][1] == epi
    assert calls[1][1] == calls[2][1] == phases
    assert calls[-1][1] == tuple(map(float, step.after_phase))
    weights = dict(model.normalized_weights)
    first, second = map(float, model.capacities)
    w, e, v = (float(weights[key]) for key in ("phase", "epi", "vf"))
    delta, contrast = _gap(phases), epi[0] - epi[1]
    pressure = -e * contrast + w * delta / math.pi + v * (second - first)
    np.testing.assert_allclose(
        tuple(map(float, step.pressure_before)),
        (pressure, -pressure),
        rtol=2e-14,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        tuple(map(float, step.after_epi)),
        (epi[0] + dt * first * pressure, epi[1] - dt * second * pressure),
        rtol=2e-14,
        atol=2e-15,
    )
    expected_gap = delta + dt * (
        second - first - 2 * float(model.coupling_strength) * math.sin(delta)
    )
    assert step.before_phase_gap == pytest.approx(delta, abs=2e-15)
    assert step.after_phase_gap == pytest.approx(expected_gap, abs=2e-15)
    assert step.pressure_before == _fresh_pressure(
        model, step.before_epi, step.before_phase
    )
    assert step.pressure_after == _fresh_pressure(
        model, step.after_epi, step.after_phase
    )
    assert step.pressure_after != step.pressure_before
    assert step.forcing_before.full_kernel_pressure == step.pressure_before
    assert step.forcing_before.normalized_weights == model.normalized_weights
    assert step.forcing_before.snapshot.stored_pressure == (0, 0)
    decompose_non_epi_forcing(step.forcing_before)
    assert step.euler.dt == step.dt == _represented(dt)
    assert step.euler.before.epi == step.before_epi
    assert step.euler.after.epi == step.after_epi
    assert step.euler.before.stored_pressure == step.pressure_before
    assert step.euler.before.capacity == step.euler.after.capacity == model.capacities
    expected = tuple(
        x + step.dt * nu * p
        for x, nu, p in zip(
            step.before_epi, model.capacities, step.pressure_before, strict=True
        )
    )
    assert step.euler.expected_epi == expected
    assert step.euler.state_defect == tuple(
        x - y for x, y in zip(step.after_epi, expected, strict=True)
    )
    assert step.euler.identity_residual == 0
    assert step.before_contrast == step.before_epi[0] - step.before_epi[1]
    assert step.after_contrast == step.after_epi[0] - step.after_epi[1]
    nu0, nu1 = model.capacities
    assert step.weighted_mean_drift == (
        nu1 * (step.after_epi[0] - step.before_epi[0])
        + nu0 * (step.after_epi[1] - step.before_epi[1])
    ) / (nu0 + nu1)
    inverse_capacity_sum = 1 / nu0 + 1 / nu1
    pressure_mean_drift = step.dt * sum(step.pressure_before) / inverse_capacity_sum
    rounding_mean_drift = (
        sum(
            defect / nu
            for defect, nu in zip(
                step.euler.state_defect, model.capacities, strict=True
            )
        )
        / inverse_capacity_sum
    )
    assert step.weighted_mean_drift == pressure_mean_drift + rounding_mean_drift


def test_phase_is_independent_of_form_but_form_responds_to_phase_when_enabled():
    model = derive_p2_phase_form_model(CAPACITIES, pressure_weights=MIX)
    first = propose_p2_phase_form_step(model, (0, 0), (0.2, 0.3), dt=0.125)
    other_form = propose_p2_phase_form_step(model, (0.25, -0.125), (0.2, 0.3), dt=0.125)
    other_phase = propose_p2_phase_form_step(model, (0, 0), (0.2, 0.5), dt=0.125)
    assert first.after_phase == other_form.after_phase
    assert first.after_epi != other_phase.after_epi
    assert first.pressure_before != other_phase.pressure_before
    off = derive_p2_phase_form_model(CAPACITIES, pressure_weights=PHASE_OFF)
    active_weights, off_weights = dict(model.normalized_weights), dict(
        off.normalized_weights
    )
    assert active_weights["epi"] == off_weights["epi"] == F(1, 2)
    assert active_weights["vf"] == off_weights["vf"] == 0
    assert off_weights["phase"] == 0 and off_weights["topo"] == F(1, 2)
    inactive = [
        propose_p2_phase_form_step(off, (0, 0), phases, dt=0.125)
        for phases in ((0.2, 0.3), (0.2, 0.5))
    ]
    assert all(step.after_epi == step.pressure_before == (0, 0) for step in inactive)
    assert all(
        step.forcing_before.snapshot.topology_gradient == (0, 0) for step in inactive
    )
    assert inactive[0].after_phase != inactive[1].after_phase

    # The same coefficient-preserving control also works with an active
    # capacity channel; removing phase must not silently rescale that source.
    full = derive_p2_phase_form_model(
        CAPACITIES, pressure_weights={"phase": 2, "epi": 3, "vf": 1, "topo": 4}
    )
    no_phase = derive_p2_phase_form_model(
        CAPACITIES, pressure_weights={"phase": 0, "epi": 3, "vf": 1, "topo": 6}
    )
    for key in ("epi", "vf"):
        assert (
            dict(full.normalized_weights)[key] == dict(no_phase.normalized_weights)[key]
        )
    capacity_driven = [
        propose_p2_phase_form_step(no_phase, (0, 0), phases, dt=0.125)
        for phases in ((0.2, 0.3), (0.2, 0.5))
    ]
    assert capacity_driven[0].after_epi == capacity_driven[1].after_epi != (0, 0)
    assert capacity_driven[0].pressure_before == capacity_driven[1].pressure_before


def test_prospective_128_step_lock_and_perturbation_controls_keep_the_phase_off_baseline():
    # Fixed before execution: capacities, coefficients, horizon, perturbation
    # and analytic target. The finite trace never fits or adjusts this target.
    dt, count, coupling = 0.25, 128, 0.1
    nu0, nu1, k = map(_represented, (*CAPACITIES, coupling))
    target_gap = math.asin(float((nu1 - nu0) / (2 * k)))
    target_contrast = target_gap / math.pi
    phase_perturbation, form_perturbation = 0.04, 0.02
    contraction = 1 - 2 * coupling * dt * math.cos(target_gap + phase_perturbation)
    form_factor = 1 - float((nu0 + nu1) / 2) * dt
    phase_bound = phase_perturbation * contraction**count
    form_bound = form_factor**count * form_perturbation + dt * float(nu0 + nu1) / (
        2 * math.pi
    ) * phase_perturbation * (contraction**count - form_factor**count) / (
        contraction - form_factor
    )
    model = derive_p2_phase_form_model(
        CAPACITIES, pressure_weights=MIX, coupling_strength=coupling
    )
    off = derive_p2_phase_form_model(
        CAPACITIES, pressure_weights=PHASE_OFF, coupling_strength=coupling
    )
    starts = (
        (model, _epi_at_mean(target_contrast), _phases(target_gap)),
        (
            model,
            _epi_at_mean(target_contrast + form_perturbation),
            _phases(target_gap + phase_perturbation),
        ),
        (off, (0.0, 0.0), (0.25, 0.25)),
    )
    endings = []
    for chosen, epi, phase in starts:
        for _ in range(count):
            step = propose_p2_phase_form_step(chosen, epi, phase, dt=dt)
            assert step.euler.identity_residual == 0
            if chosen is off:
                assert (
                    step.after_epi
                    == step.pressure_before
                    == step.pressure_after
                    == (0, 0)
                )
            epi, phase = step.after_epi, step.after_phase
        endings.append(step)
    baseline, perturbed, inactive = endings
    assert abs(baseline.after_phase_gap - target_gap) < 1e-10
    assert abs(float(baseline.after_contrast) - target_contrast) < 1e-10
    # A factor-two comparison margin leaves numerical-rounding headroom;
    # it is a finite regression criterion, not a binary64 error theorem.
    assert (
        abs(perturbed.after_phase_gap - target_gap)
        < 2 * phase_bound
        < phase_perturbation / 20
    )
    assert (
        abs(float(perturbed.after_contrast) - target_contrast)
        < 2 * form_bound
        < form_perturbation / 20
    )
    assert inactive.after_epi == (0, 0)
    assert abs(inactive.after_phase_gap - target_gap) < 0.01


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacities", (1,)),
        ("capacities", (0, 1)),
        ("capacities", (True, 1)),
        ("capacities", (float("inf"), 1)),
        ("capacities", {1, 2}),
        ("coupling_strength", -0.1),
        ("coupling_strength", True),
        ("coupling_strength", float("nan")),
        ("pressure_weights", {"phase": 1, "epi": 0}),
        ("pressure_weights", {"phase": 0, "epi": 0}),
        ("pressure_weights", {"epi": -1}),
        ("pressure_weights", {"epi": True}),
        ("pressure_weights", {"epi": 1, "unknown": 0}),
        ("pressure_weights", "epi"),
    ],
)
def test_invalid_model_inputs_are_rejected(field, value):
    arguments = {"capacities": CAPACITIES, "pressure_weights": MIX, field: value}
    with pytest.raises((TypeError, ValueError)):
        derive_p2_phase_form_model(**arguments)


@pytest.mark.parametrize(
    "epi,phase,dt",
    [
        ((0,), (0.2, 0.3), 0.125),
        ((True, 0), (0.2, 0.3), 0.125),
        ((math.nextafter(1.0, math.inf), 0), (0.2, 0.3), 0.125),
        ((0, 0), (0.2,), 0.125),
        ((0, 0), (-0.1, 0.2), 0.125),
        ((0, 0), (0, math.tau), 0.125),
        ((0, 0), (0, math.pi / 2), 0.125),
        ((0, 0), (0, float("nan")), 0.125),
        ((0, 0), (0.2, 0.3), 0),
        ((0, 0), (0.2, 0.3), True),
        ((0, 0), (0.2, 0.3), math.nextafter(1.0, math.inf)),
        ((1, 1), (0.2, 1.0), 0.125),
    ],
)
def test_invalid_steps_and_clipping_are_rejected(epi, phase, dt):
    model = derive_p2_phase_form_model(CAPACITIES, pressure_weights=MIX)
    with pytest.raises((TypeError, ValueError)):
        propose_p2_phase_form_step(model, epi, phase, dt=dt)


def test_post_step_phase_gate_and_exact_monotone_boundary_are_admitted_separately():
    model = derive_p2_phase_form_model(
        (1, 2), pressure_weights={"epi": 1}, coupling_strength=0
    )
    assert model.monotone_step_ceiling == F(1, 3)
    with pytest.raises(ValueError, match="proposed phase gap"):
        propose_p2_phase_form_step(model, (0, 0), (0, 1.4), dt=0.25)
    equal = derive_p2_phase_form_model(
        (1, 1), pressure_weights={"epi": 1}, coupling_strength=0
    )
    assert equal.monotone_step_ceiling == F(1, 2)
    accepted = propose_p2_phase_form_step(equal, (0.25, -0.25), (0.2, 0.2), dt=0.5)
    assert accepted.after_epi == (0, 0)


def test_model_recipe_cannot_be_bypassed_and_all_returned_state_is_detached():
    capacities, weights, epi, phases = (
        list(CAPACITIES),
        dict(MIX),
        [0.25, -0.125],
        [0.2, 0.3],
    )
    model = derive_p2_phase_form_model(iter(capacities), pressure_weights=weights)
    forged = replace(model, normalized_weights=tuple((key, F(0)) for key in CHANNELS))
    with pytest.raises(ValueError, match="recipe"):
        propose_p2_phase_form_step(forged, epi, phases, dt=0.125)

    class ComparableDisplay(float):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    misleading = replace(model, monotone_step_ceiling=ComparableDisplay(2.0))
    assert misleading == model
    with pytest.raises(ValueError, match="monotone step ceiling"):
        propose_p2_phase_form_step(misleading, epi, phases, dt=1.125)
    step = propose_p2_phase_form_step(model, iter(epi), iter(phases), dt=0.125)
    capacities[0] = weights["epi"] = epi[0] = phases[0] = 99
    assert model.capacities == tuple(map(_represented, CAPACITIES))
    assert step.before_epi == (F(1, 4), F(-1, 8))
    assert step.before_phase == (_represented(0.2), _represented(0.3))
    with pytest.raises(FrozenInstanceError):
        model.form_decay_rate = 0
    with pytest.raises(FrozenInstanceError):
        step.after_epi = (0, 0)


def _capacity_observation(graph):
    """Refresh the actual gate inputs, without inventing pressure or Si."""
    default_compute_delta_nfr(graph)
    compute_Si(graph, inplace=True)
    return {
        "capacity": tuple(
            _represented(get_attr(graph.nodes[i], ALIAS_VF)) for i in graph
        ),
        "pressure": tuple(
            _represented(get_attr(graph.nodes[i], ALIAS_DNFR)) for i in graph
        ),
        "sense": tuple(float(get_attr(graph.nodes[i], ALIAS_SI)) for i in graph),
        "counts": tuple(graph.nodes[i].get("stable_count", 0) for i in graph),
        "epi": tuple(_represented(graph.nodes[i]["EPI"]) for i in graph),
        "phase": tuple(_represented(graph.nodes[i]["theta"]) for i in graph),
    }


@pytest.fixture(scope="module")
def capacity_lock_audit():
    """One finite paired audit, with no physical steps during gate evaluation.

    The original target, five-call window and two later Euler steps are fixed
    before the writer executes. This does not invoke the full runtime scheduler.
    """
    model = derive_p2_phase_form_model(CAPACITIES, pressure_weights=MIX)
    delta, contrast = model.locked_phase_estimate, model.locked_contrast_estimate
    epi, phase = _epi_at_mean(contrast), (0.25, 0.25 + delta)
    branches = {}
    for name, enabled, threshold in (
        ("both", True, None),
        ("inactive", False, None),
        ("high_threshold", True, 0.875),
    ):
        graph = _p2_graph(model, epi, phase)
        inject_defaults(graph)
        if threshold is not None:
            graph.graph["SELECTOR_THRESHOLDS"] = {"si_hi": threshold}
        trace, calls = [], []
        setter = adaptation.set_vf

        def record_write(live_graph, node, value):
            calls.append((node, _represented(value)))
            return setter(live_graph, node, value)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(adaptation, "set_vf", record_write)
            for ordinal in range(1, 6):
                before = _capacity_observation(graph)
                offset = len(calls)
                if enabled:
                    adaptation.adapt_vf_after_structural_stability(graph, n_jobs=1)
                after = _capacity_observation(graph)
                trace.append((ordinal, before, after, tuple(calls[offset:])))
        final = trace[-1][2]
        post_model = derive_p2_phase_form_model(
            final["capacity"],
            pressure_weights=dict(model.configured_weights),
            coupling_strength=model.coupling_strength,
        )
        # Freeze the post-event prediction before computing the response;
        # never replace the original reference in the comparisons below.
        steps = []
        x, theta = final["epi"], final["phase"]
        for _ in range(2):
            step = propose_p2_phase_form_step(post_model, x, theta, dt=0.25)
            steps.append(step)
            x, theta = step.after_epi, step.after_phase
        branches[name] = {
            "trace": tuple(trace),
            "model": post_model,
            "steps": tuple(steps),
            "policy": tuple(
                (key, graph.graph[key])
                for key in ("VF_ADAPT_TAU", "VF_ADAPT_MU", "EPS_DNFR_STABLE")
            ),
            "threshold": graph.graph["SELECTOR_THRESHOLDS"]["si_hi"],
            "si_weights": get_Si_weights(graph),
        }
    return model, branches


@pytest.mark.parametrize(
    "branch,counts,written_nodes,expected_capacity",
    [
        ("both", (5, 5), (0, 1), (0.96, 1.04)),
        ("inactive", (0, 0), (), CAPACITIES),
        ("high_threshold", (5, 5), (0, 1), (0.96, 1.04)),
    ],
)
def test_capacity_policy_uses_fresh_inputs_and_actual_five_call_eligibility(
    capacity_lock_audit, branch, counts, written_nodes, expected_capacity
):
    original, branches = capacity_lock_audit
    run = branches[branch]
    assert dict(run["policy"]) == {
        "VF_ADAPT_TAU": 5,
        "VF_ADAPT_MU": 0.1,
        "EPS_DNFR_STABLE": 0.001,
    }
    for ordinal, before, after, writes in run["trace"]:
        assert all(abs(float(p)) <= 0.001 for p in before["pressure"])
        # Pressure is normalized even when its absolute residual is tiny.
        # Compute the diagnostic independently from the captured live state.
        alpha, beta, gamma = run["si_weights"]
        maximum = max(before["capacity"])
        pmax = max(map(abs, before["pressure"]))
        phase_gap = abs(_gap(before["phase"]))
        for i, si in enumerate(before["sense"]):
            pnorm = float(abs(before["pressure"][i]) / pmax) if pmax else 0.0
            expected_si = (
                alpha * float(before["capacity"][i] / maximum)
                + beta * (1 - phase_gap / math.pi)
                + gamma * (1 - pnorm)
            )
            assert si == pytest.approx(expected_si, abs=2e-15)
        if branch == "high_threshold":
            # Keep the originally declared threshold. The current phase
            # kernel cancels the prepared form gradient exactly in binary64,
            # so both nodes now pass. The earlier one-node observation relied
            # on a tiny pressure residual normalized to unit magnitude in Si.
            assert run["threshold"] == 0.875
            assert before["pressure"] == (0, 0)
        assert min(before["sense"]) > run["threshold"]
        expected_counts = (0, 0) if branch == "inactive" else (ordinal, ordinal)
        assert after["counts"] == expected_counts
        assert after["epi"] == before["epi"]
        assert after["phase"] == before["phase"]
        assert after["pressure"] == before["pressure"]
        assert tuple(node for node, _ in writes) == (
            written_nodes if ordinal == 5 else ()
        )
        if ordinal < 5:
            assert before["capacity"] == after["capacity"] == original.capacities
    assert run["trace"][-1][2]["counts"] == counts
    assert run["model"].capacities == tuple(map(_represented, expected_capacity))


@pytest.mark.parametrize(
    "branch,eligibility", [("both", (1, 1)), ("high_threshold", (1, 1))]
)
def test_capacity_jump_moves_the_conditional_target_without_replacing_the_old_one(
    capacity_lock_audit, branch, eligibility
):
    original, branches = capacity_lock_audit
    changed = branches[branch]["model"]
    nu0, nu1 = original.capacities
    d, sigma, mu = nu1 - nu0, nu0 + nu1, _represented(0.1)
    a0, a1 = eligibility
    ideal = (nu0 + mu * a0 * d, nu1 - mu * a1 * d)
    # Real blend identities and represented writer residuals remain separate.
    residual = tuple(x - y for x, y in zip(changed.capacities, ideal, strict=True))
    assert all(abs(float(error)) < 2e-16 for error in residual)
    new_d = changed.capacities[1] - changed.capacities[0]
    assert new_d == (1 - mu * (a0 + a1)) * d + residual[1] - residual[0]
    assert sum(changed.capacities) == sigma + mu * (a0 - a1) * d + sum(residual)
    delta = math.asin(float(new_d / (2 * original.coupling_strength)))
    assert changed.lock_status == "strict_attracting"
    assert changed.locked_phase_estimate == pytest.approx(delta, abs=2e-15)
    assert changed.locked_contrast_estimate == pytest.approx(delta / math.pi, abs=2e-15)
    assert changed.locked_contrast_estimate < original.locked_contrast_estimate - 0.01
    assert original.capacities == tuple(map(_represented, CAPACITIES))
    assert branches["inactive"]["model"] == original

    # A changed capacity metric changes the conserved-mean formula itself.
    x0, x1 = branches[branch]["trace"][-1][2]["epi"]
    old_mean = (nu1 * x0 + nu0 * x1) / sigma
    new0, new1 = changed.capacities
    new_mean = (new1 * x0 + new0 * x1) / (new0 + new1)
    expected_shift = (x0 - x1) * (new1 * nu0 - nu1 * new0) / (sigma * (new0 + new1))
    assert new_mean - old_mean == expected_shift != 0


def test_capacity_change_reaches_form_via_the_next_phase_and_pressure_snapshots(
    capacity_lock_audit,
):
    original, branches = capacity_lock_audit
    active = branches["both"]["steps"]
    control = branches["inactive"]["steps"]
    for first, second in (active, control):
        assert second.before_epi == first.after_epi
        assert second.before_phase == first.after_phase
        assert second.pressure_before == first.pressure_after
        assert first.euler.identity_residual == second.euler.identity_residual == 0
    # With v=0, the event cannot change pressure or form immediately. Its
    # detuning change first reaches phase, then the next refreshed EPI row.
    assert active[0].pressure_before == control[0].pressure_before
    assert (
        abs(float(active[0].after_contrast) - original.locked_contrast_estimate) < 1e-14
    )
    assert active[0].after_phase_gap - control[0].after_phase_gap == pytest.approx(
        -0.005, abs=2e-15
    )
    assert (
        abs(float(control[1].after_contrast) - original.locked_contrast_estimate)
        < 1e-14
    )
    # Independent two-step prediction, from h*sigma*w/pi times the first
    # phase displacement. Its sign/size was fixed before this response.
    expected_contrast_shift = 0.25 * 2 * 0.5 * (-0.005) / math.pi
    assert float(active[1].after_contrast - control[1].after_contrast) == pytest.approx(
        expected_contrast_shift, abs=2e-15
    )
