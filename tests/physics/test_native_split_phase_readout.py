"""Static constituent checks for the native split-phase collective chart.

The two declared geometries test fresh pressure/Si and conditional phase
coordination. They are supplied states, not source acquisition or a trajectory.
Mutation execution and sequential admission belong to the operator controls.
"""

import math

import networkx as nx
import pytest

from tnfr import operators
from tnfr.alias import get_attr
from tnfr.constants import DEFAULTS
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics import integrators, runtime
from tnfr.dynamics.dnfr import _resolve_dnfr_weights, default_compute_delta_nfr
from tnfr.metrics.sense_index import compute_Si, get_Si_weights

MU = (1.0, 1.25, 1.5, 0.5, 0.75)
MEAN_FORM = (0.25, 0.125, 0.375, -0.125, 0.5)
FORM_CONTRAST = (0.0625, -0.125, 0.0, 0.0625, -0.03125)
MEAN_CAPACITY = (1.0, 1.25, 0.75, 1.5, 1.0)
CAPACITY_CONTRAST = (0.125, -0.0625, 0.125, 0.0, -0.125)
SPLIT = (0.25, 0.0, 0.0, 0.0, 0.0)
GENERAL = (0.25, 0.125, -0.0625, 0.1875, 0.0)
TOL = 3e-15  # Represented trig/readout check, not an exact-circle enclosure.


@pytest.fixture(scope="module", autouse=True)
def no_native_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("Split readouts must not execute events or trajectories")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (runtime, "step"),
            (runtime, "run"),
            (operators, "apply_glyph"),
            (integrators.DefaultIntegrator, "integrate"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


def _graph(contrasts):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for i in range(5):
        for member in (0, 1):
            for neighbor in (0, 1):
                graph.add_edge(2 * i + member, 2 * ((i + 1) % 5) + neighbor, weight=1)
    for node in graph:
        i, member = divmod(node, 2)
        sign = 2 * member - 1
        graph.nodes[node].update(
            EPI=MEAN_FORM[i] + sign * FORM_CONTRAST[i],
            nu_f=MEAN_CAPACITY[i] + sign * CAPACITY_CONTRAST[i],
            theta=MU[i] + sign * contrasts[i],
            delta_nfr=0.0,
        )
    return graph


def _angle(real, imag):
    # Every supplied fine phase is inside (0, 2), so this is the declared lift.
    return math.atan2(imag, real)


def _block_targets(contrasts):
    return tuple(
        _angle(
            sum(
                math.cos(contrasts[j]) * math.cos(MU[j])
                for j in ((i - 1) % 5, (i + 1) % 5)
            ),
            sum(
                math.cos(contrasts[j]) * math.sin(MU[j])
                for j in ((i - 1) % 5, (i + 1) % 5)
            ),
        )
        for i in range(5)
    )


@pytest.mark.parametrize("contrasts", (SPLIT, GENERAL))
def test_split_coordinates_reconstruct_fresh_pressure_and_si(contrasts):
    graph = _graph(contrasts)
    phases = tuple(graph.nodes[n]["theta"] for n in graph)
    assert max(phases) - min(phases) < 2 < math.pi
    assert all(graph.nodes[n]["nu_f"] > 0 for n in graph)
    local = _block_targets(contrasts)
    weights = _resolve_dnfr_weights(graph)
    k, b, t = weights["epi"], weights["vf"], weights["phase"] / math.pi
    means, differences = [], []
    for i in range(5):
        neighbors = ((i - 1) % 5, (i + 1) % 5)
        means.append(
            t * (local[i] - MU[i])
            + k * (sum(MEAN_FORM[j] for j in neighbors) / 2 - MEAN_FORM[i])
            + b * (sum(MEAN_CAPACITY[j] for j in neighbors) / 2 - MEAN_CAPACITY[i])
        )
        differences.append(
            k * FORM_CONTRAST[i] + b * CAPACITY_CONTRAST[i] + t * contrasts[i]
        )
    expected_pressure = tuple(
        means[i] + sign * differences[i] for i in range(5) for sign in (1, -1)
    )
    peak = max(abs(mean) + abs(diff) for mean, diff in zip(means, differences))
    assert peak == pytest.approx(max(map(abs, expected_pressure)), abs=TOL, rel=0)

    # Fine production readout consumes all four actual neighbors, not the
    # collective expression. Si then consumes that newly prepared pressure.
    default_compute_delta_nfr(graph, cache_size=0, n_jobs=1)
    observed_pressure = tuple(get_attr(graph.nodes[n], ALIAS_DNFR) for n in graph)
    assert observed_pressure == pytest.approx(expected_pressure, abs=TOL, rel=0)
    alpha, beta, gamma = get_Si_weights(graph)
    vmax = max(v + abs(e) for v, e in zip(MEAN_CAPACITY, CAPACITY_CONTRAST))
    expected_si = tuple(
        alpha * (MEAN_CAPACITY[i] - sign * CAPACITY_CONTRAST[i]) / vmax
        + beta * (1 - abs(local[i] - MU[i] + sign * contrasts[i]) / math.pi)
        + gamma * (1 - abs(means[i] + sign * differences[i]) / (peak or 1))
        for i in range(5)
        for sign in (1, -1)
    )
    assert all(0 < value < 1 for value in expected_si)
    observed_si = compute_Si(graph, inplace=False, n_jobs=1)
    assert tuple(observed_si[n] for n in graph) == pytest.approx(
        expected_si, abs=TOL, rel=0
    )


def test_unchanged_block_means_lose_neighbor_response_after_splitting():
    paired, split = _graph((0.0,) * 5), _graph(SPLIT)
    for graph in (paired, split):
        assert (
            tuple(
                (graph.nodes[2 * i]["theta"] + graph.nodes[2 * i + 1]["theta"]) / 2
                for i in range(5)
            )
            == MU
        )
        default_compute_delta_nfr(graph, cache_size=0, n_jobs=1)
    for n in paired:
        assert paired.nodes[n]["EPI"] == split.nodes[n]["EPI"]
        assert paired.nodes[n]["nu_f"] == split.nodes[n]["nu_f"]
    # Arg(cos(s)+exp(i/2)) = 1/4 + atan(tan(s/2)^2*tan(1/4)).
    # Its strict positive difference follows analytically for 0<s<pi/2.
    angle_change = math.atan(math.tan(SPLIT[0] / 2) ** 2 * math.tan(0.25))
    expected = _resolve_dnfr_weights(split)["phase"] * angle_change / math.pi
    assert expected > 1e-4
    for node, sign in ((2, 1), (3, 1), (8, -1), (9, -1)):
        actual = get_attr(split.nodes[node], ALIAS_DNFR) - get_attr(
            paired.nodes[node], ALIAS_DNFR
        )
        assert actual == pytest.approx(sign * expected, abs=TOL, rel=0)


def test_signed_internal_phase_is_not_recoverable_from_resultant_weights():
    first, second = _graph(SPLIT), _graph(tuple(-a for a in SPLIT))
    assert _block_targets(SPLIT) == _block_targets(tuple(-a for a in SPLIT))
    for graph in (first, second):
        default_compute_delta_nfr(graph, cache_size=0, n_jobs=1)
    phase_weight = _resolve_dnfr_weights(first)["phase"]
    expected = 2 * phase_weight * SPLIT[0] / math.pi
    for node, sign in ((0, 1), (1, -1)):
        actual = get_attr(first.nodes[node], ALIAS_DNFR) - get_attr(
            second.nodes[node], ALIAS_DNFR
        )
        assert actual == pytest.approx(sign * expected, abs=TOL, rel=0)


def test_fine_coordinator_retains_nonzero_internal_phase_contrast():
    graph = _graph(GENERAL)
    phases = tuple(graph.nodes[n]["theta"] for n in graph)
    kg, kl = DEFAULTS["PHASE_K_GLOBAL"], DEFAULTS["PHASE_K_LOCAL"]
    own = 1 - kg - kl
    assert 0 < own < 1
    global_target = _angle(sum(map(math.cos, phases)), sum(map(math.sin, phases)))
    fine = []
    for node in graph:
        neighbors = tuple(graph.neighbors(node))
        local = _angle(
            sum(math.cos(phases[n]) for n in neighbors),
            sum(math.sin(phases[n]) for n in neighbors),
        )
        fine.append(own * phases[node] + kg * global_target + kl * local)
    block_targets = _block_targets(GENERAL)
    expected_mean = tuple(
        own * mu + kg * global_target + kl * local
        for mu, local in zip(MU, block_targets)
    )
    for i, contrast in enumerate(GENERAL):
        assert (fine[2 * i] + fine[2 * i + 1]) / 2 == pytest.approx(
            expected_mean[i], abs=TOL, rel=0
        )
        assert (fine[2 * i + 1] - fine[2 * i]) / 2 == pytest.approx(
            own * contrast, abs=TOL, rel=0
        )
        if contrast:
            assert fine[2 * i] != fine[2 * i + 1]
    # This checks the ideal constituent map at supplied admissible gains,
    # not that a complete native call selected these gains or this source.
