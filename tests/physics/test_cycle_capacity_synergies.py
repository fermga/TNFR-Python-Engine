"""Capacity energy and finite writer boundaries for the cycle forcing budget.

Exact masked averages below are conditional algebra on fixed C5 support.
The static production control refreshes its actual gate inputs. One shared
post-UM preparation supplies independent frozen studies: a 32-step detuning
control and five intervals through the first eligible feedback on a supplied
locked family. Both compose existing owners at declared event times; neither
is the complete native runtime or an infinite maintenance certificate.
"""

import math
from fractions import Fraction as F

import mpmath as mp
import networkx as nx
import pytest

from tests.joint_phase_helpers import (
    configure,
    execute_coupling_cycle_birth,
    execute_joint_step,
)
from tnfr._binary64 import uses_ieee_binary64_rounding
from tnfr.alias import get_attr, set_vf
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.adaptation import (
    _vf_adapt_chunk,
    adapt_vf_after_structural_stability,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.mathematics._phase_midpoint import _pi_bounds
from tnfr.metrics.sense_index import compute_Si
from tnfr.operators.network_stage import _detached_stage_graph
from tnfr.physics._cycle_algebra import dirichlet_energy, dot, laplacian_action
from tnfr.physics.capacity_localization import observe_cycle_capacity_balance
from tnfr.physics.cycle_relaxation import (
    bound_cycle_capacity_forcing,
    bound_cycle_relaxation,
)
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    observe_forcing_capacity_difference,
    observe_forcing_dirichlet_balance,
    observe_forcing_mean_balance,
)
from tnfr.physics.phase_response import observe_phase_lock_source
from tnfr.physics.support_transport import observe_support_transport_euler
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import angle_diff


@pytest.mark.parametrize("eligible", [(), (0,), (1, 3), tuple(range(5))])
@pytest.mark.parametrize("mu", [F(0), F(1, 10), F(1, 2), F(1)])
def test_masked_exact_average_dissipates_capacity_dirichlet_energy(eligible, mu):
    capacity = (F(2), F(1), F(3, 2), F(5, 4), F(1))
    # This owner returns L_rw=L_cycle/2, not the transport Laplacian of a
    # separately weighted EPI graph. z is the masked unnormalized action.
    laplacian = laplacian_action(capacity)
    z = tuple(2 * value if i in eligible else F(0) for i, value in enumerate(laplacian))
    following = tuple(
        (
            (1 - mu) * value + mu * (capacity[i - 1] + capacity[(i + 1) % 5]) / 2
            if i in eligible
            else value
        )
        for i, value in enumerate(capacity)
    )
    difference = dirichlet_energy(following) - dirichlet_energy(capacity)
    z_l_z = 2 * dot(z, laplacian_action(z))
    assert difference == -mu * dot(z, z) / 2 + mu**2 * z_l_z / 8
    assert z_l_z <= 4 * dot(z, z)
    assert difference <= -mu * (1 - mu) * dot(z, z) / 2 <= 0
    assert min(capacity) <= min(following) <= max(following) <= max(capacity)
    if not eligible or mu == 0:
        assert following == capacity
    elif len(eligible) == len(capacity):
        assert sum(following) == sum(capacity)

    if mu in (0, F(1, 2), 1):
        # These dyadic inputs and blends are exactly representable. Exercise
        # the existing production proposal independently of eligibility policy;
        # this does not assume exact equality for the default 0.1 blend.
        work = [(i, i, ((i - 1) % 5, (i + 1) % 5)) for i in eligible]
        proposals = dict(
            _vf_adapt_chunk((work, tuple(map(float, capacity)), float(mu)))
        )
        assert (
            tuple(F(proposals.get(i, value)) for i, value in enumerate(capacity))
            == following
        )


@pytest.mark.parametrize(
    "capacity",
    [(F(1),) * 5, (F(2), F(1), F(1), F(1), F(1)), (F(1), F(2), F(3), F(2), F(1))],
)
@pytest.mark.parametrize("weight", [F(0), F(1, 7), F(1)])
def test_same_capacity_energy_bounds_gap_forcing_and_capacity_pressure(
    capacity, weight
):
    gap_forcing = tuple(
        capacity[(i + 1) % 5] - value for i, value in enumerate(capacity)
    )
    energy = dirichlet_energy(capacity)
    assert dot(gap_forcing, gap_forcing) == 2 * energy

    # The public held-cycle observer independently materializes the capacity
    # pressure. Zero EPI removes its form contribution; other channels are
    # outside this detached exact observer's declared model.
    reading = observe_cycle_capacity_balance(
        (F(0),) * 5, capacity, epi_weight=F(1, 2), vf_weight=weight
    )
    assert dot(reading.pressure, reading.pressure) <= 2 * weight**2 * energy
    assert sum(reading.pressure) == 0


def test_partial_eligibility_can_move_the_capacity_mean_while_energy_falls():
    capacity = (F(2), F(1), F(1), F(1), F(1))
    # Only node 1 updates by the configured ideal 1/10 neighbor average.
    following = (F(2), F(21, 20), F(1), F(1), F(1))
    assert (
        following[1]
        == F(9, 10) * capacity[1] + F(1, 10) * (capacity[0] + capacity[2]) / 2
    )
    assert sum(following) / 5 - sum(capacity) / 5 == F(1, 100)
    assert dirichlet_energy(following) - dirichlet_energy(capacity) == -F(19, 400)


@pytest.mark.skipif(
    not uses_ieee_binary64_rounding(),
    reason="The stalling witness requires IEEE binary64.",
)
def test_fresh_default_gate_can_admit_all_nodes_without_removing_one_ulp_detuning():
    graph = nx.cycle_graph(5)
    inject_defaults(graph)
    graph.graph["RANDOM_SEED"] = 17
    for node in graph:
        graph.nodes[node].update(
            EPI=0.5,
            nu_f=math.nextafter(1.0, math.inf) if node == 0 else 1.0,
            theta=0.125 + math.tau * node / 5,
        )
    before = tuple(get_attr(graph.nodes[node], ALIAS_VF, strict=True) for node in graph)
    fields_before = tuple(
        (
            get_attr(graph.nodes[node], ALIAS_EPI, strict=True),
            get_attr(graph.nodes[node], ALIAS_THETA, strict=True),
        )
        for node in graph
    )
    tau = graph.graph["VF_ADAPT_TAU"]
    assert tau == 5
    assert graph.graph["VF_ADAPT_MU"] == 0.1
    assert all(graph.nodes[node].get("stable_count", 0) == 0 for node in graph)
    threshold = graph.graph["SELECTOR_THRESHOLDS"]["si_hi"]

    for call in range(1, tau + 1):
        default_compute_delta_nfr(graph)
        compute_Si(graph, inplace=True)
        assert all(
            get_attr(graph.nodes[node], ALIAS_SI, strict=True) >= threshold
            for node in graph
        )
        assert all(
            abs(get_attr(graph.nodes[node], ALIAS_DNFR, strict=True))
            <= graph.graph["EPS_DNFR_STABLE"]
            for node in graph
        )
        adapt_vf_after_structural_stability(graph)
        assert tuple(graph.nodes[node]["stable_count"] for node in graph) == (call,) * 5

    after = tuple(get_attr(graph.nodes[node], ALIAS_VF, strict=True) for node in graph)
    assert after == before
    assert max(after) - min(after) == math.ulp(1.0)
    assert dirichlet_energy(tuple(map(F, after))) == F(math.ulp(1.0)) ** 2 > 0
    assert (
        tuple(
            (
                get_attr(graph.nodes[node], ALIAS_EPI, strict=True),
                get_attr(graph.nodes[node], ALIAS_THETA, strict=True),
            )
            for node in graph
        )
        == fields_before
    )
    ideal, residual = _capacity_proposal_defect(
        tuple(map(F, before)), tuple(map(F, after)), tuple(range(5)), F(0.1)
    )
    assert any(residual)
    ideal_drop = dirichlet_energy(ideal) - dirichlet_energy(tuple(map(F, before)))
    rounding_work = 2 * dot(laplacian_action(ideal), residual) + dirichlet_energy(
        residual
    )
    assert ideal_drop < 0 < rounding_work
    assert ideal_drop + rounding_work == 0


def _capacity_proposal_defect(before, after, eligible, mu):
    """Keep the configured exact masked average and represented residual apart."""
    gradient = laplacian_action(before)
    ideal = tuple(
        value - mu * gradient[i] if i in eligible else value
        for i, value in enumerate(before)
    )
    return ideal, tuple(a - b for a, b in zip(after, ideal, strict=True))


def _capacity_vector(graph):
    return tuple(F(get_attr(graph.nodes[n], ALIAS_VF, strict=True)) for n in graph)


def _cycle_readout(graph, time):
    forcing = capture_non_epi_forcing(graph)
    source = forcing.snapshot
    phases = tuple(map(float, forcing.phase))
    gaps = tuple(angle_diff(phases[(i + 1) % 5], phases[i]) for i in range(5))
    strengths = tuple(
        sum(weight for i, _, weight in source.conductance if i == node)
        for node in range(5)
    )
    return {
        "time": time,
        "source": source,
        "forcing": forcing,
        "phase": phases,
        "gaps": gaps,
        "gap_norm": math.sqrt(sum((value - math.tau / 5) ** 2 for value in gaps)),
        "weighted_mean": dot(strengths, source.epi) / sum(strengths),
        "winding": certify_phase_winding(graph, range(5)),
    }


def _phase_step_defect_estimate(step, dt, coupling):
    """Compare one represented proposal with its ideal Euler formula at 80 dps.

    This is an independent local arithmetic estimate, not another solver or
    an interval enclosure of the continuous trajectory's error.
    """
    with mp.workdps(80):
        phases = [
            mp.mpf(value.numerator) / value.denominator for value in step.before.phase
        ]
        capacities = [
            mp.mpf(value.numerator) / value.denominator
            for value in step.before.snapshot.capacity
        ]
        timestep = mp.mpf(dt.numerator) / dt.denominator
        gain = mp.mpf(coupling.numerator) / coupling.denominator
        residuals = []
        for i, represented in enumerate(step.phase_after):
            current = (
                mp.sin(phases[(i - 1) % 5] - phases[i])
                + mp.sin(phases[(i + 1) % 5] - phases[i])
            ) / 2
            ideal = phases[i] + timestep * (capacities[i] + gain * current)
            output = mp.mpf(represented)
            turn = mp.nint((ideal - output) / (2 * mp.pi))
            residuals.append(float(output + turn * 2 * mp.pi - ideal))
        return tuple(residuals)


@pytest.fixture(scope="module")
def post_event_cycle():
    """One actual UM preparation and full-mix template for independent studies."""
    graph = execute_coupling_cycle_birth()["graph"]
    inject_defaults(graph)
    # The producer's old phase/EPI-only recipe is replaced before the template
    # is captured. The default capacity channel is part of this declared study.
    graph.graph["DNFR_WEIGHTS"] = dict(DEFAULTS["DNFR_WEIGHTS"])
    default_compute_delta_nfr(graph)
    template = bound_cycle_relaxation(
        graph, range(5), coupling_strength=F(1, 2), times=(0, 1, 2, 4)
    )
    return {"graph": graph, "template": template}


@pytest.fixture(scope="module")
def capacity_continuation(post_event_cycle):
    """One prospectively fixed mixed-pressure study, retaining every gate event."""
    graph = _detached_stage_graph(post_event_cycle["graph"])
    template = post_event_cycle["template"]
    timestep, coupling, steps = F(1, 8), F(1, 2), 32
    budget = bound_cycle_capacity_forcing(
        template,
        capacity_bounds=(F(1), F(1) + F(1, 4096)),
        phase_radius=F(3, 2),
        times=tuple(index * timestep for index in range(steps + 1)),
    )
    # Deliberate initial perturbation, not a spontaneous capacity writer.
    set_vf(graph, 0, float(template.capacity + F(1, 4096)))
    default_compute_delta_nfr(graph)
    initial_forcing = capture_non_epi_forcing(graph)
    records = [_cycle_readout(graph, F(0))]
    events = []
    for index in range(steps):
        default_compute_delta_nfr(graph)
        step = execute_joint_step(
            graph, dt=timestep, coupling_strength=coupling, time=index * timestep
        )
        euler = observe_support_transport_euler(
            step.before.snapshot, step.after_epi, timestep
        )
        # execute_joint_step has already refreshed pressure after both state
        # changes. The real adaptation gate now sees a fresh Si as well.
        compute_Si(graph, inplace=True)
        pressure = tuple(
            F(get_attr(graph.nodes[n], ALIAS_DNFR, strict=True)) for n in graph
        )
        sense = tuple(F(get_attr(graph.nodes[n], ALIAS_SI, strict=True)) for n in graph)
        capacity_before = _capacity_vector(graph)
        counts_before = tuple(graph.nodes[n].get("stable_count", 0) for n in graph)
        epi_phase_before = tuple(
            (
                get_attr(graph.nodes[n], ALIAS_EPI, strict=True),
                get_attr(graph.nodes[n], ALIAS_THETA, strict=True),
            )
            for n in graph
        )
        adapt_vf_after_structural_stability(graph)
        counts_after = tuple(graph.nodes[n]["stable_count"] for n in graph)
        eligible = tuple(
            i
            for i, count in enumerate(counts_after)
            if count >= graph.graph["VF_ADAPT_TAU"]
        )
        capacity_after = _capacity_vector(graph)
        mu = F(graph.graph["VF_ADAPT_MU"])
        _, capacity_defect = _capacity_proposal_defect(
            capacity_before, capacity_after, eligible, mu
        )
        events.append(
            {
                "time": (index + 1) * timestep,
                "integrator_time": F(graph.graph["_t"]),
                "step": step,
                "euler": euler,
                "source_work": observe_forcing_dirichlet_balance(step.before),
                "mean_balance": observe_forcing_mean_balance(step.before),
                "phase_step_defect_estimate": _phase_step_defect_estimate(
                    step, timestep, coupling
                ),
                "gate_pressure": pressure,
                "gate_sense": sense,
                "counts_before": counts_before,
                "counts_after": counts_after,
                "eligible": eligible,
                "capacity_before": capacity_before,
                "capacity_after": capacity_after,
                "capacity_defect": capacity_defect,
                "epi_phase_before": epi_phase_before,
                "epi_phase_after": tuple(
                    (
                        get_attr(graph.nodes[n], ALIAS_EPI, strict=True),
                        get_attr(graph.nodes[n], ALIAS_THETA, strict=True),
                    )
                    for n in graph
                ),
            }
        )
        records.append(_cycle_readout(graph, (index + 1) * timestep))
    return {
        "template": template,
        "budget": budget,
        "initial_forcing": initial_forcing,
        "records": tuple(records),
        "events": tuple(events),
        "tau": graph.graph["VF_ADAPT_TAU"],
        "mu": F(graph.graph["VF_ADAPT_MU"]),
        "pressure_threshold": F(graph.graph["EPS_DNFR_STABLE"]),
        "sense_threshold": F(graph.graph["SELECTOR_THRESHOLDS"]["si_hi"]),
        "seed": graph.graph["RANDOM_SEED"],
        "dt": timestep,
        "coupling": coupling,
    }


def test_actual_capacity_gate_admits_only_an_inactive_local_contrast(
    capacity_continuation,
):
    study = capacity_continuation
    assert study["seed"] == 17
    assert study["tau"] == 5 and study["mu"] == F(0.1)
    assert study["pressure_threshold"] == F(0.001)
    assert study["sense_threshold"] == F(0.5)
    initial = (F(1) + F(1, 4096),) + (F(1),) * 4
    for index, event in enumerate(study["events"]):
        assert event["time"] == (index + 1) * F(1, 8)
        assert event["integrator_time"] == event["time"]
        expected_counts = tuple(
            (
                count + 1
                if sense >= study["sense_threshold"]
                and abs(pressure) <= study["pressure_threshold"]
                else 0
            )
            for count, sense, pressure in zip(
                event["counts_before"],
                event["gate_sense"],
                event["gate_pressure"],
                strict=True,
            )
        )
        assert event["counts_after"] == expected_counts
        assert event["eligible"] == (() if index < 4 else (3,))
        assert event["epi_phase_after"] == event["epi_phase_before"]
        assert event["capacity_after"] == event["capacity_before"] == initial
        assert event["capacity_defect"] == (0,) * 5
    # Twenty-eight actual admissions update only an already uniform local
    # capacity neighborhood. They do not remove the deliberately supplied bump.
    assert study["events"][-1]["counts_after"] == (0, 0, 0, 32, 0)
    assert dirichlet_energy(study["records"][-1]["source"].capacity) == F(1, 4096) ** 2


def test_capacity_continuation_retains_signed_source_work_and_numerical_defects(
    capacity_continuation,
):
    study = capacity_continuation
    effective = dict(study["template"].capture.normalized_weights)
    assert effective == dict(study["initial_forcing"].normalized_weights)
    assert effective["vf"] > 0
    assert study["template"].capture.snapshot.capacity == (1,) * 5
    assert any(study["initial_forcing"].snapshot.capacity_gradient)
    for event in study["events"]:
        before = event["step"].before
        assert not any(before.stored_pressure_residual)
        assert before.normalized_weights == study["template"].capture.normalized_weights
        assert before.snapshot.topology_gradient == (0,) * 5
        assert event["euler"].identity_residual == 0
        assert max(map(abs, event["euler"].state_defect)) < F(1, 10**15)
        assert max(map(abs, before.kernel_pressure_defect)) < F(1, 10**15)
        assert max(map(abs, event["phase_step_defect_estimate"])) < 2e-15
        balance = event["source_work"]
        assert balance.identity_residual == 0
        assert balance.source_rate == sum(rate for _, rate in balance.channel_rates)
        capacity_work = dict(balance.channel_rates)["vf"]
        assert capacity_work == sum(
            gradient * capacity * effective["vf"] * source
            for gradient, capacity, source in zip(
                before.snapshot.dirichlet_gradient,
                before.snapshot.capacity,
                before.snapshot.capacity_gradient,
                strict=True,
            )
        )
    assert any(
        dict(event["source_work"].channel_rates)["vf"] for event in study["events"]
    )
    initial, final = study["records"][0], study["records"][-1]
    assert initial["source"].dirichlet_energy == 0 < final["source"].dirichlet_energy
    assert final["weighted_mean"] != initial["weighted_mean"]
    assert 0 < final["gap_norm"] < initial["gap_norm"]
    for record in study["records"]:
        assert record["source"].conductance == initial["source"].conductance
        assert record["winding"].winding == 1 and record["winding"].u3_admissible
        assert record["winding"].minimum_u3_margin > 0
        assert all(-1 < value < 1 for value in record["source"].epi)


def test_finite_capacity_trace_compares_with_prospective_continuous_budget(
    capacity_continuation,
):
    """Measured containment does not promote Euler to a continuous certificate."""
    study = capacity_continuation
    budget = study["budget"]
    assert budget.admitted and budget.admission_failure is None
    assert budget.phase_tube_radius_upper < budget.phase_radius
    assert budget.phase_gap_tail_upper > 0
    assert budget.epi_energy_tail_upper > 0
    records = {record["time"]: record for record in study["records"]}
    assert len(budget.samples) == len(records) == 33
    # The band was computed before the perturbation and trajectory. It covers
    # the whole finite prefix; no clipping or infinite-horizon conclusion is
    # transferred from the earlier held-capacity envelope.
    assert -1 < budget.samples[-1].epi_interval[0]
    assert budget.samples[-1].epi_interval[1] < 1
    # Coarser outward rational endpoints used by the public study summary.
    assert F(-225, 1000) <= budget.samples[-1].epi_interval[0]
    assert budget.samples[-1].epi_interval[1] <= F(475, 1000)
    for sample in budget.samples:
        observed = records[sample.time]
        forcing = observed["forcing"]
        capacity = observed["source"].capacity
        assert (
            budget.capacity_bounds[0]
            <= min(capacity)
            <= max(capacity)
            <= budget.capacity_bounds[1]
        )
        assert observed["gap_norm"] <= float(sample.gap_deviation_norm_upper) + 2e-13
        assert (
            max(map(abs, observed["gaps"]))
            <= float(budget.phase_tube_radius_upper) + 2e-13
        )
        phase_source_norm = math.sqrt(
            float(dot(forcing.phase_gradient, forcing.phase_gradient))
        )
        source_norm = math.sqrt(float(dot(forcing.forcing, forcing.forcing)))
        assert phase_source_norm <= float(sample.phase_source_norm_upper) + 2e-13
        assert source_norm <= float(sample.non_epi_source_norm_upper) + 2e-13
        assert (
            float(observed["source"].dirichlet_energy)
            <= float(sample.epi_dirichlet_energy_upper) + 2e-13
        )
        low, high = sample.epi_interval
        for time, prefix in records.items():
            if time <= sample.time:
                assert all(
                    float(low) - 2e-13 <= float(value) <= float(high) + 2e-13
                    for value in prefix["source"].epi
                )


def test_fixed_strength_mean_has_an_exact_finite_channel_telescope(
    capacity_continuation,
):
    study = capacity_continuation
    area = dict.fromkeys(
        ("diffusion", "phase", "vf", "topo", "kernel", "stored", "endpoint"), F(0)
    )
    for index, event in enumerate(study["events"]):
        balance = event["mean_balance"]
        strengths, total = balance.strengths, balance.total_strength
        after_epi = event["step"].after_epi.epi
        after_mean = dot(strengths, after_epi) / total
        endpoint = dot(strengths, event["euler"].state_defect) / total
        assert balance.mean == study["records"][index]["weighted_mean"]
        assert after_mean == study["records"][index + 1]["weighted_mean"]
        assert event["epi_phase_before"] == event["epi_phase_after"]
        assert balance.identity_residual == 0
        assert (
            after_mean - balance.mean
            == study["dt"]
            * (
                balance.modeled_rate
                + balance.kernel_defect_rate
                + balance.stored_residual_rate
            )
            + endpoint
        )
        assert (
            balance.diffusion_rate
            == -event["step"].before.epi_weight
            * dot(balance.source.capacity, balance.source.dirichlet_gradient)
            / total
        )
        area["diffusion"] += study["dt"] * balance.diffusion_rate
        for name, rate in balance.channel_rates:
            area[name] += study["dt"] * rate
        area["kernel"] += study["dt"] * balance.kernel_defect_rate
        area["stored"] += study["dt"] * balance.stored_residual_rate
        area["endpoint"] += endpoint
    observed = (
        study["records"][-1]["weighted_mean"] - study["records"][0]["weighted_mean"]
    )
    assert sum(area.values()) == observed
    assert observed != 0 and area["diffusion"] != 0 and area["vf"] != 0
    assert area["stored"] == 0


@pytest.fixture(scope="module")
def interval_locked_state(post_event_cycle):
    """One supplied analytic state shared by static and first-feedback controls."""
    template = post_event_cycle["template"]
    source = template.capture.snapshot
    weights = dict(template.capture.normalized_weights)
    strength = template.strengths
    beta = next(weight for i, j, weight in source.conductance if (i, j) == (0, 4))
    a, coupling = F(1, 4096), template.coupling_strength
    graph = nx.Graph()
    configure(graph, weights=DEFAULTS["DNFR_WEIGHTS"])
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17,
        DELTA_PHI_MAX=float(template.effective_phase_gate),
        UM_MAX_PHASE_DIFF=float(template.effective_phase_gate),
    )
    with mp.workdps(110):

        def exact(value):
            value = F(value)
            return mp.mpf(value.numerator) / value.denominator

        d, amplitude, gain = 2 * mp.pi / 5, exact(a), exact(coupling)
        kappa = 1 + amplitude / 2
        gaps = (d - amplitude, d, d, d + amplitude, d)
        shape = tuple(amplitude * (value - mp.mpf(2) / 5) for value in (1, 0, 0, 0, 1))
        phase = tuple(mp.mpf("0.125") + i * d + shape[i] for i in range(5))
        current = tuple((mp.sin(gaps[i]) - mp.sin(gaps[i - 1])) / 2 for i in range(5))
        capacity = tuple(kappa - gain * value for value in current)
        phase_pressure = tuple((gaps[i] - gaps[i - 1]) / (2 * mp.pi) for i in range(5))
        capacity_pressure = tuple(
            (capacity[i - 1] + capacity[(i + 1) % 5]) / 2 - capacity[i]
            for i in range(5)
        )
        forcing = tuple(
            exact(weights["phase"]) * g + exact(weights["vf"]) * h
            for g, h in zip(phase_pressure, capacity_pressure, strict=True)
        )
        numerator = mp.fsum(
            exact(s) * value for s, value in zip(strength, forcing, strict=True)
        )
        analytic_numerator = (1 - exact(beta)) * (
            exact(weights["phase"]) * amplitude / mp.pi
            + exact(weights["vf"]) * gain * mp.cos(d) * mp.sin(amplitude)
        )
        h_total = mp.fsum(
            exact(s) / nu for s, nu in zip(strength, capacity, strict=True)
        )
        drift = numerator / h_total
        lower_drift = (
            (1 - beta) * weights["phase"] * a / (_pi_bounds()[1] * sum(strength))
        )

        for i, node in enumerate(source.nodes):
            graph.add_node(
                node,
                EPI=float(source.epi[i]),
                theta=float(phase[i]),
                nu_f=float(capacity[i]),
                delta_nfr=0.0,
                stable_count=0,
            )
        graph.add_edges_from(
            (source.nodes[i], source.nodes[j], {"weight": float(weight)})
            for i, j, weight in source.conductance
            if i < j
        )
        ideal_d_rate = mp.fsum(
            exact(s) * nu * f
            for s, nu, f in zip(strength, capacity, forcing, strict=True)
        ) / exact(sum(strength))
    return {
        "graph": graph,
        "template": template,
        "a": a,
        "coupling": coupling,
        "beta": beta,
        "kappa": kappa,
        "ideal_capacity": capacity,
        "ideal_current": current,
        "ideal_gaps": gaps,
        "phase_pressure": phase_pressure,
        "forcing": forcing,
        "source_numerator": numerator,
        "analytic_numerator": analytic_numerator,
        "h_drift": drift,
        "lower_drift": lower_drift,
        "d_drift": ideal_d_rate,
    }


def test_interval_family_can_lock_phase_shape_while_the_form_mean_drifts(
    interval_locked_state,
):
    """A supplied static comparison state, not the actual adaptation trajectory."""
    state = interval_locked_state
    graph, template = state["graph"], state["template"]
    source, strength = template.capture.snapshot, template.strengths
    with mp.workdps(110):
        assert 0 < state["beta"] < 1
        assert all(1 <= nu <= 1 + mp.mpf(1) / 4096 for nu in state["ideal_capacity"])
        assert max(state["ideal_gaps"]) < mp.mpf(3) / 2
        assert mp.almosteq(state["source_numerator"], state["analytic_numerator"])
        assert 0 < float(state["lower_drift"]) < state["h_drift"]
        assert all(
            mp.almosteq(nu + mp.mpf("0.5") * j, state["kappa"])
            for nu, j in zip(
                state["ideal_capacity"], state["ideal_current"], strict=True
            )
        )
        captured = capture_non_epi_forcing(graph)
        assert captured.normalized_weights == template.capture.normalized_weights
        for actual, ideal in zip(
            captured.phase_gradient, state["phase_pressure"], strict=True
        ):
            assert float(actual) == pytest.approx(float(ideal), rel=0, abs=2e-15)
        for actual, ideal in zip(captured.forcing, state["forcing"], strict=True):
            assert float(actual) == pytest.approx(float(ideal), rel=0, abs=2e-15)
        actual_h_total = sum(
            s / nu for s, nu in zip(strength, captured.snapshot.capacity, strict=True)
        )
        actual_h_rate = dot(strength, captured.full_kernel_pressure) / actual_h_total
        assert float(actual_h_rate) == pytest.approx(
            float(state["h_drift"]), rel=0, abs=2e-15
        )
        assert actual_h_rate > state["lower_drift"] > 0
        mean_balance = observe_forcing_mean_balance(captured)
        assert float(mean_balance.fresh_rate) == pytest.approx(
            float(state["d_drift"]), rel=0, abs=2e-15
        )
        proposed = propose_u3_gated_phase_step(
            graph,
            source.nodes,
            captured.phase,
            captured.snapshot.capacity,
            dt=0.125,
            coupling_strength=float(state["coupling"]),
        )
        for old, new in zip(captured.phase, proposed, strict=True):
            assert angle_diff(float(new), float(old)) == pytest.approx(
                float(state["kappa"] / 8), rel=0, abs=2e-15
            )


def _capacity_metric_mean(capture, strengths):
    weights = _capacity_metric_weights(capture, strengths)
    return dot(weights, capture.snapshot.epi) / sum(weights)


def _capacity_metric_weights(capture, strengths):
    return tuple(
        strength / capacity
        for strength, capacity in zip(strengths, capture.snapshot.capacity, strict=True)
    )


@pytest.fixture(scope="module")
def first_lock_feedback(interval_locked_state):
    """Exactly five declared intervals, ending at the first default gate window."""
    state = interval_locked_state
    graph = _detached_stage_graph(state["graph"])
    strengths = state["template"].strengths
    dt, coupling = F(1, 8), state["coupling"]
    initial_counts = tuple(graph.nodes[n].get("stable_count", 0) for n in graph)
    records = []
    for index in range(5):
        default_compute_delta_nfr(graph)
        step = execute_joint_step(
            graph, dt=dt, coupling_strength=coupling, time=index * dt
        )
        compute_Si(graph, inplace=True)
        lock_before = observe_phase_lock_source(graph, coupling_strength=coupling)
        before = lock_before.capture
        sense = tuple(F(get_attr(graph.nodes[n], ALIAS_SI, strict=True)) for n in graph)
        counts_before = tuple(graph.nodes[n].get("stable_count", 0) for n in graph)
        adapt_vf_after_structural_stability(graph)
        counts_after = tuple(graph.nodes[n]["stable_count"] for n in graph)
        eligible = tuple(
            i
            for i, count in enumerate(counts_after)
            if count >= graph.graph["VF_ADAPT_TAU"]
        )
        after = capture_non_epi_forcing(graph)
        difference = observe_forcing_capacity_difference(before, after)
        mu = F(graph.graph["VF_ADAPT_MU"])
        ideal_capacity, capacity_defect = _capacity_proposal_defect(
            before.snapshot.capacity, after.snapshot.capacity, eligible, mu
        )
        mean_change = sum(difference.capacity_change) / len(difference.capacity_change)
        lock_after = tuple(
            residual + delta - mean_change
            for residual, delta in zip(
                lock_before.full_support_rate_residual,
                difference.capacity_change,
                strict=True,
            )
        )
        phase_weight = dict(before.normalized_weights)["phase"]
        phase_rate_numerator_before = -phase_weight * dot(
            strengths, laplacian_action(lock_before.full_support_rate_residual)
        )
        phase_rate_numerator_after = -phase_weight * dot(
            strengths, laplacian_action(lock_after)
        )
        records.append(
            {
                "time": (index + 1) * dt,
                "integrator_time": F(graph.graph["_t"]),
                "step": step,
                "before": before,
                "after": after,
                "sense": sense,
                "counts_before": counts_before,
                "counts_after": counts_after,
                "eligible": eligible,
                "difference": difference,
                "ideal_capacity": ideal_capacity,
                "capacity_defect": capacity_defect,
                "source_compatibility_before": dot(strengths, before.forcing),
                "source_compatibility_after": dot(strengths, after.forcing),
                "h_rate_before": dot(strengths, before.forcing)
                / sum(_capacity_metric_weights(before, strengths)),
                "h_rate_after": dot(strengths, after.forcing)
                / sum(_capacity_metric_weights(after, strengths)),
                "source_kernel_residual_before": dot(
                    strengths, before.kernel_pressure_defect
                ),
                "source_kernel_residual_after": dot(
                    strengths, after.kernel_pressure_defect
                ),
                "source_stored_residual_after": dot(
                    strengths, after.stored_pressure_residual
                ),
                "d_mean_before": observe_forcing_mean_balance(before),
                "d_mean_after": observe_forcing_mean_balance(after),
                "h_mean_before": _capacity_metric_mean(before, strengths),
                "h_mean_after": _capacity_metric_mean(after, strengths),
                "lock_before": lock_before,
                "lock_after": lock_after,
                "phase_rate_numerator_before": phase_rate_numerator_before,
                "phase_rate_numerator_after": phase_rate_numerator_after,
            }
        )
    last = records[-1]["after"]
    next_phase = propose_u3_gated_phase_step(
        graph,
        last.snapshot.nodes,
        last.phase,
        last.snapshot.capacity,
        dt=float(dt),
        coupling_strength=float(coupling),
    )
    return {
        "state": state,
        "initial_counts": initial_counts,
        "records": tuple(records),
        "dt": dt,
        "tau": graph.graph["VF_ADAPT_TAU"],
        "mu": F(graph.graph["VF_ADAPT_MU"]),
        "pressure_threshold": F(graph.graph["EPS_DNFR_STABLE"]),
        "sense_threshold": F(graph.graph["SELECTOR_THRESHOLDS"]["si_hi"]),
        "next_phase_proposal": tuple(map(float, next_phase)),
        "post_write_lock": observe_phase_lock_source(graph, coupling_strength=coupling),
    }


def test_first_locked_state_feedback_uses_fresh_default_gates(first_lock_feedback):
    study = first_lock_feedback
    assert study["initial_counts"] == (0,) * 5
    assert study["tau"] == 5 and study["mu"] == F(0.1)
    assert study["pressure_threshold"] == F(0.001)
    assert study["sense_threshold"] == F(0.5)
    assert len(study["records"]) == 5
    for index, event in enumerate(study["records"]):
        assert event["time"] == event["integrator_time"] == (index + 1) * F(1, 8)
        assert not any(event["step"].before.stored_pressure_residual)
        assert not any(event["before"].stored_pressure_residual)
        assert event["before"].snapshot.epi == event["step"].after_epi.epi
        assert event["before"].phase == tuple(map(F, event["step"].phase_after))
        expected_counts = tuple(
            (
                count + 1
                if sense >= study["sense_threshold"]
                and abs(pressure) <= study["pressure_threshold"]
                else 0
            )
            for count, sense, pressure in zip(
                event["counts_before"],
                event["sense"],
                event["before"].snapshot.stored_pressure,
                strict=True,
            )
        )
        assert event["counts_after"] == expected_counts == (index + 1,) * 5
        assert event["eligible"] == (() if index < 4 else tuple(range(5)))
        difference = event["difference"]
        assert difference.epi_offset == difference.phase_offset == 0
        assert event["after"].snapshot.epi == event["before"].snapshot.epi
        assert event["after"].phase == event["before"].phase
        assert not any(difference.identity_residual)
        assert not any(difference.stored_identity_residual)
        assert max(map(abs, event["capacity_defect"])) < F(1, 10**15)
        strengths = study["state"]["template"].strengths
        vf_weight = dict(event["before"].normalized_weights)["vf"]
        capacity_gradient = laplacian_action(event["before"].snapshot.capacity)
        masked_gradient = tuple(
            value if i in event["eligible"] else F(0)
            for i, value in enumerate(capacity_gradient)
        )
        # Exact signed source ledger for the actual mask and represented
        # writer residual; the conditional all-target sign is a separate test.
        assert event["source_compatibility_after"] - event[
            "source_compatibility_before"
        ] == study["mu"] * vf_weight * dot(
            laplacian_action(strengths), masked_gradient
        ) - vf_weight * dot(
            strengths, laplacian_action(event["capacity_defect"])
        )
        assert min(event["before"].snapshot.capacity) <= min(
            event["after"].snapshot.capacity
        )
        assert max(event["after"].snapshot.capacity) <= max(
            event["before"].snapshot.capacity
        )
        assert bool(any(difference.capacity_change)) == (index == 4)
        euler = observe_support_transport_euler(
            event["step"].before.snapshot, event["step"].after_epi, study["dt"]
        )
        assert euler.identity_residual == 0
        assert max(map(abs, euler.state_defect)) < F(1, 10**15)
        assert all(-1 < value < 1 for value in euler.expected_epi)


def test_first_capacity_write_changes_source_and_metric_without_moving_form_or_phase(
    first_lock_feedback,
):
    study = first_lock_feedback
    event = study["records"][-1]
    strengths = study["state"]["template"].strengths
    difference = event["difference"]
    change = event["source_compatibility_after"] - event["source_compatibility_before"]
    assert change == dot(strengths, difference.capacity_pressure_change)
    assert difference.phase_realization_change == (0,) * 5
    assert change < 0 < event["source_compatibility_after"]
    assert 0 < event["h_rate_after"] < event["h_rate_before"]
    assert event["d_mean_after"].mean == event["d_mean_before"].mean

    h_before = tuple(
        s / nu
        for s, nu in zip(strengths, event["before"].snapshot.capacity, strict=True)
    )
    h_after = tuple(
        s / nu
        for s, nu in zip(strengths, event["after"].snapshot.capacity, strict=True)
    )
    reweighting = sum(
        (new - old) * (x - event["h_mean_before"])
        for new, old, x in zip(
            h_after, h_before, event["before"].snapshot.epi, strict=True
        )
    ) / sum(h_after)
    assert event["h_mean_after"] - event["h_mean_before"] == reweighting < 0
    assert (
        dot(strengths, difference.modeled_pressure_before)
        == event["source_compatibility_before"]
    )
    assert (
        dot(strengths, difference.modeled_pressure_after)
        == event["source_compatibility_after"]
    )
    assert event["h_rate_before"] == event["source_compatibility_before"] / sum(
        h_before
    )
    assert event["h_rate_after"] == event["source_compatibility_after"] / sum(h_after)
    assert (
        dot(strengths, event["after"].full_kernel_pressure)
        == event["source_compatibility_after"] + event["source_kernel_residual_after"]
    )
    assert dot(strengths, event["after"].snapshot.stored_pressure) == (
        event["source_compatibility_after"]
        + event["source_kernel_residual_after"]
        + event["source_stored_residual_after"]
    )

    # Same phase coefficients plus the centered capacity jump account exactly
    # for the changed lock residual. A tolerance does not make the old residual
    # an exact lock or the new state a different long-time locked solution.
    assert event["lock_before"].full_u3_admission
    assert study["post_write_lock"].full_u3_admission
    assert event["lock_after"] == study["post_write_lock"].full_support_rate_residual
    assert max(map(abs, event["lock_before"].full_support_rate_residual)) < F(1, 10**14)
    assert max(map(abs, event["lock_after"])) > F(1, 10**6)
    old_gaps = tuple(
        angle_diff(
            float(event["after"].phase[(i + 1) % 5]), float(event["after"].phase[i])
        )
        for i in range(5)
    )
    next_gaps = tuple(
        angle_diff(
            study["next_phase_proposal"][(i + 1) % 5], study["next_phase_proposal"][i]
        )
        for i in range(5)
    )
    for i, (old, following) in enumerate(zip(old_gaps, next_gaps, strict=True)):
        assert following - old == pytest.approx(
            float(
                study["dt"]
                * (event["lock_after"][(i + 1) % 5] - event["lock_after"][i])
            ),
            rel=0,
            abs=2e-15,
        )
    assert (
        max(abs(new - old) for new, old in zip(next_gaps, old_gaps, strict=True)) > 1e-8
    )


def test_first_feedback_matches_the_locked_family_source_signs(first_lock_feedback):
    study = first_lock_feedback
    state, event = study["state"], study["records"][-1]
    weights = dict(state["template"].capture.normalized_weights)
    strengths = state["template"].strengths
    exact_rate_numerator_change = -weights["phase"] * dot(
        strengths, laplacian_action(event["difference"].capacity_change)
    )
    assert (
        event["phase_rate_numerator_after"] - event["phase_rate_numerator_before"]
        == exact_rate_numerator_change
        < 0
    )
    with mp.workdps(110):

        def exact(value):
            value = F(value)
            return mp.mpf(value.numerator) / value.denominator

        mu, gain, t = (
            exact(study["mu"]),
            exact(state["coupling"]),
            exact(state["beta"] - 1),
        )
        c = mp.cos(2 * mp.pi / 5) * mp.sin(exact(state["a"]))
        ideal_source_change = 5 * mu * exact(weights["vf"]) * gain * t * c / 4
        ideal_following_source = -t * (
            exact(weights["phase"]) * exact(state["a"]) / mp.pi
            + exact(weights["vf"]) * gain * c * (1 - 5 * mu / 4)
        )
        ideal_phase_source_rate_change = (
            5 * mu * exact(weights["phase"]) * gain * t * c / (4 * mp.pi)
        )
        assert ideal_source_change < 0 < ideal_following_source
        assert ideal_phase_source_rate_change < 0
        assert float(event["source_compatibility_before"]) == pytest.approx(
            float(state["analytic_numerator"]), rel=0, abs=2e-15
        )
        assert float(event["source_compatibility_after"]) == pytest.approx(
            float(ideal_following_source), rel=0, abs=2e-15
        )
        assert float(
            event["source_compatibility_after"] - event["source_compatibility_before"]
        ) == pytest.approx(float(ideal_source_change), rel=0, abs=2e-15)
        assert float(exact(exact_rate_numerator_change) / mp.pi) == pytest.approx(
            float(ideal_phase_source_rate_change), rel=0, abs=2e-15
        )
        # This is the instantaneous response derivative after a capacity event,
        # not an immediate phase-pressure jump or a finite-difference fit.
        assert abs(float(event["phase_rate_numerator_before"])) < 1e-14
        assert event["phase_rate_numerator_after"] < 0
