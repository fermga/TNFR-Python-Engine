"""Two finite contacts, one formation negative control and three static budgets.

Trajectories reuse the production phase proposal, NumPy pressure and scalar
nodal Euler; static budget proposals are never applied. High-precision estimates
are not validated intervals or continuous-ODE error certificates. Preparation
is not autonomous formation.
"""

import json
import math
import platform
from copy import deepcopy
from fractions import Fraction as F
from importlib.metadata import version

import mpmath as mp
import networkx as nx
import pytest

from tests.joint_phase_helpers import (
    configure,
    exact_phase_cycle_state,
    execute_joint_step,
)
from tnfr.dynamics import integrators
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.operators._phase_gate import resolve_u3_phase_neighbors
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_cycle_geometry import derive_phase_cycle_geometry
from tnfr.physics.support_transport import observe_support_transport_euler
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import angle_diff

H, STEPS = F(1, 128), 64
RINGS = (tuple(range(5)), tuple(range(5, 10)))
TOL = 1e-12  # finite arithmetic comparison; not a physical or admission threshold


def _mp(value):
    if isinstance(value, F):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _read(graph, time, lifts):
    phase = tuple(graph.nodes[node]["theta"] for node in graph)
    gaps = tuple(angle_diff(phase[j], phase[i]) for i, j in graph.edges())
    with mp.workdps(80):
        offset = tuple(
            float(_mp(lift) - _mp(time) - 2 * mp.pi * (i % 5) / 5)
            for i, lift in enumerate(lifts)
        )
    return {
        "time": time,
        "phase": phase,
        "lifts": tuple(lifts),
        "offset": offset,
        "gaps": gaps,
        "energy": math.fsum(1 - math.cos(gap) for gap in gaps),
        "epi": tuple(graph.nodes[node]["EPI"] for node in graph),
        "winding": tuple(certify_phase_winding(graph, ring) for ring in RINGS),
    }


@pytest.fixture(scope="module")
def contacts(record_testsuite_property):
    """Execute only the two predeclared preparations, each exactly once."""
    reports = []
    with pytest.MonkeyPatch.context() as patch, mp.workdps(80):
        # Pressure retains its NumPy owner; only EPI uses the scalar integrator.
        patch.setattr(integrators, "np", None)
        for beta_turn in (F(1, 48), F(1, 24)):
            graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
            graph.add_edge(0, 5)
            configure(graph)
            graph.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, use_extended_dynamics=False)
            nx.set_edge_attributes(graph, 1.0, "weight")
            nx.set_edge_attributes(graph, 1.0, "length")
            turns = tuple(F(i % 5, 5) + (beta_turn if i >= 5 else 0) for i in graph)
            exact = exact_phase_cycle_state(graph, turns)
            for node, turn in zip(graph, turns, strict=True):
                graph.nodes[node].update(
                    EPI=0.5,
                    theta=float(2 * mp.pi * _mp(turn)),
                    nu_f=1.0,
                    delta_nfr=0.0,
                    dEPI=0.0,
                )
            default_compute_delta_nfr(graph)
            edges = tuple(graph.edges())
            lifts = tuple(_mp(graph.nodes[node]["theta"]) for node in graph)
            initial_error = max(
                abs(lift - 2 * mp.pi * _mp(turn))
                for lift, turn in zip(lifts, turns, strict=True)
            )
            trace = [_read(graph, F(0), lifts)]
            records = []
            cumulative_phase_defect = mp.mpf(0)
            for index in range(STEPS):
                step = execute_joint_step(
                    graph, dt=H, coupling_strength=1, time=index * H
                )
                before = step.before
                phase = tuple(_mp(value) for value in before.phase)
                increments = tuple(
                    angle_diff(new, float(old))
                    for old, new in zip(before.phase, step.phase_after, strict=True)
                )
                phase_defect, source_defect = [], []
                for i, neighbors in enumerate(before.snapshot.support_neighbors):
                    # The accumulated tube budget belongs to these retained
                    # lifts, not to raw phases plus assumed exact turn offsets.
                    # Rounded angle_diff increments can separate those charts.
                    lift_differences = tuple(lifts[j] - lifts[i] for j in neighbors)
                    rate = 1 + sum(mp.sin(value) for value in lift_differences) / len(
                        neighbors
                    )
                    phase_defect.append(float(_mp(increments[i]) - _mp(H) * rate))
                    # Pressure is evaluated on the actual stored raw phases.
                    differences = tuple(phase[j] - phase[i] for j in neighbors)
                    ideal_source = (
                        mp.atan2(
                            sum(mp.sin(value) for value in differences),
                            sum(mp.cos(value) for value in differences),
                        )
                        / mp.pi
                    )
                    source_defect.append(
                        float(_mp(before.phase_gradient[i]) - ideal_source)
                    )
                cumulative_phase_defect += max(map(abs, map(_mp, phase_defect)))
                lifts = tuple(
                    old + _mp(delta)
                    for old, delta in zip(lifts, increments, strict=True)
                )
                following = _read(graph, (index + 1) * H, lifts)
                euler = observe_support_transport_euler(
                    before.snapshot, step.after_epi, H
                )
                records.append(
                    {
                        "step": step,
                        "increments": increments,
                        "euler": euler,
                        "phase_defect": tuple(phase_defect),
                        "source_defect": tuple(source_defect),
                        "tube_rounding_estimate": float(
                            initial_error + cumulative_phase_defect
                        ),
                        "actual_time": F(graph.graph["_t"]),
                    }
                )
                trace.append(following)
            before_cut = deepcopy(dict(graph.nodes(data=True)))
            graph.remove_edge(0, 5)
            after_cut = deepcopy(dict(graph.nodes(data=True)))
            cut_geometry = tuple(
                derive_phase_cycle_geometry(graph.subgraph(ring)) for ring in RINGS
            )
            cut_winding = tuple(certify_phase_winding(graph, ring) for ring in RINGS)
            reports.append(
                {
                    "beta_turn": beta_turn,
                    "beta": float(2 * mp.pi * _mp(beta_turn)),
                    "initial_materialization_error": float(initial_error),
                    "exact": exact,
                    "edges": edges,
                    "trace": tuple(trace),
                    "records": tuple(records),
                    "before_cut": before_cut,
                    "after_cut": after_cut,
                    "cut_geometry": cut_geometry,
                    "cut_winding": cut_winding,
                    "provenance": (
                        "numpy_default_pressure",
                        "shared_u3_phase_proposal",
                        "scalar_default_integrator_euler",
                        "binary64",
                        80,
                    ),
                }
            )
    record_testsuite_property(
        "contact_runtime",
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "networkx": nx.__version__,
                "mpmath": mp.__version__,
                "h": str(H),
                "steps": STEPS,
                "duration": str(H * STEPS),
                "coupling": 1,
                "provenance": reports[0]["provenance"],
                "arithmetic_scope": "estimates_not_validated_intervals_or_ODE_error_bounds",
            },
            sort_keys=True,
        ),
    )
    for report in reports:
        endpoint, records = report["trace"][-1], report["records"]
        max_gap = max(abs(gap) for state in report["trace"] for gap in state["gaps"])
        metrics = {
            "beta_turn": str(report["beta_turn"]),
            "endpoint_max_form_deviation": max(abs(x - 0.5) for x in endpoint["epi"]),
            "endpoint_max_phase_deformation": max(
                abs(offset - (report["beta"] if i >= 5 else 0))
                for i, offset in enumerate(endpoint["offset"])
            ),
            "maximum_all_edge_gap": max_gap,
            "minimum_acute_margin": math.pi / 2 - max_gap,
            "max_phase_proposal_defect_estimate": max(
                abs(x) for r in records for x in r["phase_defect"]
            ),
            "max_source_defect_estimate": max(
                abs(x) for r in records for x in r["source_defect"]
            ),
            "max_pressure_assembly_defect": float(
                max(
                    abs(x)
                    for r in records
                    for x in r["step"].before.kernel_pressure_defect
                )
            ),
            "max_epi_euler_defect": float(
                max(abs(x) for r in records for x in r["euler"].state_defect)
            ),
            "initial_materialization_error_estimate": report[
                "initial_materialization_error"
            ],
            "cumulative_phase_and_initial_error_estimate": records[-1][
                "tube_rounding_estimate"
            ],
            "cosine_energy_change": endpoint["energy"] - report["trace"][0]["energy"],
            "endpoint_ring_form_means": tuple(
                math.fsum(endpoint["epi"][i] for i in ring) / 5 for ring in RINGS
            ),
        }
        record_testsuite_property(
            "contact_beta_pi_" + str(report["beta_turn"].denominator // 2),
            json.dumps(metrics, sort_keys=True),
        )
    return tuple(reports)


def test_energy_barrier_is_sufficient_for_only_one_preparation_but_tube_admits_both():
    with mp.workdps(80):
        q = 2 * mp.pi / 5
        # On one original ring, the least acute-boundary cost has one gap pi/2
        # and four gaps 3*pi/8; the other ring remains at its winding minimum.
        baseline = 10 * (1 - mp.cos(q))
        ring_boundary = 1 + 4 * (1 - mp.cos(3 * mp.pi / 8)) + baseline / 2
        bridge_boundary = baseline + 1
        barrier = min(ring_boundary, bridge_boundary)
        decisions = []
        for divisor in (24, 12):
            beta = mp.pi / divisor
            decisions.append(baseline + 1 - mp.cos(beta) < barrier)
            # A cooperative secant response traps theta-t-reference in [0,beta].
            assert q + beta < mp.pi / 2
            assert 0 < H <= 1
        assert decisions == [True, False]


def test_initial_contact_response_uses_full_degree_three_normalization(contacts):
    with mp.workdps(80):
        q = 2 * mp.pi / 5
        for report in contacts:
            beta = 2 * mp.pi * _mp(report["beta_turn"])
            source = mp.atan2(mp.sin(beta), 2 * mp.cos(q) + mp.cos(beta)) / mp.pi
            first = report["records"][0]
            snapshot = first["step"].before.snapshot
            assert tuple(map(len, snapshot.support_neighbors)) == (
                3,
                2,
                2,
                2,
                2,
                3,
                2,
                2,
                2,
                2,
            )
            expected_source = (float(source), 0, 0, 0, 0, -float(source), 0, 0, 0, 0)
            assert tuple(
                map(float, first["step"].before.phase_gradient)
            ) == pytest.approx(expected_source, rel=0, abs=TOL)
            assert tuple(map(float, snapshot.rate)) == pytest.approx(
                tuple(x / 2 for x in expected_source), rel=0, abs=TOL
            )
            expected_rates = [1.0] * 10
            expected_rates[0] += float(mp.sin(beta) / 3)
            expected_rates[5] -= float(mp.sin(beta) / 3)
            assert first["increments"] == pytest.approx(
                tuple(float(H) * x for x in expected_rates), rel=0, abs=TOL
            )


def test_contact_trace_retains_each_winding_and_an_acute_piecewise_linear_path(
    contacts,
):
    for report in contacts:
        assert len(report["trace"]) == 65 and len(report["records"]) == 64
        exact = report["exact"]
        assert exact.geometry.cycle_rank == 2
        assert len(exact.geometry.bridge_edge_indices) == 1
        assert all(abs(period) == 1 for period in exact.cycle_periods)
        assert len(report["edges"]) == 11
        for index, state in enumerate(report["trace"]):
            assert state["time"] == index * H
            assert max(map(abs, state["gaps"])) < math.pi / 2
            assert all(
                item.is_defined and item.winding == 1 and item.u3_admissible
                for item in state["winding"]
            )
            allowance = (
                TOL
                if index == 0
                else TOL + report["records"][index - 1]["tube_rounding_estimate"]
            )
            assert min(state["offset"]) >= -allowance
            assert max(state["offset"]) <= report["beta"] + allowance
            if index:
                previous = report["trace"][index - 1]
                record = report["records"][index - 1]
                assert record["actual_time"] == state["time"]
                assert max(map(abs, record["increments"])) < 2 * float(H) + TOL
                affine_end = tuple(
                    gap + record["increments"][j] - record["increments"][i]
                    for gap, (i, j) in zip(
                        previous["gaps"], report["edges"], strict=True
                    )
                )
                assert affine_end == pytest.approx(state["gaps"], rel=0, abs=TOL)
                # Both endpoints in one acute branch certify only this Euler
                # interpolation; the separate analytical theorem owns the ODE.
                assert max(map(abs, affine_end)) < math.pi / 2
                assert state["energy"] <= previous["energy"] + TOL


def test_source_form_envelope_and_endpoint_defects_remain_separate(contacts):
    for report in contacts:
        assert report["provenance"] == (
            "numpy_default_pressure",
            "shared_u3_phase_proposal",
            "scalar_default_integrator_euler",
            "binary64",
            80,
        )
        assert report["initial_materialization_error"] < TOL
        assert F(1, 2) * report["beta_turn"] in (F(1, 96), F(1, 48))
        for state in report["trace"]:
            # beta/pi = 2*beta_turn, e=w=1/2, so the radius is t*beta_turn.
            radius = float(state["time"] * report["beta_turn"])
            assert max(abs(x - 0.5) for x in state["epi"]) <= radius + TOL
            assert min(state["epi"]) > 0 and max(state["epi"]) < 1
        for record in report["records"]:
            before, euler = record["step"].before, record["euler"]
            assert dict(before.normalized_weights) == {
                "phase": F(1, 2),
                "epi": F(1, 2),
                "vf": 0,
                "topo": 0,
            }
            assert not any(before.stored_pressure_residual)
            assert max(map(abs, before.phase_gradient)) <= 2 * report["beta_turn"] + TOL
            assert max(map(abs, before.kernel_pressure_defect)) < TOL
            assert max(map(abs, record["phase_defect"])) < TOL
            assert max(map(abs, record["source_defect"])) < TOL
            assert record["tube_rounding_estimate"] < TOL
            assert max(map(abs, euler.state_defect)) < TOL
            assert euler.identity_residual == 0
            # Every held, unclipped Euler proposal lies strictly inside rails.
            assert min(euler.expected_epi) > 0 and max(euler.expected_epi) < 1
        assert report["trace"][-1]["epi"] != report["trace"][0]["epi"]


def test_contact_removal_preserves_nodal_state_and_original_cycle_identity(contacts):
    for report in contacts:
        assert report["before_cut"] == report["after_cut"]
        assert all(geometry.cycle_rank == 1 for geometry in report["cut_geometry"])
        assert all(
            item.is_defined and item.winding == 1 and item.u3_admissible
            for item in report["cut_winding"]
        )
        assert report["cut_winding"] == report["trace"][-1]["winding"]


def _formation_control_graph(phases):
    """Prepare the frozen nonacute C5 control, without generating its support."""
    graph = nx.cycle_graph(5)
    configure(graph)
    graph.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, use_extended_dynamics=False)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(
            EPI=0.5, theta=float(phase), nu_f=1.0, delta_nfr=0.0, dEPI=0.0
        )
    default_compute_delta_nfr(graph)
    return graph


def _formation_analytic_state(time):
    q, eta = 2 * mp.pi / 5, mp.pi / 40
    d = 2 * mp.atan(mp.tan(q / 2) * mp.exp(-2 * _mp(time)))
    phase = tuple(
        _mp(time) + x
        for x in (
            q / 2 - d / 2,
            q / 2 + d / 2,
            4 * q + eta,
            5 * q / 2 - d / 2,
            5 * q / 2 + d / 2,
        )
    )
    source = (
        (q + d) / mp.pi,
        -3 * (q + d) / (4 * mp.pi) + eta / (2 * mp.pi),
        1 - eta / mp.pi,
        3 * (q + d) / (4 * mp.pi) + eta / (2 * mp.pi),
        -(q + d) / mp.pi,
    )
    return phase, source, d


@pytest.fixture(scope="module")
def formation_negative_control(record_testsuite_property):
    """One frozen trajectory; this fixture does not rerun either contact."""
    with pytest.MonkeyPatch.context() as patch, mp.workdps(80):
        patch.setattr(integrators, "np", None)
        initial, _, d_reference = _formation_analytic_state(F(0))
        graph = _formation_control_graph(initial)
        lifts = tuple(_mp(graph.nodes[i]["theta"]) for i in graph)
        records, trace = [], []
        for index in range(STEPS + 1):
            phases = tuple(graph.nodes[i]["theta"] for i in graph)
            admitted = tuple(
                resolve_u3_phase_neighbors(
                    graph.graph,
                    phases[i],
                    graph.neighbors(i),
                    phase_getter=lambda j: phases[j],
                    operator_code="UM",
                    require_compatible=False,
                ).neighbors
                for i in graph
            )
            state = {
                "time": index * H,
                "actual_time": F(graph.graph["_t"]),
                "epi": tuple(graph.nodes[i]["EPI"] for i in graph),
                "lifts": tuple(lifts),
                "phase": phases,
                "admitted": admitted,
                "winding": certify_phase_winding(graph, range(5)),
                "edge40_lift": lifts[0] - lifts[4],
                "d_reference": d_reference,
            }
            trace.append(state)
            if index == STEPS:
                break
            step = execute_joint_step(graph, dt=H, coupling_strength=1, time=index * H)
            increments = tuple(
                angle_diff(new, old)
                for old, new in zip(phases, step.phase_after, strict=True)
            )
            # Audit the same retained lift recurrence used for the branch path.
            rates = tuple(
                (
                    1 + sum(mp.sin(lifts[j] - lifts[i]) for j in row) / len(row)
                    if row
                    else mp.mpf(1)
                )
                for i, row in enumerate(admitted)
            )
            phase_defect = tuple(
                _mp(delta) - _mp(H) * rate
                for delta, rate in zip(increments, rates, strict=True)
            )
            euler = observe_support_transport_euler(
                step.before.snapshot, step.after_epi, H
            )
            records.append(
                {
                    "step": step,
                    "euler": euler,
                    "phase_defect": phase_defect,
                    "increments": increments,
                }
            )
            lifts = tuple(
                x + _mp(delta) for x, delta in zip(lifts, increments, strict=True)
            )
            d_reference -= 2 * _mp(H) * mp.sin(d_reference)
        crossing = tuple(
            (a["time"], b["time"])
            for a, b in zip(trace, trace[1:])
            if a["edge40_lift"] < -mp.pi < b["edge40_lift"]
        )
        mean_budget = sum(
            (
                H * abs(sum(r["step"].before.snapshot.rate) / 5 - F(1, 10))
                + abs(sum(r["euler"].state_defect) / 5)
                for r in records
            ),
            F(0),
        )
        _, _, endpoint_ode_gap = _formation_analytic_state(H * STEPS)
        metrics = {
            "h": str(H),
            "steps": STEPS,
            "duration": str(H * STEPS),
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "networkx": nx.__version__,
            "mpmath": mp.__version__,
            "provenance": [
                "numpy_default_pressure",
                "shared_u3_phase_proposal",
                "scalar_default_integrator_euler",
                "binary64",
                80,
            ],
            "arithmetic_scope": "estimates_not_validated_intervals_or_ODE_error_bounds",
            "endpoint_form_mean": math.fsum(trace[-1]["epi"]) / 5,
            "mean_forecast_error": float(
                abs(sum(map(F, trace[-1]["epi"])) / 5 - F(11, 20))
            ),
            "cumulative_mean_defect_budget": float(mean_budget),
            "endpoint_max_form_deviation": max(abs(x - 0.5) for x in trace[-1]["epi"]),
            "initial_winding": trace[0]["winding"].winding,
            "endpoint_winding": trace[-1]["winding"].winding,
            "fully_acute_retained_states": sum(
                state["winding"].u3_admissible for state in trace
            ),
            "endpoint_pair_gap_error_against_ODE": float(
                abs(trace[-1]["lifts"][1] - trace[-1]["lifts"][0] - endpoint_ode_gap)
            ),
            "pair_gap_discretization_bound": float(STEPS * H**2),
            "represented_path_crossing_brackets": [
                [str(t) for t in pair] for pair in crossing
            ],
            "analytic_ODE_crossing_time": float(mp.log(5) / 4),
            "max_phase_recurrence_defect_estimate": float(
                max(abs(x) for r in records for x in r["phase_defect"])
            ),
            "max_source_sum_defect_estimate": float(
                max(abs(sum(r["step"].before.phase_gradient) - 1) for r in records)
            ),
            "max_epi_euler_defect": float(
                max(abs(x) for r in records for x in r["euler"].state_defect)
            ),
        }
        record_testsuite_property(
            "formation_negative_control", json.dumps(metrics, sort_keys=True)
        )
    return {"trace": tuple(trace), "records": tuple(records), "crossing": crossing}


def test_formation_control_analytic_pressure_uses_all_edges_at_both_frozen_times():
    with mp.workdps(80):
        for time in (F(0), F(1, 2)):
            phase, source, _ = _formation_analytic_state(time)
            # These are two pressure evaluations, not additional trajectories.
            graph = _formation_control_graph(phase)
            observation = capture_non_epi_forcing(graph)
            assert tuple(map(len, observation.snapshot.support_neighbors)) == (2,) * 5
            assert tuple(map(float, observation.phase_gradient)) == pytest.approx(
                tuple(map(float, source)), rel=0, abs=TOL
            )
            assert float(sum(source)) == pytest.approx(1, rel=0, abs=TOL)
            assert float(sum(observation.snapshot.rate) / 5) == pytest.approx(
                0.1, rel=0, abs=TOL
            )


def test_formation_control_changes_winding_without_full_phase_admission(
    formation_negative_control,
):
    study = formation_negative_control
    with mp.workdps(80):
        for state in study["trace"]:
            assert state["actual_time"] == state["time"]
            assert state["admitted"] == ((1,), (0,), (), (4,), (3,))
            assert state["winding"].is_defined and not state["winding"].u3_admissible
            _, _, continuous_d = _formation_analytic_state(state["time"])
            # Independent scalar pair-Euler recurrence; the theorem's local
            # truncation <=h^2 gives the separate finite bound t*h.
            assert (
                abs(state["d_reference"] - continuous_d) <= _mp(state["time"] * H) + TOL
            )
            for left, right in ((0, 1), (3, 4)):
                assert (
                    abs(
                        state["lifts"][right]
                        - state["lifts"][left]
                        - state["d_reference"]
                    )
                    < TOL
                )
        assert study["trace"][0]["winding"].winding == 0
        assert study["trace"][-1]["winding"].winding == -1
        assert len(study["crossing"]) == 1
        # This is a crossing on the retained affine Euler path, not a claim
        # that a finite-step crossing equals the analytical ODE event time.
        assert study["crossing"][0][0] < study["crossing"][0][1] <= F(1, 2)
        for before, after, record in zip(
            study["trace"], study["trace"][1:], study["records"]
        ):
            assert abs(
                after["edge40_lift"]
                - before["edge40_lift"]
                - _mp(record["increments"][0])
                + _mp(record["increments"][4])
            ) < mp.mpf("1e-70")
            assert max(map(abs, record["increments"])) < 2 * float(H) + TOL
            assert max(map(abs, record["phase_defect"])) < TOL


def test_formation_control_predicts_mean_form_drift_without_clipping(
    formation_negative_control,
):
    study = formation_negative_control
    budget = F(0)
    for index, state in enumerate(study["trace"]):
        if index:
            record = study["records"][index - 1]
            before, euler = record["step"].before, record["euler"]
            assert tuple(map(len, before.snapshot.support_neighbors)) == (2,) * 5
            assert not any(before.stored_pressure_residual)
            assert abs(sum(before.phase_gradient) - 1) < TOL
            assert euler.identity_residual == 0
            assert min(euler.expected_epi) > 0 and max(euler.expected_epi) < 1
            # Mean diffusion cancels on this regular graph; source/assembly and
            # endpoint arithmetic deviations remain explicit, detached terms.
            rate_error = abs(sum(before.snapshot.rate) / 5 - F(1, 10))
            budget += H * rate_error + abs(sum(euler.state_defect) / 5)
        actual_mean = sum(map(F, state["epi"])) / 5
        assert abs(actual_mean - (F(1, 2) + state["time"] / 10)) <= budget
        assert max(abs(x - 0.5) for x in state["epi"]) <= float(state["time"] / 2) + TOL
        assert min(state["epi"]) > 0 and max(state["epi"]) < 1
    assert float(sum(map(F, study["trace"][-1]["epi"])) / 5) == pytest.approx(
        0.55, rel=0, abs=TOL
    )


@pytest.fixture(scope="module")
def contact_budget_control(record_testsuite_property):
    """Three frozen winding-one snapshots; proposals are never applied.

    The isolated rate is an algebraic comparison on the identical recipient
    state, not another graph execution or an independent recipient trajectory.
    """
    reports = []
    recipient_turns = (F(0), F(201, 800), F(403, 800), F(606, 800), F(1, 80))
    with mp.workdps(80):
        epsilon, q = mp.pi / 40, 2 * mp.pi / 5
        for gamma_turn in (F(1, 12), F(-1, 12), F(3, 8)):
            gamma = 2 * mp.pi * _mp(gamma_turn)
            bridge_admitted = abs(gamma) <= mp.pi / 2
            turns = recipient_turns + tuple(gamma_turn + F(k, 5) for k in range(5))
            ideal_phase = tuple(2 * mp.pi * _mp(turn) for turn in turns)
            graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
            graph.add_edge(0, 5)
            configure(graph)
            graph.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, use_extended_dynamics=False)
            # Retain declared configuration separately from pressure caches.
            configuration_keys = tuple(graph.graph)
            nx.set_edge_attributes(graph, 1.0, "weight")
            nx.set_edge_attributes(graph, 1.0, "length")
            for node, phase in zip(graph, ideal_phase, strict=True):
                graph.nodes[node].update(
                    EPI=0.5, theta=float(phase), nu_f=1.0, delta_nfr=0.0, dEPI=0.0
                )
            default_compute_delta_nfr(graph)
            frozen = (
                deepcopy(dict(graph.nodes(data=True))),
                deepcopy(dict(((i, j), data) for i, j, data in graph.edges(data=True))),
                deepcopy({key: graph.graph[key] for key in configuration_keys}),
            )
            observation = capture_non_epi_forcing(graph)
            phase = tuple(map(_mp, observation.phase))
            rows = observation.snapshot.support_neighbors
            admitted = tuple(
                resolve_u3_phase_neighbors(
                    graph.graph,
                    graph.nodes[i]["theta"],
                    graph.neighbors(i),
                    phase_getter=lambda j: graph.nodes[j]["theta"],
                    operator_code="UM",
                    require_compatible=False,
                ).neighbors
                for i in graph
            )
            independent_admitted = tuple(
                tuple(
                    j
                    for j in row
                    if abs(mp.arg(mp.exp(1j * (phase[j] - phase[i])))) <= mp.pi / 2
                )
                for i, row in enumerate(rows)
            )
            rates = tuple(
                (
                    sum(mp.sin(phase[j] - phase[i]) for j in row) / len(row)
                    if row
                    else mp.mpf(0)
                )
                for i, row in enumerate(independent_admitted)
            )
            ring_rows = tuple(
                tuple(j for j in row if j < 5) for row in independent_admitted[:5]
            )
            isolated_rates = tuple(
                (
                    sum(mp.sin(phase[j] - phase[i]) for j in row) / len(row)
                    if row
                    else mp.mpf(0)
                )
                for i, row in enumerate(ring_rows)
            )
            ideal_isolated = (mp.sin(epsilon), 0, 0, 0, -mp.sin(epsilon))
            ideal_rates = list(ideal_isolated) + [mp.mpf(0)] * 5
            if bridge_admitted:
                ideal_rates[0] = (mp.sin(epsilon) + mp.sin(gamma)) / 2
                ideal_rates[5] = -mp.sin(gamma) / 3
            ideal_u = (
                (mp.sin(gamma) - mp.sin(epsilon)) / 2 if bridge_admitted else mp.mpf(0)
            )
            ideal_drift = -2 * mp.sin(epsilon) - ideal_u
            gaps = tuple(
                mp.arg(mp.exp(1j * (phase[(i + 1) % 5] - phase[i]))) for i in range(5)
            )
            excess_edges = tuple(i for i, gap in enumerate(gaps) if gap > mp.pi / 2)
            sigma = int(gaps[4] > mp.pi / 2) - int(gaps[0] > mp.pi / 2)
            drift = sum(rates[(i + 1) % 5] - rates[i] for i in excess_edges)
            isolated_drift = sum(
                isolated_rates[(i + 1) % 5] - isolated_rates[i] for i in excess_edges
            )
            u = rates[0] - isolated_rates[0]
            proposal = propose_u3_gated_phase_step(
                graph,
                observation.snapshot.nodes,
                observation.phase,
                observation.snapshot.capacity,
                dt=float(H),
                coupling_strength=1.0,
            )
            increments = tuple(
                _mp(angle_diff(new, float(old)))
                for new, old in zip(proposal, observation.phase, strict=True)
            )
            phase_defect = tuple(
                increment - _mp(H) * (1 + rate)
                for increment, rate in zip(increments, rates, strict=True)
            )
            proposal_rates = tuple(increment / _mp(H) - 1 for increment in increments)
            proposal_drift = sum(
                proposal_rates[(i + 1) % 5] - proposal_rates[i] for i in excess_edges
            )
            resultants = tuple(
                sum(mp.exp(1j * (phase[j] - phase[i])) for j in row)
                for i, row in enumerate(rows)
            )
            represented_source = tuple(mp.arg(value) / mp.pi for value in resultants)
            ideal_source = (
                mp.atan2(
                    mp.sin(ideal_phase[1]) + mp.sin(ideal_phase[4]) + mp.sin(gamma),
                    mp.cos(ideal_phase[1]) + mp.cos(ideal_phase[4]) + mp.cos(gamma),
                )
                / mp.pi,
                -mp.mpf(799) / 800,
                -mp.mpf(799) / 800,
                -mp.mpf(799) / 800,
                -mp.mpf(107) / 400,
                mp.atan2(-mp.sin(gamma), 2 * mp.cos(q) + mp.cos(gamma)) / mp.pi,
                0,
                0,
                0,
                0,
            )
            source_defect = tuple(
                _mp(actual) - expected
                for actual, expected in zip(
                    observation.phase_gradient, represented_source, strict=True
                )
            )
            winding = tuple(certify_phase_winding(graph, ring) for ring in RINGS)
            after = (
                deepcopy(dict(graph.nodes(data=True))),
                deepcopy(dict(((i, j), data) for i, j, data in graph.edges(data=True))),
                deepcopy({key: graph.graph[key] for key in configuration_keys}),
            )
            report = {
                "gamma_turn": gamma_turn,
                "bridge_admitted": bridge_admitted,
                "observation": observation,
                "admitted": admitted,
                "independent_admitted": independent_admitted,
                "winding": winding,
                "frozen": frozen,
                "after": after,
                "rates": rates,
                "isolated_rates": isolated_rates,
                "ideal_rates": tuple(ideal_rates),
                "ideal_isolated": ideal_isolated,
                "u": u,
                "ideal_u": ideal_u,
                "drift": drift,
                "ideal_drift": ideal_drift,
                "isolated_drift": isolated_drift,
                "sigma": sigma,
                "excess_edges": excess_edges,
                "proposal_drift": proposal_drift,
                "phase_defect": phase_defect,
                "source_defect": source_defect,
                "represented_source": represented_source,
                "ideal_source": ideal_source,
                "min_resultant": min(map(abs, resultants)),
                "min_pressure_branch_margin": min(
                    mp.pi - abs(mp.arg(z)) for z in resultants
                ),
            }
            reports.append(report)
            record_testsuite_property(
                "contact_budget_control_gamma_" + str(gamma_turn),
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "numpy": version("numpy"),
                        "networkx": nx.__version__,
                        "mpmath": mp.__version__,
                        "phase_turns": tuple(map(str, turns)),
                        "h": str(H),
                        "scope": "three_static_snapshots_no_applied_proposals_or_trajectories",
                        "provenance": (
                            "numpy_default_pressure",
                            "shared_u3_phase_proposal",
                            "binary64",
                            80,
                        ),
                        "arithmetic_scope": "estimates_not_validated_intervals_or_ODE_error_bounds",
                        "admitted_neighbors": admitted,
                        "support_counts": tuple(map(len, rows)),
                        "ideal_coupling_rates": tuple(map(float, ideal_rates)),
                        "represented_coupling_rates": tuple(map(float, rates)),
                        "ideal_port_rate_difference": float(ideal_u),
                        "represented_port_rate_difference": float(u),
                        "ideal_excess_rate": float(ideal_drift),
                        "represented_excess_rate": float(drift),
                        "represented_isolated_excess_rate": float(isolated_drift),
                        "proposal_excess_rate": float(proposal_drift),
                        "min_resultant": float(report["min_resultant"]),
                        "min_pressure_branch_margin": float(
                            report["min_pressure_branch_margin"]
                        ),
                        "max_phase_materialization_error_estimate": float(
                            max(
                                abs(a - b)
                                for a, b in zip(phase, ideal_phase, strict=True)
                            )
                        ),
                        "max_phase_proposal_defect_estimate": float(
                            max(map(abs, phase_defect))
                        ),
                        "max_source_defect_estimate": float(
                            max(map(abs, source_defect))
                        ),
                    },
                    sort_keys=True,
                ),
            )
    return tuple(reports)


def test_contact_budget_control_retains_signed_port_balance_without_evolution(
    contact_budget_control,
):
    with mp.workdps(80):
        for report in contact_budget_control:
            assert report["frozen"] == report["after"]
            assert report["excess_edges"] == (0, 1, 2, 3) and report["sigma"] == -1
            recipient, donor = report["winding"]
            assert (
                recipient.is_defined
                and recipient.winding == 1
                and not recipient.u3_admissible
            )
            assert donor.is_defined and donor.winding == 1 and donor.u3_admissible
            # Detached transport rows are sorted; the live phase gate retains
            # neighbor insertion order. Compare membership without replacing
            # the independently retained order of either owner.
            assert (
                tuple(tuple(sorted(row)) for row in report["admitted"])
                == report["independent_admitted"]
            )
            expected = (
                (4, 5),
                (),
                (),
                (),
                (0,),
                (6, 9, 0),
                (5, 7),
                (6, 8),
                (7, 9),
                (5, 8),
            )
            if not report["bridge_admitted"]:
                expected = ((4,),) + expected[1:5] + ((6, 9),) + expected[6:]
            assert report["admitted"] == expected
            for key, ideal_key in (
                ("rates", "ideal_rates"),
                ("isolated_rates", "ideal_isolated"),
            ):
                assert (
                    max(
                        abs(a - b)
                        for a, b in zip(report[key], report[ideal_key], strict=True)
                    )
                    < TOL
                )
            assert abs(report["u"] - report["ideal_u"]) < TOL
            assert abs(report["drift"] - report["ideal_drift"]) < TOL
            assert abs(report["isolated_drift"] + 2 * mp.sin(mp.pi / 40)) < TOL
            assert abs(
                report["drift"]
                - report["isolated_drift"]
                - report["u"] * report["sigma"]
            ) < mp.mpf("1e-70")
            assert max(map(abs, report["phase_defect"])) < TOL
            # Four consecutive excess gaps telescope to the two endpoint rates.
            assert abs(report["proposal_drift"] - report["drift"]) <= 2 * max(
                map(abs, report["phase_defect"])
            ) / _mp(H)
        positive, negative, off = contact_budget_control
        assert positive["drift"] < positive["isolated_drift"] < 0
        assert negative["drift"] > 0
        assert off["u"] == 0 and off["drift"] == off["isolated_drift"]


def test_contact_budget_control_keeps_full_support_phase_source(contact_budget_control):
    with mp.workdps(80):
        for report in contact_budget_control:
            observation = report["observation"]
            rows = observation.snapshot.support_neighbors
            assert tuple(map(len, rows)) == (3, 2, 2, 2, 2, 3, 2, 2, 2, 2)
            assert rows[0] == (1, 4, 5) and rows[5] == (0, 6, 9)
            assert report["min_resultant"] > 0
            assert report["min_pressure_branch_margin"] > 0
            assert abs(report["min_resultant"] - 2 * mp.sin(3 * mp.pi / 800)) < TOL
            assert abs(report["min_pressure_branch_margin"] - mp.pi / 800) < TOL
            assert max(map(abs, report["source_defect"])) < TOL
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(
                        report["represented_source"],
                        report["ideal_source"],
                        strict=True,
                    )
                )
                < TOL
            )
            assert not any(observation.stored_pressure_residual)
            # Uniform form/capacity and explicit zero topology weight leave
            # half the full-support phase source in the refreshed nodal row.
            assert tuple(map(float, observation.snapshot.rate)) == pytest.approx(
                tuple(float(value / 2) for value in report["ideal_source"]),
                rel=0,
                abs=TOL,
            )
