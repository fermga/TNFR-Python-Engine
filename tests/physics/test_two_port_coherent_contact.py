"""Frozen two-port controls under supplied phase/form laws.

The entrance trajectory begins wound; the winding-zero formation budget is
only a nonmutating snapshot. Neither establishes autonomous pattern formation.
The distributed-contact gate barrier also has two nonmutating controls.
High-precision residual estimates are not validated continuous-ODE enclosures.
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

from tests.joint_phase_helpers import configure, execute_joint_step
from tnfr.dynamics import integrators
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.operators._phase_gate import resolve_u3_phase_neighbors
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_cycle_geometry import (
    derive_phase_chord_extension,
    derive_phase_cycle_geometry,
)
from tnfr.physics.support_transport import observe_support_transport_euler
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import angle_diff

H, STEPS = F(1, 2048), 32
RINGS = (tuple(range(5)), tuple(range(5, 10)))
CYCLES = RINGS + ((0, 1, 6, 5),)
TOL = 1e-12  # Arithmetic comparison, not a physical or gate threshold.


def _mp(value):
    if isinstance(value, F):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _read(graph, time):
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
    gaps = {
        (i, j): mp.arg(mp.exp(1j * (phase[j] - phase[i]))) for i, j in graph.edges()
    }
    independent_admitted = tuple(
        tuple(
            j
            for j in row
            if abs(mp.arg(mp.exp(1j * (phase[j] - phase[i])))) <= mp.pi / 2
        )
        for i, row in enumerate(rows)
    )
    rates = tuple(
        sum(mp.sin(phase[j] - phase[i]) for j in row) / len(row) if row else mp.mpf(0)
        for i, row in enumerate(independent_admitted)
    )
    resultants = tuple(
        sum(mp.exp(1j * (phase[j] - phase[i])) for j in row)
        for i, row in enumerate(rows)
    )
    source_defect = tuple(
        _mp(actual) - mp.arg(resultant) / mp.pi
        for actual, resultant in zip(
            observation.phase_gradient, resultants, strict=True
        )
    )
    return {
        "time": time,
        "observation": observation,
        "phase": phase,
        "admitted": admitted,
        "independent_admitted": independent_admitted,
        "rates": rates,
        "gaps": gaps,
        "resultants": resultants,
        "source_defect": source_defect,
        "min_branch_margin": min(mp.pi - abs(gap) for gap in gaps.values()),
        "min_pressure_branch_margin": min(mp.pi - abs(mp.arg(z)) for z in resultants),
        "winding": tuple(certify_phase_winding(graph, cycle) for cycle in CYCLES),
    }


@pytest.fixture(scope="module")
def two_port_contact(record_testsuite_property):
    """Apply only the predeclared 32 steps; retain both donor and recipient."""
    d_turn = F(1, 4) + F(1, 16000)
    r_turn = (1 - d_turn) / 4
    gamma_turn = (d_turn + F(1, 5)) / 2
    turns = (F(0),) + tuple(d_turn + k * r_turn for k in range(4))
    turns += tuple((gamma_turn - F(k, 5)) % 1 for k in range(5))
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(0, 5)
    one_port_geometry = derive_phase_cycle_geometry(graph)
    graph.add_edge(1, 6)
    configure(graph)
    graph.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, use_extended_dynamics=False)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    geometry = derive_phase_cycle_geometry(graph)
    extension = derive_phase_chord_extension(one_port_geometry, geometry)
    # The existing acute reconstruction deliberately rejects the initial
    # excluded edge. Joint realizability here follows from exact nodal turns.
    edge_turns = tuple(
        (turns[j] - turns[i] + F(1, 2)) % 1 - F(1, 2) for i, j in geometry.edges
    )
    periods = tuple(
        sum(
            coefficient * turn
            for coefficient, turn in zip(row, edge_turns, strict=True)
        )
        for row in geometry.cycle_rows
    )
    with pytest.MonkeyPatch.context() as patch, mp.workdps(80):
        # Retain NumPy pressure, replacing only the EPI integration path.
        patch.setattr(integrators, "np", None)
        for node, turn in zip(graph, turns, strict=True):
            graph.nodes[node].update(
                EPI=0.5,
                theta=float(2 * mp.pi * _mp(turn)),
                nu_f=1.0,
                delta_nfr=0.0,
                dEPI=0.0,
            )
        default_compute_delta_nfr(graph)
        trace = [_read(graph, F(0))]
        records = []
        for index in range(STEPS):
            before = trace[-1]
            step = execute_joint_step(graph, dt=H, coupling_strength=1, time=index * H)
            increments = tuple(
                _mp(angle_diff(new, float(old)))
                for new, old in zip(step.phase_after, step.before.phase, strict=True)
            )
            defects = tuple(
                increment - _mp(H) * (1 + rate)
                for increment, rate in zip(increments, before["rates"], strict=True)
            )
            records.append(
                {
                    "step": step,
                    "increments": increments,
                    "phase_defect": defects,
                    "euler": observe_support_transport_euler(
                        step.before.snapshot, step.after_epi, H
                    ),
                    "actual_time": F(graph.graph["_t"]),
                }
            )
            trace.append(_read(graph, (index + 1) * H))
        first_admitted = next(
            (index for index, state in enumerate(trace) if 1 in state["admitted"][0]),
            None,
        )
        metrics = {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "networkx": nx.__version__,
            "mpmath": mp.__version__,
            "phase_turns": tuple(map(str, turns)),
            "h": str(H),
            "steps": STEPS,
            "duration": str(STEPS * H),
            "support_edges": tuple(graph.edges()),
            "conductance_and_length": 1,
            "capacities": [1] * 10,
            "initial_form": [0.5] * 10,
            "normalized_pressure_weights": {
                "epi": 0.5,
                "phase": 0.5,
                "vf": 0,
                "topo": 0,
            },
            "phase_coupling": 1,
            "u3_gate": "pi/2",
            "Gamma": "none",
            "named_events": [],
            "seed": None,
            "provenance": (
                "numpy_default_pressure",
                "shared_u3_phase_proposal",
                "scalar_default_integrator_euler",
                "binary64",
                80,
            ),
            "scope": "prepared_wound_recipient_acute_entry_not_winding_zero_formation",
            "arithmetic_scope": "estimates_not_validated_intervals_or_ODE_error_bounds",
            "fundamental_cycle_periods": tuple(map(str, periods)),
            "first_admitted_endpoint": first_admitted,
            "retained_crossing_bracket": (
                [str((first_admitted - 1) * H), str(first_admitted * H)]
                if first_admitted is not None and first_admitted > 0
                else None
            ),
            "initial_excess_rate": float(trace[0]["rates"][1] - trace[0]["rates"][0]),
            "endpoint_acute_margin": float(
                min(mp.pi / 2 - abs(gap) for gap in trace[-1]["gaps"].values())
            ),
            "endpoint_max_form_deviation": max(
                abs(float(x) - 0.5) for x in trace[-1]["observation"].snapshot.epi
            ),
            "endpoint_max_donor_deformation": float(
                max(
                    abs(trace[-1]["phase"][i] - trace[0]["phase"][i] - _mp(STEPS * H))
                    for i in range(5, 10)
                )
            ),
            "min_phase_branch_margin": float(
                min(s["min_branch_margin"] for s in trace)
            ),
            "min_pressure_branch_margin": float(
                min(s["min_pressure_branch_margin"] for s in trace)
            ),
            "min_resultant": float(min(abs(z) for s in trace for z in s["resultants"])),
            "max_phase_proposal_defect_estimate": float(
                max(abs(x) for r in records for x in r["phase_defect"])
            ),
            "max_source_defect_estimate": float(
                max(abs(x) for s in trace for x in s["source_defect"])
            ),
            "max_nodal_state_defect": float(
                max(abs(x) for r in records for x in r["euler"].state_defect)
            ),
            "retained_gates": tuple(s["admitted"] for s in trace),
        }
        record_testsuite_property(
            "two_port_contact", json.dumps(metrics, sort_keys=True)
        )
    return {
        "turns": turns,
        "geometry": geometry,
        "extension": extension,
        "edge_turns": edge_turns,
        "periods": periods,
        "trace": tuple(trace),
        "records": tuple(records),
        "first_admitted": first_admitted,
        "metrics": metrics,
    }


def test_two_port_geometry_and_entrance_preserve_prepared_winding(two_port_contact):
    report = two_port_contact
    geometry, turns = report["geometry"], report["turns"]
    assert geometry.cycle_rank == 3 and geometry.bridge_edge_indices == ()
    extension = report["extension"]
    assert extension.before.cycle_rank == 2
    assert geometry.edges[extension.added_edge_index] == (1, 6)
    assert set(extension.created_cycle) == {0, 1, 5, 6}
    assert all(period.denominator == 1 for period in report["periods"])
    for cycle, expected in zip(CYCLES, (1, -1, 0), strict=True):
        exact_period = sum(
            (turns[cycle[(i + 1) % len(cycle)]] - turns[node] + F(1, 2)) % 1 - F(1, 2)
            for i, node in enumerate(cycle)
        )
        assert exact_period == expected
    trace = report["trace"]
    assert tuple(
        edge for edge, gap in trace[0]["gaps"].items() if abs(gap) > math.pi / 2
    ) == ((0, 1),)
    crossing = report["first_admitted"]
    assert crossing is not None and 0 < crossing < len(trace)
    for index, state in enumerate(trace):
        assert (1 in state["admitted"][0]) == (index >= crossing)
        assert (0 in state["admitted"][1]) == (index >= crossing)
        assert all(
            abs(gap) < math.pi / 2
            for edge, gap in state["gaps"].items()
            if edge != (0, 1)
        )
        assert state["min_branch_margin"] > 0
        for certificate, expected in zip(state["winding"], (1, -1, 0), strict=True):
            assert certificate.is_defined and certificate.winding == expected
        assert state["winding"][1].u3_admissible
        assert state["winding"][0].u3_admissible == (index >= crossing)
        assert (
            tuple(tuple(sorted(row)) for row in state["admitted"])
            == state["independent_admitted"]
        )
    assert all(abs(gap) < math.pi / 2 for gap in trace[-1]["gaps"].values())


def test_two_port_phase_execution_matches_actual_donor_coupling(two_port_contact):
    report = two_port_contact
    with mp.workdps(80):
        d = mp.pi / 2 + mp.pi / 8000
        r, gamma = (2 * mp.pi - d) / 4, (d + 2 * mp.pi / 5) / 2
        f0, f5 = (mp.sin(gamma) - mp.sin(r)) / 2, -mp.sin(gamma) / 3
        predicted = (f0, -f0, 0, 0, 0, f5, -f5, 0, 0, 0)
        initial = report["trace"][0]
        assert (
            max(abs(a - b) for a, b in zip(initial["rates"], predicted, strict=True))
            < TOL
        )
        assert initial["rates"][1] - initial["rates"][0] < 0
        for record, state in zip(report["records"], report["trace"][:-1], strict=True):
            assert record["step"].before.phase == state["observation"].phase
            assert record["actual_time"] == state["time"] + H
            assert max(map(abs, record["phase_defect"])) < TOL
            assert all(
                0 <= increment <= 2 * _mp(H) + TOL for increment in record["increments"]
            )
        assert report["metrics"]["endpoint_max_donor_deformation"] > 1e-3


def test_two_port_form_uses_full_support_without_clipping(two_port_contact):
    report = two_port_contact
    with mp.workdps(80):
        initial = report["trace"][0]
        phase = initial["phase"]
        gated_resultant = sum(
            mp.exp(1j * (phase[j] - phase[0])) for j in initial["admitted"][0]
        )
        assert abs(mp.arg(gated_resultant) - mp.arg(initial["resultants"][0])) > mp.mpf(
            "0.1"
        )
        for state in report["trace"]:
            observation = state["observation"]
            assert tuple(map(len, observation.snapshot.support_neighbors)) == (
                3,
                3,
                2,
                2,
                2,
                3,
                3,
                2,
                2,
                2,
            )
            assert dict(observation.normalized_weights) == {
                "phase": F(1, 2),
                "epi": F(1, 2),
                "vf": 0,
                "topo": 0,
            }
            assert all(mp.re(z) > mp.mpf(13) / 32 for z in state["resultants"])
            assert state["min_pressure_branch_margin"] > mp.pi / 2
            assert max(map(abs, state["source_defect"])) < TOL
            assert not any(observation.stored_pressure_residual)
            assert max(map(abs, observation.kernel_pressure_defect)) < TOL
            # A convex Euler diffusion row and |phase source| <= 1 give t/2.
            assert max(abs(x - F(1, 2)) for x in observation.snapshot.epi) <= state[
                "time"
            ] / 2 + F(TOL)
            assert observation.snapshot.capacity == (F(1),) * 10
        for record in report["records"]:
            euler = record["euler"]
            assert euler.identity_residual == 0
            assert max(map(abs, euler.state_defect)) < TOL
            assert min(euler.expected_epi) > 0 and max(euler.expected_epi) < 1


@pytest.fixture(scope="module")
def formation_budget_snapshot(record_testsuite_property):
    """Inspect one predeclared winding-zero state; never apply its proposal."""
    d_turn = F(1, 2) + F(1, 16000)
    r_turn = (1 - d_turn) / 4
    gamma_turn = (d_turn - F(1, 5)) / 2
    turns = (F(0),) + tuple(d_turn + k * r_turn for k in range(4))
    turns += tuple(gamma_turn + F(k, 5) for k in range(5))
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    configure(graph)
    graph.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, use_extended_dynamics=False)
    configuration_keys = tuple(graph.graph)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    geometry = derive_phase_cycle_geometry(graph)
    edge_turns = tuple(
        (turns[j] - turns[i] + F(1, 2)) % 1 - F(1, 2) for i, j in geometry.edges
    )
    periods = tuple(
        sum(
            coefficient * turn
            for coefficient, turn in zip(row, edge_turns, strict=True)
        )
        for row in geometry.cycle_rows
    )

    def retained_state():
        return (
            deepcopy(dict(graph.nodes(data=True))),
            deepcopy(dict(((i, j), data) for i, j, data in graph.edges(data=True))),
            deepcopy({key: graph.graph[key] for key in configuration_keys}),
        )

    with mp.workdps(80):
        ideal_phase = tuple(2 * mp.pi * _mp(turn) for turn in turns)
        for node, phase in zip(graph, ideal_phase, strict=True):
            graph.nodes[node].update(
                EPI=0.5, theta=float(phase), nu_f=1.0, delta_nfr=0.0, dEPI=0.0
            )
        default_compute_delta_nfr(graph)
        frozen = retained_state()
        state = _read(graph, F(0))
        observation = state["observation"]
        counts = tuple(map(len, state["independent_admitted"]))
        rates = state["rates"]
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
        proposal_rates = tuple(increment / _mp(H) - 1 for increment in increments)
        rate_defects = tuple(
            actual - expected
            for actual, expected in zip(proposal_rates, rates, strict=True)
        )
        potential = sum(1 - max(mp.cos(gap), 0) for gap in state["gaps"].values())
        full_potential = sum(1 - mp.cos(gap) for gap in state["gaps"].values())
        admitted_edges = tuple(
            (i, j) for i, j in graph.edges() if j in state["independent_admitted"][i]
        )
        potential_rate = sum(
            mp.sin(state["gaps"][i, j]) * (rates[j] - rates[i])
            for i, j in admitted_edges
        )
        proposal_potential_rate = sum(
            mp.sin(state["gaps"][i, j]) * (proposal_rates[j] - proposal_rates[i])
            for i, j in admitted_edges
        )
        dissipation = -sum(
            count * rate**2 for count, rate in zip(counts, rates, strict=True)
        )
        q, d = 2 * mp.pi / 5, 2 * mp.pi * _mp(d_turn)
        r, gamma = (2 * mp.pi - d) / 4, (d - q) / 2
        threshold = 10 * (1 - mp.cos(q))
        ideal_potential = (
            1 + 4 * (1 - mp.cos(r)) + 5 * (1 - mp.cos(q)) + 2 * (1 - mp.cos(gamma))
        )
        f0, f5 = (mp.sin(gamma) - mp.sin(r)) / 2, -mp.sin(gamma) / 3
        ideal_rates = (f0, -f0, 0, 0, 0, f5, -f5, 0, 0, 0)
        recipient_source = (
            mp.arg(mp.exp(1j * d) + mp.exp(-1j * r) + mp.exp(1j * gamma)) / mp.pi
        )
        donor_source = mp.arg(2 * mp.cos(q) + mp.exp(-1j * gamma)) / mp.pi
        ideal_source = (
            recipient_source,
            -recipient_source,
            0,
            0,
            0,
            donor_source,
            -donor_source,
            0,
            0,
            0,
        )
        ideal_gap_rate = mp.sin(r) - mp.sin(gamma)
        metrics = {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "networkx": nx.__version__,
            "mpmath": mp.__version__,
            "phase_turns": tuple(map(str, turns)),
            "h": str(H),
            "applied_steps": 0,
            "scope": "one_static_winding_zero_formation_budget_no_applied_proposal",
            "provenance": (
                "numpy_default_pressure",
                "shared_u3_phase_proposal",
                "binary64",
                80,
            ),
            "arithmetic_scope": "estimates_not_validated_intervals_or_ODE_certificates",
            "support_edges": tuple(graph.edges()),
            "conductance_and_length": 1,
            "capacities": [1] * 10,
            "initial_form": [0.5] * 10,
            "normalized_pressure_weights": {
                "epi": 0.5,
                "phase": 0.5,
                "vf": 0,
                "topo": 0,
            },
            "phase_coupling": 1,
            "u3_gate": "pi/2",
            "Gamma": "none",
            "named_events": [],
            "seed": None,
            "fundamental_cycle_periods": tuple(map(str, periods)),
            "oriented_cycle_windings": (0, 1, -1),
            "admitted_neighbors": state["admitted"],
            "admitted_counts": counts,
            "initial_gap_rate": float(rates[1] - rates[0]),
            "gated_potential": float(potential),
            "full_cosine_potential": float(full_potential),
            "two_wound_rings_budget_threshold": float(threshold),
            "budget_deficit": float(threshold - potential),
            "gradient_dissipation": float(dissipation),
            "potential_rate": float(potential_rate),
            "proposal_potential_rate": float(proposal_potential_rate),
            "max_proposal_rate_defect_estimate": float(max(map(abs, rate_defects))),
            "max_source_defect_estimate": float(max(map(abs, state["source_defect"]))),
            "min_resultant": float(min(map(abs, state["resultants"]))),
            "min_pressure_branch_margin": float(state["min_pressure_branch_margin"]),
        }
        record_testsuite_property(
            "formation_budget_snapshot", json.dumps(metrics, sort_keys=True)
        )
    return {
        "turns": turns,
        "geometry": geometry,
        "periods": periods,
        "state": state,
        "frozen": frozen,
        "after": retained_state(),
        "counts": counts,
        "potential": potential,
        "full_potential": full_potential,
        "ideal_potential": ideal_potential,
        "threshold": threshold,
        "potential_rate": potential_rate,
        "proposal_potential_rate": proposal_potential_rate,
        "dissipation": dissipation,
        "rate_defects": rate_defects,
        "ideal_rates": ideal_rates,
        "ideal_source": ideal_source,
        "ideal_gap_rate": ideal_gap_rate,
        "metrics": metrics,
    }


def test_formation_budget_snapshot_is_realizable_but_below_formation_budget(
    formation_budget_snapshot,
):
    report = formation_budget_snapshot
    turns, state = report["turns"], report["state"]
    assert report["geometry"].cycle_rank == 3
    assert all(period.denominator == 1 for period in report["periods"])
    for cycle, expected, certificate in zip(
        CYCLES, (0, 1, -1), state["winding"], strict=True
    ):
        period = sum(
            (turns[cycle[(i + 1) % len(cycle)]] - turns[node] + F(1, 2)) % 1 - F(1, 2)
            for i, node in enumerate(cycle)
        )
        assert period == expected
        assert certificate.is_defined and certificate.winding == expected
    assert not state["winding"][0].u3_admissible
    assert state["winding"][1].u3_admissible
    with mp.workdps(80):
        assert abs(report["potential"] - report["ideal_potential"]) < TOL
        assert report["potential"] < report["threshold"] < report["full_potential"]
        assert state["rates"][1] - state["rates"][0] < 0
        assert (
            abs(state["rates"][1] - state["rates"][0] - report["ideal_gap_rate"]) < TOL
        )


def test_formation_budget_snapshot_uses_actual_gradient_and_full_pressure_without_evolution(
    formation_budget_snapshot,
):
    report = formation_budget_snapshot
    state, observation = report["state"], report["state"]["observation"]
    assert report["frozen"] == report["after"]
    assert report["counts"] == (2, 2, 2, 2, 2, 3, 3, 2, 2, 2)
    assert (
        tuple(tuple(sorted(row)) for row in state["admitted"])
        == state["independent_admitted"]
    )
    assert tuple(map(len, observation.snapshot.support_neighbors)) == (
        3,
        3,
        2,
        2,
        2,
        3,
        3,
        2,
        2,
        2,
    )
    assert observation.snapshot.epi == (F(1, 2),) * 10
    assert observation.snapshot.capacity == (F(1),) * 10
    with mp.workdps(80):
        assert (
            max(
                abs(actual - expected)
                for actual, expected in zip(
                    state["rates"], report["ideal_rates"], strict=True
                )
            )
            < TOL
        )
        assert report["dissipation"] < 0
        assert abs(report["potential_rate"] - report["dissipation"]) < mp.mpf("1e-70")
        assert max(map(abs, report["rate_defects"])) < TOL
        assert abs(
            report["proposal_potential_rate"] - report["potential_rate"]
        ) <= 2 * 11 * max(map(abs, report["rate_defects"]))
        assert min(map(abs, state["resultants"])) > 0
        assert state["min_pressure_branch_margin"] > 0
        assert max(map(abs, state["source_defect"])) < TOL
        assert tuple(map(float, observation.phase_gradient)) == pytest.approx(
            tuple(map(float, report["ideal_source"])), rel=0, abs=TOL
        )
        gated_resultant = sum(
            mp.exp(1j * (state["phase"][j] - state["phase"][0]))
            for j in state["admitted"][0]
        )
        assert abs(mp.arg(gated_resultant) - mp.arg(state["resultants"][0])) > mp.mpf(
            "0.1"
        )
        assert not any(observation.stored_pressure_residual)
        assert max(map(abs, observation.kernel_pressure_defect)) < TOL
        assert tuple(map(float, observation.snapshot.rate)) == pytest.approx(
            tuple(float(value / 2) for value in report["ideal_source"]), rel=0, abs=TOL
        )


@pytest.fixture(scope="module")
def gate_barrier_control(record_testsuite_property):
    """Read two frozen geometries; calculate but never apply either proposal."""
    cases = {
        "high_budget_branch": (F(1, 2) + F(1, 16000), F(1, 16), F(1, 40)),
        "barrier": (F(3, 8), F(1, 12), F(5, 32)),
    }
    reports = {}
    with mp.workdps(80):
        for name, (d_turn, e_turn, a_turn) in cases.items():
            s_turn = (1 - d_turn - 2 * a_turn) / 2
            u_turn, gamma_turn = (1 - e_turn) / 4, (d_turn - e_turn) / 2
            recipient_gaps = (d_turn, a_turn, s_turn, s_turn, a_turn)
            turns = (F(0),) + tuple(sum(recipient_gaps[:i]) for i in range(1, 5))
            turns += (gamma_turn,) + tuple(
                gamma_turn + e_turn + k * u_turn for k in range(4)
            )
            graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
            graph.add_edges_from(((0, 5), (1, 6)))
            configure(graph)
            graph.graph.update(EPI_MIN=0.0, EPI_MAX=1.0, use_extended_dynamics=False)
            configuration_keys = tuple(graph.graph)
            nx.set_edge_attributes(graph, 1.0, "weight")
            nx.set_edge_attributes(graph, 1.0, "length")
            geometry = derive_phase_cycle_geometry(graph)
            edge_turns = tuple(
                (turns[j] - turns[i] + F(1, 2)) % 1 - F(1, 2) for i, j in geometry.edges
            )
            periods = tuple(
                sum(c * turn for c, turn in zip(row, edge_turns, strict=True))
                for row in geometry.cycle_rows
            )

            def retained_state():
                return (
                    deepcopy(dict(graph.nodes(data=True))),
                    deepcopy(
                        dict(((i, j), data) for i, j, data in graph.edges(data=True))
                    ),
                    deepcopy({key: graph.graph[key] for key in configuration_keys}),
                )

            ideal_phase = tuple(2 * mp.pi * _mp(turn) for turn in turns)
            for node, phase in zip(graph, ideal_phase, strict=True):
                graph.nodes[node].update(
                    EPI=0.5, theta=float(phase), nu_f=1.0, delta_nfr=0.0, dEPI=0.0
                )
            default_compute_delta_nfr(graph)
            frozen = retained_state()
            state = _read(graph, F(0))
            observation = state["observation"]
            proposal = propose_u3_gated_phase_step(
                graph,
                observation.snapshot.nodes,
                observation.phase,
                observation.snapshot.capacity,
                dt=float(H),
                coupling_strength=1.0,
            )
            proposal_rates = tuple(
                _mp(angle_diff(new, float(old))) / _mp(H) - 1
                for new, old in zip(proposal, observation.phase, strict=True)
            )
            rate_defects = tuple(
                actual - expected
                for actual, expected in zip(proposal_rates, state["rates"], strict=True)
            )
            d, e, a, s, u, gamma = (
                2 * mp.pi * _mp(turn)
                for turn in (d_turn, e_turn, a_turn, s_turn, u_turn, gamma_turn)
            )
            f0 = (mp.sin(gamma) - mp.sin(a)) / 2
            f2 = (mp.sin(s) - mp.sin(a)) / 2
            f5 = (mp.sin(e) - mp.sin(u) - mp.sin(gamma)) / 3
            ideal_rates = (f0, -f0, f2, 0, -f2, f5, -f5, 0, 0, 0)
            ideal_source = tuple(
                mp.arg(sum(mp.exp(1j * (ideal_phase[j] - ideal_phase[i])) for j in row))
                / mp.pi
                for i, row in enumerate(observation.snapshot.support_neighbors)
            )
            potential = sum(1 - max(mp.cos(gap), 0) for gap in state["gaps"].values())
            ideal_potential = (
                1
                + 2 * (1 - mp.cos(a))
                + 2 * (1 - mp.cos(s))
                + (1 - mp.cos(e))
                + 4 * (1 - mp.cos(u))
                + 2 * (1 - mp.cos(gamma))
            )
            threshold = 10 * (1 - mp.cos(2 * mp.pi / 5))
            gap_rates = (
                proposal_rates[1] - proposal_rates[0],
                proposal_rates[6] - proposal_rates[5],
            )
            z_rate = 2 * gap_rates[0] + 3 * gap_rates[1]
            budget_rate = 2 * mp.sin(a) + 2 * mp.sin(u) - 2 * mp.sin(e)
            # Subtracting binary64 proposal endpoints and dividing by H loses
            # more accuracy than the independently evaluated sine rates.
            proposal_rate_allowance = 2 * math.ulp(2 * math.pi) / float(H)
            metrics = {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "networkx": nx.__version__,
                "mpmath": mp.__version__,
                "phase_turns": tuple(map(str, turns)),
                "lifted_d_turns": str(d_turn),
                "lifted_e_turns": str(e_turn),
                "h": str(H),
                "applied_steps": 0,
                "scope": "two_static_budget_and_gate_barrier_controls_no_evolution",
                "provenance": (
                    "numpy_default_pressure",
                    "shared_u3_phase_proposal",
                    "binary64",
                    80,
                ),
                "arithmetic_scope": "estimates_not_validated_intervals_or_ODE_certificates",
                "support_edges": tuple(graph.edges()),
                "conductance_and_length": 1,
                "capacities": [1] * 10,
                "initial_form": [0.5] * 10,
                "normalized_pressure_weights": dict(
                    (key, float(value)) for key, value in observation.normalized_weights
                ),
                "phase_coupling": 1,
                "u3_gate": "pi/2",
                "Gamma": "none",
                "named_events": [],
                "seed": None,
                "fundamental_cycle_periods": tuple(map(str, periods)),
                "oriented_cycle_windings": tuple(c.winding for c in state["winding"]),
                "admitted_neighbors": state["admitted"],
                "admitted_counts": tuple(map(len, state["independent_admitted"])),
                "gated_potential": float(potential),
                "two_wound_rings_budget_threshold": float(threshold),
                "budget_surplus": float(potential - threshold),
                "proposal_lifted_d_rate": float(gap_rates[0]),
                "proposal_lifted_e_rate": float(gap_rates[1]),
                "proposal_z_rate": float(z_rate),
                "independent_z_rate": float(budget_rate),
                "barrier_lower_bound": float(2 - mp.sqrt(3)),
                "max_proposal_rate_defect_estimate": float(max(map(abs, rate_defects))),
                "proposal_rate_allowance": proposal_rate_allowance,
                "max_source_defect_estimate": float(
                    max(map(abs, state["source_defect"]))
                ),
                "min_resultant_real_part": float(
                    min(mp.re(z) for z in state["resultants"])
                ),
                "min_phase_branch_margin": float(state["min_branch_margin"]),
                "min_pressure_branch_margin": float(
                    state["min_pressure_branch_margin"]
                ),
            }
            record_testsuite_property(
                f"gate_barrier_control_{name}", json.dumps(metrics, sort_keys=True)
            )
            reports[name] = {
                "turns": turns,
                "d_turn": d_turn,
                "e_turn": e_turn,
                "geometry": geometry,
                "periods": periods,
                "state": state,
                "frozen": frozen,
                "after": retained_state(),
                "ideal_rates": ideal_rates,
                "ideal_source": ideal_source,
                "rate_defects": rate_defects,
                "proposal_rate_allowance": proposal_rate_allowance,
                "potential": potential,
                "ideal_potential": ideal_potential,
                "threshold": threshold,
                "gap_rates": gap_rates,
                "z_rate": z_rate,
                "budget_rate": budget_rate,
                "metrics": metrics,
            }
    return reports


def test_gate_barrier_control_separates_budget_from_actual_entrance(
    gate_barrier_control,
):
    with mp.workdps(80):
        for name, expected in (
            ("high_budget_branch", (0, 1, -1)),
            ("barrier", (1, 1, 0)),
        ):
            report = gate_barrier_control[name]
            turns, state = report["turns"], report["state"]
            assert report["geometry"].cycle_rank == 3
            assert all(period.denominator == 1 for period in report["periods"])
            for cycle, winding, certificate in zip(
                CYCLES, expected, state["winding"], strict=True
            ):
                period = sum(
                    (turns[cycle[(i + 1) % len(cycle)]] - turns[node] + F(1, 2)) % 1
                    - F(1, 2)
                    for i, node in enumerate(cycle)
                )
                assert period == winding
                assert certificate.is_defined and certificate.winding == winding
            assert not state["winding"][0].u3_admissible
            assert state["winding"][1].u3_admissible
            assert tuple(
                edge for edge, gap in state["gaps"].items() if abs(gap) > mp.pi / 2
            ) == ((0, 1),)
            assert state["min_branch_margin"] > 0
            assert abs(report["potential"] - report["ideal_potential"]) < TOL
            assert (
                abs(report["z_rate"] - report["budget_rate"])
                < 10 * report["proposal_rate_allowance"]
            )
        high = gate_barrier_control["high_budget_branch"]
        assert high["potential"] > high["threshold"]
        assert high["gap_rates"][0] < 0
        barrier = gate_barrier_control["barrier"]
        assert 2 * barrier["d_turn"] + 3 * barrier["e_turn"] == 1
        assert barrier["z_rate"] > 2 - mp.sqrt(3)


def test_gate_barrier_control_uses_actual_donor_and_full_pressure_without_evolution(
    gate_barrier_control,
):
    with mp.workdps(80):
        for report in gate_barrier_control.values():
            state, observation = report["state"], report["state"]["observation"]
            assert report["frozen"] == report["after"]
            assert tuple(map(len, state["independent_admitted"])) == (
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                2,
                2,
                2,
            )
            assert (
                tuple(tuple(sorted(row)) for row in state["admitted"])
                == state["independent_admitted"]
            )
            assert tuple(map(len, observation.snapshot.support_neighbors)) == (
                3,
                3,
                2,
                2,
                2,
                3,
                3,
                2,
                2,
                2,
            )
            assert observation.snapshot.epi == (F(1, 2),) * 10
            assert observation.snapshot.capacity == (F(1),) * 10
            assert dict(observation.normalized_weights) == {
                "phase": F(1, 2),
                "epi": F(1, 2),
                "vf": 0,
                "topo": 0,
            }
            assert (
                max(
                    abs(actual - expected)
                    for actual, expected in zip(
                        state["rates"], report["ideal_rates"], strict=True
                    )
                )
                < TOL
            )
            assert (
                max(map(abs, report["rate_defects"]))
                < report["proposal_rate_allowance"]
            )
            assert report["gap_rates"][1] != 0
            assert min(mp.re(z) for z in state["resultants"]) > 0
            assert state["min_pressure_branch_margin"] > mp.pi / 2
            assert max(map(abs, state["source_defect"])) < TOL
            assert tuple(map(float, observation.phase_gradient)) == pytest.approx(
                tuple(map(float, report["ideal_source"])), rel=0, abs=TOL
            )
            gated_resultant = sum(
                mp.exp(1j * (state["phase"][j] - state["phase"][0]))
                for j in state["admitted"][0]
            )
            assert abs(
                mp.arg(gated_resultant) - mp.arg(state["resultants"][0])
            ) > mp.mpf("0.1")
            assert not any(observation.stored_pressure_residual)
            assert max(map(abs, observation.kernel_pressure_defect)) < TOL
            assert tuple(map(float, observation.snapshot.rate)) == pytest.approx(
                tuple(float(value / 2) for value in report["ideal_source"]),
                rel=0,
                abs=TOL,
            )
