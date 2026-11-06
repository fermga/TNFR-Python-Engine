"""Prospective G3 P3 phase-source comparison through the shared nodal engine.

This is one fixed software experiment, not a configurable pressure law. The
second source replaces only the phase channel with the existing J_phi/pi
read-out. Held phase, capacity and support are supplied premises. Agreement
with the analytic response neither selects a physical law nor demonstrates
autonomous pattern formation. Run with ``python -m tnfr.research.phase_form_response
--output artifacts/research/g3_phase_form_response`` from the checkout.
"""

from __future__ import annotations

import argparse
import math
import platform
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np

from ..alias import get_attr, set_dnfr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..dynamics.dnfr import default_compute_delta_nfr
from ..dynamics.integrators import update_epi_via_nodal_equation
from ..physics.extended import compute_phase_current
from ..physics.forcing_realization import capture_non_epi_forcing
from ..utils.io import json_dumps
from .artifact_io import sha256_file as _hash
from .artifact_io import write_json_once
from .claims import ClaimStatus
from .core_manifests import CoreExperimentManifest, current_git_source_provenance
from .evidence_sidecar import EvidenceSidecar

_MODELS = ("argument", "sine_current")
_SOURCE_PATHS = ("src/tnfr", "pyproject.toml")


def _source(declaration, model, spread):
    """Independent regular-chart P3 source; no engine field evaluation."""
    mean = declaration["mean_gap"]
    weights = declaration["normalized_weights"]
    if model == "argument":
        phase = ((spread - mean) / math.pi, mean / math.pi, -(mean + spread) / math.pi)
    elif model == "sine_current":
        phase = (
            -math.sin(mean - spread) / math.pi,
            math.sin(mean) * math.cos(spread) / math.pi,
            -math.sin(mean + spread) / math.pi,
        )
    else:
        raise ValueError("unknown phase source")
    return tuple(
        weights["phase"] * value + weights["topo"] * topology
        for value, topology in zip(phase, (1, -1, 1), strict=True)
    )


def _prediction(declaration, model, spread, step, *, continuous):
    """Independent eigenmode solution: L_rw*(-1,1,-1)=2v; L_rw*(1,0,-1)=w."""
    source = _source(declaration, model, spread)
    even, odd = source[1], (source[0] - source[2]) / 2
    epi_weight = declaration["normalized_weights"]["epi"]
    rate = declaration["capacity"] * epi_weight
    dt = declaration["dt"]

    def response(eigenvalue):
        exponent = eigenvalue * rate
        decay = (
            -math.expm1(-exponent * step * dt)
            if continuous
            else 1 - (1 - exponent * dt) ** step
        )
        return decay / (eigenvalue * epi_weight)

    even *= response(2)
    odd *= response(1)
    initial = declaration["initial_epi"]
    return (initial - even + odd, initial + even, initial - even - odd)


def _declaration():
    """Return the fixed, pre-evaluation protocol and its analytic predictions."""
    declaration = {
        "claim_id": "G3-P3-held-phase-finite-response",
        "nodes": [0, 1, 2],
        "edges": [[0, 1], [1, 2]],
        "conductance": 1.0,
        "length": 1.0,
        "initial_epi": 0.5,
        "capacity": 1.0,
        "center_phase": math.pi / 4,
        "mean_gap": math.pi / 8,
        "spreads": [math.pi / 16, 3 * math.pi / 16],
        "raw_weights": {"phase": 2, "epi": 4, "vf": 1, "topo": 1},
        "normalized_weights": {"phase": 0.25, "epi": 0.5, "vf": 0.125, "topo": 0.125},
        "models": list(_MODELS),
        "gamma": {"type": "none"},
        "held_coordinates": ["phase", "capacity", "support", "conductance"],
        "horizon": 1.0,
        "steps": 256,
        "dt": 1 / 256,
        "solver": "Euler; refresh complete pressure before each nodal integrator call",
        "pressure_path": "NumPy fused canonical; four directed support entries exclude JIT",
        "clip_bounds": [0.0, 1.0],
        "unclipped_domain": [0.25, 0.75],
        "source_bound": 0.203125,
        "analytic_trajectory_enclosure": [0.296875, 0.703125],
        "paired_orientation": "center EPI at spread[1] minus center EPI at spread[0]",
        "budgets": {
            "pressure_realization": 2e-14,
            "step_rounding": 2e-15,
            "discrete_trajectory": 1e-11,
            "paired_continuous": 5e-6,
        },
        "precision": "IEEE-754 binary64; Fraction observes represented Euler defects",
        "scope": "prospective software control; no physical law selection or autonomous formation",
        "assumptions": [
            "fixed reciprocal P3 with unit conductance and length",
            "held acute phases, common positive capacity, Gamma absent",
            "capacity gradient is zero; nonzero topology source remains included",
            "analytic enclosure uses positive diffusion averaging and the declared source bound",
            "numerical budgets are declared acceptance policies, not interval certificates",
            "local artifact order is not externally trusted prospective registration",
        ],
    }
    predictions = {}
    for model in _MODELS:
        predictions[model] = {}
        for label, continuous in (("continuous", True), ("euler", False)):
            endpoints = [
                _prediction(
                    declaration,
                    model,
                    spread,
                    declaration["steps"],
                    continuous=continuous,
                )
                for spread in declaration["spreads"]
            ]
            predictions[model][label] = {
                "endpoints": endpoints,
                "paired_center": endpoints[1][1] - endpoints[0][1],
            }
    declaration["predictions"] = predictions
    return declaration


def _read(graph, aliases):
    return tuple(get_attr(graph.nodes[node], aliases, strict=True) for node in graph)


def _run_case(declaration, model, spread):
    """Advance actual shared owners; closed-form predictions never supply pressure."""
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, declaration["conductance"], "weight")
    nx.set_edge_attributes(graph, declaration["length"], "length")
    center, mean = declaration["center_phase"], declaration["mean_gap"]
    phases = (center + mean - spread, center, center + mean + spread)
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: declaration["initial_epi"],
                ALIAS_VF[0]: declaration["capacity"],
                ALIAS_THETA[0]: phase,
                ALIAS_DNFR[0]: 0.0,
            }
        )
    graph.graph.update(
        DNFR_WEIGHTS=dict(declaration["raw_weights"]),
        vectorized_dnfr=True,
        GAMMA=dict(declaration["gamma"]),
        DT_MIN=0.0,
        EPI_MIN=declaration["clip_bounds"][0],
        EPI_MAX=declaration["clip_bounds"][1],
        CLIP_MODE="hard",
        use_extended_dynamics=False,
    )
    held_support = tuple(graph.edges(data="weight"))
    trace = [{"time": 0.0, "epi": _read(graph, ALIAS_EPI)}]
    max_pressure = max_rounding = max_discrete = 0.0
    observed_min = observed_max = declaration["initial_epi"]
    unclipped = held = True
    dt = declaration["dt"]
    capacity = declaration["capacity"]
    weights = declaration["normalized_weights"]
    source = _source(declaration, model, spread)
    unit_phase = current = observation = None
    for step in range(declaration["steps"]):
        profile = {}
        default_compute_delta_nfr(graph, n_jobs=1, profile=profile)
        if profile["dnfr_path"] != "fused_canonical":
            raise RuntimeError("declared NumPy pressure path did not execute")
        if step == 0:
            observation = capture_non_epi_forcing(graph)
            unit_phase = tuple(map(float, observation.phase_gradient))
            current_by_node = compute_phase_current(graph)
            current = tuple(current_by_node[node] / math.pi for node in graph)
            if dict(observation.normalized_weights) != weights:
                raise RuntimeError("pressure mixture differs from the declaration")
        if model == "sine_current":
            for node, pressure, arg, sine in zip(
                graph, _read(graph, ALIAS_DNFR), unit_phase, current, strict=True
            ):
                # This explicit experimental substitution is not a new default
                # pressure implementation; every other channel was refreshed.
                set_dnfr(graph, node, pressure + weights["phase"] * (sine - arg))
        before = _read(graph, ALIAS_EPI)
        pressure = _read(graph, ALIAS_DNFR)
        # Independent P3 matrix multiplication, not another trajectory solver.
        laplacian = (
            before[0] - before[1],
            before[1] - (before[0] + before[2]) / 2,
            before[2] - before[1],
        )
        expected_pressure = tuple(
            f - weights["epi"] * lx for f, lx in zip(source, laplacian, strict=True)
        )
        max_pressure = max(
            max_pressure,
            *(abs(a - b) for a, b in zip(pressure, expected_pressure, strict=True)),
        )
        raw = tuple(
            Fraction(x) + Fraction(dt) * Fraction(capacity) * Fraction(p)
            for x, p in zip(before, pressure, strict=True)
        )
        low, high = declaration["unclipped_domain"]
        unclipped &= all(low < value < high for value in raw)
        update_epi_via_nodal_equation(
            graph, dt=dt, t=step * dt, method="euler", n_jobs=1
        )
        after = _read(graph, ALIAS_EPI)
        max_rounding = max(
            max_rounding,
            *(
                float(abs(Fraction(value) - proposal))
                for value, proposal in zip(after, raw, strict=True)
            ),
        )
        expected = _prediction(declaration, model, spread, step + 1, continuous=False)
        max_discrete = max(
            max_discrete, *(abs(a - b) for a, b in zip(after, expected, strict=True))
        )
        unclipped &= all(low < value < high for value in after)
        observed_min, observed_max = min(observed_min, *after), max(
            observed_max, *after
        )
        held &= (
            _read(graph, ALIAS_THETA) == phases
            and _read(graph, ALIAS_VF) == (capacity,) * 3
            and tuple(graph.edges(data="weight")) == held_support
            and graph.graph["GAMMA"] == declaration["gamma"]
        )
        trace.append({"time": (step + 1) * dt, "epi": after, "pressure_used": pressure})
    return {
        "model": model,
        "spread": spread,
        "phases": phases,
        "initial_arg_phase_source": unit_phase,
        "initial_sine_phase_source": current,
        "initial_kernel_assembly_defect": list(
            map(float, observation.kernel_pressure_defect)
        ),
        "initial_stored_pressure_residual": list(
            map(float, observation.stored_pressure_residual)
        ),
        "max_pressure_realization_error": max_pressure,
        "max_step_rounding_defect": max_rounding,
        "max_discrete_trajectory_error": max_discrete,
        "observed_epi_range": [observed_min, observed_max],
        "unclipped_domain_verified": unclipped,
        "held_coordinates_verified": held,
        "pressure_refresh_calls": declaration["steps"],
        "nodal_integrator_calls": len(trace) - 1,
        "trace": trace,
    }


def _save_new(path, data):
    write_json_once(path, data, sort_keys=True, newline=None)


def run_phase_form_response(output_dir, *, repository=None):
    """Write the declaration before evolution, then finite evidence and sidecar.

    No existing declaration/results are overwritten. The output directory is
    the evidence root, independently of the Git checkout supplying provenance.
    A changed declaration or working source invalidates this invocation.
    """
    repository = (
        Path(repository)
        if repository is not None
        else Path(__file__).resolve().parents[3]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ("declaration.json", "result.json", "evidence.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError(
            "use a fresh output directory; frozen evidence is not overwritten"
        )
    revision, dirty, dirty_hash = source_before = current_git_source_provenance(
        repository, _SOURCE_PATHS
    )
    declaration = _declaration()
    _save_new(output / names[0], declaration)
    declaration_hash = _hash(output / names[0])
    cases = [
        _run_case(declaration, model, spread)
        for model in _MODELS
        for spread in declaration["spreads"]
    ]
    if _hash(output / names[0]) != declaration_hash:
        raise RuntimeError("frozen declaration changed during evaluation")
    if current_git_source_provenance(repository, _SOURCE_PATHS) != source_before:
        raise RuntimeError("working source changed during evaluation")
    paired = {}
    for model in _MODELS:
        first, second = (case for case in cases if case["model"] == model)
        actual = second["trace"][-1]["epi"][1] - first["trace"][-1]["epi"][1]
        prediction = declaration["predictions"][model]
        discrete, continuous = (
            prediction["euler"]["paired_center"],
            prediction["continuous"]["paired_center"],
        )
        paired[model] = {
            "observed": actual,
            "continuous_prediction": continuous,
            "euler_prediction": discrete,
            "analytic_integration_defect": discrete - continuous,
            "engine_realization_defect": actual - discrete,
            "total_continuous_error": actual - continuous,
        }
    budgets = declaration["budgets"]
    checks = {
        "pressure_realization": all(
            case["max_pressure_realization_error"] <= budgets["pressure_realization"]
            for case in cases
        ),
        "step_rounding": all(
            case["max_step_rounding_defect"] <= budgets["step_rounding"]
            for case in cases
        ),
        "discrete_trajectory": all(
            case["max_discrete_trajectory_error"] <= budgets["discrete_trajectory"]
            for case in cases
        ),
        "paired_continuous": all(
            abs(pair["total_continuous_error"]) <= budgets["paired_continuous"]
            for pair in paired.values()
        ),
        "unclipped_domain": all(case["unclipped_domain_verified"] for case in cases),
        "held_coordinates": all(case["held_coordinates_verified"] for case in cases),
    }
    accepted = all(checks.values())
    report = {
        "claim_id": declaration["claim_id"],
        "declaration_sha256": declaration_hash,
        "paired_orientation": declaration["paired_orientation"],
        "checks": checks,
        "accepted": accepted,
        "paired": paired,
        "cases": cases,
        "physical_status": "not_admitted_by_this_software_comparison",
    }
    _save_new(output / names[1], report)
    manifest = CoreExperimentManifest(
        claim_id=declaration["claim_id"],
        git_sha=revision,
        source_dirty=dirty,
        dirty_source_hash=dirty_hash,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        graph_construction="ordered unit-conductance P3: 0--1--2",
        capacity_specification="held common capacity one",
        solver=declaration["solver"],
        result_status=ClaimStatus.MEASURED if accepted else ClaimStatus.NEGATIVE,
        timestep=declaration["dt"],
        telemetry=(
            "center spread-pair response",
            "pressure/integration/rounding defects",
        ),
        controls=(
            "same-mean different-spread pair",
            "independent P3 eigenmode solution",
            "complete pressure mixture",
        ),
        artifacts=names[:2],
    )
    sidecar = EvidenceSidecar(
        manifest=manifest,
        artifact=names[1],
        model="held-phase P3 pressure-source comparison",
        norm="maximum absolute coordinate defect; signed paired center response",
        distance_convention="unit conductance and length; unique unweighted phase neighbors",
        clock="declared structural time, not laboratory seconds",
        finite_horizon=declaration["horizon"],
        tail_status="UNASSESSED_FINITE_WINDOW",
        provenance={
            "uses_future_samples": False,
            "uses_outcome_derived_wiring": False,
            "fits_on_evaluation_data": False,
            "uses_evaluation_labels": False,
            "uses_postselection": False,
        },
        claim_statement="finite shared-engine response agrees with independently declared modal predictions within the frozen budgets",
        claim_status=manifest.result_status.value,
        scope=declaration["scope"],
        assumptions=tuple(declaration["assumptions"]),
        outcome=(
            "all prospective controls accepted"
            if accepted
            else "at least one prospective control rejected"
        ),
        source_imports=(
            "tnfr.dynamics.dnfr",
            "tnfr.dynamics.integrators",
            "tnfr.physics.forcing_realization",
            "tnfr.physics.extended",
            "tnfr.research.phase_form_response",
        ),
        dirty_source_hash=dirty_hash or "",
        graph_context={
            key: declaration[key] for key in ("nodes", "edges", "conductance", "length")
        },
        state_context={
            key: declaration[key]
            for key in (
                "initial_epi",
                "capacity",
                "center_phase",
                "mean_gap",
                "spreads",
                "held_coordinates",
                "gamma",
            )
        },
        numerical_context={
            key: declaration[key]
            for key in (
                "dt",
                "steps",
                "budgets",
                "precision",
                "pressure_path",
                "normalized_weights",
                "clip_bounds",
                "unclipped_domain",
            )
        },
        observation_context={
            "declaration_sha256": declaration_hash,
            "paired_orientation": declaration["paired_orientation"],
            "physical_status": report["physical_status"],
        },
        cost_context={
            "trajectories": len(cases),
            "nodal_integrator_calls": sum(
                case["nodal_integrator_calls"] for case in cases
            ),
        },
        artifact_hashes={name: _hash(output / name) for name in names[:2]},
        stop_reason="single declared finite comparison completed; no sweep",
    )
    sidecar.write_admitted(output / names[2], root_dir=output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_phase_form_response(args.output)
    print(
        json_dumps(
            {key: report[key] for key in ("accepted", "checks", "paired")}, indent=2
        )
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
