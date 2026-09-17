"""Detached additive-defect audit of the retained two-cycle C6 campaign.

No graph is evolved. Input bytes and their historical producer declaration
remain separate from this analysis and its conditional future-error model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.c6_winding_phase_response import _phase_readout  # noqa: E402
from benchmarks.structural_target_compatibility import _capture  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.physics._cycle_algebra import laplacian_action  # noqa: E402
from tnfr.physics.coupling_winding import (  # noqa: E402
    derive_c6_winding_joint_domain, observe_c6_winding_defect,
    bound_c6_winding_defect_prefix, bound_c6_winding_uniform_defects,
)
from tnfr.physics.forcing_realization import decompose_non_epi_forcing  # noqa: E402
from tnfr.physics.support_transport import (  # noqa: E402
    _rebuild, observe_support_transport_euler,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
NODES = tuple(range(6))
SUPPORT = tuple(tuple(sorted(((i - 1) % 6, (i + 1) % 6))) for i in NODES)
EDGES = tuple((i, j, Fraction(1)) for i, row in enumerate(SUPPORT) for j in row)
CASES = (("null", Fraction(0)), ("k1", Fraction(1, 4096)), ("k3", Fraction(1, 4096)))


def _number(value):
    if type(value) not in (str, int, float):
        raise ValueError("recorded scalar must be numeric, not a boolean or container")
    try:
        return Fraction(value)
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("recorded scalar must be finite rational data") from error


def _vector(values):
    if type(values) is not list or len(values) != 6:
        raise ValueError("recorded C6 vector must contain six ordered entries")
    return tuple(_number(value) for value in values)


def _nodes(values):
    if type(values) is not list or any(type(value) is not int for value in values):
        raise ValueError("recorded node indices must be literal integers")
    if tuple(values) != NODES:
        raise ValueError("record requires ordered C6 nodes 0..5")


def _checked_capture(data, reference, *, fresh):
    source = data["snapshot"]
    _nodes(source["nodes"])
    for edge in source["conductance"]:
        if len(edge) != 3 or any(type(index) is not int for index in edge[:2]):
            raise ValueError("conductance indices must be literal integers")
        _number(edge[2])
    if any(type(index) is not int for row in source["support_neighbors"] for index in row):
        raise ValueError("support indices must be literal integers")
    for name, value in source.items():
        if name not in ("nodes", "conductance", "support_neighbors"):
            _vector(value) if type(value) is list else _number(value)
    for name, value in data.items():
        if name not in ("snapshot", "normalized_weights"):
            _vector(value) if type(value) is list else _number(value)
    for _, weight in data["normalized_weights"]:
        _number(weight)
    capture = _capture(data)
    snapshot = _rebuild(capture.snapshot)
    if (snapshot != capture.snapshot or snapshot.conductance != EDGES
            or snapshot.support_neighbors != SUPPORT or snapshot.capacity != (1,) * 6):
        raise ValueError("capture requires rebuilt fixed unit C6 support and capacity")
    decompose_non_epi_forcing(capture)
    weights = dict(capture.normalized_weights)
    if (capture.epi_weight != reference.epi_weight or weights["phase"] != reference.phase_weight
            or weights["topo"] != 0):
        raise ValueError("capture channel coefficients differ from the declared model")
    modeled = tuple(reference.epi_weight * epi + force for epi, force in zip(
        snapshot.epi_gradient, capture.forcing, strict=True,
    ))
    if (capture.kernel_pressure_defect != tuple(a - b for a, b in zip(
        capture.full_kernel_pressure, modeled, strict=True,
    )) or capture.stored_pressure_residual != tuple(a - b for a, b in zip(
        snapshot.stored_pressure, capture.full_kernel_pressure, strict=True,
    ))):
        raise ValueError("serialized pressure-defect caches disagree with their sources")
    if fresh and any(capture.stored_pressure_residual):
        raise ValueError("declared refreshed capture retains stale pressure")
    return capture


def _state_matches(state, capture, *, phase=True):
    _nodes(state["nodes"])
    for name, expected in (("epi", capture.snapshot.epi), ("capacity", capture.snapshot.capacity),
                           ("pressure", capture.snapshot.stored_pressure)):
        if _vector(state[name]) != expected:
            raise ValueError(f"recorded endpoint {name} differs from its capture")
    if phase and _vector(state["phase"]) != capture.phase:
        raise ValueError("recorded phase differs from its capture")
    if phase:
        edges = state["edges"]
        if (type(edges) is not list or len(edges) != 6
                or any(len(edge) != 3 or any(type(index) is not int for index in edge[:2])
                       or _number(edge[2]["weight"]) != 1 or _number(edge[2]["length"]) != 1
                       for edge in edges)
                or {tuple(sorted(edge[:2])) for edge in edges} != {(i, j) for i, j, _ in EDGES if i < j}):
            raise ValueError("materialized graph edges differ from the unit C6 capture")


def _phase_coordinates(capture, readout):
    phase = _phase_readout(capture)
    scale = Fraction(math.pi)
    values = tuple(value / scale for value in phase["represented_lift"])
    if (_number(readout["represented_pi_scale"]) != scale
            or _vector(readout["phase_pi_represented"]) != values
            or _vector(readout["phase"]["represented_lift"]) != phase["represented_lift"]):
        raise ValueError("retained phase coordinates disagree with the represented chart")
    return values


def _stage(stage, before, reference, glyph, expected_history):
    if _checked_capture(stage["before_capture"], reference, fresh=True) != before:
        raise ValueError("stage does not continue the retained source capture")
    raw = _checked_capture(stage["raw_capture"], reference, fresh=False)
    after = _checked_capture(stage["after_capture"], reference, fresh=True)
    _state_matches(stage["before"]["state"], before)
    _state_matches(stage["raw_state"]["state"], raw)
    _state_matches(stage["after_refresh"]["state"], after)
    if (raw.snapshot.epi != before.snapshot.epi or after.snapshot.epi != before.snapshot.epi
            or raw.phase != after.phase or before.normalized_weights != after.normalized_weights):
        raise ValueError("UM/IL or pressure refresh changed an excluded primary input")
    result = stage["stage_result"]
    _nodes(stage["targets"])
    if (result["glyph"] != glyph or result["schedule"] != "two_phase_jacobi"
            or type(result["nodes_processed"]) is not int or result["nodes_processed"] != 6
            or len(stage["admissions"]) != 6
            or any(item["allowed"] is not True or item["candidate"] != glyph for item in stage["admissions"])):
        raise ValueError("stage declaration does not retain all-target admission")
    if any(tuple(stage["before"]["state"]["glyph_history"][str(node)]) != expected_history
           or tuple(stage["after_refresh"]["state"]["glyph_history"][str(node)]) != expected_history + (glyph,)
           for node in NODES):
        raise ValueError("stage histories do not match the declared finite word")
    if glyph == "UM":
        if _number(stage["resolved_factors"]["UM_theta_push"]) != reference.coupling_phase_factor:
            raise ValueError("recorded UM phase factor differs from the joint reference")
    elif any(_number(item["phase"]["coefficient"]) != reference.coherence_phase_factor
             for item in stage["independent_prediction"]):
        raise ValueError("recorded IL phase factor differs from the joint reference")
    return after


def _flow(cycle, before, after, reference):
    flow = cycle["flow"]
    _state_matches(flow["before"], before)
    _state_matches(flow["after_refresh"], after)
    raw = flow["raw_after_integrator"]
    _nodes(raw["nodes"])
    if (_vector(raw["epi"]) != after.snapshot.epi or _vector(raw["capacity"]) != (1,) * 6
            or _vector(raw["pressure"]) != before.snapshot.stored_pressure
            or _vector(raw["phase"]) != before.phase or before.phase != after.phase
            or before.normalized_weights != after.normalized_weights):
        raise ValueError("held flow changed an excluded coordinate or coefficient")
    duration = _number(flow["duration"])
    if (duration != reference.timestep
            or _number(raw["time"]) - _number(flow["before"]["time"]) != duration
            or _number(flow["after_refresh"]["time"]) != _number(raw["time"])):
        raise ValueError("flow duration or endpoint times disagree")
    evidence = flow["executor_evidence"]
    if (evidence["resolved_method"] != "euler" or type(evidence["resolved_substeps"]) is not int
            or evidence["resolved_substeps"] != 4 or evidence["clipping_applied"] is not False
            or evidence["gamma_is_none"] is not True or evidence["extended_dynamics_requested"] is not False
            or evidence["integrator_provenance_certified"] is not True
            or _number(evidence["duration"]) != duration):
        raise ValueError("retained flow does not declare the expected held Euler execution")
    _state_matches(evidence["captured_left"], before, phase=False)
    _nodes(evidence["captured_right"]["nodes"])
    if any(_vector(evidence["captured_right"][name]) != _vector(raw[name])
           for name in ("epi", "capacity", "pressure")):
        raise ValueError("executor right endpoint differs from the recorded raw state")
    if _checked_capture(flow["forcing_capture"], reference, fresh=True) != after:
        raise ValueError("flow forcing capture differs from its endpoint")
    return observe_support_transport_euler(before.snapshot, after.snapshot, duration)


def analyze_c6_defect_case(case):
    data = case["joint_domain_reference"]
    reference = derive_c6_winding_joint_domain(**{name: _number(data[name]) for name in (
        "coupling_phase_factor", "coherence_phase_factor", "capacity", "epi_weight",
        "phase_weight", "timestep", "epi_lower", "epi_upper",
    )})
    if reference.capacity != 1 or reference.timestep != Fraction(1, 4):
        raise ValueError("retained study requires unit capacity and duration 1/4")
    band = case["admission_band"]
    lower = max(_number(band[name]) for name in (
        "um_min_epi_magnitude", "il_min_epi", "configured_epi_min",
    ))
    if (reference.epi_lower != lower or _number(band["positive_epi_lower"]) != lower
            or lower <= _number(band["il_min_epi"])
            or reference.epi_upper != _number(band["epi_upper"])
            or reference.epi_upper != _number(band["configured_epi_max"])
            or reference.capacity < _number(band["um_min_capacity"])
            or reference.capacity <= _number(band["il_min_capacity"])):
        raise ValueError("joint model differs from the recorded application band")
    controls = case["initial"]["configured_controls"]
    if (controls["CLIP_MODE"] != "hard" or _number(controls["EPI_MIN"]) != _number(band["configured_epi_min"])
            or _number(controls["EPI_MAX"]) != reference.epi_upper):
        raise ValueError("joint band differs from the recorded hard clipping controls")
    word = case["word"]
    if (word["string_validator_passed"] is not True or word["instance_validator_passed"] is not True
            or word["context"]["initial_epi_nonzero"] is not True
            or tuple(word["names"]) != ("coupling", "coherence", "coupling", "coherence", "silence")):
        raise ValueError("the retained two-cycle word lacks both admission declarations")
    if type(case["cycle_count"]) is not int or case["cycle_count"] != 2 or len(case["cycles"]) != 2:
        raise ValueError("the retained campaign requires exactly two cycles")
    current = _checked_capture(case["initial_capture"], reference, fresh=True)
    initial = current
    initial_phase = _phase_coordinates(current, case["initial_joint_readout"])
    _state_matches(case["initial"]["state"], current)
    observations, cycles = [], []
    history, time = (), _number(case["initial"]["state"]["time"])
    for ordinal, cycle in enumerate(case["cycles"], start=1):
        if type(cycle["ordinal"]) is not int or cycle["ordinal"] != ordinal:
            raise ValueError("cycle ordinals must be contiguous")
        if _checked_capture(cycle["before_capture"], reference, fresh=True) != current:
            raise ValueError("cycle does not continue the preceding endpoint")
        source = current
        phase_before = _phase_coordinates(source, cycle["before_joint_readout"])
        for key, glyph in (("um", "UM"), ("il", "IL")):
            stage = cycle[key]
            if any(_number(stage[name]["state"]["time"]) != time
                   for name in ("before", "raw_state", "after_refresh")):
                raise ValueError("instantaneous phase stage changed the recorded physical time")
            current = _stage(stage, current, reference, glyph, history)
            history += (glyph,)
        if _checked_capture(cycle["post_il_capture"], reference, fresh=True) != current:
            raise ValueError("nodal interval does not start at the retained IL endpoint")
        phase_after = _phase_coordinates(current, cycle["post_il_joint_readout"])
        after = _checked_capture(cycle["after_capture"], reference, fresh=True)
        if _phase_coordinates(after, cycle["after_joint_readout"]) != phase_after:
            raise ValueError("held flow changed the measured phase chart")
        if _number(cycle["flow"]["before"]["time"]) != time:
            raise ValueError("flow start time does not match its preceding stage")
        budget = _flow(cycle, current, after, reference)
        observation = observe_c6_winding_defect(
            reference, phase_before_pi=phase_before, phase_after_pi=phase_after,
            epi_before=source.snapshot.epi, epi_after=after.snapshot.epi,
        )
        midpoint = tuple(-value for value in laplacian_action(phase_after))
        phase_effect = tuple(reference.forcing_step_factor * (actual - ideal) for actual, ideal in zip(
            current.phase_gradient, midpoint, strict=True,
        ))
        assembly = tuple(reference.timestep * reference.capacity * (kernel + stored) for kernel, stored in zip(
            current.kernel_pressure_defect, current.stored_pressure_residual, strict=True,
        ))
        delta = tuple(actual - expected for actual, expected in zip(
            after.snapshot.epi, observation.modeled_epi_after, strict=True,
        ))
        residual = tuple(total - phase - arithmetic - execution for total, phase, arithmetic, execution in zip(
            delta, phase_effect, assembly, budget.state_defect, strict=True,
        ))
        if any(residual):
            raise RuntimeError("phase/assembly/integrator signed defect identity failed")
        observations.append(observation)
        cycles.append({
            "ordinal": ordinal, "observation": asdict(observation),
            "pressure_budget": {
                "phase_realization_epi_effect": phase_effect,
                "pressure_assembly_epi_effect": assembly,
                "integrator_epi_effect": budget.state_defect,
                "total_epi_defect": delta, "identity_residual": residual,
            },
            "historical_conditional_status": cycle["conditional_transition"]["status"],
        })
        current, time = after, time + reference.timestep
    if _checked_capture(case["final_capture"], reference, fresh=True) != current:
        raise ValueError("final capture differs from the last audited endpoint")
    _state_matches(case["final_before_closure"]["state"], current)
    if _number(case["final_before_closure"]["state"]["time"]) != time:
        raise ValueError("final clock differs from the audited finite prefix")
    prefix = bound_c6_winding_defect_prefix(reference, observations=observations)
    # These are finite observed maxima, not uniform production error bounds.
    eps = max(item.phase_oscillation_defect for item in observations)
    centered = max(item.centered_defect_oscillation for item in observations)
    mean_prefix, mean_bound = Fraction(0), Fraction(0)
    for item in observations:
        mean_prefix += item.mean_defect
        mean_bound = max(mean_bound, abs(mean_prefix))
    uniform = bound_c6_winding_uniform_defects(
        reference, initial_epi=initial.snapshot.epi,
        initial_phase_oscillation=max(initial_phase) - min(initial_phase),
        phase_defect_bound=eps, centered_epi_defect_bound=centered, mean_prefix_bound=mean_bound,
    )
    return {
        "mode": case["mode"], "epsilon": _number(case["epsilon"]), "reference": asdict(reference),
        "cycles": cycles, "prefix_budget": asdict(prefix),
        "uniform_envelope_from_finite_extrema": {
            "conditional_envelope": asdict(uniform), "future_hypotheses_verified": False,
            "scope": "Finite extrema illustrate a conditional bound; they do not bound future defects",
        },
    }


def analyze_c6_defect_report(parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O3.a-C6-two-cycle-joint-admission":
        raise ValueError("input must be the retained B16 C6 joint campaign")
    if tuple((case["mode"], _number(case["epsilon"])) for case in parent["cases"]) != CASES:
        raise ValueError("input requires the ordered null/k1/k3 preparations")
    return {
        "cases": [analyze_c6_defect_case(case) for case in parent["cases"]],
        "runtime_executed": False,
        "scope": "Detached arithmetic and record-continuity audit; no new causal or future-runtime certificate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research/c6_winding_joint_domain.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_defect_budget.json")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite input evidence")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_defect_report(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-additive-defect-budget", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest, versions={"python": platform.python_version()},
        graph_construction="Detached B16 ordered unit C6 captures; no new graph or trajectory",
        capacity_specification="Retained uniform unit capacity; no new choices",
        solver="Offline exact rational error and prefix accounting; no integration",
        result_status=ClaimStatus.DERIVED,
        telemetry=("phase contraction excess", "signed phase/pressure/integrator defects",
                   "mean and reserve prefixes", "conditional range envelope"),
        controls=("retained null/k1/k3", "finite extrema are not future-error hypotheses"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or args.input.read_bytes() != source:
        raise RuntimeError("analysis source or input bytes changed during the audit")
    report.update(
        manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE,
        input_evidence={"path": str(args.input), "sha256": hashlib.sha256(source).hexdigest(),
                        "producer_manifest": parent["manifest"], "producer_source_scope": parent["source_scope"]},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote detached C6 defect budget to {args.output}")


if __name__ == "__main__":
    main()
