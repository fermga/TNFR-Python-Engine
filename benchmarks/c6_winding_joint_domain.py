"""Two default C6 UM/IL/flow cycles with joint admission and numeric defects.

Exactly three inherited preparations are used: null, positive k1 and positive
k3 at 2^-12. Phase/capacity/EPI bounds, complete-word validators, strict local
readiness and finite binary64 execution remain separate evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.antipodal_region_phase_response import _observed_stage  # noqa: E402
from benchmarks.c6_winding_phase_response import (  # noqa: E402
    AMPLITUDE,
    _phase_readout,
    _winding_state,
    prepare_c6_phase_response,
)
from benchmarks.capacity_feedback import (
    SOURCE_SCOPE,
    STEP,
    _artifact_payload,
)  # noqa: E402
from benchmarks.child_coupling_feedback import (
    _advance_forced_support_interval,
)  # noqa: E402
from benchmarks.compatible_capacity_regions import (
    _admissions,
    _captured_state,
)  # noqa: E402
from benchmarks.structural_perturbation_response import (  # noqa: E402
    _diagnostics,
    _plain,
    _reference,
    _word,
)
from tnfr.config.thresholds import EPI_IL_MIN, VF_IL_MIN  # noqa: E402
from tnfr.operators._coherence_stage_kernel import (
    DEFAULT_PHASE_LOCKING_COEFFICIENT,
)  # noqa: E402
from tnfr.operators.definitions import Coherence, Coupling, Silence  # noqa: E402
from tnfr.operators.factor_contracts import (
    resolve_runtime_operator_factors,
)  # noqa: E402
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_pointwise_stage,
)  # noqa: E402
from tnfr.operators.preconditions import (
    validate_coupling,
    validate_silence,
)  # noqa: E402
from tnfr.physics.coupling_winding import (  # noqa: E402
    derive_c6_winding_joint_domain,
    observe_c6_winding_joint_domain,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402

CASES = (("null", 0.0), ("k1", AMPLITUDE), ("k3", AMPLITUDE))
CYCLE_COUNT = 2


def _admission_band(graph):
    config = graph.graph.get("IL_PRECONDITIONS", {})
    values = {
        "um_min_epi_magnitude": Fraction(float(graph.graph.get("UM_MIN_EPI", 0.05))),
        "um_min_capacity": Fraction(float(graph.graph.get("UM_MIN_VF", 0.01))),
        "il_min_epi": Fraction(float(config.get("min_epi", EPI_IL_MIN))),
        "il_min_capacity": Fraction(float(config.get("min_vf", VF_IL_MIN))),
        "configured_epi_min": Fraction(float(graph.graph["EPI_MIN"])),
        "configured_epi_max": Fraction(float(graph.graph["EPI_MAX"])),
        "configured_strict_preconditions_enabled": bool(
            graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        ),
    }
    lower = max(
        values[name]
        for name in (
            "um_min_epi_magnitude",
            "il_min_epi",
            "configured_epi_min",
        )
    )
    if lower <= values["il_min_epi"]:
        raise ValueError("this inherited domain needs a positive strict IL margin")
    values.update(
        positive_epi_lower=lower,
        epi_upper=values["configured_epi_max"],
        scope=(
            "UM admits equality at its EPI-magnitude/capacity floors; IL requires "
            "strict positivity above its configured minima. Independent validators "
            "are checked without enabling or disabling a graph policy"
        ),
    )
    return values


def _joint_readout(capture, band, domain):
    phase = _phase_readout(capture)
    pi_repr = Fraction(math.pi)
    scaled = tuple(value / pi_repr for value in phase["represented_lift"])
    epi, capacity = capture.snapshot.epi, capture.snapshot.capacity
    diameter = max(scaled) - min(scaled)
    lower_reserve = min(epi) - domain.epi_phase_budget_weight * diameter
    upper_reserve = max(epi) + domain.epi_phase_budget_weight * diameter
    return {
        "phase": phase,
        "phase_pi_represented": scaled,
        "represented_pi_scale": pi_repr,
        "phase_oscillation_pi": diameter,
        "inside_represented_prepared_phase_box": max(map(abs, scaled))
        <= Fraction(1, 24),
        "lower_reserve": lower_reserve,
        "upper_reserve": upper_reserve,
        "within_represented_reserve_band": (
            lower_reserve >= band["positive_epi_lower"]
            and upper_reserve <= band["epi_upper"]
        ),
        "epi_min": min(epi),
        "epi_max": max(epi),
        "epi_lower_margins": tuple(value - band["positive_epi_lower"] for value in epi),
        "epi_upper_margins": tuple(band["epi_upper"] - value for value in epi),
        "strict_il_epi_margins": tuple(value - band["il_min_epi"] for value in epi),
        "strict_il_capacity_margins": tuple(
            value - band["il_min_capacity"] for value in capacity
        ),
        "um_capacity_margins": tuple(
            value - band["um_min_capacity"] for value in capacity
        ),
        "unit_capacity": capacity == (1,) * 6,
        "scope": (
            "Measured wrapped lift divided by represented binary64 pi. These rational "
            "coordinates are not identified with a mathematical-pi phase chart"
        ),
    }


def _repeat_refusal_control(graph):
    before = _captured_state(graph)
    _, word = _word((Coupling(), Coupling(), Silence()), initialized=True)
    candidates = tuple(asdict(validate_candidate(graph, node, "UM")) for node in graph)
    after = _captured_state(graph)
    if before != after:
        raise RuntimeError("the read-only repeated-word control modified the graph")
    return {
        "word": word,
        "live_candidates_after_one_um": _plain(candidates),
        "history_at_probe": before["state"]["glyph_history"],
        "materialized_state_preserved": True,
        "scope": (
            "Both whole-word validators and the current incremental gate are retained. "
            "No direct repeated UM is executed; a string-word refusal is not silently "
            "promoted to a generic kernel or live-candidate refusal"
        ),
    }


def _conditional_transition(reference, before, after, following, flow, band):
    left, right = _joint_readout(before, band, reference), _joint_readout(
        after, band, reference
    )
    result = {
        "phase_before_pi_represented": left["phase_pi_represented"],
        "phase_after_pi_represented": right["phase_pi_represented"],
        "epi_before": before.snapshot.epi,
        "observed_phase_contraction_residual": (
            right["phase_oscillation_pi"]
            - reference.nonlinear_oscillation_factor * left["phase_oscillation_pi"]
        ),
        "observed_phase_nesting_slacks": (
            min(right["phase_pi_represented"]) - min(left["phase_pi_represented"]),
            max(left["phase_pi_represented"]) - max(right["phase_pi_represented"]),
        ),
        "scope": (
            "Conditional exact model evaluated on declared represented chart data. "
            "Its application is separately tested; numeric phase defects can prevent "
            "this binding even when the actual runtime word succeeds"
        ),
    }
    try:
        observation = observe_c6_winding_joint_domain(
            reference,
            phase_before_pi=left["phase_pi_represented"],
            phase_after_pi=right["phase_pi_represented"],
            epi=before.snapshot.epi,
        )
    except ValueError as error:
        result.update(status="outside_conditional_transition", reason=str(error))
        return result
    observed = asdict(observation)
    observed.pop("reference", None)
    ideal = observation.epi_after
    budget = flow["regime_step_budget"]["support_budget"]
    pressure_effect = tuple(
        a - b for a, b in zip(budget["expected_epi"], ideal, strict=True)
    )
    total = tuple(a - b for a, b in zip(following.snapshot.epi, ideal, strict=True))
    residual = tuple(
        a - b - c
        for a, b, c in zip(
            total,
            pressure_effect,
            budget["state_defect"],
            strict=True,
        )
    )
    if any(residual):
        raise RuntimeError(
            "the conditional-pressure/integrator defect decomposition failed"
        )
    result.update(
        status="observed_conditional_transition",
        observation=observed,
        conditional_pressure_realization_effect=pressure_effect,
        actual_integrator_endpoint_defect=budget["state_defect"],
        runtime_minus_conditional_epi=total,
        defect_identity_residual=residual,
    )
    return result


def run_c6_joint_case(mode, epsilon):
    if type(epsilon) is not float or (mode, epsilon) not in CASES:
        raise ValueError(
            "mode and epsilon must be one of the three inherited joint controls"
        )
    graph = prepare_c6_phase_response(mode, epsilon)
    current = capture_non_epi_forcing(graph)
    original = _reference(current)
    band = _admission_band(graph)
    factors = resolve_runtime_operator_factors(
        graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph
    )
    weights = dict(current.normalized_weights)
    domain = derive_c6_winding_joint_domain(
        coupling_phase_factor=factors["UM_theta_push"],
        coherence_phase_factor=DEFAULT_PHASE_LOCKING_COEFFICIENT,
        capacity=1,
        epi_weight=weights["epi"],
        phase_weight=weights["phase"],
        timestep=STEP,
        epi_lower=band["positive_epi_lower"],
        epi_upper=band["epi_upper"],
    )
    ops = (Coupling(), Coherence(), Coupling(), Coherence(), Silence())
    word, validation = _word(ops, initialized=min(current.snapshot.epi) > 0)
    if word is None:
        raise RuntimeError(
            f"the declared two-cycle full word was refused: {validation}"
        )
    record = {
        "mode": mode,
        "epsilon": Fraction(epsilon),
        "cycle_count": CYCLE_COUNT,
        "word": validation,
        "admission_band": band,
        "joint_domain_reference": asdict(domain),
        "initial": _captured_state(graph),
        "initial_capture": asdict(current),
        "initial_joint_readout": _joint_readout(current, band, domain),
        "initial_winding": _winding_state(graph),
        "initial_diagnostics": _diagnostics(graph),
    }
    cycles = []
    for cycle in range(CYCLE_COUNT):
        before = current
        strict_um = []
        for node in graph:
            validate_coupling(graph, node)
            strict_um.append({"node": node, "passed": True})
        current, um = _observed_stage(
            graph,
            ops[2 * cycle],
            word.step(2 * cycle),
            current,
            phase_readout=_phase_readout,
        )
        um_winding = _winding_state(graph)
        if cycle == 0:
            record["repeated_um_control"] = _repeat_refusal_control(graph)
        current, il = _observed_stage(
            graph,
            ops[2 * cycle + 1],
            word.step(2 * cycle + 1),
            current,
            phase_readout=_phase_readout,
        )
        post_il = current
        il_winding = _winding_state(graph)
        current, flow = _advance_forced_support_interval(
            graph,
            original,
            _reference(post_il),
            post_il,
            post_il,
            duration=STEP,
        )
        cycles.append(
            {
                "ordinal": cycle + 1,
                "independent_strict_um_readiness": strict_um,
                "before_capture": asdict(before),
                "um": um,
                "il": il,
                "flow": flow,
                "um_winding": um_winding,
                "il_winding": il_winding,
                "post_il_capture": asdict(post_il),
                "after_capture": asdict(current),
                "before_joint_readout": _joint_readout(before, band, domain),
                "post_il_joint_readout": _joint_readout(post_il, band, domain),
                "after_joint_readout": _joint_readout(current, band, domain),
                "after_winding": _winding_state(graph),
                "conditional_transition": _conditional_transition(
                    domain,
                    before,
                    post_il,
                    current,
                    flow,
                    band,
                ),
            }
        )
    record.update(
        cycles=cycles,
        final_before_closure=_captured_state(graph),
        final_capture=asdict(current),
        final_joint_readout=_joint_readout(current, band, domain),
        final_diagnostics=_diagnostics(graph),
        final_winding=_winding_state(graph),
    )
    targets = tuple(graph)
    strict_sha = []
    for node in targets:
        validate_silence(graph, node)
        strict_sha.append({"node": node, "passed": True})
    admissions = _admissions(graph, ops[-1], word.step(4))
    metrics_start = len(graph.graph.get("operator_metrics", ()))
    closure = execute_pointwise_stage(
        graph,
        ops[-1],
        targets,
        sequence_context=word.step(4),
        collect_metrics=True,
    )
    if closure.schedule != TWO_PHASE_JACOBI or closure.nodes_processed != 6:
        raise RuntimeError("the terminal all-target SHA stage did not execute")
    record["closure_after_measurement"] = {
        "admissions": admissions,
        "independent_strict_sha_readiness": strict_sha,
        "stage_result": asdict(closure),
        "actual_operator_metrics": _plain(
            graph.graph.get("operator_metrics", ())[metrics_start:]
        ),
        "after": _captured_state(graph),
    }
    record["scope"] = (
        "Exactly two prepared default UM/IL/flow cycles with explicit stage refresh "
        "and one terminal SHA after measurement. Numeric phase/budget deviations, "
        "application readiness and conditional exact-model guarantees remain separate; "
        "no binary64 invariant domain, future admission or autonomous preparation follows"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_joint_domain.json",
    )
    args = parser.parse_args()
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    sha, dirty, digest = provenance
    cases = [run_c6_joint_case(*case) for case in CASES]
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-two-cycle-joint-admission",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        graph_construction="Inherited unit C6 winding1: null,+k1,+k3 at2^-12; no later state assignment",
        capacity_specification="Uniform nu=1, initial EPI=.5; threshold-derived positive admission band",
        solver="Shared default Euler: two h=.25 intervals, each four held-pressure substeps",
        timestep=STEP,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(
            "all-target UM IL",
            "Euler .25",
            "all-target UM IL",
            "Euler .25",
            "terminal SHA",
        ),
        telemetry=(
            "strict UM/IL/SHA and complete-word admission",
            "joint-domain reserve",
            "phase and Euler defects",
            "winding/support",
        ),
        controls=(
            "null",
            "+k1",
            "+k3",
            "read-only direct-UM-repeat validator contrast",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance:
        raise RuntimeError("source changed while running the joint C6 controls")
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": SOURCE_SCOPE,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_artifact_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite two-cycle C6 joint observations to {args.output}")


if __name__ == "__main__":
    main()
