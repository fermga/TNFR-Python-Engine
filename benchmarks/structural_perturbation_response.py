"""One fixed U/P/F response comparison from the actual retained t=9.5 graph.

The old B2.d.8 profile and metric remain the target. Default child VAL/IL
prepares P and F; F additionally requests child UM. Twelve shared Euler
intervals precede each branch's separate terminal closures. No search occurs.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, is_dataclass
from enum import Enum
from fractions import Fraction
import json
from pathlib import Path
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.canonical_winding_persistence import _tetrad  # noqa: E402
from benchmarks.child_coupling_feedback import (  # noqa: E402
    _advance_forced_support_interval, _compatibility_channels, _pattern,
    prepare_child_feedback_endpoint,
)
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.metrics.common import compute_coherence  # noqa: E402
from tnfr.operators.definitions import (  # noqa: E402
    Coherence, Coupling, Dissonance, Expansion, SelfOrganization, Silence,
)
from tnfr.operators.factor_contracts import (  # noqa: E402
    resolve_runtime_operator_factors,
)
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.grammar_execution import ValidatedSequence  # noqa: E402
from tnfr.operators.grammar_types import StructuralGrammarError  # noqa: E402
from tnfr.operators.preconditions import OperatorPreconditionError  # noqa: E402
from tnfr.physics.forced_support import (  # noqa: E402
    derive_forced_support_balance, observe_forced_support_event,
    observe_forced_support_pattern, observe_forced_support_state,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.physics.support_transport import SupportTransportSnapshot  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.rng import base_seed  # noqa: E402
from tnfr.utils import ensure_node_offset_map  # noqa: E402
from tnfr.validation import validate_sequence  # noqa: E402

CASES = ("U", "P", "F")
STEP = 0.25
SEGMENT_COUNT = 12


def _plain(value):
    """Retain replayable public attributes without address-bearing repr strings."""
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool, Fraction)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque, np.ndarray)):
        return tuple(_plain(item) for item in value)
    if isinstance(value, random.Random):
        return {"python_random_state": _plain(value.getstate())}
    if isinstance(value, np.random.Generator):
        return {"numpy_generator_state": _plain(value.bit_generator.state)}
    if is_dataclass(value):
        return _plain(asdict(value))
    raise TypeError(f"unrecorded materialized attribute type: {type(value).__name__}")


def _materialized(graph):
    """Audit channels, all node attributes and controls relevant to these kernels."""
    return {
        "state": _state(graph),
        "node_attributes": {
            node: _plain(dict(data)) for node, data in graph.nodes(data=True)
        },
        "configured_controls": {
            key: _plain(value) for key, value in graph.graph.items()
            if key.isupper() and not callable(value)
        },
        "random_provenance": {
            "resolved_base_seed": base_seed(graph),
            "node_offsets": dict(ensure_node_offset_map(graph)),
            "candidate_sample": tuple(graph.graph.get("_node_sample", ())),
            "graph_random_generators": {
                key: _plain(value) for key, value in graph.graph.items()
                if isinstance(value, (random.Random, np.random.Generator))
            },
            "stream_scope": (
                "Coupling constructs local deterministic make_rng(seed, node_offset) "
                "streams. The retained nine-node sample is the full node tuple; "
                "VAL and IL use no random stream in these default branches"
            ),
        },
    }


def _diagnostics(graph):
    return {
        "tetrad": _plain(_tetrad(graph)),
        "canonical_network_coherence": float(compute_coherence(graph)),
        "scope": "Current refreshed pressure and retained derivative diagnostics",
    }


def _reference(capture):
    return derive_forced_support_balance(
        capture.snapshot, epi_weight=capture.epi_weight, forcing=capture.forcing,
    )


def _word(operators, *, initialized):
    context = {"initial_epi_nonzero": initialized}
    names = tuple(operator.name for operator in operators)
    validation = validate_sequence(list(names), context=context)
    record = {"names": names, "context": context, "string_validator_passed": validation.passed}
    try:
        validated = ValidatedSequence(operators, context=context)
    except ValueError as error:
        record.update(instance_validator_passed=False, reason=str(error))
        return None, record
    record["instance_validator_passed"] = True
    if not validation.passed:
        record["reason"] = "string-based full-word validator refused the declared word"
        return None, record
    return validated, record


def _event(graph, child, operator, sequence_step, target, before_capture, *, refresh):
    before = _materialized(graph)
    before_reference = _reference(before_capture)
    admission = validate_candidate(
        graph, child, operator.glyph, sequence_context=sequence_step,
    )
    record = {
        "target": child, "requested_name": operator.name,
        "requested_glyph": operator.glyph.value,
        "sequence_index": sequence_step.index,
        "admission": _plain(asdict(admission)), "before": before,
        "resolved_factors": _plain(resolve_runtime_operator_factors(
            graph.graph.get("GLYPH_FACTORS"), operator.glyph, graph.graph,
        )),
        "pressure_refresh_after_event": refresh,
    }
    if not admission.allowed:
        record.update(status="controlled_obstruction", reason="live grammar refusal")
        return before_capture, record
    metrics_start = len(graph.graph.get("operator_metrics", ()))
    try:
        operator(graph, child, collect_metrics=True, sequence_context=sequence_step)
    except (OperatorPreconditionError, StructuralGrammarError, ValueError) as error:
        # IL's current strict preflight still uses a plain ValueError. Other
        # unexpected ValueError/RuntimeError failures are implementation errors.
        if isinstance(error, ValueError) and not str(error).startswith("IL precondition failed:"):
            raise
        record.update(
            status="controlled_obstruction", reason=str(error),
            exception_type=type(error).__name__, after_failure=_materialized(graph),
        )
        return before_capture, record
    raw = _materialized(graph)
    if raw["state"]["glyph_history"][child][-1] != operator.glyph.value:
        raise RuntimeError("a substituted operator cannot count as the requested event")
    raw_capture = capture_non_epi_forcing(graph)
    if refresh:
        default_compute_delta_nfr(graph)
    following = capture_non_epi_forcing(graph)
    after_reference = _reference(following)
    observation = asdict(observe_forced_support_event(
        before_reference, after_reference, before_capture.snapshot, following.snapshot,
    ))
    observation.pop("before_reference")
    observation.pop("after_reference")
    original_before, original_after = _pattern(target, before_capture), _pattern(target, following)
    record.update(
        status="executed", raw_after_event=raw, after_refresh=_materialized(graph),
        actual_operator_metrics=_plain(graph.graph.get("operator_metrics", ())[metrics_start:]),
        before_forcing_capture=asdict(before_capture),
        raw_forcing_capture=asdict(raw_capture), after_forcing_capture=asdict(following),
        before_reference=asdict(before_reference), after_reference=asdict(after_reference),
        before_compatibility_channels=_compatibility_channels(before_reference, before_capture),
        after_compatibility_channels=_compatibility_channels(after_reference, following),
        full_event_budget=observation,
        original_pattern_before=asdict(original_before),
        original_pattern_after=asdict(original_after),
        original_pattern_change=original_after.error_variance - original_before.error_variance,
    )
    return following, record


def _checkpoint(record):
    return {
        "helper": "benchmarks.child_coupling_feedback.prepare_child_feedback_endpoint",
        "case": record["case"], "retained_endpoint": record["final_before_closure"],
        "original_reference": record["original_reference"],
        "postevent_reference": record["postevent_reference"],
        "prefix_checkpoint": record["checkpoint"],
        "executed_recent_segment_count": len(record["segments"]),
        "recent_duration": record["postevent_elapsed_time"],
        "recent_mean_identity_residuals": tuple(
            segment["regime_step_budget"]["mean_identity_residual"]
            for segment in record["segments"]
        ),
        "scope": "Live THOL/parent-UM prefix and child-UM continuation; neither SHA executed",
    }


def run_structural_response_case(case):
    """Run exactly one independently prepared branch, without editing its state."""
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    graph, prefix = prepare_child_feedback_endpoint()
    child = prefix["event"]["target"]
    old = prefix["event"]["before_forcing_capture"]
    target = derive_forced_support_balance(
        SupportTransportSnapshot(**old["snapshot"]),
        epi_weight=old["epi_weight"], forcing=old["forcing"],
    )
    current = capture_non_epi_forcing(graph)
    initial_reference = _reference(current)
    initial = _materialized(graph)
    initial_pattern = _pattern(target, current)
    initial_regime = observe_forced_support_state(initial_reference, current.snapshot)
    child_ops = [Coupling()]
    if case != "U":
        child_ops.extend((Expansion(), Coherence()))
    if case == "F":
        child_ops.append(Coupling())
    child_ops.append(Silence())
    child_epi = initial["state"]["epi"][initial["state"]["nodes"].index(child)]
    child_word, child_word_record = _word(child_ops, initialized=child_epi > 0)
    parent_ops = (Coherence(), Dissonance(), SelfOrganization(), Coupling(), Silence())
    parent_epi = initial["state"]["epi"][initial["state"]["nodes"].index(0)]
    parent_word, parent_word_record = _word(parent_ops, initialized=parent_epi > 0)
    words = {
        "child": {"target": child, "retained_prefix": initial["state"]["glyph_history"][child],
                  **child_word_record},
        "parent": {"target": 0, "retained_prefix": initial["state"]["glyph_history"][0],
                   **parent_word_record},
    }
    record = {
        "case": case, "checkpoint": _checkpoint(prefix), "initial": initial,
        "original_target": asdict(target), "initial_reference": asdict(initial_reference),
        "initial_original_pattern": asdict(initial_pattern), "words": words,
        "initial_diagnostics": _diagnostics(graph), "events": [],
    }
    if child_word is None or parent_word is None:
        record.update(status="controlled_obstruction", reason="full-word validation refused")
        return record
    if words["child"]["retained_prefix"] != ("UM",) or words["parent"]["retained_prefix"] != (
        "IL", "OZ", "THOL", "UM",
    ):
        raise RuntimeError("the retained target histories differ from the declared word prefixes")
    for index, operator in enumerate(child_ops[1:-1], start=1):
        if operator.glyph.value == "UM":
            record["pre_feedback_materialized"] = _materialized(graph)
        current, event = _event(
            graph, child, operator, child_word.step(index), target, current,
            refresh=operator.glyph.value != "VAL",
        )
        record["events"].append(event)
        if event["status"] != "executed":
            record.update(status="controlled_obstruction", reason=event["reason"])
            return record
    if case != "F":
        record["pre_feedback_materialized"] = _materialized(graph)
    reference = _reference(current)
    record["postevent_reference"] = asdict(reference)
    record["conditional_fixed_model_limit"] = {
        "pattern": asdict(observe_forced_support_pattern(
            target, nodes=reference.source.nodes, epi=reference.relative_profile,
        )),
        "scope": (
            "Fixed-coefficient exact-real model limit in the original profile "
            "observable; projected uniform drift vanishes. This is not an "
            "asymptotic assertion about clipped or binary64 runtime execution"
        ),
    }
    record["postevent_diagnostics"] = _diagnostics(graph)
    frozen_capture = current
    segments = []
    for _ in range(SEGMENT_COUNT):
        current, segment = _advance_forced_support_interval(
            graph, target, reference, current, frozen_capture, duration=STEP,
        )
        segments.append(segment)
    final = _materialized(graph)
    final_pattern = _pattern(target, current)
    final_regime = observe_forced_support_state(reference, current.snapshot)
    pattern_event_change = sum((item["original_pattern_change"] for item in record["events"]), Fraction(0))
    pattern_flow_change = sum((
        item["original_pattern_after"]["error_variance"]
        - item["original_pattern_before"]["error_variance"] for item in segments
    ), Fraction(0))
    regime_event_change = sum((
        item["full_event_budget"]["variance_change"] for item in record["events"]
    ), Fraction(0))
    regime_flow_change = sum((
        item["regime_step_budget"]["after"]["error_variance"]
        - item["regime_step_budget"]["before"]["error_variance"] for item in segments
    ), Fraction(0))
    budget = {
        "original_pattern_event_change": pattern_event_change,
        "original_pattern_flow_change": pattern_flow_change,
        "original_pattern_total_change": final_pattern.error_variance - initial_pattern.error_variance,
        "original_pattern_identity_residual": (
            final_pattern.error_variance - initial_pattern.error_variance
            - pattern_event_change - pattern_flow_change
        ),
        "current_regime_event_change": regime_event_change,
        "current_regime_flow_change": regime_flow_change,
        "current_regime_total_change": final_regime.error_variance - initial_regime.error_variance,
        "current_regime_identity_residual": (
            final_regime.error_variance - initial_regime.error_variance
            - regime_event_change - regime_flow_change
        ),
    }
    if budget["original_pattern_identity_residual"] or budget["current_regime_identity_residual"]:
        raise RuntimeError("the exact finite event/flow telescope failed")
    record.update(
        status="measured", segments=segments, final_before_closure=final,
        final_original_pattern=asdict(final_pattern), final_regime_state=asdict(final_regime),
        final_diagnostics=_diagnostics(graph), finite_telescope=budget,
        physical_elapsed_time=final["state"]["time"] - initial["state"]["time"],
    )
    closures = []
    for node, word, ops in ((child, child_word, child_ops), (0, parent_word, parent_ops)):
        context = word.step(len(ops) - 1)
        admission = validate_candidate(graph, node, "SHA", sequence_context=context)
        closure = {"target": node, "admission": _plain(asdict(admission))}
        if admission.allowed:
            Silence()(graph, node, collect_metrics=True, sequence_context=context)
            closure.update(status="executed", after=_state(graph))
        else:
            closure.update(status="controlled_obstruction")
        closures.append(closure)
    record["closures_after_measurement"] = closures
    return record


def compare_structural_response(cases):
    """Apply the predeclared damage/benefit/gap rule to exact represented R0."""
    if tuple(item["case"] for item in cases) != CASES:
        raise ValueError("comparison requires the declared ordered U/P/F branches")
    if any(item["status"] != "measured" for item in cases):
        return {"status": "controlled_obstruction", "scope": "No recovery classification is available"}
    u, p, f = cases
    matches = {
        "common_initial_materialized_state": u["initial"] == p["initial"] == f["initial"],
        "common_original_target": u["original_target"] == p["original_target"] == f["original_target"],
        "matched_perturbed_pre_feedback_state": p["pre_feedback_materialized"] == f["pre_feedback_materialized"],
        "matched_VAL_IL_events": p["events"] == f["events"][:2],
        "matched_elapsed_time": all(item["physical_elapsed_time"] == STEP * SEGMENT_COUNT for item in cases),
    }
    if not all(matches.values()):
        raise RuntimeError(f"the declared matched comparison failed: {matches}")
    ru, rp, rf = (item["final_original_pattern"]["error_variance"] for item in cases)
    damage, benefit, gap = rp - ru, rp - rf, rf - ru
    if damage <= 0:
        classification = "no_damage_in_fixed_observable"
    elif benefit <= 0:
        classification = "no_beneficial_correction"
    elif benefit < damage:
        classification = "partial_correction"
    elif benefit == damage:
        classification = "gap_closed_in_fixed_observable"
    else:
        classification = "outperforms_unperturbed_in_fixed_observable"
    return {
        "status": "measured", "matching_checks": matches,
        "endpoint_pattern_error": dict(zip(CASES, (ru, rp, rf), strict=True)),
        "damage": damage, "corrective_benefit": benefit, "remaining_gap": gap,
        "damage_identity_residual": damage - benefit - gap,
        "correction_fraction": benefit / damage if damage > 0 else None,
        "classification": classification,
        "scope": (
            "Exact arithmetic on represented endpoints at t=12.5 in the fixed old "
            "profile observable. No full-state recovery, solver-error bound, "
            "autonomous repeated policy or empirical correspondence is inferred"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/structural_perturbation_response.json")
    args = parser.parse_args()
    scope = (
        "src/tnfr", "benchmarks/capacity_localization.py",
        "benchmarks/canonical_winding_persistence.py", "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_birth_transport.py", "benchmarks/forced_support_balance.py",
        "benchmarks/child_coupling_feedback.py", "benchmarks/structural_perturbation_response.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    cases = [run_structural_response_case(case) for case in CASES]
    comparison = compare_structural_response(cases)
    manifest = CoreExperimentManifest(
        claim_id="O3.a-matched-structural-perturbation-response",
        git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__},
        graph_construction="Independent live C8+child retained t=9.5 prefixes; no state injection",
        capacity_specification="Actual default child VAL/IL and optional UM writes; held during flow",
        solver="Shared default nodal Euler and declared canonical pressure refresh",
        timestep=STEP, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("parent IL OZ THOL UM", "child UM [VAL IL [UM]]", "separate final SHA"),
        telemetry=("full event/reference budgets", "fixed target damage/benefit/gap", "matched state and RNG provenance", "tetrad and executor endpoints"),
        controls=("U unperturbed", "P identical VAL/IL without feedback"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    report = {"manifest": manifest.to_dict(), "source_scope": scope, "cases": cases, "comparison": comparison}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(_payload(comparison), indent=2))
    print(f"Wrote finite matched response observations to {args.output}")


if __name__ == "__main__":
    main()
