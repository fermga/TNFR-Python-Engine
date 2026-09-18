"""Offline EN/AL reset and pre-generated-pressure accounting on retained records.

No graph is reconstructed and no native step, public operator, trajectory or
exponential is executed. Scalar kernels are evaluated on detached inputs only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import thol_native_policy_window as window  # noqa: E402
from benchmarks.thol_family_closure import (  # noqa: E402
    _capture, _equal, _match_live_state, _reference, _check_digest,
)
from benchmarks.thol_full_state_response import _add, _subtract, _paired_delta, _payload  # noqa: E402
from tnfr.operators import _op_AL, _validated_epi_assignment_value  # noqa: E402
from tnfr.operators._neighbor_epi_kernel import (  # noqa: E402
    neighbor_epi_unweighted_mean, neighbor_epi_blend_value,
    neighbor_epi_represented_affine_row,
)
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors  # noqa: E402
from tnfr.physics._cycle_algebra import dot  # noqa: E402
from tnfr.physics.hybrid_operator_stability import (  # noqa: E402
    _exact_matrix_product as mm, _exact_matrix_vector as mv,
)
from tnfr.physics.forced_support import (  # noqa: E402
    observe_forced_support_event, observe_forced_support_pattern,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

WINDOW_PATH = ROOT / "artifacts/research/thol_native_policy_window_2026_09_18.json"
WINDOW_SHA256 = "37907325606a10551c65302fcbc18ef0fc23e0f46868cc222b3bc524b0faeedb"
DT = F(1, 4)
SELECTED = ((2, "EN", F(5, 2)), (4, "AL", F(3)))


def _v(values):
    if any(type(x) not in (int, float, str, F) or (type(x) is float and not math.isfinite(x)) for x in values):
        raise ValueError("coordinates must be finite exact or represented reals")
    return tuple(F(x) for x in values)


def _identity(n):
    return tuple(tuple(F(i == j) for j in range(n)) for i in range(n))


def _sum(vectors, n):
    return tuple(sum((row[i] for row in vectors), F(0)) for i in range(n))


def _literal_number(value):
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] == "binary64":
        value = float.fromhex(value[1])
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("recorded numeric configuration must be finite")
    return value


def _configuration(record):
    raw = record["graph_attributes"]
    result = {key: _literal_number(raw[key]) for key in ("EPI_MIN", "EPI_MAX") if key in raw}
    if "CLIP_MODE" in raw:
        if type(raw["CLIP_MODE"]) is not str:
            raise ValueError("invalid recorded clip mode")
        result["CLIP_MODE"] = raw["CLIP_MODE"]
    factors = raw.get("GLYPH_FACTORS", {})
    if type(factors) is not dict:
        raise ValueError("recorded factors must be a mapping")
    result["GLYPH_FACTORS"] = {key: _literal_number(value) for key, value in factors.items()}
    return result


def _load(path, expected, claim):
    _check_digest(expected)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("retained input digest mismatch")
    result = json.loads(raw)
    manifest = CoreExperimentManifest(**result["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != claim:
        raise ValueError("retained input claim mismatch")
    return result, {"path": str(path), "sha256": expected, "historical_manifest": result["manifest"]}


def load_evidence(window_path=WINDOW_PATH, native_path=window.NATIVE_PATH, *,
                  expected_window_sha256=WINDOW_SHA256, expected_native_sha256=window.NATIVE_SHA256):
    retained, wb = _load(window_path, expected_window_sha256, "O3.a-generated-five-step-native-policy-window")
    prior, nb = _load(native_path, expected_native_sha256, "O1.b-generated-native-runtime-response")
    binding = retained["historical_inputs"]["native"]
    if (binding["sha256"] != expected_native_sha256 or binding["historical_manifest"] != prior["manifest"]
            or _payload(retained["protocol"]) != _payload(window.PROTOCOL)
            or tuple(row["branch"] for row in retained["branches"]) != window.BRANCHES
            or tuple(row["branch"] for row in prior["branches"]) != window.BRANCHES):
        raise ValueError("retained window ancestry, protocol or branch order mismatch")
    pool = retained["record_pool"]
    original = window.expand_record(retained["original_reference"], pool)
    for i, branch in enumerate(retained["branches"]):
        first = window.expand_record(retained["first_step_replays"][i], pool)
        children = prior["replayed_prior_reports"][i]["lineage"]["children"]
        window._admit_native_replay(first, prior["branches"][i], children)
        _equal(original, prior["replayed_prior_reports"][i]["original_reference"], "original target identity")
        previous = first["endpoint"]
        if branch["completed_steps"] != 5 or branch["status"] != "completed" or len(branch["steps"]) != 5:
            raise ValueError("audit requires the retained completed five-step window")
        for ordinal, reference in enumerate(branch["steps"]):
            step = window.expand_record(reference, pool)
            if step["ordinal"] != ordinal or step["status"] != "executed" or step["before"] != previous:
                raise ValueError("retained step chronology or complete boundary identity mismatch")
            if F(step["endpoint"]["state"]["time"]) != F(2) + ordinal*DT:
                raise ValueError("retained window endpoint differs")
            previous = step["endpoint"]
        _equal(previous, window.expand_record(branch["final_record"], pool), "final retained record")
    return retained, _reference(original), {"window": wb, "native": nb}


def audit_reset_step(step, original_reference):
    """Admit one complete ordered EN or AL reset and its native consumed drive."""
    trace = step["native_trace"]
    prepared, integration = window._one(trace, "_prepare_dnfr"), window._one(trace, "integrate")
    if step["status"] != "executed" or trace["status"] != "executed" or prepared is None or integration is None:
        raise ValueError("a completed generation and integration are required")
    generation = prepared["after"]["state"]
    capture = trace["captures"]["pressure_generation"]
    if not capture["available"]:
        raise ValueError("generation forcing capture is unavailable")
    snap, observation, _ = _capture(capture["payload"])
    _match_live_state(generation, capture["payload"], snap)
    nodes, n = tuple(generation["nodes"]), len(generation["nodes"])
    if nodes != original_reference.source.nodes or not 1 < n <= 16:
        raise ValueError("reset requires the original bounded ordered node space")
    calls = [row for row in trace["boundaries"] if row["boundary"] == "apply_glyph"]
    glyphs = {row["glyph"] for row in calls}
    if (len(calls) != n or tuple(row["node"] for row in calls) != nodes or len(glyphs) != 1
            or not glyphs <= {"EN", "AL"} or any(row["outcome"] != "completed" for row in calls)):
        raise ValueError("exactly one ordered EN or AL call per node is required")
    glyph = next(iter(glyphs))
    if len([row for row in trace["boundaries"] if row["boundary"] == "_refresh_delta_nfr"]) != 1:
        raise ValueError("exactly one pre-reset pressure refresh is required")
    anchors = [step["before"]["state"], generation] + [row[side]["state"] for row in calls for side in ("before", "after")]
    anchors += [integration["before"]["state"], integration["after"]["state"], step["endpoint"]["state"]]
    if any(any(state[key] != generation[key] for key in ("nodes", "capacity", "edges")) for state in anchors):
        raise ValueError("node, capacity or support changed during retained reset step")
    if any(state["pressure"] != generation["pressure"] for state in anchors[1:]):
        raise ValueError("generated pressure was not retained through reset and integration")
    if any(state["phase"] != generation["phase"] for state in anchors[1:-2]):
        raise ValueError("phase changed before integration")
    if step["before"]["state"]["epi"] != generation["epi"]:
        raise ValueError("unaccounted pre-generation EPI jump")
    start_time, end_time = F(step["before"]["state"]["time"]), F(step["endpoint"]["state"]["time"])
    if end_time-start_time != DT or F(integration["before"]["state"]["time"]) != start_time:
        raise ValueError("reset step has an unexpected physical interval")
    x0 = _v(generation["epi"])
    S, c, propagated, rows = _identity(n), (F(0),)*n, [], []
    expected = x0
    for i, row in enumerate(calls):
        before, after = row["before"], row["after"]
        x, y = _v(before["state"]["epi"]), _v(after["state"]["epi"])
        if x != expected or any(y[j] != x[j] for j in range(n) if j != i):
            raise ValueError("local reset order or single-target EPI write differs")
        cfg = _configuration(before)
        factors = resolve_runtime_operator_factors(cfg["GLYPH_FACTORS"], glyph, cfg)
        holder = SimpleNamespace(graph=cfg, EPI=float(x[i]))
        local = list(_identity(n))
        offset = [F(0)]*n
        if glyph == "EN":
            ordered = before["ordered_neighbors"]
            if tuple(node for node, _ in ordered) != nodes:
                raise ValueError("ordered neighbor record differs from node space")
            neighbors = tuple(ordered[i][1])
            indices = tuple(nodes.index(node) for node in neighbors)
            # The support observer sorts indices; EN retains actual iteration
            # order for its scalar read. Compare membership, keep both orders.
            if tuple(sorted(indices)) != snap.support_neighbors[i]:
                raise ValueError("Reception neighbors differ from captured runtime support")
            mix = factors["EN_mix"]
            local[i] = _v(neighbor_epi_represented_affine_row(n, i, indices, mix))
            mean = neighbor_epi_unweighted_mean(float(x[j]) for j in indices)
            raw = neighbor_epi_blend_value(float(x[i]), mean, mix)
            bounded = _validated_epi_assignment_value(holder, raw)
            control = {"EN_mix": F(mix), "neighbor_indices": indices, "neighbor_mean": F(mean)}
        else:
            boost = factors["AL_boost"]
            offset[i] = F(boost)
            raw = float(x[i])+boost
            _op_AL(holder, factors)
            bounded = holder.EPI
            control = {"AL_boost": F(boost)}
        if F(bounded) != y[i]:
            raise ValueError("retained local reset differs from the shared scalar kernel")
        local, offset = tuple(local), tuple(offset)
        ideal = _add(mv(local, x), offset)
        defect = _subtract(y, ideal)
        propagated = [mv(local, previous) for previous in propagated] + [defect]
        S, c = mm(local, S), _add(mv(local, c), offset)
        rows.append({"node": nodes[i], "control": control, "configuration": cfg,
                     "row": local[i], "offset": offset[i], "before_epi": x, "after_epi": y,
                     "kernel_evaluation_defect": F(raw)-ideal[i], "clipping_defect": y[i]-F(raw),
                     "local_defect": defect})
        expected = y
    xg, xi, xf = (_v(record["epi"]) for record in (
        integration["before"]["state"], integration["after"]["state"], step["endpoint"]["state"]))
    if expected != xg:
        raise ValueError("unaccounted post-reset pre-integration EPI change")
    total_defect = _sum(propagated, n)
    reset_residual = _subtract(xg, _add(_add(mv(S, x0), c), total_defect))
    strengths = [F(0)]*n
    for i, j, weight in snap.conductance:
        strengths[i] += weight
    A = [[F(0)]*n for _ in range(n)]
    for i, j, weight in snap.conductance:
        a = snap.capacity[i]*observation.epi_weight*weight/strengths[i]
        A[i][i] += a
        A[i][j] -= a
    A = tuple(map(tuple, A))
    b = tuple(nu*f for nu, f in zip(snap.capacity, observation.forcing, strict=True))
    ideal_drive = _subtract(b, mv(A, x0))
    pressure_term = tuple(DT*(nu*p-f) for nu, p, f in zip(snap.capacity, snap.stored_pressure, ideal_drive, strict=True))
    integration_defect = tuple(y-x-DT*nu*p for y, x, nu, p in zip(xi, xg, snap.capacity, snap.stored_pressure, strict=True))
    post = _subtract(xf, xi)
    ideal_end = _add(_add(mv(S, x0), c), tuple(DT*f for f in ideal_drive))
    residual = _subtract(xf, _add(ideal_end, _sum((total_defect, pressure_term, integration_defect, post), n)))
    if any(reset_residual) or any(residual):
        raise RuntimeError("exact reset/runtime accounting identity failed")
    # Reuse the fixed-reference event observer: target/metric do not change.
    event = observe_forced_support_event(original_reference, original_reference,
                                         replace(snap, epi=x0), replace(snap, epi=xg))
    patterns = {name: asdict(observe_forced_support_pattern(original_reference, nodes=nodes, epi=value))
                for name, value in (("before", x0), ("after_reset", xg), ("endpoint", xf))}
    return {"glyph": glyph, "nodes": nodes, "start_time": start_time, "end_time": end_time,
            "vectors": {"x0": x0, "xg": xg, "xi": xi, "xf": xf},
            "reset": {"S": S, "c": c, "local_rows": rows, "propagated_local_defects": propagated,
                      "total_reset_defect": total_defect, "identity_residual": reset_residual},
            "generation": {"A": A, "b": b, "observation": asdict(observation)},
            "runtime": {"ideal_endpoint": ideal_end, "pressure_term": pressure_term,
                        "integration_remainder": integration_defect, "postintegration_change": post,
                        "identity_residual": residual, "pre_generated_pressure_retained": True},
            "old_target_patterns": patterns, "old_target_reset_event": asdict(event),
            "scope": "Finite retained reset; integration remainder is not assumed to be rounding"}


def audit_pair(left, right, metric):
    """Compare two admitted resets using the same generation coefficients."""
    keys = ("glyph", "nodes", "start_time", "end_time")
    if any(left[key] != right[key] for key in keys) or len(metric) != len(left["nodes"]):
        raise ValueError("paired resets require common ordered nodes, glyph and times")
    flags = {key: left["generation"][key] == right["generation"][key] for key in ("A", "b")}
    flags.update({key: left["reset"][key] == right["reset"][key] for key in ("S", "c")})
    if not all(flags.values()):
        return {"available": False, "flags": flags, "reason": "Reset map or generation model differs"}
    n = len(metric)
    states = {name: _paired_delta(left["vectors"][key], right["vectors"][key], metric)
              for name, key in (("before", "x0"), ("after_reset", "xg"), ("endpoint", "xf"))}
    S, A = left["reset"]["S"], left["generation"]["A"]
    pair_map = tuple(tuple(s-DT*a for s, a in zip(sr, ar, strict=True)) for sr, ar in zip(S, A, strict=True))
    ideal = mv(pair_map, states["before"]["epi_difference"])
    terms = {"reset_defect": _subtract(right["reset"]["total_reset_defect"], left["reset"]["total_reset_defect"])}
    for key in ("pressure_term", "integration_remainder", "postintegration_change"):
        terms[key] = _subtract(right["runtime"][key], left["runtime"][key])
    residual = _subtract(states["endpoint"]["epi_difference"], _add(ideal, _sum(terms.values(), n)))
    if any(residual):
        raise RuntimeError("paired reset accounting identity failed")
    return {"available": True, "flags": flags, "glyph": left["glyph"], "time": left["end_time"],
            "states": states, "pair_map": pair_map, "formula": "S-hA" if left["glyph"] == "EN" else "I-hA",
            "ideal_difference": ideal, "residual_terms": terms, "identity_residual": residual,
            "mean_change_at_reset": states["after_reset"]["weighted_mean_offset"]-states["before"]["weighted_mean_offset"],
            "ideal_mean_increment": dot(metric, _subtract(ideal, states["before"]["epi_difference"]))/sum(metric),
            "scope": "Pressure was generated before the reset; this is not refreshed Euler at the reset endpoint"}


def run_study(window_path=WINDOW_PATH, native_path=window.NATIVE_PATH, *,
              expected_window_sha256=WINDOW_SHA256, expected_native_sha256=window.NATIVE_SHA256):
    retained, original, bindings = load_evidence(
        window_path, native_path, expected_window_sha256=expected_window_sha256,
        expected_native_sha256=expected_native_sha256)
    branches = []
    for branch in retained["branches"]:
        audits = []
        for ordinal, glyph, time in SELECTED:
            step = window.expand_record(branch["steps"][ordinal], retained["record_pool"])
            result = audit_reset_step(step, original)
            if result["glyph"] != glyph or result["end_time"] != time:
                raise ValueError("selected retained reset identity mismatch")
            audits.append(result)
        branches.append({"branch": branch["branch"], "resets": audits})
    paired = [audit_pair(a, b, original.metric_weights) for a, b in zip(
        branches[0]["resets"], branches[1]["resets"], strict=True)]
    for binding in bindings.values():
        if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("retained input changed during offline audit")
    return {"historical_inputs": bindings, "branches": branches, "paired": paired,
            "original_reference": asdict(original), "native_calls": 0, "new_trajectories": 0,
            "autonomous_maintenance_certified": False,
            "scope": "Four retained reset receipts only; no new horizon, policy or empirical claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-input", type=Path, default=WINDOW_PATH)
    parser.add_argument("--native-input", type=Path, default=window.NATIVE_PATH)
    parser.add_argument("--expected-window-sha256", default=WINDOW_SHA256)
    parser.add_argument("--expected-native-sha256", default=window.NATIVE_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/thol_retained_reset_audit_2026_09_18.json")
    args = parser.parse_args()
    if args.output.resolve() in (args.window_input.resolve(), args.native_input.resolve()):
        raise ValueError("output must not overwrite retained inputs")
    scope = ("src/tnfr", "benchmarks/thol_retained_reset_audit.py", "benchmarks/thol_native_policy_window.py",
             "benchmarks/thol_native_runtime_response.py", "benchmarks/thol_full_state_response.py",
             "benchmarks/thol_family_closure.py", "benchmarks/thol_pressure_feedback.py")
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(
        args.window_input, args.native_input, expected_window_sha256=args.expected_window_sha256,
        expected_native_sha256=args.expected_native_sha256)
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during offline reset audit")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.b-retained-EN-AL-reset-accounting", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version()}, graph_construction="No graph reconstruction; authenticated retained records",
        capacity_specification="Retained positive capacities and original target metric", solver="Exact rational offline accounting",
        timestep=None, seed=None, result_status=ClaimStatus.DERIVED, operator_sequence=("reception", "emission"),
        telemetry=("ordered reset maps", "local defects", "pre-generated pressure", "paired mean and shape"),
        controls=("zero native calls", "pinned retained inputs", "unchanged original target"), artifacts=(str(args.output),))
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(json.dumps(_payload(report), indent=2, allow_nan=False)+"\n"))
    print(f"Wrote retained reset audit to {args.output}")


if __name__ == "__main__":
    main()
