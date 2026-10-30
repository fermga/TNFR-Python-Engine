"""Five predeclared native steps after authenticated causal response replays."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
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

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks import thol_native_runtime_response as native  # noqa: E402
from benchmarks.thol_distributed_target import _target  # noqa: E402
from benchmarks.thol_full_state_response import (  # noqa: E402
    BRANCHES,
    COMMON_FIELDS,
    _add,
    _paired_delta,
    _payload,
    _reference,
    _subtract,
    replay_response_branch,
)
from tnfr.constants.canonical import DYNAMICS_SI_HI_THRESHOLD_CANONICAL  # noqa: E402
from tnfr.physics.support_transport import _laplacian  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

NATIVE_PATH = ROOT / "artifacts/research/thol_native_runtime_response_2026_09_18.json"
NATIVE_SHA256 = "71252d116d8933d15a797707ed9f44ed865406422da6db492b48f6989d739c95"
WINDOW_STEPS = 5
DT = Fraction(1, 4)
PROTOCOL = {
    "start_time": 1.75,
    "maximum_steps_per_branch": WINDOW_STEPS,
    "dt": DT,
    "completed_endpoint": 3.0,
    "use_Si": True,
    "apply_glyphs": True,
    "refusal_stops_branch": True,
    "retries": 0,
    "parameter_tuning": False,
    "scope": "Five additional calls, separate from the historical native-step replay",
}


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


class RecordPool:
    """Lossless canonical-JSON tree interning; no scientific field is dropped."""

    def __init__(self, *, max_nodes=500000):
        if type(max_nodes) is not int or max_nodes <= 0:
            raise ValueError("max_nodes must be a positive integer")
        self.nodes = {}
        self.max_nodes = max_nodes

    def pack(self, value):
        def encode(item):
            if isinstance(item, dict):
                body = {
                    "kind": "dict",
                    "items": [[key, encode(v)] for key, v in item.items()],
                }
            elif isinstance(item, list):
                body = {"kind": "list", "items": [encode(v) for v in item]}
            else:
                if not (item is None or type(item) in (bool, int, str, float)):
                    raise TypeError("pool accepts canonical JSON values only")
                if type(item) is float and not math.isfinite(item):
                    raise ValueError(
                        "nonfinite scalar must retain an explicit literal representation"
                    )
                return item
            digest = hashlib.sha256(_bytes(body)).hexdigest()
            if digest not in self.nodes:
                if len(self.nodes) >= self.max_nodes:
                    raise RuntimeError(
                        "lossless record pool exhausted its finite node budget"
                    )
                self.nodes[digest] = body
            elif self.nodes[digest] != body:
                raise RuntimeError("record digest collision")
            return {"$record": digest}

        return encode(_payload(value))


def expand_record(reference, nodes):
    """Validate all referenced digests, then return detached canonical JSON."""
    active = set()

    def decode(value):
        if not isinstance(value, dict):
            if value is None or type(value) in (bool, int, str, float):
                if type(value) is float and not math.isfinite(value):
                    raise ValueError("nonfinite compact scalar")
                return value
            raise ValueError("invalid compact scalar")
        if set(value) != {"$record"} or type(value["$record"]) is not str:
            raise ValueError("invalid compact reference")
        key = value["$record"]
        if (
            len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
            or key not in nodes
            or key in active
        ):
            raise ValueError("missing, malformed or cyclic compact reference")
        body = nodes[key]
        if hashlib.sha256(_bytes(body)).hexdigest() != key or set(body) != {
            "kind",
            "items",
        }:
            raise ValueError("compact record digest/schema mismatch")
        active.add(key)
        if body["kind"] == "list":
            result = [decode(v) for v in body["items"]]
        elif body["kind"] == "dict":
            entries = body["items"]
            if any(
                not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str
                for pair in entries
            ):
                raise ValueError("invalid compact mapping")
            if len({pair[0] for pair in entries}) != len(entries):
                raise ValueError("duplicate compact mapping key")
            result = {key: decode(v) for key, v in entries}
        else:
            raise ValueError("invalid compact record kind")
        active.remove(key)
        return result

    return decode(reference)


def load_evidence(
    native_path=NATIVE_PATH,
    full_response_path=native.PRIOR_PATH,
    *,
    expected_native_sha256=NATIVE_SHA256,
    expected_full_sha256=native.PRIOR_SHA256,
):
    full, full_binding = native.load_prior_evidence(
        full_response_path, expected_sha256=expected_full_sha256
    )
    if (
        type(expected_native_sha256) is not str
        or len(expected_native_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_native_sha256)
    ):
        raise ValueError("native expected digest must be lowercase SHA256")
    raw = Path(native_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_native_sha256:
        raise ValueError("native report digest mismatch")
    report = json.loads(raw)
    manifest = CoreExperimentManifest(**report["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O1.b-generated-native-runtime-response"
        or report["prior_evidence"]["sha256"] != expected_full_sha256
        or report["prior_evidence"]["historical_manifest"] != full["manifest"]
        or tuple(row["branch"] for row in report["branches"]) != BRANCHES
        or any(row["status"] != "executed" for row in report["branches"])
    ):
        raise ValueError("native report claim, ancestry or completed branches differ")
    binding = {
        "native": {
            "path": str(native_path),
            "sha256": expected_native_sha256,
            "historical_manifest": report["manifest"],
        },
        "full_response": full_binding,
    }
    return report, full, binding


def _admit_native_replay(current, historical, children):
    now, old = _payload(current), deepcopy(historical)
    full_equal, utc = now == old, []
    if current["branch"] == "child_emission":
        for payload, label in ((now, "current"), (old, "historical")):
            records = [("before", payload["before"]), ("endpoint", payload["endpoint"])]
            for i, row in enumerate(payload["native_trace"]["boundaries"]):
                records.extend(
                    (f"native_trace.boundaries[{i}].{side}", row[side])
                    for side in ("before", "after")
                )
            origins = {}
            for path, record in records:
                attrs = dict(record["node_attributes"])
                for child in children:
                    data = attrs[child]
                    value = data["emission_timestamp"]
                    if (
                        type(value) is not str
                        or datetime.fromisoformat(value).utcoffset() != timedelta(0)
                        or data["_emission_origin"] != value
                        or data["_structural_lineage"]["origin"] != value
                        or (child in origins and origins[child] != value)
                    ):
                        raise ValueError(
                            "native trace has inconsistent actual-child UTC origin"
                        )
                    origins[child] = value
                    for key in ("emission_timestamp", "_emission_origin"):
                        data[key] = "<invocation UTC origin>"
                    data["_structural_lineage"]["origin"] = "<invocation UTC origin>"
            utc.append(
                {
                    "payload": label,
                    "origins": origins,
                    "record_paths": [path for path, _ in records],
                }
            )
    if now != old:
        raise ValueError(
            "native scientific replay differs outside explicit child UTC record locations"
        )
    return {
        "full_payload_equal": full_equal,
        "scientific_payload_equal": True,
        "utc_records": utc,
        "scope": "Only three actual-child UTC aliases at enumerated record locations; all other fields exact",
    }


def _fseq(values):
    return tuple(Fraction(v) for v in values)


def _one(trace, name):
    values = [
        row
        for row in trace["boundaries"]
        if row["boundary"] == name and row["outcome"] == "completed"
    ]
    return values[0] if len(values) == 1 else None


def _target_record(original, record, capture):
    pattern = native._pattern(original, record)
    result = {"pattern": pattern, "compatibility_available": False}
    if pattern["available"] and capture["available"]:
        try:
            current = _reference(capture["payload"])
            target = _target(original, current, capture["payload"])
        except ValueError as exc:
            result["reason"] = str(exc)
        else:
            result.update(
                compatibility_available=True,
                target=asdict(target),
                scope="Original z0/H0 unchanged; current profile is a separately labeled derived observation",
            )
    return result


def _activity(capture):
    if not capture["available"]:
        return capture
    obs = capture["payload"]["observation"]
    snap = obs["snapshot"]
    return {
        "available": True,
        "positive_capacity": all(v > 0 for v in snap["capacity"]),
        "stored_nodal_rate": snap["rate"],
        "fresh_kernel_nodal_rate": tuple(
            nu * p
            for nu, p in zip(snap["capacity"], obs["full_kernel_pressure"], strict=True)
        ),
    }


def _adaptation(trace):
    row = _one(trace, "adapt_vf_after_structural_stability")
    if row is None:
        return {"available": False}
    before, after = row["before"], row["after"]
    cfg = before["resolved_adaptation_configuration"]
    # Exact literal mapping retains the already configured threshold values.
    graph_attrs = before["graph_attributes"]
    selector = graph_attrs.get("SELECTOR_THRESHOLDS", {})
    fallback = graph_attrs.get("GLYPH_THRESHOLDS", {})
    raw = selector.get("si_hi", fallback.get("hi", DYNAMICS_SI_HI_THRESHOLD_CANONICAL))
    si_hi = (
        float.fromhex(raw[1])
        if isinstance(raw, (tuple, list)) and raw[0] == "binary64"
        else float(raw)
    )
    battrs, aattrs = dict(before["node_attributes"]), dict(after["node_attributes"])
    rows = []
    for i, node in enumerate(before["state"]["nodes"]):
        previous = battrs[node].get("stable_count", 0)
        observed = aattrs[node].get("stable_count", 0)
        qualifies = (
            before["stored_Si"][i] >= si_hi
            and abs(before["state"]["pressure"][i]) <= cfg["EPS_DNFR_STABLE"]
        )
        expected = previous + 1 if qualifies else 0
        if observed != expected:
            raise RuntimeError("adaptation counter differs from its declared gate")
        rows.append(
            {
                "node": node,
                "Si": before["stored_Si"][i],
                "pressure": before["state"]["pressure"][i],
                "before_count": previous,
                "after_count": observed,
                "qualifies": qualifies,
                "eligible": observed >= cfg["VF_ADAPT_TAU"],
                "nu_before": before["state"]["capacity"][i],
                "nu_after": after["state"]["capacity"][i],
            }
        )
    return {
        "available": True,
        "configuration": {**cfg, "si_hi": si_hi},
        "nodes": rows,
        "scope": "Si was prepared before glyphs; this gate reads its stored value and adaptation-entry pressure",
    }


def _il_inputs(step):
    trace = step["native_trace"]
    prepared, integration = _one(trace, "_prepare_dnfr"), _one(trace, "integrate")
    capture = trace["captures"].get("pressure_generation", {"available": False})
    if (
        step["status"] != "executed"
        or prepared is None
        or integration is None
        or not capture["available"]
    ):
        return {
            "available": False,
            "reason": "Completed generation/integration/capture required",
        }
    generation = prepared["after"]["state"]
    consumed = integration["before"]["state"]
    obs = capture["payload"]["observation"]
    snap = obs["snapshot"]
    nodes = tuple(generation["nodes"])
    if (
        tuple(snap["nodes"]) != nodes
        or tuple(snap["epi"]) != _fseq(generation["epi"])
        or tuple(snap["capacity"]) != _fseq(generation["capacity"])
        or tuple(snap["stored_pressure"]) != _fseq(generation["pressure"])
        or tuple(obs["phase"]) != _fseq(generation["phase"])
    ):
        raise ValueError("generation capture differs from its actual native boundary")
    calls = [row for row in trace["boundaries"] if row["boundary"] == "apply_glyph"]
    flags = {
        "all_nodes_one_IL": len(calls) == len(nodes)
        and tuple(r["node"] for r in calls) == nodes
        and all(r["glyph"] == "IL" and r["outcome"] == "completed" for r in calls),
        "same_node_space": tuple(consumed["nodes"])
        == nodes
        == tuple(step["endpoint"]["state"]["nodes"]),
        "one_generated_pressure_refresh": len(
            [r for r in trace["boundaries"] if r["boundary"] == "_refresh_delta_nfr"]
        )
        == 1,
    }
    if not all(flags.values()):
        return {
            "available": False,
            "flags": flags,
            "reason": "Restricted all-node IL policy not observed",
        }
    anchors = (
        [generation]
        + [r[side]["state"] for r in calls for side in ("before", "after")]
        + [consumed]
    )
    flags["epi_capacity_support_unchanged_since_generation"] = all(
        all(
            state[key] == generation[key]
            for key in ("nodes", "epi", "capacity", "edges")
        )
        for state in anchors
    )
    flags["no_other_pressure_write"] = (
        generation["pressure"] == calls[0]["before"]["state"]["pressure"]
    )
    for i, row in enumerate(calls):
        left, right = row["before"]["state"], row["after"]["state"]
        flags["no_other_pressure_write"] &= all(
            left["pressure"][j] == right["pressure"][j]
            for j in range(len(nodes))
            if j != i
        )
        following = calls[i + 1]["before"]["state"] if i + 1 < len(calls) else consumed
        flags["no_other_pressure_write"] &= right["pressure"] == following["pressure"]
    factors = tuple(Fraction(r["resolved_IL_retention"]) for r in calls)
    flags["common_retention_within_branch"] = len(set(factors)) == 1
    try:
        flags["declared_binary64_IL_product"] = all(
            Fraction(row["after"]["state"]["pressure"][i])
            == Fraction(float(factor * Fraction(row["before"]["state"]["pressure"][i])))
            for i, (row, factor) in enumerate(zip(calls, factors, strict=True))
        )
    except OverflowError:
        flags["declared_binary64_IL_product"] = False
    flags["no_pre_generation_epi_change"] = (
        step["before"]["state"]["epi"] == generation["epi"]
    )
    if not all(flags.values()):
        return {
            "available": False,
            "flags": flags,
            "reason": "Generated/consumed boundary conditions failed",
        }
    a = factors[0]
    generated, held = _fseq(generation["pressure"]), _fseq(consumed["pressure"])
    epsilon_p = tuple(
        p - (obs["epi_weight"] * g + f)
        for p, g, f in zip(generated, snap["epi_gradient"], obs["forcing"], strict=True)
    )
    epsilon_il = tuple(p - a * q for p, q in zip(held, generated, strict=True))
    xg, xi, xf = (
        _fseq(record["epi"])
        for record in (
            consumed,
            integration["after"]["state"],
            step["endpoint"]["state"],
        )
    )
    remainder = tuple(
        y - x - DT * nu * p
        for y, x, nu, p in zip(xi, xg, snap["capacity"], held, strict=True)
    )
    return {
        "available": True,
        "flags": flags,
        "generation_observation": obs,
        "retention": a,
        "generated_pressure": generated,
        "consumed_pressure": held,
        "epsilon_pressure": epsilon_p,
        "epsilon_IL": epsilon_il,
        "integration_remainder": remainder,
        "postintegration_epi_change": _subtract(xf, xi),
        "generated_phase": _fseq(generation["phase"]),
        "consumed_phase": _fseq(consumed["phase"]),
        "scope": "Phase changes after generation are retained; stored pressure is not retrospectively refreshed",
    }


def continue_window(graph, prior_report, first_step, *, pool):
    if (
        first_step["status"] != "executed"
        or native._record(graph) != first_step["endpoint"]
    ):
        raise ValueError(
            "window requires the actual completed replayed native endpoint"
        )
    if native._state(graph)["time"] != 1.75:
        raise ValueError("five additional calls start at t=1.75")
    original = _reference(prior_report["prefix"]["coupling"]["refreshed_forcing"])
    records = []
    for ordinal in range(WINDOW_STEPS):
        before, capture0 = native._record(graph), native._capture(graph)
        tetrad0 = native._tetrad(graph)
        trace = native._trace_step(graph, capture_generation=True)
        endpoint, capture1 = native._record(graph), native._capture(graph)
        if trace["status"] == "executed" and endpoint["state"]["time"] != 1.75 + (
            ordinal + 1
        ) * float(DT):
            raise RuntimeError("completed native call has an unexpected time")
        step = {
            "ordinal": ordinal,
            "status": trace["status"],
            "before": before,
            "endpoint": endpoint,
            "before_capture": capture0,
            "endpoint_capture": capture1,
            "native_trace": trace,
            "before_tetrad": tetrad0,
            "endpoint_tetrad": native._tetrad(graph),
            "old_target_before": _target_record(original, before, capture0),
            "old_target_endpoint": _target_record(original, endpoint, capture1),
            "activity": _activity(capture1),
            "adaptation": _adaptation(trace),
        }
        step["policy_inputs"] = _il_inputs(step)
        packed = pool.pack(step)
        if expand_record(packed, pool.nodes) != _payload(step):
            raise RuntimeError("lossless step compaction failed")
        records.append(packed)
        if trace["status"] != "executed":
            break
    completed = sum(
        expand_record(ref, pool.nodes)["status"] == "executed" for ref in records
    )
    return {
        "branch": prior_report["branch"],
        "steps": records,
        "completed_steps": completed,
        "attempted_steps": len(records),
        "status": "completed" if completed == WINDOW_STEPS else "refused",
        "final_time": native._state(graph)["time"],
        "final_record": pool.pack(native._record(graph)),
    }


def paired_step(left, right, metric):
    original_space = all(
        step[key]["pattern"]["available"]
        for step in (left, right)
        for key in ("old_target_before", "old_target_endpoint")
    )
    if (
        not original_space
        or len(left["before"]["state"]["nodes"]) != len(metric)
        or left["status"] != right["status"]
        or left["status"] != "executed"
        or left["endpoint"]["state"]["time"] != right["endpoint"]["state"]["time"]
        or left["before"]["state"]["time"] != right["before"]["state"]["time"]
        or left["before"]["state"]["nodes"] != right["before"]["state"]["nodes"]
        or left["endpoint"]["state"]["nodes"] != left["before"]["state"]["nodes"]
        or right["endpoint"]["state"]["nodes"] != left["before"]["state"]["nodes"]
    ):
        return {
            "available": False,
            "reason": "Common completed times and ordered node space required",
        }
    start = _paired_delta(
        _fseq(left["before"]["state"]["epi"]),
        _fseq(right["before"]["state"]["epi"]),
        metric,
    )
    end = _paired_delta(
        _fseq(left["endpoint"]["state"]["epi"]),
        _fseq(right["endpoint"]["state"]["epi"]),
        metric,
    )
    result = {
        "available": True,
        "time": left["endpoint"]["state"]["time"],
        "before": start,
        "endpoint": end,
        "common_IL_gate": {"available": False},
    }
    l, r = left["policy_inputs"], right["policy_inputs"]
    if not l["available"] or not r["available"]:
        result["common_IL_gate"][
            "reason"
        ] = "A branch failed its actual IL policy conditions"
        return result
    lo, ro = l["generation_observation"], r["generation_observation"]
    flags = {
        key: lo["snapshot"][key] == ro["snapshot"][key]
        for key in ("nodes", "conductance", "support_neighbors", "capacity")
    }
    flags.update(
        epi_weight=lo["epi_weight"] == ro["epi_weight"],
        forcing=lo["forcing"] == ro["forcing"],
        retention=l["retention"] == r["retention"],
    )
    if not all(flags.values()):
        result["common_IL_gate"] = {
            "available": False,
            "flags": flags,
            "reason": "Generation coefficients or source differ",
        }
        return result
    snap = lo["snapshot"]
    edges = tuple((i, j, Fraction(w)) for i, j, w in snap["conductance"])
    strengths = [Fraction(0)] * len(metric)
    for i, j, w in edges:
        strengths[i] += w
    nu, e, a = (
        _fseq(snap["capacity"]),
        Fraction(lo["epi_weight"]),
        Fraction(l["retention"]),
    )
    lap = _laplacian(edges, start["epi_difference"])
    action = tuple(
        e * v * z / d if d else Fraction(0)
        for v, z, d in zip(nu, lap, strengths, strict=True)
    )
    ideal = tuple(
        d - DT * a * v for d, v in zip(start["epi_difference"], action, strict=True)
    )
    pressure = tuple(
        DT * v * a * (Fraction(b) - Fraction(c))
        for v, b, c in zip(
            nu, r["epsilon_pressure"], l["epsilon_pressure"], strict=True
        )
    )
    il = tuple(
        DT * v * (Fraction(b) - Fraction(c))
        for v, b, c in zip(nu, r["epsilon_IL"], l["epsilon_IL"], strict=True)
    )
    integration = _subtract(
        _fseq(r["integration_remainder"]), _fseq(l["integration_remainder"])
    )
    post = _subtract(
        _fseq(r["postintegration_epi_change"]), _fseq(l["postintegration_epi_change"])
    )
    residual = _subtract(
        end["epi_difference"],
        _add(_add(ideal, pressure), _add(il, _add(integration, post))),
    )
    if any(residual):
        raise RuntimeError("actual conditional common-IL paired identity failed")
    result["common_IL_gate"] = {
        "available": True,
        "flags": flags,
        "retention": a,
        "generator_action_on_difference": action,
        "ideal_difference": ideal,
        "pressure_residual_term": pressure,
        "IL_residual_term": il,
        "integration_residual_difference": integration,
        "postintegration_difference": post,
        "identity_residual": residual,
        "scope": "Conditional finite identity; no future shared-policy claim",
    }
    return result


def run_study(
    native_path=NATIVE_PATH,
    full_response_path=native.PRIOR_PATH,
    *,
    expected_native_sha256=NATIVE_SHA256,
    expected_full_sha256=native.PRIOR_SHA256,
):
    archived, full, bindings = load_evidence(
        native_path,
        full_response_path,
        expected_native_sha256=expected_native_sha256,
        expected_full_sha256=expected_full_sha256,
    )
    graphs, priors, first_steps, admission = [], [], [], []
    for i, name in enumerate(BRANCHES):
        graph, prior = replay_response_branch(name)
        admission.append(
            {"prior_full": native._admit_replay(prior, full["branches"][i])}
        )
        native._admit_replay(prior, archived["replayed_prior_reports"][i])
        graphs.append(graph)
        priors.append(prior)
    if any(priors[0][key] != priors[1][key] for key in COMMON_FIELDS):
        raise ValueError("causal prior common sources differ")
    for i, (graph, prior) in enumerate(zip(graphs, priors, strict=True)):
        first = native.run_native_branch(graph, prior)
        admission[i]["native"] = _admit_native_replay(
            first, archived["branches"][i], prior["lineage"]["children"]
        )
        first_steps.append(first)
    pool = RecordPool()
    replay_refs = [pool.pack(first) for first in first_steps]
    branches = [
        continue_window(graph, prior, first, pool=pool)
        for graph, prior, first in zip(graphs, priors, first_steps, strict=True)
    ]
    metric = priors[0]["original_reference"]["metric_weights"]
    pairs = [
        paired_step(expand_record(a, pool.nodes), expand_record(b, pool.nodes), metric)
        for a, b in zip(branches[0]["steps"], branches[1]["steps"])
    ]
    for binding in bindings.values():
        if (
            hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise RuntimeError("historical input changed during the window")
    return {
        "protocol": PROTOCOL,
        "historical_inputs": bindings,
        "replay_admission": admission,
        "original_reference": pool.pack(priors[0]["original_reference"]),
        "first_step_replays": replay_refs,
        "branches": branches,
        "paired_steps": pool.pack(pairs),
        "record_pool": pool.nodes,
        "pool_contract": "SHA256-addressed canonical JSON trees; expansion retains every recorded field",
        "pool_node_limit": pool.max_nodes,
        "autonomous_maintenance_certified": False,
        "scope": "Fixed five-call window; refusal stops a branch without retry or whole-window rollback",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-input", type=Path, default=NATIVE_PATH)
    parser.add_argument("--full-response-input", type=Path, default=native.PRIOR_PATH)
    parser.add_argument("--expected-native-sha256", default=NATIVE_SHA256)
    parser.add_argument("--expected-full-sha256", default=native.PRIOR_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/thol_native_policy_window_2026_09_18.json",
    )
    args = parser.parse_args()
    if args.output.resolve() in (
        args.native_input.resolve(),
        args.full_response_input.resolve(),
    ):
        raise ValueError("output must not overwrite historical inputs")
    scope = (
        "src/tnfr",
        "benchmarks/thol_native_policy_window.py",
        "benchmarks/thol_native_runtime_response.py",
        "benchmarks/thol_full_state_response.py",
        "benchmarks/thol_distributed_target.py",
        "benchmarks/thol_distributed_transport.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_preparation_policy.py",
        "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/selection_birth_closure.py",
        "benchmarks/capacity_localization.py",
        "benchmarks/structural_target_compatibility.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    result = run_study(
        args.native_input,
        args.full_response_input,
        expected_native_sha256=args.expected_native_sha256,
        expected_full_sha256=args.expected_full_sha256,
    )
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during policy window")
    manifest = CoreExperimentManifest(
        claim_id="O3.a-generated-five-step-native-policy-window",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Authenticated independent causal THOL/UM/optional AL and native-step replays",
        capacity_specification="Actual native adaptation and counters; unchanged defaults",
        solver="Five native composite .25 steps maximum per branch",
        timestep=0.25,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(
            "coherence",
            "dissonance",
            "self_organization",
            "coupling",
            "emission",
        ),
        telemetry=(
            "complete compact native traces",
            "generated and consumed pressure",
            "conditional paired map",
            "original target",
        ),
        controls=(
            "fixed window without retries",
            "old z0/H0",
            "authenticated historical byte inputs",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        **result,
        "experimental_status": "No empirical correspondence tested",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(
        args.output,
        lambda stream: stream.write(
            json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
        ),
    )
    print(f"Wrote native policy window to {args.output}")


if __name__ == "__main__":
    main()
