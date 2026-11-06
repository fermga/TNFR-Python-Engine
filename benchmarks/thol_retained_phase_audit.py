"""Conditioning and transported-label audit of one retained phase boundary.

Only detached coordination calculations run, with the effective gains already
recorded after adaptive selection. No native step, future state, alternate
force, or adaptive-history equivalence is inferred.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.thol_native_runtime_response import _tetrad  # noqa: E402
from benchmarks.thol_preparation_policy import _transport_state  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.alias import get_theta_attr  # noqa: E402
from tnfr.constants.aliases import (
    ALIAS_EPI,
    ALIAS_VF,
    ALIAS_THETA,
    ALIAS_DNFR,
)  # noqa: E402
from tnfr.dynamics import coordination  # noqa: E402
from tnfr.mathematics.unified_numerical import compute_phase_difference  # noqa: E402
from tnfr.metrics import trig, trig_cache  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

INPUT = ROOT / "artifacts/research/thol_native_runtime_response_2026_09_18.json"
EXPECTED_SHA256 = "71252d116d8933d15a797707ed9f44ed865406422da6db492b48f6989d739c95"
ROTATION = 1
TERMS = 40  # Arithmetic resource choice, not an epsilon or physical threshold.


def trig_bounds(value):
    """Enclose exact sin/cos of a represented angle by rational Taylor bounds.

    This reuses the degree-80 cosine/degree-81 sine construction of the earlier
    retained-resultant audit. The next identically zero Taylor coefficients permit
    remainder bounds |x|**82/82! and |x|**83/83!, respectively. No range reduction
    or binary64 trigonometric accuracy premise enters these bounds.
    """
    if type(value) not in (int, float, Fraction) or isinstance(value, bool):
        raise ValueError("angle must be an exact or represented finite real")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("angle must be finite")
    x = Fraction(value)
    if abs(x) > 8:
        raise ValueError("retained-angle arithmetic domain is [-8, 8]")
    square, cosine, sine, ct, st = x * x, Fraction(1), x, Fraction(1), x
    for k in range(1, TERMS + 1):
        ct *= -square / ((2 * k - 1) * (2 * k))
        st *= -square / ((2 * k) * (2 * k + 1))
        cosine += ct
        sine += st
    ec, es = abs(x) ** 82 / math.factorial(82), abs(x) ** 83 / math.factorial(83)
    return (cosine - ec, cosine + ec), (sine - es, sine + es)


def resultant_enclosure(phases):
    """Exact rectangle and squared norm; approximate angles remain displays."""
    if not isinstance(phases, (tuple, list)) or not 1 <= len(phases) <= 64:
        raise ValueError("one through 64 ordered retained angles required")
    values = [trig_bounds(value) for value in phases]
    real, imag = (
        tuple(sum(row[channel][j] for row in values) / len(values) for j in (0, 1))
        for channel in (0, 1)
    )

    def square(interval):
        low, high = interval
        return (
            Fraction(0) if low <= 0 <= high else min(low * low, high * high),
            max(low * low, high * high),
        )

    r2, i2 = square(real), square(imag)
    squared = tuple(a + b for a, b in zip(r2, i2, strict=True))
    c, s = (float(sum(v) / 2) for v in (real, imag))
    return {
        "real": real,
        "imag": imag,
        "squared_norm": squared,
        "nonzero_certified": squared[0] > 0,
        "midpoint_display": {
            "real": c,
            "imag": s,
            "norm": math.hypot(c, s),
            "angle": math.atan2(s, c),
        },
        "scope": "Exact represented-input enclosure; midpoint angle is not a certified angle interval",
    }


def _float_literal(value):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or value[0] != "binary64"
    ):
        raise ValueError("recorded binary64 gain required")
    result = float.fromhex(value[1])
    if not math.isfinite(result):
        raise ValueError("gain must be finite")
    return result


def load_evidence(path=INPUT, *, expected_sha256=EXPECTED_SHA256):
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("expected digest must be lowercase SHA256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("retained native bytes differ")
    report = json.loads(raw)
    manifest = CoreExperimentManifest(**report["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O1.b-generated-native-runtime-response":
        raise ValueError("wrong retained native claim")
    if tuple(branch["branch"] for branch in report["branches"]) != (
        "control",
        "child_emission",
    ):
        raise ValueError("wrong retained branch order")
    return report, {
        "path": str(path),
        "sha256": expected_sha256,
        "historical_manifest": report["manifest"],
    }


def _rotation(nodes, parent_children):
    pairs = tuple(tuple(pair) for pair in parent_children)
    parents, children = tuple(p for p, _ in pairs), tuple(c for _, c in pairs)
    if (
        parents != tuple(range(8))
        or len(set(children)) != 8
        or set(parents) & set(children)
    ):
        raise ValueError("eight actual distinct C8 parent-child pairs required")
    if set(nodes) != set(parents + children) or len(nodes) != 16:
        raise ValueError("node space differs from actual parent-child partition")
    child_by_parent = dict(pairs)
    mapping = {p: (p + ROTATION) % 8 for p in parents}
    mapping.update({c: child_by_parent[mapping[p]] for p, c in pairs})
    return mapping


def _detached_graph(record, mapping, *, node_order=None):
    """Rebuild only the fields consumed by held-gain coordination/readouts."""
    state = record["state"]
    nodes = tuple(state["nodes"])
    if set(mapping) != set(nodes) or len(set(mapping.values())) != len(nodes):
        raise ValueError("complete bijective node transport required")
    graph = nx.Graph()
    order = (
        tuple(mapping[n] for n in nodes) if node_order is None else tuple(node_order)
    )
    if len(order) != len(nodes) or set(order) != set(mapping.values()):
        raise ValueError("node enumeration must contain every transported node once")
    graph.add_nodes_from(order)
    for i, node in enumerate(nodes):
        data = {
            alias[0]: state[key][i]
            for key, alias in (
                ("epi", ALIAS_EPI),
                ("capacity", ALIAS_VF),
                ("phase", ALIAS_THETA),
                ("pressure", ALIAS_DNFR),
            )
        }
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in data.values()
        ):
            raise ValueError("finite scalar channels required")
        graph.add_node(mapping[node], **data)
    graph.add_edges_from(
        (mapping[u], mapping[v], deepcopy(data)) for u, v, data in state["edges"]
    )
    neighbors = dict(record["ordered_neighbors"])
    if set(neighbors) != set(nodes):
        raise ValueError("complete recorded neighbor order required")
    for node in nodes:
        order = tuple(mapping[n] for n in neighbors[node])
        adjacency = graph._adj[mapping[node]]
        if len(order) != len(adjacency) or set(order) != set(adjacency):
            raise ValueError("recorded neighbors differ from support")
        ordered = [(n, adjacency[n]) for n in order]
        adjacency.clear()
        adjacency.update(ordered)
    return graph


def _projection(graph):
    return {
        "nodes": tuple(graph),
        "edges": tuple((u, v, dict(d)) for u, v, d in graph.edges(data=True)),
        "ordered_neighbors": {n: tuple(graph.neighbors(n)) for n in graph},
        "phase": tuple(get_theta_attr(graph.nodes[n]) for n in graph),
    }


def compare_phases(base, other):
    """Node-aligned circular shifts and residual after one common offset.

    The relative residual is computed directly from the two anchor-relative
    patterns, not by assuming that every output difference is a gauge rotation.
    Binary64 circular differences are numerical readouts, not exact symmetry proofs.
    """
    if not len(base) or len(base) != len(other):
        raise ValueError("nonempty aligned phase vectors required")
    shifts = tuple(map(float, compute_phase_difference(other, base)))
    p = compute_phase_difference(base, [base[0]] * len(base))
    q = compute_phase_difference(other, [other[0]] * len(other))
    relative = tuple(map(float, compute_phase_difference(q, p)))
    return {
        "node_aligned_shifts": shifts,
        "anchor_common_shift": shifts[0],
        "relative_pattern_residual": relative,
        "max_relative_residual": max(map(abs, relative)),
        "exact_represented_vectors_equal": tuple(base) == tuple(other),
        "scope": "No epsilon-based gauge certification; shared circular readout at represented inputs",
    }


def _evaluate(record, gains, mapping, *, scalar=False, canonical_order=False):
    graph = _detached_graph(
        record,
        mapping,
        node_order=record["state"]["nodes"] if canonical_order else None,
    )
    initial = _projection(graph)
    with ExitStack() as context:
        if scalar:
            for owner in (coordination, trig, trig_cache):
                context.enter_context(patch.object(owner, "np", None))
        cache = trig_cache.get_trig_cache(graph)
        cosines = [cache.cos[n] for n in graph]
        sines = [cache.sin[n] for n in graph]
        mean = (lambda values: math.fsum(values) / len(values)) if scalar else np.mean
        c, s = float(mean(cosines)), float(mean(sines))
        direction = math.atan2(s, c) if scalar else float(np.arctan2(s, c))
        coordination.coordinate_global_local_phase(graph, gains[0], gains[1], n_jobs=1)
    phases = tuple(
        get_theta_attr(graph.nodes[mapping[n]]) for n in record["state"]["nodes"]
    )
    tetrad = _tetrad(graph)
    aligned = {
        key: (
            {n: value[mapping[n]] for n in record["state"]["nodes"]}
            if isinstance(value, dict)
            else value
        )
        for key, value in tetrad["fields"].items()
    }
    return {
        "initial": initial,
        "phase": phases,
        "tetrad": tetrad,
        "aligned_tetrad_fields": aligned,
        "floating_global_resultant": {
            "real": c,
            "imag": s,
            "norm": math.hypot(c, s),
            "angle": direction,
        },
        "scope": "Detached held-effective-gain coordination; scalar patches cover trig, means and coordination only",
    }


def _compare_fields(base, other):
    result = {}
    for key, left in base["aligned_tetrad_fields"].items():
        right = other["aligned_tetrad_fields"][key]
        pairs = (
            [(left[n], right[n]) for n in left]
            if isinstance(left, dict)
            else [(left, right)]
        )
        differences = tuple(
            Fraction(_float_literal(b)) - Fraction(_float_literal(a)) for a, b in pairs
        )
        result[key] = {
            "represented_equal": left == right,
            "max_absolute_represented_difference": max(map(abs, differences)),
            "scope": "Source-node-aligned shared diagnostics; represented arithmetic differences, no tolerance claim",
        }
    return result


def audit_boundary(row, parent_children):
    """Audit one complete retained boundary without changing its input record."""
    saved = deepcopy(row)
    if (
        row["boundary"] != "coordinate_global_local_phase"
        or row["outcome"] != "completed"
    ):
        raise ValueError("completed phase boundary required")
    before, after = row["before"], row["after"]
    nodes = tuple(before["state"]["nodes"])
    if tuple(after["state"]["nodes"]) != nodes:
        raise ValueError("phase boundary changed node space")
    mapping = _rotation(nodes, parent_children)
    identity = dict(zip(nodes, nodes, strict=True))
    gains = tuple(
        _float_literal(after["graph_attributes"][key])
        for key in ("PHASE_K_GLOBAL", "PHASE_K_LOCAL")
    )
    base = _evaluate(before, gains, identity)
    moved = _evaluate(before, gains, mapping)
    reordered = _evaluate(before, gains, mapping, canonical_order=True)
    scalar = _evaluate(before, gains, identity, scalar=True)
    if base["phase"] != tuple(after["state"]["phase"]):
        raise ValueError(
            "held NumPy replay differs from the actual archived coordination output"
        )
    inverse = {v: k for k, v in mapping.items()}
    transported = _transport_state(moved["initial"], inverse) == _transport_state(
        base["initial"], identity
    )
    if not transported or moved["phase"] != base["phase"]:
        raise ValueError(
            "transported-label control changed the aligned phase calculation"
        )
    for key in ("phase", "ordered_neighbors"):

        def by_node(projection):
            if key == "phase":
                return dict(zip(projection["nodes"], projection[key], strict=True))
            return projection[key]

        if by_node(moved["initial"]) != by_node(reordered["initial"]):
            raise ValueError(
                "canonical enumeration changed the named mathematical phase input"
            )
    if row != saved:
        raise RuntimeError("retained input mutated during detached audit")
    comparison = compare_phases(base["phase"], scalar["phase"])
    return {
        "source_boundary": row,
        "effective_gains": gains,
        "rotation": ROTATION,
        "node_mapping": tuple(mapping.items()),
        "transported_source_equal": transported,
        "numpy_replay_matches_archive": True,
        "base_numpy": base,
        "transported_numpy": moved,
        "canonical_order_numpy": reordered,
        "canonical_order_vs_numpy": compare_phases(base["phase"], reordered["phase"]),
        "enumeration_scope": "Same k->k+1 labels, named scalar values and transported neighbor order; canonical node enumeration alone changes reduction order",
        "scalar_arithmetic": scalar,
        "scalar_vs_numpy": comparison,
        "tetrad_scalar_equals_numpy": {
            key: scalar["aligned_tetrad_fields"][key]
            == base["aligned_tetrad_fields"][key]
            for key in ("phi_s", "grad_phi", "curv_phi", "xi_c")
        },
        "tetrad_canonical_order_equals_numpy": {
            key: reordered["aligned_tetrad_fields"][key]
            == base["aligned_tetrad_fields"][key]
            for key in ("phi_s", "grad_phi", "curv_phi", "xi_c")
        },
        "tetrad_canonical_order_comparison": _compare_fields(base, reordered),
        "tetrad_scalar_comparison": _compare_fields(base, scalar),
        "represented_resultant": resultant_enclosure(before["state"]["phase"]),
        "ideal_resultant": {
            "exact_zero": True,
            "hypothesis": "Exact phases k*pi/4 for k=0,...,7 duplicated; each antipodal pair cancels",
        },
        "conditioning": "At nonzero mean z, the differential norm of arg is 1/abs(z). No global angular error bound asserted.",
        "live_graph_writes": 0,
        "native_steps": 0,
        "scope": "Finite retained-input arithmetic/label sensitivity; no physical emergence or robust symmetry breaking",
    }


def run_study(path=INPUT, *, expected_sha256=EXPECTED_SHA256):
    report, binding = load_evidence(path, expected_sha256=expected_sha256)
    rows = []
    for branch in report["branches"]:
        candidates = [
            row
            for row in branch["native_trace"]["boundaries"]
            if row["boundary"] == "coordinate_global_local_phase"
        ]
        if branch["status"] != "executed" or len(candidates) != 1:
            raise ValueError(
                "one completed coordination boundary per retained branch required"
            )
        rows.append(candidates[0])
    if rows[0]["before"]["state"]["phase"] != rows[1]["before"]["state"]["phase"]:
        raise ValueError("retained branches do not share the declared phase input")
    lineage = report["replayed_prior_reports"][0]["lineage"]["parent_children"]
    result = audit_boundary(rows[0], lineage)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("retained bytes changed during audit")
    return {
        "input_evidence": binding,
        "selected_branch": "control",
        "both_input_phase_vectors_equal": True,
        "audit": result,
        "experimental_status": "No empirical correspondence tested",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/thol_retained_phase_audit_2026_09_18.json",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise ValueError("output must not overwrite retained input")
    scope = (
        "src/tnfr",
        "benchmarks/thol_retained_phase_audit.py",
        "benchmarks/thol_native_runtime_response.py",
        "benchmarks/thol_preparation_policy.py",
        "benchmarks/thol_pressure_feedback.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(args.input, expected_sha256=args.expected_sha256)
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during retained phase audit")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-retained-phase-conditioning",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        graph_construction="Detached projection of authenticated coordination boundary; one cycle rotation 1",
        capacity_specification="Retained scalar channels; no capacity evolution",
        solver="Exact rational Taylor enclosure and detached held-gain coordination",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=(),
        telemetry=("relative phase pattern", "tetrad", "exact resultant enclosure"),
        controls=(
            "transported node and neighbor order",
            "scalar/numpy arithmetic",
            "no trajectory extension",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    result = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(
        args.output,
        lambda stream: stream.write(
            json.dumps(_payload(result), indent=2, allow_nan=False) + "\n"
        ),
    )
    print(f"Wrote retained phase audit to {args.output}")


if __name__ == "__main__":
    main()
