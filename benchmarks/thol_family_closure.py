"""Test ancestry-family state sufficiency on four authenticated held models.

This detached analysis consumes complete retained target/lineage reports. It
does not reconstruct a graph, execute operators, simulate, or fit a memory law.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from benchmarks.thol_lineage_coordination import (  # noqa: E402
    COMMON_FIELDS,
    CONTROL_PATH,
    CONTROL_SHA256,
    compare_common_source,
    load_control_evidence,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.physics._cycle_algebra import dot  # noqa: E402
from tnfr.physics.forced_support import (  # noqa: E402
    observe_forced_support_state,
    observe_forced_support_target,
)
from tnfr.physics.structural_morphism import (  # noqa: E402
    _finite_difference,
    _finite_norm,
    _finite_product,
    _intertwining_diagnostics,
    _relative_tolerance,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.research.recorded_support import (  # noqa: E402
    bind_recorded_nodal_state,
    read_recorded_forced_reference,
    read_recorded_forcing,
    read_recorded_support_snapshot,
)
from tnfr.utils.io import safe_write  # noqa: E402

LINEAGE_PATH = ROOT / "artifacts/research/thol_lineage_coordination_2026_09_18.json"
LINEAGE_SHA256 = "8b7d6e3e217d57566788350955aa8afbba747add07a49b341167c0626b0b848a"
MODEL_ORDER = ("original", "original_parents", "born_children", "all_node_um")


def _json(value):
    return json.loads(json.dumps(_payload(value), allow_nan=False))


def _equal(left, right, label):
    if _json(left) != _json(right):
        raise ValueError(f"retained {label} differs from its reconstructed evidence")


def _digest(value):
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _check_digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected SHA256 must be 64 lowercase hexadecimal characters")


def _snapshot(raw):
    """Compatibility wrapper for the shared strict archival reader."""
    return read_recorded_support_snapshot(raw)


def _reference(raw):
    """Compatibility wrapper for the shared exact reference reconstruction."""
    return read_recorded_forced_reference(raw)


def _capture(raw):
    """Compatibility wrapper; no phase kernel or producer is rerun."""
    return read_recorded_forcing(raw)


def _admit_target(raw, original, current, capture):
    snapshot, observation, components = _capture(capture)
    if (
        observation.epi_weight != current.epi_weight
        or observation.forcing != current.forcing
    ):
        raise ValueError("captured forcing differs from the retained held model")
    value = observe_forced_support_target(
        original,
        current,
        snapshot,
        forcing_components=components,
    )
    _equal(asdict(value), raw, "complete target readout")
    return snapshot, value


def _match_live_state(state, capture, snapshot):
    """Compatibility wrapper for strict raw triad/pressure binding."""
    return bind_recorded_nodal_state(state, capture, snapshot)


def _lineage(branch):
    """Use birth receipts and retained actual parent pointers, not label parsing."""
    birth = branch["prefix"]["birth"]
    pairs = tuple(tuple(pair) for pair in birth["parent_children"])
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("birth receipt must contain nonempty parent-child pairs")
    parents, children = tuple(p for p, _ in pairs), tuple(c for _, c in pairs)
    nodes = tuple(branch["before_optional_event"]["nodes"])
    if (
        len(set(parents + children)) != len(nodes)
        or parents + children != nodes
        or len(set(map(str, nodes))) != len(nodes)
        or parents != tuple(birth["before"]["nodes"])
        or nodes != tuple(birth["after"]["nodes"])
    ):
        raise ValueError("actual ancestry must be disjoint and exhaust ordered support")
    attributes = branch["common_source"]["node_attributes"]
    if tuple(node for node, _ in attributes) != nodes:
        raise ValueError(
            "retained node attributes must preserve the complete node order"
        )
    attrs = dict(attributes)
    _equal(
        branch["common_source"]["state"],
        branch["before_optional_event"],
        "common live source state",
    )
    for state in (
        birth["after"],
        branch["before_optional_event"],
        branch["after_optional_event"],
    ):
        hierarchy = {str(parent): [child] for parent, child in pairs}
        expected_children = {**hierarchy, **{str(child): [] for child in children}}
        if state["hierarchy"] != hierarchy or state["children"] != expected_children:
            raise ValueError("retained hierarchy differs from actual birth pairs")
    for parent, child in pairs:
        if (
            attrs[parent].get("sub_nodes") != [child]
            or attrs[child].get("parent_node") != parent
        ):
            raise ValueError("retained live parentage differs from birth receipt")
    retained = branch["lineage"]
    for key, expected in (
        ("parent_children", pairs),
        ("parents", parents),
        ("children", children),
        ("current_nodes", nodes),
    ):
        _equal(retained[key], expected, f"lineage {key}")
    if (
        retained["live_parentage_verified"] is not True
        or retained["disjoint_and_exhaustive"] is not True
    ):
        raise ValueError("lineage producer did not verify actual parentage")
    selected = parents if branch["branch"] == "original_parents" else children
    _equal(retained["selected_targets"], selected, "selected ancestry cohort")
    return {
        "nodes": nodes,
        "parent_children": pairs,
        "blocks": pairs,
        "family_count": len(pairs),
        "scope": (
            "One declared partition from retained actual birth receipts and parent pointers; "
            "no partition search and no inference from node names."
        ),
    }


def load_evidence(
    control_path=CONTROL_PATH,
    lineage_path=LINEAGE_PATH,
    *,
    expected_control_sha256=CONTROL_SHA256,
    expected_lineage_sha256=LINEAGE_SHA256,
):
    _check_digest(expected_control_sha256)
    _check_digest(expected_lineage_sha256)
    controls, control_binding = load_control_evidence(
        control_path,
        expected_sha256=expected_control_sha256,
    )
    raw = Path(lineage_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_lineage_sha256:
        raise ValueError("retained lineage bytes differ from their expected digest")
    report = json.loads(raw)
    manifest = CoreExperimentManifest(**report["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O1.b-lineage-scoped-UM-fixed-target-response":
        raise ValueError("lineage evidence must preserve its original producer")
    branches = tuple(report["branches"])
    if tuple(row["branch"] for row in branches) != MODEL_ORDER[1:3]:
        raise ValueError("lineage evidence must retain both ordered cohorts")
    if any(row["status"] != "executed" for row in branches):
        raise ValueError("four-model analysis requires both executed lineage branches")
    previous = report["retained_controls"]
    for key in (
        "sha256",
        "byte_count",
        "historical_manifest",
        "historical_source_scope",
    ):
        _equal(previous[key], control_binding[key], f"lineage control binding {key}")
    if previous["historical_producer_preserved"] is not True:
        raise ValueError("lineage evidence lost the historical producer identity")
    comparisons = tuple(compare_common_source(branch, controls) for branch in branches)
    partitions = tuple(_lineage(branch) for branch in branches)
    _equal(partitions[0], partitions[1], "ancestry partition across cohorts")
    return (
        controls,
        branches,
        partitions[0],
        {
            "retained_controls": control_binding,
            "retained_lineage": {
                "path": str(lineage_path),
                "sha256": actual,
                "byte_count": len(raw),
                "historical_manifest": report["manifest"],
                "historical_source_scope": report["source_scope"],
                "historical_producer_preserved": True,
            },
            "common_source_comparison": comparisons,
        },
    )


def _numerical_crosscheck(reference, blocks, exact):
    """Materialize the existing quotient identities in binary64, without a graph."""
    nodes = reference.source.nodes
    n, m = len(nodes), len(blocks)
    index = {node: i for i, node in enumerate(nodes)}
    w = np.zeros((n, n))
    for i, j, weight in reference.source.conductance:
        w[i, j] = float(weight)
    d = w.sum(axis=1)
    nu = np.array(reference.source.capacity, dtype=float)
    h = d / nu
    a = float(reference.epi_weight) * nu[:, None] * (np.eye(n) - w / d[:, None])
    p = np.zeros((n, m))
    for j, block in enumerate(blocks):
        for node in block:
            p[index[node], j] = 1.0
    r = (p.T * h) / (p.T @ h)[:, None]
    q = _finite_difference(np.eye(n), _finite_product(p, r, "P R"), "Q")
    raq = _finite_product(_finite_product(r, a, "R A"), q, "R A Q")
    qap = _finite_product(_finite_product(q, a, "Q A"), p, "Q A P")
    k0 = _finite_product(raq, _finite_product(a, p, "A P"), "K0")

    def norm(value, name):
        return _finite_norm(value, matrix=True, name=name)

    abar = _finite_product(_finite_product(r, a, "R A"), p, "R A P")

    def diagnostic(mapping, source, target):
        residual, scale = _intertwining_diagnostics(mapping, source, target)
        tolerance = _relative_tolerance(mapping)
        return {
            "residual": residual,
            "residual_scale": scale,
            "relative_residual": residual / scale,
            "relative_tolerance": tolerance,
            "within_relative_tolerance": bool(residual / scale <= tolerance),
        }

    projection = diagnostic(r, a, abar)
    lift = diagnostic(p, abar, a)
    return {
        "projection_defect_spectral_norm": norm(raq, "projection defect"),
        "lift_defect_spectral_norm": norm(qap, "lift defect"),
        "zero_lag_kernel_spectral_norm": norm(k0, "zero-lag kernel"),
        "norm_definition": (
            "Matrix spectral 2-norm (largest singular value), evaluated with max-entry "
            "rescaling. The same norm applies to exact_cast_discrepancies and the shared "
            "intertwining residuals/scales; it is neither a Frobenius nor an infinity norm."
        ),
        "exact_cast_discrepancies": {
            name: norm(
                _finite_difference(
                    value, np.array(getattr(exact, field), dtype=float), name
                ),
                name,
            )
            for name, value, field in (
                ("R", r, "projection"),
                ("A", a, "micro_generator"),
                ("RAQ", raq, "hidden_to_macro"),
                ("QAP", qap, "macro_to_hidden"),
                ("K0", k0, "instantaneous_kernel"),
            )
        },
        "projection_intertwining": projection,
        "lift_intertwining": lift,
        "closure_within_relative_tolerance": bool(
            projection["within_relative_tolerance"]
            and lift["within_relative_tolerance"]
        ),
        "scope": (
            "Detached binary64 materialization using structural_morphism's finite matrix, "
            "intertwining residual and relative-tolerance owners. This diagnostic is not the "
            "exact closure decision. certify_morphism itself is not called because its "
            "default also samples flows. No graph, matrix exponential or trajectory is used."
        ),
    }


def _observer_reset(original, current, before, after, epi, before_state, after_state):
    projected_before = tuple(dot(row, epi) for row in before.projection)
    projected_after = tuple(dot(row, epi) for row in after.projection)
    reweight = tuple(
        dot(tuple(b - a for a, b in zip(left, right)), epi)
        for left, right in zip(before.projection, after.projection)
    )
    change = tuple(b - a for a, b in zip(projected_before, projected_after))
    if change != reweight:
        raise RuntimeError("event observer reweighting identity failed")
    profile_before = tuple(
        dot(row, original.relative_profile) for row in before.projection
    )
    profile_after = tuple(
        dot(row, current.relative_profile) for row in after.projection
    )
    centered_before = tuple(
        dot(row, before_state.relative_error) for row in before.projection
    )
    centered_after = tuple(
        dot(row, after_state.relative_error) for row in after.projection
    )
    mean_change = after_state.mean - before_state.mean
    centered_change = tuple(b - a for a, b in zip(centered_before, centered_after))
    decomposed = tuple(
        raw - mean_change - (zb - za)
        for raw, za, zb in zip(reweight, profile_before, profile_after)
    )
    if centered_change != decomposed:
        raise RuntimeError("centered observer reset identity failed")
    return {
        "epi_unchanged": True,
        "projected_epi_before": projected_before,
        "projected_epi_after": projected_after,
        "projected_epi_change": change,
        "projection_reweighting": reweight,
        "current_mean_before": before_state.mean,
        "current_mean_after": after_state.mean,
        "current_mean_reweighting": mean_change,
        "projected_relative_error_before": centered_before,
        "projected_relative_error_after": centered_after,
        "projected_relative_error_change": centered_change,
        "projected_profile_shift": tuple(
            b - a for a, b in zip(profile_before, profile_after)
        ),
        "identity_residual": tuple(Fraction(0) for _ in change),
        "scope": "Fixed EPI; changes arise from the observer's H weighting and current profile.",
    }


def run_study(
    control_path=CONTROL_PATH,
    lineage_path=LINEAGE_PATH,
    *,
    expected_control_sha256=CONTROL_SHA256,
    expected_lineage_sha256=LINEAGE_SHA256,
):
    from tnfr.physics.epi_memory import observe_forced_support_closure

    controls, lineage, partition, bindings = load_evidence(
        control_path,
        lineage_path,
        expected_control_sha256=expected_control_sha256,
        expected_lineage_sha256=expected_lineage_sha256,
    )
    base = controls[0]
    original = _reference(base["original_reference"])
    if _digest(asdict(original)) != base["original_reference_sha256"]:
        raise ValueError("original reference digest differs from complete reference")
    initial, _ = _admit_target(
        base["initial_target"],
        original,
        original,
        base["prefix"]["coupling"]["refreshed_forcing"],
    )
    if initial != original.source:
        raise ValueError("original model is not bound to its retained initial capture")
    snapshot, target = _admit_target(
        base["baseline_target"],
        original,
        original,
        base["baseline_flow"]["after_forcing"],
    )
    actual_state = base["before_optional_event"]
    if actual_state["time"] != 1.0:
        raise ValueError("family observer requires the declared common t=1 boundary")
    _match_live_state(actual_state, base["baseline_flow"]["after_forcing"], snapshot)
    blocks = partition["blocks"]
    first = observe_forced_support_closure(original, blocks, epi=snapshot.epi)
    before_state = observe_forced_support_state(original, snapshot)
    rows = []
    for name, branch in zip(MODEL_ORDER, (base, *lineage, controls[1]), strict=True):
        if name == "original":
            reference, current, readout, exact = original, snapshot, target, first
        else:
            _equal(
                branch["original_reference"],
                base["original_reference"],
                "original reference",
            )
            _equal(
                branch["after_optional_event"]["epi"],
                actual_state["epi"],
                "unchanged event EPI",
            )
            _equal(
                branch["after_optional_event"]["nodes"],
                actual_state["nodes"],
                "event node order",
            )
            if branch["after_optional_event"]["time"] != actual_state["time"]:
                raise ValueError("UM event changed the retained physical boundary")
            reference = _reference(branch["post_event_target"]["reference"])
            current, readout = _admit_target(
                branch["post_event_target"],
                original,
                reference,
                branch["event"]["coupling"]["refreshed_forcing"],
            )
            if reference.source != current or current.epi != snapshot.epi:
                raise ValueError("post-event source does not retain the common EPI")
            _match_live_state(
                branch["after_optional_event"],
                branch["event"]["coupling"]["refreshed_forcing"],
                current,
            )
            exact = observe_forced_support_closure(reference, blocks, epi=current.epi)
        state = observe_forced_support_state(reference, current)
        rows.append(
            {
                "model": name,
                "observation_time": 1.0,
                "reference_sha256": _digest(asdict(reference)),
                "actual_snapshot": asdict(current),
                "exact_observation": asdict(exact),
                "numerical_crosscheck": _numerical_crosscheck(reference, blocks, exact),
                "current_state": asdict(state),
                "old_target_readout": {
                    "pattern": asdict(readout.pattern),
                    "target_compatible": readout.target_compatible,
                    "compatibility_energy": readout.compatibility_energy,
                    "limiting_pattern": asdict(readout.limiting_pattern),
                },
                "observer_reset": _observer_reset(
                    original, reference, first, exact, current.epi, before_state, state
                ),
            }
        )
    for path, key in (
        (control_path, "retained_controls"),
        (lineage_path, "retained_lineage"),
    ):
        if (
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            != bindings[key]["sha256"]
        ):
            raise RuntimeError("retained input bytes changed during the analysis")
    return {
        **bindings,
        "partition": partition,
        "models": tuple(rows),
        "common_fields_checked": COMMON_FIELDS,
        "new_trajectories_executed": False,
        "partition_search_performed": False,
        "scope": (
            "Exact all-state affine closure or hidden-state obstruction for one actual ancestry "
            "partition and four fixed retained models. Supplied captured forcing is preserved. "
            "No autonomous macro-triad, physical correspondence, future schedule, recovery or "
            "new executed birth claim. Historical producer metadata is retained."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=CONTROL_PATH)
    parser.add_argument("--lineage", type=Path, default=LINEAGE_PATH)
    parser.add_argument("--expected-control-sha256", default=CONTROL_SHA256)
    parser.add_argument("--expected-lineage-sha256", default=LINEAGE_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "artifacts/research/thol_family_closure.json"),
    )
    args = parser.parse_args()
    if args.output.resolve() in (args.controls.resolve(), args.lineage.resolve()):
        raise ValueError("new analysis must not overwrite retained input evidence")
    scope = (
        "src/tnfr",
        "benchmarks/thol_family_closure.py",
        "benchmarks/thol_lineage_coordination.py",
        "benchmarks/thol_distributed_target.py",
        "benchmarks/thol_distributed_transport.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/capacity_localization.py",
        "benchmarks/structural_target_compatibility.py",
        "benchmarks/thol_preparation_policy.py",
        "benchmarks/selection_birth_closure.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-generated-family-affine-state-sufficiency",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="No graph construction; authenticated retained actual birth families",
        capacity_specification="Four retained fixed forced models; exact represented coefficients",
        solver="Detached exact rational closure and memory-at-zero identities; no integration",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(),
        telemetry=(
            "exact RAQ/QAP",
            "hidden-state witness",
            "K(0)",
            "observer reweighting",
            "binary64 diagnostic",
        ),
        controls=("complete retained fixed-target and lineage evidence",),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    result = run_study(
        args.controls,
        args.lineage,
        expected_control_sha256=args.expected_control_sha256,
        expected_lineage_sha256=args.expected_lineage_sha256,
    )
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("analysis source changed during execution")
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        **result,
        "experimental_status": "No empirical correspondence tested",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote detached family state-sufficiency analysis to {args.output}")


if __name__ == "__main__":
    main()
