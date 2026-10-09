"""Exact child-cohort distortion ledger from two authenticated saved reports.

No graph, scalar reset kernel, pressure kernel, coordinator or runtime is run.
Difference snapshots are detached arithmetic fields, not new physical states.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import thol_retained_reset_audit as reset  # noqa: E402
from benchmarks.thol_family_closure import (
    _capture,
    _equal,
    _reference,
    _snapshot,
)  # noqa: E402
from benchmarks.thol_full_state_response import (
    _add,
    _subtract,
    _paired_delta,
    _payload,
)  # noqa: E402
from benchmarks.thol_regional_balance_audit import _component_budget  # noqa: E402
from tnfr.physics.forcing_realization import (
    NonEpiForcingObservation,
    decompose_non_epi_forcing,
)  # noqa: E402
from tnfr.physics.support_transport import (
    _from_data,
    observe_regional_support_euler,
)  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

RESET_PATH = ROOT / "artifacts/research/thol_retained_reset_audit_2026_09_18.json"
RESET_SHA256 = "6257b5ce03a8620afe7c982f68cf2913c4b6759610a99ef047f2c878818310ec"
RECOVERY_PATH = ROOT / "artifacts/research/thol_regional_recovery_audit_2026_09_18.json"
RECOVERY_SHA256 = "5a926d82a5cbb2d2ea8f51fd1fbfe9d033aa5a59caff797f465ed0efdaeec8b0"
START, END, DT = F(9, 4), F(5, 2), F(1, 4)
BRANCHES = ("control", "child_emission")


def _generation(raw):
    """Hydrate and check the complete saved capture without evaluating a kernel."""
    snap = _snapshot(raw["snapshot"])
    names = (
        "phase",
        "forcing",
        "phase_gradient",
        "full_kernel_pressure",
        "kernel_pressure_defect",
        "stored_pressure_residual",
    )
    observation = NonEpiForcingObservation(
        snapshot=snap,
        epi_weight=F(raw["epi_weight"]),
        normalized_weights=tuple((k, F(v)) for k, v in raw["normalized_weights"]),
        **{key: reset._v(raw[key]) for key in names},
    )
    return _capture(
        {
            "observation": raw,
            "components": _payload(decompose_non_epi_forcing(observation)),
        }
    )


def _admit_reset(saved, original):
    saved = _payload(saved)
    snap, observation, components = _generation(saved["generation"]["observation"])
    nodes, n = snap.nodes, len(snap.nodes)
    if (
        saved["glyph"] != "EN"
        or F(saved["start_time"]) != START
        or F(saved["end_time"]) != END
        or tuple(saved["nodes"]) != nodes
        or nodes != original.source.nodes
    ):
        raise ValueError("retained EN interval or original node domain differs")
    for key in ("conductance", "support_neighbors", "capacity"):
        if getattr(snap, key) != getattr(original.source, key):
            raise ValueError("retained support/capacity differs from original metric")
    if observation.epi_weight != original.epi_weight:
        raise ValueError("retained EPI coefficient differs from original reference")
    vectors = {key: reset._v(saved["vectors"][key]) for key in ("x0", "xg", "xi", "xf")}
    if any(len(v) != n for v in vectors.values()) or vectors["x0"] != snap.epi:
        raise ValueError("generation snapshot and complete staged EPI disagree")
    rows = saved["reset"]["local_rows"]
    if len(rows) != n or tuple(row["node"] for row in rows) != nodes:
        raise ValueError("retained EN requires one ordered local row per node")
    matrix, offset, propagated, expected = reset._identity(n), (F(0),) * n, [], snap.epi
    for i, row in enumerate(rows):
        x, y, coefficient = (
            reset._v(row[key]) for key in ("before_epi", "after_epi", "row")
        )
        if (
            len(x) != n
            or len(y) != n
            or len(coefficient) != n
            or x != expected
            or any(y[j] != x[j] for j in range(n) if j != i)
        ):
            raise ValueError("retained EN row order or single-target write differs")
        indices = tuple(row["control"]["neighbor_indices"])
        if tuple(sorted(indices)) != snap.support_neighbors[i] or F(row["offset"]) != 0:
            raise ValueError("retained EN neighbor support or zero offset differs")
        local = list(reset._identity(n))
        local[i] = coefficient
        local = tuple(local)
        shift = tuple(F(row["offset"]) if j == i else F(0) for j in range(n))
        defect = _subtract(y, _add(reset.mv(local, x), shift))
        _equal(defect, row["local_defect"], "local reset defect")
        if F(row["kernel_evaluation_defect"]) + F(row["clipping_defect"]) != defect[i]:
            raise ValueError("saved local kernel/clipping defect split differs")
        propagated = [reset.mv(local, value) for value in propagated] + [defect]
        matrix, offset = reset.mm(local, matrix), _add(reset.mv(local, offset), shift)
        expected = y
    if expected != vectors["xg"]:
        raise ValueError("last EN endpoint differs from integrator entry")
    total = reset._sum(propagated, n)
    for key, value in (
        ("S", matrix),
        ("c", offset),
        ("propagated_local_defects", propagated),
        ("total_reset_defect", total),
        ("identity_residual", (F(0),) * n),
    ):
        _equal(value, saved["reset"][key], f"composed reset {key}")
    if _add(_add(reset.mv(matrix, snap.epi), offset), total) != vectors["xg"]:
        raise ValueError("composed retained reset identity differs")
    strengths = [F(0)] * n
    for i, j, weight in snap.conductance:
        strengths[i] += weight
    a = [[F(0)] * n for _ in range(n)]
    for i, j, weight in snap.conductance:
        value = observation.epi_weight * snap.capacity[i] * weight / strengths[i]
        a[i][i] += value
        a[i][j] -= value
    a = tuple(map(tuple, a))
    b = tuple(nu * f for nu, f in zip(snap.capacity, observation.forcing, strict=True))
    _equal(a, saved["generation"]["A"], "generation A")
    _equal(b, saved["generation"]["b"], "generation b")
    drive = tuple(
        nu * p for nu, p in zip(snap.capacity, snap.stored_pressure, strict=True)
    )
    ideal = _subtract(b, reset.mv(a, snap.epi))
    runtime = {
        "ideal_endpoint": _add(
            _add(reset.mv(matrix, snap.epi), offset), tuple(DT * v for v in ideal)
        ),
        "pressure_term": tuple(DT * v for v in _subtract(drive, ideal)),
        "integration_remainder": _subtract(
            _subtract(vectors["xi"], vectors["xg"]), tuple(DT * v for v in drive)
        ),
        "postintegration_change": _subtract(vectors["xf"], vectors["xi"]),
        "identity_residual": (F(0),) * n,
        "pre_generated_pressure_retained": True,
    }
    _equal(runtime, saved["runtime"], "runtime ledger")
    if (
        _add(
            runtime["ideal_endpoint"],
            reset._sum(
                (
                    total,
                    runtime["pressure_term"],
                    runtime["integration_remainder"],
                    runtime["postintegration_change"],
                ),
                n,
            ),
        )
        != vectors["xf"]
    ):
        raise ValueError("retained complete endpoint ledger differs")
    return {
        "snapshot": snap,
        "observation": observation,
        "components": components,
        "vectors": vectors,
        "S": matrix,
        "c": offset,
        "A": a,
        "b": b,
        "rows": rows,
        "runtime": runtime,
        "reset_defect": total,
        "saved": saved,
    }


def audit_pair(left_reset, right_reset, original_reference, children):
    """Account for one supplied paired EN interval; provenance belongs to audit_saved."""
    left, right = (
        _admit_reset(row, original_reference) for row in (left_reset, right_reset)
    )
    for key in ("A", "b", "S", "c"):
        if left[key] != right[key]:
            raise ValueError(f"paired retained {key} differs")
    for lrow, rrow in zip(left["rows"], right["rows"], strict=True):
        for key in ("node", "row", "offset", "configuration"):
            _equal(lrow[key], rrow[key], f"paired EN {key}")
        for key in ("EN_mix", "neighbor_indices"):
            _equal(
                lrow["control"][key], rrow["control"][key], f"paired EN control {key}"
            )
    snap, n = left["snapshot"], len(left["snapshot"].nodes)
    children = tuple(children)
    if (
        not children
        or len(set(children)) != len(children)
        or not set(children) < set(snap.nodes)
    ):
        raise ValueError("children must be a nonempty proper ordered subset")
    indices = tuple(snap.nodes.index(node) for node in children)
    metric = tuple(original_reference.metric_weights[i] for i in indices)
    pressure = _subtract(right["snapshot"].stored_pressure, snap.stored_pressure)
    force = _subtract(right["observation"].forcing, left["observation"].forcing)
    if any(force):
        raise ValueError(
            "common positive-capacity affine source requires equal forcing"
        )
    e = left["observation"].epi_weight
    delta = {
        key: _subtract(right["vectors"][key], left["vectors"][key])
        for key in left["vectors"]
    }

    def snapshot(epi):
        return _from_data(
            snap.nodes,
            snap.conductance,
            snap.support_neighbors,
            epi,
            snap.capacity,
            pressure,
        )

    def budget(before, after, dt):
        return observe_regional_support_euler(
            snapshot(before),
            snapshot(after),
            children,
            dt=dt,
            epi_weight=e,
            forcing=force,
        )

    def shape(epi):
        return _paired_delta(
            (F(0),) * len(indices), tuple(epi[i] for i in indices), metric
        )

    stages = []
    for ordinal, (lrow, rrow) in enumerate(
        zip(left["rows"], right["rows"], strict=True)
    ):
        before = _subtract(reset._v(rrow["before_epi"]), reset._v(lrow["before_epi"]))
        after = _subtract(reset._v(rrow["after_epi"]), reset._v(lrow["after_epi"]))
        stages.append(
            {
                "ordinal": ordinal,
                "node": lrow["node"],
                "delta_before": before,
                "delta_after": after,
                "paired_increment": _subtract(after, before),
                "control_increment": _subtract(
                    reset._v(lrow["after_epi"]), reset._v(lrow["before_epi"])
                ),
                "perturbed_increment": _subtract(
                    reset._v(rrow["after_epi"]), reset._v(rrow["before_epi"])
                ),
                "budget": asdict(budget(before, after, F(0))),
            }
        )
    integration = budget(delta["xg"], delta["xi"], DT)
    post = budget(delta["xi"], delta["xf"], F(0))
    reset_total = budget(delta["x0"], delta["xg"], F(0))
    generation_snapshot = snapshot(delta["x0"])
    lag = tuple(
        e * (a - b)
        for a, b in zip(
            generation_snapshot.epi_gradient,
            integration.before.epi_gradient,
            strict=True,
        )
    )
    pressure_parts = {
        "held_generation_lag": lag,
        "generation_kernel_defect": _subtract(
            right["observation"].kernel_pressure_defect,
            left["observation"].kernel_pressure_defect,
        ),
        "generation_stored_minus_fresh": _subtract(
            right["observation"].stored_pressure_residual,
            left["observation"].stored_pressure_residual,
        ),
    }
    if (
        reset._sum(pressure_parts.values(), n)
        != integration.balance.stored_pressure_defect
    ):
        raise RuntimeError(
            "held-pressure lag and generation defect decomposition failed"
        )
    work = {}
    for name, values in pressure_parts.items():
        item = _component_budget(integration.balance, values)
        work[name] = {
            **item,
            "weighted_total_work": DT * item["weighted_total_rate"],
            "variance_work": DT * item["variance_rate"],
        }
    mass = (
        sum((F(row["budget"]["mass_change"]) for row in stages), F(0))
        + integration.mass_change
        + post.mass_change
    )
    variance = (
        sum((F(row["budget"]["variance_change"]) for row in stages), F(0))
        + integration.variance_change
        + post.variance_change
    )
    endpoint = budget(delta["x0"], delta["xf"], F(0))
    if (
        mass != endpoint.mass_change
        or variance != endpoint.variance_change
        or sum((F(row["budget"]["variance_change"]) for row in stages), F(0))
        != reset_total.variance_change
    ):
        raise RuntimeError("paired regional stage telescope failed")
    residuals = {"reset": _subtract(right["reset_defect"], left["reset_defect"])}
    residuals.update(
        {
            key: _subtract(right["runtime"][key], left["runtime"][key])
            for key in (
                "pressure_term",
                "integration_remainder",
                "postintegration_change",
            )
        }
    )
    modes = _mode_diagnostic(
        left, delta, children, original_reference.metric_weights, residuals
    )
    return {
        "nodes": snap.nodes,
        "children": children,
        "region_indices": indices,
        "metric_weights": metric,
        "start_time": START,
        "end_time": END,
        "dt": DT,
        "delta": delta,
        "states": {key: shape(value) for key, value in delta.items()},
        "local_EN_writes": stages,
        "reset_budget": asdict(reset_total),
        "integration_budget": asdict(integration),
        "postintegration_budget": asdict(post),
        "pressure_difference": pressure,
        "forcing_difference": force,
        "pressure_work": work,
        "telescope": {
            "mass_change": mass,
            "variance_change": variance,
            "mass_residual": mass - endpoint.mass_change,
            "variance_residual": variance - endpoint.variance_change,
        },
        "mode_diagnostic": modes,
        "retained_resets": (left["saved"], right["saved"]),
        "scope": "Difference-field accounting of one child-cohort response relative to its evolving control; "
        "no new physical graph, refreshed pressure, order counterfactual or maintenance predicate",
    }


def _mode_diagnostic(left, delta, children, metric, residuals):
    """One fixed four-mode split, including every self and Gram cross term."""
    nodes, n = left["snapshot"].nodes, len(metric)
    bi = tuple(nodes.index(node) for node in children)
    pi = tuple(i for i in range(n) if i not in bi)

    def mean(value, indices):
        return sum((metric[i] * value[i] for i in indices), F(0)) / sum(
            metric[i] for i in indices
        )

    def center(value):
        m = mean(value, bi)
        return tuple(value[i] - m for i in bi)

    mb, mp = mean(delta["x0"], bi), mean(delta["x0"], pi)
    one, indicator = (F(1),) * n, tuple(F(i in bi) for i in range(n))
    modes = {
        "parent_mean": tuple(mp for _ in nodes),
        "child_parent_mean_contrast": tuple((mb - mp) * v for v in indicator),
        "child_centered": tuple(
            delta["x0"][i] - mb if i in bi else F(0) for i in range(n)
        ),
        "parent_centered": tuple(
            delta["x0"][i] - mp if i in pi else F(0) for i in range(n)
        ),
    }
    if reset._sum(modes.values(), n) != delta["x0"]:
        raise RuntimeError("fixed cohort decomposition failed")
    t = tuple(
        tuple(s - DT * a for s, a in zip(sr, ar, strict=True))
        for sr, ar in zip(left["S"], left["A"], strict=True)
    )
    outputs = {key: center(reset.mv(t, value)) for key, value in modes.items()}
    outputs.update(
        {f"residual_{key}": center(value) for key, value in residuals.items()}
    )
    reconstructed = reset._sum(outputs.values(), len(bi))
    if reconstructed != center(delta["xf"]):
        raise RuntimeError("child centered mode/residual reconstruction failed")
    labels = tuple(outputs)
    gram = []
    for i, name in enumerate(labels):
        for j in range(i, len(labels)):
            other = labels[j]
            inner = sum(
                (
                    metric[k] * a * b
                    for k, a, b in zip(bi, outputs[name], outputs[other], strict=True)
                ),
                F(0),
            )
            gram.append(
                {
                    "left": name,
                    "right": other,
                    "inner_product": inner,
                    "energy_contribution": inner / 2 if i == j else inner,
                }
            )
    energy = sum((row["energy_contribution"] for row in gram), F(0))
    direct = sum(
        (metric[k] * v * v / 2 for k, v in zip(bi, reconstructed, strict=True)), F(0)
    )
    if energy != direct:
        raise RuntimeError("complete Gram energy reconstruction failed")
    transport = center(tuple(-DT * v for v in reset.mv(left["A"], indicator)))
    return {
        "parent_mean": mp,
        "child_mean": mb,
        "T": t,
        "initial_modes": modes,
        "centered_child_contributions": outputs,
        "centered_endpoint": reconstructed,
        "gram_terms": gram,
        "energy": energy,
        "energy_residual": energy - direct,
        "centered_T_one": center(reset.mv(t, one)),
        "centered_S_child_indicator": center(reset.mv(left["S"], indicator)),
        "centered_minus_hA_child_indicator": transport,
        "child_mean_contrast_transport_to_shape_zero": not any(transport),
        "scope": "Fixed admitted S and A only; cross terms are retained. This does not isolate update order as a cause.",
    }


def audit_saved(reset_report, recovery_report):
    """Bind the two saved derivations and audit only their declared EN interval."""
    for name in ("window", "native"):
        a, b = (
            report["historical_inputs"][name]
            for report in (reset_report, recovery_report)
        )
        for key in ("sha256", "historical_manifest"):
            _equal(a[key], b[key], f"shared historical {name} {key}")
    if tuple(row["branch"] for row in reset_report["branches"]) != BRANCHES:
        raise ValueError("saved branch identity/order differs")
    original = _reference(reset_report["original_reference"])
    nodes = original.source.nodes
    _equal(nodes, recovery_report["nodes"], "regional original nodes")
    _equal(
        original.metric_weights,
        recovery_report["full_metric_weights"],
        "regional original full metric",
    )
    pairs = tuple(tuple(pair) for pair in recovery_report["actual_lineage"]["pairs"])
    children = tuple(recovery_report["actual_lineage"]["children"])
    if (
        len(nodes) != 16
        or len(pairs) != 8
        or any(len(pair) != 2 for pair in pairs)
        or len({node for pair in pairs for node in pair}) != 16
        or set(node for pair in pairs for node in pair) != set(nodes)
        or children != tuple(pair[1] for pair in pairs)
    ):
        raise ValueError("saved actual eight-family ancestry differs")
    selected = []
    for branch in reset_report["branches"]:
        candidates = [row for row in branch["resets"] if row["glyph"] == "EN"]
        if len(candidates) != 1:
            raise ValueError("one retained EN reset per branch is required")
        selected.append(candidates[0])
    regions = [
        row for row in recovery_report["regions"] if row["label"] == "actual_children"
    ]
    if len(regions) != 1:
        raise ValueError("one saved actual-child region is required")
    region = regions[0]
    indices = tuple(nodes.index(node) for node in children)
    _equal(children, region["nodes"], "child cohort")
    _equal(indices, region["full_node_indices"], "child indices")
    _equal(
        tuple(original.metric_weights[i] for i in indices),
        region["metric_weights"],
        "child metric",
    )
    result = audit_pair(*selected, original, children)
    endpoints = []
    for time, key in ((START, "x0"), (END, "xf")):
        rows = [row for row in region["endpoints"] if F(row["time"]) == time]
        if len(rows) != 1:
            raise ValueError("regional endpoint time is missing or duplicated")
        row = rows[0]
        for branch, field in zip(
            selected, ("control_epi", "perturbed_epi"), strict=True
        ):
            _equal(
                tuple(F(branch["vectors"][key][i]) for i in indices),
                row[field],
                "regional endpoint EPI",
            )
        _equal(result["states"][key], row["paired"], "regional paired endpoint readout")
        endpoints.append(row)
    return {
        "original_reference": reset_report["original_reference"],
        "actual_lineage": recovery_report["actual_lineage"],
        "common_historical_inputs": reset_report["historical_inputs"],
        "regional_endpoints": endpoints,
        "audit": result,
        "native_calls": 0,
        "kernel_calls": 0,
        "forcing_capture_calls": 0,
        "new_trajectories": 0,
        "autonomous_maintenance_certified": False,
    }


def run_study(
    reset_path=RESET_PATH,
    recovery_path=RECOVERY_PATH,
    *,
    expected_reset_sha256=RESET_SHA256,
    expected_recovery_sha256=RECOVERY_SHA256,
):
    first, rb = reset._load(
        reset_path, expected_reset_sha256, "O3.b-retained-EN-AL-reset-accounting"
    )
    second, cb = reset._load(
        recovery_path,
        expected_recovery_sha256,
        "O3.a-retained-regional-paired-response",
    )
    result = audit_saved(first, second)
    for binding in (rb, cb):
        if (
            hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise RuntimeError("input changed during retained child audit")
    return {"historical_inputs": {"reset": rb, "recovery": cb}, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-input", type=Path, default=RESET_PATH)
    parser.add_argument("--recovery-input", type=Path, default=RECOVERY_PATH)
    parser.add_argument("--expected-reset-sha256", default=RESET_SHA256)
    parser.add_argument("--expected-recovery-sha256", default=RECOVERY_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/thol_child_distortion_audit_2026_09_18.json",
    )
    args = parser.parse_args()
    if args.output.resolve() in (
        args.reset_input.resolve(),
        args.recovery_input.resolve(),
    ):
        raise ValueError("output must not overwrite retained inputs")
    scope = (
        "src/tnfr",
        "benchmarks/thol_child_distortion_audit.py",
        "benchmarks/thol_retained_reset_audit.py",
        "benchmarks/thol_family_closure.py",
        "benchmarks/thol_full_state_response.py",
        "benchmarks/thol_regional_balance_audit.py",
        "benchmarks/thol_pressure_feedback.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(
        args.reset_input,
        args.recovery_input,
        expected_reset_sha256=args.expected_reset_sha256,
        expected_recovery_sha256=args.expected_recovery_sha256,
    )
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during retained child audit")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-retained-child-distortion",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="No graph; two authenticated saved derivations",
        capacity_specification="Original full-support positive capacity and fixed child metric",
        solver="Exact retained paired stage ledger; no kernels or solver execution",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=("reception",),
        telemetry=(
            "per-write paired child distortion",
            "held-pressure lag",
            "finite regional Euler budget",
            "complete mode Gram terms",
        ),
        controls=(
            "one fixed EN interval",
            "actual child lineage",
            "no new trajectories",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(
        args.output,
        lambda stream: stream.write(
            json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
        ),
    )
    print(f"Wrote retained child distortion audit to {args.output}")


if __name__ == "__main__":
    main()
