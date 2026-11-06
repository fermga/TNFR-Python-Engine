"""Admit common declared EN maps from two authenticated retained experiments.

Only saved records and exact arithmetic are used. No graph, operator, reset
kernel, pressure kernel, coordinator or trajectory is evaluated. The earlier
authenticated EN rows supply the declared coefficients; matching captured
configuration and ordered support permit their reuse on the later records.
The later local realization defect remains unclassified (rounding/clipping
are not separately inferred from an endpoint).
"""

from __future__ import annotations

from fractions import Fraction as F
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import thol_child_distortion_audit as child  # noqa: E402
from benchmarks import thol_retained_reset_audit as reset  # noqa: E402
from benchmarks.thol_family_closure import (
    _capture,
    _equal,
    _match_live_state,
    _reference,
)  # noqa: E402
from benchmarks.thol_full_state_response import _add, _subtract  # noqa: E402
from benchmarks.thol_native_policy_window import expand_record  # noqa: E402
from benchmarks.thol_regional_balance_audit import _bind_source_edges  # noqa: E402

RESET_PATH, RESET_SHA256 = child.RESET_PATH, child.RESET_SHA256
RESTORATION_PATH = ROOT / "artifacts/research/thol_regional_restoration_2026_09_18.json"
RESTORATION_SHA256 = "7ebf3b9cd372f70a5b5d170b16e50fae22ec2a56595c73adb3ce82866c596259"
START, END, DT = child.START, child.END, child.DT


def _one(trace, boundary):
    rows = [row for row in trace["boundaries"] if row["boundary"] == boundary]
    if len(rows) != 1 or rows[0]["outcome"] != "completed":
        raise ValueError(f"one completed {boundary} boundary is required")
    return rows[0]


def _common(left, right, *, source_required):
    for key in ("S", "c", "A"):
        if left[key] != right[key]:
            raise ValueError(f"declared common {key} differs")
    if source_required and left["b"] != right["b"]:
        raise ValueError("paired non-EPI source does not cancel")
    for key in ("nodes", "conductance", "support_neighbors", "capacity"):
        if getattr(left["snapshot"], key) != getattr(right["snapshot"], key):
            raise ValueError(f"common {key} differs")
    if left["observation"].epi_weight != right["observation"].epi_weight:
        raise ValueError("common EPI weight differs")
    for lrow, rrow in zip(left["rows"], right["rows"], strict=True):
        for key in ("node", "row", "offset", "configuration"):
            _equal(lrow[key], rrow[key], f"common local {key}")
        for key in ("EN_mix", "neighbor_indices"):
            _equal(lrow["control"][key], rrow["control"][key], f"common local {key}")


def admit_trace(step, original, template):
    """Bind a saved EN trace to admitted declared rows, retaining total defects.

    ``template`` must come from an authenticated saved reset through the
    existing arithmetic admission. This function alone authenticates no file
    and certifies no general binary64 kernel identity.
    """
    trace = step["native_trace"]
    if step["status"] != "executed" or trace["status"] != "executed":
        raise ValueError("a completed saved step is required")
    recorded_ordinals = tuple(row["ordinal"] for row in trace["boundaries"])
    if (
        any(type(value) is not int or value < 0 for value in recorded_ordinals)
        or tuple(sorted(set(recorded_ordinals))) != recorded_ordinals
    ):
        raise ValueError("boundary list order or ordinals differ")
    prepared, integration = (_one(trace, key) for key in ("_prepare_dnfr", "integrate"))
    refresh = _one(trace, "_refresh_delta_nfr")
    calls = [row for row in trace["boundaries"] if row["boundary"] == "apply_glyph"]
    capture = trace["captures"]["pressure_generation"]
    if not capture["available"]:
        raise ValueError("generation capture is unavailable")
    snap, observation, components = _capture(capture["payload"])
    _match_live_state(prepared["after"]["state"], capture["payload"], snap)
    nodes, n = snap.nodes, len(snap.nodes)
    if nodes != original.source.nodes or not 1 < n <= 16:
        raise ValueError("saved trace differs from original bounded node space")
    if (
        len(calls) != n
        or tuple(row["node"] for row in calls) != nodes
        or any(row["glyph"] != "EN" or row["outcome"] != "completed" for row in calls)
    ):
        raise ValueError("exactly one completed ordered EN call per node is required")
    ordinals = tuple(row["ordinal"] for row in calls)
    if (
        tuple(sorted(set(ordinals))) != ordinals
        or prepared["ordinal"] >= ordinals[0]
        or refresh["ordinal"] >= ordinals[0]
        or ordinals[-1] >= integration["ordinal"]
    ):
        raise ValueError("saved generation, EN and integration order differs")
    for key in ("conductance", "support_neighbors", "capacity"):
        if getattr(snap, key) != getattr(original.source, key):
            raise ValueError("saved support/capacity differs from original metric")
    if observation.epi_weight != original.epi_weight:
        raise ValueError("saved EPI coefficient differs from original reference")
    records = [step["before"], prepared["after"], refresh["after"]]
    records += [row[side] for row in calls for side in ("before", "after")]
    records += [integration["before"], integration["after"], step["endpoint"]]
    if (
        any(F(record["state"]["time"]) != START for record in records[:-2])
        or any(F(record["state"]["time"]) != END for record in records[-2:])
        or F(prepared["before"]["state"]["time"]) != START
        or F(refresh["before"]["state"]["time"]) != START
    ):
        raise ValueError("internal EN interval clocks differ")
    for record in records:
        state = record["state"]
        if (
            tuple(state["nodes"]) != nodes
            or reset._v(state["capacity"]) != snap.capacity
        ):
            raise ValueError("saved node/capacity changed inside the interval")
        _bind_source_edges(state, snap)
    for record in records[1:]:
        if reset._v(record["state"]["pressure"]) != snap.stored_pressure:
            raise ValueError("pre-generated pressure was not retained")
    phase = prepared["after"]["state"]["phase"]
    if any(record["state"]["phase"] != phase for record in records[1:-2]):
        raise ValueError("phase changed before integration")
    if reset._v(step["before"]["state"]["epi"]) != snap.epi:
        raise ValueError("unassigned pre-generation EPI change")
    if (
        F(step["before"]["state"]["time"]) != START
        or F(step["endpoint"]["state"]["time"]) != END
        or F(integration["before"]["state"]["time"]) != START
    ):
        raise ValueError("saved EN physical interval differs")
    arguments = integration["effective_arguments"]
    if (
        integration["integrator_type"] != "tnfr.dynamics.integrators.DefaultIntegrator"
        or arguments["method"] != "euler"
        or F(reset._literal_number(arguments["dt"])) != DT
    ):
        raise ValueError("saved integration is not the declared Euler interval")
    entry = trace["captures"]["integrator_entry"]
    if not entry["available"]:
        raise ValueError("integrator-entry capture is unavailable")
    entry_snap, entry_observation, _ = _capture(entry["payload"])
    _match_live_state(integration["before"]["state"], entry["payload"], entry_snap)
    for key in (
        "nodes",
        "conductance",
        "support_neighbors",
        "capacity",
        "stored_pressure",
    ):
        if getattr(entry_snap, key) != getattr(snap, key):
            raise ValueError("integration-entry domain differs from generation")
    if entry_observation.epi_weight != observation.epi_weight:
        raise ValueError("integration-entry EPI coefficient changed")
    matrix, offset = reset._identity(n), (F(0),) * n
    propagated, rows, expected = [], [], snap.epi
    for i, (call, declared) in enumerate(zip(calls, template["rows"], strict=True)):
        before, after = call["before"], call["after"]
        x, y = (reset._v(record["state"]["epi"]) for record in (before, after))
        if x != expected or any(y[j] != x[j] for j in range(n) if j != i):
            raise ValueError("ordered EN single-target EPI write differs")
        for record in (before, after):
            _equal(
                reset._configuration(record),
                declared["configuration"],
                "EN configuration",
            )
            ordered = record["ordered_neighbors"]
            if tuple(node for node, _ in ordered) != nodes:
                raise ValueError("ordered neighbor domain differs")
            indices = tuple(nodes.index(node) for node in ordered[i][1])
            if indices != tuple(declared["control"]["neighbor_indices"]):
                raise ValueError("EN neighbor order differs from declared template")
        row = reset._v(declared["row"])
        if declared["node"] != nodes[i] or F(declared["offset"]) != 0:
            raise ValueError("declared EN target or zero offset differs")
        local = list(reset._identity(n))
        local[i] = row
        local = tuple(local)
        defect = _subtract(y, reset.mv(local, x))
        propagated = [reset.mv(local, value) for value in propagated] + [defect]
        matrix = reset.mm(local, matrix)
        rows.append(
            {
                "node": nodes[i],
                "control": {
                    key: declared["control"][key]
                    for key in ("EN_mix", "neighbor_indices")
                },
                "configuration": declared["configuration"],
                "row": row,
                "offset": F(0),
                "before_epi": x,
                "after_epi": y,
                "local_defect": defect,
                "defect_scope": "Unclassified realized endpoint minus declared affine row",
            }
        )
        expected = y
    vectors = {
        "x0": snap.epi,
        "xg": reset._v(integration["before"]["state"]["epi"]),
        "xi": reset._v(integration["after"]["state"]["epi"]),
        "xf": reset._v(step["endpoint"]["state"]["epi"]),
    }
    if expected != vectors["xg"]:
        raise ValueError("unassigned post-EN pre-integration EPI change")
    # A is inherited only after its defining support, capacity and EPI weight
    # match the original and the admitted template; b comes from this capture.
    a = template["A"]
    b = tuple(nu * f for nu, f in zip(snap.capacity, observation.forcing, strict=True))
    total = reset._sum(propagated, n)
    drive = tuple(
        nu * p for nu, p in zip(snap.capacity, snap.stored_pressure, strict=True)
    )
    ideal = _subtract(b, reset.mv(a, snap.epi))
    runtime = {
        "ideal_endpoint": _add(
            reset.mv(matrix, snap.epi), tuple(DT * v for v in ideal)
        ),
        "pressure_term": tuple(DT * v for v in _subtract(drive, ideal)),
        "integration_remainder": _subtract(
            _subtract(vectors["xi"], vectors["xg"]), tuple(DT * v for v in drive)
        ),
        "postintegration_change": _subtract(vectors["xf"], vectors["xi"]),
        "identity_residual": (F(0),) * n,
        "pre_generated_pressure_retained": True,
    }
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
        raise RuntimeError("retained endpoint identity failed")
    result = {
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
        "reset_basis": "Authenticated declared rows with matching captured configuration/order; new kernel/clipping split unidentified",
    }
    _common(template, result, source_required=False)
    return result


def _pair(left, right):
    _common(left, right, source_required=True)
    delta = {
        key: _subtract(right["vectors"][key], left["vectors"][key])
        for key in left["vectors"]
    }
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
    return {
        "control": left,
        "perturbed": right,
        "delta": delta,
        "residuals": residuals,
        "paired_source_difference": _subtract(right["b"], left["b"]),
    }


def admit_reports(saved, restoration):
    """Arithmetic admission of supplied reports; file authentication is separate."""
    original = _reference(saved["original_reference"])
    pool = restoration["record_pool"]
    restored_original = expand_record(restoration["original_reference"], pool)
    _equal(saved["original_reference"], restored_original, "fixed original reference")
    replayed = expand_record(restoration["replayed_control_report"], pool)
    children = tuple(restoration["children"])
    if (
        children != tuple(replayed["lineage"]["children"])
        or not children
        or len(set(children)) != len(children)
        or not set(children) < set(original.source.nodes)
    ):
        raise ValueError("fixed actual child lineage differs")
    if tuple(row["branch"] for row in saved["branches"]) != (
        "control",
        "child_emission",
    ):
        raise ValueError("old paired branch identity differs")
    old = []
    for branch in saved["branches"]:
        selected = [row for row in branch["resets"] if row["glyph"] == "EN"]
        if len(selected) != 1:
            raise ValueError("one old EN interval is required")
        old.append(child._admit_reset(selected[0], original))
    cohort = _pair(*old)
    if (
        restoration["status"] != "completed"
        or restoration["aligned_completed_steps"] != 6
        or tuple(row["branch"] for row in restoration["branches"])
        != ("control", "localized_child_emission")
    ):
        raise ValueError("completed fixed localized branch pair is required")
    localized = []
    for branch_index, branch in enumerate(restoration["branches"]):
        if (
            branch["completed_steps"] != 6
            or branch["attempted_steps"] != 6
            or len(branch["steps"]) != 6
        ):
            raise ValueError("localized window must contain its six retained steps")
        previous = expand_record(restoration["initial_records"][branch_index], pool)
        selected = []
        for ordinal, packed in enumerate(branch["steps"]):
            step = expand_record(packed, pool)
            if (
                step["ordinal"] != ordinal
                or step["status"] != "executed"
                or step["before"] != previous
                or F(step["before"]["state"]["time"]) != F(3, 2) + ordinal * DT
                or F(step["endpoint"]["state"]["time"]) != F(3, 2) + (ordinal + 1) * DT
            ):
                raise ValueError("localized saved chronology differs")
            if F(step["before"]["state"]["time"]) == START:
                selected.append(step)
            previous = step["endpoint"]
        _equal(
            previous,
            expand_record(branch["terminal_record"], pool),
            "localized terminal record",
        )
        if len(selected) != 1:
            raise ValueError("one localized EN interval is required")
        localized.append(admit_trace(selected[0], original, old[0]))
    second = _pair(*localized)
    for row in (old[1], *localized):
        _common(old[0], row, source_required=False)
    n = len(original.source.nodes)
    t = tuple(
        tuple(old[0]["S"][i][j] - DT * old[0]["A"][i][j] for j in range(n))
        for i in range(n)
    )
    return {
        "original_reference": original,
        "children": children,
        "witnesses": {"cohort": cohort, "localized": second},
        "common_coefficients": {
            "S": old[0]["S"],
            "A": old[0]["A"],
            "T": t,
            "c": old[0]["c"],
            "metric_weights": original.metric_weights,
            "cross_experiment_b_equal": old[0]["b"] == localized[0]["b"],
            "paired_b_cancels": True,
            "dt": DT,
        },
        "native_calls": 0,
        "new_trajectories": 0,
        "kernel_calls": 0,
        "scope": "Common declared affine coefficients, with independent retained realization defects; no common kernel source identity or future policy guarantee",
    }


def load_comparable_witnesses(
    reset_path=RESET_PATH,
    restoration_path=RESTORATION_PATH,
    *,
    expected_reset_sha256=RESET_SHA256,
    expected_restoration_sha256=RESTORATION_SHA256,
):
    """Authenticate the two immutable reports, then admit their saved witnesses."""
    old, first = reset._load(
        reset_path, expected_reset_sha256, "O3.b-retained-EN-AL-reset-accounting"
    )
    new, second = reset._load(
        restoration_path,
        expected_restoration_sha256,
        "O3.a-localized-regional-restoration",
    )
    result = admit_reports(old, new)
    for path, expected in (
        (reset_path, expected_reset_sha256),
        (restoration_path, expected_restoration_sha256),
    ):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("historical input changed during offline admission")
    result["historical_inputs"] = {"reset": first, "restoration": second}
    return result
