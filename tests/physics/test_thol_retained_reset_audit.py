"""Offline EN/AL accounting controls; no graph or runtime trajectory executes."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.thol_preparation_policy import _literal
from benchmarks.thol_pressure_feedback import _payload
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import NonEpiForcingObservation
from tnfr.physics.support_transport import _from_data

F = Fraction
ROOT = Path(__file__).resolve().parents[2]


def _apply(matrix, vector):
    return tuple(
        sum((a * b for a, b in zip(row, vector, strict=True)), F(0)) for row in matrix
    )


def _product(left, right):
    return tuple(
        tuple(
            sum((a * b for a, b in zip(row, column, strict=True)), F(0))
            for column in zip(*right, strict=True)
        )
        for row in left
    )


def _identity(size):
    return tuple(tuple(F(i == j) for j in range(size)) for i in range(size))


def _fixture(glyph="EN", *, initial=(0.5, 0.0), factor=None):
    """Construct detached two-node receipt fixtures, never execute a graph.

    The only state transition here is explicit test data for finite arithmetic
    admission. Shared owners construct the passive reference, not the outcomes.
    """
    factor = (0.25 if glyph == "EN" else 0.125) if factor is None else factor
    pressure = (initial[1] - initial[0], initial[0] - initial[1])
    edges = ((0, 1, F(1)), (1, 0, F(1)))
    support = ((1,), (0,))
    snap = _from_data((0, 1), edges, support, initial, (1, 1), pressure)
    reference = derive_forced_support_balance(snap, epi_weight=1, forcing=(0, 0))
    obs = NonEpiForcingObservation(
        snapshot=snap,
        phase=(F(0), F(0)),
        epi_weight=F(1),
        forcing=(F(0), F(0)),
        phase_gradient=(F(0), F(0)),
        normalized_weights=(
            ("phase", F(0)),
            ("epi", F(1)),
            ("vf", F(0)),
            ("topo", F(0)),
        ),
        full_kernel_pressure=tuple(map(F, pressure)),
        kernel_pressure_defect=tuple(
            F(p) - g for p, g in zip(pressure, snap.epi_gradient, strict=True)
        ),
        stored_pressure_residual=(F(0), F(0)),
    )
    configuration = {
        "GLYPH_FACTORS": {
            "EN_mix": factor if glyph == "EN" else 0.25,
            "AL_boost": factor if glyph == "AL" else 0.125,
        },
        "EPI_MIN": -10.0,
        "EPI_MAX": 10.0,
    }

    def record(values, time=2.25):
        return {
            "state": {
                "nodes": [0, 1],
                "epi": list(values),
                "capacity": [1.0, 1.0],
                "phase": [0.0, 0.0],
                "pressure": list(pressure),
                "time": time,
                "edges": [[0, 1, {"weight": 1.0}]],
            },
            "ordered_neighbors": [[0, [1]], [1, [0]]],
            "node_attributes": [
                [node, {"EPI": _literal(value), "epi_kind": "AL"}]
                for node, value in enumerate(values)
            ],
            "graph_attributes": _literal(deepcopy(configuration)),
        }

    current = tuple(initial)
    generation = record(current)
    rows = [
        {
            "ordinal": 0,
            "boundary": "_prepare_dnfr",
            "outcome": "completed",
            "before": deepcopy(generation),
            "after": deepcopy(generation),
        },
        {
            "ordinal": 1,
            "boundary": "_refresh_delta_nfr",
            "outcome": "completed",
            "before": deepcopy(generation),
            "after": deepcopy(generation),
        },
    ]
    for node in (0, 1):
        before = record(current)
        updated = list(current)
        if glyph == "EN":
            updated[node] = (1.0 - factor) * current[node] + factor * current[1 - node]
        else:
            updated[node] = current[node] + factor
        current = tuple(updated)
        rows.append(
            {
                "ordinal": len(rows),
                "boundary": "apply_glyph",
                "outcome": "completed",
                "node": node,
                "glyph": glyph,
                "before": before,
                "after": record(current),
            }
        )
    reset = current
    after = tuple(x + 0.25 * p for x, p in zip(reset, pressure, strict=True))
    rows.append(
        {
            "ordinal": len(rows),
            "boundary": "integrate",
            "outcome": "completed",
            "before": record(reset),
            "after": record(after, 2.5),
        }
    )
    payload = {
        "observation": asdict(obs),
        "components": tuple((key, (F(0), F(0))) for key in ("phase", "vf", "topo")),
    }
    step = {
        "ordinal": 0,
        "status": "executed",
        "before": record(initial),
        "endpoint": record(after, 2.5),
        "native_trace": {
            "status": "executed",
            "boundaries": rows,
            "captures": {
                "pressure_generation": {"available": True, "payload": payload}
            },
        },
    }
    return _payload(step), reference


def _reset_oracle(step):
    """Direct ordered row updates, independent of the producer's row helper."""
    nodes = step["before"]["state"]["nodes"]
    size = len(nodes)
    composite, offset, propagated = _identity(size), (F(0),) * size, (F(0),) * size
    rows, defects, individual = [], [], []
    calls = [
        row
        for row in step["native_trace"]["boundaries"]
        if row["boundary"] == "apply_glyph"
    ]
    for call in calls:
        index = nodes.index(call["node"])
        local, shift = [list(row) for row in _identity(size)], [F(0)] * size
        factors = call["before"]["graph_attributes"]["GLYPH_FACTORS"]
        if call["glyph"] == "EN":
            mix = float.fromhex(factors["EN_mix"][1])
            neighbors = dict(call["before"]["ordered_neighbors"])[call["node"]]
            local[index] = [F(0)] * size
            local[index][index] = F(1.0 - mix)
            for neighbor in neighbors:
                j = nodes.index(neighbor)
                local[index][j] = F(float(local[index][j]) + mix / len(neighbors))
        else:
            shift[index] = F(float.fromhex(factors["AL_boost"][1]))
        local = tuple(map(tuple, local))
        x, y = (
            tuple(map(F, call[side]["state"]["epi"])) for side in ("before", "after")
        )
        ideal = tuple(a + b for a, b in zip(_apply(local, x), shift, strict=True))
        defect = tuple(a - b for a, b in zip(y, ideal, strict=True))
        individual = [_apply(local, value) for value in individual] + [defect]
        propagated = tuple(
            a + b for a, b in zip(_apply(local, propagated), defect, strict=True)
        )
        offset = tuple(a + b for a, b in zip(_apply(local, offset), shift, strict=True))
        composite = _product(local, composite)
        rows.append(local)
        defects.append(defect)
    return {
        "S": composite,
        "c": offset,
        "local_matrices": rows,
        "local_defects": defects,
        "propagated_individual": individual,
        "total_defect": propagated,
    }


def _assert_reset(audit, step):
    expected = _reset_oracle(step)
    reset = audit["reset"]
    assert reset["S"] == expected["S"] and reset["c"] == expected["c"]
    assert tuple(reset["propagated_local_defects"]) == tuple(
        expected["propagated_individual"]
    )
    assert reset["total_reset_defect"] == expected["total_defect"]
    for row, local, defect in zip(
        reset["local_rows"],
        expected["local_matrices"],
        expected["local_defects"],
        strict=True,
    ):
        index = audit["nodes"].index(row["node"])
        assert row["row"] == local[index] and row["local_defect"] == defect
        assert row["kernel_evaluation_defect"] + row["clipping_defect"] == defect[index]
    x0, xg, xi, xf = (audit["vectors"][key] for key in ("x0", "xg", "xi", "xf"))
    assert xg == tuple(
        a + b + c
        for a, b, c in zip(
            _apply(reset["S"], x0), reset["c"], reset["total_reset_defect"], strict=True
        )
    )
    raw = step["native_trace"]["captures"]["pressure_generation"]["payload"][
        "observation"
    ]
    snapshot = raw["snapshot"]
    weights = {(i, j): F(w) for i, j, w in snapshot["conductance"]}
    nu, e = tuple(map(F, snapshot["capacity"])), F(raw["epi_weight"])
    strengths = tuple(
        sum(w for (j, _), w in weights.items() if i == j) for i in range(len(x0))
    )
    a = tuple(
        tuple(
            e * nu[i] * (F(i == j) - weights.get((i, j), 0) / strengths[i])
            for j in range(len(x0))
        )
        for i in range(len(x0))
    )
    b = tuple(v * F(f) for v, f in zip(nu, raw["forcing"], strict=True))
    assert audit["generation"]["A"] == a and audit["generation"]["b"] == b
    ideal = tuple(
        reset_value + offset + F(1, 4) * (force - rate)
        for reset_value, offset, force, rate in zip(
            _apply(reset["S"], x0), reset["c"], b, _apply(a, x0), strict=True
        )
    )
    assert audit["runtime"]["ideal_endpoint"] == ideal
    runtime_names = ("pressure_term", "integration_remainder", "postintegration_change")
    actual_terms = [reset["total_reset_defect"]] + [
        audit["runtime"][name] for name in runtime_names
    ]
    assert xf == tuple(
        value + sum(parts, F(0))
        for value, parts in zip(ideal, zip(*actual_terms, strict=True), strict=True)
    )
    assert not any(reset["identity_residual"]) and not any(
        audit["runtime"]["identity_residual"]
    )


@pytest.mark.parametrize("glyph", ("EN", "AL"))
def test_detached_reset_matches_independent_ordered_algebra_without_runtime(glyph):
    from benchmarks.thol_retained_reset_audit import audit_reset_step

    step, reference = _fixture(glyph)
    saved = deepcopy(step)
    with patch(
        "tnfr.dynamics.step",
        side_effect=AssertionError("offline audit cannot advance dynamics"),
    ):
        audit = audit_reset_step(step, reference)
    _assert_reset(audit, step)
    assert step == saved


def test_reception_is_sequential_and_flow_keeps_pre_reset_pressure():
    from benchmarks.thol_retained_reset_audit import audit_reset_step

    step, reference = _fixture("EN")
    audit = audit_reset_step(step, reference)
    assert audit["reset"]["S"] == ((F(3, 4), F(1, 4)), (F(3, 16), F(13, 16)))
    assert audit["vectors"]["xg"] == (F(3, 8), F(3, 32))
    assert audit["vectors"]["xf"] == (F(1, 4), F(7, 32))
    refreshed_endpoint = (F(39, 128), F(21, 128))
    assert audit["runtime"]["ideal_endpoint"] != refreshed_endpoint
    assert audit["runtime"]["pre_generated_pressure_retained"] is True


def test_nonzero_local_defects_are_transported_through_later_reception_rows():
    from benchmarks.thol_retained_reset_audit import audit_reset_step

    step, reference = _fixture("EN", initial=(0.3, 0.2), factor=0.1)
    audit = audit_reset_step(step, reference)
    _assert_reset(audit, step)
    raw_defects = tuple(row["local_defect"] for row in audit["reset"]["local_rows"])
    assert any(any(row) for row in raw_defects)
    assert tuple(audit["reset"]["propagated_local_defects"]) != raw_defects


@pytest.mark.parametrize("glyph", ("EN", "AL"))
def test_paired_reset_map_mean_change_and_residuals(glyph):
    from benchmarks.thol_retained_reset_audit import audit_pair, audit_reset_step

    left, reference = _fixture(glyph)
    right, _ = _fixture(glyph, initial=(0.75, 0.0))
    a, b = audit_reset_step(left, reference), audit_reset_step(right, reference)
    paired = audit_pair(a, b, reference.metric_weights)
    assert paired["available"]
    expected_map = tuple(
        tuple(s - F(1, 4) * g for s, g in zip(sr, gr, strict=True))
        for sr, gr in zip(a["reset"]["S"], a["generation"]["A"], strict=True)
    )
    assert paired["pair_map"] == expected_map
    delta0 = tuple(
        y - x for x, y in zip(a["vectors"]["x0"], b["vectors"]["x0"], strict=True)
    )
    deltag = tuple(
        y - x for x, y in zip(a["vectors"]["xg"], b["vectors"]["xg"], strict=True)
    )
    deltaf = tuple(
        y - x for x, y in zip(a["vectors"]["xf"], b["vectors"]["xf"], strict=True)
    )
    assert (
        paired["mean_change_at_reset"] == sum(deltag, F(0)) / 2 - sum(delta0, F(0)) / 2
    )
    assert deltaf == tuple(
        value + sum(parts, F(0))
        for value, parts in zip(
            _apply(expected_map, delta0),
            zip(*paired["residual_terms"].values(), strict=True),
            strict=True,
        )
    )
    if glyph == "AL":
        assert (
            a["reset"]["S"] == _identity(2)
            and a["reset"]["c"] == b["reset"]["c"] == (F(1, 8),) * 2
        )
        assert paired["mean_change_at_reset"] == 0
    else:
        assert paired["mean_change_at_reset"] == -F(1, 128)


@pytest.mark.parametrize(
    "corruption",
    ("order", "target", "space", "factor", "pressure", "endpoint", "extra_refresh"),
)
def test_corrupt_record_boundaries_are_rejected(corruption):
    from benchmarks.thol_retained_reset_audit import audit_reset_step

    step, reference = _fixture("EN")
    rows = step["native_trace"]["boundaries"]
    first, second = rows[2], rows[3]
    if corruption == "order":
        rows[2], rows[3] = second, first
    elif corruption == "target":
        first["node"] = 1
    elif corruption == "space":
        first["before"]["state"]["nodes"].reverse()
    elif corruption == "factor":
        first["before"]["graph_attributes"]["GLYPH_FACTORS"]["EN_mix"] = [
            "binary64",
            float(0.125).hex(),
        ]
    elif corruption == "pressure":
        first["after"]["state"]["pressure"][0] = 0.0
    elif corruption == "endpoint":
        first["after"]["state"]["epi"][0] += 0.125
    else:
        rows.append(deepcopy(rows[1]))
    with pytest.raises((ValueError, RuntimeError)):
        audit_reset_step(step, reference)


def test_pair_abstains_if_declared_reset_map_differs():
    from benchmarks.thol_retained_reset_audit import audit_pair, audit_reset_step

    left, reference = _fixture("AL", factor=0.125)
    right, _ = _fixture("AL", factor=0.25)
    result = audit_pair(
        audit_reset_step(left, reference),
        audit_reset_step(right, reference),
        reference.metric_weights,
    )
    assert not result["available"] and not result["flags"]["c"]


def test_changed_input_bytes_are_rejected_before_payload_admission(tmp_path):
    from benchmarks.thol_retained_reset_audit import load_evidence

    path = tmp_path / "retained.json"
    raw = b'{"payload": 1}'
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="digest"):
        load_evidence(
            path, path, expected_window_sha256=digest, expected_native_sha256=digest
        )


def test_pinned_retained_four_resets_without_any_new_native_step():
    from benchmarks.thol_retained_reset_audit import WINDOW_PATH, run_study, window

    if not WINDOW_PATH.exists() or not window.NATIVE_PATH.exists():
        pytest.skip("historical research artifacts are optional local evidence")
    hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (WINDOW_PATH, window.NATIVE_PATH)
    }
    with patch(
        "tnfr.dynamics.step", side_effect=AssertionError("no new native invocation")
    ):
        report = run_study()
    archived = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
    for branch, source in zip(report["branches"], archived["branches"], strict=True):
        for audit, ordinal in zip(branch["resets"], (2, 4), strict=True):
            step = window.expand_record(
                source["steps"][ordinal], archived["record_pool"]
            )
            _assert_reset(audit, step)
    assert report["native_calls"] == report["new_trajectories"] == 0
    assert not report["autonomous_maintenance_certified"]
    assert [paired["formula"] for paired in report["paired"]] == ["S-hA", "I-hA"]
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
