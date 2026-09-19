"""Portable detached regional response controls; no archived outcome is required."""

import hashlib
import json
from copy import deepcopy
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from benchmarks import thol_regional_recovery_audit as audit
from tnfr.physics.support_transport import _from_data


def _fixture(*, zero_control=False, uniform_error=False):
    """Explicit synthetic records, not a simulated or causally admitted study."""
    nodes = tuple(range(16))
    pairs = tuple((i, i + 8) for i in range(8))
    hierarchy = {str(i): [i + 8] for i in range(8)}
    children = {**hierarchy, **{str(i): [] for i in range(8, 16)}}
    edges = [(i, (i + 1) % 16, {"weight": float(1 + i % 3)}) for i in nodes]
    conductance = tuple(
        sorted(
            (a, b, F(data["weight"]))
            for i, j, data in edges
            for a, b in ((i, j), (j, i))
        )
    )
    support = tuple(
        tuple(sorted({j for i, j, _w in conductance if i == node})) for node in nodes
    )
    capacity = tuple(F(1 + i % 2) for i in nodes)
    initial = tuple(F(i % 4) + F(i, 8) for i in nodes)
    source = _from_data(nodes, conductance, support, initial, capacity, (F(0),) * 16)
    metric = tuple(
        sum(w for i, _j, w in conductance if i == node) / capacity[node]
        for node in nodes
    )
    original = SimpleNamespace(source=source, metric_weights=metric)

    def state(values, time):
        return {
            "nodes": list(nodes),
            "epi": list(values),
            "capacity": list(capacity),
            "time": time,
            "edges": deepcopy(edges),
            "hierarchy": deepcopy(hierarchy),
            "children": deepcopy(children),
        }

    birth_after = state(initial, F(1, 2))
    birth = {
        "parent_children": pairs,
        "before": {"nodes": tuple(range(8))},
        "after": birth_after,
        "children": [
            {"parent": p, "child": c, "node_data": {"parent_node": p}} for p, c in pairs
        ],
    }
    prior = {
        "prefix": {"birth": birth},
        "lineage": {
            "parent_children": pairs,
            "parents": tuple(range(8)),
            "children": tuple(range(8, 16)),
            "nodes": nodes,
        },
        "common_source": {
            "state": birth_after,
            "node_attributes": [
                (i, {"sub_nodes": [i + 8]} if i < 8 else {"parent_node": i - 8})
                for i in nodes
            ],
        },
    }
    pool = audit.reset.window.RecordPool()
    branches, firsts, priors = [], [], []
    for branch_index, name in enumerate(audit.reset.window.BRANCHES):
        records = []
        for k, time in enumerate(audit.TIMES):
            control = (
                (F(3),) * 16
                if zero_control
                else tuple((1 - F(k, 16)) * x + F(k, 32) for x in initial)
            )
            errors = (
                (F(1, 4),) * 16
                if uniform_error
                else tuple(
                    F(i + 1, 32) * (1 - F(k, 8)) if i >= 8 else F(0) for i in nodes
                )
            )
            values = tuple(
                x + branch_index * d for x, d in zip(control, errors, strict=True)
            )
            records.append({"state": state(values, time)})
        firsts.append(
            pool.pack({"branch": name, "status": "executed", "endpoint": records[0]})
        )
        steps = [
            pool.pack(
                {
                    "ordinal": k,
                    "status": "executed",
                    "before": records[k],
                    "endpoint": records[k + 1],
                }
            )
            for k in range(5)
        ]
        branches.append(
            {
                "branch": name,
                "steps": steps,
                "status": "completed",
                "completed_steps": 5,
                "final_record": pool.pack(records[-1]),
            }
        )
        priors.append({"branch": name, **deepcopy(prior)})
    retained = {
        "first_step_replays": firsts,
        "branches": branches,
        "record_pool": pool.nodes,
    }
    return retained, original, {"replayed_prior_reports": priors}


def _mutate_endpoint(retained, branch, index, change):
    """Keep record digests and chronology coherent while changing a primitive."""
    pool = audit.reset.window.RecordPool()
    decoded = []
    for b, item in enumerate(retained["branches"]):
        first = audit.reset.window.expand_record(
            retained["first_step_replays"][b], retained["record_pool"]
        )
        steps = [
            audit.reset.window.expand_record(ref, retained["record_pool"])
            for ref in item["steps"]
        ]
        records = [first["endpoint"]] + [step["endpoint"] for step in steps]
        if b == branch:
            change(records[index])
        first["endpoint"] = records[0]
        retained["first_step_replays"][b] = pool.pack(first)
        for k, step in enumerate(steps):
            step["before"], step["endpoint"] = records[k], records[k + 1]
        decoded.append((item, steps, records[-1]))
    for item, steps, last in decoded:
        item["steps"] = [pool.pack(step) for step in steps]
        item["final_record"] = pool.pack(last)
    retained["record_pool"] = pool.nodes


def _oracle(values, weights):
    mean = sum(w * x for w, x in zip(weights, values, strict=True)) / sum(weights)
    centered = tuple(x - mean for x in values)
    return (
        mean,
        centered,
        sum(w * x * x for w, x in zip(weights, centered, strict=True)) / 2,
    )


def test_all_fixed_regions_times_and_independent_weighted_arithmetic():
    retained, original, native = _fixture()
    result = audit.audit_window(retained, original, native)
    assert len(result["regions"]) == 9 and result["regional_endpoint_count"] == 54
    assert result["actual_lineage"]["pairs"] == tuple((i, i + 8) for i in range(8))
    for region in result["regions"]:
        weights = tuple(original.metric_weights[i] for i in region["full_node_indices"])
        assert region["metric_weights"] == weights
        assert tuple(row["time"] for row in region["endpoints"]) == audit.TIMES
        first = region["endpoints"][0]["control_epi"]
        for row in region["endpoints"]:
            x, y = row["control_epi"], row["perturbed_epi"]
            dm, dz, error = _oracle(
                tuple(b - a for a, b in zip(x, y, strict=True)), weights
            )
            cm, cz, variance = _oracle(x, weights)
            drift_mean, drift, drift_energy = _oracle(
                tuple(b - a for a, b in zip(first, x, strict=True)), weights
            )
            assert (
                row["paired"]["weighted_mean_offset"],
                row["paired"]["centered_epi_difference"],
                row["paired"]["centered_H_energy"],
            ) == (dm, dz, error)
            assert (
                row["control_mean"],
                row["control_centered_epi"],
                row["control_variance"],
            ) == (cm, cz, variance)
            assert (
                row["control_mean_drift_from_start"],
                row["control_centered_drift_from_start"],
                row["control_centered_drift_energy"],
            ) == (drift_mean, drift, drift_energy)
            assert row["relative_error"]["value"] == error / variance
        summary = region["endpoint_summary"]
        assert summary["centered_error_decreased"]
        assert summary["control_variance_decreased"]
        assert not summary["control_centered_form_unchanged_at_every_endpoint"]
    assert not result["autonomous_maintenance_certified"]
    assert (
        result["native_calls"]
        == result["forcing_capture_calls"]
        == result["coordination_calls"]
        == 0
    )
    json.dumps(audit._payload(result), allow_nan=False)


def test_uniform_offset_has_zero_centered_error_but_retains_mean():
    result = audit.audit_window(*_fixture(uniform_error=True))
    for region in result["regions"]:
        for row in region["endpoints"]:
            assert row["paired"]["centered_H_energy"] == 0
            assert row["paired"]["weighted_mean_offset"] == F(1, 4)
            assert row["relative_error"]["value"] == 0
        assert not region["endpoint_summary"]["centered_error_decreased"]


@pytest.mark.parametrize(
    "uniform,case",
    [
        (False, "zero_control_with_positive_error"),
        (True, "zero_control_and_zero_error"),
    ],
)
def test_zero_control_contrast_keeps_ratio_unavailable(uniform, case):
    result = audit.audit_window(*_fixture(zero_control=True, uniform_error=uniform))
    for region in result["regions"]:
        for row in region["endpoints"]:
            assert row["relative_error"] == {
                "available": False,
                "value": None,
                "zero_case": case,
            }
        assert region["endpoint_summary"]["relative_error_change"] is None
        assert region["endpoint_summary"]["relative_error_decreased"] is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda record: record["state"]["capacity"].__setitem__(0, "3"), "capacity"),
        (
            lambda record: record["state"]["edges"][0][2].__setitem__("weight", 7.0),
            "conductance",
        ),
        (lambda record: record["state"]["edges"].pop(), "conductance"),
        (lambda record: record["state"]["nodes"].reverse(), "nodes"),
        (
            lambda record: record["state"].__setitem__("time", "9/4"),
            "physical endpoints",
        ),
        (
            lambda record: record["state"]["epi"].__setitem__(0, float("nan")),
            "nonfinite|finite",
        ),
        (lambda record: record["state"]["children"]["0"].clear(), "hierarchy"),
    ],
)
def test_endpoint_primitive_mismatch_is_rejected(mutation, match):
    retained, original, native = _fixture()
    # NaN is rejected by the shared codec before it can reach arithmetic.
    with pytest.raises((ValueError, TypeError), match=match):
        _mutate_endpoint(retained, 1, 3, mutation)
        audit.audit_window(retained, original, native)


def test_wrong_cached_metric_and_parentage_are_rejected():
    retained, original, native = _fixture()
    changed = SimpleNamespace(source=original.source, metric_weights=(F(1),) * 16)
    with pytest.raises(ValueError, match="metric"):
        audit.audit_window(retained, changed, native)
    native["replayed_prior_reports"][1]["prefix"]["birth"]["children"][0]["node_data"][
        "parent_node"
    ] = 7
    with pytest.raises(ValueError, match="parent pointers"):
        audit.audit_window(retained, original, native)


def test_compact_tamper_and_incomplete_window_reject():
    retained, original, native = _fixture()
    key = retained["first_step_replays"][0]["$record"]
    retained["record_pool"][key]["items"].append(["extra", True])
    with pytest.raises(ValueError, match="digest"):
        audit.audit_window(retained, original, native)
    retained, original, native = _fixture()
    retained["branches"][1]["steps"].pop()
    with pytest.raises(ValueError, match="completed"):
        audit.audit_window(retained, original, native)


def test_inputs_unchanged_and_no_runtime_or_capture_is_called(monkeypatch):
    retained, original, native = _fixture()
    before = deepcopy((retained, native))

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "offline regional audit must not execute a runtime or capture"
        )

    monkeypatch.setattr(audit.reset.window.native, "run_native_branch", forbidden)
    monkeypatch.setattr(audit.reset.window.native, "_trace_step", forbidden)
    monkeypatch.setattr(audit.reset.window.native, "_capture", forbidden)
    monkeypatch.setattr(audit.reset.window, "continue_window", forbidden)
    from tnfr.dynamics import coordination
    from tnfr.physics import forcing_realization

    monkeypatch.setattr(coordination, "coordinate_global_local_phase", forbidden)
    monkeypatch.setattr(forcing_realization, "capture_non_epi_forcing", forbidden)
    audit.audit_window(retained, original, native)
    assert (retained, native) == before


def test_run_study_uses_pinned_loader_and_rechecks_bytes(tmp_path, monkeypatch):
    retained, original, native = _fixture()
    paths = {name: tmp_path / f"{name}.json" for name in ("window", "native")}
    for path in paths.values():
        path.write_text("{}", encoding="utf8")
    bindings = {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    }
    seen = []

    def load(left, right, **kwargs):
        seen.append((left, right, kwargs))
        return retained, original, bindings

    monkeypatch.setattr(audit.reset, "load_evidence", load)
    monkeypatch.setattr(audit.reset, "_load", lambda *args: (native, {}))
    args = {
        "expected_window_sha256": bindings["window"]["sha256"],
        "expected_native_sha256": bindings["native"]["sha256"],
    }
    result = audit.run_study(paths["window"], paths["native"], **args)
    assert seen == [(paths["window"], paths["native"], args)]
    assert result["historical_inputs"] == bindings
    paths["native"].write_text('{"changed":true}', encoding="utf8")
    with pytest.raises(RuntimeError, match="changed"):
        audit.run_study(paths["window"], paths["native"], **args)
