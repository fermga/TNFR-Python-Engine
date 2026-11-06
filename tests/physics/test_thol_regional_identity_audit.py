"""Portable temporal regional evidence; no graph or native execution."""

import hashlib
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F

import pytest

from benchmarks import thol_regional_identity_audit as audit
from benchmarks.thol_preparation_policy import _literal
from benchmarks.thol_pressure_feedback import _payload
from tests.physics.test_thol_regional_balance_audit import _write_fixture
from tests.physics.test_thol_regional_balance_audit import prior as _prior_fixture
from tnfr.physics.forcing_realization import (
    NonEpiForcingObservation,
    decompose_non_epi_forcing,
)
from tnfr.physics.support_transport import _from_data

# Register the existing portable fixture without introducing a second constructor.
prior = _prior_fixture


def _capture(state, template, *, phase_increment=F(0)):
    """Encode declared forcing data; do not evaluate a phase kernel."""
    raw = template["observation"]
    original = raw["snapshot"]
    snapshot = _from_data(
        state["nodes"],
        [(i, j, F(w)) for i, j, w in original["conductance"]],
        original["support_neighbors"],
        state["epi"],
        state["capacity"],
        state["pressure"],
    )
    weights = tuple((name, F(value)) for name, value in raw["normalized_weights"])
    w = dict(weights)
    pg = tuple(F(value) + phase_increment for value in raw["phase_gradient"])
    forcing = tuple(
        w["phase"] * p + w["vf"] * v + w["topo"] * t
        for p, v, t in zip(
            pg, snapshot.capacity_gradient, snapshot.topology_gradient, strict=True
        )
    )
    model = tuple(
        w["epi"] * g + f for g, f in zip(snapshot.epi_gradient, forcing, strict=True)
    )
    fresh = tuple(F(float(value + F(1, 32))) for value in model)
    observation = NonEpiForcingObservation(
        snapshot,
        tuple(map(F, state["phase"])),
        w["epi"],
        forcing,
        pg,
        weights,
        fresh,
        tuple(p - q for p, q in zip(fresh, model, strict=True)),
        tuple(p - q for p, q in zip(snapshot.stored_pressure, fresh, strict=True)),
    )
    return _payload(
        {
            "available": True,
            "payload": {
                "observation": asdict(observation),
                "components": decompose_non_epi_forcing(observation),
            },
        }
    )


def _record(state, prior, *, glyph_count=0):
    attributes = deepcopy(prior["common_source"]["node_attributes"])
    for _, data in attributes:
        data.update(
            glyph_history=_literal(["IL"] * glyph_count),
            _operator_step=glyph_count,
            _grammar_u2_debt=0,
            _grammar_prior_coherence=True,
            stable_count=0,
        )
    return {
        "state": deepcopy(state),
        "node_attributes": attributes,
        "ordered_neighbors": [
            [node, [state["nodes"][j] for j in row]]
            for node, row in zip(
                state["nodes"],
                prior["original_reference"]["source"]["support_neighbors"],
                strict=True,
            )
        ],
        "graph_attributes": {
            "GLYPH_FACTORS": _literal({"IL_dnfr_factor": 0.5}),
            "_dnfr_weights": {
                key: _literal(float(F(value)))
                for key, value in prior["prefix"]["coupling"]["refreshed_forcing"][
                    "observation"
                ]["normalized_weights"]
            },
            "history": {
                "since_AL": {str(n): 1 for n in state["nodes"]},
                "since_EN": {str(n): 1 for n in state["nodes"]},
            },
        },
    }


@pytest.fixture
def native_fixture(prior):
    """Declare a 16-node native-shaped receipt, never execute its transitions."""
    prior = deepcopy(prior)
    template = prior["prefix"]["coupling"]["refreshed_forcing"]
    start = deepcopy(prior["prefix"]["coupling"]["after_refreshed"])
    start["time"] = 1.5
    start["phase"][0] = math.tau
    generated = deepcopy(start)
    generated["pressure"] = [
        float(F(value))
        for value in _capture(start, template)["payload"]["observation"][
            "full_kernel_pressure"
        ]
    ]
    records = []

    def row(name, before, after, **extra):
        records.append(
            {
                "ordinal": len(records),
                "boundary": name,
                "outcome": "completed",
                "before": _record(before, prior),
                "after": _record(after, prior),
                **extra,
            }
        )
        return records[-1]

    row("_prepare_dnfr", start, generated)
    row("_refresh_delta_nfr", start, generated)
    row("compute_Si", generated, generated)
    current = deepcopy(generated)
    calls = []
    for i, node in enumerate(start["nodes"]):
        after = deepcopy(current)
        after["pressure"][i] = 0.5 * current["pressure"][i]
        entry = row("apply_glyph", current, after, node=node, glyph="IL")
        calls.append({k: entry[k] for k in ("node", "glyph", "ordinal", "outcome")})
        current = after
    consumed = deepcopy(current)
    integrated = deepcopy(consumed)
    integrated["time"] = 1.75
    integrated["epi"] = [
        float(F(x) + F(1, 4) * F(nu) * F(p))
        for x, nu, p in zip(
            consumed["epi"], consumed["capacity"], consumed["pressure"], strict=True
        )
    ]
    integration = row(
        "integrate",
        consumed,
        integrated,
        effective_arguments={
            "dt": _literal(0.25),
            "method": "euler",
            "t": None,
            "n_jobs": None,
        },
        integrator_type="tnfr.dynamics.integrators.DefaultIntegrator",
    )
    normalized = deepcopy(integrated)
    normalized["phase"][0] = 0.0
    coordinated = deepcopy(normalized)
    coordinated["phase"] = [0.125] * 16
    phase = row("coordinate_global_local_phase", normalized, coordinated)
    endpoint = deepcopy(coordinated)
    endpoint["time"] = 1.75
    row("adapt_vf_after_structural_stability", coordinated, coordinated)
    prior["endpoint"] = deepcopy(start)
    branch = {
        "branch": "control",
        "status": "executed",
        "before": _record(start, prior),
        "endpoint": _record(endpoint, prior),
        "before_capture": _capture(start, template),
        "endpoint_capture": _capture(endpoint, template, phase_increment=F(1, 16)),
        "native_trace": {
            "status": "executed",
            "failure": None,
            "boundaries": records,
            "actual_glyph_calls": calls,
            "native_calls": 1,
            "selector_requests": [{"node": n, "glyph": "IL"} for n in start["nodes"]],
            "captures": {"integrator_entry": _capture(consumed, template)},
            "clamp_interval": {
                "after_integrate_ordinal": integration["ordinal"],
                "before_phase_ordinal": phase["ordinal"],
            },
        },
    }
    return prior, _payload(branch)


def _one(branch, name):
    return next(
        row for row in branch["native_trace"]["boundaries"] if row["boundary"] == name
    )


def _coordinates(branch):
    """Independent exact coefficients from raw primitive receipt fields."""
    source = _one(branch, "integrate")["before"]["state"]
    target = branch["endpoint"]["state"]
    nodes = source["nodes"]
    index = {node: i for i, node in enumerate(nodes)}
    weights = {}
    for left, right, data in source["edges"]:
        i, j = index[left], index[right]
        weights[i, j] = weights[j, i] = F(data["weight"])
    degree = tuple(
        sum((w for (j, _), w in weights.items() if j == i), F(0)) for i in range(16)
    )
    nu = tuple(map(F, source["capacity"]))
    metric = tuple(d / n for d, n in zip(degree, nu, strict=True))
    x, xf, pressure = (
        tuple(map(F, values))
        for values in (source["epi"], target["epi"], source["pressure"])
    )
    rate = tuple(n * p for n, p in zip(nu, pressure, strict=True))
    expected = tuple(a + F(1, 4) * r for a, r in zip(x, rate, strict=True))
    defect = tuple(a - b for a, b in zip(xf, expected, strict=True))
    return degree, metric, x, xf, rate, expected, defect, weights


def test_nine_exact_finite_budgets_and_regional_source_terms(native_fixture):
    prior, branch = native_fixture
    result = audit.audit_branch(prior, branch)
    degree, metric, x, xf, rate, expected, defect, weights = _coordinates(branch)
    assert result["regional_observation_count"] == 9
    assert result["native_calls"] == result["new_trajectories"] == 0
    assert result["actual_lineage"]["pairs"] == tuple(
        map(tuple, prior["lineage"]["parent_children"])
    )
    assert [item["label"] for item in result["regions"]] == [
        f"ancestry_pair_{i}" for i in range(8)
    ] + ["actual_children"]
    for item in result["regions"]:
        finite = item["finite_budget"]
        balance = finite["balance"]
        ids = tuple(balance["region_indices"])
        htotal = sum(metric[i] for i in ids)

        def total(values):
            return sum((metric[i] * values[i] for i in ids), F(0))

        def mean(values):
            return total(values) / htotal

        def variance(values):
            return (
                sum((metric[i] * (values[i] - mean(values)) ** 2 for i in ids), F(0))
                / 2
            )

        assert balance["strengths"] == degree and balance["metric_weights"] == metric
        assert finite["expected_epi"] == expected and finite["state_defect"] == defect
        assert finite["mass_change"] == total(xf) - total(x)
        assert finite["mass_drift_term"] == total(rate) / 4
        assert finite["mass_defect_term"] == total(defect)
        assert finite["after_mean"] == mean(xf)
        assert finite["variance_change"] == variance(xf) - variance(x)
        drift = sum((metric[i] * (x[i] - mean(x)) * rate[i] for i in ids), F(0)) / 4
        quadratic = (
            sum((metric[i] * (rate[i] - mean(rate)) ** 2 for i in ids), F(0)) / 32
        )
        linear_defect = sum(
            (
                metric[i] * (expected[i] - mean(expected)) * (defect[i] - mean(defect))
                for i in ids
            ),
            F(0),
        )
        assert finite["variance_drift_term"] == drift
        assert finite["variance_quadratic_term"] == quadratic
        assert finite["variance_defect_linear_term"] == linear_defect
        assert finite["variance_defect_quadratic_term"] == variance(defect)
        assert finite[
            "variance_change"
        ] == drift + quadratic + linear_defect + variance(defect)
        assert (
            finite["mass_identity_residual"]
            == finite["variance_identity_residual"]
            == 0
        )
        cut = -F(1, 2) * sum(
            (
                w * (x[i] - x[j])
                for (i, j), w in weights.items()
                if i in ids and j not in ids
            ),
            F(0),
        )
        assert balance["mass_boundary_rate"] == cut
        for name, values in result["pressure_split"].items():
            terms = item["first_order_pressure_terms"][name]
            mrate = sum((degree[i] * values[i] for i in ids), F(0))
            vrate = sum((degree[i] * (x[i] - mean(x)) * values[i] for i in ids), F(0))
            assert terms["weighted_total_rate"] == mrate
            assert terms["variance_rate"] == vrate
            assert terms["weighted_total_first_order_term"] == mrate / 4
            assert terms["variance_first_order_term"] == vrate / 4
    # These are full-graph degrees, not the induced parent-child pair degree.
    assert degree == (F(5, 2),) * 8 + (F(1, 2),) * 8
    assert metric == (F(5, 2),) * 8 + (F(1),) * 8
    assert (
        result["regions"][-1]["finite_budget"]["balance"]["internal_dissipation"] == 0
    )


def test_intentional_il_write_source_change_and_normalization_are_distinct(
    native_fixture,
):
    prior, branch = native_fixture
    result = audit.audit_branch(prior, branch)
    split = result["pressure_split"]
    generated = tuple(
        map(F, _one(branch, "_prepare_dnfr")["after"]["state"]["pressure"])
    )
    assert split["IL_associated_write"] == tuple(-p / 2 for p in generated)
    assert any(split["IL_associated_write"])
    assert not any(split["generation_minus_fresh_kernel"])
    assert split["generation_minus_model"] == split["fresh_kernel_minus_model"]
    assert (
        result["IL_factor_certified"] is False
    )  # Attribution is not an IL kernel certificate.
    assert result["generation_source_reuse"]["derived_reuse"] is True
    assert "pressure_generation" not in branch["native_trace"]["captures"]
    assert result["source_channel_delta_endpoint_minus_entry"] == {
        "phase": (F(1, 128),) * 16,
        "vf": (F(0),) * 16,
        "topo": (F(0),) * 16,
    }
    stages = result["phase_by_stage"]
    assert stages["integration_exit"][0] == F(math.tau)
    assert stages["after_bracketed_normalization"][0] == 0
    assert stages["endpoint"] == stages["after_coordination"] == (F(1, 8),) * 16
    assert stages["integration_entry"] == stages["generation"]


def test_same_lineage_is_not_relative_form_or_mean_preservation(native_fixture):
    prior, branch = native_fixture
    result = audit.audit_branch(prior, branch)
    assert all(not row["identity"]["same_relative_epi"] for row in result["regions"])
    assert all(not row["identity"]["same_absolute_epi"] for row in result["regions"])
    assert all(row["identity"]["same_capacity"] for row in result["regions"])
    assert any(row["identity"]["weighted_mean_change"] for row in result["regions"])
    assert "persistence" in result["scope"]


def test_relative_identity_permits_translation_but_reports_mean_change(native_fixture):
    prior, branch = native_fixture
    # Change only declared post-integration EPI; a finite endpoint defect remains
    # explicit. This is a synthetic readout distinction, not a replay claim.
    shifted = [x + 0.25 for x in branch["before"]["state"]["epi"]]
    integration = _one(branch, "integrate")
    for row in branch["native_trace"]["boundaries"]:
        for side in ("before", "after"):
            if row["ordinal"] > integration["ordinal"] or (
                row is integration and side == "after"
            ):
                row[side]["state"]["epi"] = shifted.copy()
    branch["endpoint"]["state"]["epi"] = shifted.copy()
    branch["endpoint_capture"] = _capture(
        branch["endpoint"]["state"],
        prior["prefix"]["coupling"]["refreshed_forcing"],
        phase_increment=F(1, 16),
    )
    result = audit.audit_branch(prior, branch)
    for item in result["regions"]:
        identity = item["identity"]
        assert identity["same_relative_epi"] and not identity["same_absolute_epi"]
        assert identity["weighted_mean_change"] == F(1, 4)
        assert any(item["finite_budget"]["state_defect"])


@pytest.mark.parametrize(
    "change",
    (
        "prior_endpoint",
        "interval",
        "dt",
        "integration_clock",
        "boundary_order",
        "duplicate_ordinal",
        "IL_order",
        "IL_summary",
        "IL_nonlocal_pressure",
        "preflow_epi",
        "postflow_epi",
        "support",
        "capacity",
        "parent_pointer",
        "node_order",
        "generation_phase",
        "generation_neighbors",
        "generation_config",
        "vectorized_config",
        "intermediate_phase",
        "normalized_weights",
        "clamp_bracket",
        "clamp_nonphase",
        "endpoint_pressure",
        "endpoint_phase",
        "snapshot_cache",
        "reference_cache",
        "forcing_components",
    ),
)
def test_inconsistent_native_receipt_is_rejected(native_fixture, change):
    prior, branch = native_fixture
    trace = branch["native_trace"]
    generation, integration, phase = (
        _one(branch, name)
        for name in ("_prepare_dnfr", "integrate", "coordinate_global_local_phase")
    )
    calls = [row for row in trace["boundaries"] if row["boundary"] == "apply_glyph"]
    if change == "prior_endpoint":
        prior["endpoint"]["epi"][0] += 0.125
    elif change == "interval":
        branch["endpoint"]["state"]["time"] = 2.0
    elif change == "dt":
        integration["effective_arguments"]["dt"] = _literal(0.5)
    elif change == "integration_clock":
        integration["before"]["state"]["time"] = 1.25
    elif change == "boundary_order":
        trace["boundaries"][0], trace["boundaries"][1] = (
            trace["boundaries"][1],
            trace["boundaries"][0],
        )
    elif change == "duplicate_ordinal":
        trace["boundaries"][1]["ordinal"] = 0
    elif change == "IL_order":
        calls[0]["node"], calls[1]["node"] = calls[1]["node"], calls[0]["node"]
    elif change == "IL_summary":
        trace["actual_glyph_calls"][0]["glyph"] = "EN"
    elif change == "IL_nonlocal_pressure":
        calls[0]["after"]["state"]["pressure"][1] += 0.125
    elif change == "preflow_epi":
        calls[0]["after"]["state"]["epi"][0] += 0.125
    elif change == "postflow_epi":
        phase["after"]["state"]["epi"][0] += 0.125
    elif change == "support":
        calls[0]["after"]["state"]["edges"][0][2]["weight"] = 0.75
    elif change == "capacity":
        calls[0]["after"]["state"]["capacity"][0] = 0.75
    elif change == "parent_pointer":
        calls[0]["after"]["node_attributes"][8][1]["parent_node"] = "p1"
    elif change == "node_order":
        calls[0]["after"]["state"]["nodes"].reverse()
    elif change == "generation_phase":
        generation["after"]["state"]["phase"][0] = 0.25
    elif change == "generation_neighbors":
        generation["after"]["ordered_neighbors"][0][1].reverse()
    elif change == "generation_config":
        generation["after"]["graph_attributes"]["_dnfr_hook_name"] = "changed-source"
    elif change == "vectorized_config":
        generation["after"]["graph_attributes"]["vectorized_dnfr"] = False
    elif change == "intermediate_phase":
        calls[1]["before"]["state"]["phase"][0] = 0.25
    elif change == "normalized_weights":
        for record in (generation["after"], integration["before"]):
            record["graph_attributes"]["_dnfr_weights"]["epi"] = _literal(0.75)
    elif change == "clamp_bracket":
        trace["clamp_interval"]["before_phase_ordinal"] = integration["ordinal"]
    elif change == "clamp_nonphase":
        phase["before"]["state"]["pressure"][0] += 0.125
    elif change == "endpoint_pressure":
        branch["endpoint"]["state"]["pressure"][0] += 0.125
    elif change == "endpoint_phase":
        branch["endpoint"]["state"]["phase"][0] += 0.125
    elif change == "snapshot_cache":
        trace["captures"]["integrator_entry"]["payload"]["observation"]["snapshot"][
            "epi_gradient"
        ][0] = "999"
    elif change == "reference_cache":
        prior["original_reference"]["metric_weights"][0] = "999"
    elif change == "forcing_components":
        trace["captures"]["integrator_entry"]["payload"]["components"][0][1][0] = "999"
    else:
        raise AssertionError("unhandled corruption control")
    with pytest.raises((ValueError, TypeError)):
        audit.audit_branch(prior, branch)


def _write_native_fixture(path, prior, branch):
    _write_fixture(path, prior)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["branches"][0] = branch
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbid_runtime(monkeypatch):
    from benchmarks import thol_full_state_response, thol_native_runtime_response
    from tnfr.dynamics import coordination, dnfr, runtime
    from tnfr.physics import forcing_realization

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "retained regional audit must not execute a graph or phase kernel"
        )

    monkeypatch.setattr(runtime, "step", forbidden)
    monkeypatch.setattr(thol_full_state_response, "replay_response_branch", forbidden)
    monkeypatch.setattr(thol_native_runtime_response, "run_study", forbidden)
    monkeypatch.setattr(coordination, "coordinate_global_local_phase", forbidden)
    monkeypatch.setattr(dnfr, "_compute_dnfr_common", forbidden)
    monkeypatch.setattr(forcing_realization, "capture_non_epi_forcing", forbidden)


def test_portable_cli_digest_provenance_and_original_bytes(
    native_fixture, tmp_path, monkeypatch
):
    prior, branch = native_fixture
    path, output = tmp_path / "fixture.json", tmp_path / "audit.json"
    digest = _write_native_fixture(path, prior, branch)
    original = path.read_bytes()
    _forbid_runtime(monkeypatch)
    monkeypatch.setattr(
        audit, "current_git_source_provenance", lambda *args: ("a" * 40, False, None)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--input",
            str(path),
            "--expected-sha256",
            digest,
            "--output",
            str(output),
        ],
    )
    audit.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["manifest"]["result_status"] == "derived"
    assert result["historical_input"]["sha256"] == digest
    assert result["regional_observation_count"] == 9
    assert result["native_calls"] == result["new_trajectories"] == 0
    assert path.read_bytes() == original
    canonical = json.dumps(
        branch, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert result["retained_branch"]["sha256"] == hashlib.sha256(canonical).hexdigest()
    monkeypatch.setattr(
        sys, "argv", ["audit", "--input", str(path), "--output", str(path)]
    )
    with pytest.raises(ValueError, match="overwrite"):
        audit.main()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="digest"):
        audit.run_study(path, expected_sha256=digest)


def test_pinned_interval_if_available_without_runtime_or_kernel(monkeypatch):
    if not audit.INPUT_PATH.exists():
        pytest.skip("ignored retained native interval is not available")
    _forbid_runtime(monkeypatch)
    result = audit.run_study()
    assert result["interval"] == (F(3, 2), F(7, 4))
    assert result["regional_observation_count"] == 9
    assert result["native_calls"] == result["new_trajectories"] == 0
    for region in result["regions"]:
        budget = region["finite_budget"]
        assert (
            budget["mass_identity_residual"]
            == budget["variance_identity_residual"]
            == 0
        )
    assert result["generation_source_reuse"]["derived_reuse"]
    assert any(result["pressure_split"]["IL_associated_write"])
