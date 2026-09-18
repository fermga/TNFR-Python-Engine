"""Portable arithmetic admission; no archived trajectories or kernels run."""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
import hashlib
import json

import pytest

from benchmarks import thol_regional_response_admission as admission
from benchmarks import thol_child_distortion_audit as child
from tests.physics.test_thol_child_distortion_audit import _fixture
from tnfr.physics.forcing_realization import decompose_non_epi_forcing
from tnfr.physics.support_transport import _from_data
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import CoreExperimentManifest


def _trace(saved):
    """Embed exact dyadic detached fixture receipts in the retained trace schema."""
    snap, obs, _ = child._generation(saved["generation"]["observation"])
    nodes, n = snap.nodes, len(snap.nodes)
    config = saved["reset"]["local_rows"][0]["configuration"]

    def record(epi, time=F(9, 4)):
        return {"state": {"nodes": list(nodes), "epi": [float(F(x)) for x in epi],
                          "capacity": list(map(float, snap.capacity)),
                          "pressure": list(map(float, snap.stored_pressure)),
                          "phase": list(map(float, obs.phase)), "time": float(time),
                          "edges": [(nodes[i], nodes[j], {"weight": float(w)})
                                    for i, j, w in snap.conductance if i < j]},
                "ordered_neighbors": [(node, [nodes[j] for j in snap.support_neighbors[i]])
                                      for i, node in enumerate(nodes)],
                "graph_attributes": deepcopy(config)}

    def capture(observation):
        return {"available": True, "payload": {
            "observation": child._payload(asdict(observation)),
            "components": child._payload(decompose_non_epi_forcing(observation))}}

    vectors = saved["vectors"]
    before = record(vectors["x0"])
    boundaries = [{"ordinal": 0, "boundary": "_prepare_dnfr", "outcome": "completed",
                   "before": deepcopy(before), "after": deepcopy(before)},
                  {"ordinal": 1, "boundary": "_refresh_delta_nfr", "outcome": "completed",
                   "before": deepcopy(before), "after": deepcopy(before)}]
    for i, row in enumerate(saved["reset"]["local_rows"]):
        boundaries.append({"ordinal": i+2, "boundary": "apply_glyph", "outcome": "completed",
                           "node": nodes[i], "glyph": "EN", "before": record(row["before_epi"]),
                           "after": record(row["after_epi"])})
    boundaries.append({"ordinal": n+2, "boundary": "integrate", "outcome": "completed",
                       "integrator_type": "tnfr.dynamics.integrators.DefaultIntegrator",
                       "effective_arguments": {"dt": ["binary64", "0x1.0000000000000p-2"], "method": "euler"},
                       "before": record(vectors["xg"]), "after": record(vectors["xi"], F(5, 2))})
    entry_snap = _from_data(nodes, snap.conductance, snap.support_neighbors,
                            tuple(F(x) for x in vectors["xg"]), snap.capacity, snap.stored_pressure)
    fresh = tuple(obs.epi_weight*g+f for g, f in zip(entry_snap.epi_gradient, obs.forcing, strict=True))
    entry_obs = replace(obs, snapshot=entry_snap, full_kernel_pressure=fresh,
                        kernel_pressure_defect=(F(0),)*n,
                        stored_pressure_residual=tuple(p-v for p, v in zip(snap.stored_pressure, fresh, strict=True)))
    return {"status": "executed", "before": before, "endpoint": record(vectors["xf"], F(5, 2)),
            "native_trace": {"status": "executed", "boundaries": boundaries,
                             "captures": {"pressure_generation": capture(obs), "integrator_entry": capture(entry_obs)}}}


def _case(defects=False):
    # The two-node fixture has only dyadic coordinates and explicit defects;
    # float serialization therefore introduces no accidental test-data error.
    left, right, original, _ = _fixture(n=2, defects=defects)
    template = child._admit_reset(left, original)
    return _trace(right), original, template, child._admit_reset(right, original)


@pytest.mark.parametrize("defects", (False, True))
def test_common_declared_rows_and_complete_realized_remainders(defects, monkeypatch):
    step, original, template, expected = _case(defects)

    def forbidden(*args, **kwargs):
        raise AssertionError("no kernel, runtime or historical producer may run")

    for name in ("neighbor_epi_unweighted_mean", "neighbor_epi_blend_value",
                 "neighbor_epi_represented_affine_row", "_op_AL", "audit_reset_step", "run_study"):
        monkeypatch.setattr(admission.reset, name, forbidden)
    actual = admission.admit_trace(step, original, template)
    for key in ("S", "A", "b", "c", "vectors", "runtime", "reset_defect"):
        assert actual[key] == expected[key]
    assert all("kernel_evaluation_defect" not in row and "clipping_defect" not in row for row in actual["rows"])
    assert bool(any(actual["reset_defect"])) == defects
    pair = admission._pair(template, actual)
    assert pair["paired_source_difference"] == (F(0), F(0))
    assert pair["delta"]["xf"] == tuple(b-a for a, b in zip(template["vectors"]["xf"], actual["vectors"]["xf"], strict=True))


@pytest.mark.parametrize("mutation", (
    "status", "time", "order", "glyph", "missing_call", "phase", "pressure", "configuration",
    "neighbor", "entry_capture", "single_target", "capacity", "edge", "integrator", "dt", "entry_state",
    "pre_generation_jump", "post_reset_jump", "missing_refresh", "generation_capture",
    "generation_time", "glyph_time", "integration_time", "boundary_list_order"))
def test_trace_domain_and_causal_order_mutations_rejected(mutation):
    step, original, template, _ = _case()
    trace = step["native_trace"]
    calls = [row for row in trace["boundaries"] if row["boundary"] == "apply_glyph"]
    integration = trace["boundaries"][-1]
    if mutation == "status":
        trace["status"] = "failed"
    elif mutation == "time":
        step["endpoint"]["state"]["time"] = 3.0
    elif mutation == "order":
        calls[0]["ordinal"] = 99
    elif mutation == "glyph":
        calls[0]["glyph"] = "IL"
    elif mutation == "missing_call":
        trace["boundaries"].remove(calls[0])
    elif mutation == "phase":
        calls[0]["after"]["state"]["phase"][0] = 0.5
    elif mutation == "pressure":
        integration["after"]["state"]["pressure"][0] += 0.25
    elif mutation == "configuration":
        calls[0]["before"]["graph_attributes"]["GLYPH_FACTORS"]["EN_mix"] = 0.5
    elif mutation == "neighbor":
        calls[0]["before"]["ordered_neighbors"][0][1].clear()
    elif mutation == "entry_capture":
        trace["captures"]["integrator_entry"]["available"] = False
    elif mutation == "single_target":
        calls[0]["after"]["state"]["epi"][1] += 0.25
    elif mutation == "capacity":
        calls[0]["after"]["state"]["capacity"][0] *= 2
    elif mutation == "edge":
        calls[0]["after"]["state"]["edges"][0][2]["weight"] = 2.0
    elif mutation == "integrator":
        integration["integrator_type"] = "custom"
    elif mutation == "dt":
        integration["effective_arguments"]["dt"] = 0.5
    elif mutation == "entry_state":
        integration["before"]["state"]["epi"][0] += 0.25
    elif mutation == "pre_generation_jump":
        step["before"]["state"]["epi"][0] += 0.25
    elif mutation == "post_reset_jump":
        calls[-1]["after"]["state"]["epi"][-1] += 0.25
    elif mutation == "missing_refresh":
        trace["boundaries"] = [row for row in trace["boundaries"] if row["boundary"] != "_refresh_delta_nfr"]
    elif mutation == "generation_capture":
        trace["captures"]["pressure_generation"]["available"] = False
    elif mutation == "generation_time":
        trace["boundaries"][0]["after"]["state"]["time"] = 2.0
    elif mutation == "glyph_time":
        calls[0]["after"]["state"]["time"] = 2.0
    elif mutation == "integration_time":
        integration["after"]["state"]["time"] = 2.25
    elif mutation == "boundary_list_order":
        trace["boundaries"][0], trace["boundaries"][1] = trace["boundaries"][1], trace["boundaries"][0]
    with pytest.raises((ValueError, RuntimeError)):
        admission.admit_trace(step, original, template)


def test_cross_experiment_source_equality_is_separate_from_paired_cancellation():
    _, _, template, expected = _case()
    expected = {**expected, "b": (F(1), F(0))}
    admission._common(template, expected, source_required=False)
    with pytest.raises(ValueError, match="source"):
        admission._pair(template, expected)


@pytest.mark.parametrize("key", ("S", "A", "c"))
def test_declared_common_maps_are_not_assumed(key):
    _, _, template, expected = _case()
    altered = deepcopy(expected)
    altered[key] = ((F(99),),) if key in ("S", "A") else (F(99),)
    with pytest.raises(ValueError, match=key):
        admission._common(template, altered, source_required=False)


def _file(path, claim):
    manifest = CoreExperimentManifest(
        claim_id=claim, git_sha="a"*40, source_dirty=False, versions={"python": "test"},
        graph_construction="synthetic", capacity_specification="unit", solver="saved arithmetic",
        timestep=0.25, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("reception",), telemetry=("fixture",), controls=("fixture",), artifacts=(str(path),))
    path.write_text(json.dumps({"manifest": manifest.to_dict()}))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authentication_precedes_arithmetic_admission(tmp_path, monkeypatch):
    old, new = tmp_path/"old.json", tmp_path/"new.json"
    a = _file(old, "O3.b-retained-EN-AL-reset-accounting")
    b = _file(new, "O3.a-localized-regional-restoration")
    seen = []
    monkeypatch.setattr(admission, "admit_reports", lambda *args: seen.append(args) or {"fixture": True})
    value = admission.load_comparable_witnesses(old, new, expected_reset_sha256=a, expected_restoration_sha256=b)
    assert len(seen) == 1 and value["historical_inputs"]["restoration"]["sha256"] == b
    old.write_text(old.read_text()+" ")
    with pytest.raises(ValueError, match="digest"):
        admission.load_comparable_witnesses(old, new, expected_reset_sha256=a, expected_restoration_sha256=b)
    assert len(seen) == 1


def test_wrong_historical_claim_refused(tmp_path, monkeypatch):
    old, new = tmp_path/"old.json", tmp_path/"new.json"
    a, b = _file(old, "wrong-claim"), _file(new, "O3.a-localized-regional-restoration")
    monkeypatch.setattr(admission, "admit_reports", lambda *args: pytest.fail("unadmitted claim"))
    with pytest.raises(ValueError, match="claim"):
        admission.load_comparable_witnesses(old, new, expected_reset_sha256=a, expected_restoration_sha256=b)


def test_input_mutation_during_read_only_admission_is_detected(tmp_path, monkeypatch):
    old, new = tmp_path/"old.json", tmp_path/"new.json"
    a = _file(old, "O3.b-retained-EN-AL-reset-accounting")
    b = _file(new, "O3.a-localized-regional-restoration")

    def changed(*args):
        new.write_text(new.read_text()+" ")
        return {}

    monkeypatch.setattr(admission, "admit_reports", changed)
    with pytest.raises(RuntimeError, match="changed"):
        admission.load_comparable_witnesses(old, new, expected_reset_sha256=a, expected_restoration_sha256=b)
