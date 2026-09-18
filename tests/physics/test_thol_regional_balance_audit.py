"""Portable detached regional receipts; no live graph or trajectory is used."""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
import hashlib
import json
import sys

import pytest

from benchmarks import thol_regional_balance_audit as audit
from benchmarks.thol_distributed_target import _digest
from benchmarks.thol_pressure_feedback import _payload
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import NonEpiForcingObservation, decompose_non_epi_forcing
from tnfr.physics.support_transport import _from_data
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import CoreExperimentManifest


@pytest.fixture
def prior():
    """Synthetic declared receipt, not evidence of executed THOL or Coupling."""
    parents = tuple(f"p{i}" for i in range(8))
    children = tuple(f"c{i}" for i in range(8))
    nodes = parents+children
    pairs = tuple(zip(parents, children))
    undirected = [(i, (i+1) % 8, F(1)) for i in range(8)]
    undirected += [(i, i+8, F(1, 2)) for i in range(8)]
    directed, support = [], [set() for _ in nodes]
    for i, j, w in undirected:
        directed.extend(((i, j, w), (j, i, w)))
        support[i].add(j)
        support[j].add(i)
    support = tuple(tuple(sorted(row)) for row in support)
    x = tuple(F(1)+F(i % 2) for i in range(8))+tuple(F(i+1, 32) for i in range(8))
    nu = (F(1),)*8+(F(1, 2),)*8
    snap = _from_data(nodes, directed, support, x, nu, (0,)*16)
    phase = tuple(F(1 if i % 2 else -1, 8) for i in range(16))
    weights = (("phase", F(1, 8)), ("epi", F(1, 2)), ("vf", F(1, 4)), ("topo", F(1, 8)))
    forcing = tuple(phase[i]/8+snap.capacity_gradient[i]/4+snap.topology_gradient[i]/8 for i in range(16))
    model = tuple(g/2+f for g, f in zip(snap.epi_gradient, forcing, strict=True))
    fresh = tuple(F(float(p))+F(1, 32) for p in model)
    stored = tuple(F(float(p+F(1, 16))) for p in fresh)
    snap = _from_data(nodes, directed, support, x, nu, stored)
    observation = NonEpiForcingObservation(
        snap, (F(0),)*16, F(1, 2), forcing, phase, weights, fresh,
        tuple(p-q for p, q in zip(fresh, model)), tuple(p-q for p, q in zip(stored, fresh)))
    reference = derive_forced_support_balance(snap, epi_weight=F(1, 2), forcing=forcing)
    hierarchy = {p: [c] for p, c in pairs}
    child_map = {**hierarchy, **{c: [] for c in children}}
    state = {"time": .5, "nodes": nodes, "epi": tuple(map(float, x)), "capacity": tuple(map(float, nu)),
             "pressure": tuple(map(float, stored)), "phase": (0.0,)*16,
             "edges": tuple((nodes[i], nodes[j], {"weight": float(w)}) for i, j, w in undirected),
             "hierarchy": hierarchy, "children": child_map}
    attributes = [(p, {"sub_nodes": [c]}) for p, c in pairs]+[(c, {"parent_node": p}) for p, c in pairs]
    return _payload({
        "branch": "control", "status": "executed",
        "prefix": {"coupling": {"after_refreshed": state,
                                "refreshed_forcing": {"observation": asdict(observation),
                                                      "components": decompose_non_epi_forcing(observation)}},
                   "birth": {"parent_children": pairs, "before": {"nodes": parents}, "after": state,
                             "children": [{"parent": p, "child": c, "node_data": {"parent_node": p}} for p, c in pairs]}},
        "original_reference": asdict(reference), "original_reference_sha256": _digest(reference),
        "lineage": {"parent_children": pairs, "parents": parents, "children": children, "nodes": nodes},
        "common_source": {"state": state, "node_attributes": attributes},
    })


def test_nine_observations_use_one_full_snapshot_and_independent_budgets(prior, monkeypatch):
    original = audit.observe_regional_support_balance
    seen = []

    def record(snapshot, region, **kwargs):
        seen.append((snapshot, tuple(region)))
        return original(snapshot, region, **kwargs)

    monkeypatch.setattr(audit, "observe_regional_support_balance", record)
    result = audit.audit_prior(prior)
    assert len(seen) == 9 and all(snap is seen[0][0] for snap, _ in seen)
    assert result["snapshot_count"] == 1 and result["native_calls"] == result["new_trajectories"] == 0
    assert [region for _, region in seen] == list(map(tuple, prior["lineage"]["parent_children"]))+[tuple(prior["lineage"]["children"])]
    snap = seen[0][0]
    weights = {(i, j): w for i, j, w in snap.conductance}
    d = tuple(sum((w for (source, _), w in weights.items() if source == i), F(0)) for i in range(16))
    h = tuple(di/nu for di, nu in zip(d, snap.capacity))
    for item in result["regions"]:
        b = item["balance"]
        ids = set(b["region_indices"])
        mass = sum((h[i]*snap.epi[i] for i in ids), F(0))
        mean = mass/sum(h[i] for i in ids)
        cut_mass = -F(1, 2)*sum((w*(snap.epi[i]-snap.epi[j]) for (i, j), w in weights.items()
                                if i in ids and j not in ids), F(0))
        direct_rate = sum((h[i]*snap.rate[i] for i in ids), F(0))
        direct_variance_rate = sum((h[i]*(snap.epi[i]-mean)*snap.rate[i] for i in ids), F(0))
        assert b["strengths"] == d and b["metric_weights"] == h
        assert b["weighted_total"] == mass and b["mass_boundary_rate"] == cut_mass
        assert b["stored_mass_rate"] == direct_rate and b["stored_variance_rate"] == direct_variance_rate
        residuals = ("model_mass_identity_residual", "mass_identity_residual",
                     "model_variance_identity_residual", "variance_identity_residual")
        assert all(b[key] == 0 for key in residuals)
        assert tuple(item["forcing_channels"]) == ("phase", "vf", "topo")
        for field, total in (("weighted_total_rate", b["mass_defect_rate"]), ("variance_rate", b["variance_defect_rate"])):
            assert item["fresh_kernel_defect"][field]+item["stored_minus_fresh_residual"][field] == total
    # The parent retains degree 2.5, rather than the induced pair degree .5.
    assert result["regions"][0]["balance"]["strengths"][0] == F(5, 2)
    assert result["regions"][-1]["balance"]["internal_dissipation"] == 0


def test_child_response_is_conditional_on_parents_and_current_forcing(prior):
    result = audit.audit_prior(prior)
    observation = prior["prefix"]["coupling"]["refreshed_forcing"]["observation"]
    for i, row in enumerate(result["child_cohort_response"]["rows"]):
        assert row["parent_weights"] == ((f"p{i}", F(1)),)
        expected = F(1+i % 2)+2*F(observation["forcing"][i+8])
        assert row["conditional_equilibrium_epi"] == expected
        assert row["relaxation_rate"] == F(1, 4)
        assert row["model_nodal_rate"] == -F(1, 4)*(row["current_epi"]-expected)
        assert row["equilibrium_pressure_residual"] == row["affine_rate_identity_residual"] == 0
    assert "Parents actually evolve" in result["child_cohort_response"]["scope"]
    assert result["regional_formation_or_persistence_certified"] is False


@pytest.mark.parametrize("change", ("time", "epi", "capacity", "phase", "pressure", "edge_weight", "snapshot_cache",
                                    "reference_cache", "reference_digest", "lineage", "hierarchy", "parent_pointer", "birth_pointer"))
def test_modified_source_reference_or_actual_ancestry_rejected(prior, change):
    p = deepcopy(prior)
    state = p["prefix"]["coupling"]["after_refreshed"]
    if change == "time":
        state["time"] = .75
    elif change in ("epi", "capacity", "phase", "pressure"):
        state[change][0] += .125
    elif change == "edge_weight":
        state["edges"][0][2]["weight"] *= 2
    elif change == "snapshot_cache":
        p["prefix"]["coupling"]["refreshed_forcing"]["observation"]["snapshot"]["epi_gradient"][0] = "99"
    elif change == "reference_cache":
        p["original_reference"]["metric_weights"][0] = "99"
    elif change == "reference_digest":
        p["original_reference_sha256"] = "0"*64
    elif change == "lineage":
        p["lineage"]["parent_children"][0][1] = "c1"
    elif change == "hierarchy":
        state["hierarchy"]["p0"] = ["c1"]
    elif change == "parent_pointer":
        p["common_source"]["node_attributes"][8][1]["parent_node"] = "p1"
    else:
        p["prefix"]["birth"]["children"][0]["node_data"]["parent_node"] = "p1"
    with pytest.raises((ValueError, TypeError)):
        audit.audit_prior(p)


def test_child_child_support_refuses_independent_child_equilibria(prior):
    raw = prior["prefix"]["coupling"]["refreshed_forcing"]
    snap, obs, _ = audit._capture(raw)
    support = list(snap.support_neighbors)
    support[8] += (9,)
    with pytest.raises(ValueError, match="child-child"):
        audit._held_parent_response(replace(snap, support_neighbors=tuple(support)), obs,
                                    tuple(prior["lineage"]["parents"]), tuple(prior["lineage"]["children"]))


def _write_fixture(path, prior):
    manifest = CoreExperimentManifest(
        claim_id=audit.INPUT_CLAIM, git_sha="a"*40, source_dirty=False, versions={"python": "synthetic-test"},
        graph_construction="Synthetic detached observer test; no executed birth evidence",
        capacity_specification="Supplied positive test capacities", solver="No trajectory", timestep=None, seed=None,
        result_status=ClaimStatus.DERIVED, operator_sequence=(), telemetry=("fixture",), controls=("no runtime",), artifacts=(str(path),))
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "branches": [{"branch": "control"}, {"branch": "child_emission"}],
              "replayed_prior_reports": [prior, {"branch": "child_emission"}]}
    path.write_text(json.dumps(report), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_input_digest_checked_before_scientific_fields(prior, tmp_path):
    path = tmp_path/"fixture.json"
    digest = _write_fixture(path, prior)
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="digest"):
        audit.run_study(path, expected_sha256=digest)


def test_cli_preserves_original_bytes_and_has_no_solver_calls(prior, tmp_path, monkeypatch):
    path, output = tmp_path/"fixture.json", tmp_path/"audit.json"
    digest = _write_fixture(path, prior)
    monkeypatch.setattr(audit, "current_git_source_provenance", lambda *args: ("a"*40, False, None))
    monkeypatch.setattr(sys, "argv", ["audit", "--input", str(path), "--expected-sha256", digest, "--output", str(output)])
    audit.main()
    result = json.loads(output.read_text())
    assert result["manifest"]["result_status"] == "derived"
    assert result["historical_input"]["sha256"] == digest
    assert result["regional_observation_count"] == 9 and result["native_calls"] == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    monkeypatch.setattr(sys, "argv", ["audit", "--input", str(path), "--output", str(path)])
    with pytest.raises(ValueError, match="overwrite"):
        audit.main()


def test_pinned_source_snapshot_if_available(monkeypatch):
    if not audit.INPUT_PATH.exists():
        pytest.skip("ignored retained source is not available")

    def forbidden(*args, **kwargs):
        raise AssertionError("offline regional audit must not execute a graph")

    from tnfr.dynamics import runtime
    from benchmarks import thol_full_state_response
    monkeypatch.setattr(runtime, "step", forbidden)
    monkeypatch.setattr(thol_full_state_response, "replay_response_branch", forbidden)
    result = audit.run_study()
    assert len(result["regions"]) == 9 and result["child_cohort_response"]["no_child_child_support_edges"]
    assert all(row["equilibrium_pressure_residual"] == 0 for row in result["child_cohort_response"]["rows"])
