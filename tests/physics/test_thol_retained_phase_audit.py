"""Portable exact enclosure and detached retained-phase controls."""

import hashlib
import json
import math
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction

import networkx as nx
import pytest

from benchmarks import thol_retained_phase_audit as audit
from benchmarks.thol_pressure_feedback import _payload
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import CoreExperimentManifest


@pytest.fixture(scope="module")
def boundary():
    parents = tuple(range(8))
    pairs = tuple((i, f"child_{i}") for i in parents)
    nodes = parents + tuple(c for _, c in pairs)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((i, (i + 1) % 8) for i in parents)
    graph.add_edges_from(pairs)
    phases = tuple(
        ((i % 8) * math.pi / 4 + math.pi) % (2 * math.pi) - math.pi for i in range(16)
    )
    state = {
        "nodes": nodes,
        "edges": tuple((u, v, {"weight": 1.0}) for u, v in graph.edges),
        "epi": (0.25,) * 16,
        "capacity": (1.0,) * 16,
        "pressure": (0.1,) * 16,
        "phase": phases,
    }
    before = {
        "state": state,
        "ordered_neighbors": tuple((n, tuple(graph.neighbors(n))) for n in nodes),
    }
    gains = (0.05, 0.1)
    output = audit._evaluate(before, gains, dict(zip(nodes, nodes, strict=True)))
    after = {
        "state": {**state, "phase": output["phase"]},
        "graph_attributes": {
            key: ("binary64", value.hex())
            for key, value in zip(
                ("PHASE_K_GLOBAL", "PHASE_K_LOCAL"), gains, strict=True
            )
        },
    }
    return {
        "boundary": "coordinate_global_local_phase",
        "outcome": "completed",
        "before": before,
        "after": after,
    }, pairs


def test_exact_taylor_zero_parity_and_independent_high_precision_values():
    assert audit.trig_bounds(0) == (
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )
    with localcontext() as context:
        context.prec = 180
        for value in (Fraction(1), Fraction(3), Fraction(7)):
            bounds = audit.trig_bounds(value)
            opposite = audit.trig_bounds(-value)
            assert bounds[0] == opposite[0]
            assert bounds[1] == (-opposite[1][1], -opposite[1][0])
            x = Decimal(value.numerator) / Decimal(value.denominator)
            cosine = sum(
                (-1) ** k * x ** (2 * k) / Decimal(math.factorial(2 * k))
                for k in range(150)
            )
            sine = sum(
                (-1) ** k * x ** (2 * k + 1) / Decimal(math.factorial(2 * k + 1))
                for k in range(150)
            )
            for point, (low, high) in zip((cosine, sine), bounds, strict=True):
                assert (
                    Decimal(low.numerator) / Decimal(low.denominator)
                    < point
                    < Decimal(high.numerator) / Decimal(high.denominator)
                )


@pytest.mark.parametrize(
    "value", (True, "1", float("nan"), float("inf"), 9, complex(1, 0))
)
def test_enclosure_domain_fails_closed(value):
    with pytest.raises(ValueError):
        audit.trig_bounds(value)


def test_resultant_enclosure_squares_intervals_and_requires_finite_nonempty_input():
    result = audit.resultant_enclosure([0, 0])
    assert result["real"] == (1, 1) and result["imag"] == (0, 0)
    assert result["squared_norm"] == (1, 1) and result["nonzero_certified"]
    for values in ([], [0] * 65, iter([0])):
        with pytest.raises(ValueError):
            audit.resultant_enclosure(values)


def test_relative_pattern_separates_common_rotation_from_nonuniform_change():
    base = [0.0, 0.25, 0.5]
    common = audit.compare_phases(base, [0.5, 0.75, 1.0])
    assert common["anchor_common_shift"] == pytest.approx(0.5)
    assert (
        common["max_relative_residual"] < 1e-15
    )  # Numerical test, not a certified gauge tolerance.
    changed = audit.compare_phases(base, [0.5, 0.875, 1.0])
    assert changed["relative_pattern_residual"][1] == pytest.approx(0.125)
    assert changed["max_relative_residual"] > 0.1


def test_detached_audit_preserves_inputs_and_calls_no_native_step(
    boundary, monkeypatch
):
    from tnfr import dynamics

    def forbidden(*args, **kwargs):
        raise AssertionError("native trajectory forbidden")

    monkeypatch.setattr(dynamics, "step", forbidden)
    row, pairs = boundary
    saved = deepcopy(row)
    owners = (audit.coordination, audit.trig, audit.trig_cache)
    old_numpy = tuple(owner.np for owner in owners)
    result = audit.audit_boundary(row, pairs)
    assert row == saved and tuple(owner.np for owner in owners) == old_numpy
    assert (
        result["rotation"] == 1
        and result["live_graph_writes"] == result["native_steps"] == 0
    )
    assert result["transported_source_equal"] and result["numpy_replay_matches_archive"]
    assert result["base_numpy"]["phase"] == result["transported_numpy"]["phase"]
    moved, reordered = (
        result["transported_numpy"]["initial"],
        result["canonical_order_numpy"]["initial"],
    )
    assert moved["nodes"] != reordered["nodes"]
    assert dict(zip(moved["nodes"], moved["phase"], strict=True)) == dict(
        zip(reordered["nodes"], reordered["phase"], strict=True)
    )
    assert moved["ordered_neighbors"] == reordered["ordered_neighbors"]
    mapping = dict(result["node_mapping"])
    for name in ("transported_numpy", "canonical_order_numpy"):
        observed = result[name]
        for field in ("phi_s", "grad_phi", "curv_phi"):
            for node in row["before"]["state"]["nodes"]:
                assert (
                    observed["aligned_tetrad_fields"][field][node]
                    == observed["tetrad"]["fields"][field][mapping[node]]
                )
    assert (
        result["ideal_resultant"]["exact_zero"]
        and "Exact phases" in result["ideal_resultant"]["hypothesis"]
    )
    assert set(result["base_numpy"]["tetrad"]["fields"]) == {
        "phi_s",
        "grad_phi",
        "curv_phi",
        "xi_c",
    }


@pytest.mark.parametrize(
    "mutation", ("endpoint", "outcome", "lineage", "neighbors", "order")
)
def test_declared_boundary_and_transport_admission_rejects_malformed_inputs(
    boundary, mutation
):
    row, pairs = deepcopy(boundary)
    if mutation == "endpoint":
        row["after"]["state"]["phase"] = (0.0,) * 16
    elif mutation == "outcome":
        row["outcome"] = "raised"
    elif mutation == "lineage":
        pairs = pairs[:-1] + (pairs[0],)
    elif mutation == "neighbors":
        row["before"]["ordered_neighbors"] = tuple(
            (n, ()) for n, _ in row["before"]["ordered_neighbors"]
        )
    else:
        nodes = row["before"]["state"]["nodes"]
        with pytest.raises(ValueError):
            audit._detached_graph(
                row["before"],
                dict(zip(nodes, nodes, strict=True)),
                node_order=(nodes[0],) * 16,
            )
        return
    with pytest.raises(ValueError):
        audit.audit_boundary(row, pairs)


def _write_input(path, boundary):
    row, pairs = boundary
    manifest = CoreExperimentManifest(
        claim_id="O1.b-generated-native-runtime-response",
        git_sha="a" * 40,
        source_dirty=False,
        versions={"python": "test"},
        graph_construction="Explicit portable synthetic boundary fixture",
        capacity_specification="positive",
        solver="synthetic fixture",
        timestep=0.25,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(),
        telemetry=("fixture",),
        controls=("synthetic",),
        artifacts=(str(path),),
    )
    payload = {
        "manifest": manifest.to_dict(),
        "branches": [
            {
                "branch": name,
                "status": "executed",
                "native_trace": {"boundaries": [row]},
            }
            for name in ("control", "child_emission")
        ],
        "replayed_prior_reports": [{"lineage": {"parent_children": pairs}}],
    }
    raw = json.dumps(_payload(payload), allow_nan=False).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_portable_pinned_input_and_cli_manifest_no_archive_dependency(
    boundary, tmp_path, monkeypatch
):
    source, output = tmp_path / "source.json", tmp_path / "audit.json"
    digest = _write_input(source, boundary)
    with pytest.raises(ValueError, match="bytes differ"):
        audit.run_study(source, expected_sha256="0" * 64)
    before = source.read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--input",
            str(source),
            "--expected-sha256",
            digest,
            "--output",
            str(output),
        ],
    )
    audit.main()
    result = json.loads(output.read_bytes())
    CoreExperimentManifest(**result["manifest"]).validate_for_admission()
    assert result["manifest"]["timestep"] is None and result["manifest"]["seed"] is None
    assert result["audit"]["native_steps"] == 0
    monkeypatch.setattr(
        "sys.argv", ["audit", "--input", str(source), "--output", str(source)]
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        audit.main()
    assert source.read_bytes() == before


@pytest.mark.skipif(
    not audit.INPUT.exists(),
    reason="Optional historical integration; portable controls do not need ignored evidence",
)
def test_pinned_actual_boundary_conditioning_is_not_a_claim_of_physical_symmetry_breaking():
    result = audit.run_study()["audit"]
    assert result["represented_resultant"]["nonzero_certified"]
    assert result["numpy_replay_matches_archive"] and result["transported_source_equal"]
    assert result["canonical_order_vs_numpy"]["max_relative_residual"] > 0.3
    assert result["scalar_vs_numpy"]["max_relative_residual"] < 1e-14
    assert result["tetrad_scalar_equals_numpy"]["phi_s"]
    assert not result["tetrad_canonical_order_equals_numpy"]["grad_phi"]
    assert "no physical emergence" in result["scope"]
