"""Manufactured arithmetic/evidence controls; no retained target is loaded.

The synthetic saved cells below test serialized arithmetic and association.
They are not trajectories or independently admitted derivative certificates.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.mathematics import _validated_taylor as taylor
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.research import maintained_sha_direct_response as direct
from tnfr.research import maintained_sha_motion_source as admission
from tnfr.research import maintained_sha_preparation_response as owner
from tnfr.research import sine_preparation_bounds as arithmetic
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def forbid_retained_readers_and_evolution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructed controls cannot admit or evolve a retained target")

    monkeypatch.setattr(admission, "load_sha_successor", forbidden)
    monkeypatch.setattr(direct, "load_sha_retention", forbidden)
    monkeypatch.setattr(taylor, "picard_tube", forbidden)
    monkeypatch.setattr(taylor, "validated_box_taylor_step", forbidden)
    monkeypatch.setattr(taylor, "validated_taylor_step", forbidden)


def _path(*, scale=Q(1), child=Q(1)):
    graph = nx.Graph()
    for node, form, phase, capacity in zip(
        ("parent", "child", "rest", "remote"),
        (0, child, Q(1, 4), Q(-1, 2)),
        (0, 0, Q(1, 8), Q(-1, 4)),
        (2, 3, 2, 4),
        strict=True,
    ):
        graph.add_node(node, EPI=scale * form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.add_weighted_edges_from(
        (("parent", "child", 2), ("parent", "rest", 3), ("rest", "remote", 5))
    )
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return graph


def _source(graph):
    return owner._capture_cut_source(
        graph, parent="parent", child="child", reference_model=owner.MODEL
    )


def _nominal(*, connected_radius=Q(1), cut_radius=Q(1), gap=4):
    return owner.NominalPreparationBounds(I(gap), I(0), connected_radius, cut_radius)


def test_two_supports_keep_actual_degrees_and_weighted_sums():
    source = _source(_path())
    pi = pi_interval()
    connected = owner._coefficient_bounds(source, cut=False, pi=pi)
    cut = owner._coefficient_bounds(source, cut=True, pi=pi)
    assert connected.dynamic_indices == (0, 1, 2, 3)
    assert connected.degrees == (2, 1, 2, 1)
    assert connected.weighted_row_sums == (5, 2, 8, 5)
    assert cut.dynamic_indices == (0, 2, 3)
    assert cut.degrees == (1, 2, 1)
    assert cut.weighted_row_sums == (3, 8, 5)
    assert connected.a_numerator == cut.a_numerator == 8
    assert connected.b_numerator == cut.b_numerator == 40
    assert connected.A_upper >= Q(8) / pi.lo
    assert connected.B_upper >= Q(40) / pi.lo


def test_constructed_family_success_has_shared_centers_and_frozen_child_error():
    source = _source(_path(child=Q(16)))
    result = owner._bound_preparation_response(source, _nominal())
    assert result.status == "family_ordering_certified"
    assert tuple(hull.center for hull in result.hulls) == source.epi + source.phase
    assert result.family_members == 3**8
    assert result.connected_comparison.endpoints[0][0] == Q(1, 2**48)
    assert result.cut_comparison.endpoints[0][0] == Q(1, 2**53)
    assert result.child_form_error == Q(1, 2**48)
    assert result.child_phase_error == Q(1, 2**1074)
    U, V = result.cut_comparison.final
    without_child = V * V + source.reference_conductance * U * U
    assert result.cut_error_squared > without_child
    assert result.nominal_gap_gate.margin == Q(4) - result.pi.hi
    assert result.connected_radius_gate.margin == 1 - result.connected_error_squared
    assert result.cut_radius_gate.margin == 1 - result.cut_error_squared


def test_exactly_one_hull_per_coordinate_and_128_registered_comparison_calls(
    monkeypatch,
):
    hulls, steps = [], []
    original_hull = owner.adjacent_coordinate_hull
    original_step = arithmetic.comparison_flow_upper

    def hull(value):
        hulls.append(value)
        return original_hull(value)

    def step(matrix, initial, duration):
        steps.append((matrix, initial, duration))
        return original_step(matrix, initial, duration)

    monkeypatch.setattr(owner, "adjacent_coordinate_hull", hull)
    monkeypatch.setattr(arithmetic, "comparison_flow_upper", step)
    result = owner._bound_preparation_response(_source(_path()), _nominal())
    assert len(hulls) == 8 and len(steps) == 128
    assert all(row[2] == Q(1, 128) for row in steps)
    assert len(result.connected_comparison.endpoints) == 65
    assert len(result.cut_comparison.endpoints) == 65
    assert steps[64][1] == result.cut_comparison.endpoints[0]


@pytest.mark.parametrize("which", ("connected", "cut", "gap"))
def test_insufficient_sufficient_gate_is_a_normal_result_not_a_counterexample(which):
    source = _source(_path())
    nominal = _nominal()
    if which == "connected":
        nominal = replace(nominal, connected_radius_squared=Q(1, 2**500))
    elif which == "cut":
        nominal = replace(nominal, cut_radius_squared=Q(1, 2**500))
    else:
        nominal = replace(nominal, connected_clockwise=I(3))
    result = owner._bound_preparation_response(source, nominal)
    gate = getattr(
        result, "nominal_gap_gate" if which == "gap" else which + "_radius_gate"
    )
    assert gate.status == "insufficient_bound" and gate.margin <= 0
    assert result.status == "insufficient_bound"
    assert len(result.connected_comparison.endpoints) == 65
    assert len(result.cut_comparison.endpoints) == 65


@pytest.mark.parametrize("margin", (Q(0), Q(-1), Q(1)))
def test_only_strict_positive_exact_gate_passes(margin):
    assert (owner._gate(margin).status == "pass") is (margin > 0)
    with pytest.raises(TypeError):
        owner._gate(True)


@pytest.mark.parametrize("center", (Q(1, 3), True, 1.0))
def test_every_center_roundtrips_before_any_neighbor_is_constructed(
    monkeypatch, center
):
    source = _source(_path())
    source = replace(source, phase=source.phase[:-1] + (center,))

    def forbidden(*args):
        pytest.fail("all center admissions must precede the first neighbor")

    monkeypatch.setattr(owner, "adjacent_coordinate_hull", forbidden)
    with pytest.raises((ValueError, TypeError)):
        owner._bound_preparation_response(source, _nominal())


def test_fixed_work_policy_rejects_before_hulls_or_adaptive_subdivision(monkeypatch):
    source = _source(_path())
    source = replace(source, capacity=tuple(10000 * value for value in source.capacity))

    def forbidden(*args):
        pytest.fail("unsupported comparison work must not construct a family")

    monkeypatch.setattr(owner, "adjacent_coordinate_hull", forbidden)
    with pytest.raises(ValueError, match="64-step"):
        owner._bound_preparation_response(source, _nominal())


@pytest.mark.parametrize(
    "field,value",
    (
        ("connected_radius_squared", True),
        ("cut_radius_squared", Q(0)),
        ("connected_cells_checked", True),
        ("cut_cells_checked", 255),
    ),
)
def test_missing_or_unadmitted_nominal_prerequisites_reject(field, value):
    with pytest.raises((ValueError, TypeError)):
        owner._bound_preparation_response(
            _source(_path()), replace(_nominal(), **{field: value})
        )


def _fixed_graph():
    graph = nx.Graph()
    graph.add_nodes_from(tuple(range(20)) + ("2_sub_0",))
    pairs = tuple((i, j) for i in range(20) for j in range(i + 1, 20))
    graph.add_edges_from(pairs[:52], weight=1)
    graph.add_edge(2, "2_sub_0", weight=1)
    for node in graph:
        graph.nodes[node].update(
            EPI=int(node == "2_sub_0"), theta=0, nu_f=1, delta_nfr=99
        )
    graph.graph.update(_t=owner.TIMES[-1], GAMMA={"type": "none"})
    return graph


def _synthetic_cells(observation, phase_rate):
    # Algebraically valid observation records only: no claim these constant
    # boxes follow a complete nonlinear trajectory. Producer premises are
    # deliberately external to the saved-arithmetic verifier under test.
    cells = []
    for index in range(256):
        r, delta, r_dot, delta_dot = I(1), I(0), I(0), I(phase_rate)
        radius = delta**2 + observation.reference_conductance * r**2
        numerator = sqrt(I(observation.reference_conductance)) * (
            delta * r_dot - r * delta_dot
        )
        rate = numerator / radius
        cells.append(
            direct.DirectResponseCell(
                index,
                Q(index, 512),
                Q(1, 512),
                r,
                delta,
                r_dot,
                delta_dot,
                radius,
                numerator,
                rate,
                Q(1, 512) * rate,
                "clockwise" if phase_rate > 0 else "stationary",
                I(observation.form_mean),
                I(observation.phase_lift_mean),
                (True, True),
                (),
            )
        )
    return tuple(cells), direct._aggregate(cells, full_window=True)


def _synthetic_cut_observation(connected, source):
    beta = tuple(
        source.capacity[i] / degree if degree else Q(0)
        for i, degree in enumerate(source.cut_degrees)
    )
    rho = tuple(
        Q(degree) / source.capacity[i] for i, degree in enumerate(source.cut_degrees)
    )
    signed, linear = [], [Q(0)] * len(source.nodes)
    for (i, j), weight in zip(source.cut_edges, source.cut_conductances, strict=True):
        coefficient = -beta[source.parent_index] * (
            (i == source.parent_index) - (j == source.parent_index)
        )
        signed.append(coefficient)
        linear[i] += coefficient * weight
        linear[j] -= coefficient * weight
    return replace(
        connected,
        edges=source.cut_edges,
        conductances=source.cut_conductances,
        degrees=source.cut_degrees,
        beta=beta,
        mean_weights=rho,
        form_mean=source.form_mean,
        phase_lift_mean=source.phase_lift_mean,
        form_rate_edge_coefficients=tuple(signed),
        phase_rate_coefficients=tuple(linear),
    )


@pytest.fixture(scope="module")
def saved_constructed_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("constructed-preparation-evidence")
    graph = _fixed_graph()
    source = owner._capture_cut_source(
        graph, parent=2, child="2_sub_0", reference_model=owner.MODEL
    )
    connected = direct._capture_direct_source(
        graph, parent=2, child="2_sub_0", reference_model=owner.MODEL
    )
    cut = _synthetic_cut_observation(connected, source)
    connected_cells, connected_aggregate = _synthetic_cells(connected, 8)
    cut_cells, cut_aggregate = _synthetic_cells(cut, 0)
    receipts = ({"path": "constructed-only", "bytes": 1, "sha256": "0" * 64},)
    records = {
        "connected_result": {
            "report": {
                "source": asdict(connected),
                "source_receipts": receipts,
                "cells": tuple(map(asdict, connected_cells)),
                "aggregate": asdict(connected_aggregate),
            }
        },
        "cut_result": {
            "report": {
                "preparation": {
                    "source": asdict(source),
                    "observation_source": asdict(cut),
                    "source_receipts": receipts,
                    "initial": tuple(
                        asdict(I(values[i]))
                        for values in (source.epi, source.phase)
                        for i in source.host_indices
                    ),
                },
                "status": "completed",
                "accepted_steps": 256,
                "certified_time": Q(1, 2),
                "cells": tuple(map(asdict, cut_cells)),
                "aggregate": asdict(cut_aggregate),
            }
        },
    }
    byte_receipts = {}
    for key, record in records.items():
        path = root / owner._PATHS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_once(path, record)
        byte_receipts[key] = file_receipt(path, label=owner._PATHS[key])
    verifier = root / owner._PATHS["observation_verifier"]
    verifier.write_bytes((ROOT / owner._PATHS["observation_verifier"]).read_bytes())
    byte_receipts["observation_verifier"] = file_receipt(
        verifier, label=owner._PATHS["observation_verifier"]
    )
    for branch in ("connected", "cut"):
        keys = [branch + "_result"] + (
            ["observation_verifier"] if branch == "cut" else []
        )
        record = {
            "evidence": tuple(
                dict(byte_receipts[key], path=Path(byte_receipts[key]["path"]).name)
                for key in keys
            )
        }
        if branch == "cut":
            record["admitted_complete_cut_certificate"] = True
        key = branch + "_final_state"
        path = root / owner._PATHS[key]
        write_json_once(path, record)
        byte_receipts[key] = file_receipt(path, label=owner._PATHS[key])
    evidence = owner.FrozenNominalEvidence(root, **byte_receipts)
    return graph, source, receipts, evidence


def test_real_immutable_verifier_rebuilds_both_constructed_cell_windows(
    saved_constructed_case,
):
    _, source, receipts, evidence = saved_constructed_case
    nominal = owner._verify_nominal_evidence(source, receipts, evidence)
    assert nominal.connected_clockwise == I(4)
    assert nominal.cut_clockwise == I(0)
    assert nominal.connected_radius_squared == nominal.cut_radius_squared == 1
    assert nominal.connected_cells_checked == nominal.cut_cells_checked == 256
    assert len(nominal.evidence_receipts) == 5


@pytest.mark.parametrize(
    "field", ("epi", "cut_degrees", "reference_conductance", "structural_time")
)
def test_fresh_complete_source_must_match_before_saved_verifier_or_hulls(
    monkeypatch, saved_constructed_case, field
):
    _, source, receipts, evidence = saved_constructed_case
    value = getattr(source, field)
    changed = value[:-1] + (value[-1] + 1,) if type(value) is tuple else value + 1
    source = replace(source, **{field: changed})

    def forbidden(*args):
        pytest.fail("source association must precede verification and hulls")

    monkeypatch.setattr(owner, "_load_observation_verifier", forbidden)
    monkeypatch.setattr(owner, "adjacent_coordinate_hull", forbidden)
    with pytest.raises(ValueError, match="fresh complete primitives"):
        owner._verify_nominal_evidence(source, receipts, evidence)


def test_same_numeric_value_with_invalid_physical_type_cannot_match(
    saved_constructed_case,
):
    _, source, receipts, evidence = saved_constructed_case
    source = replace(source, epi=(False,) + source.epi[1:])
    with pytest.raises(ValueError, match="fresh complete primitives"):
        owner._verify_nominal_evidence(source, receipts, evidence)


def test_fresh_source_receipts_cannot_be_replaced_by_cached_receipts(
    saved_constructed_case,
):
    _, source, _, evidence = saved_constructed_case
    with pytest.raises(ValueError, match="fresh source evidence"):
        owner._verify_nominal_evidence(source, ({"cached": True},), evidence)


@pytest.mark.parametrize("bad", (True, 0, -1, 1.0))
def test_receipt_byte_count_is_a_strict_positive_integer(bad):
    receipt = {"path": "a.json", "bytes": bad, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="typed evidence receipt"):
        owner._receipt(receipt, "a.json")


def test_modified_record_bytes_are_rejected_before_json_or_physics(tmp_path):
    path = tmp_path / owner._PATHS["connected_result"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}")
    receipt = file_receipt(path, label=owner._PATHS["connected_result"])
    evidence = owner.FrozenNominalEvidence(tmp_path, receipt, {}, {}, {}, {})
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="evidence bytes differ"):
        owner._evidence_record(evidence, "connected_result")


def test_a_rehashed_alternate_verifier_cannot_replace_the_immutable_owner(tmp_path):
    path = tmp_path / owner._PATHS["observation_verifier"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"raise AssertionError('must never execute')\n")
    receipt = file_receipt(path, label=owner._PATHS["observation_verifier"])
    evidence = owner.FrozenNominalEvidence(tmp_path, {}, {}, {}, {}, receipt)
    with pytest.raises(ValueError, match="immutable nominal"):
        owner._load_observation_verifier(evidence)


def test_final_state_must_bind_exactly_one_corresponding_result():
    receipt = {"path": "folder/result.json", "bytes": 1, "sha256": "0" * 64}
    row = dict(receipt, path="result.json")
    owner._final_state_link({"evidence": (row,)}, receipt)
    for rows in ((), (row, row), (dict(row, bytes=True),)):
        with pytest.raises(ValueError, match="bind"):
            owner._final_state_link({"evidence": rows}, receipt)


def test_public_wrapper_freshly_admits_then_binds_then_builds_all42_hulls(
    monkeypatch, saved_constructed_case
):
    graph, _, receipts, evidence = saved_constructed_case
    before = deepcopy(
        (
            dict(graph.graph),
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
        )
    )
    stale = SimpleNamespace(graph=None, receipts=({"cached": True},))
    fresh = SimpleNamespace(
        graph=graph,
        receipts=deepcopy(receipts),
        repository_root=evidence.repository_root,
    )
    calls = []
    original_capture = owner._capture_cut_source
    original_verify = owner._verify_nominal_evidence

    def admit(value):
        assert value is stale
        calls.append("fresh")
        return fresh

    def capture(value, **kwargs):
        assert value is fresh.graph
        calls.append("capture")
        return original_capture(value, **kwargs)

    def verify(source, source_receipts, supplied_evidence):
        calls.append("nominal")
        return original_verify(source, source_receipts, supplied_evidence)

    monkeypatch.setattr(owner, "validate_live_sha_successor", admit)
    monkeypatch.setattr(owner, "_capture_cut_source", capture)
    monkeypatch.setattr(owner, "_verify_nominal_evidence", verify)
    result = owner.audit_sha_preparation_response(stale, evidence=evidence)
    assert calls == ["fresh", "capture", "nominal"]
    assert len(result.hulls) == 42 and result.family_members == 3**42
    assert result.status == "family_ordering_certified"
    assert (
        result.source_receipts == receipts
        and result.source_receipts is not fresh.receipts
    )
    assert (
        dict(graph.graph),
        tuple(graph.nodes(data=True)),
        tuple(graph.edges(data=True)),
    ) == before


@pytest.mark.parametrize("invalid", ("shape", "clock", "repository"))
def test_public_bad_fresh_association_rejects_before_capture(
    monkeypatch, saved_constructed_case, invalid
):
    original, _, receipts, evidence = saved_constructed_case
    graph = deepcopy(original)
    if invalid == "shape":
        graph.remove_edge(0, 1)
    elif invalid == "clock":
        graph.graph["_t"] = 0
    root = (
        evidence.repository_root / "different"
        if invalid == "repository"
        else evidence.repository_root
    )
    fresh = SimpleNamespace(graph=graph, receipts=receipts, repository_root=root)
    monkeypatch.setattr(owner, "validate_live_sha_successor", lambda value: fresh)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid public source must not reach capture")

    monkeypatch.setattr(owner, "_capture_cut_source", forbidden)
    with pytest.raises(ValueError):
        owner.audit_sha_preparation_response(object(), evidence=evidence)


def test_unadmitted_cached_object_cannot_trigger_source_read(saved_constructed_case):
    evidence = saved_constructed_case[-1]
    with pytest.raises(ValueError, match="file-backed committed SHA admission"):
        owner.audit_sha_preparation_response(
            SimpleNamespace(graph=None), evidence=evidence
        )
