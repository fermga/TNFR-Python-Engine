"""Prospective zero-form control and explicit competing-outcome reporting."""

import hashlib
import json
import math
import zipfile
from copy import deepcopy
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from benchmarks import relational_transit_proof as campaign
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.relational_transit import certify_relational_transit_capture


def normalized(record):
    return json.loads(
        json.dumps(record, default=campaign._project), object_hook=campaign._exact
    )


@pytest.fixture(scope="module")
def prospective():
    return normalized(campaign.prepare_protocol(zero_form=True))


def test_only_initial_form_is_ablated_and_no_limit_is_selected(prospective):
    original, _ = campaign._retained()
    reference = campaign._reference_proof()
    for key in ("nodes", "edges", "cycles", "initial_phase", "capacity", "model"):
        assert prospective[key] == original[key] == reference["protocol"][key]
    assert prospective["initial_form"] == [0.0] * 10
    assert prospective["numerical_policy"] == reference["protocol"]["numerical_policy"]
    assert prospective["requested_sector"] is None
    assert prospective["reference_target_sector"] == 1
    assert set(prospective["competing_outcomes"]) == {"-1", "0", "1", "unavailable"}
    assert prospective["terminal_prediction"].startswith("unselected")


def test_independent_initial_response_bounds(prospective):
    report = prospective["initial_derivation"]
    q, r, a, b = report["initial_rate_bounds"]
    assert Q(10885, 100000) < q[0] <= q[1] < Q(10886, 100000)
    assert Q(-24255, 100000) < r[0] <= r[1] < Q(-24254, 100000)
    assert a == b == [Q(0), Q(0)]
    a2, b2 = report["initial_acceleration_bounds"][2:]
    assert Q(1730, 100000) < a2[0] <= a2[1] < Q(1732, 100000)
    assert Q(-3003, 100000) < b2[0] <= b2[1] < Q(-3001, 100000)
    assert Q(79365, 10000) < report["initial_storage_bounds"][0]
    assert report["initial_storage_bounds"][1] < Q(79366, 10000)
    assert report["storage_third_derivative_bounds"][1] < 0
    assert min(report["resultant_real_lower_bounds"]) > 0


@pytest.mark.parametrize("sector", (1, 0, -1, None))
def test_outcomes_distinguish_consensus_negative_and_unavailable(
    prospective, monkeypatch, sector
):
    calls = []
    monkeypatch.setattr(
        campaign, "prepare_protocol", lambda **kwargs: deepcopy(prospective)
    )

    def stub(graph, **kwargs):
        calls.append(kwargs)
        assert all(data["EPI"] == 0 for _, data in graph.nodes(data=True))
        return SimpleNamespace(
            admitted=sector is not None, initial_winding_zero=True, target_sector=sector
        )

    monkeypatch.setattr(campaign.owner, "certify_relational_transit_capture", stub)
    monkeypatch.setattr(
        campaign, "relational_report_to_dict", lambda report: {"stub": True}
    )
    report = campaign.evaluate_protocol(prospective)
    assert calls[0]["requested_sector"] is None
    assert report["terminal_basin_admitted"] == (sector is not None)
    assert report["continuous_capture_admitted"] == (sector == 1)
    assert report["outcome"] == (str(sector) if sector is not None else "unavailable")
    assert report["different_limit_certified"] == (
        sector != 1 if sector is not None else None
    )


@pytest.mark.parametrize(
    "changed_key", ("initial_form", "initial_phase", "requested_sector")
)
def test_changed_control_rejects_before_execution(
    prospective, monkeypatch, changed_key
):
    monkeypatch.setattr(
        campaign, "prepare_protocol", lambda **kwargs: deepcopy(prospective)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("changed control executed")

    monkeypatch.setattr(campaign.owner, "certify_relational_transit_capture", forbidden)
    changed = deepcopy(prospective)
    changed[changed_key] = "not the frozen control"
    with pytest.raises(ValueError, match="frozen proof protocol"):
        campaign.evaluate_protocol(changed)


def test_regular_consensus_basin_outside_old_atan_series_domain():
    import networkx as nx

    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.0, theta=(1.0, -1.0, -1.0, 0.0, 1.0)[node % 5], nu_f=1.0
        )
    # This is an independently known consensus-basin state, not the ablation.
    result = certify_relational_transit_capture(
        graph,
        model=RelationalExchangeModel(1.0, phase_domain="positive_resultant"),
        cycles=(tuple(range(5)), tuple(range(5, 10))),
        horizon=Q(1, 128),
        time_step=Q(1, 128),
        order=4,
        requested_sector=None,
    )
    assert result.admitted and result.target_sector == 0
    assert result.initial.admitted and result.initial.target_sector == 0


def test_uniform_form_response_for_unequal_capacities_uses_existing_joint_law():
    import networkx as nx

    graph = nx.path_graph(2)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True)
    for node, capacity, phase in ((0, 1.0, -0.25), (1, 2.0, 0.25)):
        graph.nodes[node].update(EPI=3.0, theta=phase, nu_f=capacity)
    model = RelationalExchangeModel(1.0, phase_domain="positive_resultant")
    field = evaluate_relational_exchange(graph, model=model)
    delta, w = 0.5, 0.5
    expected_rate = (w * delta / math.pi, -2 * w * delta / math.pi)
    assert field.form_rate == pytest.approx(expected_rate, abs=2e-15)
    assert field.phase_rate == (0.0, 0.0)
    # The phase row is linear in x at fixed theta. Its tangent along x_dot
    # therefore gives theta_ddot exactly, without a second-order evolution law.
    for node, rate in enumerate(field.form_rate):
        graph.nodes[node]["EPI"] = rate
    tangent = evaluate_relational_exchange(graph, model=model).phase_rate
    H = math.pi * math.sin(delta) / delta
    acceleration = w * w * 3 * delta / (math.pi * H)
    assert tangent == pytest.approx((acceleration, -2 * acceleration), abs=2e-15)
    # A true balanced phase and uniform form, instead, remain stationary.
    for node in graph:
        graph.nodes[node].update(EPI=3.0, theta=0.25)
    balanced = evaluate_relational_exchange(graph, model=model)
    assert balanced.form_rate == balanced.phase_rate == (0.0, 0.0)


@pytest.fixture(scope="module")
def retained_ablation():
    base = campaign.ROOT / "docs/assets/relational_zero_form_response"
    hashes = {
        "result.json": "bb30b7b2ac8812871b7f7667c4a898d79293a4e25e7b96bff13ec29fdf5996c4",
        "result.protocol.json": "920413db8d67dc46c13914bdd550bb097b416ade8dadd0417883b5b5eeca49d1",
        "result.source.zip": "dcd7b49d4f596c9703ea191b74fadb0d28c6952d529363ce559b245984dbe8ee",
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest
    result, protocol = (
        campaign._read(base / name) for name in ("result.json", "result.protocol.json")
    )
    assert result["protocol"] == protocol
    return base, protocol, result, result["certificate"]["report"]


def test_frozen_ablation_and_reference_provenance(retained_ablation):
    base, protocol, result, _ = retained_ablation
    reference = campaign._reference_proof()
    for key in (
        "nodes",
        "edges",
        "cycles",
        "initial_phase",
        "capacity",
        "model",
        "numerical_policy",
    ):
        assert protocol[key] == reference["protocol"][key]
    assert protocol["initial_form"] == [0.0] * 10
    assert protocol["reference_proof_sha256"] == campaign.REFERENCE_PROOF_SHA256
    with zipfile.ZipFile(base / "result.source.zip") as archive:
        assert set(archive.namelist()) == set(protocol["source_sha256"])
        for name, digest in protocol["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    assert result["terminal_basin_admitted"] and result["outcome"] == "0"
    assert result["different_limit_certified"]
    assert not result["phase_geometry_sufficient_for_positive_capture"]
    assert not result["continuous_capture_admitted"]
    assert not result["original_finite_prediction_passed"]
    assert reference["continuous_capture_admitted"]


def test_complete_retained_ablation_tubes_and_consensus_gate(retained_ablation):
    _, _, _, report = retained_ablation

    def intervals(records):
        return tuple(I(item["lo"], item["hi"]) for item in records)

    box = intervals(report["initial_box"])
    assert box[0] == box[1] == I(0)
    time = Q(0)
    for step in report["steps"]:
        assert step["time"] == time and step["duration"] == Q(1, 8)
        tube = intervals(step["tube"])
        assert all(x.subset_of(b) for x, b in zip(box, tube))
        assert campaign.owner._regular_bounds(tube) == tuple(
            step["resultant_real_lower_bounds"]
        )
        assert min(step["resultant_real_lower_bounds"]) > 0
        rate = campaign.owner._flow(tube, Q(1, 2), Q(1, 2), Q(1))
        image = tuple(x + f * I(0, Q(1, 8)) for x, f in zip(box, rate))
        margin = min(min(x.lo - b.lo, b.hi - x.hi) for x, b in zip(image, tube))
        assert margin == step["picard_interior_margin"] > 0
        box = intervals(step["endpoint"])
        assert all(x.subset_of(b) for x, b in zip(box, tube))
        time += step["duration"]
    assert time == report["horizon"] == report["validated_horizon"] == 32
    assert len(report["steps"]) == 256
    assert report["target_sector"] == 0 and report["rectangle_kind"] == "consensus"
    assert report["requested_sector"] is None and report["initial_winding_zero"]
    assert all(i.lo > 0 for i in intervals(report["rectangle_margin_bounds"]))
    storage = campaign.owner._storage(box, Q(1))
    assert storage == I(**report["endpoint_storage"])
    assert storage.hi < Q(2117, 1000) < 7
    assert max(i.width for i in box) < Q(15, 10**14)
    assert report["failed_tube"] is None and not report["unavailable_reasons"]


def test_post_evaluation_winding_at_retained_endpoints_is_transient(retained_ablation):
    _, _, _, report = retained_ablation
    frames = [(Q(0), report["initial_box"])] + [
        (step["time"] + step["duration"], step["endpoint"]) for step in report["steps"]
    ]
    pi = pi_interval()
    positive_times, zero_times = [], []
    for time, box in frames:
        a, b = (I(value["lo"], value["hi"]) for value in box[2:])
        assert a.lo > 0 and a.hi < pi.lo
        assert b.abs_max < pi.lo and (a - b).abs_max < pi.lo
        # Raw edge sum telescopes. Only edge -2a can need a +2pi wrap.
        if 2 * a.lo > pi.hi:
            positive_times.append(time)
        else:
            assert 2 * a.hi < pi.lo
            zero_times.append(time)
    assert positive_times == [Q(i, 8) for i in range(16, 47)]
    assert zero_times == [Q(i, 8) for i in (*range(16), *range(47, 257))]
    # This classifies retained endpoints, not an uninterrupted lifetime between them.
