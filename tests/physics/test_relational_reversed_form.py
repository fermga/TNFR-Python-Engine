"""Equal-storage form reversal and explicit competing-basin reporting."""

import hashlib
import json
import zipfile
from copy import deepcopy
from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import pytest

from benchmarks import relational_transit_proof as campaign
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.mathematics._rational_interval import I, pi_interval


def normalized(record):
    return json.loads(
        json.dumps(record, default=campaign._project), object_hook=campaign._exact
    )


@pytest.fixture(scope="module")
def prospective():
    return normalized(campaign.prepare_protocol(reverse_form=True))


def test_only_initial_form_orientation_changes(prospective):
    original, _ = campaign._retained()
    reference = campaign._reference_proof()["protocol"]
    for key in ("nodes", "edges", "cycles", "initial_phase", "capacity", "model"):
        assert prospective[key] == original[key] == reference[key]
    assert prospective["intervention"] == "reversed_initial_form"
    assert prospective["initial_form"] == [-value for value in original["initial_form"]]
    assert prospective["numerical_policy"] == reference["numerical_policy"]
    assert prospective["requested_sector"] is None
    assert prospective["reference_target_sector"] == 1
    assert set(prospective["competing_outcomes"]) == {"-1", "0", "1", "unavailable"}
    assert prospective["terminal_prediction"].startswith("unselected")


def test_zero_form_and_reversal_are_mutually_exclusive():
    with pytest.raises(ValueError):
        campaign.prepare_protocol(zero_form=True, reverse_form=True)


def test_static_derivation_retains_exact_orientation_and_equal_energy(prospective):
    derivation = prospective["initial_derivation"]
    reference, reverse = derivation["reference"], derivation["control"]
    q = 1 + Q(1, 1 << 54)
    a = Q(prospective["initial_phase"][0])
    assert reference["coordinates"] == [q, Q(0), a, a]
    assert reverse["coordinates"] == [-q, Q(0), a, a]
    assert reference["form_storage"] == reverse["form_storage"] == Q(4, 5) * q**2
    assert reference["storage_rate"] == reverse["storage_rate"] == -Q(2, 3) * q**2
    assert reference["total_storage_bounds"] == reverse["total_storage_bounds"]
    lo, hi = reverse["total_storage_bounds"]
    assert Q(87365, 10000) < lo <= hi < Q(87366, 10000)
    reference_rates, reverse_rates = (
        record["phase_rate_bounds"] for record in (reference, reverse)
    )
    assert reverse_rates == [[-hi, -lo] for lo, hi in reference_rates]
    assert reverse_rates[0][1] < 0 < reference_rates[0][0]
    assert reference_rates[1] == reverse_rates[1] == [Q(0), Q(0)]
    # Equal storage and its first derivative do not fix the next derivative.
    assert reference["exchange_flux_bounds"][1] < 0
    assert reverse["exchange_flux_bounds"][0] > 0
    assert derivation["acceleration_difference_bounds"][0] > 0
    assert all(derivation["gates"].values())


def test_reversal_preserves_storage_and_loss_but_reverses_phase_rate(prospective):
    original, _ = campaign._retained()
    graph = nx.Graph()
    graph.add_nodes_from(prospective["nodes"])
    graph.add_edges_from(prospective["edges"])
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True)
    for node, phase, capacity in zip(
        prospective["nodes"], prospective["initial_phase"], prospective["capacity"]
    ):
        graph.nodes[node].update(theta=phase, nu_f=capacity)
    model = RelationalExchangeModel(**prospective["model"])
    fields = []
    for form in (original["initial_form"], prospective["initial_form"]):
        for node, value in zip(prospective["nodes"], form):
            graph.nodes[node]["EPI"] = value
        fields.append(evaluate_relational_exchange(graph, model=model))
    forward, reverse = fields
    # The exact represented Dirichlet sum is independent of contrast orientation.
    expected_form_storage = sum(
        (Q(original["initial_form"][i]) - Q(original["initial_form"][j])) ** 2 / 2
        for i, j in original["edges"]
    )
    assert forward.form_storage == reverse.form_storage == expected_form_storage
    assert forward.phase_storage == reverse.phase_storage
    assert forward.storage == reverse.storage
    assert forward.continuous_loss == reverse.continuous_loss
    assert reverse.phase_rate == tuple(-value for value in forward.phase_rate)
    assert reverse.phase_rate[0] < 0 < forward.phase_rate[0]
    assert reverse.phase_rate[4] == forward.phase_rate[4] == 0
    # The phase source is held, so this is not time reversal of the joint law.
    assert reverse.phase_source == forward.phase_source
    assert reverse.form_rate != tuple(-value for value in forward.form_rate)


@pytest.mark.parametrize("sector", (1, 0, -1, None))
def test_outcomes_preserve_consensus_and_do_not_invent_a_limit(
    prospective, monkeypatch, sector
):
    calls = []
    monkeypatch.setattr(
        campaign, "prepare_protocol", lambda **kwargs: deepcopy(prospective)
    )

    def stub(graph, **kwargs):
        calls.append(kwargs)
        assert [graph.nodes[node]["EPI"] for node in prospective["nodes"]] == (
            prospective["initial_form"]
        )
        return SimpleNamespace(
            admitted=sector is not None, initial_winding_zero=True, target_sector=sector
        )

    monkeypatch.setattr(campaign.owner, "certify_relational_transit_capture", stub)
    monkeypatch.setattr(
        campaign, "relational_report_to_dict", lambda report: {"stub": True}
    )
    report = campaign.evaluate_protocol(prospective)
    assert len(calls) == 1 and calls[0]["requested_sector"] is None
    assert report["terminal_basin_admitted"] == (sector is not None)
    assert report["target_sector"] == sector
    assert report["continuous_capture_admitted"] == (sector == 1)
    assert report["outcome"] == (str(sector) if sector is not None else "unavailable")
    assert report["same_positive_limit"] == (
        sector == 1 if sector is not None else None
    )
    assert report["different_limit_certified"] == (
        sector != 1 if sector is not None else None
    )
    assert report["initial_storage_selector_refuted"] == (
        sector != 1 if sector is not None else None
    )


@pytest.mark.parametrize(
    "changed_key",
    (
        "initial_form",
        "initial_phase",
        "capacity",
        "model",
        "numerical_policy",
        "requested_sector",
        "intervention",
    ),
)
def test_changed_frozen_control_rejects_before_execution(
    prospective, monkeypatch, changed_key
):
    monkeypatch.setattr(
        campaign, "prepare_protocol", lambda **kwargs: deepcopy(prospective)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("changed frozen control executed")

    monkeypatch.setattr(campaign.owner, "certify_relational_transit_capture", forbidden)
    changed = deepcopy(prospective)
    changed[changed_key] = "not the frozen control"
    with pytest.raises(ValueError, match="frozen proof protocol"):
        campaign.evaluate_protocol(changed)


def test_cli_refuses_conflicting_interventions(tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as error:
        campaign.main(
            ["--prepare", "--zero-form", "--reverse-form", "--output", str(output)]
        )
    assert error.value.code == 2
    assert not output.with_suffix(".protocol.json").exists()
    assert not output.with_suffix(".source.zip").exists()


def test_cli_reversal_freeze_requires_matching_evaluation_flag(tmp_path, monkeypatch):
    calls = []

    def preparation(**kwargs):
        calls.append(kwargs)
        return {"source_sha256": {}, "intervention": "reversed_initial_form"}

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched CLI intervention executed")

    monkeypatch.setattr(campaign, "prepare_protocol", preparation)
    monkeypatch.setattr(campaign, "evaluate_protocol", forbidden)
    output = tmp_path / "result.json"
    assert campaign.main(["--prepare", "--reverse-form", "--output", str(output)]) == 0
    assert calls[0].get("reverse_form") is True
    assert not calls[0].get("zero_form", False)
    with pytest.raises(ValueError, match="frozen protocol"):
        campaign.main(["--output", str(output)])
    assert output.with_suffix(".protocol.json").exists()
    assert output.with_suffix(".source.zip").exists()
    assert not output.exists()


@pytest.fixture(scope="module")
def retained_reversal():
    base = campaign.ROOT / "docs/assets/relational_reversed_form_response"
    hashes = {
        "result.json": "f3de2b74e9f0ab9ba3b621dfc471266e22f23a391bb1f4f76153538d924e2963",
        "result.protocol.json": "050e265497f9bd7c483c20ab77bbf58baf763bb13e42575d8227bca9801bd9a8",
        "result.source.zip": "9816d178479941e1d9ac007452b06c41d75e2fe7eb5d20d83c81bb1c42494e20",
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest
    result, protocol = (
        campaign._read(base / name) for name in ("result.json", "result.protocol.json")
    )
    assert result["protocol"] == protocol
    return base, protocol, result, result["certificate"]["report"]


def test_retained_reversal_provenance_and_equal_storage(retained_reversal):
    base, protocol, result, report = retained_reversal
    original, _ = campaign._retained()
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
    assert protocol["initial_form"] == [-value for value in original["initial_form"]]
    assert protocol["reference_proof_sha256"] == campaign.REFERENCE_PROOF_SHA256
    assert len(protocol["source_sha256"]) == 604
    with zipfile.ZipFile(base / "result.source.zip") as archive:
        assert set(archive.namelist()) == set(protocol["source_sha256"])
        for name, digest in protocol["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    q = -(1 + Q(1, 1 << 54))
    assert I(**report["initial_box"][0]) == I(q)
    assert I(**report["initial_box"][1]) == I(0)
    initial = protocol["initial_derivation"]
    for record in (initial["reference"], initial["control"]):
        assert record["form_storage"] == Q(4, 5) * q**2
        assert record["storage_rate"] == -Q(2, 3) * q**2
    assert (
        initial["reference"]["total_storage_bounds"]
        == initial["control"]["total_storage_bounds"]
    )
    assert result["terminal_basin_admitted"] and result["outcome"] == "0"
    assert result["different_limit_certified"]
    assert result["initial_storage_selector_refuted"]
    assert not result["same_positive_limit"]
    assert not result["continuous_capture_admitted"]
    assert not result["original_finite_prediction_passed"]
    assert reference["continuous_capture_admitted"]


def test_retained_reversal_tube_chain_and_terminal_consensus(retained_reversal):
    _, _, _, report = retained_reversal

    def intervals(records):
        return tuple(I(**record) for record in records)

    box = intervals(report["initial_box"])
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
    assert box == intervals(report["endpoint"])
    assert report["target_sector"] == 0 and report["rectangle_kind"] == "consensus"
    assert report["requested_sector"] is None and report["initial_winding_zero"]
    assert all(bound.lo > 0 for bound in intervals(report["rectangle_margin_bounds"]))
    pi = pi_interval()
    assert box[2].abs_max < 2 * pi.lo / 3
    assert box[3].abs_max < pi.lo / 2
    storage = campaign.owner._storage(box, Q(1))
    assert storage == I(**report["endpoint_storage"])
    assert storage.hi < Q(138, 100) < 7
    assert max(bound.width for bound in box) < Q(2, 10**14)
    assert report["failed_tube"] is None and not report["unavailable_reasons"]


def test_retained_whole_time_tubes_exclude_any_winding_change(retained_reversal):
    _, _, _, report = retained_reversal
    pi = pi_interval()
    margins = []
    for step in report["steps"]:
        a, b = (I(**record) for record in step["tube"][2:])
        # Each ring has raw phase increments (-2a, a-b, b, b, a-b).
        # Their exact symbolic sum is zero. If none reaches a wrap boundary,
        # every continuous state in the tube therefore has winding zero.
        raw_edges = (-2 * a, a - b, b, b, a - b)
        margins.append(min(pi.lo - edge.abs_max for edge in raw_edges))
    assert len(margins) == 256
    assert min(margins) > Q(26165, 10**6)
