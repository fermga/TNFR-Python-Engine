"""Optional shared observations on one detached, explicitly prepared state.

Scientific witnesses use structural-observation-design-20261022's fixed
protocol. These calls install no laws and evaluate no trajectories.
"""

import json
import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics import fields, winding_certificates
from tnfr.sdk import Network, diagnose_network, study


def _radial_star(form=0.0):
    graph = nx.star_graph(6)
    for node, radius in enumerate((0, 1, 1, 2, 2, 3, 3)):
        graph.nodes[node].update(
            EPI=form, nu_f=1.0, theta=0.0, delta_nfr=2.0 ** (radius + 1) - 1
        )
        if node:
            graph.edges[0, node].update(length=float(radius), weight=1.0)
    return graph


def _directed_star(center_coherence=1.0):
    graph = nx.DiGraph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=1 / center_coherence - 1)
    for radius in range(1, 5):
        for _ in range(3):
            node = len(graph)
            graph.add_node(
                node, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=2.0**radius - 1
            )
            graph.add_edge(0, node, length=float(radius), weight=1.0)
    return graph


def _cycle(sign=1):
    graph = nx.cycle_graph(8)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.0, nu_f=1.0, theta=sign * math.tau * node / 8, delta_nfr=node / 8
        )
    nx.set_edge_attributes(graph, 1.0, "length")
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def test_default_output_and_optional_work_remain_unchanged(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unrequested optional owner was invoked")

    monkeypatch.setattr(fields, "observe_coherence_profile", forbidden)
    monkeypatch.setattr(winding_certificates, "certify_phase_winding", forbidden)
    monkeypatch.setattr(study, "_phase_evidence_projection", forbidden)
    network = Network(_radial_star())
    default = diagnose_network(network)
    assert set(default) == {
        "schema_version",
        "name",
        "state",
        "metrics",
        "nodal",
        "tetrad",
        "scope",
    }
    assert default["schema_version"] == "tnfr.diagnostics.v1"
    assert (
        diagnose_network(
            network,
            include_coherence_profile=False,
            include_phase_evidence=False,
            winding_cycles=(),
        )
        == default
    )


def test_optional_result_preserves_every_default_field_and_is_json_serializable():
    network = Network(_radial_star())
    baseline = diagnose_network(network)
    extended = diagnose_network(
        network, include_coherence_profile=True, include_phase_evidence=True
    )
    optional = extended.pop("observations")
    assert extended == baseline
    assert optional["request"] == {
        "include_coherence_profile": True,
        "include_phase_evidence": True,
        "winding_cycles": [],
    }
    assert optional["coherence_profile"]["available"]
    assert optional["phase_evidence"]["available"]
    json.dumps(optional, allow_nan=False)


def test_profile_phase_and_detached_graph_are_each_collected_once(monkeypatch):
    counts = {"copy": 0, "phase": 0, "profile": 0, "fit": 0}
    snapshot_ids = []

    def wrapper(original, name):
        def observed(graph, *args, **kwargs):
            counts[name] += 1
            if name != "copy":
                snapshot_ids.append(id(graph))
            return original(graph, *args, **kwargs)

        return observed

    monkeypatch.setattr(
        study, "copy_graph_state", wrapper(study.copy_graph_state, "copy")
    )
    monkeypatch.setattr(
        study,
        "observe_phase_curvature",
        wrapper(study.observe_phase_curvature, "phase"),
    )
    monkeypatch.setattr(
        fields,
        "observe_coherence_profile",
        wrapper(fields.observe_coherence_profile, "profile"),
    )
    from tnfr.physics import _coherence_fit

    monkeypatch.setattr(
        _coherence_fit,
        "fit_coherence_length",
        wrapper(_coherence_fit.fit_coherence_length, "fit"),
    )
    monkeypatch.setattr(
        study,
        "estimate_coherence_length_with_provenance",
        lambda _: pytest.fail("profile must reuse its fit"),
    )
    graph = _radial_star()
    result = diagnose_network(
        Network(graph), include_coherence_profile=True, include_phase_evidence=True
    )
    assert counts == {"copy": 1, "phase": 1, "profile": 1, "fit": 1}
    assert len(set(snapshot_ids)) == 1 and snapshot_ids[0] != id(graph)
    profile = result["observations"]["coherence_profile"]["value"]
    assert profile["fit"]["value"] == result["tetrad"]["xi_c"]["value"]


def test_directed_profile_adds_information_missing_from_full_tetrad():
    reports = [
        diagnose_network(
            Network(_directed_star(center)), include_coherence_profile=True
        )
        for center in (1.0, 0.5)
    ]
    before, after = reports
    for key in ("phi_s", "grad_phi", "k_phi"):
        assert before["tetrad"][key] == after["tetrad"][key]
    assert before["tetrad"]["phi_s"]["value"] == pytest.approx(
        [499 / 48] + [0.0] * 12, rel=2e-13
    )
    assert before["tetrad"]["xi_c"]["value"] == pytest.approx(
        1 / math.log(2), rel=2e-13
    )
    assert after["tetrad"]["xi_c"]["value"] == pytest.approx(
        before["tetrad"]["xi_c"]["value"], rel=2e-13
    )
    first = before["observations"]["coherence_profile"]["value"]
    second = after["observations"]["coherence_profile"]["value"]
    assert first["fit"]["intercept"] == pytest.approx(0, abs=2e-14)
    assert second["fit"]["intercept"] == pytest.approx(-math.log(2), abs=2e-14)
    assert before["nodal"][0]["value"]["expected_depi_dt"] == 0
    assert after["nodal"][0]["value"]["expected_depi_dt"] == 1


def test_existing_state_projection_preserves_form_and_capacity_witnesses():
    first, second = [
        diagnose_network(Network(_radial_star(form)), include_coherence_profile=True)
        for form in (0.0, 1.0)
    ]
    assert first["tetrad"] == second["tetrad"]
    assert (
        first["observations"]["coherence_profile"]
        == second["observations"]["coherence_profile"]
    )
    assert first["state"]["nodes"][0]["epi"]["value"] == 0
    assert second["state"]["nodes"][0]["epi"]["value"] == 1

    graph = nx.path_graph(4)
    for node, (form, pressure, phase) in enumerate(
        zip((0, 1, -1, 0.5), (-1, 0.5, 0.5, 0), (0, 0.2, 0.4, 0.6))
    ):
        graph.nodes[node].update(EPI=form, delta_nfr=pressure, theta=phase, nu_f=1.0)
    slow = diagnose_network(Network(graph))
    nx.set_node_attributes(graph, 2.0, "nu_f")
    fast = diagnose_network(Network(graph))
    assert slow["tetrad"] == fast["tetrad"]
    assert all(row["nu_f"]["value"] == 2 for row in fast["state"]["nodes"])
    for left, right in zip(slow["nodal"], fast["nodal"]):
        assert (
            right["value"]["expected_depi_dt"] == 2 * left["value"]["expected_depi_dt"]
        )


def test_declared_cycle_orientation_distinguishes_equal_tetrads():
    reports = [
        diagnose_network(
            Network(_cycle(sign)),
            winding_cycles=(tuple(range(8)), tuple(reversed(range(8)))),
        )
        for sign in (1, -1)
    ]
    for key in ("phi_s", "grad_phi", "k_phi"):
        assert reports[0]["tetrad"][key]["value"] == pytest.approx(
            reports[1]["tetrad"][key]["value"], abs=2e-14, rel=2e-13
        )
    for sign, report in zip((1, -1), reports):
        cycles = report["observations"]["winding_cycles"]
        assert cycles[0]["value"]["winding"] == sign
        assert cycles[1]["value"]["winding"] == -sign
        assert all(row["available"] for row in cycles)
        assert cycles[0]["value"]["orientation"] == "declared"
        assert cycles[0]["value"]["minimum_branch_margin"] > 0
        assert "coherence_profile" not in report["observations"]
        assert "phase_evidence" not in report["observations"]


def test_phase_evidence_retains_exact_component_sums_and_distinct_cancellation():
    graph = nx.star_graph(4)
    for node, phase in zip(graph, (0.3, 0.0, 0.0, math.pi, -math.pi)):
        graph.nodes[node].update(EPI=1.0, nu_f=1.0, theta=phase, delta_nfr=0.0)
    report = diagnose_network(Network(graph), include_phase_evidence=True)
    evidence = report["observations"]["phase_evidence"]
    assert evidence["available"] and not report["tetrad"]["k_phi"]["available"]
    payload = evidence["value"]
    center = payload["rows"][0]
    assert center["resultant"]["joint_zero"]
    assert center["resultant"]["angle"] is None
    for row in payload["rows"]:
        result = row["resultant"]
        for column, name in enumerate(("real_sum", "imag_sum")):
            exact = sum(
                (
                    Fraction(payload["components"][node][column])
                    for node in row["neighbors"]
                ),
                Fraction(0),
            )
            assert result[name] == {
                "numerator": exact.numerator,
                "denominator": exact.denominator,
            }
    json.dumps(report, allow_nan=False)


def test_invalid_optional_channels_fail_independently_without_repair():
    graph = _cycle()
    graph.nodes[0]["theta"] = math.nan
    report = diagnose_network(
        Network(graph),
        include_phase_evidence=True,
        include_coherence_profile=True,
        winding_cycles=(tuple(range(8)),),
    )
    observations = report["observations"]
    assert observations["coherence_profile"]["available"]
    assert not observations["phase_evidence"]["available"]
    assert not observations["winding_cycles"][0]["available"]
    assert observations["winding_cycles"][0]["value"]["status"] == "undefined"
    assert report["tetrad"]["phi_s"]["available"]
    assert math.isnan(graph.nodes[0]["theta"])


def test_invalid_metric_profile_does_not_hide_phase_or_winding():
    graph = _cycle()
    graph.edges[0, 1]["length"] = -1
    report = diagnose_network(
        Network(graph),
        include_phase_evidence=True,
        include_coherence_profile=True,
        winding_cycles=(tuple(range(8)),),
    )
    observations = report["observations"]
    assert not observations["coherence_profile"]["available"]
    assert observations["phase_evidence"]["available"]
    assert observations["winding_cycles"][0]["available"]


@pytest.mark.parametrize("cycle", ((0, 1, 2), (0, 1, 99), (0, 1, 0), ()))
def test_absent_or_invalid_oriented_cycles_retain_owner_certificate(cycle):
    graph = _cycle()
    result = diagnose_network(Network(graph), winding_cycles=(cycle,))["observations"][
        "winding_cycles"
    ][0]
    assert not result["available"]
    assert result["value"]["status"] == "undefined"
    assert result["value"]["winding"] is None
    assert not result["value"]["cycle_exists"]
    if 99 in cycle:
        assert result["value"]["cycle_nodes"][-1] is None
    json.dumps(result, allow_nan=False)


def test_branch_ambiguity_retains_a_present_but_undefined_cycle():
    graph = _cycle()
    graph.nodes[1]["theta"] = math.pi
    result = diagnose_network(Network(graph), winding_cycles=(tuple(range(8)),))[
        "observations"
    ]["winding_cycles"][0]
    assert not result["available"]
    assert result["value"]["cycle_exists"]
    assert result["value"]["winding"] is None
    assert "branch" in result["error"]["message"]


class _OpaqueNode:
    def __deepcopy__(self, memo):
        raise AssertionError("opaque node identities must not be copied")

    def __str__(self):
        return "same display label"


def test_opaque_identity_indices_survive_all_optional_projections():
    nodes = tuple(_OpaqueNode() for _ in range(8))
    graph = nx.relabel_nodes(_cycle(), dict(enumerate(nodes)))
    result = diagnose_network(
        Network(graph),
        include_coherence_profile=True,
        include_phase_evidence=True,
        winding_cycles=(nodes,),
    )
    observations = result["observations"]
    assert observations["coherence_profile"]["available"]
    assert observations["coherence_profile"]["value"]["nodes"] == list(range(8))
    assert observations["phase_evidence"]["value"]["nodes"] == list(range(8))
    cycle = observations["winding_cycles"][0]
    assert cycle["available"] and cycle["value"]["cycle_nodes"] == list(range(8))
    assert len(set(cycle["value"]["cycle_labels"])) == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "name", ("include_coherence_profile", "include_phase_evidence")
)
@pytest.mark.parametrize("value", (0, 1, "false", "true", None, [], {}))
def test_flag_admission_precedes_any_snapshot_or_field_read(name, value, monkeypatch):
    monkeypatch.setattr(
        study, "copy_graph_state", lambda _: pytest.fail("invalid request copied graph")
    )
    with pytest.raises(TypeError, match="boolean"):
        diagnose_network(Network(nx.Graph()), **{name: value})


@pytest.mark.parametrize(
    "cycles",
    (
        None,
        "012",
        {0, 1, 2},
        ((node for node in range(3)),),
        ({0, 1, 2},),
        ("012",),
        ([[0], 1, 2],),
    ),
)
def test_cycle_request_admission_precedes_any_snapshot(cycles, monkeypatch):
    monkeypatch.setattr(
        study, "copy_graph_state", lambda _: pytest.fail("invalid request copied graph")
    )
    with pytest.raises(TypeError):
        diagnose_network(Network(nx.Graph()), winding_cycles=cycles)


def test_numpy_booleans_are_explicit_flags_and_state_is_not_mutated():
    graph = _radial_star()
    saved = deepcopy(graph)
    result = diagnose_network(
        Network(graph),
        include_coherence_profile=np.bool_(True),
        include_phase_evidence=np.bool_(False),
    )
    assert result["observations"]["request"]["include_coherence_profile"] is True
    assert nx.utils.graphs_equal(graph, saved)
    result["observations"]["coherence_profile"]["value"]["coherence"][0] = 99
    assert graph.nodes[0]["delta_nfr"] == 1


@pytest.mark.parametrize("size", (0, 1))
def test_empty_support_keeps_profile_evidence_with_unavailable_xi(size):
    report = diagnose_network(
        Network(nx.empty_graph(size)),
        include_coherence_profile=True,
        include_phase_evidence=True,
    )
    assert report["observations"]["coherence_profile"]["available"]
    assert report["observations"]["coherence_profile"]["value"]["fit"]["value"] is None
    assert report["observations"]["phase_evidence"]["available"]
    assert not report["tetrad"]["xi_c"]["available"]
    assert report["tetrad"]["xi_c"]["provenance"]["method"] == "unavailable"
    json.dumps(report, allow_nan=False)
