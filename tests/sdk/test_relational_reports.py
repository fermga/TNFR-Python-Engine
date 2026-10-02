"""Public wiring and exact evidence export, without another research campaign."""

import json
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from types import ModuleType

import networkx as nx
import pytest

from tnfr.sdk import (
    Network,
    RelationalExchangeModel,
    export_to_json,
    relational_report_to_dict,
)


def _network():
    graph = nx.Graph()
    graph.add_edge(("port", 0), "right")
    for node, form in zip(graph, (0.25, -0.25), strict=True):
        graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0)
    return Network(graph)


def test_pattern_sdk_is_a_thin_delegate(monkeypatch):
    from tnfr.physics import relational_observations as owner

    network, model = _network(), RelationalExchangeModel(1)
    reference = {node: 0 for node in network.G}
    regions, cycles = (("right",),), ()
    calls, marker = [], object()

    def observe(graph, **kwargs):
        calls.append((graph, kwargs))
        return marker

    monkeypatch.setattr(owner, "observe_relational_pattern", observe)
    assert (
        network.relational_pattern(
            model, reference_phase=reference, regions=regions, cycles=cycles
        )
        is marker
    )
    assert calls == [
        (
            network.G,
            dict(
                model=model, reference_phase=reference, regions=regions, cycles=cycles
            ),
        )
    ]


def test_exact_field_step_and_pattern_export_reuses_atomic_writer(tmp_path):
    network, model = _network(), RelationalExchangeModel(1)
    field = network.relational_exchange(model)
    step = network.step_relational(model, dt=1 / 64)
    pattern = network.relational_pattern(
        model, reference_phase={node: 0 for node in network.G}, regions=(("right",),)
    )
    for report, exact in (
        (field, field.storage),
        (step, step.energy_step_defect),
        (pattern, pattern.regions[0].phase_error_mean),
    ):
        data = relational_report_to_dict(report)
        path = tmp_path / f"{type(report).__name__}.json"
        export_to_json(data, path)
        assert json.loads(path.read_text(encoding="utf-8")) == data
        assert data["schema"] == "tnfr.relational-report.v1"
        body = data["report"]
        if report is field:
            encoded = body["storage"]
            assert body["nodes"] == [["port", 0], "right"]
            body["epi"][0] = 999
            assert field.epi[0] == 0.25
        elif report is step:
            encoded = body["energy_step_defect"]
            assert body["before"]["nodes"] == body["after"]["nodes"]
        else:
            encoded = body["regions"][0]["phase_error_mean"]
            assert body["regions"][0]["transport"]["mass_identity_residual"] == {
                "numerator": 0,
                "denominator": 1,
            }
        assert Fraction(encoded["numerator"], encoded["denominator"]) == exact


def test_pattern_export_retains_signed_work_and_paired_cut_rates_without_evolution(
    tmp_path,
):
    network, model = _network(), RelationalExchangeModel(1)
    pattern = network.relational_pattern(
        model, reference_phase=dict.fromkeys(network.G, 0), regions=(("right",),)
    )
    data = relational_report_to_dict(pattern)
    path = tmp_path / "signed-work.json"
    export_to_json(data, path)
    assert json.loads(path.read_text(encoding="utf-8")) == data
    body, region = data["report"], data["report"]["regions"][0]
    assert body["field"]["work"]["dissipation"] == [
        {"numerator": 1, "denominator": 8},
        {"numerator": 1, "denominator": 8},
    ]
    assert region["work"]["form_work"] == {"numerator": -1, "denominator": 8}
    assert region["work"]["exchange"] == {"numerator": 0, "denominator": 1}
    assert region["boundary"]["cut"]["region"] == ["right"]
    assert region["boundary"]["cut"]["outward_cut_current"] == {
        "numerator": -1,
        "denominator": 2,
    }
    assert region["boundary"]["form_boundary_rate"] == {
        "numerator": 1,
        "denominator": 4,
    }
    assert region["boundary"]["phase_boundary_rate"] == {
        "numerator": -1,
        "denominator": 4,
    }
    assert region["boundary"]["form_identity_residual"] == {
        "numerator": 0,
        "denominator": 1,
    }
    phase_defect = pattern.regions[0].boundary.phase_rate_residual
    assert region["boundary"]["phase_rate_residual"] == {
        "numerator": phase_defect.numerator,
        "denominator": phase_defect.denominator,
    }
    body["field"]["work"]["exchange"][0]["numerator"] = 99
    region["boundary"]["cut"]["region"].append("changed")
    assert pattern.field.work.exchange == (Fraction(0), Fraction(0))
    assert pattern.regions[0].boundary.cut.region == ("right",)


def test_pattern_export_retains_unavailable_divided_rates_with_zero_capacity():
    network, model = _network(), RelationalExchangeModel(1)
    network.G.nodes["right"]["nu_f"] = 0.0
    pattern = network.relational_pattern(
        model, reference_phase=dict.fromkeys(network.G, 0), regions=(("right",),)
    )
    region = relational_report_to_dict(pattern)["report"]["regions"][0]
    assert (
        region["boundary"]["weighted_rate_unavailable_reason"]
        == "zero_capacity_in_region"
    )
    assert region["boundary"]["form_weighted_rate"] is None
    assert region["boundary"]["phase_weighted_rate"] is None
    assert region["work"]["exchange"] == {"numerator": 0, "denominator": 1}
    assert region["boundary"]["cut"]["outward_cut_current"] == {
        "numerator": -1,
        "denominator": 2,
    }
    assert region["phase_response"]["total_rate"] == {
        "numerator": 0,
        "denominator": 1,
    }
    assert region["phase_response"]["identity_residual"] == {
        "numerator": 0,
        "denominator": 1,
    }


def test_sdk_exports_unweighted_phase_response_and_actual_rounding_evidence(tmp_path):
    network, model = _network(), RelationalExchangeModel(1)
    pattern = network.relational_pattern(
        model, reference_phase=dict.fromkeys(network.G, 0), regions=(("right",),)
    )
    data = relational_report_to_dict(pattern)
    path = tmp_path / "phase-response.json"
    export_to_json(data, path)
    assert json.loads(path.read_text(encoding="utf-8")) == data
    field = data["report"]["field"]
    response = data["report"]["regions"][0]["phase_response"]
    mobility = pattern.field.phase_mobility[1]
    assert field["phase_mobility"][1] == {
        "numerator": mobility.numerator,
        "denominator": mobility.denominator,
    }
    actual = Fraction(pattern.field.phase_rate[1])
    assert response["total_rate"] == {
        "numerator": actual.numerator,
        "denominator": actual.denominator,
    }
    rounding = pattern.field.phase_rate_rounding_defect[1]
    assert response["rounding_residual"] == field["phase_rate_rounding_defect"][1]
    assert response["rounding_residual"] == {
        "numerator": rounding.numerator,
        "denominator": rounding.denominator,
    }
    for key in (
        "mobility_variance",
        "form_gradient_variance",
        "mobility_gradient_covariance",
        "covariance_rate",
        "covariance_rate_squared_bound",
        "identity_residual",
    ):
        assert response[key] == {"numerator": 0, "denominator": 1}
    response["mean_mobility"]["numerator"] = 999
    field["phase_rate_rounding_defect"][1]["numerator"] = 999
    assert pattern.regions[0].phase_response.mean_mobility == mobility
    assert pattern.field.phase_rate_rounding_defect[1] == rounding


def test_projection_rejects_opaque_labels_and_nonfinite_tampering(tmp_path):
    network, model = _network(), RelationalExchangeModel(1)
    field = network.relational_exchange(model)
    opaque = object()
    altered = replace(field, nodes=(opaque, "right"))
    path = tmp_path / "retained.json"
    path.write_text("retained", encoding="utf-8")
    with pytest.raises(TypeError, match="node labels"):
        export_to_json(relational_report_to_dict(altered), path)
    assert path.read_text(encoding="utf-8") == "retained"
    with pytest.raises(ValueError, match="nonfinite"):
        relational_report_to_dict(replace(field, epi=(float("nan"), -0.25)))
    with pytest.raises(TypeError, match="expected a relational"):
        relational_report_to_dict(model)


def test_undefined_cycle_cannot_smuggle_opaque_labels_through_record_projection():
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    network, model = _network(), RelationalExchangeModel(1)
    report = network.relational_pattern(
        model,
        reference_phase={node: 0 for node in network.G},
        regions=(("right",),),
        cycles=((("port", 0), "right", OpaqueLabel(3)),),
    )
    assert not report.winding[0].is_defined
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


@pytest.mark.parametrize(
    "method, owner_name, target",
    (
        ("relational_capture", "certify_relational_capture", {}),
        (
            "relational_local_capture",
            "certify_relational_local_capture",
            {"target_sector": -1},
        ),
        (
            "relational_sector_capture",
            "certify_relational_sector_capture",
            {"target_sector": -1},
        ),
    ),
)
def test_capture_sdk_delegates_without_own_admission(
    monkeypatch, method, owner_name, target
):
    from tnfr.physics import relational_capture as owner

    network, model = _network(), RelationalExchangeModel(1)
    cycles, calls, marker = (("supplied",),), [], object()

    def capture(graph, **kwargs):
        calls.append((graph, kwargs))
        return marker

    monkeypatch.setattr(owner, owner_name, capture)
    assert getattr(network, method)(model, cycles=cycles, **target) is marker
    assert calls == [(network.G, dict(model=model, cycles=cycles, **target))]


def _stub_transit_owner(monkeypatch):
    """Replace trajectory execution with only its declared report interface."""

    @dataclass(frozen=True)
    class RelationalTransitCertificate:
        initial: object
        elapsed: Fraction

    owner = ModuleType("tnfr.physics.relational_transit")
    owner.RelationalTransitCertificate = RelationalTransitCertificate
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    return owner


@pytest.mark.parametrize("requested_order", (None, 8))
@pytest.mark.parametrize("requested_sector", (None, 1, 0, -1))
def test_transit_sdk_delegates_exact_options_without_execution(
    monkeypatch, requested_order, requested_sector
):
    owner = _stub_transit_owner(monkeypatch)
    network, model = _network(), RelationalExchangeModel(1)
    cycles, calls, marker = (("supplied",),), [], object()
    horizon, time_step = Fraction(1, 2), Fraction(1, 128)

    def capture(graph, **kwargs):
        calls.append((graph, kwargs))
        return marker

    owner.certify_relational_transit_capture = capture
    options = {} if requested_order is None else {"order": requested_order}
    assert (
        network.relational_transit_capture(
            model=model,
            cycles=cycles,
            horizon=horizon,
            time_step=time_step,
            requested_sector=requested_sector,
            **options,
        )
        is marker
    )
    assert calls == [
        (
            network.G,
            dict(
                model=model,
                cycles=cycles,
                horizon=horizon,
                time_step=time_step,
                order=12 if requested_order is None else requested_order,
                requested_sector=requested_sector,
            ),
        )
    ]


def _initial_capture():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=0.0, nu_f=1.0)
    return Network(graph).relational_capture(
        RelationalExchangeModel(1), cycles=(tuple(range(5)), tuple(range(5, 10)))
    )


def test_transit_export_preserves_nested_initial_evidence_without_execution(
    monkeypatch, tmp_path
):
    owner = _stub_transit_owner(monkeypatch)
    initial = _initial_capture()
    report = owner.RelationalTransitCertificate(initial, Fraction(1, 7))
    output = relational_report_to_dict(report)
    path = tmp_path / "transit.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    assert output["report_type"] == "RelationalTransitCertificate"
    assert output["report"]["elapsed"] == {"numerator": 1, "denominator": 7}
    assert output["report"]["initial"] == relational_report_to_dict(initial)["report"]
    output["report"]["initial"]["field"]["epi"][0] = 123
    assert initial.field.epi[0] == 0


@pytest.mark.parametrize("location", ("field", "winding"))
def test_transit_export_checks_nested_initial_labels(monkeypatch, location):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    owner = _stub_transit_owner(monkeypatch)
    initial = _initial_capture()
    opaque = OpaqueLabel(3)
    if location == "field":
        initial = replace(
            initial,
            field=replace(initial.field, nodes=(opaque,) + initial.field.nodes[1:]),
        )
    else:
        winding = replace(initial.winding[0], cycle_nodes=(opaque,))
        initial = replace(initial, winding=(winding, initial.winding[1]))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(
            owner.RelationalTransitCertificate(initial, Fraction(0))
        )


def test_local_capture_report_exports_complete_exact_endpoint_evidence(tmp_path):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=0.0, nu_f=1.0)
    graph.nodes[7]["EPI"] = 1 / 1024
    report = Network(graph).relational_local_capture(
        RelationalExchangeModel(1),
        cycles=(tuple(range(5)), tuple(range(5, 10))),
        target_sector=0,
    )
    assert report.admitted
    output = relational_report_to_dict(report)
    path = tmp_path / "local-capture.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    assert output["report_type"] == "RelationalLocalCaptureCertificate"
    upper = report.quotient_squared_bounds[1]
    assert output["report"]["quotient_squared_bounds"][1] == {
        "numerator": upper.numerator,
        "denominator": upper.denominator,
    }
    assert output["report"]["target_sector"] == 0
    output["report"]["field"]["epi"][7] = 123.0
    assert report.field.epi[7] == graph.nodes[7]["EPI"] == 1 / 1024


@pytest.mark.parametrize("beta, heterogeneous", ((1.0, False), (2.0, True)))
def test_sector_capture_export_retains_exact_acute_gap_and_barrier_evidence(
    tmp_path, beta, heterogeneous
):
    import math

    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    phase = (4 * math.pi / 5, -4 * math.pi / 5, -2 * math.pi / 5, 0, 2 * math.pi / 5)
    for node in graph:
        capacity = 1.0 + node / 16 if heterogeneous else 1.0
        graph.nodes[node].update(EPI=0.0, theta=phase[node % 5], nu_f=capacity)
    report = Network(graph).relational_sector_capture(
        RelationalExchangeModel(beta), cycles=(tuple(range(5)), tuple(range(5, 10)))
    )
    assert report.admitted
    output = relational_report_to_dict(report)
    path = tmp_path / "sector-capture.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    assert output["report_type"] == "RelationalSectorCaptureCertificate"
    lower = report.capture_barrier_bounds[0]
    assert output["report"]["capture_barrier_bounds"][0] == {
        "numerator": lower.numerator,
        "denominator": lower.denominator,
    }
    assert output["report"]["ring_windings"] == [1, 1]
    assert output["report"]["bridge_winding"] == 0
    assert len(output["report"]["edge_gap_affine"]) == graph.number_of_edges()
    assert output["report"]["positive_capacity"] is True
    assert output["report"]["unit_capacity"] is (not heterogeneous)
    assert output["report"]["unit_storage_scale"] is (beta == 1)
    assert output["report"]["field"]["model"]["storage_scale"] == beta
    assert output["report"]["field"]["capacity"] == [
        graph.nodes[node]["nu_f"] for node in graph
    ]
    output["report"]["field"]["epi"][0] = 123.0
    assert report.field.epi[0] == graph.nodes[0]["EPI"] == 0.0


def _attachment_networks():
    left = _network()
    right = Network(
        nx.relabel_nodes(left.G, {("port", 0): ("target", 1), "right": "outer"})
    )
    return left, right


def test_attachment_sdk_is_a_thin_delegate_and_requires_another_network(monkeypatch):
    from tnfr.physics import relational_observations as owner

    left, right = _attachment_networks()
    model, bridge = RelationalExchangeModel(1), (("port", 0), ("target", 1))
    calls, marker = [], object()

    def observe(left_graph, right_graph, **kwargs):
        calls.append((left_graph, right_graph, kwargs))
        return marker

    monkeypatch.setattr(owner, "observe_relational_attachment", observe)
    assert left.relational_attachment(right, model, bridge=bridge) is marker
    assert calls == [(left.G, right.G, dict(model=model, bridge=bridge))]
    with pytest.raises(TypeError, match="other must be a Network"):
        left.relational_attachment(right.G, model, bridge=bridge)
    assert len(calls) == 1


@pytest.fixture(scope="module")
def attachment_report():
    left, right = _attachment_networks()
    return left.relational_attachment(
        right,
        RelationalExchangeModel(1),
        bridge=(("port", 0), ("target", 1)),
    )


def test_attachment_export_retains_complete_fields_and_exact_detached_changes(
    attachment_report, tmp_path
):
    report = attachment_report
    output = relational_report_to_dict(report)
    path = tmp_path / "attachment.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    assert output["report_type"] == "RelationalAttachmentObservation"
    body = output["report"]
    assert body["bridge"] == [["port", 0], ["target", 1]]
    assert body["components"] == [
        relational_report_to_dict(field)["report"] for field in report.components
    ]
    assert body["joined"] == relational_report_to_dict(report.joined)["report"]
    encoded_loss = body["continuous_loss_change"]
    assert Fraction(encoded_loss["numerator"], encoded_loss["denominator"]) == (
        report.continuous_loss_change
    )
    assert body["represented_zero_supply_passive"] is True
    for name in ("form_rate_change", "phase_rate_change", "pressure_change"):
        recovered = tuple(
            Fraction(value["numerator"], value["denominator"]) for value in body[name]
        )
        assert recovered == getattr(report, name)
    encoded = body["transport_reset"]["energy_change"]
    assert Fraction(encoded["numerator"], encoded["denominator"]) == (
        report.transport_reset.energy_change
    )
    body["components"][0]["epi"][0] = 123.0
    body["joined"]["relative_resultant"][0][0] = 456.0
    body["ports"][0]["after"]["degree"] = 999
    assert report.components[0].epi[0] == 0.25
    assert report.joined.relative_resultant[0][0] != 456.0
    assert report.ports[0].after.degree != 999


def test_attachment_supply_export_preserves_exact_work_and_scope(
    attachment_report, tmp_path
):
    report = attachment_report
    tiny = Fraction(1, 2**1200)
    assessment = report.assess_supply(-tiny)
    output = relational_report_to_dict(assessment)
    assert output["schema"] == "tnfr.relational-report.v1"
    assert output["report_type"] == "RelationalAttachmentSupplyAssessment"
    body = output["report"]
    assert body["required_supply"] == {"numerator": 0, "denominator": 1}
    assert body["supplied_work"] == {"numerator": -1, "denominator": 2**1200}
    assert body["supply_margin"] == body["supplied_work"]
    assert body["represented_balance_satisfied"] is False
    assert "caller_supplied_work_not_authenticated" in body["scope"]
    assert "represented_storage_not_an_ideal_trigonometric_certificate" in body["scope"]
    path = tmp_path / "attachment_supply.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    body["supplied_work"]["numerator"] = 0
    body["scope"].clear()
    assert assessment.supplied_work == -tiny
    assert assessment.scope
    assert report.storage_change == 0


@pytest.mark.parametrize(
    "location",
    (
        "component",
        "component_edge",
        "joined",
        "joined_edge",
        "bridge",
        "port_before",
        "port_after",
        "reset",
        "cut",
        "cut_edge",
    ),
)
def test_attachment_export_checks_all_node_label_locations(attachment_report, location):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report, opaque = attachment_report, OpaqueLabel(3)
    if location == "component":
        first = report.components[0]
        report = replace(
            report,
            components=(
                replace(first, nodes=(opaque,) + first.nodes[1:]),
                report.components[1],
            ),
        )
    elif location == "component_edge":
        first = report.components[0]
        report = replace(
            report,
            components=(
                replace(first, edges=((opaque, first.nodes[1]),)),
                report.components[1],
            ),
        )
    elif location == "joined":
        report = replace(
            report,
            joined=replace(report.joined, nodes=(opaque,) + report.joined.nodes[1:]),
        )
    elif location == "joined_edge":
        report = replace(
            report,
            joined=replace(report.joined, edges=((opaque, report.joined.nodes[1]),)),
        )
    elif location == "bridge":
        report = replace(report, bridge=(opaque, report.bridge[1]))
    elif location.startswith("port_"):
        name = location.removeprefix("port_")
        port = report.ports[0]
        port = replace(port, **{name: replace(getattr(port, name), node=opaque)})
        report = replace(report, ports=(port, report.ports[1]))
    elif location == "reset":
        before = report.transport_reset.before
        report = replace(
            report,
            transport_reset=replace(
                report.transport_reset,
                before=replace(before, nodes=(opaque,) + before.nodes[1:]),
            ),
        )
    elif location == "cut":
        report = replace(report, cut=replace(report.cut, region=(opaque,)))
    else:
        report = replace(
            report, cut=replace(report.cut, cut_edges=((opaque, "outer", Fraction(1)),))
        )
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


def _relocation_network():
    left, right = _attachment_networks()
    graph = nx.compose(left.G, right.G)
    graph.add_edge(("port", 0), ("target", 1), weight=1.0)
    graph.nodes[("port", 0)]["EPI"] = 1.0
    return Network(graph)


def test_relocation_sdk_delegates_the_complete_supplied_contract(monkeypatch):
    from tnfr.physics import relational_observations as owner

    network = _relocation_network()
    model = RelationalExchangeModel(1)
    old, new = (("port", 0), ("target", 1)), ("right", "outer")
    calls, marker = [], object()

    def observe(graph, **kwargs):
        calls.append((graph, kwargs))
        return marker

    monkeypatch.setattr(owner, "observe_relational_relocation", observe)
    assert (
        network.relational_relocation(model, remove_bridge=old, add_bridge=new)
        is marker
    )
    assert calls == [(network.G, dict(model=model, remove_bridge=old, add_bridge=new))]


@pytest.fixture(scope="module")
def relocation_report():
    return _relocation_network().relational_relocation(
        RelationalExchangeModel(1),
        remove_bridge=(("port", 0), ("target", 1)),
        add_bridge=("right", "outer"),
    )


def test_relocation_export_retains_full_fields_partition_and_budget(
    relocation_report, tmp_path
):
    report = relocation_report
    output = relational_report_to_dict(report)
    assert output["report_type"] == "RelationalRelocationObservation"
    body = output["report"]
    assert body["before"] == relational_report_to_dict(report.before)["report"]
    assert body["after"] == relational_report_to_dict(report.after)["report"]
    assert body["components"] == [[["port", 0], "right"], [["target", 1], "outer"]]
    assert body["remove_bridge"] == [["port", 0], ["target", 1]]
    assert body["add_bridge"] == ["right", "outer"]
    assert body["represented_zero_supply_passive"] is True
    for name in ("storage_change", "continuous_loss_change"):
        value = body[name]
        assert Fraction(value["numerator"], value["denominator"]) == getattr(
            report, name
        )
    assert report.storage_change < 0
    assessment = report.assess_supply(report.storage_change)
    assert assessment.supply_margin == 0
    supply = relational_report_to_dict(assessment)
    assert supply["report"]["represented_balance_satisfied"] is True
    path = tmp_path / "relocation.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    body["after"]["epi"][0] = 123.0
    body["components"][0].clear()
    assert report.after.epi[0] == 1.0
    assert report.components[0] == (("port", 0), "right")


@pytest.mark.parametrize(
    "location",
    (
        "before",
        "after_edge",
        "component",
        "remove_bridge",
        "add_bridge",
        "port",
        "reset",
        "cut_before",
        "cut_after_edge",
    ),
)
def test_relocation_export_rejects_opaque_labels_in_nested_evidence(
    relocation_report, location
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report, opaque = relocation_report, OpaqueLabel(3)
    if location == "before":
        report = replace(report, before=replace(report.before, nodes=(opaque,)))
    elif location == "after_edge":
        report = replace(
            report, after=replace(report.after, edges=((opaque, "outer"),))
        )
    elif location == "component":
        report = replace(report, components=((opaque,), report.components[1]))
    elif location in ("remove_bridge", "add_bridge"):
        report = replace(report, **{location: (opaque, "outer")})
    elif location == "port":
        port = report.ports[0]
        report = replace(
            report,
            ports=(
                replace(port, after=replace(port.after, node=opaque)),
                *report.ports[1:],
            ),
        )
    elif location == "reset":
        report = replace(
            report,
            transport_reset=replace(
                report.transport_reset,
                after=replace(report.transport_reset.after, nodes=(opaque,)),
            ),
        )
    elif location == "cut_before":
        report = replace(
            report, cut_before=replace(report.cut_before, environment=(opaque,))
        )
    else:
        report = replace(
            report,
            cut_after=replace(
                report.cut_after, cut_edges=((opaque, "outer", Fraction(1)),)
            ),
        )
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)
