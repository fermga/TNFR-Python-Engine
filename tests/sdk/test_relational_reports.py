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


def test_phase_moment_sdk_export_preserves_exact_information_obstruction(tmp_path):
    from tnfr.physics.phase_response import assess_phase_moment_information

    left = ((Fraction(1), Fraction(0)),) * 3 + ((Fraction(3, 5), Fraction(4, 5)),) * 3
    right = ((Fraction(4, 5), Fraction(3, 5)),) * 5 + (
        (Fraction(4, 5), Fraction(-3, 5)),
    )
    report = assess_phase_moment_information(left, right, epsilon=1)
    direct = report.to_dict()
    evidence = relational_report_to_dict(report)
    assert direct["schema"] == "tnfr.phase-moment-information.v1"
    assert evidence["schema"] == "tnfr.relational-report.v1"
    assert evidence["report_type"] == "PhaseMomentInformationAssessment"
    assert evidence["report"] == direct["report"]
    body = evidence["report"]
    assert body["degree"] == 6
    assert body["first_resultants_equal"]
    assert body["strictly_acute"] == [True, True]
    assert body["sufficiency_obstruction"]
    assert body["cubic_source_difference_pi_numerator"] == {
        "numerator": -14,
        "denominator": 125,
    }
    assert body["cubic_storage_difference"] == {
        "numerator": -24,
        "denominator": 125,
    }
    destination = tmp_path / "phase-information.json"
    destination.write_text("previous content", encoding="utf-8")
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    body["cubic_source_difference_pi_numerator"]["numerator"] = 999
    assert report.cubic_source_difference_pi_numerator == Fraction(-14, 125)
    assert direct["report"]["cubic_source_difference_pi_numerator"]["numerator"] == -14


def test_bridge_channel_sdk_export_preserves_target_and_clock(tmp_path):
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
    from tnfr.physics.relational_sine_resonance import assess_sine_bridge_channels

    graph = _network().G
    graph.graph["GAMMA"] = {"type": "none"}
    model = RelationalExchangeModel(
        4, epi_weight=0, phase_weight=2, phase_domain="regular"
    )
    source = bound_relational_sine_exchange(graph, reference_model=model)
    report = assess_sine_bridge_channels(
        source,
        target_phase_turns=(Fraction(1, 7),) * 2,
        bridge=(("port", 0), "right"),
    )
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "SineBridgeChannelAssessment"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    # The shared model normalizes the supplied (0, 2) mix to (0, 1).
    assert body["clock_rate_pi_numerator"] == {"numerator": 1, "denominator": 1}
    assert body["target_phase_turns"] == [{"numerator": 1, "denominator": 7}] * 2
    assert body["source"]["phase"] == [{"numerator": 0, "denominator": 1}] * 2
    assert not body["channel_difference_is_positive"]
    path = tmp_path / "bridge-channels.json"
    export_to_json(evidence, path)
    assert json.loads(path.read_text(encoding="utf-8")) == evidence
    body["target_phase_turns"][0]["numerator"] = 42
    assert report.target_phase_turns == (Fraction(1, 7),) * 2


def test_bridge_memory_sdk_exports_exact_hidden_source_and_static_scope(tmp_path):
    from tnfr.physics.relational_sine_bridge_memory import assess_sine_bridge_memory
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

    graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
    graph.add_edge(0, 6)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )
    report = assess_sine_bridge_memory(
        source,
        left_cycle=tuple(range(6)),
        right_cycle=tuple(range(6, 12)),
        target_phase_turns=tuple(Fraction(i % 6, 6) for i in range(12)),
    )
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "SineBridgeMemoryAssessment"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["coordinate_memory"]["visible_indices"] == [0, 4]
    assert len(body["hidden_initial_projection_rows"]) == 6
    assert len(body["hidden_initial_projection_rows"][0]) == 24
    assert body["static_visible_generator"][0][1] == {
        "numerator": -2,
        "denominator": 13,
    }
    assert body["low_frequency_rate_matrix"][1][1] == {
        "numerator": 449,
        "denominator": 169,
    }
    assert (
        "static_generator_is_zero_frequency_resolvent_algebra_not_a_memory_integral"
        in body["scope"]
    )
    destination = tmp_path / "bridge-memory.json"
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    body["static_visible_generator"][0][1]["numerator"] = 99
    assert report.static_visible_generator[0][1] == Fraction(-2, 13)


def test_bridge_storage_family_export_retains_alternative_law_and_target(tmp_path):
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
    from tnfr.physics.relational_sine_resonance import assess_bridge_storage_family

    graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
    graph.add_edge(0, 6)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    report = assess_bridge_storage_family(
        source,
        left_cycle=tuple(range(6)),
        right_cycle=tuple(range(6, 12)),
        target_phase_turns=tuple(Fraction(i % 6, 6) for i in range(12)),
        epsilon=Fraction(4, 9),
    )
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "BridgeStorageFamilyAssessment"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["law"] == "normalized_sine_cubic_reciprocal_exchange"
    assert body["epsilon"] == {"numerator": 4, "denominator": 9}
    assert body["channel_difference"] == {"numerator": 0, "denominator": 1}
    assert body["source"]["phase"] == [{"numerator": 0, "denominator": 1}] * 12
    assert body["target_phase_turns"][1] == {"numerator": 1, "denominator": 6}
    assert len(body["full_tangent_generator"]) == 24
    assert body["local_phase_radius_turns"] == {"numerator": 1, "denominator": 24}
    destination = tmp_path / "bridge-storage.json"
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    body["epsilon"]["numerator"] = 999
    assert report.epsilon == Fraction(4, 9)


def _return_path_source():
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 10), (10, 5), (1, 6)))
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def test_return_path_storage_export_retains_implicit_geometry_and_law(tmp_path):
    from tnfr.physics.phase_cycle_geometry import assess_return_path_storage_geometry

    report = assess_return_path_storage_geometry(
        _return_path_source(),
        left_cycle=tuple(range(5)),
        right_cycle=tuple(range(5, 10)),
        mediator=10,
        epsilon=Fraction(1),
        refinements=8,
    )
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "ReturnPathStorageGeometryAssessment"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["law"] == "normalized_sine_cubic_reciprocal_exchange"
    assert body["epsilon"] == {"numerator": 1, "denominator": 1}
    assert body["root_turn_bracket"]["refinements"] == 8
    assert body["named_cycle_periods"] == [
        {"numerator": value, "denominator": 1} for value in (1, -1, 0)
    ]
    assert len(body["nodal_turn_affine_coefficients"]) == 11
    assert len(body["edge_turn_affine_coefficients"]) == 13
    assert body["source"]["phase"] == [{"numerator": 0, "denominator": 1}] * 11
    destination = tmp_path / "return-path-storage.json"
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    body["root_turn_bracket"]["lower"]["numerator"] = 999
    assert report.root_turn_bracket.lower < Fraction(1, 5)


def test_geometry_inference_export_keeps_unbounded_and_response_availability(tmp_path):
    from tnfr.physics.phase_cycle_geometry import assess_return_path_geometry_response

    report = assess_return_path_geometry_response(
        _return_path_source(),
        left_cycle=tuple(range(5)),
        right_cycle=tuple(range(5, 10)),
        mediator=10,
        special_turn_bounds=(Fraction(1, 12), Fraction(1, 5)),
        form_direction=(0,) * 10 + (1,),
        observation_origin="supplied_mathematical_interval",
    )
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "ReturnPathGeometryResponseAssessment"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["coefficient_status"] == "unbounded_above"
    assert body["coefficient_lower"] == {"numerator": 0, "denominator": 1}
    assert body["coefficient_upper"] is None
    assert body["response_acceleration_bounds"] is None
    assert body["response_status"] == "unavailable"
    destination = tmp_path / "geometry-inference.json"
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    body["coefficient_lower"]["numerator"] = 999
    assert report.coefficient_lower == 0


def test_conservative_entry_and_storage_balance_export_keep_distinct_clocks(tmp_path):
    from tnfr.physics.relational_sine_entry import (
        analyze_sine_conservative_source_geometry,
        certify_sine_conservative_winding_entry,
    )

    source = replace(_return_path_source(), epi=(0, -24, 0, 0, 0, 0, 0, 0, 0, 0, 24))
    certificate = certify_sine_conservative_winding_entry(
        source,
        cycle=(5, 6, 7, 8, 9),
        scaled_window=(Fraction(1, 4), Fraction(1, 3)),
        edge_turn_offsets=(1, 0, 0, 0, 0),
    )
    balance = source.regional_storage_balance(region=certificate.cycle)
    geometry = analyze_sine_conservative_source_geometry(
        source, receiver=certificate.cycle
    )
    for report in (certificate, balance, geometry):
        evidence = relational_report_to_dict(report)
        assert evidence["report_type"] == type(report).__name__
        assert evidence["report"] == report.to_dict()["report"]
        path = tmp_path / f"{type(report).__name__}.json"
        export_to_json(evidence, path)
        assert json.loads(path.read_text(encoding="utf-8")) == evidence
    body = relational_report_to_dict(certificate)["report"]
    assert body["clock"] == "tau=t/pi"
    assert body["acquisition_certified"] and body["certified_winding"] == -1
    assert not body["acute_acquisition_certified"]
    assert geometry.relative_source_rank == 2
    body["edge_turn_offsets"][0] = 0
    assert certificate.edge_turn_offsets[0] == 1
    ledger = relational_report_to_dict(balance)["report"]
    assert ledger["clock"] == "structural_t"
    assert ledger["regional_form_storage"] == {"numerator": 0, "denominator": 1}
    assert ledger["complement_form_storage"] == {"numerator": 864, "denominator": 1}
    assert ledger["boundary_form_storage"] == {"numerator": 576, "denominator": 1}


def test_conservative_phase_transport_sdk_preserves_clock_and_exact_contrasts(tmp_path):
    from tnfr.physics.relational_sine_entry import (
        analyze_sine_conservative_phase_transport,
    )

    source = replace(_return_path_source(), epi=(0, -24, 0, 0, 0, 0, 0, 0, 0, 0, 24))
    contrast = tuple(int(i == 9) - int(i == 7) for i in range(11))
    report = analyze_sine_conservative_phase_transport(source, contrasts=(contrast,))
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "SineConservativePhaseTransport"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["clock"] == "tau=t/pi"
    assert body["contrast_initial_velocity"] == [{"numerator": 0, "denominator": 1}]
    assert body["contrast_initial_jerk"] == [{"numerator": -80, "denominator": 3}]
    assert body["contrast_acceleration_bounds"] == [{"numerator": 3, "denominator": 1}]
    path = tmp_path / "conservative-phase-transport.json"
    export_to_json(evidence, path)
    assert json.loads(path.read_text(encoding="utf-8")) == evidence
    body["contrasts"][0][7]["numerator"] = 0
    assert report.contrasts[0][7] == -1


@pytest.fixture(scope="module")
def conservative_handoff_report():
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
    from tnfr.physics.relational_sine_entry import assess_sine_conservative_handoff

    graph = nx.cycle_graph(5)
    graph.add_edges_from((i, i + 5) for i in range(5))
    for i in graph:
        graph.nodes[i].update(EPI=0 if i < 5 else -192 * (i - 7), theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    return assess_sine_conservative_handoff(source, cycle=range(5))


def test_conservative_handoff_export_keeps_entry_and_exit_distinct(
    conservative_handoff_report, tmp_path
):
    report = conservative_handoff_report
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "SineConservativeHandoff"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["entry_subbarrier_certified"]
    assert body["forced_exit_certified"] and body["handoff_obstruction_certified"]
    assert body["omega"] == {"numerator": 64, "denominator": 1}
    assert body["scaled_exit_time"] == {"numerator": 5, "denominator": 192}
    destination = tmp_path / "conservative-handoff.json"
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    body["omega"]["numerator"] = 0
    assert report.omega == 64


@pytest.mark.parametrize("location", ("cycle", "source"))
def test_conservative_handoff_export_admits_all_labels(
    conservative_handoff_report, location
):
    opaque = object()
    report = conservative_handoff_report
    if location == "cycle":
        report = replace(report, cycle=(opaque, *report.cycle[1:]))
    else:
        report = replace(
            report,
            source=replace(report.source, nodes=(opaque, *report.source.nodes[1:])),
        )
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


@pytest.fixture(scope="module")
def phase_offset_reports():
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
    from tnfr.physics.relational_sine_partition import (
        assess_sine_phase_offset_partition,
    )

    graph = nx.cycle_graph(5)
    graph.add_edges_from((i, i + 5) for i in range(5))
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    partition = assess_sine_phase_offset_partition(
        source,
        blocks=(tuple(range(5)), tuple(range(5, 10))),
        phase_offset_turns=tuple(Fraction(i, 5) for i in range(5)) * 2,
    )
    return partition, partition.evaluate((1, 0), (0, Fraction(1, 8)))


@pytest.mark.parametrize("index", (0, 1))
def test_phase_offset_sdk_export_preserves_template_and_detached_state(
    phase_offset_reports, index, tmp_path
):
    report = phase_offset_reports[index]
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == type(report).__name__
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["clock"] == "tau=t/pi"
    family = body if index == 0 else body["partition"]
    assert family["invariance_certified"]
    assert family["source"]["phase"] == [{"numerator": 0, "denominator": 1}] * 10
    assert family["phase_offset_turns"][1] == {"numerator": 1, "denominator": 5}
    if index == 1:
        assert body["fine_form"][:5] == [{"numerator": 1, "denominator": 1}] * 5
        assert body["block_phase_rates"] == [
            {"numerator": 1, "denominator": 3},
            {"numerator": -1, "denominator": 1},
        ]
    destination = tmp_path / "phase-offset-report.json"
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    family["phase_offset_turns"][1]["numerator"] = 99
    assert phase_offset_reports[0].phase_offset_turns[1] == Fraction(1, 5)


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("location", ("source", "blocks"))
def test_phase_offset_export_validates_nested_node_labels(
    phase_offset_reports, index, location
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    partition, state = phase_offset_reports
    if location == "source":
        partition = replace(
            partition,
            source=replace(partition.source, nodes=(OpaqueLabel(0), *range(1, 10))),
        )
    else:
        partition = replace(
            partition, blocks=((OpaqueLabel(0), *range(1, 5)), tuple(range(5, 10)))
        )
    report = partition if index == 0 else replace(state, partition=partition)
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


@pytest.fixture(scope="module")
def collective_pulse_report(phase_offset_reports):
    from tnfr.physics.relational_sine_partition import observe_sine_collective_pulse

    # Install declared exact primitives explicitly: graph capture alone would
    # retain binary64 materializations of non-dyadic rational input.
    form = (
        tuple(Fraction(value, 400) for value in (139, 99, 119, 119, 119))
        + (-Fraction(357, 400),) * 5
    )
    source = replace(phase_offset_reports[0].source, epi=form)
    return observe_sine_collective_pulse(
        source, cycle=tuple(range(5)), contact_turn_offsets=(0,) * 5
    )


def test_collective_pulse_sdk_exports_signed_split_and_conditional_jet(
    collective_pulse_report, tmp_path
):
    report = collective_pulse_report
    exported = relational_report_to_dict(report)
    assert exported["report_type"] == "SineCollectivePulseBalance"
    assert exported["report"] == report.to_dict()["report"]
    body = exported["report"]
    assert body["form_gap"] == {"numerator": 119, "denominator": 100}
    assert body["flat_phase_jet_available"]
    assert body["fourth_energy_derivative"] == {
        "numerator": 14161,
        "denominator": 337500,
    }
    assert body["contact_turn_offsets"] == [0] * 5
    assert body["clock"] == "tau=t/pi"
    path = tmp_path / "collective-pulse.json"
    export_to_json(exported, path)
    assert json.loads(path.read_text(encoding="utf-8")) == exported
    body["source"]["epi"][0]["numerator"] = 999
    assert report.source.epi[0] == Fraction(139, 400)


@pytest.mark.parametrize("location", ("source", "cycle"))
def test_collective_pulse_sdk_rejects_opaque_labels(collective_pulse_report, location):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report = collective_pulse_report
    if location == "source":
        report = replace(
            report, source=replace(report.source, nodes=(OpaqueLabel(0), *range(1, 10)))
        )
    else:
        report = replace(report, cycle=(OpaqueLabel(0), *range(1, 5)))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


@pytest.fixture(scope="module")
def moving_pattern_windows(phase_offset_reports):
    from tnfr.physics.relational_sine_partition import assess_sine_moving_pattern_window

    partition = phase_offset_reports[0]
    pulse = partition.evaluate(
        (Fraction(1, 20), -Fraction(3, 20)), (-Fraction(2, 5),) * 2
    )
    stationary = partition.evaluate((0, 0), (-Fraction(2, 5),) * 2)
    common = dict(
        form_error_bounds=(Fraction(1, 1000),) * 10,
        phase_error_bounds=(Fraction(1, 1000),) * 10,
        receiver_phase_radius=Fraction(1, 10),
        contact_phase_radius=Fraction(1, 10),
        reference_contact_bound=Fraction(1, 4),
    )
    return {
        "pulse": assess_sine_moving_pattern_window(pulse, scaled_horizon=5, **common),
        "stationary": assess_sine_moving_pattern_window(
            stationary, scaled_horizon=5, **common
        ),
        "unavailable": assess_sine_moving_pattern_window(
            pulse, scaled_horizon=10**6, **common
        ),
    }


@pytest.mark.parametrize("kind", ("pulse", "stationary", "unavailable"))
def test_moving_pattern_window_sdk_preserves_distinct_retention_and_entry(
    moving_pattern_windows, kind, tmp_path
):
    report = moving_pattern_windows[kind]
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "SineMovingPatternWindow"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["clock"] == "tau=t/pi"
    assert body["whole_window_retention_certified"] == (kind != "unavailable")
    assert body["phase_flat_acquisition_status"] == (
        "excluded" if kind == "stationary" else "not_excluded"
    )
    if kind == "unavailable":
        assert body["receiver_storage_excess_upper_bound"] is None
    else:
        assert body["scaled_horizon"] == {"numerator": 5, "denominator": 1}
    destination = tmp_path / f"moving-pattern-{kind}.json"
    export_to_json(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    body["form_error_bounds"][0]["numerator"] = 999
    assert report.form_error_bounds[0] == Fraction(1, 1000)


@pytest.mark.parametrize("location", ("source", "blocks"))
def test_moving_pattern_window_export_admits_nested_labels(
    moving_pattern_windows, location
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report = moving_pattern_windows["pulse"]
    partition = report.reference.partition
    if location == "source":
        partition = replace(
            partition,
            source=replace(partition.source, nodes=(OpaqueLabel(0), *range(1, 10))),
        )
    else:
        partition = replace(
            partition, blocks=((OpaqueLabel(0), *range(1, 5)), tuple(range(5, 10)))
        )
    altered = replace(report, reference=replace(report.reference, partition=partition))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(altered)


def test_regional_organization_sdk_retains_full_forecast_and_distinct_outcome(tmp_path):
    from tnfr.physics.relational_sine_forecast import bound_sine_flow
    from tnfr.physics.relational_sine_regional import (
        assess_sine_regional_channels,
        assess_sine_regional_organization,
    )

    forecast = bound_sine_flow(
        (0,) * 10 + (1,),
        neighbors=tuple(((i - 1) % 5, (i + 1) % 5) for i in range(5)),
        visible_capacity=(1,) * 4,
        model=RelationalExchangeModel(1, epi_weight=0, phase_domain="regular"),
        observation_time=0,
        end_time=Fraction(1, 16),
        time_step=Fraction(1, 16),
        order=2,
    )
    report = assess_sine_regional_organization(
        forecast,
        cycle_indices=(0, 1, 2, 3, 4),
        minimum_duration=Fraction(1, 32),
        acute_margin=Fraction(1, 16),
    )
    evidence = relational_report_to_dict(report)
    assert evidence["report_type"] == "SineRegionalOrganization"
    assert evidence["report"] == report.to_dict()["report"]
    body = evidence["report"]
    assert body["clock"] == "structural_t"
    assert body["horizon_complete"]
    assert body["outcome"] == "acute_winding_excluded_on_horizon"
    assert len(body["forecast"]["initial_box"]) == 11
    assert body["forecast"]["validated_end_time"] == {"numerator": 1, "denominator": 16}
    path = tmp_path / "regional-organization.json"
    export_to_json(evidence, path)
    assert json.loads(path.read_text(encoding="utf-8")) == evidence
    body["cycle_indices"][0] = 4
    assert report.cycle_indices == (0, 1, 2, 3, 4)
    channels = assess_sine_regional_channels(forecast, cycle_indices=(0, 1, 2, 3, 4))
    projected = relational_report_to_dict(channels)
    assert projected["report_type"] == "SineRegionalChannelHistory"
    assert projected["report"] == channels.to_dict()["report"]
    assert projected["report"]["acute_accessibility_barrier"] == {
        "numerator": 7,
        "denominator": 2,
    }
    assert projected["report"]["initial_phase_flat_certified"]
    path = tmp_path / "regional-channels.json"
    export_to_json(projected, path)
    assert json.loads(path.read_text(encoding="utf-8")) == projected


def test_pair_emission_sdk_delegates_explicit_action_and_lifts(monkeypatch):
    from tnfr.physics import relational_sine_scale as owner

    network, model = _network(), RelationalExchangeModel(1)
    pairs = ((("port", 0), "right"),)
    turns = (2, 2)
    result = object()

    def assess(graph, **kwargs):
        assert graph is network.G
        assert kwargs == {
            "reference_model": model,
            "pairs": pairs,
            "pair_index": 0,
            "boost": Fraction(1, 8),
            "phase_turns": turns,
        }
        return result

    monkeypatch.setattr(owner, "assess_sine_pair_emission", assess)
    assert (
        network.relational_sine_pair_emission(
            model, pairs=pairs, pair_index=0, boost=Fraction(1, 8), phase_turns=turns
        )
        is result
    )


def test_pair_emission_exact_export_keeps_target_distinction(tmp_path):
    graph = nx.complete_bipartite_graph(2, 2)
    for node, form in zip(graph, (0.25, 0.75, 0.0, 0.0), strict=True):
        graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0)
    graph.graph["GAMMA"] = {"type": "none"}
    report = Network(graph).relational_sine_pair_emission(
        RelationalExchangeModel(1, epi_weight=0, phase_domain="regular"),
        pairs=((0, 1), (2, 3)),
        pair_index=0,
        boost=0.125,
    )
    evidence = relational_report_to_dict(report)
    path = tmp_path / "pair-emission.json"
    export_to_json(evidence, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema"] == "tnfr.relational-report.v1"
    assert saved["report_type"] == "SinePairEmissionAssessment"
    body = saved["report"]
    assert not body["single_member_source_orbit_equal"]
    assert body["whole_pair_structural_descent_certified"]
    assert not body["runtime_admission_certified"]
    assert not body["event_occurrence_derived"]
    for field, numerator in (("first_member", 9), ("second_member", 25)):
        assert body[field]["internal_form_squared"] == {
            "numerator": numerator,
            "denominator": 256,
        }
    assert body["whole_pair"]["form_mean"] == {"numerator": 5, "denominator": 8}
    body["first_member"]["form_pair"][0]["numerator"] = 999
    assert report.first_member.form_pair[0] == Fraction(3, 8)

    increment = report.form_increment(outcome="whole_pair")
    assert increment.comparison is report.comparison
    increment_data = relational_report_to_dict(increment)
    assert increment_data["report_type"] == "SineFormIncrementAssessment"
    assert increment_data["report"]["weighted_form_change"] == {
        "numerator": 1,
        "denominator": 2,
    }
    assert increment_data["report"]["closed_flow_endpoint_obstructed"]
    assert not increment_data["report"]["endpoint_reachability_certified"]
    transfer = report.comparison.regional_transfer(region=(0, 1))
    transfer_data = relational_report_to_dict(transfer)
    assert transfer_data["report_type"] == "SineRegionalTransfer"
    assert transfer_data["report"]["region_indices"] == [0, 1]
    assert transfer_data["report"]["regional_weighted_form"] == {
        "numerator": 2,
        "denominator": 1,
    }
    path = tmp_path / "form-balance.json"
    export_to_json({"transfer": transfer_data, "increment": increment_data}, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"transfer": transfer_data, "increment": increment_data}


@pytest.mark.parametrize(
    "kind",
    (
        "identity",
        "replica",
        "replica_capacity",
        "replica_equilibria",
        "phase_pairs",
        "joint_pairs",
        "joint_pairing_projection",
        "joint_pairing_window",
        "state_pairing",
        "pairing_transition",
        "pairing_window",
        "pair_support_symmetry",
        "mixed_pair_state",
        "pair_emission",
        "regional_transfer",
        "form_increment",
        "mobility_comparison",
        "mobility_relative_balance",
        "mobility_geometry",
        "pairing_mobility",
        "replica_persistence",
        "replica_pulse",
        "replica_variation",
        "replica_splitting",
    ),
)
def test_extended_sine_export_delegates_owner_verdicts(monkeypatch, kind):
    from tnfr.physics.relational_sine_comparison import (
        SineFormIncrementAssessment,
        SineMobilityComparison,
        SineMobilityRelativeBalance,
        SineRegionalTransfer,
    )
    from tnfr.physics.relational_sine_recovery import SineCycleIdentityAssessment
    from tnfr.physics.relational_sine_scale import (
        JointPairObservation,
        PhasePairObservation,
        SineJointPairingProjection,
        SineJointPairingWindowAssessment,
        SineMixedPairStateAssessment,
        SineMobilityGeometryAssessment,
        SinePairEmissionAssessment,
        SinePairingMobilityAssessment,
        SinePairingTransitionAssessment,
        SinePairingWindowAssessment,
        SinePairSupportSymmetryAssessment,
        SineReplicaCapacityAssessment,
        SineReplicaEquilibriaAssessment,
        SineReplicaPersistenceAssessment,
        SineReplicaPulseAssessment,
        SineReplicaPulseSplitting,
        SineReplicaPulseVariation,
        SineReplicaScaleAssessment,
        SineStatePairingAssessment,
    )

    if kind == "identity":
        report_type = SineCycleIdentityAssessment
        schema = "tnfr.relational-sine-cycle-identity.v1"
        body = {
            "family_almost_everywhere_recurrence_certified": True,
            "source_set_trapping_certified": False,
            "relative_source_family_membership": "unresolved",
            "individual_recurrence_status": "unavailable_for_chosen_state",
        }
    elif kind == "replica":
        report_type = SineReplicaScaleAssessment
        schema = "tnfr.relational-sine-replica-scale.v1"
        body = {
            "synchronized_submanifold_invariant": True,
            "source_in_synchronized_submanifold": False,
            "same_law_reduced_flow_certified_for_source": False,
            "all_state_coarse_closure_obstructed": True,
            "unordered_pair_state_closure_certified": True,
            "unordered_pair_state_identifies_swap_orbits": True,
        }
    elif kind == "replica_capacity":
        report_type = SineReplicaCapacityAssessment
        schema = "tnfr.relational-sine-replica-capacity.v1"
        body = {
            "family": {
                "family_admitted": False,
                "source_set_trapping_certified": False,
                "individual_recurrence_status": "unavailable_for_chosen_state",
            },
            "all_time_internal_activity_status": "not_certified_by_this_reader",
        }
    elif kind == "replica_equilibria":
        report_type = SineReplicaEquilibriaAssessment
        schema = "tnfr.relational-sine-replica-equilibria.v1"
        body = {
            "source_equilibrium_status": "unavailable",
            "source_equilibrium_winding": None,
            "acute_equilibrium_classification_certified": True,
        }
    elif kind == "phase_pairs":
        report_type = PhasePairObservation
        schema = "tnfr.phase-pairs.v1"
        body = {"candidate_pairs": None, "status": "unavailable"}
    elif kind == "joint_pairs":
        report_type = JointPairObservation
        schema = "tnfr.joint-pairs.v1"
        body = {"candidate_pairs": None, "status": "unavailable"}
    elif kind == "joint_pairing_projection":
        report_type = SineJointPairingProjection
        schema = "tnfr.relational-sine-joint-pairing-projection.v1"
        body = {
            "candidate_pairs": None,
            "status": "unavailable",
            "same_pairing_as_initial_box": None,
        }
    elif kind == "joint_pairing_window":
        report_type = SineJointPairingWindowAssessment
        schema = "tnfr.relational-sine-joint-pairing-window.v1"
        body = {
            "reference_identity_certified": True,
            "candidate_pairs": None,
            "status": "unavailable",
        }
    elif kind == "state_pairing":
        report_type = SineStatePairingAssessment
        schema = "tnfr.relational-sine-state-pairing.v1"
        body = {
            "capacity_admission_status": "rejected",
            "capacity": None,
            "all_time_pairing_persistence_certified": False,
        }
    elif kind == "pairing_transition":
        report_type = SinePairingTransitionAssessment
        schema = "tnfr.relational-sine-pairing-transition.v1"
        body = {
            "forward": {
                "status": "certified_matching",
                "local_interval_existence_certified": True,
                "certified_time_horizon": None,
                "support_admission_status": "rejected",
            },
            "backward": {"status": "unavailable"},
        }
    elif kind == "pairing_window":
        report_type = SinePairingWindowAssessment
        schema = "tnfr.relational-sine-pairing-window.v1"
        body = {
            "status": "unavailable",
            "candidate_pairs": None,
            "support_admission_status": "not_attempted",
        }
    elif kind == "pair_support_symmetry":
        report_type = SinePairSupportSymmetryAssessment
        schema = "tnfr.relational-sine-pair-support-symmetry.v1"
        body = {
            "independent_pair_swaps_equivariant": True,
            "unordered_pair_quotient_status": "certified_by_independent_swap_symmetry",
            "strict_replica_admission_status": "rejected",
            "witness": None,
        }
    elif kind == "mixed_pair_state":
        report_type = SineMixedPairStateAssessment
        schema = "tnfr.relational-sine-mixed-pair-state.v1"
        body = {
            "coordinates": ({"mode": "ordered"}, {"mode": "unordered"}),
            "support_symmetry": {"strict_replica_admission_status": "rejected"},
        }
    elif kind == "pair_emission":
        report_type = SinePairEmissionAssessment
        schema = "tnfr.relational-sine-pair-emission.v1"
        body = {
            "single_member_source_orbit_equal": False,
            "whole_pair_structural_descent_certified": True,
            "runtime_admission_certified": False,
            "event_occurrence_derived": False,
        }
    elif kind == "regional_transfer":
        report_type = SineRegionalTransfer
        schema = "tnfr.relational-sine-regional-transfer.v1"
        body = {
            "global_weighted_form_conserved": True,
            "global_weighted_form_rate_bounds": {
                "lo": {"numerator": -1, "denominator": 100},
                "hi": {"numerator": 1, "denominator": 100},
            },
        }
    elif kind == "form_increment":
        report_type = SineFormIncrementAssessment
        schema = "tnfr.relational-sine-form-increment.v1"
        body = {
            "closed_flow_endpoint_obstructed": False,
            "endpoint_reachability_certified": False,
        }
    elif kind == "mobility_comparison":
        report_type = SineMobilityComparison
        schema = "tnfr.relational-sine-mobility-comparison.v1"
        body = {
            "law": "current_squared_reciprocal_mobility",
            "epsilon": {"numerator": 1, "denominator": 1},
        }
    elif kind == "mobility_relative_balance":
        report_type = SineMobilityRelativeBalance
        schema = "tnfr.relational-sine-mobility-relative-balance.v1"
        body = {
            "relative_field_closure_certified": True,
            "constant_mobility_mean_and_volume_identities_certified": False,
        }
    elif kind == "mobility_geometry":
        report_type = SineMobilityGeometryAssessment
        schema = "tnfr.relational-sine-mobility-geometry.v1"
        body = {
            "source_set_trapping_certified": True,
            "relative_family_recurrence_status": "unavailable_invariant_measure_unproved",
            "full_state_recurrence_status": "not_assessed_removed_origins",
        }
    elif kind == "pairing_mobility":
        report_type = SinePairingMobilityAssessment
        schema = "tnfr.relational-sine-pairing-mobility.v1"
        body = {"certified_time_horizon": None}
    elif kind == "replica_persistence":
        report_type = SineReplicaPersistenceAssessment
        schema = "tnfr.relational-sine-replica-persistence.v1"
        body = {
            "family_admitted": True,
            "source_set_trapping_certified": False,
            "individual_recurrence_status": "unavailable_for_chosen_state",
        }
    elif kind == "replica_pulse":
        report_type = SineReplicaPulseAssessment
        schema = "tnfr.relational-sine-replica-pulse.v1"
        body = {
            "nonlinear_periodic_exchange_certified": True,
            "graph_membership_certified": False,
            "all_fine_edges_acute_status": "excluded",
        }
    elif kind == "replica_variation":
        report_type = SineReplicaPulseVariation
        schema = "tnfr.relational-sine-replica-pulse-variation.v1"
        body = {
            "variation_identity_certified": True,
            "periodic_reference_certified": True,
            "orbital_stability_status": "not_assessed",
        }
    else:
        report_type = SineReplicaPulseSplitting
        schema = "tnfr.relational-sine-replica-pulse-splitting.v1"
        body = {
            "sufficiently_small_nonlinear_orbital_instability_certified": True,
            "amplitude_upper_bound": None,
            "finite_preparation_assessed": False,
            "return_multipliers_computed": False,
        }
    report = object.__new__(report_type)
    calls = []

    def project(self):
        calls.append(self)
        return {"schema": schema, "report": body}

    monkeypatch.setattr(report_type, "to_dict", project)
    result = relational_report_to_dict(report)
    assert calls == [report]
    assert result == {
        "schema": "tnfr.relational-report.v1",
        "report_type": report_type.__name__,
        "report": body,
    }


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


def test_seeded_formation_obstruction_export_preserves_scope_and_exact_deficits():
    from tnfr.physics.relational_capture import (
        certify_relational_seeded_formation_obstruction,
    )

    report = certify_relational_seeded_formation_obstruction()
    data = relational_report_to_dict(report)
    assert data["report_type"] == "RelationalSeededFormationObstruction"
    body = data["report"]
    assert body["status"] == "obstructed" and body["obstruction_certified"]
    assert "admitted" not in body
    for case, original in zip(body["cases"], report.cases):
        assert case["matching_ports"] == [
            list(pair) for pair in original.matching_ports
        ]
        deficit = case["additional_storage_gap_lower_bound"]
        assert Fraction(deficit["numerator"], deficit["denominator"]) == (
            original.additional_storage_gap_lower_bound
        )
        assert original.additional_storage_gap_lower_bound > 0
    assert json.loads(json.dumps(data)) == data
    body["cases"][0]["additional_storage_gap_lower_bound"]["numerator"] = 0
    assert report.cases[0].additional_storage_gap_lower_bound > 0


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


def _replace_report_path(report, path, value):
    """Alter one retained record without changing any other captured evidence."""
    if not path:
        return value
    key, *remaining = path
    if isinstance(key, int):
        return tuple(
            _replace_report_path(item, remaining, value) if index == key else item
            for index, item in enumerate(report)
        )
    return replace(
        report, **{key: _replace_report_path(getattr(report, key), remaining, value)}
    )


@pytest.fixture(scope="module")
def native_label_reports():
    from tnfr.dynamics.relational import RelationalExchangeStep

    network, model = _network(), RelationalExchangeModel(1)
    for node in network.G:
        network.G.nodes[node]["EPI"] = 0
    field = network.relational_exchange(model)
    # A constructed equilibrium step exercises the report format without evolution.
    step = RelationalExchangeStep(
        before=field,
        after=field,
        dt=1.0,
        t_before=0.0,
        t_after=1.0,
        epi_update_defect=(Fraction(0),) * 2,
        phase_update_defect=(Fraction(0),) * 2,
        clock_defect=Fraction(0),
        energy_change=Fraction(0),
        energy_step_defect=Fraction(0),
    )
    return {
        "field": field,
        "step": step,
        "uniform_tangent": network.relational_uniform_tangent(model),
        "consensus_tangent": network.relational_consensus_tangent(model),
        "pattern": network.relational_pattern(
            model,
            reference_phase=dict.fromkeys(network.G, 0),
            regions=(("right",),),
        ),
    }


@pytest.mark.parametrize(
    "kind,path",
    (
        ("field", ("edges",)),
        ("step", ("before", "edges")),
        ("step", ("after", "edges")),
        ("uniform_tangent", ("field", "edges")),
        ("consensus_tangent", ("field", "edges")),
        ("pattern", ("field", "edges")),
    ),
)
def test_native_report_export_rejects_dataclass_edge_labels(
    native_label_reports, kind, path
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report = native_label_reports[kind]
    assert json.loads(json.dumps(relational_report_to_dict(report)))
    altered = _replace_report_path(report, path, ((("nested", OpaqueLabel(3)), 1),))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(altered)


@pytest.mark.parametrize(
    "path",
    (
        ("nodes",),
        ("transport", "source", "nodes"),
        ("transport", "region"),
        ("transport", "environment"),
        ("boundary", "cut", "nodes"),
        ("boundary", "cut", "region"),
        ("boundary", "cut", "environment"),
    ),
)
def test_pattern_export_rejects_dataclass_labels_in_nested_regional_evidence(
    native_label_reports, path
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report = native_label_reports["pattern"]
    altered = _replace_report_path(report, ("regions", 0, *path), (OpaqueLabel(3),))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(altered)


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
        ("relational_consensus_capture", "certify_relational_consensus_capture", {}),
        (
            "relational_consensus_formation_obstruction",
            "certify_relational_consensus_formation_obstruction",
            {},
        ),
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


def test_sector_geometry_sdk_delegates_without_law_admission(monkeypatch):
    from tnfr.physics import relational_capture as owner

    network = _network()
    cycles, calls, marker = (("supplied",),), [], object()

    def observe(graph, **kwargs):
        calls.append((graph, kwargs))
        return marker

    monkeypatch.setattr(owner, "observe_relational_sector_geometry", observe)
    assert (
        network.relational_sector_geometry(
            storage_scale=2, cycles=cycles, target_sector=-1
        )
        is marker
    )
    assert calls == [
        (network.G, dict(storage_scale=2, cycles=cycles, target_sector=-1))
    ]


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


@pytest.mark.parametrize("location", ("field", "field_edge", "cycles", "winding"))
@pytest.mark.parametrize("report_kind", ("transit", "consensus"))
def test_capture_wrapper_export_checks_nested_initial_labels(
    monkeypatch, location, report_kind
):
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
    elif location == "field_edge":
        initial = replace(initial, field=replace(initial.field, edges=((opaque, 1),)))
    elif location == "cycles":
        initial = replace(initial, cycles=((opaque,), initial.cycles[1]))
    else:
        winding = replace(initial.winding[0], cycle_nodes=(opaque,))
        initial = replace(initial, winding=(winding, initial.winding[1]))
    with pytest.raises(TypeError, match="node labels"):
        report = (
            owner.RelationalTransitCertificate(initial, Fraction(0))
            if report_kind == "transit"
            else replace(_consensus_capture(), initial=initial)
        )
        relational_report_to_dict(report)


def _consensus_capture(*, phase_perturbation=0):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    form = (1, -1, Fraction(-3, 2), 0, Fraction(3, 2))
    for node in graph:
        graph.nodes[node].update(EPI=form[node % 5], theta=0, nu_f=1)
    graph.nodes[0]["theta"] = phase_perturbation
    return Network(graph).relational_consensus_capture(
        RelationalExchangeModel(1, epi_weight=1, phase_weight=1),
        cycles=(tuple(range(5)), tuple(range(5, 10))),
    )


@pytest.mark.parametrize("phase_perturbation", (0, Fraction(1, 1024)))
def test_consensus_capture_export_preserves_initial_and_analytic_availability(
    tmp_path, phase_perturbation
):
    report = _consensus_capture(phase_perturbation=phase_perturbation)
    output = relational_report_to_dict(report)
    path = tmp_path / "consensus-capture.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    assert output["report_type"] == "RelationalConsensusCaptureCertificate"
    data = output["report"]
    assert data["initial"] == relational_report_to_dict(report.initial)["report"]
    assert not report.initial.energy_admitted
    if phase_perturbation:
        assert data["status"] == "unavailable"
        assert data["target_sector"] is None
        assert data["endpoint_storage_upper_bound"] is None
    else:
        assert data["status"] == "admitted"
        assert data["target_sector"] == 0
        assert data["initial_form_storage"] == {"numerator": 9, "denominator": 1}
        assert data["endpoint_storage_upper_bound"] == {
            "numerator": 6,
            "denominator": 1,
        }
    data["initial"]["field"]["epi"][0] = 999
    assert report.initial.field.epi[0] == 1


def _full_form_consensus_obstruction(*, phase=0):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    for node in graph:
        graph.nodes[node].update(EPI=Fraction(node, 16), theta=0, nu_f=1)
    graph.nodes[0]["theta"] = phase
    return Network(graph).relational_consensus_formation_obstruction(
        RelationalExchangeModel(1),
        cycles=(tuple(range(5)), tuple(range(5, 10))),
    )


@pytest.mark.parametrize("phase", (0, Fraction(1, 1024)))
def test_full_form_obstruction_export_keeps_exclusion_and_continuation_separate(
    tmp_path, phase
):
    report = _full_form_consensus_obstruction(phase=phase)
    output = relational_report_to_dict(report)
    path = tmp_path / "formation-obstruction.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    assert output["report_type"] == "RelationalConsensusFormationObstruction"
    data = output["report"]
    assert data["field"] == relational_report_to_dict(report.field)["report"]
    assert data["continuation_status"] == "not_certified"
    if phase:
        assert data["status"] == "unavailable"
        assert data["all_regular_time_phase_storage_upper_bound"] is None
        assert data["excluded_target_sectors"] is None
    else:
        assert data["status"] == "admitted"
        assert data["excluded_target_sectors"] == [-1, 1]
        upper = data["all_regular_time_phase_storage_upper_bound"]
        assert Fraction(upper["numerator"], upper["denominator"]) == (
            Fraction(7, 10) * report.initial_form_storage
        )
    data["cycles"][0].clear()
    data["field"]["epi"][0] = 999
    assert report.cycles[0] == tuple(range(5))
    assert report.field.epi[0] == 0


@pytest.mark.parametrize("location", ("node", "edge", "cycle"))
def test_full_form_obstruction_export_rejects_opaque_labels(location):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report, opaque = _full_form_consensus_obstruction(), OpaqueLabel(3)
    if location == "node":
        report = replace(report, field=replace(report.field, nodes=(opaque,)))
    elif location == "edge":
        report = replace(report, field=replace(report.field, edges=((opaque, 1),)))
    else:
        report = replace(report, cycles=((opaque,), report.cycles[1]))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


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

    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    for path, labels in (
        (("cycles",), ((OpaqueLabel(3),), report.cycles[1])),
        (("bridge_cycle",), (OpaqueLabel(3),)),
        (("field", "edges"), ((OpaqueLabel(3), 1),)),
    ):
        with pytest.raises(TypeError, match="node labels"):
            relational_report_to_dict(_replace_report_path(report, path, labels))


@pytest.fixture(scope="module")
def sector_geometry_report():
    import math

    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    # No capacity or evolution law is needed for the geometric observation.
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=2 * math.pi * (node % 5) / 5)
    return Network(graph).relational_sector_geometry(
        storage_scale=2, cycles=(tuple(range(5)), tuple(range(5, 10)))
    )


def test_sector_geometry_exports_exact_sublevel_without_law_claims(
    sector_geometry_report, tmp_path
):
    report = sector_geometry_report
    assert report.admitted
    output = relational_report_to_dict(report)
    path = tmp_path / "sector-geometry.json"
    export_to_json(output, path)
    assert json.loads(path.read_text(encoding="utf-8")) == output
    assert output["report_type"] == "RelationalSectorGeometry"
    data = output["report"]
    assert data["ring_windings"] == [1, 1]
    assert data["bridge_winding"] == 0
    lower = report.sublevel_acute_margin_lower_bound
    assert data["sublevel_acute_margin_lower_bound"] == {
        "numerator": lower.numerator,
        "denominator": lower.denominator,
    }
    assert all(
        key not in data for key in ("field", "model", "capacity", "target_sector")
    )
    assert not any(key.startswith("future_") for key in data)
    data["epi"][0] = 123.0
    assert report.epi[0] == 0.0


@pytest.mark.parametrize("label_location", ("nodes", "edges", "cycles", "bridge_cycle"))
def test_sector_geometry_export_rejects_opaque_labels_everywhere(
    sector_geometry_report, label_location
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    label = OpaqueLabel(7)
    labels = (label,) if label_location in ("nodes", "bridge_cycle") else ((label,),)
    report = replace(sector_geometry_report, **{label_location: labels})
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


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


@pytest.fixture(scope="module")
def analytic_sine_reports():
    """Zero-horizon consensus controls exercise projection, not formation runs."""
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
    from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry
    from tnfr.physics.relational_sine_reduction import (
        bound_sine_slow_phase,
        certify_sine_slow_capture,
    )

    labels = (("node", 0), "node-1", "node-2", "node-3", "node-4")
    graph = nx.relabel_nodes(nx.cycle_graph(5), dict(enumerate(labels)))
    for node in graph:
        graph.nodes[node].update(EPI=Fraction(1, 8), theta=Fraction(1, 4), nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph, reference_model=RelationalExchangeModel(1, phase_domain="regular")
    )
    return {
        "prepared_entry": certify_sine_prepared_entry(
            source, scaled_time=0, edge_turn_offsets=(0,) * 5
        ),
        "slow_phase": bound_sine_slow_phase(source, slow_time=0),
        "slow_capture": certify_sine_slow_capture(
            source, slow_time=0, target_phase_turns=(0,) * 5
        ),
    }


@pytest.mark.parametrize("kind", ("prepared_entry", "slow_phase", "slow_capture"))
def test_analytic_sine_reports_delegate_exact_zero_horizon_evidence(
    analytic_sine_reports, kind, tmp_path
):
    report = analytic_sine_reports[kind]
    generic = relational_report_to_dict(report)
    direct = report.to_dict()
    assert generic["schema"] == "tnfr.relational-report.v1"
    assert (
        generic["report_type"]
        == {
            "prepared_entry": "SinePreparedEntry",
            "slow_phase": "SineSlowPhaseBound",
            "slow_capture": "SineSlowCapture",
        }[kind]
    )
    assert generic["report"] == direct["report"]
    body = generic["report"]
    assert body["source"]["nodes"] == [
        ["node", 0],
        "node-1",
        "node-2",
        "node-3",
        "node-4",
    ]
    assert body["source"]["epi"] == [{"numerator": 1, "denominator": 8}] * 5
    assert body["source"]["phase"] == [{"numerator": 1, "denominator": 4}] * 5
    zero = {"numerator": 0, "denominator": 1}
    if kind == "prepared_entry":
        assert body["scaled_time"] == body["horizon"] == zero
        assert body["initial_form_storage"] == zero
        assert body["initial_cycle_periods"] == [0]
        assert body["capture"]["cycle_periods"] == [0]
        assert body["capture"]["status"] == "admitted"
        assert body["status"] == "unavailable"  # Capture is not winding acquisition.
    elif kind == "slow_phase":
        assert body["slow_time"] == zero
        assert body["horizon_bounds"] == {"lo": zero, "hi": zero}
        assert body["composite_phase_error_upper_bound"] == zero
        assert "admitted" not in body  # An error bound has no accuracy verdict.
    else:
        assert body["actual_phase_distance_upper_bound"] == zero
        assert body["capture"]["status"] == "admitted"
        assert body["capture"]["observation_time"] is None
        assert body["slow_phase"]["horizon_bounds"] == {"lo": zero, "hi": zero}
    path = tmp_path / f"{kind}.json"
    export_to_json(generic, path)
    assert json.loads(path.read_text(encoding="utf-8")) == generic
    body["source"]["epi"][0]["numerator"] = 999
    assert report.source.epi[0] == Fraction(1, 8)


@pytest.mark.parametrize("kind", ("prepared_entry", "slow_phase", "slow_capture"))
def test_analytic_sine_delegation_retains_owner_node_label_rejection(
    analytic_sine_reports, kind
):
    report = analytic_sine_reports[kind]
    source = replace(report.source, nodes=(object(), *report.source.nodes[1:]))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(replace(report, source=source))


@pytest.fixture(scope="module")
def sine_pattern_composition_reports():
    """Two static P2 sources exercise composition export without a flow."""
    from tnfr.physics.relational_sine_composition import assess_sine_pattern_composition
    from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

    model = RelationalExchangeModel(1, phase_domain="regular")
    sources = []
    for labels, capacities in (
        ((("left", 0), "left-1"), (1, 2)),
        ((("right", 0), "right-1"), (3, 4)),
    ):
        graph = nx.path_graph(labels)
        for node, capacity in zip(graph, capacities, strict=True):
            graph.nodes[node].update(EPI=0, theta=0, nu_f=capacity)
        graph.graph["GAMMA"] = {"type": "none"}
        sources.append(
            bound_relational_sine_pattern(
                graph,
                reference_node=labels[0],
                reference_model=model,
                form_error_bounds=(0, 0),
                phase_error_bounds=(0, 0),
            )
        )
    return {
        kind: assess_sine_pattern_composition(
            *sources,
            bridge=("left-1", ("right", 0)),
            observation_time=Fraction(5, 3),
            edge_turn_offsets=(0, 0, 0),
            form_origin_difference=Fraction(1, 3),
            phase_origin_difference=phase,
            work_allowance=0,
        )
        for kind, phase in (("available", Fraction(1, 7)), ("unavailable", None))
    }


@pytest.mark.parametrize("kind", ("available", "unavailable"))
def test_sine_pattern_composition_sdk_export_retains_exact_frames_and_availability(
    sine_pattern_composition_reports, kind, tmp_path
):
    report = sine_pattern_composition_reports[kind]
    generic, direct = relational_report_to_dict(report), report.to_dict()
    assert generic["schema"] == "tnfr.relational-report.v1"
    assert generic["report_type"] == "SinePatternComposition"
    assert direct["schema"] == "tnfr.relational-sine-pattern-composition.v1"
    assert generic["report"] == direct["report"]
    body = generic["report"]
    assert body["status"] == kind
    assert body["nodes"] == [["left", 0], "left-1", ["right", 0], "right-1"]
    assert body["bridge"] == ["left-1", ["right", 0]]
    assert body["observation_time"] == {"numerator": 5, "denominator": 3}
    assert body["form_origin_difference"] == {"numerator": 1, "denominator": 3}
    for field, ideal in (
        ("bridge_form_storage_bounds", Fraction(1, 18)),
        ("representative_weighted_form_mean_bounds", Fraction(11, 105)),
    ):
        encoded = body[field]
        decoded = tuple(
            Fraction(encoded[end]["numerator"], encoded[end]["denominator"])
            for end in ("lo", "hi")
        )
        bounds = getattr(report, field)
        assert decoded == (bounds.lo, bounds.hi)
        assert decoded[0] <= ideal <= decoded[1]
    if kind == "available":
        assert body["phase_origin_difference"] == {"numerator": 1, "denominator": 7}
        assert (
            body["joined"]["nominal_form"][2:]
            == [{"numerator": 1, "denominator": 3}] * 2
        )
        assert body["capture"]["status"] == "admitted"
        assert body["capture"]["weighted_form_mean"] is None
        assert body["budget_status"] == "exceeds_allowance"
    else:
        assert body["phase_origin_difference"] is None
        assert body["joined"] is body["capture"] is None
        assert body["bridge_storage_bounds"] is body["joined_storage_bounds"] is None
        assert body["unavailable_reasons"] == ["phase_origin_difference_not_supplied"]
        assert body["budget_status"] == "unresolved"
    path = tmp_path / f"sine-composition-{kind}.json"
    export_to_json(generic, path)
    assert json.loads(path.read_text(encoding="utf-8")) == generic
    body["left"]["nominal_form"][0]["numerator"] = 999
    assert report.left.nominal_form[0] == Fraction(0)


@pytest.mark.parametrize("location", ("nodes", "bridge", "left", "joined"))
def test_sine_composition_export_rejects_opaque_labels(
    sine_pattern_composition_reports, location
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report = sine_pattern_composition_reports["available"]
    opaque = OpaqueLabel(7)
    if location in ("nodes", "bridge"):
        labels = getattr(report, location)
        altered = replace(report, **{location: (opaque, *labels[1:])})
    else:
        nested = getattr(report, location)
        altered = replace(
            report,
            **{location: replace(nested, nodes=(opaque, *nested.nodes[1:]))},
        )
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(altered)


@pytest.mark.parametrize(
    "path", (("nodes",), ("edges",), ("geometry", "nodes"), ("source", "nodes"))
)
def test_sine_composition_export_validates_nested_capture_labels(
    sine_pattern_composition_reports, path
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    report = sine_pattern_composition_reports["available"]
    opaque = OpaqueLabel(7)
    labels = ((opaque, "left-1"),) if path == ("edges",) else (opaque,)
    capture = _replace_report_path(report.capture, path, labels)
    altered = replace(report, capture=capture)
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(altered)
    with pytest.raises(TypeError, match="node labels"):
        capture.to_dict()


@pytest.fixture(scope="module")
def prepared_composition_exports():
    """Zero-time sources separate export/capture wiring from acquisition."""
    from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

    model = RelationalExchangeModel(1, phase_domain="regular")
    entries = []
    for labels in ((0, 1), (2, 3)):
        graph = nx.path_graph(labels)
        for node in graph:
            graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
        graph.graph["GAMMA"] = {"type": "none"}
        source = bound_relational_sine_pattern(
            graph,
            reference_node=labels[0],
            reference_model=model,
            form_error_bounds=(0, 0),
            phase_error_bounds=(0, 0),
        )
        entries.append(
            source.certify_prepared_entry(scaled_time=0, edge_turn_offsets=(0,))
        )
    return tuple(
        entries[0].compose_with(
            entries[1],
            bridge=(1, 2),
            left_initial_time=Fraction(3, 7),
            right_initial_time=Fraction(3, 7),
            edge_turn_offsets=(0, 0, 0),
            form_origin_difference=origin,
            phase_origin_difference=0,
            work_allowance=Fraction(1, 4),
        )
        for origin in (Fraction(1, 3), None)
    )


@pytest.mark.parametrize("index", (0, 1))
def test_prepared_composition_exact_sdk_export(
    prepared_composition_exports, index, tmp_path
):
    report = prepared_composition_exports[index]
    generic = relational_report_to_dict(report)
    assert generic["report_type"] == "SinePreparedComposition"
    assert generic["report"] == report.to_dict()["report"]
    body = generic["report"]
    assert body["source"]["observation_time"] == {"numerator": 3, "denominator": 7}
    assert body["source"]["left"]["status"] == "unavailable"
    assert not report.acquisition_and_capture_certified
    if index == 0:
        assert body["status"] == "available"
        assert body["capture"]["status"] == "admitted"
        assert body["budget_status"] == "within_allowance"
        assert body["source"]["form_origin_difference"] == {
            "numerator": 1,
            "denominator": 3,
        }
        assert body["capture"]["source"] == body["source"]
    else:
        assert body["status"] == "unavailable"
        assert body["capture"] is None and body["joined_storage_bounds"] is None
        assert body["budget_status"] == "unresolved"
    path = tmp_path / "prepared-composition.json"
    export_to_json(generic, path)
    assert json.loads(path.read_text(encoding="utf-8")) == generic
    body["source"]["left"]["source"]["nominal_form"][0]["numerator"] = 42
    assert report.left.source.nominal_form[0] == 0


@pytest.mark.parametrize(
    "path",
    (
        ("source", "left", "source", "nodes"),
        ("source", "right", "geometry", "nodes"),
        ("source", "left", "capture", "nodes"),
        ("capture", "source", "right", "source", "reference_node"),
        ("capture", "geometry", "nodes"),
    ),
)
def test_prepared_composition_nested_label_admission(
    prepared_composition_exports, path
):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    value = OpaqueLabel(1)
    if path[-1] != "reference_node":
        value = (value,)
    report = _replace_report_path(prepared_composition_exports[0], path, value)
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


@pytest.mark.parametrize(
    "owner_name,class_name,assessor,evaluator",
    [
        (
            "relational_sine_reduced_class_ports",
            class_name,
            "assess_sine_reduced_class_ports",
            "evaluate_sine_reduced_class_ports",
        )
        for class_name in ("SineReducedClassPortState", "SineReducedClassPorts")
    ]
    + [
        (
            "relational_sine_port_composition",
            class_name,
            "assess_sine_port_composition",
            "evaluate_sine_port_composition",
        )
        for class_name in ("SinePortCompositionState", "SinePortComposition")
    ]
    + [
        (
            "relational_sine_port_relaxation",
            "SinePortRelaxation",
            "assess_sine_port_relaxation",
            None,
        ),
        (
            "relational_sine_port_form_tracking",
            "SinePortFormTracking",
            "assess_sine_port_form_tracking",
            None,
        ),
        (
            "relational_sine_two_port_compatibility",
            "SineTwoPortCompatibility",
            "assess_sine_two_port_compatibility",
            None,
        ),
        (
            "relational_sine_two_port_compatibility",
            "SineTwoPortHandoffObstruction",
            "assess_sine_two_port_handoff_obstruction",
            None,
        ),
        (
            "relational_sine_two_port_transit",
            "SineTwoPortTransit",
            "assess_sine_two_port_transit",
            None,
        ),
        (
            "relational_sine_two_port_capture",
            "SineTwoPortCapture",
            "assess_sine_two_port_capture",
            None,
        ),
        (
            "relational_sine_two_port_probe",
            "SineTwoPortProbe",
            "assess_sine_two_port_probe",
            None,
        ),
        (
            "relational_sine_two_port_dipole",
            "SineTwoPortDipole",
            "assess_sine_two_port_dipole",
            None,
        ),
        (
            "relational_sine_two_port_inference",
            "SineTwoPortInference",
            "infer_sine_two_port_geometry",
            None,
        ),
        (
            "relational_sine_two_port_readout",
            "SineTwoPortReadout",
            "bound_sine_two_port_readout",
            None,
        ),
        (
            "relational_sine_two_pulse_inference",
            "SineTwoPulseInference",
            "infer_sine_two_pulse_geometry_gain",
            None,
        ),
        (
            "relational_sine_clock_inference",
            "SineClockInference",
            "infer_sine_geometry_gain_clock",
            None,
        ),
        (
            "relational_sine_curvature_inference",
            "SineCurvatureInference",
            "infer_sine_geometry_gain_clock_curvature",
            None,
        ),
    ],
)
def test_reduced_port_sdk_wiring_does_not_evaluate_research(
    monkeypatch, owner_name, class_name, assessor, evaluator
):
    from importlib import import_module

    owner = import_module(f"tnfr.physics.{owner_name}")

    report_type = getattr(owner, class_name)
    report = object.__new__(report_type)
    body = {"unavailable": None, "retained_coordinates": 20}
    monkeypatch.setattr(report_type, "to_dict", lambda self: {"report": body})

    def forbidden(**kwargs):
        pytest.fail("SDK projection must not evaluate a research model")

    monkeypatch.setattr(owner, assessor, forbidden)
    if evaluator is not None:
        monkeypatch.setattr(owner, evaluator, forbidden)
    assert relational_report_to_dict(report) == {
        "schema": "tnfr.relational-report.v1",
        "report_type": class_name,
        "report": body,
    }
