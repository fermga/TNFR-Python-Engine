"""Reports preserve circular phase, current telemetry and existing files."""

import json
import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import set_dnfr, set_theta
from tnfr.metrics.coherence import compute_coherence
from tnfr.sdk import fluent
from tnfr.sdk.fluent import NetworkResults, TNFRNetwork
from tnfr.sdk.simple import Network
from tnfr.sdk.utils import (
    compare_networks,
    compute_network_statistics,
    export_to_json,
    format_comparison_table,
    import_from_json,
)


def test_optional_result_averages_do_not_break_summary():
    results = NetworkResults(1.0, {}, {}, nx.Graph())
    summary = results.summary()
    assert "Coherence C(t): 1.000" in summary
    assert "not measured" in summary


def test_zero_result_averages_are_measured_values():
    results = NetworkResults(1.0, {}, {}, nx.Graph(), avg_vf=0.0, avg_phase=0.0)
    summary = results.summary()
    assert "0.000 Hz_str (computed)" in summary
    assert "0.000 rad (computed)" in summary


def test_snapshot_summary_retains_correlation_when_conservation_is_unavailable():
    results = NetworkResults(
        1.0,
        {},
        {},
        nx.Graph(),
        unified_fields={
            "complex_field": {"correlation": -0.25},
            "tensor_invariants": {
                "conservation_quality": None,
                "conservation_sample_available": False,
                "conservation_scope": "single_snapshot_no_temporal_balance",
            },
        },
    )
    summary = results.summary()
    assert "Correlation: -0.250" in summary
    assert "Conservation Quality: unavailable (single snapshot)" in summary


def test_mean_phase_respects_the_phase_wrap():
    network = TNFRNetwork().add_nodes(2)
    for node, phase in zip(network.graph, [0.1, 2.0 * math.pi - 0.1]):
        set_theta(network.graph, node, phase)
    phase = network.measure().avg_phase
    assert math.atan2(math.sin(phase), math.cos(phase)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("rotation", [0.0, 0.1, 1.0])
def test_balanced_antipodal_phases_have_no_reportable_mean(rotation):
    network = TNFRNetwork().add_nodes(2)
    for node, phase in zip(network.graph, [rotation, rotation + math.pi]):
        set_theta(network.graph, node, phase)
    assert network.measure().avg_phase is None


def test_export_reports_current_graph_after_a_previous_measurement():
    network = TNFRNetwork().add_nodes(2)
    before = network.measure().coherence
    for node in network.graph:
        set_dnfr(network.graph, node, 10.0)
    expected = compute_coherence(network.graph)
    assert expected != before
    assert network.export_to_dict()["metrics"]["coherence"] == pytest.approx(expected)


def test_save_refreshes_once_and_uses_the_shared_report_schema(tmp_path, monkeypatch):
    network = TNFRNetwork("saved-report").add_nodes(2)
    before = network.measure().coherence
    for node in network.graph:
        set_dnfr(network.graph, node, 10.0)
    expected = compute_coherence(network.graph)
    assert expected != before
    reads = []
    original = network.measure

    def measure():
        reads.append(None)
        return original()

    monkeypatch.setattr(network, "measure", measure)
    destination = tmp_path / "nested" / "report.json"
    assert network.save(destination) is network
    assert len(reads) == 1
    saved = import_from_json(destination)
    assert saved["name"] == "saved-report"
    assert saved["metadata"]["nodes"] == 2
    assert saved["metrics"]["coherence"] == pytest.approx(expected)
    assert saved == json.loads(json.dumps(network.export_to_dict()))


def test_failed_save_preserves_existing_file(tmp_path):
    network = TNFRNetwork().add_nodes(2)
    destination = tmp_path / "result.json"
    previous = '{"previous": true}'
    destination.write_text(previous, encoding="utf-8")
    for node in network.graph:
        set_dnfr(network.graph, node, math.nan)
    with pytest.raises(ValueError):
        network.save(destination)
    assert destination.read_text(encoding="utf-8") == previous
    assert list(tmp_path.iterdir()) == [destination]


def test_save_rejects_an_uninitialized_network_before_writing(tmp_path):
    destination = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="No network created"):
        TNFRNetwork().save(destination)
    assert not destination.exists()


def test_failed_json_export_preserves_existing_file(tmp_path):
    destination = tmp_path / "result.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(TypeError):
        export_to_json({"valid": 1, "unsupported": object()}, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    "numeric_key, text_key", [(1, "1"), (1.0, "1.0"), (True, "true"), (None, "null")]
)
def test_nested_json_key_collisions_preserve_existing_report(
    tmp_path, numeric_key, text_key
):
    destination = tmp_path / "result.json"
    previous = '{"previous": true}'
    destination.write_text(previous, encoding="utf-8")
    data = {"rows": [(0, {numeric_key: "first", text_key: "second"})]}
    with pytest.raises(ValueError, match="keys collide after serialization"):
        export_to_json(data, destination)
    assert destination.read_text(encoding="utf-8") == previous
    assert list(tmp_path.iterdir()) == [destination]


def test_noncolliding_json_keys_retain_standard_encoder_behavior(tmp_path):
    destination = tmp_path / "result.json"
    data = {
        1: [{"shared": "first"}, {"shared": "second"}],
        "1.0": "text",
        2.5: "float",
        False: "boolean",
        None: "none",
    }
    export_to_json(data, destination)
    assert import_from_json(destination) == {
        "1": [{"shared": "first"}, {"shared": "second"}],
        "1.0": "text",
        "2.5": "float",
        "false": "boolean",
        "null": "none",
    }


@pytest.mark.parametrize(
    "payload",
    [
        '{"nodes": 1, "nodes": 2}',
        '{"rows": [{"seed": 1, "seed": 2}]}',
        r'{"nodes": 1, "no\u0064es": 2}',
    ],
)
def test_json_import_rejects_duplicate_decoded_names(tmp_path, payload):
    source = tmp_path / "ambiguous.json"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="keys collide"):
        import_from_json(source)
    assert source.read_text(encoding="utf-8") == payload


@pytest.mark.parametrize(
    "token", ["NaN", "Infinity", "-Infinity", "1e999", "1e-999", "-1e-999"]
)
def test_json_import_rejects_nonfinite_or_lost_nonzero_numbers(tmp_path, token):
    source = tmp_path / "invalid.json"
    source.write_text('{"rows": [{"value": ' + token + "}]}", encoding="utf-8")
    with pytest.raises(ValueError, match="finite|underflows"):
        import_from_json(source)


def test_json_number_roundtrip_retains_zero_subnormal_and_integer_domains(tmp_path):
    source = tmp_path / "numbers.json"
    tiny = float.fromhex("0x0.0000000000001p-1022")
    values = {"tiny": tiny, "negative_zero": -0.0, "large_integer": 10**400}
    export_to_json(values, source)
    restored = import_from_json(source)
    assert restored == values
    assert math.copysign(1.0, restored["negative_zero"]) == -1.0


def test_json_zero_does_not_depend_on_decimal_exponent_limits(tmp_path):
    source = tmp_path / "zero.json"
    source.write_text('{"zero": 0e-9999999999999999999999999999}', encoding="utf-8")
    assert import_from_json(source) == {"zero": 0.0}


def test_json_export_uses_shared_atomic_writer_for_new_directories(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    export_to_json({"phase": 0.0, "label": "\u03c0"}, destination)
    assert import_from_json(destination) == {"phase": 0.0, "label": "\u03c0"}


def test_encoding_failure_preserves_error_details_and_existing_file(tmp_path):
    destination = tmp_path / "result.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(UnicodeEncodeError) as failure:
        export_to_json({"invalid_unicode": "\ud800"}, destination)
    assert failure.value.encoding == "utf-8"
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    "pressures, mean, std",
    [
        ([1e308, 1e308], 1e308, 0.0),
        ([-1e308, 1e308], 0.0, 1e308),
        ([1e308, 1.0, -1e308], 1.0 / 3.0, math.sqrt(2.0 / 3.0) * 1e308),
    ],
)
def test_reports_share_finite_signed_reductions(pressures, mean, std):
    graph = nx.path_graph(len(pressures))
    results = NetworkResults(
        0.5, dict.fromkeys(graph, 0.5), dict(enumerate(pressures)), graph
    )
    summary = results.to_dict()["summary_stats"]
    stats = compute_network_statistics(results)
    compared = compare_networks({"sample": results})["sample"]
    assert summary["avg_delta_nfr"] == pytest.approx(mean)
    assert stats["avg_delta_nfr"] == summary["avg_delta_nfr"]
    assert compared["avg_delta_nfr"] == summary["avg_delta_nfr"]
    assert stats["std_delta_nfr"] == pytest.approx(std)
    assert "inf" not in results.summary()
    json.dumps(results.to_dict(), allow_nan=False)


def test_near_antipodal_phase_uses_one_sdk_availability_policy():
    network = TNFRNetwork().add_nodes(2)
    for node, phase in zip(network.graph, [0.0, math.pi - 1e-13]):
        set_theta(network.graph, node, phase)
    assert Network(network.graph).avg_phase() is None
    assert network.measure().avg_phase is None


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_json_export_preserves_existing_file(tmp_path, invalid):
    destination = tmp_path / "result.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        export_to_json({"measurement": invalid}, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.iterdir()) == [destination]


def test_fluent_metric_exports_do_not_alias_the_captured_result():
    network = TNFRNetwork().add_nodes(2)
    result = network.measure()
    data = result.to_dict()
    node = next(iter(network.graph))
    original_si = result.sense_indices[node]

    data["sense_indices"][node] = -1.0
    data["delta_nfr"][node] = 99.0

    assert result.sense_indices[node] == original_si
    assert result.delta_nfr[node] == 0.0
    assert result.to_dict()["summary_stats"]["avg_delta_nfr"] == 0.0


def test_fluent_measure_observes_one_detached_graph_without_live_cache_writes(
    monkeypatch,
):
    network = TNFRNetwork().add_nodes(2).connect_nodes(connection_pattern="ring")
    for node in network.graph:
        set_dnfr(network.graph, node, 0.5)
    before_nodes = deepcopy(dict(network.graph.nodes(data=True)))
    before_edges = deepcopy(list(network.graph.edges(data=True)))
    before_metadata = dict(network.graph.graph)
    observed_graphs = []
    original_coherence, original_si = fluent.compute_coherence, fluent.compute_Si

    def coherence(graph):
        observed_graphs.append(graph)
        return original_coherence(graph)

    def sense(graph, *, inplace):
        observed_graphs.append(graph)
        return original_si(graph, inplace=inplace)

    monkeypatch.setattr(fluent, "compute_coherence", coherence)
    monkeypatch.setattr(fluent, "compute_Si", sense)
    result = network.measure()

    assert observed_graphs == [result.graph, result.graph]
    assert result.graph is not network.graph
    assert result.coherence == pytest.approx(2.0 / 3.0)
    assert dict(network.graph.nodes(data=True)) == before_nodes
    assert list(network.graph.edges(data=True)) == before_edges
    assert network.graph.graph == before_metadata


def test_optional_field_failure_retains_provenance_and_core_measurements(tmp_path):
    network = TNFRNetwork().add_nodes(2).connect_nodes(connection_pattern="ring")
    for node in network.graph:
        set_dnfr(network.graph, node, 1.0)
        set_theta(network.graph, node, 0.0)
    for _, _, data in network.graph.edges(data=True):
        data["length"] = -1.0
    result = network.measure()

    assert result.coherence == 0.5
    assert result.unified_fields is None
    assert result.unified_fields_available is False
    assert result.unified_fields_error["type"] == "ValueError"
    assert "edge length" in result.unified_fields_error["message"]
    assert "Unified Field Telemetry: unavailable" in result.summary()
    destination = tmp_path / "report.json"
    export_to_json(result, destination)
    data = import_from_json(destination)
    assert data["unified_fields_available"] is False
    assert data["unified_fields_error"] == result.unified_fields_error
    data["unified_fields_error"]["message"] = "changed export"
    assert "edge length" in result.unified_fields_error["message"]


def test_core_measurement_failure_propagates_without_replacing_cached_evidence(
    monkeypatch,
):
    network = TNFRNetwork().add_nodes(2)
    previous = network.measure()

    def refuse(graph):
        assert graph is not network.graph
        raise ValueError("invalid stored pressure")

    monkeypatch.setattr(fluent, "compute_coherence", refuse)
    with pytest.raises(ValueError, match="invalid stored pressure"):
        network.measure()
    assert network._results is previous


def test_fluent_export_retains_opaque_scalar_map_node_identity():
    class Node:
        def __deepcopy__(self, memo):
            raise AssertionError("node identity is not mutable report data")

    node = Node()
    result = NetworkResults(1.0, {node: 0.5}, {node: 0.0}, nx.Graph())
    exported = result.to_dict()
    assert next(iter(exported["sense_indices"])) is node
    assert next(iter(exported["delta_nfr"])) is node


def test_comparison_table_uses_all_rows_and_keeps_missing_separate_from_zero():
    table = format_comparison_table(
        {"missing": {}, "measured": {"coherence": 0.0, "avg_si": 0.25}}
    )
    lines = table.splitlines()
    assert "coherence" in lines[0] and "avg_si" in lines[0]
    assert (
        next(line for line in lines if line.startswith("missing")).count("unavailable")
        == 2
    )
    assert "0.000" in next(line for line in lines if line.startswith("measured"))


@pytest.mark.parametrize(
    "invalid", [None, False, "0.0", math.nan, Fraction(1, 10**400)]
)
def test_comparison_table_does_not_materialize_invalid_values_as_measured_zero(invalid):
    table = format_comparison_table({"sample": {"coherence": invalid}})
    assert "unavailable" in table
    assert "0.000" not in table
