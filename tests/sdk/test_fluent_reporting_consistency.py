"""Reports preserve circular phase, current telemetry and existing files."""

import json
import math

import networkx as nx
import pytest

from tnfr.alias import set_dnfr, set_theta
from tnfr.metrics.coherence import compute_coherence
from tnfr.sdk.fluent import NetworkResults, TNFRNetwork
from tnfr.sdk.utils import export_to_json, import_from_json


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


def test_failed_json_export_preserves_existing_file(tmp_path):
    destination = tmp_path / "result.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(TypeError):
        export_to_json({"valid": 1, "unsupported": object()}, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.iterdir()) == [destination]


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
