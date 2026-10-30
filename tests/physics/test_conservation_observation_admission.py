"""Conservation diagnostics require actual interval and sector evidence."""

import math
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.conservation import (
    ConservationSnapshot,
    ConservationTracker,
    LyapunovResult,
    _energy_from_snapshot,
    analyze_sector_coupling,
    capture_conservation_snapshot,
    compute_energy_functional,
    compute_lyapunov_derivative,
    compute_noether_charge,
    compute_ward_identity,
    decompose_conservation_residual,
    detect_grammar_violations_from_conservation,
    verify_conservation_balance,
    verify_sequence_ward_identity,
)


def _graph():
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(
            theta=(0.0, 0.3, 0.9)[node],
            delta_nfr=(0.0, 2.0, -1.0)[node],
            EPI=0.0,
            nu_f=1.0,
        )
    return graph


@pytest.fixture
def snapshot():
    return capture_conservation_snapshot(_graph())


def test_sector_divergence_reads_neighbors_instead_of_local_current_magnitudes():
    graph = _graph()
    before = capture_conservation_snapshot(graph)
    result = decompose_conservation_residual(before, before)
    for node in graph:
        for field, result_key in (
            (before.j_phi, "j_phi_div"),
            (before.j_dnfr, "j_dnfr_div"),
        ):
            expected = (
                sum(field[neighbor] - field[node] for neighbor in graph[node])
                / graph.degree[node]
            )
            assert result[result_key][node] == pytest.approx(expected)
        assert result["potential_residual"][node] + result["geometric_residual"][
            node
        ] == pytest.approx(before.divergence[node])
    # Local proportional splitting gave a negative phase component at node 2.
    assert before.divergence[2] < 0.0 < result["j_phi_div"][2]


def test_captured_sector_maps_are_detached_and_sum_to_total(snapshot):
    assert snapshot.divergence_phi is not snapshot.j_phi
    assert snapshot.divergence_dnfr is not snapshot.j_dnfr
    for node in snapshot.charge_density:
        assert (
            snapshot.divergence[node]
            == snapshot.divergence_phi[node] + snapshot.divergence_dnfr[node]
        )


@pytest.mark.parametrize(
    "reader", [decompose_conservation_residual, analyze_sector_coupling]
)
def test_legacy_snapshot_has_balance_but_no_invented_sector_evidence(snapshot, reader):
    legacy = replace(snapshot, divergence_phi=None, divergence_dnfr=None)
    assert math.isfinite(verify_conservation_balance(legacy, legacy).rms_residual)
    with pytest.raises(ValueError, match="sector divergences are unavailable"):
        reader(legacy, legacy)


@pytest.mark.parametrize(
    "dt", [0.0, -1.0, math.nan, math.inf, True, np.bool_(True), "1"]
)
@pytest.mark.parametrize(
    "reader",
    [
        verify_conservation_balance,
        decompose_conservation_residual,
        analyze_sector_coupling,
        compute_lyapunov_derivative,
    ],
)
def test_invalid_clock_cannot_supply_a_conservation_or_stability_verdict(
    snapshot, dt, reader
):
    with pytest.raises(ValueError, match="dt"):
        reader(snapshot, snapshot, dt=dt)


@pytest.mark.parametrize("dt", [0.0, -1.0, math.nan, math.inf, True])
def test_ward_reader_uses_the_same_interval_admission(snapshot, dt):
    with pytest.raises(ValueError, match="dt"):
        compute_ward_identity(snapshot, snapshot, "observed", dt=dt)


@pytest.mark.parametrize(
    "reader",
    [
        verify_conservation_balance,
        compute_lyapunov_derivative,
        decompose_conservation_residual,
    ],
)
def test_support_change_needs_explicit_correspondence_and_source_accounting(
    snapshot, reader
):
    graph = _graph()
    graph.remove_node(2)
    with pytest.raises(ValueError, match="identical node support"):
        reader(snapshot, capture_conservation_snapshot(graph))


@pytest.mark.parametrize(
    "reader", [verify_conservation_balance, compute_lyapunov_derivative]
)
def test_empty_observation_is_not_perfect_balance_or_zero_energy_stability(
    snapshot, reader
):
    empty = capture_conservation_snapshot(nx.Graph())
    with pytest.raises(ValueError, match="nonempty node support"):
        reader(snapshot, empty)
    with pytest.raises(ValueError, match="nonempty node support"):
        reader(empty, empty)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, True, np.bool_(True)])
def test_invalid_authoritative_divergence_is_not_zero_filled(snapshot, bad):
    altered = replace(snapshot, divergence={**snapshot.divergence, 0: bad})
    with pytest.raises(ValueError, match="divergence"):
        verify_conservation_balance(snapshot, altered)


def test_incomplete_divergence_is_not_zero_filled(snapshot):
    altered = replace(snapshot, divergence={0: 0.0})
    with pytest.raises(ValueError, match="complete node support"):
        verify_conservation_balance(snapshot, altered)


@pytest.mark.parametrize("timestamp", [0.0, -1.0, math.nan, math.inf, True])
def test_tracker_rejects_bad_timestamp_without_partial_record(timestamp):
    tracker = ConservationTracker(_graph())
    tracker.record(0.0)
    before = deepcopy(tracker._snapshots), deepcopy(tracker.report())
    with pytest.raises(ValueError):
        tracker.record(timestamp)
    assert (tracker._snapshots, tracker.report()) == before
    tracker.record(1.0)
    assert tracker.report().times == [0.0, 1.0]


def test_tracker_rejects_support_change_without_partial_record():
    graph = _graph()
    tracker = ConservationTracker(graph)
    tracker.record(0.0)
    before = deepcopy(tracker._snapshots), deepcopy(tracker.report())
    graph.remove_node(2)
    with pytest.raises(ValueError, match="identical node support"):
        tracker.record(1.0)
    assert (tracker._snapshots, tracker.report()) == before


def test_empty_ward_series_does_not_claim_conservation():
    result = verify_sequence_ward_identity([])
    assert result["sample_available"] is False
    assert result["sequence_conserved"] is False
    assert result["aggregate_balance_within_alert"] is False


@pytest.mark.parametrize("threshold", [-1.0, math.nan, math.inf, True, np.bool_(True)])
def test_lyapunov_numerical_tolerance_is_admitted_explicitly(snapshot, threshold):
    with pytest.raises(ValueError, match="stability_threshold"):
        compute_lyapunov_derivative(snapshot, snapshot, stability_threshold=threshold)


def test_small_positive_energy_change_is_not_exact_nonincrease(snapshot):
    after = replace(
        snapshot,
        phi_s={node: value * 1.000000000001 for node, value in snapshot.phi_s.items()},
    )
    result = compute_lyapunov_derivative(snapshot, after)
    assert 0.0 < result.energy_derivative < 1e-6
    assert result.is_stable is True  # compatibility tolerance alert
    assert result.energy_nonincreasing is False


@pytest.mark.parametrize("magnitude", [1e200, float.fromhex("0x0.0000000000001p-1022")])
def test_rms_preserves_representable_residual_scale(magnitude):
    zero = {0: 0.0, 1: 0.0}
    snapshot = ConservationSnapshot(
        charge_density=zero.copy(),
        phi_s=zero.copy(),
        grad_phi=zero.copy(),
        k_phi=zero.copy(),
        j_phi=zero.copy(),
        j_dnfr=zero.copy(),
        divergence={0: magnitude, 1: magnitude},
    )
    result = verify_conservation_balance(snapshot, snapshot)
    assert result.mean_residual == magnitude
    assert result.rms_residual == magnitude
    assert result.std_residual == 0.0
    assert result.grammar_violation_index == magnitude


def _numeric_snapshot(charges):
    nodes = range(len(charges))
    zero = {node: 0.0 for node in nodes}
    return ConservationSnapshot(
        charge_density=dict(enumerate(charges)),
        phi_s=dict(enumerate(charges)),
        k_phi=zero.copy(),
        grad_phi=zero.copy(),
        j_phi=zero.copy(),
        j_dnfr=zero.copy(),
        divergence=zero.copy(),
        divergence_phi=zero.copy(),
        divergence_dnfr=zero.copy(),
    )


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_sector_correlation_is_scaled_and_reports_legacy_gate_availability(scale):
    before = _numeric_snapshot((0.0, 0.0, 0.0))
    after = replace(
        before,
        phi_s={0: -scale, 1: 0.0, 2: scale},
        k_phi={0: scale, 1: 0.0, 2: -scale},
    )
    result = analyze_sector_coupling(before, after)
    if scale > 1e-15:
        assert result["cross_coupling_available"] is True
        assert result["cross_coupling_strength"] == pytest.approx(-1.0)
    else:
        assert result["cross_coupling_available"] is False
        assert result["cross_coupling_status"] == "below_legacy_dispersion_cut"
        assert (
            result["cross_coupling_strength"] == 0.0
        )  # compatibility, not a measurement


def test_unrepresentable_sector_asymmetry_is_explicitly_unavailable():
    result = analyze_sector_coupling(
        _numeric_snapshot((0.0, 0.0, 0.0)),
        _numeric_snapshot((-1e308, 0.0, 1e308)),
    )
    assert result["dominant_sector"] == "potential"
    assert result["sector_asymmetry"] is None
    assert result["sector_asymmetry_available"] is False


@pytest.mark.parametrize("magnitude", [1e308, np.float32(2e38)])
def test_balance_and_sector_secants_recover_finite_rates_after_large_subtraction(
    magnitude,
):
    before = _numeric_snapshot((-magnitude, magnitude))
    after = _numeric_snapshot((magnitude, -magnitude))
    interval = float(magnitude)
    balance = verify_conservation_balance(before, after, dt=interval)
    sectors = decompose_conservation_residual(before, after, dt=interval)
    # Each exact represented change is twice the supplied interval. Opposite
    # signed nodes keep both total charges and their drift representable.
    assert balance.delta_rho == balance.residual == {0: 2.0, 1: -2.0}
    assert balance.charge_drift == balance.mean_residual == 0.0
    assert balance.rms_residual == 2.0
    assert sectors["phi_s_drift"] == balance.delta_rho


@pytest.mark.parametrize(
    "reader", [verify_conservation_balance, decompose_conservation_residual]
)
def test_nonzero_unrepresentable_charge_rate_cannot_become_perfect_balance(reader):
    with pytest.raises(ValueError, match="underflow"):
        reader(_numeric_snapshot((0.0,)), _numeric_snapshot((1e-300,)), dt=1e100)


def test_nonzero_unrepresentable_energy_rate_cannot_claim_nonincrease():
    with pytest.raises(ValueError, match="energy derivative.*underflow"):
        compute_lyapunov_derivative(
            _numeric_snapshot((0.0,)), _numeric_snapshot((1e-150,)), dt=1e100
        )
    # Legacy/deserialized results may already carry a rounded zero derivative.
    # The endpoint energies still determine the accurate named sign read-out.
    legacy = LyapunovResult(0.0, 5e-301, 0.0, 0.0, True, False)
    assert legacy.energy_nonincreasing is False


def test_snapshot_nonzero_source_cannot_disappear_during_input_materialization():
    invalid = _numeric_snapshot((Fraction(1, 10**400),))
    with pytest.raises(ValueError, match="underflow"):
        verify_conservation_balance(invalid, invalid)


def test_returned_tracker_evidence_is_detached_without_copying_node_identity():
    nodes = (object(), object())
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    tracker = ConservationTracker(graph)
    returned = tracker.record(0.0)
    assert tuple(returned.charge_density) == nodes
    returned.charge_density[nodes[0]] = 9.0
    tracker.record(1.0)
    assert tracker.latest_balance.delta_rho == dict.fromkeys(nodes, 0.0)

    report = tracker.report()
    report.times.append(7.0)
    report.conservation_quality[-1] = -999.0
    retained = tracker.report()
    assert retained.times == [0.0, 1.0]
    assert retained.conservation_quality == [1.0, 1.0]


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -1.0, 0.0, True, "1"])
def test_balance_alert_cut_must_be_a_finite_positive_real(snapshot, invalid):
    balance = verify_conservation_balance(snapshot, snapshot)
    with pytest.raises(ValueError, match="max_allowed_residual"):
        detect_grammar_violations_from_conservation(
            balance, {"max_allowed_residual": invalid}
        )


def test_live_charge_and_tracker_share_extreme_finite_cancellation():
    graph = nx.complete_graph(4)
    for node, pressure in enumerate((1e308, 1e308, -1e308, -1e308)):
        graph.nodes[node].update(delta_nfr=pressure, theta=0.0)
    # Unit K4 potential at each node sums the other three pressures, hence
    # Phi=(-1e308,-1e308,+1e308,+1e308), and total charge is exactly zero.
    assert compute_noether_charge(graph) == 0.0
    tracker = ConservationTracker(graph)
    tracker.record(0.0)
    tracker.record(1.0)
    balance = tracker.latest_balance
    assert balance.total_charge_before == balance.total_charge_after == 0.0
    assert balance.charge_drift == 0.0
    assert tracker.report().total_charge == [0.0, 0.0]
    assert math.isfinite(balance.rms_residual)
    assert balance.rms_residual > 1e308


def test_live_and_snapshot_energy_recover_positive_subnormal_network_total():
    graph = nx.Graph()
    graph.add_edges_from((2 * i, 2 * i + 1) for i in range(4))
    before = capture_conservation_snapshot(graph)
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 1.1e-162
    after = capture_conservation_snapshot(graph)
    # Eight equal potentials; the other four field channels vanish. Each
    # individual square rounds to zero, but 8 * Phi^2 / 2 rounds to minsubnormal.
    expected = float(4 * Fraction.from_float(1.1e-162) ** 2)
    assert expected == math.ulp(0.0)
    assert compute_energy_functional(graph) == _energy_from_snapshot(after) == expected
    change = compute_lyapunov_derivative(before, after)
    assert change.energy_derivative == expected
    assert change.energy_nonincreasing is False


def test_live_and_snapshot_energy_recover_finite_normalized_square():
    graph = nx.path_graph(2)
    graph.edges[0, 1]["length"] = 1e-77
    graph.nodes[0]["delta_nfr"] = 1.5
    snapshot = capture_conservation_snapshot(graph)
    # Inverse-square aggregation amplifies one potential while pressure flux
    # stays finite. Its raw square overflows; the half-square total does not.
    potential = Fraction.from_float(snapshot.phi_s[1])
    expected = float(potential**2 / 2 + Fraction(9, 4))
    assert expected > 1e308
    assert math.isfinite(expected)
    assert (
        compute_energy_functional(graph) == _energy_from_snapshot(snapshot) == expected
    )
