"""Tests for TNFR phase-transition telemetry and operational phase labels.

Validates the signed global order/chirality z-scores, time-series measurements
and reproducibility. It does not assert a universal phase-transition theorem.

Tests verify:
1.  Symmetric phase: uniform/equilibrium graphs → ⟨𝒮⟩ = 0 (NON_LIFE)
2.  Signed global imbalance is distinct from local heterogeneous activity
3.  Preferred handedness uses |⟨χ⟩|, not ⟨|χ|⟩
4.  Effective time-series exponent is measured without a universal comparator
5.  Susceptibility: sampled peak along a declared transition sequence
6.  Coherence length: finite non-negative sampled diagnostic
7.  Phase classification consistency across topologies
8.  Transition time detection via interpolation
9.  Sampled susceptibility-peak time
10. PhaseSnapshot capture correctness
11. Time-series detect_phase_transition() pipeline
12. Power-law fit of a protocol-dependent effective exponent
13. Constants consistency with canonical module
14. Dataclass integrity (PhaseTransitionTelemetry fields)
15. Reproducibility under seed control

TIER: CORE TELEMETRY — operational classification and measured scaling.
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA
from tnfr.physics import phase_transition as transition_module
from tnfr.physics.phase_transition import (
    Phase,
    PhaseTransitionTelemetry,
    capture_phase_snapshot,
    classify_phase,
    compute_chirality_statistics,
    compute_order_parameter,
    detect_phase_transition,
    fit_critical_exponent,
    symmetry_zscore,
)
from tnfr.physics.unified import (
    compute_chirality_field,
    compute_symmetry_breaking_field,
)

# ============================================================================
# Fixtures — build TNFR graphs in controlled structural regimes
# ============================================================================


@pytest.fixture
def uniform_graph() -> nx.Graph:
    """Graph with uniform phases and zero ΔNFR → symmetric (non-life)."""
    G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
    inject_defaults(G)
    # All defaults are uniform → ⟨𝒮⟩ = 0
    return G


@pytest.fixture
def heterogeneous_graph() -> nx.Graph:
    """Graph with diverse phases and heterogeneous ΔNFR → broken symmetry."""
    rng = np.random.default_rng(42)
    G = nx.watts_strogatz_graph(30, 4, 0.3, seed=42)
    inject_defaults(G)
    for node in G.nodes():
        G.nodes[node]["theta"] = rng.uniform(0, 2 * np.pi)
        G.nodes[node]["phase"] = G.nodes[node]["theta"]
        G.nodes[node]["delta_nfr"] = rng.uniform(0.1, 1.5)
    return G


def _build_transition_sequence(n_steps: int = 20, seed: int = 42) -> tuple:
    """Build a time series that transitions from uniform to heterogeneous.

    Returns (graphs, times) where phases gradually diversify.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    times = []
    for step in range(n_steps):
        t = float(step)
        times.append(t)
        G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
        inject_defaults(G)
        # Gradually increase phase diversity and ΔNFR heterogeneity
        scale = step / max(n_steps - 1, 1)  # 0 → 1
        for node in G.nodes():
            G.nodes[node]["theta"] = rng.uniform(0, 2 * np.pi * scale)
            G.nodes[node]["phase"] = G.nodes[node]["theta"]
            G.nodes[node]["delta_nfr"] = rng.uniform(0.0, 1.5 * scale)
        graphs.append(G)
    return graphs, times


# ============================================================================
# Test 1: Constants consistency with canonical module
# ============================================================================


@pytest.fixture(scope="module")
def transition_sample():
    """One retained seeded ramp serves distinct read-only telemetry assertions."""
    graphs, times = _build_transition_sequence(n_steps=20, seed=42)
    return times, detect_phase_transition(graphs, times)


class TestEmergentClassification:
    """The classifier uses a standardized spatial imbalance.

    Audit 2026: the magic scales (γ/π)² and γ/(π+γ) were proven INERT
    (they sat in a two-order-of-magnitude gap; sweeping them changed no
    classification) and were removed. Coupled nodes do not justify an
    independent-sample significance interpretation.
    """

    def test_zscore_zero_for_uniform_zero_field(self):
        """A perfectly uniform zero field has z = 0 (symmetric)."""
        assert symmetry_zscore(0.0, 0.0, 30) == 0.0

    def test_zscore_infinite_for_uniform_nonzero_field(self):
        """A uniform non-zero field (Var=0, mean>0) is fully broken: z = ∞."""
        assert symmetry_zscore(0.5, 0.0, 30) == math.inf

    def test_zscore_is_mean_over_standard_error(self):
        """The legacy z-score API computes |mean| / sqrt(Var/N)."""
        z = symmetry_zscore(0.2, 0.01, 25)
        assert z == pytest.approx(0.2 / math.sqrt(0.01 / 25), rel=1e-12)

    def test_zscore_zero_nodes_is_zero(self):
        """An empty system has zero standardized imbalance."""
        assert symmetry_zscore(1.0, 1.0, 0) == 0.0

    @pytest.mark.parametrize(
        "mean_abs,variance,n",
        [(-0.1, 1.0, 2), (math.nan, 1.0, 2), (0.1, -1.0, 2), (0.1, 1.0, -1)],
    )
    def test_zscore_rejects_invalid_inputs(self, mean_abs, variance, n):
        with pytest.raises(ValueError):
            symmetry_zscore(mean_abs, variance, n)

    @pytest.mark.parametrize(
        "order_z,chirality_z",
        [(math.nan, 0.0), (0.0, math.nan), (-1.0, 0.0), (0.0, -1.0)],
    )
    def test_phase_classifier_rejects_invalid_ratios(self, order_z, chirality_z):
        with pytest.raises(ValueError):
            classify_phase(order_z, chirality_z)


# ============================================================================
# Test 2: Symmetric phase (non-life)
# ============================================================================


class TestSymmetricPhase:
    """Uniform equilibrium graphs must be classified as NON_LIFE."""

    def test_uniform_graph_order_parameter_zero(self, uniform_graph):
        """⟨𝒮⟩ = 0 for uniform phases and ΔNFR."""
        op = compute_order_parameter(uniform_graph)
        assert op["mean"] == pytest.approx(0.0, abs=1e-10)
        assert op["abs_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_graph_chirality_zero(self, uniform_graph):
        """⟨χ⟩ = 0 for uniform phases."""
        chi = compute_chirality_statistics(uniform_graph)
        assert chi["mean"] == pytest.approx(0.0, abs=1e-10)
        assert chi["abs_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_graph_classified_non_life(self, uniform_graph):
        """Uniform graph → Phase.NON_LIFE."""
        snap = capture_phase_snapshot(uniform_graph)
        assert snap.phase == Phase.NON_LIFE

    def test_uniform_no_homochirality(self, uniform_graph):
        """Uniform graph has no homochirality."""
        snap = capture_phase_snapshot(uniform_graph)
        assert snap.has_homochirality is False

    def test_susceptibility_zero_uniform(self, uniform_graph):
        """χ_𝒮 = 0 when all 𝒮(i) = 0."""
        op = compute_order_parameter(uniform_graph)
        assert op["susceptibility"] == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Test 3: Broken symmetry (life phase)
# ============================================================================


class TestBrokenSymmetry:
    """The declared heterogeneous fixture has nonzero local activity."""

    def test_heterogeneous_nonzero_order_parameter(self, heterogeneous_graph):
        """⟨|𝒮|⟩ > 0 for diverse phases and ΔNFR."""
        op = compute_order_parameter(heterogeneous_graph)
        assert op["abs_mean"] > 0

    def test_heterogeneous_nonzero_chirality(self, heterogeneous_graph):
        """⟨|χ|⟩ > 0 for diverse phases (broken mirror symmetry)."""
        chi = compute_chirality_statistics(heterogeneous_graph)
        assert chi["abs_mean"] > 0

    def test_random_heterogeneity_without_signed_chirality_is_critical(
        self, heterogeneous_graph
    ):
        """Local chiral activity does not imply preferred handedness."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.phase == Phase.CRITICAL

    def test_random_heterogeneity_does_not_force_homochirality(
        self, heterogeneous_graph
    ):
        """Opposing local chirality cancels in the signed global mean."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        chi = compute_chirality_statistics(heterogeneous_graph)
        assert chi["abs_mean"] > 10.0 * abs(chi["mean"])
        assert snap.has_homochirality is False

    def test_positive_susceptibility(self, heterogeneous_graph):
        """χ_𝒮 > 0 in the broken phase (finite fluctuations)."""
        op = compute_order_parameter(heterogeneous_graph)
        assert op["susceptibility"] > 0


# ============================================================================
# Test 4: Chirality and homochirality
# ============================================================================


class TestChirality:
    """Verify chirality field behaviour and homochirality detection."""

    def test_chirality_abs_mean_geq_abs_mean(self, heterogeneous_graph):
        """⟨|χ|⟩ ≥ |⟨χ⟩| by Jensen's inequality."""
        chi = compute_chirality_statistics(heterogeneous_graph)
        assert chi["abs_mean"] >= abs(chi["mean"]) - 1e-12

    def test_chirality_per_node_consistency(self, heterogeneous_graph):
        """compute_chirality_statistics consistent with unified.compute_chirality_field."""
        field = compute_chirality_field(heterogeneous_graph)
        stats = compute_chirality_statistics(heterogeneous_graph)
        values = np.array(list(field.values()))
        assert stats["mean"] == pytest.approx(float(np.mean(values)), rel=1e-10)
        assert stats["abs_mean"] == pytest.approx(
            float(np.mean(np.abs(values))), rel=1e-10
        )


# ============================================================================
# Test 5: Phase classification logic
# ============================================================================


class TestPhaseClassification:
    """Verify the configured standardized-imbalance classifier and its boundary."""

    def test_zero_zscore_is_non_life(self):
        """order_z = 0 maps to the operational NON_LIFE label."""
        assert classify_phase(0.0, 0.0) == Phase.NON_LIFE

    def test_subsigma_order_is_non_life(self):
        """order_z ≤ 1 → NON_LIFE even with high chirality_z."""
        assert classify_phase(0.9, 5.0) == Phase.NON_LIFE

    def test_boundary_z_equals_one_is_non_life(self):
        """order_z = 1 exactly → NON_LIFE (the cut is z > 1)."""
        assert classify_phase(1.0, 2.0) == Phase.NON_LIFE

    def test_high_order_high_chirality_is_life(self):
        """order_z > 1 AND chirality_z > 1 → LIFE."""
        assert classify_phase(5.0, 5.0) == Phase.LIFE

    def test_high_order_low_chirality_is_critical(self):
        """order_z > 1 but chirality_z ≤ 1 → CRITICAL."""
        assert classify_phase(5.0, 0.5) == Phase.CRITICAL


# ============================================================================
# Test 6: PhaseSnapshot capture
# ============================================================================


class TestPhaseSnapshot:
    """Verify PhaseSnapshot dataclass integrity."""

    def test_snapshot_order_parameter_abs_nonneg(self, heterogeneous_graph):
        """|⟨𝒮⟩| ≥ 0 always."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.order_parameter_abs >= 0
        assert snap.order_parameter_abs == pytest.approx(abs(snap.order_parameter))

    def test_snapshot_coherence_length_nonneg(self, heterogeneous_graph):
        """ξ_C ≥ 0."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.coherence_length >= 0


# ============================================================================
# Test 7: Time-series phase transition detection
# ============================================================================


class TestPhaseTransitionDetection:
    """Verify detect_phase_transition() on evolving graph sequences."""

    def test_transition_detected_in_diversifying_sequence(self, transition_sample):
        """The declared ramp exposes aligned time-series coordinates."""
        times, tel = transition_sample

        assert isinstance(tel, PhaseTransitionTelemetry)
        assert len(tel.times) == 20
        assert len(tel.order_parameter) == 20
        assert len(tel.phase_classification) == 20
        assert len(tel.order_parameter_abs) == 20
        assert len(tel.chirality_mean) == 20
        assert len(tel.chirality_abs_mean) == 20
        assert len(tel.susceptibility) == 20
        assert len(tel.coherence_length) == 20

    def test_early_steps_non_life(self, transition_sample):
        """First time steps (uniform) → NON_LIFE."""
        times, tel = transition_sample
        # Step 0 is fully uniform
        assert tel.phase_classification[0] == Phase.NON_LIFE

    def test_late_random_heterogeneity_need_not_break_global_symmetry(
        self, transition_sample
    ):
        """Large local diversity may still cancel in signed global means."""
        times, tel = transition_sample
        assert tel.phase_classification[-1] == Phase.NON_LIFE

    def test_order_parameter_increases(self, transition_sample):
        """⟨|𝒮|⟩ increases as diversity grows."""
        times, tel = transition_sample
        # Final order parameter should exceed initial
        assert tel.order_parameter_abs[-1] > tel.order_parameter_abs[0]

    def test_transition_time_exists(self, transition_sample):
        """Transition time is detected (not None)."""
        times, tel = transition_sample
        # Should detect a crossing
        assert tel.transition_time is not None
        assert tel.transition_time >= times[0]
        assert tel.transition_time <= times[-1]

    def test_critical_time_exists(self, transition_sample):
        """Critical time (peak susceptibility) is detected."""
        times, tel = transition_sample
        assert tel.critical_time == times[int(np.argmax(tel.susceptibility))]

    def test_measured_exponent_is_only_exponent(self, transition_sample):
        """Only the MEASURED exponent is stored; no derived 'theoretical' one.

        Audit 2026: the exponent is protocol-dependent (measured), not a
        universal γ/π constant, so theoretical_exponent was removed.
        """
        times, tel = transition_sample
        assert hasattr(tel, "measured_exponent")
        assert not hasattr(tel, "theoretical_exponent")


# ============================================================================
# Test 8: Susceptibility along the declared sampled ramp
# ============================================================================


# ============================================================================
# Test 9: Protocol-dependent effective-exponent fitting
# ============================================================================


class TestEffectiveExponentFit:
    """Verify protocol-dependent effective-exponent estimation."""

    def test_fit_result_has_no_theoretical(self):
        """The fit returns only measured observables (no derived 'theoretical').

        Audit 2026: there is no universal γ/π exponent to compare against.
        """
        result = fit_critical_exponent([0, 1, 2], np.array([0.0, 0.1, 0.5]), None)
        assert "theoretical" not in result
        assert "exponent" in result and "r_squared" in result

    def test_fit_with_insufficient_data_returns_none(self):
        """Too few data points → exponent is None."""
        result = fit_critical_exponent([0, 1], np.array([0.0, 0.1]), 0.5)
        assert result["exponent"] is None
        assert result["r_squared"] is None

    def test_fit_never_substitutes_before_peak_samples(self):
        """Two after-peak points remain insufficient despite earlier data."""
        times = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        order = np.array([9.0, 4.0, 1.0, 0.0, 1.0, 4.0])

        result = fit_critical_exponent(times, order, critical_time=0.0)

        assert result == {"exponent": None, "r_squared": None}

    def test_fit_uses_consistent_neutral_exponent_convention(self):
        times = [0.0, 1.0, 2.0, 4.0]
        order = np.sqrt(np.asarray(times, dtype=float))

        result = fit_critical_exponent(times, order, critical_time=0.0)

        assert result["exponent"] == pytest.approx(0.5)
        assert result["r_squared"] == pytest.approx(1.0)

    def test_constant_postcritical_response_has_perfect_zero_slope_fit(self):
        result = fit_critical_exponent(
            [0.0, 1.0, 2.0, 4.0],
            np.array([0.0, 2.0, 2.0, 2.0]),
            critical_time=0.0,
        )

        assert result["exponent"] == pytest.approx(0.0, abs=1e-14)
        assert result["r_squared"] == pytest.approx(1.0)


# ============================================================================
# Test 10: Multi-topology validation
# ============================================================================


class TestMultiTopology:
    """Phase classification must work across different network topologies."""

    @pytest.mark.parametrize(
        "graph_fn,seed",
        [
            (lambda s: nx.watts_strogatz_graph(20, 4, 0.3, seed=s), 42),
            (lambda s: nx.barabasi_albert_graph(20, 2, seed=s), 42),
            (lambda s: nx.grid_2d_graph(5, 4), 42),
            (lambda s: nx.erdos_renyi_graph(20, 0.3, seed=s), 42),
        ],
        ids=["watts_strogatz", "barabasi_albert", "grid_2d", "erdos_renyi"],
    )
    def test_uniform_is_non_life(self, graph_fn, seed):
        """Uniform initialization → NON_LIFE across topologies."""
        G = graph_fn(seed)
        inject_defaults(G)
        snap = capture_phase_snapshot(G)
        assert snap.phase == Phase.NON_LIFE

    @pytest.mark.parametrize(
        "graph_fn,seed",
        [
            (lambda s: nx.watts_strogatz_graph(30, 4, 0.3, seed=s), 42),
            (lambda s: nx.barabasi_albert_graph(30, 2, seed=s), 42),
            (lambda s: nx.erdos_renyi_graph(30, 0.3, seed=s), 42),
        ],
        ids=["watts_strogatz", "barabasi_albert", "erdos_renyi"],
    )
    def test_selected_heterogeneous_fixtures_cross_order_cut(self, graph_fn, seed):
        """The seeded fixtures cross the operational signed-order cut."""
        rng = np.random.default_rng(seed)
        G = graph_fn(seed)
        inject_defaults(G)
        for node in G.nodes():
            G.nodes[node]["theta"] = rng.uniform(0, 2 * np.pi)
            G.nodes[node]["phase"] = G.nodes[node]["theta"]
            G.nodes[node]["delta_nfr"] = rng.uniform(0.1, 1.5)
        snap = capture_phase_snapshot(G)
        assert snap.phase in (Phase.LIFE, Phase.CRITICAL)
        assert snap.order_parameter_abs > 0


# ============================================================================
# Test 11: Consistency with unified.py field computations
# ============================================================================


class TestUnifiedConsistency:
    """Order parameter and chirality must match unified.py single source of truth."""

    def test_order_parameter_matches_symmetry_breaking_field(self, heterogeneous_graph):
        """compute_order_parameter delegates to unified.compute_symmetry_breaking_field."""
        S_field = compute_symmetry_breaking_field(heterogeneous_graph)
        values = np.array(list(S_field.values()))
        op = compute_order_parameter(heterogeneous_graph)
        assert op["mean"] == pytest.approx(float(np.mean(values)), rel=1e-10)
        assert op["variance"] == pytest.approx(float(np.var(values)), rel=1e-10)


# ============================================================================
# Test 12: Reproducibility under seed control
# ============================================================================


class TestReproducibility:
    """Identical seeds must produce identical phase transition telemetry."""

    def test_snapshot_reproducible(self):
        """Same graph → same snapshot."""
        G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
        inject_defaults(G)
        rng = np.random.default_rng(42)
        for node in G.nodes():
            G.nodes[node]["theta"] = rng.uniform(0, 2 * np.pi)
            G.nodes[node]["phase"] = G.nodes[node]["theta"]
            G.nodes[node]["delta_nfr"] = rng.uniform(0.1, 1.0)

        snap1 = capture_phase_snapshot(G)
        snap2 = capture_phase_snapshot(G)
        assert snap1.order_parameter == snap2.order_parameter
        assert snap1.chirality_mean == snap2.chirality_mean
        assert snap1.phase == snap2.phase

    def test_transition_detection_reproducible(self):
        """Same sequence → same telemetry."""
        g1, t1 = _build_transition_sequence(n_steps=10, seed=77)
        g2, t2 = _build_transition_sequence(n_steps=10, seed=77)
        tel1 = detect_phase_transition(g1, t1)
        tel2 = detect_phase_transition(g2, t2)
        np.testing.assert_array_almost_equal(tel1.order_parameter, tel2.order_parameter)
        assert tel1.transition_time == tel2.transition_time


# ============================================================================
# Test 13: Coherence length behaviour
# ============================================================================


# ============================================================================
# Test 14: Edge cases
# ============================================================================


class TestEdgeCases:
    """Handle degenerate inputs gracefully."""

    def test_single_node_graph(self):
        """Single-node graph should not crash."""
        G = nx.Graph()
        G.add_node(0)
        inject_defaults(G)
        snap = capture_phase_snapshot(G)
        assert snap.phase == Phase.NON_LIFE

    def test_two_node_graph(self):
        """Two-node graph computes without error."""
        G = nx.path_graph(2)
        inject_defaults(G)
        snap = capture_phase_snapshot(G)
        assert isinstance(snap.phase, Phase)

    def test_empty_time_series(self):
        """Zero-length time series returns empty telemetry."""
        tel = detect_phase_transition([], [])
        assert len(tel.times) == 0
        assert tel.transition_time is None
        assert tel.critical_time is None

    def test_single_step_time_series(self):
        """Single-step time series computes without crash."""
        G = nx.watts_strogatz_graph(10, 4, 0.3, seed=42)
        inject_defaults(G)
        tel = detect_phase_transition([G], [0.0])
        assert len(tel.phase_classification) == 1

    def test_time_series_requires_one_time_per_graph(self):
        """A partial time coordinate cannot silently truncate diagnostics."""
        G = nx.path_graph(3)
        inject_defaults(G)
        with pytest.raises(ValueError, match="same number"):
            detect_phase_transition([G, G.copy()], [0.0])

    @pytest.mark.parametrize(
        "times",
        [[0.0, 0.0], [1.0, 0.0], [0.0, float("nan")]],
        ids=["duplicate", "decreasing", "non_finite"],
    )
    def test_time_series_requires_finite_strictly_increasing_times(self, times):
        G = nx.path_graph(3)
        inject_defaults(G)
        with pytest.raises(ValueError):
            detect_phase_transition([G, G.copy()], times)

    def test_classifier_inputs_are_exposed_in_telemetry(self):
        """The values governing labels remain directly auditable."""
        G = nx.path_graph(3)
        inject_defaults(G)
        snapshot = capture_phase_snapshot(G)
        telemetry = detect_phase_transition([G], [0.0])

        assert snapshot.node_count == 3
        assert snapshot.order_zscore == telemetry.order_zscore[0]
        assert snapshot.chirality_zscore == telemetry.chirality_zscore[0]
        assert telemetry.node_count.tolist() == [3]

    @pytest.mark.parametrize(
        "times,order,critical_time,error",
        [
            ([0.0, 1.0], np.array([0.1]), 0.5, "same number"),
            ([0.0, 1.0], np.array([0.1, -0.2]), 0.5, "negative"),
            ([0.0, 1.0], np.array([0.1, np.nan]), 0.5, "finite"),
            ([0.0, 1.0], np.array([0.1, 0.2]), np.nan, "critical_time"),
        ],
    )
    def test_exponent_fit_rejects_invalid_coordinates(
        self, times, order, critical_time, error
    ):
        with pytest.raises(ValueError, match=error):
            fit_critical_exponent(times, order, critical_time)


# ============================================================================
# Test 15: Order parameter statistics correctness
# ============================================================================


class TestOrderParameterStatistics:
    """Verify mathematical correctness of order parameter statistics."""

    def test_n_nodes_correct(self, heterogeneous_graph):
        """n_nodes matches graph size."""
        op = compute_order_parameter(heterogeneous_graph)
        assert op["n_nodes"] == heterogeneous_graph.number_of_nodes()

    def test_abs_mean_geq_abs_of_mean(self, heterogeneous_graph):
        """⟨|𝒮|⟩ ≥ |⟨𝒮⟩| by Jensen's inequality."""
        op = compute_order_parameter(heterogeneous_graph)
        assert op["abs_mean"] >= abs(op["mean"]) - 1e-12

    def test_susceptibility_equals_n_times_var(self, heterogeneous_graph):
        """χ_𝒮 = N · Var(𝒮)."""
        op = compute_order_parameter(heterogeneous_graph)
        expected = op["n_nodes"] * op["variance"]
        assert op["susceptibility"] == pytest.approx(expected, rel=1e-10)


def test_standardized_imbalance_does_not_underflow_its_variance_divisor():
    variance = math.ulp(0.0)
    expected = 1e-162 / math.sqrt(variance) * 2.0
    observed = symmetry_zscore(1e-162, variance, 4)
    assert observed == pytest.approx(expected)
    assert observed < 1.0
    assert classify_phase(observed, 0.0) is Phase.NON_LIFE


@pytest.mark.parametrize("scale", [1.0, 1e-200, math.ulp(0.0)])
def test_field_classification_keeps_small_nonconstant_spread(monkeypatch, scale):
    graph = nx.empty_graph(3)
    inject_defaults(graph)
    monkeypatch.setattr(
        transition_module,
        "compute_symmetry_breaking_field",
        lambda graph: {0: -scale, 1: scale, 2: scale},
    )
    snapshot = capture_phase_snapshot(graph)
    assert snapshot.order_zscore == pytest.approx(math.sqrt(3.0 / 8.0))
    assert snapshot.phase is Phase.NON_LIFE
    if scale < 1e-100:
        assert compute_order_parameter(graph)["variance"] == 0.0


def test_uniform_large_field_mean_is_finite_without_sum_overflow(monkeypatch):
    monkeypatch.setattr(
        transition_module,
        "compute_symmetry_breaking_field",
        lambda graph: {0: 1e308, 1: 1e308, 2: 1e308},
    )
    statistics = compute_order_parameter(nx.empty_graph(3))
    assert statistics["mean"] == statistics["abs_mean"] == 1e308
    assert statistics["variance"] == statistics["susceptibility"] == 0.0
    assert statistics["standardized_imbalance"] == math.inf


def test_field_variance_rejects_unrepresentable_spread(monkeypatch):
    monkeypatch.setattr(
        transition_module,
        "compute_symmetry_breaking_field",
        lambda graph: {0: -1e308, 1: 1e308},
    )
    with pytest.raises(ValueError, match="variance exceeds"):
        compute_order_parameter(nx.path_graph(2))


def test_disconnected_copy_can_change_label_without_local_dynamics():
    graph = nx.path_graph(3)
    inject_defaults(graph)
    for node, phase, pressure in zip(graph, [0.0, 0.2, 0.6], [0.0, 0.3, 0.6]):
        graph.nodes[node][ALIAS_THETA[0]] = phase
        graph.nodes[node][ALIAS_DNFR[0]] = pressure
    duplicated = nx.disjoint_union(graph, graph)
    original = capture_phase_snapshot(graph)
    copies = capture_phase_snapshot(duplicated)
    assert copies.order_parameter == pytest.approx(original.order_parameter)
    assert copies.order_zscore == pytest.approx(math.sqrt(2) * original.order_zscore)
    assert copies.chirality_zscore == pytest.approx(
        math.sqrt(2) * original.chirality_zscore
    )
    assert original.phase is Phase.NON_LIFE
    assert copies.phase is Phase.LIFE


def test_susceptibility_combines_size_before_subnormal_materialization(monkeypatch):
    monkeypatch.setattr(
        transition_module,
        "compute_symmetry_breaking_field",
        lambda graph: {0: 0.0, 1: 2e-162, 2: 0.0, 3: 2e-162},
    )
    statistics = compute_order_parameter(nx.empty_graph(4))
    assert statistics["variance"] == 0.0
    assert statistics["susceptibility"] == math.ulp(0.0)
    assert statistics["standardized_imbalance"] == pytest.approx(2.0)


def test_unavailable_coherence_length_keeps_snapshot_and_series_provenance():
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    snapshot = capture_phase_snapshot(graph)
    telemetry = detect_phase_transition([graph], [0.0])
    assert math.isnan(snapshot.coherence_length)
    assert not snapshot.coherence_length_available
    assert snapshot.coherence_length_provenance.method == "unavailable"
    assert math.isnan(telemetry.coherence_length[0])
    assert telemetry.coherence_length_available.tolist() == [False]
    assert telemetry.coherence_length_provenance[0].method == "unavailable"


@pytest.mark.parametrize(
    "times,values,expected",
    [
        ([0.0, 1.0, 2.0], [2.0, 0.0, 2.0], 0.0),
        ([0.0], [2.0], 0.0),
        ([0.0, 3.0], [math.nextafter(1.0, 0.0), math.nextafter(1.0, 2.0)], 1.0),
        ([-1e308, 1e308], [0.0, 2.0], 0.0),
    ],
)
def test_first_observed_crossing_retains_initial_occupation_and_finite_time(
    times, values, expected
):
    assert transition_module._find_crossing_time(
        times, np.array(values), 1.0
    ) == pytest.approx(expected)


def test_exponent_fit_has_no_fixed_time_or_amplitude_cutoff():
    result = fit_critical_exponent(
        [1e-14, 2e-14, 3e-14, 4e-14],
        np.array([1e-16, 2e-16, 3e-16, 4e-16]),
        critical_time=0.0,
    )
    assert result["exponent"] == pytest.approx(1.0)
    assert result["r_squared"] == pytest.approx(1.0)


def test_exponent_fit_retains_imperfect_nearly_constant_log_response():
    result = fit_critical_exponent(
        [1.0, 2.0, 4.0], np.array([1.0, 1.0, 1.0 + 1e-8]), critical_time=0.0
    )
    assert result["r_squared"] == pytest.approx(0.75)


def test_exponent_fit_handles_a_finite_log_of_an_overflowing_time_distance():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    result = fit_critical_exponent(
        [0.0, largest / 2.0, largest],
        np.array([1.0, 1.5, 2.0]),
        critical_time=-largest,
    )
    assert result["exponent"] == pytest.approx(1.0)
    assert result["r_squared"] == pytest.approx(1.0)


@pytest.mark.parametrize("invalid", [True, "1", 1 + 0j])
def test_transition_public_controls_reject_coerced_labels(invalid):
    with pytest.raises((TypeError, ValueError)):
        symmetry_zscore(invalid, 1.0, 2)
    with pytest.raises((TypeError, ValueError)):
        classify_phase(invalid, 0.0)
    with pytest.raises((TypeError, ValueError)):
        fit_critical_exponent([0.0, 1.0, 2.0], [0.0, invalid, 2.0], 0.0)
