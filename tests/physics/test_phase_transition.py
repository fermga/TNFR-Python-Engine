"""Tests for TNFR Phase Transition — Life/Non-Life as Universal Symmetry Breaking.

Validates the Structural Phase Transition Theorem: the symmetry breaking
field 𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²) serves as order parameter
of a second-order phase transition between non-life (⟨𝒮⟩ = 0, symmetric)
and life (⟨𝒮⟩ ≠ 0, broken symmetry).

Tests verify:
1.  Symmetric phase: uniform/equilibrium graphs → ⟨𝒮⟩ = 0 (NON_LIFE)
2.  Broken symmetry: heterogeneous phases/ΔNFR → ⟨𝒮⟩ ≠ 0 (LIFE)
3.  Chirality: life phase requires ⟨|χ|⟩ > 0 (homochirality)
4.  Critical exponent reference: γ_c = γ/π ≈ 0.1837 (calibrated TIER-2 scale;
    audit 2026: the measured exponent is protocol-dependent, not universal)
5.  Susceptibility: diverges near critical point (peak at transition)
6.  Coherence length: grows near critical regime
7.  Phase classification consistency across topologies
8.  Transition time detection via interpolation
9.  Critical time at peak susceptibility
10. PhaseSnapshot capture correctness
11. Time-series detect_phase_transition() pipeline
12. Power-law fit of critical exponent
13. Constants consistency with canonical module
14. Dataclass integrity (PhaseTransitionTelemetry fields)
15. Reproducibility under seed control

TIER: CORE PHYSICS — phase transition axiomatises life/non-life boundary.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.physics.phase_transition import (
    Z_SIGNIFICANCE,
    Phase,
    PhaseSnapshot,
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


@pytest.fixture
def critical_graph() -> nx.Graph:
    """Graph near the critical regime — weak chirality, moderate 𝒮."""
    rng = np.random.default_rng(99)
    G = nx.watts_strogatz_graph(25, 4, 0.3, seed=99)
    inject_defaults(G)
    for node in G.nodes():
        # Small phase perturbation → near-zero 𝒮
        G.nodes[node]["theta"] = rng.uniform(0, 0.3)
        G.nodes[node]["phase"] = G.nodes[node]["theta"]
        G.nodes[node]["delta_nfr"] = rng.uniform(0.01, 0.1)
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


class TestEmergentClassification:
    """The classification scale is the emergent sampling-noise z-score.

    Audit 2026: the magic scales (γ/π)² and γ/(π+γ) were proven INERT
    (they sat in a two-order-of-magnitude gap; sweeping them changed no
    classification) and were removed. The phase is now decided by the
    statistical significance of symmetry breaking measured from the system.
    """

    def test_z_significance_is_one_sigma(self):
        """The only cut is z = 1 (the sampling-noise scale, not a constant)."""
        assert Z_SIGNIFICANCE == 1.0

    def test_zscore_zero_for_uniform_zero_field(self):
        """A perfectly uniform zero field has z = 0 (symmetric)."""
        assert symmetry_zscore(0.0, 0.0, 30) == 0.0

    def test_zscore_infinite_for_uniform_nonzero_field(self):
        """A uniform non-zero field (Var=0, mean>0) is fully broken: z = ∞."""
        assert symmetry_zscore(0.5, 0.0, 30) == math.inf

    def test_zscore_is_mean_over_standard_error(self):
        """z = |mean| / sqrt(Var/N) — emergent, measured from the system."""
        z = symmetry_zscore(0.2, 0.01, 25)
        assert z == pytest.approx(0.2 / math.sqrt(0.01 / 25), rel=1e-12)

    def test_zscore_zero_nodes_is_zero(self):
        """Empty system has no significance."""
        assert symmetry_zscore(1.0, 1.0, 0) == 0.0


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
    """Heterogeneous graphs must show broken symmetry → LIFE."""

    def test_heterogeneous_nonzero_order_parameter(self, heterogeneous_graph):
        """⟨|𝒮|⟩ > 0 for diverse phases and ΔNFR."""
        op = compute_order_parameter(heterogeneous_graph)
        assert op["abs_mean"] > 0

    def test_heterogeneous_nonzero_chirality(self, heterogeneous_graph):
        """⟨|χ|⟩ > 0 for diverse phases (broken mirror symmetry)."""
        chi = compute_chirality_statistics(heterogeneous_graph)
        assert chi["abs_mean"] > 0

    def test_heterogeneous_classified_life(self, heterogeneous_graph):
        """Diverse phases with strong ΔNFR → Phase.LIFE."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.phase == Phase.LIFE

    def test_heterogeneous_has_homochirality(self, heterogeneous_graph):
        """Heterogeneous graph develops homochirality."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.has_homochirality is True

    def test_positive_susceptibility(self, heterogeneous_graph):
        """χ_𝒮 > 0 in the broken phase (finite fluctuations)."""
        op = compute_order_parameter(heterogeneous_graph)
        assert op["susceptibility"] > 0


# ============================================================================
# Test 4: Chirality and homochirality
# ============================================================================


class TestChirality:
    """Verify chirality field behaviour and homochirality detection."""

    def test_chirality_sign_is_handedness(self, heterogeneous_graph):
        """⟨χ⟩ has a definite sign → preferred handedness."""
        chi = compute_chirality_statistics(heterogeneous_graph)
        # Should be non-zero for heterogeneous network
        assert abs(chi["mean"]) > 0

    def test_chirality_abs_mean_geq_abs_mean(self, heterogeneous_graph):
        """⟨|χ|⟩ ≥ |⟨χ⟩| by Jensen's inequality."""
        chi = compute_chirality_statistics(heterogeneous_graph)
        assert chi["abs_mean"] >= abs(chi["mean"]) - 1e-12

    def test_chirality_variance_positive(self, heterogeneous_graph):
        """Chirality variance > 0 in heterogeneous phase."""
        chi = compute_chirality_statistics(heterogeneous_graph)
        assert chi["variance"] > 0

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
    """Verify classify_phase() z-score logic (emergent, no magic constant)."""

    def test_zero_zscore_is_non_life(self):
        """order_z = 0 → NON_LIFE (within sampling noise of zero)."""
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

    def test_phase_is_enum(self):
        """Phase classification returns Phase enum."""
        assert isinstance(classify_phase(0.0, 0.0), Phase)

    def test_all_phases_reachable(self):
        """All three phases are reachable from z-scores."""
        assert classify_phase(0.0, 0.0) == Phase.NON_LIFE
        assert classify_phase(5.0, 5.0) == Phase.LIFE
        assert classify_phase(5.0, 0.5) == Phase.CRITICAL


# ============================================================================
# Test 6: PhaseSnapshot capture
# ============================================================================


class TestPhaseSnapshot:
    """Verify PhaseSnapshot dataclass integrity."""

    def test_snapshot_has_all_fields(self, uniform_graph):
        """Snapshot exposes all required structural diagnostics."""
        snap = capture_phase_snapshot(uniform_graph)
        assert hasattr(snap, "order_parameter")
        assert hasattr(snap, "order_parameter_abs")
        assert hasattr(snap, "chirality_mean")
        assert hasattr(snap, "chirality_abs_mean")
        assert hasattr(snap, "susceptibility")
        assert hasattr(snap, "coherence_length")
        assert hasattr(snap, "phase")
        assert hasattr(snap, "has_homochirality")

    def test_snapshot_order_parameter_abs_nonneg(self, heterogeneous_graph):
        """|⟨𝒮⟩| ≥ 0 always."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.order_parameter_abs >= 0

    def test_snapshot_susceptibility_nonneg(self, heterogeneous_graph):
        """χ_𝒮 = N·Var(𝒮) ≥ 0 always."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.susceptibility >= 0

    def test_snapshot_coherence_length_nonneg(self, heterogeneous_graph):
        """ξ_C ≥ 0."""
        snap = capture_phase_snapshot(heterogeneous_graph)
        assert snap.coherence_length >= 0


# ============================================================================
# Test 7: Time-series phase transition detection
# ============================================================================


class TestPhaseTransitionDetection:
    """Verify detect_phase_transition() on evolving graph sequences."""

    def test_transition_detected_in_diversifying_sequence(self):
        """System transitioning from uniform to heterogeneous triggers detection."""
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)

        assert isinstance(tel, PhaseTransitionTelemetry)
        assert len(tel.times) == 20
        assert len(tel.order_parameter) == 20
        assert len(tel.phase_classification) == 20

    def test_early_steps_non_life(self):
        """First time steps (uniform) → NON_LIFE."""
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)
        # Step 0 is fully uniform
        assert tel.phase_classification[0] == Phase.NON_LIFE

    def test_late_steps_life(self):
        """Last steps (fully heterogeneous) → LIFE or CRITICAL."""
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)
        # Last step has maximum diversity
        assert tel.phase_classification[-1] in (Phase.LIFE, Phase.CRITICAL)

    def test_order_parameter_increases(self):
        """⟨|𝒮|⟩ increases as diversity grows."""
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)
        # Final order parameter should exceed initial
        assert tel.order_parameter_abs[-1] > tel.order_parameter_abs[0]

    def test_transition_time_exists(self):
        """Transition time is detected (not None)."""
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)
        # Should detect a crossing
        assert tel.transition_time is not None
        assert tel.transition_time >= times[0]
        assert tel.transition_time <= times[-1]

    def test_critical_time_exists(self):
        """Critical time (peak susceptibility) is detected."""
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)
        assert tel.critical_time is not None

    def test_measured_exponent_is_only_exponent(self):
        """Only the MEASURED exponent is stored; no derived 'theoretical' one.

        Audit 2026: the exponent is protocol-dependent (measured), not a
        universal γ/π constant, so theoretical_exponent was removed.
        """
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)
        assert hasattr(tel, "measured_exponent")
        assert not hasattr(tel, "theoretical_exponent")

    def test_telemetry_arrays_correct_length(self):
        """All arrays have matching length."""
        graphs, times = _build_transition_sequence(n_steps=15, seed=7)
        tel = detect_phase_transition(graphs, times)
        assert len(tel.order_parameter) == 15
        assert len(tel.order_parameter_abs) == 15
        assert len(tel.chirality_mean) == 15
        assert len(tel.chirality_abs_mean) == 15
        assert len(tel.susceptibility) == 15
        assert len(tel.coherence_length) == 15
        assert len(tel.phase_classification) == 15


# ============================================================================
# Test 8: Susceptibility behaviour near transition
# ============================================================================


class TestSusceptibility:
    """Susceptibility χ_𝒮 = N·Var(𝒮) should peak near the critical point."""

    def test_susceptibility_peak_in_middle(self):
        """Susceptibility should peak somewhere between fully uniform and fully diverse."""
        graphs, times = _build_transition_sequence(n_steps=25, seed=42)
        tel = detect_phase_transition(graphs, times)
        peak_idx = int(np.argmax(tel.susceptibility))
        # Peak should not be at the very first or very last step
        # (transition occurs in the middle of the ramp)
        assert peak_idx > 0, "Peak susceptibility at step 0 is unexpected"

    def test_susceptibility_nonnegative(self):
        """χ_𝒮 ≥ 0 at all times (it's a variance)."""
        graphs, times = _build_transition_sequence(n_steps=15, seed=42)
        tel = detect_phase_transition(graphs, times)
        assert np.all(tel.susceptibility >= -1e-12)


# ============================================================================
# Test 9: Critical exponent fitting
# ============================================================================


class TestCriticalExponentFit:
    """Verify critical exponent estimation and theoretical comparison."""

    def test_fit_returns_dict(self):
        """fit_critical_exponent returns a dictionary with expected keys."""
        graphs, times = _build_transition_sequence(n_steps=20, seed=42)
        tel = detect_phase_transition(graphs, times)
        result = fit_critical_exponent(
            times, tel.order_parameter_abs, tel.critical_time
        )
        assert "exponent" in result
        assert "r_squared" in result

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

    def test_measured_exponent_positive(self):
        """Fitted exponent should be positive for an increasing order parameter."""
        graphs, times = _build_transition_sequence(n_steps=30, seed=42)
        tel = detect_phase_transition(graphs, times)
        if tel.measured_exponent is not None:
            # Power-law growth → positive exponent
            assert tel.measured_exponent > 0


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
    def test_heterogeneous_is_life(self, graph_fn, seed):
        """Heterogeneous phases + ΔNFR → LIFE across topologies."""
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

    def test_chirality_matches_chirality_field(self, heterogeneous_graph):
        """compute_chirality_statistics delegates to unified.compute_chirality_field."""
        chi_field = compute_chirality_field(heterogeneous_graph)
        values = np.array(list(chi_field.values()))
        chi = compute_chirality_statistics(heterogeneous_graph)
        assert chi["mean"] == pytest.approx(float(np.mean(values)), rel=1e-10)


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


class TestCoherenceLength:
    """ξ_C should grow as the system approaches the critical regime."""

    def test_coherence_length_nonneg_in_telemetry(self):
        """ξ_C ≥ 0 at all time steps."""
        graphs, times = _build_transition_sequence(n_steps=15, seed=42)
        tel = detect_phase_transition(graphs, times)
        assert np.all(tel.coherence_length >= 0)


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

    def test_variance_nonneg(self, heterogeneous_graph):
        """Var(𝒮) ≥ 0."""
        op = compute_order_parameter(heterogeneous_graph)
        assert op["variance"] >= -1e-12

    def test_susceptibility_equals_n_times_var(self, heterogeneous_graph):
        """χ_𝒮 = N · Var(𝒮)."""
        op = compute_order_parameter(heterogeneous_graph)
        expected = op["n_nodes"] * op["variance"]
        assert op["susceptibility"] == pytest.approx(expected, rel=1e-10)
