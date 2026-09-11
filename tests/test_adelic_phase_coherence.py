"""Adelic phase order is normalized independently of state-vector norm."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.dynamics.adelic import AdelicDynamics, AdelicState


def test_aligned_unit_norm_amplitudes_have_unit_phase_order() -> None:
    amplitudes = np.ones(4, dtype=complex) / np.sqrt(4.0)
    state = AdelicState(primes=[2, 3, 5, 7], amplitudes=amplitudes)

    assert state.phase_coherence == pytest.approx(1.0)
    assert state.coherence == pytest.approx(state.phase_coherence)


def test_phase_order_is_amplitude_scale_invariant_and_detects_cancellation() -> None:
    aligned = AdelicState(primes=[2, 3], amplitudes=np.array([2.0, 2.0j]))
    rescaled = AdelicState(primes=[2, 3], amplitudes=7.0 * aligned.amplitudes)
    cancelled = AdelicState(primes=[2, 3], amplitudes=np.array([1.0, -1.0]))

    assert rescaled.phase_coherence == pytest.approx(aligned.phase_coherence)
    assert cancelled.phase_coherence == pytest.approx(0.0)


def test_resonance_trajectory_labels_phase_order_explicitly() -> None:
    trajectory = AdelicDynamics(max_prime=5).run_resonance_search(
        start_t=0.0, end_t=0.2, dt=0.1
    )

    assert trajectory["phase_coherence"] == trajectory["coherence"]
    assert trajectory["phase_coherence"][0] == pytest.approx(1.0)