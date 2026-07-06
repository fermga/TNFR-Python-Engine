"""Tests for the data-agnostic signal confronter (the empirical arm).

confront_signal reads ANY real multichannel signal through the emergent
phase-locking geometry (L_rw) and the universal coherence kernel; the coherence
length uses the emergent spectral gap (never nan on a valid graph).
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics.canonical import estimate_coherence_length
from tnfr.validation import SignalConfrontation, confront_signal
from tnfr.validation.multichannel_interface import (
    build_coupling_graph,
    phase_amplitude_matrices,
)


def _coherent_signal(n_channels: int = 12, n_samples: int = 1024, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 64.0
    base = 2.0 * np.pi * 6.0 * t
    return np.array(
        [np.sin(base + rng.normal(0.0, 0.05)) for _ in range(n_channels)]
    )


def test_confront_signal_returns_canonical_readouts():
    rep = confront_signal(_coherent_signal())
    assert isinstance(rep, SignalConfrontation)
    assert rep.n_channels == 12 and rep.n_samples == 1024
    assert 0.0 < rep.coherence <= 1.0
    assert rep.pulse_fundamental > 0.0
    # xi_C is finite via the emergent spectral gap (never nan on a valid graph)
    assert rep.xi_c == rep.xi_c and rep.xi_c > 0.0
    assert isinstance(rep.diffusive_face_valid, bool)
    assert isinstance(rep.at_equilibrium, bool)
    assert "SignalConfrontation" in rep.summary()


def test_confront_signal_rejects_too_few_channels():
    with pytest.raises(ValueError):
        confront_signal(np.zeros((2, 512)))


def test_estimate_coherence_length_robust_on_uniform_field():
    # A uniformly coherent field degenerates the autocorrelation exp-decay fit
    # (nan); the public estimator falls back to the EMERGENT spectral gap
    # 1/sqrt(lambda_2), so xi_C reads the emergent geometry and is never nan.
    sig = _coherent_signal()
    phase, amp = phase_amplitude_matrices(sig)
    graph = build_coupling_graph(phase, amp, k_neighbours=4)
    xi = estimate_coherence_length(graph)
    assert xi == xi and xi > 0.0  # finite, positive (not nan)


def _relaxational_signal(
    n_channels: int = 12, n_samples: int = 1024, seed: int = 1
):
    # Brownian drift: power decays monotonically from low frequency, with no
    # resonant peak -> over-damped / relaxational (Q small).
    rng = np.random.default_rng(seed)
    return np.cumsum(
        rng.normal(0.0, 1.0, size=(n_channels, n_samples)), axis=1
    )


def test_quality_factor_discriminates_wave_from_relaxation():
    from tnfr.validation import estimate_quality_factor

    q_osc = estimate_quality_factor(_coherent_signal())
    q_relax = estimate_quality_factor(_relaxational_signal())
    assert q_osc > 0.5      # a clean oscillation is under-damped
    assert q_osc > q_relax  # sharper than a relaxational drift


def _travelling_wave(
    n_channels: int = 12, n_samples: int = 1024, seed: int = 0
):
    # spatially-structured oscillation (a phase ramp across channels): the
    # emergent modes carry the oscillation -> WAVE face.  (A rank-1 uniform
    # sinusoid would put all oscillatory energy in the trivial mode.)
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 64.0
    return np.array([
        np.sin(2 * np.pi * 6.0 * t + 0.6 * k + rng.normal(0.0, 0.02))
        for k in range(n_channels)
    ])


def test_confront_signal_emergent_wave_face():
    # A spatially-structured oscillation reads as the conservative WAVE face
    # via the EMERGENT modal dynamics (not the raw input spectrum).
    rep = confront_signal(_travelling_wave())
    assert rep.wave_fraction > 0.5
    assert rep.diffusive_face_valid is False


def test_confront_signal_emergent_diffusive_face():
    # A relaxational (Brownian) field reads as the over-damped DIFFUSIVE face
    # -- the discrimination the input-spectrum Q could not certify.
    rep = confront_signal(_relaxational_signal())
    assert rep.wave_fraction <= 0.5
    assert rep.diffusive_face_valid is True


def test_emergent_wave_fraction_discriminates():
    from tnfr.validation import emergent_wave_fraction

    assert emergent_wave_fraction(_travelling_wave()) > 0.5
    assert emergent_wave_fraction(_relaxational_signal()) <= 0.5


def _ring_diffusion(n: int = 16, T: int = 2000, c0: float = 0.3, seed: int = 0):
    # a genuine graph-diffusion field on a ring: the nodal equation's own
    # dynamics, so the one-step nodal predictor should recover it.
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    for i in range(n):
        w[i, (i + 1) % n] = w[i, (i - 1) % n] = 1.0
    lrw = np.eye(n) - w / w.sum(axis=1, keepdims=True)
    x = np.zeros((n, T))
    x[:, 0] = rng.normal(0.0, 1.0, n)
    for t in range(1, T):
        x[:, t] = x[:, t - 1] - c0 * (lrw @ x[:, t - 1]) \
            + 0.02 * rng.normal(0.0, 1.0, n)
    return x


def test_nodal_prediction_skill_on_diffusion():
    # A genuine graph-diffusion field: the nodal one-step predictor explains
    # increment variance beyond persistence AND beats per-channel AR-1.
    from tnfr.validation import nodal_prediction_skill

    rep = nodal_prediction_skill(_ring_diffusion())
    assert rep.beats_persistence
    assert rep.nodal_skill > 0.05
    assert rep.nodal_skill > rep.ar1_skill
    assert "NodalPredictionSkill" in rep.summary()


def test_nodal_prediction_skill_null_on_independent():
    # Independent per-channel AR-1 (no inter-channel coupling): the nodal
    # diffusion term carries little one-step skill.
    from tnfr.validation import nodal_prediction_skill

    rng = np.random.default_rng(3)
    n, T = 12, 2000
    x = np.zeros((n, T))
    for t in range(1, T):
        x[:, t] = 0.9 * x[:, t - 1] + rng.normal(0.0, 1.0, n)
    rep = nodal_prediction_skill(x)
    assert rep.nodal_skill < 0.1
