#!/usr/bin/env python3
"""Example 159 — the empirical-confrontation pipeline (TNFR-IA -> engine).

Promotes the TNFR-IA empirical arm's workflow to a runnable engine example:
map an objective multichannel signal onto the canonical TNFR magnitudes and
then ask the decisive question the two-face machinery answers — *which face is
the data on?* — using only engine primitives.

Pipeline (all canonical, engine-native):
  signal  -> analytic phase/amplitude        (phase_amplitude_matrices)
          -> emergent phase-locking graph     (build_coupling_graph, U3/PLV)
          -> emergent geometry (L_rw modes)   (compute_emergent_pulse)
          -> xi_C decay fit / spectral fallback
                                      (estimate_coherence_length_with_provenance)
          -> the two-face diagnosis           (verify_overdamped_projection,
                                               damped_wave_rates)

The confrontation (measured here):
  M1  Canonical read-outs of the data: the emergent pulse (fundamental
      omega = sqrt(lambda_2), dominant beat, vibration energy), the fitted
      coherence length xi_C from a correlation-decay fit when identifiable,
      with the graph-spectral scale 1/sqrt(lambda_2) as fallback, plus the
      Kuramoto order R and local phase fields.
  M2  The two-face dial on the *same* emergent geometry: at large damping the
      engine certificate verify_overdamped_projection is VALID (the diffusive /
      over-damped face); below the critical damping gamma_c = 2*sqrt(lambda_max)
      the modes are under-damped (oscillatory / conservative face).
  M3  The cross-program face map. A persistent-rhythm (oscillatory) signal lives
      on the UNDER-DAMPED conservative face — the TNFR-IA finding that real EEG
      sits there, its local tetrad carrying state. The *linear* part of
      Navier-Stokes lives on the OVER-DAMPED diffusive face (the re-founded NS
      program). The same engine certificate places each — the theory is
      falsifiable against data because the face is *measured*, not assumed.

HONEST SCOPE: this is the empirical *instrument*, not a competitive model.
On accessible-signal tasks the canonical read-outs tie simple baselines
(see the TNFR-IA findings); their value is canonical, emergent parsimony and
interpretability. Closes no open problem.

Run:
    python examples/10_applications/159_empirical_confrontation_pipeline.py
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - best-effort console setup
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

from tnfr.physics.fields import (  # noqa: E402
    estimate_coherence_length_with_provenance,
)
from tnfr.validation import (  # noqa: E402
    build_coupling_graph,
    confront_signal,
    nodal_prediction_skill,
    phase_amplitude_matrices,
)

K_NEIGHBOURS = 4


def coupled_oscillators(
    *, n_channels: int = 12, n_samples: int = 2048, fs: float = 64.0,
    spread: float = 0.05, seed: int = 0,
) -> np.ndarray:
    """A coherent multichannel oscillator ensemble (a stand-in real signal)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    base = 2.0 * np.pi * 6.0 * t
    return np.array(
        [np.sin(base + rng.normal(0.0, spread)) for _ in range(n_channels)]
    )


def load_signal() -> tuple[np.ndarray, str]:
    """Bring your own real signal: ``python 159_...py path/to/signal.npy``.

    The array must be shape ``(n_channels, n_samples)`` -- EEG, grid telemetry,
    any real multichannel measurement. With no argument a synthetic coherent
    oscillator ensemble stands in.
    """
    if len(sys.argv) > 1:
        arr = np.load(sys.argv[1])
        return np.asarray(arr, dtype=float), sys.argv[1]
    return coupled_oscillators(), "synthetic coupled oscillators"


def main() -> None:
    print("=" * 72)
    print("EMPIRICAL CONFRONTATION -- canonical multichannel read-outs")
    print("=" * 72)
    signals, source = load_signal()
    print(f"\nsource: {source}  "
          f"(shape {signals.shape[0]}ch x {signals.shape[1]})")

    # The confrontation report includes the emergent geometry and its spectral
    # xi_C comparison. Rebuild the deterministic coupling graph to request the
    # primary decay-fit estimator together with explicit provenance.
    rep = confront_signal(signals, k_neighbours=K_NEIGHBOURS)
    phase, amplitude = phase_amplitude_matrices(signals)
    coupling_graph = build_coupling_graph(
        phase, amplitude, k_neighbours=K_NEIGHBOURS
    )
    xi_estimate = estimate_coherence_length_with_provenance(coupling_graph)

    print("\nCanonical read-outs of the emergent coupling graph:")
    print(f"   Kuramoto R           = {rep.kuramoto_R:.3f}")
    print(f"   coherence C          = {rep.coherence:.3f}  "
          f"(at ΔNFR=0 attractor: {rep.at_equilibrium})")
    print(
        f"   tetrad |∇φ|,|K_φ|    = {rep.grad_phi:.3f}, {rep.k_phi:.3f}  "
        "(local phase-field summaries)"
    )
    print(f"   Φ_s                   = {rep.phi_s:.3f}")
    print(
        f"   ξ_C estimate          = {xi_estimate.value:.3f}  "
        f"(method={xi_estimate.method}, fit={xi_estimate.fit_available})"
    )
    print(f"                          {xi_estimate.fit_quality}")
    print(
        f"   ξ_C spectral compare = {rep.xi_c:.3f}  "
        "(1/sqrt(lambda_2) on the same coupling graph)"
    )
    print(f"   pulse ω₀, beat, E    = {rep.pulse_fundamental:.3f}, "
          f"{rep.dominant_beat:.3f}, {rep.vibration_energy:.3f}")
    face = ("DIFFUSIVE (over-damped)" if rep.diffusive_face_valid
            else "WAVE (under-damped)")
    print(f"   face (emergent)      = {face}  "
          f"(wave_frac = {rep.wave_fraction:.2f}, Q = {rep.quality_factor:.2f})")

    dyn = nodal_prediction_skill(signals)
    print(f"   nodal 1-step skill   = {dyn.nodal_skill:+.3f} vs AR-1 "
          f"{dyn.ar1_skill:+.3f}")
    print("                          (the evolution law ∂EPI/∂t=ν_f·ΔNFR, "
          "diffusion-selective)")

    print("\nThe cross-program face map (the face is MEASURED, not assumed):")
    print("   * oscillatory data (persistent rhythm) -> UNDER-DAMPED")
    print("     conservative face  [TNFR-IA: real EEG lives here; the local")
    print("     tetrad |∇φ|,|K_φ| carries seizure state cross-patient, AUC~0.67]")
    print("   * linear Navier-Stokes (first-order viscous) -> OVER-DAMPED")
    print("     diffusive face  [blow-up = the NONLINEAR K_φ cascade]")

    print("\n" + "=" * 72)
    print(
        "VERDICT: the engine maps an admissible multichannel signal to canonical\n"
        "magnitudes, while xi_C reports a decay fit or its explicit spectral\n"
        "fallback. This produces testable read-outs for comparison with data.\n"
        "Canonical, parsimonious, data-agnostic; it\n"
        "does not claim to beat task-specific baselines. See\n"
        "docs/EMPIRICAL_CONFRONTATION_EEG.md. Closes nothing."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
