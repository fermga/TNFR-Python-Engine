#!/usr/bin/env python3
"""Example 159 — the empirical-confrontation pipeline (TNFR-IA -> engine).

Map a multichannel signal onto scoped graph read-outs and retrospective
modal AR(2) root statistics. A phase-locking graph is an observational
construction, not a verified canonical coupling or physical pressure law.

Pipeline (scoped, engine-native):
  signal  -> analytic phase/amplitude        (phase_amplitude_matrices)
          -> phase-locking graph               (build_coupling_graph, PLV)
          -> emergent geometry (L_rw modes)   (compute_emergent_pulse)
          -> xi_C decay fit / spectral fallback
                                      (estimate_coherence_length_with_provenance)
          -> modal-root diagnostic             (confront_signal)

The confrontation (measured here):
  M1  Scoped read-outs of the data: the emergent pulse (fundamental
      omega = sqrt(lambda_2), dominant beat, vibration energy), the fitted
      coherence length xi_C from a correlation-decay fit when identifiable,
      with the graph-spectral scale 1/sqrt(lambda_2) as fallback, plus the
      Kuramoto order R and local phase fields.
  M2  Real/complex fitted roots and their moduli are separate diagnostics.
      Real roots may grow, and complex roots may decay. Short, constant or
      degenerate data abstain; computational failures retain a reason.
  M3  The legacy nodal/AR-1 scores fit and evaluate the same window. Positive
      fitted skill is not held-out forecasting evidence or model admission.

This example certifies no physical diffusion, conservation, stability or
experimental TNFR law. Diagnostic transforms use the supplied full signal
window. Separate Euler and continuous P2 software fixtures demonstrate
pre-evaluation forecasts without admitting that signal as a physical model.

Run:
    python examples/10_applications/159_empirical_confrontation_pipeline.py
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - best-effort console setup
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

from tnfr.physics.fields import estimate_coherence_length_with_provenance  # noqa: E402
from tnfr.validation import (  # noqa: E402
    P2MeasurementBounds,
    build_coupling_graph,
    calibrate_p2_transport,
    confront_signal,
    forecast_p2_transport,
    nodal_prediction_skill,
    phase_amplitude_matrices,
    score_p2_transport,
    write_p2_transport_forecast,
)
from tnfr.validation.nodal_prediction import (  # noqa: E402
    NodalMeasurementRun,
    calibrate_nodal_prediction,
    forecast_nodal_response,
    score_nodal_forecast,
    write_nodal_forecast,
)

K_NEIGHBOURS = 4


def coupled_oscillators(
    *,
    n_channels: int = 12,
    n_samples: int = 2048,
    fs: float = 64.0,
    spread: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic phase-offset sinusoids, not a TNFR trajectory or real data."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    base = 2.0 * np.pi * 6.0 * t
    return np.array([np.sin(base + rng.normal(0.0, spread)) for _ in range(n_channels)])


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


def demonstrate_reserved_prediction() -> None:
    """Exercise the forecast boundary on an independent P2 Euler formula.

    This nine-sample software fixture uses the exact-model antisymmetric
    factor ``1-2*capacity*dt``. It is not an experiment, acquired measurement
    or an admitted physical model for the signal loaded above.
    """
    import networkx as nx

    times = tuple(float(i / 8) for i in range(9))
    factor = 1 - 2 * 0.4 / 8
    training_amplitude = 0.7 * factor ** np.arange(9)
    calibration_run = NodalMeasurementRun(
        run_id="synthetic-calibration",
        acquisition_id="synthetic-calibration-preparation",
        channel_ids=("left", "right"),
        timestamps=times,
        samples=(1 + training_amplitude, 1 - training_amplitude),
        value_unit="fixture_units",
        time_unit="fixture_seconds",
    )
    calibration = calibrate_nodal_prediction(
        [calibration_run],
        graph=nx.Graph([("left", "right")]),
        offsets=(0.0, 0.0),
        scales=(1.0, 1.0),
        structural_time_per_unit=1.0,
        support_provenance="independently declared synthetic P2 edge",
        measurement_provenance="software unit map; no physical instrument",
    )
    # Only the reserved initialization and schedule are available at issuance.
    forecast = forecast_nodal_response(
        calibration,
        evaluation_run_id="synthetic-reserved",
        evaluation_acquisition_id="synthetic-reserved-preparation",
        initial_measurement=(1.5, 2.5),
        timestamps=times,
        absolute_error_bound=1e-12,
        max_structural_step=0.125,
        max_steps=8,
    )
    issued_hash = forecast.content_hash
    with TemporaryDirectory(prefix="tnfr-forecast-demo-") as directory:
        write_nodal_forecast(forecast, Path(directory) / "forecast.json")
        # Materialize the reserved response only AFTER issuing/saving forecast.
        reserved_amplitude = -0.5 * factor ** np.arange(9)
        observation = NodalMeasurementRun(
            run_id="synthetic-reserved",
            acquisition_id="synthetic-reserved-preparation",
            channel_ids=("left", "right"),
            timestamps=times,
            samples=(2 + reserved_amplitude, 2 - reserved_amplitude),
            value_unit="fixture_units",
            time_unit="fixture_seconds",
        )
        score = score_nodal_forecast(
            forecast,
            calibration,
            observation,
            expected_forecast_hash=issued_hash,
        )
    print("\nSeparate synthetic P2 forecast-boundary demonstration:")
    print(f"   frozen capacity       = {calibration.capacity:.6f}")
    print(f"   issued forecast hash  = {forecast.content_hash}")
    print(f"   reserved max error    = {score.max_absolute_error:.3e}")
    print(f"   declared budget met   = {score.meets_declared_error_bound}")
    print(f"   physical status       = {score.physical_status}")


def demonstrate_continuous_p2_intervals() -> None:
    """Compare an independently generated continuous software fixture.

    Decimal exponentials generate observations independently of the owner's
    rational log/exp enclosures. The declared 1e-10 coordinate bound covers
    conversion of these small fixture values to binary64; it is not an
    instrument calibration or a physical measurement error estimate.
    """
    import networkx as nx

    def continuous_samples(mean: str, amplitude: str, times):
        with localcontext() as context:
            context.prec = 90
            rate = Decimal("0.4")
            center, initial = Decimal(mean), Decimal(amplitude)
            contrasts = [
                initial * (-2 * rate * Decimal.from_float(time)).exp() for time in times
            ]
            return (
                tuple(float(center + value) for value in contrasts),
                tuple(float(center - value) for value in contrasts),
            )

    calibration_times = (0.0, 1.0)
    calibration_run = NodalMeasurementRun(
        run_id="continuous-calibration",
        acquisition_id="continuous-calibration-preparation",
        channel_ids=("left", "right"),
        timestamps=calibration_times,
        samples=continuous_samples("1", "0.7", calibration_times),
        value_unit="fixture_units",
        time_unit="fixture_seconds",
    )
    measurement = P2MeasurementBounds(
        offsets=(0, 0),
        scales=(1, 1),
        epi_error=(Fraction(1, 10**10), Fraction(1, 10**10)),
        timestamp_error=0,
        structural_time_per_unit=(1, 1),
        provenance="declared Decimal/binary64 software fixture bounds only",
    )
    calibration = calibrate_p2_transport(
        [calibration_run],
        graph=nx.Graph([("left", "right")]),
        measurement=measurement,
        support_provenance="independently declared synthetic P2 edge",
    )
    times = tuple(float(i / 8) for i in range(9))
    forecast = forecast_p2_transport(
        calibration,
        evaluation_run_id="continuous-reserved",
        evaluation_acquisition_id="continuous-reserved-preparation",
        initial_measurement=(1.5, 2.5),
        timestamps=times,
    )
    issued_hash = forecast.content_hash
    with TemporaryDirectory(prefix="tnfr-continuous-p2-demo-") as directory:
        write_p2_transport_forecast(forecast, Path(directory) / "forecast.json")
        # Future response values are generated only after the exact tube is saved.
        observation = NodalMeasurementRun(
            run_id="continuous-reserved",
            acquisition_id="continuous-reserved-preparation",
            channel_ids=("left", "right"),
            timestamps=times,
            samples=continuous_samples("2", "-0.5", times),
            value_unit="fixture_units",
            time_unit="fixture_seconds",
        )
        comparison = score_p2_transport(
            forecast,
            calibration,
            observation,
            expected_forecast_hash=issued_hash,
        )
    lower, upper = (float(value) for value in calibration.capacity)
    print("\nSeparate continuous P2 interval demonstration:")
    print(f"   capacity enclosure    = [{lower:.17g}, {upper:.17g}]")
    print("                          (display only; saved endpoints are exact)")
    print(f"   issued forecast hash  = {issued_hash}")
    print(f"   reserved comparison   = {comparison.status}")
    print(f"   physical status       = {comparison.physical_status}")
    print("                          (overlapping outer boxes do not prove model fit)")


def main() -> None:
    print("=" * 72)
    print("EMPIRICAL CONFRONTATION -- canonical multichannel read-outs")
    print("=" * 72)
    signals, source = load_signal()
    print(f"\nsource: {source}  " f"(shape {signals.shape[0]}ch x {signals.shape[1]})")

    # The confrontation report includes the emergent geometry and its spectral
    # xi_C comparison. Rebuild the deterministic coupling graph to request the
    # primary decay-fit estimator together with explicit provenance.
    rep = confront_signal(signals, k_neighbours=K_NEIGHBOURS)
    phase, amplitude = phase_amplitude_matrices(signals)
    coupling_graph = build_coupling_graph(phase, amplitude, k_neighbours=K_NEIGHBOURS)
    xi_estimate = estimate_coherence_length_with_provenance(coupling_graph)

    print("\nScoped read-outs of the emergent coupling graph:")
    print(f"   Kuramoto R           = {rep.kuramoto_R:.3f}")
    print(
        f"   static coherence C₀ = {rep.coherence:.3f}  "
        f"(mean |ΔNFR| within tolerance: {rep.at_equilibrium}; dEPI=0)"
    )
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
    print(
        f"   pulse ω₀, beat, E    = {rep.pulse_fundamental:.3f}, "
        f"{rep.dominant_beat:.3f}, {rep.vibration_energy:.3f}"
    )
    modal = rep.modal_diagnostic
    if modal is None:
        print("   modal roots           = unresolved (missing provenance)")
    else:
        print(
            f"   modal roots           = {modal.status}: "
            f"{modal.root_classification}; fitted stability={modal.stability}"
        )
        print(f"                          {modal.reason}")
    fraction = (
        "unavailable" if rep.wave_fraction is None else f"{rep.wave_fraction:.2f}"
    )
    print(f"   complex-root fraction = {fraction}, Q={rep.quality_factor:.2f}")
    print("                          (descriptive; no physical regime certificate)")

    dyn = nodal_prediction_skill(signals)
    print(
        f"   same-window fit       = {dyn.nodal_skill:+.3f} vs AR-1 "
        f"{dyn.ar1_skill:+.3f}; c={dyn.diffusivity:+.3f}"
    )
    print("                          (fit and score share data; no held-out evidence)")
    print(f"   fitted capacity sign  = {dyn.capacity_domain}")

    demonstrate_reserved_prediction()
    demonstrate_continuous_p2_intervals()

    print("\n" + "=" * 72)
    print(
        "The graph read-outs and modal roots describe this signal window.\n"
        "xi_C retains its decay-fit or spectral-fallback provenance.\n"
        "Calibration, physical model admission and held-out prediction remain\n"
        "separate tasks. These diagnostics alone do not validate a TNFR law."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
