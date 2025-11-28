"""
Life Demo: Minimal usage of tnfr.physics.life to detect life emergence.

Run:
    python -m examples.life_demo
"""
from __future__ import annotations
import numpy as np
from tnfr.physics import detect_life_emergence


def _logistic(t: np.ndarray, L=10.0, k=0.6, t0=8.0) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (t - t0)))


def main() -> None:
    T = 400
    times = np.linspace(0.0, 20.0, T)

    # Synthetic ‖EPI‖ trajectory that saturates (steady-state)
    epi_series = _logistic(times, L=10.0, k=0.6, t0=8.0)
    dEPI_dt = np.gradient(epi_series, times)

    # Hostile environment: small negative external ΔNFR
    dnfr_external = np.full_like(epi_series, -0.05)
    d_dnfr_external_dt = np.zeros_like(epi_series)

    # Parameters (from derivation): ensure crossing
    epsilon = 0.8
    gamma = 1.0
    epi_max = 10.0

    telem = detect_life_emergence(
        times=times,
        epi_series=epi_series,
        dEPI_dt=dEPI_dt,
        dnfr_external=dnfr_external,
        d_dnfr_external_dt=d_dnfr_external_dt,
        epsilon=epsilon,
        gamma=gamma,
        epi_max=epi_max,
    )

    print("Life Telemetry Summary:")
    print(f"  threshold_time: {telem.life_threshold_time}")
    print(f"  A(t) max: {float(np.max(telem.autopoietic_coefficient)):.3f}")
    print(f"  S(t) median: {float(np.median(telem.self_org_index)):.3f}")
    print(f"  M(t) final: {float(telem.stability_margin[-1]):.3f}")
    print(f"  Vi(t) mean: {float(np.mean(telem.vitality_index)):.3f}")


if __name__ == "__main__":
    main()
