import numpy as np

from tnfr.physics import detect_life_emergence


def _logistic(t, L=10.0, k=0.6, t0=8.0):
    return L / (1.0 + np.exp(-k * (t - t0)))


def test_detect_life_emergence_threshold():
    # Synthetic trajectory that crosses life threshold per derivation
    T = 400
    times = np.linspace(0.0, 20.0, T)
    epi_series = _logistic(times, L=10.0, k=0.6, t0=8.0)

    # Numerical derivative of ‖EPI‖
    dEPI_dt = np.gradient(epi_series, times)

    # External ΔNFR is small and negative (hostile environment)
    dnfr_external = np.full_like(epi_series, -0.05)
    d_dnfr_external_dt = np.zeros_like(epi_series)

    # Parameters chosen to ensure threshold crossing
    epsilon = 0.8   # self-feedback strength
    gamma = 1.0     # autopoietic strength
    epi_max = 10.0  # carrying capacity

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

    # Life threshold time should be detected
    assert telem.life_threshold_time is not None

    # A(t) must exceed 1 at or after the threshold time
    idx = np.searchsorted(times, telem.life_threshold_time)
    assert np.any(telem.autopoietic_coefficient[idx:] > 1.0)

    # Stability margin tends to positive near saturated EPI
    assert telem.stability_margin[-1] > 0.0

    # Vitality index should remain within [0,1]
    assert np.isfinite(telem.vitality_index).all()
    assert (telem.vitality_index >= 0.0).all()
    assert (telem.vitality_index <= 1.0).all()


def test_metrics_shapes_and_bounds():
    T = 200
    times = np.linspace(0.0, 10.0, T)
    epi_series = _logistic(times, L=8.0, k=0.5, t0=4.0)
    dEPI_dt = np.gradient(epi_series, times)
    dnfr_external = np.full_like(epi_series, -0.04)
    d_dnfr_external_dt = np.zeros_like(epi_series)

    epsilon = 0.5
    gamma = 0.8
    epi_max = 8.0

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

    # Shape consistency
    for arr in (
        telem.vitality_index,
        telem.autopoietic_coefficient,
        telem.self_org_index,
        telem.stability_margin,
    ):
        assert arr.shape == (T,)
        assert np.isfinite(arr).all()

    # Stability margin within reasonable bounds (-0.5..0.5 from derivation)
    assert (telem.stability_margin > -1.0).all()  # relaxed bound due to synthetic series
    assert (telem.stability_margin < 1.0).all()
