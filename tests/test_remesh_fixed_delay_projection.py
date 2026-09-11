"""Direct checks for the corrected fixed-delay REMESH surrogate."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.riemann.operator_catalog_discipline_signature import (
    compute_operator_catalog_discipline_signature,
)
from tnfr.riemann.remesh_infinity_residue_split import (
    build_resonant_bin_mask,
    compute_residue_split_certificate,
    split_residue_by_remesh_infinity,
)


def test_fixed_bins_use_delay_gcd() -> None:
    mask = build_resonant_bin_mask(32, tau_l=4, tau_g=8)

    assert np.flatnonzero(mask).tolist() == [0, 8, 16, 24]


def test_mask_matches_unit_eigenvalues_of_cyclic_filter() -> None:
    n_samples = 72
    tau_l, tau_g = 6, 9
    alpha = 0.5
    beta = (1.0 - alpha) ** 2
    gamma = alpha * (1.0 - alpha)
    delta = alpha
    omega = 2.0 * np.pi * np.arange(n_samples) / n_samples
    multipliers = (
        beta
        + gamma * np.exp(-1j * omega * tau_l)
        + delta * np.exp(-1j * omega * tau_g)
    )

    mask = build_resonant_bin_mask(
        n_samples, tau_l=tau_l, tau_g=tau_g
    )

    assert np.array_equal(mask, np.isclose(multipliers, 1.0, atol=1e-12))


def test_coprime_delays_fix_only_dc_mode() -> None:
    mask = build_resonant_bin_mask(24, tau_l=3, tau_g=8)

    assert np.flatnonzero(mask).tolist() == [0]


def test_projection_is_orthogonal_and_reconstructs_signal() -> None:
    signal = np.random.default_rng(19).normal(size=32)
    range_part, kernel_part = split_residue_by_remesh_infinity(
        signal, tau_l=4, tau_g=8
    )
    range_twice, _ = split_residue_by_remesh_infinity(
        range_part, tau_l=4, tau_g=8
    )
    kernel_range, _ = split_residue_by_remesh_infinity(
        kernel_part, tau_l=4, tau_g=8
    )

    assert np.allclose(range_part + kernel_part, signal)
    assert np.allclose(range_twice, range_part)
    assert np.allclose(kernel_range, 0.0)
    assert float(np.dot(range_part, kernel_part)) == pytest.approx(0.0, abs=1e-12)


def test_certificate_reports_gcd_period_and_exact_controls() -> None:
    certificate = compute_residue_split_certificate(
        n_primes=5,
        max_power=2,
        tau_l=4,
        tau_g=8,
        n_periods=4,
    )

    assert certificate.lcm_period == 8
    assert certificate.fixed_period_gcd == 4
    assert certificate.range_control_resonant == pytest.approx(1.0)
    assert certificate.range_control_nonresonant == pytest.approx(0.0, abs=1e-12)
    assert certificate.ratio_in_range + certificate.ratio_in_kernel == pytest.approx(
        1.0
    )


def test_registry_signature_states_its_completeness_boundary() -> None:
    certificate = compute_operator_catalog_discipline_signature()

    assert certificate.S_OC == 0.0
    assert any("not completeness" in note for note in certificate.notes)
