"""Finite-size phase-scaling diagnostics and scope boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics import analyze_phase_finite_size_scaling


def _synthetic_sweep():
    sizes = np.array([16, 32, 64, 128])
    controls = np.array([0.2, 0.4, 0.6])
    replicate_factors = np.array([0.9, 1.0, 1.1])

    order = np.empty((4, 3, 3))
    susceptibility = np.empty_like(order)
    coherence_length = np.empty_like(order)
    control_profiles = np.array([0.4, 1.0, 0.6])
    for i, size in enumerate(sizes):
        for j, profile in enumerate(control_profiles):
            order[i, j] = -(size ** -0.25) * profile * replicate_factors
            susceptibility[i, j] = (
                size**0.5 * profile * replicate_factors
            )
            coherence_length[i, j] = (
                size**0.4 * profile * replicate_factors
            )
    return sizes, controls, order, susceptibility, coherence_length


def test_recovers_declared_node_count_slopes_and_peak_control():
    sizes, controls, order, susceptibility, coherence_length = _synthetic_sweep()
    result = analyze_phase_finite_size_scaling(
        sizes, controls, order, susceptibility, coherence_length
    )

    assert result.pseudocritical_control == (0.4, 0.4, 0.4, 0.4)
    assert result.pseudocritical_controls == ((0.4,),) * 4
    assert result.peak_is_unique == (True,) * 4
    assert result.all_peaks_unique
    assert result.all_peaks_interior
    assert result.peak_at_control_boundary == (False,) * 4
    assert result.replicate_count == 3
    assert result.susceptibility_size_fit is not None
    assert result.order_size_fit is not None
    assert result.coherence_length_size_fit is not None
    assert result.susceptibility_size_fit.slope == pytest.approx(0.5)
    assert result.order_size_fit.slope == pytest.approx(-0.25)
    assert result.coherence_length_size_fit.slope == pytest.approx(0.4)
    assert result.susceptibility_size_fit.r_squared == pytest.approx(1.0)
    assert all(value is not None for value in result.peak_susceptibility_sem)
    assert result.status == "finite_size_measurement"


def test_two_dimensional_input_is_one_replicate_without_fake_sem():
    sizes = [8, 16, 32]
    controls = [0.1, 0.2]
    order = np.array([[0.1, 0.2], [0.08, 0.15], [0.06, 0.1]])
    susceptibility = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])

    result = analyze_phase_finite_size_scaling(
        sizes, controls, order, susceptibility
    )

    assert result.replicate_count == 1
    assert result.peak_susceptibility_sem == (None, None, None)
    assert result.coherence_length_at_peak is None
    assert result.coherence_length_size_fit is None
    assert not result.all_peaks_interior
    assert result.status == "finite_size_measurement_unbracketed"


def test_sampled_peak_can_shift_with_size():
    sizes = [10, 20, 40]
    controls = [0.2, 0.4, 0.6]
    order = np.ones((3, 3))
    susceptibility = np.array(
        [[3.0, 2.0, 1.0], [1.0, 4.0, 2.0], [1.0, 2.0, 5.0]]
    )

    result = analyze_phase_finite_size_scaling(
        sizes, controls, order, susceptibility
    )

    assert result.pseudocritical_control == (0.2, 0.4, 0.6)


def test_plateau_reports_all_maximizers_and_any_boundary_contact():
    sizes = [8, 16, 32]
    controls = [0.0, 1.0, 2.0]
    order = np.array(
        [
            [[0.1, 0.3], [0.4, 0.6], [0.8, 1.0]],
            [[0.1, 0.3], [0.5, 0.7], [0.9, 1.1]],
            [[0.1, 0.3], [0.6, 0.8], [1.0, 1.2]],
        ]
    )
    susceptibility = np.array(
        [
            [[1.0, 1.0], [3.0, 5.0], [5.0, 3.0]],
            [[2.0, 2.0], [6.0, 10.0], [10.0, 6.0]],
            [[4.0, 4.0], [12.0, 20.0], [20.0, 12.0]],
        ]
    )

    result = analyze_phase_finite_size_scaling(
        sizes, controls, order, susceptibility
    )

    assert result.pseudocritical_control == (None, None, None)
    assert result.pseudocritical_controls == ((1.0, 2.0),) * 3
    assert result.peak_is_unique == (False, False, False)
    assert not result.all_peaks_unique
    assert result.peak_at_control_boundary == (True, True, True)
    assert not result.all_peaks_interior
    assert result.order_at_peak == pytest.approx((0.7, 0.8, 0.9))
    assert result.peak_susceptibility == pytest.approx((4.0, 8.0, 16.0))
    assert result.status == "finite_size_measurement_unbracketed_ambiguous_peak"
    assert any("plateau" in item for item in result.limitations)


@pytest.mark.parametrize(
    "sizes,controls,order,susceptibility,error",
    [
        ([8, 16], [0.1, 0.2], np.ones((2, 2)), np.ones((2, 2)), "three"),
        ([8, True, 16], [0.1, 0.2], np.ones((3, 2)), np.ones((3, 2)), "integer"),
        ([8, 8, 16], [0.1, 0.2], np.ones((3, 2)), np.ones((3, 2)), "increasing"),
        ([8, 16, 32], [0.2, 0.1], np.ones((3, 2)), np.ones((3, 2)), "increasing"),
        ([8, 16, 32], [0.1, 0.2], np.ones((3, 2)), -np.ones((3, 2)), "negative"),
        ([8, 16, 32], [0.1, 0.2], np.ones((3, 3)), np.ones((3, 2)), "shape"),
    ],
)
def test_rejects_invalid_protocols(sizes, controls, order, susceptibility, error):
    with pytest.raises(ValueError, match=error):
        analyze_phase_finite_size_scaling(
            sizes, controls, order, susceptibility
        )


def test_result_is_reproducible_and_does_not_claim_universality():
    inputs = _synthetic_sweep()
    first = analyze_phase_finite_size_scaling(*inputs)
    second = analyze_phase_finite_size_scaling(*inputs)

    assert first == second
    assert any("cannot establish" in item for item in first.limitations)


def test_nonpositive_peak_withholds_fit_without_dropping_a_size():
    sizes = [8, 16, 32, 64]
    controls = [0.0, 1.0]
    order = np.ones((4, 2))
    susceptibility = np.array(
        [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
    )

    result = analyze_phase_finite_size_scaling(
        sizes, controls, order, susceptibility
    )

    assert result.susceptibility_size_fit is None
    assert result.nonpositive_fit_observables == ("peak_susceptibility",)
    assert result.status.endswith("nonpositive_fit_input")
    assert any("no sizes were dropped" in item for item in result.limitations)


def test_positive_observable_mean_cannot_silently_underflow_to_zero():
    tiny = np.nextafter(0.0, 1.0)
    order = np.ones((3, 2, 2), dtype=float)
    susceptibility = np.full((3, 2, 2), tiny, dtype=float)
    susceptibility[:, :, 1] = 0.0

    with pytest.raises(ValueError, match="below nonzero floating-point range"):
        analyze_phase_finite_size_scaling(
            [8, 16, 32],
            [0.0, 1.0],
            order,
            susceptibility,
        )
def test_boolean_observations_are_not_silently_converted_to_numbers():
    sizes = [8, 16, 32]
    controls = [0.0, 1.0]
    order = np.ones((3, 2), dtype=object)
    order[1, 1] = True

    with pytest.raises(ValueError, match="not booleans"):
        analyze_phase_finite_size_scaling(
            sizes, controls, order, np.ones((3, 2))
        )


def test_numpy_boolean_control_is_not_silently_converted_to_zero():
    with pytest.raises(ValueError, match="finite real coordinates"):
        analyze_phase_finite_size_scaling(
            [8, 16, 32],
            [np.bool_(False), 1.0],
            np.ones((3, 2)),
            np.ones((3, 2)),
        )


def test_finite_maximum_observations_keep_finite_means_and_uncertainties():
    maximum = np.finfo(float).max
    values = np.full((3, 2, 2), maximum)

    result = analyze_phase_finite_size_scaling(
        [8, 16, 32], [0.0, 1.0], values, values, values
    )

    assert result.peak_susceptibility == (maximum,) * 3
    assert result.order_at_peak == (maximum,) * 3
    assert result.coherence_length_at_peak == (maximum,) * 3
    assert result.peak_susceptibility_sem == (0.0,) * 3
    assert result.susceptibility_size_fit is not None
    assert np.isfinite(result.susceptibility_size_fit.slope)


def test_text_observations_and_unresolved_float_sizes_are_rejected():
    with pytest.raises(ValueError, match="real numeric"):
        analyze_phase_finite_size_scaling(
            [8, 16, 32],
            [0.0, 1.0],
            [["1.0", "1.0"]] * 3,
            np.ones((3, 2)),
        )

    with pytest.raises(ValueError, match="numerical fit"):
        analyze_phase_finite_size_scaling(
            [2**53, 2**53 + 1, 2**53 + 2],
            [0.0, 1.0],
            np.ones((3, 2)),
            np.ones((3, 2)),
        )
