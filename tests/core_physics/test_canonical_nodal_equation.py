"""Admission and finite-output boundaries of the shared nodal row owner."""

import math
from fractions import Fraction

import pytest

from tnfr.dynamics.canonical import (
    compute_canonical_nodal_derivative,
    compute_extended_nodal_system,
)
from tnfr.errors.contextual import FrequencyError, NetworkConfigError, TNFRValueError


@pytest.mark.parametrize(
    "capacity",
    [
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="text"),
        pytest.param(-Fraction(1, 2**2000), id="negative-underflow"),
        pytest.param(Fraction(1, 2**2000), id="positive-underflow"),
        pytest.param(10**400, id="conversion-overflow"),
    ],
)
def test_capacity_cannot_be_coerced_into_an_admitted_coefficient(capacity):
    with pytest.raises(FrequencyError) as caught:
        compute_canonical_nodal_derivative(capacity, 1.0)
    assert caught.value.context["vf"] == capacity


@pytest.mark.parametrize("pressure", [False, "0.25", Fraction(1, 2**2000), 10**400])
def test_zero_capacity_does_not_authorize_invalid_pressure(pressure):
    with pytest.raises(NetworkConfigError) as caught:
        compute_canonical_nodal_derivative(0.0, pressure)
    assert caught.value.context["parameter"] == "delta_nfr"


@pytest.mark.parametrize(
    "capacity,pressure,expected",
    [
        (0, 1e308, 0.0),
        (1e308, 1.0, 1e308),
        (Fraction(3, 2), Fraction(-1, 4), -0.375),
        (math.ulp(0.0), 1, math.ulp(0.0)),
    ],
)
def test_zero_large_and_small_representable_capacity_remain_admitted(
    capacity, pressure, expected
):
    result = compute_canonical_nodal_derivative(capacity, pressure)
    assert result.derivative == expected
    assert result.validated is True
    assert type(result.nu_f) is float
    assert type(result.delta_nfr) is float


@pytest.mark.parametrize(
    "overrides,parameter",
    [
        ({"theta": True}, "phase"),
        ({"j_phi": "0.25"}, "J_φ"),
        ({"j_dnfr_divergence": Fraction(1, 2**2000)}, "flux_divergence"),
        ({"coupling_strength": -Fraction(1, 2**2000)}, "coupling_strength"),
    ],
)
def test_extension_uses_the_same_raw_real_admission(overrides, parameter):
    inputs = dict(nu_f=1.0, delta_nfr=0.5, theta=0.0, j_phi=0.0, j_dnfr_divergence=0.0)
    inputs.update(overrides)
    with pytest.raises(NetworkConfigError) as caught:
        compute_extended_nodal_system(**inputs)
    assert caught.value.context["parameter"] == parameter


@pytest.mark.parametrize(
    "overrides,parameter",
    [
        ({"nu_f": 1e308, "delta_nfr": 2.0}, "dEPI_dt"),
        ({"j_phi": 1e308, "coupling_strength": 100.0}, "dtheta_dt"),
        ({"j_dnfr_divergence": 1.7e308}, "ddelta_nfr_dt"),
        ({"delta_nfr": 1e308}, "dtheta_dt"),
    ],
)
def test_extension_cannot_certify_nonrepresentable_derivatives(overrides, parameter):
    inputs = dict(nu_f=0.0, delta_nfr=0.0, theta=0.0, j_phi=0.0, j_dnfr_divergence=0.0)
    inputs.update(overrides)
    with pytest.raises(NetworkConfigError) as caught:
        compute_extended_nodal_system(**inputs)
    assert caught.value.context["parameter"] == parameter


@pytest.mark.parametrize("extended", [False, True])
def test_explicit_unvalidated_path_retains_its_scope(extended):
    function = (
        compute_extended_nodal_system
        if extended
        else compute_canonical_nodal_derivative
    )
    args = (1e308, 2.0, 0.0, 0.0, 0.0) if extended else (1e308, 2.0)
    result = function(*args, validate_units=False)
    derivative = result.classical_derivative if extended else result.derivative
    assert math.isinf(derivative)
    assert result.validated is False
    with pytest.raises(TNFRValueError, match="validate_units must be a boolean"):
        function(*args, validate_units="false")


def test_frequency_error_describes_the_actual_nonnegative_domain():
    error = FrequencyError(-math.ulp(0.0), operation="validation")
    assert error.context["vf"] == -math.ulp(0.0)
    assert (
        error.context["valid_range"]
        == "finite represented real values >= 0 (zero admitted)"
    )
    assert "Zero is valid" in error.suggestion
    assert "Typical range" not in str(error)
