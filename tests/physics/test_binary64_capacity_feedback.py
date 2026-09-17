"""Finite controls for the analytic unit-binade Coupling capacity class."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

from tnfr import _binary64
from tnfr.operators import _coupling_stage_kernel as kernel
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics.capacity_feedback import (
    derive_p2_binary64_coupling_lattice, observe_p2_binary64_coupling_lattice,
)


F = Fraction
SPACING = F(1, 2**52)


def _gamma():
    return canonical_glyph_factor_defaults()["UM_vf_sync"]


def _reference():
    return derive_p2_binary64_coupling_lattice(coupling_factor=_gamma())


def _capacity(index):
    return float(1 + index * SPACING)


def test_actual_default_factor_satisfies_all_exact_analytic_margins():
    ref = _reference()
    assert ref.exact_coupling_factor == F(5734161139222659, 72057594037927936)
    assert ref.lattice_step == SPACING
    assert ref.relative_product_error == F(1, 2**53)
    assert ref.max_index == 2**52
    assert ref.fixed_max_index == 6
    assert ref.proof_lower_gain == F(5, 64)
    assert all(getattr(ref, name) > 0 for name in (
        "fixed_boundary_margin", "descent_boundary_margin", "product_upper_margin",
        "proof_lower_gain_margin",
    ))
    assert ref.exact_coupling_factor * SPACING > F(1, 2**1022)
    assert ref.uniform_horizon == 512 and ref.uniform_horizon_verified
    # Independent rational form of the integer proof, without class enumeration.
    upper_after_horizon = (
        F(59, 64)**512 * (ref.max_index - F(32, 5)) + F(32, 5)
    )
    assert F(32, 5) < upper_after_horizon < 7


@pytest.mark.parametrize(("index", "following"), (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    (7, 6), (8, 7), (13, 12), (80, 74),
    (2**52, 4145214556169080),
))
def test_independent_lattice_endpoints_and_conditional_eventual_indices(index, following):
    ref = _reference()
    result = observe_p2_binary64_coupling_lattice(
        ref, capacity_before=_capacity(index), capacity_after=_capacity(following),
    )
    assert result.index_before == index
    assert result.index_after == following
    assert result.index_decrease == index - following
    assert result.is_fixed == (index <= 6)
    assert result.eventual_index == min(index, 6)
    assert result.eventual_gap == min(index, 6) * SPACING
    assert result.eventual_capacity == _capacity(min(index, 6))
    assert result.termination_horizon_from_before == (0 if index <= 6 else 512)


def test_one_ulp_and_six_ulp_stalls_have_positive_defects_against_the_exact_model():
    ref = _reference()
    for index in (1, 6):
        value = _capacity(index)
        result = observe_p2_binary64_coupling_lattice(ref, capacity_before=value, capacity_after=value)
        assert result.capacity_gap_defect == ref.exact_coupling_factor * index * SPACING
        assert result.capacity_gap_defect > 0
        assert result.ideal_capacity_gap_after < index * SPACING


def test_first_descending_state_and_upper_binade_have_independent_signed_defects():
    ref = _reference()
    seven = observe_p2_binary64_coupling_lattice(ref, capacity_before=_capacity(7), capacity_after=_capacity(6))
    assert seven.capacity_gap_defect == (7 * ref.exact_coupling_factor - 1) * SPACING
    assert seven.capacity_gap_defect < 0
    upper = observe_p2_binary64_coupling_lattice(
        ref, capacity_before=2.0, capacity_after=_capacity(4145214556169080),
    )
    assert upper.capacity_gap_defect == 3 * SPACING / 16
    assert 1.0 < upper.capacity_after < 2.0


@pytest.mark.parametrize(("offset", "rounded_product_ulps", "expected_index"), (
    (9, F(716770142402831, 2), 4145214556169072),
    (21, F(716770142402829, 2), 4145214556169060),
))
def test_final_halfway_cases_round_the_capacity_index_to_even(
    offset, rounded_product_ulps, expected_index,
):
    ref = _reference()
    index = 2**52 - offset
    before = _capacity(index)
    # These two halfway cases exercise opposite final rounding directions.
    product = ref.coupling_factor * (1.0 - before)
    assert -F(product) / SPACING == rounded_product_ulps
    exact_final_index = index - rounded_product_ulps
    assert exact_final_index.denominator == 2
    assert expected_index % 2 == 0
    assert abs(exact_final_index - expected_index) == F(1, 2)
    result = observe_p2_binary64_coupling_lattice(
        ref, capacity_before=before, capacity_after=_capacity(expected_index),
    )
    assert result.index_after == expected_index
    if offset == 9:
        # Rounding the decrement first would select the wrong odd capacity.
        assert index - round(rounded_product_ulps) != expected_index


def test_observer_calls_the_shared_production_capacity_kernel(monkeypatch):
    original = kernel.coupling_capacity_blend
    calls = []

    def recording(capacity, neighbors, factor):
        calls.append((capacity, neighbors, factor))
        return original(capacity, neighbors, factor)

    monkeypatch.setattr(kernel, "coupling_capacity_blend", recording)
    result = observe_p2_binary64_coupling_lattice(
        _reference(), capacity_before=_capacity(7), capacity_after=_capacity(6),
    )
    assert calls == [(_capacity(7), (1.0,), _gamma())]
    assert result.index_after == 6


@pytest.mark.parametrize(("before", "after"), (
    (_capacity(1), 1.0), (_capacity(7), _capacity(7)), (_capacity(8), _capacity(6)),
))
def test_a_fabricated_or_idealized_endpoint_is_rejected(before, after):
    with pytest.raises(ValueError, match="production kernel"):
        observe_p2_binary64_coupling_lattice(_reference(), capacity_before=before, capacity_after=after)


@pytest.mark.parametrize("value", (
    1, True, F(1), "1.0", float("nan"), float("inf"),
    math.nextafter(1.0, 0.0), math.nextafter(2.0, math.inf),
))
def test_endpoint_inputs_require_actual_finite_floats_in_the_unit_binade(value):
    with pytest.raises(ValueError):
        observe_p2_binary64_coupling_lattice(_reference(), capacity_before=value, capacity_after=1.0)
    with pytest.raises(ValueError):
        observe_p2_binary64_coupling_lattice(_reference(), capacity_before=1.0, capacity_after=value)


@pytest.mark.parametrize("factor", (0.0, 0.01, 0.075, 0.1, 0.5, 1.0, 0, True, F(1, 12), float("nan")))
def test_unsupported_factor_or_nonfloat_factor_is_rejected(factor):
    with pytest.raises(ValueError):
        derive_p2_binary64_coupling_lattice(coupling_factor=factor)


def test_another_factor_is_accepted_only_inside_the_same_explicit_proved_class():
    ref = derive_p2_binary64_coupling_lattice(coupling_factor=0.08)
    assert ref.fixed_boundary_margin > 0 and ref.proof_lower_gain_margin > 0
    result = observe_p2_binary64_coupling_lattice(
        ref, capacity_before=_capacity(7), capacity_after=_capacity(6),
    )
    assert result.eventual_index == 6


def test_failed_binary64_platform_check_rejects_derivation_and_observation(monkeypatch):
    ref = _reference()
    monkeypatch.setattr(_binary64, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(ValueError, match="IEEE binary64"):
        _reference()
    with pytest.raises(ValueError, match="IEEE binary64"):
        observe_p2_binary64_coupling_lattice(ref, capacity_before=1.0, capacity_after=1.0)


def test_reference_caches_are_rebuilt_and_numeric_observation_is_frozen():
    ref = _reference()
    forged = replace(
        ref, exact_coupling_factor=F(1), lattice_step=F(1), max_index=1,
        fixed_max_index=99, fixed_boundary_margin=F(-1), uniform_horizon=0,
        uniform_horizon_verified=False,
    )
    expected = observe_p2_binary64_coupling_lattice(ref, capacity_before=_capacity(7), capacity_after=_capacity(6))
    actual = observe_p2_binary64_coupling_lattice(forged, capacity_before=_capacity(7), capacity_after=_capacity(6))
    assert actual == expected
    with pytest.raises(FrozenInstanceError):
        actual.eventual_index = 0
    with pytest.raises(TypeError, match="reference"):
        observe_p2_binary64_coupling_lattice(None, capacity_before=1.0, capacity_after=1.0)
    with pytest.raises(ValueError, match="six-fixed-gap"):
        observe_p2_binary64_coupling_lattice(
            replace(ref, coupling_factor=0.1), capacity_before=1.0, capacity_after=1.0,
        )
