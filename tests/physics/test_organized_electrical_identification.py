"""Synthetic component admission without a physical or reserved experiment.

Independent exact controls check the component shapes and eliminated scales.
The prospective sine-amplitude test consumes primitive measurements, not an
earlier verdict, and imports no acquired source or trajectory machinery.
"""

import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_product,
)
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_electrical_identification as owner


PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
WHEEL = tuple((a, b) for a, b in PAIRS if b == 5 or (a - b) % 5 in (1, 4))


def _cap_coefficient(a, b):
    if b == 5:
        return Q(11, 396)
    return Q(5 if (a - b) % 5 in (1, 4) else 41, 396)


def _primitives(*, c=Q(2), g=Q(3), a0=Q(5)):
    return {
        "capacitances": tuple(c * _cap_coefficient(*pair) for pair in PAIRS),
        "conductances": (g / 6,) * 15,
        "josephson_constant": a0,
    }


def _laplacian(pairs, values):
    rows = [[Q(0)] * 6 for _ in range(6)]
    for (a, b), value in zip(pairs, values):
        rows[a][a] += value
        rows[b][b] += value
        rows[a][b] -= value
        rows[b][a] -= value
    return tuple(tuple(row) for row in rows)


def _expected_amplitude(*, c=Q(2), g=Q(3), a0=Q(5)):
    # Independent Machin enclosure: form the final dimensional quantity only
    # after the exact ratio, avoiding products near the absolute interval grid.
    pi = pi_interval()
    return I(
        a0 * g**2 / (c * (1023 * pi.hi) ** 2), a0 * g**2 / (c * (1023 * pi.lo) ** 2)
    )


@pytest.fixture(autouse=True)
def no_trajectory_or_frozen_execution(request):
    if (
        request.node.name
        == "test_cold_identification_has_no_source_or_solver_dependency"
    ):
        yield
    else:
        with forbid_sine_regeneration():
            yield


def test_pair_registration_and_capacity_shape_follow_the_exact_wheel_inverse():
    assert owner._MUTUAL_PAIRS == PAIRS
    assert owner._WHEEL_PAIRS == WHEEL
    assert len(PAIRS) == 15 and len(WHEEL) == 10
    laplacian = _laplacian(WHEEL, (Q(1),) * 10)
    shifted = tuple(tuple(value + Q(1, 6) for value in row) for row in laplacian)
    inverse = exact_matrix_inverse(shifted)
    pseudoinverse = tuple(tuple(value - Q(1, 6) for value in row) for row in inverse)
    capacitors = _primitives(c=Q(1))["capacitances"]
    assert _laplacian(PAIRS, capacitors) == pseudoinverse
    projection = tuple(tuple(Q(a == b) - Q(1, 6) for b in range(6)) for a in range(6))
    assert exact_square_matrix_product(laplacian, pseudoinverse) == projection


@pytest.mark.parametrize("c,g,a0", ((Q(2), Q(3), Q(5)), (Q(7, 4), Q(9, 5), Q(2, 7))))
def test_independent_capacity_conductance_calibration(c, g, a0):
    result = owner._calibrate_organized_electrical_components(
        **_primitives(c=c, g=g, a0=a0)
    )
    assert result.status == "component_shape_not_excluded"
    assert not result.exclusions
    assert result.capacitance_scale_bounds.contains(c)
    assert result.conductance_scale_bounds.contains(g)
    assert result.clock_scale_bounds.contains(4 * g / c)
    expected = _expected_amplitude(c=c, g=g, a0=a0)
    assert result.predicted_sine_amplitude_bounds.lo <= expected.hi
    assert result.predicted_sine_amplitude_bounds.hi >= expected.lo
    pi = pi_interval()
    charge = result.charge_scale_bounds
    assert charge.lo <= a0 * g / (9 * 1023 * pi.hi)
    assert charge.hi >= a0 * g / (9 * 1023 * pi.lo)


@pytest.mark.parametrize("gamma", (Q(1, 7), Q(3, 11), Q(2)))
def test_generic_exact_scale_elimination_control(gamma):
    # This algebraic control does not configure the actual fixed-gamma kernel.
    a0, clock, charge = Q(5, 3), Q(7, 2), Q(11, 13)
    c = 36 * charge / (a0 * clock * gamma)
    g = 9 * charge / (a0 * gamma)
    j = 9 * clock * charge * gamma / 4
    assert 4 * g / c == clock
    assert a0 * gamma * g / 9 == charge
    assert c * j / (a0 * g**2) == gamma**2


def test_reserved_sine_amplitude_does_not_calibrate_charge_or_clock():
    primitives = _primitives()
    before = owner._calibrate_organized_electrical_components(**primitives)
    j = _expected_amplitude()
    matched = owner._compare_organized_electrical_sine_amplitudes(
        **primitives, sine_amplitudes=(j,) * 10
    )
    doubled = owner._compare_organized_electrical_sine_amplitudes(
        **primitives, sine_amplitudes=(I(2 * j.lo, 2 * j.hi),) * 10
    )
    assert matched.status == "necessary_relation_not_excluded"
    assert doubled.status == "sine_amplitude_relation_excluded"
    assert matched.calibration == doubled.calibration == before
    pi = pi_interval()
    ratio = matched.dimensionless_ratio_bounds
    assert ratio.lo <= 1 / (1023 * pi.hi) ** 2
    assert ratio.hi >= 1 / (1023 * pi.lo) ** 2


@pytest.mark.parametrize("field", ("capacitances", "conductances"))
def test_same_total_component_trace_cannot_replace_the_full_shape(field):
    primitives = _primitives()
    changed = list(primitives[field])
    delta = min(changed[0], changed[1]) / 4
    changed[0] += delta
    changed[1] -= delta
    assert sum(changed) == sum(primitives[field])
    original_matrix = _laplacian(PAIRS, primitives[field])
    changed_matrix = _laplacian(PAIRS, changed)
    assert sum(row[i] for i, row in enumerate(original_matrix)) == sum(
        row[i] for i, row in enumerate(changed_matrix)
    )
    assert original_matrix != changed_matrix
    result = owner._calibrate_organized_electrical_components(
        **dict(primitives, **{field: tuple(changed)})
    )
    assert result.status == "component_shape_excluded"
    assert result.exclusions


@pytest.mark.parametrize("field", ("capacitances", "conductances", "sine_amplitudes"))
def test_one_changed_component_is_detected_from_primitives(field):
    primitives = dict(_primitives(), sine_amplitudes=(_expected_amplitude(),) * 10)
    values = list(primitives[field])
    value = values[0]
    values[0] = I(2 * value.lo, 2 * value.hi) if isinstance(value, I) else 2 * value
    primitives[field] = values
    result = owner._compare_organized_electrical_sine_amplitudes(**primitives)
    expected = (
        "sine_amplitude_uniformity_excluded"
        if field == "sine_amplitudes"
        else "component_shape_excluded"
    )
    assert result.status == expected
    assert result.exclusions


def test_overlapping_component_ranges_remain_unresolved_not_equality():
    primitives = _primitives()
    for field in ("capacitances", "conductances"):
        primitives[field] = tuple(
            I(value / 2, 2 * value) for value in primitives[field]
        )
    j = _expected_amplitude()
    result = owner._compare_organized_electrical_sine_amplitudes(
        **primitives, sine_amplitudes=(I(j.lo / 2, 2 * j.hi),) * 10
    )
    assert result.status == "necessary_relation_not_excluded"
    assert not result.exclusions
    assert result.calibration.clock_scale_bounds.width > 0
    assert result.dimensionless_ratio_bounds.width > 0


def test_cached_success_cannot_substitute_for_re_admitted_component_records():
    primitives = _primitives()
    calibration = owner._calibrate_organized_electrical_components(**primitives)
    fabricated = replace(
        calibration, status="component_shape_not_excluded", exclusions=()
    )
    with pytest.raises((TypeError, ValueError)):
        owner._calibrate_organized_electrical_components(
            **dict(primitives, capacitances=fabricated)
        )
    changed = list(primitives["capacitances"])
    changed[-1] *= 2
    report = owner._compare_organized_electrical_sine_amplitudes(
        **dict(primitives, capacitances=changed),
        sine_amplitudes=(calibration.predicted_sine_amplitude_bounds,) * 10,
    )
    assert report.status == "component_shape_excluded"
    assert calibration.status == "component_shape_not_excluded"


def test_represented_float_primitives_are_normalized_before_storage():
    primitives = _primitives(a0=Q.from_float(0.1))
    capacitors = list(primitives["capacitances"])
    conductances = [0.5] * 15
    result = owner._calibrate_organized_electrical_components(
        capacitances=iter(capacitors),
        conductances=iter(conductances),
        josephson_constant=0.1,
    )
    assert result.status == "component_shape_not_excluded"
    assert result.josephson_constant_bounds.contains(Q.from_float(0.1))
    assert result.conductance_bounds == (I(Q(1, 2)),) * 15
    recorded = result.capacitance_bounds
    capacitors[0] *= 2
    conductances[0] *= 2
    assert result.capacitance_bounds == recorded
    assert result.capacitance_bounds[0].contains(primitives["capacitances"][0])


def test_invalid_excess_stream_is_not_exhausted():
    consumed = []

    def values():
        for index in range(16):
            consumed.append(index)
            yield 1
        pytest.fail("component admission consumed beyond its one-item excess bound")

    with pytest.raises(ValueError):
        owner._calibrate_organized_electrical_components(
            **dict(_primitives(), capacitances=values())
        )
    assert consumed == list(range(16))


@pytest.mark.parametrize(
    "field", ("capacitances", "conductances", "sine_amplitudes", "josephson_constant")
)
@pytest.mark.parametrize(
    "bad",
    (0, -1, False, True, float("nan"), float("inf"), "1", complex(1), Decimal("1")),
)
def test_invalid_physical_scalar_rejected_before_arithmetic(field, bad):
    primitives = dict(_primitives(), sine_amplitudes=(_expected_amplitude(),) * 10)
    if field == "josephson_constant":
        primitives[field] = bad
    else:
        values = list(primitives[field])
        values[-1] = bad
        primitives[field] = values
    with pytest.raises((TypeError, ValueError)):
        owner._compare_organized_electrical_sine_amplitudes(**primitives)


@pytest.mark.parametrize("field", ("capacitances", "conductances", "sine_amplitudes"))
@pytest.mark.parametrize("bad", (None, "123", b"123", {1, 2}, {0: 1}, (), (1,) * 16))
def test_component_vectors_require_order_and_exact_length(field, bad):
    primitives = dict(_primitives(), sine_amplitudes=(_expected_amplitude(),) * 10)
    primitives[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._compare_organized_electrical_sine_amplitudes(**primitives)


@pytest.mark.parametrize("bad", (I(0, 1), I(-1, 1), I(-2, -1), Q(1, 10**100)))
def test_nonpositive_or_unresolvable_interval_is_not_admitted(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._calibrate_organized_electrical_components(
            **dict(_primitives(), josephson_constant=bad)
        )


def test_small_dimensional_components_preserve_positive_finite_calibration():
    c, g, a0 = Q(1, 10**15), Q(1, 10**6), Q(3, 10**16)
    j = _expected_amplitude(c=c, g=g, a0=a0)
    report = owner._compare_organized_electrical_sine_amplitudes(
        **_primitives(c=c, g=g, a0=a0), sine_amplitudes=(j,) * 10
    )
    assert report.status == "necessary_relation_not_excluded"
    assert report.calibration.clock_scale_bounds.contains(4 * 10**9)
    assert report.calibration.charge_scale_bounds.lo > 0
    assert report.calibration.predicted_sine_amplitude_bounds.lo > 0
    assert report.dimensionless_ratio_bounds.lo > 0
    assert report.dimensionless_ratio_bounds.width < Q(1, 10**12)


def test_cold_identification_has_no_source_or_solver_dependency(
    source_tree_environment,
):
    script = """
import importlib.abc
import sys
from fractions import Fraction as Q

attempted = []
forbidden = (
    'tnfr.physics._organized_electrical_source',
    'tnfr.physics._organized_joint_resources',
    'tnfr.physics._organized_outer_winding',
    'tnfr.physics._organized_source_capture',
    'tnfr.physics._organized_parent_acquisition',
    'tnfr.physics._sine_flow',
    'tnfr.research',
    'tnfr.mathematics._validated_taylor',
    'tnfr.mathematics._validated_metric',
)
class WithoutSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(forbidden):
            attempted.append(fullname)
            raise AssertionError('component calculation imported source: ' + fullname)

sys.meta_path.insert(0, WithoutSource())
from tnfr.physics import _organized_electrical_identification as owner
values = []
for a in range(6):
    for b in range(a + 1, 6):
        numerator = 11 if b == 5 else (5 if (a - b) % 5 in (1, 4) else 41)
        values.append(Q(numerator, 396))
report = owner._compare_organized_electrical_sine_amplitudes(
    capacitances=values, conductances=(Q(1, 6),) * 15,
    josephson_constant=1, sine_amplitudes=(1,) * 10,
)
assert report.status == 'sine_amplitude_relation_excluded'
assert not attempted
assert not any(name.startswith(forbidden) for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=source_tree_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
