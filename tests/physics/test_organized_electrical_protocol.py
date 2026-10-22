"""Independent synthetic controls for the fixed five-child electrical bridge.

Matrix, row, storage and work identities are exact algebraic controls. Voltage
records solve a static quadrature identity and are not trajectories, acquired
sources or reserved responses. No physical calibration or experiment runs.
"""

import subprocess
import sys
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_product,
)
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_electrical_protocol as owner
from tnfr.physics import _organized_electrical_readout as readout

PAIRS = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
RING = tuple((i, j) for i, j in PAIRS if (i - j) % 5 in (1, 4))
CHORDS = tuple(pair for pair in PAIRS if pair not in RING)


def _components(c=Q(2), g=Q(3), a0=Q(5)):
    return {
        "capacitances": (c / 5,) * 5,
        "conductances": (g / 5,) * 10,
        "josephson_constant": a0,
    }


def _laplacian(pairs, values, size=5):
    matrix = [[Q(0)] * size for _ in range(size)]
    for (a, b), value in zip(pairs, values):
        matrix[a][a] += value
        matrix[b][b] += value
        matrix[a][b] -= value
        matrix[b][a] -= value
    return tuple(tuple(row) for row in matrix)


def _action(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _expected_amplitude(c=Q(2), g=Q(3), a0=Q(5)):
    pi = pi_interval()
    return I(
        a0 * g**2 / (c * (1023 * pi.hi) ** 2), a0 * g**2 / (c * (1023 * pi.lo) ** 2)
    )


@pytest.fixture(autouse=True)
def no_source_assessment_or_trajectory(request):
    if request.node.name == "test_cold_calibration_does_not_import_source_or_solver":
        yield
        return
    from tnfr.physics import _organized_electrical_source as source
    from tnfr.physics import _organized_finite_observation as finite
    from tnfr.physics import _organized_finite_observation_source as finite_source

    def forbidden(*args, **kwargs):
        pytest.fail("a frozen or acquired source was executed by a synthetic control")

    pure_bound = finite._bound_organized_finite_observation

    def synthetic_bound(*, raw_readout_error):
        if raw_readout_error == Q(1, 10**14):
            pytest.fail("the reserved finite-observation corner was rerun")
        return pure_bound(raw_readout_error=raw_readout_error)

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(source, "_assess_organized_electrical_source", forbidden)
        patch.setattr(source, "_assess_organized_joint_resources", forbidden)
        patch.setattr(finite_source, "_assess_organized_finite_observation", forbidden)
        patch.setattr(finite, "_bound_organized_finite_observation", synthetic_bound)
        yield


@pytest.fixture(scope="module")
def matrices():
    lap = _laplacian(RING, (Q(1),) * 5)
    shifted = tuple(tuple(x + Q(1, 5) for x in row) for row in lap)
    inverse = exact_matrix_inverse(shifted)
    pinv = tuple(tuple(x - Q(1, 5) for x in row) for row in inverse)
    projection = tuple(tuple(Q(i == j) - Q(1, 5) for j in range(5)) for i in range(5))
    return lap, pinv, projection


def test_passive_C5_matrices_follow_independent_exact_inverse(matrices):
    lap, pinv, projection = matrices
    assert owner._C5_PAIRS == PAIRS
    assert owner._C5_RING_PAIRS == RING
    assert owner._C5_CAPACITOR_PAIRS == CHORDS
    assert len(RING) == len(CHORDS) == 5
    assert exact_square_matrix_product(lap, pinv) == projection
    assert _laplacian(CHORDS, (Q(1, 5),) * 5) == pinv
    assert _laplacian(PAIRS, (Q(1, 5),) * 10) == projection
    assert all(pinv[i][i] == Q(2, 5) for i in range(5))
    assert all(pinv[i][j] == 0 for i, j in RING)
    assert all(pinv[i][j] == -Q(1, 5) for i, j in CHORDS)
    assert max(sum(abs(x) for x in row) for row in pinv) == Q(4, 5)


@pytest.mark.parametrize("vector", ((1, 2, -1, 3, -5), (1, 0, 0, 0, 0), (7,) * 5))
def test_branch_energies_prove_passivity_without_claiming_full_rank(vector):
    for pairs, values in ((CHORDS, (Q(2, 5),) * 5), (PAIRS, (Q(3, 5),) * 10)):
        matrix = _laplacian(pairs, values)
        quadratic = _dot(vector, _action(matrix, vector))
        assert quadratic == sum(
            w * (vector[a] - vector[b]) ** 2 for (a, b), w in zip(pairs, values)
        )
        assert quadratic >= 0
        assert (quadratic == 0) == (len(set(vector)) == 1)
        assert _action(matrix, (Q(1),) * 5) == (Q(0),) * 5


@pytest.mark.parametrize("gamma", (Q(1, 7), Q(3, 11), Q(2)))
def test_full_rows_energy_loss_and_supplied_work_match_with_one_clock(matrices, gamma):
    lap, pinv, projection = matrices
    a0, c, g = Q(5, 3), Q(7, 2), Q(11, 13)
    clock, charge = 4 * g / c, a0 * gamma * g / 9
    j, voltage, energy = (
        a0 * gamma**2 * g**2 / c,
        a0 * gamma * g / c,
        a0**2 * gamma**2 * g**2 / (9 * c),
    )
    U = (Q(1), Q(-2), Q(3), Q(-5), Q(7))
    forcing = (Q(1), Q(-1), Q(2), Q(3), Q(-5))
    # Independently supplied antisymmetric edge currents exercise both rows.
    S = [Q(0)] * 5
    for (a, b), current in zip(RING, (Q(2, 7), Q(-1, 3), Q(4, 5), Q(-7, 8), Q(1, 9))):
        S[a] -= current
        S[b] += current
    LU = _action(lap, U)
    PU = _action(projection, U)
    V = tuple(voltage * x for x in LU)
    C = tuple(tuple(c * x for x in row) for row in pinv)
    G = tuple(tuple(g * x for x in row) for row in projection)
    q = tuple(9 * charge * x for x in PU)
    assert _action(C, V) == q
    U_prime = tuple((-x + gamma * s) / 4 + f for x, s, f in zip(LU, S, forcing))
    electric_row = tuple(
        -gv + j * s + 9 * charge * clock * f
        for gv, s, f in zip(_action(G, V), S, forcing)
    )
    assert electric_row == tuple(9 * charge * clock * x for x in U_prime)
    assert tuple(v / a0 for v in V) == tuple(clock * gamma * x / 4 for x in LU)
    assert _dot(V, _action(C, V)) / 2 == energy * 9 * _dot(U, LU) / 2
    assert a0 * j == 9 * energy
    assert energy == a0 * clock * charge * gamma / 4
    assert _dot(V, _action(G, V)) == energy * clock * 9 * _dot(LU, LU) / 4
    physical_input = tuple(9 * charge * clock * f for f in forcing)
    assert _dot(V, physical_input) == energy * clock * 9 * _dot(LU, forcing)


@pytest.mark.parametrize("c,g,a0", ((Q(2), Q(3), Q(5)), (Q(7, 4), Q(9, 5), Q(2, 7))))
def test_independent_C5_calibration_and_predicted_scales(c, g, a0):
    result = owner._calibrate_c5_electrical_components(**_components(c, g, a0))
    assert result.status == "component_shape_not_excluded"
    assert result.capacitance_scale_bounds.contains(c)
    assert result.conductance_scale_bounds.contains(g)
    assert result.clock_scale_bounds.contains(4 * g / c)
    pi = pi_interval()
    gl, gh = 1 / (1023 * pi.hi), 1 / (1023 * pi.lo)
    for interval, low, high in (
        (result.charge_scale_bounds, a0 * gl * g / 9, a0 * gh * g / 9),
        (result.voltage_scale_bounds, a0 * gl * g / c, a0 * gh * g / c),
        (
            result.energy_scale_bounds,
            a0**2 * gl**2 * g**2 / (9 * c),
            a0**2 * gh**2 * g**2 / (9 * c),
        ),
        (
            result.predicted_sine_amplitude_bounds,
            a0 * gl**2 * g**2 / c,
            a0 * gh**2 * g**2 / c,
        ),
    ):
        assert interval.lo <= low <= high <= interval.hi


def test_C5_sine_response_never_calibrates_its_prediction():
    j = _expected_amplitude()
    yes = owner._compare_c5_electrical_sine_amplitudes(
        **_components(), sine_amplitudes=(j,) * 5
    )
    no = owner._compare_c5_electrical_sine_amplitudes(
        **_components(), sine_amplitudes=(I(2 * j.lo, 2 * j.hi),) * 5
    )
    assert yes.status == "necessary_relation_not_excluded"
    assert no.status == "sine_amplitude_relation_excluded"
    assert yes.calibration == no.calibration
    pi = pi_interval()
    assert yes.dimensionless_ratio_bounds.lo <= 1 / (1023 * pi.hi) ** 2
    assert yes.dimensionless_ratio_bounds.hi >= 1 / (1023 * pi.lo) ** 2


@pytest.mark.parametrize("field", ("capacitances", "conductances", "sine_amplitudes"))
def test_one_changed_branch_is_an_exclusion_not_a_fit(field):
    values = dict(_components(), sine_amplitudes=(_expected_amplitude(),) * 5)
    changed = list(values[field])
    changed[0] = 2 * changed[0]
    values[field] = changed
    result = owner._compare_c5_electrical_sine_amplitudes(**values)
    assert result.status == (
        "sine_amplitude_uniformity_excluded"
        if field == "sine_amplitudes"
        else "component_shape_excluded"
    )
    assert result.exclusions


def test_uncertainty_overlap_is_only_a_necessary_condition():
    values = {
        field: tuple(I(x / 2, 2 * x) for x in vector)
        for field, vector in _components().items()
        if field != "josephson_constant"
    }
    result = owner._calibrate_c5_electrical_components(
        **values, josephson_constant=I(4, 6)
    )
    assert result.status == "component_shape_not_excluded"
    assert result.clock_scale_bounds.width > 0
    assert result.energy_scale_bounds.width > 0


def test_final_endpoint_arithmetic_preserves_small_dimensional_energy():
    # Squaring a0 on the interval grid would lose positivity; this final
    # energy is nonetheless resolved because the other scales compensate.
    c, g, a0 = Q(1, 10**15), Q(1, 10**6), Q(3, 10**16)
    result = owner._calibrate_c5_electrical_components(**_components(c, g, a0))
    assert result.energy_scale_bounds.lo > 0
    assert result.voltage_scale_bounds.lo > 0
    assert result.clock_scale_bounds.contains(4 * 10**9)


def test_unresolved_positive_derived_energy_is_an_explicit_domain_error():
    # All primitive components are resolved, but this final positive energy
    # lies below the fixed grid; returning a zero scale would be unsound.
    with pytest.raises(ValueError, match="energy_scale.*unsupported interval domain"):
        owner._calibrate_c5_electrical_components(
            **_components(c=Q(1), g=Q(1), a0=Q(1, 10**18))
        )


@pytest.mark.parametrize(
    "field", ("capacitances", "conductances", "sine_amplitudes", "josephson_constant")
)
@pytest.mark.parametrize(
    "bad",
    (
        0,
        -1,
        False,
        True,
        float("nan"),
        float("inf"),
        "1",
        complex(1),
        Decimal("1"),
        I(0, 1),
        Q(1, 10**100),
    ),
)
def test_invalid_component_primitives_reject_before_arithmetic(field, bad):
    values = dict(_components(), sine_amplitudes=(_expected_amplitude(),) * 5)
    if field == "josephson_constant":
        values[field] = bad
    else:
        values[field] = (*values[field][:-1], bad)
    with pytest.raises((ValueError, TypeError)):
        owner._compare_c5_electrical_sine_amplitudes(**values)


@pytest.mark.parametrize("field", ("capacitances", "conductances", "sine_amplitudes"))
@pytest.mark.parametrize("bad", (None, {1, 2}, {0: 1}, "123", (), (1,) * 11))
def test_component_vectors_have_fixed_order_and_size(field, bad):
    values = dict(_components(), sine_amplitudes=(_expected_amplitude(),) * 5)
    values[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._compare_c5_electrical_sine_amplitudes(**values)


def test_shared_transport_and_C5_comparison_are_separate_bounds():
    report = owner._bound_c5_electrical_departure(
        initial_hidden_form=Q(1, 100),
        initial_hidden_phase=Q(1, 200),
        duration=Q(3, 2),
        collective_form_upper=4,
    )
    g, time = report.gamma_bounds.hi, Q(3, 2)
    A = (Q(1, 100) + 2 * g * time * Q(1, 200)) / (1 - (2 * g * time) ** 2)
    B = (Q(1, 200) + 2 * g * time * Q(1, 100)) / (1 - (2 * g * time) ** 2)
    assert report.hidden_form_upper == A and report.hidden_phase_upper == B
    # They solve the two bounding integral equations without a time solver.
    assert A == Q(1, 100) + 2 * g * time * B
    assert B == Q(1, 200) + 2 * g * time * A
    rho = g * B**2 / 9
    assert (
        report.mean_form_error_upper
        == time * rho + g * time * report.mean_phase_error_upper
    )
    assert report.mean_phase_error_upper == g * time * report.mean_form_error_upper
    assert report.storage_remainder_lower == -4 * B**2
    assert report.storage_remainder_upper == 4 * (A**2 + B**2)
    assert report.integrated_hidden_loss_upper == 16 * time * A**2
    assert report.integrated_hidden_work_upper == 4 * 4 * time * rho


def test_zero_duration_retains_hidden_state_without_invented_flow():
    report = owner._bound_c5_electrical_departure(
        initial_hidden_form=2,
        initial_hidden_phase=3,
        duration=0,
        collective_form_upper=0,
    )
    assert report.hidden_form_upper == 2 and report.hidden_phase_upper == 3
    assert report.mean_form_error_upper == report.mean_phase_error_upper == 0
    assert (
        report.integrated_hidden_loss_upper == report.integrated_hidden_work_upper == 0
    )
    assert report.storage_remainder_lower < 0 < report.storage_remainder_upper


@pytest.mark.parametrize(
    "field",
    (
        "initial_hidden_form",
        "initial_hidden_phase",
        "duration",
        "collective_form_upper",
    ),
)
@pytest.mark.parametrize(
    "bad", (-1, False, True, float("nan"), float("inf"), "1", I(0, 1))
)
def test_hidden_departure_rejects_invalid_norms_clocks_and_bounds(field, bad):
    values = dict(
        initial_hidden_form=0,
        initial_hidden_phase=0,
        duration=1,
        collective_form_upper=1,
    )
    with pytest.raises((TypeError, ValueError)):
        owner._bound_c5_electrical_departure(**dict(values, **{field: bad}))


def test_hidden_transport_requires_its_actual_stronger_domain():
    pi = pi_interval()
    g = I(1 / (1023 * pi.hi), 1 / (1023 * pi.lo)).hi
    with pytest.raises(ValueError, match="gamma\\*lambda\\*time/4 < 1"):
        owner._bound_c5_electrical_departure(
            initial_hidden_form=0,
            initial_hidden_phase=0,
            duration=1 / (2 * g),
            collective_form_upper=0,
        )


def test_actual_fine_sine_rows_obey_quadratic_closure_and_signed_storage(matrices):
    mp = mpmath.mp.clone()
    mp.dps = 80
    gamma = 1 / (1023 * mp.pi)
    lap5 = matrices[0]
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9)
    ) + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
    lap45 = _laplacian(edges, (Q(1),) * 90, size=45)
    U = tuple(Q(c - 2, 20) for c in range(5))
    phi = tuple(mp.mpf(c) / 4 for c in range(5))
    u = tuple(Q((c + 1) * (j - 4), 1000) for c in range(5) for j in range(9))
    z = tuple(Q((c - 2) * (j - 4), 2000) for c in range(5) for j in range(9))

    def real(value):
        return mp.mpf(value.numerator) / value.denominator

    x = tuple(U[c] + u[9 * c + j] for c in range(5) for j in range(9))
    theta = tuple(
        phi[c] + 2 * mp.pi * (j - 4) / 9 + real(z[9 * c + j])
        for c in range(5)
        for j in range(9)
    )
    S45, S5 = [mp.mpf(0)] * 45, [mp.mpf(0)] * 5
    for collection, phases, currents in ((edges, theta, S45), (RING, phi, S5)):
        for a, b in collection:
            current = mp.sin(phases[b] - phases[a])
            currents[a] += current
            currents[b] -= current
    closure = tuple(
        gamma * (sum(S45[9 * c : 9 * c + 9]) / 9 - S5[c]) / 4 for c in range(5)
    )
    a2, b2 = _dot(u, u), _dot(z, z)
    assert abs(sum(closure)) < mp.mpf("1e-75")
    assert sum(abs(x) for x in closure) < gamma * real(b2) / 9
    H45 = real(_dot(x, _action(lap45, x))) / 2 + sum(
        1 - mp.cos(theta[b] - theta[a]) for a, b in edges
    )
    H5 = real(_dot(U, _action(lap5, U))) / 2 + sum(
        1 - mp.cos(phi[b] - phi[a]) for a, b in RING
    )
    remainder = H45 - 45 * (1 - mp.cos(2 * mp.pi / 9)) - 9 * H5
    assert -4 * real(b2) < remainder < 4 * real(a2 + b2)
    Lu = _action(lap45, u)
    assert _dot(Lu, Lu) / 4 <= 16 * a2


def _empty_records():
    return [[[[Q(0)] * 5 for _ in range(2)] for _ in range(3)] for _ in range(5)]


def _voltage_infer(voltages, **kwargs):
    values = dict(
        _components(),
        voltages=voltages,
        voltage_error=Q(1, 10**20),
        mean_form=Q(1, 1000),
    )
    values.update(kwargs)
    return readout._infer_c5_voltage_winding(**values)


def _synthetic_voltages(matrices, winding):
    # Static endpoint identities only. They do not claim a realizable history.
    lap, _, _ = matrices
    mp = mpmath.mp.clone()
    mp.dps = 105
    h = Q(1, 10**5)
    end_inverse = exact_matrix_inverse(
        tuple(
            tuple(Q(i == j) + h * x / 8 for j, x in enumerate(row))
            for i, row in enumerate(lap)
        )
    )
    gamma = 1 / (1023 * mp.pi)
    angle = 2 * mp.pi * winding / 5
    odd_half = gamma * mp.mpf("0.00001") * mp.sin(mp.mpf("0.1")) * mp.cos(angle) / 4
    even_half = (
        gamma * mp.mpf("0.00001") * (1 - mp.cos(mp.mpf("0.1"))) * mp.sin(angle) / 4
    )
    calibration = owner._calibrate_c5_electrical_components(**_components())
    voltage_scale = calibration.voltage_scale_bounds.midpoint
    records = _empty_records()
    for c in range(5):
        receiver = (c + 1) % 5
        for j, increment in enumerate(
            (mp.mpf(0), odd_half + even_half, -odd_half + even_half)
        ):
            rhs = [Q(0)] * 5
            rhs[receiver] = Q(str(increment))
            rhs[(receiver + 1) % 5] = -Q(str(increment))
            endpoint = _action(end_inverse, rhs)
            records[c][j][1] = [voltage_scale * x for x in _action(lap, endpoint)]
    return records


@pytest.mark.parametrize("winding", (-1, 0, 1))
def test_voltage_quadrature_reconstructs_static_compatible_identity(matrices, winding):
    result = _voltage_infer(_synthetic_voltages(matrices, winding))
    assert result.identified_winding == winding
    assert result.form_readout_error_upper < Q(1, 10**14)
    assert result.inference.candidate_windings == (winding,)
    assert all(
        sum(row) / 5 == Q(1, 1000)
        for block in result.form_readings
        for plateau in block
        for row in plateau
    )


def test_common_voltage_gauge_cancels_before_scale_uncertainty(matrices):
    values = _synthetic_voltages(matrices, 1)
    original = _voltage_infer(values)
    shifted = [
        [
            [
                [x + Q(10**500) * (1 + c + j + e) for x in row]
                for e, row in enumerate(plateau)
            ]
            for j, plateau in enumerate(block)
        ]
        for c, block in enumerate(values)
    ]
    result = _voltage_infer(shifted)
    assert result.transformed_voltage_readings == original.transformed_voltage_readings
    assert result.form_readings == original.form_readings
    assert result.form_readout_error_upper == original.form_readout_error_upper
    assert result.identified_winding == 1


def test_readout_scale_uncertainty_and_all_error_channels_are_retained(matrices):
    records = _synthetic_voltages(matrices, 0)
    base = _voltage_infer(records)
    report = _voltage_infer(
        records, mean_form_error=Q(1, 10**17), additional_form_error=Q(1, 10**17)
    )
    assert report.form_readout_error_upper == base.form_readout_error_upper + Q(
        2, 10**17
    )
    lo, hi = report.inverse_voltage_scale_endpoints
    largest = max(
        abs(x)
        for block in report.transformed_voltage_readings
        for plateau in block
        for row in plateau
        for x in row
    )
    assert (
        report.form_readout_error_upper
        == Q(2, 10**17) + hi * Q(4, 5) * report.voltage_error + (hi - lo) * largest / 2
    )
    uncertain = _voltage_infer(records, josephson_constant=I(4, 6))
    assert uncertain.status == "unavailable_frozen_readout_allowance"
    assert uncertain.inference is None and uncertain.identified_winding is None


def test_frozen_budget_cannot_be_replaced_by_more_tolerant_generic_inverse(matrices):
    result = _voltage_infer(
        _synthetic_voltages(matrices, 1), additional_form_error=Q(2, 10**14)
    )
    assert result.status == "unavailable_frozen_readout_allowance"
    assert result.inference is None and result.identified_winding is None
    assert result.unresolved_conditions == (
        "reconstructed_form_error_exceeds_frozen_allowance",
    )


def test_winding_and_voltage_do_not_determine_the_floating_mean(matrices):
    records = _synthetic_voltages(matrices, -1)
    small = _voltage_infer(records, mean_form=0)
    large = _voltage_infer(records, mean_form=1)
    assert small.identified_winding == -1
    assert large.identified_winding is None
    assert not any(large.inference.raw_readings_compatible)
    assert (
        large.inference.corrected_plateau_increments
        == small.inference.corrected_plateau_increments
    )


def test_all_raw_readings_are_admitted_even_when_components_exclude():
    records = _empty_records()
    records[-1][-1][-1][-1] = True
    with pytest.raises((TypeError, ValueError)):
        _voltage_infer(records, capacitances=(Q(1), Q(2), Q(1), Q(1), Q(1)))
    result = _voltage_infer(
        _empty_records(), capacitances=(Q(1), Q(2), Q(1), Q(1), Q(1))
    )
    assert result.status == "unavailable_component_shape"
    assert result.inference is None and result.form_readings is None


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("delta", (-1, 1))
def test_voltage_reading_dimensions_are_fixed_and_ordered(depth, delta):
    records = _empty_records()
    row = records
    for _ in range(depth):
        row = row[0]
    if delta < 0:
        row.pop()
    else:
        row.append(deepcopy(row[-1]))
    with pytest.raises(ValueError):
        _voltage_infer(records)


@pytest.mark.parametrize(
    "field", ("voltage_error", "mean_form_error", "additional_form_error")
)
@pytest.mark.parametrize("bad", (-1, False, True, float("nan"), float("inf"), "1"))
def test_error_channels_require_nonnegative_physical_scalars(field, bad):
    with pytest.raises((ValueError, TypeError)):
        _voltage_infer(_empty_records(), **{field: bad})


def test_cold_calibration_does_not_import_source_or_solver(source_tree_environment):
    script = """
import importlib.abc
import sys
from fractions import Fraction as Q
forbidden = ('tnfr.physics._organized_electrical_source',
             'tnfr.physics._organized_joint_resources',
             'tnfr.physics._organized_finite_observation_source',
             'tnfr.research', 'tnfr.mathematics._validated_taylor')
class NoSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(forbidden):
            raise AssertionError('component admission imported source: ' + fullname)
sys.meta_path.insert(0, NoSource())
from tnfr.physics._organized_electrical_protocol import _compare_c5_electrical_sine_amplitudes
report = _compare_c5_electrical_sine_amplitudes(capacitances=(Q(1, 5),) * 5,
    conductances=(Q(1, 5),) * 10, josephson_constant=1, sine_amplitudes=(1,) * 5)
assert report.status == 'sine_amplitude_relation_excluded'
assert not any(x.startswith(forbidden) for x in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=source_tree_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
