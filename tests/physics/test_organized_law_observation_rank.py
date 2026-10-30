"""Static premises of the finite-rank proof, without sampled response matrices."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._interval_taylor import cos as jet_cos
from tnfr.mathematics._interval_taylor import sinc
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics._organized_computable_prediction import _forecast_stages
from tnfr.research._organized_forecast_inputs import _sample_times


def _close(left, right):
    difference = left - right
    assert difference.contains(0)
    assert difference.width < Q(1, 10**24)


def _first_ramps():
    return {
        stage.child: stage
        for stage in _forecast_stages()
        if stage.name == "ramp" and stage.base == 0
    }


def test_every_first_ramp_has_both_endpoints_in_the_original_thirty_readings():
    stages, times = _forecast_stages(), _sample_times()
    assert len(times) == len(set(times)) == 30
    assert times[0] == 200 and times[-1] == Q(4280003, 20000)
    ramps = _first_ramps()
    assert set(ramps) == set(range(5))
    for child, ramp in ramps.items():
        previous, following = stages[
            stages.index(ramp) - 1 : stages.index(ramp) + 2 : 2
        ]
        assert previous.name == following.name == "plateau"
        assert previous.child == following.child == child
        assert previous.base == 0 and following.base == Q(1, 10)
        assert previous.start + previous.duration == ramp.start
        assert ramp.start + ramp.duration == following.start
        assert ramp.duration == 1
        assert ramp.start in times and following.start in times


@pytest.mark.parametrize("edge", range(5))
def test_each_edge_receives_opposite_cubic_probes_from_its_two_endpoints(edge):
    ramps = _first_ramps()
    tail, head = edge, (edge + 1) % 5
    for time in (Q(0), Q(1, 4), Q(1, 2), Q(1)):
        expected = Q(1, 10) * time**2 * (3 - 2 * time)
        for child, sign in ((tail, -1), (head, 1)):
            ramp = ramps[child]
            command = ramp.command(ramp.start + time)
            assert sum(command) == 0
            assert command[head] - command[tail] == sign * expected
            changed = {
                node for node in range(5) if command[(node + 1) % 5] != command[node]
            }
            assert changed == ({(child - 1) % 5, child} if time else set())


@pytest.mark.parametrize("mode", (1, 2))
@pytest.mark.parametrize("child", range(5))
def test_adjacent_incidence_columns_are_independent_in_each_c5_eigenspace(mode, child):
    angle = 2 * mode * pi_interval() / 5
    coordinates = tuple((cos(node * angle), sin(node * angle)) for node in range(5))
    eigenvalue = (2 - 2 * cos(angle)) / 4
    expected = (5 + (1 if mode == 2 else -1) * sqrt(5)) / 8
    _close(eigenvalue, expected)
    assert eigenvalue.lo > Q(1, 3)
    for node in range(5):
        for axis in (0, 1):
            laplacian = (
                2 * coordinates[node][axis]
                - coordinates[(node - 1) % 5][axis]
                - coordinates[(node + 1) % 5][axis]
            ) / 4
            _close(laplacian, eigenvalue * coordinates[node][axis])
    columns = tuple(
        tuple(
            coordinates[edge][axis] - coordinates[(edge + 1) % 5][axis]
            for axis in (0, 1)
        )
        for edge in ((child - 1) % 5, child)
    )
    determinant = columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0]
    assert determinant.lo > 0
    _close(determinant, (2 - 2 * cos(angle)) * sin(angle))


def test_first_six_positive_riccati_coefficients_match_independent_cos_over_sinc_jet():
    variable = Jet((I(0), I(1)) + (I(0),) * 11)
    cotangent_product = jet_cos(variable) / sinc(variable)
    coefficients = [Q(1, 3)]
    for degree in range(2, 7):
        coefficients.append(
            sum(
                coefficients[k - 1] * coefficients[degree - k - 1]
                for k in range(1, degree)
            )
            / (2 * degree + 1)
        )
    _close(cotangent_product.coeffs[0], I(1))
    for degree, coefficient in enumerate(coefficients, 1):
        assert coefficient > 0
        _close(cotangent_product.coeffs[2 * degree], I(-coefficient))
        _close(cotangent_product.coeffs[2 * degree - 1], I(0))
