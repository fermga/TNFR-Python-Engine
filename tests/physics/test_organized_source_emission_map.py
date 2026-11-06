"""Static scope controls for fixed-time Emission on an open conserved leaf.

Analytic continuation is a theorem, not a conclusion from sampled endpoints.
The auxiliary countermodels below isolate its open-domain, analyticity,
fixed-duration and positive-gain hypotheses; they are not organized-source
realizations. Existing source-port/capture owners verify the full fine law,
conservation and equilibrium rows, without needing a new source assessment.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators.al_sha_stage_proposals import emission_epi_proposal
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as port

GAIN = Q(1, 8)


@pytest.fixture(scope="module", autouse=True)
def no_response_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Emission-map controls cannot execute or rebuild a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (capture, "_assess_organized_source_capture"),
            (capture, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def barrier():
    return capture._derive_organized_source_capture_barrier()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _leaf_chart(barrier, free):
    """Eliminate only the last form and offset using the two conserved sums."""
    assert len(free) == 88
    form_shift = tuple(free[:44]) + (-sum(free[:44]),)
    phase_shift = tuple(free[44:]) + (-sum(free[44:]),)
    return (
        tuple(a + b for a, b in zip(barrier.target_form, form_shift)),
        tuple(a + b for a, b in zip(barrier.target_phase_offsets, phase_shift)),
    )


def _chart_inverse(barrier, form, offsets):
    return tuple(a - b for a, b in zip(form[:44], barrier.target_form[:44])) + tuple(
        a - b for a, b in zip(offsets[:44], barrier.target_phase_offsets[:44])
    )


def test_full_leaf_chart_retains_all_eighty_eight_independent_directions(barrier):
    zero = (Q(0),) * 88
    assert _leaf_chart(barrier, zero) == (
        barrier.target_form,
        barrier.target_phase_offsets,
    )
    # An explicit left inverse checks independence, not only the number of
    # coordinates. Both signs remain on the same affine leaf.
    for axis in range(88):
        for sign in (-1, 1):
            free = tuple(Q(sign, 64) if i == axis else Q(0) for i in range(88))
            form, offsets = _leaf_chart(barrier, free)
            assert sum(form) == 0
            assert sum(offsets) == Q(9, 8)
            assert _chart_inverse(barrier, form, offsets) == free
    left = tuple(Q(i % 7 - 3, 1024) for i in range(88))
    right = tuple(Q(i % 5 - 2, 512) for i in range(88))
    middle = tuple((a + b) / 2 for a, b in zip(left, right))
    images = tuple(_leaf_chart(barrier, point) for point in (left, right, middle))
    for channel in (0, 1):
        assert images[2][channel] == tuple(
            (a + b) / 2 for a, b in zip(images[0][channel], images[1][channel])
        )


def test_positive_child_increment_can_preserve_the_global_form_sum(barrier):
    free = tuple(Q(i % 7 - 3, 1024) for i in range(88))
    form, offsets = _leaf_chart(barrier, free)
    changed = list(form)
    policy = {"CLIP_MODE": "hard", "EPI_MIN": -1.0, "EPI_MAX": 1.0}
    for node in range(9):
        before, after = emission_epi_proposal(policy, form[node], float(GAIN))
        assert -1 < form[node] < form[node] + GAIN < 1
        assert Q.from_float(before) == form[node]
        changed[node] = Q.from_float(after)
    # This independently supplied compensation is not part of Emission, an
    # executed support event or a claim that the continuous law realizes it.
    for node in range(36, 45):
        changed[node] -= GAIN
    assert sum(changed) == sum(form) == 0
    assert sum(changed[:9]) / 9 - sum(form[:9]) / 9 == GAIN
    assert sum(changed[36:]) / 9 - sum(form[36:]) / 9 == -GAIN
    assert tuple(changed[9:36]) == form[9:36]
    recovered = _chart_inverse(barrier, changed, offsets)
    assert _leaf_chart(barrier, recovered) == (tuple(changed), offsets)


def _oscillator(mp, x, y, time):
    return (
        x * mp.cos(time) + y * mp.sin(time),
        -x * mp.sin(time) + y * mp.cos(time),
    )


def test_analytic_fixed_time_gain_can_hold_on_a_slice_without_an_open_domain(mp):
    # The analytic oscillator x'=y, y'=-x has quarter-period map (y,-x).
    # The gain slice y=x+b is codimension one; transverse changes destroy it.
    time = mp.pi / 2
    for x in (-Q(1, 4), Q(0), Q(1, 4)):
        y = x + GAIN
        exact_endpoint = (y, -x)
        actual = _oscillator(mp, _mp(mp, x), _mp(mp, y), time)
        for observed, expected in zip(actual, exact_endpoint):
            assert abs(observed - _mp(mp, expected)) < mp.mpf("1e-70")
        assert exact_endpoint[0] - x == GAIN
        for transverse in (-Q(1, 64), Q(1, 64)):
            perturbed = _oscillator(mp, _mp(mp, x), _mp(mp, y + transverse), time)[0]
            gain = perturbed - _mp(mp, x)
            assert abs(gain - _mp(mp, GAIN + transverse)) < mp.mpf("1e-70")
            assert abs(gain - _mp(mp, GAIN)) > _mp(mp, abs(transverse) / 2)
    # Independently check the declared vector field from the closed formula.
    x, y, probe = mp.mpf(1) / 4, mp.mpf(1) / 2, mp.mpf(1) / 7
    point = _oscillator(mp, x, y, probe)
    velocity = tuple(
        mp.diff(lambda time: _oscillator(mp, x, y, time)[axis], probe)
        for axis in (0, 1)
    )
    assert abs(velocity[0] - point[1]) < mp.mpf("1e-70")
    assert abs(velocity[1] + point[0]) < mp.mpf("1e-70")


def _smooth_step(mp, value):
    # The standard flat extension is C-infinity, but not real analytic at
    # 0 or 1. Its smoothness is analytical, not inferred from these samples.
    if value <= 0:
        return mp.mpf(0)
    if value >= 1:
        return mp.mpf(1)
    left, right = mp.exp(-1 / value), mp.exp(-1 / (1 - value))
    return left / (left + right)


def test_smooth_nonanalytic_field_has_a_fixed_time_translation_away_from_equilibrium(
    mp,
):
    assert _smooth_step(mp, mp.mpf(0)) == 0
    assert 0 < _smooth_step(mp, mp.mpf(1) / 4) < 1
    assert _smooth_step(mp, mp.mpf(1) / 2) == mp.mpf(1) / 2
    # For x'=b*chi(x), the whole open interval (2,3) stays in chi=1 until
    # T=1. Its exact solution there is x(t)=x(0)+bt; no solver is involved.
    for initial in (Q(9, 4), Q(5, 2), Q(11, 4)):
        for time in (Q(0), Q(1, 4), Q(1, 2), Q(1)):
            point = initial + GAIN * time
            assert point > 1
            velocity = _mp(mp, GAIN) * _smooth_step(mp, _mp(mp, point))
            assert velocity == _mp(mp, GAIN)
        assert (initial + GAIN) - initial == GAIN
    assert _mp(mp, GAIN) * _smooth_step(mp, mp.mpf(0)) == 0


def test_state_dependent_duration_allows_translation_for_an_analytic_flow(mp):
    # The analytic flow x'=x fixes zero. On x>0, T(x)=log(1+b/x) gives
    # exp(T(x))*x=x+b; this is not one common-time flow map.
    times = []
    for initial in (GAIN, 2 * GAIN, 4 * GAIN):
        x, gain = _mp(mp, initial), _mp(mp, GAIN)
        time = mp.log(1 + gain / x)
        times.append(time)
        endpoint = mp.exp(time) * x
        assert time > 0
        assert abs(endpoint - x - gain) < mp.mpf("1e-70")
        fixed_gain = mp.exp(mp.log(2)) * x - x
        assert abs(fixed_gain - x) < mp.mpf("1e-70")
        if initial != GAIN:
            assert fixed_gain > gain
    assert times[0] > times[1] > times[2] > 0
    assert mp.exp(times[0]) * 0 == 0


def test_zero_gain_does_not_exclude_a_nontrivial_periodic_analytic_flow(mp):
    # A full-period identity is compatible with a fixed point and an analytic
    # nonzero vector field elsewhere. The obstruction needs b>0.
    for initial in ((Q(0), Q(0)), (Q(1, 4), Q(1, 2)), (-Q(1, 2), Q(1, 8))):
        x, y = (_mp(mp, value) for value in initial)
        endpoint = _oscillator(mp, x, y, 2 * mp.pi)
        assert max(abs(endpoint[0] - x), abs(endpoint[1] - y)) < mp.mpf("1e-70")
        if initial != (Q(0), Q(0)):
            halfway = _oscillator(mp, x, y, mp.pi)
            assert max(abs(halfway[0] - x), abs(halfway[1] - y)) > mp.mpf(".1")
