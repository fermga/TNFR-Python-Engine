"""Manufactured differential inclusions; no research-source evaluation."""

from fractions import Fraction as Q
from types import SimpleNamespace

import mpmath as mp
import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_metric import validated_metric_taylor_step
from tnfr.mathematics._validated_metric_inclusion import (
    metric_forcing_gain_upper,
    metric_inclusion_observation_boxes,
    validated_metric_inclusion_step,
)


def everywhere(_):
    return (Q(1),)


def zero(state):
    return tuple(0*x for x in state)


def as_mp(x):
    x = Q(x)
    return mp.mpf(x.numerator)/x.denominator


def contains(interval, value):
    return as_mp(interval.lo) <= value <= as_mp(interval.hi)


def scalar_step(flow=zero, **kwargs):
    options = dict(forcing_bound=Q(1, 5), growth_rate=0, order=10)
    options.update(kwargs)
    return validated_metric_inclusion_step(
        (Q(0),), Q(1, 10), ((Q(1),),), Q(1, 4), flow, everywhere, **options
    )


def test_zero_drift_retains_initial_ball_and_continuously_added_forcing():
    step, failed, reason = scalar_step()
    assert step is not None and failed is reason is None
    assert step.forcing_gain_upper_bound == Q(1, 4)
    assert step.endpoint_radius >= Q(3, 20)
    assert step.endpoint_radius < Q(3, 20)+Q(1, 2**100)
    assert step.nominal_step.initial_radius == Q(1, 10)
    assert step.nominal_step.picard_interior_margin > 0
    assert step.nominal_step.tube[0].hi > Q(3, 20)
    assert step.endpoint[0].contains(Q(3, 20))
    assert step.endpoint[0].contains(-Q(3, 20))


@pytest.mark.parametrize('rate', (-Q(1), Q(0), Q(1)))
def test_exact_scalar_forced_envelope_with_signed_growth_and_chaining(rate):
    flow = lambda state: (rate*state[0],)
    center, radius = (Q(1),), Q(1, 10)
    forcing, duration = Q(1, 50), Q(1, 8)
    for index in range(4):
        step, failed, reason = validated_metric_inclusion_step(
            center, radius, ((Q(1),),), duration, flow, everywhere,
            forcing_bound=forcing, growth_rate=rate, order=12, time=index*duration,
        )
        assert step is not None and failed is reason is None
        center, radius = step.endpoint_center, step.endpoint_radius
    with mp.workdps(80):
        total = as_mp(4*duration)
        growth = mp.exp(as_mp(rate)*total)
        gain = total if rate == 0 else (growth-1)/as_mp(rate)
        exact_center = growth
        exact_radius = growth/10+as_mp(forcing)*gain
        assert contains(step.endpoint[0], exact_center+exact_radius)
        assert contains(step.endpoint[0], exact_center-exact_radius)
        assert abs(exact_center-as_mp(center[0]))+exact_radius <= as_mp(radius)
        if rate < 0:
            assert step.forcing_gain_upper_bound < duration
            assert radius < Q(1, 10)


def test_forcing_is_not_differentiated_by_jet_calls():
    calls = {'interval': 0, 'jet': 0}

    def drift(state):
        calls['jet' if isinstance(state[0], Jet) else 'interval'] += 1
        return (0*state[0],)

    step, _, _ = scalar_step(drift, forcing_bound=Q(1))
    assert step is not None and all(calls.values())
    assert all(x == I(0) for x in step.nominal_step.center_polynomial_bounds)
    assert all(x == I(0) for x in step.nominal_step.local_remainder_bounds)
    assert step.forcing_radius_upper_bound == Q(1, 4)


def test_measurable_switching_disturbance_is_enclosed_inside_the_step():
    step, _, _ = validated_metric_inclusion_step(
        (Q(0),), 0, ((1,),), Q(1, 2), zero, everywhere,
        forcing_bound=1, growth_rate=0, order=8,
    )
    bins = metric_inclusion_observation_boxes(step, zero, bin_count=2)
    # d=+1 until1/4 and -1 afterward: endpoint0 hides its interior excursion.
    assert bins[0][2][0].contains(Q(1, 4))
    assert bins[1][2][0].contains(Q(0))
    assert bins[0][2][0].hi < Q(3, 10)
    assert bins[0][2][0].hi >= Q(1, 4)


def test_non_diagonal_rotating_metric_preserves_forcing_radius():
    metric = ((Q(2), Q(1)), (Q(1), Q(2)))

    def flow(state):
        x, y = state
        return (-x-2*y, 2*x+y)

    duration, forcing = Q(1, 4), Q(1, 5)
    step, failed, reason = validated_metric_inclusion_step(
        (Q(0), Q(0)), 0, metric, duration, flow, everywhere,
        forcing_bound=forcing, growth_rate_bounds=(-1, 1), order=8,
    )
    assert step is not None and failed is reason is None
    assert step.nominal_step.growth_rate_upper_bound == 0
    # The nominal order+1 remainder is evaluated on the whole forced tube,
    # so it need not vanish even though this center orbit is identically zero.
    assert forcing*duration <= step.endpoint_radius < forcing*duration+Q(1, 10**5)
    # d(t)=E exp(A*t)u, ||u||_W=1, yields y(t)=E*t*exp(A*t)u.
    with mp.workdps(80):
        t = as_mp(duration)
        cc, ss = mp.cos(mp.sqrt(3)*t), mp.sin(mp.sqrt(3)*t)/mp.sqrt(3)
        u = 1/mp.sqrt(6)
        exact = (as_mp(forcing)*t*u*(cc-3*ss),
                 as_mp(forcing)*t*u*(cc+3*ss))
        assert all(contains(box, x) for box, x in zip(step.endpoint, exact))
        norm2 = 2*exact[0]**2+2*exact[0]*exact[1]+2*exact[1]**2
        assert norm2 <= as_mp(step.endpoint_radius)**2


def test_nonlinear_growth_is_certified_on_the_disturbed_tube():
    flow = lambda state: (state[0]**2,)
    arguments = ((Q(1, 2),), Q(1, 100), ((1,),), Q(1, 16), flow, everywhere)
    step, failed, reason = validated_metric_inclusion_step(
        *arguments, forcing_bound=Q(1, 20), growth_rate_bounds=(0, 4),
        growth_bisections=10, order=10,
    )
    nominal, _, _ = validated_metric_taylor_step(
        *arguments, growth_rate_bounds=(0, 4), growth_bisections=10, order=10,
    )
    assert step is not None and failed is reason is None
    assert step.nominal_step.tube[0].hi > nominal.tube[0].hi
    assert step.nominal_step.growth_certificate.tube == step.nominal_step.tube
    assert step.nominal_step.growth_rate_upper_bound >= 2*step.nominal_step.tube[0].hi
    with mp.workdps(80):
        root = mp.sqrt(mp.mpf(1)/20)
        exact = root*mp.tan(mp.atan(mp.mpf(51)/100/root)+root/16)
        assert contains(step.endpoint[0], exact)


def test_refinement_only_reduces_duhamel_after_robust_tube_admission():
    seen = []

    def refine(tube):
        seen.append(tube)
        return Q(1, 100)

    refined, _, _ = scalar_step(forcing_refinement=refine)
    robust, _, _ = scalar_step()
    assert seen == [refined.nominal_step.tube]
    assert refined.nominal_step == robust.nominal_step
    assert refined.endpoint_radius < robust.endpoint_radius
    assert refined.effective_forcing_bound == Q(1, 100)


@pytest.mark.parametrize('bad', (True, 0.2, -1, Q(2)))
def test_invalid_refined_bound_returns_unavailable(bad):
    step, tube, reason = scalar_step(forcing_refinement=lambda _: bad)
    assert step is None and tube is not None
    assert reason.startswith('inclusion_bound_unavailable:')


@pytest.mark.parametrize('bad', (True, 0.2, -1))
def test_invalid_forcing_rejected_before_callbacks(bad):
    def forbidden(_):
        pytest.fail('invalid primitive reached a field/domain callback')

    with pytest.raises((TypeError, ValueError)):
        validated_metric_inclusion_step(
            (0,), 0, ((1,),), Q(1, 4), forbidden, forbidden,
            forcing_bound=bad, growth_rate=0, order=8,
        )


def test_robust_forcing_domain_failure_does_not_return_nominal_success():
    domain = lambda tube: (Q(1, 8)-tube[0].abs_max,)
    step, tube, reason = validated_metric_inclusion_step(
        (0,), 0, ((1,),), Q(1, 4), zero, domain,
        forcing_bound=1, growth_rate=0, order=8,
    )
    assert step is None and tube is not None
    assert reason == 'whole_disturbed_tube_domain_not_admitted'


def test_failed_fixed_growth_bracket_is_not_expanded():
    step, tube, reason = scalar_step(
        lambda state: (2*state[0],), growth_rate=None, growth_rate_bounds=(0, 1),
    )
    assert step is None and tube is not None
    assert reason == 'declared_growth_upper_endpoint_not_certified'


def test_external_growth_certificate_is_called_once_and_binds_exact_tube():
    seen = []
    metric = ((Q(1),),)

    def certifier(tube):
        seen.append(tube)
        return SimpleNamespace(certified=True, tube=tube, metric=metric,
                               growth_rate_upper_bound=Q(1, 2), reason=None)

    step, failed, reason = scalar_step(
        lambda state: (state[0]/2,), growth_rate=None, growth_certifier=certifier,
    )
    assert step is not None and failed is reason is None
    assert seen == [step.growth_proof_tube]
    assert step.growth_proof_tube == step.nominal_step.tube
    assert step.external_growth_certificate.growth_rate_upper_bound == Q(1, 2)
    assert step.nominal_step.growth_certificate is None


@pytest.mark.parametrize('mismatch', ('metric', 'boolean_metric', 'tube', 'clock', 'rate', 'verdict'))
def test_external_growth_mismatch_is_not_admitted(mismatch):
    def certifier(tube):
        values = dict(certified=True, tube=tube, metric=((Q(1),),),
                      growth_rate_upper_bound=Q(0), prescribed_time_interval=None)
        if mismatch == 'metric':
            values['metric'] = ((Q(2),),)
        elif mismatch == 'boolean_metric':
            values['metric'] = ((True,),)
        elif mismatch == 'tube':
            values['tube'] = (I(0),)
        elif mismatch == 'clock':
            values['prescribed_time_interval'] = I(0, 1)
        elif mismatch == 'rate':
            values['growth_rate_upper_bound'] = False
        else:
            values['certified'] = 1
        return SimpleNamespace(**values)

    step, tube, reason = scalar_step(growth_rate=None, growth_certifier=certifier)
    assert step is None and tube is not None
    assert reason.startswith('external_growth_unavailable:')


def test_external_failed_proof_stops_before_taylor_jets():
    def flow(state):
        assert not isinstance(state[0], Jet), 'failed proof reached Taylor jets'
        return (0*state[0],)

    def certifier(tube):
        return SimpleNamespace(certified=False, tube=tube, metric=((Q(1),),),
                               growth_rate_upper_bound=None, reason='fixed_bound_failed')

    step, tube, reason = scalar_step(flow, growth_rate=None, growth_certifier=certifier)
    assert step is None and tube is not None and reason == 'fixed_bound_failed'


def test_recomputed_tube_cannot_escape_external_proof_coverage():
    changed = False

    def flow(state):
        return (0*state[0]+int(changed),)

    def certifier(tube):
        nonlocal changed
        changed = True
        return SimpleNamespace(certified=True, tube=tube, metric=((Q(1),),),
                               growth_rate_upper_bound=Q(0), reason=None)

    step, tube, reason = scalar_step(flow, growth_rate=None, growth_certifier=certifier)
    assert step is None and tube is not None
    assert reason == 'nominal_tube_exceeds_external_growth_proof_tube'


def test_prescribed_time_drift_keeps_forcing_out_of_clock_jets():
    seen = []

    def flow(time, state):
        return (time+0*state[0],)

    def refine(clock, tube):
        seen.append(clock)
        return Q(1, 100)

    step, failed, reason = validated_metric_inclusion_step(
        (0,), 0, ((1,),), Q(1, 4), flow, lambda clock, tube: (Q(1),),
        forcing_bound=Q(1, 50), forcing_refinement=refine,
        growth_rate=0, order=8, time=2, time_dependent=True,
    )
    assert step is not None and failed is reason is None
    assert seen == [I(2, Q(9, 4))]
    assert step.nominal_step.center_solution_endpoint[0].contains(Q(17, 32))
    assert step.endpoint[0].contains(Q(17, 32)+Q(1, 400))
    assert step.endpoint[0].contains(Q(17, 32)-Q(1, 400))


def test_zero_forcing_preserves_nominal_certificate_values():
    flow = lambda state: (-state[0],)
    args = ((Q(1),), Q(1, 10), ((1,),), Q(1, 8), flow, everywhere)
    old, _, _ = validated_metric_taylor_step(*args, growth_rate=-1, order=10)
    new, _, _ = validated_metric_inclusion_step(
        *args, forcing_bound=0, growth_rate=-1, order=10,
    )
    assert new.nominal_step == old
    assert new.endpoint == old.endpoint
    assert new.endpoint_radius == old.endpoint_radius


@pytest.mark.parametrize('rate', (-1, 0, 1))
def test_duhamel_gain_bounds_independent_analytic_integral(rate):
    upper = metric_forcing_gain_upper(rate, Q(1, 2))
    with mp.workdps(80):
        exact = mp.mpf('0.5') if rate == 0 else (mp.exp(mp.mpf(rate)/2)-1)/rate
        assert exact <= as_mp(upper) < exact+mp.mpf('1e-20')
    assert metric_forcing_gain_upper(rate, 0) == 0


def test_gain_rejects_unsupported_exponent_instead_of_extending_policy():
    with pytest.raises(ValueError, match='abs'):
        metric_forcing_gain_upper(3, Q(1, 2))
