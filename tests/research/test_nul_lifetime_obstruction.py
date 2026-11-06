"""Exact controls for the continuous acute-class escape proof, not a flow run.

Pi-dependent phases below are symbolic real-law data, never materialized as
runtime graph inputs. A finite binary64 family has a different uniformity claim.
"""

from fractions import Fraction as Q
from itertools import combinations

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.research.nul_reset import bound_nul_reset


def _edges():
    weights = {
        (i, j): Q(1)
        for i, j in combinations(range(20), 2)
        if i // 10 == j // 10 and (i % 10 // 2 - j % 10 // 2) % 5 in (1, 4)
    }
    weights.update(
        {
            (0, 1): Q(7, 8),
            (0, 10): Q(15, 16),
            (0, 11): Q(15, 16),
            (1, 10): Q(13, 16),
            (1, 11): Q(13, 16),
        }
    )
    return weights


def test_actual_scalar_image_and_exact_gradients_point_out_of_acute_face():
    factor = canonical_glyph_factor_defaults()["NUL_scale"]
    assert Q(7, 8) < Q(factor) < 1
    image = bound_nul_reset(
        factor=factor,
        capacity=1.0,
        form_bounds=(0.5,) * 3,
        pressure_bounds=(0.0,) * 3,
    )
    assert image.image_form.nominal == Q(factor) / 2
    assert image.capacity_after == Q(factor)
    c = image.image_form.nominal
    forms = (c, Q(-1)) + (Q(0),) * 18
    gradients, degrees = [Q(0)] * 20, [0] * 20
    for (i, j), weight in _edges().items():
        current = weight * (forms[i] - forms[j])
        gradients[i] += current
        gradients[j] -= current
        degrees[i] += 1
        degrees[j] += 1
    assert len(_edges()) == 45
    assert (degrees[1], degrees[2]) == (7, 4)
    assert gradients[2] == 1 - c > Q(1, 2)
    assert gradients[1] == -Q(13, 2) - Q(7, 8) * c < -4
    # Nodes 1 and 2 both retain unit capacity; the nominal target is node 0.
    phase_gap_numerator = gradients[2] / degrees[2] - gradients[1] / degrees[1]
    assert phase_gap_numerator > Q(39, 56)
    assert phase_gap_numerator / pi_interval().hi > Q(39, 224)


def test_all_symbolic_initial_edges_stay_in_the_same_reference_sector():
    pi = pi_interval()
    assert Q(25, 8) < pi.lo < pi.hi < Q(22, 7)
    phase_pi = [Q(2 * (i % 10 // 2), 5) for i in range(20)]
    phase_delta = [0] * 20
    phase_pi[1], phase_delta[1] = -Q(1, 10), 1
    reference_integers = {}
    for i, j in _edges():
        wrap = int(i // 10 == j // 10 and j % 10 // 2 - i % 10 // 2 == 4)
        reference_integers[i, j] = wrap
        a = phase_pi[j] - phase_pi[i] - 2 * wrap
        b = phase_delta[j] - phase_delta[i]
        if (i, j) in ((1, 2), (1, 3)):
            # Exactly pi/2-delta, strictly acute for every 0<delta<1/128.
            assert (a, b) == (Q(1, 2), -1)
            assert pi.lo / 2 - Q(1, 128) > 0
        else:
            gap = I(a) * pi + I(0, Q(1, 128)) * b
            assert gap.lo > -pi.lo / 2
            assert gap.hi < pi.lo / 2
    for base in (0, 10):
        cycle = [base + 2 * k for k in range(5)]
        winding = 0
        for i, j in zip(cycle, cycle[1:] + cycle[:1]):
            key = tuple(sorted((i, j)))
            winding -= reference_integers[key] * (1 if i < j else -1)
        assert winding == 1


def test_global_acceleration_bound_forces_exit_before_eight_delta():
    # |x_dot| <= 1/pi and weighted strength <= bare degree imply
    # |g'(t)-g'(0)| <= 4t/pi^2 globally, without an acute-domain premise.
    horizon = Q(1, 16)
    initial_speed_lower = Q(39, 224)
    speed_change_upper = 4 * horizon / 3**2
    speed_lower = initial_speed_lower - speed_change_upper
    assert speed_lower == Q(295, 2016) > Q(1, 8)
    # These rational controls check the same symbolic inequality at several
    # scales; the proof quantifies over real delta, not sampled float states.
    for exponent in (8, 16, 40, 128):
        delta = Q(1, 2**exponent)
        exit_bound = 8 * delta
        assert exit_bound < horizon
        assert speed_lower * exit_bound > delta
    assert Q(1, 20) - Q(23, 576) == Q(29, 2880)
