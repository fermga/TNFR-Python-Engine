"""Independent phase geometry behind the regional accessibility discriminator.

These are static paths and analytic controls, not research trajectories of
the complete nodal law. Finite path samples supplement the owned proof; they
do not certify a dynamical source reaching an acute winding sector.
"""

import mpmath as mp
import pytest


def _phase_storage(phases, cosine):
    return sum(
        1 - cosine(phases[(i + 1) % len(phases)] - phases[i])
        for i in range(len(phases))
    )


def _principal_gaps(phases):
    return tuple(
        mp.fmod(phases[(i + 1) % len(phases)] - phases[i] + 3 * mp.pi, 2 * mp.pi)
        - mp.pi
        for i in range(len(phases))
    )


def test_nodal_path_derivative_identifies_the_intervening_phase_saddle():
    sym = pytest.importorskip("sympy")
    angle = sym.Symbol("angle", real=True)
    phases = tuple(i * angle for i in range(5))
    potential = _phase_storage(phases, sym.cos)
    derivative = sym.diff(potential, angle)
    factorization = 8 * sym.sin(5 * angle / 2) * sym.cos(3 * angle / 2)
    assert sym.trigsimp(sym.expand_trig(derivative - factorization)) == 0

    # On (0, 2*pi/5), the sine factor is positive and the cosine factor
    # changes sign exactly once, at pi/3. This is the interior maximum.
    saddle = sym.pi / 3
    target = 2 * sym.pi / 5
    assert 0 < saddle < target
    assert sym.simplify(derivative.subs(angle, saddle)) == 0
    assert sym.simplify(sym.diff(derivative, angle).subs(angle, saddle)) == -6
    saddle_storage = sym.simplify(potential.subs(angle, saddle))
    target_storage = sym.simplify(potential.subs(angle, target))
    assert saddle_storage == sym.Rational(7, 2)
    assert 0 < target_storage < saddle_storage


def test_critical_branch_enumeration_has_no_value_between_twist_and_saddle():
    sym = pytest.importorskip("sympy")
    energies = set()
    for obtuse_count in range(6):
        coefficient = 5 - 2 * obtuse_count
        for winding in range(-5, 6):
            alpha = (2 * winding - obtuse_count) * sym.pi / coefficient
            if not -sym.pi / 2 <= alpha <= sym.pi / 2:
                continue
            gaps = [alpha] * (5 - obtuse_count) + [sym.pi - alpha] * obtuse_count
            # Every actual nodal current vanishes when successive edge
            # sines agree. Check this, rather than only evaluating a formula.
            assert all(
                sym.simplify(sym.sin(gap) - sym.sin(gaps[0])) == 0 for gap in gaps
            )
            assert sym.simplify(sum(gaps) - 2 * sym.pi * winding) == 0
            energies.add(sym.simplify(sum(1 - sym.cos(gap) for gap in gaps)))
    positive_values = sorted((value for value in energies if value > 0), key=float)
    assert positive_values[0] == 5 * (1 - sym.cos(2 * sym.pi / 5))
    assert positive_values[1] == sym.Rational(7, 2)
    assert all(value >= 4 for value in positive_values[2:])


def test_acute_nodal_interpolation_preserves_sector_and_lowers_phase_storage():
    with mp.workdps(85):
        source_gaps = tuple(mp.pi * value / 40 for value in (19, 17, 15, 13, 16))
        source = [mp.mpf(0)]
        for gap in source_gaps[:-1]:
            source.append(source[-1] + gap)
        target = tuple(2 * mp.pi * i / 5 for i in range(5))

        def storage_at(fraction):
            phases = tuple(
                (1 - fraction) * a + fraction * b for a, b in zip(source, target)
            )
            return _phase_storage(phases, mp.cos)

        initial_storage = storage_at(mp.mpf(0))
        gap_displacements = tuple(2 * mp.pi / 5 - value for value in source_gaps)
        # All interpolated principal gaps stay in the source/target hull.
        # This positive bound applies to the whole interpolation interval.
        hessian_lower_bound = mp.cos(max(source_gaps)) * sum(
            value**2 for value in gap_displacements
        )
        assert hessian_lower_bound > 0
        assert abs(mp.diff(storage_at, mp.mpf(1))) < mp.mpf("1e-75")
        for numerator in range(9):
            fraction = mp.mpf(numerator) / 8
            phases = tuple(
                (1 - fraction) * a + fraction * b for a, b in zip(source, target)
            )
            principal = _principal_gaps(phases)
            assert all(0 < value < mp.pi / 2 for value in principal)
            assert abs(sum(principal) - 2 * mp.pi) < mp.mpf("1e-75")
            assert mp.diff(storage_at, fraction, 2) >= hessian_lower_bound
            assert storage_at(fraction) <= initial_storage + mp.mpf("1e-75")


def test_first_slip_can_give_nonacute_winding_below_the_acute_accessibility_barrier():
    with mp.workdps(85):
        seam_phases = tuple(i * mp.pi / 4 for i in range(5))
        seam_storage = _phase_storage(seam_phases, mp.cos)
        assert abs(seam_storage - (6 - 2 * mp.sqrt(2))) < mp.mpf("1e-75")
        # Cross the seam along the nodal path, without reaching its saddle.
        angle = mp.pi / 4 + mp.pi / 1000
        phases = tuple(i * angle for i in range(5))
        principal = _principal_gaps(phases)
        storage = _phase_storage(phases, mp.cos)
        assert abs(sum(principal) - 2 * mp.pi) < mp.mpf("1e-75")
        assert max(principal) > mp.pi / 2
        assert seam_storage < storage < mp.mpf(7) / 2
