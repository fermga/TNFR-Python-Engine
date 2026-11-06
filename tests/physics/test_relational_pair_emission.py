"""Exact action-descent and reset-work controls, without an event trajectory.

These symbolic checks use the fine member map and an independent graph
quadratic form. Runtime admission and report wiring have separate owners.
"""

import pytest


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


def test_singleton_reset_realisability_and_exact_target_obstruction(symbolic):
    s = symbolic
    center, internal, sine = s.symbols("X u v", real=True)
    first_step, second_step, boost = s.symbols("a b h", nonnegative=True)
    fine = (center + internal, center - internal)

    def quotient(forms):
        return (
            (forms[0] + forms[1]) / 2,
            ((forms[0] - forms[1]) / 2) ** 2,
            (forms[0] - forms[1]) * sine / 2,
        )

    first = quotient((fine[0] + first_step, fine[1]))
    second = quotient((fine[0], fine[1] + second_step))
    # The two potentially clipped increments must agree before the internal
    # coordinates can agree. Equal means alone discard both obstructions.
    assert s.expand(first[0] - second[0]) == (first_step - second_step) / 2
    differences = tuple(
        s.expand((left - right).subs({first_step: boost, second_step: boost}))
        for left, right in zip(first, second)
    )
    assert differences == (0, 2 * boost * internal, boost * sine)
    for _, squared_form, correlation in (first, second):
        # sin(delta)^2=1-R^2 on every stratum, including either zero axis.
        assert s.expand(correlation**2 - squared_form * sine**2) == 0
    assert all(value.subs(boost, 0) == 0 for value in differences)
    assert all(value.subs({internal: 0, sine: 0}) == 0 for value in differences)
    assert differences[1].subs(sine, 0) != 0
    assert differences[2].subs(internal, 0) != 0


def test_whole_pair_action_descends_for_arbitrary_shared_point_map(symbolic):
    s = symbolic
    first_image, second_image, sine = s.symbols("y_plus y_minus v", real=True)
    retained = (
        (first_image + second_image) / 2,
        (first_image - second_image) ** 2 / 4,
        (first_image - second_image) * sine / 2,
    )
    swap = {first_image: second_image, second_image: first_image, sine: -sine}
    assert all(s.expand(value.xreplace(swap) - value) == 0 for value in retained)

    center, internal, boost = s.symbols("X u b", real=True)
    unprojected = {
        first_image: center + internal + boost,
        second_image: center - internal + boost,
    }
    assert tuple(s.expand(v.subs(unprojected)) for v in retained) == (
        center + boost,
        internal**2,
        internal * sine,
    )
    # This symmetry proof allows arbitrary nonlinear images. Preservation of
    # U is an additional affine-map property, not a consequence of the swap.
    assert s.expand(retained[1] - internal**2) != 0


def test_fine_graph_reset_work_matches_quotient_but_does_not_select_port(symbolic):
    s = symbolic
    center, neighbor, internal, neighbor_internal = s.symbols("X Y u v", real=True)
    first_step, second_step, boost, phase_sine = s.symbols("a b h z", real=True)
    # One base edge becomes K_(2,2), with no within-pair edges. Construct its
    # Laplacian from edge differences rather than a quotient-rate formula.
    basis = s.eye(4)
    laplacian = s.zeros(4)
    for first, second in ((0, 2), (0, 3), (1, 2), (1, 3)):
        incidence = basis[:, first] - basis[:, second]
        laplacian += incidence * incidence.T
    forms = s.Matrix(
        (
            center + internal,
            center - internal,
            neighbor + neighbor_internal,
            neighbor - neighbor_internal,
        )
    )
    jump = s.Matrix((first_step, second_step, 0, 0))
    pressure_form = laplacian * forms
    work = s.expand(
        ((forms + jump).T * laplacian * (forms + jump))[0] / 2
        - (forms.T * laplacian * forms)[0] / 2
    )
    balance = (pressure_form.T * jump)[0] + (jump.T * laplacian * jump)[0] / 2
    assert s.expand(work - balance) == 0

    def quotient_storage(mean, offset):
        return 2 * ((mean - neighbor) ** 2 + offset**2 + neighbor_internal**2)

    after_mean = center + (first_step + second_step) / 2
    after_offset = internal + (first_step - second_step) / 2
    quotient_jump = quotient_storage(after_mean, after_offset) - quotient_storage(
        center, internal
    )
    assert s.expand(work - quotient_jump) == 0
    local_work = work.subs({first_step: boost, second_step: 0})
    assert s.expand(local_work - boost * pressure_form[0] - boost**2) == 0

    # At equal form, different phases still distinguish the local outputs.
    # The storage cost cannot see that choice because phase is not reset.
    other_work = work.subs({first_step: 0, second_step: boost})
    assert s.expand((local_work - other_work).subs(internal, 0)) == 0
    first_correlation = boost * phase_sine / 2
    second_correlation = -boost * phase_sine / 2
    assert s.expand(first_correlation - second_correlation) == boost * phase_sine
