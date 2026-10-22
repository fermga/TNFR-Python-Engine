"""Exact phase-channel aliases, complete-row coverings and scope controls.

These static identities use primitive Fourier laws and supplied states. They
neither regenerate research responses nor identify a preferred physical law.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from functools import reduce
from math import pi

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.operators._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors
from tnfr.physics._composition_compatible_constitutive import (
    _derive_composition_compatible_law,
)
from tnfr.physics._phase_boundary_information import (
    _derive_boundary_information,
    _derive_phase_response_faithfulness,
    _evaluate_boundary_response,
)
from tnfr.physics._sine_class_constitutive_comparison import (
    _current_fourier_coefficients,
)
from tnfr.physics.phase_resultant_sectors import _winding


@pytest.fixture(scope="module", autouse=True)
def no_response_or_source_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _laurent_current(coefficients, phasor):
    """Independent signed Fourier expansion, with no trigonometric sampling."""
    return sum(
        sp.Rational(scale) * (phasor**order - phasor ** (-order)) / (2 * sp.I)
        for order, scale in coefficients
    )


def _current(coefficients, gap, query=(1, 0)):
    return _evaluate_boundary_response(
        coefficients, gap_phasors=(gap,), query_phasor=query
    ).current


@pytest.mark.parametrize(
    "support", [(2, 4), (8, 12), (6, 10, 14), (6, 10, 15), (2, 3), (1, 9, 18)]
)
def test_common_rotation_polynomial_independently_certifies_shift_group(support):
    z = sp.Symbol("z")
    coefficients = tuple(
        (order, Q((-1) ** i, i + 2)) for i, order in enumerate(support)
    )
    report = _derive_phase_response_faithfulness(coefficients)
    # A rotation is invisible to every query iff every active complex Fourier
    # coefficient is unchanged. Their common roots are obtained without gcd
    # of the integer orders or the production function's reported subgroup.
    common = reduce(sp.gcd, (sp.Poly(z**order - 1, z) for order in support)).monic()
    degree = common.degree()
    assert common.as_expr() == z**degree - 1
    assert sp.gcd(common, common.diff()).degree() == 0
    assert report.shift_subgroup_order == degree
    assert report.fundamental_period_turns == Q(1, degree)
    assert report.faithful == (common.as_expr() == z - 1)
    assert report.coefficients == coefficients


def test_even_law_alias_is_all_query_identity_but_extra_observation_separates():
    coefficients = ((2, Q(2, 5)), (4, Q(-3, 7)))
    z = sp.Symbol("z", nonzero=True)
    assert (
        sp.expand(
            _laurent_current(coefficients, -z) - _laurent_current(coefficients, z)
        )
        == 0
    )
    gap, shifted = (Q(3, 5), Q(4, 5)), (Q(-3, 5), Q(-4, 5))
    for query in ((1, 0), (0, 1), (Q(5, 13), Q(12, 13))):
        assert _current(coefficients, gap, query) == _current(
            coefficients, shifted, query
        )
    report = _derive_phase_response_faithfulness(coefficients)
    assert not report.faithful
    # Primitive phase, or an added third-harmonic phase readout, distinguishes
    # these gaps. Observation augmentation does not change the supplied law.
    phase = sp.Rational(3, 5) + sp.I * sp.Rational(4, 5)
    assert phase != -phase
    assert sp.expand(phase**3 - (-phase) ** 3) != 0
    assert (
        sp.gcd(sp.Symbol("w") ** 2 - 1, sp.Symbol("w") ** 3 - 1) == sp.Symbol("w") - 1
    )


def test_coprime_harmonics_need_no_first_harmonic_or_injective_single_reading():
    coefficients = ((2, Q(1, 3)), (3, Q(2, 7)))
    report = _derive_phase_response_faithfulness(coefficients)
    assert report.faithful
    assert 1 not in dict(report.coefficients)
    assert _derive_boundary_information(coefficients).minimum_regular_dimension == 4
    # Equal current at one registration is not equality of the response
    # profile. The quarter-turn query detects the active odd harmonic.
    assert _current(coefficients, (1, 0)) == _current(coefficients, (-1, 0)) == 0
    difference = _current(coefficients, (1, 0), (0, 1)) - _current(
        coefficients, (-1, 0), (0, 1)
    )
    assert difference == 2 * dict(coefficients)[3] != 0


def test_tiny_exact_harmonic_breaks_alias_and_fresh_inputs_replace_cached_claims():
    tiny = Q(1, 10**1000)
    coefficients = [(4, Q(-2, 5)), (2, Q(3, 7)), (1, Q(0))]
    before = _derive_phase_response_faithfulness(coefficients)
    coefficients[-1] = (1, tiny)
    after = _derive_phase_response_faithfulness(coefficients)
    assert not before.faithful and after.faithful
    assert before.coefficients == tuple(coefficients[:-1])
    assert after.coefficients == tuple(coefficients)
    assert _current(coefficients, (0, 1)) - _current(coefficients, (0, -1)) == 2 * tiny
    assert 0 < 2 * tiny < Q(1, 10**999)
    with pytest.raises(FrozenInstanceError):
        before.faithful = True
    # Reports cannot be supplied in place of newly admitted primitive rows.
    with pytest.raises(TypeError):
        _derive_phase_response_faithfulness(after)


@pytest.mark.parametrize("coefficients", [(), ((1, Q(0)), (2, -0.0))])
def test_zero_law_has_whole_circle_aliases_without_a_least_period(coefficients):
    z = sp.Symbol("z", nonzero=True)
    assert _laurent_current(coefficients, z) == 0
    report = _derive_phase_response_faithfulness(coefficients)
    assert report.shift_subgroup_order is report.fundamental_period_turns is None
    assert not report.faithful
    for gap in ((1, 0), (0, 1), (Q(3, 5), Q(4, 5))):
        assert _current(coefficients, gap, (Q(5, 13), Q(-12, 13))) == 0


@pytest.mark.parametrize("deformation", [Q(0), Q(1, 10**400), Q(1, 324)])
def test_retained_counterfamilies_remain_faithful_without_becoming_same_law(
    deformation,
):
    laws = (
        _current_fourier_coefficients(deformation),
        _derive_composition_compatible_law(
            deformation=deformation
        ).current_coefficients,
    )
    z = sp.Symbol("z", nonzero=True)
    first_harmonic = _laurent_current(((1, 1),), z)
    for coefficients in laws:
        report = _derive_phase_response_faithfulness(coefficients)
        assert dict(report.coefficients)[1] > 0
        assert report.faithful
        correction = sp.expand(_laurent_current(coefficients, z) - first_harmonic)
        assert (correction == 0) == (deformation == 0)
    # Over the whole supplied composition interval its first coefficient
    # cannot vanish, independently of these representative exact members.
    assert 1 - Q(3, 2) * Q(1, 324) > 0


def test_faithful_unit_slope_law_need_not_have_positive_acute_curvature():
    angle = sp.Symbol("angle", real=True)
    coefficients = ((1, Q(1, 4)), (3, Q(1, 4)))
    potential = sum(
        sp.Rational(scale) * (1 - sp.cos(order * angle)) / order
        for order, scale in coefficients
    )
    current = sp.diff(potential, angle)
    curvature = sp.diff(current, angle)
    assert all(scale > 0 for _, scale in coefficients)
    assert _derive_phase_response_faithfulness(coefficients).faithful
    assert curvature.subs(angle, 0) == 1
    assert curvature.subs(angle, sp.pi / 3) == -sp.Rational(5, 8)
    assert 0 < sp.pi / 3 < sp.pi / 2


def test_pure_harmonic_covering_transports_both_rows_storage_and_loss():
    n = sp.Symbol("n", integer=True, positive=True)
    gamma = sp.Symbol("gamma", positive=True)
    form = sp.Matrix(sp.symbols("x0:3", real=True))
    phase = sp.Matrix(sp.symbols("theta0:3", real=True))
    incidence = sp.Matrix([[-1, 1, 0], [0, -1, 1], [-1, 0, 1]])
    weights = sp.diag(2, 3, 5)
    laplacian = incidence.T * weights * incidence

    def rows_and_storage(x, theta, harmonic):
        gaps = incidence * theta
        gradient = (
            incidence.T
            * weights
            * gaps.applyfunc(lambda gap: sp.sin(harmonic * gap) / harmonic)
        )
        dx = (-laplacian * x - gamma * gradient) / 4
        dtheta = gamma * laplacian * x / 4
        storage = (x.T * laplacian * x)[0] / 2 + sum(
            weights[i, i] * (1 - sp.cos(harmonic * gap)) / harmonic**2
            for i, gap in enumerate(gaps)
        )
        return dx, dtheta, storage, gradient

    dx, dtheta, storage, gradient = rows_and_storage(form, phase, n)
    mapped_dx, mapped_dtheta, mapped_storage, _ = rows_and_storage(
        n * form, n * phase, 1
    )
    assert (n * dx - mapped_dx).applyfunc(sp.expand) == sp.zeros(3, 1)
    assert (n * dtheta - mapped_dtheta).applyfunc(sp.expand) == sp.zeros(3, 1)
    assert sp.expand(n**2 * storage - mapped_storage) == 0
    loss = (laplacian * form).dot(laplacian * form) / 4
    assert sp.expand((laplacian * form).dot(dx) + gradient.dot(dtheta) + loss) == 0
    assert (
        sp.expand(
            (laplacian * (n * form)).dot(laplacian * (n * form)) / 4 - n**2 * loss
        )
        == 0
    )
    # Scaling phase alone fails both complete-row coefficients at unchanged
    # clock; matching the periodic current is not enough.
    wrong_dx, wrong_dtheta, _, _ = rows_and_storage(form, n * phase, 1)
    assert (dx - wrong_dx - gamma * (n - 1) * gradient / 4).applyfunc(
        sp.expand
    ) == sp.zeros(3, 1)
    assert (
        n * dtheta - wrong_dtheta - (n - 1) * gamma * laplacian * form / 4
    ).applyfunc(sp.expand) == sp.zeros(3, 1)
    assert (n * dtheta - wrong_dtheta).subs(n, 2) != sp.zeros(3, 1)


def test_covering_collisions_are_not_global_charts_or_u3_equivalence():
    root = -sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2
    assert root != 1
    assert sp.expand(root**3) == 1
    z = sp.Symbol("z", nonzero=True)
    assert (
        sp.expand(
            _laurent_current(((3, Q(1, 3)),), root * z)
            - _laurent_current(((3, Q(1, 3)),), z)
        ).simplify()
        == 0
    )
    assert sp.diff(3 * sp.Symbol("theta"), sp.Symbol("theta")) != 0
    # On an oriented three-cycle the quotient also loses primitive winding.
    # Exact turns use the shared branch-aware winding reduction; no evolving
    # state or numerical phase reconstruction is involved.
    turns = (Q(0), Q(1, 3), Q(2, 3))
    covered = tuple((3 * turn) % 1 for turn in turns)
    assert _winding(turns) == 1
    assert _winding(covered) == 0
    # The n=2 pressure channel aliases gaps 0 and pi at every query, while
    # the existing primitive-phase operator gate admits only the former.
    aliases = ((2, Q(1, 2)),)
    assert _current(aliases, (1, 0), (0, 1)) == _current(aliases, (-1, 0), (0, 1))
    phases = {"neighbor": 0.0}
    admitted = resolve_u3_phase_neighbors(
        {}, 0.0, phases, phase_getter=phases.__getitem__, operator_code="UM"
    )
    assert admitted.neighbors == ("neighbor",)
    phases["neighbor"] = pi
    with pytest.raises(U3PhaseGateError, match="no compatible neighbor"):
        resolve_u3_phase_neighbors(
            {}, 0.0, phases, phase_getter=phases.__getitem__, operator_code="UM"
        )


@pytest.mark.parametrize(
    "invalid", [((2, True),), ((2, 1), (2, 0)), ((1.0, 1),), ((2, float("nan")),)]
)
def test_new_reader_reuses_primitive_law_admission(invalid):
    with pytest.raises((TypeError, ValueError)):
        _derive_phase_response_faithfulness(invalid)
