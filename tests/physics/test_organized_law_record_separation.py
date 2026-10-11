"""Proof controls and synthetic admission; retained moments are not replayed."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as Q

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I, sqrt
from tnfr.physics import _organized_law_record_separation as separation
from tnfr.physics._organized_computable_prediction import _forecast_stages
from tnfr.physics._organized_law_contrast_kernels import _MomentKernel
from tnfr.physics._organized_law_joint_bounds import _bound_organized_law_joint_phases


@pytest.fixture(scope="module")
def late():
    return separation._bound_late_history(_bound_organized_law_joint_phases())


def _synthetic_moments():
    mu = (5 + sqrt(5)) / 8
    return tuple(
        _MomentKernel(2, harmonic, mu, I(harmonic), I(harmonic**2), Q(0), Q(0))
        for harmonic in (1, 2, 9, 18, 36)
    )


def test_late_blocks_use_all_recorded_ramps_and_keep_the_reference(late):
    h = Q(1, 100000)
    assert late.interval_start == 6 * (1 + h)
    assert late.reference_time == 9 * (1 + h)
    assert late.interval_end == 12 * (1 + h)
    assert late.maximum_reference_distance == 3 * (1 + h)
    stages = _forecast_stages()
    raw_times = {
        time - 200
        for stage in stages
        if stage.name == "plateau"
        for time in (stage.start, stage.start + stage.duration)
    }
    for child, block in zip((2, 3), late.ramp_endpoints):
        assert len(block) == 3
        for index, (start, end) in enumerate(block):
            assert start == 3 * child * (1 + h) + index * (1 + h) + h
            assert end - start == 1
            assert start in raw_times and end in raw_times
    assert late.interval_end < max(raw_times)


@pytest.mark.parametrize("child", (2, 3))
@pytest.mark.parametrize("plateau_index", (0, 1, 2))
def test_late_grid_rejects_a_changed_constant_plateau_level(
    monkeypatch, child, plateau_index
):
    cells = list(separation._historical_cells())
    selected = [
        index
        for index, cell in enumerate(cells)
        if cell.child == child and cell.name == "plateau"
    ]
    index = selected[plateau_index]
    cells[index] = replace(cells[index], base=Q(1, 5))
    monkeypatch.setattr(separation, "_historical_cells", lambda: tuple(cells))
    with pytest.raises(ArithmeticError, match="constant short plateaus"):
        separation._late_grid()


@pytest.mark.parametrize(
    "mu,duration", ((Q(0), Q(16)), (Q(1, 3), Q(7)), (Q(9, 10), Q(12)))
)
def test_decay_encloses_an_independent_high_precision_exponential(mu, duration):
    value = separation._decay(I(mu), duration)
    with localcontext() as context:
        context.prec = 80
        truth = (
            -Decimal(mu.numerator)
            / Decimal(mu.denominator)
            * Decimal(duration.numerator)
            / Decimal(duration.denominator)
        ).exp()
        lower = Decimal(value.lo.numerator) / Decimal(value.lo.denominator)
        upper = Decimal(value.hi.numerator) / Decimal(value.hi.denominator)
        assert lower <= truth <= upper


@pytest.mark.parametrize("bad", (True, False, float("nan"), float("inf"), "1", None))
def test_decay_rejects_invalid_physical_clocks_before_arithmetic(bad):
    with pytest.raises((TypeError, ValueError)):
        separation._decay(I(Q(1, 2)), bad)


@pytest.mark.parametrize(
    "mu,duration", ((I(-1), 1), (I(0, 2), 1), (I(1), -1), (I(1), 17))
)
def test_decay_declared_domain_is_retained(mu, duration):
    with pytest.raises(ValueError):
        separation._decay(mu, duration)


def test_c5_projector_and_dual_rows_independently_give_the_proof_norms():
    incidence = np.zeros((5, 5))
    for edge in range(5):
        incidence[edge, edge] = -1
        incidence[(edge + 1) % 5, edge] = 1
    generator = incidence @ incidence.T / 4
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    root = 5**0.5
    for mode, expected_row in ((1, 1 + 1 / root), (2, 1 + 3 / root)):
        mu = (5 + (-1 if mode == 1 else 1) * root) / 8
        vectors = eigenvectors[:, np.isclose(eigenvalues, mu)]
        projector = vectors @ vectors.T
        assert np.max(np.sum(abs(projector), axis=1)) == pytest.approx(
            (2 + 2 * root) / 5
        )
        assert np.max(np.sum(abs(incidence.T @ projector), axis=1)) == pytest.approx(
            expected_row
        )
        for child, selected in ((2, 1), (3, 0)):
            columns = projector @ incidence[:, [child - 1, child]]
            row = np.linalg.solve(columns.T @ columns, columns.T)[selected]
            assert row @ incidence[:, 2] == pytest.approx(1)
            assert row @ incidence[:, 1 if child == 2 else 3] == pytest.approx(0)
            if mode == 2:
                assert np.sum(abs(row @ incidence)) == pytest.approx(3 + root)


def test_late_bound_separates_heat_forcing_and_joint_difference_errors(late):
    assert late.admitted
    assert late.phase_motion_upper == late.heat_phase_upper + late.forced_phase_upper
    assert 0 < late.phase_motion_upper < late.phase_motion_cap == Q(1, 10**6)
    assert late.form_difference_radius_ratio < 44
    assert (
        late.relative_phase_current_radius_ratio
        == 22 * late.gamma_upper**2 * late.maximum_reference_distance
    )
    assert (
        late.current_remainder_radius_ratio
        == 38 * late.phase_motion_cap + late.relative_phase_current_radius_ratio
    )
    assert late.current_remainder_radius_ratio < Q(46, 10**6)


def test_centered_three_ramp_parities_and_combined_weights():
    # Synthetic scalar quadrature, unrelated to the retained kernels. Direct
    # symmetric integration tests command signs, parity and all three weights.
    mu, frequency, eps, b_s, b_c = 0.8, 3, 0.1, 0.7, -0.4
    nodes, weights = np.polynomial.legendre.leggauss(48)
    r = nodes / 2
    xi = eps * (1.5 * r - 2 * r**3)
    q = (eps / 2 + xi, -2 * xi, -eps / 2 + xi)
    residuals = tuple(
        tuple(
            float(
                np.dot(
                    weights / 2,
                    np.exp(mu * (r - 0.5))
                    * (
                        b_s * np.cos(frequency * sign * command)
                        + b_c * np.sin(frequency * sign * command)
                    ),
                )
            )
            for command in q
        )
        for sign in (-1, 1)
    )
    t = tuple(values[0] - values[2] for values in residuals)
    q_values = tuple(values[0] + values[2] - 2 * values[1] for values in residuals)
    e0 = -np.exp(mu / 2) * sum(t) / 8
    o0 = np.exp(mu / 2) * (t[1] - t[0]) / 8
    we = np.exp(mu / 2) * sum(q_values) / 8
    wo = np.exp(mu / 2) * (q_values[1] - q_values[0]) / 8
    positive_r, positive_weights = (nodes + 1) / 4, weights / 4
    positive_xi = eps * (1.5 * positive_r - 2 * positive_r**3)

    def c(n):
        return np.dot(
            positive_weights, np.sinh(mu * positive_r) * np.sin(n * positive_xi)
        )

    def d(n):
        return np.dot(
            positive_weights, np.cosh(mu * positive_r) * np.cos(n * positive_xi)
        )

    assert e0 == pytest.approx(
        b_s * np.sin(frequency * eps / 2) * c(frequency), abs=1e-15
    )
    assert o0 == pytest.approx(
        b_c * np.sin(frequency * eps / 2) * d(frequency), abs=1e-15
    )
    assert we == pytest.approx(
        b_s * (np.cos(frequency * eps / 2) * d(frequency) - d(2 * frequency)), abs=1e-15
    )
    assert wo == pytest.approx(
        b_c * (np.cos(frequency * eps / 2) * c(frequency) + c(2 * frequency)), abs=1e-15
    )
    for alpha in (Q(1, 2), Q(2)):
        expected_e = abs(1 + alpha) + 2 * abs(alpha) + abs(alpha - 1)
        expected_o = abs(1 - alpha) + 2 * abs(alpha) + abs(1 + alpha)
        assert expected_e == expected_o
        assert expected_e == (2 + 2 * alpha if alpha < 1 else 4 * alpha)


@pytest.mark.parametrize("replacement", (True, 2.0, Q(2), 1))
def test_moment_mode_is_re_admitted_before_any_coefficient_arithmetic(replacement):
    moments = _synthetic_moments()
    bad = (replace(moments[0], mode=replacement),) + moments[1:]
    with pytest.raises(ValueError):
        separation._admit_moments(bad, (5 + sqrt(5)) / 8)


@pytest.mark.parametrize("kind", ("order", "mu", "noninterval", "length", "mapping"))
def test_primitive_associations_and_intervals_are_re_admitted(kind):
    moments = _synthetic_moments()
    mu = (5 + sqrt(5)) / 8
    if kind == "order":
        moments = tuple(reversed(moments))
    elif kind == "mu":
        moments = (replace(moments[0], mu=I(0)),) + moments[1:]
    elif kind == "noninterval":
        moments = (replace(moments[0], sine_hyperbolic=float("nan")),) + moments[1:]
    elif kind == "length":
        moments = moments[:-1]
    else:
        moments = {item.harmonic: item for item in moments}
    with pytest.raises((TypeError, ValueError)):
        separation._admit_moments(moments, mu)


def test_unused_producer_diagnostics_do_not_replace_primitive_moments():
    moments = _synthetic_moments()
    changed = tuple(
        replace(item, sine_truncation=True, cosine_truncation=float("nan"))
        for item in moments
    )
    expected = separation._admit_moments(moments, moments[0].mu)
    assert separation._admit_moments(changed, moments[0].mu) == expected


def test_unresolved_fundamental_denominator_abstains_without_a_record_verdict():
    moments = tuple(
        replace(item, sine_hyperbolic=I(0), cosine_hyperbolic=I(0))
        for item in _synthetic_moments()
    )
    result = separation._certify_organized_law_record_separation(moments=moments)
    assert result.late_bounds.admitted
    assert result.contrast is None and result.circle is None
    assert not result.available and not result.separation_certified
    assert result.reason == "fundamental_elimination_denominator_contains_zero"
