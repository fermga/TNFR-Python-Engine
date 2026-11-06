"""Independent coefficient controls for the frozen all-zero-pressure bound.

The six carried gain pairs, phase coordinate bases and support sign vectors
are algebraic controls, not preparations. No source, controller resultant,
native trajectory or refinement of the recorded inconclusive bound is run.
"""

import builtins
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

import pytest

from tests.research import test_native_mutation_feasibility as prior
from tnfr.config.defaults_core import CoreDefaults
from tnfr.research import _native_mutation_feasibility as shared
from tnfr.research import _native_mutation_zero_pressure as zero

no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard

PAIRS = tuple(product(prior.BRANCHES, ("stable", "transition")))
BASIS = tuple(tuple(Q(i == j) for i in range(5)) for j in range(5))


def _multiply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _fine_laplacian(vector):
    fine = prior._paired(vector)
    return tuple(
        sum(fine[j] for j in prior._neighbors(2 * i)) / 4 - fine[2 * i]
        for i in range(5)
    )


@pytest.fixture(scope="module")
def obstruction(renewal_guard):
    return zero.derive_native_mutation_zero_obstruction()


def test_fresh_zero_masks_determine_capacity_order_and_the_correct_phase_unit(
    obstruction,
):
    law, cfg = prior._defaults(), CoreDefaults()
    alpha, beta, gamma = law["stability"]
    k, b, w = law["pressure"]
    d = w / prior.PI
    assert obstruction.law.pressure == law["pressure"]
    assert obstruction.law.stability == law["stability"]
    assert obstruction.premises.dt == prior.DT
    assert obstruction.capacity_floor == (law["hi"] - beta - gamma) / alpha
    assert obstruction.zero_ratio_intercept == (law["lo"] - beta - gamma) / alpha
    assert obstruction.zero_ratio_slope == beta / (alpha * prior.PI)
    assert obstruction.capacity_order_margin == (law["hi"] - law["lo"] - beta) / alpha
    assert obstruction.capacity_order_margin > 0
    assert obstruction.endpoint_arc_bound == 2 * k / d + law["shift"]
    assert obstruction.endpoint_arc_bound != 2 * k / w + law["shift"]
    # This diagonal bound uses the zero-ratio target mask, not the larger
    # unrestricted OZ diagonal retained by the independent reducer.
    assert law["f"] * (law["lo"] - gamma) / alpha <= law["ell"]
    expected_maximum = law["ell"] * Q(cfg.VF_MAX)
    assert obstruction.maximum_flow_diagonal == expected_maximum
    assert (
        obstruction.pressure_dominance_margin
        == (1 - 2 * prior.DT * k * expected_maximum)
        > 0
    )
    # Independent endpoint-potential sign identity on formal coordinate
    # bases: q=0 makes k*x+b*v+d*y constant. These are not admitted states.
    for capacities, phases in product(BASIS, BASIS):
        forms = tuple(-(b * v + d * y) / k for v, y in zip(capacities, phases))
        assert (
            _fine_laplacian(
                tuple(
                    k * x + b * v + d * y for x, v, y in zip(forms, capacities, phases)
                )
            )
            == prior.ZERO
        )
        for j in range(1, 5):
            assert k * (forms[0] - forms[j]) == (
                b * (capacities[j] - capacities[0]) + d * (phases[j] - phases[0])
            )


def test_phase_inverse_and_forcing_are_identified_by_ten_constituent_rows(obstruction):
    law = prior._defaults()
    assert (
        tuple((cell.first_branch, cell.second_branch) for cell in obstruction.cells)
        == PAIRS
    )
    for cell in obstruction.cells:
        first = prior._gains(law, law["gains"], cell.first_branch)
        second = prior._gains(law, first, cell.second_branch)
        assert cell.first_gains == first and cell.second_gains == second
        for vector in BASIS:
            fine_y = prior._phase_fine(prior._paired(vector), first)
            y = tuple(fine_y[2 * i] for i in range(5))
            assert _multiply(cell.phase_matrix, vector) == y
            assert _multiply(cell.phase_inverse, y) == vector
            inverse_column = _multiply(cell.phase_inverse, vector)
            fine_return = prior._phase_fine(prior._paired(inverse_column), first)
            assert tuple(fine_return[2 * i] for i in range(5)) == vector
            # H*C=L*(I-C): identify H without constructing another inverse
            # or copying its five-cycle coefficient formula.
            expected = _fine_laplacian(tuple(x - new for x, new in zip(vector, y)))
            assert _multiply(cell.forcing_matrix, y) == expected
        assert all(sum(row) == 0 for row in cell.forcing_matrix)
        assert all(
            cell.forcing_matrix[i][j] == cell.forcing_matrix[0][(j - i) % 5]
            for i in range(5)
            for j in range(5)
        )


def test_forcing_oscillation_norm_is_an_exact_support_bound(obstruction):
    for cell in obstruction.cells:
        bound = cell.forcing_oscillation_norm
        for row in cell.forcing_matrix:
            assert bound == sum(abs(value) for value in row) / 2 > 0
            positive = tuple(Q(value > 0) for value in row)
            negative = tuple(Q(value < 0) for value in row)
            assert sum(a * b for a, b in zip(row, positive)) == bound
            assert sum(a * b for a, b in zip(row, negative)) == -bound
            # Zero row sum removes every common origin. The two sign
            # choices attain the interval support extrema for width one.
            translated = tuple(Q(7, 4) + 2 * value for value in positive)
            assert sum(a * b for a, b in zip(row, translated)) == 2 * bound


def test_single_capacity_quadratic_has_its_declared_exact_global_maximum(obstruction):
    law, cfg = prior._defaults(), CoreDefaults()
    k, b, w = law["pressure"]
    d, maximum = w / prior.PI, Q(cfg.VF_MAX)
    r0, r1 = obstruction.zero_ratio_intercept, obstruction.zero_ratio_slope
    for cell in obstruction.cells:
        g2, local2 = cell.second_gains
        a2 = 1 - g2 - local2
        ceiling = min(a2 * law["shift"] / (g2 + local2), 2 * k / d)
        assert cell.gap_ceiling == ceiling
        assert cell.ratio_ceiling == r0 + r1 * ceiling < obstruction.capacity_floor
        constant, linear, quadratic = cell.capacity_quadratic
        assert (constant, linear, quadratic) == (
            r0 * 2 * k,
            r1 * 2 * k - r0 * d,
            -r1 * d,
        )
        assert quadratic < 0
        vertex = cell.capacity_vertex
        assert 0 <= vertex <= ceiling
        derivative = linear + 2 * quadratic * vertex
        assert (
            (vertex == 0 and derivative <= 0)
            or (vertex == ceiling and derivative >= 0)
            or derivative == 0
        )
        numerator = cell.capacity_numerator_max
        assert numerator == (r0 + r1 * vertex) * (2 * k - d * vertex)
        for z in (Q(0), ceiling, vertex):
            value = (r0 + r1 * z) * (2 * k - d * z)
            assert (
                numerator - value
                == (-derivative * (z - vertex) - quadratic * (z - vertex) ** 2)
                >= 0
            )
        assert (
            cell.capacity_ceiling
            == min(
                maximum * cell.ratio_ceiling,
                numerator / (b * (obstruction.capacity_floor - cell.ratio_ceiling)),
            )
            > 0
        )


def test_target_forcing_identity_and_frozen_rate_composition(obstruction):
    law, cfg = prior._defaults(), CoreDefaults()
    k, _, w = law["pressure"]
    d, maximum, tau = w / prior.PI, Q(cfg.VF_MAX), prior.DT * k
    for cell in obstruction.cells:
        global1, local1 = cell.first_gains
        a1 = 1 - global1 - local1
        determinant = a1 * (a1 - local1 / 2) - local1**2 / 4
        ch, ct = a1 / determinant - 1, local1 / (2 * determinant)
        assert determinant > 0 and min(ch, ct) >= 0
        for vector in BASIS:
            fine = prior._phase_fine(prior._paired(vector), cell.first_gains)
            y = tuple(fine[2 * i] for i in range(5))
            source_gap = (vector[1] + vector[4]) / 2 - vector[0]
            current_gap = (y[1] + y[4]) / 2 - y[0]
            near_far = (y[1] + y[4] - y[2] - y[3]) / 2
            assert source_gap - current_gap == ch * current_gap + ct * near_far
        global2, local2 = cell.second_gains
        a2 = 1 - global2 - local2
        assert cell.phase_diameter_ceiling == min(
            a2 * law["shift"] / (global2 + local2 / 2),
            (1 - global2) * obstruction.endpoint_arc_bound,
        )
        assert cell.pressure_peak_ceiling * obstruction.pressure_dominance_margin == (
            d * cell.forcing_oscillation_norm * cell.phase_diameter_ceiling
        )
        assert cell.target_forcing_ceiling == d * (
            ch * cell.gap_ceiling + ct * a2 * (law["shift"] + cell.gap_ceiling)
        )
        assert cell.target_denominator == 1 - tau * law["f"] * cell.capacity_ceiling > 0
        assert cell.rate_ceiling * cell.target_denominator == (
            law["f"]
            * cell.capacity_ceiling
            * (
                cell.target_forcing_ceiling
                + tau * law["ell"] * maximum * cell.pressure_peak_ceiling
            )
        )
        assert cell.threshold_gap == law["xi"] - cell.rate_ceiling < 0
        assert cell.status == "inconclusive"


def test_no_report_artifact_source_or_other_kernel_can_supply_the_coefficients(
    obstruction, renewal_guard
):
    _, forbidden = renewal_guard
    fake = replace(obstruction, pressure_dominance_margin=Q(1))
    for report in (obstruction, fake, {"law": obstruction.law}):
        with pytest.raises(TypeError):
            zero.derive_native_mutation_zero_obstruction(report)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(prior.guards.domain.gain, "_source", forbidden)
        patch.setattr(zero, "reduce_native_mutation_zero_pressure", forbidden)
        rebuilt = zero.derive_native_mutation_zero_obstruction()
    assert rebuilt == obstruction
    assert rebuilt.law is not obstruction.law
    assert rebuilt.premises is not obstruction.premises
    with pytest.raises(FrozenInstanceError):
        rebuilt.pressure_dominance_margin = Q(1)
    with pytest.raises(FrozenInstanceError):
        rebuilt.cells[0].status = "excluded"


@pytest.mark.parametrize("change", ("capacity_order", "pressure_dominance"))
def test_changed_owned_law_cannot_inherit_the_obstruction_domain(
    change, monkeypatch, renewal_guard
):
    cfg = CoreDefaults()
    if change == "capacity_order":
        cfg.SI_WEIGHTS["beta"] = 100
    else:
        cfg.GLYPH_FACTORS["OZ_dnfr_factor"] = 100
    monkeypatch.setattr(shared, "CoreDefaults", lambda: cfg)
    with pytest.raises(ValueError):
        zero.derive_native_mutation_zero_obstruction()
