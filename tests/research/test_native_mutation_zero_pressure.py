"""Independent controls for exact zero-successor-pressure elimination.

The frozen capacity frame, seven formal phase frames and six declared branch
pairs identify the reduction algebra. They are not admitted preparations or
native trajectories. Direct ten-constituent fields, strict scalar rows and
common offsets are checked without evaluating a selector or phase resultant.
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
PHASE_FRAMES = (
    ("zero", prior.ZERO),
    *((f"e{j}", tuple(Q(i == j) for i in range(5))) for j in range(5)),
    ("declared", prior.PHASES),
)


def _matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _product(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(5)) for j in range(5))
        for i in range(5)
    )


def _identity():
    return tuple(tuple(Q(i == j) for j in range(5)) for i in range(5))


def _fine_oracle(spec, pressure):
    """Reconstruct the original compiler gauge by integrating ring differences."""
    law = prior._defaults()
    k, b, w = law["pressure"]
    capacities = tuple(Q(value) for value in spec.capacities)
    theta = tuple(Q(value) - Q(spec.phases[0]) for value in spec.phases)
    potential = prior._potential(pressure)
    x0 = prior._paired(
        tuple(
            (potential[i] - b * capacities[i] - w * theta[i] / prior.PI) / k
            for i in range(5)
        )
    )
    fine_capacity, phase = prior._paired(capacities), prior._paired(theta)
    actual_pressure = prior._fine_field(law, x0, fine_capacity, phase)
    first_rate = tuple(
        capacity * (law["f"] if node < 2 else law["ell"]) * p
        for node, (capacity, p) in enumerate(zip(fine_capacity, actual_pressure))
    )
    x1 = tuple(value + prior.DT * rate for value, rate in zip(x0, first_rate))
    first_gains = prior._gains(law, law["gains"], spec.first_branch)
    first_phase = prior._phase_fine(phase, first_gains)
    q = prior._fine_field(law, x1, fine_capacity, first_phase)
    second_rate = tuple(
        capacity * (1 if node < 2 else law["ell"]) * p
        for node, (capacity, p) in enumerate(zip(fine_capacity, q))
    )
    x2 = tuple(value + prior.DT * rate for value, rate in zip(x1, second_rate))
    mutated = tuple(
        value - (law["shift"] if node < 2 else 0)
        for node, value in enumerate(first_phase)
    )
    second_gains = prior._gains(law, first_gains, spec.second_branch)
    terminal = prior._phase_fine(mutated, second_gains)
    return {
        "forms": (x0, x1, x2),
        "pressure": (actual_pressure, q),
        "phase": (phase, first_phase, mutated, terminal),
        "capacity": fine_capacity,
        "rates": (first_rate, second_rate),
    }


def _independent_transport(spec):
    """Five columns from the ten-node held-rate row, without an inverse."""
    law = prior._defaults()
    k, _, w = law["pressure"]
    capacities = prior._paired(spec.capacities)
    columns = []
    for block in range(5):
        p = tuple(Q(node // 2 == block) for node in range(10))
        rate = tuple(
            v * (law["f"] if node < 2 else law["ell"]) * value
            for node, (v, value) in enumerate(zip(capacities, p))
        )
        columns.append(
            tuple(
                p[2 * i]
                + prior.DT
                * k
                * (sum(rate[j] for j in prior._neighbors(2 * i)) / 4 - rate[2 * i])
                for i in range(5)
            )
        )
    matrix = tuple(tuple(columns[j][i] for j in range(5)) for i in range(5))
    theta = prior._paired(tuple(value - spec.phases[0] for value in spec.phases))
    gains = prior._gains(law, law["gains"], spec.first_branch)
    first = prior._phase_fine(theta, gains)
    change = tuple(new - old for old, new in zip(theta, first))
    forcing = tuple(
        w
        * (sum(change[j] for j in prior._neighbors(2 * i)) / 4 - change[2 * i])
        / prior.PI
        for i in range(5)
    )
    return matrix, forcing


@pytest.fixture(scope="module")
def reductions(renewal_guard):
    # Exactly the seven formal frames and six carried branch pairs frozen
    # in the protocol; no source family or trigonometric predicate is sampled.
    return {
        (first, second, name): zero.reduce_native_mutation_zero_pressure(
            prior._spec(first, second, phases=phases)
        )
        for first, second in PAIRS
        for name, phases in PHASE_FRAMES
    }


def test_uniform_owned_bound_proves_column_dominance_without_inverse_positivity(
    reductions,
):
    cfg, law = CoreDefaults(), prior._defaults()
    certificate = zero.derive_native_mutation_zero_invertibility()
    maximum = Q(cfg.VF_MAX) * max(law["f"], law["ell"])
    step = prior.DT * law["pressure"][0] * maximum
    assert certificate.maximum_flow_diagonal == maximum
    assert certificate.maximum_pressure_step == step
    assert certificate.column_dominance_margin == 1 - 2 * step > 0
    assert certificate.dimension == 5
    assert "not inverse positivity" in certificate.scope
    assert certificate.law.pressure == law["pressure"]
    for result in reductions.values():
        matrix = result.transport_matrix
        assert result.invertibility == certificate
        assert all(value >= 0 for row in matrix for value in row)
        for j in range(5):
            assert sum(matrix[i][j] for i in range(5)) == 1
            dominance = matrix[j][j] - sum(
                abs(matrix[i][j]) for i in range(5) if i != j
            )
            assert dominance >= certificate.column_dominance_margin > 0
        assert _product(matrix, result.transport_inverse) == _identity()
        assert _product(result.transport_inverse, matrix) == _identity()
        # Nonnegative transport is compatible with a signed inverse; never
        # transfer Markov positivity to the solved source pressure vector.
        assert any(value < 0 for row in result.transport_inverse for value in row)


def test_unique_pressure_solves_the_independent_complete_transport(reductions):
    for result in reductions.values():
        matrix, forcing = _independent_transport(result.spec)
        assert result.transport_matrix == matrix
        assert result.phase_forcing == forcing
        assert sum(forcing) == sum(result.source_pressure) == 0
        residual = tuple(
            a + b for a, b in zip(_matvec(matrix, result.source_pressure), forcing)
        )
        assert residual == result.successor_pressure == prior.ZERO
    for first, second in PAIRS:
        null = reductions[first, second, "zero"]
        assert null.source_pressure == prior.ZERO
        basis_pressures = tuple(
            reductions[first, second, f"e{i}"].source_pressure for i in range(5)
        )
        assert (
            tuple(sum(values[j] for values in basis_pressures) for j in range(5))
            == prior.ZERO
        )
        declared = tuple(
            sum(prior.PHASES[i] * basis_pressures[i][j] for i in range(5))
            for j in range(5)
        )
        assert reductions[first, second, "declared"].source_pressure == declared
    # The second selected gain does not change the already prepared q.
    for first, (name, _) in product(prior.BRANCHES, PHASE_FRAMES):
        assert (
            reductions[first, "stable", name].source_pressure
            == reductions[first, "transition", name].source_pressure
        )


def test_direct_fine_fields_rates_and_endpoint_potential_match(reductions):
    law = prior._defaults()
    k, b, w = law["pressure"]
    for result in reductions.values():
        fine = _fine_oracle(result.spec, result.source_pressure)
        assert fine["pressure"][0] == prior._paired(result.source_pressure)
        assert fine["pressure"][1] == (Q(0),) * 10
        assert result.form_offsets == tuple(values[::2] for values in fine["forms"])
        assert result.first_rates == fine["rates"][0][::2]
        assert result.second_rates == fine["rates"][1][::2] == prior.ZERO
        assert result.form_offsets[2] == result.form_offsets[1]
        assert result.first_phase == fine["phase"][1][::2]
        assert result.mutation_phase == fine["phase"][2][::2]
        assert result.terminal_phase == fine["phase"][3][::2]
        potential = tuple(
            k * x + b * v + w * phase / prior.PI
            for x, v, phase in zip(fine["forms"][1], fine["capacity"], fine["phase"][1])
        )
        assert len(set(potential)) == 1
        terminal_pressure = prior._fine_field(
            law, fine["forms"][2], fine["capacity"], fine["phase"][3]
        )
        phase_change = tuple(
            new - old for old, new in zip(fine["phase"][1], fine["phase"][3])
        )
        fresh_difference = tuple(
            w
            * (
                sum(phase_change[j] for j in prior._neighbors(node)) / 4
                - phase_change[node]
            )
            / prior.PI
            for node in range(10)
        )
        assert terminal_pressure == fresh_difference
    # A formal coefficient countercontrol, not an admitted Mutation: zero
    # stored second pressure does not freeze phase or imply fresh equilibrium.
    null = reductions["stable", "stable", "zero"]
    fine = _fine_oracle(null.spec, null.source_pressure)
    assert any(
        prior._fine_field(law, fine["forms"][2], fine["capacity"], fine["phase"][3])
    )


def test_true_shared_offset_interval_retains_all_thirty_original_rails(reductions):
    for result in reductions.values():
        offsets = (*result.form_offsets[0], *result.form_offsets[1])
        expected = (-1 - min(offsets), 1 - max(offsets))
        assert result.offset_interval == expected
        assert (
            result.offset_check.lhs,
            result.offset_check.rhs,
            result.offset_check.relation,
        ) == (*expected, "<=")
        assert result.offset_check.satisfied == (max(offsets) - min(offsets) <= 2)
        for cell in result.cells:
            rails = [row for row in cell.rows if row.label.startswith("rail.")]
            assert len(rails) == 30
            assert all(not row.strict and row.unit == "form" for row in rails)
            assert (
                max(
                    row.rhs / row.offset_coefficient
                    for row in rails
                    if row.offset_coefficient < 0
                )
                == expected[0]
            )
            assert (
                min(
                    row.rhs / row.offset_coefficient
                    for row in rails
                    if row.offset_coefficient > 0
                )
                == expected[1]
            )
            if result.offset_check.satisfied:
                for endpoint in expected:
                    assert all(
                        row.offset_coefficient * endpoint <= row.rhs for row in rails
                    )


def test_exact_touch_and_arbitrarily_small_gap_use_the_real_interval_helper():
    rails = (Q(-1), Q(1))
    offsets = (Q(-1), Q(1), *(Q(0),) * 8)
    touching = zero._common_offset_interval(offsets, rails)
    assert touching == (Q(0), Q(0))
    assert shared.ParameterCheck("formal_touch", *touching, "<=").satisfied
    tiny = Q(1, 2**120)
    separated = zero._common_offset_interval(
        (offsets[0], offsets[1] + tiny, *offsets[2:]), rails
    )
    assert separated == (Q(0), -tiny)
    assert not shared.ParameterCheck("formal_gap", *separated, "<=").satisfied
    translated = zero._common_offset_interval(
        tuple(value + Q(7, 4) for value in offsets), rails
    )
    assert translated == (Q(-7, 4), Q(-7, 4))


def test_all_nine_substituted_cells_match_independent_signed_masks(reductions):
    law = prior._defaults()
    gamma = law["stability"][2]
    for result in reductions.values():
        fine = _fine_oracle(result.spec, result.source_pressure)
        assert tuple(cell.key for cell in result.cells) == tuple(
            shared.CellKey(owner, None) for owner in prior.SOURCE_OWNERS
        )
        p = result.source_pressure
        peak = max(map(abs, p))
        ratio = tuple(abs(value) / peak if peak else Q(0) for value in p)
        source_base, next_base = (
            prior._si_base(law, fine["capacity"], phase)[::2]
            for phase in fine["phase"][:2]
        )
        source_si = tuple(
            base - gamma * relative for base, relative in zip(source_base, ratio)
        )
        direct_masks = (
            peak > 0
            and p[0] > Q(1e-9)
            and ratio[0] > law["cut"]
            and result.spec.capacities[0] * law["f"] * p[0] > law["xi"]
            and source_si[0] <= law["lo"]
            and all(value >= law["hi"] for value in source_si[1:])
            and next_base[0] <= law["lo"]
            and all(value >= law["hi"] for value in next_base[1:])
        )
        for cell in result.cells:
            expected = prior._expected_residuals(law, fine, cell.key)
            assert len(cell.rows) == len(expected) == 68
            assert {row.label for row in cell.rows} == set(expected)
            for row in cell.rows:
                residual, strict, unit = expected[row.label]
                coefficient = (
                    (1 if row.label.endswith("upper") else -1)
                    if row.label.startswith("rail.")
                    else 0
                )
                assert (row.offset_coefficient, row.rhs, row.strict, row.unit) == (
                    coefficient,
                    -residual,
                    strict,
                    unit,
                )
            assert {check.label for check in cell.parameter_checks} == {
                row.label for row in cell.rows if row.offset_coefficient == 0
            }
            for check in cell.parameter_checks:
                residual, strict, _ = expected[check.label]
                assert (check.lhs, check.rhs, check.relation) == (
                    0,
                    -residual,
                    "<" if strict else "<=",
                )
            block, sign = cell.key.source_peak
            compatible = (
                direct_masks
                and sign * p[block] == peak
                and result.offset_check.satisfied
            )
            assert (cell.status == "algebraically_compatible") == compatible
            assert "negative signed-zero" in cell.zero_pressure_direction
        assert (result.algebraic_status == "algebraically_compatible") == any(
            cell.status == "algebraically_compatible" for cell in result.cells
        )


def test_unavailable_direction_and_controller_geometry_are_not_discharged(reductions):
    for result in reductions.values():
        assert all(
            value.status == "unevaluated" for value in result.controller_obligations
        )
        assert "negative binary64 sign bit" in " ".join(result.obligations)
        assert "not acquired native observations" in " ".join(result.obligations)
        assert result.premises.traces[2][0] == ("AL", "IL", "EN", "OZ", "ZHIR")
    null = reductions["stable", "stable", "zero"]
    outward = next(
        check for check in null.phase_geometry if check.label == "negative_outward_gap"
    )
    assert outward.lhs == outward.rhs == 0
    assert outward.relation == "<" and not outward.satisfied
    assert all(cell.status == "algebraically_excluded" for cell in null.cells)
    assert all(
        not next(
            check
            for check in cell.parameter_checks
            if check.label == "source.peak.positive"
        ).satisfied
        for cell in null.cells
    )


def test_common_phase_translation_and_represented_inputs_normalize_first(reductions):
    original = reductions["stable", "stable", "declared"]
    translated = zero.reduce_native_mutation_zero_pressure(
        prior._spec(
            capacities=list(map(float, prior.CAPACITIES)),
            phases=[float(value + Q(7, 4)) for value in prior.PHASES],
        )
    )
    assert translated == original
    assert translated.law is not original.law


@pytest.mark.parametrize(
    "change",
    [
        {"capacities": (True, 2, 3, 4, 5)},
        {"capacities": (0, 2, 3, 4, 5)},
        {"capacities": {1, 2, 3, 4, 5}},
        {"capacities": (float("nan"), 2, 3, 4, 5)},
        {"phases": (0, 0, 0, 0)},
        {"phases": (0, 0, float("inf"), 0, 0)},
        {"first_branch": True},
        {"second_branch": "assumed"},
    ],
)
def test_raw_invalid_primitives_do_not_reach_elimination(
    change, monkeypatch, renewal_guard
):
    _, forbidden = renewal_guard
    monkeypatch.setattr(zero, "exact_matrix_inverse", forbidden)
    with pytest.raises((TypeError, ValueError)):
        zero.reduce_native_mutation_zero_pressure(replace(prior._spec(), **change))


@pytest.mark.parametrize("drift", ("strict_rail", "offset_dependent_mask"))
def test_canonical_row_drift_cannot_silently_change_offset_reduction(
    drift, monkeypatch, renewal_guard
):
    original_cell = zero._cell

    def changed_cell(prepared, key):
        cell = original_cell(prepared, key)
        rows = list(cell.rows)
        index = next(
            i
            for i, row in enumerate(rows)
            if row.label.startswith("rail.") == (drift == "strict_rail")
        )
        row = rows[index]
        rows[index] = (
            replace(row, strict=True)
            if drift == "strict_rail"
            else replace(row, coefficients=(*row.coefficients[:4], Q(1)))
        )
        return replace(cell, rows=tuple(rows))

    monkeypatch.setattr(zero, "_cell", changed_cell)
    with pytest.raises(ValueError, match="supported weak-rail structure"):
        zero.reduce_native_mutation_zero_pressure(prior._spec())


def test_changed_owned_law_cannot_bypass_universal_invertibility(
    monkeypatch, renewal_guard
):
    cfg = CoreDefaults()
    cfg.GLYPH_FACTORS["OZ_dnfr_factor"] = 100
    monkeypatch.setattr(shared, "CoreDefaults", lambda: cfg)
    with pytest.raises(ValueError, match="strict pressure-invertibility"):
        zero.derive_native_mutation_zero_invertibility()
    with pytest.raises(ValueError, match="strict pressure-invertibility"):
        zero.reduce_native_mutation_zero_pressure(prior._spec())


def test_reports_artifacts_and_native_producers_cannot_supply_premises(
    reductions, renewal_guard
):
    original = reductions["stable", "stable", "declared"]
    fake = replace(original, source_pressure=(Q(0),) * 5, algebraic_status="admitted")
    for value in (original, fake, {"spec": original.spec}):
        with pytest.raises(TypeError):
            zero.reduce_native_mutation_zero_pressure(value)
    with pytest.raises(TypeError):
        zero.derive_native_mutation_zero_invertibility(original.invertibility)
    _, forbidden = renewal_guard
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(prior.guards.domain.gain, "_source", forbidden)
        rebuilt = zero.reduce_native_mutation_zero_pressure(fake.spec)
    assert rebuilt == original and rebuilt.law is not original.law
    with pytest.raises(FrozenInstanceError):
        rebuilt.algebraic_status = "admitted"
    with pytest.raises(FrozenInstanceError):
        rebuilt.cells[0].rows[0].rhs = Q(0)
