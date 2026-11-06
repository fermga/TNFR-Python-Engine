"""Independent algebra controls for the conditional Mutation slice compiler.

The frozen frames and coordinate bases identify affine coefficients, not
admitted physical sources. No system is solved or tested for joint feasibility.
Nine declared gain pairs, all 81 peak cells and exact contradiction witnesses
are checked under guards forbidding native execution and old response fixtures.
"""

import builtins
import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path

import pytest

from tests.physics import test_native_mutation_renewal as guards
from tnfr.config.defaults_core import CoreDefaults
from tnfr.research import _native_mutation_feasibility as compiler

no_execution = guards.no_execution
calculus_guard = guards.calculus_guard
renewal_guard = guards.renewal_guard

DT, PI = Q(1, 16), Q(math.pi)
CAPACITIES = (Q(1, 2), Q(2), Q(3), Q(4), Q(5))
PHASES = (Q(0), Q(1, 8), Q(1, 4), Q(3, 8), Q(1, 2))
BRANCHES = ("stable", "transition", "dissonant")
SOURCE_OWNERS = ((0, 1),) + tuple(product(range(1, 5), (-1, 1)))
NEXT_OWNERS = tuple(product(range(1, 5), (-1, 1)))
ZERO = (Q(0),) * 5
BASIS = (ZERO,) + tuple(tuple(Q(i == j) for i in range(5)) for j in range(5))


def _spec(first="stable", second="stable", *, capacities=CAPACITIES, phases=PHASES):
    return compiler.NativeMutationSlice(capacities, phases, first, second)


def _defaults():
    """Read scalar owners directly; no compiled law or old research report."""
    cfg = CoreDefaults()
    pressure = {key: Q(value) for key, value in cfg.DNFR_WEIGHTS.items()}
    stability = {key: Q(value) for key, value in cfg.SI_WEIGHTS.items()}
    return {
        "pressure": tuple(
            pressure[key] / sum(pressure.values()) for key in ("epi", "vf", "phase")
        ),
        "stability": tuple(
            stability[key] / sum(stability.values())
            for key in ("alpha", "beta", "gamma")
        ),
        "f": Q(cfg.GLYPH_FACTORS["OZ_dnfr_factor"]),
        "ell": Q(cfg.GLYPH_FACTORS["IL_dnfr_factor"]),
        "shift": Q(cfg.GLYPH_FACTORS["ZHIR_theta_shift_factor"]) * PI / 4,
        "cut": Q(cfg.SELECTOR_THRESHOLDS["dnfr_hi"]),
        "lo": Q(cfg.SELECTOR_THRESHOLDS["si_lo"]),
        "hi": Q(cfg.SELECTOR_THRESHOLDS["si_hi"]),
        "xi": Q(cfg.ZHIR_THRESHOLD_XI),
        "gains": (Q(cfg.PHASE_K_GLOBAL), Q(cfg.PHASE_K_LOCAL)),
        "policy": {
            key: Q(value) for key, value in cfg.PHASE_ADAPT.items() if key != "enabled"
        },
    }


def _neighbors(node):
    block = node // 2
    return tuple(
        2 * other + parity
        for other in ((block - 1) % 5, (block + 1) % 5)
        for parity in (0, 1)
    )


def _paired(values):
    return tuple(value for value in values for _ in range(2))


def _fine_field(law, forms, capacities, phases):
    k, b, w = law["pressure"]
    return tuple(
        k * (sum(forms[j] for j in _neighbors(n)) / 4 - forms[n])
        + b * (sum(capacities[j] for j in _neighbors(n)) / 4 - capacities[n])
        + w * (sum(phases[j] for j in _neighbors(n)) / 4 - phases[n]) / PI
        for n in range(10)
    )


def _potential(pressure):
    """Integrate ring differences, independent of the compiler's C5 inverse."""
    first_edge = -Q(2, 5) * sum((5 - j) * pressure[j] for j in range(1, 5))
    values = tuple(
        i * first_edge + 2 * sum((i - j) * pressure[j] for j in range(1, i))
        for i in range(5)
    )
    average = sum(values) / 5
    result = tuple(value - average for value in values)
    assert (
        tuple(
            (result[(i - 1) % 5] + result[(i + 1) % 5]) / 2 - result[i]
            for i in range(5)
        )
        == pressure
    )
    return result


def _gains(law, old, branch):
    policy = law["policy"]
    choices = {
        "stable": (policy["kG_min"], policy["kL_min"]),
        "transition": (
            (policy["kG_min"] + policy["kG_max"]) / 2,
            (policy["kL_min"] + policy["kL_max"]) / 2,
        ),
        "dissonant": (policy["kG_max"], (policy["kL_min"] + policy["kL_max"]) / 2),
    }
    return tuple(
        current + policy["up" if goal > current else "down"] * (goal - current)
        for current, goal in zip(old, choices[branch])
    )


def _phase_fine(phases, gains):
    global_gain, local_gain = gains
    # Only the common global angle is omitted. Its gain still acts on every
    # relative coordinate and its branch obligation remains unevaluated.
    return tuple(
        (1 - global_gain - local_gain) * value
        + local_gain * sum(phases[j] for j in _neighbors(n)) / 4
        for n, value in enumerate(phases)
    )


def _fine_coefficients(spec, coordinates, law):
    """Formal basis substitution, not a selector or trajectory evaluation."""
    pressure = (*coordinates[:4], -sum(coordinates[:4]))
    potential = _potential(pressure)
    k, b, w = law["pressure"]
    theta = tuple(Q(value) - Q(spec.phases[0]) for value in spec.phases)
    forms = _paired(
        tuple(
            (potential[i] - b * spec.capacities[i] - w * theta[i] / PI) / k
            + coordinates[4]
            for i in range(5)
        )
    )
    capacity, phase = _paired(spec.capacities), _paired(theta)
    fine_pressure = _fine_field(law, forms, capacity, phase)
    assert fine_pressure == _paired(pressure)
    next_forms = tuple(
        x + DT * v * (law["f"] if n < 2 else law["ell"]) * p
        for n, (x, v, p) in enumerate(zip(forms, capacity, fine_pressure))
    )
    first_gains = _gains(law, law["gains"], spec.first_branch)
    first_phase = _phase_fine(phase, first_gains)
    following_pressure = _fine_field(law, next_forms, capacity, first_phase)
    final_forms = tuple(
        x + DT * v * (1 if n < 2 else law["ell"]) * p
        for n, (x, v, p) in enumerate(zip(next_forms, capacity, following_pressure))
    )
    mutation_phase = tuple(
        value - (law["shift"] if n < 2 else 0) for n, value in enumerate(first_phase)
    )
    second_gains = _gains(law, first_gains, spec.second_branch)
    terminal_phase = _phase_fine(mutation_phase, second_gains)
    return {
        "forms": (forms, next_forms, final_forms),
        "pressure": (fine_pressure, following_pressure),
        "phase": (phase, first_phase, mutation_phase, terminal_phase),
        "gains": (first_gains, second_gains),
        "capacity": capacity,
    }


def _si_base(law, capacity, phase):
    alpha, beta, gamma = law["stability"]
    return tuple(
        alpha * v / max(capacity)
        + beta * (1 - abs(sum(phase[j] for j in _neighbors(n)) / 4 - phase[n]) / PI)
        + gamma
        for n, v in enumerate(capacity)
    )


def _expected_residuals(law, fine, key):
    """Independent constituent formulas, before affine coefficient extraction."""
    rows = {}

    def add(label, value, strict=False, unit="pressure"):
        rows[label] = (value, strict, unit)

    for t, values in enumerate(fine["forms"]):
        for i in range(5):
            add(f"rail.t{t}.b{i}.upper", values[2 * i] - 1, unit="form")
            add(f"rail.t{t}.b{i}.lower", -values[2 * i] - 1, unit="form")

    p, q = (values[::2] for values in fine["pressure"])
    s0, s1 = (
        _si_base(law, fine["capacity"], phase)[::2] for phase in fine["phase"][:2]
    )
    gamma = law["stability"][2]

    def peak(prefix, values, owner):
        result = owner[1] * values[owner[0]]
        add(f"{prefix}.peak.positive", -result, True)
        for i, value in enumerate(values):
            add(f"{prefix}.peak.b{i}.upper", value - result)
            add(f"{prefix}.peak.b{i}.lower", -value - result)
        return result

    def il(prefix, values, base, maximum):
        for i in range(1, 5):
            add(
                f"{prefix}.il.b{i}.positive",
                gamma * values[i] + (law["hi"] - base[i]) * maximum,
            )
            add(
                f"{prefix}.il.b{i}.negative",
                -gamma * values[i] + (law["hi"] - base[i]) * maximum,
            )

    peak0 = peak("source", p, key.source_peak)
    add("source.target.si_low", (s0[0] - law["lo"]) * peak0 - gamma * p[0])
    add("source.target.oz_ratio", law["cut"] * peak0 - p[0], True)
    add("source.target.oz_tiny", Q(1e-9) - p[0], True)
    add(
        "source.target.acquired_rate",
        law["xi"] - fine["capacity"][0] * law["f"] * p[0],
        True,
        "form/time",
    )
    il("source", p, s0, peak0)
    if key.successor_peak is None:
        for i, value in enumerate(q):
            add(f"successor.zero.b{i}.upper", value)
            add(f"successor.zero.b{i}.lower", -value)
        add("successor.zero.target.si_low", s1[0] - law["lo"], unit="dimensionless")
        for i in range(1, 5):
            add(f"successor.zero.il.b{i}", law["hi"] - s1[i], unit="dimensionless")
    else:
        peak1 = peak("successor", q, key.successor_peak)
        add("successor.target.si_low", (s1[0] - law["lo"]) * peak1 + gamma * q[0])
        add("successor.target.nonpositive", q[0])
        add("successor.target.zhir_ratio", -q[0] - law["cut"] * peak1)
        il("successor", q, s1, peak1)
    return rows


def _value(expression, coordinates):
    return expression.constant + sum(
        a * x for a, x in zip(expression.coefficients, coordinates)
    )


@pytest.fixture(scope="module")
def compiled(renewal_guard):
    # This entire finite coefficient frame was frozen before compilation.
    return {
        (first, second): compiler.compile_native_mutation_slice(_spec(first, second))
        for first, second in product(BRANCHES, repeat=2)
    }


@pytest.fixture(scope="module")
def uniform(renewal_guard):
    return compiler.compile_native_mutation_slice(
        _spec(capacities=(Q(1),) * 5, phases=ZERO)
    )


def test_all_cells_and_units_are_explicit_without_claiming_feasibility(compiled):
    keys = {
        compiler.CellKey(source, following)
        for source in SOURCE_OWNERS
        for following in (*NEXT_OWNERS, None)
    }
    for result in compiled.values():
        assert len(result.cells) == 81
        assert {cell.key for cell in result.cells} == keys
        assert sum(cell.key.successor_peak is None for cell in result.cells) == 9
        assert result.status in ("conditional", "parameter_incompatible")
        assert all(
            obligation.status == "unevaluated"
            for obligation in result.controller_obligations
        )
        for cell in result.cells:
            labels = {row.label for row in cell.rows}
            assert (
                len(labels)
                == len(cell.rows)
                == (68 if cell.key.successor_peak is None else 75)
            )
            assert sum(row.unit == "form" for row in cell.rows) == 30
            assert sum(row.strict for row in cell.rows) == (
                4 if cell.key.successor_peak is None else 5
            )
            assert sum(row.unit == "form/time" for row in cell.rows) == 1
            assert "negative" in cell.zero_pressure_direction
            assert "zero" in cell.zero_pressure_direction


def test_all_affine_rows_equal_independent_ten_node_fields(compiled):
    law = _defaults()
    for result in compiled.values():
        assert result.law.pressure == law["pressure"]
        assert result.law.stability == law["stability"]
        for coordinates in BASIS:
            fine = _fine_coefficients(result.spec, coordinates, law)
            for t, fields in enumerate(result.form_boundaries):
                assert (
                    tuple(_value(field, coordinates) for field in fields)
                    == fine["forms"][t][::2]
                )
            assert (
                tuple(_value(field, coordinates) for field in result.pressure)
                == fine["pressure"][0][::2]
            )
            assert (
                tuple(_value(field, coordinates) for field in result.successor_pressure)
                == fine["pressure"][1][::2]
            )
            assert (
                sum(_value(field, coordinates) for field in result.successor_pressure)
                == 0
            )
            for cell in result.cells:
                expected = _expected_residuals(law, fine, cell.key)
                assert set(expected) == {row.label for row in cell.rows}
                for row in cell.rows:
                    residual = (
                        sum(a * x for a, x in zip(row.coefficients, coordinates))
                        - row.rhs
                    )
                    assert (residual, row.strict, row.unit) == expected[row.label], (
                        result.spec.first_branch,
                        result.spec.second_branch,
                        cell.key,
                        row.label,
                    )


def test_phase_branch_obligations_and_associated_history_are_retained(compiled):
    law = _defaults()
    for result in compiled.values():
        fine = _fine_coefficients(result.spec, ZERO, law)
        assert result.first_phase == fine["phase"][1][::2]
        assert result.mutation_phase == fine["phase"][2][::2]
        assert result.terminal_phase == fine["phase"][3][::2]
        source, first, mutated, terminal = (values[::2] for values in fine["phase"])
        expected_geometry = {
            "source_lift_width": (max(source) - min(source), Q(3), "<="),
            "first_common_arc": (max(first) - min(first), PI, "<"),
            "mutation_common_arc": (max(mutated) - min(mutated), PI, "<"),
            "negative_outward_gap": (first[0] - (first[1] + first[4]) / 2, Q(0), "<"),
            "complete_renewal": (
                max(first) - min(first),
                max(terminal) - min(terminal),
                "<",
            ),
            "target_not_capacity_maximum": (CAPACITIES[0], max(CAPACITIES), "<"),
            **{
                f"terminal_target_minimum_{i}": (terminal[0], terminal[i], "<=")
                for i in range(1, 5)
            },
        }
        assert {
            check.label: (check.lhs, check.rhs, check.relation)
            for check in result.phase_geometry
        } == expected_geometry
        for step, obligation in enumerate(result.controller_obligations):
            branch = (result.spec.first_branch, result.spec.second_branch)[step]
            phases = fine["phase"][2 * step][::2]
            assert obligation.phase_differences == tuple(
                phases[i] - phases[j] for i, j in combinations(range(5), 2)
            )
            assert (obligation.constant, obligation.cosine_coefficient) == (
                Q(1, 5),
                Q(2, 25),
            )
            assert obligation.disruptor_load == (Q(1, 20), Q(2, 25))[step]
            assert obligation.outgoing_gains == fine["gains"][step]
            assert obligation.incoming_gains == (
                law["gains"] if step == 0 else fine["gains"][0]
            )
            policy = law["policy"]
            if branch == "stable":
                assert obligation.lower_squared == policy["R_hi"] ** 2
                assert obligation.upper_squared is None
            elif branch == "dissonant":
                assert obligation.lower_squared is None
                assert obligation.upper_squared == policy["R_lo"] ** 2
            else:
                assert obligation.lower_squared == policy["R_lo"] ** 2
                assert obligation.upper_squared == policy["R_hi"] ** 2
            assert (
                obligation.lower_strict
                == obligation.upper_strict
                == (branch == "transition")
            )
        assert result.premises.dt == DT
        assert result.premises.times == (Q(1), Q(17, 16), Q(9, 8))
        assert result.premises.support == tuple(_neighbors(n) for n in range(10))
        assert result.premises.traces[0] == (("AL", "IL", "EN"),) * 5
        assert result.premises.traces[2][0] == ("AL", "IL", "EN", "OZ", "ZHIR")
        assert result.premises.traces[2][1:] == (("AL", "IL", "EN", "IL", "IL"),) * 4
        assert result.premises.node_ordinals == (3, 4, 5)
        assert result.premises.runtime_ordinals == result.premises.lags == (0, 1, 2)
        assert result.premises.possible_counts == ((0,), (0, 1), (0, 1, 2))
        for block, history in enumerate(result.temporal_histories):
            assert history == tuple(
                (time, result.form_boundaries[t][block])
                for t, time in enumerate(result.premises.times)
            )
        assert result.status == (
            "conditional"
            if all(check.satisfied for check in result.phase_geometry)
            else "parameter_incompatible"
        )


def test_origin_normalization_and_reflection_preserve_the_same_coefficients(compiled):
    original = compiled["stable", "stable"]
    translated = compiler.compile_native_mutation_slice(
        _spec(
            capacities=list(map(float, CAPACITIES)),
            phases=[float(value + Q(7, 4)) for value in PHASES],
        )
    )
    assert translated == original
    order = (0, 4, 3, 2, 1)
    reflected = compiler.compile_native_mutation_slice(
        _spec(
            capacities=tuple(CAPACITIES[i] for i in order),
            phases=tuple(PHASES[i] for i in order),
        )
    )
    for coordinates in BASIS:
        p = (*coordinates[:4], -sum(coordinates[:4]))
        reflected_coordinates = (*tuple(p[i] for i in order)[:4], coordinates[4])
        for left, right in zip(original.form_boundaries, reflected.form_boundaries):
            assert tuple(
                _value(field, reflected_coordinates) for field in right
            ) == tuple(_value(left[i], coordinates) for i in order)
        assert tuple(
            _value(field, reflected_coordinates)
            for field in reflected.successor_pressure
        ) == tuple(_value(original.successor_pressure[i], coordinates) for i in order)
    assert reflected.terminal_phase == tuple(original.terminal_phase[i] for i in order)
    assert (
        reflected.controller_obligations[0].disruptor_load
        == original.controller_obligations[0].disruptor_load
    )


def test_common_offset_rails_have_exact_pairwise_elimination(compiled):
    result = compiled["stable", "stable"]
    rails = {row.label: row for row in result.cells[0].rows}
    names = [(t, i) for t in range(3) for i in range(5)]
    for (t, i), (s, j) in product(names, repeat=2):
        upper, lower = rails[f"rail.t{t}.b{i}.upper"], rails[f"rail.t{s}.b{j}.lower"]
        coefficients = tuple(
            a + b for a, b in zip(upper.coefficients, lower.coefficients)
        )
        first, second = result.form_boundaries[t][i], result.form_boundaries[s][j]
        assert coefficients == tuple(
            a - b for a, b in zip(first.coefficients, second.coefficients)
        )
        assert coefficients[4] == 0
        assert upper.rhs + lower.rhs == 2 - first.constant + second.constant
        assert not upper.strict and not lower.strict
    # This is Fourier elimination of one shared offset, not 15 separately
    # chosen offsets or a claim that these 225 remaining rows are feasible.
    assert all(
        field.coefficients[4] == 1
        for boundary in result.form_boundaries
        for field in boundary
    )


def test_zero_peak_masks_cannot_be_removed_by_cross_multiplication(uniform):
    key = compiler.CellKey((0, 1), None)
    cell = next(cell for cell in uniform.cells if cell.key == key)
    rows = {row.label: row for row in cell.rows}
    low = rows["successor.zero.target.si_low"]
    assert low.coefficients == ZERO
    assert low.rhs == uniform.law.si_low - 1 < 0
    assert not low.strict and low.unit == "dimensionless"
    assert all(
        rows[f"successor.zero.il.b{i}"].rhs == 1 - uniform.law.si_high
        for i in range(1, 5)
    )
    assert "successor.peak.positive" not in rows
    assert "no native realization" in cell.zero_pressure_direction
    proof = compiler.check_cell_contradiction(uniform.spec, key, [(low.label, 1)])
    assert proof.proven and proof.weighted_rhs < 0
    assert not proof.positive_strict_weight


def test_exact_strict_and_non_strict_contradiction_rules(uniform):
    key = compiler.CellKey((0, 1), (1, 1))
    coefficient = 1 - uniform.law.si_low - uniform.law.stability[2]
    assert coefficient > 0
    weights = [("source.target.si_low", 1), ("source.peak.positive", coefficient)]
    result = compiler.check_cell_contradiction(uniform.spec, key, weights)
    assert result.proven and result.positive_strict_weight
    assert result.weighted_rhs == 0 and result.coefficients == ZERO
    assert result.scope == "conditional fixed-parameter cell only"
    assert result.spec == uniform.spec
    assert result.law == uniform.law and result.law is not uniform.law
    assert result.premises == uniform.premises
    assert result.premises is not uniform.premises
    assert result.controller_obligations == uniform.controller_obligations
    assert result.phase_geometry == uniform.phase_geometry
    empty = compiler.check_cell_contradiction(uniform.spec, key, [])
    assert not empty.proven and empty.weighted_rhs == 0
    zero_strict = compiler.check_cell_contradiction(
        uniform.spec, key, [("source.peak.b0.upper", 1), ("source.peak.positive", 0)]
    )
    assert not zero_strict.proven and not zero_strict.positive_strict_weight
    positive_rhs = compiler.check_cell_contradiction(
        uniform.spec, key, [("rail.t0.b0.upper", 1), ("rail.t0.b0.lower", 1)]
    )
    assert positive_rhs.weighted_rhs == 2 and not positive_rhs.proven
    with pytest.raises(ValueError, match="cancel exactly"):
        compiler.check_cell_contradiction(
            uniform.spec,
            key,
            [
                ("source.target.si_low", 1),
                ("source.peak.positive", coefficient + Q(1, 2**100)),
            ],
        )


def test_strict_source_boundaries_and_signed_zero_are_not_rounded(compiled):
    result = compiled["stable", "stable"]
    key = compiler.CellKey((0, 1), (1, 1))
    rows = {
        row.label: row
        for row in next(cell for cell in result.cells if cell.key == key).rows
    }
    for label, value in (
        ("source.target.oz_tiny", Q(1e-9)),
        (
            "source.target.acquired_rate",
            result.law.xi / (CAPACITIES[0] * result.law.oz_factor),
        ),
    ):
        coordinates = (value, Q(0), Q(0), Q(0), Q(0))
        row = rows[label]
        assert row.strict
        assert sum(a * x for a, x in zip(row.coefficients, coordinates)) == row.rhs
    assert rows["source.target.oz_ratio"].strict
    assert not rows["successor.target.nonpositive"].strict
    assert not rows["successor.target.zhir_ratio"].strict
    assert (
        rows["successor.target.nonpositive"].coefficients
        == result.successor_pressure[0].coefficients
    )
    assert (
        rows["successor.target.nonpositive"].rhs
        == -result.successor_pressure[0].constant
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacities", (True, 2, 3, 4, 5)),
        ("capacities", (0, 2, 3, 4, 5)),
        ("capacities", (-1, 2, 3, 4, 5)),
        ("capacities", (7, 2, 3, 4, 5)),
        ("capacities", (float("nan"), 2, 3, 4, 5)),
        ("capacities", {1, 2, 3, 4, 5}),
        ("capacities", {i: i + 1 for i in range(5)}),
        ("capacities", (1, 2, 3, 4)),
        ("phases", (False, 0, 0, 0, 0)),
        ("phases", (float("inf"), 0, 0, 0, 0)),
        ("phases", (1j, 0, 0, 0, 0)),
        ("phases", "00000"),
        ("first_branch", True),
        ("second_branch", "Stable"),
    ],
)
def test_invalid_primitives_reject_before_compilation(field, value, renewal_guard):
    with pytest.raises((TypeError, ValueError)):
        compiler.compile_native_mutation_slice(replace(_spec(), **{field: value}))


@pytest.mark.parametrize(
    "weights",
    [
        [("source.peak.b0.upper", True)],
        [("source.peak.b0.upper", -1)],
        [("source.peak.b0.upper", float("nan"))],
        [("source.peak.b0.upper", float("inf"))],
        [("source.peak.b0.upper", 1j)],
        [("source.peak.b0.upper", 0), ("source.peak.b0.upper", 0)],
        [("invented.row", 0)],
        [(0, 0)],
        [("source.peak.b0.upper",)],
        [("source.peak.b0.upper", 0, 0)],
        {("source.peak.b0.upper", 0)},
        {"source.peak.b0.upper": 0},
    ],
)
def test_invalid_certificates_are_not_inconclusive_or_feasible(uniform, weights):
    with pytest.raises((TypeError, ValueError)):
        compiler.check_cell_contradiction(
            uniform.spec, compiler.CellKey((0, 1), None), weights
        )


@pytest.mark.parametrize(
    "key",
    [
        compiler.CellKey((False, 1), None),
        compiler.CellKey((0.0, 1), None),
        compiler.CellKey((0, -1), None),
        compiler.CellKey((0, 0), None),
        compiler.CellKey((5, 1), None),
        compiler.CellKey((0, 1), (0, 1)),
        compiler.CellKey((0, 1), (1, True)),
        ((0, 1), None),
    ],
)
def test_invalid_signed_owner_admission_is_exact(uniform, key):
    with pytest.raises((TypeError, ValueError)):
        compiler.check_cell_contradiction(uniform.spec, key, [])


def test_only_rebuilt_primitives_can_authorize_a_certificate(uniform):
    key = compiler.CellKey((0, 1), None)
    weights = [("successor.zero.target.si_low", 1)]
    with pytest.raises(TypeError):
        compiler.check_cell_contradiction(uniform, key, weights)
    with pytest.raises(TypeError):
        compiler.compile_native_mutation_slice({"capacities": (1,) * 5, "phases": ZERO})
    fake = replace(uniform, cells=(), status="feasible")
    with pytest.raises(TypeError):
        compiler.check_cell_contradiction(fake, key, weights)
    assert compiler.check_cell_contradiction(fake.spec, key, weights).proven
    with pytest.raises(FrozenInstanceError):
        uniform.law.xi = Q(0)
    with pytest.raises(FrozenInstanceError):
        uniform.cells[0].rows[0].rhs = Q(100)


def test_changed_fixed_shift_default_is_rejected(monkeypatch, renewal_guard):
    cfg = CoreDefaults()
    cfg.GLYPH_FACTORS["ZHIR_theta_shift"] = Q(1, 8)
    monkeypatch.setattr(compiler, "CoreDefaults", lambda: cfg)
    with pytest.raises(ValueError, match="fixed Mutation shift"):
        compiler.compile_native_mutation_slice(_spec())
    with pytest.raises(ValueError, match="fixed Mutation shift"):
        compiler.check_cell_contradiction(_spec(), compiler.CellKey((0, 1), None), [])


def test_compilation_reads_no_artifact_or_old_source_builder(compiled, renewal_guard):
    _, forbidden = renewal_guard
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(guards.domain.gain, "_source", forbidden)
        rebuilt = compiler.compile_native_mutation_slice(_spec())
        checked = compiler.check_cell_contradiction(
            _spec(), compiler.CellKey((0, 1), None), []
        )
    assert rebuilt == compiled["stable", "stable"]
    assert checked.spec == rebuilt.spec and checked.law == rebuilt.law
    assert not checked.proven


def test_guarded_owner_never_consumes_native_or_response_producers(renewal_guard):
    targets, forbidden = renewal_guard
    assert len(targets) >= 30
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
