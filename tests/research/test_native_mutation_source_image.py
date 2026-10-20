"""Detached exact controls for the fixed phase-image theorem.

Formal coefficient bases and the single frozen rational phasor counterexample
are lemma controls, never native parameter preparations or source witnesses.
No prior controller bound, trigonometric kernel, source or solver is evaluated.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import combinations

import pytest

from tests.physics import test_native_mutation_renewal as guards
from tnfr.config.defaults_core import CoreDefaults
from tnfr.research import _native_mutation_controller as previous
from tnfr.research import _native_mutation_feasibility as compiler
from tnfr.research import _native_mutation_source_image as image

no_execution = guards.no_execution
calculus_guard = guards.calculus_guard
renewal_guard = guards.renewal_guard


@pytest.fixture(scope="module", autouse=True)
def source_image_guard(renewal_guard):
    targets, forbidden = renewal_guard
    additions = (
        (previous, "derive_native_mutation_controller_domain"),
        (previous, "certified_cosine_bounds"),
        (compiler, "_prepare"),
        (compiler, "compile_native_mutation_slice"),
        (compiler, "project_native_mutation_masks"),
    )
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in additions:
            patch.setattr(owner, name, forbidden)
        yield (*targets, *additions), forbidden


@pytest.fixture(scope="module")
def restriction(source_image_guard):
    targets, forbidden = source_image_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    return image.derive_native_mutation_source_image()


def _sum(*polynomials):
    result = {}
    for polynomial in polynomials:
        for power, value in polynomial.items():
            result[power] = result.get(power, Q(0)) + value
    return {power: value for power, value in result.items() if value}


def _product(left, right):
    terms = []
    for x, a in left.items():
        for y, b in right.items():
            terms.append({tuple(i + j for i, j in zip(x, y)): a * b})
    return _sum(*terms)


def _negative(polynomial):
    return {power: -value for power, value in polynomial.items()}


def test_pair_averaging_resultant_identities_are_exact_coefficients():
    # Variables are a, Re(Z), Im(Z), delta. They are formal coefficients,
    # not a parameter grid or values admitted as physical phase evidence.
    a, real, imag, delta = (
        {tuple(int(i == j) for i in range(4)): Q(1)} for j in range(4)
    )
    before = _sum(_product(_sum(a, real), _sum(a, real)), _product(imag, imag))
    after = _sum(
        _product(_sum(a, real, delta), _sum(a, real, delta)), _product(imag, imag)
    )
    projected_increment = _product(delta, _sum(a, real))
    assert _sum(after, _negative(before)) == _sum(
        projected_increment, projected_increment, _product(delta, delta)
    )
    complement = _sum(_product(real, real), _product(imag, imag))
    midpoint_projection = _product(a, _sum(a, real))
    assert _sum(before, _negative(complement)) == _sum(
        midpoint_projection, midpoint_projection, _negative(_product(a, a))
    )
    # Thus a>0 and a+Re(Z)<=0 force |S|²<=|Z|²-a²<9.
    # With |S|>=3 this is impossible; the first identity is nonnegative
    # for every averaging increment delta>=0, including delta=0.


def test_coordinator_matrix_has_the_majorization_and_contraction_premises(restriction):
    cfg = CoreDefaults()
    down = Q(cfg.PHASE_ADAPT["down"])
    independent = tuple(
        (1 - down) * Q(initial) + down * Q(cfg.PHASE_ADAPT[key])
        for initial, key in (
            (cfg.PHASE_K_GLOBAL, "kG_min"),
            (cfg.PHASE_K_LOCAL, "kL_min"),
        )
    )
    assert restriction.first_stable_gains == independent
    global_gain, local_gain = independent
    matrix = restriction.normalized_phase_matrix
    assert restriction.contraction == 1 - global_gain
    assert 0 < restriction.contraction < 1
    assert all(value >= 0 for row in matrix for value in row)
    assert all(sum(row) == 1 for row in matrix)
    assert all(sum(row[j] for row in matrix) == 1 for j in range(5))
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            coefficient = value * restriction.contraction
            expected = Q(0)
            if i == j:
                expected = 1 - global_gain - local_gain
            elif (j - i) % 5 in (1, 4):
                expected = local_gain / 2
            assert coefficient == expected
    # For any k output entries, input weights lie in [0,1] and sum to k.
    # Descending prefix indicators generate the cone of sorted vectors
    # modulo constants, proving the required majorization coefficient rows.
    for size in range(1, 5):
        for chosen in combinations(range(5), size):
            weights = tuple(sum(matrix[i][j] for i in chosen) for j in range(5))
            assert sum(weights) == size and all(0 <= value <= 1 for value in weights)
            for prefix in range(1, 6):
                assert sum(weights[:prefix]) <= min(size, prefix)


def test_low_resultant_counterexample_blocks_unconditional_monotonicity():
    # The sole frozen rational phasor control. All five have unit norm.
    pair = ((Q(24, 25), Q(7, 25)), (Q(24, 25), Q(-7, 25)))
    other = Q(-4, 5), Q(3, 5)
    phasors = (*pair, other, other, other)
    assert all(x * x + y * y == 1 for x, y in phasors)
    # Half-gap epsilon has tan(epsilon)=7/24. The opposite endpoint
    # gap is <pi because epsilon<atan(3/4); its sine is positive.
    assert Q(7, 24) < Q(3, 4)
    assert other[1] * pair[0][0] + other[0] * pair[0][1] == Q(44, 125) > 0
    before = tuple(sum(point[i] for point in phasors) for i in range(2))
    after = 2 + 3 * other[0], 3 * other[1]
    before_squared, after_squared = (
        sum(coordinate**2 for coordinate in point) for point in (before, after)
    )
    assert after_squared - before_squared == -Q(44, 625)
    assert after_squared < before_squared < 9
    assert before_squared / 25 < Q(3, 5) ** 2


def test_joint_source_bound_excludes_only_the_named_conditional_pair(restriction):
    cfg = CoreDefaults()
    high, low = Q(cfg.PHASE_ADAPT["R_hi"]), Q(cfg.PHASE_ADAPT["R_lo"])
    assert restriction.block_count == 5
    assert restriction.averaging_resultant_threshold == Q(3, 5) <= high
    assert restriction.source_lift_width_ceiling == 3
    assert restriction.source_resultant_lower == high
    assert (
        restriction.mutation_resultant_loss_bound == restriction.law.mutation_shift / 5
    )
    assert (
        restriction.successor_resultant_lower
        == high - restriction.mutation_resultant_loss_bound
    )
    assert restriction.dissonant_threshold == low
    assert restriction.strict_gap == restriction.successor_resultant_lower - low > 0
    traces = restriction.execution.traces
    counts = tuple(
        sum(code in {"OZ", "ZHIR"} for row in part for code in row)
        for part in traces[1:]
    )
    lengths = tuple(sum(map(len, part)) for part in traces[1:])
    assert restriction.disruptor_loads == tuple(
        Q(a, b) for a, b in zip(counts, lengths)
    )
    assert restriction.disruptor_loads == (Q(1, 20), Q(2, 25))
    assert max(restriction.disruptor_loads) < Q(cfg.PHASE_ADAPT["disr_lo"])
    assert restriction.excluded_branch_pair == ("stable", "dissonant")
    assert restriction.status == "conditionally_excluded"
    assert "source" in " ".join(restriction.obligations)


@pytest.mark.parametrize("change", ("low_resultant", "cut_equality", "fixed_shift"))
def test_changed_default_domains_cannot_inherit_the_exclusion(
    change, monkeypatch, restriction
):
    cfg = CoreDefaults()
    if change == "low_resultant":
        cfg.PHASE_ADAPT["R_lo"], cfg.PHASE_ADAPT["R_hi"] = Q(2, 5), Q(1, 2)
        message = "high-resultant theorem"
    elif change == "cut_equality":
        cfg.PHASE_ADAPT["R_hi"] = (
            Q(cfg.PHASE_ADAPT["R_lo"]) + restriction.mutation_resultant_loss_bound
        )
        message = "does not exclude dissonance"
    else:
        cfg.GLYPH_FACTORS["ZHIR_theta_shift"] = Q(1, 8)
        message = "fixed Mutation shift"
    monkeypatch.setattr(compiler, "CoreDefaults", lambda: cfg)
    with pytest.raises(ValueError, match=message):
        image.derive_native_mutation_source_image()


def test_report_is_immutable_and_cannot_supply_its_own_admission(restriction):
    with pytest.raises(FrozenInstanceError):
        restriction.status = "feasible"
    with pytest.raises(FrozenInstanceError):
        restriction.law.mutation_shift = Q(0)
    forged = replace(restriction, strict_gap=Q(100))
    with pytest.raises(TypeError):
        image.derive_native_mutation_source_image(forged)
    assert restriction.execution.node_ordinals == (3, 4, 5)
    assert "fresh CoreDefaults" in restriction.law.provenance


def test_high_resultant_threshold_equality_is_admitted(monkeypatch, source_image_guard):
    cfg = CoreDefaults()
    cfg.PHASE_ADAPT["R_hi"], cfg.PHASE_ADAPT["R_lo"] = Q(3, 5), Q(2, 5)
    monkeypatch.setattr(compiler, "CoreDefaults", lambda: cfg)
    boundary = image.derive_native_mutation_source_image()
    assert boundary.source_resultant_lower == boundary.averaging_resultant_threshold
    assert boundary.strict_gap > 0
    assert boundary.law.phase_policy.R_lo == Q(2, 5)
    assert boundary.status == "conditionally_excluded"


def test_second_actual_load_must_remain_in_the_low_policy_sector(
    monkeypatch, source_image_guard
):
    cfg = CoreDefaults()
    cfg.PHASE_ADAPT["disr_lo"] = Q(1, 20)
    monkeypatch.setattr(compiler, "CoreDefaults", lambda: cfg)
    with pytest.raises(ValueError, match="low-load controller context"):
        image.derive_native_mutation_source_image()
