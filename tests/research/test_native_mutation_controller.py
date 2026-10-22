"""Independent controls for the frozen nine-pair controller restriction.

The five endpoint counts per carried gain pair are algebraic extrema of an
enlarged phase box, not parameter sources. There is no feasibility search,
native call, solver, new phase preparation or refinement of the cosine work.
"""

from collections import Counter
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import combinations, product
from math import factorial

import pytest

from tests.physics import test_native_mutation_renewal as guards
from tnfr.config.defaults_core import CoreDefaults
from tnfr.research import _native_mutation_controller as controller

no_execution = guards.no_execution
calculus_guard = guards.calculus_guard
renewal_guard = guards.renewal_guard
BRANCHES = ("stable", "transition", "dissonant")


@pytest.fixture(scope="module")
def restriction(renewal_guard):
    targets, forbidden = renewal_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    return controller.derive_native_mutation_controller_domain()


def _carried_gains(policy, incoming, branch):
    if branch == "stable":
        target = policy["kG_min"], policy["kL_min"]
    elif branch == "dissonant":
        target = policy["kG_max"], (policy["kL_min"] + policy["kL_max"]) / 2
    else:
        target = (policy["kG_min"] + policy["kG_max"]) / 2, (
            policy["kL_min"] + policy["kL_max"]
        ) / 2
    return tuple(
        (1 - policy["up"] if target[i] > incoming[i] else 1 - policy["down"])
        * incoming[i]
        + (policy["up"] if target[i] > incoming[i] else policy["down"]) * target[i]
        for i in range(2)
    )


def test_fixed_work_and_carried_default_controller_memory(restriction):
    assert restriction.terms == 8 and restriction.bits == 128
    assert len(restriction.cells) == 9
    assert {
        (cell.first_branch, cell.second_branch) for cell in restriction.cells
    } == set(product(BRANCHES, repeat=2))
    cfg = CoreDefaults()
    policy = {
        key: Q(value) for key, value in cfg.PHASE_ADAPT.items() if key != "enabled"
    }
    gains0 = Q(cfg.PHASE_K_GLOBAL), Q(cfg.PHASE_K_LOCAL)
    assert restriction.loads == (Q(1, 20), Q(2, 25))
    assert max(restriction.loads) < policy["disr_lo"] < policy["disr_hi"]
    for cell in restriction.cells:
        first = _carried_gains(policy, gains0, cell.first_branch)
        second = _carried_gains(policy, first, cell.second_branch)
        assert (cell.first_gains, cell.second_gains) == (first, second)
        assert second != _carried_gains(policy, gains0, cell.second_branch)
        assert tuple(endpoint.high_count for endpoint in cell.endpoints) == tuple(
            range(5)
        )
    assert restriction.execution.node_ordinals == (3, 4, 5)
    assert restriction.execution.runtime_ordinals == (0, 1, 2)
    assert restriction.execution.traces[1][0] == ("AL", "IL", "EN", "OZ")
    assert restriction.execution.traces[2][0] == ("AL", "IL", "EN", "OZ", "ZHIR")


def test_five_cycle_row_geometry_gives_the_renewal_diameter_bound(restriction):
    for cell in restriction.cells:
        global_gain, local_gain = cell.second_gains
        amplitude = 1 - global_gain - local_gain
        matrix = tuple(
            tuple(
                (
                    amplitude
                    if j == i
                    else local_gain / 2 if (i - j) % 5 in (1, 4) else Q(0)
                )
                for j in range(5)
            )
            for i in range(5)
        )
        assert amplitude >= local_gain / 2 > 0
        assert all(sum(row) == 1 - global_gain for row in matrix)
        norms = []
        for i, j in combinations(range(5), 2):
            difference = tuple(a - b for a, b in zip(matrix[i], matrix[j]))
            assert sum(difference) == 0
            norm = sum(max(value, Q(0)) for value in difference)
            # The sign corner realizes the affine extremum on the unit cube;
            # this is a coefficient identity, not an admitted phase source.
            corner = tuple(Q(value > 0) for value in difference)
            assert sum(a * b for a, b in zip(difference, corner)) == norm
            expected = (
                amplitude if (i - j) % 5 in (1, 4) else amplitude + local_gain / 2
            )
            assert norm == expected
            norms.append(norm)
        oscillation = max(norms)
        shift_column = tuple(row[0] for row in matrix)
        assert max(shift_column) - min(shift_column) == amplitude
        assert oscillation == 1 - global_gain - local_gain / 2 < 1
        assert (1 - oscillation) * cell.diameter_ceiling == amplitude * cell.shift
        assert 0 < cell.shift < cell.diameter_ceiling
        assert cell.diameter_ceiling + cell.shift < 3


def test_cosine_enclosures_retain_fixed_alternating_tail_and_rounding(restriction):
    grid = Q(1, 1 << 128)
    for cell in restriction.cells:
        for angle, (lower, upper) in zip(cell.cosine_angles, cell.cosine_enclosures):
            assert 0 < angle < 3
            polynomial = sum(
                Q((-1) ** j, factorial(2 * j)) * angle ** (2 * j) for j in range(8)
            )
            remainder = angle**16 / factorial(16)
            # On this short arc the first omitted alternating term is
            # positive and subsequent magnitudes strictly decrease.
            assert angle * angle < 17 * 18
            assert lower <= polynomial <= polynomial + remainder <= upper
            assert 0 <= polynomial - lower < grid
            assert 0 <= upper - polynomial - remainder < grid
            assert upper - lower <= remainder + 2 * grid


def test_pair_difference_counts_independently_recover_all_corner_bounds(restriction):
    for cell in restriction.cells:
        shift, diameter = cell.shift, cell.diameter_ceiling
        cosine = dict(zip(cell.cosine_angles, cell.cosine_enclosures))
        cosine[Q(0)] = Q(1), Q(1)
        for endpoint in cell.endpoints:
            high, low = endpoint.high_count, endpoint.low_count
            assert high + low == 4
            # This symbolic count pattern is exactly the frozen relaxation;
            # it is not a source admitted through the first coordinator.
            pattern = (-shift,) + (Q(0),) * low + (diameter,) * high
            differences = Counter(abs(a - b) for a, b in combinations(pattern, 2))
            direct = tuple(
                (
                    Q(5)
                    + 2
                    * sum(
                        count * cosine[gap][side] for gap, count in differences.items()
                    )
                )
                / 25
                for side in (0, 1)
            )
            assert direct == (endpoint.r_squared_lower, endpoint.r_squared_upper)
        assert cell.universal_r_squared_lower == min(
            x.r_squared_lower for x in cell.endpoints
        )
        assert cell.relaxed_minimum_upper == min(
            x.r_squared_upper for x in cell.endpoints
        )
        assert cell.universal_r_squared_lower <= cell.relaxed_minimum_upper


def test_exact_branch_exclusions_preserve_all_unresolved_pairs(restriction):
    low_cut, high_cut = restriction.r_squared_thresholds
    expected_excluded = {("transition", "dissonant"), ("dissonant", "dissonant")}
    observed = set()
    for cell in restriction.cells:
        impossible = (
            cell.second_branch == "dissonant"
            and cell.universal_r_squared_lower > low_cut
        ) or (
            cell.second_branch == "transition"
            and cell.universal_r_squared_lower >= high_cut
        )
        assert (cell.status == "excluded") == impossible
        if impossible:
            observed.add((cell.first_branch, cell.second_branch))
            assert cell.exclusion_reason is not None
        else:
            assert cell.status == "conditional" and cell.exclusion_reason is None
    assert observed == expected_excluded
    assert sum(cell.status == "conditional" for cell in restriction.cells) == 7


def test_the_retained_relaxation_really_crosses_the_unresolved_cut(restriction):
    cell = next(
        x
        for x in restriction.cells
        if (x.first_branch, x.second_branch) == ("stable", "dissonant")
    )
    low_cut = restriction.r_squared_thresholds[0]
    assert cell.universal_r_squared_lower <= cell.relaxed_minimum_upper < low_cut
    assert low_cut - cell.relaxed_minimum_upper > Q(1, 2000)
    # The shared enclosure proves that a RELAXED corner minimum is below
    # the cut. More cosine precision cannot turn this box into exclusion.
    # That corner need not have an admitted first-controller preimage.
    assert cell.status == "conditional"
    assert cell.relaxed_minimum_upper < 1
    assert "conditional means not excluded" in " ".join(restriction.obligations)


def test_result_immutability_and_no_cached_admission_consumer(restriction):
    with pytest.raises(FrozenInstanceError):
        restriction.terms = 16
    with pytest.raises(FrozenInstanceError):
        restriction.cells[0].status = "excluded"
    forged = replace(restriction, cells=())
    # The no-argument owner has no path accepting an edited report as proof.
    with pytest.raises(TypeError):
        controller.derive_native_mutation_controller_domain(forged)
