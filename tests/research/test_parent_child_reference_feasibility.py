"""Independent algebraic remainders for fixed-reference feasibility bounds.

The frozen parent-child-reference-admission protocol permits free symbolic
proof controls. These tests import no source consumer and evaluate no retained
preparation, numerical root, trajectory, event or parameter family.
"""

import sympy as s


def _zero(expression):
    assert s.cancel(s.expand(expression)) == 0


def test_weighted_full_row_bound_retains_the_pairwise_cauchy_remainder():
    weights = s.symbols("w0 w1 w2", positive=True)
    differences = s.symbols("z0 z1 z2", real=True)
    strength = sum(weights)
    gradient = sum(w * z for w, z in zip(weights, differences, strict=True))
    twice_form_storage = sum(
        w * z**2 for w, z in zip(weights, differences, strict=True)
    )
    remainder = sum(
        weights[i] * weights[j] * (differences[i] - differences[j]) ** 2
        for i in range(len(weights))
        for j in range(i + 1, len(weights))
    )
    _zero(strength * twice_form_storage - gradient**2 - remainder)
    assert remainder.is_nonnegative
    # Applying this to each full neighbor row bounds its form gradient by
    # sqrt(2*H*strength). Phase-coefficient rate bounds retain those complete
    # rows, including edges that are outside the marked parent's boundary.


def test_center_offset_bound_uses_the_joint_original_form_storage():
    w, sigma, gamma = s.symbols("w sigma gamma", positive=True)
    r, boundary_gradient = s.symbols("r boundary_gradient", real=True)
    kappa = w + gamma**2 * sigma
    centered_gradient = boundary_gradient + gamma * sigma * r
    reference_form = gamma * centered_gradient / kappa
    joint_form_storage = w * r**2 + boundary_gradient**2 / sigma
    remainder = sigma * w * (r - gamma * boundary_gradient / w) ** 2
    _zero(sigma * kappa * joint_form_storage / w - centered_gradient**2 - remainder)
    _zero(
        gamma**2 * sigma * joint_form_storage / kappa
        - w * reference_form**2
        - w * gamma**2 * remainder / kappa**2
    )
    assert remainder.is_nonnegative
    # Together with joint_form_storage <= 2*H, this proves the form part
    # of B_global without separately double-counting pair and host energy.


def test_accumulated_work_and_larger_offset_cannot_improve_the_ratio():
    root_upper, remainder = s.symbols("root_upper remainder", positive=True)
    initial_offset, offset_increment, work, duration = s.symbols(
        "initial_offset offset_increment work duration", nonnegative=True
    )
    excitation = remainder + work * duration
    initial_ratio = root_upper * initial_offset / excitation
    retained_ratio = root_upper * (initial_offset + offset_increment) / remainder
    difference = (
        root_upper
        * (excitation * offset_increment + initial_offset * work * duration)
        / (remainder * excitation)
    )
    _zero(retained_ratio - initial_ratio - difference)
    assert difference.is_nonnegative
    # The positive remainder is q-g*T. Any chosen B>=B0 therefore gives
    # rho(T)>=rho_Q before the strictly positive rotation gate is applied.


def test_rotation_rate_decreases_throughout_its_positive_admitted_domain():
    clock, upper, slack = s.symbols("clock upper slack", positive=True)
    ratio, increment = s.symbols("ratio increment", nonnegative=True)
    larger_ratio = ratio + increment
    lower = upper * larger_ratio + slack

    def rate(value):
        return clock * (lower - upper * value) / (1 + value) ** 2

    difference = (
        clock
        * increment
        * (upper * (1 + larger_ratio) ** 2 + slack * (2 + ratio + larger_ratio))
        / ((1 + ratio) ** 2 * (1 + larger_ratio) ** 2)
    )
    _zero(rate(ratio) - rate(larger_ratio) - difference)
    assert difference.is_nonnegative
    assert rate(larger_ratio).is_positive
    # The slack parametrizes lambda-Lambda*larger_ratio>0; monotonicity is
    # needed on this admitted domain, not on all possible ratios. Combined
    # with the previous control, a certified full-turn horizon is at least
    # 2*pi/Omega_opt. A uniform bound on G also satisfies g>=G(t0) by
    # definition; neither fact promotes G(t0) to a future upper bound.
