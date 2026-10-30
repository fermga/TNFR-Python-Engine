"""Independent fine-coordinate controls for the principal-chamber reduction.

These fixed static configurations are algebra controls, not equilibria or
responses of the retained source. No operator, trajectory or producer is run.
"""

from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tnfr.dynamics import integrators, relational
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions import Coupling, SelfOrganization
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research.artifact_io import file_receipt
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.maintained_successor_stationary import (
    SELECTED_SHIFTS,
    SELECTED_SUPPORT,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_thol"
PRIOR = ROOT / "docs/assets/local_capacity_history"
CORE = (Q(11, 10), Q(6, 5), Q(12, 5), Q(1, 10), Q(11, 10), Q(19, 10))


def _mp(mp, value):
    exact = Q(value)
    return mp.mpf(exact.numerator) / exact.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("principal geometry must not execute an event or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (maintained_parent_thol_execution, "execute_network_operator_stage"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "rk4_coupled_step"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
            (Coupling, "_execute"),
            (SelfOrganization, "_prepare_execution"),
            (SelfOrganization, "_prepare_bifurcation"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def study():
    from tnfr.research.maintained_successor_principal import evaluate_successor_core

    paths = tuple(
        sorted(
            path
            for directory in (EVIDENCE, PRIOR)
            for path in directory.iterdir()
            if path.is_file()
        )
    )
    receipts = tuple(file_receipt(path) for path in paths)
    source = load_thol_successor(EVIDENCE, repository_root=ROOT)
    signature = _signature(source.graph)
    result = evaluate_successor_core(source.graph, coordinates=CORE)
    yield source, result
    assert _signature(source.graph) == signature
    assert receipts == tuple(file_receipt(path) for path in paths)


def _core_phases(mp, coordinates):
    parent, partner, a2, b0, b1, b2 = coordinates
    qa = (2 * mp.pi - a2) / 3
    qb = (2 * mp.pi + b0 - b2) / 3
    return (
        mp.mpf(0),
        mp.mpf(0),
        parent,
        partner,
        a2,
        a2,
        a2 + qa,
        a2 + qa,
        a2 + 2 * qa,
        a2 + 2 * qa,
        b0,
        b0,
        b1,
        b1,
        b2,
        b2,
        b2 + qb,
        b2 + qb,
        b2 + 2 * qb,
        b2 + 2 * qb,
        parent,
    )


def _fine_values(mp, phases):
    gradient = mp.matrix(21, 1)
    hessian = mp.matrix(21)
    potential = mp.mpf(0)
    for left, right in SELECTED_SUPPORT:
        difference = phases[right] - phases[left]
        sine, cosine = mp.sin(difference), mp.cos(difference)
        potential += 1 - cosine
        gradient[left] -= sine
        gradient[right] += sine
        hessian[left, left] += cosine
        hessian[right, right] += cosine
        hessian[left, right] -= cosine
        hessian[right, left] -= cosine
    return potential, gradient, hessian


@pytest.fixture(scope="module")
def oracle():
    mp = mpmath.mp.clone()
    mp.dps = 90
    coordinates = tuple(_mp(mp, value) for value in CORE)
    phases = _core_phases(mp, coordinates)
    potential, gradient, hessian = _fine_values(mp, phases)
    jacobian = mp.matrix(
        [
            [
                mp.diff(
                    lambda *args: _core_phases(mp, args)[i],
                    coordinates,
                    tuple(int(k == j) for k in range(6)),
                )
                for j in range(6)
            ]
            for i in range(21)
        ]
    )
    return mp, phases, potential, gradient, hessian, jacobian


def test_fresh_source_keeps_all_53_branches_and_33_cycles(study):
    _, result = study
    assert result.chamber.edges == SELECTED_SUPPORT
    assert result.chamber.ideal.shifts == SELECTED_SHIFTS
    assert result.chamber.conditional.shifts == SELECTED_SHIFTS
    assert result.chamber.cycle_rank == 53 - 21 + 1 == 33


def test_full_fine_interval_rows_match_independent_sums(study, oracle):
    _, result = study
    candidate = result.candidate
    mp, phases, potential, gradient, hessian, _ = oracle
    _contains(mp, candidate.potential, potential)
    for interval, value in zip(candidate.phases, phases, strict=True):
        _contains(mp, interval, value)
    for interval, value in zip(candidate.gradient, gradient, strict=True):
        _contains(mp, interval, value)
    for row, expected in zip(candidate.hessian, hessian.tolist(), strict=True):
        for interval, value in zip(row, expected, strict=True):
            _contains(mp, interval, value)
    # This admitted algebra probe deliberately is not a stationary candidate.
    assert max(abs(value) for value in gradient) > mp.mpf("0.1")


def test_six_core_is_the_full_hessian_pullback(study, oracle):
    _, result = study
    candidate = result.candidate
    mp, _, _, gradient, hessian, jacobian = oracle
    expected_gradient = jacobian.T * gradient
    expected_hessian = jacobian.T * hessian * jacobian
    for interval, value in zip(candidate.core_gradient, expected_gradient, strict=True):
        _contains(mp, interval, value)
    for row, expected in zip(
        candidate.core_hessian, expected_hessian.tolist(), strict=True
    ):
        for interval, value in zip(row, expected, strict=True):
            _contains(mp, interval, value)
    # Removing a common phase to retain any positive weighted mean changes
    # neither curvature nor the actual fine relative directions.
    rho = mp.matrix([mp.mpf(i + 1) / (i + 2) for i in range(21)])
    centered = jacobian - mp.ones(21, 1) * (rho.T * jacobian) / mp.fsum(rho)
    assert mp.norm(centered.T * hessian * centered - expected_hessian) < mp.mpf("1e-80")


def test_omitted_twin_directions_have_the_reported_curvatures(study, oracle):
    _, result = study
    mp, _, _, _, hessian, _ = oracle
    pairs = (
        (0, 1),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, 17),
        (18, 19),
    )
    for pair, interval in zip(
        pairs, result.candidate.transverse_eigenvalues, strict=True
    ):
        direction = mp.matrix(21, 1)
        direction[pair[0]], direction[pair[1]] = 1, -1
        eigenvalue = (direction.T * hessian * direction)[0] / 2
        _contains(mp, interval, eigenvalue)
        assert mp.norm(hessian * direction - eigenvalue * direction) < mp.mpf("1e-80")


def _four_phases(mp, coordinates):
    a, mean_a, b, mean_b = coordinates
    origin = mean_a - a
    a2 = mean_a + a
    b0, b2 = mean_b - b, mean_b + b
    partner = mp.arg(1 + 4 * mp.cos(a) * mp.exp(1j * mean_a))
    b1 = mp.arg(1 + 4 * mp.cos(b) * mp.exp(1j * mean_b))
    return tuple(
        value + origin
        for value in _core_phases(
            mp,
            (
                -origin,
                partner - origin,
                a2 - origin,
                b0 - origin,
                b1 - origin,
                b2 - origin,
            ),
        )
    )


def _four_potential(mp, coordinates):
    a, mean_a, b, mean_b = coordinates
    ra = abs(1 + 4 * mp.cos(a) * mp.exp(1j * mean_a))
    rb = abs(1 + 4 * mp.cos(b) * mp.exp(1j * mean_b))
    return (
        51
        - 4 * mp.cos(a) * mp.cos(mean_a)
        - ra
        - 12 * mp.cos((2 * mp.pi - 2 * a) / 3)
        - 4 * mp.cos(b) * mp.cos(mean_b)
        - 2 * rb
        - 12 * mp.cos((2 * mp.pi - 2 * b) / 3)
        - 4 * mp.cos(mean_a - a - mean_b + b)
    )


@pytest.mark.parametrize("half_b", [Q(9, 10), Q(8, 5)])
def test_four_potential_retains_nonacute_edges_and_full_fine_hessian(half_b):
    from tnfr.research.maintained_successor_principal import _evaluate_reduction

    mp = mpmath.mp.clone()
    mp.dps = 90
    exact_coordinates = (Q(6, 5), Q(1, 10), half_b, 0)
    coordinates = tuple(_mp(mp, value) for value in exact_coordinates)
    result = _evaluate_reduction(
        exact_coordinates, edges=SELECTED_SUPPORT, shifts=SELECTED_SHIFTS
    )
    phases = _four_phases(mp, coordinates)
    potential, _, hessian = _fine_values(mp, phases)
    gaps = tuple(
        phases[j] - phases[i] - 2 * mp.pi * turn
        for (i, j), turn in zip(SELECTED_SUPPORT, SELECTED_SHIFTS, strict=True)
    )
    assert max(abs(gap) for gap in gaps) < mp.pi
    if half_b > Q(3, 2):
        assert max(abs(gap) for gap in gaps) > mp.pi / 2
    assert abs(potential - _four_potential(mp, coordinates)) < mp.mpf("1e-80")
    _contains(mp, result.potential, potential)
    for i, interval in enumerate(result.gradient):
        value = mp.diff(
            lambda *args: _four_potential(mp, args),
            coordinates,
            tuple(int(k == i) for k in range(4)),
        )
        _contains(mp, interval, value)
    jacobian = mp.matrix(
        [
            [
                mp.diff(
                    lambda *args: _four_phases(mp, args)[i],
                    coordinates,
                    tuple(int(k == j) for k in range(4)),
                )
                for j in range(4)
            ]
            for i in range(21)
        ]
    )
    reduced_hessian = mp.matrix(
        [
            [
                mp.diff(
                    lambda *args: _four_potential(mp, args),
                    coordinates,
                    tuple(int(k == i) + int(k == j) for k in range(4)),
                )
                for j in range(4)
            ]
            for i in range(4)
        ]
    )
    assert mp.norm(jacobian.T * hessian * jacobian - reduced_hessian) < mp.mpf("1e-75")
    for row, expected in zip(result.hessian, reduced_hessian.tolist(), strict=True):
        for interval, value in zip(row, expected, strict=True):
            _contains(mp, interval, value)


def test_changing_a_lift_without_its_branches_is_not_the_same_candidate():
    from tnfr.research.maintained_successor_principal import _evaluate_core

    changed = list(CORE)
    changed[0] += 7  # More than one complete turn; all other lifts are retained.
    with pytest.raises(ValueError, match="principal"):
        _evaluate_core(changed, edges=SELECTED_SUPPORT, shifts=SELECTED_SHIFTS)


@pytest.mark.parametrize("component", ["A", "B"])
def test_zero_cosine_three_block_chain_has_both_cubic_signs(component):
    mp = mpmath.mp.clone()
    mp.dps = 90
    parameters = [mp.mpf(1), mp.mpf(1), mp.pi / 2, mp.mpf(0), mp.mpf(1), mp.pi / 2]
    phases = list(_core_phases(mp, parameters))
    indices = ((6, 7), (8, 9)) if component == "A" else ((16, 17), (18, 19))
    baseline = _fine_values(mp, phases)[0]
    for sign in (-1, 1):
        increment = mp.mpf(sign) / 1024
        perturbed = phases.copy()
        for multiple, pair in enumerate(indices, 1):
            for node in pair:
                perturbed[node] += multiple * increment
        difference = _fine_values(mp, perturbed)[0] - baseline
        expected = 8 * mp.sin(increment) * (1 - mp.cos(increment))
        assert abs(difference - expected) < mp.mpf("1e-80")
        assert sign * difference > 0
        assert (
            max(
                abs(perturbed[j] - perturbed[i] - 2 * mp.pi * turn)
                for (i, j), turn in zip(SELECTED_SUPPORT, SELECTED_SHIFTS, strict=True)
            )
            < mp.pi
        )


def test_odd_degree_alone_does_not_make_the_resultant_nonzero():
    """The exclusion also needs stationary PSD zero-row reasoning."""
    mp = mpmath.mp.clone()
    mp.dps = 90
    angle = mp.acos(-mp.mpf(1) / 4)
    neighbors = [angle, angle, -angle, -angle, mp.mpf(0)]
    assert abs(mp.fsum(mp.exp(1j * value) for value in neighbors)) < mp.mpf("1e-80")
    assert any(abs(mp.cos(value)) > mp.mpf("0.2") for value in neighbors)
    # If a PSD zero diagonal forced all five incident cosines to zero, each
    # incident sine would instead be +1 or -1; their odd sum cannot vanish.
    assert all(2 * positives - 5 != 0 for positives in range(6))
