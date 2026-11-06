"""Formal coefficient controls for the joint two-boundary IL-floor theorem.

Four observation bases, two cyclic distances and C5 rotations/reflections
prove reconstruction. Tensor controls check the adjacent held-current row.
These are algebraic identities, not capacity/phase sources or trajectories.
"""

import builtins
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path

import pytest

from tests.research import test_native_mutation_feasibility as prior
from tnfr.config.defaults_core import CoreDefaults
from tnfr.research import _native_mutation_feasibility as shared
from tnfr.research import _native_mutation_joint_boundary as boundary

no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard
OBSERVATION_BASIS = tuple(tuple(Q(i == j) for i in range(4)) for j in range(4))


def _multiply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _laplacian(values):
    return tuple(
        (values[(i - 1) % 5] + values[(i + 1) % 5]) / 2 - values[i] for i in range(5)
    )


def _relabel(values, offset, orientation):
    output = [Q(0)] * 5
    for i, value in enumerate(values):
        output[(offset + orientation * i) % 5] = value
    return tuple(output)


@pytest.fixture(scope="module")
def theorem(renewal_guard):
    return boundary.derive_native_mutation_joint_boundary()


def test_floor_is_the_exact_zero_pressure_and_phase_boundary(theorem):
    law = prior._defaults()
    alpha, beta, gamma = law["stability"]
    assert theorem.law.stability == law["stability"]
    assert theorem.capacity_floor == (law["hi"] - beta - gamma) / alpha
    assert 0 < theorem.capacity_floor < 1
    assert beta > 0 and gamma > 0
    assert alpha * theorem.capacity_floor + beta + gamma == law["hi"]
    # At the exact floor, the IL residual is this sum of nonnegative
    # terms. Its coefficients are strictly positive, so both must vanish.
    residual_coefficients = (beta / prior.PI, gamma)
    assert all(value > 0 for value in residual_coefficients)
    assert sum(a * b for a, b in zip(residual_coefficients, (Q(0), Q(0)))) == 0
    # At peak zero, the actual convention sets rho=0 AND all pressures=0;
    # it cannot substitute an unconstrained p for a vanished normalized row.
    assert theorem.execution.dt == prior.DT
    assert theorem.execution.possible_counts == ((0,), (0, 1), (0, 1, 2))
    assert "strictly outward" in " ".join(theorem.obligations)
    assert "not all Mutation" in " ".join(theorem.obligations)
    assert (
        theorem.scope == "conditional two-boundary IL-floor theorem; no source verdict"
    )


def test_two_reconstruction_maps_are_exact_right_inverses_on_four_formal_bases(theorem):
    assert tuple(item.distance for item in theorem.reconstructions) == (1, 2)
    for instrument in theorem.reconstructions:
        left, right = instrument.observation_nodes
        assert (left, right) == (0, instrument.distance)
        assert len(instrument.matrix) == 5
        assert all(len(row) == 4 for row in instrument.matrix)
        for observations in OBSERVATION_BASIS:
            h = _multiply(instrument.matrix, observations)
            lap = _laplacian(h)
            assert sum(h) == 0
            assert (h[left], h[right], lap[left], lap[right]) == observations
        assert _multiply(instrument.matrix, (Q(0),) * 4) == (Q(0),) * 5


def test_rotations_and_reflections_cover_every_two_point_observation(theorem):
    covered = set()
    for instrument, offset, orientation in product(
        theorem.reconstructions, range(5), (-1, 1)
    ):
        left, right = offset, (offset + orientation * instrument.distance) % 5
        covered.add(tuple(sorted((left, right))))
        for observations in OBSERVATION_BASIS:
            canonical = _multiply(instrument.matrix, observations)
            h = _relabel(canonical, offset, orientation)
            lap = _laplacian(h)
            assert sum(h) == 0
            assert (h[left], h[right], lap[left], lap[right]) == observations
    assert covered == set(combinations(range(5), 2))
    assert set(combinations(range(1, 5), 2)) <= covered


def test_actual_first_gain_rows_recover_the_joint_phase_observations(theorem):
    law = prior._defaults()
    assert tuple(row.first_branch for row in theorem.first_gain_rows) == prior.BRANCHES
    for row in theorem.first_gain_rows:
        expected = prior._gains(law, law["gains"], row.first_branch)
        assert row.gains == expected
        global_gain, local_gain = expected
        assert local_gain > 0
        for instrument, observations in product(
            theorem.reconstructions, OBSERVATION_BASIS
        ):
            h = _multiply(instrument.matrix, observations)
            lap = _laplacian(h)
            g = _multiply(row.phase_matrix, h)
            assert g == tuple(
                (1 - global_gain) * value + local_gain * change
                for value, change in zip(h, lap)
            )
            recovered_lap = tuple(
                row.laplacian_recovery[0] * a + row.laplacian_recovery[1] * b
                for a, b in zip(h, g)
            )
            assert recovered_lap == lap
            # L annihilates the common global angle, and commutes with
            # this coordinator. This is a phase-gap, not absolute-angle row.
            assert _laplacian(_multiply(row.phase_matrix, h)) == _multiply(
                row.phase_matrix, _laplacian(h)
            )
            assert sum(g) == 0


def test_owned_support_and_adjacent_current_balance_match_fine_constituents(theorem):
    law = prior._defaults()
    k = law["pressure"][0]
    assert theorem.adjacent_current_balances == ((1, 2), (4, 3))
    for vector in tuple(tuple(Q(i == j) for i in range(5)) for j in range(5)):
        fine = prior._paired(vector)
        direct = tuple(
            sum(fine[j] for j in prior._neighbors(2 * i)) / 4 - fine[2 * i]
            for i in range(5)
        )
        assert _multiply(theorem.laplacian, vector) == direct == _laplacian(vector)
    # Formal bilinear capacity/pressure bases; zero basis coordinates are
    # coefficients, not proposed active physical capacities or NFR states.
    basis = ((Q(1), Q(0)), (Q(0), Q(1)))
    for (floor_block, other), capacities, pressures in product(
        theorem.adjacent_current_balances, basis, basis
    ):
        u, v_other = capacities
        p0, p_other = pressures
        flow = [Q(0)] * 5
        flow[0] = law["f"] * u * p0
        flow[other] = law["ell"] * v_other * p_other
        fine_flow = prior._paired(flow)
        # At saturation p_i=h_i=g_i=0. The complete transport at i is
        # therefore exactly the neighbor contribution to q_i below.
        q_row = (
            prior.DT
            * k
            * (
                sum(fine_flow[j] for j in prior._neighbors(2 * floor_block)) / 4
                - fine_flow[2 * floor_block]
            )
        )
        expected = (
            prior.DT * k * (law["f"] * u * p0 + law["ell"] * v_other * p_other) / 2
        )
        assert q_row == expected
    assert prior.DT * k / 2 > 0


def test_no_report_or_artifact_can_replace_fresh_owned_premises(theorem, renewal_guard):
    _, forbidden = renewal_guard
    with pytest.raises(TypeError):
        boundary.derive_native_mutation_joint_boundary(theorem)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(prior.guards.domain.gain, "_source", forbidden)
        rebuilt = boundary.derive_native_mutation_joint_boundary()
    assert rebuilt == theorem
    assert rebuilt.law is not theorem.law
    assert rebuilt.execution is not theorem.execution
    with pytest.raises(FrozenInstanceError):
        rebuilt.capacity_floor = Q(0)
    with pytest.raises(FrozenInstanceError):
        rebuilt.reconstructions[0].distance = 0


def test_changed_floor_law_is_rejected(monkeypatch, renewal_guard):
    cfg = CoreDefaults()
    cfg.SI_WEIGHTS.update({"alpha": 1, "beta": 1, "gamma": 1})
    monkeypatch.setattr(shared, "CoreDefaults", lambda: cfg)
    with pytest.raises(ValueError, match="positive IL capacity floor"):
        boundary.derive_native_mutation_joint_boundary()


def test_zero_local_gain_cannot_authorize_the_joint_reconstruction(
    monkeypatch, renewal_guard
):
    monkeypatch.setattr(boundary, "_gains", lambda *args: (Q(1, 8), Q(0)))
    with pytest.raises(ValueError, match="positive first phase row"):
        boundary.derive_native_mutation_joint_boundary()


def test_nonpaired_support_cannot_authorize_the_c5_obstruction(
    monkeypatch, renewal_guard
):
    original = boundary._owned_law

    def changed_support():
        law, execution = original()
        support = list(execution.support)
        support[1] = (0, 2, 3, 4)
        return law, replace(execution, support=tuple(support))

    monkeypatch.setattr(boundary, "_owned_law", changed_support)
    with pytest.raises(ValueError, match="paired unit C5 pressure support"):
        boundary.derive_native_mutation_joint_boundary()
