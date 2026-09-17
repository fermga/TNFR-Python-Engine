"""Exact count algebra oracles and canonical-boundary checks for C6 returns.

The small synthetic mode graph tests the incidence/displacement theorem only;
its deliberately restrictive guards make no assertion of nodal reachability.
Public-entry tests independently exercise real canonical source rebuilding.
"""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice


DIVISORS = (1, 2, 3, 4, 5, 6)
ZERO = ((0,) * 7,) * 7


@pytest.fixture(scope="module")
def algebra():
    rows = tuple((.4 + index / 1000,) + (.5,) * 5 for index in range(13))
    shifts = [(0,) * 6]
    for axis, divisor in enumerate(DIVISORS):
        shifts.extend(tuple(sign * divisor * int(i == axis) for i in range(6)) for sign in (1, -1))
    edges = tuple(owner.C6CarriedReturnTransition(
        source, None, target, ZERO, None, owner._move(ZERO, shifts[a]), shifts[a],
    ) for a, source in enumerate(rows) for target in rows)
    state = NodalRemainderState(rows[0], (F(0),) * 6, .375, .625)
    zones = tuple(C6CarriedForwardZone(row, ZERO) for row in rows)
    envelope = owner.C6CarriedReturnEnvelope(
        None, state, 1., rows, tuple(tuple(map(float, shift)) for shift in shifts), F(1),
        state.exact_epi, (), zones, zones, edges, (), len(edges), True, True, (),
        "fixed_point", 0, 0, 1, len(rows),
    )
    circulation = (1,) * len(edges)
    generators = []
    for axis in range(6):
        index = 2 * axis + 1
        generators.append(tuple(int(i == index * len(rows) + index) for i in range(len(edges))))
    return envelope, circulation, tuple(generators)


@pytest.fixture(scope="module")
def certificate(algebra):
    return owner._certify_return_count_relaxation(*algebra)


def test_coordinate_cycle_basis_and_positive_circulation_are_verified_exactly(certificate, algebra):
    assert certificate.coordinate_divisors == DIVISORS
    assert certificate.positive_circulation == algebra[1]
    assert certificate.coordinate_generators == algebra[2]
    assert certificate.exact_cycle_displacement_lattice_certified
    assert certificate.nonnegative_counts_cover_coordinate_cosets
    assert certificate.abstract_mode_walk_exists_for_each_coordinate_coset_point
    assert not certificate.joint_guard_satisfaction_certified
    assert not certificate.actual_origin_reachability_certified
    assert not certificate.conditional_invariance_certified
    assert not certificate.conditional_boundedness_certified
    assert not certificate.future_runtime_certified and not certificate.asymptotic_convergence_certified


def test_saved_paths_have_the_certified_direction_and_use_complete_relation_edges(certificate):
    envelope = certificate.return_envelope
    origin = envelope.state.epi
    assert len(certificate.origin_to_mode_paths) == len(certificate.mode_to_origin_paths) == 13
    for target, indices in certificate.origin_to_mode_paths:
        current = origin
        for index in indices:
            edge = envelope.return_relation[index]
            assert edge.source_epi == current
            current = edge.target_epi
        assert current == target
    for source, indices in certificate.mode_to_origin_paths:
        current = source
        for index in indices:
            edge = envelope.return_relation[index]
            assert edge.source_epi == current
            current = edge.target_epi
        assert current == origin


@pytest.mark.parametrize("target_index", (0, 1, 12))
@pytest.mark.parametrize("coefficients", ((0,) * 6, (1, -2, 3, -4, 5, -6),
                                          (2**90, -2**91, 2**92, -2**93, 2**94, -2**95)))
def test_constructor_satisfies_all_integer_counts_and_exact_displacements(certificate, target_index, coefficients):
    envelope = certificate.return_envelope
    target = envelope.epi_states[target_index]
    displacement = tuple(a * b for a, b in zip(coefficients, DIVISORS, strict=True))
    witness = certificate.construct_counts(target_epi=target, displacement=displacement)
    assert witness.target_epi == target and witness.displacement == displacement
    assert all(type(value) is int and value > 0 for value in witness.edge_counts)
    # Independent incidence and nodal-displacement sums; no production helper.
    for row in envelope.epi_states:
        outward = sum(count for count, edge in zip(witness.edge_counts, envelope.return_relation, strict=True)
                      if edge.source_epi == row)
        inward = sum(count for count, edge in zip(witness.edge_counts, envelope.return_relation, strict=True)
                     if edge.target_epi == row)
        assert outward - inward == int(row == envelope.state.epi) - int(row == target)
    assert tuple(sum(count * edge.shift[i] for count, edge in zip(
        witness.edge_counts, envelope.return_relation, strict=True,
    )) for i in range(6)) == displacement
    assert not witness.actual_origin_reachability_certified
    assert not witness.joint_guard_satisfaction_certified


def test_count_feasibility_does_not_force_the_endpoint_to_satisfy_its_rn_guard(certificate):
    witness = certificate.construct_counts(
        target_epi=certificate.return_envelope.state.epi, displacement=(1000,) + (0,) * 5,
    )
    assert all(value > 0 for value in witness.edge_counts)
    # Every synthetic source guard is the singleton zero; this endpoint is
    # outside that geometry despite its certified count representation.
    assert witness.displacement[0] > ZERO[0][6]
    assert not witness.joint_guard_satisfaction_certified


@pytest.mark.parametrize("tamper", ("zero", "negative", "bool", "length", "container", "unbalanced", "shift"))
def test_invalid_positive_circulation_is_rejected(algebra, tamper):
    envelope, circulation, generators = algebra
    bad = list(circulation)
    if tamper == "zero":
        bad[0] = 0
    elif tamper == "negative":
        bad[0] = -1
    elif tamper == "bool":
        bad[0] = True
    elif tamper == "length":
        bad.pop()
    elif tamper == "unbalanced":
        bad[1] += 1
    elif tamper == "shift":
        bad[14] += 1
    with pytest.raises((TypeError, ValueError)):
        owner._certify_return_count_relaxation(envelope, bad if tamper == "container" else tuple(bad), generators)


@pytest.mark.parametrize("tamper", ("length", "bool", "unbalanced", "shift", "outer_container"))
def test_invalid_signed_coordinate_generators_are_rejected(algebra, tamper):
    envelope, circulation, generators = algebra
    bad = [list(row) for row in generators]
    if tamper == "length":
        bad[0].pop()
    elif tamper == "bool":
        bad[0][0] = False
    elif tamper == "unbalanced":
        bad[0][1] += 1
    elif tamper == "shift":
        bad[0][3 * 13 + 3] += 1
    else:
        with pytest.raises(TypeError):
            owner._certify_return_count_relaxation(envelope, circulation, bad)
        return
    with pytest.raises((TypeError, ValueError)):
        owner._certify_return_count_relaxation(envelope, circulation, tuple(map(tuple, bad)))


def test_disconnected_mode_graph_is_rejected_even_with_valid_balanced_cycle_candidates(algebra):
    envelope, _, _ = algebra
    edges = tuple(edge for edge in envelope.return_relation if edge.source_epi == edge.target_epi)
    disconnected = replace(envelope, return_relation=edges)
    generators = tuple(tuple(int(i == 2 * axis + 1) for i in range(13)) for axis in range(6))
    with pytest.raises(ValueError, match="strongly connected"):
        owner._certify_return_count_relaxation(disconnected, (1,) * 13, generators)


@pytest.mark.parametrize("flag", ("relation_complete", "transient_isolation_certified"))
def test_incomplete_relation_premises_are_rejected(algebra, flag):
    envelope, circulation, generators = algebra
    with pytest.raises(ValueError, match="completely constructed"):
        owner._certify_return_count_relaxation(replace(envelope, **{flag: False}), circulation, generators)


@pytest.mark.parametrize("tamper", ("gcd", "bool", "length", "container", "target"))
def test_constructor_rejects_nongrid_or_invalid_input(certificate, tamper):
    target = certificate.return_envelope.state.epi
    displacement = (0,) * 6
    if tamper == "gcd":
        displacement = (0, 1, 0, 0, 0, 0)
    elif tamper == "bool":
        displacement = (True,) + (0,) * 5
    elif tamper == "length":
        displacement = (0,) * 5
    elif tamper == "container":
        displacement = [0] * 6
    else:
        target = (.5,) * 6
    with pytest.raises((TypeError, ValueError)):
        certificate.construct_counts(target_epi=target, displacement=displacement)


def test_real_canonical_stationary_coordinate_is_explicitly_outside_full_rank_count_theorem():
    reference = derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)
    state = NodalRemainderState((.5,) * 6, (F(0),) * 6, .375, .625)
    with pytest.raises(ValueError, match="positive increment gcd"):
        owner.derive_c6_carried_return_count_relaxation(
            reference, state=state, epi_states=(state.epi,), timestep=1., transient_epi_states=(),
            positive_circulation=(1,), coordinate_generators=((0,),) * 6,
        )


def test_public_entry_rebuilds_canonical_source_before_interpreting_proof_candidates():
    reference = derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)
    state = NodalRemainderState((.5,) * 6, (F(0),) * 6, .375, .625)
    forged = replace(reference, source=replace(reference.source, phase=(float("nan"),) * 6))
    with pytest.raises(ValueError):
        owner.derive_c6_carried_return_count_relaxation(
            forged, state=state, epi_states=(state.epi,), timestep=1., transient_epi_states=(),
            positive_circulation=(), coordinate_generators=(),
        )


def test_public_construction_budget_never_promotes_an_incomplete_return_relation():
    reference = derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)
    state = NodalRemainderState((.5,) * 6, (F(0),) * 6, .375, .625)
    rows = (state.epi, (.5000000000000001,) * 6)
    with pytest.raises(ValueError, match="completely constructed"):
        owner.derive_c6_carried_return_count_relaxation(
            reference, state=state, epi_states=rows, timestep=1., transient_epi_states=(),
            positive_circulation=(), coordinate_generators=(), max_intersections=1,
        )
