"""Independent full-support controls for the analytical acute obstruction.

The retained THOL input is only read. Static high-precision calculations
check the face witness and algebra, not an executed UM endpoint or flow.
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
from tnfr.research.maintained_successor_stationary import certify_successor_stationary

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_thol"
PRIOR = ROOT / "docs/assets/local_capacity_history"


def _mp(mp, value):
    exact = Q(value)
    return mp.mpf(exact.numerator) / exact.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("stationary admission must not execute an event or trajectory")

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
    certificate = certify_successor_stationary(source.graph)
    yield source, certificate
    assert _signature(source.graph) == signature
    assert receipts == tuple(file_receipt(path) for path in paths)


def _face_map(mp, parameters):
    """Independently embed five real coordinates in all 21 marked lifts."""
    parent, partner, chain, origin, short_gap = parameters
    return (
        mp.mpf(0),
        mp.mpf(0),
        parent,
        partner,
        2 * mp.pi - 3 * chain,
        2 * mp.pi - 3 * chain,
        2 * mp.pi - 2 * chain,
        2 * mp.pi - 2 * chain,
        2 * mp.pi - chain,
        2 * mp.pi - chain,
        origin,
        origin,
        origin + short_gap,
        origin + short_gap,
        origin + mp.pi / 2,
        origin + mp.pi / 2,
        origin + mp.pi,
        origin + mp.pi,
        origin + 3 * mp.pi / 2,
        origin + 3 * mp.pi / 2,
        parent,
    )


def _gradient(mp, phases, edges):
    values = [mp.mpf(0) for _ in phases]
    for left, right in edges:
        current = mp.sin(phases[left] - phases[right])
        values[left] += current
        values[right] -= current
    return tuple(values)


def _potential(mp, phases, edges):
    return mp.fsum(1 - mp.cos(phases[j] - phases[i]) for i, j in edges)


def _pullback(mp, gradient):
    """Differentiate the full potential through the affine face embedding."""
    return (
        gradient[2] + gradient[20],
        gradient[3],
        -mp.fsum(
            coefficient * gradient[index]
            for coefficient, indices in ((3, (4, 5)), (2, (6, 7)), (1, (8, 9)))
            for index in indices
        ),
        mp.fsum(gradient[10:20]),
        gradient[12] + gradient[13],
    )


@pytest.fixture(scope="module")
def oracle(study):
    _, certificate = study
    mp = mpmath.mp.clone()
    mp.dps = 90
    edges = certificate.chamber.edges

    def restricted(*parameters):
        return _pullback(mp, _gradient(mp, _face_map(mp, parameters), edges))

    center = tuple(_mp(mp, value) for value in certificate.face.center)
    root = tuple(mp.findroot(restricted, center, tol=mp.mpf("1e-80")))
    phases = _face_map(mp, root)
    return mp, root, phases, restricted


def test_every_edge_branch_and_independent_cycle_is_retained(study):
    _, certificate = study
    chamber = certificate.chamber
    edges = chamber.edges
    expected_wraps = {
        (left, right)
        for pair_left, pair_right in (((0, 1), (8, 9)), ((10, 11), (18, 19)))
        for left in pair_left
        for right in pair_right
    }
    assert len(edges) == 53
    assert chamber.cycle_rank == len(edges) - len(chamber.reset.nodes) + 1 == 33
    assert {
        edge for edge, turn in zip(edges, chamber.ideal.shifts, strict=True) if turn
    } == expected_wraps
    assert all(
        turn == int(edge in expected_wraps)
        for edge, turn in zip(edges, chamber.ideal.shifts, strict=True)
    )
    assert chamber.conditional.shifts == chamber.ideal.shifts
    assert chamber.conditional.minimum_acute_margin.lo > Q(1, 100)
    chords = tuple(cycle.chord_edge_index for cycle in chamber.cycles)
    for row, cycle in enumerate(chamber.cycles):
        incidence = [0] * len(chamber.reset.nodes)
        for coefficient, (left, right) in zip(
            cycle.edge_coefficients, edges, strict=True
        ):
            incidence[left] -= coefficient
            incidence[right] += coefficient
        assert incidence == [0] * len(incidence)
        # The identity submatrix on distinct chords proves row independence.
        assert tuple(cycle.edge_coefficients[column] for column in chords) == tuple(
            int(index == row) for index in range(len(chords))
        )
        assert cycle.winding == -sum(
            coefficient * shift
            for coefficient, shift in zip(
                cycle.edge_coefficients, chamber.ideal.shifts, strict=True
            )
        )


def test_interval_face_contains_independent_full_potential_root(study, oracle):
    _, certificate = study
    mp, root, phases, restricted = oracle
    face = certificate.face
    for interval, value in zip(face.parameters, root, strict=True):
        _contains(mp, interval, value)
    for interval, value in zip(face.phases, phases, strict=True):
        _contains(mp, interval, value)
    assert max(abs(value) for value in restricted(*root)) < mp.mpf("1e-75")
    for interval, (left, right), turn in zip(
        face.edge_differences,
        certificate.chamber.edges,
        certificate.chamber.ideal.shifts,
        strict=True,
    ):
        _contains(mp, interval, phases[right] - phases[left] - 2 * mp.pi * turn)
    assert face.root_inclusion_certified is True
    assert face.q < 1
    assert face.image_radius_bound < face.radius


def test_full_gradient_is_nonzero_and_has_a_feasible_face_normal(study, oracle):
    _, certificate = study
    mp, _, phases, _ = oracle
    chamber, face = certificate.chamber, certificate.face
    gradient = _gradient(mp, phases, chamber.edges)
    tolerance = mp.mpf("1e-75")
    multiplier = gradient[14]
    assert multiplier > mp.mpf("0.014")
    _contains(mp, face.multiplier, multiplier)
    for index, value in enumerate(gradient):
        expected = (
            multiplier
            if index in (14, 15)
            else (-multiplier if index in (10, 11) else mp.mpf(0))
        )
        assert abs(value - expected) < tolerance
        _contains(mp, face.full_gradient_bounds[index], value)

    normal = [0] * 21
    active = []
    for index, ((left, right), turn) in enumerate(
        zip(chamber.edges, chamber.ideal.shifts, strict=True)
    ):
        gap = phases[right] - phases[left] - 2 * mp.pi * turn
        if abs(abs(gap) - mp.pi / 2) < tolerance:
            active.append(index)
            sign = 1 if gap > 0 else -1
            normal[left] -= sign
            normal[right] += sign
        else:
            assert abs(gap) < mp.pi / 2
    assert tuple(active) == face.active_edge_indices
    assert len(active) == 12
    for value, direction in zip(gradient, normal, strict=True):
        assert abs(value + multiplier * direction / 2) < tolerance
    assert face.kkt_certified is True
    assert face.closed_stationary_excluded is True


def test_separation_applies_to_full_asymmetric_states_and_common_phase(study, oracle):
    _, certificate = study
    mp, _, boundary, _ = oracle
    chamber = certificate.chamber
    source = tuple(_mp(mp, value.midpoint) for value in chamber.ideal.phases)
    # This perturbation breaks a source twin; no symmetry restriction is imposed.
    asymmetric = list(source)
    asymmetric[0] += mp.mpf(1) / 1000
    boundary_gradient = _gradient(mp, boundary, chamber.edges)
    for state in (source, asymmetric):
        gaps = tuple(
            state[right] - state[left] - 2 * mp.pi * turn
            for (left, right), turn in zip(
                chamber.edges, chamber.ideal.shifts, strict=True
            )
        )
        assert all(abs(gap) < mp.pi / 2 for gap in gaps)
        displacement = tuple(
            value - base for value, base in zip(state, boundary, strict=True)
        )
        assert (
            mp.fsum(g * d for g, d in zip(boundary_gradient, displacement, strict=True))
            > 0
        )
        gradient = _gradient(mp, state, chamber.edges)
        monotonicity = mp.fsum(
            (value - base) * delta
            for value, base, delta in zip(
                gradient, boundary_gradient, displacement, strict=True
            )
        )
        assert monotonicity > 0
    translated = tuple(value + mp.mpf(7) / 3 for value in boundary)
    assert max(
        abs(a - b)
        for a, b in zip(
            _gradient(mp, translated, chamber.edges), boundary_gradient, strict=True
        )
    ) < mp.mpf("1e-75")


def test_radial_storage_path_preserves_exact_model_invariants_algebraically(
    study, oracle
):
    """Dyadic midpoint controls exercise the general path identity, not flow."""
    _, certificate = study
    mp, _, face, _ = oracle
    chamber = certificate.chamber
    x = tuple(_mp(mp, value) for value in chamber.reset.epi)
    theta = tuple(_mp(mp, value.midpoint) for value in chamber.ideal.phases)
    weights = tuple(_mp(mp, value.midpoint) for value in chamber.conductances)
    capacities = tuple(
        _mp(mp, value.midpoint) for value in chamber.reset.capacity_after
    )
    degrees = [0] * len(x)
    for left, right in chamber.edges:
        degrees[left] += 1
        degrees[right] += 1
    rho = tuple(
        degree / capacity for degree, capacity in zip(degrees, capacities, strict=True)
    )

    def mean(values):
        return mp.fsum(
            weight * value for weight, value in zip(rho, values, strict=True)
        ) / mp.fsum(rho)

    def form_storage(values):
        return mp.fsum(
            weight * (values[i] - values[j]) ** 2 / 2
            for weight, (i, j) in zip(weights, chamber.edges, strict=True)
        )

    form_mean, phase_mean = mean(x), mean(theta)
    boundary = tuple(value + phase_mean - mean(face) for value in face)
    q0 = form_storage(x)
    assert q0 > 0
    _contains(mp, certificate.source_form_storage, q0)
    assert certificate.conditional_source_form_storage.lo > 0
    v0 = _potential(mp, theta, chamber.edges)
    h0 = q0 + v0
    assert _potential(mp, boundary, chamber.edges) < v0
    for parameter in (mp.mpf(0), mp.mpf(1) / 2, mp.mpf(1), mp.mpf(10001) / 10000):
        phases = tuple(
            a + parameter * (b - a) for a, b in zip(theta, boundary, strict=True)
        )
        potential = _potential(mp, phases, chamber.edges)
        assert h0 > potential
        factor = mp.sqrt((h0 - potential) / q0)
        forms = tuple(form_mean + factor * (value - form_mean) for value in x)
        assert abs(mean(phases) - phase_mean) < mp.mpf("1e-75")
        assert abs(mean(forms) - form_mean) < mp.mpf("1e-75")
        assert abs(form_storage(forms) + potential - h0) < mp.mpf("1e-75")
        gaps = tuple(
            phases[j] - phases[i] - 2 * mp.pi * turn
            for (i, j), turn in zip(chamber.edges, chamber.ideal.shifts, strict=True)
        )
        if parameter < 1:
            assert max(abs(value) for value in gaps) < mp.pi / 2
        elif parameter > 1:
            assert all(
                abs(gaps[index]) > mp.pi / 2
                for index in certificate.face.active_edge_indices
            )


def test_static_analysis_leaves_retained_event_history_and_support_untouched(study):
    source, certificate = study
    assert source.graph.number_of_nodes() == 21
    assert source.graph.number_of_edges() == 45
    assert source.graph.degree("2_sub_0") == 0
    assert tuple(source.graph.nodes[2]["glyph_history"]) == ("VAL", "THOL")
    assert tuple(source.graph.nodes["2_sub_0"]["glyph_history"]) == ()
    assert len(certificate.chamber.edges) == 53
    assert certificate.chamber.numerical_allowance == Q(1, 2**40)
